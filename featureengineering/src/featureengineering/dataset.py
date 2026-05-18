from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .settings import ProjectPaths, configure_paths


_MAX_CACHE_SIZE = 16


# ── allowed stock pool ──────────────────────────────────────────────────────
_ALLOWED_CODES: set[str] | None = None


def _pad_code(code: str) -> str:
    """Strip exchange suffix and zero-pad to 6 digits, e.g. '000001.SZ' → '000001'."""
    code = str(code).strip()
    if code.upper().endswith(".SZ"):
        code = code[:-3]
    elif code.upper().endswith(".SH"):
        code = code[:-3]
    elif code.upper().endswith(".BSE"):
        code = code[:-4]
    return code.zfill(6)


def _load_allowed_codes(path: Path) -> set[str]:
    global _ALLOWED_CODES
    if _ALLOWED_CODES is not None:
        return _ALLOWED_CODES
    codes: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.strip()
            if code:
                codes.add(_pad_code(code))
    _ALLOWED_CODES = codes
    return _ALLOWED_CODES


def _normalize_index_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a flat parquet DataFrame into a (Date, Code) MultiIndex panel.

    Handles:
      - Flat tables with ``trade_date`` / ``stock_code`` columns (daily data)
      - Tables that already have a ``date`` / ``code`` MultiIndex (legacy)
    """
    frame = df.copy()

    # Already a MultiIndex
    if isinstance(frame.index, pd.MultiIndex):
        names = [str(name).lower() if name is not None else "" for name in frame.index.names]
        if names == ["date", "code"] or names == ["code", "date"]:
            frame = frame.reset_index()
    elif {"trade_date", "stock_code"}.issubset(frame.columns):
        frame = frame.rename(columns={"trade_date": "date", "stock_code": "code"})
    elif {"date", "code"}.issubset(frame.columns):
        pass
    elif {"Date", "Code"}.issubset(frame.columns):
        frame = frame.rename(columns={"Date": "date", "Code": "code"})
    else:
        raise ValueError(
            "Expected a panel with trade_date/stock_code or date/code columns, "
            f"got columns: {list(frame.columns)}"
        )

    if {"date", "code"}.issubset(frame.columns):
        frame["date"] = frame["date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
        frame["code"] = frame["code"].apply(_pad_code)
        # Drop helper columns that are not factors
        drop_cols = [c for c in frame.columns if c in ("stock_name", "name")]
        if drop_cols:
            frame = frame.drop(columns=drop_cols)
        frame = frame.set_index(["date", "code"]).sort_index()
        frame.index = frame.index.set_names(["Date", "Code"])
        frame = frame.reorder_levels(["Date", "Code"]).sort_index()
        return frame

    raise ValueError("Failed to normalize panel index.")


def _filter_by_date_range(
    df: pd.DataFrame,
    min_date: str | None,
    max_date: str | None,
    lookback_days: int = 0,
) -> pd.DataFrame:
    """Filter a (Date, Code) MultiIndex DataFrame to a date range.

    If *min_date* is provided, dates >= (min_date - lookback_days) are kept.
    If *max_date* is provided, dates <= max_date are kept.
    When both are None, returns the DataFrame unchanged.
    """
    if min_date is None and max_date is None:
        return df

    dates = df.index.get_level_values("Date")

    if min_date is not None:
        from datetime import datetime, timedelta
        min_dt = datetime.strptime(min_date, "%Y%m%d")
        lookback_dt = min_dt - timedelta(days=lookback_days)
        cutoff = lookback_dt.strftime("%Y%m%d")
        df = df.loc[dates >= cutoff]

    if max_date is not None:
        dates = df.index.get_level_values("Date")
        df = df.loc[dates <= max_date]

    return df


def _build_daily_financial(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    value_cols: list[str],
    on_progress: Callable[[str, int, int], None] | None = None,
    date_col: str = "ann_date",
) -> pd.DataFrame:
    """Convert report-frequency financial data into a daily-forward-filled panel.

    Uses a vectorized pivot-ffill-stack approach instead of per-stock iteration
    to avoid O(n_stocks) Python-level overhead.
    """
    def _report(stage, current, total):
        if on_progress:
            on_progress(stage, current, total)

    df = df.copy()
    df["ann_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["stock_code"] = df["stock_code"].apply(_pad_code)
    calendar_dates = pd.to_datetime(calendar["date"]).sort_values()

    # Keep latest report per (stock_code, ann_date)
    df = df.sort_values(["stock_code", "ann_date"])
    df = df.drop_duplicates(subset=["stock_code", "ann_date"], keep="last")

    if not value_cols:
        return pd.DataFrame()

    _report("ffill", 0, 1)

    # Pivot all value columns in a single pass (was: per-column loop)
    pivot = df.pivot_table(
        index="ann_date", columns="stock_code", values=value_cols, aggfunc="last"
    )
    pivot = pivot.reindex(calendar_dates).ffill()

    # Stack stock_code level back to rows.
    # After pivot_table with a list of values, columns is always a MultiIndex
    # with (value_col, stock_code) levels; stack the stock_code level.
    if isinstance(pivot.columns, pd.MultiIndex):
        result = pivot.stack(level=1, future_stack=True)
    else:
        result = pivot.stack(future_stack=True).to_frame(value_cols[0])

    _report("ffill", 1, 1)

    _report("index", 0, 1)
    result.index = result.index.set_names(["date", "code"])
    result = result.reset_index()
    result["date"] = result["date"].dt.strftime("%Y%m%d")
    result["code"] = result["code"].apply(_pad_code)
    result = result.set_index(["date", "code"]).sort_index()
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    _report("index", 1, 1)
    return result


@dataclass
class DataRepository:
    paths: ProjectPaths | None = None
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    _cache_order: deque[str] = field(default_factory=deque)
    on_progress: Callable[[str, int, int], None] | None = None

    def __post_init__(self) -> None:
        if self.paths is None:
            self.paths = configure_paths()

    @property
    def allowed_codes(self) -> set[str]:
        return _load_allowed_codes(self.paths.stock_pool_file)

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _read_parquet_columns(self, path: Path, columns: list[str]) -> pd.DataFrame:
        """Read only specified columns from a parquet file (columnar access)."""
        return pd.read_parquet(path, columns=columns)

    def _filter_by_allowed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose Code is in the allowed stock pool."""
        allowed = self.allowed_codes
        if not allowed:
            return df
        codes = df.index.get_level_values("Code")
        mask = codes.isin(allowed)
        return df.loc[mask]

    def _cache_put(self, key: str, value: pd.DataFrame) -> None:
        """Store in cache with LRU eviction when the cache exceeds _MAX_CACHE_SIZE."""
        if len(self._cache) >= _MAX_CACHE_SIZE:
            oldest = self._cache_order.popleft()
            del self._cache[oldest]
        self._cache[key] = value
        self._cache_order.append(key)

    def _cache_get(self, key: str) -> pd.DataFrame | None:
        """Retrieve from cache and bump the key to MRU position."""
        if key not in self._cache:
            return None
        # Move key to end (MRU)
        self._cache_order.remove(key)
        self._cache_order.append(key)
        return self._cache[key]

    # ── daily panel loading ───────────────────────────────────────────

    def load_panel(
        self,
        relative_path: str,
        min_date: str | None = None,
        max_date: str | None = None,
        lookback_days: int = 0,
    ) -> pd.DataFrame:
        """Load a daily-frequency parquet file and normalize to (Date, Code) panel.

        When *min_date* or *max_date* are provided, the returned panel is
        filtered to the specified date range (with *lookback_days* buffer
        subtracted from *min_date* to provide context for rolling windows).
        """
        cache_key = relative_path
        cached = self._cache_get(cache_key)
        if cached is None:
            if self.on_progress:
                self.on_progress("load", 0, 1)
            raw = self._read_parquet(self.paths.source_root / relative_path)
            normalized = _normalize_index_frame(raw)
            self._cache_put(cache_key, self._filter_by_allowed(normalized))
            if self.on_progress:
                self.on_progress("load", 1, 1)
            full = self._cache[cache_key]
        else:
            full = cached
        return _filter_by_date_range(full, min_date, max_date, lookback_days)

    # ── report-frequency financial loading ────────────────────────────

    def load_financial_panel(
        self,
        relative_path: str,
        value_cols: list[str] | None = None,
        date_col: str = "ann_date",
        min_date: str | None = None,
        max_date: str | None = None,
        lookback_days: int = 0,
    ) -> pd.DataFrame:
        """Load a report-frequency financial table and forward-fill to daily panel.

        Parameters
        ----------
        relative_path : str
            Path to the parquet file relative to source_root.
        value_cols : list[str] or None
            Columns to forward-fill. If None, auto-detect all value columns.
        date_col : str
            Date column to use as ffill anchor (default: ann_date).
            Use 'end_date' for data like pledge_stat.parquet.
        min_date, max_date : str or None
            Date range filter (YYYYMMDD). When provided, the returned panel
            is filtered after forward-filling.
        lookback_days : int
            Extra days subtracted from *min_date* for rolling-window context.
        """
        cols_tag = "" if value_cols is None else f"_{'_'.join(sorted(value_cols))}"
        cache_key = f"__financial__{relative_path}{cols_tag}_{date_col}"
        cached = self._cache_get(cache_key)
        if cached is None:
            if value_cols is not None:
                needed = list(value_cols) + ["stock_code", date_col]
                raw = self._read_parquet_columns(self.paths.source_root / relative_path, needed)
            else:
                raw = self._read_parquet(self.paths.source_root / relative_path)
            calendar = self._cache_get("__calendar__")
            if calendar is None:
                calendar = self._read_parquet(self.paths.source_root / "calendar.parquet")
                self._cache_put("__calendar__", calendar)
            if value_cols is None:
                exclude = {"stock_code", "ann_date", "f_ann_date", "end_date",
                           "report_type", "comp_type", "end_type", "update_time",
                           "name", "stock_name"}
                value_cols = [c for c in raw.columns if c not in exclude]
            self._cache_put(cache_key, self._filter_by_allowed(
                _build_daily_financial(raw, calendar, value_cols,
                                       on_progress=self.on_progress,
                                       date_col=date_col)
            ))

        full = self._cache[cache_key]
        return _filter_by_date_range(full, min_date, max_date, lookback_days)

    # ── stock pool ─────────────────────────────────────────────────────

    def load_stock_pool(self) -> pd.DataFrame:
        """Return listed stocks filtered to the allowed pool."""
        cache_key = "__stock_pool__"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        raw = self._read_parquet(self.paths.source_root / "stock_list.parquet")
        pool = raw[raw["list_status"] == "L"].copy()
        pool["code"] = pool["stock_code"].apply(_pad_code)
        pool = pool[["code", "industry", "area"]].reset_index(drop=True)
        pool = pool.rename(columns={"code": "Code"})
        allowed = self.allowed_codes
        if allowed:
            pool = pool[pool["Code"].isin(allowed)]
        result = pool.sort_values("Code").reset_index(drop=True)
        self._cache_put(cache_key, result)
        return result

    def available_codes(self) -> pd.Index:
        return pd.Index(self.load_stock_pool()["Code"], name="Code")

    def load_industry_map(self) -> pd.Series:
        """Return a Series mapping stock_code → industry."""
        pool = self.load_stock_pool()
        return pool.set_index("Code")["industry"]
