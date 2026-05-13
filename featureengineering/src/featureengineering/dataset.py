from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .settings import ProjectPaths, configure_paths


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


def _build_daily_financial(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    value_cols: list[str],
    on_progress: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """Convert report-frequency financial data into a daily-forward-filled panel.

    Parameters
    ----------
    df : DataFrame
        Raw report-level data with columns ``stock_code``, ``ann_date``, ``end_date``,
        and the target value columns.
    calendar : DataFrame
        Trading calendar with column ``date`` (YYYY-MM-DD strings).
    value_cols : list[str]
        Columns to forward-fill.
    on_progress : callable or None
        Called as on_progress(stage, current, total) for sub-progress reporting.
    """
    def _report(stage, current, total):
        if on_progress:
            on_progress(stage, current, total)

    df = df.copy()
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df["stock_code"] = df["stock_code"].apply(_pad_code)
    calendar_dates = pd.to_datetime(calendar["date"]).sort_values()

    # Build a stock-date grid
    codes = sorted(df["stock_code"].unique())
    n_codes = len(codes)

    # For each (stock_code, ann_date), keep the latest report before ann_date
    df = df.sort_values(["stock_code", "ann_date"])
    df = df.drop_duplicates(subset=["stock_code", "ann_date"], keep="last")

    _report("ffill", 0, n_codes)

    panels = []
    for i, (code, group) in enumerate(df.groupby("stock_code")):
        group = group.set_index("ann_date").sort_index()
        sub = group[value_cols].reindex(calendar_dates, method="ffill")
        sub["code"] = code
        sub = sub.reset_index().rename(columns={"index": "date"})
        panels.append(sub)
        if (i + 1) % 200 == 0 or i + 1 == n_codes:
            _report("ffill", i + 1, n_codes)

    if not panels:
        return pd.DataFrame(columns=value_cols)

    _report("concat", 0, 1)
    result = pd.concat(panels, ignore_index=True)
    _report("concat", 1, 1)

    _report("index", 0, 1)
    result["code"] = result["code"].apply(_pad_code)
    result["date"] = result["date"].dt.strftime("%Y%m%d")
    result = result.set_index(["date", "code"]).sort_index()
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    _report("index", 1, 1)
    return result


@dataclass
class DataRepository:
    paths: ProjectPaths | None = None
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    on_progress: Callable[[str, int, int], None] | None = None

    def __post_init__(self) -> None:
        if self.paths is None:
            self.paths = configure_paths()

    @property
    def allowed_codes(self) -> set[str]:
        return _load_allowed_codes(self.paths.stock_pool_file)

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _filter_by_allowed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose Code is in the allowed stock pool."""
        allowed = self.allowed_codes
        if not allowed:
            return df
        codes = df.index.get_level_values("Code")
        mask = codes.isin(allowed)
        return df.loc[mask]

    # ── daily panel loading ───────────────────────────────────────────

    def load_panel(self, relative_path: str) -> pd.DataFrame:
        """Load a daily-frequency parquet file and normalize to (Date, Code) panel."""
        if relative_path not in self._cache:
            if self.on_progress:
                self.on_progress("load", 0, 1)
            raw = self._read_parquet(self.paths.source_root / relative_path)
            normalized = _normalize_index_frame(raw)
            self._cache[relative_path] = self._filter_by_allowed(normalized)
            if self.on_progress:
                self.on_progress("load", 1, 1)
        return self._cache[relative_path]

    # ── report-frequency financial loading ────────────────────────────

    def load_financial_panel(
        self, relative_path: str, value_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Load a report-frequency financial table and forward-fill to daily panel."""
        cols_tag = "" if value_cols is None else f"_{'_'.join(sorted(value_cols))}"
        cache_key = f"__financial__{relative_path}{cols_tag}"
        if cache_key not in self._cache:
            raw = self._read_parquet(self.paths.source_root / relative_path)
            calendar = self._read_parquet(self.paths.source_root / "calendar.parquet")
            if value_cols is None:
                exclude = {"stock_code", "ann_date", "f_ann_date", "end_date",
                           "report_type", "comp_type", "end_type"}
                value_cols = [c for c in raw.columns if c not in exclude]
            self._cache[cache_key] = self._filter_by_allowed(
                _build_daily_financial(raw, calendar, value_cols,
                                       on_progress=self.on_progress)
            )
        return self._cache[cache_key]

    # ── stock pool ─────────────────────────────────────────────────────

    def load_stock_pool(self) -> pd.DataFrame:
        """Return listed stocks filtered to the allowed pool."""
        cache_key = "__stock_pool__"
        if cache_key not in self._cache:
            raw = self._read_parquet(self.paths.source_root / "stock_list.parquet")
            pool = raw[raw["list_status"] == "L"].copy()
            pool["code"] = pool["stock_code"].apply(_pad_code)
            pool = pool[["code", "industry", "area"]].reset_index(drop=True)
            pool = pool.rename(columns={"code": "Code"})
            allowed = self.allowed_codes
            if allowed:
                pool = pool[pool["Code"].isin(allowed)]
            self._cache[cache_key] = pool.sort_values("Code").reset_index(drop=True)
        return self._cache[cache_key]

    def available_codes(self) -> pd.Index:
        return pd.Index(self.load_stock_pool()["Code"], name="Code")

    def load_industry_map(self) -> pd.Series:
        """Return a Series mapping stock_code → industry."""
        pool = self.load_stock_pool()
        return pool.set_index("Code")["industry"]
