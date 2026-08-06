from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .settings import ProjectPaths, configure_paths


_MAX_CACHE_SIZE = 16
_MARGIN_CALENDAR_BUFFER_DAYS = 14


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



def _fill_source_gaps(
    df: pd.DataFrame,
    trading_dates: list[str],
    extend_to_date: str | None = None,
) -> pd.DataFrame:
    """Forward-fill whole-date source gaps on the canonical daily calendar."""
    if df.empty:
        return df
    if not hasattr(df.index, "names") or list(df.index.names) != ["Date", "Code"]:
        return df
    ref_dates = trading_dates
    current_dates = df.index.get_level_values("Date").unique()
    current_min, current_max = current_dates.min(), current_dates.max()

    # ── 1. fill internal gaps ──────────────────────────────────────────
    ref_in_range = [d for d in ref_dates if current_min <= d <= current_max]
    missing = sorted(set(ref_in_range) - set(current_dates))

    result = df
    codes = df.index.get_level_values("Code").unique()
    current_dates_sorted = sorted(current_dates)

    if missing:
        fill_parts = []
        for miss_d in missing:
            prev = None
            for d in reversed(current_dates_sorted):
                if d < miss_d:
                    prev = d
                    break
            if prev is None:
                continue
            prev_data = result.loc[prev]
            new_idx = pd.MultiIndex.from_product(
                [[miss_d], codes], names=["Date", "Code"]
            )
            new_data = prev_data.reindex(codes)
            new_data.index = new_idx
            fill_parts.append(new_data)
        if fill_parts:
            fill_df = pd.concat(fill_parts)
            fill_df = fill_df.reorder_levels(["Date", "Code"]).sort_index()
            result = pd.concat([result, fill_df])
            result = result[~result.index.duplicated(keep="last")]
            result = result.sort_index()

    # ── 2. extend forward to extend_to_date ────────────────────────────
    if extend_to_date is not None:
        current_max_str = result.index.get_level_values("Date").max()
        if current_max_str < extend_to_date:
            ext_dates = [
                d for d in ref_dates if current_max_str < d <= extend_to_date
            ]
            if ext_dates:
                # 只向"最后数据日期实际存在的代码"扩展,不能使用全历史 codes:
                # 对已退出面板的代码(如历史上曾融资、现已退出的股票),全历史
                # reindex 会在尾日造出仅尾日存在的行,后续 groupby shift(1)
                # 按行位置平移时会把它们数年前的最后值拉进尾日 → 尾日因子
                # 混入陈年伪值(曾导致 margin 因子尾日 n=1782 而非 1742)。
                last_data = result.loc[current_max_str]
                tail_codes = last_data.index.unique()
                for ext_d in ext_dates:
                    new_idx = pd.MultiIndex.from_product(
                        [[ext_d], tail_codes], names=["Date", "Code"]
                    )
                    new_data = last_data.reindex(tail_codes)
                    new_data.index = new_idx
                    result = pd.concat([result, new_data])
                result = result.sort_index()

    return result


def _lag_panel_one_trading_day(
    df: pd.DataFrame,
    trading_dates: list[str],
) -> pd.DataFrame:
    """Move each observation to the next canonical trading date.

    ``margin_detail`` is published one trading day late.  Relabelling raw date
    T-1 as availability date T is stricter than ``groupby().shift(1)``: if a
    stock is missing on T-1, T remains missing instead of silently reusing an
    older observation.
    """
    if df.empty:
        return df
    next_date = dict(zip(trading_dates[:-1], trading_dates[1:]))
    source_dates = df.index.get_level_values("Date")
    available_dates = source_dates.map(next_date)
    keep = available_dates.notna()
    if not keep.any():
        return df.iloc[0:0].copy()

    result = df.loc[keep].copy()
    result.index = pd.MultiIndex.from_arrays(
        [
            available_dates[keep],
            df.index.get_level_values("Code")[keep],
        ],
        names=["Date", "Code"],
    )
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()

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
    elif {"trade_date", "ths_code"}.issubset(frame.columns):
        frame = frame.rename(columns={"trade_date": "date", "ths_code": "code"})
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
    # Only forward-fill to trading days (is_open=1), skipping weekends and holidays.
    # calendar.parquet contains both trading and non-trading days — without this
    # filter, the resulting panel has ~52% useless rows for every financial factor.
    if "is_open" in calendar.columns:
        trading = calendar.loc[calendar["is_open"].astype(bool)]
    else:
        trading = calendar
    calendar_dates = pd.to_datetime(trading["date"]).sort_values()

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
    _trading_dates_cache: list[str] | None = None
    on_progress: Callable[[str, int, int], None] | None = None

    def __post_init__(self) -> None:
        if self.paths is None:
            self.paths = configure_paths()

    @property
    def allowed_codes(self) -> set[str]:
        return _load_allowed_codes(self.paths.stock_pool_file)

    def _read_parquet(self, path: Path, filters: list[tuple] | None = None) -> pd.DataFrame:
        return pd.read_parquet(path, filters=filters)

    def _read_parquet_columns(self, path: Path, columns: list[str],
                              filters: list[tuple] | None = None) -> pd.DataFrame:
        """Read only specified columns from a parquet file (columnar access)."""
        return pd.read_parquet(path, columns=columns, filters=filters)

    def _filter_by_allowed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose Code is in the allowed stock pool."""
        allowed = self.allowed_codes
        if not allowed:
            return df
        codes = df.index.get_level_values("Code")
        mask = codes.isin(allowed)
        return df.loc[mask]

    def _drop_backfill_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop is_backfill==True rows (re-downloaded backfill data).

        Point-in-time discipline: backfilled rows may carry information that
        was not available at the recorded date.  The column is object-dtype
        with False/None/True values; ``ne(True)`` keeps both False and missing
        (None) rows and drops only True.  Sources without the column are
        passed through unchanged.  Index is preserved.
        """
        if "is_backfill" not in df.columns:
            return df
        return df.loc[df["is_backfill"].ne(True)]

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

    def _trading_dates(self) -> list[str]:
        """Return the canonical trading calendar from unadjusted daily data."""
        if self._trading_dates_cache is None:
            raw = self._read_parquet_columns(
                self.paths.source_root / "daily.parquet", ["trade_date"]
            )
            dates = (
                raw["trade_date"]
                .astype(str)
                .str.replace("-", "", regex=False)
                .str.slice(0, 8)
            )
            self._trading_dates_cache = sorted(dates.dropna().unique().tolist())
        return self._trading_dates_cache

    # ── daily panel loading ───────────────────────────────────────────

    # Date column candidates for parquet predicate pushdown, in priority order.
    _DATE_COL_CANDIDATES = ["trade_date", "end_date", "date", "ann_date", "f_ann_date"]

    def _build_date_filters(
        self,
        path: Path,
        min_date: str | None,
        max_date: str | None,
        lookback_days: int = 0,
    ) -> tuple[list[tuple] | None, str | None]:
        """Build parquet-level row filters for date range, if possible.

        Reads the file schema to detect the date column.  Returns
        ``(filters, date_col)`` where *filters* may be None if no
        pushdown is possible (missing date column, no range supplied).
        """
        if min_date is None and max_date is None:
            return None, None
        import pyarrow.parquet as _pq
        try:
            schema = _pq.read_schema(path)
        except Exception:
            return None, None
        date_col = None
        for c in self._DATE_COL_CANDIDATES:
            if c in schema.names:
                date_col = c
                break
        if date_col is None:
            return None, None
        from datetime import datetime as _dt, timedelta as _td
        filters: list[tuple] = []
        if min_date is not None:
            min_dt = _dt.strptime(min_date, "%Y%m%d") - _td(days=lookback_days)
            # Source data uses YYYY-MM-DD format (verified against
            # daily.parquet, finance.parquet, etc.).
            filters.append((date_col, ">=", min_dt.strftime("%Y-%m-%d")))
        if max_date is not None:
            # max_date is YYYYMMDD, convert to YYYY-MM-DD for the filter
            max_dt = _dt.strptime(max_date, "%Y%m%d")
            filters.append((date_col, "<=", max_dt.strftime("%Y-%m-%d")))
        return filters, date_col

    def load_panel(
        self,
        relative_path: str,
        min_date: str | None = None,
        max_date: str | None = None,
        lookback_days: int = 0,
    ) -> pd.DataFrame:
        """Load a daily-frequency parquet file and normalize to (Date, Code) panel.

        When *min_date* or *max_date* are provided, parquet predicate pushdown
        is used to reduce I/O.  Filtered reads are **not** cached — only
        full-file reads populate the LRU cache.
        """
        cache_key = relative_path
        filepath = self.paths.source_root / relative_path
        is_margin = "margin_detail" in relative_path
        # A fourteen-calendar-day read buffer covers weekends and the longest
        # regular mainland-China public-holiday closures.  The buffer is only
        # used to obtain T-1 and is removed again after calendar relabelling.
        margin_buffer = _MARGIN_CALENDAR_BUFFER_DAYS if is_margin else 0

        # Try parquet-level pushdown first
        pq_filters, _ = self._build_date_filters(
            filepath, min_date, max_date, lookback_days + margin_buffer
        )

        if pq_filters is not None:
            # Partial read — don't cache (the full file isn't in memory).
            raw = self._read_parquet(filepath, filters=pq_filters)
            normalized = _normalize_index_frame(raw)
            result = self._filter_by_allowed(normalized)
            result = self._drop_backfill_rows(result)
            if is_margin:
                result = _lag_panel_one_trading_day(result, self._trading_dates())
                result = _filter_by_date_range(
                    result, min_date, max_date, lookback_days=lookback_days
                )
            else:
                # Still apply memory-side filtering for safety if parquet
                # predicate pushdown used a source-specific date dtype.
                result = _filter_by_date_range(
                    result, min_date, max_date, lookback_days=lookback_days
                )
                result = _fill_source_gaps(result, self._trading_dates())
            return result

        # Full-file read path (cached)
        cached = self._cache_get(cache_key)
        if cached is None:
            if self.on_progress:
                self.on_progress("load", 0, 1)
            raw = self._read_parquet(filepath)
            normalized = _normalize_index_frame(raw)
            allowed = self._filter_by_allowed(normalized)
            self._cache_put(cache_key, self._drop_backfill_rows(allowed))
            if self.on_progress:
                self.on_progress("load", 1, 1)
            full = self._cache[cache_key]
        else:
            full = cached
        if is_margin:
            full = _filter_by_date_range(
                full, min_date, max_date, lookback_days + margin_buffer
            )
            full = _lag_panel_one_trading_day(full, self._trading_dates())
            full = _filter_by_date_range(
                full, min_date, max_date, lookback_days=lookback_days
            )
        else:
            full = _filter_by_date_range(
                full, min_date, max_date, lookback_days=lookback_days
            )
            full = _fill_source_gaps(full, self._trading_dates())
        return full

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
        is_margin = "margin_detail" in relative_path
        margin_extra = _MARGIN_CALENDAR_BUFFER_DAYS if is_margin else 0
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
        full = _filter_by_date_range(full, min_date, max_date, lookback_days + margin_extra)
        if is_margin:
            full = _lag_panel_one_trading_day(full, self._trading_dates())
            full = _filter_by_date_range(
                full, min_date, max_date, lookback_days=lookback_days
            )
        else:
            full = _fill_source_gaps(full, self._trading_dates())
        return full

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
        pool = pool.drop_duplicates(subset=["Code"])
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

    # ── factor panel loading (Class 4 coupling factors) ──────────────────

    def load_factor_panel(
        self,
        name: str,
        min_date: str | None = None,
        max_date: str | None = None,
        lookback_days: int = 0,
    ) -> pd.Series:
        """Load a pre-computed factor from a .fea file and return as (Date, Code) Series.

        The .fea file is in wide format (Date index, Code columns).  We convert
        it to a (Date, Code)-indexed Series and filter on the requested date range.
        """
        filepath = self.paths.factor_output_dir / f"{name}.fea"
        if not filepath.exists():
            raise FileNotFoundError(f"Factor file not found: {filepath}")

        df = pd.read_feather(filepath)

        # Wide format: Date index (str YYYYMMDD), Code columns
        if df.columns[0] != "Date" and "Date" not in df.columns:
            # Assume the index is Date
            pass
        else:
            if "Date" in df.columns:
                df = df.set_index("Date")

        df.index = df.index.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

        # Stack to (Date, Code) MultiIndex
        stacked = df.stack(future_stack=True)
        stacked.index = stacked.index.set_names(["Date", "Code"])
        stacked.name = name

        # Ensure numeric dtype: PyArrow-backed feather files may produce
        # object-dtype Series after stacking, which breaks numpy ufuncs
        # (np.log, np.sign, etc.) that rely on native float64.
        stacked = pd.to_numeric(stacked, errors="coerce")

        # Filter by date range
        if min_date is not None:
            from datetime import datetime, timedelta
            min_dt = datetime.strptime(min_date, "%Y%m%d")
            lookback_dt = min_dt - timedelta(days=lookback_days)
            cutoff = lookback_dt.strftime("%Y%m%d")
            stacked = stacked.loc[stacked.index.get_level_values("Date") >= cutoff]
        if max_date is not None:
            stacked = stacked.loc[stacked.index.get_level_values("Date") <= max_date]

        return stacked
