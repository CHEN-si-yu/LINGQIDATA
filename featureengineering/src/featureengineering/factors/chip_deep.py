"""Phase 2 factors — cyq_chips raw histogram.

These are heavy-data factors (461M rows source).  Must be built sequentially,
separately from the parallel Phase 1 pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


def _pad_code(code: str) -> str:
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _compute_chip_peak(
    source_path: str,
    allowed_codes: set[str],
    on_progress=None,
) -> pd.Series:
    """Compute chip peak price for each (date, code) from raw cyq_chips.

    Processes 445 row groups (~1M rows each), finding the price bucket
    with maximum chip percentage per stock per date.

    Returns a Series with (Date, Code) MultiIndex.
    """
    pf = pq.ParquetFile(source_path)
    n_groups = pf.metadata.num_row_groups
    MAX_PRICE = 10000.0  # sanity cap

    peaks: list[tuple[str, str, float]] = []

    for i in range(n_groups):
        tab = pf.read_row_group(i, columns=["trade_date", "stock_code", "price", "percent"])

        # Convert to pandas for groupby aggregation
        df = tab.to_pandas()
        df["stock_code"] = df["stock_code"].apply(_pad_code)
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)

        # Filter to allowed stocks
        if allowed_codes:
            df = df[df["stock_code"].isin(allowed_codes)]

        if df.empty:
            if on_progress:
                on_progress("chip_peak", i + 1, n_groups)
            continue

        # Filter outliers
        df = df[(df["price"] > 0) & (df["price"] < MAX_PRICE) & (df["percent"] > 0)]

        # Per group, find the price where percent is max
        idx = df.groupby(["trade_date", "stock_code"], sort=False)["percent"].idxmax()
        idx = idx.dropna().astype(int)
        best = df.loc[idx, ["trade_date", "stock_code", "price"]]
        peaks.extend(zip(best["trade_date"], best["stock_code"], best["price"]))

        if on_progress:
            on_progress("chip_peak", i + 1, n_groups)

    if not peaks:
        return pd.Series(dtype=float, name="chip_peak")

    result = pd.DataFrame(peaks, columns=["Date", "Code", "peak_price"])
    result = result.set_index(["Date", "Code"]).sort_index()
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    s = result["peak_price"]
    # Deduplicate index (keep last if duplicates across row groups)
    s = s.groupby(s.index.names).last()
    return s


def _forward_fill_to_daily(
    monthly_series: pd.Series,
    calendar: pd.DataFrame,
    allowed_codes: set[str],
) -> pd.Series:
    """Forward-fill a monthly/(Date,Code) series to daily trading calendar."""
    df = monthly_series.reset_index()
    wide = df.pivot(index="Date", columns="Code", values=monthly_series.name)
    # Only trading days
    if "is_open" in calendar.columns:
        trading = calendar.loc[calendar["is_open"].astype(bool)]
    else:
        trading = calendar
    cal_dates = trading["date"].astype(str).str.replace("-", "").str.slice(0, 8)
    wide.index = wide.index.astype(str)
    wide = wide.reindex(cal_dates).ffill()
    stacked = wide.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked.name = monthly_series.name
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()
    if allowed_codes:
        codes_mask = stacked.index.get_level_values("Code").isin(allowed_codes)
        stacked = stacked.loc[codes_mask]
    return stacked


@register_factor(
    name="chip_peak_distance",
    description="筹码峰距离因子，(close-chip_peak_price)/close截面排名。价格在最大筹码峰上方=支撑排前。",
    category="price",
    thesis="最大筹码峰是最密集的持仓成本区，价格在峰上方时有强支撑，在峰下方时变为强阻力。",
    dependencies=("cyq_chips.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_chip_peak_distance(context: FactorContext):
    source_path = str(context.repo.paths.source_root / "cyq_chips.parquet")
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    on_progress = context.repo.on_progress

    peak_series = _compute_chip_peak(
        source_path, context.repo.allowed_codes, on_progress=on_progress,
    )

    # Forward-fill peak to daily (peak is computed per-date from histogram)
    # Peak data inherits the dates of cyq_chips which are daily — no ffill needed
    common = close.index.intersection(peak_series.index)
    distance = (close.loc[common] - peak_series.loc[common]) / close.loc[common].replace(0, np.nan)
    return cross_sectional_rank(distance)


def _compute_chip_below_ratio(
    source_path: str,
    daily_adj_path: str,
    allowed_codes: set[str],
    on_progress=None,
) -> pd.Series:
    """Compute below-price chip area ratio for each (date, code) from cyq_chips.

    For each stock per date, sum the chip percentages for all price buckets
    below the day's close price, divided by total chip area.

    Returns a Series with (Date, Code) MultiIndex.
    """
    import pyarrow.dataset as ds

    # Load close prices for all dates
    close_df = pd.read_parquet(daily_adj_path, columns=["trade_date", "stock_code", "close"])
    close_df["trade_date"] = close_df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    close_df["stock_code"] = close_df["stock_code"].apply(_pad_code)
    if allowed_codes:
        close_df = close_df[close_df["stock_code"].isin(allowed_codes)]
    close_map = close_df.set_index(["trade_date", "stock_code"])["close"]

    pf = pq.ParquetFile(source_path)
    n_groups = pf.metadata.num_row_groups
    MAX_PRICE = 10000.0

    # Accumulate: (date, code) -> (below_area, total_area)
    accum: dict[tuple[str, str], tuple[float, float]] = {}

    for i in range(n_groups):
        tab = pf.read_row_group(i, columns=["trade_date", "stock_code", "price", "percent"])
        df = tab.to_pandas()
        df["stock_code"] = df["stock_code"].apply(_pad_code)
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)

        if allowed_codes:
            df = df[df["stock_code"].isin(allowed_codes)]

        if df.empty:
            if on_progress:
                on_progress("chip_below", i + 1, n_groups)
            continue

        df = df[(df["price"] > 0) & (df["price"] < MAX_PRICE) & (df["percent"] > 0)]

        # Merge close price onto each row
        df = df.merge(
            close_map.rename("close").reset_index(),
            left_on=["trade_date", "stock_code"],
            right_on=["trade_date", "stock_code"],
            how="inner",
        )

        if df.empty:
            if on_progress:
                on_progress("chip_below", i + 1, n_groups)
            continue

        df["below"] = df["price"] <= df["close"]
        df["below_area"] = np.where(df["below"], df["percent"], 0.0)

        grouped = df.groupby(["trade_date", "stock_code"], sort=False)
        below_sum = grouped["below_area"].sum()
        total_sum = grouped["percent"].sum()

        for (d, c), b, t in zip(
            below_sum.index, below_sum.values, total_sum.loc[below_sum.index].values
        ):
            key = (str(d), str(c))
            prev_b, prev_t = accum.get(key, (0.0, 0.0))
            accum[key] = (prev_b + b, prev_t + t)

        if on_progress:
            on_progress("chip_below", i + 1, n_groups)

    if not accum:
        return pd.Series(dtype=float, name="chip_below_ratio")

    records = [
        (d, c, below / total if total > 0 else np.nan)
        for (d, c), (below, total) in accum.items()
    ]
    result = pd.DataFrame(records, columns=["Date", "Code", "chip_below_ratio"])
    result = result.set_index(["Date", "Code"]).sort_index()
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    s = result["chip_below_ratio"]
    s = s.groupby(s.index.names).last()
    return s


@register_factor(
    name="chip_peak_ratio",
    description="下方筹码占比因子，当前价格以下的筹码面积占总筹码面积比例截面排名。",
    category="price",
    thesis="价格下方的筹码面积占比越高，意味着越多的持仓者处于盈利状态、且下方支撑越密集。适中的下方筹码占比（50-70%）是最优区间——既有支撑又未形成过重的获利抛压。该因子与筹码峰（最密集价格位置）互补：一个看筹码分布形状的'深度'，一个看'位置'。",
    dependencies=("cyq_chips.parquet", "daily_adj.parquet"),
)
def factor_chip_peak_ratio(context: FactorContext):
    source_path = str(context.repo.paths.source_root / "cyq_chips.parquet")
    daily_adj_path = str(context.repo.paths.source_root / "daily_adj.parquet")

    on_progress = context.repo.on_progress

    ratio_series = _compute_chip_below_ratio(
        source_path, daily_adj_path, context.repo.allowed_codes,
        on_progress=on_progress,
    )
    return cross_sectional_rank(ratio_series)


# ── Chip below momentum ────────────────────────────────────────────────


@register_factor(
    name="chip_below_momentum",
    description="下方筹码动量因子，chip_peak_ratio的5日变化截面排名。",
    category="price",
    thesis="下方筹码占比的快速增加意味着价格在向上突破——越来越多的筹码从上方套牢转为下方盈利。这种'支撑区间侵蚀'是上升趋势强度的领先指标：价格推升+下方筹码同步扩张=多方在持续消耗上方套牢盘，趋势基础稳固。",
    dependencies=("cyq_chips.parquet", "daily_adj.parquet"),
)
def factor_chip_below_momentum(context: FactorContext):
    source_path = str(context.repo.paths.source_root / "cyq_chips.parquet")
    daily_adj_path = str(context.repo.paths.source_root / "daily_adj.parquet")

    on_progress = context.repo.on_progress

    ratio_series = _compute_chip_below_ratio(
        source_path, daily_adj_path, context.repo.allowed_codes,
        on_progress=on_progress,
    )
    chg = ratio_series.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(chg)
