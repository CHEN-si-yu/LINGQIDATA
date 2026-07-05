"""
Extended limit event factors — Class 1.

Additional limit-up / limit-down event factors using previously unused fields
from limit_up.parquet and limit_list.parquet.

The limit_event_deep.py module covers deep pattern analysis; unused_fields_factors.py
covers open_times, first_time, and turnover_ratio from limit_list.  This module
targets the remaining unused fields: seal quality, timing nuances, board type
signals, and aggregated counts.

Data sources:
- ``limit_up.parquet`` (daily event)
- ``limit_list.parquet`` (daily event)
- ``daily_adj.parquet`` (for panel alignment)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_sum,
    safe_divide,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pad_code(code: str) -> str:
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _parse_time_to_minutes(series: pd.Series) -> pd.Series:
    """Convert time strings (HH:MM:SS or HHMMSS) to minutes from midnight."""
    s = series.astype(str).str.strip().str.replace(":", "", regex=False)
    hh = pd.to_numeric(s.str[:2], errors="coerce")
    mm = pd.to_numeric(s.str[2:4], errors="coerce")
    ss = pd.to_numeric(s.str[4:6], errors="coerce").fillna(0)
    return hh * 60 + mm + ss / 60.0


def _load_limit_up_panel(context: FactorContext) -> pd.DataFrame:
    """Load limit_up.parquet as (Date, Code) MultiIndex panel."""
    cache = getattr(_load_limit_up_panel, "_cache", None)
    if cache is not None:
        return cache

    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "limit_up.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)
    raw = raw.set_index(["trade_date", "stock_code"])
    raw.index = raw.index.set_names(["Date", "Code"])
    raw = raw.reorder_levels(["Date", "Code"]).sort_index()

    _load_limit_up_panel._cache = raw
    return raw


@register_factor(
    name="ext_limit_seal_quality",
    description="封板质量因子，封单额/当日成交额截面排名（高值排前）。",
    category="event",
    thesis="封单额占当日成交额的比例反映涨停板的封板质量。高比率意味着"
    "大量的买单在涨停价排队等待成交，卖盘稀少，封板牢固。"
    "封板质量高的涨停板第二天继续上涨的概率更大。使用limit_list的fd_amount/amount。",
    dependencies=("limit_list.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_seal_quality(context: FactorContext):
    ll = context.load("limit_list.parquet")
    daily_adj = context.load("daily_adj.parquet")

    ratio = safe_divide(ll["fd_amount"], ll["amount"])
    ratio = ratio.reindex(daily_adj.index)

    ratio_ma = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return cross_sectional_rank(ratio_ma)


@register_factor(
    name="ext_limit_seal_flow_ratio",
    description="封单流通比因子，封单量/流通股数(sealed_flow_ratio)截面排名（高值排前）。",
    category="event",
    thesis="封单量占流通股数的比例（sealed_flow_ratio）从另一个维度衡量封板强度。"
    "该指标来自limit_up.parquet，与limit_list的fd_amount/amount互补。"
    "封单流通比高意味着有大量资金愿意在涨停价接盘。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_seal_flow_ratio(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    sfr = lu["sealed_flow_ratio"].reindex(daily_adj.index)

    sfr_ma = sfr.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return cross_sectional_rank(sfr_ma)


@register_factor(
    name="ext_limit_open_count_ratio",
    description="涨停开板率因子（低值排前），open_count/consecutive_days截面排名。",
    category="event",
    thesis="涨停期间日均开板次数反映封板的稳定性。开板次数多说明多空分歧大，"
    "封板不牢固。零开板（一字板）是最强的封板形态。"
    "open_count来自limit_up.parquet，consecutive_days来自同一数据源。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_open_count_ratio(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    days = lu["consecutive_days"].replace(0, np.nan)
    ratio = safe_divide(lu["open_count"], days)
    ratio = ratio.reindex(daily_adj.index)

    ratio_ma = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(-ratio_ma)


@register_factor(
    name="ext_limit_first_seal_speed",
    description="首次封板速度因子（快封排前），first_limit_time转换为距开盘分钟数。",
    category="event",
    thesis="首次封板时间越早，封板资金越坚决。开盘秒板（<5分钟）是极强的做多信号。"
    "尾盘拉板（>200分钟）往往是弱势封板或诱多陷阱。"
    "first_limit_time来自limit_up.parquet，与limit_list的first_time互补。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_first_seal_speed(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    minutes = _parse_time_to_minutes(lu["first_limit_time"])
    # Market opens at 9:30 = 570 min
    minutes_from_open = minutes - 570.0
    minutes_from_open = minutes_from_open.clip(lower=0)
    minutes_from_open = minutes_from_open.reindex(daily_adj.index)

    speed_ma = minutes_from_open.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(-speed_ma)  # faster (fewer minutes) = higher


@register_factor(
    name="ext_limit_final_seal_speed",
    description="最终封板时间因子（早封排前），final_limit_time距开盘分钟数。",
    category="event",
    thesis="最终封板时间可能与首次不同（开板后回封）。最终封板时间早说明"
    "全天大部分时间在封板状态，抛压小。最终封板时间晚（尾盘才最终封住）"
    "意味着多空博弈激烈，次日低开概率较高。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_final_seal_speed(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    minutes = _parse_time_to_minutes(lu["final_limit_time"])
    minutes_from_open = minutes - 570.0
    minutes_from_open = minutes_from_open.clip(lower=0)
    minutes_from_open = minutes_from_open.reindex(daily_adj.index)

    speed_ma = minutes_from_open.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(-speed_ma)


@register_factor(
    name="ext_limit_board_type_signal",
    description="涨停板型因子，首板排前（连板排后），boards字段编码。",
    category="event",
    thesis="首板涨停（涨停的第一天）与连板涨停（连续多日涨停）有不同的交易含义。"
    "首板后续溢价空间更大（风险收益比更优），而连板后期风险显著增加。"
    "boards字段直接标注了'首板涨停'等板型分类。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_board_type_signal(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    # boards contains labels like "首板涨停", "2连板", etc.
    # Map: first board = 3, consecutive = inverse of count
    is_first_board = lu["boards"].astype(str).str.contains("首板", na=False).astype(float)
    is_first_board = is_first_board.reindex(daily_adj.index).fillna(0)

    signal = is_first_board.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(signal)


@register_factor(
    name="ext_limit_type_quality",
    description="涨停类型质量因子，缩量涨停排前（放量涨停排后）。",
    category="event",
    thesis="缩量涨停意味着持筹者惜售、封板轻松，是强势涨停的特征。"
    "放量涨停意味着多空在涨停价换手充分，可能是主力在涨停板出货。"
    "limit_type字段直接标注了涨停类型（缩量涨停/放量涨停等）。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_type_quality(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    # Shrinking volume limit-up = highest quality
    is_shrinking = lu["limit_type"].astype(str).str.contains("缩量", na=False).astype(float)
    is_shrinking = is_shrinking.reindex(daily_adj.index).fillna(0)

    quality = is_shrinking.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(quality)


@register_factor(
    name="ext_limit_seal_duration",
    description="封板持续时间因子（长封排前），(final_limit_time - first_limit_time)截面排名。",
    category="event",
    thesis="首次封板到最终封板之间的时间窗口反映了封板的稳定性。"
    "首次和最终封板时间重合（一字板到收盘）是最强的形态。"
    "大时间差说明中途被打开过，封板不稳定。"
    "使用limit_up的first_limit_time和final_limit_time。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_seal_duration(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    first_min = _parse_time_to_minutes(lu["first_limit_time"])
    final_min = _parse_time_to_minutes(lu["final_limit_time"])

    # Duration = time between first and final seal (smaller = tighter seal)
    # If same → 0 duration → strongest
    duration = (final_min - first_min).abs()
    duration = duration.reindex(daily_adj.index)

    dur_ma = duration.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(-dur_ma)  # shorter duration = higher rank


@register_factor(
    name="ext_limit_up_count_20d",
    description="20日涨停次数因子，近20日涨停天数截面排名。",
    category="event",
    thesis="20日涨停次数是涨停活跃度的直接度量。涨停次数多说明个股处于强势期，"
    "动量效应明显。但极端高频（>5次/20日）可能面临监管问询和停牌风险。"
    "适中频率（1-3次/20日）的涨停动量最可持续。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_up_count_20d(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    is_limit = lu["is_limit_up"].reindex(daily_adj.index).fillna(0).astype(float)
    count_20d = is_limit.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    return cross_sectional_rank(count_20d)


@register_factor(
    name="ext_limit_consecutive_max_20d",
    description="20日最大连板天数因子，近20日内最大连续涨停天数截面排名。",
    category="event",
    thesis="最大连板天数反映个股在最近20日内的最强涨停动量。"
    "连板天数多是短线强势股的标志，但也意味着追高风险。"
    "连板天数从峰值回落可能是趋势减弱的信号。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_ext_limit_consecutive_max_20d(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    cd = lu["consecutive_days"].reindex(daily_adj.index).fillna(0)
    max_cd = cd.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).max()
    )
    return cross_sectional_rank(max_cd)
