"""
Extended limit event factors (扩展涨跌停事件因子) — Class 1.

These factors provide deep analysis of limit-up/limit-down events: seal quality,
momentum continuation/reversal, cluster effects, contagion risk, recovery
patterns, and composite quality scoring.

Data sources:
- ``limit_up.parquet`` (daily, 2019-01-02 onward)
- ``limit_list.parquet`` (daily, 2020-01-02 onward — acceptable per constraints)
- ``daily_adj.parquet``
- ``ths_constituent_stocks.parquet``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_code(code: str) -> str:
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)

def _parse_time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a Series of time values (string or numeric) to seconds since 00:00.

    Handles formats like ``"09:30:00"``, ``"093000"``, ``"09:30"``, or
    numeric seconds-since-midnight.  Returns float (seconds from midnight).
    """
    s = series.astype(str).str.strip().str.replace(":", "", regex=False)
    # Parse HHMMSS or HHMM
    hh = pd.to_numeric(s.str[:2], errors="coerce")
    mm = pd.to_numeric(s.str[2:4], errors="coerce")
    ss = pd.to_numeric(s.str[4:6], errors="coerce").fillna(0)
    return hh * 3600 + mm * 60 + ss

_MARKET_OPEN_SEC = 9 * 3600 + 30 * 60  # 34200 = 09:30
_MARKET_CLOSE_SEC = 15 * 3600          # 54000 = 15:00
_TRADING_SECONDS = _MARKET_CLOSE_SEC - _MARKET_OPEN_SEC  # 19800

def _seconds_from_open(series: pd.Series) -> pd.Series:
    """Convert time series to seconds from market open (09:30), clipped to [0, 19800]."""
    seconds = _parse_time_to_seconds(series)
    return (seconds - _MARKET_OPEN_SEC).clip(0, _TRADING_SECONDS)

def _load_limit_up_panel(context: FactorContext) -> pd.DataFrame:
    """Load limit_up.parquet as (Date, Code) MultiIndex panel with key fields."""
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

def _build_limit_event_daily(context: FactorContext) -> pd.DataFrame:
    """Build a daily (Date, Code) panel of aggregated limit event features.

    Merges limit_up and limit_list data, forward-filling event features.
    """
    cache = getattr(_build_limit_event_daily, "_cache", None)
    if cache is not None:
        return cache

    daily_adj = context.load("daily_adj.parquet")
    ref_idx = daily_adj.index  # canonical (Date, Code) MultiIndex

    lu = _load_limit_up_panel(context)

    # Key event features to extract
    features = pd.DataFrame(index=ref_idx)

    # Consecutive limit-up days (0 if not limit-up today)
    features["consecutive_days"] = lu["consecutive_days"].reindex(ref_idx).fillna(0)
    features["sealed_volume"] = lu["sealed_volume"].reindex(ref_idx)
    features["sealed_amount"] = lu["sealed_amount"].reindex(ref_idx)
    features["sealed_turnover_ratio"] = lu["sealed_turnover_ratio"].reindex(ref_idx)
    features["open_count"] = lu["open_count"].reindex(ref_idx).fillna(0)
    features["is_limit_up"] = lu["is_limit_up"].reindex(ref_idx).fillna(0).astype(float)
    features["is_limit_down"] = ((lu.get("limit_type") == "D") | (lu.get("limit_type") == "跌停")).reindex(ref_idx).fillna(0).astype(float)

    for col in features.columns:
        features[col] = features[col].fillna(0 if col in ("consecutive_days", "open_count", "is_limit_up", "is_limit_down") else np.nan)

    _build_limit_event_daily._cache = features
    return features

# ═══════════════════════════════════════════════════════════════════════════════
# A — Seal Quality & Limit-Up Strength
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="limit_up_seal_quality",
    description="封板质量因子，(封单额/成交额+封板时间占比+1/(开板次数+1))/3截面排名（封板质量高排前）。",
    category="event",
    thesis="涨停封板质量是判断涨停是否'真涨停'的核心——大封单、快速封板、零开板是高质量涨停的三要素。高质量涨停次日高开概率大，低质量涨停（频繁开板、尾盘封板）则大概率次日回落。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_seal_quality(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    # Seal quality components
    sealed_amount = lu["sealed_amount"]
    amount = daily_adj["amount"].reindex(sealed_amount.index)
    seal_ratio = sealed_amount / amount.replace(0, np.nan)

    # Time to seal: seconds from market open, earlier = better
    first_time = lu.get("first_limit_time", pd.Series(np.nan, index=lu.index))
    sec_from_open = _seconds_from_open(first_time)
    # Normalize: 0=instant, _TRADING_SECONDS=end of day. 1 - time/total = earlier is higher
    time_score = 1.0 - sec_from_open.fillna(_TRADING_SECONDS) / _TRADING_SECONDS

    # Open count: fewer openings = better seal
    open_count = lu.get("open_count", pd.Series(0, index=lu.index))
    stability = 1.0 / (open_count + 1.0)

    quality = (seal_ratio.fillna(0) + time_score.fillna(0) + stability.fillna(0)) / 3.0

    # Forward-fill signal for 3 days after event
    quality = quality.reindex(daily_adj.index).fillna(0)
    quality = quality.groupby(level="Code").transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )

    return cross_sectional_rank(quality)

@register_factor(
    name="limit_seal_durability_rank",
    description="封板耐久性因子，最终封板时间/全天交易时间截面排名（取负向=尾盘封板=不可靠排后）。",
    category="event",
    thesis="封板时间越早，封板越可靠——早盘封板的涨停股次日延续涨停概率远高于尾盘封板股。尾盘封板往往是短线资金'做收盘'的行为，缺乏真实买盘支撑。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_seal_durability_rank(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    final_time = lu.get("final_limit_time", pd.Series(np.nan, index=lu.index))
    first_time = lu.get("first_limit_time", pd.Series(np.nan, index=lu.index))

    # Convert to seconds from market open
    final_sec = _seconds_from_open(final_time)
    first_sec = _seconds_from_open(first_time)

    # Duration = final_time - first_time (how long it stayed sealed)
    duration = final_sec.fillna(0) - first_sec.fillna(_TRADING_SECONDS)
    durability = duration.clip(lower=0) / _TRADING_SECONDS

    durability = durability.reindex(daily_adj.index).fillna(0)
    durability = durability.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).max()
    )

    return cross_sectional_rank(durability)

@register_factor(
    name="limit_break_count_20d",
    description="开板次数因子，20日盘中开板累计次数截面排名（取负向=频繁开板=封板无力排后）。",
    category="event",
    thesis="涨停板反复开板是封板力量不足的表现——每次开板都意味着有大量卖盘在涨停价出货。20日累计开板次数高的股票封板质量系统性偏低，后续动量延续性差。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_break_count_20d(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    open_count = lu.get("open_count", pd.Series(0, index=lu.index)).fillna(0)
    open_count = open_count.reindex(daily_adj.index).fillna(0)

    break_20d = open_count.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=1).sum()
    )

    return cross_sectional_rank(-break_20d)

# ═══════════════════════════════════════════════════════════════════════════════
# B — Limit-Up Momentum & Reversal
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="limit_up_premium_5d",
    description="涨停溢价因子，涨停日后5日超额收益截面排名（涨停动量延续=高溢价排前）。",
    category="event",
    thesis="涨停后的短期溢价效应在A股中显著存在——高质量涨停之后5日平均仍有正超额收益，尤其是在板块风口上的涨停。但低质量涨停后的溢价迅速衰减甚至转负。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_premium_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    is_limit_up = lu["is_limit_up"].reindex(ret.index).fillna(0).astype(float)

    # Lag limit-up by 6 days so the 5d post-event window (T-5..T-1) has elapsed.
    # The now-known 5d return after those historical events is the premium signal.
    lu_lagged = is_limit_up.groupby(level="Code").shift(6).fillna(0)

    # 5-day return ending at T-1 (fully known at time T)
    post_ret_5d = ret.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum().shift(1)
    )

    premium = lu_lagged * post_ret_5d
    premium = premium.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=10).mean()
    )

    return cross_sectional_rank(premium)

@register_factor(
    name="limit_up_reversal_risk",
    description="涨停反转风险因子，涨停次日低开概率×低开幅度截面排名（取负向=反转风险高排后）。",
    category="event",
    thesis="涨停次日低开是'假突破'的典型特征——追涨停的资金被套，形成短期套牢盘。涨停反转风险高的股票说明涨停的质量差，市场不认可涨停价位。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_reversal_risk(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    close = daily_adj["close"]
    open_price = daily_adj["open"]
    ret_overnight = (open_price / close.groupby(level="Code").shift(1).replace(0, np.nan)) - 1.0

    is_limit_up = lu["is_limit_up"].reindex(ret_overnight.index).fillna(0).astype(float)

    # Reversal = negative overnight return after limit-up
    reversal = is_limit_up * (-ret_overnight).clip(lower=0)
    reversal_5d = reversal.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    return cross_sectional_rank(-reversal_5d)

@register_factor(
    name="limit_up_momentum_chain",
    description="连板延续因子，涨停日连续涨停天数×次日涨停概率截面排名（连板惯性排前）。",
    category="event",
    thesis="连板股具有显著的'惯性效应'——连续涨停的天数越多（在一定范围内），次日继续涨停的概率也越高，因为市场关注度和情绪在加强。但超过5连板后监管干预风险上升。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_momentum_chain(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    consecutive = lu["consecutive_days"].reindex(daily_adj.index).fillna(0)
    # Cap at 7+ consecutive days (regulatory risk)
    chain = consecutive.clip(0, 7)

    return cross_sectional_rank(chain)

# ═══════════════════════════════════════════════════════════════════════════════
# C — Cluster & Contagion Effects
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="limit_sector_momentum_spillover",
    description="涨停板块溢出因子，同板块有涨停的个股次日超额收益截面排名（溢出受益排前）。",
    category="event",
    thesis="板块内有股票涨停会对同板块其他股票产生正向溢出——资金在追逐龙头的同时会扩散到板块内其他标的，形成板块联动。涨停溢出效应是短线交易中重要的alpha来源。",
    dependencies=("limit_up.parquet", "daily_adj.parquet", "stock_list.parquet"),
)
def factor_limit_sector_momentum_spillover(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()

    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # Use T-1 return (known at T) instead of T+1 return (unknown at T).
    # Lag the sector limit-up signal by 2 days so the "next day" return has elapsed.
    known_ret = ret.groupby(level="Code").shift(1)

    is_lu = lu["is_limit_up"].reindex(ret.index).fillna(0)
    # Lag is_lu by 2 days: event at T-2, "next day" (=T-1) return is now known
    is_lu_lagged = is_lu.groupby(level="Code").shift(2).fillna(0)

    codes = is_lu_lagged.index.get_level_values("Code")
    industries = codes.map(industry_map)

    df = pd.DataFrame({"lu": is_lu_lagged.values, "ret": known_ret.values, "industry": industries.values}, index=ret.index)
    df = df.dropna(subset=["industry"])

    # For each date+industry, if there WAS a limit-up 2 days ago, the now-known
    # next-day return of non-LU stocks represents the spillover effect.
    df["has_lu"] = df.groupby(["Date", "industry"])["lu"].transform("max")
    df["spillover"] = df["has_lu"] * (1 - df["lu"]) * df["ret"]

    # Smooth with rolling mean to capture persistent spillover patterns
    spillover_smoothed = df["spillover"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )

    return cross_sectional_rank(spillover_smoothed)

# ═══════════════════════════════════════════════════════════════════════════════
# D — Event Frequency & Volatility
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="limit_event_frequency_20d",
    description="涨跌停频率因子，20日涨跌停事件总次数截面排名（取负向=频繁触板=不稳定排后）。",
    category="event",
    thesis="频繁触发涨跌停的股票波动率极高、投机性强——适合短线交易但不适合中长线持有。涨跌停频率是'投机热度'的量化指标，高频触板股票的交易成本（滑点、流动性）也更高。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_event_frequency_20d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    is_event = (lu["is_limit_up"].reindex(daily_adj.index).fillna(0) > 0).astype(float)
    freq = is_event.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=1).sum()
    )

    return cross_sectional_rank(-freq)

@register_factor(
    name="limit_event_volatility",
    description="涨跌停波动率因子，涨跌停事件前后5日波动率比值截面排名（取负向=波动放大=不稳定排后）。",
    category="event",
    thesis="涨跌停事件前后的波动率变化反映了事件对价格稳定性的冲击——波动率显著放大说明事件引发了市场分歧，后续走势不确定。波动率在事件后收敛则说明市场对定价形成了共识。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_event_volatility(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol = ret.groupby(level="Code").transform(lambda s: s.rolling(5, min_periods=3).std())

    # Vol before event (T-10 to T-6) vs current vol (T-4 to T, both known at T)
    vol_pre = vol.groupby(level="Code").shift(10)
    vol_current = vol

    is_event = (lu["is_limit_up"].reindex(vol.index).fillna(0) > 0).astype(float)

    # Ratio of current vol to pre-event vol — elevated vol around event = risk
    vol_ratio = vol_current / vol_pre.replace(0, np.nan)
    vol_change = is_event * vol_ratio

    vol_change = vol_change.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )

    return cross_sectional_rank(-vol_change)
@register_factor(
    name="limit_up_next_day_gap",
    description="涨停次日跳空因子，涨停次日开盘价/涨停价-1截面排名（高开=强势延续排前）。",
    category="event",
    thesis="涨停次日高开是涨停动量延续的最直接信号——高开幅度越大，说明隔夜买盘越强，涨停的动量延续概率越高。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_next_day_gap(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    open_ = daily_adj["open"]
    is_lu = lu["is_limit_up"].reindex(close.index).fillna(0).astype(float)
    is_lu_yesterday = is_lu.groupby(level="Code").shift(1).fillna(0)
    yesterday_close = close.groupby(level="Code").shift(1)
    gap = safe_divide(open_ - yesterday_close, yesterday_close.abs() + 1e-8)
    next_day_gap = gap.where(is_lu_yesterday > 0, np.nan)
    next_day_gap = next_day_gap.groupby(level="Code").transform(lambda s: s.ffill(limit=1))
    return cross_sectional_rank(next_day_gap)


@register_factor(
    name="limit_up_quality_composite",
    description="涨停质量综合评分因子，(封板速度+封单比+耐久度+连板数排名)/4截面排名。",
    category="event",
    thesis="综合多维度涨停质量评分的因子比单一维度更稳健——快速封板+大封单+持久封板+合适的连板数是高质量涨停的完整画像。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_quality_composite(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    first_time = pd.to_datetime(limit_up["first_limit_time"], format="%H%M%S", errors="coerce")
    minutes_from_open = (first_time - first_time.dt.normalize() - pd.Timedelta(hours=9, minutes=30)).dt.total_seconds() / 60
    seal_speed_rank = cross_sectional_rank(-minutes_from_open.fillna(480))
    seal_ratio = safe_divide(limit_up["sealed_turnover_ratio"], 1.0)
    seal_ratio_rank = cross_sectional_rank(seal_ratio)
    consecutive_days = limit_up.get("consecutive_days", pd.Series(1, index=limit_up.index))
    consecutive_rank = cross_sectional_rank(consecutive_days.clip(upper=8))
    last_time = pd.to_datetime(limit_up["final_limit_time"], format="%H%M%S", errors="coerce")
    duration = (last_time - first_time).dt.total_seconds() / 60
    duration_rank = cross_sectional_rank(duration.fillna(0))
    composite = (seal_speed_rank.fillna(0) + seal_ratio_rank.fillna(0) +
                 consecutive_rank.fillna(0) + duration_rank.fillna(0)) / 4.0
    return cross_sectional_rank(composite)

