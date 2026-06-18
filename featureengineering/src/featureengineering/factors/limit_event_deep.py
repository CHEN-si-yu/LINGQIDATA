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
from ..utils import cross_sectional_rank


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
    name="limit_up_opening_strength",
    description="涨停开盘强度因子，开盘到涨停时间（秒）截面排名（取负向=慢封排后）。越早封板越强。",
    category="event",
    thesis="从开盘到触及涨停板的速度是涨停强度的最佳单一指标——集合竞价阶段即封板（一字板）最强，开盘30分钟内封板次之，尾盘封板最弱。封板速度反映了买盘的急切程度和实力。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_opening_strength(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    first_time = lu.get("first_limit_time", pd.Series(np.nan, index=lu.index))
    sec_from_open = _seconds_from_open(first_time)
    # Earlier = stronger, so invert: -seconds_from_open
    strength = -sec_from_open.fillna(_TRADING_SECONDS) / _TRADING_SECONDS  # -1 = instant, 0 = end of day

    strength = strength.reindex(daily_adj.index).fillna(0)
    strength = strength.groupby(level="Code").transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )

    return cross_sectional_rank(strength)


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


@register_factor(
    name="limit_event_momentum_break_5d",
    description="涨停动量打断因子，-(涨停前5日正动量+涨停后5日负动量)截面排名（趋势中断=不好排后）。",
    category="event",
    thesis="涨停打断了原有的价格趋势——如果涨停前在下跌而涨停后也未能延续，说明涨停是'一日游'行情。真正的趋势启动是涨停前有蓄势、涨停后能延续。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_event_momentum_break_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    is_limit_up = lu["is_limit_up"].reindex(ret.index).fillna(0).astype(float)

    # Pre-event momentum: returns from T-5 to T-1 (known at time T)
    pre_ret_5d = ret.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum().shift(1)
    )

    # Post-event: lag limit-up by 6 days, use the now-known 5d return (T-5..T-1)
    # as the post-event outcome for those historical events
    lu_lagged = is_limit_up.groupby(level="Code").shift(6).fillna(0)
    post_ret_5d = pre_ret_5d  # same window: 5-day return ending at T-1

    # Momentum break from historical events: positive pre but negative post
    break_signal = lu_lagged * (pre_ret_5d.clip(lower=0)) * (-post_ret_5d.clip(upper=0))

    break_20d = break_signal.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=10).mean()
    )

    return cross_sectional_rank(-break_20d)


# ═══════════════════════════════════════════════════════════════════════════════
# C — Cluster & Contagion Effects
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="limit_up_cluster_effect",
    description="涨停集群效应因子，同行业同日涨停股票数截面排名（板块共振=趋势确认排前）。",
    category="event",
    thesis="同一行业内多只股票同日涨停是板块行情的强烈确认——集群涨停意味着行业基本面或政策面出现了系统性催化剂，而非个股孤立事件。集群中的涨停股延续性更强。",
    dependencies=("limit_up.parquet", "stock_list.parquet"),
)
def factor_limit_up_cluster_effect(context: FactorContext):
    lu = _load_limit_up_panel(context)
    industry_map = context.repo.load_industry_map()

    is_lu = lu["is_limit_up"]
    codes = is_lu.index.get_level_values("Code")
    dates = is_lu.index.get_level_values("Date")
    industries = codes.map(industry_map)

    df = pd.DataFrame({"is_limit_up": is_lu.values, "industry": industries.values}, index=is_lu.index)
    df = df.dropna(subset=["industry"])

    cluster = df.groupby(["Date", "industry"])["is_limit_up"].transform("sum")
    return cross_sectional_rank(cluster)


@register_factor(
    name="limit_down_contagion_risk",
    description="跌停传染风险因子，同行业同日跌停股票数截面排名（跌停蔓延=系统性风险排后）。",
    category="event",
    thesis="跌停的行业传染效应在A股中十分显著——一只股票跌停（尤其是龙头）可能引发同行业其他股票的恐慌性抛售。跌停集群是行业层面的系统性风险信号。",
    dependencies=("limit_up.parquet", "stock_list.parquet"),
)
def factor_limit_down_contagion_risk(context: FactorContext):
    lu = _load_limit_up_panel(context)
    industry_map = context.repo.load_industry_map()

    # Detect limit down: is_limit_up=0 and stock is in limit_up table (might be limit down)
    is_ld = lu.get("is_limit_down", pd.Series(0, index=lu.index)).fillna(0)
    codes = is_ld.index.get_level_values("Code")
    industries = codes.map(industry_map)

    df = pd.DataFrame({"is_limit_down": is_ld.values, "industry": industries.values}, index=is_ld.index)
    df = df.dropna(subset=["industry"])

    contagion = df.groupby(["Date", "industry"])["is_limit_down"].transform("sum")
    return cross_sectional_rank(-contagion)


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
    name="limit_event_return_asymmetry",
    description="涨跌停收益不对称因子，涨停后5日正收益概率-跌停后5日正收益概率截面排名。",
    category="event",
    thesis="涨跌停后收益的不对称性反映股票的' resilience '（反弹能力）——涨停后继续涨+跌停后能反弹的股票具有最强的alpha特征。涨跌停只是短期冲击，不对称性度量了股票吸收冲击的能力。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_event_return_asymmetry(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # Lag events by 6 days so the 5d post-event window (T-5..T-1) has elapsed.
    # The now-known 5d return tells us whether past limit-ups/downs were followed
    # by positive returns — this historical asymmetry is the signal for today.
    post_ret_5d = ret.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum().shift(1)
    )

    is_lu = lu["is_limit_up"].reindex(ret.index).fillna(0).astype(float)
    is_ld = lu.get("is_limit_down", pd.Series(0, index=lu.index)).reindex(ret.index).fillna(0).astype(float)

    is_lu_lagged = is_lu.groupby(level="Code").shift(6).fillna(0)
    is_ld_lagged = is_ld.groupby(level="Code").shift(6).fillna(0)

    lu_win = (is_lu_lagged * (post_ret_5d > 0).astype(float))
    ld_win = (is_ld_lagged * (post_ret_5d > 0).astype(float))

    lu_win_60d = lu_win.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=10).sum())
    ld_win_60d = ld_win.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=10).sum())
    lu_count_60d = is_lu_lagged.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=10).sum())
    ld_count_60d = is_ld_lagged.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=10).sum())

    asym = lu_win_60d / lu_count_60d.replace(0, np.nan) - ld_win_60d / ld_count_60d.replace(0, np.nan)

    return cross_sectional_rank(asym)


# ═══════════════════════════════════════════════════════════════════════════════
# E — Gap Fill & Recovery
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="limit_gap_fill_probability",
    description="跳空回补概率因子，涨停跳空缺口在20日内被回补的天数占比截面排名（取负向=高回补=假突破排后）。",
    category="event",
    thesis="涨停形成的跳空缺口如果在短期内被回补，说明涨停价位不被市场认可——'缺口必补'的规律在A股中概率较高。未回补的缺口是强势的确认，是趋势延续的标志。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_gap_fill_probability(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    open_p = daily_adj["open"]
    close_p = daily_adj["close"]
    pre_close = close_p.groupby(level="Code").shift(1)

    # Gap up size
    gap = (open_p / pre_close.replace(0, np.nan)) - 1.0

    is_lu = lu["is_limit_up"].reindex(gap.index).fillna(0).astype(float)
    gap_on_lu = is_lu * gap.clip(lower=0)

    # Check if low in next 20 days fills the gap
    low = daily_adj["low"]
    gap_filled = (low.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=1).min()
    ) <= pre_close).astype(float)

    fill_prob = gap_filled.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=10).mean()
    )

    return cross_sectional_rank(-fill_prob)


@register_factor(
    name="limit_down_recovery_prob",
    description="跌停恢复概率因子，跌停后5日内回补跌幅超50%的概率截面排名（快速恢复=韧性排前）。",
    category="event",
    thesis="跌停后的恢复速度是股票质量的重要指标——优质公司在跌停后能快速反弹，因为基本面支撑使得跌停价成为'黄金坑'。恢复慢的股票说明跌停反映了真实的基本面问题。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_down_recovery_prob(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    is_ld = lu.get("is_limit_down", pd.Series(0, index=lu.index)).reindex(ret.index).fillna(0).astype(float)

    # Lag limit-down by 6 days so the 5d post-event window (T-5..T-1) has elapsed.
    # The now-known cumulative return tells us whether past limit-downs recovered.
    is_ld_lagged = is_ld.groupby(level="Code").shift(6).fillna(0)

    # 5-day cumulative return ending at T-1 (fully known at time T)
    cum_5d = ret.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum().shift(1)
    )

    recovered = (is_ld_lagged * (cum_5d > 0).astype(float))
    ld_count = is_ld_lagged.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=10).sum())

    recovery_prob = recovered.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=10).sum()
    ) / ld_count.replace(0, np.nan)

    return cross_sectional_rank(recovery_prob)


# ═══════════════════════════════════════════════════════════════════════════════
# F — Composite & Extended
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="limit_up_quality_composite",
    description="涨停质量综合评分因子，(封板速度+封单比+耐久度+连板数排名)/4截面排名。",
    category="event",
    thesis="综合多维度涨停质量评分的因子比单一维度更稳健——快速封板+大封单+持久封板+合适的连板数是高质量涨停的完整画像。用于筛选真正的强势涨停股。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_quality_composite(context: FactorContext):
    lu = _load_limit_up_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    # Speed component
    first_time = lu.get("first_limit_time", pd.Series(np.nan, index=lu.index))
    sec_from_open = _seconds_from_open(first_time)
    speed = -sec_from_open.fillna(_TRADING_SECONDS) / _TRADING_SECONDS

    # Seal ratio component
    sealed_amount = lu["sealed_amount"]
    amount = daily_adj["amount"].reindex(sealed_amount.index)
    seal_ratio = sealed_amount / amount.replace(0, np.nan)

    # Durability
    final_time = lu.get("final_limit_time", pd.Series(np.nan, index=lu.index))
    final_sec = _seconds_from_open(final_time)
    duration = (final_sec.fillna(0) - sec_from_open.fillna(_TRADING_SECONDS)).clip(lower=0) / _TRADING_SECONDS

    # Consecutive days (capped)
    consecutive = lu["consecutive_days"].reindex(daily_adj.index).fillna(0).clip(0, 7)

    # Combine
    composite = (
        speed.fillna(0).reindex(daily_adj.index).fillna(0) +
        seal_ratio.fillna(0).reindex(daily_adj.index).fillna(0) +
        duration.fillna(0).reindex(daily_adj.index).fillna(0) +
        consecutive.fillna(0) / 7.0
    ) / 4.0

    return cross_sectional_rank(composite)


@register_factor(
    name="limit_up_next_day_gap",
    description="涨停次日跳空因子，涨停次日开盘价/涨停价-1截面排名（高开=强势延续排前）。",
    category="event",
    thesis="涨停次日高开是涨停动量延续的最直接信号——高开幅度越大，说明隔夜买盘越强，涨停的动量延续概率越高。低开甚至平开的涨停次日大概率是假突破。",
    dependencies=("limit_up.parquet", "daily_adj.parquet"),
)
def factor_limit_up_next_day_gap(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    lu = _load_limit_up_panel(context)

    open_p = daily_adj["open"]
    close_p = daily_adj["close"]

    is_lu = lu["is_limit_up"].reindex(open_p.index).fillna(0).astype(float)

    # Lag limit-up by 2 days so the "next day" (=T-1) gap has fully materialised.
    # We measure the overnight gap that actually occurred after past limit-ups.
    is_lu_lagged = is_lu.groupby(level="Code").shift(2).fillna(0)

    # Overnight gap at T-1: (open at T-1 / close at T-2 - 1), both known at T
    prev_open = open_p.groupby(level="Code").shift(1)
    prev_close = close_p.groupby(level="Code").shift(2)
    overnight_gap = (prev_open / prev_close.replace(0, np.nan)) - 1.0

    next_day_gap = is_lu_lagged * overnight_gap

    next_day_gap = next_day_gap.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )

    return cross_sectional_rank(next_day_gap)


@register_factor(
    name="limit_seal_volume_ratio",
    description="封单量比因子，涨停封单量/自由流通股本截面排名（高封单比=买盘强大排前）。",
    category="event",
    thesis="封单量相对自由流通股本的比例是涨停买盘实力最直观的体现——封单占比高意味着有大量资金在涨停价排队买入，这些股票次日继续走强的概率显著更高。",
    dependencies=("limit_up.parquet", "finance.parquet"),
)
def factor_limit_seal_volume_ratio(context: FactorContext):
    lu = _load_limit_up_panel(context)
    finance = context.load("finance.parquet")

    sealed_vol = lu["sealed_volume"]
    float_share = finance["float_share"]

    common = sealed_vol.index.intersection(float_share.index)
    ratio = sealed_vol.loc[common] / float_share.loc[common].replace(0, np.nan)

    ratio = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).max()
    )

    return cross_sectional_rank(ratio)


@register_factor(
    name="limit_sector_leader_signal",
    description="板块龙头涨停信号因子，是否为同行业最早封板的股票截面排名（龙头=排前）。",
    category="event",
    thesis="同行业内最早封板的股票是市场公认的板块龙头——最早涨停意味着最强的买盘和最敏锐的资金嗅觉。龙头涨停股是板块行情的'发令枪'，后续涨幅空间最大。",
    dependencies=("limit_up.parquet", "stock_list.parquet"),
)
def factor_limit_sector_leader_signal(context: FactorContext):
    lu = _load_limit_up_panel(context)
    industry_map = context.repo.load_industry_map()

    first_time = lu.get("first_limit_time", pd.Series(np.nan, index=lu.index))
    first_sec = _seconds_from_open(first_time)
    is_lu = lu["is_limit_up"]

    codes = first_sec.index.get_level_values("Code")
    industries = codes.map(industry_map)

    df = pd.DataFrame({"first_time": first_sec.values, "industry": industries.values, "lu": is_lu.values}, index=first_sec.index)
    df = df.dropna(subset=["industry"])
    df = df[df["lu"] > 0]

    # Within each date+industry, find earliest (lowest first_time)
    df["earliest"] = df.groupby(["Date", "industry"])["first_time"].transform("min")
    df["is_leader"] = (df["first_time"] == df["earliest"]).astype(float)

    leader = df["is_leader"]
    return cross_sectional_rank(leader)
