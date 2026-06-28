"""
Extended dragon tiger board factors (扩展龙虎榜因子) — Class 1.

Deep analysis of dragon tiger board (龙虎榜) data: institution quality tracking,
pattern recognition, sentiment indices, spillover effects, crowded trade warnings,
and smart money tracking.

Data sources:
- ``dragon_tiger.parquet`` (daily, 2019-01-02 onward)
- ``top_list.parquet`` (daily, 2019-01-02 onward)
- ``daily_adj.parquet``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_sum

def _pad_code(code: str) -> str:
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)

def _load_dt_daily(context: FactorContext) -> pd.DataFrame:
    """Aggregate dragon_tiger daily data into a (Date, Code) panel."""
    cache = getattr(_load_dt_daily, "_cache", None)
    if cache is not None:
        return cache

    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "dragon_tiger.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)

    # Aggregate by date+stock
    agg = raw.groupby(["trade_date", "stock_code"]).agg(
        total_buy=("buy_amount", "sum"),
        total_sell=("sell_amount", "sum"),
        net_amount=("net_buy_amount", "sum"),
        org_count=("org_name", "nunique"),
        buy_orgs=("buy_amount", lambda x: (x > 0).sum()),
        sell_orgs=("sell_amount", lambda x: (x > 0).sum()),
        max_buy=("buy_amount", "max"),
        max_sell=("sell_amount", "max"),
    )
    agg.index = agg.index.set_names(["Date", "Code"])
    agg = agg.reorder_levels(["Date", "Code"]).sort_index()

    _load_dt_daily._cache = agg
    return agg

def _load_top_list_daily(context: FactorContext) -> pd.DataFrame:
    """Load top_list as (Date, Code) panel."""
    cache = getattr(_load_top_list_daily, "_cache", None)
    if cache is not None:
        return cache

    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "top_list.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)
    raw = raw.set_index(["trade_date", "stock_code"])
    raw.index = raw.index.set_names(["Date", "Code"])
    raw = raw.reorder_levels(["Date", "Code"]).sort_index()

    _load_top_list_daily._cache = raw
    return raw

# ═══════════════════════════════════════════════════════════════════════════════
# A — Institution Quality & Patterns
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="dt_appearance_frequency_20d",
    description="龙虎榜频率因子，20日龙虎榜出现次数截面排名（频繁上榜=高关注度排前）。",
    category="event",
    thesis="频繁上龙虎榜说明股票处于高关注度状态——交易活跃、资金进出量大。适度频率（2-5次/月）是健康的活跃度，过高频率（>10次/月）则可能是过度投机或出货。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_dt_appearance_frequency_20d(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    appeared = (dt["org_count"] > 0).astype(float)
    appeared = appeared.reindex(daily_adj.index).fillna(0)

    freq = appeared.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=1).sum()
    )

    return cross_sectional_rank(freq)

# ═══════════════════════════════════════════════════════════════════════════════
# B — Pattern Recognition: Momentum & Reversal
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="dt_pattern_momentum",
    description="龙虎榜动量模式因子，上榜日净买入+次日正收益截面排名（动量确认排前）。",
    category="event",
    thesis="龙虎榜净买入且次日继续上涨是标准的'机构建仓'模式——机构在买入后没有遭遇抛压，说明市场认可该价位，后续有望继续走强。净买入但次日下跌则可能是'接盘'模式。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_dt_pattern_momentum(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    net = dt["net_amount"].reindex(daily_adj.index).fillna(0)
    close = daily_adj["close"]

    # Lag DT net by 2 days so the "next day" (=T-1) return has fully materialised.
    net_lagged = net.groupby(level="Code").shift(2).fillna(0)

    # Yesterday's return (known at T) as the "next day" return for the lagged event
    known_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1).shift(1))

    net_rank = net_lagged.groupby(level="Date").rank(pct=True)
    ret_rank = known_ret.groupby(level="Date").rank(pct=True)

    momentum = net_rank * ret_rank

    momentum = momentum.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )

    return cross_sectional_rank(momentum)

@register_factor(
    name="dt_pattern_reversal",
    description="龙虎榜反转模式因子，-(上榜日净买入+次日负收益)截面排名（买入被套=短期压力排后）。",
    category="event",
    thesis="龙虎榜净买入但次日下跌意味着上榜资金被套——套牢的游资/机构可能在后续止损卖出，形成短期抛压。这是龙虎榜交易中需要警惕的'接盘侠'模式。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_dt_pattern_reversal(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    net = dt["net_amount"].reindex(daily_adj.index).fillna(0)
    close = daily_adj["close"]

    # Lag DT net by 2 days so the "next day" (=T-1) return has fully materialised.
    net_lagged = net.groupby(level="Code").shift(2).fillna(0)

    # Yesterday's return (known at T) as the "next day" return for the lagged event
    known_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1).shift(1))

    net_rank = net_lagged.groupby(level="Date").rank(pct=True)
    neg_ret_rank = (-known_ret).groupby(level="Date").rank(pct=True)

    reversal = net_rank * neg_ret_rank

    reversal = reversal.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )

    return cross_sectional_rank(-reversal)

# ═══════════════════════════════════════════════════════════════════════════════
# C — Top List Deep Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="top_list_repeat_appearance",
    description="连续上榜因子，连续出现在龙虎榜的天数截面排名（持续上榜=持续关注排前）。",
    category="event",
    thesis="连续多天上榜说明交易持续活跃——这种持续性是短线强势股的重要特征。连续上榜3天以上的股票通常处于主升浪或主跌浪中，趋势明确。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_top_list_repeat_appearance(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    appeared = (dt["org_count"].reindex(daily_adj.index).fillna(0) > 0).astype(float)

    # Count consecutive appearance days
    def _consecutive(s):
        consecutive = pd.Series(0, index=s.index)
        streak = 0
        for i, v in enumerate(s.values):
            if v > 0:
                streak += 1
            else:
                streak = 0
            consecutive.iloc[i] = streak
        return consecutive

    consecutive = appeared.groupby(level="Code").transform(_consecutive)

    return cross_sectional_rank(consecutive)

# ═══════════════════════════════════════════════════════════════════════════════
# D — Sentiment & Crowding
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="dt_crowded_trade_warning",
    description="龙虎榜拥挤预警因子，-(20日龙虎榜出现次数×平均净买入额)截面排名（过度关注=拥挤风险排后）。",
    category="event",
    thesis="过度出现在龙虎榜上且持续被净买入可能意味着'过度拥挤'——当太多资金已经进入后，后续增量买盘可能不足。龙虎榜拥挤预警是对冲动量反转风险的有效工具。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_dt_crowded_trade_warning(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    appeared = (dt["org_count"].reindex(daily_adj.index).fillna(0) > 0).astype(float)
    net = dt["net_amount"].reindex(daily_adj.index).fillna(0)

    freq_20d = appeared.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=1).sum())
    avg_net = net.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=5).mean())

    crowd = freq_20d * avg_net.abs()

    return cross_sectional_rank(-crowd)

# ═══════════════════════════════════════════════════════════════════════════════
# E — Spillover & Smart Money
# ═══════════════════════════════════════════════════════════════════════════════
@register_factor(
    name="dt_smart_money_tracking",
    description="龙虎榜聪明钱跟踪因子，过去净买入后5日正收益概率>50%时的当前净买入截面排名。",
    category="event",
    thesis="'聪明钱'的特征是其买入行为具有正向预测能力——跟踪历史上买入胜率高的龙虎榜信号，可以在聪明钱再次出手时跟随。这是龙虎榜信号的'质量过滤器'。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_dt_smart_money_tracking(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    net = dt["net_amount"].reindex(daily_adj.index).fillna(0)
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # Lag positive-net events by 6 days so the 5d post-event window has elapsed.
    # Compute historical win rate from past events whose outcomes are now known.
    is_positive = (net > 0).astype(float)
    is_positive_lagged = is_positive.groupby(level="Code").shift(6).fillna(0)

    # 5-day cumulative return ending at T-1 (fully known at time T)
    post_ret_5d = ret.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum().shift(1)
    )

    is_win = (is_positive_lagged * (post_ret_5d > 0).astype(float)).groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=10).sum()
    )
    is_count = is_positive_lagged.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=10).sum())

    win_rate = is_win / is_count.replace(0, np.nan)

    # Current positive net * historical win rate = smart money confidence
    smart_signal = (win_rate > 0.5).astype(float) * net

    return cross_sectional_rank(smart_signal)
@register_factor(
    name="dt_net_amount_persistence_10d",
    description="龙虎榜净买持续性因子，10日内净买入方向一致天数占比截面排名（持续净买=坚定看多排前）。",
    category="event",
    thesis="10日内净买入方向的一致性反映上榜资金的持续态度——10天中8天净买入比10天累计净买入额更能说明机构对该股的信心。一致净买入的信号强度远高于忽买忽卖。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_dt_net_amount_persistence_10d(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    net = dt["net_amount"].reindex(daily_adj.index).fillna(0)
    is_positive = (net > 0).astype(float)
    is_negative = (net < 0).astype(float)

    pos_10d = is_positive.groupby(level="Code").transform(lambda s: s.rolling(10, min_periods=1).sum())
    neg_10d = is_negative.groupby(level="Code").transform(lambda s: s.rolling(10, min_periods=1).sum())

    persistence = (pos_10d - neg_10d) / 10.0  # -1 to 1, higher = more persistent buying

    return cross_sectional_rank(persistence)

