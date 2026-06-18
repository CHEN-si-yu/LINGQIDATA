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
    name="dt_net_flow_trend_5d",
    description="龙虎榜净流趋势因子，5日累计净买入额截面排名（净买入持续=资金看好排前）。",
    category="event",
    thesis="龙虎榜净买入的5日累计额反映上榜后的资金持续态度——连续净买入比单日净买入更能代表机构对该股的信心。累计净买入大的股票中期趋势延续概率高。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_net_flow_trend_5d(context: FactorContext):
    dt = _load_dt_daily(context)
    net = dt["net_amount"]
    net_5d = rolling_group_sum(net, 5)
    return cross_sectional_rank(net_5d)


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


@register_factor(
    name="dt_institution_consistency",
    description="机构操作一致性因子，同一龙虎榜买卖方向相同机构占比截面排名（一致=确定性高排前）。",
    category="event",
    thesis="龙虎榜上机构操作方向的一致性反映了专业资金的共识程度——买入席位全是净买入（无卖出）比买卖参半更有说服力。高一致性的龙虎榜买卖信号更可靠。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_institution_consistency(context: FactorContext):
    dt = _load_dt_daily(context)

    # Consistency = |buy_orgs - sell_orgs| / total_orgs
    consistency = (dt["buy_orgs"] - dt["sell_orgs"]).abs() / dt["org_count"].replace(0, np.nan)

    return cross_sectional_rank(consistency)


@register_factor(
    name="dt_institution_size_rank",
    description="机构买入规模因子，最大单笔买入金额截面排名（大买单=机构参与深排前）。",
    category="event",
    thesis="龙虎榜上最大单笔买入金额反映了参与机构/游资的资金体量——亿元级别的大买单通常来自顶级游资或机构席位，其研究能力和市场影响力更强。大买单上榜的股票后续表现系统性地优于小买单上榜股。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_institution_size_rank(context: FactorContext):
    dt = _load_dt_daily(context)
    max_buy = dt["max_buy"]

    max_buy_5d = max_buy.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).max()
    )

    return cross_sectional_rank(max_buy_5d)


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


@register_factor(
    name="dt_reversal_prob_5d",
    description="龙虎榜反转概率因子，-(上榜后5日正收益概率)截面排名（低反转=动量可靠排前）。",
    category="event",
    thesis="上榜后5日正收益的概率反映了龙虎榜信号的可靠性——概率>60%的股票龙虎榜信号值得跟踪，概率<40%的则信号可能是噪音或反向指标。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_dt_reversal_prob_5d(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # Lag DT appearance by 6 days so the 5d post-event window (T-5..T-1) has elapsed.
    # The now-known 5d return after past DT appearances determines the signal.
    appeared = (dt["org_count"].reindex(ret.index).fillna(0) > 0).astype(float)
    appeared_lagged = appeared.groupby(level="Code").shift(6).fillna(0)

    # 5-day cumulative return ending at T-1 (fully known at time T)
    post_ret_5d = ret.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum().shift(1)
    )

    win = appeared_lagged * (post_ret_5d > 0).astype(float)

    win_sum = win.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=10).sum())
    appear_sum = appeared_lagged.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=10).sum())

    prob = win_sum / appear_sum.replace(0, np.nan)

    return cross_sectional_rank(-prob)


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


@register_factor(
    name="top_list_amount_momentum",
    description="龙虎榜成交额动量因子，上榜日成交额5日变化率截面排名（成交放大=活跃度升排前）。",
    category="event",
    thesis="上榜成交额的边际变化比绝对成交额更有意义——成交额持续放大的龙虎榜股票处于'升温'阶段，成交额萎缩则意味着关注度在下降。成交额动量是龙虎榜信号的'加速度'指标。",
    dependencies=("top_list.parquet",)
)
def factor_top_list_amount_momentum(context: FactorContext):
    tl = _load_top_list_daily(context)
    amount = tl["amount"]

    chg = amount.groupby(level="Code").transform(lambda s: s.pct_change(5))

    return cross_sectional_rank(chg)


@register_factor(
    name="top_list_inst_dominance_change",
    description="机构主导度变化因子，龙虎榜净买入/成交额5日变化截面排名（机构占比提升排前）。",
    category="event",
    thesis="机构净买入在成交额中的占比变化反映机构参与度的升降——占比上升意味着机构正在加大对该股的配置，是中长期看好的信号。机构主导度的提升往往领先于股价上涨。",
    dependencies=("dragon_tiger.parquet", "top_list.parquet"),
)
def factor_top_list_inst_dominance_change(context: FactorContext):
    dt = _load_dt_daily(context)
    tl = _load_top_list_daily(context)

    net = dt["net_amount"]
    amount = tl["amount"]

    common = net.index.intersection(amount.index)
    dominance = net.loc[common] / amount.loc[common].replace(0, np.nan).abs()
    dom_chg = dominance.groupby(level="Code").transform(lambda s: s.diff(5))

    return cross_sectional_rank(dom_chg)


# ═══════════════════════════════════════════════════════════════════════════════
# D — Sentiment & Crowding
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="dt_sentiment_index",
    description="龙虎榜情绪指数因子，(净买入额/成交额+买入机构数/卖出机构数)/2截面排名。",
    category="event",
    thesis="龙虎榜情绪综合了资金流向和机构参与两个维度——净买入大+买入机构多是积极的龙虎榜信号。情绪指数是短线交易中判断龙虎榜'含金量'的快捷指标。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_sentiment_index(context: FactorContext):
    dt = _load_dt_daily(context)

    net = dt["net_amount"]
    total = dt["total_buy"] + dt["total_sell"]

    net_ratio = net / total.replace(0, np.nan)
    buy_sell_org_ratio = dt["buy_orgs"] / dt["sell_orgs"].replace(0, np.nan)

    sentiment = (net_ratio.fillna(0) + buy_sell_org_ratio.fillna(1) / buy_sell_org_ratio.fillna(1).max()) / 2.0

    return cross_sectional_rank(sentiment)


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


@register_factor(
    name="dt_buy_sell_concentration",
    description="买卖集中度因子，最大单笔买入/最大单笔卖出截面排名（买入集中=机构主导排前）。",
    category="event",
    thesis="最大买单与最大卖单的比值反映了龙虎榜上多空双方的实力对比——比值>2意味着买方中有一家实力远超卖方任何一家，这种'主力碾压'式的龙虎榜后续上涨概率高。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_buy_sell_concentration(context: FactorContext):
    dt = _load_dt_daily(context)

    concentration = dt["max_buy"] / dt["max_sell"].replace(0, np.nan)

    concentration = concentration.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    return cross_sectional_rank(concentration)


# ═══════════════════════════════════════════════════════════════════════════════
# E — Spillover & Smart Money
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="dt_sector_spillover",
    description="龙虎榜板块溢出因子，同行业有龙虎榜净买入的个股次日超额收益截面排名。",
    category="event",
    thesis="龙虎榜的板块溢出效应与涨停集群效应类似——行业内一只股票被机构大幅净买入，会引发市场对同行业其他股票的关注和跟风买入。龙虎榜溢出效应是行业轮动的重要微观驱动。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet", "stock_list.parquet"),
)
def factor_dt_sector_spillover(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()

    net = dt["net_amount"]
    close = daily_adj["close"]

    # Use T-1 return (known at T) and lag sector DT signal by 2 days.
    known_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1).shift(1))

    # Lag net signal: event at T-2, "next day" (=T-1) return is now known
    net_lagged = net.groupby(level="Code").shift(2)

    codes = net_lagged.index.get_level_values("Code")
    industries = codes.map(industry_map)

    df = pd.DataFrame({"net": net_lagged.values, "ret": known_ret.reindex(net_lagged.index).values,
                       "industry": industries.values}, index=net_lagged.index)
    df = df.dropna(subset=["industry"])
    df["has_positive_dt"] = (df["net"] > 0).astype(float)

    # Spillover: if a stock in the industry had positive DT 2 days ago,
    # the now-known next-day return of other stocks represents the spillover
    df["sector_has_dt"] = df.groupby(["Date", "industry"])["has_positive_dt"].transform("max")
    df["spillover"] = df["sector_has_dt"] * df["ret"]

    # Smooth with rolling mean
    spillover_smoothed = df["spillover"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )

    return cross_sectional_rank(spillover_smoothed)


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
    name="dt_org_type_diversity",
    description="机构多样性因子，龙虎榜参与席位数量截面排名（多机构参与=共识强排前）。",
    category="event",
    thesis="参与龙虎榜的机构数量反映了市场对该股的关注广度——单家机构独买可能是'独庄'行为，多家机构共同买入则是市场共识的体现。席位多样性越高，龙虎榜信号越可靠。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_org_type_diversity(context: FactorContext):
    dt = _load_dt_daily(context)
    diversity = dt["org_count"]

    diversity_5d = diversity.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).max()
    )

    return cross_sectional_rank(diversity_5d)


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


@register_factor(
    name="dt_buy_sell_asymmetry",
    description="龙虎榜买卖不对称因子，(总买入-总卖出)/(总买入+总卖出)截面排名（正=净买方主导排前）。",
    category="event",
    thesis="买方和卖方金额的不对称性是龙虎榜最基础也最有效的信号——净买入占比高的龙虎榜代表买方力量明显强于卖方。这个简单的指标在A股龙虎榜策略中长期有效。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_buy_sell_asymmetry(context: FactorContext):
    dt = _load_dt_daily(context)

    asymmetry = (dt["total_buy"] - dt["total_sell"]) / (dt["total_buy"] + dt["total_sell"]).replace(0, np.nan)

    asymmetry = asymmetry.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    return cross_sectional_rank(asymmetry)


@register_factor(
    name="dt_sector_leader_confirmation",
    description="龙虎榜龙头确认因子，板块涨幅前20%+龙虎榜净买入截面排名（板块领涨+机构买入双确认排前）。",
    category="event",
    thesis="在板块涨幅领先的股票中，龙虎榜净买入是板块龙头地位的'机构认证'——板块涨+机构买意味着该股不仅走势强，而且有专业资金背书。双确认信号的胜率显著高于单信号。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet", "stock_list.parquet"),
)
def factor_dt_sector_leader_confirmation(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()

    close = daily_adj["close"]
    ret_5d = close.groupby(level="Code").transform(lambda s: s.pct_change(5))
    net = dt["net_amount"]

    codes = ret_5d.index.get_level_values("Code")
    industries = codes.map(industry_map)

    df_ret = pd.DataFrame({"ret_5d": ret_5d.values, "industry": industries.values}, index=ret_5d.index)
    df_ret = df_ret.dropna(subset=["industry"])
    df_ret["sector_rank"] = df_ret.groupby(["Date", "industry"])["ret_5d"].rank(pct=True)

    is_leader = (df_ret["sector_rank"] > 0.8).astype(float)

    common = net.index.intersection(is_leader.index)
    leader = is_leader.loc[common]
    net_aligned = net.loc[common]

    net_rank = net_aligned.groupby(level="Date").rank(pct=True)

    confirmation = leader * net_rank

    return cross_sectional_rank(confirmation)
