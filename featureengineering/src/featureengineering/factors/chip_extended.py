"""
Extended chip distribution factors (筹码成本分布因子) — Class 1.

基于 cyq_perf.parquet 面板数据的筹码成本分布因子。
利用成本分位数(cost_5pct~cost_95pct)、加权均价(weight_avg)、获利盘比例(winner_rate)
构建筹码形态、趋势和压力/支撑因子。

数据源: cyq_perf.parquet
字段: cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct, weight_avg, winner_rate
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide


# ── 筹码离散度与形态 ─────────────────────────────────────────────────────

@register_factor(
    name="chip_dispersion_width",
    description="筹码成本离散度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（取负向=宽散=分歧大排后）。",
    category="coupling",
    thesis="筹码成本分布的宽度反映市场参与者的分歧程度——成本高度集中（窄）意味着持仓者成本接近、形成共识，容易形成趋势；成本分散（宽）意味着多空分歧大、方向不明。窄幅集中的股票突破方向更确定。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_dispersion_width(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    width = (cyq["cost_95pct"] - cyq["cost_5pct"]) / cyq["cost_50pct"].replace(0, np.nan)
    return cross_sectional_rank(-width)


@register_factor(
    name="chip_skewness_ratio",
    description="筹码成本偏度因子，(cost_50pct-cost_5pct)/(cost_95pct-cost_50pct)截面排名（>1=低位密集看涨排前）。",
    category="coupling",
    thesis="筹码偏度是判断筹码在'低位密集'还是'高位密集'的核心指标——>1意味着中位数更靠近下沿（低位密集，看涨），<1意味着中位数更靠近上沿（高位密集，看跌）。这是筹码分布中最有预测力的单一形态指标。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_skewness_ratio(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    lower_half = cyq["cost_50pct"] - cyq["cost_5pct"]
    upper_half = cyq["cost_95pct"] - cyq["cost_50pct"]
    skew = safe_divide(lower_half, upper_half)
    return cross_sectional_rank(skew)


@register_factor(
    name="chip_concentration_zone",
    description="筹码集中区位因子，-(|winner_rate-0.5|)截面排名（获利盘50%=多空平衡=方向将出排前）。",
    category="coupling",
    thesis="获利盘比例在50%附近是多空力量最平衡的状态——平衡即将被打破，方向选择在即。获利盘>80%（超买）或<20%（超卖）则趋势可能耗尽。50%±15%是最佳'突破前夜'区间。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_concentration_zone(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    wr = cyq["winner_rate"]
    # Distance from 0.5, negative = closer to balance
    zone = -np.abs(wr - 0.5)
    return cross_sectional_rank(zone)


# ── 筹码趋势 ─────────────────────────────────────────────────────────────

@register_factor(
    name="chip_cost_momentum_20d",
    description="筹码成本重心趋势因子，weight_avg的20日变化率截面排名（成本上移=资金抬轿排前）。",
    category="coupling",
    thesis="筹码加权平均成本的变化方向反映资金流向——成本重心持续上移说明新增资金愿意以更高价格买入（看涨），成本重心下移则说明持仓者在不断降低预期。成本趋势是持仓者集体智慧的表达。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_cost_momentum_20d(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    wa = cyq["weight_avg"]
    mom = wa.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    return cross_sectional_rank(mom)


@register_factor(
    name="winner_rate_momentum_5d",
    description="获利盘变化率因子，winner_rate的5日变化截面排名（获利盘快速增加=短期过热排后）。",
    category="coupling",
    thesis="获利盘比例的短期变化速度是超买超卖的灵敏指标——5日内获利盘从30%飙升至70%意味着大量持仓者快速获利，短期获利了结压力上升（负向信号）。反之获利盘从70%跌至30%则恐慌盘出清（正向信号）。",
    dependencies=("cyq_perf.parquet",),
)
def factor_winner_rate_momentum_5d(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    wr = cyq["winner_rate"]
    chg = wr.groupby(level="Code").transform(
        lambda s: s.diff(5)
    )
    return cross_sectional_rank(-chg)


@register_factor(
    name="chip_peak_shift",
    description="筹码峰移动因子，cost_50pct的20日变化率截面排名（中位数成本上移=看涨排前）。",
    category="coupling",
    thesis="筹码中位数成本（50分位）的位置变化反映筹码峰在向哪个方向移动——中位数成本上升意味着市场整体成本在抬升、接盘力量充足；中位数成本下降则说明支撑在走弱。与cost_momentum互补：一个看均值移动，一个看中位数移动。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_peak_shift(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    c50 = cyq["cost_50pct"]
    shift = c50.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    return cross_sectional_rank(shift)


# ── 筹码压力/支撑 ─────────────────────────────────────────────────────────

@register_factor(
    name="chip_above_below_ratio",
    description="筹码压力比因子，(price-cost_85pct)/(cost_15pct-price)截面排名（上方套牢>下方获利=压力大排后）。",
    category="coupling",
    thesis="上方套牢盘与下方获利盘的比例关系衡量筹码的'重力方向'——上方套牢盘越重（高cost_85pct），股价上涨阻力越大；下方获利盘越多（低cost_15pct），股价下跌支撑越强。比值>2=压力远大于支撑。",
    dependencies=("cyq_perf.parquet", "daily_adj.parquet"),
)
def factor_chip_above_below_ratio(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    daily = context.load("daily_adj.parquet")

    close = daily["close"]
    cost_85 = cyq["cost_85pct"]
    cost_15 = cyq["cost_15pct"]

    common = close.index.intersection(cost_85.index).intersection(cost_15.index)
    above = close.loc[common] - cost_85.loc[common]  # distance above 85th pct
    below = cost_15.loc[common] - close.loc[common]  # distance below 15th pct

    ratio = safe_divide(above, below)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="chip_support_strength",
    description="筹码支撑强度因子，cost_15pct处的筹码密度×1/(价格-成本15pct距离)截面排名。",
    category="coupling",
    thesis="底部的筹码密集区形成支撑——筹码支撑强度=底部筹码密度×距离倒数。底部筹码越多、价格离支撑越近，支撑力越强。这是技术分析中'筹码密集区是强支撑'的量化表达。",
    dependencies=("cyq_perf.parquet", "daily_adj.parquet"),
)
def factor_chip_support_strength(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    daily = context.load("daily_adj.parquet")

    close = daily["close"]
    cost_15 = cyq["cost_15pct"]
    cost_5 = cyq["cost_5pct"]

    common = close.index.intersection(cost_15.index)
    price = close.loc[common]
    c15 = cost_15.loc[common]
    c5 = cost_5.loc[common]

    # Density proxy: how tight the lower cost range is (tighter = denser)
    density = 1.0 / (c15 - c5).replace(0, np.nan).abs().clip(lower=0.01)
    # Distance from price to support (c15), closer = stronger support
    distance = (price - c15).abs().clip(lower=0.01)
    strength = density / distance

    return cross_sectional_rank(strength)
