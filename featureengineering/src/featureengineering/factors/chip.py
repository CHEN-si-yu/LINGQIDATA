from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── Chip structure (cyq_perf) ────────────────────────────────────────────

@register_factor(
    name="winner_rate",
    description="获利盘比例因子，winner_rate截面排名（取负向=高获利盘为反转信号）。",
    category="price",
    thesis="高获利盘比例意味着多数持仓者处于盈利状态，短期存在获利了结压力，是反转信号。",
    dependencies=("cyq_perf.parquet",),
)
def factor_winner_rate(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    return cross_sectional_rank(-perf["winner_rate"])


@register_factor(
    name="chip_concentration",
    description="筹码集中度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（取负向=窄区间排前）。",
    category="price",
    thesis="筹码分布区间窄=筹码密集，突破后趋势性强。密集区间突破方向的持续性优于分散区间。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_concentration(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    spread = (perf["cost_95pct"] - perf["cost_5pct"]) / perf["cost_50pct"].replace(0, np.nan)
    return cross_sectional_rank(-spread)


@register_factor(
    name="chip_position",
    description="筹码位置因子，(close-cost_5pct)/(cost_95pct-cost_5pct)截面排名。",
    category="price",
    thesis="当前价在筹码分布中的相对位置反映获利盘大小，高位=接近套牢区上沿，上行阻力增大。",
    dependencies=("daily_adj.parquet", "cyq_perf.parquet"),
)
def factor_chip_position(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    perf = context.load("cyq_perf.parquet")
    close = daily_adj["close"]
    lo = perf["cost_5pct"]
    hi = perf["cost_95pct"]
    common = close.index.intersection(lo.index).intersection(hi.index)
    position = (close.loc[common] - lo.loc[common]) / (hi.loc[common] - lo.loc[common]).replace(0, np.nan)
    return cross_sectional_rank(-position)


@register_factor(
    name="cost_displacement",
    description="成本偏离因子，(close-weight_avg)/weight_avg截面排名（取负向=大幅偏离排后）。",
    category="price",
    thesis="现价偏离加权平均成本越大，获利回吐/抄底反弹的均值回归动力越强。",
    dependencies=("daily_adj.parquet", "cyq_perf.parquet"),
)
def factor_cost_displacement(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    perf = context.load("cyq_perf.parquet")
    close = daily_adj["close"]
    wavg = perf["weight_avg"]
    common = close.index.intersection(wavg.index)
    displacement = (close.loc[common] - wavg.loc[common]) / wavg.loc[common].replace(0, np.nan)
    return cross_sectional_rank(-displacement)


@register_factor(
    name="winner_rate_change_5d",
    description="获利盘5日变化因子，winner_rate - winner_rate.shift(5)截面排名（取负向）。",
    category="price",
    thesis="获利盘比例快速扩大=上涨过程中筹码加速获利，往往是短期赶顶信号。",
    dependencies=("cyq_perf.parquet",),
)
def factor_winner_rate_change_5d(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    wr = perf["winner_rate"]
    change = wr.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(-change)


@register_factor(
    name="chip_cost_skew",
    description="成本分布偏度因子，(cost_50pct-cost_15pct)/(cost_85pct-cost_50pct)截面排名。",
    category="price",
    thesis="偏度>1=上方套牢盘重于下方获利盘（负向），偏度<1=下方支撑强于上方压力（正向）。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_cost_skew(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    lower = (perf["cost_50pct"] - perf["cost_15pct"]).replace(0, np.nan)
    upper = (perf["cost_85pct"] - perf["cost_50pct"]).replace(0, np.nan)
    skew = lower / upper.replace(0, np.nan)
    return cross_sectional_rank(skew)


# ── Cost skew momentum ─────────────────────────────────────────────────


@register_factor(
    name="cost_skew_momentum_5d",
    description="成本偏度5日变化因子，chip_cost_skew的5日差分截面排名。",
    category="price",
    thesis="成本分布偏度从'下方支撑>上方压力'转变为'上方压力>下方支撑'是趋势衰竭的领先信号。持续上涨理应将筹码从下方搬运到上方（偏度自然下降），但如果偏度快速逆转（5日变化转正），说明上方套牢盘在快速增厚——这是多空力量对比变化的微观信号。",
    dependencies=("cyq_perf.parquet",),
)
def factor_cost_skew_momentum_5d(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    lower = (perf["cost_50pct"] - perf["cost_15pct"]).replace(0, np.nan)
    upper = (perf["cost_85pct"] - perf["cost_50pct"]).replace(0, np.nan)
    skew = lower / upper.replace(0, np.nan)
    skew_chg = skew.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(-skew_chg)
