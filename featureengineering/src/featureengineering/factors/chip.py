from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


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


# ── Chip morphology (new factors) ──────────────────────────────────────────


@register_factor(
    name="chip_range_normalized",
    description="归一化筹码区间因子，(cost_85pct-cost_15pct)/cost_50pct截面排名（取负向=窄区间排前）。",
    category="price",
    thesis="归一化后的筹码分布区间宽度反映了筹码的相对密集程度。区间窄=筹码密集，突破后趋势性强；区间宽=筹码分散，有效性降低。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_range_normalized(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    spread = (perf["cost_85pct"] - perf["cost_15pct"]) / perf["cost_50pct"].replace(0, np.nan)
    return cross_sectional_rank(-spread)


@register_factor(
    name="chip_support_distance",
    description="筹码支撑距离因子，close/cost_15pct-1截面排名。距离支撑位越近=反弹潜力越大。",
    category="price",
    thesis="收盘价接近15%分位成本线意味着当前价格处于筹码密集支撑区附近，技术性反弹概率增大。",
    dependencies=("daily_adj.parquet", "cyq_perf.parquet"),
)
def factor_chip_support_distance(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    perf = context.load("cyq_perf.parquet")
    close = daily_adj["close"]
    support = perf["cost_15pct"]
    common = close.index.intersection(support.index)
    result = close.loc[common] / support.loc[common].replace(0, np.nan) - 1
    return cross_sectional_rank(result)


@register_factor(
    name="chip_resistance_distance",
    description="筹码阻力距离因子，cost_85pct/close-1截面排名（取负向=接近阻力排后）。",
    category="price",
    thesis="收盘价接近85%分位成本线意味着上方套牢盘压力近在咫尺，短期上行阻力增大，存在回落风险。",
    dependencies=("daily_adj.parquet", "cyq_perf.parquet"),
)
def factor_chip_resistance_distance(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    perf = context.load("cyq_perf.parquet")
    close = daily_adj["close"]
    resistance = perf["cost_85pct"]
    common = close.index.intersection(resistance.index)
    result = resistance.loc[common] / close.loc[common].replace(0, np.nan) - 1
    return cross_sectional_rank(-result)


@register_factor(
    name="chip_weighted_cost_momentum_5d",
    description="加权成本5日动量因子，(weight_avg-weight_avg.shift(5))/weight_avg截面排名。衡量平均成本迁移速度。",
    category="price",
    thesis="加权平均成本的快速变化反映了筹码在投资者之间的快速转移。成本快速上移=增量资金入场，成本快速下移=恐慌性抛售。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_weighted_cost_momentum_5d(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    wavg = perf["weight_avg"]
    mom = wavg.groupby(level="Code").transform(lambda s: s.diff(5)) / wavg.replace(0, np.nan)
    return cross_sectional_rank(mom)


@register_factor(
    name="chip_winner_rate_stability_20d",
    description="获利盘比例20日稳定性因子，winner_rate的20日滚动标准差截面排名（取负向=高波动排后）。",
    category="price",
    thesis="获利盘比例的剧烈波动意味着筹码结构不稳定，多空分歧大，趋势难以持续。稳定的获利盘比例是健康趋势的特征。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_winner_rate_stability_20d(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    stable = perf["winner_rate"].groupby(level="Code").transform(lambda s: s.rolling(20).std())
    return cross_sectional_rank(-stable)


@register_factor(
    name="chip_historical_position",
    description="历史位置因子，(close-his_low)/(his_high-his_low)截面排名（取负向=接近历史高位排后）。",
    category="price",
    thesis="当前价格在历史最低到最高区间内的相对位置。接近历史高位=获利盘积累较大，上行阻力增大；接近历史低位=超卖反弹机会。",
    dependencies=("daily_adj.parquet", "cyq_perf.parquet"),
)
def factor_chip_historical_position(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    perf = context.load("cyq_perf.parquet")
    close = daily_adj["close"]
    lo = perf["his_low"]
    hi = perf["his_high"]
    common = close.index.intersection(lo.index).intersection(hi.index)
    position = (close.loc[common] - lo.loc[common]) / (hi.loc[common] - lo.loc[common]).replace(0, np.nan)
    return cross_sectional_rank(-position)


@register_factor(
    name="chip_cost_asymmetry",
    description="成本分布不对称因子，(cost_95pct-cost_50pct)/(cost_50pct-cost_5pct)截面排名（取负向=上重下轻排后）。",
    category="price",
    thesis="成本分布不对称度>1意味着上方套牢盘筹码量多于下方获利盘，上方抛压更重，上行阻力大。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_cost_asymmetry(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    upper = (perf["cost_95pct"] - perf["cost_50pct"]).replace(0, np.nan)
    lower = (perf["cost_50pct"] - perf["cost_5pct"]).replace(0, np.nan)
    asym = upper / lower.replace(0, np.nan)
    return cross_sectional_rank(-asym)


@register_factor(
    name="chip_concentration_change_5d",
    description="筹码集中度5日变化因子，concentration的5日差分截面排名。集中度收窄=吸筹，放宽=派发。",
    category="price",
    thesis="筹码集中度的变化方向揭示了主力资金的动向：集中度快速收窄（区间收缩）=资金吸筹，集中度放宽（区间扩散）=派发或恐慌。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_concentration_change_5d(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    conc = (perf["cost_95pct"] - perf["cost_5pct"]) / perf["cost_50pct"].replace(0, np.nan)
    conc_chg = conc.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(conc_chg)


@register_factor(
    name="chip_winner_rate_acceleration",
    description="获利盘比例加速度因子，winner_rate_change_5d的5日差分（二阶导数）截面排名（取负向=加速获利排后）。",
    category="price",
    thesis="获利盘比例增长的速度本身在加快=上涨进入加速阶段，这通常是短期赶顶的强烈信号。正加速度意味着越来越多人快速进入盈利状态。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_winner_rate_acceleration(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    wr = perf["winner_rate"]
    wr_chg = wr.groupby(level="Code").transform(lambda s: s.diff(5))
    accel = wr_chg.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(-accel)


# ── Supplementary chip factors ─────────────────────────────────────────────


@register_factor(
    name="chip_cost_kurtosis_20d",
    description="20日成本分布峰度因子 (高尖峰排前)。",
    category="price",
    thesis="成本分布的尖峰形态意味着筹码高度集中在窄区间，支撑/阻力更明确",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_cost_kurtosis_20d(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    # Approximate kurtosis proxy using cost percentiles
    spread_90_10 = perf["cost_95pct"] - perf["cost_5pct"]
    spread_70_30 = perf["cost_85pct"] - perf["cost_15pct"]
    kurtosis_proxy = safe_divide(spread_70_30, spread_90_10 + 1e-8)
    # Lower ratio = more peaked distribution (tighter middle relative to tails)
    return cross_sectional_rank(-kurtosis_proxy)


@register_factor(
    name="chip_weighted_cost_volatility_20d",
    description="20日加权成本波动率因子 (成本稳定排前, 负向)。",
    category="price",
    thesis="加权成本波动大意味着筹码大幅换手、持仓成本不稳定，持仓者面临更大不确定性",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_weighted_cost_volatility_20d(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    wa = perf["weight_avg"]
    wa_std = wa.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    wa_mean = wa.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    cv = safe_divide(wa_std, wa_mean + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="chip_winner_rate_ma5",
    description="获利盘比例5日均值因子 (获利盘多排后, 负向)。",
    category="price",
    thesis="获利盘比例均值平滑后过滤单日噪音，高获利盘=潜在抛压",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_winner_rate_ma5(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    wr = perf["winner_rate"]
    ma5 = rolling_group_mean(wr, 5)
    return cross_sectional_rank(-ma5)


@register_factor(
    name="chip_concentration_ma5",
    description="筹码集中度5日均值因子 (低集中排后, 高集中度的负向截面排名取负=高集中排后)。",
    category="price",
    thesis="筹码集中度均值化后稳定性提升，高集中度意味着筹码被少数人掌控=拉升易但出货难",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_concentration_ma5(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    concentration = safe_divide(
        perf["cost_95pct"] - perf["cost_5pct"],
        perf["cost_50pct"] + 1e-8
    )
    ma5 = rolling_group_mean(concentration, 5)
    return cross_sectional_rank(-ma5)


@register_factor(
    name="chip_support_strength_20d",
    description="20日筹码支撑强度因子 (成本密集区数量/支撑距离)。",
    category="price",
    thesis="筹码密集区离当前价格越近且越集中，支撑越强，下跌空间越小",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_support_strength_20d(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    # Support strength: how concentrated + how close the cost distribution is below current price
    cost_15 = perf["cost_15pct"]
    cost_50 = perf["cost_50pct"]
    cost_85 = perf["cost_85pct"]
    # Lower cost_50 relative to cost_85 = support band is narrow and close
    support = safe_divide(cost_50 - cost_15, cost_85 - cost_50 + 1e-8)
    # Normalize: more mass below = stronger support
    return cross_sectional_rank(support)
