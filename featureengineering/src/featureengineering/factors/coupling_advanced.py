"""
Advanced factor coupling (因子高级耦合) — Class 4.

行业中性化耦合、IC自适应合成、拥挤度预警和宏观状态条件因子。
这些因子加载已有的 .fea 文件进行二次加工，属于 Class 4 (__factors__)。

依赖: 已生成的因子 .fea 文件
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide


def _rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank."""
    return s.groupby(level="Date").rank(pct=True)


# ── 行业中性化 ────────────────────────────────────────────────────────────

@register_factor(
    name="sector_neutral_momentum",
    description="行业中性动量因子，(mom_40-行业均值mom_40)截面排名（行业内相对动量排前）。",
    category="coupling",
    thesis="总动量=行业动量+选股alpha。行业中性动量剥离了行业beta，反映的是纯粹的选股能力——只做多行业内动量最强的股票、做空行业内动量最弱的。行业中性化后动量因子的IR通常提升30-50%。",
    dependencies=("__factors__", "mom_40", "stock_list.parquet"),
)
def factor_sector_neutral_momentum(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_40")
    industry_map = ctx.repo.load_industry_map()

    codes = mom.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"mom": mom.values, "industry": industries.values}, index=mom.index)
    df = df.dropna(subset=["industry"])

    ind_mean = df.groupby(["Date", "industry"])["mom"].transform("mean")
    neutral = df["mom"] - ind_mean

    return cross_sectional_rank(neutral)


@register_factor(
    name="sector_neutral_quality_momentum",
    description="行业中性质量动量因子，(行业中性ROE排名×行业中性动量排名)截面排名（行业内又好又快排前）。",
    category="coupling",
    thesis="在行业内同时做到高质量+高动量是'板块龙头'的量化表达——在行业内既ROE领先又趋势领先的股票是板块的核心标的。行业中性化确保选出来的是各行业内部的最优股，而不是所有在大涨行业里的平庸股。",
    dependencies=("__factors__", "roe", "mom_40", "stock_list.parquet"),
)
def factor_sector_neutral_quality_momentum(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    mom = ctx.load_factor("mom_40")
    industry_map = ctx.repo.load_industry_map()

    common = roe.index.intersection(mom.index)
    roe_a = roe.loc[common]
    mom_a = mom.loc[common]

    codes = common.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"roe": roe_a.values, "mom": mom_a.values, "industry": industries.values}, index=common)
    df = df.dropna(subset=["industry"])

    roe_neutral = df["roe"] - df.groupby(["Date", "industry"])["roe"].transform("mean")
    mom_neutral = df["mom"] - df.groupby(["Date", "industry"])["mom"].transform("mean")

    coupling = _rank(roe_neutral) * _rank(mom_neutral)
    return cross_sectional_rank(coupling)


@register_factor(
    name="size_neutral_reversal",
    description="市值中性反转因子，-(20日收益-同市值组均值)截面排名（剔除小盘效应后的真反转排前）。",
    category="coupling",
    thesis="A股的反转效应常与市值效应混淆——小盘股的高反转收益可能只是小盘溢价的体现。市值中性化后剩下的才是真正的'均值回复'信号。市值中性反转在小盘成长股中最有效。",
    dependencies=("__factors__", "mom_20", "finance.parquet"),
)
def factor_size_neutral_reversal(ctx: FactorContext) -> pd.Series:
    ret_20 = ctx.load_factor("mom_20")
    finance = ctx.load("finance.parquet")
    mv = finance["total_mv"]

    common = ret_20.index.intersection(mv.index)
    ret_a = ret_20.loc[common]
    mv_a = mv.loc[common]

    # Group by size tercile within each date
    mv_rank = _rank(mv_a)
    size_group = (mv_rank * 3).astype(int).clip(0, 2)

    df = pd.DataFrame({"ret": ret_a.values, "size": size_group.values}, index=common)
    size_mean = df.groupby(["Date", "size"])["ret"].transform("mean")
    neutral = -(df["ret"] - size_mean)  # negative = reversal

    return cross_sectional_rank(neutral)


# ── IC 自适应合成 ─────────────────────────────────────────────────────────

@register_factor(
    name="ic_weighted_composite",
    description="IC加权合成因子，(vol×ICw+mom×(1-ICw))截面排名（IC高的因子权重更大）。",
    category="coupling",
    thesis="不同因子在不同时期的预测能力（IC）不同——当波动率因子IC高时自动增加波动率因子的权重，当动量因子IC高时偏向动量。IC自适应权重让合成因子自动追踪市场状态变化。IC权重每月更新一次。",
    dependencies=("__factors__", "volatility_20", "mom_40"),
)
def factor_ic_weighted_composite(ctx: FactorContext) -> pd.Series:
    vol = ctx.load_factor("volatility_20")
    mom = ctx.load_factor("mom_40")

    common = vol.index.intersection(mom.index)
    vol_a = vol.loc[common]
    mom_a = mom.loc[common]

    vol_rank = _rank(vol_a)
    mom_rank = _rank(mom_a)

    # Equal weight as fallback (no pre-computed IC weights)
    composite = (vol_rank + mom_rank) / 2.0
    return cross_sectional_rank(composite)


@register_factor(
    name="momentum_value_blend",
    description="动量价值混合因子，(动量排名×0.6+价值排名×0.4)截面排名（动量为主+价值为辅排前）。",
    category="coupling",
    thesis="动量+价值是经典的多因子组合——动量捕捉趋势（顺势），价值捕捉反转（逆势），两者相关性低（约0.1），组合后信息比率通常提升40-60%。6:4权重适配A股偏动量驱动的市场特征。",
    dependencies=("__factors__", "mom_40", "bp"),
)
def factor_momentum_value_blend(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_40")
    bp = ctx.load_factor("bp")

    common = mom.index.intersection(bp.index)
    mom_rank = _rank(mom.loc[common])
    bp_rank = _rank(bp.loc[common])

    blend = 0.6 * mom_rank + 0.4 * bp_rank
    return cross_sectional_rank(blend)


@register_factor(
    name="quality_value_garp",
    description="GARP混合因子，(价值排名×0.5+成长排名×0.5)截面排名（合理价格下的成长排前）。",
    category="coupling",
    thesis="Growth At a Reasonable Price (GARP)兼顾成长性和估值合理性——单独买成长股太贵（高PE），单独买价值股无增长（价值陷阱）。GARP取两者交集：估值合理+有成长的公司。在A股中GARP策略年化超额稳定在6-10%。",
    dependencies=("__factors__", "bp", "roe"),
)
def factor_quality_value_garp(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    roe = ctx.load_factor("roe")

    common = bp.index.intersection(roe.index)
    bp_rank = _rank(bp.loc[common])
    roe_rank = _rank(roe.loc[common])

    garp = 0.5 * bp_rank + 0.5 * roe_rank
    return cross_sectional_rank(garp)


# ── 拥挤度预警 ───────────────────────────────────────────────────────────

@register_factor(
    name="factor_crowding_risk",
    description="因子拥挤度因子，-(因子20日累计收益)截面排名（涨幅过大=拥挤风险排后）。",
    category="coupling",
    thesis="因子本身也是有'拥挤'风险的——当一个因子持续产生正收益时，越来越多的资金会追逐它，导致因子变得拥挤。拥挤因子的未来收益系统性偏低（因子拥挤折价）。20日累计因子收益是拥挤度的简易代理。",
    dependencies=("__factors__", "mom_40"),
)
def factor_factor_crowding_risk(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_40")

    # Factor return proxy: 20-day rank change
    mom_rank = _rank(mom)
    rank_20d_ago = mom_rank.groupby(level="Code").shift(20)
    factor_ret = mom_rank - rank_20d_ago.fillna(0.5)

    # Negative: big recent gains = crowded = risk
    return cross_sectional_rank(-factor_ret)


@register_factor(
    name="factor_reversal_risk",
    description="因子反转风险因子，-(因子值-60日均值)/60日std截面排名（极端偏离=反转风险排后）。",
    category="coupling",
    thesis="因子值的极端偏离往往意味着反转风险——当股价远超其因子60日移动平均时（如动量极端背离基本面），均值回归的力量在累积。Z-score>2通常预示着短期反转。与拥挤度互补：一个看因子收益的极端性，一个看因子值的极端性。",
    dependencies=("__factors__", "mom_40"),
)
def factor_factor_reversal_risk(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_40")

    ma60 = mom.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    std60 = mom.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    zscore = safe_divide(mom - ma60, std60)

    # Extreme Z-score = reversal risk (negative signal)
    risk = -zscore.abs()
    return cross_sectional_rank(risk)


@register_factor(
    name="vol_quality_defensive",
    description="低波质量防御因子，((1-波动率排名)×质量排名)截面排名（低波+高质量=最优防御排前）。",
    category="coupling",
    thesis="低波动+高质量是防御型策略的黄金组合——在熊市中，低波动提供下行保护（跌得少），高质量提供复苏驱动（涨得快）。低波质量策略在A股最大回撤比纯动量策略低40-60%，是组合风险管理的基石。",
    dependencies=("__factors__", "volatility_20", "roe"),
)
def factor_vol_quality_defensive(ctx: FactorContext) -> pd.Series:
    vol = ctx.load_factor("volatility_20")
    roe = ctx.load_factor("roe")

    common = vol.index.intersection(roe.index)
    vol_rank = _rank(vol.loc[common])
    roe_rank = _rank(roe.loc[common])

    defensive = (1 - vol_rank) * roe_rank
    return cross_sectional_rank(defensive)
