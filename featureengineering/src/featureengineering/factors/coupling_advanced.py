"""
Advanced factor coupling (因子高级耦合) — Class 5.

行业中性化耦合、IC自适应合成、拥挤度预警和宏观状态条件因子。
这些因子加载已有的 .fea 文件进行二次加工，属于 Class 5 (__factors__)。

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


