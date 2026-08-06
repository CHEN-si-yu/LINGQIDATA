"""
Coupling extended factors — Class 5 (cross-factor interaction, survivors only).
"""
from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ── Factor Momentum ──────────────────────────────────────────────────────────

@register_factor(
    name="coupling_value_momentum_20d",
    description="价值因子(bp)自身20日动量。bp值持续上升=估值优势在扩大，排名高。",
    category="coupling",
    thesis="因子值的时序动量捕捉了因子暴露的边际变化。bp持续上升=股价下跌快于账面价值——估值优势在扩大。",
    dependencies=("__factors__", "bp"),
)
def factor_coupling_value_momentum_20d(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    mom = bp.groupby(level="Code").transform(lambda s: s.diff(20))
    return cross_sectional_rank(mom)


@register_factor(
    name="coupling_quality_momentum_20d",
    description="盈利质量因子(sp_ttm+bp rank)20日动量。质量改善=排名高。",
    category="coupling",
    thesis="用bp和sp_ttm合成质量代理，然后做20日动量。质量因子持续改善=盈利能力提升。",
    dependencies=("__factors__", "bp", "sp_ttm"),
)
def factor_coupling_quality_momentum_20d(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    sp = ctx.load_factor("sp_ttm")
    quality = sp.groupby(level="Date").rank(pct=True) + bp.groupby(level="Date").rank(pct=True)
    mom = quality.groupby(level="Code").transform(lambda s: s.diff(20))
    return cross_sectional_rank(mom)


@register_factor(
    name="coupling_liquidity_momentum_20d",
    description="流动性因子(turnover_20)自身20日变化。换手率上升=关注度提升，排名高。",
    category="coupling",
    thesis="换手率因子的时序变化捕捉了市场关注度的边际变化。",
    dependencies=("__factors__", "turnover_20"),
)
def factor_coupling_liquidity_momentum_20d(ctx: FactorContext) -> np.ndarray:
    # turnover_20 的 .fea 为 rank(-换手),高=低换手;描述要求"换手率上升=关注度
    # 提升,排名高",故对 (1.0 - to) 取差分。修复前对 to 取 diff = 换手率下降排前,
    # 与描述相反(2026-08-05)。
    to = ctx.load_factor("turnover_20")
    mom = (1.0 - to).groupby(level="Code").transform(lambda s: s.diff(20))
    return cross_sectional_rank(mom)


# ── Factor Cross-Interactions ────────────────────────────────────────────────

@register_factor(
    name="coupling_value_quality_resonance",
    description="价值(bp)+质量(sp_ttm)共振。两者同时高排名=优质低估，超级信号。",
    category="coupling",
    thesis="当价值因子和质量因子同时给出高排名时(低估+高盈利)，是A股最可靠的alpha组合。",
    dependencies=("__factors__", "bp", "sp_ttm"),
)
def factor_coupling_value_quality_resonance(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    sp = ctx.load_factor("sp_ttm")
    resonance = bp * sp
    return cross_sectional_rank(resonance)


@register_factor(
    name="coupling_valuation_sentiment_divergence",
    description="估值(bp)与情绪(turnover_20)的背离。低估值+高换手=价值发现，排名高。",
    category="coupling",
    thesis="低估值(bp高)同时高换手意味着市场在积极交易这只低估股票——可能是价值发现过程。",
    dependencies=("__factors__", "bp", "turnover_20"),
)
def factor_coupling_valuation_sentiment_divergence(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    to = ctx.load_factor("turnover_20")
    signal = bp * to
    return cross_sectional_rank(signal)


# ── Multi-Factor Composites ──────────────────────────────────────────────────

@register_factor(
    name="coupling_smallcap_value_combo",
    description="小市值(log_total_mv取反=小盘高排)+低估值(bp)。小盘价值股效应。",
    category="coupling",
    thesis="小盘价值股在全球市场均有显著溢价。在A股，小盘+低估值组合在牛市和震荡市中表现突出。",
    dependencies=("__factors__", "bp", "log_total_mv"),
)
def factor_coupling_smallcap_value_combo(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    size = ctx.load_factor("log_total_mv")
    combo = bp * size
    return cross_sectional_rank(combo)
