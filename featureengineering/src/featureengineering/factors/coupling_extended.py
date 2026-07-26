"""
Coupling extended factors — Class 5 (cross-factor interaction).

These factors load pre-computed factor values via ctx.load_factor() and
build derived signals from cross-factor interactions:
  - Factor momentum: factor values own time-series momentum
  - Factor interactions: product / divergence of two factors
  - Factor regime composites: conditional combinations

All factors use dependencies=("__factors__", "factor_a", "factor_b", ...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Factor Momentum — time-series momentum of factor values
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_value_momentum_20d",
    description="价值因子(bp)自身20日动量。bp值持续上升=估值优势在扩大，排名高。",
    category="coupling",
    thesis="因子值的时序动量捕捉了因子暴露的边际变化。bp持续上升=股价下跌快于账面价值"
           "——估值优势在扩大(价值陷阱风险也需注意)。因子的时序变化比单期截面值包含更多信息。",
    dependencies=("__factors__", "bp"),
)
def factor_coupling_value_momentum_20d(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    mom = bp.groupby(level="Code").transform(lambda s: s.diff(20))
    return cross_sectional_rank(mom)


@register_factor(
    name="coupling_quality_momentum_20d",
    description="盈利质量因子(roe proxy: 1/pb * 1/pe_ttm)20日动量。质量改善=排名高。",
    category="coupling",
    thesis="用bp和sp_ttm合成ROE代理(sp_ttm/bp ≈ 净利润/净资产 ≈ ROE)，"
           "然后做20日动量。质量因子持续改善=盈利能力提升，是最可靠的alpha来源之一。",
    dependencies=("__factors__", "bp", "sp_ttm"),
)
def factor_coupling_quality_momentum_20d(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    sp = ctx.load_factor("sp_ttm")
    # ROE proxy = sales-to-price / book-to-price = sales / book ≈ ROE * (P/B) consistency
    # Actually: sp_ttm/bp = (S/P) / (B/P) = S/B = asset turnover proxy
    # Better: use sp and bp ranks to form quality composite
    quality = sp.groupby(level="Date").rank(pct=True) + bp.groupby(level="Date").rank(pct=True)
    mom = quality.groupby(level="Code").transform(lambda s: s.diff(20))
    return cross_sectional_rank(mom)


@register_factor(
    name="coupling_liquidity_momentum_20d",
    description="流动性因子(turnover_20)自身20日变化。换手率上升=关注度提升，排名高。",
    category="coupling",
    thesis="换手率因子的时序变化捕捉了市场关注度的边际变化。"
           "换手率持续上升=市场关注度在增加、流动性在改善——短期可能伴随价格上涨。",
    dependencies=("__factors__", "turnover_20"),
)
def factor_coupling_liquidity_momentum_20d(ctx: FactorContext) -> np.ndarray:
    to = ctx.load_factor("turnover_20")
    mom = to.groupby(level="Code").transform(lambda s: s.diff(20))
    return cross_sectional_rank(mom)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Factor Cross-Interactions
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_value_quality_resonance",
    description="价值(bp)+质量(sp_ttm)共振。两者同时高排名=优质低估，超级信号。",
    category="coupling",
    thesis="当价值因子和质量因子同时给出高排名时(低估+高盈利)，"
           "是A股最可靠的alpha组合——优质公司的估值洼地。两者乘积捕捉了共振强度。",
    dependencies=("__factors__", "bp", "sp_ttm"),
)
def factor_coupling_value_quality_resonance(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    sp = ctx.load_factor("sp_ttm")
    resonance = bp * sp
    return cross_sectional_rank(resonance)


@register_factor(
    name="coupling_momentum_reversal_balance",
    description="动量(mom_20)与反转(reversal_5)的差值。正=趋势占优，负=反转占优，排名高。",
    category="coupling",
    thesis="动量与反转因子的差值捕捉了市场当前的主导风格。正值=动量风格占优"
           "(趋势延续)，负值=反转风格占优(均值回归)。该因子本身是一个风格轮动信号。",
    dependencies=("__factors__", "mom_20", "reversal_5"),
)
def factor_coupling_momentum_reversal_balance(ctx: FactorContext) -> np.ndarray:
    mom = ctx.load_factor("mom_20")
    rev = ctx.load_factor("reversal_5")
    balance = mom - rev
    return cross_sectional_rank(balance)


@register_factor(
    name="coupling_volume_volatility_ratio",
    description="量(volume_ratio)与波动(volatility_20)的比率。高量低波=稳健放量，排名高。",
    category="coupling",
    thesis="成交量与波动率的比率。高量低波=成交量放大但波动率未放大——"
           "意味着有新增资金平稳入场(机构吸筹特征)；低量高波=缩量剧烈波动——"
           "意味着筹码不稳定、方向不明。",
    dependencies=("__factors__", "volume_ratio", "volatility_20"),
)
def factor_coupling_volume_volatility_ratio(ctx: FactorContext) -> np.ndarray:
    vr = ctx.load_factor("volume_ratio")
    vol = ctx.load_factor("volatility_20")
    ratio = safe_divide(vr, vol + 0.01)
    ratio = ratio.clip(0, 10)
    return cross_sectional_rank(ratio)


@register_factor(
    name="coupling_chip_momentum_combo",
    description="筹码集中度(cr3)与价格动量(mom_60)的组合。筹码集中+趋势向上=主力控盘拉升。",
    category="coupling",
    thesis="筹码集中度高(cr3因子排名高)+中长期价格动量强(mom_60排名高)"
           "=主力控盘拉升阶段——这是最强势的牛股特征。两者乘积捕捉共振强度。",
    dependencies=("__factors__", "chip_cr3_factor", "mom_60"),
)
def factor_coupling_chip_momentum_combo(ctx: FactorContext) -> np.ndarray:
    cr3 = ctx.load_factor("chip_cr3_factor")
    mom = ctx.load_factor("mom_60")
    combo = cr3 * mom
    return cross_sectional_rank(combo)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Factor Divergence / Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_fundamental_price_divergence",
    description="基本面(bp)与技术面(mom_20)的背离。bp高+mom低=超跌价值，排名高。",
    category="coupling",
    thesis="当基本面因子(价值bp)与技术面因子(动量mom)出现背离时——"
           "基本面好但技术面差(超跌价值股)，或者基本面差但技术面好(泡沫股)。"
           "该因子做多超跌价值(bp高+mom低)，是经典的反转价值策略。",
    dependencies=("__factors__", "bp", "mom_20"),
)
def factor_coupling_fundamental_price_divergence(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")
    divergence = bp - mom
    return cross_sectional_rank(divergence)


@register_factor(
    name="coupling_valuation_sentiment_divergence",
    description="估值(bp)与情绪(turnover_20)的背离。低估值+高换手=价值发现，排名高。",
    category="coupling",
    thesis="低估值(bp高)同时高换手意味着市场在积极交易这只低估股票——"
           "可能是价值发现过程。高估值(bp低)同时低换手意味着无人问津的高价股——"
           "可能是流动性陷阱。bp高+turnover高=价值被发现中。",
    dependencies=("__factors__", "bp", "turnover_20"),
)
def factor_coupling_valuation_sentiment_divergence(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    to = ctx.load_factor("turnover_20")
    signal = bp * to
    return cross_sectional_rank(signal)


@register_factor(
    name="coupling_risk_adjusted_momentum",
    description="风险调整动量=mom_60/volatility_60。高动量低波动=高质量趋势，排名高。",
    category="coupling",
    thesis="动量除以波动率得到风险调整后的趋势强度。高动量+低波动=稳定上涨(优质趋势)；"
           "高动量+高波动=投机性上涨(脆弱趋势)。该比率过滤了高波动的虚假趋势信号。",
    dependencies=("__factors__", "mom_60", "volatility_60"),
)
def factor_coupling_risk_adjusted_momentum(ctx: FactorContext) -> np.ndarray:
    mom = ctx.load_factor("mom_60")
    vol = ctx.load_factor("volatility_60")
    adj = safe_divide(mom, vol + 0.01)
    adj = adj.clip(0, 10)
    return cross_sectional_rank(adj)


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Multi-Factor Composites
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_value_mom_quality_trio",
    description="价值+动量+质量三因子等权组合。经典Fama-French三因子在A股的截面实现。",
    category="coupling",
    thesis="bp(价值)+mom_60(动量)+sp_ttm(质量/盈利)三因子等权组合。"
           "这三者是全球股市最稳健的alpha来源，在A股同样有效。等权组合降低了"
           "单一因子失效的风险，是稳健的多因子复合信号。",
    dependencies=("__factors__", "bp", "mom_60", "sp_ttm"),
)
def factor_coupling_value_mom_quality_trio(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_60")
    sp = ctx.load_factor("sp_ttm")
    trio = bp + mom + sp
    return cross_sectional_rank(trio)


@register_factor(
    name="coupling_low_risk_quality_combo",
    description="低风险(volatility_20) + 高质量(sp_ttm) + 低估值(bp)的三低组合。防御型优质信号。",
    category="coupling",
    thesis="低波动+高质量+低估值的组合代表了又好又便宜又稳的股票——"
           "在A股市场中这类股票长期提供稳健超额收益(低波动异象+价值溢价+质量溢价)。"
           "特别适合震荡市和熊市中的防御配置。",
    dependencies=("__factors__", "bp", "sp_ttm", "volatility_20"),
)
def factor_coupling_low_risk_quality_combo(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    sp = ctx.load_factor("sp_ttm")
    vol = ctx.load_factor("volatility_20")
    # vol is neg-ranked (low vol = high rank), so we can add directly
    combo = bp + sp + vol
    return cross_sectional_rank(combo)


@register_factor(
    name="coupling_smallcap_value_combo",
    description="小市值(log_total_mv取反=小盘高排)+低估值(bp)。小盘价值股效应。",
    category="coupling",
    thesis="小盘价值股(Fama-French SMB+HML的交叉)在全球市场均有显著溢价。"
           "在A股，小盘+低估值组合在牛市和震荡市中表现突出。"
           "log_total_mv取负排名(小盘排前)×bp(价值排前)=小盘价值信号。",
    dependencies=("__factors__", "bp", "log_total_mv"),
)
def factor_coupling_smallcap_value_combo(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    size = ctx.load_factor("log_total_mv")
    combo = bp * size
    return cross_sectional_rank(combo)


# ═══════════════════════════════════════════════════════════════════════════════
# Section E: Cross-Asset / Macro Style Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_defensive_score",
    description="防御性评分：低波动+低换手+低估值。三个防御特征的等权组合，熊市中排名高。",
    category="coupling",
    thesis="防御型股票的三个特征——低波动(volatility_20排前)、低换手(turnover_20排前"
           "意味着筹码稳定)、低估值(bp排前)。三者等权组合识别出最具防御性的股票，"
           "在熊市和震荡市中应具有抗跌性。",
    dependencies=("__factors__", "bp", "turnover_20", "volatility_20"),
)
def factor_coupling_defensive_score(ctx: FactorContext) -> np.ndarray:
    bp = ctx.load_factor("bp")
    to = ctx.load_factor("turnover_20")
    vol = ctx.load_factor("volatility_20")
    score = bp + to + vol
    return cross_sectional_rank(score)


@register_factor(
    name="coupling_aggressive_score",
    description="激进型评分：高动量+高换手+高波动。三个进攻特征的等权组合，牛市中排名高。",
    category="coupling",
    thesis="激进型股票的三个特征——高动量(mom_60排前)、高换手(turnover_20排后=高换手排前"
           "意味着交易活跃)、高波动(volatility_20排后=高波动排前)。"
           "三者等权组合识别出最具进攻性的股票，在牛市中应具有高弹性。",
    dependencies=("__factors__", "mom_60", "turnover_20", "volatility_20"),
)
def factor_coupling_aggressive_score(ctx: FactorContext) -> np.ndarray:
    mom = ctx.load_factor("mom_60")
    to = ctx.load_factor("turnover_20")
    vol = ctx.load_factor("volatility_20")
    # For aggressive: want high mom, high turnover (low turnover_20 rank), high vol (low volatility_20 rank)
    score = mom + (1.0 - to) + (1.0 - vol)
    return cross_sectional_rank(score)
