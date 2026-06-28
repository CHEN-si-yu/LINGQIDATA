"""
Factor-coupling (因子耦合) module — Class 4 factors.

These factors load pre-computed Class 1/2/3 factors from .fea files and build
derived signals from their interactions: co-momentum, divergence, resonance,
conditional effects, stability, timing, and cross-factor residuals.

All factors declare ``"__factors__"`` in their dependencies so the build
orchestrator routes them to the Class 4 (coupling) pipeline.  The remaining
dependency strings are factor names that must already exist as .fea files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    event_decay,
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    safe_divide,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (0-1)."""
    return s.groupby(level="Date").rank(pct=True)

def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score per date."""
    mu = s.groupby(level="Date").transform("mean")
    sg = s.groupby(level="Date").transform("std")
    return safe_divide(s - mu, sg + 1e-8)

def _delta(s: pd.Series, window: int) -> pd.Series:
    """Per-stock difference over *window* periods."""
    return s.groupby(level="Code").diff(window)

def _momentum(s: pd.Series, window: int) -> pd.Series:
    """Per-stock percentage change over *window* periods."""
    return s.groupby(level="Code").transform(lambda x: x.pct_change(window))

# ═══════════════════════════════════════════════════════════════════════════════
# A — Factor Momentum (因子动量)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="bp_factor_momentum_20",
    description="BP因子20日动量 (BP因子值的变化率)。",
    category="coupling",
    thesis="BP自身的动量捕捉估值修复的启动——BP快速上升的股票正在被市场重估",
    dependencies=("__factors__", "bp"),
)
def factor_bp_factor_momentum_20(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = _momentum(bp, 20)
    return cross_sectional_rank(mom)

@register_factor(
    name="roe_factor_momentum_60",
    description="ROE因子60日动量 (ROE因子值的变化率)。",
    category="coupling",
    thesis="ROE因子值的趋势变化反映盈利改善的可持续性——ROE因子在上升的股票值得关注",
    dependencies=("__factors__", "roe"),
)
def factor_roe_factor_momentum_60(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    mom = _momentum(roe, 60)
    return cross_sectional_rank(mom)

@register_factor(
    name="mom20_factor_momentum_20",
    description="20日动量因子自身的20日变化 (因子加速度)。",
    category="coupling",
    thesis="动量因子的动量是趋势的二阶导——不仅趋势好而且趋势在加速",
    dependencies=("__factors__", "mom_20"),
)
def factor_mom20_factor_momentum_20(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_20")
    delta = _delta(mom, 20)
    return cross_sectional_rank(delta)

@register_factor(
    name="turnover_factor_momentum_20",
    description="换手率因子20日动量 (关注度变化)。",
    category="coupling",
    thesis="换手率因子趋势上升=市场关注度在持续提升，是正面信号",
    dependencies=("__factors__", "turnover_20"),
)
def factor_turnover_factor_momentum_20(ctx: FactorContext) -> pd.Series:
    to = ctx.load_factor("turnover_20")
    mom = _momentum(to, 20)
    return cross_sectional_rank(mom)

@register_factor(
    name="volatility_factor_momentum_20",
    description="波动率因子20日动量 (波动率变化方向, 降波排前)。",
    category="coupling",
    thesis="波动率因子下行=风险压缩过程，是市场对股票的确定性定价增强的信号",
    dependencies=("__factors__", "volatility_20"),
)
def factor_volatility_factor_momentum_20(ctx: FactorContext) -> pd.Series:
    vol = ctx.load_factor("volatility_20")
    delta = _delta(vol, 20)
    return cross_sectional_rank(-delta)

@register_factor(
    name="winner_rate_factor_momentum_20",
    description="获利盘因子20日动量 (筹码结构变化方向)。",
    category="coupling",
    thesis="获利盘比例趋势下降意味着筹码在从分散到集中——主力收集筹码的迹象",
    dependencies=("__factors__", "winner_rate"),
)
def factor_winner_rate_factor_momentum_20(ctx: FactorContext) -> pd.Series:
    wr = ctx.load_factor("winner_rate")
    delta = _delta(wr, 20)
    return cross_sectional_rank(-delta)

@register_factor(
    name="mf_flow_factor_momentum_20",
    description="主力资金流因子20日动量 (资金态度变化)。",
    category="coupling",
    thesis="主力资金因子的趋势上升意味着机构态度在边际改善",
    dependencies=("__factors__", "mf_net_inflow_ratio"),
)
def factor_mf_flow_factor_momentum_20(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    delta = _delta(mf, 20)
    return cross_sectional_rank(delta)

# ═══════════════════════════════════════════════════════════════════════════════
# B — Factor Divergence (因子背离)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="value_momentum_divergence",
    description="价值-动量背离因子 (高BP+低动量=价值陷阱风险, 取负=深度价值)。",
    category="coupling",
    thesis="高BP+正动量=价值重估确认(买入信号)，高BP+负动量=价值陷阱(回避)。捕捉两者背离的价值发现机会",
    dependencies=("__factors__", "bp", "mom_20"),
)
def factor_value_momentum_divergence(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")
    bp_r = _rank(bp)
    mom_r = _rank(mom)
    # High BP + negative momentum = value trap (negative signal)
    # High BP + positive momentum = value discovery (positive signal)
    divergence = bp_r * mom_r
    return cross_sectional_rank(divergence)

@register_factor(
    name="quality_price_divergence",
    description="质量-价格背离因子 (高ROE+低收益=质量回调买入机会)。",
    category="coupling",
    thesis="高质量公司短期下跌是买入良机——基本面的高质量与价格的暂时疲软形成背离",
    dependencies=("__factors__", "roe", "mom_20"),
)
def factor_quality_price_divergence(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    mom = ctx.load_factor("mom_20")
    roe_r = _rank(roe)
    mom_r = _rank(mom)
    # High quality + price weakness = dip opportunity
    divergence = roe_r * (1 - mom_r)
    return cross_sectional_rank(divergence)

@register_factor(
    name="flow_price_divergence",
    description="资金-价格背离因子 (主力流入+价格下跌=吸筹信号)。",
    category="coupling",
    thesis="主力资金在价格下跌时持续买入是吸筹的经典特征——聪明钱与价格走势的背离",
    dependencies=("__factors__", "mf_net_inflow_ratio", "mom_20"),
)
def factor_flow_price_divergence(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    mom = ctx.load_factor("mom_20")
    mf_r = _rank(mf)
    mom_r = _rank(mom)
    # Inflow + weak price = accumulation
    divergence = mf_r * (1 - mom_r)
    return cross_sectional_rank(divergence)

@register_factor(
    name="chip_price_divergence",
    description="筹码-价格背离因子 (获利盘减+价格上涨=派发风险, 取负)。",
    category="coupling",
    thesis="价格上涨但获利盘比例快速下降=筹码从获利者转向新买家，可能是主力派发",
    dependencies=("__factors__", "winner_rate", "mom_20"),
)
def factor_chip_price_divergence(ctx: FactorContext) -> pd.Series:
    wr = ctx.load_factor("winner_rate")
    mom = ctx.load_factor("mom_20")
    # High price momentum + declining winner rate = distribution warning
    wr_change = _delta(wr, 5)
    wr_declining = _rank(-wr_change)
    mom_r = _rank(mom)
    divergence = wr_declining * mom_r
    return cross_sectional_rank(-divergence)

@register_factor(
    name="turnover_price_divergence",
    description="换手-价格背离因子 (高换手+下跌=恐慌抛售, 取负)。",
    category="coupling",
    thesis="高换手伴随价格下跌是恐慌性抛售(或主力对倒出货)，低换手下跌是自然调整",
    dependencies=("__factors__", "turnover_20", "mom_20"),
)
def factor_turnover_price_divergence(ctx: FactorContext) -> pd.Series:
    to = ctx.load_factor("turnover_20")
    mom = ctx.load_factor("mom_20")
    to_r = _rank(to)
    mom_r = _rank(mom)
    # High turnover + negative momentum = panic selling (negative)
    divergence = to_r * (1 - mom_r)
    return cross_sectional_rank(-divergence)

@register_factor(
    name="bp_roe_divergence",
    description="BP-ROE背离因子 (高BP+低ROE=价值陷阱)。",
    category="coupling",
    thesis="低PB+低ROE是价值陷阱的经典画像——便宜是因为基本面差，不是市场误定价",
    dependencies=("__factors__", "bp", "roe"),
)
def factor_bp_roe_divergence(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    roe = ctx.load_factor("roe")
    bp_r = _rank(bp)
    roe_r = _rank(roe)
    # High BP + low ROE = value trap (negative)
    # High BP + high ROE = genuine undervaluation (positive)
    divergence = bp_r * roe_r
    return cross_sectional_rank(divergence)

@register_factor(
    name="size_quality_divergence",
    description="规模-质量背离因子 (小市值+高ROE是优质成长, 大市值+低ROE是僵尸企业)。",
    category="coupling",
    thesis="小盘+高ROE是未来的大盘成长股，大盘+低ROE是成熟但低效的僵尸企业",
    dependencies=("__factors__", "log_circ_mv", "roe"),
)
def factor_size_quality_divergence(ctx: FactorContext) -> pd.Series:
    size = ctx.load_factor("log_circ_mv")
    roe = ctx.load_factor("roe")
    # log_circ_mv is negative (small ranks high), so we invert for clarity
    small_r = _rank(-size)
    roe_r = _rank(roe)
    # Small + high ROE = quality small cap (positive)
    divergence = small_r * roe_r
    return cross_sectional_rank(divergence)

@register_factor(
    name="vol_momentum_divergence",
    description="波动-动量背离因子 (低波+正动量=优质趋势, 高波+正动量=风险趋势)。",
    category="coupling",
    thesis="低波动率+高动量是最优质的趋势——涨得稳而不是涨得急",
    dependencies=("__factors__", "volatility_20", "mom_20"),
)
def factor_vol_momentum_divergence(ctx: FactorContext) -> pd.Series:
    vol = ctx.load_factor("volatility_20")
    mom = ctx.load_factor("mom_20")
    # volatility_20 is negative (low vol ranks high)
    low_vol_r = _rank(vol)
    mom_r = _rank(mom)
    # Low vol + strong momentum = quality trend
    return cross_sectional_rank(low_vol_r * mom_r)

@register_factor(
    name="fundflow_retail_inst_divergence",
    description="主力-散户资金背离因子 (大单净买+小单净卖=机构吸筹)。",
    category="coupling",
    thesis="大单(机构)买入而小单(散户)卖出是机构吸筹的清晰信号",
    dependencies=("__factors__", "mf_big_order_ratio", "mf_small_order_ratio"),
)
def factor_fundflow_retail_inst_divergence(ctx: FactorContext) -> pd.Series:
    big = ctx.load_factor("mf_big_order_ratio")
    small = ctx.load_factor("mf_small_order_ratio")
    big_r = _rank(big)
    small_r = _rank(-small)  # low small order ratio = less retail selling
    # Big buying + small selling = institutional accumulation
    divergence = big_r * small_r
    return cross_sectional_rank(divergence)

# ═══════════════════════════════════════════════════════════════════════════════
# C — Factor Resonance (因子共振)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="value_quality_resonance",
    description="价值-质量共振因子 (BP+ROE排名均值)。",
    category="coupling",
    thesis="便宜且高质量的公司在任何市场环境下都有安全边际，两个维度共振是最稳健的alpha",
    dependencies=("__factors__", "bp", "roe"),
)
def factor_value_quality_resonance(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    roe = ctx.load_factor("roe")
    return cross_sectional_rank((_rank(bp) + _rank(roe)) / 2.0)

@register_factor(
    name="trend_quality_resonance",
    description="趋势-质量共振因子 (动量+低波+ROE三维度均值)。",
    category="coupling",
    thesis="趋势、低波和高质量三者共振是高确定性的盈利机会——涨得稳、风险低、基本面好",
    dependencies=("__factors__", "mom_20", "volatility_20", "roe"),
)
def factor_trend_quality_resonance(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_20")
    vol = ctx.load_factor("volatility_20")
    roe = ctx.load_factor("roe")
    return cross_sectional_rank((_rank(mom) + _rank(vol) + _rank(roe)) / 3.0)

@register_factor(
    name="low_risk_value_resonance",
    description="低风险-价值共振因子 (低波+低回撤+高BP)。",
    category="coupling",
    thesis="便宜且风险低的股票是防御型投资者的理想标的——下行保护+估值回归双重收益",
    dependencies=("__factors__", "bp", "volatility_20", "max_drawdown_60"),
)
def factor_low_risk_value_resonance(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    vol = ctx.load_factor("volatility_20")
    dd = ctx.load_factor("max_drawdown_60")
    return cross_sectional_rank((_rank(bp) + _rank(vol) + _rank(dd)) / 3.0)

@register_factor(
    name="momentum_value_resonance",
    description="动量-价值共振因子 (趋势+便宜)。",
    category="coupling",
    thesis="价值与趋势的结合避免了纯价值因子的'价值陷阱'和纯动量因子的'追高'——买入正在上涨的便宜股票",
    dependencies=("__factors__", "bp", "mom_60"),
)
def factor_momentum_value_resonance(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_60")
    return cross_sectional_rank((_rank(bp) + _rank(mom)) / 2.0)

@register_factor(
    name="quality_growth_resonance",
    description="质量-成长共振因子 (ROE+营收增速)。",
    category="coupling",
    thesis="高ROE且高成长的股票是GARP策略的核心——以合理价格买入优质成长",
    dependencies=("__factors__", "roe", "or_yoy"),
)
def factor_quality_growth_resonance(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    growth = ctx.load_factor("or_yoy")
    return cross_sectional_rank((_rank(roe) + _rank(growth)) / 2.0)

@register_factor(
    name="chip_momentum_resonance",
    description="筹码-动量共振因子 (筹码CR3集中度+动量趋势)。",
    category="coupling",
    thesis="筹码集中叠加上涨动量是爆发力最强的组合——主力已控盘、趋势已形成",
    dependencies=("__factors__", "chip_cr3_factor", "mom_20"),
)
def factor_chip_momentum_resonance(ctx: FactorContext) -> pd.Series:
    chip = ctx.load_factor("chip_cr3_factor")
    mom = ctx.load_factor("mom_20")
    return cross_sectional_rank((_rank(chip) + _rank(mom)) / 2.0)

@register_factor(
    name="flow_momentum_resonance",
    description="资金-动量共振因子 (主力流入+价格动量)。",
    category="coupling",
    thesis="主力资金与价格双双走强是最强的买入信号——聪明钱与市场形成共振",
    dependencies=("__factors__", "mf_net_inflow_ratio", "mom_20"),
)
def factor_flow_momentum_resonance(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    mom = ctx.load_factor("mom_20")
    return cross_sectional_rank((_rank(mf) + _rank(mom)) / 2.0)

@register_factor(
    name="bull_factor_alignment_5",
    description="5因子多头一致性得分 (BP+ROE+MOM+低波+资金流>中位数的因子数)。",
    category="coupling",
    thesis="多个维度同时看多是高置信度的买入信号——单一因子可能错判，但多因子共振很少同时错",
    dependencies=("__factors__", "bp", "roe", "mom_20", "volatility_20", "mf_net_inflow_ratio"),
)
def factor_bull_factor_alignment_5(ctx: FactorContext) -> pd.Series:
    factors = ctx.load_factors(["bp", "roe", "mom_20", "volatility_20", "mf_net_inflow_ratio"])
    # Count how many factors rank above median (volatility inverted: low vol = good)
    score = pd.Series(0.0, index=factors.index)
    for col in ["bp", "roe", "mom_20", "mf_net_inflow_ratio"]:
        if col in factors.columns:
            score += (factors[col].groupby(level="Date").rank(pct=True) > 0.5).astype(float)
    if "volatility_20" in factors.columns:
        score += (factors["volatility_20"].groupby(level="Date").rank(pct=True) < 0.5).astype(float)
    return cross_sectional_rank(score)

@register_factor(
    name="factor_unanimity_8",
    description="8因子综合一致性得分 (8个核心因子同向比率)。",
    category="coupling",
    thesis="因子一致性是市场方向确信度的综合度量——所有因子同时指向同一方向是强信号",
    dependencies=("__factors__", "bp", "roe", "mom_20", "volatility_20",
                  "turnover_20", "mf_net_inflow_ratio", "winner_rate", "f_score"),
)
def factor_factor_unanimity_8(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20", "volatility_20", "turnover_20",
                    "mf_net_inflow_ratio", "winner_rate", "f_score"]
    factors = ctx.load_factors(factor_names)
    score = pd.Series(0.0, index=factors.index)
    positive_direction = {"bp", "roe", "mom_20", "mf_net_inflow_ratio", "f_score"}
    for col in factor_names:
        if col not in factors.columns:
            continue
        r = factors[col].groupby(level="Date").rank(pct=True)
        if col in positive_direction:
            score += (r > 0.5).astype(float)
        elif col == "volatility_20":
            score += (r < 0.5).astype(float)
        elif col == "turnover_20":
            score += (r < 0.5).astype(float)
        elif col == "winner_rate":
            score += (r < 0.5).astype(float)
    return cross_sectional_rank(score)

# ═══════════════════════════════════════════════════════════════════════════════
# D — Conditional Factors (条件因子)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# E — Factor Stability (因子稳定性)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="factor_rank_stability_20",
    description="因子排名20日稳定性 (bp/roe/mom三个排名的20日相关性均值, 高稳定排前)。",
    category="coupling",
    thesis="排名稳定的股票意味着其因子特征明确、不受短期噪音干扰——确定性溢价",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_factor_rank_stability_20(ctx: FactorContext) -> pd.Series:
    factors = ctx.load_factors(["bp", "roe", "mom_20"])
    stability = pd.Series(0.0, index=factors.index)

    for col in factors.columns:
        rank = factors[col].groupby(level="Date").rank(pct=True)
        # Rolling correlation of current rank with lagged rank
        rank_lag20 = rank.groupby(level="Code").shift(20)
        # Approximate stability: negative abs difference
        diff_abs = (rank - rank_lag20).abs()
        stability_col = -diff_abs.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).mean()
            )
        stability += stability_col.fillna(0)

    stability = stability / len(factors.columns)
    return cross_sectional_rank(stability)

@register_factor(
    name="factor_rank_volatility_60",
    description="因子排名60日波动率 (排名变动大=因子漂移风险, 低波动排前)。",
    category="coupling",
    thesis="因子排名高波动意味着股票的因子特征不稳定——今天是成长股，明天变价值股？",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_factor_rank_volatility_60(ctx: FactorContext) -> pd.Series:
    factors = ctx.load_factors(["bp", "roe", "mom_20"])
    vol_sum = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        rank = factors[col].groupby(level="Date").rank(pct=True)
        col_vol = rank.groupby(level="Code").transform(
            lambda s: s.rolling(60, min_periods=30).std()
            )
        vol_sum += col_vol.fillna(0)
    avg_vol = vol_sum / len(factors.columns)
    return cross_sectional_rank(-avg_vol)

@register_factor(
    name="factor_profile_shift_20",
    description="因子画像突变因子 (5因子排名20日变化的绝对值均值, 突变大排后)。",
    category="coupling",
    thesis="因子画像突然大幅变化意味着基本面或市场认知的重大变动，增加了不确定性",
    dependencies=("__factors__", "bp", "roe", "mom_20", "volatility_20", "turnover_20"),
)
def factor_factor_profile_shift_20(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20", "volatility_20", "turnover_20"]
    factors = ctx.load_factors(factor_names)
    shift_sum = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        rank = factors[col].groupby(level="Date").rank(pct=True)
        rank_lag20 = rank.groupby(level="Code").shift(20)
        shift = (rank - rank_lag20).abs()
        shift_sum += shift.fillna(0)
    avg_shift = shift_sum / len(factors.columns)
    return cross_sectional_rank(-avg_shift)

# ═══════════════════════════════════════════════════════════════════════════════
# F — Factor Timing / Regime (因子择时)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="value_factor_zscore_252",
    description="价值因子(BP)252日历史Z-score (相对自身历史的高估/低估)。",
    category="coupling",
    thesis="BP相对于自身历史水平处于高位时价值因子更有效——估值回复的引力更强",
    dependencies=("__factors__", "bp"),
)
def factor_value_factor_zscore_252(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")

    def _rolling_zscore(s, window):
        mean = s.rolling(window, min_periods=window // 2).mean()
        std = s.rolling(window, min_periods=window // 2).std()
        return safe_divide(s - mean, std + 1e-8)

    z = bp.groupby(level="Code").transform(lambda s: _rolling_zscore(s, 252))
    return cross_sectional_rank(z)

@register_factor(
    name="momentum_factor_zscore_60",
    description="动量因子60日历史Z-score (相对自身历史的动量强度)。",
    category="coupling",
    thesis="动量因子在其历史高位时趋势更确定——正反馈机制在加速",
    dependencies=("__factors__", "mom_20"),
)
def factor_momentum_factor_zscore_60(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_20")

    def _rolling_zscore(s, window):
        mean = s.rolling(window, min_periods=window // 2).mean()
        std = s.rolling(window, min_periods=window // 2).std()
        return safe_divide(s - mean, std + 1e-8)

    z = mom.groupby(level="Code").transform(lambda s: _rolling_zscore(s, 60))
    return cross_sectional_rank(z)

@register_factor(
    name="factor_crowding_proxy_20",
    description="因子拥挤度代理变量 (因子截面离散度/均值, 高拥挤排后)。",
    category="coupling",
    thesis="因子截面离散度下降意味着大家都在追逐同一批股票——拥挤交易容易踩踏",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_factor_crowding_proxy_20(ctx: FactorContext) -> pd.Series:
    factors = ctx.load_factors(["bp", "roe", "mom_20"])
    crowding = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        cv = factors[col].groupby(level="Date").transform(
            lambda s: s.std() / (pd.to_numeric(s, errors="coerce").abs().mean() + 1e-8)
            )
        crowding += cv.fillna(0)
    crowding = crowding / len(factors.columns)
    return cross_sectional_rank(-crowding)

@register_factor(
    name="factor_regime_change_60",
    description="因子格局长变化检测 (因子排名自相关60日变化, 格局变=风险)。",
    category="coupling",
    thesis="因子之间的相关结构发生突变意味着市场风格在切换——风格切换期的选股风险加大",
    dependencies=("__factors__", "bp", "roe", "mom_20", "volatility_20"),
)
def factor_factor_regime_change_60(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20", "volatility_20"]
    factors = ctx.load_factors(factor_names)

    ranks = pd.DataFrame(index=factors.index)
    for col in factors.columns:
        ranks[col] = factors[col].groupby(level="Date").rank(pct=True)

    # Compute pairwise rolling correlation difference (current vs 60d ago)
    regime_change = pd.Series(0.0, index=ranks.index)
    n_pairs = 0
    for i, c1 in enumerate(factors.columns):
        for c2 in factors.columns[i + 1:]:
            # Rank correlation proxy: Spearman via rank difference
            pair_corr = ranks[c1] * ranks[c2]  # product proxy for correlation
            corr_now = pair_corr.groupby(level="Code").transform(
                lambda s: s.rolling(20, min_periods=10).mean()
                )
            corr_old = corr_now.groupby(level="Code").shift(60)
            change = (corr_now - corr_old).abs()
            regime_change += change.fillna(0)
            n_pairs += 1

    if n_pairs > 0:
        regime_change = regime_change / n_pairs
    return cross_sectional_rank(-regime_change)

# ═══════════════════════════════════════════════════════════════════════════════
# G — Cross-Factor Residual (因子残差/正交)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="momentum_orthogonal_to_value",
    description="动量对价值正交化因子 (剔除价值成分的纯动量)。",
    category="coupling",
    thesis="剔除BP影响后的动量是更纯粹的趋势信号——不受估值变化的混淆",
    dependencies=("__factors__", "mom_20", "bp"),
)
def factor_momentum_orthogonal_to_value(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_20")
    bp = ctx.load_factor("bp")
    # Residual: mom - beta * bp, where beta = cov(date_cs) / var(date_cs)
    mom_z = _zscore(mom)
    bp_z = _zscore(bp)
    # Simple orthogonalization: subtract the projected component
    beta = (mom_z * bp_z).groupby(level="Date").transform("mean") / (
        (bp_z ** 2).groupby(level="Date").transform("mean") + 1e-8
    )
    residual = mom_z - beta * bp_z
    return cross_sectional_rank(residual)

@register_factor(
    name="quality_orthogonal_to_size",
    description="质量对规模正交化因子 (剔除规模效应的纯质量)。",
    category="coupling",
    thesis="ROE与小盘效应有天然的正相关性——剔除这个关系后的纯质量信号更干净",
    dependencies=("__factors__", "roe", "log_circ_mv"),
)
def factor_quality_orthogonal_to_size(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    size = ctx.load_factor("log_circ_mv")
    roe_z = _zscore(roe)
    size_z = _zscore(size)
    beta = (roe_z * size_z).groupby(level="Date").transform("mean") / (
        (size_z ** 2).groupby(level="Date").transform("mean") + 1e-8
    )
    residual = roe_z - beta * size_z
    return cross_sectional_rank(residual)

@register_factor(
    name="vol_orthogonal_to_beta",
    description="波动率对Beta正交化因子 (剔除市场敏感度的纯特质波动)。",
    category="coupling",
    thesis="剔除市场Beta后的波动率是更纯粹的异质风险——捕捉公司自身的不确定性",
    dependencies=("__factors__", "volatility_20", "beta_60"),
)
def factor_vol_orthogonal_to_beta(ctx: FactorContext) -> pd.Series:
    vol = ctx.load_factor("volatility_20")
    beta = ctx.load_factor("beta_60")
    vol_z = _zscore(vol)
    beta_z = _zscore(beta)
    b = (vol_z * beta_z).groupby(level="Date").transform("mean") / (
        (beta_z ** 2).groupby(level="Date").transform("mean") + 1e-8
    )
    residual = vol_z - b * beta_z
    return cross_sectional_rank(residual)

@register_factor(
    name="turnover_orthogonal_to_mv",
    description="换手率对市值正交化因子 (剔除规模效应的纯流动性)。",
    category="coupling",
    thesis="小盘股天然换手率高——剔除市值影响后的换手率才是真正的流动性偏好度量",
    dependencies=("__factors__", "turnover_20", "log_circ_mv"),
)
def factor_turnover_orthogonal_to_mv(ctx: FactorContext) -> pd.Series:
    to = ctx.load_factor("turnover_20")
    size = ctx.load_factor("log_circ_mv")
    to_z = _zscore(to)
    size_z = _zscore(size)
    b = (to_z * size_z).groupby(level="Date").transform("mean") / (
        (size_z ** 2).groupby(level="Date").transform("mean") + 1e-8
    )
    residual = to_z - b * size_z
    return cross_sectional_rank(residual)

# ═══════════════════════════════════════════════════════════════════════════════
# H — Coupling / Cross-Factor Spreads & Interactions
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="value_momentum_spread",
    description="价值动量价差因子，bp排名-mom_20排名截面排名（正=价值>动量=价值风格占优排前）。",
    category="enhanced",
    thesis="价值与动量的截面排名差异反映风格轮动——正值意味着股票在价值维度排名远高于动量维度(可能处于风格切换的早期阶段)。",
    dependencies=("__factors__", "bp", "mom_20"),
)
def factor_value_momentum_spread(context: FactorContext):
    df = context.load_factors(["bp", "mom_20"])
    bp_r = df["bp"].groupby(level="Date").rank(pct=True)
    mom_r = df["mom_20"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(bp_r - mom_r)

@register_factor(
    name="quality_value_bias",
    description="质量价值偏差因子，roe排名-bp排名截面排名（正=质量>价值=质量溢价排前）。",
    category="enhanced",
    thesis="高质量股票的价值折价程度——质量排名远高于价值排名时，可能市场低估了其质量（价值发现机会），但也可能是质量溢价过高。",
    dependencies=("__factors__", "roe", "bp"),
)
def factor_quality_value_bias(context: FactorContext):
    df = context.load_factors(["roe", "bp"])
    roe_r = df["roe"].groupby(level="Date").rank(pct=True)
    bp_r = df["bp"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(roe_r - bp_r)

@register_factor(
    name="momentum_volatility_ratio",
    description="动量效率比因子，mom_20排名/(1+volatility_20排名)截面排名（动量效率高排前）。",
    category="enhanced",
    thesis="动量的效率维度——动量高且波动低=单位波动撬动的价格变动大、趋势高效；动量高但波动大=噪音多、方向不确定。",
    dependencies=("__factors__", "mom_20", "volatility_20"),
)
def factor_momentum_volatility_ratio(context: FactorContext):
    df = context.load_factors(["mom_20", "volatility_20"])
    mom_r = df["mom_20"].groupby(level="Date").rank(pct=True)
    vol_r = df["volatility_20"].groupby(level="Date").rank(pct=True)
    efficiency = mom_r / (1 + vol_r)
    return cross_sectional_rank(efficiency)

@register_factor(
    name="value_reversal_crossover",
    description="价值反转交叉因子，bp排名+reversal_5排名截面排名（低估值+短期下跌=超跌价值排前）。",
    category="enhanced",
    thesis="低估值叠加短期下跌是经典的逆向价值买点——估值已便宜+短期又在跌，可能是最后的恐慌盘，均值回归概率高。",
    dependencies=("__factors__", "bp", "reversal_5"),
)
def factor_value_reversal_crossover(context: FactorContext):
    df = context.load_factors(["bp", "reversal_5"])
    bp_r = df["bp"].groupby(level="Date").rank(pct=True)
    rev_r = df["reversal_5"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(bp_r + rev_r)

@register_factor(
    name="fund_flow_price_resonance",
    description="资金价格共振因子，net_mf_amount_intensity排名×mom_20排名截面排名（资金+价格双强排前）。",
    category="enhanced",
    thesis="主力资金流向与价格趋势的同向共振——资金和价格同时发出做多信号时，确定性远高于单一信号。共振越强、趋势持续性越好。",
    dependencies=("__factors__", "net_mf_amount_intensity", "mom_20"),
)
def factor_fund_flow_price_resonance(context: FactorContext):
    df = context.load_factors(["net_mf_amount_intensity", "mom_20"])
    mf_r = df["net_mf_amount_intensity"].groupby(level="Date").rank(pct=True)
    mom_r = df["mom_20"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(mf_r * mom_r)

@register_factor(
    name="illusory_diversification",
    description="虚假分散因子，(sector_diversification排名×volatility_20排名)截面排名（分散但高波=虚假分散排后）。",
    category="enhanced",
    thesis="分散度与波动的交互——在多个行业分散但波动依然高的公司，所谓的'分散'可能是低质量业务拼凑的假象，而非真正的风险分散。",
    dependencies=("__factors__", "sector_diversification", "volatility_20"),
)
def factor_illusory_diversification(context: FactorContext):
    df = context.load_factors(["sector_diversification", "volatility_20"])
    div_r = df["sector_diversification"].groupby(level="Date").rank(pct=True)
    vol_r = df["volatility_20"].groupby(level="Date").rank(pct=True)
    illusion = div_r * (1 - vol_r)
    return cross_sectional_rank(illusion)

@register_factor(
    name="chip_momentum_confirmation",
    description="筹码动量确认因子，chip_cr3_factor排名×mom_20排名截面排名（筹码集中+动量=强烈趋势排前）。",
    category="enhanced",
    thesis="筹码集中且价格动量向上是最强的趋势信号——筹码集中意味着主力控盘，价格动量意味着趋势已启动，两者共振预示着主升浪。",
    dependencies=("__factors__", "chip_cr3_factor", "mom_20"),
)
def factor_chip_momentum_confirmation(context: FactorContext):
    df = context.load_factors(["chip_cr3_factor", "mom_20"])
    chip_r = df["chip_cr3_factor"].groupby(level="Date").rank(pct=True)
    mom_r = df["mom_20"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(chip_r * mom_r)

@register_factor(
    name="small_quality_premium",
    description="小盘质量溢价因子，log_circ_mv排名×roe排名截面排名（小市值+高质量=高成长空间排前）。",
    category="enhanced",
    thesis="小盘+高质量是最佳组合——小市值提供成长空间、高质量提供安全边际。但小盘股通常质量参差，高质量的小盘股是alpha矿。",
    dependencies=("__factors__", "log_circ_mv", "roe"),
)
def factor_small_quality_premium(context: FactorContext):
    df = context.load_factors(["log_circ_mv", "roe"])
    size_r = (1 - df["log_circ_mv"].groupby(level="Date").rank(pct=True))
    roe_r = df["roe"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(size_r * roe_r)

@register_factor(
    name="turnover_shock_20",
    description="换手率异动因子，换手率20日Z-score截面排名（异常高换手排后）。",
    category="enhanced",
    thesis="换手率突然飙升(偏离正常水平多个标准差)通常伴随事件驱动——可能是机构出货、消息泄露或市场关注度突然上升。异常换手后的方向取决于配合的价格信号。",
    dependencies=("__factors__", "turnover_20"),
)
def factor_turnover_shock_20(context: FactorContext):
    df = context.load_factors(["turnover_20"])
    to = df["turnover_20"]
    to_mean = to.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=20).mean())
    to_std = to.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=20).std())
    zscore = (to - to_mean) / to_std.replace(0, np.nan)
    return cross_sectional_rank(-zscore.abs())

@register_factor(
    name="profitability_size_interaction",
    description="盈利规模交互因子，roe排名×log_total_mv排名截面排名（大市值+高盈利=龙头质量排前）。",
    category="enhanced",
    thesis="大市值+高盈利=行业龙头+盈利能力强——ROE与规模的交叉捕捉的是'盈利的规模效应'：大公司中只有ROE高的才是真正的龙头，小公司高ROE可能是阶段性的。",
    dependencies=("__factors__", "roe", "log_total_mv"),
)
def factor_profitability_size_interaction(context: FactorContext):
    df = context.load_factors(["roe", "log_total_mv"])
    roe_r = df["roe"].groupby(level="Date").rank(pct=True)
    mv_r = df["log_total_mv"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(roe_r * mv_r)

@register_factor(
    name="reversal_vol_coupling",
    description="反转波动耦合因子，reversal_5排名×(1-volatility_20排名)截面排名（低波动下的反转更可靠排前）。",
    category="enhanced",
    thesis="低波动环境下的反转信号更可靠——高波动中的反转可能只是噪音(随机游走)，低波动中的趋势反转才是真正的信号转变。",
    dependencies=("__factors__", "reversal_5", "volatility_20"),
)
def factor_reversal_vol_coupling(context: FactorContext):
    df = context.load_factors(["reversal_5", "volatility_20"])
    rev_r = df["reversal_5"].groupby(level="Date").rank(pct=True)
    vol_r = df["volatility_20"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(rev_r * (1 - vol_r))

@register_factor(
    name="bp_vol_interaction",
    description="估值波动交互因子，bp排名/(1+volatility_20排名)截面排名。",
    category="enhanced",
    thesis="低估值+低波动是防御型价值——估值便宜且价格稳定，适合作为组合的压舱石。高估值+高波动是进攻型成长——价格活跃但估值贵。",
    dependencies=("__factors__", "bp", "volatility_20"),
)
def factor_bp_vol_interaction(context: FactorContext):
    df = context.load_factors(["bp", "volatility_20"])
    bp_r = df["bp"].groupby(level="Date").rank(pct=True)
    vol_r = df["volatility_20"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(bp_r / (1 + vol_r))

@register_factor(
    name="multi_factor_consensus",
    description="多因子共识度因子，[价值+动量+质量+低波]四个维度排名的最小值截面排名（全方位优质排前）。",
    category="enhanced",
    thesis="多因子策略的核心困境是各因子信号方向不一致——共识度因子取各维度排名的最小值，确保选出的股票在主要维度上不存在明显短板。",
    dependencies=("__factors__", "bp", "mom_20", "roe", "volatility_20"),
)
def factor_multi_factor_consensus(context: FactorContext):
    df = context.load_factors(["bp", "mom_20", "roe", "volatility_20"])
    bp_r = df["bp"].groupby(level="Date").rank(pct=True)
    mom_r = df["mom_20"].groupby(level="Date").rank(pct=True)
    roe_r = df["roe"].groupby(level="Date").rank(pct=True)
    vol_r = 1 - df["volatility_20"].groupby(level="Date").rank(pct=True)
    consensus = pd.concat([bp_r, mom_r, roe_r, vol_r], axis=1).min(axis=1)
    return cross_sectional_rank(consensus)

@register_factor(
    name="momentum_reversal_balance",
    description="动量反转平衡因子，mom_20排名-reversal_5排名截面排名。",
    category="enhanced",
    thesis="动量与反转的平衡反映趋势的成熟度——正向(动量>反转)=趋势仍强、中期趋势未到反转阶段；负向(反转>动量)=趋势可能已到末端、反转压力上升。",
    dependencies=("__factors__", "mom_20", "reversal_5"),
)
def factor_momentum_reversal_balance(context: FactorContext):
    df = context.load_factors(["mom_20", "reversal_5"])
    mom_r = df["mom_20"].groupby(level="Date").rank(pct=True)
    rev_r = df["reversal_5"].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(mom_r - rev_r)

@register_factor(
    name="depth_quality_composite",
    description="深度质量综合因子，(roe+roa+gross_margin+ocf_to_profit+asset_turn-accruals)五正一负等权截面排名。",
    category="enhanced",
    thesis="五正一负的质量评估框架——ROE+ROA+毛利率+现金流质量+资产周转-应计利润。正面维度越多越好，负面维度越低越好。比quality_composite(四正)多覆盖了周转和应计两个维度。",
    dependencies=("__factors__", "roe", "roa", "gross_margin", "ocf_to_profit", "assets_turn", "accruals_ratio"),
)
def factor_depth_quality_composite(context: FactorContext):
    df = context.load_factors(["roe", "roa", "gross_margin", "ocf_to_profit", "assets_turn", "accruals_ratio"])
    roe_r = df["roe"].groupby(level="Date").rank(pct=True)
    roa_r = df["roa"].groupby(level="Date").rank(pct=True)
    gm_r = df["gross_margin"].groupby(level="Date").rank(pct=True)
    ocf_r = df["ocf_to_profit"].groupby(level="Date").rank(pct=True)
    at_r = df["assets_turn"].groupby(level="Date").rank(pct=True)
    acc_r = 1 - df["accruals_ratio"].groupby(level="Date").rank(pct=True)
    composite = (roe_r + roa_r + gm_r + ocf_r + at_r + acc_r) / 6.0
    return cross_sectional_rank(composite)

# ═══════════════════════════════════════════════════════════════════════════════
# I — Factor Lead-Lag / Information Flow (因子领先滞后)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="fund_flow_lead_momentum_10",
    description="资金流领先动量因子，滞后10日的主力资金排名与当前动量排名的对齐度（资金先行排前）。",
    category="coupling",
    thesis="主力资金流向领先价格动量约10个交易日——聪明钱先布局，价格后反应。资金流入领先于价格上涨的股票正在被机构建仓。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "mom_20"),
)
def factor_fund_flow_lead_momentum_10(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    mom = ctx.load_factor("mom_20")
    mf_rank = mf.groupby(level="Date").rank(pct=True)
    mom_rank = mom.groupby(level="Date").rank(pct=True)
    mf_lag10 = mf_rank.groupby(level="Code").shift(10)
    # High when past fund flow was strong AND current momentum is strong (flow led price)
    lead_score = mf_lag10 * mom_rank
    return cross_sectional_rank(lead_score)

@register_factor(
    name="chip_lead_momentum_10",
    description="筹码领先动量因子，滞后10日获利盘变化与当前动量的一致性（筹码改善领先价格排前）。",
    category="coupling",
    thesis="筹码结构的改善领先于价格趋势的形成——获利盘比例从低位回升意味着主力完成洗盘、筹码重新集中，随后价格启动。",
    dependencies=("__factors__", "winner_rate", "mom_20"),
)
def factor_chip_lead_momentum_10(ctx: FactorContext) -> pd.Series:
    wr = ctx.load_factor("winner_rate")
    mom = ctx.load_factor("mom_20")
    wr_change = wr.groupby(level="Code").diff(5)
    wr_improving = wr_change.groupby(level="Date").rank(pct=True)
    wr_lag10 = wr_improving.groupby(level="Code").shift(10)
    mom_rank = mom.groupby(level="Date").rank(pct=True)
    lead_score = wr_lag10 * mom_rank
    return cross_sectional_rank(lead_score)

@register_factor(
    name="turnover_lead_volatility_10",
    description="换手领先波动因子，滞后10日换手率变化与当前波动率变化的一致性（换手预示波动排后=风险预警排前）。",
    category="coupling",
    thesis="换手率的异常变化领先于波动率的扩张——换手先放大、波动后跟上。高换手领先于高波动时是风险预警信号(排后)，低换手预示低波动是稳定信号(排前)。",
    dependencies=("__factors__", "turnover_20", "volatility_20"),
)
def factor_turnover_lead_volatility_10(ctx: FactorContext) -> pd.Series:
    to = ctx.load_factor("turnover_20")
    vol = ctx.load_factor("volatility_20")
    to_change = to.groupby(level="Code").diff(10)
    to_change_r = to_change.groupby(level="Date").rank(pct=True)
    to_lag10 = to_change_r.groupby(level="Code").shift(10)
    vol_change = vol.groupby(level="Code").diff(10)
    vol_change_r = vol_change.groupby(level="Date").rank(pct=True)
    # When past turnover surge predicts current volatility surge = risk warning (rank low)
    lead_risk = to_lag10 * vol_change_r
    return cross_sectional_rank(-lead_risk)

# ═══════════════════════════════════════════════════════════════════════════════
# J — Asymmetric Factor Response (非对称因子响应)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="bp_down_market_defense",
    description="BP下行市场防御因子，在下跌市场中BP因子的相对强度（熊市防御力排前）。",
    category="coupling",
    thesis="低估值股票在市场下跌时具有天然的下行保护——已经便宜的股票在恐慌中再跌的空间有限，BP在下行市场中的防御价值是价值因子的重要alpha来源。",
    dependencies=("__factors__", "bp", "mom_20"),
)
def factor_bp_down_market_defense(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")
    bp_rank = bp.groupby(level="Date").rank(pct=True)
    # mom_20's cross-sectional mean proxies market direction
    mkt_return = mom.groupby(level="Date").transform("mean")
    # Weight BP more when market is down (negative market return)
    down_market_weight = (-mkt_return).clip(lower=0)
    down_market_weight_r = down_market_weight.groupby(level="Date").rank(pct=True)
    defense_score = bp_rank * (0.5 + 0.5 * down_market_weight_r)
    return cross_sectional_rank(defense_score)

@register_factor(
    name="quality_bear_market_alpha",
    description="质量熊市Alpha因子，高波动(熊市)环境下ROE质量溢价（熊市质量溢价排前）。",
    category="coupling",
    thesis="高ROE公司在市场高波动(通常伴随下跌)期间获得质量溢价——资金在恐慌时弃劣留优、涌向盈利确定性强的公司。质量因子的alpha在波动率高的时期更显著。",
    dependencies=("__factors__", "roe", "volatility_20"),
)
def factor_quality_bear_market_alpha(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    vol = ctx.load_factor("volatility_20")
    roe_rank = roe.groupby(level="Date").rank(pct=True)
    vol_rank = vol.groupby(level="Date").rank(pct=True)
    # Quality gets stronger bid in high vol environment; amplify roe_rank by vol_rank
    bear_alpha = roe_rank * (0.3 + 0.7 * vol_rank)
    return cross_sectional_rank(bear_alpha)

@register_factor(
    name="mom_up_down_asymmetry",
    description="动量非对称因子，上行动量与下行动量的差值（上行强+下行弱=趋势质量高排前）。",
    category="coupling",
    thesis="动量效应的非对称性——好的动量是涨得多跌得少(强趋势)，坏的动量是涨得少跌得多(弱趋势)。上行动量与下行动量的差异度量趋势的内在质量。",
    dependencies=("__factors__", "up_volatility_20", "down_volatility_20"),
)
def factor_mom_up_down_asymmetry(ctx: FactorContext) -> pd.Series:
    up_vol = ctx.load_factor("up_volatility_20")
    down_vol = ctx.load_factor("down_volatility_20")
    up_r = _rank(up_vol)
    down_r = _rank(down_vol)
    # High up_vol + low down_vol = strong bullish asymmetry
    asymmetry = up_r - down_r
    return cross_sectional_rank(asymmetry)

@register_factor(
    name="low_vol_tail_hedge",
    description="低波动尾部对冲因子，低波动+低回撤的尾部保护综合得分（极端风险保护力排前）。",
    category="coupling",
    thesis="低波动且低回撤的股票在市场尾部事件中提供最佳保护——波动低意味着日常风险小，回撤低意味着极端事件中的韧性。两者结合是尾部对冲的最优标的特征。",
    dependencies=("__factors__", "volatility_20", "max_drawdown_60"),
)
def factor_low_vol_tail_hedge(ctx: FactorContext) -> pd.Series:
    vol = ctx.load_factor("volatility_20")
    dd = ctx.load_factor("max_drawdown_60")
    vol_r = vol.groupby(level="Date").rank(pct=True)
    dd_r = dd.groupby(level="Date").rank(pct=True)
    # Both low vol and low drawdown are good — higher score = better tail hedge
    hedge_score = (1 - vol_r) * (1 - dd_r)
    return cross_sectional_rank(hedge_score)

# ═══════════════════════════════════════════════════════════════════════════════
# K — Higher-Order Factor Interactions (高阶因子交互)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="value_quality_safety_triple",
    description="价值质量安全三因子筛选，(BP排名×ROE排名×(1-资产负债率排名))截面排名（三重安全边际排前）。",
    category="coupling",
    thesis="便宜+高质量+低杠杆是最稳健的投资三角——BP提供估值保护、ROE提供盈利确认、低杠杆提供财务安全。三个维度同时满足的股票在三重安全边际下具备最强的alpha持续性。",
    dependencies=("__factors__", "bp", "roe", "debt_to_assets"),
)
def factor_value_quality_safety_triple(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    roe = ctx.load_factor("roe")
    debt = ctx.load_factor("debt_to_assets")
    bp_r = _rank(bp)
    roe_r = _rank(roe)
    debt_r = _rank(debt)  # low debt ranks high
    triple = bp_r * roe_r * debt_r
    return cross_sectional_rank(triple)

@register_factor(
    name="momentum_quality_liquidity_triple",
    description="动量质量流动性三因子筛选，动量排名×ROE排名×(1-高换手惩罚)截面排名（高质量流动性趋势排前）。",
    category="coupling",
    thesis="动量+质量+适度流动性的三维筛选——有基本面支撑的趋势且换手不过热(非投机性交易)是最健康的上涨形态。过度换手伴随的趋势可能只是短线资金博弈。",
    dependencies=("__factors__", "mom_20", "roe", "turnover_20"),
)
def factor_momentum_quality_liquidity_triple(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_20")
    roe = ctx.load_factor("roe")
    turnover = ctx.load_factor("turnover_20")
    mom_r = _rank(mom)
    roe_r = _rank(roe)
    to_r = _rank(turnover)
    # Penalize extreme turnover (top 20% = likely speculative)
    liq_penalty = 1.0 - 0.5 * to_r.clip(upper=0.8)
    triple = mom_r * roe_r * liq_penalty
    return cross_sectional_rank(triple)

@register_factor(
    name="chip_flow_momentum_triple",
    description="筹码资金动量三因子共振，(获利盘×主力资金×动量)三维度乘积截面排名。",
    category="coupling",
    thesis="筹码健康+资金流入+价格动量是最强的三重确认信号——筹码结构好(主力锁仓)、资金在买、价格在涨，三者共振的股票往往处于主升浪阶段。",
    dependencies=("__factors__", "winner_rate", "mf_net_inflow_ratio", "mom_20"),
)
def factor_chip_flow_momentum_triple(ctx: FactorContext) -> pd.Series:
    wr = ctx.load_factor("winner_rate")
    mf = ctx.load_factor("mf_net_inflow_ratio")
    mom = ctx.load_factor("mom_20")
    wr_r = _rank(wr)
    mf_r = _rank(mf)
    mom_r = _rank(mom)
    triple = wr_r * mf_r * mom_r
    return cross_sectional_rank(triple)

@register_factor(
    name="growth_value_quality_golden",
    description="成长价值质量金三角因子，(营收增速+BP+ROE)三个排名均值截面排名。",
    category="coupling",
    thesis="成长性+低估值+高质量是投资的金三角——成长提供空间、价值提供安全边际、质量提供确定性。三者兼顾的GARP策略在A股长周期中表现最优。",
    dependencies=("__factors__", "or_yoy", "bp", "roe"),
)
def factor_growth_value_quality_golden(ctx: FactorContext) -> pd.Series:
    growth = ctx.load_factor("or_yoy")
    bp = ctx.load_factor("bp")
    roe = ctx.load_factor("roe")
    golden = (_rank(growth) + _rank(bp) + _rank(roe)) / 3.0
    return cross_sectional_rank(golden)

# ═══════════════════════════════════════════════════════════════════════════════
# L — Factor Time-Series Dynamics (因子时序动态)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="factor_autocorrelation_break",
    description="因子自相关断裂因子，核心因子排名自相关20日变化（自相关下降=因子规律被打破排后）。",
    category="coupling",
    thesis="因子自身的时序自相关结构突然断裂意味着股票的因子特征发生了本质变化——比如从价值股变为成长股、从低波动变为高波动。自相关断裂是因子失效的预警信号。",
    dependencies=("__factors__", "bp", "roe", "mom_20", "volatility_20"),
)
def factor_factor_autocorrelation_break(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20", "volatility_20"]
    factors = ctx.load_factors(factor_names)
    break_score = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        rank = factors[col].groupby(level="Date").rank(pct=True)
        rank_lag = rank.groupby(level="Code").shift(5)
        # Rolling rank autocorrelation proxy: negative abs diff over rolling window
        abs_diff = (rank - rank_lag).abs()
        rolling_instability = abs_diff.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).mean()
            )
        # Current vs past instability — break = sudden increase in instability
        instability_change = rolling_instability - rolling_instability.groupby(level="Code").shift(20)
        break_score += instability_change.fillna(0)
    break_score = break_score / len(factors.columns)
    return cross_sectional_rank(-break_score)

@register_factor(
    name="factor_trend_exhaustion",
    description="因子趋势衰竭因子，因子20日变化率的减速信号（趋势衰竭=即将反转排后）。",
    category="coupling",
    thesis="因子的变化速度在放缓意味着因子的趋势方向可能即将反转——就像价格上涨减速是顶部的技术信号一样，因子值的二阶导(加速度)转负是因子趋势衰竭的前兆。",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_factor_trend_exhaustion(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20"]
    factors = ctx.load_factors(factor_names)
    exhaustion = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        # First derivative: 5-day change
        velocity = factors[col].groupby(level="Code").diff(5)
        # Second derivative: change in velocity (acceleration)
        acceleration = velocity.groupby(level="Code").diff(10)
        accel_rank = acceleration.groupby(level="Date").rank(pct=True)
        exhaustion += (1 - accel_rank).fillna(0)
    exhaustion = exhaustion / len(factors.columns)
    return cross_sectional_rank(exhaustion)

@register_factor(
    name="factor_rolling_drawdown_60",
    description="因子滚动回撤因子，BP因子从60日高点的回撤程度（回撤大=因子超跌排前）。",
    category="coupling",
    thesis="因子值从自身近期高点的回撤可能意味着因子被过度抛售——当BP因子(估值)回撤到极端位置时，往往对应着市场对该股票的过度悲观，是逆向买入的时机。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_rolling_drawdown_60(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")

    def _rolling_dd(s: pd.Series) -> pd.Series:
        rolling_max = s.rolling(60, min_periods=30).max()
        dd = (s - rolling_max) / rolling_max.replace(0, np.nan)
        return dd

    bp_dd = bp.groupby(level="Code").transform(_rolling_dd)
    # Large drawdown = factor oversold = potential bounce (rank high)
    return cross_sectional_rank(bp_dd)

@register_factor(
    name="factor_mean_reversion_signal",
    description="因子均值回复信号，BP因子偏离252日均值的Z-score（偏离大=均值回复力强排前）。",
    category="coupling",
    thesis="因子值围绕其长期均值波动——大幅偏离后向均值回复是统计规律。BP因子偏离长期均值越大，均值回复的引力越强，捕捉的是因子层面的统计套利机会。",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_factor_mean_reversion_signal(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20"]
    factors = ctx.load_factors(factor_names)
    mr_score = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        s = factors[col]

        def _deviation_from_ma(x: pd.Series) -> pd.Series:
            ma = x.rolling(252, min_periods=60).mean()
            std = x.rolling(252, min_periods=60).std()
            return safe_divide(x - ma, std + 1e-8)

        z = s.groupby(level="Code").transform(_deviation_from_ma)
        mr_score += z.abs().fillna(0)
    mr_score = mr_score / len(factors.columns)
    return cross_sectional_rank(mr_score)

# ═══════════════════════════════════════════════════════════════════════════════
# M — Cross-Sectional Factor Structure (因子截面结构)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="factor_cs_tail_ratio",
    description="因子截面尾部比率因子，因子排名90分位/10分位值的比率（尾部发散=极端分化排后）。",
    category="coupling",
    thesis="因子排名的尾部比率反映了截面极端分化程度——当因子值在头部和尾部极度发散时，表明市场在剧烈定价分化，选股确定性下降。尾部收敛时因子的选股能力更可靠。",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_factor_cs_tail_ratio(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20"]
    factors = ctx.load_factors(factor_names)
    # Per-stock tail contribution: how far is each stock's rank from the median?
    # Stocks consistently at distribution extremes get high tail contribution scores.
    tail_score = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        rank = factors[col].groupby(level="Date").rank(pct=True)
        # Per-stock distance from median: stocks near 0 or 1 are in the tails
        dist_from_median = (rank - 0.5).abs()
        tail_score += dist_from_median.fillna(0)
    tail_score = tail_score / len(factors.columns)
    # High tail score = stock at factor distribution extremes across dimensions
    # This means the stock is being extremely priced on multiple factors
    # (both positive and negative extremes possible — rank this low as
    #  factor consensus is unclear when a stock is at one extreme on some
    #  factors and another extreme on others)
    return cross_sectional_rank(-tail_score)

@register_factor(
    name="factor_rank_bimodality",
    description="因子排名双峰性因子，因子排名的双峰程度（双峰=市场分裂排后）。",
    category="coupling",
    thesis="因子截面排名的双峰分布意味着市场在将该因子分裂为两个极端阵营——例如一半股票极度高估、一半极度低估。双峰越明显，因子的线性选股能力越弱，需要非线性的long-short策略。",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_factor_rank_bimodality(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20"]
    factors = ctx.load_factors(factor_names)
    # Per-stock bimodality: stock's own distance from median averaged across factors.
    # Stocks at extremes (near rank 0 or 1) across multiple factors contribute to 
    # distribution bimodality. High bimodality = stock being polarized on many dimensions.
    bimodality = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        rank = factors[col].groupby(level="Date").rank(pct=True)
        dist_from_median = (rank - 0.5).abs()
        bimodality += dist_from_median.fillna(0)
    bimodality = bimodality / len(factors.columns)
    # Bimodality = stock at extremes across dimensions = market polarization signal
    # Rank low (negative): stocks in the middle of distributions are preferred
    return cross_sectional_rank(-bimodality)

@register_factor(
    name="factor_pair_correlation_dispersion",
    description="因子对相关性离散因子，核心因子对20日相关性的离散度（相关离散=因子结构不稳定排后）。",
    category="coupling",
    thesis="因子之间相关关系的稳定性反映了市场定价结构的一致性——当因子对相关性大幅发散时，意味着市场对不同定价维度的关系在重构(风格切换前兆)，选股不确定性上升。",
    dependencies=("__factors__", "bp", "roe", "mom_20", "volatility_20"),
)
def factor_factor_pair_correlation_dispersion(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20", "volatility_20"]
    factors = ctx.load_factors(factor_names)
    ranks = pd.DataFrame(index=factors.index)
    for col in factors.columns:
        ranks[col] = factors[col].groupby(level="Date").rank(pct=True)

    # Compute all pairwise rolling rank correlations (proxied by rank product stability)
    corr_disp = pd.Series(0.0, index=ranks.index)
    n_pairs = 0
    for i, c1 in enumerate(factors.columns):
        for c2 in factors.columns[i + 1:]:
            pair_prod = ranks[c1] * ranks[c2]
            pair_ma = pair_prod.groupby(level="Code").transform(
                lambda s: s.rolling(20, min_periods=10).mean()
                )
            pair_std = pair_prod.groupby(level="Code").transform(
                lambda s: s.rolling(20, min_periods=10).std()
                )
            # Dispersion = std/abs(mean) for each pair
            pair_disp = safe_divide(pair_std, pair_ma.abs() + 1e-8)
            corr_disp += pair_disp.fillna(0)
            n_pairs += 1
    if n_pairs > 0:
        corr_disp = corr_disp / n_pairs
    return cross_sectional_rank(-corr_disp)

# ═══════════════════════════════════════════════════════════════════════════════
# N — Factor Efficiency (因子效率)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="factor_rolling_ir_proxy",
    description="因子滚动信息比代理因子，因子60日变化均值/60日变化标准差（因子趋势效率排前）。",
    category="coupling",
    thesis="因子自身的风险调整后趋势强度——因子值在一段时间内的变化除以其波动率，类似因子层面的信息比率(IR)。高IR意味着因子的变化方向明确且稳定、非随机游走。",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_factor_rolling_ir_proxy(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20"]
    factors = ctx.load_factors(factor_names)
    ir_score = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        # 5-day change as the "return" of the factor
        delta = factors[col].groupby(level="Code").diff(5)
        mean_delta = delta.groupby(level="Code").transform(
            lambda s: s.rolling(60, min_periods=30).mean()
            )
        std_delta = delta.groupby(level="Code").transform(
            lambda s: s.rolling(60, min_periods=30).std()
            )
        ir = safe_divide(mean_delta, std_delta + 1e-8)
        ir_score += ir.fillna(0)
    ir_score = ir_score / len(factors.columns)
    return cross_sectional_rank(ir_score)

@register_factor(
    name="factor_consistency_ratio",
    description="因子一致性比率因子，BP因子60日变化方向一致的天数占比（持续方向=高信度排前）。",
    category="coupling",
    thesis="因子变化方向的持续性是因子信度的度量——如果BP因子连续多日在上升(估值在修复)，上升趋势比忽上忽下的震荡走势更可信。一致性比率高的因子信号更可靠。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_consistency_ratio(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    bp_delta = bp.groupby(level="Code").diff(1)

    def _consistency(s: pd.Series) -> pd.Series:
        # Direction = sign of change
        direction = np.sign(s)
        # Rolling count of positive minus negative
        net_direction = direction.rolling(60, min_periods=30).sum()
        count = direction.abs().rolling(60, min_periods=30).sum()
        return safe_divide(net_direction.abs(), count + 1e-8)

    consistency = bp_delta.groupby(level="Code").transform(_consistency)
    return cross_sectional_rank(consistency)

@register_factor(
    name="factor_gain_to_pain_ratio",
    description="因子盈亏比因子，BP因子60日累计正向变化/累计负向变化绝对值（正向主导排前）。",
    category="coupling",
    thesis="因子变化的盈亏比——因子在向好方向上的累计变化幅度除以向差方向上的累计变化幅度。盈亏比高意味着因子的正向变化远大于负向变化，因子正处於有力的改善趋势中。",
    dependencies=("__factors__", "bp", "roe"),
)
def factor_factor_gain_to_pain_ratio(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe"]
    factors = ctx.load_factors(factor_names)
    gpr_score = pd.Series(0.0, index=factors.index)
    for col in factors.columns:
        delta = factors[col].groupby(level="Code").diff(1)

        def _gpr(s: pd.Series) -> pd.Series:
            pos = s.clip(lower=0).rolling(60, min_periods=30).sum()
            neg = s.clip(upper=0).abs().rolling(60, min_periods=30).sum()
            return safe_divide(pos, neg + 1e-8)

        gpr = delta.groupby(level="Code").transform(_gpr)
        gpr_score += gpr.fillna(0)
    gpr_score = gpr_score / len(factors.columns)
    return cross_sectional_rank(gpr_score)

# ═══════════════════════════════════════════════════════════════════════════════
# O — Stress & Tail Event Response (压力与尾部事件)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="multi_factor_stress_beta",
    description="多因子压力Beta因子，(beta_60+波动率+回撤+尾部风险)四维度综合压力敏感度（压力敏感度低排前）。",
    category="coupling",
    thesis="股票对市场压力的敏感度可以通过多个风险维度综合评估——高Beta+高波动+大回撤+高尾部风险的股票在市场承压时首当其冲。低压力敏感度的股票是组合的压舱石。",
    dependencies=("__factors__", "beta_60", "volatility_20", "max_drawdown_60", "tail_risk_dd_20"),
)
def factor_multi_factor_stress_beta(ctx: FactorContext) -> pd.Series:
    beta = ctx.load_factor("beta_60")
    vol = ctx.load_factor("volatility_20")
    dd = ctx.load_factor("max_drawdown_60")
    tail = ctx.load_factor("tail_risk_dd_20")
    # All four are negative indicators (higher = riskier)
    stress = (_rank(beta) + _rank(vol) + _rank(dd) + _rank(tail)) / 4.0
    return cross_sectional_rank(-stress)

@register_factor(
    name="factor_liquidity_fragility",
    description="因子流动性脆弱性因子，(高换手+高非流动性+高波动)三维度流动性压力得分（流动性脆弱排后）。",
    category="coupling",
    thesis="高换手率但低流动性(高Amihud)的股票存在流动性幻觉——交易活跃但每笔交易对价格冲击大，在流动性危机中容易暴跌。流动性脆弱性高的股票在市场流动性枯竭时最危险。",
    dependencies=("__factors__", "turnover_20", "amihud_illiq_20", "volatility_20"),
)
def factor_factor_liquidity_fragility(ctx: FactorContext) -> pd.Series:
    turnover = ctx.load_factor("turnover_20")
    illiq = ctx.load_factor("amihud_illiq_20")
    vol = ctx.load_factor("volatility_20")
    # High turnover + high illiquidity + high vol = liquidity fragility
    fragility = (_rank(turnover) + _rank(illiq) + _rank(vol)) / 3.0
    return cross_sectional_rank(-fragility)

@register_factor(
    name="factor_crash_resilience",
    description="因子崩盘韧性因子，(低Beta+低回撤+低杠杆+低波动)四维度韧性得分（抗崩盘力排前）。",
    category="coupling",
    thesis="低Beta、低回撤、低杠杆、低波动的四低组合是极端市场环境中的避风港——这四个维度共同构成股票在崩盘中的韧性画像。韧性高的股票在市场崩盘后恢复更快。",
    dependencies=("__factors__", "beta_60", "max_drawdown_60", "debt_to_assets", "volatility_20"),
)
def factor_factor_crash_resilience(ctx: FactorContext) -> pd.Series:
    beta = ctx.load_factor("beta_60")
    dd = ctx.load_factor("max_drawdown_60")
    debt = ctx.load_factor("debt_to_assets")
    vol = ctx.load_factor("volatility_20")
    # All four are negative indicators — invert for resilience score
    resilience = ((1 - _rank(beta)) + (1 - _rank(dd)) +
                  (1 - _rank(debt)) + (1 - _rank(vol))) / 4.0
    return cross_sectional_rank(resilience)

# ═══════════════════════════════════════════════════════════════════════════════
# P — Factor Cross-Sectional Interaction Depth (因子截面交互深度)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="value_lowvol_nonlinear",
    description="价值低波非线性耦合因子，BP排名×(1-波动率排名)的平方根（非线性交互捕获极端组合排前）。",
    category="coupling",
    thesis="价值与低波动的交互是非线性的——极端便宜+极端低波的组合(双重极端)的alpha远超两者的简单加总。非线性耦合(乘积开方)放大了双重极端值的影响，捕获的是'深度价值+深度低波'的稀缺组合。",
    dependencies=("__factors__", "bp", "volatility_20"),
)
def factor_value_lowvol_nonlinear(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    vol = ctx.load_factor("volatility_20")
    bp_r = _rank(bp)
    low_vol_r = 1 - _rank(vol)
    # Geometric coupling — emphasizes joint extremes
    nonlinear = np.sqrt(bp_r * low_vol_r + 1e-8)
    return cross_sectional_rank(nonlinear)

@register_factor(
    name="quality_momentum_nonlinear",
    description="质量动量非线性耦合因子，(ROE排名×动量排名)的幂次变换（幂次放大联合强信号排前）。",
    category="coupling",
    thesis="高质量+强动量的联合效应是非线性的——两个维度都处于高位的股票(前20%×前20%)享受确定性溢价和市场追捧的双重叠加，alpha呈指数级放大而非线性叠加。幂次变换(pow 1.5)强调联合强信号。",
    dependencies=("__factors__", "roe", "mom_20"),
)
def factor_quality_momentum_nonlinear(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    mom = ctx.load_factor("mom_20")
    roe_r = _rank(roe)
    mom_r = _rank(mom)
    # Power transform to amplify joint strong signals
    nonlinear = (roe_r * mom_r) ** 1.5
    return cross_sectional_rank(nonlinear)

@register_factor(
    name="factor_relative_value_spread",
    description="因子相对价值价差因子，BP排名与行业中性BP排名的差值（相对行业高估/低估排前）。",
    category="coupling",
    thesis="股票在其行业内的相对估值位置——BP排名高于行业中性BP排名意味着股票在行业内被低估(相对行业的价值洼地)，低于行业中性排名意味着行业内相对高估。跨行业的估值比较需要行业中性调整。",
    dependencies=("__factors__", "bp", "bp_industry_neutral"),
)
def factor_factor_relative_value_spread(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    bp_ind = ctx.load_factor("bp_industry_neutral")
    bp_r = _rank(bp)
    bp_ind_r = _rank(bp_ind)
    # Positive = raw BP ranks higher than industry-neutral = within-industry undervaluation
    spread = bp_r - bp_ind_r
    return cross_sectional_rank(spread)

@register_factor(
    name="sentiment_fundamental_gap",
    description="情绪基本面缺口因子，(主力资金+换手率)情绪排名均值-基本面排名均值（情绪过热=排后）。",
    category="coupling",
    thesis="市场情绪(资金+换手)与基本面(ROE+BP)的差距——情绪远高于基本面时是泡沫信号(排后)，情绪远低于基本面时是被市场忽视的价值机会(排前)。情绪基本面的缺口收敛是alpha的来源。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "turnover_20", "roe", "bp"),
)
def factor_sentiment_fundamental_gap(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    turnover = ctx.load_factor("turnover_20")
    roe = ctx.load_factor("roe")
    bp = ctx.load_factor("bp")
    sentiment = (_rank(mf) + _rank(turnover)) / 2.0
    fundamental = (_rank(roe) + _rank(bp)) / 2.0
    # Sentiment > fundamental = potential bubble (rank low)
    # Sentiment < fundamental = neglected gem (rank high)
    gap = fundamental - sentiment
    return cross_sectional_rank(gap)

@register_factor(
    name="short_term_long_term_alignment",
    description="短长期因子对齐因子，短期动量排名+长期动量排名的等权（短长期一致=趋势确认排前）。",
    category="coupling",
    thesis="短期信号与长期信号的方向一致性是趋势质量的确认——短期动量与长期动量方向一致时趋势更可靠，短期与长期方向背离时(如短期涨但长期仍在跌)趋势不稳固。短长期对齐是多时间尺度的共振确认。",
    dependencies=("__factors__", "mom_5", "mom_60"),
)
def factor_short_term_long_term_alignment(ctx: FactorContext) -> pd.Series:
    mom5 = ctx.load_factor("mom_5")
    mom60 = ctx.load_factor("mom_60")
    alignment = (_rank(mom5) + _rank(mom60)) / 2.0
    return cross_sectional_rank(alignment)

@register_factor(
    name="factor_dispersion_risk",
    description="因子离散风险因子，核心因子截面排名的标准差均值（离散大=分歧大=风险高排后）。",
    category="coupling",
    thesis="多个因子的截面排名分歧程度——同一只股票在不同因子上的排名标准差大意味着多空信号冲突、投资者对该股票的分歧大。排名一致的股票(各因子同向)确定性更高。",
    dependencies=("__factors__", "bp", "roe", "mom_20", "volatility_20", "turnover_20"),
)
def factor_factor_dispersion_risk(ctx: FactorContext) -> pd.Series:
    factor_names = ["bp", "roe", "mom_20", "volatility_20", "turnover_20"]
    factors = ctx.load_factors(factor_names)
    ranks = pd.DataFrame(index=factors.index)
    for col in factors.columns:
        ranks[col] = factors[col].groupby(level="Date").rank(pct=True)
    # Per-stock standard deviation of ranks across factors
    rank_std = ranks.std(axis=1)
    rank_mean = ranks.mean(axis=1)
    # Coefficient of variation: higher = more disagreement across factors
    dispersion = safe_divide(rank_std, rank_mean + 1e-8)
    return cross_sectional_rank(-dispersion)

# ═══════════════════════════════════════════════════════════════════════════════
# Z — Event Decay Wrappers (事件因子指数衰减连续化)
# ═══════════════════════════════════════════════════════════════════════════════
# These factors take sparse event-driven factors (NaN > 90%) and convert them
# into continuous signals via forward-fill + exponential decay.
# Half-life: 3d (封板/seal), 5d (龙虎榜/涨跌停)))

