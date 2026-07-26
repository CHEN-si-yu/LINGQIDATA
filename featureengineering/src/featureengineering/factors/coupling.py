"""
Factor-coupling (因子耦合) module — Class 5 factors.

These factors load pre-computed Class 1/2/3/4 factors from .fea files and build
derived signals from their interactions: co-momentum, divergence, resonance,
conditional effects, stability, timing, and cross-factor residuals.

All factors declare ``"__factors__"`` in their dependencies so the build
orchestrator routes them to the Class 5 (coupling) pipeline.  The remaining
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

