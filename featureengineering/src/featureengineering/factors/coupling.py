"""
Factor-coupling module — Class 5 factors (surviving after pct_change-based factor deletion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (0-1)."""
    return s.groupby(level="Date").rank(pct=True)


def _delta(s: pd.Series, window: int) -> pd.Series:
    """Per-stock difference over *window* periods."""
    return s.groupby(level="Code").diff(window)


def _momentum(s: pd.Series, window: int) -> pd.Series:
    """Per-stock percentage change over *window* periods."""
    return s.groupby(level="Code").transform(
        lambda x: x.pct_change(window, fill_method=None)
    )


# ── Factor Momentum ──────────────────────────────────────────────────────────

@register_factor(
    name="bp_factor_momentum_20",
    description="BP因子20日动量 (BP因子值的变化率)。",
    category="coupling",
    thesis="BP自身的动量捕捉估值修复的启动——BP快速上升的股票正在被市场重估",
    dependencies=("__factors__", "bp"),
)
def factor_bp_factor_momentum_20(ctx: FactorContext):
    bp = ctx.load_factor("bp")
    mom = _momentum(bp, 20)
    return cross_sectional_rank(mom)


@register_factor(
    name="turnover_factor_momentum_20",
    description="换手率因子20日动量 (关注度变化)。",
    category="coupling",
    thesis="换手率因子趋势上升=市场关注度在持续提升，是正面信号",
    dependencies=("__factors__", "turnover_20"),
)
def factor_turnover_factor_momentum_20(ctx: FactorContext):
    # turnover_20 的 .fea 为 rank(-换手),高=低换手;描述要求"关注度提升(换手率
    # 上升)排前",故对 (1.0 - to) 取动量(与 coupling_liquidity_momentum_20d 同口径)。
    # 修复前直接对 to 取动量 = 低换手程度增强排前,与描述相反(2026-08-05)。
    to = ctx.load_factor("turnover_20")
    mom = _momentum(1.0 - to, 20)
    return cross_sectional_rank(mom)


@register_factor(
    name="winner_rate_factor_momentum_20",
    description="获利盘因子20日动量 (筹码结构变化方向)。",
    category="coupling",
    thesis="获利盘比例趋势下降意味着筹码在从分散到集中——主力收集筹码的迹象",
    dependencies=("__factors__", "winner_rate"),
)
def factor_winner_rate_factor_momentum_20(ctx: FactorContext):
    # winner_rate 的 .fea 为 rank(-获利盘),高=低获利盘;
    # delta 高 = winner_rate 因子上升 = 原始获利盘下降。描述要求"获利盘下降排前",故 rank(delta)。
    wr = ctx.load_factor("winner_rate")
    delta = _delta(wr, 20)
    return cross_sectional_rank(delta)


@register_factor(
    name="mf_flow_factor_momentum_20",
    description="主力资金流因子20日动量 (资金态度变化)。",
    category="coupling",
    thesis="主力资金因子的趋势上升意味着机构态度在边际改善",
    dependencies=("__factors__", "mf_net_inflow_ratio"),
)
def factor_mf_flow_factor_momentum_20(ctx: FactorContext):
    mf = ctx.load_factor("mf_net_inflow_ratio")
    delta = _delta(mf, 20)
    return cross_sectional_rank(delta)


# ── Factor Divergence ────────────────────────────────────────────────────────

@register_factor(
    name="fundflow_retail_inst_divergence",
    description="主力-散户资金背离因子 (大单净买+小单净卖=机构吸筹)。",
    category="coupling",
    thesis="大单(机构)买入而小单(散户)卖出是机构吸筹的清晰信号",
    dependencies=("__factors__", "mf_big_order_ratio", "mf_small_order_ratio"),
)
def factor_fundflow_retail_inst_divergence(ctx: FactorContext):
    big = ctx.load_factor("mf_big_order_ratio")
    small = ctx.load_factor("mf_small_order_ratio")
    big_r = _rank(big)
    small_r = _rank(-small)
    divergence = big_r * small_r
    return cross_sectional_rank(divergence)


# ── Factor Resonance ─────────────────────────────────────────────────────────

@register_factor(
    name="value_factor_zscore_252",
    description="价值因子(BP)252日历史Z-score (相对自身历史的高估/低估)。",
    category="coupling",
    thesis="BP相对于自身历史水平处于高位时价值因子更有效——估值回复的引力更强",
    dependencies=("__factors__", "bp"),
)
def factor_value_factor_zscore_252(ctx: FactorContext):
    bp = ctx.load_factor("bp")

    def _rolling_zscore(s, window):
        mean = s.rolling(window, min_periods=window // 2).mean()
        std = s.rolling(window, min_periods=window // 2).std()
        return safe_divide(s - mean, std + 1e-8)

    z = bp.groupby(level="Code").transform(lambda s: _rolling_zscore(s, 252))
    return cross_sectional_rank(z)


# ── Orthogonalization ────────────────────────────────────────────────────────

@register_factor(
    name="turnover_orthogonal_to_mv",
    description="换手率对市值正交化因子 (剔除规模效应的纯流动性)。",
    category="coupling",
    thesis="小盘股天然换手率高——剔除市值影响后的换手率才是真正的流动性偏好度量",
    dependencies=("__factors__", "turnover_20", "log_circ_mv"),
)
def factor_turnover_orthogonal_to_mv(ctx: FactorContext):
    to = ctx.load_factor("turnover_20")
    size = ctx.load_factor("log_circ_mv")

    def _zscore(s):
        mu = s.groupby(level="Date").transform("mean")
        sg = s.groupby(level="Date").transform("std")
        return safe_divide(s - mu, sg + 1e-8)

    to_z = _zscore(to)
    size_z = _zscore(size)
    b = (to_z * size_z).groupby(level="Date").transform("mean") / (
        (size_z ** 2).groupby(level="Date").transform("mean") + 1e-8
    )
    residual = to_z - b * size_z
    return cross_sectional_rank(residual)


# ── Cross-Factor ─────────────────────────────────────────────────────────────

@register_factor(
    name="turnover_shock_20",
    description="换手率异动因子，换手率20日Z-score截面排名（异常高换手排后）。",
    category="enhanced",
    thesis="换手率突然飙升(偏离正常水平多个标准差)通常伴随事件驱动",
    dependencies=("__factors__", "turnover_20"),
)
def factor_turnover_shock_20(context: FactorContext):
    df = context.load_factors(["turnover_20"])
    to = df["turnover_20"]
    to_mean = to.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=20).mean())
    to_std = to.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=20).std())
    zscore = (to - to_mean) / to_std.replace(0, np.nan)
    return cross_sectional_rank(-zscore.abs())


# ── Higher-Order Interactions ────────────────────────────────────────────────

@register_factor(
    name="factor_rolling_drawdown_60",
    description="因子滚动回撤因子，BP因子从60日高点的回撤程度（回撤大=因子超跌排前）。",
    category="coupling",
    thesis="因子值从自身近期高点的回撤可能意味着因子被过度抛售",
    dependencies=("__factors__", "bp"),
)
def factor_factor_rolling_drawdown_60(ctx: FactorContext):
    bp = ctx.load_factor("bp")

    def _rolling_dd(s: pd.Series) -> pd.Series:
        rolling_max = s.rolling(60, min_periods=30).max()
        dd = (s - rolling_max) / rolling_max.replace(0, np.nan)
        return dd

    bp_dd = bp.groupby(level="Code").transform(_rolling_dd)
    return cross_sectional_rank(bp_dd)


@register_factor(
    name="factor_consistency_ratio",
    description="因子一致性比率因子，BP因子60日变化方向一致的天数占比（持续方向=高信度排前）。",
    category="coupling",
    thesis="因子变化方向的持续性是因子信度的度量",
    dependencies=("__factors__", "bp"),
)
def factor_factor_consistency_ratio(ctx: FactorContext):
    bp = ctx.load_factor("bp")
    bp_delta = bp.groupby(level="Code").diff(1)

    def _consistency(s: pd.Series) -> pd.Series:
        direction = np.sign(s)
        net_direction = direction.rolling(60, min_periods=30).sum()
        count = direction.abs().rolling(60, min_periods=30).sum()
        return safe_divide(net_direction.abs(), count + 1e-8)

    consistency = bp_delta.groupby(level="Code").transform(_consistency)
    return cross_sectional_rank(consistency)
