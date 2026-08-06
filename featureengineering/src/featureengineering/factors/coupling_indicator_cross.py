"""
Class 5 indicator cross-source coupling factors (survivors only).
"""
from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── MACD Coupling ────────────────────────────────────────────────────────────

@register_factor(
    name="coupling_macd_value_resonance",
    description="MACD趋势强度 × 价值(bp)共振。MACD走强+低估值=技术面与基本面双重确认。",
    category="coupling",
    thesis="MACD走强=技术面趋势向上，bp高=基本面低估——两者同时成立时，是便宜且开始涨的经典共振信号。",
    dependencies=("__factors__", "macd_trend_strength", "bp"),
)
def factor_coupling_macd_value_resonance(ctx: FactorContext) -> np.ndarray:
    macd = ctx.load_factor("macd_trend_strength")
    bp = ctx.load_factor("bp")
    resonance = macd * bp
    return cross_sectional_rank(resonance)


@register_factor(
    name="coupling_macd_chip_divergence",
    description="MACD趋势强度 vs 筹码集中度(cr3)背离。MACD强+筹码散=虚涨，MACD弱+筹码集中=蓄力。",
    category="coupling",
    thesis="MACD走强但筹码分散=技术面虚涨、缺乏主力支撑；MACD走弱但筹码高度集中=主力控盘洗盘。",
    dependencies=("__factors__", "macd_trend_strength", "chip_cr3_factor"),
)
def factor_coupling_macd_chip_divergence(ctx: FactorContext) -> np.ndarray:
    macd = ctx.load_factor("macd_trend_strength")
    cr3 = ctx.load_factor("chip_cr3_factor")
    divergence = macd - cr3
    return cross_sectional_rank(divergence)


# ── RSI Coupling ─────────────────────────────────────────────────────────────

@register_factor(
    name="coupling_rsi_value_combo",
    description="RSI超卖(低RSI=排前) × 低估值(bp)。超卖+低估=抄底双信号。",
    category="coupling",
    thesis="RSI超卖=短期技术面超跌，bp高=基本面低估——两者同时出现时，是又便宜又超跌的双重抄底信号。",
    dependencies=("__factors__", "rsi_14_excess", "bp"),
)
def factor_coupling_rsi_value_combo(ctx: FactorContext) -> np.ndarray:
    rsi = ctx.load_factor("rsi_14_excess")
    bp = ctx.load_factor("bp")
    combo = rsi * bp
    return cross_sectional_rank(combo)


@register_factor(
    name="coupling_rsi_turnover_divergence",
    description="RSI超卖 × 换手率(turnover_20)。超卖+高换手=恐慌抛售中的抄底机会。",
    category="coupling",
    thesis="超卖+高换手=恐慌性抛售——散户在恐慌中割肉，机构在低位接盘。",
    dependencies=("__factors__", "rsi_14_excess", "turnover_20"),
)
def factor_coupling_rsi_turnover_divergence(ctx: FactorContext) -> np.ndarray:
    rsi = ctx.load_factor("rsi_14_excess")
    to = ctx.load_factor("turnover_20")
    high_to = 1.0 - to
    signal = rsi * high_to
    return cross_sectional_rank(signal)


@register_factor(
    name="coupling_rsi_moneyflow_resonance",
    description="RSI趋势 × 主力资金净流入。RSI走强+主力流入=技术和资金双重看多。",
    category="coupling",
    thesis="RSI日内走高=当日买盘持续强于卖盘，主力净流入=大资金在买入——技术面和资金面双重确认。",
    dependencies=("__factors__", "rsi_intraday_trend", "mf_net_inflow_ratio"),
)
def factor_coupling_rsi_moneyflow_resonance(ctx: FactorContext) -> np.ndarray:
    rsi_trend = ctx.load_factor("rsi_intraday_trend")
    mf = ctx.load_factor("mf_net_inflow_ratio")
    resonance = rsi_trend * mf
    return cross_sectional_rank(resonance)


# ── Bollinger Coupling ───────────────────────────────────────────────────────

@register_factor(
    name="coupling_boll_squeeze_value",
    description="布林挤压(boll_squeeze) × 低估值(bp)。布林收窄(蓄力)+低估=突破前的最优买点。",
    category="coupling",
    thesis="布林带收窄=波动率压缩，价格在狭窄区间整理——是突破前的蓄力阶段。收窄+低估=一只便宜的股票正在筑底蓄力。",
    dependencies=("__factors__", "boll_squeeze", "bp"),
)
def factor_coupling_boll_squeeze_value(ctx: FactorContext) -> np.ndarray:
    boll = ctx.load_factor("boll_squeeze")
    bp = ctx.load_factor("bp")
    signal = (1.0 - boll) * bp
    return cross_sectional_rank(signal)


# ── MA Alignment Coupling ────────────────────────────────────────────────────

@register_factor(
    name="coupling_ma_chip_resonance",
    description="均线多头排列 × 筹码集中(cr3)。均线完美+筹码集中=主力高度控盘的上升趋势。",
    category="coupling",
    thesis="均线多头排列=趋势完美，筹码高度集中=主力控盘——两者叠加是主力控盘拉升的最强技术形态。",
    dependencies=("__factors__", "ma_alignment_score", "chip_cr3_factor"),
)
def factor_coupling_ma_chip_resonance(ctx: FactorContext) -> np.ndarray:
    ma = ctx.load_factor("ma_alignment_score")
    cr3 = ctx.load_factor("chip_cr3_factor")
    resonance = ma * cr3
    return cross_sectional_rank(resonance)


# ── KDJ Coupling ─────────────────────────────────────────────────────────────

@register_factor(
    name="coupling_kdj_moneyflow_divergence",
    description="KDJ金叉净数(kdj_cross_net) vs 主力资金背离。KDJ金叉+主力流出=技术虚涨，KDJ死叉+主力流入=洗盘。",
    category="coupling",
    thesis="KDJ频繁金叉但主力持续流出=散户推动的技术反弹(虚涨)；KDJ频繁死叉但主力持续流入=主力在吸筹(洗盘)。",
    dependencies=("__factors__", "kdj_cross_net", "mf_net_inflow_ratio"),
)
def factor_coupling_kdj_moneyflow_divergence(ctx: FactorContext) -> np.ndarray:
    kdj = ctx.load_factor("kdj_cross_net")
    mf = ctx.load_factor("mf_net_inflow_ratio")
    divergence = kdj - mf
    return cross_sectional_rank(divergence)


# ── Multi-Indicator Composite ────────────────────────────────────────────────

@register_factor(
    name="coupling_indicator_consensus_value",
    description="技术指标共识度(indicator_consensus) × 低估值(bp)。多指标一致看多+低估=最强信号。",
    category="coupling",
    thesis="当MACD/KDJ/RSI/Bollinger等多个指标一致发出看多信号，且股票处于低估状态时——技术面和基本面达成了罕见的全票通过。",
    dependencies=("__factors__", "indicator_consensus", "bp"),
)
def factor_coupling_indicator_consensus_value(ctx: FactorContext) -> np.ndarray:
    consensus = ctx.load_factor("indicator_consensus")
    bp = ctx.load_factor("bp")
    signal = consensus * bp
    return cross_sectional_rank(signal)
