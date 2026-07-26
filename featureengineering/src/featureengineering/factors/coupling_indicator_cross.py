"""
Class 5 indicator cross-source coupling factors.

Bridges the gap: all 16 indicator_1min technical indicator fields are concentrated
in a single file (indicator_minute.py) with ZERO cross-factor coupling.

These factors load pre-computed indicator factors via ctx.load_factor() and
combine them with factors from other classes (valuation, price, chip, fund_flow).

Pattern:
  - Load an indicator_minute factor (MACD/KDJ/RSI/Bollinger/MA family)
  - Load a factor from another class (bp, mom_20, winner_rate, mf_*, etc.)
  - Compute interaction: resonance (product), divergence (difference), ratio

All use dependencies=("__factors__", "indicator_factor", "other_factor", ...)
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: MACD × Fundamental/Price Coupling
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_macd_value_resonance",
    description="MACD趋势强度 × 价值(bp)共振。MACD走强+低估值=技术面与基本面双重确认。",
    category="coupling",
    thesis=(
        "MACD趋势强度因子(macd_trend_strength)与价值因子(bp)的乘积。"
        "MACD走强=技术面趋势向上，bp高=基本面低估——两者同时成立时，"
        "是便宜且开始涨的经典共振信号。解决了indicator_1min因子"
        "从未与valuation因子做过交叉耦合的问题。"
    ),
    dependencies=("__factors__", "macd_trend_strength", "bp"),
)
def factor_coupling_macd_value_resonance(ctx: FactorContext) -> np.ndarray:
    macd = ctx.load_factor("macd_trend_strength")
    bp = ctx.load_factor("bp")
    resonance = macd * bp
    return cross_sectional_rank(resonance)


@register_factor(
    name="coupling_macd_momentum_combo",
    description="MACD信号 × 动量(mom_20)。技术指标+价格趋势双重确认，信号增强。",
    category="coupling",
    thesis=(
        "MACD信号因子(macd_signal_cross)与动量因子(mom_20)的乘积。"
        "MACD金叉+动量强势=多重技术确认——趋势更可靠。"
        "MACD和动量来自完全不同的数据源(1分钟指标 vs 日频价格),提供真正的多维度确认。"
    ),
    dependencies=("__factors__", "macd_signal_cross", "mom_20"),
)
def factor_coupling_macd_momentum_combo(ctx: FactorContext) -> np.ndarray:
    macd = ctx.load_factor("macd_signal_cross")
    mom = ctx.load_factor("mom_20")
    combo = macd * mom
    return cross_sectional_rank(combo)


@register_factor(
    name="coupling_macd_chip_divergence",
    description="MACD趋势强度 vs 筹码集中度(cr3)背离。MACD强+筹码散=虚涨,MACD弱+筹码集中=蓄力。",
    category="coupling",
    thesis=(
        "MACD趋势强度与筹码集中度的差值。MACD走强但筹码分散=技术面虚涨、缺乏主力支撑；"
        "MACD走弱但筹码高度集中=主力控盘洗盘——技术面走弱是假象,蓄力后可能爆发。"
        "这是indicator_1min与cyq_chips两个Class之间的首次跨类耦合。"
    ),
    dependencies=("__factors__", "macd_trend_strength", "chip_cr3_factor"),
)
def factor_coupling_macd_chip_divergence(ctx: FactorContext) -> np.ndarray:
    macd = ctx.load_factor("macd_trend_strength")
    cr3 = ctx.load_factor("chip_cr3_factor")
    divergence = macd - cr3
    return cross_sectional_rank(divergence)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: RSI × Value/Sentiment Coupling
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_rsi_value_combo",
    description="RSI超卖(低RSI=排前) × 低估值(bp)。超卖+低估=抄底双信号。",
    category="coupling",
    thesis=(
        "RSI因子(rsi_14_excess)取反(超卖排前)与价值因子(bp)的乘积。"
        "RSI超卖=短期技术面超跌，bp高=基本面低估——两者同时出现时，"
        "是又便宜又超跌的双重抄底信号。RSI是1分钟K线计算的日内指标,"
        "比日频RSI更灵敏——能更快捕捉到超卖机会。"
    ),
    dependencies=("__factors__", "rsi_14_excess", "bp"),
)
def factor_coupling_rsi_value_combo(ctx: FactorContext) -> np.ndarray:
    rsi = ctx.load_factor("rsi_14_excess")
    bp = ctx.load_factor("bp")
    # rsi_14_excess is neg-ranked (RSI high = rank low), so high rank = oversold
    # bp is pos-ranked (high PB-inverse = cheap = high rank)
    combo = rsi * bp
    return cross_sectional_rank(combo)


@register_factor(
    name="coupling_rsi_turnover_divergence",
    description="RSI超卖 × 换手率(turnover_20)。超卖+高换手=恐慌抛售中的抄底机会。",
    category="coupling",
    thesis=(
        "RSI超卖排前与高换手率(1.0-turnover_20排前=高换手)的乘积。"
        "超卖+高换手=恐慌性抛售——散户在恐慌中割肉,机构在低位接盘。"
        "这是经典的别人恐惧我贪婪量化实现。而超卖+低换手=无人问津的阴跌——不值得抄底。"
    ),
    dependencies=("__factors__", "rsi_14_excess", "turnover_20"),
)
def factor_coupling_rsi_turnover_divergence(ctx: FactorContext) -> np.ndarray:
    rsi = ctx.load_factor("rsi_14_excess")
    to = ctx.load_factor("turnover_20")
    # High turnover = low turnover_20 rank, so we invert
    high_to = 1.0 - to
    signal = rsi * high_to
    return cross_sectional_rank(signal)


@register_factor(
    name="coupling_rsi_moneyflow_resonance",
    description="RSI趋势 × 主力资金净流入。RSI走强+主力流入=技术和资金双重看多。",
    category="coupling",
    thesis=(
        "RSI日内趋势因子(rsi_intraday_trend)与主力资金净流入因子(mf_net_inflow_ratio)的乘积。"
        "RSI日内走高=当日买盘持续强于卖盘，主力净流入=大资金在买入——"
        "日内技术形态与资金流向相互印证。技术面和资金面双重确认的信号比单一维度更可靠。"
    ),
    dependencies=("__factors__", "rsi_intraday_trend", "mf_net_inflow_ratio"),
)
def factor_coupling_rsi_moneyflow_resonance(ctx: FactorContext) -> np.ndarray:
    rsi_trend = ctx.load_factor("rsi_intraday_trend")
    mf = ctx.load_factor("mf_net_inflow_ratio")
    resonance = rsi_trend * mf
    return cross_sectional_rank(resonance)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Bollinger Band × Risk/Volatility Coupling
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_boll_volatility_resonance",
    description="布林带宽度(boll_width) × 波动率(volatility_20)共振。布林扩张+高波动=趋势行情确认。",
    category="coupling",
    thesis=(
        "布林带宽度因子(boll_width_20)与波动率因子(volatility_20)的共振。"
        "布林宽度扩大=价格脱离均线(趋势启动)，波动率上升=市场活跃——"
        "两者同时出现是趋势行情的强烈确认。布林宽度来自1分钟数据计算的日内指标,"
        "比日频布林带更及时地捕捉到波动率突变。"
    ),
    dependencies=("__factors__", "boll_width_20", "volatility_20"),
)
def factor_coupling_boll_volatility_resonance(ctx: FactorContext) -> np.ndarray:
    boll = ctx.load_factor("boll_width_20")
    vol = ctx.load_factor("volatility_20")
    # boll_width_20 is neg-ranked (narrow = squeeze = high rank)
    # volatility_20 is neg-ranked (low vol = high rank)
    # We want both HIGH (wide + volatile), so invert both
    resonance = (1.0 - boll) * (1.0 - vol)
    return cross_sectional_rank(resonance)


@register_factor(
    name="coupling_boll_momentum_signal",
    description="布林带位置(boll_position) × 动量(mom_60)。价格在布林上轨+动量强势=超强趋势。",
    category="coupling",
    thesis=(
        "布林带位置因子(boll_position,高=接近上轨)与中长期动量(mom_60)的乘积。"
        "价格在布林上轨附近且60日动量强势=价格持续沿上轨运行(bandwalk)——"
        "这是最强的趋势形态。价格在上轨但动量走弱=布林带突破衰竭——即将反转。"
    ),
    dependencies=("__factors__", "boll_position", "mom_60"),
)
def factor_coupling_boll_momentum_signal(ctx: FactorContext) -> np.ndarray:
    boll = ctx.load_factor("boll_position")
    mom = ctx.load_factor("mom_60")
    signal = boll * mom
    return cross_sectional_rank(signal)


@register_factor(
    name="coupling_boll_squeeze_value",
    description="布林挤压(boll_squeeze) × 低估值(bp)。布林收窄(蓄力)+低估=突破前的最优买点。",
    category="coupling",
    thesis=(
        "布林挤压因子(boll_squeeze,高=收窄)与价值因子(bp)的乘积。"
        "布林带收窄=波动率压缩,价格在狭窄区间整理——是突破前的蓄力阶段。"
        "收窄+低估=一只便宜的股票正在筑底蓄力——可能是突破前的最佳介入时机。"
        "这是indicator_1min(Bollinger)与valuation(bp)的跨类耦合。"
    ),
    dependencies=("__factors__", "boll_squeeze", "bp"),
)
def factor_coupling_boll_squeeze_value(ctx: FactorContext) -> np.ndarray:
    boll = ctx.load_factor("boll_squeeze")
    bp = ctx.load_factor("bp")
    # boll_squeeze is neg-ranked (narrow = squeeze = low rank...)
    # Actually check: "boll_squeeze" spec is ("boll_squeeze", "neg")
    # So: high squeeze metric -> low rank. But we want high squeeze = high rank
    # So invert: 1.0 - boll = high when squeeze is strong
    signal = (1.0 - boll) * bp
    return cross_sectional_rank(signal)


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: MA Alignment × Multi-Factor Combos
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_ma_value_momentum",
    description="均线多头排列(ma_alignment) × 价值(bp) × 动量(mom_20)三重确认。",
    category="coupling",
    thesis=(
        "均线多头排列因子(ma_alignment_score)与价值、动量的乘积。"
        "均线多头排列=MA5>MA10>MA20>MA30>MA60——完美的上升趋势结构。"
        "叠加低估值(bp高)+强动量(mom_20高)=技术完美+基本面好+趋势强的三重确认。"
        "这是将1分钟均线排列信号与日频因子做跨时间维度耦合的尝试。"
    ),
    dependencies=("__factors__", "ma_alignment_score", "bp", "mom_20"),
)
def factor_coupling_ma_value_momentum(ctx: FactorContext) -> np.ndarray:
    ma = ctx.load_factor("ma_alignment_score")
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")
    trio = ma * bp * mom
    return cross_sectional_rank(trio)


@register_factor(
    name="coupling_ma_chip_resonance",
    description="均线多头排列 × 筹码集中(cr3)。均线完美+筹码集中=主力高度控盘的上升趋势。",
    category="coupling",
    thesis=(
        "均线多头排列因子与筹码集中度因子(cr3)的乘积。"
        "均线多头排列=趋势完美，筹码高度集中=主力控盘——"
        "两者叠加是主力控盘拉升的最强技术形态。均线排列来自1分钟MA计算，"
        "筹码集中来自cyq_chips——两个完全不同的数据源和Class之间的耦合。"
    ),
    dependencies=("__factors__", "ma_alignment_score", "chip_cr3_factor"),
)
def factor_coupling_ma_chip_resonance(ctx: FactorContext) -> np.ndarray:
    ma = ctx.load_factor("ma_alignment_score")
    cr3 = ctx.load_factor("chip_cr3_factor")
    resonance = ma * cr3
    return cross_sectional_rank(resonance)


# ═══════════════════════════════════════════════════════════════════════════════
# Section E: KDJ × Reversal/Timing Coupling
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_kdj_reversal_combo",
    description="KDJ超卖(kdj_j_value_close排前=J值低) × 反转因子(reversal_5)。超卖+反转倾向=抄底。",
    category="coupling",
    thesis=(
        "KDJ的J值因子(kdj_j_value_close,neg-ranked=J高排后=J低排前=超卖)与"
        "反转因子(reversal_5,做多近期亏损股)的乘积。KDJ超卖+反转因子多=技术超卖+"
        "统计反转倾向——双重确认的抄底信号。KDJ来自1分钟数据,反应比日频KDJ快得多。"
    ),
    dependencies=("__factors__", "kdj_j_value_close", "reversal_5"),
)
def factor_coupling_kdj_reversal_combo(ctx: FactorContext) -> np.ndarray:
    kdj = ctx.load_factor("kdj_j_value_close")
    rev = ctx.load_factor("reversal_5")
    # kdj_j_value_close is neg-ranked: low J = oversold = high rank
    combo = kdj * rev
    return cross_sectional_rank(combo)


@register_factor(
    name="coupling_kdj_moneyflow_divergence",
    description="KDJ金叉净数(kdj_cross_net) vs 主力资金背离。KDJ金叉+主力流出=技术虚涨,KDJ死叉+主力流入=洗盘。",
    category="coupling",
    thesis=(
        "KDJ金叉净数(kdj_cross_net,金叉多=排前)与主力资金因子的差值。"
        "KDJ频繁金叉但主力持续流出=散户推动的技术反弹(虚涨)——不可持续；"
        "KDJ频繁死叉但主力持续流入=主力在吸筹(洗盘)——积极的背离信号。"
        "技术面与资金面的背离是最有价值的交易信号之一。"
    ),
    dependencies=("__factors__", "kdj_cross_net", "mf_net_inflow_ratio"),
)
def factor_coupling_kdj_moneyflow_divergence(ctx: FactorContext) -> np.ndarray:
    kdj = ctx.load_factor("kdj_cross_net")
    mf = ctx.load_factor("mf_net_inflow_ratio")
    divergence = kdj - mf
    return cross_sectional_rank(divergence)


# ═══════════════════════════════════════════════════════════════════════════════
# Section F: Multi-Indicator Composite
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="coupling_indicator_consensus_value",
    description="技术指标共识度(indicator_consensus) × 低估值(bp)。多指标一致看多+低估=最强信号。",
    category="coupling",
    thesis=(
        "技术指标共识度因子(indicator_consensus,多指标一致看多=排前)"
        "与价值因子的乘积。当MACD/KDJ/RSI/Bollinger等多个指标一致发出看多信号，"
        "且股票处于低估状态时——技术面和基本面达成了罕见的全票通过。"
        "这是Class 4(1分钟指标综合)与Class 1(估值)之间的Class 5耦合。"
    ),
    dependencies=("__factors__", "indicator_consensus", "bp"),
)
def factor_coupling_indicator_consensus_value(ctx: FactorContext) -> np.ndarray:
    consensus = ctx.load_factor("indicator_consensus")
    bp = ctx.load_factor("bp")
    signal = consensus * bp
    return cross_sectional_rank(signal)


@register_factor(
    name="coupling_indicator_dispersion_momentum",
    description="指标离散度(indicator_dispersion) × 动量(mom_60)。指标分歧大+动量强=趋势中分歧(持续)。",
    category="coupling",
    thesis=(
        "技术指标离散度因子(indicator_dispersion,高=各指标方向不一致)与动量因子的组合。"
        "指标分歧大+动量强=趋势中正常的分歧(多空博弈)——趋势可能持续；"
        "指标分歧大+动量弱=方向不明(震荡)——应回避；"
        "指标分歧小+动量强=一致性强趋势——最确定的行情。"
    ),
    dependencies=("__factors__", "indicator_dispersion", "mom_60"),
)
def factor_coupling_indicator_dispersion_momentum(ctx: FactorContext) -> np.ndarray:
    dispersion = ctx.load_factor("indicator_dispersion")
    mom = ctx.load_factor("mom_60")
    # dispersion is neg-ranked (high dispersion = lots of disagreement = low rank)
    # We want: high agreement (1.0-dispersion) × high momentum
    signal = (1.0 - dispersion) * mom
    return cross_sectional_rank(signal)
