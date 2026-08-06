"""
Extended factor coupling (扩展因子耦合) — Class 5.

These factors load pre-computed .fea factors and build high-dimensional
derived signals: triple interactions, factor quality metrics, orthogonalization,
sentiment divergence, factor dynamics, crowding warnings, and tail risk.

All factors declare ``"__factors__"`` in their dependencies.
The remaining dependency strings name existing .fea files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_std,
    safe_divide,
)

def _rank(s: pd.Series) -> pd.Series:
    return s.groupby(level="Date").rank(pct=True)

def _zscore(s: pd.Series) -> pd.Series:
    mu = s.groupby(level="Date").transform("mean")
    sg = s.groupby(level="Date").transform("std")
    return safe_divide(s - mu, sg + 1e-8)

def _delta(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level="Code").diff(window)

def _momentum(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level="Code").transform(lambda x: x.pct_change(window))

# ═══════════════════════════════════════════════════════════════════════════════
# A — Interactions & Resonance
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="fundflow_value_interaction",
    description="资金流-价值交互因子，主力净流入排名×bp排名截面排名（资金流入+低估=戴维斯双击前兆排前）。",
    category="coupling",
    thesis="主力资金的流入方向与价值的结合是最强的'聪明钱'信号——主力资金流入低估值股票意味着机构在系统性布局价值洼地，是戴维斯双击(估值修复+盈利增长)的前兆。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "bp"),
)
def factor_fundflow_value_interaction(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    bp = ctx.load_factor("bp")

    common = mf.index.intersection(bp.index)
    interaction = _rank(mf.loc[common]) * _rank(bp.loc[common])
    return cross_sectional_rank(interaction)

@register_factor(
    name="factor_trend_strength_60",
    description="因子趋势强度因子，bp因子60日均值偏移/60日std截面排名（强趋势=因子方向可靠排前）。",
    category="coupling",
    thesis="因子值的趋势强度衡量因子信号的'可信度'——因子在持续改善(如BP持续上升)比因子在某一时点的水平更具信息量。趋势强度高的因子信号更可能持续而非反转。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_trend_strength_60(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    ma60 = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).mean())
    std60 = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).std())
    strength = (bp - ma60) / std60.replace(0, np.nan)
    return cross_sectional_rank(strength)

@register_factor(
    name="factor_mean_reversion_20",
    description="因子均值回复因子，-(bp偏离20日均值的标准差倍数)截面排名（取负向=过度偏离=回复压力排后）。",
    category="coupling",
    thesis="因子值对短期均值的偏离具有均值回复特征——偏离过大的股票在因子维度上'超买'或'超卖'。因子均值回复是因子择时的重要信号。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_mean_reversion_20(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    ma20 = bp.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    std20 = bp.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
    z = (bp - ma20) / std20.replace(0, np.nan)
    return cross_sectional_rank(-z.abs())

@register_factor(
    name="factor_momentum_decay",
    description="因子动量衰减因子，bp的5日动量/20日动量截面排名（衰减=短期弱于长期=动能减弱排后）。",
    category="coupling",
    thesis="因子短期动量与长期动量的比值反映因子趋势的'健康度'——短期动量<长期动量意味着趋势在减速(衰减)，可能即将反转。比值稳定在1附近则趋势健康。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_momentum_decay(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom5 = _momentum(bp, 5)
    mom20 = _momentum(bp, 20)
    common = mom5.index.intersection(mom20.index)
    decay = mom5.loc[common] / mom20.loc[common].replace(0, np.nan)
    return cross_sectional_rank(decay)

@register_factor(
    name="factor_signal_to_noise_60",
    description="因子信噪比因子，bp的60日均值/std截面排名（高信噪比=因子信号清晰排前）。",
    category="coupling",
    thesis="因子的信噪比(均值/标准差)是因子质量的度量——高信噪比意味着因子的信号稳定、噪音小，低信噪比的因子信号可能只是随机波动。选择高信噪比因子是量化投资的基础原则。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_signal_to_noise_60(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    ma60 = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).mean())
    std60 = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).std())
    snr = ma60.abs() / std60.replace(0, np.nan)
    return cross_sectional_rank(snr)

@register_factor(
    name="factor_turnover_ratio_20",
    description="因子换手率因子，bp截面排名20日变化绝对值截面排名（取负向=高换手=不稳定排后）。",
    category="coupling",
    thesis="因子截面排名的变化(因子换手率)反映了因子信号的不稳定性——因子换手率过高意味着因子信号每天都在变，难以形成稳定的alpha。低因子换手率意味着信号的持续性。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_turnover_ratio_20(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    rank = _rank(bp)
    chg = rank.groupby(level="Code").diff(20).abs()
    return cross_sectional_rank(-chg)


@register_factor(
    name="factor_consistency_score",
    description="因子一致性因子，bp 60日在极端分位(>0.8或<0.2)的占比截面排名（持续极端=信号强烈排前）。",
    category="coupling",
    thesis="因子持续处于极端分位意味着该股票在该因子维度上有稳定的特征——而非偶尔极端。持续在极端分位的股票最具因子特征的代表性，是因子策略最核心的标的。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_consistency_score(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    rank = _rank(bp)
    is_extreme = ((rank > 0.8) | (rank < 0.2)).astype(float)
    consistency = is_extreme.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    return cross_sectional_rank(consistency)

# ═══════════════════════════════════════════════════════════════════════════════
# D — Factor Crowding & Risk
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="factor_crowding_warning",
    description="因子拥挤预警因子，-(bp排名60日中同方向占比>80%)截面排名（拥挤=一致预期风险排后）。",
    category="coupling",
    thesis="当因子排名持续处于同一方向(>80%的时间在顶部或底部)，意味着该因子的拥挤度极高——过多资金在追逐同一个因子信号，反转风险在累积。因子拥挤是量化策略最大的尾部风险。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_crowding_warning(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    rank = _rank(bp)
    is_high = (rank > 0.8).astype(float)
    is_low = (rank < 0.2).astype(float)

    high_pct = is_high.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    low_pct = is_low.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )

    crowding = np.maximum(high_pct, low_pct)
    return cross_sectional_rank(-crowding)

@register_factor(
    name="factor_drawdown_60_deep",
    description="因子滚动回撤因子，-(bp 60日滚动最大回撤)截面排名（深度回撤=因子失效风险排后）。",
    category="coupling",
    thesis="因子值的滚动回撤衡量因子本身的'表现'——因子值持续下跌意味着该股票在持续失去该因子特征。深度的因子回撤可能意味着基本面发生了转折性变化。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_rolling_drawdown_60(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    peak = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).max())
    dd = (bp / peak.replace(0, np.nan)) - 1.0
    return cross_sectional_rank(dd)


@register_factor(
    name="sentiment_value_gap",
    description="情绪-价值差因子，-(资金流因子排名-bp排名)截面排名（取负向=情绪脱离价值=风险排后）。",
    category="coupling",
    thesis="资金情绪与基本价值的差距是'泡沫/恐慌'的度量——资金大幅流入但价值排名很低=情绪脱离基本面(泡沫风险)。资金大幅流出但价值排名很高=恐慌超卖(价值机会)。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "bp"),
)
def factor_sentiment_value_gap(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    bp = ctx.load_factor("bp")

    common = mf.index.intersection(bp.index)
    gap = _rank(mf.loc[common]) - _rank(bp.loc[common])
    return cross_sectional_rank(-gap.abs())

@register_factor(
    name="factor_cycle_position",
    description="因子周期位置因子，bp偏离2年均值的符号×(偏离持续的月数)截面排名（正偏离+持续长=周期高位排前）。",
    category="coupling",
    thesis="因子值在自身历史周期中的位置是因子择时的核心——偏离历史均值的符号和持续时间共同决定了当前处于周期的什么阶段。周期分析可以帮助避免在因子周期顶部买入。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_cycle_position(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    ma_500 = bp.groupby(level="Code").transform(
        lambda s: s.rolling(500, min_periods=120).mean()
    )
    deviation = bp - ma_500

    # Count consecutive periods of same-sign deviation
    sign = np.sign(deviation)
    def _consecutive(s):
        result = pd.Series(0, index=s.index)
        cnt = 0
        prev = 0
        for i, v in enumerate(s.values):
            if np.isnan(v):
                result.iloc[i] = np.nan
                continue
            if v == prev and v != 0:
                cnt += 1
            else:
                cnt = 1
            prev = v if v != 0 else prev
            result.iloc[i] = cnt
        return result

    consecutive = sign.groupby(level="Code").transform(_consecutive)
    cycle = sign * consecutive

    return cross_sectional_rank(cycle)

@register_factor(
    name="factor_volatility_regime_shift",
    description="因子波动率状态转换因子，bp 20日std/60日std截面排名（短期波动>长期波动=状态转入高波排后）。",
    category="coupling",
    thesis="因子波动率的结构性变化(regime shift)对因子策略有重大影响——波动率突然放大意味着因子可能进入了新的状态，历史参数不再适用。识别波动率状态转换是动态因子配置的前提。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_volatility_regime_shift(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    std20 = rolling_group_std(bp, 20)
    std60 = rolling_group_std(bp, 60)
    ratio = std20 / std60.replace(0, np.nan)
    return cross_sectional_rank(-ratio)

@register_factor(
    name="factor_profile_shift_deep",
    description="因子轮廓位移因子，bp 20日前排名与当前排名的均方差截面排名（位移大=剧烈变化排后）。",
    category="coupling",
    thesis="因子截面排名的位移(profile shift)反映因子结构是否在发生根本性变化——突然的大幅位移意味着因子与股票的关系在重构，历史规律可能不再适用。稳定的因子轮廓意味着稳定的alpha预期。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_profile_shift_deep(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    rank = _rank(bp)
    rank_20 = rank.groupby(level="Code").shift(20)
    shift = (rank - rank_20).abs()
    return cross_sectional_rank(-shift)

@register_factor(
    name="factor_ic_ir_proxy_60",
    description="因子IC IR代理因子，bp 60日(均值/std)×sqrt(60)截面排名（高信息比率=因子有效性强排前）。",
    category="coupling",
    thesis="因子自身的信噪比(均值/std)乘以sqrt(N)是IC IR(信息比率)的代理——衡量因子信号相对于噪音的强度。高IC IR的因子信号更可靠，是因子权重配置的核心依据。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_ic_ir_proxy_60(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    ma60 = bp.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    std60 = bp.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    ir = (ma60.abs() / std60.replace(0, np.nan)) * np.sqrt(60)
    return cross_sectional_rank(ir)

@register_factor(
    name="factor_multi_horizon_momentum",
    description="多周期因子动量因子，(bp 5日动量排名+bp 20日动量排名+bp 60日动量排名)/3截面排名。多周期共振。",
    category="coupling",
    thesis="多周期因子动量的共振比单一周期更可靠——短中长三个周期的动量方向一致时，因子趋势最为确定。多周期共振可以过滤掉短期噪音和虚假反转。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_multi_horizon_momentum(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom5 = _momentum(bp, 5)
    mom20 = _momentum(bp, 20)
    mom60 = _momentum(bp, 60)

    common = mom5.index.intersection(mom20.index).intersection(mom60.index)
    composite = (
        _rank(mom5.loc[common]) + _rank(mom20.loc[common]) + _rank(mom60.loc[common])
    ) / 3.0

    return cross_sectional_rank(composite)

