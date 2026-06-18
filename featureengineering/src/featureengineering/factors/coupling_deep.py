"""
Extended factor coupling (扩展因子耦合) — Class 4.

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
    rolling_group_mean,
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


def _rolling_corr(a: pd.Series, b: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Per-code rolling correlation between two (Date, Code) MultiIndex Series."""
    mp = min_periods or max(1, window // 2)
    result = pd.Series(np.nan, index=a.index)
    for code in a.index.get_level_values("Code").unique():
        try:
            sa = a.xs(code, level="Code")
            sb = b.xs(code, level="Code")
            corr = sa.rolling(window, min_periods=mp).corr(sb)
            idx = pd.MultiIndex.from_arrays(
                [corr.index, [code] * len(corr)], names=["Date", "Code"]
            )
            tmp = pd.Series(corr.values, index=idx)
            result.update(tmp)
        except KeyError:
            continue
    return result.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# A — Triple Interactions & Resonance
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="value_quality_momentum_triple",
    description="价值质量动量三因子，(bp排名×roe排名×mom_20排名)截面排名。三维共振=最强信号。",
    category="coupling",
    thesis="BP(便宜)+ROE(优质)+动量(趋势)的三维共振是量化选股的'圣杯'组合——三个维度同时排名靠前的股票历史上超额收益最稳定。单维度好可能是陷阱，三维共振才是真正的alpha。",
    dependencies=("__factors__", "bp", "roe", "mom_20"),
)
def factor_value_quality_momentum_triple(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    roe = ctx.load_factor("roe")
    mom = ctx.load_factor("mom_20")

    common = bp.index.intersection(roe.index).intersection(mom.index)
    triple = _rank(bp.loc[common]) * _rank(roe.loc[common]) * _rank(mom.loc[common])
    return cross_sectional_rank(triple)


@register_factor(
    name="bp_mom_resonance",
    description="BP-动量共振因子，(bp排名-0.5)×(mom_20排名-0.5)×2截面排名（正向共振=高估值+正动量排前）。",
    category="coupling",
    thesis="BP和动量的共振(同向)比背离更具信息量——高BP+正动量=价值重估确认，低BP+负动量=价值陷阱。共振强度量化了价值和趋势的'协调度'。",
    dependencies=("__factors__", "bp", "mom_20"),
)
def factor_bp_mom_resonance(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")

    common = bp.index.intersection(mom.index)
    bp_dev = _rank(bp.loc[common]) - 0.5
    mom_dev = _rank(mom.loc[common]) - 0.5
    resonance = bp_dev * mom_dev * 2.0  # range [-0.5, 0.5]
    return cross_sectional_rank(resonance)


@register_factor(
    name="roe_turnover_interaction",
    description="ROE-换手率交互因子，(roe排名)×(1-换手率排名)截面排名（高质量+低换手=价值沉淀排前）。",
    category="coupling",
    thesis="ROE和换手率的交互识别'被低估的优质公司'——高ROE但低换手意味着优质公司尚未被市场充分发现，是价值投资的核心标的。高ROE+高换手则可能已被充分定价。",
    dependencies=("__factors__", "roe", "turnover_20"),
)
def factor_roe_turnover_interaction(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    turnover = ctx.load_factor("turnover_20")

    common = roe.index.intersection(turnover.index)
    interaction = _rank(roe.loc[common]) * (1 - _rank(turnover.loc[common]))
    return cross_sectional_rank(interaction)


@register_factor(
    name="size_momentum_interaction",
    description="规模-动量交互因子，(1-规模排名)×动量排名截面排名（小盘+高动量=成长爆发排前）。",
    category="coupling",
    thesis="规模和动量的交互捕捉小盘成长股的爆发力——小市值股票的高动量往往伴随更大的上涨空间。大盘股的动量虽然可靠但弹性有限，小盘+高动量是A股中弹性最大的因子组合。",
    dependencies=("__factors__", "log_total_mv", "mom_20"),
)
def factor_size_momentum_interaction(ctx: FactorContext) -> pd.Series:
    size = ctx.load_factor("log_total_mv")
    mom = ctx.load_factor("mom_20")

    common = size.index.intersection(mom.index)
    interaction = (1 - _rank(size.loc[common])) * _rank(mom.loc[common])
    return cross_sectional_rank(interaction)


@register_factor(
    name="volatility_reversal_interaction",
    description="波动率-反转交互因子，(高波动率排名)×(低动量排名)截面排名（高波+超跌=反弹潜力排前）。",
    category="coupling",
    thesis="高波动率股票在经历大幅下跌后的反弹力度最强——波动率提供了反弹的'弹簧'，低动量提供了反弹的'位置'(超跌)。高波+超跌是经典的均值回复策略标的。",
    dependencies=("__factors__", "volatility_20", "mom_20"),
)
def factor_volatility_reversal_interaction(ctx: FactorContext) -> pd.Series:
    vol = ctx.load_factor("volatility_20")
    mom = ctx.load_factor("mom_20")

    common = vol.index.intersection(mom.index)
    interaction = _rank(vol.loc[common]) * (1 - _rank(mom.loc[common]))
    return cross_sectional_rank(interaction)


@register_factor(
    name="chip_momentum_quality",
    description="筹码-动量-质量三因子，(筹码集中排名×动量排名×ROE排名)截面排名。",
    category="coupling",
    thesis="筹码集中度、价格动量和盈利质量的三维共振是中线选股的最强组合——筹码集中意味着主力在收集，动量确认趋势，质量保证基本面支撑。三维共振的股票是中线最确定的alpha来源。",
    dependencies=("__factors__", "winner_rate", "mom_20", "roe"),
)
def factor_chip_momentum_quality(ctx: FactorContext) -> pd.Series:
    wr = ctx.load_factor("winner_rate")
    mom = ctx.load_factor("mom_20")
    roe = ctx.load_factor("roe")

    common = wr.index.intersection(mom.index).intersection(roe.index)
    triple = (1 - _rank(wr.loc[common])) * _rank(mom.loc[common]) * _rank(roe.loc[common])
    return cross_sectional_rank(triple)


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
    name="sector_momentum_coupling",
    description="板块-个股动量耦合因子，板块动量排名×个股动量排名截面排名。板块与个股动量共振。",
    category="coupling",
    thesis="板块动量与个股动量的耦合效应是行业轮动策略的核心——在强势板块中选择强势个股(双强)是顺势而为的最佳策略。板块弱而个股强可能不可持续，板块强而个股弱则有补涨空间。",
    dependencies=("__factors__", "sector_mv_rank", "mom_20"),
)
def factor_sector_momentum_coupling(ctx: FactorContext) -> pd.Series:
    sector = ctx.load_factor("sector_mv_rank")
    mom = ctx.load_factor("mom_20")

    common = sector.index.intersection(mom.index)
    coupling = _rank(sector.loc[common]) * _rank(mom.loc[common])
    return cross_sectional_rank(coupling)


@register_factor(
    name="index_weight_momentum_coupling",
    description="指数权重-动量耦合因子，沪深300权重排名×动量排名截面排名。被动资金+趋势双击。",
    category="coupling",
    thesis="指数权重高的股票叠加正动量是'被动+主动'双驱动——不仅受益于指数基金的被动配置，还得到了主动趋势资金的青睐。这种双驱动在牛市环境中表现最优。",
    dependencies=("__factors__", "index_weight_hs300", "mom_20"),
)
def factor_index_weight_momentum_coupling(ctx: FactorContext) -> pd.Series:
    iw = ctx.load_factor("index_weight_hs300")
    mom = ctx.load_factor("mom_20")

    common = iw.index.intersection(mom.index)
    coupling = _rank(iw.loc[common]) * _rank(mom.loc[common])
    return cross_sectional_rank(coupling)


@register_factor(
    name="low_vol_quality_coupling",
    description="低波-质量耦合因子，(1-波动率排名)×ROE排名截面排名（高质量+低波动=防御成长排前）。",
    category="coupling",
    thesis="低波动+高质量是防御型成长策略的核心——此类公司在熊市中跌幅小(低波的保护)，在牛市中也能涨(质量的驱动)。低波质量组合在A股中长期夏普比率最高。",
    dependencies=("__factors__", "volatility_20", "roe"),
)
def factor_low_vol_quality_coupling(ctx: FactorContext) -> pd.Series:
    vol = ctx.load_factor("volatility_20")
    roe = ctx.load_factor("roe")

    common = vol.index.intersection(roe.index)
    coupling = (1 - _rank(vol.loc[common])) * _rank(roe.loc[common])
    return cross_sectional_rank(coupling)


# ═══════════════════════════════════════════════════════════════════════════════
# B — Factor Quality & Dynamics
# ═══════════════════════════════════════════════════════════════════════════════


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
    name="factor_outlier_frequency",
    description="因子极端值频率因子，-(bp 60日滚动|zscore|>2的次数占比)截面排名（频繁极端=不稳定排后）。",
    category="coupling",
    thesis="因子值频繁出现极端值(zscore>2)意味着该股票在该因子维度上波动剧烈——可能是数据质量问题，也可能是公司基本面确实在大幅变动。无论原因，高极端值频率都会降低因子信号的可靠性。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_outlier_frequency(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    z = _zscore(bp)
    is_outlier = (z.abs() > 2.0).astype(float)
    freq = is_outlier.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    return cross_sectional_rank(-freq)



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
# C — Orthogonalization & Neutral
# ═══════════════════════════════════════════════════════════════════════════════




@register_factor(
    name="multi_factor_orthogonal",
    description="多因子正交残差因子，mom_20对size+bp+volatility回归残差截面排名（纯动量alpha）。",
    category="coupling",
    thesis="将动量对规模、价值和波动率同时正交化后，得到的残差是最纯净的动量alpha——它剔除了常见风险因子的影响，代表了无法被其他因子解释的独立趋势信号。",
    dependencies=("__factors__", "mom_20", "log_total_mv", "bp", "volatility_20"),
)
def factor_multi_factor_orthogonal(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_20")
    size = ctx.load_factor("log_total_mv")
    bp = ctx.load_factor("bp")
    vol = ctx.load_factor("volatility_20")

    common = mom.index.intersection(size.index).intersection(bp.index).intersection(vol.index)
    mom_a = mom.loc[common]
    size_a = size.loc[common]
    bp_a = bp.loc[common]
    vol_a = vol.loc[common]

    residual = pd.Series(np.nan, index=mom_a.index)
    for date in mom_a.index.get_level_values("Date").unique():
        m = mom_a.xs(date, level="Date")
        s = size_a.xs(date, level="Date")
        b = bp_a.xs(date, level="Date")
        v = vol_a.xs(date, level="Date")
        mask = m.notna() & s.notna() & b.notna() & v.notna()
        if mask.sum() < 50:
            continue
        X = np.column_stack([s[mask].values, b[mask].values, v[mask].values])
        y = m[mask].values
        try:
            coeffs = np.linalg.lstsq(
                np.column_stack([np.ones(len(y)), X]), y, rcond=None
            )[0]
            predicted = coeffs[0] + X @ coeffs[1:]
            resid_vals = y - predicted
            idx = pd.MultiIndex.from_arrays(
                [[date] * len(resid_vals), s[mask].index], names=["Date", "Code"]
            )
            tmp = pd.Series(resid_vals, index=idx)
            residual.update(tmp)
        except np.linalg.LinAlgError:
            continue

    return cross_sectional_rank(residual)


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
    name="factor_anti_crowding",
    description="反拥挤信号因子，bp排名处于中位(0.3-0.7)+近期从极端回归截面排名（均值回复机会排前）。",
    category="coupling",
    thesis="因子从极端分位回归到中位是'反拥挤'机会——被过度追捧或抛弃的股票正在回归正常，拥挤风险在下降。反拥挤是逆向投资在因子维度的应用。",
    dependencies=("__factors__", "bp"),
)
def factor_factor_anti_crowding(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    rank = _rank(bp)
    is_mid = ((rank > 0.3) & (rank < 0.7)).astype(float)

    # Was extreme 20 days ago
    rank_20 = rank.groupby(level="Code").shift(20)
    was_extreme = ((rank_20 > 0.85) | (rank_20 < 0.15)).astype(float)

    anti_crowd = is_mid * was_extreme
    return cross_sectional_rank(anti_crowd)



@register_factor(
    name="tail_dependence_score",
    description="尾部依赖评分因子，bp极端低(<0.1分位)时mom_20也极端低的概率截面排名（取负向=高尾部依赖=系统风险排后）。",
    category="coupling",
    thesis="因子间的尾部依赖(Tail Dependence)是极端风险的重要度量——当一个因子处于极端值时另一个因子也倾向于极端，说明两者在危机中会同时恶化。高尾部依赖因子组合的分散化效果在危机中大打折扣。",
    dependencies=("__factors__", "bp", "mom_20"),
)
def factor_tail_dependence_score(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")

    common = bp.index.intersection(mom.index)
    bp_aligned = bp.loc[common]
    mom_aligned = mom.loc[common]

    bp_rank = _rank(bp_aligned)
    mom_rank = _rank(mom_aligned)

    # Lower tail: both in bottom 10%
    bp_low = (bp_rank < 0.1).astype(float)
    mom_low = (mom_rank < 0.1).astype(float)
    joint_low = bp_low * mom_low

    tail_dep = joint_low.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    ) / bp_low.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    ).replace(0, np.nan)

    return cross_sectional_rank(-tail_dep)


# ═══════════════════════════════════════════════════════════════════════════════
# E — Sentiment & Regime
# ═══════════════════════════════════════════════════════════════════════════════


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
    name="sentiment_quality_conflict",
    description="情绪-质量背离因子，-(资金流高排名+ROE低排名)截面排名（炒作无业绩=危险信号排后）。",
    category="coupling",
    thesis="资金追捧但盈利质量差是'题材炒作'的典型特征——这种背离在A股中短期可能赚钱，但中长期必然回归。情绪-质量背离度高的股票应回避。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "roe"),
)
def factor_sentiment_quality_conflict(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    roe = ctx.load_factor("roe")

    common = mf.index.intersection(roe.index)
    conflict = _rank(mf.loc[common]) * (1 - _rank(roe.loc[common]))
    return cross_sectional_rank(-conflict)


@register_factor(
    name="sentiment_momentum_confirmation",
    description="情绪-动量确认因子，资金流排名×动量排名截面排名（情绪+趋势双确认=最强短线排前）。",
    category="coupling",
    thesis="资金情绪和价格趋势的同向确认是短线交易的最强信号——资金在买+价格在涨=市场共识。两者背离(资金在买但价格在跌)则需要更深入分析。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "mom_20"),
)
def factor_sentiment_momentum_confirmation(ctx: FactorContext) -> pd.Series:
    mf = ctx.load_factor("mf_net_inflow_ratio")
    mom = ctx.load_factor("mom_20")

    common = mf.index.intersection(mom.index)
    confirm = _rank(mf.loc[common]) * _rank(mom.loc[common])
    return cross_sectional_rank(confirm)



@register_factor(
    name="momentum_value_resonance_deep",
    description="动量价值深度共振因子，(mom_20排名×bp排名)×(1+|mom_20排名-bp排名|)截面排名。",
    category="coupling",
    thesis="动量与价值不仅需要同向(共振)，还需要强度匹配——高动量+高价值的'完美共振'比一高一低的'弱共振'更有投资价值。深度共振评分综合了方向一致性和强度匹配度。",
    dependencies=("__factors__", "mom_20", "bp"),
)
def factor_momentum_value_resonance_deep(ctx: FactorContext) -> pd.Series:
    mom = ctx.load_factor("mom_20")
    bp = ctx.load_factor("bp")

    common = mom.index.intersection(bp.index)
    mom_r = _rank(mom.loc[common])
    bp_r = _rank(bp.loc[common])

    resonance = mom_r * bp_r * (1 - np.abs(mom_r - bp_r))
    return cross_sectional_rank(resonance)


@register_factor(
    name="quality_growth_nonlinear",
    description="质量成长非线性因子，roe排名×or_yoy因子排名×(两者同向=正否则负)截面排名。",
    category="coupling",
    thesis="质量和成长的交互是非线性的——高ROE+高成长是最优组合，但高ROE+负成长(成熟期)和低ROE+高成长(投入期)各有不同的投资逻辑。两者同向时(高高或低低)信号更明确，异向时需要更细致的分析。",
    dependencies=("__factors__", "roe", "mom_20"),
)
def factor_quality_growth_nonlinear(ctx: FactorContext) -> pd.Series:
    roe = ctx.load_factor("roe")
    mom = ctx.load_factor("mom_20")

    common = roe.index.intersection(mom.index)
    roe_r = _rank(roe.loc[common])
    mom_r = _rank(mom.loc[common])

    # Same direction = positive product
    nonlinear = roe_r * mom_r * np.sign(roe_r - 0.5) * np.sign(mom_r - 0.5)
    return cross_sectional_rank(nonlinear)


# ═══════════════════════════════════════════════════════════════════════════════
# F — Factor Dynamics Extended
# ═══════════════════════════════════════════════════════════════════════════════


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
    name="factor_diversification_ratio",
    description="因子分散化比率因子，(bp+mom_20+roe)三个因子排名差异的std截面排名（取负向=差异大=互补强排后）。",
    category="coupling",
    thesis="多个因子排名之间的差异(分散化比率)衡量因子组合的多样性——三个因子排名差异越大,组合的分散化越充分。分散化比率高的股票在单一因子失效时仍有其他因子提供支撑。",
    dependencies=("__factors__", "bp", "mom_20", "roe"),
)
def factor_factor_diversification_ratio(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")
    roe = ctx.load_factor("roe")

    common = bp.index.intersection(mom.index).intersection(roe.index)
    bp_r = _rank(bp.loc[common])
    mom_r = _rank(mom.loc[common])
    roe_r = _rank(roe.loc[common])

    # Standard deviation of the 3 ranks (higher = more diverse)
    rank_df = pd.DataFrame({"bp": bp_r, "mom": mom_r, "roe": roe_r})
    dispersion = rank_df.std(axis=1)

    return cross_sectional_rank(dispersion)


@register_factor(
    name="factor_weight_ewma_vol",
    description="逆波动率加权因子，(bp/std_bp+mom_20/std_mom+roe/std_roe)截面排名。低波动因子权重更高。",
    category="coupling",
    thesis="逆波动率加权是最简单的动态因子配置方法——给近期波动率低的因子更高权重(信号更稳定)，波动率高的因子低权重(信号噪音大)。相比等权组合，逆波动率加权的IR通常更高。",
    dependencies=("__factors__", "bp", "mom_20", "roe"),
)
def factor_factor_weight_ewma_vol(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")
    roe = ctx.load_factor("roe")

    common = bp.index.intersection(mom.index).intersection(roe.index)
    bp_a = bp.loc[common]
    mom_a = mom.loc[common]
    roe_a = roe.loc[common]

    bp_std = rolling_group_std(bp_a, 60)
    mom_std = rolling_group_std(mom_a, 60)
    roe_std = rolling_group_std(roe_a, 60)

    # Inverse vol weighted
    w_bp = 1.0 / bp_std.replace(0, np.nan)
    w_mom = 1.0 / mom_std.replace(0, np.nan)
    w_roe = 1.0 / roe_std.replace(0, np.nan)
    w_sum = w_bp + w_mom + w_roe

    composite = (bp_a * w_bp + mom_a * w_mom + roe_a * w_roe) / w_sum.replace(0, np.nan)
    return cross_sectional_rank(composite)


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



@register_factor(
    name="factor_style_rotation_20",
    description="因子风格轮动信号因子，(bp排名20日变化-mom_20排名20日变化)截面排名（正=转向价值排前）。",
    category="coupling",
    thesis="因子排名的相对变化捕捉因子层面的风格轮动——价值排名上升+动量排名下降=市场正在转向价值风格。因子风格轮动信号是宏观因子配置的核心参考。",
    dependencies=("__factors__", "bp", "mom_20"),
)
def factor_factor_style_rotation_20(ctx: FactorContext) -> pd.Series:
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("mom_20")

    common = bp.index.intersection(mom.index)
    bp_chg = _rank(bp.loc[common]).groupby(level="Code").diff(20)
    mom_chg = _rank(mom.loc[common]).groupby(level="Code").diff(20)

    rotation = bp_chg.fillna(0) - mom_chg.fillna(0)
    return cross_sectional_rank(rotation)
