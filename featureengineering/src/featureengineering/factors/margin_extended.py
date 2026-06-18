"""
Extended margin trading factors (扩展融资融券因子) — Class 1.

These factors extend the basic margin factors in ``margin.py`` with flow
dynamics, short-selling signals, squeeze detection, leverage metrics,
composite indicators, and divergence/confirmation signals.

Data source: ``margin_detail.parquet`` (daily, 2019-01-02 onward).
Fields: rzye (融资余额), rzmre (融资买入额), rzche (融资偿还额),
        rqye (融券余额), rqmcl (融券卖出量), rzrqye (融资融券余额),
        rqyl (融券余量), is_backfill
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    rolling_group_sum,
    safe_divide,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _net_margin_flow(margin: pd.DataFrame) -> pd.Series:
    """Daily net margin flow = 融资买入 - 融资偿还."""
    return margin["rzmre"] - margin["rzche"]


# ═══════════════════════════════════════════════════════════════════════════════
# A — Net Flow & Flow Dynamics
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="margin_net_flow_5d",
    description="融资净流因子，5日累计净融资流（买入-偿还）截面排名（净流入=看多排前）。",
    category="margin",
    thesis="融资净流入是杠杆资金态度的最直接指标——连续净买入意味着高风险偏好资金在持续做多。5日累计平滑了单日噪音，反映的是中期趋势而非偶尔的脉冲。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_net_flow_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    net = _net_margin_flow(margin)
    net_5d = rolling_group_sum(net, 5)
    return cross_sectional_rank(net_5d)


@register_factor(
    name="margin_flow_volatility_20",
    description="融资流波动率因子，20日净融资流标准差截面排名（取负向=高波动排后）。",
    category="margin",
    thesis="融资流的稳定性反映杠杆资金态度的坚定程度——融资流忽进忽出（高波动）意味着杠杆资金信心不足、短线投机为主。低波动的持续融资流入才是健康的加杠杆行为。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_flow_volatility_20(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    net = _net_margin_flow(margin)
    vol = rolling_group_std(net, 20)
    return cross_sectional_rank(-vol)





@register_factor(
    name="margin_flow_reversal_5d",
    description="融资流反转因子，-(5日净流与20日净流方向相反的程度)截面排名（反转=态度转向排后）。",
    category="margin",
    thesis="融资资金方向的反转（从流入转流出或反之）可能意味着杠杆资金嗅到了风险或机会——突然的反转往往领先于价格的反转，因为杠杆资金对强制平仓风险极为敏感。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_flow_reversal_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    net = _net_margin_flow(margin)

    net_5d = rolling_group_sum(net, 5)
    net_20d = rolling_group_sum(net, 20)

    # Directional alignment: positive if same sign, negative if opposite
    alignment = np.sign(net_5d) * np.sign(net_20d)
    # Reversal = -alignment (high when signs differ)
    reversal = -alignment

    return cross_sectional_rank(-reversal)  # No reversal = good


@register_factor(
    name="margin_buy_climax",
    description="融资买入高潮因子，融资买入额相对60日均值的偏离度截面排名（取负向=极端买入=潜在顶部排后）。",
    category="margin",
    thesis="融资买入的极端值往往是情绪高点的标志——融资买入突然飙升至历史均值2倍以上通常发生在股价已大幅上涨之后，是'最后的追涨'。对极端买入保持警惕，具有逆向投资价值。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_buy_climax(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    buy = margin["rzmre"]

    buy_ma60 = rolling_group_mean(buy, 60)
    buy_std60 = rolling_group_std(buy, 60)

    # How many standard deviations above mean
    climax = (buy - buy_ma60) / buy_std60.replace(0, np.nan)

    return cross_sectional_rank(-climax)


@register_factor(
    name="margin_panic_repay",
    description="恐慌偿还因子，-(融资偿还额超过60日均值3σ的程度)截面排名（恐慌偿还=强制平仓风险排后）。",
    category="margin",
    thesis="融资偿还的极端飙升往往是强制平仓的结果——股价快速下跌触发维持担保比例红线，券商强制卖出导致偿还额骤增。这是恐慌的量化指标，通常标志着阶段性底部或继续下跌的分水岭。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_panic_repay(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    repay = margin["rzche"]

    repay_ma60 = rolling_group_mean(repay, 60)
    repay_std60 = rolling_group_std(repay, 60)

    panic = (repay - repay_ma60) / repay_std60.replace(0, np.nan)

    return cross_sectional_rank(-panic)


# ═══════════════════════════════════════════════════════════════════════════════
# B — Short Selling Signals
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="short_sell_activity_5d",
    description="融券活跃度因子，5日融券卖出量均值截面排名（取负向=高融券=看空排后）。",
    category="margin",
    thesis="融券卖出量是A股中稀缺的直接做空指标——融券卖出活跃的股票面临真实的卖空压力。在A股融券成本较高的环境下，持续被融券做空的股票通常有基本面或估值上的硬伤。",
    dependencies=("margin_detail.parquet",),
)
def factor_short_sell_activity_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    short_sell = margin["rqmcl"]
    short_5d = rolling_group_mean(short_sell, 5)
    return cross_sectional_rank(-short_5d)


@register_factor(
    name="short_sell_momentum_5d",
    description="融券动量因子，融券卖出量5日变化率截面排名（取负向=融券加速=做空加剧排后）。",
    category="margin",
    thesis="融券卖出量的边际变化比绝对水平更具信号价值——融券量在增加意味着做空力量在集结，可能在为更大的下跌做准备。融券边际增加的股票短期面临更大的下行压力。",
    dependencies=("margin_detail.parquet",),
)
def factor_short_sell_momentum_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    short_sell = margin["rqmcl"]
    chg = short_sell.groupby(level="Code").transform(lambda s: s.pct_change(5))
    return cross_sectional_rank(-chg)


@register_factor(
    name="short_sell_to_turnover",
    description="融券换手比因子，融券卖出量/日换手率截面排名（取负向=高融券占比排后）。",
    category="margin",
    thesis="融券量相对换手率的比值衡量做空力量在市场交易中的占比——高比值意味着每100股交易中就有显著的做空盘，空方力量强大。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_short_sell_to_turnover(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    finance = context.load("finance.parquet")

    short_sell = margin["rqmcl"]
    turnover = finance["turnover_rate"]

    common = short_sell.index.intersection(turnover.index)
    ratio = short_sell.loc[common] / turnover.loc[common].replace(0, np.nan)

    return cross_sectional_rank(-ratio)


@register_factor(
    name="short_sell_concentration",
    description="融券集中度因子，融券卖出量20日日间变异系数截面排名（取负向=集中做空=冲击大排后）。",
    category="margin",
    thesis="融券卖出在时间上的集中度反映了做空的'组织性'——集中在某几天大量融券（高CV）可能是有组织的做空行为，相比分散做空具有更大的价格冲击力。",
    dependencies=("margin_detail.parquet",),
)
def factor_short_sell_concentration(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    short_sell = margin["rqmcl"]

    mean_20 = rolling_group_mean(short_sell, 20)
    std_20 = rolling_group_std(short_sell, 20)

    cv = std_20 / mean_20.replace(0, np.nan)  # coefficient of variation
    return cross_sectional_rank(-cv)


@register_factor(
    name="short_squeeze_potential",
    description="逼空潜力因子，高融券余额+5日正收益截面排名（融券多+价格上涨=空头被挤压排前）。",
    category="margin",
    thesis="当融券余额高企而股价逆势上涨时，空头面临巨大亏损——被迫回补的压力会进一步推高股价，形成逼空(squeeze)行情。A股中虽不如美股常见，但在融券集中的中小市值标的中时有发生。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_short_squeeze_potential(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    rqye = margin["rqye"]  # 融券余额
    close = daily_adj["close"]
    ret_5d = close.groupby(level="Code").transform(lambda s: s.pct_change(5))

    common = rqye.index.intersection(ret_5d.index)
    rqye_aligned = rqye.loc[common]
    ret_aligned = ret_5d.loc[common]

    # Squeeze = high short balance + rising price
    rqye_rank = rqye_aligned.groupby(level="Date").rank(pct=True)
    ret_rank = ret_aligned.groupby(level="Date").rank(pct=True)

    squeeze = rqye_rank * ret_rank
    return cross_sectional_rank(squeeze)


@register_factor(
    name="short_cover_rally_signal",
    description="空头回补信号因子，融券余额下降+价格上涨截面排名（空头回补=买入力量排前）。",
    category="margin",
    thesis="融券余额下降+价格上涨是空头回补(short cover)的典型特征——做空者买入平仓，形成额外的买入需求。空头回补行情往往是短期、快速且幅度可观的，是事件驱动策略的优质信号。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_short_cover_rally_signal(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    rqye = margin["rqye"]
    close = daily_adj["close"]

    rqye_chg = rqye.groupby(level="Code").transform(lambda s: s.diff(5))
    ret_5d = close.groupby(level="Code").transform(lambda s: s.pct_change(5))

    common = rqye_chg.index.intersection(ret_5d.index)
    rqye_chg_aligned = rqye_chg.loc[common]
    ret_aligned = ret_5d.loc[common]

    # Cover rally = decreasing short + rising price
    short_decline_rank = (-rqye_chg_aligned).groupby(level="Date").rank(pct=True)
    price_rise_rank = ret_aligned.groupby(level="Date").rank(pct=True)

    cover_signal = short_decline_rank * price_rise_rank
    return cross_sectional_rank(cover_signal)


# ═══════════════════════════════════════════════════════════════════════════════
# C — Leverage & Cost Metrics
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="margin_to_float_mv",
    description="融资市值比因子，融资余额/流通市值截面排名（取负向=高杠杆排后=风险信号）。",
    category="margin",
    thesis="融资余额占流通市值的比重是杠杆率的直接度量——高杠杆率意味着股价中包含了大量借来的资金，一旦市场转向，强制平仓的连锁反应将放大跌幅。融资/市值是预测尾部风险的重要指标。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_margin_to_float_mv(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    finance = context.load("finance.parquet")

    rzye = margin["rzye"]
    circ_mv = finance["circ_mv"]

    common = rzye.index.intersection(circ_mv.index)
    ratio = rzye.loc[common] / circ_mv.loc[common].replace(0, np.nan)

    return cross_sectional_rank(-ratio)


@register_factor(
    name="margin_to_turnover",
    description="融资成交比因子，日融资买入额/日成交额截面排名（高融资占比=杠杆驱动排前）。",
    category="margin",
    thesis="融资买入在日成交额中的占比反映了当天交易的'杠杆驱动'程度——高占比意味着当日成交主要由杠杆资金推动，价格上涨的'质量'相对较低（借来的钱而非自有资金）。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_margin_to_turnover(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    rzmre = margin["rzmre"]
    amount = daily_adj["amount"]

    common = rzmre.index.intersection(amount.index)
    ratio = rzmre.loc[common] / amount.loc[common].replace(0, np.nan)

    return cross_sectional_rank(ratio)


@register_factor(
    name="margin_leverage_change_5d",
    description="杠杆率变化因子，融资余额/流通市值 5日变化截面排名（去杠杆=风险厌恶排后）。",
    category="margin",
    thesis="杠杆率的边际变化反映风险偏好的转变——杠杆率下降意味着资金在主动或被强制去杠杆，是风险厌恶上升的信号。杠杆率持续下降的股票面临持续的卖出压力。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_margin_leverage_change_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    finance = context.load("finance.parquet")

    rzye = margin["rzye"]
    circ_mv = finance["circ_mv"]

    common = rzye.index.intersection(circ_mv.index)
    leverage = rzye.loc[common] / circ_mv.loc[common].replace(0, np.nan)
    leverage_chg = leverage.groupby(level="Code").transform(lambda s: s.diff(5))

    return cross_sectional_rank(leverage_chg)


@register_factor(
    name="margin_cost_burden",
    description="融资成本负担因子，-(融资余额×融资利率8%)/净利润TTM估计截面排名（高成本负担=财务压力排后）。",
    category="margin",
    thesis="融资利息是融资交易最被忽视的成本——年化8%的融资利率意味着每1亿融资余额每年产生800万利息费用。对净利润微薄的公司而言，融资利息可能侵蚀相当比例的利润，形成'越跌越融、越融越亏'的恶性循环。",
    dependencies=("margin_detail.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_margin_cost_burden(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    fin = context.load_financial("financial_indicator.parquet", value_cols=["profit_dedt"])

    rzye = margin["rzye"]
    profit = fin["profit_dedt"]

    common = rzye.index.intersection(profit.index)
    rzye_aligned = rzye.loc[common]
    profit_aligned = profit.loc[common]

    # Annual interest cost ≈ 8% of margin balance
    interest_cost = rzye_aligned * 0.08
    # TTM profit proxy: quarterly profit × 4
    burden = interest_cost / profit_aligned.replace(0, np.nan).abs()

    return cross_sectional_rank(-burden)


@register_factor(
    name="margin_utilization_rate",
    description="融资使用率因子，融资余额/全市场融资余额截面排名（融资集中度=该股在融资体系中的重要性）。",
    category="margin",
    thesis="融资余额的市场占比反映该股在融资体系中的'系统重要性'——融资集中在少数股票中意味着这些股票承载了最多的杠杆风险。融资集中度高的股票在市场下跌时面临更大的系统性去杠杆压力。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_utilization_rate(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    rzye = margin["rzye"]

    total_market = rzye.groupby(level="Date").sum()
    dates = rzye.index.get_level_values("Date")
    total_mapped = pd.Series(total_market.loc[dates].values, index=rzye.index)

    utilization = rzye / total_mapped.replace(0, np.nan)

    return cross_sectional_rank(utilization)


# ═══════════════════════════════════════════════════════════════════════════════
# D — Composite Indicators
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="margin_bullish_composite",
    description="多头融资综合因子，(买入排名+净流排名+余额增长排名)/3截面排名。",
    category="margin",
    thesis="融资买入量+净流入+余额增长三个维度同时向好是杠杆资金全面看多的最强信号——三维共振排除了单一指标的噪音，是顺势做多的可靠确认。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_bullish_composite(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    buy = margin["rzmre"]
    net = _net_margin_flow(margin)
    balance_chg = margin["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(5))

    buy_rank = buy.groupby(level="Date").rank(pct=True)
    net_rank = net.groupby(level="Date").rank(pct=True)
    bal_rank = balance_chg.groupby(level="Date").rank(pct=True)

    composite = (buy_rank + net_rank + bal_rank) / 3.0
    return cross_sectional_rank(composite)


@register_factor(
    name="margin_bearish_composite",
    description="空头融资综合因子，(偿还排名+融券排名+余额下降排名)/3截面排名（取负向=看空信号排后）。",
    category="margin",
    thesis="融资偿还增加+融券卖出增加+融资余额下降是杠杆资金全面撤退的标志——三维同时恶化往往预示着更大级别的下跌行情。空头综合指标是大盘系统性风险预警的微观基础。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_bearish_composite(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    repay = margin["rzche"]
    short_sell = margin["rqmcl"]
    balance_chg = -margin["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(5))

    repay_rank = repay.groupby(level="Date").rank(pct=True)
    short_rank = short_sell.groupby(level="Date").rank(pct=True)
    bal_rank = balance_chg.groupby(level="Date").rank(pct=True)

    composite = (repay_rank + short_rank + bal_rank) / 3.0
    return cross_sectional_rank(-composite)


@register_factor(
    name="margin_extreme_positioning",
    description="融资极端定位因子，融资指标（余额+买入+净流）在各自1年历史中的综合分位截面排名（极端高位=过热排后）。",
    category="margin",
    thesis="融资指标处于历史极端高位是'融资过热'的量化表达——与估值指标的历史分位类似，融资指标的历史高分位往往对应着行情的后期阶段。逆向降低极端融资股票的配置是风险管理的有效手段。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_extreme_positioning(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    rzye = margin["rzye"]
    buy = margin["rzmre"]
    net = _net_margin_flow(margin)

    # 252-day rolling percentile rank
    rzye_pct = rzye.groupby(level="Code").transform(
        lambda s: s.rolling(252, min_periods=60).rank(pct=True))
    buy_pct = buy.groupby(level="Code").transform(
        lambda s: s.rolling(252, min_periods=60).rank(pct=True))
    net_5d = rolling_group_sum(net, 5)
    net_pct = net_5d.groupby(level="Code").transform(
        lambda s: s.rolling(252, min_periods=60).rank(pct=True))

    extreme = (rzye_pct + buy_pct + net_pct) / 3.0
    return cross_sectional_rank(-extreme)


# ═══════════════════════════════════════════════════════════════════════════════
# E — Divergence & Confirmation Signals
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="margin_price_divergence_5d",
    description="融资价格背离因子，-(5日价格收益)×(5日融资净流入)截面排名（价格跌+融资买=背离抄底排前）。",
    category="margin",
    thesis="价格下跌但融资逆势买入是最有价值的背离信号——杠杆资金在别人恐惧时贪婪，这种'聪明钱'行为往往预示着价格即将反弹。价格-融资背离是事件驱动和逆向投资策略的核心信号。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_margin_price_divergence_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    net = _net_margin_flow(margin)
    net_5d = rolling_group_sum(net, 5)

    close = daily_adj["close"]
    ret_5d = close.groupby(level="Code").transform(lambda s: s.pct_change(5))

    common = net_5d.index.intersection(ret_5d.index)
    net_aligned = net_5d.loc[common]
    ret_aligned = ret_5d.loc[common]

    net_rank = net_aligned.groupby(level="Date").rank(pct=True)
    ret_rank = ret_aligned.groupby(level="Date").rank(pct=True)

    # Divergence = net buying + price decline (inverse of ret)
    divergence = net_rank * (1 - ret_rank)
    return cross_sectional_rank(divergence)


@register_factor(
    name="margin_breakout_confirmation",
    description="融资突破确认因子，价格创20日新高+融资买入创20日新高截面排名（价格突破获融资确认排前）。",
    category="margin",
    thesis="价格突破伴随融资买入的同步新高是技术分析方法中'价量配合'在融资维度的体现——融资资金的积极参与确认了突破的有效性，降低了假突破的概率。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_margin_breakout_confirmation(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    buy = margin["rzmre"]
    close = daily_adj["close"]

    buy_20d_high = buy.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max())
    price_20d_high = close.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max())

    price_breakout = (close >= price_20d_high.shift(1)).astype(float)
    buy_breakout = (buy >= buy_20d_high.shift(1)).astype(float)

    common = price_breakout.index.intersection(buy_breakout.index)
    confirmation = price_breakout.loc[common] * buy_breakout.loc[common]

    # Rolling 5-day average of confirmation signals
    conf_5d = confirmation.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    return cross_sectional_rank(conf_5d)


@register_factor(
    name="margin_flush_out_signal",
    description="融资出清信号因子，5日跌幅>10%+融资余额5日降幅>10%截面排名（暴跌+去杠杆=恐慌底排前）。",
    category="margin",
    thesis="股价暴跌叠加融资余额骤降是典型的'强制平仓底'信号——高杠杆持仓在股价快速下跌时被迫平仓，形成'多杀多'的踩踏。当去杠杆过程基本完成时，卖压衰竭，往往形成阶段性底部。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_margin_flush_out_signal(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    rzye = margin["rzye"]
    close = daily_adj["close"]

    rzye_chg = rzye.groupby(level="Code").transform(lambda s: s.pct_change(5))
    price_chg = close.groupby(level="Code").transform(lambda s: s.pct_change(5))

    common = rzye_chg.index.intersection(price_chg.index)
    rzye_aligned = rzye_chg.loc[common]
    price_aligned = price_chg.loc[common]

    # Severe decline + margin reduction = flush out
    price_severe = (-price_aligned).clip(lower=0)  # how negative
    margin_reduce = (-rzye_aligned).clip(lower=0)  # how much margin reduced

    flush = price_severe * margin_reduce
    return cross_sectional_rank(flush)


# ═══════════════════════════════════════════════════════════════════════════════
# F — Extended Margin Factors
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="margin_trend_strength_20",
    description="融资趋势强度因子，净融资流20日方向一致性（同向天数/20）截面排名（强趋势排前）。",
    category="margin",
    thesis="融资流的方向一致性（净流入连续天数占比）是趋势强度的度量——连续20天中18天净流入比20天累计净流入更能反映杠杆资金的坚定态度。方向一致性高的股票趋势更可靠。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_trend_strength_20(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    net = _net_margin_flow(margin)

    is_positive = (net > 0).astype(float)
    trend_strength = rolling_group_mean(is_positive, 20)

    return cross_sectional_rank(trend_strength)


@register_factor(
    name="short_interest_ratio_change",
    description="融券占比变化因子，融券余量/流通股本 5日变化截面排名（取负向=融券增加排后）。",
    category="margin",
    thesis="融券余量占流通股本的比例是更精确的做空压力指标（比融券卖出量更能反映存量空头）——该比例的上升意味着空头在累积而非平仓，持续上升是强烈的看空信号。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_short_interest_ratio_change(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    finance = context.load("finance.parquet")

    rqyl = margin["rqyl"]  # 融券余量
    float_share = finance["float_share"]

    common = rqyl.index.intersection(float_share.index)
    si_ratio = rqyl.loc[common] / float_share.loc[common].replace(0, np.nan)
    si_chg = si_ratio.groupby(level="Code").transform(lambda s: s.diff(5))

    return cross_sectional_rank(-si_chg)


@register_factor(
    name="margin_sentiment_divergence",
    description="融资情绪背离因子，融资买入5日变化率-价格5日变化率截面排名（融资比价格更积极=先行指标排前）。",
    category="margin",
    thesis="融资情绪相对价格走势的领先/滞后关系具有择时价值——融资买入先于价格上涨，说明杠杆资金在提前布局；价格上涨后才跟进融资，则是追涨行为。融资先行的股票更具alpha特征。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_margin_sentiment_divergence(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    buy = margin["rzmre"]
    close = daily_adj["close"]

    buy_chg = buy.groupby(level="Code").transform(lambda s: s.pct_change(5))
    price_chg = close.groupby(level="Code").transform(lambda s: s.pct_change(5))

    common = buy_chg.index.intersection(price_chg.index)
    sentiment_lead = buy_chg.loc[common] - price_chg.loc[common]

    return cross_sectional_rank(sentiment_lead)


@register_factor(
    name="margin_flow_acceleration",
    description="融资流加速度因子，5日净流变化-20日净流变化/4截面排名（流入加速排前）。",
    category="margin",
    thesis="融资流的加速度（二阶导数）比一阶导（净流本身）更灵敏——净流从负转正或从小正转为大正的过程是杠杆资金态度转变的关键窗口。加速度是融资信号的'领先指标'。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_flow_acceleration(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    net = _net_margin_flow(margin)

    net_5d = rolling_group_sum(net, 5)
    net_20d = rolling_group_sum(net, 20)

    # Acceleration = shorter-term change - longer-term average change
    accel = net_5d.diff(5) - net_20d.diff(20) / 4.0

    return cross_sectional_rank(accel)


@register_factor(
    name="dual_margin_signal",
    description="融资融券双信号因子，(融资净流入排名-融券卖出排名)截面排名（融资强+融券弱=一致看多排前）。",
    category="margin",
    thesis="融资和融券从两个方向反映市场态度——融资净流入（看多）+融券卖出减少（空头退缩）是双重确认的看多信号。两者方向一致时信号最强，方向矛盾时需要警惕市场分歧。",
    dependencies=("margin_detail.parquet",),
)
def factor_dual_margin_signal(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    net = rolling_group_sum(_net_margin_flow(margin), 5)
    short = rolling_group_mean(margin["rqmcl"], 5)

    net_rank = net.groupby(level="Date").rank(pct=True)
    short_rank = short.groupby(level="Date").rank(pct=True)

    # Bullish when net rank high and short rank low
    dual = net_rank - short_rank
    return cross_sectional_rank(dual)


@register_factor(
    name="margin_concentration_hhi",
    description="融资集中度HHI因子，各股融资余额在总融资中的占比平方和截面排名（集中度高=系统风险大排后）。",
    category="margin",
    thesis="融资资金的集中度是市场杠杆风险的系统性指标——融资高度集中在少数股票中意味着去杠杆风险也集中。这些股票一旦下跌，连锁平仓效应会放大。对融资集中度高的股票需要给予流动性风险折扣。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_concentration_hhi(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    rzye = margin["rzye"]

    total = rzye.groupby(level="Date").sum()
    dates = rzye.index.get_level_values("Date")
    total_mapped = pd.Series(total.loc[dates].values, index=rzye.index)

    share = rzye / total_mapped.replace(0, np.nan)
    hhi = share ** 2  # Individual contribution to HHI

    return cross_sectional_rank(hhi)


@register_factor(
    name="margin_smart_money_proxy",
    description="融资聪明钱因子，5日融资买入/成交额比的变化率截面排名（融资占比上升=杠杆资金信心增强排前）。",
    category="margin",
    thesis="融资买入占比的边际变化可能包含'聪明钱'信息——在市场下跌时仍然逆势增加融资买入占比的资金，通常对基本面有更深入的研究和更强的信心，是对冲基金和游资的行为特征。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_margin_smart_money_proxy(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    buy = margin["rzmre"]
    amount = daily_adj["amount"]

    common = buy.index.intersection(amount.index)
    buy_ratio = buy.loc[common] / amount.loc[common].replace(0, np.nan)
    ratio_chg = buy_ratio.groupby(level="Code").transform(lambda s: s.diff(5))

    return cross_sectional_rank(ratio_chg)
