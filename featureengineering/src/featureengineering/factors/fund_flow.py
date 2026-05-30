from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


def _total_amount(ff):
    """Return total turnover amount from main fund flow, zero replaced with NaN."""
    return (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)


# ── Main Fund Net Inflow ────────────────────────────────────────────────

@register_factor(
    name="mf_net_inflow_ratio",
    description="主力资金净流入率因子，主力净流入额/成交额截面排名。",
    category="fund_flow",
    thesis="主力净流入率是日内聪明钱行为的直接度量，持续净流入预示后续上涨动力。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_inflow_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    ratio = ff["net_mf_amount"] / _total_amount(ff)
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_net_inflow_5d",
    description="5日累计主力净流入率因子截面排名。",
    category="fund_flow",
    thesis="短期累计主力资金行为比单日更具稳定性，过滤噪音。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_inflow_5d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    daily_ratio = ff["net_mf_amount"] / _total_amount(ff)
    cum_ratio = daily_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    return cross_sectional_rank(cum_ratio)


# ── Order-size analysis ─────────────────────────────────────────────────

@register_factor(
    name="mf_big_order_ratio",
    description="大单+特大单净买入率因子，(特大+大净买入)/总成交额截面排名。",
    category="fund_flow",
    thesis="特大单和大单通常代表机构行为，净买入占比高是专业资金看多的信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_big_order_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    big_net = (
        ff["buy_lg_amount"] - ff["sell_lg_amount"]
        + ff["buy_elg_amount"] - ff["sell_elg_amount"]
    )
    ratio = big_net / _total_amount(ff)
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_small_order_ratio",
    description="小单净买入率因子（负值=散户净卖出，排名高=散户流出多）。",
    category="fund_flow",
    thesis="散户净卖出+机构净买入的组合是较强的看多信号，反向使用小单数据更有效。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_small_order_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
    ratio = small_net / _total_amount(ff)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="mf_big_small_divergence",
    description="大小单背离因子，(大单净买-小单净买)/总成交额截面排名。",
    category="fund_flow",
    thesis="大小单背离度越大，说明机构与散户行为分歧越大，分歧顶点常伴随趋势转折。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_big_small_divergence(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    big_net = (
        ff["buy_lg_amount"] - ff["sell_lg_amount"]
        + ff["buy_elg_amount"] - ff["sell_elg_amount"]
    )
    small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
    divergence = (big_net - small_net) / _total_amount(ff)
    return cross_sectional_rank(divergence)


# ── Margin trading ──────────────────────────────────────────────────────

@register_factor(
    name="margin_buy_strength",
    description="融资买入强度因子，融资买入额/成交额截面排名。",
    category="fund_flow",
    thesis="融资买入强度反映杠杆做多意愿，高融资买入意味着投资者对后市乐观。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_buy_strength(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    # margin doesn't have amount, skip this approach
    # Use financing buy vs financing repay
    net_finance = margin["rzmre"] - margin["rzche"]  # buy - repay
    ratio = net_finance / margin["rzye"].replace(0, np.nan)  # relative to balance
    return cross_sectional_rank(ratio)


@register_factor(
    name="margin_balance_change",
    description="融资余额变化率因子，融资余额日变动率截面排名。",
    category="fund_flow",
    thesis="融资余额变化反映杠杆资金对后市的边际看法。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_balance_change(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    rzye = margin["rzye"]
    change = rzye.groupby(level="Code").transform(lambda s: s.pct_change(1))
    return cross_sectional_rank(change)


@register_factor(
    name="margin_short_pressure",
    description="融券压力因子，融券余额/两融总余额截面排名（高比例排后=看空压力）。",
    category="fund_flow",
    thesis="融券余额占比反映做空力量强度，高融券占比对股价构成压力。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_short_pressure(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    short_ratio = margin["rqye"] / margin["rzrqye"].replace(0, np.nan)
    return cross_sectional_rank(-short_ratio)


# ── Fund flow trend ───────────────────────────────────────────────────────

@register_factor(
    name="mf_net_inflow_trend_5d",
    description="5日主力净流入率趋势因子，近5日净流入率线性回归斜率截面排名。",
    category="fund_flow",
    thesis="主力资金流的趋势方向比单日流向量更具信息量，持续流入斜率反映资金态度的一致性强弱。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_inflow_trend_5d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    daily_ratio = ff["net_mf_amount"] / _total_amount(ff)

    def _trend_slope(y):
        y = y[~np.isnan(y)]
        if len(y) < 3:
            return np.nan
        x = np.arange(len(y), dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = (x * x).sum()
        if denom == 0:
            return np.nan
        return (x * y).sum() / denom

    slope = daily_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).apply(_trend_slope, raw=True)
    )
    return cross_sectional_rank(slope)


# ── Big order divergence ─────────────────────────────────────────────────

@register_factor(
    name="big_order_divergence",
    description="大单净流入与涨跌幅背离因子，rank(大单净买入率)-rank(pct_chg)截面排名。",
    category="fund_flow",
    thesis="大单净流入与价格涨跌的背离反映聪明钱与价格行为的分歧，背离度越大预示未来价格修正越强。",
    dependencies=("main_fund_flow.parquet", "daily_adj.parquet"),
)
def factor_big_order_divergence(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    daily_adj = context.load("daily_adj.parquet")

    big_net = (
        ff["buy_lg_amount"] - ff["sell_lg_amount"]
        + ff["buy_elg_amount"] - ff["sell_elg_amount"]
    )
    big_net_rate = big_net / _total_amount(ff)

    with np.errstate(invalid="ignore"):
        rank_big = big_net_rate.groupby(level="Date").rank(pct=True)
        rank_pct = daily_adj["pct_chg"].groupby(level="Date").rank(pct=True)

    # Align on common index
    common = rank_big.index.intersection(rank_pct.index)
    divergence = rank_big.loc[common] - rank_pct.loc[common]
    return cross_sectional_rank(divergence)


# ── Margin net open interest ─────────────────────────────────────────────

@register_factor(
    name="margin_net_open",
    description="融资净开仓强度因子，(融资买入额-融资偿还额)/融资余额截面排名。",
    category="fund_flow",
    thesis="融资净开仓是杠杆资金日内净流向的度量——净买入>净偿还=资金净流入。该比率标准化后可跨股票比较杠杆资金的边际参与意愿。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_net_open(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    net_open = (margin["rzmre"] - margin["rzche"]) / margin["rzye"].replace(0, np.nan)
    return cross_sectional_rank(net_open)


# ── Short selling intensity ──────────────────────────────────────────────

@register_factor(
    name="short_sell_intensity",
    description="融券卖出强度因子，(融券卖出量-融券偿还量)/融券余量截面排名（高=做空增加，排后）。",
    category="fund_flow",
    thesis="融券净卖出代表空头力量的边际变化——净卖出增加意味着更多投资者在借券做空，是负面信号。与margin_short_pressure互补：一个看空头余额占比（存量），一个看空头行为变化（流量）。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_short_sell_intensity(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")
    rqmcl = margin["rqmcl"]
    vol = daily_adj["vol"]
    common = rqmcl.index.intersection(vol.index)
    intensity = rqmcl.loc[common] / vol.loc[common].replace(0, np.nan)
    return cross_sectional_rank(-intensity)


# ── Volume-based fund flow ─────────────────────────────────────────────


@register_factor(
    name="net_mf_amount_intensity",
    description="成交量基础主力净流入因子，(主力净流入量/总成交量)截面排名。",
    category="fund_flow",
    thesis="所有现有主力资金流因子均基于成交金额，但金额受股价高低影响大——高价股在金额排名中天然占优。成交量基础的主力净流入率从'股数'维度衡量主力行为，消除价格偏差后更公平地跨股票比较主力参与度。与mf_net_inflow_ratio互补：一个看金额权重，一个看量权重。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_net_mf_amount_intensity(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    ratio = ff["net_mf_amount"] / _total_amount(ff)
    return cross_sectional_rank(ratio)


# ── ELG (extra-large order) ratio ────────────────────────────────────────────

@register_factor(
    name="mf_elg_order_ratio",
    description="特大单净买入率因子，(特大单净买入额/总成交额)截面排名。",
    category="fund_flow",
    thesis="特大单代表机构大额交易，净买入占比高反映专业性资金对后市的明确看多态度。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_elg_order_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    elg_net = ff["buy_elg_amount"] - ff["sell_elg_amount"]
    ratio = elg_net / _total_amount(ff)
    return cross_sectional_rank(ratio)


# ── Medium order ratio ──────────────────────────────────────────────────────

@register_factor(
    name="mf_mid_order_ratio",
    description="中单净买入率因子，(中单净买入额/总成交额)截面排名。",
    category="fund_flow",
    thesis="中单代表专业但非机构级别的交易者（如大户/游资），其净流向反映中等规模资金的短期态度。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_mid_order_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    md_net = ff["buy_md_amount"] - ff["sell_md_amount"]
    ratio = md_net / _total_amount(ff)
    return cross_sectional_rank(ratio)


# ── Fund flow continuity ───────────────────────────────────────────────────

@register_factor(
    name="mf_flow_continuity",
    description="主力资金连续流入天数因子，统计各股票连续净流入天数截面排名。",
    category="fund_flow",
    thesis="连续净流入天数越长，说明主力资金对该股的持续看好态度越坚定，短期股价支撑越强。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_flow_continuity(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    positive = ff["net_mf_amount"] > 0

    def _streak_series(s):
        """Compute consecutive True streak for a single stock."""
        groups = (s != s.shift(1)).cumsum()
        streak = s.groupby(groups).cumcount() + 1
        return streak.where(s, 0)

    streak = positive.groupby(level="Code").transform(_streak_series)
    return cross_sectional_rank(streak)


# ── Net inflow volatility ──────────────────────────────────────────────────

@register_factor(
    name="mf_net_inflow_volatility_20d",
    description="20日主力净流入率波动率因子，(负向排名)净流入率波动越大排名越低。",
    category="fund_flow",
    thesis="主力资金流向波动率高说明资金态度分歧大、缺乏一致方向，高波动区间后市走势不确定性增加，应给予低评分。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_inflow_volatility_20d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    daily_ratio = ff["net_mf_amount"] / _total_amount(ff)
    vol_20d = daily_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-vol_20d)


# ── ELG vs small divergence ────────────────────────────────────────────────

@register_factor(
    name="mf_elg_small_divergence",
    description="特大单与小单背离因子，(特大单净买入-小单净买入)/总成交额截面排名。",
    category="fund_flow",
    thesis="特大单（机构）与小单（散户）行为背离度越大，说明精英资金与散户分歧越严重，背离极端值往往预示趋势转折。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_elg_small_divergence(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    elg_net = ff["buy_elg_amount"] - ff["sell_elg_amount"]
    small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
    divergence = (elg_net - small_net) / _total_amount(ff)
    return cross_sectional_rank(divergence)


# ── Big order turnover ratio ───────────────────────────────────────────────

@register_factor(
    name="mf_big_order_turnover_ratio",
    description="大额订单成交占比因子，(大单+特大单成交量)/总成交量截面排名。",
    category="fund_flow",
    thesis="大额订单成交量占比高说明市场由机构主导，机构参与度高的股票信息传递效率更高，定价更有效。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_big_order_turnover_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    big_vol = (
        ff["buy_lg_amount"] + ff["buy_elg_amount"]
        + ff["sell_lg_amount"] + ff["sell_elg_amount"]
    )
    total_vol = (
        big_vol
        + ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
    ).replace(0, np.nan)
    ratio = big_vol / total_vol
    return cross_sectional_rank(ratio)


# ── Amount-weighted direction composite ─────────────────────────────────────

@register_factor(
    name="mf_amount_weighted_direction",
    description="金额加权方向复合因子，四档订单方向信号按金额占比加权求和截面排名。",
    category="fund_flow",
    thesis="不同规模订单的资金方向信号强度不同，按金额占比加权综合各档位方向信号，比单一净流入更全面反映市场多空结构。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_amount_weighted_direction(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_amt = _total_amount(ff)

    # Direction sign for each tier: +1 if buy > sell, -1 if sell > buy
    sm_dir = np.sign(ff["buy_sm_amount"] - ff["sell_sm_amount"])
    md_dir = np.sign(ff["buy_md_amount"] - ff["sell_md_amount"])
    lg_dir = np.sign(ff["buy_lg_amount"] - ff["sell_lg_amount"])
    elg_dir = np.sign(ff["buy_elg_amount"] - ff["sell_elg_amount"])

    # Amount proportion weight for each tier
    sm_w = (ff["buy_sm_amount"] + ff["sell_sm_amount"]) / total_amt
    md_w = (ff["buy_md_amount"] + ff["sell_md_amount"]) / total_amt
    lg_w = (ff["buy_lg_amount"] + ff["sell_lg_amount"]) / total_amt
    elg_w = (ff["buy_elg_amount"] + ff["sell_elg_amount"]) / total_amt

    composite = sm_dir * sm_w + md_dir * md_w + lg_dir * lg_w + elg_dir * elg_w
    return cross_sectional_rank(composite)


# ── Supplementary fund flow factors ────────────────────────────────────────


@register_factor(
    name="mf_cumulative_flow_20d",
    description="20日累计主力净流入率因子。",
    category="fund_flow",
    thesis="累计净流入/流出反映中期资金态度，持续的净流入比单日信号更可靠",
    dependencies=("main_fund_flow.parquet", "calendar.parquet"),
)
def factor_mf_cumulative_flow_20d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_amt = _total_amount(ff)
    net_amt = ff["net_mf_amount"]
    net_rate = safe_divide(net_amt, total_amt)
    cum_rate = net_rate.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    return cross_sectional_rank(cum_rate)


@register_factor(
    name="mf_flow_reversal_20d",
    description="20日主力资金反转因子 (从流出的流出反转为流入)。",
    category="fund_flow",
    thesis="主力从净流出转为净流入是重要的拐点信号，捕捉资金态度的边际变化",
    dependencies=("main_fund_flow.parquet", "calendar.parquet"),
)
def factor_mf_flow_reversal_20d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_amt = _total_amount(ff)
    net_rate = safe_divide(ff["net_mf_amount"], total_amt)
    cum_10 = rolling_group_mean(net_rate, 10)
    cum_20 = rolling_group_mean(net_rate, 20)
    # Reversal: recent 10d positive while longer 20d negative
    reversal = cum_10 - cum_20
    return cross_sectional_rank(reversal)


@register_factor(
    name="mf_big_order_stability_20d",
    description="20日大单净买入率稳定性因子 (高稳定排前)。",
    category="fund_flow",
    thesis="大单行为一致性反映机构意图明确，频繁方向切换意味着不确定性和噪音交易",
    dependencies=("main_fund_flow.parquet", "calendar.parquet"),
)
def factor_mf_big_order_stability_20d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_amt = _total_amount(ff)
    big_net = ff["buy_lg_amount"] + ff["buy_elg_amount"] - ff["sell_lg_amount"] - ff["sell_elg_amount"]
    big_rate = safe_divide(big_net, total_amt)
    rate_std = rolling_group_std(big_rate, 20)
    rate_mean = rolling_group_mean(big_rate, 20)
    stability = safe_divide(rate_mean.abs() + 1e-8, rate_std + 1e-8)
    return cross_sectional_rank(stability)


@register_factor(
    name="mf_big_small_convergence_20d",
    description="20日大单/小单收敛因子 (大单趋势-小单趋势)。",
    category="fund_flow",
    thesis="大单趋势与小单趋势的背离收敛包含信息——大单领先小单转向是机构先行的信号",
    dependencies=("main_fund_flow.parquet", "calendar.parquet"),
)
def factor_mf_big_small_convergence_20d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_amt = _total_amount(ff)
    big_net = ff["buy_lg_amount"] + ff["buy_elg_amount"] - ff["sell_lg_amount"] - ff["sell_elg_amount"]
    small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
    big_rate = safe_divide(big_net, total_amt)
    small_rate = safe_divide(small_net, total_amt)
    big_trend = rolling_group_mean(big_rate, 5)
    small_trend = rolling_group_mean(small_rate, 5)
    convergence = big_trend - small_trend
    return cross_sectional_rank(convergence)


@register_factor(
    name="mf_open_close_divergence_10d",
    description="10日开盘/收盘资金流向背离因子。",
    category="fund_flow",
    thesis="开盘和收盘阶段的资金行为反映不同类型投资者：开盘=跟随/散户，尾盘=机构调仓。两者背离有信号意义",
    dependencies=("main_fund_flow.parquet", "calendar.parquet"),
)
def factor_mf_open_close_divergence_10d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_amt = _total_amount(ff)
    net_rate = safe_divide(ff["net_mf_amount"], total_amt)
    # Proxy: difference between daily net rate and its 5d trend
    trend_5 = rolling_group_mean(net_rate, 5)
    divergence = safe_divide(net_rate - trend_5, trend_5.abs() + 1e-8)
    div_std = rolling_group_std(divergence, 10)
    return cross_sectional_rank(div_std)


@register_factor(
    name="margin_net_buy_pressure_10d",
    description="10日融资净买入压力因子 (融资买入增加排前)。",
    category="fund_flow",
    thesis="融资净买入加速是杠杆资金看多的信号，融资余额趋势性增加预示短期强势",
    dependencies=("margin_detail.parquet", "calendar.parquet"),
)
def factor_margin_net_buy_pressure_10d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    rzmre = margin["rzmre"]  # margin buy amount
    rzche = margin["rzche"]  # margin repayment
    net_buy = rzmre - rzche
    net_pressure = net_buy.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).sum()
    )
    return cross_sectional_rank(net_pressure)


@register_factor(
    name="short_sell_change_5d",
    description="5日融券卖出量变化因子 (增加排后, 负向)。",
    category="fund_flow",
    thesis="融券卖出增加意味着空头力量增强，是负向信号",
    dependencies=("margin_detail.parquet", "calendar.parquet"),
)
def factor_short_sell_change_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    rqmcl = margin["rqmcl"]
    ma_5 = rqmcl.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    ma_20 = rqmcl.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    change = safe_divide(ma_5 - ma_20, ma_20 + 1e-8)
    return cross_sectional_rank(-change)


# ── Order size divergence depth ───────────────────────────────────────────

@register_factor(
    name="big_vs_small_divergence_5d",
    description="大小单背离5日因子，(大单净流入率-小单净流入率)的5日变化截面排名。",
    category="fund_flow",
    thesis="大单资金与小单资金方向的背离变化是聪明的信号——大单资金相对小单的流入加速意味着机构/大户正在加速建仓，而散户可能还在犹豫或减持。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_big_vs_small_divergence_5d(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    big_net = (mf["buy_lg_amount"] + mf["buy_elg_amount"] - mf["sell_lg_amount"] - mf["sell_elg_amount"]) / _total_amount(mf)
    small_net = (mf["buy_sm_amount"] - mf["sell_sm_amount"]) / _total_amount(mf)
    divergence = big_net - small_net
    div_5d = divergence.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(div_5d)


@register_factor(
    name="medium_order_flow",
    description="中单资金流因子，中单净买入/总成交量截面排名。",
    category="fund_flow",
    thesis="中单资金流往往被忽视——中单代表中等资金量级的投资者行为，在大单和小单之间提供了额外信息维度。中单净流入可能是机构隐藏建仓意图的手段。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_medium_order_flow(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    md_net = (mf["buy_md_amount"] - mf["sell_md_amount"]) / _total_amount(mf)
    return cross_sectional_rank(md_net)


@register_factor(
    name="super_large_order_intensity",
    description="超大单强度因子，超大单净买入/总成交量截面排名。",
    category="fund_flow",
    thesis="超大单(每单>500万)是机构定制化交易和主力大额博弈的直接体现——超大单净流入持续为正意味着主力在持续收集筹码。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_super_large_order_intensity(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    elg_net = (mf["buy_elg_amount"] - mf["sell_elg_amount"]) / _total_amount(mf)
    return cross_sectional_rank(elg_net)


@register_factor(
    name="small_order_crowding",
    description="小单拥挤度因子，-(小单买入量/总成交量)截面排名（高小单占比=散户追涨排后）。",
    category="fund_flow",
    thesis="小单成交量占比过高意味着散户主导交易——散户追涨是经典的短期见顶信号，机构通常在小单占比极端时反向操作。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_small_order_crowding(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    small_pct = (mf["buy_sm_amount"] + mf["sell_sm_amount"]) / _total_amount(mf)
    return cross_sectional_rank(-small_pct)


@register_factor(
    name="net_mf_flow_persistence",
    description="主力资金净流入持续性因子，近5日净流入为正的天数截面排名。",
    category="fund_flow",
    thesis="主力资金持续净流入比单日净流入更有意义——持续流入代表机构系统性建仓而非一日游。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_net_mf_flow_persistence(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    net_pos = (mf["net_mf_amount"] > 0).astype(float)
    persistence = net_pos.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    return cross_sectional_rank(persistence)


@register_factor(
    name="mf_net_amount_intensity",
    description="主力资金净额强度因子，net_mf_amount/流通市值截面排名。",
    category="fund_flow",
    thesis="主力资金净买入相对流通市值的比例——消除规模效应后，小市值股票的主力建仓信号更灵敏。",
    dependencies=("main_fund_flow.parquet", "finance.parquet"),
)
def factor_mf_net_amount_intensity(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    finance = context.load("finance.parquet")
    net_amount = mf["net_mf_amount"]
    circ_mv = finance["circ_mv"]
    common = net_amount.index.intersection(circ_mv.index)
    intensity = net_amount.loc[common] / circ_mv.loc[common].replace(0, np.nan)
    return cross_sectional_rank(intensity)


@register_factor(
    name="order_concentration",
    description="订单集中度因子，-(中单+小单)/总成交截面排名（大单+超大单占比高=机构主导排前）。",
    category="fund_flow",
    thesis="成交量的订单结构反映市场参与者的构成——大单占比越高意味着机构/大户参与度越高，信息的alpha质量越高。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_order_concentration(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    small_mid = (mf["buy_sm_amount"] + mf["sell_sm_amount"] +
                  mf["buy_md_amount"] + mf["sell_md_amount"])
    total = _total_amount(mf)
    concentration = small_mid / total.replace(0, np.nan)
    return cross_sectional_rank(-concentration)


@register_factor(
    name="net_mf_amount_momentum_5d",
    description="主力资金净额5日动量因子，net_mf_amount的5日变化截面排名。",
    category="fund_flow",
    thesis="主力资金净买入额的边际变化——净买入正在加速(无论正负)意味着主力行为在改变，是趋势转变的早期信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_net_mf_amount_momentum_5d(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    net_amt = mf["net_mf_amount"]
    mom = net_amt.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(mom)


@register_factor(
    name="fund_flow_volatility_20",
    description="资金流波动性因子，-(net_mf_amount/vol的20日标准差)截面排名（资金流不稳定排后）。",
    category="fund_flow",
    thesis="主力资金净流向频繁变换方向意味着多空分歧大、主力意图不明确——资金流方向稳定是趋势确定性高的信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_fund_flow_volatility_20(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    net_ratio = mf["net_mf_amount"] / _total_amount(mf)
    vol_20 = net_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-vol_20)


@register_factor(
    name="large_order_timing_signal",
    description="大单时机信号因子，大单净买入/成交量×20日价格位置截面排名（低位大单流入=最佳买点排前）。",
    category="fund_flow",
    thesis="大单流入配合价格在低位是最优的信号组合——机构在低位大额建仓意味着他们对当前价格水平认可，且预期未来上涨空间大。",
    dependencies=("main_fund_flow.parquet", "daily_adj.parquet"),
)
def factor_large_order_timing_signal(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    daily_adj = context.load("daily_adj.parquet")
    big_net = (mf["buy_lg_amount"] + mf["buy_elg_amount"] - mf["sell_lg_amount"] - mf["sell_elg_amount"]) / _total_amount(mf)
    close = daily_adj["close"]
    high_20 = close.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max())
    low_20 = close.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).min())
    position = (close - low_20) / (high_20 - low_20).replace(0, np.nan)
    # Low position (close to low) + positive big net = strong buy signal
    signal = big_net * (1 - position)
    return cross_sectional_rank(signal)
