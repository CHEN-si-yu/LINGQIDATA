from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


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
