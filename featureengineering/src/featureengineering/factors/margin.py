from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── Margin leverage ratio ─────────────────────────────────────────────────

@register_factor(
    name="margin_leverage_ratio",
    description="融资余额/流通市值因子，融资盘相对规模截面排名（高杠杆排后）。",
    category="margin",
    thesis="融资余额占流通市值比例反映杠杆资金对个股的参与深度，过高杠杆意味着潜在的多杀多踩踏风险。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_margin_leverage_ratio(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    finance = context.load("finance.parquet")

    rzye = margin["rzye"]
    circ_mv = finance["circ_mv"]

    common = rzye.index.intersection(circ_mv.index)
    ratio = rzye.loc[common] / circ_mv.loc[common].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Margin buy intensity 5d ───────────────────────────────────────────────

@register_factor(
    name="margin_buy_intensity_5d",
    description="5日融资买入强度因子，5日累计融资买入额/5日总成交额截面排名。",
    category="margin",
    thesis="融资买入强度反映杠杆资金的持续参与热情，高强度买入区间往往伴随趋势行情。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_margin_buy_intensity_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    daily_adj = context.load("daily_adj.parquet")

    rzmre = margin["rzmre"]
    amount = daily_adj["amount"]

    common = rzmre.index.intersection(amount.index)

    cum_buy = rzmre.loc[common].groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    cum_amount = amount.loc[common].groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    intensity = cum_buy / cum_amount.replace(0, np.nan)
    return cross_sectional_rank(intensity)


# ── Margin balance growth 5d ─────────────────────────────────────────────

@register_factor(
    name="margin_balance_growth_5d",
    description="融资余额5日增长率因子，rzye的5日增长率截面排名（融资余额增长=杠杆资金看多排前）。",
    category="fund_flow",
    thesis="融资余额增长代表杠杆资金持续看好后市——融资余额是杠杆资金的存量指标，余额增长意味着更多资金愿意加杠杆买入。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_balance_growth_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    balance = margin["rzye"]
    growth = balance.groupby(level="Code").transform(lambda s: s.pct_change(5))
    return cross_sectional_rank(growth)


# ── Margin repay acceleration ───────────────────────────────────────────

@register_factor(
    name="margin_repay_acceleration",
    description="融资偿还加速度因子，-(rzche的5日增长率)截面排名（偿还加速=去杠杆排后）。",
    category="fund_flow",
    thesis="融资偿还额的加速增长意味着杠杆资金在加速撤离——偿还速度超过买入速度时多头杠杆率下降，市场承压。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_repay_acceleration(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    repay = margin["rzche"]
    growth = repay.groupby(level="Code").transform(lambda s: s.pct_change(5))
    return cross_sectional_rank(-growth)


# ── Margin balance to float mv ──────────────────────────────────────────

@register_factor(
    name="margin_balance_to_float_mv",
    description="融资余额/流通市值截面排名。",
    category="fund_flow",
    thesis="融资余额相对流通市值的比例衡量杠杆渗透深度——高比例意味着大量的股价是由借来的钱支撑的，下跌时面临强制平仓的连锁风险。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_margin_balance_to_float_mv(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    finance = context.load("finance.parquet")
    rzye = margin["rzye"]
    circ_mv = finance["circ_mv"]
    common = rzye.index.intersection(circ_mv.index)
    ratio = rzye.loc[common] / circ_mv.loc[common].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Margin buy sell imbalance ───────────────────────────────────────────

@register_factor(
    name="margin_buy_sell_imbalance",
    description="融资买卖失衡因子，(rzmre-rzche)/(rzmre+rzche)的5日均值截面排名。",
    category="fund_flow",
    thesis="融资买入与偿还的5日平均失衡——持续净买入意味着杠杆资金持续看多，持续净偿还意味着杠杆资金在系统性减仓。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_buy_sell_imbalance(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    imbalance = (margin["rzmre"] - margin["rzche"]) / (margin["rzmre"] + margin["rzche"]).replace(0, np.nan)
    avg_5d = imbalance.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    return cross_sectional_rank(avg_5d)


# ── Margin balance stability 20d ────────────────────────────────────────

@register_factor(
    name="margin_balance_stability_20d",
    description="融资余额稳定性因子，-(rzye的20日变异系数)截面排名（余额稳定=杠杆质量高排前）。",
    category="fund_flow",
    thesis="融资余额大幅波动意味着杠杆资金短线进出频繁——稳定增长的融资余额代表机构型杠杆资金的长期持仓。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_balance_stability_20d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    balance = margin["rzye"]
    std_20 = balance.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    mean_20 = balance.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    cv = std_20 / mean_20.replace(0, np.nan)
    return cross_sectional_rank(-cv)

