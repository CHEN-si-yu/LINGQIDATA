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


