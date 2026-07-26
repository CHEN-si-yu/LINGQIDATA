"""
Factor implementations based on previously unused upstream data fields.

This module fills gaps identified in the comprehensive field audit (2026-07-02).
All factors use fields that were NOT referenced in any existing factor code,
yet have excellent data quality (>95% coverage, <5% NaN for target stock pool).

Data coverage baseline: 1,782 stocks in Code_num.txt, 2020-01-01 to 2026-07-01.

Sections
--------
A.  Financial Indicator (financial_indicator.parquet) — unused fields
B.  Balance Sheet (balancesheet.parquet) — unused fields
C.  Income Statement (income.parquet) — unused fields
D.  Cash Flow (cashflow.parquet) — unused fields
E.  Main Fund Flow Volume (main_fund_flow.parquet) — volume-based ratios
F.  Finance Panel (finance.parquet) — unused field
G.  Comprehensive / Composite factors from unused fields
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

# ═══════════════════════════════════════════════════════════════════════════
# A.  Financial Indicator — previously unused fields
#    Source: financial_indicator.parquet (163 columns, 85 unused)
#    Coverage: 89,090 rows, 1782/1782 stocks, 2020-03-31 to 2026-03-31
# ═══════════════════════════════════════════════════════════════════════════

# ── A.1  Free Cash Flow Quality ────────────────────────────────────────────

def _total_vol(ff):
    """Return total volume from main fund flow, zero replaced with NaN."""
    return (
        ff["buy_sm_vol"] + ff["sell_sm_vol"]
        + ff["buy_md_vol"] + ff["sell_md_vol"]
        + ff["buy_lg_vol"] + ff["sell_lg_vol"]
        + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    ).replace(0, np.nan)


@register_factor(
    name="mf_big_order_vol_ratio",
    description="大单+特大单成交量占比因子，基于成交量口径的机构行为度量。",
    category="fund_flow",
    thesis="成交量口径的大单占比与成交额口径互补：成交量剔除了高价股偏差，"
    "更能反映真实的交易笔数集中度。两个口径的信号差异本身也是信息。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_big_order_vol_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    big_vol = (
        ff["buy_lg_vol"] - ff["sell_lg_vol"]
        + ff["buy_elg_vol"] - ff["sell_elg_vol"]
    )
    ratio = big_vol / _total_vol(ff)
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_amount_vol_divergence",
    description="资金流量价背离因子，（大单净买入额占比 - 大单净买入量占比）截面排名。",
    category="fund_flow",
    thesis="成交额占比与成交量占比的差异反映了不同价位上的交易分布。"
    "大单净买入额占比 > 大单净买入量占比意味着大资金偏好高价买入（更强的买入意愿）。"
    "额量背离是聪明钱强度的重要信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_amount_vol_divergence(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    # Big order net amount ratio
    big_net_amt = (
        ff["buy_lg_amount"] - ff["sell_lg_amount"]
        + ff["buy_elg_amount"] - ff["sell_elg_amount"]
    )
    total_amt = (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)
    amt_ratio = big_net_amt / total_amt

    # Big order net volume ratio
    big_net_vol = (
        ff["buy_lg_vol"] - ff["sell_lg_vol"]
        + ff["buy_elg_vol"] - ff["sell_elg_vol"]
    )
    vol_ratio = big_net_vol / _total_vol(ff)

    divergence = amt_ratio - vol_ratio
    return cross_sectional_rank(divergence)


@register_factor(
    name="mf_net_vol_intensity",
    description="主力净流入量/总成交量因子，成交量口径的资金净流向强度。",
    category="fund_flow",
    thesis="净流入量占比为正意味着买方在数量上占优，配合净流入额占比使用"
    "可以识别量价配合vs量价背离的信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_vol_intensity(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    ratio = ff["net_mf_vol"] / _total_vol(ff)
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_order_size_entropy",
    description="订单规模分布的熵因子（高值排前），度量资金参与结构的多样性。",
    category="fund_flow",
    thesis="当各规模订单（小/中/大/特大）的参与比例较为均匀时，市场参与结构健康。"
    "当某一类订单占比畸高时（例如全靠大单拉动），行情的持续性存疑。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_order_size_entropy(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_vol = _total_vol(ff)

    # Volume share per order size category
    sm_share = (ff["buy_sm_vol"] + ff["sell_sm_vol"]) / total_vol
    md_share = (ff["buy_md_vol"] + ff["sell_md_vol"]) / total_vol
    lg_share = (ff["buy_lg_vol"] + ff["sell_lg_vol"]) / total_vol
    elg_share = (ff["buy_elg_vol"] + ff["sell_elg_vol"]) / total_vol

    # Entropy: -sum(p * ln(p)), higher = more diverse participation
    eps = 1e-10
    entropy = -(
        sm_share * np.log(sm_share + eps)
        + md_share * np.log(md_share + eps)
        + lg_share * np.log(lg_share + eps)
        + elg_share * np.log(elg_share + eps)
    )
    return cross_sectional_rank(entropy)


# ═══════════════════════════════════════════════════════════════════════════
# F.  Finance Panel — ps (Price-to-Sales unadjusted)
#     Source: finance.parquet
#     Only 1 unused field: ps (vs ps_ttm which is already used)
#     ps is the unadjusted trailing P/S using reported revenue
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="sp_raw",
    description="未调整市销率因子（低值排前），ps = 总市值/最近报告期营收。",
    category="valuation",
    thesis="ps字段提供了一种不同于ps_ttm的市销率计算口径（使用最近一期报告而非TTM），"
    "两者之间的差异本身包含了信息（TTM调整的方向和幅度）。",
    dependencies=("finance.parquet",),
)
def factor_sp_raw(context: FactorContext):
    finance = context.load("finance.parquet")
    return cross_sectional_rank(-finance["ps"])


# ═══════════════════════════════════════════════════════════════════════════
# G.  Composite Factors from Unused Fields
# ═══════════════════════════════════════════════════════════════════════════

