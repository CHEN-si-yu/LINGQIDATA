"""
Fund flow volume-based extended factors — Class 1 panel factors.

Exploits the heavily underutilized _vol (volume) fields from main_fund_flow.parquet.
The 8 _vol fields (buy/sell × sm/md/lg/elg) are only used in 3 files vs 7 for _amount.

Key innovation: vol/amount ratio = average price per trade for each order size.
Divergence between vol-based and amount-based flow signals = behavior alpha.

Uses:
  - buy_sm_vol, sell_sm_vol, buy_md_vol, sell_md_vol
  - buy_lg_vol, sell_lg_vol, buy_elg_vol, sell_elg_vol
  - net_mf_vol
  - Corresponding _amount fields for ratio computation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, rolling_group_mean


def _total_vol(ff: pd.DataFrame) -> pd.Series:
    """Total buy+sell volume across all order sizes."""
    cols = [
        "buy_sm_vol", "sell_sm_vol", "buy_md_vol", "sell_md_vol",
        "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol",
    ]
    total = ff[cols[0]].copy()
    for c in cols[1:]:
        total = total + ff[c]
    return total.replace(0, np.nan)


def _total_amount(ff: pd.DataFrame) -> pd.Series:
    """Total buy+sell amount across all order sizes."""
    cols = [
        "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
        "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
    ]
    total = ff[cols[0]].copy()
    for c in cols[1:]:
        total = total + ff[c]
    return total.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Vol/Amount Ratio — Average Trade Price by Order Size
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_large_order_avg_price",
    description="大单+超大单成交均价相对总成交均价。高比值=机构高价成交(拉升建仓),排名高。",
    category="fund_flow",
    thesis=(
        "大单和超大单的成交均价(amount/vol)相对总成交均价的比值。"
        "大单均价高于总均价=机构在较高价位积极买入(拉升式建仓)——看多信号；"
        "大单均价低于总均价=机构在压低价格吸筹或高位出货——需结合方向判断。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_large_order_avg_price(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    large_vol = ff["buy_lg_vol"] + ff["sell_lg_vol"] + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    large_amt = ff["buy_lg_amount"] + ff["sell_lg_amount"] + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    total_vol = _total_vol(ff)
    total_amt = _total_amount(ff)
    large_avg = safe_divide(large_amt, large_vol)
    total_avg = safe_divide(total_amt, total_vol)
    ratio = safe_divide(large_avg, total_avg)
    ratio = ratio.clip(0.5, 2.0)
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_vol_amount_divergence",
    description="资金流量的量-额背离：净买入量占比-净买入额占比。正=量大但额小(低价成交),负=量小但额大(高价成交)。",
    category="fund_flow",
    thesis=(
        "净买入的成交量占比与成交额占比的差值。正值=成交量净买入多于成交额净买入"
        "(低价位大量成交,散户主导,机构可能在压价吸筹)；"
        "负值=成交额净买入多于成交量净买入(高价位少量成交,机构拉升中买入)。"
        "量-额背离揭示了资金行为的质量差异——同样的净流入,高价vs低价含义完全不同。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_vol_amount_divergence(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    total_vol = _total_vol(ff)
    total_amt = _total_amount(ff)
    net_vol_ratio = safe_divide(ff["net_mf_vol"], total_vol)
    net_amt_ratio = safe_divide(ff["net_mf_amount"], total_amt)
    divergence = net_vol_ratio - net_amt_ratio
    divergence = divergence.clip(-0.5, 0.5)
    return cross_sectional_rank(divergence)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Volume Concentration by Order Size
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_vol_concentration_large",
    description="大单+超大单成交量占比。高占比=机构交易量集中,排名高。",
    category="fund_flow",
    thesis=(
        "大单和超大单成交量占总成交量的比例。高比例意味着机构和主力资金主导了当日交易量——"
        "这是专业资金活跃度的直接证据。与金额集中度配合使用,量+额双高=最可靠的机构信号。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_vol_concentration_large(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    large_vol = ff["buy_lg_vol"] + ff["sell_lg_vol"] + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    ratio = safe_divide(large_vol, _total_vol(ff))
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_vol_retail_ratio",
    description="小单成交量占比取反。小单量占比高=散户活跃,噪音大,排名低。",
    category="fund_flow",
    thesis=(
        "小单成交量占总成交量的比例取反排名。小单量占比高意味着散户交易活跃——"
        "噪音交易者主导、价格发现效率低。低小单量占比意味着机构主导——价格信号更有效。"
        "这是vol维度对散户参与度的度量,与amount维度的mf_retail_dominance互补。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_vol_retail_ratio(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    retail_vol = ff["buy_sm_vol"] + ff["sell_sm_vol"]
    ratio = safe_divide(retail_vol, _total_vol(ff))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Net Flow Volume Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_net_vol_ratio_5d",
    description="净买入成交量占比5日变化。量能角度净流入的边际改善,排名高。",
    category="fund_flow",
    thesis=(
        "净买入成交量占比(net_mf_vol/total_vol)的5日均值。"
        "成交量维度的净流入捕捉了交易量的方向性——净流入量持续增加="
        "更多成交量在买方,即使金额净流入不大,量能的方向性也很重要。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_vol_ratio_5d(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    net_vol_ratio = safe_divide(ff["net_mf_vol"], _total_vol(ff))
    ratio_ma5 = net_vol_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    ratio_ma5 = ratio_ma5.clip(-0.5, 0.5)
    return cross_sectional_rank(ratio_ma5)


@register_factor(
    name="mf_large_vol_net_5d",
    description="大单净买入量占比5日均值。持续的大单量净流入=机构持续建仓(量能角度)。",
    category="fund_flow",
    thesis=(
        "大单和超大单的净买入量占总成交量比例的5日均值。"
        "与金额维度的大单净流入互补——成交量维度的大单净流入更能"
        "反映机构的手数行为(买入多少股),而不仅仅是金额行为(花了多少钱)。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_large_vol_net_5d(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    large_net_vol = (ff["buy_lg_vol"] + ff["buy_elg_vol"]
                     - ff["sell_lg_vol"] - ff["sell_elg_vol"])
    ratio = safe_divide(large_net_vol, _total_vol(ff))
    ratio_ma5 = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    ratio_ma5 = ratio_ma5.clip(-0.5, 0.5)
    return cross_sectional_rank(ratio_ma5)
