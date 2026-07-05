"""
Extended fund flow factors — Class 1.

Amount-based fund flow factors using previously unused amount columns from
main_fund_flow.parquet.  The existing fund_flow.py uses net_mf_amount and
some ratios; unused_fields_factors.py uses the volume columns for 4 factors.
This module focuses on the **amount (成交额) columns per order size** which
remain untapped — providing a complementary perspective to volume-based metrics.

Data source: ``main_fund_flow.parquet`` (20 columns)

Order size classification: SM < 40K < MD < 200K < LG < 1M < ELG
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    safe_divide,
)


def _total_amount(ff: pd.DataFrame) -> pd.Series:
    """Total trading amount across all order sizes, zero → NaN."""
    return (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)


@register_factor(
    name="ext_mf_small_order_amount_ratio",
    description="小额订单成交额占比因子，散户参与度度量（低值排前）。",
    category="fund_flow",
    thesis="小额订单（<4万元）成交额占比高意味着散户交易活跃。"
    "散户占比高的股票往往存在行为偏差（过度交易、追涨杀跌），"
    "机构化的股票小额订单占比低。低散户占比是A股机构化趋势下的质量信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_ext_mf_small_order_amount_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    sm_amount = ff["buy_sm_amount"] + ff["sell_sm_amount"]
    ratio = safe_divide(sm_amount, _total_amount(ff))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_mf_medium_order_amount_ratio",
    description="中单成交额占比因子，中单（4-20万）成交额/总成交额截面排名。",
    category="fund_flow",
    thesis="中单成交额占比反映了中等资金（游资/大户）的参与程度。"
    "与散户和机构不同，中等资金的行为模式更加灵活，是市场活跃度的中间层指标。"
    "中单占比异常波动往往预示着资金结构的变化。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_ext_mf_medium_order_amount_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    md_amount = ff["buy_md_amount"] + ff["sell_md_amount"]
    ratio = safe_divide(md_amount, _total_amount(ff))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_mf_big_order_net_amount_ratio",
    description="大单净买入金额占比因子，大单+特大单净买入额/总成交额截面排名。",
    category="fund_flow",
    thesis="成交额口径的大单净买入占比与成交量口径（mf_big_order_vol_ratio）互补："
    "成交额口径考虑了价格信息，反映了机构资金的'质量'（高价买入=更强的信心）。"
    "两个口径的差异本身就是重要信号（参考mf_amount_vol_divergence）。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_ext_mf_big_order_net_amount_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    big_net_amt = (
        ff["buy_lg_amount"] - ff["sell_lg_amount"]
        + ff["buy_elg_amount"] - ff["sell_elg_amount"]
    )
    ratio = safe_divide(big_net_amt, _total_amount(ff))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_mf_amount_concentration",
    description="成交额集中度因子（低值排前），各订单规模成交额占比的Herfindahl指数。",
    category="fund_flow",
    thesis="成交额在各订单规模间的分布集中度反映了市场参与者的多样性。"
    "高度集中（如大单占比>70%）意味着单一资金类型主导，行情持续性存疑。"
    "适度分散的成交结构更健康（各类资金共同参与）。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_ext_mf_amount_concentration(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total = _total_amount(ff)

    sm_share = (ff["buy_sm_amount"] + ff["sell_sm_amount"]) / total
    md_share = (ff["buy_md_amount"] + ff["sell_md_amount"]) / total
    lg_share = (ff["buy_lg_amount"] + ff["sell_lg_amount"]) / total
    elg_share = (ff["buy_elg_amount"] + ff["sell_elg_amount"]) / total

    # Herfindahl-Hirschman Index: sum of squared shares
    hhi = sm_share**2 + md_share**2 + lg_share**2 + elg_share**2
    return cross_sectional_rank(-hhi)  # lower concentration = higher rank


@register_factor(
    name="ext_mf_small_order_avg_price",
    description="小额订单均价因子（低值排前），小额成交均价/VWAP截面排名。",
    category="fund_flow",
    thesis="小额订单的成交均价与VWAP的比较反映散户交易的执行质量。"
    "散户成交均价偏高（>VWAP）可能意味着追高买入行为，"
    "散户成交均价偏低（<VWAP）可能是恐慌性抛售。"
    "该比率偏离1的程度是散户情绪的温度计。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_ext_mf_small_order_avg_price(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total = _total_amount(ff)
    total_vol = (
        ff["buy_sm_vol"] + ff["sell_sm_vol"]
        + ff["buy_md_vol"] + ff["sell_md_vol"]
        + ff["buy_lg_vol"] + ff["sell_lg_vol"]
        + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    ).replace(0, np.nan)

    vwap = total / total_vol  # volume-weighted average price

    sm_total_vol = ff["buy_sm_vol"] + ff["sell_sm_vol"]
    sm_total_amt = ff["buy_sm_amount"] + ff["sell_sm_amount"]
    sm_avg_price = safe_divide(sm_total_amt, sm_total_vol)

    ratio = safe_divide(sm_avg_price, vwap)
    # Deviation from 1 (overpaying or underpaying relative to VWAP)
    deviation = (ratio - 1.0).abs()
    return cross_sectional_rank(-deviation)


@register_factor(
    name="ext_mf_large_order_avg_price",
    description="大单均价因子，大单+特大单成交均价/VWAP截面排名（高值排前）。",
    category="fund_flow",
    thesis="大单成交均价高于VWAP意味着机构愿意以高于市场均价的价格买入——"
    "这是机构做多意愿强烈的信号。机构低价买入是正常行为，高价买入是超常行为。"
    "大单均价/VWAP的比率是机构买卖紧迫度的代理变量。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_ext_mf_large_order_avg_price(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total = _total_amount(ff)
    total_vol = (
        ff["buy_sm_vol"] + ff["sell_sm_vol"]
        + ff["buy_md_vol"] + ff["sell_md_vol"]
        + ff["buy_lg_vol"] + ff["sell_lg_vol"]
        + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    ).replace(0, np.nan)

    vwap = total / total_vol

    lg_total_vol = (
        ff["buy_lg_vol"] + ff["sell_lg_vol"]
        + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    )
    lg_total_amt = (
        ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    )
    lg_avg_price = safe_divide(lg_total_amt, lg_total_vol)

    ratio = safe_divide(lg_avg_price, vwap)
    return cross_sectional_rank(ratio)
