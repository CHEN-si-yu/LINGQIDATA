"""
Fund flow deep extended factors — Class 1 panel factors.

Extends fund flow analysis beyond the existing 30 factors in fund_flow.py.
Uses main_fund_flow.parquet for additional order-size decomposition.

Fields used:
  - buy_sm_amount, sell_sm_amount: small order
  - buy_md_amount, sell_md_amount: medium order
  - buy_lg_amount, sell_lg_amount: large order
  - buy_elg_amount, sell_elg_amount: extra-large order
  - net_mf_amount, net_mf_vol
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, rolling_group_mean


def _total_amount(ff: pd.DataFrame) -> pd.Series:
    """Total buy+sell amount across all order sizes."""
    cols = [
        "buy_sm_amount", "sell_sm_amount",
        "buy_md_amount", "sell_md_amount",
        "buy_lg_amount", "sell_lg_amount",
        "buy_elg_amount", "sell_elg_amount",
    ]
    total = ff[cols[0]].copy()
    for c in cols[1:]:
        total = total + ff[c]
    return total.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Order Size Concentration (Herfindahl-style)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_order_concentration",
    description="主力资金订单集中度=大单+超大单占比，Herfindahl指数式度量。高集中度=机构主导，排名高。",
    category="fund_flow",
    thesis=(
        "大单和超大单成交额占总成交额的比例。高比例意味着机构和主力资金主导了当日交易——"
        "这是专业资金的足迹。低比例意味着散户交易占主导，价格发现效率低。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_order_concentration(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    big = ff["buy_lg_amount"] + ff["sell_lg_amount"] + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    total = _total_amount(ff)
    concentration = safe_divide(big, total)
    return cross_sectional_rank(concentration)


@register_factor(
    name="mf_retail_dominance",
    description="散户交易占比=(小单买入+小单卖出)/总成交额取反。高散户占比=噪音交易多，排名低。",
    category="fund_flow",
    thesis=(
        "小单成交额占总成交额的比例取反排名。小单占比高意味着散户交易主导——噪音交易多、"
        "价格发现效率低。小单占比低意味着机构主导——定价更有效。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_retail_dominance(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    retail = ff["buy_sm_amount"] + ff["sell_sm_amount"]
    ratio = safe_divide(retail, _total_amount(ff))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Smart/Dumb Money Divergence
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_smart_dumb_divergence",
    description="聪明钱vs散户分歧=(大单净买/大单总额)-(小单净买/小单总额)。正值=机构买散户卖，排名高。",
    category="fund_flow",
    thesis=(
        "大单净流向与小单净流向的标准化差值。正值=大单净买入但小单净卖出——"
        "机构在收集筹码而散户在卖出(聪明钱信号)；负值=机构卖出但散户买入——"
        "机构在派发筹码(危险信号)。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_smart_dumb_divergence(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    big_net = (ff["buy_lg_amount"] + ff["buy_elg_amount"]
               - ff["sell_lg_amount"] - ff["sell_elg_amount"])
    big_total = (ff["buy_lg_amount"] + ff["sell_lg_amount"]
                 + ff["buy_elg_amount"] + ff["sell_elg_amount"])
    small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
    small_total = ff["buy_sm_amount"] + ff["sell_sm_amount"]
    big_ratio = safe_divide(big_net, big_total)
    small_ratio = safe_divide(small_net, small_total)
    divergence = big_ratio - small_ratio
    divergence = divergence.clip(-1, 1)
    return cross_sectional_rank(divergence)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Flow Persistence and Acceleration
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_flow_acceleration_5d",
    description="主力资金净流入加速度=5日净流入变化率。加速流入=增量资金积极，排名高。",
    category="fund_flow",
    thesis=(
        "主力资金净流入的5日二阶变化(加速度)。净流入本身是速度概念，加速度则捕捉了"
        "资金态度的边际变化——加速流入意味着资金越来越积极(乐观信号)；"
        "减速(从流入转为流出)意味着资金态度转变(谨慎信号)。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_flow_acceleration_5d(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    net = ff["net_mf_amount"]
    velocity = safe_divide(net, _total_amount(ff))
    accel = velocity.groupby(level="Code").transform(lambda s: s.diff(5))
    accel = accel.clip(-0.5, 0.5)
    return cross_sectional_rank(accel)


@register_factor(
    name="mf_flow_stability_20d",
    description="主力资金流向20日稳定性=连续同向天数占比。频繁转向=信号不可靠，排名低。",
    category="fund_flow",
    thesis=(
        "主力资金净流入方向在20日内的稳定性。持续同向(一直流入或一直流出)意味着"
        "主力态度明确——方向稳定。频繁转向(今天进明天出)意味着主力态度摇摆——"
        "信号噪音大、可靠性低。排名取反使得不稳定者排后。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_flow_stability_20d(context: FactorContext) -> np.ndarray:
    ff = context.load("main_fund_flow.parquet")
    net = ff["net_mf_amount"]
    # 修正(2026-08-05):
    # 1) NaN 处理:np.sign(NaN)=NaN,NaN!=x 恒 True——每股首行(prev 为 NaN)及
    #    任何 NaN 日都会被误计为一次"方向变化"。须同时要求 net 与 prev 均有效。
    # 2) 方向:原 rank(-stability) 使不稳定者排名反而靠前;描述要求"稳定者排前、
    #    不稳定者排后",应 rank(stability)。
    prev = net.groupby(level="Code").shift(1)
    sign_changes = (
        (np.sign(net) != np.sign(prev)) & net.notna() & prev.notna()
    ).astype(float)
    change_count = sign_changes.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    stability = 1.0 - change_count / 20.0
    return cross_sectional_rank(stability)
