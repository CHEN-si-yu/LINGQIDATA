"""
Deep margin factors — Class 1 panel factors.

Advanced margin flow analysis:
  - Flow asymmetry: directional persistence of net margin flow
  - Repay shock: sudden margin liquidation detection

Data source: margin_detail.parquet
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Flow Asymmetry — directional persistence
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="margin_flow_asymmetry_10d",
    description="10日累计融资净买入/累计总交易额，资金流向方向持续性。持续净买入=排名高。",
    category="fund_flow",
    thesis="过去10日累计融资净买入/(累计买入+累计偿还)衡量融资净流入的方向持续性。"
           "接近1意味着持续10日几乎全是净买入——杠杆资金方向极其一致；"
           "接近-1意味着持续净卖出。资金流向的持续性比单日方向更能预测未来走势。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_flow_asymmetry_10d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    net = m["rzmre"] - m["rzche"]
    total = m["rzmre"] + m["rzche"]
    net_10d = net.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).sum()
    )
    total_10d = total.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).sum()
    )
    asymmetry = safe_divide(net_10d, total_10d)
    asymmetry = asymmetry.clip(-1, 1)
    return cross_sectional_rank(asymmetry)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Repay Shock — margin liquidation detection
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="margin_repay_shock",
    description="融资偿还冲击=当日偿还额/20日均偿还额取反。突然放大=恐慌平仓，排名低。",
    category="fund_flow",
    thesis="单日融资偿还额相对20日均值的倍数取反。偿还突然放大(偿还冲击)意味着融资盘集中平仓"
           "——可能是股价触发平仓线或投资者主动止损。偿还冲击越大的股票越看空(排名取反)。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_repay_shock(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    rzche = m["rzche"]
    repay_ma20 = rzche.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    shock = safe_divide(rzche, repay_ma20)
    shock = shock.clip(0, 5)
    return cross_sectional_rank(-shock)
