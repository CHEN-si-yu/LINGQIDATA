"""
Advanced margin / short-selling factors — Class 1 panel factors.

These factors extend beyond the basic margin and short-selling signals
with cross-data-source interactions:
  - margin_detail + finance: total leverage to market cap

Data sources: margin_detail.parquet, finance.parquet

时点对齐:margin_detail 因上游延迟一天,数据层已统一映射为
Date=T 上的 margin 值 = 原始 T-1；其他日频字段仍使用 T 日值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    safe_divide,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Total Leverage Ratio — margin_detail + finance
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="total_leverage_ratio",
    description="融资融券总余额/总市值，衡量杠杆化程度。高杠杆=波动风险大，排名取反。",
    category="fund_flow",
    thesis="融资融券总余额(rzrqye)/总市值(total_mv)衡量股票的杠杆化程度。"
           "高杠杆意味着一部分市值被杠杆资金锁定——这些资金在市场下跌时面临"
           "强制平仓风险，可能放大跌幅。低杠杆意味着更健康的持仓结构。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_total_leverage_ratio(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    f = context.load("finance.parquet")
    # Date=T: margin 是可获得的原始 T-1 值，市值按用户约束使用 T 日值。
    mv_t = f["total_mv"].reindex(m.index)
    ratio = safe_divide(
        m["rzrqye"],
        mv_t,
    )
    ratio = ratio.clip(0, 0.5)
    return cross_sectional_rank(-ratio)
