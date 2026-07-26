"""
Advanced margin / short-selling factors — Class 1 panel factors.

These factors extend beyond the basic margin and short-selling signals
with cross-data-source interactions:
  - margin_detail + daily_adj: price-margin confirmation
  - margin_detail + daily: leverage bet ratio (long vs short)
  - margin_detail + finance: total leverage to market cap

Data sources: margin_detail.parquet, daily_adj.parquet, daily.parquet, finance.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    safe_divide,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Price-Margin Confirmation — margin_detail + daily_adj
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="margin_price_confirmation_5d",
    description="融资余额5日变化率×当日收益率，量价确认信号。量价齐升=排名高，背离=排名低。",
    category="fund_flow",
    thesis="融资余额变化率乘以当日价格变化率，构成量价确认信号。融资增加+价格上涨(量价齐升)"
           "意味着杠杆资金推动的上涨有基本面支撑；融资增加+价格下跌(量价背离)意味着"
           "杠杆资金接盘但股价不涨——典型的机构出货信号。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_margin_price_confirmation_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    d = context.load("daily_adj.parquet")
    rzye_chg = m["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(5))
    common_idx = rzye_chg.index.intersection(d.index)
    confirmation = rzye_chg.loc[common_idx] * d.loc[common_idx, "pct_chg"] / 100.0
    confirmation = confirmation.clip(-0.5, 0.5)
    return cross_sectional_rank(confirmation)


@register_factor(
    name="short_price_reversal_5d",
    description="融券余额5日变化率×当日收益率取反。空头加仓+价格上涨=恶劣信号，排名低。",
    category="fund_flow",
    thesis="融券余额变化率乘以当日收益率取反排名。融券增加+价格上涨(空头在上涨中加仓)"
           "意味着聪明钱认为上涨不可持续——是恶劣的看空信号；融券减少+价格下跌"
           "(空头在下跌中回补)意味着空头开始离场，可能反转。",
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_short_price_reversal_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    d = context.load("daily_adj.parquet")
    rqye_chg = m["rqye"].groupby(level="Code").transform(lambda s: s.pct_change(5))
    common_idx = rqye_chg.index.intersection(d.index)
    reversal = rqye_chg.loc[common_idx] * d.loc[common_idx, "pct_chg"] / 100.0
    reversal = reversal.clip(-0.5, 0.5)
    return cross_sectional_rank(-reversal)


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
    common_idx = m.index.intersection(f.index)
    ratio = safe_divide(
        m.loc[common_idx, "rzrqye"],
        f.loc[common_idx, "total_mv"],
    )
    ratio = ratio.clip(0, 0.5)
    return cross_sectional_rank(-ratio)
