"""
Margin short-selling deep factors — Class 1 panel factors.

Exploits underutilized margin_detail.parquet fields:
  - rqyl: short-selling remaining shares (融券余量)
  - rqmcl: short-selling volume (融券卖出量)

These capture short-selling intensity and short-squeeze risk.
NaN ~9% (2022 后实测), acceptable per skill.md <20% threshold.

时点对齐:margin_detail 因上游延迟一天,数据层已统一映射为 Date=T 上的
margin 值 = 原始 T-1；daily 等正常日频数据在 Date=T 直接使用 T 日值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Short Interest / Concentration (rqyl)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="short_interest_volatility_20d",
    description="融券余量20日波动率取反。余量剧烈波动=空头态度摇摆/不稳定,排名低。",
    category="fund_flow",
    thesis=(
        "融券余量20日变异系数(std/mean)。高波动意味着空头在频繁开仓平仓——"
        "做空方向不坚定、观点摇摆(噪音空头)；低波动意味着空头仓位稳定——"
        "做空信念坚定(信息型空头)。低波动的持续做空比高波动的频繁进出更有预测力。"
    ),
    dependencies=("margin_detail.parquet",),
)
def factor_short_interest_volatility_20d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    rqyl = m["rqyl"]
    roll_std = rqyl.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    roll_mean = rqyl.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    cv = safe_divide(roll_std, roll_mean)
    cv = cv.clip(0, 2)
    return cross_sectional_rank(-cv)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Short Selling Flow Dynamics (rqmcl)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="short_sell_volume_ratio",
    description="融券卖出占比：rqmcl/vol。融券卖出量相对总成交量的占比,高速=活跃做空。",
    category="fund_flow",
    thesis=(
        "融券卖出量相对总成交量的占比。高占比=当日做空交易在总交易中占比大——"
        "做空意愿强烈、空头攻击火力集中。该比率提供了做空行为的市场份额视角。"
        "需要daily.parquet的成交量做分母。"
    ),
    dependencies=("margin_detail.parquet", "daily.parquet"),
)
def factor_short_sell_volume_ratio(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    d = context.load("daily.parquet")
    # Date=T 使用当时可获得的 T-1 融券卖出量和 T 日总成交量。
    vol_t = d["vol"].reindex(m.index)
    ratio = safe_divide(m["rqmcl"], vol_t)
    ratio = ratio.clip(0, 1)
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Short Squeeze Risk — combined margin + short signals
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="short_squeeze_risk",
    description="逼空风险=融券余量/融资余额。高比值=大量做空仓位vs低做多杠杆,逼空风险大,排名高(反转做多信号)。",
    category="fund_flow",
    thesis=(
        "融券余量(rqyl)/融资余额(rzye)衡量做空仓位相对做多杠杆的规模。"
        "比值异常高意味着大量资金在做空该股票——一旦股价反弹,空头被迫回补(逼空),"
        "可能引发剧烈上涨。该因子是经典的short-squeeze预警信号——"
        "做空过度拥挤的股票存在逼空风险,是潜在的反转做多机会。"
    ),
    dependencies=("margin_detail.parquet",),
)
def factor_short_squeeze_risk(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    squeeze = safe_divide(m["rqyl"], m["rzye"])
    squeeze = squeeze.clip(0, 10)
    return cross_sectional_rank(squeeze)
