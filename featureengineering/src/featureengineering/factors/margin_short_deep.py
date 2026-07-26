"""
Margin short-selling deep factors — Class 1 panel factors.

Exploits underutilized margin_detail.parquet fields:
  - rqyl: short-selling remaining shares (融券余量) — only 2 files
  - rqmcl: short-selling volume (融券卖出量) — only 3 files
  - exchange_id: exchange identifier (SZ/ SH) — only 1 file

These capture short-selling intensity, short-squeeze risk, and exchange-specific patterns.
NaN ~15% (margin coverage ~85%), acceptable per skill.md <20% threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, rolling_group_mean, rolling_group_std


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Short Interest / Concentration (rqyl)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="short_interest_ratio_5d",
    description="融券余量5日变化率。余量攀升=做空力量持续加码,排名低。",
    category="fund_flow",
    thesis=(
        "融券余量(rqyl)5日变化率取反排名。rqyl是已融出但尚未偿还的股数——"
        "它反映了市场上活跃的做空仓位规模。rqyl持续攀升=空头在加仓(看空)；"
        "rqyl快速下降=空头集中平仓(回补),短期可能推涨。"
    ),
    dependencies=("margin_detail.parquet",),
)
def factor_short_interest_ratio_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    chg = m["rqyl"].groupby(level="Code").transform(lambda s: s.pct_change(5))
    chg = chg.clip(-0.5, 1.0)
    return cross_sectional_rank(-chg)


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
    name="short_sell_velocity_5d",
    description="融券卖出速度5日变化：rqmcl/vol。卖出量相对总成交量的占比变化,高速=活跃做空。",
    category="fund_flow",
    thesis=(
        "融券卖出量相对总成交量的占比。高占比=当日做空交易在总交易中占比大——"
        "做空意愿强烈、空头攻击火力集中。该比率提供了做空行为的市场份额视角。"
        "需要daily_adj.parquet的成交量做分母。"
    ),
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_short_sell_velocity_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    d = context.load("daily_adj.parquet")
    common = m.index.intersection(d.index)
    ratio = safe_divide(m.loc[common, "rqmcl"], d.loc[common, "vol"])
    ratio = ratio.clip(0, 1)
    return cross_sectional_rank(ratio)


@register_factor(
    name="short_sell_momentum_10d",
    description="融券卖出量10日动量。卖出量持续放大=空头越战越勇,排名低。",
    category="fund_flow",
    thesis=(
        "融券卖出量(rqmcl)10日滚动均值的环比变化。卖出量持续放大意味着做空交易越来越活跃——"
        "空头信心增强(或者股价持续跌,吸引更多做空者)；卖出量萎缩意味着做空兴趣减退。"
    ),
    dependencies=("margin_detail.parquet",),
)
def factor_short_sell_momentum_10d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    rqmcl = m["rqmcl"]
    ma10 = rqmcl.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).mean()
    )
    momentum = safe_divide(ma10.diff(5), ma10.shift(5))
    momentum = momentum.clip(-0.5, 1.0)
    return cross_sectional_rank(-momentum)


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


@register_factor(
    name="margin_short_flow_divergence_5d",
    description="融资净买入vs融券卖出量的5日背离。融资买+融券卖少=多头主导,排名高。",
    category="fund_flow",
    thesis=(
        "融资净买入(rzmre-rzche)与融券卖出量(rqmcl)的5日滚动背离。"
        "融资净买入增加+融券卖出量减少=做多力量增强+做空力量减弱——多头全面占优；"
        "融资净买入减少+融券卖出量增加=多头撤退+空头进攻——信号最差。"
    ),
    dependencies=("margin_detail.parquet",),
)
def factor_margin_short_flow_divergence_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    margin_net = m["rzmre"] - m["rzche"]
    margin_net_ma5 = margin_net.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    short_sell_ma5 = m["rqmcl"].groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    margin_chg = margin_net_ma5.groupby(level="Code").transform(lambda s: s.pct_change(5))
    short_chg = short_sell_ma5.groupby(level="Code").transform(lambda s: s.pct_change(5))
    divergence = margin_chg - short_chg
    divergence = divergence.clip(-1.0, 1.0)
    return cross_sectional_rank(divergence)
