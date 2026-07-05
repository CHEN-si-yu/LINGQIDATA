"""
Short selling factors -- Class 1 panel factors.

These use completely untapped margin_detail.parquet short-selling columns:
  - rqye: short-selling balance (融券余额)
  - rqmcl: short-selling volume (融券卖出量)
  - rqyl: short-selling remaining shares (融券余量)

Short-side factors are almost entirely absent from the 1,223 existing factors.
Short interest and short squeeze are well-documented alpha sources globally,
and A-shares now have a functioning securities lending market.

All factors use context.load() for daily-panel access to margin_detail.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Short Interest Level Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="short_balance_ratio",
    description="融券余额占比因子，rqye/circ_mv截面排名（高融券占比=看空压力大排后）。",
    category="risk",
    thesis=(
        "Short-selling balance relative to market cap measures bearish "
        "sentiment from informed short sellers. High short interest "
        "indicates sophisticated investors are betting against the stock. "
        "Academic research shows high short interest predicts negative "
        "future returns in most markets."
    ),
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_short_balance_ratio(context: FactorContext):
    md = context.load("margin_detail.parquet")
    fin = context.load("finance.parquet")

    rqye = md["rqye"].fillna(0)
    circ_mv = fin["circ_mv"]

    common = rqye.index.intersection(circ_mv.index)
    ratio = safe_divide(rqye.loc[common], circ_mv.loc[common] + 1e-10)
    ratio = ratio.clip(0, 0.1)  # winsorize
    return cross_sectional_rank(-ratio)  # low short interest = good


@register_factor(
    name="short_volume_intensity",
    description="融券卖出强度因子，rqmcl/float_share截面排名（高融券量=做空活跃排后）。",
    category="risk",
    thesis=(
        "Daily short-selling volume relative to float measures the "
        "intensity of short-selling activity. High short volume means "
        "active bearish positioning. Unlike short balance (stock), "
        "short volume (flow) captures the immediate sentiment change."
    ),
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_short_volume_intensity(context: FactorContext):
    md = context.load("margin_detail.parquet")
    fin = context.load("finance.parquet")

    rqmcl = md["rqmcl"].fillna(0)
    float_share = fin["float_share"]

    common = rqmcl.index.intersection(float_share.index)
    intensity = safe_divide(rqmcl.loc[common], float_share.loc[common] + 1e-10)
    intensity = intensity.clip(0, 0.05)
    return cross_sectional_rank(-intensity)


@register_factor(
    name="short_cover_potential",
    description="轧空潜力因子，rqyl/rqmcl截面排名（高未偿还比例=轧空风险排前=利好多头）。",
    category="risk",
    thesis=(
        "Short remaining shares relative to daily short volume = "
        "days-to-cover for shorts. High ratio means shorts cannot exit "
        "quickly, creating short-squeeze potential. When positive news "
        "hits, shorts are forced to buy back, amplifying upside. "
        "This is the classic short-squeeze setup factor."
    ),
    dependencies=("margin_detail.parquet",),
)
def factor_short_cover_potential(context: FactorContext):
    md = context.load("margin_detail.parquet")

    rqyl = md["rqyl"].fillna(0)
    rqmcl = md["rqmcl"].fillna(0)

    # Days to cover
    dtc = safe_divide(rqyl, rqmcl + 1e-10)
    dtc = dtc.clip(0, 50)
    return cross_sectional_rank(dtc)  # high days-to-cover = squeeze potential


# ═══════════════════════════════════════════════════════════════════════════════
# Short Flow Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="short_balance_change_5d",
    description="融券余额变化因子，rqye的5日变化率截面排名（融券增加=看空加剧排后）。",
    category="risk",
    thesis=(
        "5-day change in short balance captures the direction of "
        "short-selling pressure. Increasing shorts = bearish conviction "
        "building. Decreasing shorts = bears covering (bullish). "
        "Flow-based signal is more timely than level-based."
    ),
    dependencies=("margin_detail.parquet",),
)
def factor_short_balance_change_5d(context: FactorContext):
    md = context.load("margin_detail.parquet")
    rqye = md["rqye"].fillna(0)

    chg = rqye.groupby(level="Code").transform(
        lambda s: s.pct_change(5)
    )
    chg = chg.clip(-1, 5)
    return cross_sectional_rank(-chg)  # decreasing shorts = bullish


@register_factor(
    name="short_squeeze_signal",
    description="轧空信号因子，(短期余额↓+高days-to-cover)×价格动量截面排名。",
    category="risk",
    thesis=(
        "Short squeeze signal combines three conditions: "
        "1) Shorts are covering (balance decreasing) "
        "2) Covering will take time (high days-to-cover) "
        "3) Price is already moving against shorts (positive momentum). "
        "When all three align, a short squeeze may be underway."
    ),
    dependencies=("margin_detail.parquet", "daily_adj.parquet"),
)
def factor_short_squeeze_signal(context: FactorContext):
    md = context.load("margin_detail.parquet")
    daily = context.load("daily_adj.parquet")

    rqye = md["rqye"].fillna(0)
    rqmcl = md["rqmcl"].fillna(0)
    close = daily["close"]

    # Short covering: negative 5d change in balance
    cover = -rqye.groupby(level="Code").transform(
        lambda s: s.pct_change(5)
    )

    # Days to cover
    dtc = safe_divide(rqye, rqmcl + 1e-10)

    # Price momentum
    mom = close.groupby(level="Code").transform(
        lambda s: s.pct_change(5)
    )

    common = cover.index.intersection(dtc.index).intersection(mom.index)
    cover_a = cover.loc[common].clip(-1, 5)
    dtc_a = dtc.loc[common].clip(0, 50)
    # Normalize dtc to 0-1 rank
    dtc_rank = dtc_a.groupby(level="Date").rank(pct=True)
    mom_a = mom.loc[common].clip(-0.5, 0.5)

    # Signal: covering + high dtc + positive momentum
    signal = cover_a * dtc_rank * (mom_a - mom_a.groupby(level="Date").transform("mean"))
    return cross_sectional_rank(signal)


# ═══════════════════════════════════════════════════════════════════════════════
# Total Leverage Indicator
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="total_margin_ratio",
    description="总杠杆率因子，rzrqye/circ_mv截面排名（高杠杆=风险积聚排后）。",
    category="risk",
    thesis=(
        "Total margin+short balance relative to market cap = total "
        "leverage concentration in a stock. High leverage means the "
        "stock is heavily bet on (both long and short). Extreme leverage "
        "often precedes volatility expansion as positions unwind."
    ),
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_total_margin_ratio(context: FactorContext):
    md = context.load("margin_detail.parquet")
    fin = context.load("finance.parquet")

    rzrqye = md["rzrqye"].fillna(0)
    circ_mv = fin["circ_mv"]

    common = rzrqye.index.intersection(circ_mv.index)
    ratio = safe_divide(rzrqye.loc[common], circ_mv.loc[common] + 1e-10)
    ratio = ratio.clip(0, 0.3)
    return cross_sectional_rank(-ratio)


