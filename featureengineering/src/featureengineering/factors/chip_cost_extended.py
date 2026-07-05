
"""
Chip cost extended factors -- Class 1 panel factors from cyq_perf.parquet.

cyq_perf.parquet contains pre-computed chip cost distribution percentiles:
  - cost_5pct through cost_95pct: price levels at which X% of chips are profitable
  - his_low / his_high: all-time historical low/high reference prices
  - weight_avg: volume-weighted average holding cost

These are completely untapped -- existing chip factors use the per-stock
cyq_chips/ directory (Class 2), not this pre-aggregated cyq_perf.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide


@register_factor(
    name="cost_distribution_width",
    description="筹码成本分布宽度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（窄分布=筹码集中排前）。",
    category="price",
    thesis=(
        "Width of chip cost distribution (95th - 5th percentile) "
        "relative to median cost measures holder concentration. "
        "Narrow = holders agree on value = accumulation. "
        "Wide = holders have diverse costs = potential volatility."
    ),
    dependencies=("cyq_perf.parquet",),
)
def factor_cost_distribution_width(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    width = cyq["cost_95pct"] - cyq["cost_5pct"]
    median = cyq["cost_50pct"]
    rel_width = safe_divide(width, median + 1e-10)
    rel_width = rel_width.clip(0, 5)
    return cross_sectional_rank(-rel_width)


@register_factor(
    name="cost_skew_ratio",
    description="筹码成本偏度比率因子，(cost_50pct-cost_5pct)/(cost_95pct-cost_50pct)截面排名（右偏=获利盘主导排前）。",
    category="price",
    thesis=(
        "Cost distribution skew: right-skew means more chips below median "
        "(most holders in profit = bullish). Left-skew means more chips "
        "above median (most holders underwater = selling pressure)."
    ),
    dependencies=("cyq_perf.parquet",),
)
def factor_cost_skew_ratio(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    lower_range = cyq["cost_50pct"] - cyq["cost_5pct"]
    upper_range = cyq["cost_95pct"] - cyq["cost_50pct"]
    skew = safe_divide(lower_range, upper_range + 1e-10)
    skew = skew.clip(0.1, 10)
    return cross_sectional_rank(skew)


@register_factor(
    name="distance_from_all_time_high",
    description="距历史最高价距离因子，(close-his_high)/his_high截面排名（接近新高=强势排前）。",
    category="price",
    thesis=(
        "Distance from all-time high: near-high stocks have no underwater "
        "holders = limited overhead selling pressure. Far-from-high stocks "
        "carry supply from holders waiting to break even."
    ),
    dependencies=("cyq_perf.parquet", "daily_adj.parquet"),
)
def factor_distance_from_all_time_high(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    daily = context.load("daily_adj.parquet")
    his_high = cyq["his_high"]
    close = daily["close"]
    common = his_high.index.intersection(close.index)
    distance = safe_divide(
        close.loc[common] - his_high.loc[common],
        his_high.loc[common] + 1e-10
    )
    distance = distance.clip(-1, 1)
    return cross_sectional_rank(distance)


@register_factor(
    name="distance_from_all_time_low",
    description="距历史最低价距离因子，(close-his_low)/his_low截面排名（远离最低=趋势健康排前）。",
    category="price",
    thesis=(
        "Distance from all-time low measures recovery strength. "
        "Far from lows = strong positive momentum and improvement. "
        "Near lows = potential structural decline."
    ),
    dependencies=("cyq_perf.parquet", "daily_adj.parquet"),
)
def factor_distance_from_all_time_low(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    daily = context.load("daily_adj.parquet")
    his_low = cyq["his_low"]
    close = daily["close"]
    common = his_low.index.intersection(close.index)
    distance = safe_divide(
        close.loc[common] - his_low.loc[common],
        his_low.loc[common] + 1e-10
    )
    distance = distance.clip(-1, 20)
    return cross_sectional_rank(distance)


@register_factor(
    name="avg_cost_premium",
    description="平均成本溢价因子，(close-weight_avg)/weight_avg截面排名（现价高于均价=多数人盈利排前）。",
    category="price",
    thesis=(
        "Premium of current price over volume-weighted average cost. "
        "Positive = average holder in profit = bullish. "
        "Negative = average holder underwater = selling pressure on rallies."
    ),
    dependencies=("cyq_perf.parquet", "daily_adj.parquet"),
)
def factor_avg_cost_premium(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    daily = context.load("daily_adj.parquet")
    weight_avg = cyq["weight_avg"]
    close = daily["close"]
    common = weight_avg.index.intersection(close.index)
    premium = safe_divide(
        close.loc[common] - weight_avg.loc[common],
        weight_avg.loc[common] + 1e-10
    )
    premium = premium.clip(-1, 5)
    return cross_sectional_rank(premium)


@register_factor(
    name="cost_convergence_signal",
    description="成本收敛信号因子，(cost_95pct-cost_5pct)的20日变化率截面排名（分布收窄=筹码集中排前）。",
    category="price",
    thesis=(
        "Cost distribution convergence: narrowing = holders agree on value "
        "(accumulation). Widening = new holders at diverse prices "
        "(distribution). Rate of convergence is the second derivative."
    ),
    dependencies=("cyq_perf.parquet",),
)
def factor_cost_convergence_signal(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    width = cyq["cost_95pct"] - cyq["cost_5pct"]
    chg = width.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    chg = chg.clip(-1, 1)
    return cross_sectional_rank(-chg)
