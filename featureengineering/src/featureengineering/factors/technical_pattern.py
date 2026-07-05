"""
Technical pattern factors -- Class 1 panel factors from daily_adj.parquet.

These implement novel technical indicators inspired by quantitative strategies:
  - RSRS (Resistance Support Relative Strength): OLS regression of high on low
  - Donchian Channel position: position within N-day price channel
  - ATR-normalized momentum: trend strength adjusted for volatility

All use daily high/low/close data which is already loaded by many existing factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# RSRS (Resistance Support Relative Strength) Factors
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_rsrs_beta(high, low, window=18):
    """Compute RSRS beta: OLS slope of high ~ low over rolling window.

    RSRS quantifies the relationship between daily resistance (high) and
    support (low). A rising beta means resistance is rising faster than
    support -- bullish momentum. A falling beta means support is weakening.

    Returns: (beta, r_squared) per stock per date.
    """
    # Compute rolling sums for OLS: beta = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x2) - sum(x)^2)
    x = low
    y = high
    n = window

    sum_x = x.groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n//2).sum())
    sum_y = y.groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n//2).sum())
    sum_xy = (x * y).groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n//2).sum())
    sum_x2 = (x * x).groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n//2).sum())
    sum_y2 = (y * y).groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n//2).sum())

    denominator = n * sum_x2 - sum_x * sum_x
    beta = safe_divide(n * sum_xy - sum_x * sum_y, denominator)

    # R-squared
    numerator_r2 = (n * sum_xy - sum_x * sum_y) ** 2
    denominator_r2 = denominator * (n * sum_y2 - sum_y * sum_y)
    r2 = safe_divide(numerator_r2, denominator_r2)

    return beta, r2


@register_factor(
    name="rsrs_beta_18",
    description="RSRS Beta因子，18日high~low回归斜率截面排名（高beta=阻力上升快于支撑=看涨排前）。",
    category="price",
    thesis=(
        "RSRS (Resistance Support Relative Strength) measures the dynamic "
        "relationship between daily highs and lows via OLS regression. "
        "When beta is high, resistance levels are rising faster than support "
        "levels -- a bullish regime shift. When beta is low or falling, "
        "support is weakening -- a bearish signal. "
        "Originally developed for index timing, applied cross-sectionally "
        "this identifies stocks undergoing support/resistance regime changes."
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_rsrs_beta_18(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    high = daily["high"]
    low = daily["low"]
    beta, _ = _compute_rsrs_beta(high, low, window=18)
    return cross_sectional_rank(beta)


@register_factor(
    name="rsrs_r2_18",
    description="RSRS R-squared因子，18日high~low回归拟合优度截面排名（高R2=支撑阻力关系清晰排前）。",
    category="price",
    thesis=(
        "RSRS R-squared measures the quality of the support/resistance "
        "relationship. High R2 means the high-low relationship is tight "
        "and predictable -- clear support/resistance structure. "
        "Low R2 means the relationship is breaking down -- uncertainty, "
        "potential regime change. Low R2 precedes volatility expansion."
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_rsrs_r2_18(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    high = daily["high"]
    low = daily["low"]
    _, r2 = _compute_rsrs_beta(high, low, window=18)
    return cross_sectional_rank(r2)


@register_factor(
    name="rsrs_zscore_18",
    description="RSRS Z-score因子，beta相对自身历史400日的标准化偏离截面排名（极端偏离=支撑阻力位重塑排前）。",
    category="price",
    thesis=(
        "RSRS standardized score measures how extreme the current beta is "
        "relative to its own history. A large positive z-score means the "
        "support/resistance structure is undergoing a significant bullish "
        "shift. This is the signal used in the original RSRS timing strategy."
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_rsrs_zscore_18(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    high = daily["high"]
    low = daily["low"]
    beta, _ = _compute_rsrs_beta(high, low, window=18)

    # Z-score relative to 400-day rolling window
    roll_mean = rolling_group_mean(beta, 400, min_periods=100)
    roll_std = rolling_group_std(beta, 400, min_periods=100)
    zscore = safe_divide(beta - roll_mean, roll_std + 1e-8)
    return cross_sectional_rank(zscore)


@register_factor(
    name="rsrs_right_deviation",
    description="RSRS右偏离因子，(zscore * beta * r2)截面排名（量价验证=信号可靠排前）。",
    category="price",
    thesis=(
        "RSRS Right Deviation combines three dimensions: "
        "1) Z-score: how extreme is the regime shift? "
        "2) Beta: what direction is the shift? "
        "3) R-squared: how reliable is the measured relationship? "
        "This is the composite signal from the original RSRS strategy, "
        "applied cross-sectionally to rank stocks by support/resistance quality."
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_rsrs_right_deviation(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    high = daily["high"]
    low = daily["low"]
    beta, r2 = _compute_rsrs_beta(high, low, window=18)

    roll_mean = rolling_group_mean(beta, 400, min_periods=100)
    roll_std = rolling_group_std(beta, 400, min_periods=100)
    zscore = safe_divide(beta - roll_mean, roll_std + 1e-8)

    # Combined score: right-deviation = zscore * beta * r2
    score = zscore * beta * r2.fillna(0)
    return cross_sectional_rank(score)


# ═══════════════════════════════════════════════════════════════════════════════
# Donchian Channel Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="donchian_position_20",
    description="Donchian通道位置因子，(close-20日最低)/(20日最高-20日最低)截面排名（突破高位=强势排前）。",
    category="price",
    thesis=(
        "Donchian Channel position measures where price sits within its "
        "20-day range. Values near 1.0 = price at 20-day high (breakout), "
        "near 0.0 = price at 20-day low (breakdown). "
        "Turtle traders buy breakouts from 20-day highs. This factor "
        "captures the same signal in cross-sectional form. "
        "Channel position is comparable across stocks regardless of price level."
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_donchian_position_20(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]

    highest = high.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    lowest = low.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).min()
    )

    position = safe_divide(close - lowest, highest - lowest + 1e-10)
    position = position.clip(0, 1)
    return cross_sectional_rank(position)


@register_factor(
    name="donchian_breakout_strength",
    description="Donchian突破强度因子，(close-20日最高)/ATR截面排名（突破幅度相对波动率排前）。",
    category="price",
    thesis=(
        "Donchian breakout strength normalizes the breakout distance by ATR. "
        "A breakout 2 ATR above the channel high is more significant than "
        "a breakout 0.1 ATR above. ATR-normalization makes breakouts "
        "comparable across stocks with different volatility. "
        "Strong breakouts are more likely to develop into sustained trends."
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_donchian_breakout_strength(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]

    highest = high.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )

    # ATR: average true range
    tr1 = high - low
    tr2 = (high - close.groupby(level="Code").shift(1)).abs()
    tr3 = (low - close.groupby(level="Code").shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    strength = safe_divide(close - highest, atr + 1e-10)
    return cross_sectional_rank(strength)


# ═══════════════════════════════════════════════════════════════════════════════
# ATR-Normalized Momentum Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="atr_momentum_20",
    description="ATR标准化动量因子，(close-close_20d)/ATR_20截面排名（单位风险的收益效率排前）。",
    category="price",
    thesis=(
        "ATR-normalized momentum divides 20-day return by 20-day ATR, "
        "measuring return per unit of risk. This is the risk-adjusted "
        "trend strength -- a stock that gained 10% with 1% daily ATR "
        "shows stronger trending behavior than one that gained 10% "
        "with 5% daily ATR. Comparable across volatility regimes."
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_atr_momentum_20(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]

    # 20-day return
    ret_20 = close.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )

    # ATR
    tr1 = high - low
    tr2 = (high - close.groupby(level="Code").shift(1)).abs()
    tr3 = (low - close.groupby(level="Code").shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    momentum = safe_divide(ret_20, atr + 1e-10)
    return cross_sectional_rank(momentum)


@register_factor(
    name="atr_ratio_20",
    description="ATR比率因子，ATR_20/close截面排名（高相对波幅=高风险排后）。",
    category="risk",
    thesis=(
        "ATR as a percentage of price measures relative volatility. "
        "Unlike standard deviation which treats all deviations equally, "
        "ATR focuses on the true range (including overnight gaps). "
        "This captures gap risk that standard deviation misses. "
        "ATR/Price is directly comparable across stocks."
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_atr_ratio_20(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]

    tr1 = high - low
    tr2 = (high - close.groupby(level="Code").shift(1)).abs()
    tr3 = (low - close.groupby(level="Code").shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    atr_pct = safe_divide(atr, close + 1e-10)
    return cross_sectional_rank(-atr_pct)  # low relative ATR = stable


