"""
Technical pattern factors -- Class 1 panel factors from daily.parquet.

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
from .momentum_rebuilt import _adjusted_close


# ═══════════════════════════════════════════════════════════════════════════════
# RSRS (Resistance Support Relative Strength) Factors
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_rsrs_beta(high, low, window=18):
    """Compute RSRS beta: OLS slope of high ~ low over rolling window.

    RSRS quantifies the relationship between daily resistance (high) and
    support (low). A rising beta means resistance is rising faster than
    support -- bullish momentum. A falling beta means support is weakening.

    注意:调用方须传入折算到复权空间的 high/low(scale=后复权基座/close),
    未复权 high/low 在除权日同幅阶跃,会污染 18 日回归的斜率与 r²。

    Returns: (beta, r_squared) per stock per date.
    """
    # Compute rolling sums for OLS: beta = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x2) - sum(x)^2)
    x = low
    y = high
    n = window

    # The closed-form OLS equations below use n explicitly, so values are
    # valid only once the complete n-observation window is present.
    sum_x = x.groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n).sum())
    sum_y = y.groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n).sum())
    sum_xy = (x * y).groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n).sum())
    sum_x2 = (x * x).groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n).sum())
    sum_y2 = (y * y).groupby(level="Code").transform(lambda s: s.rolling(n, min_periods=n).sum())

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
    dependencies=("daily.parquet",),
)
def factor_rsrs_beta_18(context: FactorContext):
    daily = context.load("daily.parquet")
    # 折算到复权空间再回归,避免除权日 high/low 阶跃污染斜率(见 _compute_rsrs_beta)
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    high = daily["high"] * scale
    low = daily["low"] * scale
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
    dependencies=("daily.parquet",),
)
def factor_rsrs_r2_18(context: FactorContext):
    daily = context.load("daily.parquet")
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    high = daily["high"] * scale
    low = daily["low"] * scale
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
    dependencies=("daily.parquet",),
)
def factor_rsrs_zscore_18(context: FactorContext):
    daily = context.load("daily.parquet")
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    high = daily["high"] * scale
    low = daily["low"] * scale
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
    dependencies=("daily.parquet",),
)
def factor_rsrs_right_deviation(context: FactorContext):
    daily = context.load("daily.parquet")
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    high = daily["high"] * scale
    low = daily["low"] * scale
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
    dependencies=("daily.parquet",),
)
def factor_donchian_position_20(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    # 用每日复权系数 (后复权基座/close) 折算 high/low 后再取 20 日极值,
    # 避免除权日污染通道上下轨 (与 trend_pattern.donchian_position_60 同口径, 2026-08-05)
    scale = _adjusted_close(daily) / close.replace(0, np.nan)
    adj_high = daily["high"] * scale
    adj_low = daily["low"] * scale
    adj = _adjusted_close(daily)

    previous_high = adj_high.groupby(level="Code").shift(1)
    highest = previous_high.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    lowest = adj_low.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).min()
    )

    position = safe_divide(adj - lowest, highest - lowest + 1e-10)
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
    dependencies=("daily.parquet",),
)
def factor_donchian_breakout_strength(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]
    # 通道上轨用复权折算后的 high (除权日不产生假突破), 2026-08-05
    scale = _adjusted_close(daily) / close.replace(0, np.nan)
    adj = _adjusted_close(daily)
    adj_high = high * scale

    highest = adj_high.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )

    # ATR: average true range (pre_close is dividend-adjusted at ex-dividend dates)
    tr1 = high - low
    tr2 = (high - daily["pre_close"]).abs()
    tr3 = (low - daily["pre_close"]).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ATR 折算到复权空间(×scale)与分子 adj-highest 同口径——未复权 ATR 与
    # 复权基座分子混除会造成单位/量级错配(2026-08-05 修复)
    atr = (tr * scale).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    strength = safe_divide(adj - highest, atr + 1e-10)
    return cross_sectional_rank(strength)


# ═══════════════════════════════════════════════════════════════════════════════
# ATR-Normalized Momentum Factors
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
    dependencies=("daily.parquet",),
)
def factor_atr_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]
    # ATR 折算到复权空间(×scale)后与复权基座 adj 同口径,避免除权日 close
    # 跳变造成相对波幅虚高(2026-08-05 修复)
    scale = _adjusted_close(daily) / close.replace(0, np.nan)
    adj = _adjusted_close(daily)

    tr1 = high - low
    tr2 = (high - daily["pre_close"]).abs()
    tr3 = (low - daily["pre_close"]).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = (tr * scale).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    atr_pct = safe_divide(atr, adj + 1e-10)
    return cross_sectional_rank(-atr_pct)  # low relative ATR = stable

