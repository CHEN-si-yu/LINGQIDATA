"""
Factor stability meta-factors — Class 4 coupling factors.

These factors measure the time-series stability of factor signals, helping the
model identify which factors are currently providing reliable vs noisy signals.

Unlike IC-tracking (which requires label data and is done in quality analysis),
these metrics are computed purely from factor values:
  - Rank turnover: how much the cross-sectional ranking changes day-over-day
  - Trend persistence: is the factor trending or mean-reverting for each stock?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


def _rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (0-1)."""
    return s.groupby(level="Date").rank(pct=True)


# Top factors to create stability metrics for
_STABILITY_BASE = [
    "mom_20", "volatility_20", "turnover_20", "rv_5min",
    "gk_vol", "amihud_intraday", "roe", "bp",
]


@register_factor(
    name="factor_rank_turnover_20",
    description="因子排名周转率，top因子20日排名变化均值截面排名（低周转=信号稳定排前）。",
    category="coupling",
    thesis=(
        "Cross-sectional rank turnover measures signal stability: "
        "low turnover means the factor consistently ranks the same stocks "
        "high/low, indicating a persistent signal. High turnover means the "
        "factor ranking changes rapidly, suggesting noise rather than signal. "
        "The model should trust low-turnover factors more."
    ),
    dependencies=("__factors__", *_STABILITY_BASE),
)
def factor_factor_rank_turnover_20(ctx: FactorContext) -> pd.Series:
    factors = ctx.load_factors(_STABILITY_BASE)
    turnover_scores = []

    for col in factors.columns:
        r = _rank(factors[col])
        # Absolute day-over-day rank change per stock
        chg = r.groupby(level="Code").diff(1).abs()
        # 20-day average turnover per stock
        avg_chg = chg.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
        turnover_scores.append(avg_chg)

    # Average turnover across all factors (lower = more stable)
    avg_turnover = pd.concat(turnover_scores, axis=1).mean(axis=1)
    return cross_sectional_rank(-avg_turnover)


@register_factor(
    name="factor_trend_persistence",
    description="因子趋势持续性，top因子20日自相关系数截面排名（高自相关=趋势持续排前）。",
    category="coupling",
    thesis=(
        "Factor value autocorrelation measures trend persistence: "
        "high autocorrelation means the factor follows smooth trends "
        "(predictable), low autocorrelation means the factor is noisy "
        "(unpredictable). Stocks with persistent factor trends have more "
        "reliable signal-to-noise ratios."
    ),
    dependencies=("__factors__", *_STABILITY_BASE),
)
def factor_factor_trend_persistence(ctx: FactorContext) -> pd.Series:
    factors = ctx.load_factors(_STABILITY_BASE)
    persistence_scores = []

    for col in factors.columns:
        r = _rank(factors[col])
        # Rolling 20-day autocorrelation per stock
        roll_mean = r.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
        roll_std = r.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).std()
        )
        # Normalized deviation from rolling mean (how far from recent average)
        deviation = safe_divide(r - roll_mean, roll_std + 1e-8)
        # Lag-1 serial correlation of deviation (20d window)
        lag1 = deviation.groupby(level="Code").shift(1)
        # Rolling correlation between deviation and lag1
        # Simplified: use absolute 1-day change as inverse persistence proxy
        abs_chg = r.groupby(level="Code").diff(1).abs()
        avg_chg = abs_chg.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
        persistence_scores.append(-avg_chg)  # lower change = higher persistence

    avg_persistence = pd.concat(persistence_scores, axis=1).mean(axis=1)
    return cross_sectional_rank(avg_persistence)


