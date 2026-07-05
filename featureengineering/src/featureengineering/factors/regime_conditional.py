"""
Regime-conditional factor variants — Class 4 coupling factors.

Market regime significantly affects which factors work.  These factors condition
top-|IC| signals on:

  - Volatility regime: High (CSI300 RV > 80th pctl) vs Low (< 20th pctl)
  - Trend regime:    Bull (CSI300 > MA60) vs Bear (CSI300 < MA60)

Each base factor gets two conditional variants that are non-neutral only in
their respective regimes, allowing the model to learn different factor-response
functions per market state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — regime detection
# ═══════════════════════════════════════════════════════════════════════════════

_CSI300_CODE = "000300"


def _compute_market_rv(ctx: FactorContext) -> pd.Series:
    """Compute CSI300 20-day realised volatility."""
    daily = ctx.load("daily_adj.parquet")
    csi = daily[daily.index.get_level_values("Code") == _CSI300_CODE]
    if csi.empty:
        ret = daily["close"].groupby(level="Code").pct_change()
        mkt_ret = ret.groupby(level="Date").mean()
        rv = mkt_ret.rolling(20, min_periods=10).std()
        return rv
    close = csi["close"]
    ret = close.groupby(level="Code").pct_change()
    rv = ret.rolling(20, min_periods=10).std()
    return rv.groupby(level="Date").first()


def _get_vol_regime(ctx: FactorContext) -> pd.Series:
    """Return Series with vol regime: 2=high, 1=mid, 0=low."""
    mkt_rv = _compute_market_rv(ctx)
    high_thresh = mkt_rv.expanding(60).quantile(0.8)
    low_thresh = mkt_rv.expanding(60).quantile(0.2)
    high_thresh = high_thresh.fillna(mkt_rv.quantile(0.8))
    low_thresh = low_thresh.fillna(mkt_rv.quantile(0.2))

    regime = pd.Series(1, index=mkt_rv.index)
    regime[mkt_rv > high_thresh] = 2
    regime[mkt_rv < low_thresh] = 0
    return regime


def _get_trend_regime(ctx: FactorContext) -> pd.Series:
    """Return Series with trend regime: 1=bull, 0=bear."""
    daily = ctx.load("daily_adj.parquet")
    csi = daily[daily.index.get_level_values("Code") == _CSI300_CODE]
    if csi.empty:
        # Fallback: market average close
        mkt_close = daily["close"].groupby(level="Date").mean()
    else:
        mkt_close = csi["close"].groupby(level="Date").first()

    ma60 = mkt_close.rolling(60, min_periods=20).mean()
    return (mkt_close > ma60).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# Regime-Conditional Factors — Volatility Regime (highvol / lowvol)
# ═══════════════════════════════════════════════════════════════════════════════

_REGIME_BASE_FACTORS = [
    ("rv_5min",              "5min RV"),
    ("gk_vol",               "GK volatility"),
    ("parkinson_vol",        "Parkinson volatility"),
    ("volatility_20",        "20d volatility"),
    ("amihud_intraday",      "Amihud illiquidity"),
    ("turnover_20",          "20d turnover"),
    ("mom_20",               "20d momentum"),
    ("herding_intensity",    "herding intensity"),
    ("max_ret_20",           "20d max return"),
    ("idiosyncratic_vol_60", "60d idiosyncratic vol"),
]


def _make_vol_regime_factor(base_name, desc_en, regime_level, regime_label):
    """Create vol-regime-conditional factor."""
    regime_desc = "high" if regime_level == 2 else "low"

    thesis_high = (
        "In high-volatility regimes, volatility/fear dominate pricing: "
        "extreme vol spikes often signal panic selling (overreaction -> reversal). "
        "Separating high-vol regime allows the model to learn mean-reversion patterns."
    )
    thesis_low = (
        "In low-volatility regimes, volatility differences reflect genuine risk "
        "differences rather than panic: low-vol stocks earn a stability premium. "
        "Separating low-vol regime isolates the pure risk-pricing channel."
    )
    thesis = thesis_high if regime_level == 2 else thesis_low

    @register_factor(
        name=base_name + "_" + regime_label,
        description=(
            desc_en + " factor in " + regime_desc
            + "-volatility regime only (neutral otherwise), cross-sectional rank."
        ),
        category="coupling",
        thesis=thesis,
        dependencies=("__factors__", base_name, "daily_adj.parquet"),
    )
    def _compute(ctx: FactorContext) -> pd.Series:
        raw = ctx.load_factor(base_name)
        regime = _get_vol_regime(ctx)
        dates = raw.index.get_level_values("Date")
        regime_aligned = regime.reindex(dates)
        mask = regime_aligned.values == regime_level
        result = raw.copy()
        result.values[~mask] = np.nan
        return cross_sectional_rank(result)

    _compute.__name__ = "factor_" + base_name + "_" + regime_label
    _compute.__qualname__ = _compute.__name__
    return _compute


# Generate vol-regime variants
for _bn, _desc in _REGIME_BASE_FACTORS:
    _make_vol_regime_factor(_bn, _desc, 2, "highvol")
    _make_vol_regime_factor(_bn, _desc, 0, "lowvol")


# ═══════════════════════════════════════════════════════════════════════════════
# Regime-Conditional Factors — Trend Regime (bull / bear)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_trend_regime_factor(base_name, desc_en, is_bull):
    """Create trend-regime-conditional factor."""
    label = "bull" if is_bull else "bear"
    regime_desc = "bull" if is_bull else "bear"

    thesis_bull = (
        "In bull markets, momentum and growth factors dominate: trends "
        "self-reinforce and reversals are often brief pullbacks. "
        "Isolating bull regime lets the model weight trend-following signals."
    )
    thesis_bear = (
        "In bear markets, defensive and reversal factors dominate: low-vol, "
        "quality, and oversold-bounce signals gain strength. "
        "Isolating bear regime lets the model weight mean-reversion signals."
    )
    thesis = thesis_bull if is_bull else thesis_bear

    @register_factor(
        name=base_name + "_" + label,
        description=(
            desc_en + " factor in " + regime_desc
            + " market only (neutral otherwise), cross-sectional rank."
        ),
        category="coupling",
        thesis=thesis,
        dependencies=("__factors__", base_name, "daily_adj.parquet"),
    )
    def _compute(ctx: FactorContext) -> pd.Series:
        raw = ctx.load_factor(base_name)
        regime = _get_trend_regime(ctx)
        dates = raw.index.get_level_values("Date")
        regime_aligned = regime.reindex(dates)
        target = 1 if is_bull else 0
        mask = regime_aligned.values == target
        result = raw.copy()
        result.values[~mask] = np.nan
        return cross_sectional_rank(result)

    _compute.__name__ = "factor_" + base_name + "_" + label
    _compute.__qualname__ = _compute.__name__
    return _compute


# Generate trend-regime variants for top 6
for _bn, _desc in _REGIME_BASE_FACTORS[:6]:
    _make_trend_regime_factor(_bn, _desc, True)
    _make_trend_regime_factor(_bn, _desc, False)


