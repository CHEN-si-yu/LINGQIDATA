"""
Minute-level technical indicator factors — Class 4 per-stock factors.

These factors aggregate pre-computed 1-minute technical indicators from the
``indicator_1min/`` per-stock directory into daily cross-sectional signals.

Indicator schema (per stock parquet):
  trade_time, stock_code,
  MACD: dif, dea, macd
  KDJ:  k, d, j
  RSI:  rsi
  Bollinger: boll_mid, boll_upper, boll_lower
  MAs:  ma5, ma10, ma20, ma30, ma60
  Vol MAs: mavol5, mavol10

Architecture follows the unified builder pattern (mirroring intraday.py):
  1. One pass per stock file → all 70+ daily metrics computed in memory
  2. Stack into panel format → factor compute functions extract their metric
  3. Cross-sectional rank within each date

Build Class: Class 4 (per-stock directory, following indicator_1min pattern).
Dependency marker: "indicator_1min".
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_code(code: str) -> str:
    """Normalise a stock code: strip exchange suffix, zero-pad to 6 digits."""
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _make_multiindex_series(
    values: np.ndarray, dates: pd.Index, code: str, name: str = ""
) -> pd.Series:
    """Build a (Date, Code) MultiIndex Series from per-stock values.

    Uses ``from_arrays`` — exactly matching the Class 3 (intraday.py) pattern.
    """
    idx = pd.MultiIndex.from_arrays(
        [dates, [code] * len(dates)], names=["Date", "Code"]
    )
    return pd.Series(values, index=idx, name=name)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Metric Computation  (one stock → all daily indicator metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def _indicator_all_metrics(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Compute **all** daily indicator metrics for one stock in a single pass.

    Returns a DataFrame indexed by date (YYYYMMDD string) with columns for
    every metric needed by the registered indicator factors (~70+ columns).
    """
    df = stock_df.copy()

    # ── Ensure trade_time is datetime ────────────────────────────────────
    ts = df["trade_time"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts)
    df["trade_date"] = ts.dt.strftime("%Y%m%d")
    df["minute"] = ts.dt.hour * 60 + ts.dt.minute

    grouped = df.groupby("trade_date")
    results: dict[str, pd.Series] = {}

    # ═════════════════════════════════════════════════════════════════════
    # Section A — MACD family  (10 metrics)
    # ═════════════════════════════════════════════════════════════════════

    # End-of-day snapshots
    results["dif_open"] = grouped["dif"].first()
    results["dif_close"] = grouped["dif"].last()
    results["dea_close"] = grouped["dea"].last()
    results["macd_open"] = grouped["macd"].first()
    results["macd_close"] = grouped["macd"].last()

    # Daily statistics
    results["dif_mean"] = grouped["dif"].mean()
    results["macd_std"] = grouped["macd"].std()
    results["macd_max"] = grouped["macd"].max()
    results["macd_min"] = grouped["macd"].min()

    # MACD daily range
    results["macd_range"] = results["macd_max"] - results["macd_min"]

    # MACD trend strength: |dif - dea| / |dea| at close
    dif_c = results["dif_close"]
    dea_c = results["dea_close"]
    results["macd_trend_strength"] = safe_divide(
        (dif_c - dea_c).abs(), dea_c.abs()
    )

    # MACD consistency: fraction of minutes where macd_bar > 0
    df["macd_positive"] = (df["macd"] > 0).astype(float)
    results["macd_consistency"] = grouped["macd_positive"].mean()

    # MACD divergence: intraday correlation of MA5 trend vs MACD trend
    df["ma5_vs_open"] = (
        df["ma5"] / df.groupby("trade_date")["ma5"].transform("first") - 1.0
    )
    df["macd_vs_open"] = (
        df["macd"] - df.groupby("trade_date")["macd"].transform("first")
    )

    def _pearson(x, y):
        """Safe Pearson correlation, returns NaN if insufficient variation."""
        if len(x) < 5:
            return np.nan
        if x.std() == 0 or y.std() == 0:
            return np.nan
        with np.errstate(invalid='ignore'):
            return x.corr(y)

    results["macd_price_corr"] = grouped.apply(
        lambda g: _pearson(g["ma5_vs_open"], g["macd_vs_open"])
    )

    # MACD DIF intraday slope: (dif_close - dif_open) / |dif_open|
    dif_o = results["dif_open"]
    results["macd_dif_slope"] = safe_divide(dif_c - dif_o, dif_o.abs())

    # MACD bar acceleration: (macd_close - macd_open) / |macd_open|
    macd_o = results["macd_open"]
    results["macd_acceleration"] = safe_divide(
        results["macd_close"] - macd_o, macd_o.abs()
    )

    # MACD zero cross: whether DIF crossed zero within the day
    df["dif_sign"] = np.sign(df["dif"])
    df["dif_sign_change"] = (
        df.groupby("trade_date")["dif_sign"]
        .transform(lambda s: s.diff().abs())
        .fillna(0)
    )
    results["macd_zero_cross"] = grouped["dif_sign_change"].sum() / 2.0

    # MACD bar sign change count: # of times macd bar flips sign
    df["macd_sign"] = np.sign(df["macd"])
    df["macd_sign_change"] = (
        df.groupby("trade_date")["macd_sign"]
        .transform(lambda s: s.diff().abs())
        .fillna(0)
    )
    results["macd_bar_sign_change"] = grouped["macd_sign_change"].sum() / 2.0

    # MACD extreme ratio: fraction of minutes with |macd| > 2 * std(macd)
    macd_std_per_stock = grouped["macd"].transform("std")
    df["macd_extreme"] = (
        (df["macd"].abs() > 2.0 * macd_std_per_stock.replace(0, np.nan))
        .astype(float)
    )
    results["macd_extreme_ratio"] = grouped["macd_extreme"].mean()

    # MACD signal raw: sign(macd_close) * sign(dif_close - dea_close)
    results["macd_signal_raw"] = (
        np.sign(results["macd_close"].fillna(0))
        * np.sign((dif_c - dea_c).fillna(0))
    )

    # ═════════════════════════════════════════════════════════════════════
    # Section B — KDJ family  (12 metrics)
    # ═════════════════════════════════════════════════════════════════════

    results["k_open"] = grouped["k"].first()
    results["k_close"] = grouped["k"].last()
    results["d_open"] = grouped["d"].first()
    results["d_close"] = grouped["d"].last()
    results["j_open"] = grouped["j"].first()
    results["j_close"] = grouped["j"].last()
    results["k_mean"] = grouped["k"].mean()
    results["j_max"] = grouped["j"].max()
    results["j_min"] = grouped["j"].min()

    # K-D distance at close: (K - D) / D
    k_c = results["k_close"]
    d_c = results["d_close"]
    results["k_d_distance"] = safe_divide(k_c - d_c, d_c.abs() + 1e-10)

    # KDJ golden cross: K crosses above D count
    df["k_above_d"] = (df["k"] > df["d"]).astype(int)
    df["k_cross_up"] = (
        df.groupby("trade_date")["k_above_d"]
        .transform(lambda s: s.diff().clip(lower=0))
    )
    results["kdj_cross_count"] = grouped["k_cross_up"].sum()

    # KDJ dead cross: K crosses below D count
    df["k_below_d"] = (df["k"] < df["d"]).astype(int)
    df["k_cross_down"] = (
        df.groupby("trade_date")["k_below_d"]
        .transform(lambda s: s.diff().clip(lower=0))
    )
    results["kdj_dead_cross_count"] = grouped["k_cross_down"].sum()

    # J reversal risk: |j - 50| — extreme J = reversal probability
    results["j_reversal_risk"] = (results["j_close"] - 50.0).abs()

    # J range and volatility
    results["j_range"] = results["j_max"] - results["j_min"]
    results["j_volatility"] = grouped["j"].std()

    # D-line stability: std(D) / mean(D)  (coefficient of variation)
    d_std = grouped["d"].std()
    d_mean = grouped["d"].mean()
    results["d_stability"] = safe_divide(d_std, d_mean)

    # KDJ overbought / oversold time fractions
    df["kdj_overbought"] = (df["k"] > 80).astype(float)
    df["kdj_oversold"] = (df["k"] < 20).astype(float)
    results["kdj_overbought_frac"] = grouped["kdj_overbought"].mean()
    results["kdj_oversold_frac"] = grouped["kdj_oversold"].mean()

    # KDJ bull fraction: K > D time ratio
    results["kdj_bull_frac"] = grouped["k_above_d"].mean()

    # KDJ bull cross net: gold crosses - dead crosses
    results["kdj_cross_net"] = (
        results["kdj_cross_count"] - results["kdj_dead_cross_count"]
    )

    # ═════════════════════════════════════════════════════════════════════
    # Section C — RSI family  (10 metrics)
    # ═════════════════════════════════════════════════════════════════════

    results["rsi_open"] = grouped["rsi"].first()
    results["rsi_close"] = grouped["rsi"].last()
    results["rsi_mean"] = grouped["rsi"].mean()
    results["rsi_std"] = grouped["rsi"].std()
    results["rsi_max"] = grouped["rsi"].max()
    results["rsi_min"] = grouped["rsi"].min()

    # RSI range
    results["rsi_range"] = results["rsi_max"] - results["rsi_min"]

    # RSI extreme time fraction (>70 or <30)
    df["rsi_extreme"] = ((df["rsi"] > 70) | (df["rsi"] < 30)).astype(float)
    results["rsi_extreme_frac"] = grouped["rsi_extreme"].mean()

    # RSI overbought / oversold fractions
    df["rsi_overbought"] = (df["rsi"] > 70).astype(float)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(float)
    results["rsi_overbought_frac"] = grouped["rsi_overbought"].mean()
    results["rsi_oversold_frac"] = grouped["rsi_oversold"].mean()

    # RSI intraday trend: correlation of RSI with minute position
    def _time_corr(series):
        if len(series) < 5 or series.std() == 0:
            return np.nan
        with np.errstate(invalid='ignore'):
            return series.corr(pd.Series(range(len(series)), index=series.index))

    results["rsi_time_corr"] = grouped["rsi"].apply(_time_corr)

    # RSI volatility (coefficient of variation within day)
    rsi_cv = safe_divide(results["rsi_std"], results["rsi_mean"])
    results["rsi_volatility"] = rsi_cv

    # RSI-50 deviation at close
    results["rsi_excess"] = results["rsi_close"] - 50.0

    # ═════════════════════════════════════════════════════════════════════
    # Section D — Bollinger Bands family  (12 metrics)
    # ═════════════════════════════════════════════════════════════════════

    results["boll_mid_open"] = grouped["boll_mid"].first()
    results["boll_mid_close"] = grouped["boll_mid"].last()
    results["boll_upper_close"] = grouped["boll_upper"].last()
    results["boll_lower_close"] = grouped["boll_lower"].last()

    # Daily close proxy: last intraday ma5 value
    close_proxy = grouped["ma5"].last()
    results["close"] = close_proxy

    # Bollinger position: (close - mid) / (upper - lower)
    results["boll_position"] = safe_divide(
        close_proxy - results["boll_mid_close"],
        results["boll_upper_close"] - results["boll_lower_close"],
    )

    # Bollinger width: (upper - lower) / mid
    results["boll_width"] = safe_divide(
        results["boll_upper_close"] - results["boll_lower_close"],
        results["boll_mid_close"],
    )

    # Bollinger squeeze: 1 / width  (narrow band = high squeeze)
    results["boll_squeeze"] = safe_divide(
        1.0, results["boll_width"] + 1e-10
    )

    # Bollinger touch fractions
    df["above_upper"] = (df["ma5"] > df["boll_upper"]).astype(float)
    df["below_lower"] = (df["ma5"] < df["boll_lower"]).astype(float)
    results["boll_touch_upper"] = grouped["above_upper"].mean()
    results["boll_touch_lower"] = grouped["below_lower"].mean()
    results["boll_touch_net"] = (
        results["boll_touch_upper"] - results["boll_touch_lower"]
    )

    # Bollinger bandwalk: max consecutive minutes outside bands
    def _max_consecutive(series):
        """Max consecutive True values in a boolean series."""
        if len(series) == 0:
            return 0
        # Identify run boundaries
        groups = (series != series.shift()).cumsum()
        runs = series.groupby(groups).sum()
        return runs.max() if len(runs) > 0 else 0

    results["boll_bandwalk_upper"] = grouped["above_upper"].apply(_max_consecutive)
    results["boll_bandwalk_lower"] = grouped["below_lower"].apply(_max_consecutive)

    # Bollinger mid slope: intraday trend of mid band
    boll_mid_o = results["boll_mid_open"]
    results["boll_mid_slope"] = safe_divide(
        results["boll_mid_close"] - boll_mid_o, boll_mid_o.abs()
    )

    # Bollinger width change: expansion / contraction indicator
    boll_width_open = safe_divide(
        grouped["boll_upper"].first() - grouped["boll_lower"].first(),
        boll_mid_o.abs(),
    )
    results["boll_width_change"] = results["boll_width"] - boll_width_open

    # Bollinger bandwidth relative to historical intraday
    band_mean = (results["boll_upper_close"] + results["boll_lower_close"]) / 2.0
    results["boll_band_deviation"] = safe_divide(
        close_proxy - band_mean,
        results["boll_upper_close"] - results["boll_lower_close"],
    )

    # ═════════════════════════════════════════════════════════════════════
    # Section E — Moving Average family  (18 metrics)
    # ═════════════════════════════════════════════════════════════════════

    results["ma5_open"] = grouped["ma5"].first()
    results["ma5_close"] = grouped["ma5"].last()
    results["ma10_close"] = grouped["ma10"].last()
    results["ma20_close"] = grouped["ma20"].last()
    results["ma30_close"] = grouped["ma30"].last()
    results["ma60_close"] = grouped["ma60"].last()

    # MA alignment score: count of bull-aligned MA pairs (0-4)
    ma5 = results["ma5_close"]
    ma10 = results["ma10_close"]
    ma20 = results["ma20_close"]
    ma30 = results["ma30_close"]
    ma60 = results["ma60_close"]
    results["ma_alignment"] = (
        (ma5 > ma10).astype(int)
        + (ma10 > ma20).astype(int)
        + (ma20 > ma30).astype(int)
        + (ma30 > ma60).astype(int)
    )

    # MA dispersion: std(all MAs) / mean(all MAs)
    ma_cols = ["ma5_close", "ma10_close", "ma20_close", "ma30_close", "ma60_close"]
    ma_df = pd.DataFrame({k: results[k] for k in ma_cols})
    results["ma_dispersion"] = safe_divide(
        ma_df.std(axis=1), ma_df.mean(axis=1).abs()
    )

    # MA slopes (intraday trend)
    ma5_o = results["ma5_open"]
    results["ma5_slope"] = safe_divide(ma5 - ma5_o, ma5_o.abs())
    ma20_o = grouped["ma20"].first()
    results["ma20_slope"] = safe_divide(
        ma20 - ma20_o, ma20_o.abs()
    )

    # Price vs MA deviations (normalized)
    results["price_vs_ma5"] = safe_divide(close_proxy - ma5, ma5.abs())
    results["price_vs_ma10"] = safe_divide(close_proxy - ma10, ma10.abs())
    results["price_vs_ma20"] = safe_divide(close_proxy - ma20, ma20.abs())
    results["price_vs_ma30"] = safe_divide(close_proxy - ma30, ma30.abs())
    results["price_vs_ma60"] = safe_divide(close_proxy - ma60, ma60.abs())

    # MA convergence: |ma5 - ma60| / ma60  (short-long spread)
    results["ma_convergence"] = safe_divide(
        (ma5 - ma60).abs(), ma60.abs()
    )

    # MA cross count: number of MA cross events within day
    df["ma5_above_ma20"] = (df["ma5"] > df["ma20"]).astype(int)
    df["ma5_ma20_cross"] = (
        df.groupby("trade_date")["ma5_above_ma20"]
        .transform(lambda s: s.diff().abs())
        .fillna(0)
    )
    results["ma_cross_count"] = grouped["ma5_ma20_cross"].sum() / 2.0

    # MA bull/bear ratio
    ma_pairs_bull = (
        (ma5 > ma10).astype(int)
        + (ma5 > ma20).astype(int)
        + (ma5 > ma60).astype(int)
        + (ma10 > ma20).astype(int)
        + (ma10 > ma60).astype(int)
        + (ma20 > ma60).astype(int)
    )
    results["ma_bull_bear_ratio"] = ma_pairs_bull / 6.0

    # MA curvature: ma5_slope - ma20_slope  (acceleration proxy)
    results["ma_curvature"] = (
        results["ma5_slope"].fillna(0) - results["ma20_slope"].fillna(0)
    )

    # MA 10/30 closing values for extended use
    results["ma10_open"] = grouped["ma10"].first()
    results["ma30_open"] = grouped["ma30"].first()

    # ═════════════════════════════════════════════════════════════════════
    # Section F — Volume MA family  (8 metrics)
    # ═════════════════════════════════════════════════════════════════════

    results["mavol5_open"] = grouped["mavol5"].first()
    results["mavol5_close"] = grouped["mavol5"].last()
    results["mavol10_close"] = grouped["mavol10"].last()

    # Volume MA ratio: mavol5 / mavol10
    mavol5 = results["mavol5_close"]
    mavol10 = results["mavol10_close"]
    results["mavol_ratio"] = safe_divide(mavol5, mavol10)

    # Volume MA slope
    mavol5_o = results["mavol5_open"]
    results["mavol5_slope"] = safe_divide(mavol5 - mavol5_o, mavol5_o.abs())

    # Volume expansion: (mavol5_close / mavol5_open) - 1
    results["mavol_expansion"] = safe_divide(mavol5, mavol5_o) - 1.0

    # Volume MA ratio stability (within-day std)
    df["mavol_ratio_intra"] = safe_divide(df["mavol5"], df["mavol10"])
    results["mavol_ratio_std"] = (
        df.groupby("trade_date")["mavol_ratio_intra"].std()
    )

    # Volume MA ratio trend: correlation with minute position
    results["mavol_ratio_trend"] = (
        df.groupby("trade_date")["mavol_ratio_intra"].apply(_time_corr)
    )

    # Volume MA stability: std(mavol5) / mean(mavol5)
    mavol5_std = grouped["mavol5"].std()
    mavol5_mean = grouped["mavol5"].mean()
    results["mavol_stability"] = safe_divide(mavol5_std, mavol5_mean)

    # Volume-price confirmation: mavol5_slope sign == ma5_slope sign
    results["volume_price_confirmation"] = (
        (np.sign(results["mavol5_slope"].fillna(0))
         == np.sign(results["ma5_slope"].fillna(0)))
        .astype(float)
    )

    # ═════════════════════════════════════════════════════════════════════
    # Section G — Cross-indicator composites  (10 metrics)
    # ═════════════════════════════════════════════════════════════════════

    # MACD-KDJ alignment: sign(macd_close) * sign(k - d)
    results["macd_kdj_alignment"] = (
        np.sign(results["macd_close"].fillna(0))
        * np.sign((k_c - d_c).fillna(0))
    )

    # RSI-Bollinger combo: rsi_zscore * boll_position
    rsi_z = safe_divide(
        results["rsi_close"] - 50.0, results["rsi_std"].fillna(1.0)
    )
    results["rsi_boll_combo"] = rsi_z * results["boll_position"].fillna(0)

    # MA-RSI divergence: correlation of ma5 and rsi within day
    results["ma_rsi_divergence"] = grouped.apply(
        lambda g: _pearson(g["ma5"], g["rsi"])
    )

    # Indicator consensus: fraction of 6 indicators pointing same direction
    ind_up = (
        (np.sign(results["macd_close"].fillna(0)) > 0).astype(int)
        + (results["k_close"].fillna(50) > results["d_close"].fillna(50)).astype(int)
        + (results["rsi_close"].fillna(50) > 50).astype(int)
        + (results["boll_position"].fillna(0) > 0).astype(int)
        + (results["ma_alignment"].fillna(2) >= 3).astype(int)
        + (results["mavol_ratio"].fillna(1) > 1).astype(int)
    )
    results["indicator_consensus"] = ind_up / 6.0

    # Trend confirmation: ma_alignment * sign(macd_close)
    results["trend_confirmation"] = (
        results["ma_alignment"].fillna(2).astype(float)
        * np.sign(results["macd_close"].fillna(0))
    )

    # KDJ-Bollinger combo: (j - 50) / 50 + boll_position
    results["kdj_boll_combo"] = (
        safe_divide(results["j_close"] - 50.0, 50.0)
        + results["boll_position"].fillna(0)
    )

    # MACD-RSI combo: sign(macd_close) * (rsi - 50)
    results["macd_rsi_combo"] = (
        np.sign(results["macd_close"].fillna(0))
        * (results["rsi_close"].fillna(50) - 50.0)
    )

    # Indicator dispersion: std of indicator signals
    signal_cols = [
        "macd_signal_raw", "k_d_distance", "rsi_excess",
        "boll_position", "ma_alignment",
    ]
    # Build a temporary DataFrame of signals
    signals = {}
    for c in signal_cols:
        s = results.get(c, pd.Series(0.0, index=close_proxy.index))
        # Normalize to roughly [-1, 1] range
        if s.std() > 0:
            signals[c] = (s - s.mean()) / s.std()
        else:
            signals[c] = s.fillna(0)
    sig_df = pd.DataFrame(signals, index=close_proxy.index)
    results["indicator_dispersion"] = sig_df.std(axis=1)

    # Multi-indicator extreme: avg of z-scores
    results["multi_indicator_extreme"] = sig_df.abs().mean(axis=1)

    # ═════════════════════════════════════════════════════════════════════
    # Section H — Session-based metrics  (6 metrics)
    # ═════════════════════════════════════════════════════════════════════

    morning = df[(df["minute"] >= 570) & (df["minute"] <= 690)]
    afternoon = df[(df["minute"] >= 780) & (df["minute"] <= 900)]

    # Morning session indicator trends
    if not morning.empty:
        mg = morning.groupby("trade_date")
        results["am_macd_close"] = mg["macd"].last()
        results["am_rsi_close"] = mg["rsi"].last()
        results["am_macd_mean"] = mg["macd"].mean()
        results["am_rsi_mean"] = mg["rsi"].mean()
        # Morning MACD trend: correlation of macd with minute
        results["am_macd_trend"] = mg["macd"].apply(_time_corr)
        results["am_rsi_trend"] = mg["rsi"].apply(_time_corr)
    else:
        for k in ["am_macd_close", "am_rsi_close", "am_macd_mean",
                   "am_rsi_mean", "am_macd_trend", "am_rsi_trend"]:
            results[k] = pd.Series(np.nan, index=close_proxy.index)

    # Afternoon session indicator trends
    if not afternoon.empty:
        ag = afternoon.groupby("trade_date")
        results["pm_macd_close"] = ag["macd"].last()
        results["pm_rsi_close"] = ag["rsi"].last()
        results["pm_macd_mean"] = ag["macd"].mean()
        results["pm_rsi_mean"] = ag["rsi"].mean()
        results["pm_macd_trend"] = ag["macd"].apply(_time_corr)
        results["pm_rsi_trend"] = ag["rsi"].apply(_time_corr)
    else:
        for k in ["pm_macd_close", "pm_rsi_close", "pm_macd_mean",
                   "pm_rsi_mean", "pm_macd_trend", "pm_rsi_trend"]:
            results[k] = pd.Series(np.nan, index=close_proxy.index)

    # AM/PM MACD ratio
    results["am_pm_macd_ratio"] = safe_divide(
        results.get("am_macd_mean", pd.Series(np.nan, index=close_proxy.index)),
        results.get("pm_macd_mean", pd.Series(np.nan, index=close_proxy.index)),
    )

    # AM/PM RSI ratio
    results["am_pm_rsi_ratio"] = safe_divide(
        results.get("am_rsi_mean", pd.Series(np.nan, index=close_proxy.index)),
        results.get("pm_rsi_mean", pd.Series(np.nan, index=close_proxy.index)),
    )

    # ═════════════════════════════════════════════════════════════════════
    # Return the full metrics DataFrame
    # ═════════════════════════════════════════════════════════════════════

    return pd.DataFrame(results, index=close_proxy.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Factor → Metric Mapping
# ═══════════════════════════════════════════════════════════════════════════════

#: Map factor_name → (metric_column, direction)
#: direction: "pos"=rank(raw), "neg"=rank(-raw), "neg_abs"=rank(-raw.abs())
INDICATOR_FACTOR_SPEC: dict[str, tuple[str, str]] = {
    # ── MACD family ─────────────────────────────────────────────────────
    "macd_signal_cross":           ("macd_signal_raw",      "pos"),
    "macd_trend_strength":         ("macd_trend_strength",  "pos"),
    "macd_daily_consistency":      ("macd_consistency",     "pos"),
    "macd_price_divergence":       ("macd_price_corr",      "neg"),
    "macd_dif_slope":              ("macd_dif_slope",       "pos"),
    "macd_acceleration":           ("macd_acceleration",    "pos"),
    "macd_zero_cross":             ("macd_zero_cross",      "neg"),
    "macd_bar_sign_change":        ("macd_bar_sign_change", "neg"),
    "macd_extreme_ratio":          ("macd_extreme_ratio",   "neg"),
    "macd_daily_range":            ("macd_range",           "neg"),
    # ── KDJ family ──────────────────────────────────────────────────────
    "kdj_j_value_close":           ("j_close",              "neg"),
    "kdj_k_d_distance":            ("k_d_distance",         "pos"),
    "kdj_cross_signal":            ("kdj_cross_count",      "pos"),
    "kdj_j_reversal_risk":         ("j_reversal_risk",      "neg"),
    "kdj_j_range":                 ("j_range",              "neg"),
    "kdj_j_volatility":            ("j_volatility",         "neg"),
    "kdj_d_stability":             ("d_stability",          "neg"),
    "kdj_overbought_frac":         ("kdj_overbought_frac",  "neg"),
    "kdj_oversold_frac":           ("kdj_oversold_frac",    "pos"),
    "kdj_bull_frac":               ("kdj_bull_frac",        "pos"),
    "kdj_cross_net":               ("kdj_cross_net",        "pos"),
    # ── RSI family ──────────────────────────────────────────────────────
    "rsi_14_excess":               ("rsi_excess",           "neg"),
    "rsi_extreme_fraction":        ("rsi_extreme_frac",     "pos"),
    "rsi_intraday_trend":          ("rsi_time_corr",        "pos"),
    "rsi_range":                   ("rsi_range",            "neg"),
    "rsi_volatility":              ("rsi_volatility",       "neg"),
    "rsi_overbought_frac":         ("rsi_overbought_frac",  "neg"),
    "rsi_oversold_frac":           ("rsi_oversold_frac",    "pos"),
    # ── Bollinger family ────────────────────────────────────────────────
    "boll_position":               ("boll_position",        "pos"),
    "boll_width_20":               ("boll_width",           "neg"),
    "boll_touch_ratio":            ("boll_touch_net",       "pos"),
    "boll_squeeze":                ("boll_squeeze",         "neg"),
    "boll_bandwalk_upper":         ("boll_bandwalk_upper",  "neg"),
    "boll_bandwalk_lower":         ("boll_bandwalk_lower",  "pos"),
    "boll_mid_slope":              ("boll_mid_slope",       "pos"),
    "boll_width_change":           ("boll_width_change",    "pos"),
    "boll_band_deviation":         ("boll_band_deviation",  "neg_abs"),
    # ── MA family ───────────────────────────────────────────────────────
    "ma_alignment_score":          ("ma_alignment",         "pos"),
    "ma_dispersion":               ("ma_dispersion",        "neg"),
    "price_vs_ma5_deviation":      ("price_vs_ma5",         "neg_abs"),
    "price_vs_ma20_deviation":     ("price_vs_ma20",        "neg_abs"),
    "price_vs_ma60_deviation":     ("price_vs_ma60",        "neg_abs"),
    "ma5_slope":                   ("ma5_slope",            "pos"),
    "ma20_slope":                  ("ma20_slope",           "pos"),
    "ma_convergence":              ("ma_convergence",       "neg"),
    "ma_cross_count":              ("ma_cross_count",       "pos"),
    "ma_bull_bear_ratio":          ("ma_bull_bear_ratio",   "pos"),
    "ma_curvature":                ("ma_curvature",         "pos"),
    # ── Volume MA family ────────────────────────────────────────────────
    "mavol_ratio_signal":          ("mavol_ratio",          "pos"),
    "mavol5_slope":                ("mavol5_slope",         "pos"),
    "mavol_expansion":             ("mavol_expansion",      "pos"),
    "mavol_ratio_std":             ("mavol_ratio_std",      "neg"),
    "mavol_ratio_trend":           ("mavol_ratio_trend",    "pos"),
    "mavol_stability":             ("mavol_stability",      "neg"),
    "volume_price_confirmation":   ("volume_price_confirmation", "pos"),
    # ── Cross-indicator composites ──────────────────────────────────────
    "macd_kdj_alignment":          ("macd_kdj_alignment",   "pos"),
    "rsi_boll_combo":              ("rsi_boll_combo",       "pos"),
    "ma_rsi_divergence":           ("ma_rsi_divergence",    "neg"),
    "indicator_consensus":         ("indicator_consensus",  "pos"),
    "trend_confirmation":          ("trend_confirmation",   "pos"),
    "kdj_boll_combo":              ("kdj_boll_combo",       "pos"),
    "macd_rsi_combo":              ("macd_rsi_combo",       "pos"),
    "indicator_dispersion":        ("indicator_dispersion", "neg"),
    "multi_indicator_extreme":     ("multi_indicator_extreme", "neg"),
    # ── Session-based ───────────────────────────────────────────────────
    "am_macd_trend":               ("am_macd_trend",        "pos"),
    "pm_macd_trend":               ("pm_macd_trend",        "pos"),
    "am_rsi_trend":                ("am_rsi_trend",         "pos"),
    "pm_rsi_trend":                ("pm_rsi_trend",         "pos"),
    "am_pm_macd_ratio":            ("am_pm_macd_ratio",     "pos"),
    "am_pm_rsi_ratio":             ("am_pm_rsi_ratio",      "pos"),
    # ── New: additional transforms of existing metrics (10) ──
    "macd_signal_momentum":         ("macd_signal_raw",      "momentum"),
    "macd_trend_momentum_rev":      ("macd_trend_strength",  "momentum_rev"),
    "kdj_j_momentum":               ("j_close",              "momentum_rev"),
    "kdj_cross_net_momentum":       ("kdj_cross_net",        "momentum"),
    "rsi_trend_momentum":           ("rsi_time_corr",        "momentum"),
    "boll_position_momentum":       ("boll_position",        "momentum"),
    "boll_squeeze_pos":             ("boll_squeeze",         "pos"),
    "ma_alignment_momentum":        ("ma_alignment",         "momentum"),
    "mavol_ratio_momentum":         ("mavol_ratio",          "momentum"),
    "indicator_consensus_momentum": ("indicator_consensus",  "momentum"),
}

#: All metric columns produced by ``_indicator_all_metrics`` that map to factors.
_INDICATOR_METRIC_COLS: set[str] = {m for m, _ in INDICATOR_FACTOR_SPEC.values()}


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Worker  (runs in a ProcessPoolExecutor worker)
# ═══════════════════════════════════════════════════════════════════════════════

def _process_indicator_batch(
    batch: list[tuple[str, str]],
    min_trade_time: str | None = None,
) -> dict[str, list[pd.Series]]:
    """Process a batch of indicator_1min stock files in a worker process.

    Args:
        batch: list of ``(code, filepath_str)`` tuples.
        min_trade_time: if set, filter rows to ``trade_time >= min_trade_time``
            before computing metrics (YYYY-MM-DD HH:MM:SS format).

    Returns:
        dict mapping metric_col → list of pd.Series (one per stock).
    """
    accum: dict[str, list[pd.Series]] = defaultdict(list)
    for code, fpath_str in batch:
        try:
            stock_df = pd.read_parquet(fpath_str)
        except Exception:
            continue
        if stock_df.empty:
            continue
        if min_trade_time is not None:
            stock_df = stock_df[stock_df["trade_time"] >= min_trade_time]
            if stock_df.empty:
                continue
        try:
            daily = _indicator_all_metrics(stock_df)
        except Exception:
            continue
        for col in _INDICATOR_METRIC_COLS:
            if col not in daily.columns:
                continue
            s = daily[col].dropna()
            if s.empty:
                continue
            accum[col].append(
                _make_multiindex_series(s.values, s.index, code, col)
            )
    return dict(accum)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Builder  (4-phase, single pass over indicator_1min/)
# ═══════════════════════════════════════════════════════════════════════════════

def build_indicator_1min_new(
    factor_names: list[str],
    paths,                      # ProjectPaths
    force: bool = False,
    max_workers: int | None = None,
    quality_check_days: int = 0,
) -> list:
    """Build all Class 4 indicator_1min factors in a single pass.

    Each per-stock indicator parquet file is opened **once**, all 70+ raw
    metrics are computed together, then fanned out to every requested factor.
    Multi-stock batching runs in parallel via ProcessPoolExecutor.

    Phases:
      1. File discovery & validation
      2. Parallel metric computation (stocks batched across workers)
      3. Factor frame assembly
      4. Writing .fea output files

    Parameters
    ----------
    quality_check_days:
        If > 0, inspect the last N trading days of existing .fea files.
        Factors whose recent rows are all-NaN will be force-rebuilt.

    Returns a list of ``BuildResult`` objects compatible with CLI expectations.
    """
    from ..builder import (
        BuildResult,
        _resolve_effective_end_date,
        _read_factor_max_date,
        get_factor,
    )
    from ..dataset import _load_allowed_codes
    from ..storage import (
        ensure_single_factor_frame,
        write_factor,
        write_factor_incremental,
    )

    allowed = _load_allowed_codes(paths.stock_pool_file)
    ind_dir = Path(str(paths.source_root)) / "indicator_1min"

    if not ind_dir.exists():
        raise FileNotFoundError(f"indicator_1min data directory not found: {ind_dir}")

    # ── Phase 1 header ───────────────────────────────────────────────────
    total_factors = len(factor_names)
    n_workers = min(max_workers or 32, (os.cpu_count() or 4))

    print(f"\n{'='*64}")
    print(f"  CLASS 4 — UNIFIED SINGLE-PASS BUILDER  (indicator_1min)")
    print(f"  Factors: {total_factors}   Workers: {n_workers}")
    print(f"  Strategy: open each stock file once, compute all metrics together")
    print(f"{'='*64}")

    # ── Phase 1: File discovery ─────────────────────────────────────────
    t_phase1 = time.perf_counter()
    print(f"\n  Phase 1/4 — File discovery")

    all_files = sorted(
        [f for f in os.listdir(ind_dir) if f.endswith(".parquet")]
    )
    file_map: dict[str, str] = {}  # code → filepath
    for fname in all_files:
        # Handle naming: 000001.SZ.parquet or 000001.parquet
        code = _pad_code(os.path.splitext(fname)[0])
        if not allowed or code in allowed:
            file_map[code] = str(ind_dir / fname)

    total_stocks = len(file_map)
    print(f"  Stocks found: {total_stocks}  (filtered from {len(all_files)} files)")

    if not file_map:
        return [BuildResult(
            factor_name=n, action="skip", elapsed=0.0, rows=0,
            factor_path=paths.factor_output_dir / f"{n}.fea",
            manifest_path=paths.manifest_output_dir / f"{n}.json",
        ) for n in factor_names]

    # ── Incremental path permanently disabled — always full rebuild ──
    effective_end = _resolve_effective_end_date(paths.source_root)

    # All factors are always active
    active_names = list(factor_names)
    total_factors = len(factor_names)

    # Always full rebuild — no incremental date filtering
    min_trade_time: str | None = None
    earliest_existing: str | None = None
    mode_str = "full rebuild (incremental disabled)"
    print(f"  Mode: {mode_str}")
    t_phase1_elapsed = time.perf_counter() - t_phase1
    print(f"  Phase 1 done  ({t_phase1_elapsed:.1f}s)")

    # ── Phase 2: Parallel metric computation ─────────────────────────────
    t_phase2 = time.perf_counter()
    print(f"\n  Phase 2/4 — Computing indicator metrics  "
          f"[ProcessPoolExecutor x{n_workers}]")

    BATCH_SIZE = 40
    # file_map is {code: filepath} → list of (code, filepath) tuples
    file_items = list(file_map.items())
    batches = [
        file_items[i : i + BATCH_SIZE]
        for i in range(0, len(file_items), BATCH_SIZE)
    ]
    total_batches = len(batches)
    batch_file_counts = [len(b) for b in batches]

    from tqdm import tqdm

    accumulators: dict[str, list[pd.Series]] = defaultdict(list)
    completed_stocks = 0
    batch_errors = 0

    pbar = tqdm(
        total=total_stocks, desc="  Stocks", unit="stk",
        bar_format="{desc:>12}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {postfix}]",
    )

    t_batch_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_indicator_batch, batch, min_trade_time): idx
            for idx, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            n_in_batch = batch_file_counts[batch_idx]
            try:
                batch_result = fut.result()
            except Exception as e:
                batch_errors += 1
                completed_stocks += n_in_batch
                pbar.update(n_in_batch)
                if batch_errors <= 3:
                    pbar.write(
                        f"  [!] Batch {batch_idx} failed: "
                        f"{type(e).__name__}: {e}"
                    )
                continue
            for col, series_list in batch_result.items():
                accumulators[col].extend(series_list)
            completed_stocks += n_in_batch
            elapsed_batch = time.perf_counter() - t_batch_start
            rate = completed_stocks / elapsed_batch if elapsed_batch > 0 else 0
            pbar.set_postfix_str(f"{rate:.0f} stk/s")
            pbar.update(n_in_batch)

    pbar.close()

    if batch_errors:
        print(f"  [!] {batch_errors}/{total_batches} batches failed")
    total_accumulated = sum(len(v) for v in accumulators.values())
    if not accumulators:
        print(
            f"  [!] FATAL: all batches failed — no metrics accumulated. "
            f"Check that indicator_1min/*.parquet files are readable."
        )
    else:
        print(
            f"  Accumulated {total_accumulated} metric series across "
            f"{len(accumulators)} columns"
        )

    t_phase2_elapsed = time.perf_counter() - t_phase2
    rate_p2 = total_stocks / t_phase2_elapsed if t_phase2_elapsed > 0 else 0
    print(f"  Phase 2 done  ({t_phase2_elapsed:.1f}s, {rate_p2:.0f} stocks/s)")

    # ── Concatenate raw metrics ──────────────────────────────────────────
    t_concat = time.perf_counter()
    raw_metrics: dict[str, pd.Series] = {}
    for col in _INDICATOR_METRIC_COLS:
        parts = accumulators.get(col, [])
        if parts:
            s = pd.concat(parts)
            # Drop duplicates via O(N) Index.duplicated
            if s.index.has_duplicates:
                s = s[~s.index.duplicated(keep="last")]
            if isinstance(s.index, pd.MultiIndex):
                s.index = s.index.set_names(["Date", "Code"])
            raw_metrics[col] = s
        else:
            raw_metrics[col] = pd.Series(dtype=float, name=col)

    # Incremental filter — only when truly in incremental mode.
    if not force and earliest_existing and min_trade_time is not None:
        for col in list(raw_metrics.keys()):
            s = raw_metrics[col]
            if not s.empty:
                raw_metrics[col] = s[
                    s.index.get_level_values("Date") > earliest_existing
                ]

    print(f"  Metrics concatenated  ({time.perf_counter() - t_concat:.1f}s)")

    # ── Phase 3: Factor frame assembly ───────────────────────────────────
    t_phase3 = time.perf_counter()
    print(f"\n  Phase 3/4 — Building factor frames")

    output: dict[str, pd.DataFrame] = {}
    errors_build: list[str] = []

    pbar_factor = tqdm(
        total=total_factors, desc="  Factors", unit="fac",
        bar_format="{desc:>12}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}] {postfix}",
    )

    for name in factor_names:
        pbar_factor.set_postfix_str(f"{name}")
        pbar_factor.update(0)

        if name not in INDICATOR_FACTOR_SPEC:
            output[name] = pd.DataFrame()
            errors_build.append(name)
            pbar_factor.update(1)
            continue

        metric_col, direction = INDICATOR_FACTOR_SPEC[name]
        raw = raw_metrics.get(metric_col)
        if raw is None or raw.empty:
            output[name] = pd.DataFrame()
            pbar_factor.update(1)
            continue

        try:
            if direction == "pos":
                ranked = cross_sectional_rank(raw)
            elif direction == "neg":
                ranked = cross_sectional_rank(-raw)
            elif direction == "neg_abs":
                ranked = cross_sectional_rank(-raw.abs())
            else:
                output[name] = pd.DataFrame()
                pbar_factor.update(1)
                continue

            frame = ensure_single_factor_frame(ranked, name)
            output[name] = frame
        except Exception:
            output[name] = pd.DataFrame()
            errors_build.append(name)

        pbar_factor.update(1)

    pbar_factor.close()

    t_phase3_elapsed = time.perf_counter() - t_phase3
    rate_p3 = total_factors / t_phase3_elapsed if t_phase3_elapsed > 0 else 0
    print(f"  Phase 3 done  ({t_phase3_elapsed:.1f}s, {rate_p3:.1f} factors/s)")

    # ── Phase 4: Writing output files ────────────────────────────────────
    t_phase4 = time.perf_counter()
    print(f"\n  Phase 4/4 — Writing .fea output files")

    results: list = []
    write_errors = 0

    pbar_write = tqdm(
        total=total_factors, desc="  Write", unit="file",
        bar_format="{desc:>12}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}] {postfix}",
    )

    for name in factor_names:
        frame = output.get(name)
        factor_path = paths.factor_output_dir / f"{name}.fea"

        if frame is None or frame.empty:
            results.append(BuildResult(
                factor_name=name, action="error", elapsed=0.0, rows=0,
                factor_path=factor_path,
                manifest_path=paths.manifest_output_dir / f"{name}.json",
            ))
            write_errors += 1
            pbar_write.set_postfix_str(f"ERR {name}")
            pbar_write.update(1)
            continue

        spec = get_factor(name)
        try:
            if not factor_path.exists() or min_trade_time is None:
                write_factor(spec, frame, paths=paths)
                action = "rebuild"
            else:
                write_factor_incremental(spec, frame, paths=paths)
                action = "incremental"
            results.append(BuildResult(
                factor_name=name, action=action, elapsed=0.0,
                rows=len(frame), factor_path=factor_path,
                manifest_path=paths.manifest_output_dir / f"{name}.json",
            ))
            pbar_write.set_postfix_str(f"OK {name}")
            pbar_write.update(1)
        except Exception:
            results.append(BuildResult(
                factor_name=name, action="error", elapsed=0.0, rows=0,
                factor_path=factor_path,
                manifest_path=paths.manifest_output_dir / f"{name}.json",
            ))
            write_errors += 1
            pbar_write.set_postfix_str(f"ERR {name}")
            pbar_write.update(1)

    pbar_write.close()

    t_phase4_elapsed = time.perf_counter() - t_phase4
    rate_p4 = total_factors / t_phase4_elapsed if t_phase4_elapsed > 0 else 0
    print(f"  Phase 4 done  ({t_phase4_elapsed:.1f}s, {rate_p4:.1f} files/s)")

    # ── Final summary ────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_phase1
    built = sum(1 for r in results if r.action == "rebuild")
    skipped = sum(1 for r in results if r.action == "skip")
    errors_total = len(errors_build) + write_errors

    print(f"\n{'─'*64}")
    print(f"  BUILD COMPLETE")
    print(f"  Total time:        {total_elapsed:.1f}s")
    print(f"  Stocks processed:  {total_stocks}")
    print(f"  Factors built:     {built}  |  Skipped: {skipped}   |  Errors: {errors_total}")
    print(f"  Phase breakdown:")
    print(f"    Phase 1 (discovery):  {t_phase1_elapsed:>6.1f}s")
    print(f"    Phase 2 (metrics):    {t_phase2_elapsed:>6.1f}s  ({rate_p2:.0f} stocks/s)")
    print(f"    Phase 3 (assembly):   {t_phase3_elapsed:>6.1f}s  ({rate_p3:.1f} factors/s)")
    print(f"    Phase 4 (write):      {t_phase4_elapsed:>6.1f}s  ({rate_p4:.1f} files/s)")
    if errors_total:
        print(f"  Error list: {', '.join(errors_build)}")
    print(f"{'─'*64}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback: per-factor sequential reader  (for backward-compatible single-factor
# builds via the old build_many / build_many_parallel code paths)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_indicator_factor(
    source_root: Path,
    allowed_codes: set[str],
    metric_col: str,
    on_progress=None,
) -> pd.Series:
    """Compute one indicator metric for all stocks from per-stock 1-min files.

    Parameters
    ----------
    source_root : Path
        Root directory containing indicator_1min/ subdirectory.
    allowed_codes : set
        Set of allowed stock codes (6-digit).
    metric_col : str
        Column name from ``_indicator_all_metrics`` to extract.
    on_progress : callable or None

    Returns
    -------
    pd.Series with (Date, Code) MultiIndex.
    """
    ind_dir = source_root / "indicator_1min"
    if not ind_dir.exists():
        raise FileNotFoundError(f"indicator_1min data directory not found: {ind_dir}")

    files = sorted([f for f in os.listdir(ind_dir) if f.endswith(".parquet")])
    parts = []
    total = len(files)

    for i, fname in enumerate(files):
        code = _pad_code(os.path.splitext(fname)[0])
        if allowed_codes and code not in allowed_codes:
            if on_progress:
                on_progress("indicator", i + 1, total)
            continue

        filepath = ind_dir / fname
        try:
            stock_df = pd.read_parquet(filepath)
        except Exception:
            if on_progress:
                on_progress("indicator", i + 1, total)
            continue

        if stock_df.empty:
            if on_progress:
                on_progress("indicator", i + 1, total)
            continue

        try:
            daily = _indicator_all_metrics(stock_df)
        except Exception:
            if on_progress:
                on_progress("indicator", i + 1, total)
            continue

        if metric_col not in daily.columns:
            if on_progress:
                on_progress("indicator", i + 1, total)
            continue

        s = daily[metric_col].dropna()
        if s.empty:
            if on_progress:
                on_progress("indicator", i + 1, total)
            continue

        s = pd.DataFrame(
            {metric_col: s.values},
            index=pd.MultiIndex.from_arrays(
                [s.index, [code] * len(s)], names=["Date", "Code"]
            ),
        )[metric_col]
        parts.append(s)

        if on_progress:
            on_progress("indicator", i + 1, total)

    if not parts:
        return pd.Series(
            dtype=float, name=metric_col,
            index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Code"]),
        )

    result = pd.concat(parts)
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    result = result.groupby(list(result.index.names)).last()
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.set_names(["Date", "Code"])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Indicator Factors
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each registered factor has a real implementation that calls
# ``_compute_indicator_factor``.  The unified builder (``build_indicator_1min_new``)
# bypasses them entirely for efficiency, but these remain as the authoritative
# single-factor code path and carry all metadata (description, thesis, dependencies).

# ── MACD Factors ────────────────────────────────────────────────────────────

@register_factor(
    name="macd_signal_cross",
    description="MACD金叉死叉信号因子（日内收盘macd柱>0且dif>dea截面排名，金叉状态排前）。",
    category="intraday",
    thesis=(
        "日内收盘时的MACD柱(macd = dif - dea)方向是日内趋势的终态判断。"
        "macd>0且dif>dea：全天MACD处于多头状态，收盘确认金叉有效性；"
        "macd<0且dif<dea：全天MACD处于空头状态，收盘确认死叉。"
        "收盘MACD状态比盘中任何单分钟的MACD状态更有信息量——"
        "因为它包含了全天的多空博弈结果。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_signal_cross(context: FactorContext):
    source_root = context.repo.paths.source_root
    sig = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_signal_raw",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(sig)


@register_factor(
    name="macd_trend_strength",
    description="MACD趋势强度因子（(dif-dea)/|dea|截面排名，趋势强度高排前）。",
    category="intraday",
    thesis=(
        "DIF与DEA之间的距离(归一化)反映了MACD趋势的强度。"
        "|DIF-DEA|/|DEA|大的股票处于强趋势中（无论方向），应有趋势持续性；"
        "该比率小的股票处于盘整中，趋势信号不可靠。"
        "该因子评估了'MACD信号的可信度'而非方向。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_trend_strength(context: FactorContext):
    source_root = context.repo.paths.source_root
    ts = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_trend_strength",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ts)


@register_factor(
    name="macd_daily_consistency",
    description="MACD日内一致性因子（日内macd柱>0的分钟占比截面排名，高一致性=趋势明确排前）。",
    category="intraday",
    thesis=(
        "MACD柱方向在全天各分钟的占比反映了日内趋势的一致性。"
        "一致性接近100%或0%意味着全天单边走势，趋势十分明确；"
        "一致性接近50%意味着macd柱频繁翻红翻绿，日内多空拉锯。"
        "高一致性的交易日提供了更可靠的趋势信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_daily_consistency(context: FactorContext):
    source_root = context.repo.paths.source_root
    cons = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_consistency",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(cons)


@register_factor(
    name="macd_price_divergence",
    description="MACD价格背离因子（MA5趋势与macd柱趋势的日内相关性截面排名，负相关=背离信号排前）。",
    category="intraday",
    thesis=(
        "日内价格运动方向与MACD柱运动方向的相关系数度量了量价配合程度。"
        "正相关=价格上涨+MACD柱增大（健康上涨），或价格下跌+MACD柱减小（健康下跌）；"
        "负相关=价格上涨但MACD柱在减小（顶背离），或价格下跌但MACD柱在增大（底背离）。"
        "负相关=MACD背离，是经典的反转预警信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_price_divergence(context: FactorContext):
    source_root = context.repo.paths.source_root
    corr = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_price_corr",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-corr)


@register_factor(
    name="macd_dif_slope",
    description="MACD-DIF日内斜率因子（(dif_close-dif_open)/|dif_open|截面排名，DIF上升=强势排前）。",
    category="intraday",
    thesis=(
        "DIF日内斜率反映了MACD快线在当天的运动方向和速度。"
        "DIF日内持续上升意味着短期动能持续增强，属于强势信号；"
        "DIF日内持续下降意味着动能衰竭，即使是macd柱为正也在减速。"
        "DIF斜率比收盘DIF绝对值更早地发出趋势变化信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_dif_slope(context: FactorContext):
    source_root = context.repo.paths.source_root
    slope = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_dif_slope",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(slope)


@register_factor(
    name="macd_acceleration",
    description="MACD柱加速度因子（(macd_close-macd_open)/|macd_open|截面排名，加速扩张排前）。",
    category="intraday",
    thesis=(
        "MACD柱(即dif-dea)在一天内的变化率衡量了趋势是否在加速。"
        "MACD柱正在扩大=动能加速度为正，趋势还有延续空间；"
        "MACD柱正在缩小=动能在衰竭，趋势可能即将反转。"
        "这是'二阶导数'信号，比单纯的MACD方向更灵敏。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_acceleration(context: FactorContext):
    source_root = context.repo.paths.source_root
    acc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_acceleration",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(acc)


@register_factor(
    name="macd_zero_cross",
    description="MACD零轴穿越因子（DIF穿越零轴的频率截面排名，频繁穿越=趋势混乱排后）。",
    category="intraday",
    thesis=(
        "DIF在一天内穿越零轴的次数反映了趋势的稳定性。"
        "零次穿越=全天DIF维持在同一侧（多或空），趋势稳固；"
        "多次穿越=DIF在正负间反复，方向不明确，趋势混乱。"
        "频繁零轴穿越的股票后续走势难以预测，应避免。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_zero_cross(context: FactorContext):
    source_root = context.repo.paths.source_root
    zc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_zero_cross",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-zc)


@register_factor(
    name="macd_bar_sign_change",
    description="MACD柱翻红翻绿因子（日内macd柱正负切换次数截面排名，频繁切换=趋势不稳排后）。",
    category="intraday",
    thesis=(
        "MACD柱在一天内的正负切换次数反映了多空力量的拉锯程度。"
        "切换次数少=全天单边趋势明确；"
        "切换次数多=多空反复拉锯，方向不明。"
        "频繁的柱状图正负切换意味着量化/程序化交易的来回博弈，趋势不可靠。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_bar_sign_change(context: FactorContext):
    source_root = context.repo.paths.source_root
    sc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_bar_sign_change",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-sc)


@register_factor(
    name="macd_extreme_ratio",
    description="MACD极端值占比因子（|macd|>2σ的分钟占比截面排名，高极端值=异常行情排后）。",
    category="intraday",
    thesis=(
        "MACD柱绝对值异常放大（超过2倍标准差）的分钟占比反映了行情的极端程度。"
        "高比例=全天MACD柱频繁出现极端值，可能是资金博弈激烈或恐慌；"
        "低比例=MACD柱在正常范围内波动，行情平稳。"
        "极端占比异常升高往往是变盘或行情衰竭的前兆。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_extreme_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    er = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_extreme_ratio",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-er)


@register_factor(
    name="macd_daily_range",
    description="MACD日内振幅因子（macd_max-macd_min截面排名，振幅大=波动剧烈排后）。",
    category="intraday",
    thesis=(
        "MACD柱在一天内的最大-最小值差反映了MACD的振幅，即多空力量的波动范围。"
        "振幅大=多空博弈激烈，趋势不稳定；"
        "振幅小=MACD柱变化平缓，趋势温和但稳定。"
        "MACD振幅与价格波动率高度相关，但提供了纯技术指标层面的正交信息。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_daily_range(context: FactorContext):
    source_root = context.repo.paths.source_root
    rng = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_range",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-rng)


# ── KDJ Factors ─────────────────────────────────────────────────────────────

@register_factor(
    name="kdj_j_value_close",
    description="KDJ-J值因子（日内收盘J值截面排名，高J=超买排后，低J=超卖排前）。",
    category="intraday",
    thesis=(
        "J值(3K-2D)是KDJ中最敏感的线，率先反应动量的极端状态。"
        "J>100：超买区域，短期内大概率回调；"
        "J<0：超卖区域，短期内大概率反弹。"
        "日末收盘J值包含了全天的多空博弈结果，比盘中J值更稳定。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_j_value_close(context: FactorContext):
    source_root = context.repo.paths.source_root
    jv = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "j_close",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-jv)


@register_factor(
    name="kdj_k_d_distance",
    description="KDJ-K-D距离因子（(K-D)/|D|截面排名，正=多头动能排前）。",
    category="intraday",
    thesis=(
        "K线与D线的距离(归一化)度量了KDJ的多空强度。"
        "(K-D)/|D|>0：K在D之上，多头排列；数值越大，多头动能越强。"
        "(K-D)/|D|<0：K在D之下，空头排列；数值越小，空头动能越强。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_k_d_distance(context: FactorContext):
    source_root = context.repo.paths.source_root
    kd = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "k_d_distance",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(kd)


@register_factor(
    name="kdj_cross_signal",
    description="KDJ金叉信号因子（日内K上穿D的次数截面排名，金叉次数多=强势排前）。",
    category="intraday",
    thesis=(
        "K线上穿D线（金叉）是KDJ的经典做多信号。"
        "在日内1分钟频率上，金叉可以发生多次。"
        "金叉次数多意味着K线多次试图并成功突破D线——多头反复确认。"
        "单次金叉可能是噪音，多次金叉信号可靠性更高。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_cross_signal(context: FactorContext):
    source_root = context.repo.paths.source_root
    kc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "kdj_cross_count",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(kc)


@register_factor(
    name="kdj_j_reversal_risk",
    description="KDJ-J值反转风险因子（|j_close-50|截面排名，极端J值=高反转概率排后）。",
    category="intraday",
    thesis=(
        "J值偏离50的绝对值度量了J值的极端程度。"
        "J值越极端（远高100或远低于0），反转概率越高。"
        "该因子独立于J的方向，纯粹度量了'距离中性的偏离'——"
        "无论是极度超买还是极度超卖，都意味着短期方向不可持续。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_j_reversal_risk(context: FactorContext):
    source_root = context.repo.paths.source_root
    jr = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "j_reversal_risk",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-jr)


@register_factor(
    name="kdj_j_range",
    description="KDJ-J值振幅因子（j_max-j_min截面排名，J值波动大=多空分歧大排后）。",
    category="intraday",
    thesis=(
        "J值在一天内的波动范围反映了多空分歧的激烈程度。"
        "J值振幅大=日内经历从超买到超卖的剧烈摆动，行情极不稳定；"
        "J值振幅小=日内多空力量相对均衡或单边温和运行。"
        "高振幅往往预示着后续的均值回归。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_j_range(context: FactorContext):
    source_root = context.repo.paths.source_root
    jr = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "j_range",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-jr)


@register_factor(
    name="kdj_j_volatility",
    description="KDJ-J值波动因子（std(J)截面排名，J值波动率高=不稳定排后）。",
    category="intraday",
    thesis=(
        "J值的日内标准差度量了J值的不稳定性。"
        "高波动=J值频繁上下跳动，技术信号不可靠；"
        "低波动=J值变化平滑，技术信号可信度高。"
        "该因子评估了KDJ信号的质量而非方向。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_j_volatility(context: FactorContext):
    source_root = context.repo.paths.source_root
    jv = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "j_volatility",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-jv)


@register_factor(
    name="kdj_d_stability",
    description="KDJ-D线稳定性因子（std(D)/mean(D)截面排名，D线不稳定=信号质量差排后）。",
    category="intraday",
    thesis=(
        "D线(慢线)的变异系数反映了KDJ慢线的稳定性。"
        "D线是K线的3日平滑，理论上应该比较稳定。"
        "如果D线在一天内也有较大波动（高CV），说明即使是慢线也在被剧烈拉扯，"
        "这是一个强烈的趋势不确定信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_d_stability(context: FactorContext):
    source_root = context.repo.paths.source_root
    ds = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "d_stability",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-ds)


@register_factor(
    name="kdj_overbought_frac",
    description="KDJ超买时间占比因子（K>80分钟占比截面排名，高占比=强势超买排后）。",
    category="intraday",
    thesis=(
        "K值在超买区域(>80)的停留时间占比反映了买盘的持续性。"
        "高占比=全天大部分时间处于超买状态，买盘极其强劲但也透支了购买力；"
        "低占比=未能进入超买或仅在超买区域短暂停留。"
        "持续的超买状态既可能是强势上涨的确认，也可能意味着顶部即将到来。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_overbought_frac(context: FactorContext):
    source_root = context.repo.paths.source_root
    ob = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "kdj_overbought_frac",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-ob)


@register_factor(
    name="kdj_oversold_frac",
    description="KDJ超卖时间占比因子（K<20分钟占比截面排名，高占比=深度超卖排前=抄底信号）。",
    category="intraday",
    thesis=(
        "K值在超卖区域(<20)的停留时间占比反映了卖盘的持续性。"
        "高占比=全天大部分时间处于超卖状态，恐慌性抛售充分释放；"
        "低占比=未进入超卖或仅在超卖短暂停留。"
        "持续超卖后的反弹概率显著升高，是逆向买入信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_oversold_frac(context: FactorContext):
    source_root = context.repo.paths.source_root
    os = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "kdj_oversold_frac",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(os)


@register_factor(
    name="kdj_bull_frac",
    description="KDJ多头时间占比因子（K>D的分钟占比截面排名，高占比=日内持续多头排前）。",
    category="intraday",
    thesis=(
        "K线在D线之上的时间占比是KDJ多头排列的持续性指标。"
        "接近100%=全天K都在D之上，多头控制全局；"
        "接近0%=全天K都在D之下，空头控制全局；"
        "接近50%=K-D反复交叉，方向不明确。"
        "该因子提供了一个连续的多头强度度量，比离散的'金叉次数'更平滑。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_bull_frac(context: FactorContext):
    source_root = context.repo.paths.source_root
    bf = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "kdj_bull_frac",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(bf)


@register_factor(
    name="kdj_cross_net",
    description="KDJ金叉净数量因子（金叉次数-死叉次数截面排名，净多头=强势排前）。",
    category="intraday",
    thesis=(
        "金叉与死叉的净差值综合了多头和空头信号的博弈结果。"
        "正数且大=日内多头信号远多于空头，强势确认；"
        "负数且小=空头信号主导，弱势。"
        "净金叉数量比单独的金叉次数更有信息量——它同时惩罚了空头信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_cross_net(context: FactorContext):
    source_root = context.repo.paths.source_root
    cn = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "kdj_cross_net",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(cn)


# ── RSI Factors ─────────────────────────────────────────────────────────────

@register_factor(
    name="rsi_14_excess",
    description="RSI超买超卖因子（(rsi_close-50)截面排名，高RSI=超买排后）。",
    category="intraday",
    thesis=(
        "RSI偏离50的程度直接度量了短期超买/超卖状态。"
        "使用1分钟RSI的收盘值（最后一分钟）作为当日RSI的终态判断。"
        "RSI>70=超买（排后），RSI<30=超卖（排前），中间区域线性过渡。"
    ),
    dependencies=("indicator_1min",),
)
def factor_rsi_14_excess(context: FactorContext):
    source_root = context.repo.paths.source_root
    rsi = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "rsi_excess",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-rsi)


@register_factor(
    name="rsi_extreme_fraction",
    description="RSI极端时间占比因子（RSI>70或<30的分钟占比截面排名，高占比=极端行情排前）。",
    category="intraday",
    thesis=(
        "RSI在极端区域(>70或<30)停留的时间占比反映了行情的极端程度。"
        "高占比=全天大部分时间处于超买或超卖状态，单边行情特征明显；"
        "低占比=RSI在中性区域波动，典型的震荡行情。"
        "极端时间占比异常升高往往是变盘的前兆。"
    ),
    dependencies=("indicator_1min",),
)
def factor_rsi_extreme_fraction(context: FactorContext):
    source_root = context.repo.paths.source_root
    ef = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "rsi_extreme_frac",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ef)


@register_factor(
    name="rsi_intraday_trend",
    description="RSI日内趋势因子（RSI与时间的相关性截面排名，正相关=日内动量积聚排前）。",
    category="intraday",
    thesis=(
        "RSI在一天中与分钟序号的相关系数反映了日内动量的积聚方向。"
        "正相关=RSI随交易推进而上升（盘中资金持续流入，动量积聚）；"
        "负相关=RSI随交易推进而下降（盘中资金持续流出，动能衰竭）。"
    ),
    dependencies=("indicator_1min",),
)
def factor_rsi_intraday_trend(context: FactorContext):
    source_root = context.repo.paths.source_root
    tc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "rsi_time_corr",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(tc)


@register_factor(
    name="rsi_range",
    description="RSI日内振幅因子（rsi_max-rsi_min截面排名，振幅大=剧烈波动排后）。",
    category="intraday",
    thesis=(
        "RSI在一天内的波动范围反映了日内情绪的摇摆程度。"
        "振幅大=RSI经历了从超买到超卖的剧烈摆动（或反之），行情极不稳定；"
        "振幅小=RSI在狭窄范围内波动，情绪稳定。"
    ),
    dependencies=("indicator_1min",),
)
def factor_rsi_range(context: FactorContext):
    source_root = context.repo.paths.source_root
    rr = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "rsi_range",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-rr)


@register_factor(
    name="rsi_volatility",
    description="RSI波动率因子（std(RSI)/mean(RSI)截面排名，RSI不稳定=信号质量差排后）。",
    category="intraday",
    thesis=(
        "RSI的变异系数度量了RSI本身的稳定性。"
        "高CV=RSI频繁上下跳动，技术信号不可靠；"
        "低CV=RSI变化平滑，信号质量高。"
        "该因子评估了RSI信号的信噪比。"
    ),
    dependencies=("indicator_1min",),
)
def factor_rsi_volatility(context: FactorContext):
    source_root = context.repo.paths.source_root
    rv = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "rsi_volatility",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-rv)


@register_factor(
    name="rsi_overbought_frac",
    description="RSI超买时间占比因子（RSI>70分钟占比截面排名，高占比=强势超买排后）。",
    category="intraday",
    thesis=(
        "RSI在超买区域(>70)停留的时间占比是独立的超买强度度量。"
        "与rsi_extreme_fraction不同（它混合了超买和超卖），"
        "该因子只捕获单边的超买压力。"
    ),
    dependencies=("indicator_1min",),
)
def factor_rsi_overbought_frac(context: FactorContext):
    source_root = context.repo.paths.source_root
    ro = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "rsi_overbought_frac",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-ro)


@register_factor(
    name="rsi_oversold_frac",
    description="RSI超卖时间占比因子（RSI<30分钟占比截面排名，高占比=深度超卖排前）。",
    category="intraday",
    thesis=(
        "RSI在超卖区域(<30)停留的时间占比是独立的超卖强度度量。"
        "持续超卖后的反弹概率显著升高，是逆向买入信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_rsi_oversold_frac(context: FactorContext):
    source_root = context.repo.paths.source_root
    ro = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "rsi_oversold_frac",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ro)


# ── Bollinger Factors ───────────────────────────────────────────────────────

@register_factor(
    name="boll_position",
    description="布林带位置因子（(ma5_close-boll_mid)/(boll_upper-boll_lower)截面排名，上轨附近=强势排前）。",
    category="intraday",
    thesis=(
        "收盘价(ma5 proxy)在布林带中的相对位置是一个标准化的动量信号。"
        "接近上轨(+0.5~+1.0)=强势，往往有继续向上突破的动能；"
        "接近下轨(-1.0~-0.5)=弱势，但也是潜在的反弹位置。"
    ),
    dependencies=("indicator_1min",),
)
def factor_boll_position(context: FactorContext):
    source_root = context.repo.paths.source_root
    bp = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "boll_position",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(bp)


@register_factor(
    name="boll_width_20",
    description="布林带宽度因子（(upper-lower)/mid截面排名，窄带=变盘前兆排前）。",
    category="intraday",
    thesis=(
        "布林带宽度是波动率的直接度量。带宽窄=波动率低='暴风雨前的宁静'，"
        "往往预示着即将出现大行情。带宽宽=高波动环境，行情已经在进行中。"
        "该因子捕捉了'波动率收缩-扩张'周期的收缩端。"
    ),
    dependencies=("indicator_1min",),
)
def factor_boll_width_20(context: FactorContext):
    source_root = context.repo.paths.source_root
    bw = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "boll_width",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-bw)



@register_factor(
    name="boll_squeeze",
    description="布林带挤压因子（1/布林带宽截面排名，带宽极窄=挤压程度高=变盘迫近排前）。",
    category="intraday",
    thesis=(
        "布林带挤压(带宽的倒数)是波动率压缩到极致时的信号。"
        "挤压程度高=波动率被极度压缩，价格在极窄的区间内运行——"
        "这是经典的技术性变盘前兆(Bollinger Squeeze)。"
        "挤压后的突破方向由其他因素决定，但挤压本身提供了'即将行动'的时间窗口。"
    ),
    dependencies=("indicator_1min",),
)
def factor_boll_squeeze(context: FactorContext):
    source_root = context.repo.paths.source_root
    sq = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "boll_squeeze",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-sq)




@register_factor(
    name="boll_mid_slope",
    description="布林带中轨斜率因子（(boll_mid_close-boll_mid_open)/|boll_mid_open|截面排名，中轨上移=趋势向上排前）。",
    category="intraday",
    thesis=(
        "布林带中轨(20期均线)的日内斜率反映了短期趋势的方向和强度。"
        "中轨是20期均线——其斜率为正意味着均线正在上移，趋势向上；"
        "斜率为负意味着均线下移，趋势向下。"
        "中轨斜率是布林带体系中的'方向性'信息，与带宽(波动率)正交。"
    ),
    dependencies=("indicator_1min",),
)
def factor_boll_mid_slope(context: FactorContext):
    source_root = context.repo.paths.source_root
    ms = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "boll_mid_slope",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ms)


@register_factor(
    name="boll_width_change",
    description="布林带宽变化因子（收盘带宽-开盘带宽截面排名，带宽扩张=波动率上升排前）。",
    category="intraday",
    thesis=(
        "布林带宽度在一天内的变化方向反映了波动率的日内趋势。"
        "带宽扩张=波动率在上升，往往伴随趋势启动或加速；"
        "带宽收缩=波动率在下降，往往伴随趋势衰竭或盘整。"
        "带宽变化方向比带宽的绝对水平更有预测价值——"
        "扩张中的带宽意味着'波动率正在从低点回升'。"
    ),
    dependencies=("indicator_1min",),
)
def factor_boll_width_change(context: FactorContext):
    source_root = context.repo.paths.source_root
    wc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "boll_width_change",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(wc)


@register_factor(
    name="boll_band_deviation",
    description="布林带偏离因子（(ma5-boll_mid)/(upper-lower)截面排名，偏离大=脱离均衡排后）。",
    category="intraday",
    thesis=(
        "价格(ma5)与布林带中心(中轨)的距离，以带宽为单位标准化。"
        "该因子与boll_position类似，但使用中轨作为基准（而非相对上下轨的位置）。"
        "偏离越大=价格越远离均衡位置，均值回归压力越大。"
        "取绝对值排名，双向极端偏离均排后。"
    ),
    dependencies=("indicator_1min",),
)
def factor_boll_band_deviation(context: FactorContext):
    source_root = context.repo.paths.source_root
    bd = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "boll_band_deviation",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-bd.abs())


# ── MA Factors ──────────────────────────────────────────────────────────────

@register_factor(
    name="ma_alignment_score",
    description="MA均线排列因子（ma5>ma10>ma20>ma30>ma60的满足数量截面排名，多头排列排前）。",
    category="intraday",
    thesis=(
        "多周期均线的排列顺序是多时间框架趋势一致性的直接度量。"
        "5条均线全部多头排列(score=4)=最强多头趋势，趋势的持续性最高；"
        "全部空头排列(score=0)=最强空头趋势；"
        "均线交织(score=2左右)=方向不明确，震荡行情。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma_alignment_score(context: FactorContext):
    source_root = context.repo.paths.source_root
    ma = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma_alignment",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ma)


@register_factor(
    name="ma_dispersion",
    description="MA离散度因子（各均线间距的变异系数截面排名，低离散=均线收敛=变盘前兆排前）。",
    category="intraday",
    thesis=(
        "五条均线(MA5/10/20/30/60)的离散度反映了不同时间框架交易者的共识程度。"
        "离散度低=均线收敛在一起=不同时间框架的交易者成本趋于一致="
        "'暴风雨前的宁静'，即将出现方向性突破。"
        "离散度高=均线发散=趋势已经在进行中，顺势而为。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma_dispersion(context: FactorContext):
    source_root = context.repo.paths.source_root
    md = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma_dispersion",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-md)



@register_factor(
    name="price_vs_ma20_deviation",
    description="价格偏离MA20因子（(close-ma20)/|ma20|截面排名，偏离大=均值回归压力大排后）。",
    category="intraday",
    thesis=(
        "价格对20日均线(MA20)的偏离百分比是一个经典的均值回归信号。"
        "大幅高于MA20=短期内涨幅过大，存在获利了结压力；"
        "大幅低于MA20=短期内跌幅过大，存在技术性反弹需求。"
        "使用1分钟MA20（而非日线MA20）的优势在于更及时地反映日内趋势。"
    ),
    dependencies=("indicator_1min",),
)
def factor_price_vs_ma20_deviation(context: FactorContext):
    source_root = context.repo.paths.source_root
    pv = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "price_vs_ma20",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-pv.abs())


@register_factor(
    name="price_vs_ma60_deviation",
    description="价格偏离MA60因子（(close-ma60)/|ma60|截面排名，偏离大=长期趋势偏离排后）。",
    category="intraday",
    thesis=(
        "价格对60日均线(MA60)的偏离反映了中长期趋势的偏离程度。"
        "MA60是经典的中期趋势线——价格大幅偏离MA60意味着"
        "与中期趋势的背离，回归概率较高。"
    ),
    dependencies=("indicator_1min",),
)
def factor_price_vs_ma60_deviation(context: FactorContext):
    source_root = context.repo.paths.source_root
    pv = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "price_vs_ma60",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-pv.abs())


@register_factor(
    name="ma5_slope",
    description="MA5斜率因子（(ma5_close-ma5_open)/|ma5_open|截面排名，均线上移=趋势向上排前）。",
    category="intraday",
    thesis=(
        "MA5在一天内的变化率反映了最短周期均线的日内趋势。"
        "MA5日内上升=价格在5分钟尺度上持续走高，短期动能向上；"
        "MA5日内下降=价格在5分钟尺度上持续走低，短期动能向下。"
        "MA5斜率是微观趋势的最直接度量。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma5_slope(context: FactorContext):
    source_root = context.repo.paths.source_root
    ms = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma5_slope",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ms)


@register_factor(
    name="ma20_slope",
    description="MA20斜率因子（(ma20_close-ma20_open)/|ma20_open|截面排名，中期均线上移排前）。",
    category="intraday",
    thesis=(
        "MA20(布林带中轨)的日内斜率反映了中期趋势的日内变化。"
        "MA20日内上升=中期均线正在上移，趋势健康向上；"
        "MA20日内下降=中期均线正在下移，趋势走弱。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma20_slope(context: FactorContext):
    source_root = context.repo.paths.source_root
    ms = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma20_slope",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ms)


@register_factor(
    name="ma_convergence",
    description="MA均线收敛因子（|ma5-ma60|/|ma60|截面排名，短长均线距离近=收敛排前=变盘信号）。",
    category="intraday",
    thesis=(
        "最短周期均线(MA5)与最长周期均线(MA60)的距离度量了"
        "短期和中期趋势的离散程度。"
        "距离小=短中期均线收敛，多时间框架交易者趋于一致，即将变盘；"
        "距离大=均线发散，趋势明确。"
        "该因子与ma_dispersion互补——ma_dispersion衡量5条均线的离散度，"
        "而ma_convergence关注最短与最长均线的距离。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma_convergence(context: FactorContext):
    source_root = context.repo.paths.source_root
    mc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma_convergence",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-mc)


@register_factor(
    name="ma_cross_count",
    description="MA均线交叉因子（ma5上穿/下穿ma20次数截面排名，交叉多=方向切换频繁排后）。",
    category="intraday",
    thesis=(
        "MA5与MA20在一天内的交叉次数反映了短期-中期趋势的一致性。"
        "零次交叉=MA5全天保持在MA20的同一侧，短中期方向一致；"
        "多次交叉=MA5频繁穿越MA20，短中期方向反复切换。"
        "高频交叉意味着技术信号极度混乱，方向不可预测。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma_cross_count(context: FactorContext):
    source_root = context.repo.paths.source_root
    cc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma_cross_count",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-cc)


@register_factor(
    name="ma_bull_bear_ratio",
    description="MA多空比率因子（所有MA对中多头排列对数/总对数截面排名，高比率=全面多头排前）。",
    category="intraday",
    thesis=(
        "在6个MA对组合(5-10,5-20,5-60,10-20,10-60,20-60)中，"
        "短MA > 长MA的占比。6/6=全面多头，0/6=全面空头。"
        "比ma_alignment_score(0-4)更细粒度，覆盖了更多的MA对组合。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma_bull_bear_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    br = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma_bull_bear_ratio",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(br)


@register_factor(
    name="ma_curvature",
    description="MA曲率因子（ma5_slope-ma20_slope截面排名，正曲率=短周期加速快于中期排前）。",
    category="intraday",
    thesis=(
        "MA5斜率与MA20斜率的差值度量了均线的'曲率'——即加速度。"
        "正曲率=短周期均线比中期均线上移得更快，趋势正在加速；"
        "负曲率=短周期均线的上升速度不及中期，或在更快地下跌——趋势减速。"
        "曲率是趋势的二阶导数，比斜率更早地发出趋势变化信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma_curvature(context: FactorContext):
    source_root = context.repo.paths.source_root
    mc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma_curvature",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(mc)


# ── Volume MA Factors ───────────────────────────────────────────────────────

@register_factor(
    name="mavol_ratio_signal",
    description="成交量均线比率因子（mavol5/mavol10截面排名，比率>1=短期放量排前）。",
    category="intraday",
    thesis=(
        "短期(5分钟)成交量均线与中期(10分钟)成交量均线的比率是经典的放量/缩量信号。"
        "比率>1=短期成交活跃度高于中期，正在放量（可能伴随突破）；"
        "比率<1=短期成交萎缩，正在缩量（变盘前的沉寂）。"
        "与日线的量比不同，1分钟级别的vol均线比更能捕捉日内微观放量。"
    ),
    dependencies=("indicator_1min",),
)
def factor_mavol_ratio_signal(context: FactorContext):
    source_root = context.repo.paths.source_root
    mr = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "mavol_ratio",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(mr)


@register_factor(
    name="mavol5_slope",
    description="成交量MA5斜率因子（(mavol5_close-mavol5_open)/|mavol5_open|截面排名，成交量均线上移=放量排前）。",
    category="intraday",
    thesis=(
        "短期成交量均线(MAVol5)的日内斜率反映了成交活跃度的变化方向。"
        "MAVol5上升=成交量在5分钟尺度上持续放大，资金参与度提升；"
        "MAVol5下降=成交量萎缩，资金参与度降低。"
        "放量往往伴随趋势启动或加速，缩量往往预示趋势衰竭。"
    ),
    dependencies=("indicator_1min",),
)
def factor_mavol5_slope(context: FactorContext):
    source_root = context.repo.paths.source_root
    ms = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "mavol5_slope",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ms)


@register_factor(
    name="mavol_expansion",
    description="成交量扩张因子（mavol5_close/mavol5_open-1截面排名，成交量扩张=活跃度升排前）。",
    category="intraday",
    thesis=(
        "MAVol5在一天内的膨胀率度量了成交活跃度的日内变化。"
        "正值且大=成交量在日内显著放大，资金正在涌入/涌出；"
        "负值=成交量在日内萎缩，交易兴趣降低。"
        "成交量扩张是趋势可靠性的重要确认——"
        "价格上涨+成交量扩张=健康上涨；价格上涨+成交量萎缩=上涨乏力。"
    ),
    dependencies=("indicator_1min",),
)
def factor_mavol_expansion(context: FactorContext):
    source_root = context.repo.paths.source_root
    me = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "mavol_expansion",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(me)


@register_factor(
    name="mavol_ratio_std",
    description="成交量比率波动因子（std(mavol5/mavol10)截面排名，比率波动大=成交量不稳定排后）。",
    category="intraday",
    thesis=(
        "MAVol比率(5/10)在一天内的波动性反映了成交量节奏的稳定性。"
        "高波动=成交量忽大忽小，资金进出无序；"
        "低波动=成交量节奏稳定，交易行为有规律。"
        "成交量的不稳定性往往伴随着价格的不确定性。"
    ),
    dependencies=("indicator_1min",),
)
def factor_mavol_ratio_std(context: FactorContext):
    source_root = context.repo.paths.source_root
    rs = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "mavol_ratio_std",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-rs)


@register_factor(
    name="mavol_ratio_trend",
    description="成交量比率趋势因子（mavol比率的日内时间相关性截面排名，比率上升=持续放量排前）。",
    category="intraday",
    thesis=(
        "MAVol比率(5/10)与时间的相关性反映了成交量变化的持续性。"
        "正相关=成交量比率在日内持续走高——放量趋势具有持续性；"
        "负相关=成交量比率在日内走低——缩量趋势持续。"
        "持续放量比间歇性放量更能确认趋势的有效性。"
    ),
    dependencies=("indicator_1min",),
)
def factor_mavol_ratio_trend(context: FactorContext):
    source_root = context.repo.paths.source_root
    rt = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "mavol_ratio_trend",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(rt)


@register_factor(
    name="mavol_stability",
    description="成交量稳定性因子（std(mavol5)/mean(mavol5)截面排名，成交量不稳定排后）。",
    category="intraday",
    thesis=(
        "MAVol5的变异系数度量了短期成交量均线的稳定性。"
        "高CV=成交量忽大忽小，波动剧烈，资金行为不稳定；"
        "低CV=成交量平稳，交易节奏健康。"
        "成交量的稳定性本身就是一个信号——"
        "稳定的成交量意味着成熟的交易结构，不稳定的成交量意味着情绪化交易。"
    ),
    dependencies=("indicator_1min",),
)
def factor_mavol_stability(context: FactorContext):
    source_root = context.repo.paths.source_root
    ms = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "mavol_stability",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-ms)


@register_factor(
    name="volume_price_confirmation",
    description="量价配合确认因子（mavol5_slope符号==ma5_slope符号截面排名，量价同步=趋势确认排前）。",
    category="intraday",
    thesis=(
        "成交量MA5斜率与价格MA5斜率的方向一致性是经典的量价配合确认。"
        "方向一致=量价同步（价涨量增/价跌量减），趋势有成交量支撑；"
        "方向不一致=量价背离（价涨量缩/价跌量增），趋势不可靠。"
        "该因子将'量价配合'概念量化为1分钟级别的微观信号。"
    ),
    dependencies=("indicator_1min",),
)
def factor_volume_price_confirmation(context: FactorContext):
    source_root = context.repo.paths.source_root
    vp = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "volume_price_confirmation",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(vp)


# ── Cross-Indicator Composite Factors ───────────────────────────────────────

@register_factor(
    name="macd_kdj_alignment",
    description="MACD-KDJ一致性因子（sign(macd)*sign(K-D)截面排名，双指标同向=趋势确认排前）。",
    category="intraday",
    thesis=(
        "MACD方向和KDJ方向的一致性度量了两个最常用的趋势指标是否互相确认。"
        "+1=MACD多头+KDJ多头（双重确认，信号最强）；"
        "-1=MACD空头+KDJ空头（双重确认下跌）；"
        "0=两个指标方向矛盾（需要更多信息判断）。"
        "双指标同向确认的信号远优于单一指标。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_kdj_alignment(context: FactorContext):
    source_root = context.repo.paths.source_root
    ma = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_kdj_alignment",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ma)


@register_factor(
    name="rsi_boll_combo",
    description="RSI-布林带组合因子（rsi_zscore*boll_position截面排名，双双极端=强烈信号排前）。",
    category="intraday",
    thesis=(
        "RSI的Z-score与布林带位置的乘积捕捉了两个指标同时发出极端信号的时刻。"
        "RSI极端超买+价格在上轨附近=强烈的超买信号（可能反转）；"
        "RSI极端超卖+价格在下轨附近=强烈的超卖信号（可能反弹）。"
        "同向极端=两个独立的指标体系互相验证，信号可靠性显著提高。"
    ),
    dependencies=("indicator_1min",),
)
def factor_rsi_boll_combo(context: FactorContext):
    source_root = context.repo.paths.source_root
    rc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "rsi_boll_combo",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(rc)


@register_factor(
    name="ma_rsi_divergence",
    description="MA-RSI背离因子（ma5与rsi的日内相关性截面排名，负相关=量价背离排前=反转信号）。",
    category="intraday",
    thesis=(
        "MA5(价格代理)与RSI在一天内的相关性度量了价格和动量的配合程度。"
        "正相关=价格上涨+RSI上升（健康），或价格下跌+RSI下降（健康）；"
        "负相关=价格上涨但RSI在下降（顶背离），或价格下跌但RSI在上升（底背离）。"
        "负相关=经典的技术背离，是强烈的反转预警。"
        "该因子将日线级别的'RSI背离'概念应用到了1分钟级别。"
    ),
    dependencies=("indicator_1min",),
)
def factor_ma_rsi_divergence(context: FactorContext):
    source_root = context.repo.paths.source_root
    rd = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "ma_rsi_divergence",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-rd)


@register_factor(
    name="indicator_consensus",
    description="指标共识度因子（6个指标中同向占比截面排名，高共识=多指标共振排前）。",
    category="intraday",
    thesis=(
        "MACD、KDJ(K-D)、RSI(>50)、布林带位置(>0)、MA排列(≥3)、MAVol比率(>1)"
        "六个独立技术指标中看多方向的比例。"
        "6/6=所有指标一致看多，趋势确认度最高；"
        "0/6=所有指标一致看空；"
        "3/6=指标之间互相矛盾，方向不确定。"
        "多指标共振是技术分析中的'圣杯'——信号可靠性随共识度指数级提升。"
    ),
    dependencies=("indicator_1min",),
)
def factor_indicator_consensus(context: FactorContext):
    source_root = context.repo.paths.source_root
    ic = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "indicator_consensus",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ic)


@register_factor(
    name="trend_confirmation",
    description="趋势确认因子（ma_alignment*sign(macd_close)截面排名，均线+MACD双确认排前）。",
    category="intraday",
    thesis=(
        "MA排列分数(0-4)与MACD方向(+/-)的乘积综合了两个最重要的趋势指标。"
        "正值大=MA多头排列+MACD为正——双重确认上升趋势；"
        "负值大=MA空头排列+MACD为负——双重确认下降趋势。"
        "MA排列和MACD各自独立地度量趋势，它们的乘积提供了更强的趋势确认。"
    ),
    dependencies=("indicator_1min",),
)
def factor_trend_confirmation(context: FactorContext):
    source_root = context.repo.paths.source_root
    tc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "trend_confirmation",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(tc)


@register_factor(
    name="kdj_boll_combo",
    description="KDJ-布林带组合因子（(j-50)/50+boll_position截面排名，双重极端=强烈信号排前）。",
    category="intraday",
    thesis=(
        "KDJ的J值(标准化的-1到+1)与布林带位置的组合捕捉了超买超卖+价格位置的共振。"
        "J极端超买+价格在上轨=双重超买，回调概率极高；"
        "J极端超卖+价格在下轨=双重超卖，反弹概率极高。"
        "两个指标从不同角度（动量+波动率）给出了相同的极端判断，信号可靠性高。"
    ),
    dependencies=("indicator_1min",),
)
def factor_kdj_boll_combo(context: FactorContext):
    source_root = context.repo.paths.source_root
    kc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "kdj_boll_combo",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(kc)


@register_factor(
    name="macd_rsi_combo",
    description="MACD-RSI组合因子（sign(macd)*(rsi-50)截面排名，双动量指标同向排前）。",
    category="intraday",
    thesis=(
        "MACD方向(正/负)与RSI偏离中性的程度(RSI-50)的乘积综合了两个动量指标。"
        "MACD为正+RSI>50=双动量向上，趋势强劲；"
        "MACD为负+RSI<50=双动量向下，趋势疲弱。"
        "MACD和RSI是技术分析中最常用的两个动量指标——"
        "它们的组合信号比单独使用任何一个都更可靠。"
    ),
    dependencies=("indicator_1min",),
)
def factor_macd_rsi_combo(context: FactorContext):
    source_root = context.repo.paths.source_root
    mc = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "macd_rsi_combo",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(mc)


@register_factor(
    name="indicator_dispersion",
    description="指标离散度因子（5个标准化指标信号的std截面排名，离散度高=指标矛盾排后）。",
    category="intraday",
    thesis=(
        "五个标准化指标信号(MACD/KDJ/RSI/布林带/MA排列)的截面标准差。"
        "标准差大=不同指标给出的信号相互矛盾——"
        "有些指标看多，有些看空，综合方向不明确；"
        "标准差小=所有指标给出的信号一致——方向明确。"
        "该因子度量了'技术分析的噪音水平'——噪音越低越好。"
    ),
    dependencies=("indicator_1min",),
)
def factor_indicator_dispersion(context: FactorContext):
    source_root = context.repo.paths.source_root
    id_val = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "indicator_dispersion",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-id_val)


@register_factor(
    name="multi_indicator_extreme",
    description="多指标极端值因子（5个标准化指标的|z-score|均值截面排名，多指标同时极端=变盘排后）。",
    category="intraday",
    thesis=(
        "五个标准化指标信号的绝对Z-score均值。"
        "均值高=多个指标同时处于极端区域——强烈的超买或超卖；"
        "均值低=所有指标都在正常范围内。"
        "多指标同时极端是'过度延伸'的量化表达——"
        "当所有技术指标都指向极端时，反转的概率急剧升高。"
    ),
    dependencies=("indicator_1min",),
)
def factor_multi_indicator_extreme(context: FactorContext):
    source_root = context.repo.paths.source_root
    me = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "multi_indicator_extreme",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(-me)


# ── Session-based Factors ───────────────────────────────────────────────────

@register_factor(
    name="am_macd_trend",
    description="早盘MACD趋势因子（上午MACD与时间的相关性截面排名，正相关=早盘动能积聚排前）。",
    category="intraday",
    thesis=(
        "早盘(9:30-11:30)MACD的日内趋势反映了上午交易时段的多空力量演变。"
        "早盘MACD持续上升=多头在上午逐步取得优势；"
        "早盘MACD持续下降=空头在上午逐步施压。"
        "A股有'上午定调'的说法，早盘的MACD趋势往往决定了全天的基调。"
    ),
    dependencies=("indicator_1min",),
)
def factor_am_macd_trend(context: FactorContext):
    source_root = context.repo.paths.source_root
    at = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "am_macd_trend",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(at)


@register_factor(
    name="pm_macd_trend",
    description="午盘MACD趋势因子（下午MACD与时间的相关性截面排名，正相关=午盘动能持续排前）。",
    category="intraday",
    thesis=(
        "午盘(13:00-15:00)MACD的日内趋势反映了下午交易时段的多空演变。"
        "午盘MACD上升=多头在下午持续发力，全日强势；"
        "午盘MACD下降=多头在下午衰竭或空头反攻。"
        "A股的'下午反转'现象可以通过该因子捕捉——"
        "早盘强势但午盘MACD趋势转负的股票容易出现下午跳水。"
    ),
    dependencies=("indicator_1min",),
)
def factor_pm_macd_trend(context: FactorContext):
    source_root = context.repo.paths.source_root
    pt = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "pm_macd_trend",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(pt)


@register_factor(
    name="am_rsi_trend",
    description="早盘RSI趋势因子（上午RSI与时间的相关性截面排名，正相关=早盘动量积聚排前）。",
    category="intraday",
    thesis=(
        "早盘RSI的日内趋势是上午动量方向的独立度量（与MACD正交）。"
        "RSI持续走高=买盘在整个上午持续占优；"
        "RSI持续走低=卖盘在上午主导。"
    ),
    dependencies=("indicator_1min",),
)
def factor_am_rsi_trend(context: FactorContext):
    source_root = context.repo.paths.source_root
    at = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "am_rsi_trend",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(at)


@register_factor(
    name="pm_rsi_trend",
    description="午盘RSI趋势因子（下午RSI与时间的相关性截面排名，正相关=午盘动量持续排前）。",
    category="intraday",
    thesis=(
        "午盘RSI的日内趋势是下午动量方向的独立度量。"
        "RSI持续走高=买盘在下午持续发力；"
        "RSI持续走低=下午卖压加重。"
        "该因子与pm_macd_trend互补——MACD偏趋势，RSI偏动量。"
    ),
    dependencies=("indicator_1min",),
)
def factor_pm_rsi_trend(context: FactorContext):
    source_root = context.repo.paths.source_root
    pt = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "pm_rsi_trend",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(pt)


@register_factor(
    name="am_pm_macd_ratio",
    description="早午盘MACD比率因子（上午MACD均值/下午MACD均值截面排名，比率>1=早盘强于午盘=动能衰减排后）。",
    category="intraday",
    thesis=(
        "上午MACD与下午MACD的比率反映了动能的日内分布。"
        "比率>1=上午MACD强于下午，动能可能在日内衰减；"
        "比率<1=下午MACD强于上午，动能可能在下午加速。"
        "'上午冲高下午回落'是A股的经典模式，该因子量化了这个现象。"
    ),
    dependencies=("indicator_1min",),
)
def factor_am_pm_macd_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    ap = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "am_pm_macd_ratio",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ap)


@register_factor(
    name="am_pm_rsi_ratio",
    description="早午盘RSI比率因子（上午RSI均值/下午RSI均值截面排名，比率>1=早盘动量强于午盘排后）。",
    category="intraday",
    thesis=(
        "上午RSI与下午RSI的比率反映了动量的日内迁移。"
        "比率>1=早盘RSI更高，动量在上午集中释放，下午可能衰减；"
        "比率<1=午盘RSI更高，动量在下午增强。"
        "'上午拉高出货'的模式可以通过高比率+低绝对RSI来识别。"
    ),
    dependencies=("indicator_1min",),
)
def factor_am_pm_rsi_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    ap = _compute_indicator_factor(
        source_root, context.repo.allowed_codes, "am_pm_rsi_ratio",
        on_progress=context.repo.on_progress,
    )
    return cross_sectional_rank(ap)
