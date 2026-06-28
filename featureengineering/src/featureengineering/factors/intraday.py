"""Intraday microstructure factors from 1-minute data.

Realized volatility, intraday momentum/reversal, liquidity,
VWAP-based, volume concentration, and auction-period factors.

These factors share history_1min data.  The unified builder
(``build_intraday_unified``) reads each per-stock file **once** and
computes all 14 factors in a single pass with thread-parallel I/O,
reducing wall-clock time by ~30× compared to per-factor file scans.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ═══════════════════════════════════════════════════════════════════════════════
# Unified metric computation  (one stock → all metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def _intraday_all_metrics(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Compute **all** daily intraday metrics for one stock in a single pass.

    Returns a DataFrame indexed by date (YYYYMMDD string) with columns for
    every metric needed by the 14 registered intraday factors.
    """
    df = stock_df.copy()
    # Use .dt accessor — trade_time is datetime64[us], not string
    ts = df["trade_time"]
    df["trade_date"] = ts.dt.strftime("%Y%m%d")
    df["minute"] = ts.dt.hour * 60 + ts.dt.minute

    grouped = df.groupby("trade_date")

    # ── Daily OHLCV ───────────────────────────────────────────────────────
    daily_open = grouped["open"].first()
    daily_close = grouped["close"].last()
    daily_high = grouped["high"].max()
    daily_low = grouped["low"].min()
    daily_vol = grouped["vol"].sum()
    daily_amount = grouped["amount"].sum()

    results: dict[str, pd.Series] = {}
    results["open"] = daily_open
    results["close"] = daily_close
    results["high"] = daily_high
    results["low"] = daily_low
    results["vol"] = daily_vol
    results["amount"] = daily_amount
    results["vwap"] = daily_amount / daily_vol.replace(0, np.nan)

    # ── Morning session (09:30-11:30) ─────────────────────────────────────
    morning = df[(df["minute"] >= 570) & (df["minute"] <= 690)]
    _am_keys = ["am_open", "am_close", "am_high", "am_low", "am_vol", "am_amount", "am_vwap"]
    if not morning.empty:
        mg = morning.groupby("trade_date")
        results["am_open"] = mg["open"].first()
        results["am_close"] = mg["close"].last()
        results["am_high"] = mg["high"].max()
        results["am_low"] = mg["low"].min()
        results["am_vol"] = mg["vol"].sum()
        results["am_amount"] = mg["amount"].sum()
        results["am_vwap"] = results["am_amount"] / results["am_vol"].replace(0, np.nan)
    else:
        for k in _am_keys:
            results[k] = pd.Series(np.nan, index=daily_open.index)

    # ── Afternoon session (13:00-15:00) ───────────────────────────────────
    afternoon = df[df["minute"] >= 780]
    _pm_keys = ["pm_open", "pm_close", "pm_high", "pm_low", "pm_vol", "pm_amount", "pm_vwap"]
    if not afternoon.empty:
        ag = afternoon.groupby("trade_date")
        results["pm_open"] = ag["open"].first()
        results["pm_close"] = ag["close"].last()
        results["pm_high"] = ag["high"].max()
        results["pm_low"] = ag["low"].min()
        results["pm_vol"] = ag["vol"].sum()
        results["pm_amount"] = ag["amount"].sum()
        results["pm_vwap"] = results["pm_amount"] / results["pm_vol"].replace(0, np.nan)
    else:
        for k in _pm_keys:
            results[k] = pd.Series(np.nan, index=daily_open.index)

    # ── Opening 30 min (09:30-10:00) ──────────────────────────────────────
    open_30 = df[(df["minute"] >= 570) & (df["minute"] <= 600)]
    _o30_keys = ["open_30_high", "open_30_low", "open_30_vol", "open_30_amount", "open_30_close"]
    if not open_30.empty:
        og = open_30.groupby("trade_date")
        results["open_30_high"] = og["high"].max()
        results["open_30_low"] = og["low"].min()
        results["open_30_vol"] = og["vol"].sum()
        results["open_30_amount"] = og["amount"].sum()
        results["open_30_close"] = og["close"].last()
    else:
        for k in _o30_keys:
            results[k] = pd.Series(np.nan, index=daily_open.index)

    # ── Closing 30 min (14:30-15:00) ──────────────────────────────────────
    close_30 = df[(df["minute"] >= 870) & (df["minute"] <= 900)]
    _c30_keys = ["close_30_high", "close_30_low", "close_30_vol", "close_30_amount",
                  "close_30_open", "close_30_close"]
    if not close_30.empty:
        cg = close_30.groupby("trade_date")
        results["close_30_high"] = cg["high"].max()
        results["close_30_low"] = cg["low"].min()
        results["close_30_vol"] = cg["vol"].sum()
        results["close_30_amount"] = cg["amount"].sum()
        results["close_30_open"] = cg["open"].first()
        results["close_30_close"] = cg["close"].last()
    else:
        for k in _c30_keys:
            results[k] = pd.Series(np.nan, index=daily_open.index)

    # ── 5-minute realised volatility  (vectorized, was apply-lambda) ─────
    df["close_5min_ago"] = df.groupby("trade_date")["close"].shift(5)
    df["ret_5min"] = df["close"] / df["close_5min_ago"].replace(0, np.nan) - 1.0
    df["ret_5min_sq"] = df["ret_5min"] ** 2
    results["rv_5min"] = np.sqrt(df.groupby("trade_date")["ret_5min_sq"].sum())

    # ── 15-minute realised volatility ─────────────────────────────────────
    df["close_15min_ago"] = df.groupby("trade_date")["close"].shift(15)
    df["ret_15min"] = df["close"] / df["close_15min_ago"].replace(0, np.nan) - 1.0
    df["ret_15min_sq"] = df["ret_15min"] ** 2
    results["rv_15min"] = np.sqrt(df.groupby("trade_date")["ret_15min_sq"].sum())

    # ── Bipower variation & realised jump  (vectorized) ───────────────────
    df["abs_ret_5min"] = df["ret_5min"].abs()
    df["abs_ret_5min_lag"] = df.groupby("trade_date")["abs_ret_5min"].shift(1)
    df["bv_term"] = df["abs_ret_5min"] * df["abs_ret_5min_lag"]
    bv_5min = (np.pi / 2) * df.groupby("trade_date")["bv_term"].sum()
    results["bv_5min"] = bv_5min
    rj = results["rv_5min"] ** 2 - bv_5min
    rj = rj.clip(lower=0)
    results["rj_5min"] = np.sqrt(rj)

    # ── Amihud illiquidity (5-min)  (vectorized) ──────────────────────────
    df["amihud_term"] = df["ret_5min"].abs() / df["amount"].replace(0, np.nan)
    results["amihud_5min"] = df.groupby("trade_date")["amihud_term"].mean()

    # ── 5-min return skewness (for rv_skew_intraday) ──────────────────────
    results["ret_5min_skew"] = df.groupby("trade_date")["ret_5min"].skew()

    # ── Volume concentration ──────────────────────────────────────────────
    first_last_vol = (
        results.get("open_30_vol", pd.Series(0, index=daily_vol.index)).fillna(0)
        + results.get("close_30_vol", pd.Series(0, index=daily_vol.index)).fillna(0)
    )
    results["vol_concentration"] = first_last_vol / daily_vol.replace(0, np.nan)

    # ── AM/PM volume ratio ────────────────────────────────────────────────
    results["am_pm_vol_ratio"] = (
        results.get("am_vol", pd.Series(np.nan, index=daily_vol.index))
        / results.get("pm_vol", pd.Series(np.nan, index=daily_vol.index)).replace(0, np.nan)
    )

    # ── High-low range ───────────────────────────────────────────────────
    results["hl_range"] = (daily_high / daily_low.replace(0, np.nan)) - 1.0

    # ── Open auction return ───────────────────────────────────────────────
    results["open_auction_ret"] = daily_open / daily_close.shift(1).replace(0, np.nan) - 1.0

    # ── Lunch break effect ────────────────────────────────────────────────
    results["lunch_break_ret"] = (
        results.get("pm_open", pd.Series(np.nan, index=daily_open.index))
        / results.get("am_close", pd.Series(np.nan, index=daily_open.index)).replace(0, np.nan)
        - 1.0
    )

    # ── Intraday momentum (open_30 close / open - 1) ──────────────────────
    results["intra_mom_ret"] = (
        results.get("open_30_close", pd.Series(np.nan, index=daily_open.index))
        / daily_open.replace(0, np.nan) - 1.0
    )

    # ── Intraday reversal (close_30 close / close_30 open - 1) ────────────
    results["intra_rev_ret"] = (
        results.get("close_30_close", pd.Series(np.nan, index=daily_open.index))
        / results.get("close_30_open", pd.Series(np.nan, index=daily_open.index)).replace(0, np.nan)
        - 1.0
    )

    # ── VWAP deviation ───────────────────────────────────────────────────
    results["vwap_dev"] = (daily_close - results["vwap"]) / results["vwap"].replace(0, np.nan)

    # ── VWAP momentum (5d / 20d) ──────────────────────────────────────────
    vwap_s = results["vwap"]
    vwap_5d = vwap_s.rolling(5, min_periods=3).mean()
    vwap_20d = vwap_s.rolling(20, min_periods=10).mean()
    results["vwap_mom_5d"] = vwap_5d / vwap_20d.replace(0, np.nan) - 1.0

    # ── Afternoon momentum ────────────────────────────────────────────────
    results["pm_momentum"] = (
        results.get("pm_close", pd.Series(np.nan, index=daily_open.index))
        / results.get("pm_open", pd.Series(np.nan, index=daily_open.index)).replace(0, np.nan)
        - 1.0
    )

    # ── 10-minute realised volatility ─────────────────────────────────────
    df["close_10min_ago"] = df.groupby("trade_date")["close"].shift(10)
    df["ret_10min"] = df["close"] / df["close_10min_ago"].replace(0, np.nan) - 1.0
    df["ret_10min_sq"] = df["ret_10min"] ** 2
    results["rv_10min"] = np.sqrt(df.groupby("trade_date")["ret_10min_sq"].sum())

    # ── 30-minute realised volatility ─────────────────────────────────────
    df["close_30min_ago"] = df.groupby("trade_date")["close"].shift(30)
    df["ret_30min"] = df["close"] / df["close_30min_ago"].replace(0, np.nan) - 1.0
    df["ret_30min_sq"] = df["ret_30min"] ** 2
    results["rv_30min"] = np.sqrt(df.groupby("trade_date")["ret_30min_sq"].sum())

    # ── Realised semivariance (downside only, Barndorff-Nielsen et al. 2010) ─
    df["ret_5min_neg_sq"] = df["ret_5min"].clip(upper=0) ** 2
    results["rsv_5min"] = np.sqrt(df.groupby("trade_date")["ret_5min_neg_sq"].sum())

    # ── Parkinson range-based volatility (Parkinson 1980) ──────────────────
    _hl_ratio = daily_high / daily_low.replace(0, np.nan)
    results["parkinson_vol"] = np.sqrt(
        np.log(_hl_ratio.where(_hl_ratio > 0, np.nan)) ** 2 / (4.0 * np.log(2.0))
    )

    # ── Garman-Klass OHLC volatility (Garman & Klass 1980) ─────────────────
    _co_ratio = daily_close / daily_open.replace(0, np.nan)
    log_hl = np.log(_hl_ratio.where(_hl_ratio > 0, np.nan))
    log_co = np.log(_co_ratio.where(_co_ratio > 0, np.nan))
    gk_var = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    results["gk_vol"] = np.sqrt(gk_var.clip(lower=0))

    # ── Close position: where close falls within day's range (0=low, 1=high) ─
    results["close_position"] = (daily_close - daily_low) / (
        daily_high - daily_low
    ).replace(0, np.nan)

    # ── Relative spread: intraday range normalised by VWAP ─────────────────
    results["relative_spread"] = (daily_high - daily_low) / results["vwap"].replace(
        0, np.nan
    )

    # ── Kyle's lambda style price impact: |ΔP| / volume ────────────────────
    results["price_impact"] = (daily_close - daily_open).abs() / daily_vol.replace(
        0, np.nan
    )

    # ── Volume stability: std(5-min vols) / mean(5-min vols) ───────────────
    df["minute_5min"] = (df["minute"] // 5) * 5
    vol_5min = df.groupby(["trade_date", "minute_5min"])["vol"].sum()
    vol_5min_std = vol_5min.groupby("trade_date").std()
    vol_5min_mean = vol_5min.groupby("trade_date").mean()
    results["vol_stability"] = vol_5min_std / vol_5min_mean.replace(0, np.nan)

    # ── AM volume share: morning volume fraction ───────────────────────────
    results["am_vol_share"] = (
        results.get("am_vol", pd.Series(np.nan, index=daily_vol.index))
        / daily_vol.replace(0, np.nan)
    )

    # ── Intraday trend: fraction of positive 5-min returns ─────────────────
    df["ret_5min_pos"] = (df["ret_5min"] > 0).astype(float)
    results["intra_trend"] = df.groupby("trade_date")["ret_5min_pos"].mean()

    # ── RV trend: 5d MA / 20d MA of rv_5min ───────────────────────────────
    rv_5min_s = results["rv_5min"]
    rv_5min_ma5 = rv_5min_s.rolling(5, min_periods=3).mean()
    rv_5min_ma20 = rv_5min_s.rolling(20, min_periods=10).mean()
    results["rv_trend_5d"] = rv_5min_ma5 / rv_5min_ma20.replace(0, np.nan) - 1.0

    # ── Range / RV ratio: daily range relative to continuous volatility ────
    results["range_rv_ratio"] = results["hl_range"] / results["rv_5min"].replace(
        0, np.nan
    )

    # ── Volume / RV ratio: volume per unit of realised risk ────────────────
    results["volume_rv_ratio"] = np.log(daily_vol.replace(0, np.nan)) / results[
        "rv_5min"
    ].replace(0, np.nan)

    # ── Absolute overnight gap magnitude ───────────────────────────────────
    results["gap_abs"] = (
        daily_open / daily_close.shift(1).replace(0, np.nan) - 1.0
    ).abs()

    # ── Statistical: 5-min return standard deviation ───────────────────────
    results["ret_5min_std"] = df.groupby("trade_date")["ret_5min"].std()

    # ── Statistical: 5-min return kurtosis (tail-fatness) ───────────────────
    ret_5min_mean = df.groupby("trade_date")["ret_5min"].transform("mean")
    ret_demean = df["ret_5min"] - ret_5min_mean
    df["ret_demean_sq"] = ret_demean ** 2
    df["ret_demean_4"] = ret_demean ** 4
    m2 = df.groupby("trade_date")["ret_demean_sq"].mean()
    m4 = df.groupby("trade_date")["ret_demean_4"].mean()
    results["ret_5min_kurt"] = m4 / (m2 ** 2).replace(0, np.nan)

    # ── Statistical: 5-min return first-order autocorrelation ──────────────
    df["ret_5min_lag1"] = df.groupby("trade_date")["ret_5min"].shift(1)
    df["ret_ac_prod"] = df["ret_5min"] * df["ret_5min_lag1"]
    results["ret_autocorr"] = df.groupby("trade_date")["ret_ac_prod"].sum() / df.groupby(
        "trade_date"
    )["ret_5min_sq"].sum().replace(0, np.nan)

    # ── Statistical: volatility of volatility (CV of |ret_5min|) ───────────
    df["abs_ret_5min"] = df["ret_5min"].abs()
    abs_ret_mean = df.groupby("trade_date")["abs_ret_5min"].mean()
    abs_ret_std = df.groupby("trade_date")["abs_ret_5min"].std()
    results["vol_of_vol"] = abs_ret_std / abs_ret_mean.replace(0, np.nan)

    # ── Statistical: maximum absolute 5-min return ─────────────────────────
    results["max_abs_ret_5min"] = df.groupby("trade_date")["abs_ret_5min"].max()

    # ── Statistical: positive RV fraction (variance from up moves) ─────────
    df["ret_5min_pos_sq"] = df["ret_5min"].clip(lower=0) ** 2
    pos_rv_sq = df.groupby("trade_date")["ret_5min_pos_sq"].sum()
    results["pos_rv_ratio"] = pos_rv_sq / df.groupby("trade_date")[
        "ret_5min_sq"
    ].sum().replace(0, np.nan)

    # ── Statistical: Morning vs Afternoon RV ratio ─────────────────────────
    morning_rv_mask = (df["minute"] >= 575) & (df["minute"] <= 690)
    afternoon_rv_mask = (df["minute"] >= 785) & (df["minute"] <= 900)
    am_rv_sq = df.loc[morning_rv_mask].groupby("trade_date")["ret_5min_sq"].sum()
    pm_rv_sq = df.loc[afternoon_rv_mask].groupby("trade_date")["ret_5min_sq"].sum()
    results["am_rv_5min"] = np.sqrt(am_rv_sq.reindex(daily_open.index).fillna(0))
    am_rv_sq_aligned = am_rv_sq.reindex(daily_open.index).fillna(0)
    pm_rv_sq_aligned = pm_rv_sq.reindex(daily_open.index).fillna(0)
    results["am_pm_rv_ratio"] = am_rv_sq_aligned / pm_rv_sq_aligned.replace(0, np.nan)

    # ── Statistical: realized quarticity (variance of variance) ────────────
    n_obs = df.groupby("trade_date").size()
    results["realized_quarticity"] = (
        (n_obs / 3.0)
        * df.groupby("trade_date")["ret_demean_4"].sum()
        / (df.groupby("trade_date")["ret_demean_sq"].sum() ** 2).replace(0, np.nan)
    )

    # ── 60-minute realised volatility ─────────────────────────────────────
    df["close_60min_ago"] = df.groupby("trade_date")["close"].shift(60)
    df["ret_60min"] = df["close"] / df["close_60min_ago"].replace(0, np.nan) - 1.0
    df["ret_60min_sq"] = df["ret_60min"] ** 2
    results["rv_60min"] = np.sqrt(df.groupby("trade_date")["ret_60min_sq"].sum())

    # ── Daily RV (all 1-min returns) ──────────────────────────────────────
    df["ret_1min"] = df["close"] / df.groupby("trade_date")["close"].shift(1).replace(0, np.nan) - 1.0
    df["ret_1min_sq"] = df["ret_1min"] ** 2
    results["rv_daily"] = np.sqrt(df.groupby("trade_date")["ret_1min_sq"].sum())

    # ── Daily bipower variation ───────────────────────────────────────────
    df["abs_ret_1min"] = df["ret_1min"].abs()
    df["abs_ret_1min_lag"] = df.groupby("trade_date")["abs_ret_1min"].shift(1)
    df["bv_daily_term"] = df["abs_ret_1min"] * df["abs_ret_1min_lag"]
    bv_daily = (np.pi / 2) * df.groupby("trade_date")["bv_daily_term"].sum()
    results["bv_daily"] = bv_daily
    rj_daily = results["rv_daily"] ** 2 - bv_daily
    rj_daily = rj_daily.clip(lower=0)
    results["rjump_daily"] = np.sqrt(rj_daily)

    # ── Upside/Downside semivariance (daily) ──────────────────────────────
    df["ret_1min_pos_sq"] = df["ret_1min"].clip(lower=0) ** 2
    df["ret_1min_neg_sq"] = df["ret_1min"].clip(upper=0) ** 2
    results["rv_semi_up"] = np.sqrt(df.groupby("trade_date")["ret_1min_pos_sq"].sum())
    results["rv_semi_down"] = np.sqrt(df.groupby("trade_date")["ret_1min_neg_sq"].sum())

    # ── Volume profile skew & kurtosis ────────────────────────────────────
    df["minute_5min_bucket"] = (df["minute"] // 5) * 5
    vol_5min_bucket = df.groupby(["trade_date", "minute_5min_bucket"])["vol"].sum()
    vol_5min_mean_b = vol_5min_bucket.groupby("trade_date").mean()
    vol_5min_std_b = vol_5min_bucket.groupby("trade_date").std()
    vol_dev = (vol_5min_bucket - vol_5min_mean_b) / vol_5min_std_b.replace(0, np.nan)
    results["volume_profile_skew"] = (vol_dev ** 3).groupby("trade_date").mean()
    results["volume_profile_kurt"] = (vol_dev ** 4).groupby("trade_date").mean()

    # ── Volume peak time (minute of max 5-min volume) ─────────────────────
    vol_5min_wide = vol_5min_bucket.unstack("minute_5min_bucket")
    # Safe idxmax — all-NaN rows raise ValueError
    valid_mask = vol_5min_wide.notna().any(axis=1)
    vol_peak = pd.Series(np.nan, index=vol_5min_wide.index, dtype=float)
    if valid_mask.any():
        vol_peak[valid_mask] = vol_5min_wide.loc[valid_mask].idxmax(axis=1).astype(float)
    results["volume_peak_time"] = vol_peak

    # ── Relative volume by session ────────────────────────────────────────
    first_hour_mask = (df["minute"] >= 570) & (df["minute"] <= 630)
    last_hour_mask = (df["minute"] >= 840) & (df["minute"] <= 900)
    midday_mask = (df["minute"] >= 660) & (df["minute"] <= 810)
    results["rel_vol_first_hour"] = df.loc[first_hour_mask].groupby("trade_date")["vol"].sum() / daily_vol.replace(0, np.nan)
    results["rel_vol_last_hour"] = df.loc[last_hour_mask].groupby("trade_date")["vol"].sum() / daily_vol.replace(0, np.nan)
    results["rel_vol_midday"] = df.loc[midday_mask].groupby("trade_date")["vol"].sum() / daily_vol.replace(0, np.nan)

    # ── Path efficiency: |close - open| / (high - low) ────────────────────
    results["path_efficiency"] = (daily_close - daily_open).abs() / (daily_high - daily_low).replace(0, np.nan)

    # ── Intraday max drawdown (from peak) ─────────────────────────────────
    cum_max = df.groupby("trade_date")["close"].cummax()
    df["drawdown"] = df["close"] / cum_max.replace(0, np.nan) - 1.0
    results["intraday_max_drawdown"] = df.groupby("trade_date")["drawdown"].min()

    # ── Intraday max runup (from trough) ──────────────────────────────────
    cum_min = df.groupby("trade_date")["close"].cummin()
    df["runup"] = df["close"] / cum_min.replace(0, np.nan) - 1.0
    results["intraday_max_runup"] = df.groupby("trade_date")["runup"].max()

    # ── Close auction impact (last 3 minutes return) ──────────────────────
    close_3min = df[df["minute"] >= 897]
    if not close_3min.empty:
        c3g = close_3min.groupby("trade_date")
        results["close_auction_impact"] = c3g["close"].last() / c3g["open"].first().replace(0, np.nan) - 1.0
    else:
        results["close_auction_impact"] = pd.Series(np.nan, index=daily_open.index)

    # ── AM/PM return ratio ────────────────────────────────────────────────
    am_ret = results.get("am_close", pd.Series(np.nan, index=daily_open.index)) / results.get("am_open", pd.Series(np.nan, index=daily_open.index)).replace(0, np.nan) - 1.0
    pm_ret = results.get("pm_close", pd.Series(np.nan, index=daily_open.index)) / results.get("pm_open", pd.Series(np.nan, index=daily_open.index)).replace(0, np.nan) - 1.0
    results["am_pm_return_ratio"] = am_ret / pm_ret.replace(0, np.nan)

    # ── Return asymmetry: (mean - median) of 5-min returns ────────────────
    ret_5min_mean2 = df.groupby("trade_date")["ret_5min"].mean()
    ret_5min_median = df.groupby("trade_date")["ret_5min"].median()
    results["return_asymmetry"] = ret_5min_mean2 - ret_5min_median

    # ── Up minutes ratio ──────────────────────────────────────────────────
    df["ret_1min_pos_flag"] = (df["ret_1min"] > 0).astype(float)
    results["up_minutes_ratio"] = df.groupby("trade_date")["ret_1min_pos_flag"].mean()

    # ── Volume-weighted return ────────────────────────────────────────────
    df["vw_ret_term"] = df["ret_1min"] * df["vol"]
    results["volume_weighted_ret"] = df.groupby("trade_date")["vw_ret_term"].sum() / daily_vol.replace(0, np.nan)

    # ── Realized spread: mean |ret_5min| ──────────────────────────────────
    results["realized_spread_5min"] = df.groupby("trade_date")["abs_ret_5min"].mean()

    # ── Price impact asymmetry: |ret|/vol for up vs down ──────────────────
    df["impact_up"] = df["ret_1min"].clip(lower=0).abs() / df["vol"].replace(0, np.nan)
    df["impact_down"] = df["ret_1min"].clip(upper=0).abs() / df["vol"].replace(0, np.nan)
    impact_up_mean = df.groupby("trade_date")["impact_up"].mean()
    impact_down_mean = df.groupby("trade_date")["impact_down"].mean()
    results["price_impact_asymmetry"] = impact_up_mean / impact_down_mean.replace(0, np.nan)

    # ── Liquidity resilience: vol autocorrelation after extreme return ────
    extreme_mask = df["abs_ret_5min"] > df.groupby("trade_date")["abs_ret_5min"].transform(lambda x: x.rolling(20, min_periods=5).mean() + 2 * x.rolling(20, min_periods=5).std())

    def _safe_autocorr(x):
        """Autocorrelation with manual Pearson computation (avoids np.corrcoef)."""
        x = x.dropna()
        if len(x) <= 5:
            return np.nan
        x_lag = x.shift(1)
        # Keep only overlapping (non-NaN) pairs after the shift
        mask = x.notna() & x_lag.notna()
        n = mask.sum()
        if n <= 5:
            return np.nan
        a = x.loc[mask]
        b = x_lag.loc[mask]
        a_c = a - a.mean()
        b_c = b - b.mean()
        cov = (a_c * b_c).sum()
        denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
        if denom == 0:
            return np.nan
        return cov / denom

    results["liquidity_resilience"] = -df.loc[extreme_mask].groupby("trade_date")["vol"].apply(_safe_autocorr)

    # ── Turnover concentration: top 30 min volume / total ─────────────────
    vol_5min_top = vol_5min_bucket.groupby("trade_date").max()
    results["turnover_concentration"] = vol_5min_top / daily_vol.replace(0, np.nan)

    # ── Flash crash risk: max 5-min drawdown ──────────────────────────────
    df["ret_5min_cum"] = df.groupby("trade_date")["ret_5min"].cumsum()
    df["ret_5min_peak"] = df.groupby("trade_date")["ret_5min_cum"].cummax()
    df["ret_5min_drawdown"] = df["ret_5min_cum"] - df["ret_5min_peak"]
    results["flash_crash_risk"] = -df.groupby("trade_date")["ret_5min_drawdown"].min()

    # ── Extreme move count: |ret_5min| > 3σ within the day ────────────────
    ret_5min_std_daily = df.groupby("trade_date")["ret_5min"].transform("std")
    df["is_extreme"] = (df["abs_ret_5min"] > 3 * ret_5min_std_daily).astype(float)
    results["extreme_move_count"] = df.groupby("trade_date")["is_extreme"].sum()

    # ── Jump ratio: rjump_daily / rv_daily ────────────────────────────────
    results["jump_ratio"] = results["rjump_daily"] / results["rv_daily"].replace(0, np.nan)

    # ── Volume U-shape score ──────────────────────────────────────────────
    u_template = np.array([1.5, 1.0, 0.7, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.2, 2.0,
                           1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.6, 0.8, 1.0, 1.3, 1.7])
    n_buckets = len(u_template)
    vol_bucket_norm = vol_5min_wide.div(vol_5min_wide.mean(axis=1), axis=0)
    # Take up to n_buckets columns
    common_buckets = vol_bucket_norm.columns[:n_buckets]
    vol_arr = vol_bucket_norm[common_buckets].values
    template_arr = u_template[:len(common_buckets)]
    # Correlation of each day's vol profile with U-template
    vol_centered = vol_arr - vol_arr.mean(axis=1, keepdims=True)
    template_centered = template_arr - template_arr.mean()
    cov = (vol_centered * template_centered).sum(axis=1)
    var_vol = (vol_centered ** 2).sum(axis=1)
    var_template = (template_centered ** 2).sum()
    results["volume_u_shape_score"] = pd.Series(
        cov / np.sqrt(var_vol * var_template + 1e-10), index=vol_bucket_norm.index
    )

    # ── Hourly RV decomposition ────────────────────────────────────────────
    for h, (start_m, end_m, label) in enumerate([
        (570, 630, "rv_hourly_1"), (630, 690, "rv_hourly_2"),
        (780, 840, "rv_hourly_3"), (840, 900, "rv_hourly_4"),
    ]):
        hour_mask = (df["minute"] >= start_m) & (df["minute"] <= end_m)
        hour_rv = np.sqrt(df.loc[hour_mask].groupby("trade_date")["ret_1min_sq"].sum())
        results[label] = hour_rv.reindex(daily_open.index)

    # ── Hourly RV dispersion ──────────────────────────────────────────────
    hourly_cols = ["rv_hourly_1", "rv_hourly_2", "rv_hourly_3", "rv_hourly_4"]
    hourly_df = pd.DataFrame({c: results[c] for c in hourly_cols})
    results["rv_hourly_dispersion"] = hourly_df.std(axis=1) / hourly_df.mean(axis=1).replace(0, np.nan)

    # ── Abs autocorrelation (market quality proxy) ────────────────────────
    df["abs_ret_5min_lag1"] = df.groupby("trade_date")["abs_ret_5min"].shift(1)
    df["abs_ac_prod"] = df["abs_ret_5min"] * df["abs_ret_5min_lag1"]
    df["abs_ret_5min_sq"] = df["abs_ret_5min"] ** 2
    results["ret_autocorr_abs"] = df.groupby("trade_date")["abs_ac_prod"].sum() / df.groupby(
        "trade_date"
    )["abs_ret_5min_sq"].sum().replace(0, np.nan)

    # ── AM/PM HL range ────────────────────────────────────────────────────
    results["am_hl_range"] = (
        results.get("am_high", pd.Series(np.nan, index=daily_open.index))
        / results.get("am_low", pd.Series(np.nan, index=daily_open.index)).replace(0, np.nan)
    ) - 1.0
    results["pm_hl_range"] = (
        results.get("pm_high", pd.Series(np.nan, index=daily_open.index))
        / results.get("pm_low", pd.Series(np.nan, index=daily_open.index)).replace(0, np.nan)
    ) - 1.0

    # ── Opening/Closing 5-min momentum ────────────────────────────────────
    open_5 = df[(df["minute"] >= 570) & (df["minute"] <= 575)]
    close_5 = df[(df["minute"] >= 895) & (df["minute"] <= 900)]
    if not open_5.empty:
        o5g = open_5.groupby("trade_date")
        results["open_5min_momentum"] = o5g["close"].last() / o5g["open"].first().replace(0, np.nan) - 1.0
    else:
        results["open_5min_momentum"] = pd.Series(np.nan, index=daily_open.index)
    if not close_5.empty:
        c5g = close_5.groupby("trade_date")
        results["close_5min_momentum"] = c5g["close"].last() / c5g["open"].first().replace(0, np.nan) - 1.0
    else:
        results["close_5min_momentum"] = pd.Series(np.nan, index=daily_open.index)

    # ── Cumulative return path efficiency ─────────────────────────────────
    # Ratio of |total return| / sum(|ret_1min|)
    total_ret = daily_close / daily_open.replace(0, np.nan) - 1.0
    sum_abs_ret = df.groupby("trade_date")["abs_ret_1min"].sum()
    results["cumulative_ret_path"] = total_ret.abs() / sum_abs_ret.replace(0, np.nan)

    # ── PM momentum ───────────────────────────────────────────────────────
    results["pm_momentum"] = results.get("pm_close", pd.Series(np.nan, index=daily_open.index)) / results.get("pm_open", pd.Series(np.nan, index=daily_open.index)).replace(0, np.nan) - 1.0

    # ── RV rolling std (vol-of-vol at 5-daily level) ──────────────────────
    rv_5min_s2 = results["rv_5min"]
    results["rv_rolling_5d_std"] = rv_5min_s2.rolling(5, min_periods=3).std()

    # ── RV term structure slope: rv_5min / rv_60min ───────────────────────
    results["rv_term_structure_slope"] = results["rv_5min"] / results["rv_60min"].replace(0, np.nan) - 1.0

    return pd.DataFrame(results, index=daily_open.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified intraday builder  (single pass, all 14 factors)
# ═══════════════════════════════════════════════════════════════════════════════

#: Map factor name → (metric_column, direction)
#: direction="neg" → ``cross_sectional_rank(-value)`` (higher raw = lower rank).
INTRADAY_FACTOR_SPEC: dict[str, tuple[str, str]] = {
    # ── Realised volatility (existing) ─────────────────────────────────
    "rv_5min":                ("rv_5min",           "neg"),
    "rv_15min":               ("rv_15min",          "neg"),
    "rv_10min":               ("rv_10min",          "neg"),
    "rv_30min":               ("rv_30min",          "neg"),
    "rjump_5min":             ("rj_5min",           "neg"),
    "rsv_5min":               ("rsv_5min",          "neg"),
    # ── Range-based volatility estimators ──────────────────────────────
    "parkinson_vol":          ("parkinson_vol",     "neg"),
    "gk_vol":                 ("gk_vol",            "neg"),
    # ── Liquidity / price impact ───────────────────────────────────────
    "amihud_intraday":        ("amihud_5min",       "neg"),
    "price_impact_intraday":  ("price_impact",      "neg"),
    # ── Volume dynamics ────────────────────────────────────────────────
    "vol_concentration":      ("vol_concentration", "neg"),
    "am_pm_vol_ratio":        ("am_pm_vol_ratio",   "pos"),
    "am_vol_share":           ("am_vol_share",      "pos"),
    "vol_stability":          ("vol_stability",     "neg"),
    # ── Price pattern ──────────────────────────────────────────────────
    "hl_range_intraday":      ("hl_range",          "neg"),
    "close_position":         ("close_position",    "pos"),
    "relative_spread":        ("relative_spread",   "neg"),
    "open_auction_ret":       ("open_auction_ret",  "pos"),
    "gap_abs_intraday":       ("gap_abs",           "neg"),
    "lunch_break_effect":     ("lunch_break_ret",   "pos"),
    # ── Intraday return dynamics ───────────────────────────────────────
    "intraday_momentum":      ("intra_mom_ret",     "pos"),
    "intraday_reversal":      ("intra_rev_ret",     "neg"),
    "rv_skew_intraday":       ("ret_5min_skew",     "pos"),
    "intra_trend":            ("intra_trend",       "pos"),
    # ── VWAP-based ─────────────────────────────────────────────────────
    "vwap_deviation":         ("vwap_dev",          "pos"),
    "vwap_momentum_5d":       ("vwap_mom_5d",       "pos"),
    # ── Composite / ratio ──────────────────────────────────────────────
    "rv_trend_5d":            ("rv_trend_5d",       "neg"),
    "range_rv_ratio":         ("range_rv_ratio",    "neg"),
    "volume_rv_ratio":        ("volume_rv_ratio",   "pos"),
    # ── Statistical / distributional ────────────────────────────────────
    "ret_std_intraday":       ("ret_5min_std",      "neg"),
    "ret_kurt_intraday":      ("ret_5min_kurt",     "neg"),
    "ret_autocorr_5min":      ("ret_autocorr",      "pos"),
    "vol_of_vol_intraday":    ("vol_of_vol",        "neg"),
    "max_ret_intraday":       ("max_abs_ret_5min",  "neg"),
    "pos_rv_ratio":           ("pos_rv_ratio",      "pos"),
    "am_pm_rv_ratio":         ("am_pm_rv_ratio",    "pos"),
    "rq_intraday":            ("realized_quarticity","neg"),
    # ── New: extended RV estimators ─────────────────────────────────────
    "rv_60min":               ("rv_60min",          "neg"),
    "rv_daily":               ("rv_daily",          "neg"),
    "bv_daily":               ("bv_daily",          "neg"),
    "rjump_daily":            ("rjump_daily",       "neg"),
    "rv_semi_up":             ("rv_semi_up",        "pos"),
    "rv_semi_down":           ("rv_semi_down",      "neg"),
    # ── New: volume distribution ────────────────────────────────────────
    "volume_profile_skew":    ("volume_profile_skew", "neg"),
    "volume_profile_kurt":    ("volume_profile_kurt", "neg"),
    "volume_peak_time":       ("volume_peak_time",  "neg"),
    "rel_vol_first_hour":     ("rel_vol_first_hour", "pos"),
    "rel_vol_last_hour":      ("rel_vol_last_hour", "neg"),
    "rel_vol_midday":         ("rel_vol_midday",    "neg"),
    # ── New: price path ─────────────────────────────────────────────────
    "path_efficiency":        ("path_efficiency",   "pos"),
    "intraday_max_drawdown":  ("intraday_max_drawdown","neg"),
    "intraday_max_runup":     ("intraday_max_runup","pos"),
    "close_auction_impact":   ("close_auction_impact","pos"),
    "am_pm_return_ratio":     ("am_pm_return_ratio","pos"),
    "return_asymmetry_intraday": ("return_asymmetry","neg"),
    "up_minutes_ratio":       ("up_minutes_ratio",  "pos"),
    "volume_weighted_ret":    ("volume_weighted_ret","pos"),
    # ── New: liquidity & microstructure ─────────────────────────────────
    "realized_spread_5min":   ("realized_spread_5min","neg"),
    "price_impact_asymmetry": ("price_impact_asymmetry","neg"),
    "liquidity_resilience":   ("liquidity_resilience","pos"),
    "turnover_concentration_intraday": ("turnover_concentration","neg"),
    "flash_crash_risk":       ("flash_crash_risk",  "neg"),
    "extreme_move_count":     ("extreme_move_count","neg"),
    "jump_ratio_intraday":    ("jump_ratio",        "neg"),
    "volume_u_shape_score":   ("volume_u_shape_score","neg"),
    # ── New: hourly decomposition ───────────────────────────────────────
    "rv_hourly_1":            ("rv_hourly_1",       "neg"),
    "rv_hourly_4":            ("rv_hourly_4",       "neg"),
    "rv_hourly_dispersion":   ("rv_hourly_dispersion","neg"),
    # ── New: autocorrelation & range ────────────────────────────────────
    "ret_autocorr_abs":       ("ret_autocorr_abs",  "pos"),
    "am_hl_range_intraday":   ("am_hl_range",       "neg"),
    "pm_hl_range_intraday":   ("pm_hl_range",       "neg"),
    "close_5min_momentum":    ("close_5min_momentum","neg"),
    "open_5min_momentum":     ("open_5min_momentum","pos"),
    "cumulative_ret_path":    ("cumulative_ret_path","pos"),
    "pm_momentum_intraday":   ("pm_momentum",       "pos"),
    # ── New: vol-of-vol & term structure ────────────────────────────────
    "rv_rolling_5d_std":      ("rv_rolling_5d_std", "neg"),
    "rv_term_structure_slope":("rv_term_structure_slope","neg"),
}

#: All metric columns produced by ``_intraday_all_metrics`` that map to factors.
_INTRADAY_METRIC_COLS = {metric for metric, _ in INTRADAY_FACTOR_SPEC.values()}


def _make_multiindex_series(values: np.ndarray, dates: pd.Index, code: str,
                            name: str) -> pd.Series:
    """Build a (Date, Code) MultiIndex Series from a date-indexed array."""
    idx = pd.MultiIndex.from_arrays([dates, [code] * len(dates)], names=["Date", "Code"])
    return pd.Series(values, index=idx, name=name)


def _process_stock_batch(batch: list[tuple[str, str]],
                        min_trade_time: str | None = None) -> dict[str, list[pd.Series]]:
    """Process a batch of stock files in a worker process.

    Args:
        batch: list of (code, filepath_str) tuples.
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
            daily = _intraday_all_metrics(stock_df)
        except Exception:
            continue
        for col in _INTRADAY_METRIC_COLS:
            if col not in daily.columns:
                continue
            s = daily[col].dropna()
            if s.empty:
                continue
            accum[col].append(_make_multiindex_series(s.values, s.index, code, col))
    return dict(accum)


# ── New unified builder entry point (--new flag → now the default) ──────────

def build_intraday_new(
    factor_names: list[str],
    paths,                      # ProjectPaths
    force: bool = False,
    max_workers: int | None = None,
) -> list:
    """Build all Class 3 intraday factors in a single pass with enhanced progress.

    Each per-stock 1-min parquet file is opened **once**, all 14 raw metrics are
    computed together, then fanned out to every requested factor.  Multi-stock
    batching runs in parallel via ProcessPoolExecutor.

    The progress display uses a redesigned multi-phase layout:
      Phase 1 — File discovery & validation
      Phase 2 — Parallel metric computation (stocks batched across workers)
      Phase 3 — Factor frame assembly
      Phase 4 — Writing .fea output files

    Returns a list of ``BuildResult`` objects compatible with CLI expectations.
    """
    from ..builder import (
        BuildResult,
        _resolve_effective_end_date,
        _read_factor_max_date,
        get_factor,
    )
    from ..dataset import _load_allowed_codes
    from ..storage import write_factor, write_factor_incremental, ensure_single_factor_frame

    allowed = _load_allowed_codes(paths.stock_pool_file)
    min_dir = paths.source_root / "history_1min"

    if not min_dir.exists():
        raise FileNotFoundError(f"1-min data directory not found: {min_dir}")

    # ── Phase 1 header ───────────────────────────────────────────────────
    total_factors = len(factor_names)
    n_workers = max_workers if max_workers else min(32, (os.cpu_count() or 4))

    print(f"\n{'='*64}")
    print(f"  CLASS 3 — UNIFIED SINGLE-PASS BUILDER")
    print(f"  Factors: {total_factors}   Workers: {n_workers}")
    print(f"  Strategy: open each stock file once, compute all metrics together")
    print(f"{'='*64}")

    # ── Phase 1: File discovery ──────────────────────────────────────────
    t_phase1 = time.perf_counter()
    print(f"\n  Phase 1/4 — File discovery")

    all_files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
    file_map: dict[str, str] = {}
    for fname in all_files:
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if not allowed or code in allowed:
            file_map[code] = str(min_dir / fname)

    total_stocks = len(file_map)
    print(f"  Stocks found: {total_stocks}  (filtered from {len(all_files)} files)")

    if not file_map:
        return [BuildResult(
            factor_name=n, action="skip", elapsed=0.0, rows=0,
            factor_path=paths.factor_output_dir / f"{n}.fea",
            manifest_path=paths.manifest_output_dir / f"{n}.json",
        ) for n in factor_names]

    # ── Determine incremental filter ─────────────────────────────────────
    effective_end = _resolve_effective_end_date(paths.source_root)
    min_trade_time: str | None = None
    earliest_existing: str | None = None
    if not force:
        for name in factor_names:
            fp = paths.factor_output_dir / f"{name}.fea"
            if fp.exists():
                fm = _read_factor_max_date(fp)
                if fm:
                    if earliest_existing is None or fm < earliest_existing:
                        earliest_existing = fm
        if earliest_existing and earliest_existing < effective_end:
            from datetime import datetime as _dt, timedelta as _td
            lookback_dt = _dt.strptime(earliest_existing, "%Y%m%d") - _td(days=30)
            min_trade_time = lookback_dt.strftime("%Y-%m-%d") + " 00:00:00"

    mode_str = "incremental" if min_trade_time else "full rebuild"
    print(f"  Mode: {mode_str}" + (f"  (since {min_trade_time})" if min_trade_time else ""))

    t_phase1_elapsed = time.perf_counter() - t_phase1
    print(f"  Phase 1 done  ({t_phase1_elapsed:.1f}s)")

    # ── Phase 2: Parallel metric computation ─────────────────────────────
    t_phase2 = time.perf_counter()
    print(f"\n  Phase 2/4 — Computing intraday metrics  "
          f"[ProcessPoolExecutor x{n_workers}]")

    BATCH_SIZE = 25
    file_items = list(file_map.items())
    batches = [file_items[i:i + BATCH_SIZE] for i in range(0, len(file_items), BATCH_SIZE)]
    total_batches = len(batches)
    batch_file_counts = [len(b) for b in batches]

    from tqdm import tqdm

    accumulators: dict[str, list[pd.Series]] = defaultdict(list)
    completed_stocks = 0

    pbar = tqdm(
        total=total_stocks, desc="  Stocks", unit="stk",
        bar_format="{desc:>12}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {postfix}]",
    )

    t_batch_start = time.perf_counter()
    batch_errors = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_stock_batch, batch, min_trade_time): idx
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
                    pbar.write(f"  [!] Batch {batch_idx} failed: {type(e).__name__}: {e}")
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
        print(f"  [!] FATAL: all batches failed — no metrics accumulated. "
              f"Check that history_1min/*.parquet files are readable.")
    else:
        print(f"  Accumulated {total_accumulated} metric series across "
              f"{len(accumulators)} columns")

    t_phase2_elapsed = time.perf_counter() - t_phase2
    rate_p2 = total_stocks / t_phase2_elapsed if t_phase2_elapsed > 0 else 0
    print(f"  Phase 2 done  ({t_phase2_elapsed:.1f}s, {rate_p2:.0f} stocks/s)")

    # ── Concatenate raw metrics ──────────────────────────────────────────
    t_concat = time.perf_counter()
    raw_metrics: dict[str, pd.Series] = {}
    for col in _INTRADAY_METRIC_COLS:
        parts = accumulators.get(col, [])
        if parts:
            s = pd.concat(parts)
            s = s.groupby(list(s.index.names)).last()
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
        pbar_factor.update(0)  # refresh display

        if name not in INTRADAY_FACTOR_SPEC:
            output[name] = pd.DataFrame()
            errors_build.append(name)
            pbar_factor.update(1)
            continue

        metric_col, direction = INTRADAY_FACTOR_SPEC[name]
        raw = raw_metrics.get(metric_col)
        if raw is None or raw.empty:
            output[name] = pd.DataFrame()
            pbar_factor.update(1)
            continue

        try:
            if direction == "pos":
                ranked = cross_sectional_rank(raw)
            else:  # "neg"
                ranked = cross_sectional_rank(-raw)
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
            # Full rebuild (min_trade_time is None) → always overwrite.
            if force or not factor_path.exists() or min_trade_time is None:
                write_factor(spec, frame, paths=paths)
            else:
                write_factor_incremental(spec, frame, paths=paths)
            results.append(BuildResult(
                factor_name=name, action="rebuild", elapsed=0.0,
                rows=len(frame), factor_path=factor_path,
                manifest_path=paths.manifest_output_dir / f"{name}.json",
            ))
            pbar_write.set_postfix_str(f"OK {name}")
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
    print(f"  Factors built:     {built}  |  Skipped: {skipped}  |  Errors: {errors_total}")
    print(f"  Phase breakdown:")
    print(f"    Phase 1 (discovery):  {t_phase1_elapsed:>6.1f}s")
    print(f"    Phase 2 (metrics):    {t_phase2_elapsed:>6.1f}s  ({rate_p2:.0f} stocks/s)")
    print(f"    Phase 3 (assembly):   {t_phase3_elapsed:>6.1f}s  ({rate_p3:.1f} factors/s)")
    print(f"    Phase 4 (write):      {t_phase4_elapsed:>6.1f}s  ({rate_p4:.1f} files/s)")
    if errors_total:
        print(f"  Error list: {', '.join(errors_build)}")
    print(f"{'─'*64}\n")

    return results


# ── Public builder entry point (legacy, kept for backward compat) ────────────

def build_intraday_unified(
    paths,                      # ProjectPaths
    factor_names: list[str],
    allowed_codes: set[str],
    force: bool = False,
    on_progress=None,           # (stage: str, current: int, total: int) -> None
    max_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all intraday factors in a **single pass** over per-stock files.

    Uses ProcessPoolExecutor to parallelise the CPU-bound metric computation
    across batches of stocks.  Each stock file is read once and
    ``_intraday_all_metrics`` is called once, producing 48 metric columns that
    are fanned out to the 14 registered intraday factors.

    Returns ``{factor_name: wide_DataFrame}`` ready for ``write_factor``.
    """
    import os as _os
    from ..settings import ProjectPaths
    from ..storage import ensure_single_factor_frame
    from ..builder import _resolve_effective_end_date, _read_factor_max_date

    min_dir = paths.source_root / "history_1min"
    if not min_dir.exists():
        raise FileNotFoundError(f"1-min data directory not found: {min_dir}")

    # ── Determine incremental filter date ────────────────────────────────
    effective_end = _resolve_effective_end_date(paths.source_root)

    # Find the earliest existing max date across all factors being built.
    # This is the "last already-computed date" — we only need data AFTER it.
    min_trade_time: str | None = None
    earliest_existing: str | None = None
    if not force:
        for name in factor_names:
            factor_path = paths.factor_output_dir / f"{name}.fea"
            if factor_path.exists():
                fm = _read_factor_max_date(factor_path)
                if fm:
                    if earliest_existing is None or fm < earliest_existing:
                        earliest_existing = fm
        if earliest_existing and earliest_existing < effective_end:
            # Subtract 30 calendar days for rolling-window lookback
            # (vwap_momentum_5d needs 20 trading days of VWAP history).
            from datetime import datetime as _dt, timedelta as _td
            lookback_dt = _dt.strptime(earliest_existing, "%Y%m%d") - _td(days=30)
            min_trade_time = lookback_dt.strftime("%Y-%m-%d") + " 00:00:00"

    # ── Build file list ──────────────────────────────────────────────────
    all_files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
    file_map: dict[str, str] = {}  # code → filepath_str
    for fname in all_files:
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if not allowed_codes or code in allowed_codes:
            file_map[code] = str(min_dir / fname)

    if not file_map:
        return {n: pd.DataFrame() for n in factor_names}

    total_files = len(file_map)
    if on_progress:
        on_progress("intraday_scan", 0, total_files)

    # ── Determine parallelism ─────────────────────────────────────────────
    if max_workers is None:
        max_workers = min(32, (_os.cpu_count() or 4))
    # Split into batches so each worker gets a meaningful amount of work.
    batch_size = max(1, total_files // max_workers)
    file_items = list(file_map.items())
    batches = [file_items[i:i + batch_size] for i in range(0, len(file_items), batch_size)]

    # ── Process batches in parallel ───────────────────────────────────────
    accumulators: dict[str, list[pd.Series]] = defaultdict(list)
    completed = 0
    t_start = time.perf_counter()
    batch_file_counts = [len(b) for b in batches]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_stock_batch, batch, min_trade_time): idx
            for idx, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            try:
                batch_result = fut.result()
            except Exception:
                completed += batch_file_counts[batch_idx]
                if on_progress:
                    on_progress("intraday_scan", completed, total_files)
                continue
            for col, series_list in batch_result.items():
                accumulators[col].extend(series_list)
            completed += batch_file_counts[batch_idx]
            if on_progress:
                on_progress("intraday_scan", completed, total_files)

    scan_elapsed = time.perf_counter() - t_start

    # ── Concatenate per-stock results ─────────────────────────────────────
    if on_progress:
        on_progress("intraday_concat", 0, len(factor_names))

    raw_metrics: dict[str, pd.Series] = {}
    for col in _INTRADAY_METRIC_COLS:
        parts = accumulators.get(col, [])
        if parts:
            s = pd.concat(parts)
            # Deduplicate on full index; use list() so groupby always sees a
            # plain list of level names (FrozenList may behave differently
            # across pandas versions).
            s = s.groupby(list(s.index.names)).last()
            # Guarantee consistent MultiIndex names for downstream code that
            # calls get_level_values("Date") / groupby(level="Date").
            if isinstance(s.index, pd.MultiIndex):
                s.index = s.index.set_names(["Date", "Code"])
            raw_metrics[col] = s
        else:
            raw_metrics[col] = pd.Series(dtype=float, name=col)

    # ── Pre-filter to new dates (incremental mode) ──────────────────────
    # Filter raw_metrics BEFORE cross_sectional_rank so we only rank
    # dates that are actually new.  Cross-sectional rank is computed
    # per-date, so filtering first is mathematically identical.
    earliest_existing_for_filter = earliest_existing  # may be None (rebuild)
    if not force and earliest_existing_for_filter and min_trade_time is not None:
        for col in list(raw_metrics.keys()):
            s = raw_metrics[col]
            if not s.empty:
                raw_metrics[col] = s[s.index.get_level_values("Date") > earliest_existing_for_filter]

    # ── Build final factor frames ─────────────────────────────────────────
    output: dict[str, pd.DataFrame] = {}
    for idx, name in enumerate(factor_names):
        if on_progress:
            on_progress("intraday_build", idx + 1, len(factor_names))

        if name not in INTRADAY_FACTOR_SPEC:
            continue
        metric_col, direction = INTRADAY_FACTOR_SPEC[name]
        raw = raw_metrics.get(metric_col)
        if raw is None or raw.empty:
            output[name] = pd.DataFrame()
            continue

        ranked = cross_sectional_rank(raw if direction == "pos" else -raw)
        frame = ensure_single_factor_frame(ranked, name)

        # No per-factor file read needed here — the date filter was
        # already applied above (or we're in full-rebuild mode).
        output[name] = frame

    return output


# Backward-compatible helper kept for single-factor builds and
# non-unified code paths.  It now delegates to the unified builder
# when multiple factors are needed, but retains the original loop
# for isolated calls.
def _compute_intraday_factor(source_root: Path, allowed_codes: set[str],
                              metric_col: str, on_progress=None) -> pd.Series:
    """Compute one intraday metric for all stocks from per-stock 1-min files.

    Parameters
    ----------
    source_root : Path
        Root directory containing history_1min/ subdirectory.
    allowed_codes : set
        Set of allowed stock codes (6-digit).
    metric_col : str
        Column name from _intraday_all_metrics to extract.
    on_progress : callable or None

    Returns
    -------
    pd.Series with (Date, Code) MultiIndex.
    """
    import os
    min_dir = source_root / "history_1min"
    if not min_dir.exists():
        raise FileNotFoundError(f"1-min data directory not found: {min_dir}")

    files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
    parts = []
    total = len(files)

    for i, fname in enumerate(files):
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if allowed_codes and code not in allowed_codes:
            if on_progress:
                on_progress("intraday", i + 1, total)
            continue

        filepath = min_dir / fname
        try:
            stock_df = pd.read_parquet(filepath)
        except Exception:
            if on_progress:
                on_progress("intraday", i + 1, total)
            continue

        if stock_df.empty:
            if on_progress:
                on_progress("intraday", i + 1, total)
            continue

        try:
            daily = _intraday_all_metrics(stock_df)
        except Exception:
            if on_progress:
                on_progress("intraday", i + 1, total)
            continue

        if metric_col not in daily.columns:
            if on_progress:
                on_progress("intraday", i + 1, total)
            continue

        s = daily[metric_col].dropna()
        if s.empty:
            if on_progress:
                on_progress("intraday", i + 1, total)
            continue

        s = pd.DataFrame({metric_col: s.values},
                         index=pd.MultiIndex.from_arrays(
                             [s.index, [code] * len(s)],
                             names=["Date", "Code"]
                         ))[metric_col]
        parts.append(s)

        if on_progress:
            on_progress("intraday", i + 1, total)

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


# ── Realized Volatility Factors ─────────────────────────────────────────

@register_factor(
    name="rv_5min",
    description="5分钟已实现波动率因子，基于1分钟数据的5分钟收益平方和开根截面排名（低波排前）。",
    category="intraday",
    thesis="高频已实现波动率比日度波动率精确得多(Andersen&Bollerslev 1998)，捕捉日内信息流驱动的真实波动。低RV股票经风险调整后收益更高。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_5min(context: FactorContext):
    source_root = context.repo.paths.source_root
    rv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_5min",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


@register_factor(
    name="rv_15min",
    description="15分钟已实现波动率因子，基于1分钟数据的15分钟收益平方和开根截面排名（低波排前）。",
    category="intraday",
    thesis="15分钟已实现波动率比5分钟更平滑，减少 microstructure noise，同时比日度波动率更及时。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_15min(context: FactorContext):
    source_root = context.repo.paths.source_root
    rv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_15min",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


@register_factor(
    name="rjump_5min",
    description="已实现跳跃因子，RV_5min² - BV_5min²的正部开根截面排名（高跳跃排后=风险信号）。",
    category="intraday",
    thesis="价格跳跃代表信息冲击或流动性断裂(Barndorff-Nielsen&Shephard 2004)。高跳跃股票面临更大的尾部风险，未来收益的波动率更高且偏负。跳跃强度是传统波动率无法捕捉的风险维度。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rjump_5min(context: FactorContext):
    source_root = context.repo.paths.source_root
    rj = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rj_5min",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rj)


@register_factor(
    name="rv_skew_intraday",
    description="日内已实现偏度因子，5分钟收益的截面偏度排名。正偏=上涨跳跃多，负偏=下跌跳跃多。",
    category="intraday",
    thesis="日内收益偏度反映日内价格路径的不对称性。正偏（多数上涨在下午发生）通常比负偏（恐慌性下跌）的后续表现更好。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_skew_intraday(context: FactorContext):
    """Compute 5-min return skewness from per-stock files."""
    import os
    source_root = context.repo.paths.source_root
    min_dir = source_root / "history_1min"
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress

    files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
    parts = []
    total = len(files)

    for i, fname in enumerate(files):
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if allowed and code not in allowed:
            if on_progress: on_progress("rv_skew", i + 1, total)
            continue

        filepath = min_dir / fname
        try:
            stock_df = pd.read_parquet(filepath)
        except Exception:
            if on_progress: on_progress("rv_skew", i + 1, total)
            continue

        if stock_df.empty:
            if on_progress: on_progress("rv_skew", i + 1, total)
            continue

        stock_df["trade_date"] = pd.to_datetime(stock_df["trade_time"]).dt.strftime("%Y%m%d")
        stock_df["close_5min_ago"] = stock_df.groupby("trade_date")["close"].shift(5)
        stock_df["ret_5min"] = stock_df["close"] / stock_df["close_5min_ago"].replace(0, np.nan) - 1.0
        skew = stock_df.groupby("trade_date")["ret_5min"].skew().dropna()
        if skew.empty:
            if on_progress: on_progress("rv_skew", i + 1, total)
            continue

        s = pd.DataFrame({"skew": skew.values},
                         index=pd.MultiIndex.from_arrays(
                             [skew.index, [code] * len(skew)], names=["Date", "Code"]
                         ))["skew"]
        parts.append(s)

        if on_progress: on_progress("rv_skew", i + 1, total)

    if not parts:
        return cross_sectional_rank(pd.Series(dtype=float))

    result = pd.concat(parts)
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    result = result.groupby(list(result.index.names)).last()
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.set_names(["Date", "Code"])
    return cross_sectional_rank(result)


# ── Intraday Liquidity Factors ──────────────────────────────────────────

@register_factor(
    name="amihud_intraday",
    description="高频Amihud非流动性因子，5分钟|ret|/amount均值截面排名（取负向=高流动性排前）。",
    category="intraday",
    thesis="基于5分钟的高频Amihud比日度版本精确一个数量级，避免了日度数据的Epps效应。高非流动性=高交易成本=需要更高收益补偿。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_amihud_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    amihud = _compute_intraday_factor(source_root, context.repo.allowed_codes, "amihud_5min",
                                       on_progress=context.repo.on_progress)
    return cross_sectional_rank(-amihud)


@register_factor(
    name="hl_range_intraday",
    description="日内振幅因子，1分钟数据得到的日度(high/low-1)截面排名（低振幅=筹码稳定排前）。",
    category="intraday",
    thesis="1分钟数据计算的日内振幅比日线OHLC更精确（排除集合竞价失真）。低日内振幅代表交易结构健康、多空力量均衡。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_hl_range_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    hl = _compute_intraday_factor(source_root, context.repo.allowed_codes, "hl_range",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-hl)


# ── Intraday Momentum / Reversal ────────────────────────────────────────

@register_factor(
    name="intraday_momentum",
    description="日内动量因子，开盘30分钟收益截面排名。衡量隔夜信息消化后的早盘方向。",
    category="intraday",
    thesis="开盘30分钟的方向往往由隔夜信息和集合竞价阶段的市场情绪决定，延续性较强。'开盘定方向'在A股中具有显著的日内动量效应。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_intraday_momentum(context: FactorContext):
    source_root = context.repo.paths.source_root
    # Compute open_30 return: (open_30 close / open) - 1
    import os
    min_dir = source_root / "history_1min"
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress

    files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
    parts = []
    total = len(files)

    for i, fname in enumerate(files):
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if allowed and code not in allowed:
            if on_progress: on_progress("intra_mom", i + 1, total)
            continue

        filepath = min_dir / fname
        try:
            stock_df = pd.read_parquet(filepath)
        except Exception:
            if on_progress: on_progress("intra_mom", i + 1, total)
            continue

        if stock_df.empty:
            if on_progress: on_progress("intra_mom", i + 1, total)
            continue

        stock_df["trade_date"] = pd.to_datetime(stock_df["trade_time"]).dt.strftime("%Y%m%d")
        stock_df["minute"] = pd.to_datetime(stock_df["trade_time"]).dt.hour * 60 + \
                             pd.to_datetime(stock_df["trade_time"]).dt.minute

        grouped = stock_df.groupby("trade_date")
        daily_open = grouped["open"].first()

        open_30 = stock_df[(stock_df["minute"] >= 570) & (stock_df["minute"] <= 600)]
        if open_30.empty:
            if on_progress: on_progress("intra_mom", i + 1, total)
            continue

        open_30_close = open_30.groupby("trade_date")["close"].last()
        ret = open_30_close / daily_open.replace(0, np.nan) - 1.0
        ret = ret.dropna()
        if ret.empty:
            if on_progress: on_progress("intra_mom", i + 1, total)
            continue

        s = pd.DataFrame({"ret": ret.values},
                         index=pd.MultiIndex.from_arrays(
                             [ret.index, [code] * len(ret)], names=["Date", "Code"]
                         ))["ret"]
        parts.append(s)
        if on_progress: on_progress("intra_mom", i + 1, total)

    if not parts:
        return cross_sectional_rank(pd.Series(dtype=float))

    result = pd.concat(parts)
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    result = result.groupby(list(result.index.names)).last()
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.set_names(["Date", "Code"])
    return cross_sectional_rank(result)


@register_factor(
    name="intraday_reversal",
    description="尾盘反转因子，收盘30分钟收益截面排名（取负向=尾盘拉升排后，易次日低开）。",
    category="intraday",
    thesis="尾盘拉升（尤其是最后5-10分钟）往往是主力做收盘价的刻意行为，次日经常低开。尾盘反转效应是A股T+1制度下的特殊alpha来源。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_intraday_reversal(context: FactorContext):
    source_root = context.repo.paths.source_root
    import os
    min_dir = source_root / "history_1min"
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress

    files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
    parts = []
    total = len(files)

    for i, fname in enumerate(files):
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if allowed and code not in allowed:
            if on_progress: on_progress("intra_rev", i + 1, total)
            continue

        filepath = min_dir / fname
        try:
            stock_df = pd.read_parquet(filepath)
        except Exception:
            if on_progress: on_progress("intra_rev", i + 1, total)
            continue

        if stock_df.empty:
            if on_progress: on_progress("intra_rev", i + 1, total)
            continue

        stock_df["trade_date"] = pd.to_datetime(stock_df["trade_time"]).dt.strftime("%Y%m%d")
        stock_df["minute"] = pd.to_datetime(stock_df["trade_time"]).dt.hour * 60 + \
                             pd.to_datetime(stock_df["trade_time"]).dt.minute

        close_30 = stock_df[(stock_df["minute"] >= 870) & (stock_df["minute"] <= 900)]
        if close_30.empty:
            if on_progress: on_progress("intra_rev", i + 1, total)
            continue

        cg = close_30.groupby("trade_date")
        ret = cg["close"].last() / cg["open"].first().replace(0, np.nan) - 1.0
        ret = ret.dropna()
        if ret.empty:
            if on_progress: on_progress("intra_rev", i + 1, total)
            continue

        s = pd.DataFrame({"ret": ret.values},
                         index=pd.MultiIndex.from_arrays(
                             [ret.index, [code] * len(ret)], names=["Date", "Code"]
                         ))["ret"]
        parts.append(s)
        if on_progress: on_progress("intra_rev", i + 1, total)

    if not parts:
        return cross_sectional_rank(pd.Series(dtype=float))

    result = pd.concat(parts)
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    result = result.groupby(list(result.index.names)).last()
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.set_names(["Date", "Code"])
    return cross_sectional_rank(-result)


# ── Lunch break effect (A-share specific) ───────────────────────────────

@register_factor(
    name="lunch_break_effect",
    description="午间休市效应因子，下午开盘价/上午收盘价-1截面排名（高值=午间利好堆积）。",
    category="intraday",
    thesis="A股特有的11:30-13:00午间休市期间信息持续累积，下午开盘跳空幅度反映午间信息的冲击强度。持续的午间正跳空代表信息面偏积极。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_lunch_break_effect(context: FactorContext):
    source_root = context.repo.paths.source_root
    lb = _compute_intraday_factor(source_root, context.repo.allowed_codes, "lunch_break_ret",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(lb)


# ── VWAP Factors ────────────────────────────────────────────────────────

@register_factor(
    name="vwap_deviation",
    description="VWAP偏离因子，(收盘-VWAP)/VWAP截面排名。收盘价相对日均价的位置。",
    category="intraday",
    thesis="收盘价高于VWAP说明尾盘有资金主动推升，买方主导日内的均价水平。持续的VWAP正偏离（收盘>VWAP）是机构主动建仓的信号，尤其在连续多日正偏离后趋势延续性强。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_vwap_deviation(context: FactorContext):
    source_root = context.repo.paths.source_root
    import os
    min_dir = source_root / "history_1min"
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress

    files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
    parts = []
    total = len(files)

    for i, fname in enumerate(files):
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if allowed and code not in allowed:
            if on_progress: on_progress("vwap", i + 1, total)
            continue

        filepath = min_dir / fname
        try:
            stock_df = pd.read_parquet(filepath)
        except Exception:
            if on_progress: on_progress("vwap", i + 1, total)
            continue

        if stock_df.empty:
            if on_progress: on_progress("vwap", i + 1, total)
            continue

        stock_df["trade_date"] = pd.to_datetime(stock_df["trade_time"]).dt.strftime("%Y%m%d")
        grouped = stock_df.groupby("trade_date")
        daily_vol = grouped["vol"].sum()
        daily_amount = grouped["amount"].sum()
        daily_close = grouped["close"].last()
        vwap = daily_amount / daily_vol.replace(0, np.nan)
        dev = (daily_close - vwap) / vwap.replace(0, np.nan)
        dev = dev.dropna()
        if dev.empty:
            if on_progress: on_progress("vwap", i + 1, total)
            continue

        s = pd.DataFrame({"dev": dev.values},
                         index=pd.MultiIndex.from_arrays(
                             [dev.index, [code] * len(dev)], names=["Date", "Code"]
                         ))["dev"]
        parts.append(s)
        if on_progress: on_progress("vwap", i + 1, total)

    if not parts:
        return cross_sectional_rank(pd.Series(dtype=float))

    result = pd.concat(parts)
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    result = result.groupby(list(result.index.names)).last()
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.set_names(["Date", "Code"])
    return cross_sectional_rank(result)


# ── Volume Concentration ────────────────────────────────────────────────

@register_factor(
    name="vol_concentration",
    description="成交量集中度因子，(开盘30分+收盘30分)成交量/全日成交量截面排名（取负向=过于集中排后）。",
    category="intraday",
    thesis="成交量过度集中在开盘和收盘时段（A股经典的U型分布极端化）往往意味着信息不对称严重或主力刻意为之。适中的成交量分布代表自然的交易节奏。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_vol_concentration(context: FactorContext):
    source_root = context.repo.paths.source_root
    vc = _compute_intraday_factor(source_root, context.repo.allowed_codes, "vol_concentration",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-vc)


@register_factor(
    name="am_pm_vol_ratio",
    description="上午/下午成交量比因子截面排名。上午放量=信息消化积极，下午放量=尾盘博弈。",
    category="intraday",
    thesis="上午成交量占比高说明市场对隔夜信息反应积极，通常与正收益相关。下午成交量异常放大（尤其是最后半小时）往往是短期博弈行为，质量较差。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_am_pm_vol_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    ratio = _compute_intraday_factor(source_root, context.repo.allowed_codes, "am_pm_vol_ratio",
                                      on_progress=context.repo.on_progress)
    return cross_sectional_rank(ratio)


# ── Open Auction ────────────────────────────────────────────────────────

@register_factor(
    name="open_auction_ret",
    description="集合竞价收益率因子，(开盘价/前日收盘-1)截面排名。隔夜信息冲击的直接度量。",
    category="intraday",
    thesis="A股集合竞价(9:15-9:25)产生的开盘价包含了隔夜全部信息的综合定价。大幅高开往往延续（动量），大幅低开如果盘中回升则构成反转信号。此因子与overnight_gap互补——使用1分钟首笔数据更精准。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_open_auction_ret(context: FactorContext):
    source_root = context.repo.paths.source_root
    oa = _compute_intraday_factor(source_root, context.repo.allowed_codes, "open_auction_ret",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(oa)


# ── VWAP Momentum ───────────────────────────────────────────────────────

@register_factor(
    name="vwap_momentum_5d",
    description="VWAP动量因子，(5日VWAP均值/20日VWAP均值-1)截面排名。",
    category="intraday",
    thesis="VWAP比收盘价更能代表'真实成交价格'，VWAP动量排除了尾盘操纵的影响，比收盘价动量更纯净地反映资金的平均建仓成本变化方向。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_vwap_momentum_5d(context: FactorContext):
    source_root = context.repo.paths.source_root
    vw = _compute_intraday_factor(source_root, context.repo.allowed_codes, "vwap_mom_5d",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(vw)


# ═══════════════════════════════════════════════════════════════════════════
# Class 3 — Additional intraday factors (15 new, total 29)
# ═══════════════════════════════════════════════════════════════════════════

# ── Extended Realised Volatility ──────────────────────────────────────────

@register_factor(
    name="rv_10min",
    description="10分钟已实现波动率因子，基于1分钟数据的10分钟收益平方和开根截面排名（低波排前）。",
    category="intraday",
    thesis="10分钟频率的RV在5分钟（高噪声）和15分钟（响应慢）之间取得平衡，捕捉中等频率的波动动态。多频RV联合使用可提取波动率期限结构信息。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_10min(context: FactorContext):
    source_root = context.repo.paths.source_root
    rv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_10min",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


@register_factor(
    name="rv_30min",
    description="30分钟已实现波动率因子，基于1分钟数据的30分钟收益平方和开根截面排名（低波排前）。",
    category="intraday",
    thesis="30分钟频率的RV滤除了高频 microstructure noise，更接近'持久波动率'成分。低频RV对跳跃更稳健，与高频RV的差异可反映噪声大小。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_30min(context: FactorContext):
    source_root = context.repo.paths.source_root
    rv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_30min",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


@register_factor(
    name="rsv_5min",
    description="已实现半方差因子（下行风险），仅5分钟负收益的平方和开根截面排名（高下行波排后）。",
    category="intraday",
    thesis="下行已实现半方差(Barndorff-Nielsen et al. 2010)将波动率分解为非对称成分。投资者只厌恶下行波动，上行波动反映正信息冲击。RSV比RV更能捕捉真正的'坏波动'，对尾部风险定价更精准。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rsv_5min(context: FactorContext):
    source_root = context.repo.paths.source_root
    rsv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rsv_5min",
                                    on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rsv)


# ── Range-based Volatility Estimators ─────────────────────────────────────

@register_factor(
    name="parkinson_vol",
    description="Parkinson波动率估计因子，基于日内最高最低价的ln(H/L)/sqrt(4ln2)截面排名（低波排前）。",
    category="intraday",
    thesis="Parkinson(1980)证明基于日内最高最低价的波动率估计量比收盘价波动率效率高5倍。HL范围包含了日内全部价格路径信息，而不仅仅是收盘时点。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_parkinson_vol(context: FactorContext):
    source_root = context.repo.paths.source_root
    pv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "parkinson_vol",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-pv)


@register_factor(
    name="gk_vol",
    description="Garman-Klass波动率估计因子，基于OHLC四价的高效波动率截面排名（低波排前）。",
    category="intraday",
    thesis="Garman-Klass(1980)利用OHLC四价信息的波动率估计量效率是收盘价波动率的7.4倍。相比Parkinson仅用HL，GK额外利用OC信息区分趋势日和震荡日，对开盘跳空更稳健。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_gk_vol(context: FactorContext):
    source_root = context.repo.paths.source_root
    gk = _compute_intraday_factor(source_root, context.repo.allowed_codes, "gk_vol",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-gk)


# ── Price Pattern ─────────────────────────────────────────────────────────

@register_factor(
    name="close_position",
    description="收盘位置因子，(收盘-最低)/(最高-最低)截面排名。收盘在高位=买方主导全日。",
    category="intraday",
    thesis="收盘价在日内价格区间的位置反映了多空力量博弈的最终结果。收盘在区间上沿（close_position≈1）意味着买方在尾盘占据主导，次日延续概率较高。收盘在低位（≈0）则卖方压力较大。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_close_position(context: FactorContext):
    source_root = context.repo.paths.source_root
    cp = _compute_intraday_factor(source_root, context.repo.allowed_codes, "close_position",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(cp)


@register_factor(
    name="relative_spread",
    description="相对价差因子，(最高-最低)/VWAP截面排名（取负向=高振幅排后）。VWAP标准化后的日内波动幅度。",
    category="intraday",
    thesis="以VWAP标准化的日内振幅剔除了价格水平的影响，使得高低价差在不同价格区间的股票之间可比。高相对价差代表日内价格波动剧烈、多空分歧大，通常伴随更高的未来波动和不确定性。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_relative_spread(context: FactorContext):
    source_root = context.repo.paths.source_root
    rs = _compute_intraday_factor(source_root, context.repo.allowed_codes, "relative_spread",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rs)


@register_factor(
    name="gap_abs_intraday",
    description="隔夜跳空绝对值因子，|开盘/前收-1|截面排名（取负向=大跳空排后）。隔夜信息冲击的绝对程度。",
    category="intraday",
    thesis="隔夜跳空的绝对值（不论方向）衡量信息冲击的强度。大幅跳空（无论高开低开）意味着信息不确定性高、定价分歧大。持续的隔夜大幅波动往往伴随更高的未来波动率和更差的风险调整收益。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_gap_abs_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    ga = _compute_intraday_factor(source_root, context.repo.allowed_codes, "gap_abs",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-ga)


# ── Liquidity / Price Impact ──────────────────────────────────────────────

@register_factor(
    name="price_impact_intraday",
    description="价格冲击因子（Kyle's Lambda），|收盘-开盘|/成交量截面排名（取负向=高冲击排后）。",
    category="intraday",
    thesis="Kyle(1985)的价格冲击系数衡量单位成交量引起的价格变动。高价格冲击意味着市场深度浅、流动性差，交易成本高。1分钟数据计算的日内价格冲击比日度版本更能捕捉微观层面的流动性枯竭。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_price_impact_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    pi = _compute_intraday_factor(source_root, context.repo.allowed_codes, "price_impact",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-pi)


# ── Volume Dynamics ───────────────────────────────────────────────────────

@register_factor(
    name="vol_stability",
    description="成交量稳定性因子，5分钟成交量std/均值截面排名（取负向=不稳定排后）。成交量日内分布的规律性。",
    category="intraday",
    thesis="成交量日内分布的稳定性反映了交易结构的健康程度。稳定的成交量模式（低vol_stability）意味着流动性供给可预测，交易成本可控。成交量忽大忽小往往与信息不对称和主力操纵行为相关。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_vol_stability(context: FactorContext):
    source_root = context.repo.paths.source_root
    vs = _compute_intraday_factor(source_root, context.repo.allowed_codes, "vol_stability",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-vs)


@register_factor(
    name="am_vol_share",
    description="上午成交量占比因子，上午成交量/全日成交量截面排名。上午占比高=信息消化积极。",
    category="intraday",
    thesis="上午成交量占比反映了隔夜信息和开盘信息的消化强度。上午成交占比高的股票说明市场对信息的反应积极且集中，而非拖到尾盘博弈。与am_pm_vol_ratio相比，am_vol_share更直观地衡量上午的相对活跃度。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_am_vol_share(context: FactorContext):
    source_root = context.repo.paths.source_root
    avs = _compute_intraday_factor(source_root, context.repo.allowed_codes, "am_vol_share",
                                    on_progress=context.repo.on_progress)
    return cross_sectional_rank(avs)


# ── Return Dynamics ───────────────────────────────────────────────────────

@register_factor(
    name="intra_trend",
    description="日内趋势强度因子，正收益5分钟区间占比截面排名。日内方向一致性度量。",
    category="intraday",
    thesis="日内收益方向的一致性（正收益5分钟区间的占比）反映了买方力量的持续性。高intra_trend意味着买家在全天持续主导，而非仅在个别时段发力。日内趋势的延续性与短期动量效应正相关。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_intra_trend(context: FactorContext):
    source_root = context.repo.paths.source_root
    it_ = _compute_intraday_factor(source_root, context.repo.allowed_codes, "intra_trend",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(it_)


# ── Composite / Ratio Factors ─────────────────────────────────────────────

@register_factor(
    name="rv_trend_5d",
    description="波动率趋势因子，rv_5min的5日均值/20日均值-1截面排名（取负向=波动加速排后）。",
    category="intraday",
    thesis="波动率本身的趋势（波动率动量）包含了独立于波动率水平的信息。波动率处于上升趋势（高rv_trend_5d）意味着不确定性在加剧，即使当前波动率水平不高，也预示着风险上升。波动率上升期的股票未来收益分布更加左偏。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_trend_5d(context: FactorContext):
    source_root = context.repo.paths.source_root
    rt = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_trend_5d",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rt)


@register_factor(
    name="range_rv_ratio",
    description="振幅波动比因子，(high/low-1)/rv_5min截面排名（取负向=高比值排后）。跳成分相对于连续波动的比例。",
    category="intraday",
    thesis="日内振幅相对已实现波动的比值越高，说明价格变动中有更大的跳跃成分（而非连续扩散）。跳跃主导的价格过程意味着更大的尾部风险和更低的夏普比率。此因子与rjump_5min互补——前者是比值视角，后者是绝对跳跃强度。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_range_rv_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    rr = _compute_intraday_factor(source_root, context.repo.allowed_codes, "range_rv_ratio",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rr)


@register_factor(
    name="volume_rv_ratio",
    description="成交量波动比因子，log(成交量)/rv_5min截面排名。单位波动对应的交易活跃度。",
    category="intraday",
    thesis="每单位已实现波动率对应的成交量衡量了'信息的定价效率'。高volume_rv_ratio意味着大量交易产生了相对较小的价格波动，说明市场吸收信息的能力强、流动性充足。低比值可能暗示流动性脆弱——少量交易即引发较大波动。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_volume_rv_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    vr = _compute_intraday_factor(source_root, context.repo.allowed_codes, "volume_rv_ratio",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(vr)


# ═══════════════════════════════════════════════════════════════════════════════
# Class 3 — Statistical / distributional intraday factors (8 new)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ret_std_intraday",
    description="5分钟收益标准差因子，日内5分钟收益截面标准差排名（取负向=高离散排后）。不同于RV用平方和，std衡量收益围绕均值的离散度。",
    category="intraday",
    thesis="5分钟收益的标准差与RV高度相关但存在差异：趋势日（均值≠0）的std < sqrt(RV^2/n)，震荡日的std ≈ sqrt(RV^2/n)。std/RV的差异隐含日内方向性信息。高std代表价格在短时间内剧烈双向波动，信息不确定性大。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_ret_std_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    rs = _compute_intraday_factor(source_root, context.repo.allowed_codes, "ret_5min_std",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rs)


@register_factor(
    name="ret_kurt_intraday",
    description="5分钟收益峰度因子，日内5分钟收益分布的峰度截面排名（取负向=厚尾排后）。衡量日内极端波动的集中度。",
    category="intraday",
    thesis="金融时间序列的尖峰厚尾特征在日内高频数据中更加明显。高峰度意味着收益分布更集中（多数小幅波动）但尾部更厚（少数极端波动）。高峰度股票的日内价格过程包含更多跳跃成分，未来波动率更不稳定。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_ret_kurt_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    rk = _compute_intraday_factor(source_root, context.repo.allowed_codes, "ret_5min_kurt",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rk)


@register_factor(
    name="ret_autocorr_5min",
    description="5分钟收益自相关因子，日内5分钟收益一阶自相关系数截面排名。正自相关=日内动量，负自相关=均值回复。",
    category="intraday",
    thesis="高频收益的自相关结构反映市场微观结构特征：负自相关通常源于bid-ask bounce和存货管理（做市商行为），正自相关则暗示信息渐进扩散或趋势交易。A股市场中，负自相关较强的股票往往是做市商活跃的标的，流动性更好；正自相关则反映散户追涨杀跌。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_ret_autocorr_5min(context: FactorContext):
    source_root = context.repo.paths.source_root
    ac = _compute_intraday_factor(source_root, context.repo.allowed_codes, "ret_autocorr",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(ac)


@register_factor(
    name="vol_of_vol_intraday",
    description="波动率的波动率因子，|ret_5min|的std/mean截面排名（取负向=波动不稳定排后）。波动率自身的变异系数。",
    category="intraday",
    thesis="波动率的波动率（vol-of-vol）衡量日内波动率的聚集和分散程度(Corsi et al. 2008)。高vol-of-vol意味着波动率在日内剧烈变化（如早盘高波、尾盘低波），价格过程远非平稳。波动率的不确定性本身是第二阶风险，需要额外的风险溢价。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_vol_of_vol_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    vv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "vol_of_vol",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-vv)


@register_factor(
    name="max_ret_intraday",
    description="最大5分钟收益因子，日内最大|ret_5min|截面排名（取负向=极端波动排后）。捕获日内最剧烈的价格冲击。",
    category="intraday",
    thesis="日内最大绝对收益（最小最大值）捕获了传统矩（方差、偏度、峰度）无法完全概括的极端事件信息。单次极端5分钟波动往往对应着大单冲击、乌龙指或突发公告。极端值的大小与未来短期波动率正相关，对流动性较差的股票尤为显著。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_max_ret_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    mr = _compute_intraday_factor(source_root, context.repo.allowed_codes, "max_abs_ret_5min",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-mr)


@register_factor(
    name="pos_rv_ratio",
    description="正收益波动占比因子，正5分钟收益平方和/总RV²截面排名。上涨驱动的波动占比。",
    category="intraday",
    thesis="将已实现方差分解为正收益贡献和负收益贡献(Barndorff-Nielsen et al. 2010)，捕捉波动的非对称性。高pos_rv_ratio意味着日内波动主要由上涨推动——买家主导价格发现过程。持续的高正波动占比是bullish信号。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_pos_rv_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    pr = _compute_intraday_factor(source_root, context.repo.allowed_codes, "pos_rv_ratio",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(pr)


@register_factor(
    name="am_pm_rv_ratio",
    description="上午/下午波动率比因子，上午已实现方差/下午已实现方差截面排名。波动率的日内分布不对称性。",
    category="intraday",
    thesis="上午和下午的波动率分布反映不同类型的信息：上午波动主要由隔夜信息消化驱动，下午波动更多由盘中事件和尾盘博弈驱动。高am_pm_rv_ratio意味着信息冲击集中在上午（高效率定价），低比值则暗示不确定性延续到下午（定价效率低）。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_am_pm_rv_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    ar = _compute_intraday_factor(source_root, context.repo.allowed_codes, "am_pm_rv_ratio",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(ar)


@register_factor(
    name="rq_intraday",
    description="已实现四次变差因子（Realized Quarticity），5分钟收益四次方和截面排名（取负向=高方差波动排后）。度量波动的波动。",
    category="intraday",
    thesis="已实现四次变差(RQ)是已实现方差(RV²)的方差的估计量(Barndorff-Nielsen & Shephard 2002)。RQ衡量波动率的不确定性——即使两只股票的RV相同，RQ更高的股票未来波动率更不稳定，跳跃风险更大。RQ是波动率预测精度的关键变量。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rq_intraday(context: FactorContext):
    source_root = context.repo.paths.source_root
    rq = _compute_intraday_factor(source_root, context.repo.allowed_codes, "realized_quarticity",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rq)


# ═══════════════════════════════════════════════════════════════════════════════
# Class 3 — Extended intraday factors (Task 3, ~40 new factors)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Extended RV ──────────────────────────────────────────────────────────────

@register_factor(
    name="rv_60min",
    description="60分钟已实现波动率因子，1分钟数据60分钟收益平方和开根截面排名（低波排前）。",
    category="intraday",
    thesis="60分钟频率的RV捕捉更长周期的波动动态——相比5/15/30分钟RV，60分钟RV对 microstructure noise 更稳健，接近'长期波动率'成分。多频RV联合提供波动率期限结构信息。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_60min(context: FactorContext):
    rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_60min",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


@register_factor(
    name="rv_daily",
    description="日度已实现波动率因子，1分钟数据全日收益平方和开根截面排名（低波排前）。",
    category="intraday",
    thesis="日度RV使用全部240个1分钟收益，是最精确的日度波动率估计量。相比收盘价波动率，日度RV利用了全天的价格路径信息，精确度高一个数量级(Andersen et al. 2001)。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_daily(context: FactorContext):
    rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_daily",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


@register_factor(
    name="bv_daily",
    description="日度双幂变差因子，1分钟数据双幂变差截面排名（低波排前）。连续价格变动的稳健波动估计。",
    category="intraday",
    thesis="双幂变差(BV)是对跳跃稳健的波动率估计量(Barndorff-Nielsen & Shephard 2004)——通过相邻收益的乘积而非平方来降低跳跃的影响。BV与RV的差异(RV²-BV²)可分离出跳跃成分。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_bv_daily(context: FactorContext):
    bv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "bv_daily",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-bv)


@register_factor(
    name="rjump_daily",
    description="日度已实现跳跃因子，sqrt(max(RV²-BV²,0))截面排名（高跳跃=风险信号排后）。",
    category="intraday",
    thesis="日度已实现跳跃分离了价格过程中的不连续成分——跳跃代表信息冲击或流动性断裂。高跳跃股票面临更大的尾部风险和更不稳定的波动率。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rjump_daily(context: FactorContext):
    rj = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rjump_daily",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rj)


@register_factor(
    name="rv_semi_up",
    description="上行已实现半方差因子，仅正1分钟收益平方和开根截面排名（上涨波动=正面信号排前）。",
    category="intraday",
    thesis="上行半方差(Barndorff-Nielsen et al. 2010)将波动率分解为'好波动'(上涨驱动)和'坏波动'(下跌驱动)——上涨波动率高的股票处于资金主动买入阶段，是积极的价格发现过程。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_semi_up(context: FactorContext):
    rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_semi_up",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(rv)


@register_factor(
    name="rv_semi_down",
    description="下行已实现半方差因子，仅负1分钟收益平方和开根截面排名（高下行波=风险排后）。",
    category="intraday",
    thesis="下行半方差是投资者真正厌恶的'坏波动'——下行RV高的股票在下跌行情中跌幅更大，需要更高的风险溢价。下行RV对尾部风险的预测能力优于对称RV。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_semi_down(context: FactorContext):
    rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_semi_down",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


# ── Volume Distribution ──────────────────────────────────────────────────────

@register_factor(
    name="volume_profile_skew",
    description="成交量分布偏度因子，5分钟成交量日内分布的偏度截面排名（取负向=偏度极端排后）。",
    category="intraday",
    thesis="成交量分布的偏度反映成交活跃度的日内倾斜——正偏(集中在早盘)意味着信息消化高效，负偏(集中在尾盘)可能是被动交易或操纵行为。极端的成交量偏度是不健康的交易结构。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_volume_profile_skew(context: FactorContext):
    vs = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_profile_skew",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-vs.abs())


@register_factor(
    name="volume_profile_kurt",
    description="成交量分布峰度因子，5分钟成交量日内分布的峰度截面排名（取负向=高峰度=不均排后）。",
    category="intraday",
    thesis="成交量峰度高意味着交易集中在少数几个5分钟区间——极端的成交量集中往往对应着大单冲击或主力行为。均匀分布的成交量代表自然的、健康的交易节奏。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_volume_profile_kurt(context: FactorContext):
    vk = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_profile_kurt",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-vk)


@register_factor(
    name="volume_peak_time",
    description="成交量峰值时间因子，日内最大5分钟成交量所在分钟截面排名（取负向=尾盘放量=异常排后）。",
    category="intraday",
    thesis="成交量峰值出现的时间包含信息——早盘峰值(正常)+午盘峰值(可接受)+尾盘峰值(警惕)。尾盘突然放量往往是主力做收盘价或短线资金集中进出的标志，次日走势不确定。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_volume_peak_time(context: FactorContext):
    vpt = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_peak_time",
                                    on_progress=context.repo.on_progress)
    return cross_sectional_rank(-vpt)


@register_factor(
    name="rel_vol_first_hour",
    description="首小时量比因子，开盘首小时成交量/全日成交量截面排名（早盘活跃=信息驱动排前）。",
    category="intraday",
    thesis="开盘首小时是A股全天成交最活跃的时段——首小时成交占比高意味着市场对隔夜信息的反应积极高效，价格发现质量高。首小时占比持续高的股票流动性更好。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rel_vol_first_hour(context: FactorContext):
    rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rel_vol_first_hour",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(rv)


@register_factor(
    name="rel_vol_last_hour",
    description="尾小时量比因子，收盘前1小时成交量/全日成交量截面排名（取负向=尾盘博弈=不可靠排后）。",
    category="intraday",
    thesis="尾盘最后1小时成交占比过高是A股T+1制度下的特殊风险信号——尾盘拉抬或打压无法当日了结，次日走势往往反向。尾盘量比高意味着短线博弈而非价值投资。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rel_vol_last_hour(context: FactorContext):
    rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rel_vol_last_hour",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


@register_factor(
    name="rel_vol_midday",
    description="午间量比因子，11:00-13:30成交量/全日成交量截面排名（取负向=午间异常放量排后）。",
    category="intraday",
    thesis="午间时段通常是A股成交最清淡的时段——午间异常放量往往意味着有特定消息或主力在这个'低关注窗口'进行操作。午间量比高是信息不对称的信号。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rel_vol_midday(context: FactorContext):
    rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rel_vol_midday",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rv)


# ── Price Path ───────────────────────────────────────────────────────────────

@register_factor(
    name="path_efficiency",
    description="价格路径效率因子，|收盘-开盘|/(最高-最低)截面排名（高效=趋势性强排前）。",
    category="intraday",
    thesis="价格路径效率衡量日内价格是趋势性运动还是震荡运动——高效率(接近1)意味着价格从开盘到收盘基本单向运动，趋势明确。低效率(接近0)意味着日内大幅双向波动后回到原点，多空分歧大。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_path_efficiency(context: FactorContext):
    pe = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "path_efficiency",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(pe)


@register_factor(
    name="intraday_max_drawdown",
    description="日内最大回撤因子，日内从最高点到后续最低点的最大跌幅截面排名（取负向=深回撤排后）。",
    category="intraday",
    thesis="日内最大回撤衡量盘中持仓可能遭遇的最差情景——大回撤意味着即使收盘涨了，盘中也有大量资金被套。日内回撤小的股票持有人体验好、止损盘少、趋势更稳定。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_intraday_max_drawdown(context: FactorContext):
    dd = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "intraday_max_drawdown",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-dd.abs())


@register_factor(
    name="intraday_max_runup",
    description="日内最大拉升因子，日内从最低点到后续最高点的最大涨幅截面排名（强拉升=买方力量排前）。",
    category="intraday",
    thesis="日内最大拉升反映买方在盘中的反击力度——在经历低点后能快速大幅拉升说明下方支撑坚实、买方力量强。大拉升后的股票短期动量和信心都更强。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_intraday_max_runup(context: FactorContext):
    ru = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "intraday_max_runup",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(ru)


@register_factor(
    name="close_auction_impact",
    description="收盘竞价影响因子，最后3分钟收益截面排名（正拉升=尾盘抢筹排前）。",
    category="intraday",
    thesis="A股收盘前3分钟(14:57-15:00)是集合竞价阶段——这3分钟的涨跌反映了资金对收盘价的态度。收盘竞价拉升通常是机构为了提高净值或技术面做收盘价，次日高开概率较大。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_close_auction_impact(context: FactorContext):
    ca = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "close_auction_impact",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(ca)


@register_factor(
    name="am_pm_return_ratio",
    description="上午/下午收益比因子，上午收益/下午收益截面排名（上午领涨=主动买入排前）。",
    category="intraday",
    thesis="上午和下午的收益分配反映不同类型资金的行为——上午收益主要由隔夜信息和机构调仓驱动，下午收益更多受短线资金和情绪影响。上午领涨的股票信息优势更强。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_am_pm_return_ratio(context: FactorContext):
    ar = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "am_pm_return_ratio",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(ar)


@register_factor(
    name="return_asymmetry_intraday",
    description="日内收益不对称因子，-(5分钟收益均值-中位数)截面排名（取负向=不对称=偏度大排后）。",
    category="intraday",
    thesis="收益均值与中位数的差异反映分布的偏度——均值>中位数意味着少数极端正收益拉高了均值（正偏），均值<中位数则是少数极端负收益（负偏）。不对称性大意味着价格过程包含跳跃。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_return_asymmetry_intraday(context: FactorContext):
    ra = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "return_asymmetry",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-ra.abs())


@register_factor(
    name="up_minutes_ratio",
    description="上涨分钟占比因子，1分钟正收益分钟数/总分钟数截面排名（买盘主导排前）。",
    category="intraday",
    thesis="上涨分钟的占比是日内买方力量的直接度量——>50%的分钟在上涨意味着买方在全天持续主导，而非仅在个别时段发力。高上涨占比的股票日内定价效率高、趋势可靠。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_up_minutes_ratio(context: FactorContext):
    um = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "up_minutes_ratio",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(um)


@register_factor(
    name="volume_weighted_ret",
    description="成交量加权收益因子，Σ(ret_i×vol_i)/Σvol_i截面排名（量价配合=真实涨跌排前）。",
    category="intraday",
    thesis="成交量加权收益比简单收益更能反映'真实'的价格变动——在成交量大的价位上的价格变动比成交量小的价位上的变动更有信息含量。VW收益排除了无量空涨/空跌的噪音。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_volume_weighted_ret(context: FactorContext):
    vw = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_weighted_ret",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(vw)


# ── Liquidity & Microstructure ──────────────────────────────────────────────

@register_factor(
    name="realized_spread_5min",
    description="已实现价差因子，5分钟|ret|均值截面排名（取负向=高价差=高交易成本排后）。",
    category="intraday",
    thesis="5分钟绝对收益的均值是bid-ask spread的代理变量——在有效市场中，价格在两个方向间弹跳产生小的绝对收益。绝对收益均值大意味着实际买卖价差大、交易成本高。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_realized_spread_5min(context: FactorContext):
    rs = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "realized_spread_5min",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rs)


@register_factor(
    name="price_impact_asymmetry",
    description="价格冲击不对称因子，-(上涨冲击/下跌冲击)截面排名（取负向=不对称=上涨费劲排后）。",
    category="intraday",
    thesis="上涨和下跌的价格冲击比反映了买卖方流动性供给的不对称——上涨比下跌需要更大成交量（高比值）意味着卖方挂单稀疏、上涨阻力大。低比值则意味着买方承接力弱。接近1的对称性是最优的交易结构。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_price_impact_asymmetry(context: FactorContext):
    pa = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "price_impact_asymmetry",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-np.abs(pa - 1.0))


@register_factor(
    name="liquidity_resilience",
    description="流动性弹性因子，极端收益后成交量的恢复速度截面排名（恢复快=流动性好排前）。",
    category="intraday",
    thesis="流动性弹性(Liquidity Resilience)衡量市场在经历大单冲击后恢复正常交易的速度——高弹性意味着即使出现大单冲击，流动性也能迅速恢复，交易成本不会持续升高。这是市场质量的高级度量。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_liquidity_resilience(context: FactorContext):
    lr = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "liquidity_resilience",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(lr)


@register_factor(
    name="turnover_concentration_intraday",
    description="换手率集中度因子，最大5分钟成交量/全日成交量截面排名（取负向=集中度过高排后）。",
    category="intraday",
    thesis="换手率在单一5分钟区间过度集中往往是大单冲击或主力对倒——集中度>30%意味着三分之一的日成交在5分钟内完成，不利于中小投资者的交易执行。适度分散的成交是健康流动性的标志。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_turnover_concentration_intraday(context: FactorContext):
    tc = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "turnover_concentration",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-tc)


@register_factor(
    name="flash_crash_risk",
    description="闪崩风险因子，日内5分钟累计收益最大回撤截面排名（取负向=闪崩风险高排后）。",
    category="intraday",
    thesis="闪崩(Flash Crash)是短时间内价格剧烈下跌的现象——即使最终收盘涨回来，盘中闪崩也会触发大量止损单，对持仓者造成实质性伤害。闪崩风险高的股票需要额外的尾部风险溢价。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_flash_crash_risk(context: FactorContext):
    fc = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "flash_crash_risk",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-fc)


@register_factor(
    name="extreme_move_count",
    description="极端波动次数因子，日内|ret_5min|>3σ的次数截面排名（取负向=频繁极端波动排后）。",
    category="intraday",
    thesis="日内极端5分钟波动的次数直接度量了价格的'不平静'程度——即使总体RV不高，频繁的极端波动也意味着价格过程远非高斯，跳跃风险高。极端波动次数是对标准波动率的重要补充。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_extreme_move_count(context: FactorContext):
    em = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "extreme_move_count",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-em)


@register_factor(
    name="jump_ratio_intraday",
    description="跳跃占比因子，日度已实现跳跃/日度RV截面排名（取负向=跳跃主导=不稳定排后）。",
    category="intraday",
    thesis="跳跃成分占总波动的比例反映了价格过程的'光滑度'——跳跃占比高意味着价格变动主要由不连续的冲击驱动(而非连续的扩散)，预测难度更大、套利风险更高。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_jump_ratio_intraday(context: FactorContext):
    jr = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "jump_ratio",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-jr)


@register_factor(
    name="volume_u_shape_score",
    description="U型分布评分因子，成交量日内分布与U型模板的相关性截面排名（取负向=极端U型=操纵风险排后）。",
    category="intraday",
    thesis="A股成交量呈U型分布(早盘和尾盘量大、午间量小)是正常现象——但极端的U型(过于集中首尾)往往与操纵行为有关。适度的U型是健康的，极端的U型可能暗示主力在特定时段操作。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_volume_u_shape_score(context: FactorContext):
    us = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_u_shape_score",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-us.abs())


# ── Hourly Decomposition ─────────────────────────────────────────────────────

@register_factor(
    name="rv_hourly_1",
    description="第一小时波动率因子，9:30-10:30已实现波动率截面排名（取负向=开盘高波排后）。",
    category="intraday",
    thesis="开盘第一小时的波动率是隔夜信息冲击的集中释放——第一小时波动率异常高意味着隔夜发生了重大事件，市场需要时间消化。持续的高开盘波动往往伴随更高的后续波动。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_hourly_1(context: FactorContext):
    rh = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_hourly_1",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rh)


@register_factor(
    name="rv_hourly_4",
    description="第四小时波动率因子，14:00-15:00已实现波动率截面排名（取负向=尾盘高波排后）。",
    category="intraday",
    thesis="尾盘最后一小时的波动率包含了T+1制度下的特殊博弈——尾盘波动率异常高往往与短线资金的日内了结和隔夜避险行为有关，对次日开盘有预测意义。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_hourly_4(context: FactorContext):
    rh = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_hourly_4",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rh)


@register_factor(
    name="rv_hourly_dispersion",
    description="小时波动率离散度因子，四个小时RV的标准差/均值截面排名（取负向=波动集中=不稳定排后）。",
    category="intraday",
    thesis="四个小时的波动率离散度反映日内波动的'均匀性'——波动集中在某一个小时(高离散)意味着信息冲击集中在该时段，其余时间缺乏定价活动。低离散(均匀分布)代表全天持续的信息流和稳定的交易节奏。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_hourly_dispersion(context: FactorContext):
    rd = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_hourly_dispersion",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-rd)


# ── Autocorrelation & Range ──────────────────────────────────────────────────

@register_factor(
    name="ret_autocorr_abs",
    description="绝对收益自相关因子，|ret_5min|的一阶自相关系数截面排名（正自相关=波动聚集排前=预测性好）。",
    category="intraday",
    thesis="绝对收益的自相关(而非收益本身的自相关)是波动率聚集的微观体现——正自相关意味着大波动后继续大波动(波动聚集)，这是金融市场普遍存在的特征。较高的正自相关意味着波动率具有一定的可预测性(可使用GARCH类模型)。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_ret_autocorr_abs(context: FactorContext):
    ra = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "ret_autocorr_abs",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(ra)


@register_factor(
    name="am_hl_range_intraday",
    description="上午振幅因子，上午最高/上午最低-1截面排名（取负向=上午高振幅=分歧大排后）。",
    category="intraday",
    thesis="上午的振幅反映了隔夜信息消化过程中的多空分歧——上午振幅大意味着市场对新信息的价格发现还处于激烈博弈阶段，定价尚未收敛。低上午振幅意味着市场对信息的解读较为一致。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_am_hl_range_intraday(context: FactorContext):
    ah = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "am_hl_range",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-ah)


@register_factor(
    name="pm_hl_range_intraday",
    description="下午振幅因子，下午最高/下午最低-1截面排名（取负向=下午高振幅=尾盘博弈排后）。",
    category="intraday",
    thesis="下午的振幅更多反映盘中新增信息和T+1博弈——下午振幅异常放大往往与短线资金的短线操作和尾盘避险/抢筹有关。下午振幅大的股票隔夜风险更高。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_pm_hl_range_intraday(context: FactorContext):
    ph = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "pm_hl_range",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-ph)


@register_factor(
    name="close_5min_momentum",
    description="尾盘5分钟动量因子，最后5分钟收益截面排名（取负向=尾盘拉升=次日易低开排后）。",
    category="intraday",
    thesis="尾盘最后5分钟的收益率是A股T+1制度下最具争议的信号——尾盘拉升往往是为了做高收盘价(无法当日卖出)，次日大概率低开。尾盘打压则可能是为了次日低价吸筹。尾盘动量对次日开盘有显著的负向预测能力。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_close_5min_momentum(context: FactorContext):
    cm = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "close_5min_momentum",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-cm)


@register_factor(
    name="open_5min_momentum",
    description="开盘5分钟动量因子，前5分钟收益截面排名（开盘强势=隔夜利好排前）。",
    category="intraday",
    thesis="开盘前5分钟是集合竞价后的第一个连续交易时段——这5分钟的走势是市场对集合竞价定价的'确认'或'否定'。开盘5分钟继续上涨意味着隔夜利好被确认，有延续性。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_open_5min_momentum(context: FactorContext):
    om = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "open_5min_momentum",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(om)


@register_factor(
    name="cumulative_ret_path",
    description="累积收益路径效率因子，|总收益|/Σ|1分钟收益|截面排名（高效率=趋势明确排前）。",
    category="intraday",
    thesis="累积收益路径效率是趋势强度的度量——如果股票全天持续上涨(|总收益|/Σ|每分钟收益|接近1)，说明买方持续主导。如果比值小(0.1-0.3)，说明日内涨跌互现、无明确方向。高效率的日内路径预示着更强的短期趋势。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_cumulative_ret_path(context: FactorContext):
    cr = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "cumulative_ret_path",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(cr)


@register_factor(
    name="pm_momentum_intraday",
    description="下午动量因子，下午收盘/下午开盘-1截面排名（下午走强=买盘持续排前）。",
    category="intraday",
    thesis="下午的走势(尤其是下午开盘后)是A股短线交易的重要参考——经过午间休市的信息消化后，下午开盘的方向往往代表了机构资金的最终判断。下午持续走强比上午冲高下午回落更可靠。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_pm_momentum_intraday(context: FactorContext):
    pm = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "pm_momentum",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(pm)


# ── Vol-of-Vol & Term Structure ──────────────────────────────────────────────

@register_factor(
    name="rv_rolling_5d_std",
    description="波动率波动因子，rv_5min的5日标准差截面排名（取负向=波动率不稳定排后）。",
    category="intraday",
    thesis="波动率本身的波动率(vol-of-vol)是二阶风险——两只股票当前波动率相同，但波动率更不稳定的那只未来风险更大。Vol-of-vol是预测波动率突变的重要先行指标。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_rolling_5d_std(context: FactorContext):
    vs = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_rolling_5d_std",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-vs)


@register_factor(
    name="rv_term_structure_slope",
    description="波动率期限结构斜率因子，rv_5min/rv_60min-1截面排名（取负向=陡峭=短期波动高排后）。",
    category="intraday",
    thesis="短期与长期RV的比值是波动率期限结构的斜率——比值>1意味着短期波动率高于长期(波动率期限结构向下倾斜)，往往对应着短期事件冲击或市场压力。比值<1(正常状态)意味着波动率期限结构向上。",
    dependencies=("history_1min", "calendar.parquet"),
)
def factor_rv_term_structure_slope(context: FactorContext):
    ts = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_term_structure_slope",
                                   on_progress=context.repo.on_progress)
    return cross_sectional_rank(-ts)
