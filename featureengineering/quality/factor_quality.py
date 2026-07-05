#!/usr/bin/env python3
"""
Factor Data Quality Analyzer
============================
Analyzes 960 factors for data quality issues:
  1. High NaN ratio — factors with excessive missing values
  2. Low IC vs 1d_ret — factors with negligible predictive power
  3. Low cross-sectional variance — factors that are nearly constant
  4. Extreme outlier ratio — factors dominated by a few extreme values

Uses CPU parallel acceleration (ProcessPoolExecutor) with tqdm progress bars.
All 960 factors share an identical wide-format structure (Date × Code).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis as kurtosis_scipy
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*correlation coefficient is not defined.*")
warnings.filterwarnings("ignore", message=".*Precision loss occurred.*")

# ── Paths (override via CLI) ─────────────────────────────────────────────────
FACTOR_DIR = Path("/root/autodl-fs/lingqiData/featureengineering/data/factors")
TARGET_DIR = Path("/root/autodl-fs/lingqiData/featureengineering/data/targets")
OUTPUT_DIR = Path("/root/autodl-fs/lingqiData/featureengineering/quality")
TARGET_NAME = "label_ret_1d"

logger = logging.getLogger(__name__)


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    """Single-factor quality metrics."""
    factor_name: str
    n_dates: int
    n_codes: int
    # NaN
    nan_ratio: float            # 0-1, overall fraction of NaN
    nan_ratio_per_date_mean: float  # mean of per-date NaN ratios
    nan_ratio_per_date_max: float   # worst single-date NaN ratio
    # Variance
    xs_var_mean: float          # mean cross-sectional variance across dates
    xs_var_std: float           # std of cross-sectional variance
    xs_std_mean: float          # mean cross-sectional std
    # IC (Rank Spearman vs 1d_ret)
    ic_n_dates: int             # number of valid IC dates (>=10 stocks)
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float            # IC mean / IC std
    rank_ic_pos_ratio: float    # fraction of dates with IC > 0
    rank_ic_tstat: float
    # Distribution
    global_mean: float
    global_std: float
    skewness: float             # cross-sectional skewness (mean across dates)
    kurtosis: float             # cross-sectional kurtosis (mean across dates)
    extreme_ratio: float        # fraction of values beyond 5σ per date

    # Flags (set after thresholds applied)
    flag_high_nan: bool = False
    flag_low_ic: bool = False
    flag_low_variance: bool = False
    flag_extreme: bool = False


# ── Per-factor computation ───────────────────────────────────────────────────

def _compute_single_factor(args: tuple[str, str, str]) -> QualityResult | None:
    """Compute quality metrics for a single factor (worker function).

    Parameters
    ----------
    args : tuple
        (factor_path, target_path, factor_name)
    """
    factor_path, target_path, factor_name = args

    try:
        factor = pd.read_feather(factor_path)
        target = pd.read_feather(target_path)
        # Ensure float64 — some factors may have object columns (Python None mixed in)
        factor = factor.apply(pd.to_numeric, errors="coerce")
        target = target.apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        logger.error("Failed to read %s: %s", factor_name, e)
        return None

    # ── Basic shape ──
    n_dates = len(factor)
    n_codes = len(factor.columns)

    # ── NaN metrics ──
    total_cells = n_dates * n_codes if n_dates > 0 and n_codes > 0 else 1
    nan_count = int(factor.isna().sum().sum())
    nan_ratio = nan_count / total_cells

    # Per-date NaN ratios
    per_date_nan = factor.isna().sum(axis=1) / n_codes
    nan_ratio_per_date_mean = float(per_date_nan.mean())
    nan_ratio_per_date_max = float(per_date_nan.max())

    # ── Variance metrics ──
    factor_values = factor.values  # (dates, codes) float array
    xs_std_arr = np.nanstd(factor_values, axis=1)  # per-date cross-sectional std
    xs_var_arr = xs_std_arr ** 2

    xs_var_mean = float(np.nanmean(xs_var_arr))
    xs_var_std = float(np.nanstd(xs_var_arr))
    xs_std_mean = float(np.nanmean(xs_std_arr))

    # ── Distribution (global) ──
    flat = factor_values.ravel()
    flat_finite = flat[np.isfinite(flat)]
    global_mean = float(np.mean(flat_finite)) if len(flat_finite) > 0 else float("nan")
    global_std = float(np.std(flat_finite)) if len(flat_finite) > 0 else float("nan")

    # Cross-sectional skewness & kurtosis (mean across dates)
    skew_vals = []
    kurt_vals = []
    extreme_ratios = []
    for i in range(n_dates):
        row = factor_values[i]
        row_finite = row[np.isfinite(row)]
        if len(row_finite) < 10:
            continue
        skew_vals.append(float(skew(row_finite)))
        kurt_vals.append(float(kurtosis_scipy(row_finite)))
        # Extreme: beyond 5σ
        r_std = np.std(row_finite)
        if r_std > 0:
            extreme_ratios.append(float(np.sum(np.abs(row_finite - np.mean(row_finite)) > 5 * r_std) / len(row_finite)))

    skewness = float(np.mean(skew_vals)) if skew_vals else float("nan")
    kurtosis = float(np.mean(kurt_vals)) if kurt_vals else float("nan")
    extreme_ratio = float(np.mean(extreme_ratios)) if extreme_ratios else float("nan")

    # ── Rank IC vs target ──
    common_codes = factor.columns.intersection(target.columns).sort_values()
    common_dates = factor.index.intersection(target.index).sort_values()

    ic_values = []
    if len(common_codes) >= 10 and len(common_dates) > 0:
        f_aligned = factor.loc[common_dates, common_codes]
        t_aligned = target.loc[common_dates, common_codes]
        for date in common_dates:
            f_row = f_aligned.loc[date]
            t_row = t_aligned.loc[date]
            mask = f_row.notna() & t_row.notna()
            if mask.sum() < 10:
                continue
            ic = f_row[mask].corr(t_row[mask], method="spearman")
            ic_values.append(float(ic))

    ic_series = pd.Series(ic_values).dropna()
    ic_n_dates = len(ic_series)
    if ic_n_dates > 0:
        rank_ic_mean = float(ic_series.mean())
        rank_ic_std = float(ic_series.std(ddof=1))
        rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else float("nan")
        rank_ic_pos_ratio = float((ic_series > 0).mean())
        rank_ic_tstat = rank_ic_mean / (rank_ic_std / np.sqrt(ic_n_dates)) if rank_ic_std > 0 else float("nan")
    else:
        rank_ic_mean = float("nan")
        rank_ic_std = float("nan")
        rank_icir = float("nan")
        rank_ic_pos_ratio = float("nan")
        rank_ic_tstat = float("nan")

    return QualityResult(
        factor_name=factor_name,
        n_dates=n_dates,
        n_codes=n_codes,
        nan_ratio=nan_ratio,
        nan_ratio_per_date_mean=nan_ratio_per_date_mean,
        nan_ratio_per_date_max=nan_ratio_per_date_max,
        xs_var_mean=xs_var_mean,
        xs_var_std=xs_var_std,
        xs_std_mean=xs_std_mean,
        ic_n_dates=ic_n_dates,
        rank_ic_mean=rank_ic_mean,
        rank_ic_std=rank_ic_std,
        rank_icir=rank_icir,
        rank_ic_pos_ratio=rank_ic_pos_ratio,
        rank_ic_tstat=rank_ic_tstat,
        global_mean=global_mean,
        global_std=global_std,
        skewness=skewness,
        kurtosis=kurtosis,
        extreme_ratio=extreme_ratio,
    )


# ── Main runner ──────────────────────────────────────────────────────────────

def analyze_factors(
    factor_dir: Path = FACTOR_DIR,
    target_dir: Path = TARGET_DIR,
    target_name: str = TARGET_NAME,
    factor_names: list[str] | None = None,
    max_workers: int | None = None,
    chunk_size: int = 1,
) -> pd.DataFrame:
    """Run quality analysis on all (or selected) factor files.

    Parameters
    ----------
    factor_dir : Path
        Directory containing .fea factor files.
    target_dir : Path
        Directory containing target .fea files.
    target_name : str
        Target file name without .fea extension.
    factor_names : list[str] | None
        If provided, only analyze these factor names. Otherwise analyze all .fea files.
    max_workers : int | None
        Number of parallel workers. Defaults to CPU count - 2.
    chunk_size : int
        Chunk size for ProcessPoolExecutor (not used, kept for API compatibility).

    Returns
    -------
    pd.DataFrame
        Results sorted by factor name.
    """
    target_path = str(target_dir / f"{target_name}.fea")

    if factor_names is None:
        factor_names = sorted(
            p.stem for p in factor_dir.glob("*.fea") if p.suffix == ".fea"
        )

    # Build work items
    work_items = [
        (str(factor_dir / f"{fn}.fea"), target_path, fn)
        for fn in factor_names
    ]

    if max_workers is None:
        max_workers = max(1, (len(work_items) or 1) - 2)

    print(f"Analyzing {len(work_items)} factors using {max_workers} workers...")
    print(f"Target: {target_name}")
    print()

    results: list[QualityResult] = []
    failed: list[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_single_factor, item): item[2]
            for item in work_items
        }
        with tqdm(total=len(futures), desc="Analyzing factors", unit="factor",
                  ncols=100) as pbar:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                    else:
                        failed.append(name)
                except Exception as e:
                    logger.error("Error processing %s: %s", name, e)
                    failed.append(name)
                pbar.update(1)

    if failed:
        print(f"\n[WARNING] {len(failed)} factors failed: {failed[:10]}..."
              if len(failed) > 10 else f"\n[WARNING] {len(failed)} factors failed: {failed}")

    df = _results_to_dataframe(results)
    return df


def _results_to_dataframe(results: list[QualityResult]) -> pd.DataFrame:
    """Convert results list to DataFrame."""
    records = []
    for r in results:
        records.append({
            "factor": r.factor_name,
            "n_dates": r.n_dates,
            "n_codes": r.n_codes,
            "nan_ratio": r.nan_ratio,
            "nan_ratio_per_date_mean": r.nan_ratio_per_date_mean,
            "nan_ratio_per_date_max": r.nan_ratio_per_date_max,
            "xs_var_mean": r.xs_var_mean,
            "xs_var_std": r.xs_var_std,
            "xs_std_mean": r.xs_std_mean,
            "ic_n_dates": r.ic_n_dates,
            "rank_ic_mean": r.rank_ic_mean,
            "rank_ic_std": r.rank_ic_std,
            "rank_icir": r.rank_icir,
            "rank_ic_pos_ratio": r.rank_ic_pos_ratio,
            "rank_ic_tstat": r.rank_ic_tstat,
            "global_mean": r.global_mean,
            "global_std": r.global_std,
            "skewness": r.skewness,
            "kurtosis": r.kurtosis,
            "extreme_ratio": r.extreme_ratio,
            "flag_high_nan": r.flag_high_nan,
            "flag_low_ic": r.flag_low_ic,
            "flag_low_variance": r.flag_low_variance,
            "flag_extreme": r.flag_extreme,
        })
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("factor").reset_index(drop=True)
    return df


# ── Flagging / Screening ─────────────────────────────────────────────────────

def apply_thresholds(
    df: pd.DataFrame,
    nan_threshold: float = 0.20,
    icir_threshold: float = 0.05,
    ic_pos_ratio_threshold: float = 0.50,
    var_threshold_percentile: float = 1.0,
    extreme_threshold: float = 0.05,
) -> pd.DataFrame:
    """Apply quality flag thresholds to the results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Results from analyze_factors.
    nan_threshold : float
        Factors with nan_ratio above this are flagged. Default 20%.
    icir_threshold : float
        Factors with abs(ICIR) below this are flagged as low IC. Default 0.05.
        Additionally, factors with abs(ICIR) < 0.10 AND IC>0 ratio in [0.48, 0.52]
        (no directional signal) are also flagged.
    var_threshold_percentile : float
        Factors with xs_std_mean below this percentile of all factors are flagged.
        Default 1% (bottom 1% by variance).
    extreme_threshold : float
        Factors with extreme_ratio above this are flagged.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with flag columns updated.
    """
    df = df.copy()

    # High NaN
    df["flag_high_nan"] = df["nan_ratio"] > nan_threshold

    # Low IC:
    #   (a) |ICIR| < 0.05  → truly negligible predictive power
    #   (b) |ICIR| < 0.10  AND  IC>0 ratio ∈ [0.48, 0.52]  → near-random direction
    df["flag_low_ic"] = (
        (df["rank_icir"].abs() < icir_threshold) |
        ((df["rank_icir"].abs() < icir_threshold * 2) &
         (df["rank_ic_pos_ratio"] > 0.48) & (df["rank_ic_pos_ratio"] < 0.52))
    )

    # Low variance (bottom N percentile)
    var_cutoff = df["xs_std_mean"].quantile(var_threshold_percentile / 100.0)
    df["flag_low_variance"] = df["xs_std_mean"] <= var_cutoff

    # Extreme values
    df["flag_extreme"] = df["extreme_ratio"] > extreme_threshold

    return df


# ── Correlation / Redundancy Analysis ─────────────────────────────────────────

def _extract_sampled_snapshot(args: tuple[str, list[str], str]) -> dict[str, np.ndarray] | None:
    """Extract factor values on sampled dates for one factor (worker).

    Parameters
    ----------
    args : tuple
        (factor_path, sampled_dates, factor_name)

    Returns
    -------
    dict mapping factor_name -> 2D array (n_sampled_dates, n_codes), or None on failure.
    """
    factor_path, sampled_dates, factor_name = args
    try:
        factor = pd.read_feather(factor_path)
        # Get common dates
        common = factor.index.intersection(sampled_dates)
        if len(common) == 0:
            return None
        # Extract rows for sampled dates, return as numpy array
        snapshot = factor.loc[common].values  # (n_dates, n_codes)
        return {factor_name: snapshot}
    except Exception as e:
        logger.error("Failed to extract snapshot for %s: %s", factor_name, e)
        return None


def analyze_correlations(
    factor_dir: Path,
    factor_names: list[str],
    target_path: str | None = None,
    max_workers: int | None = None,
    corr_threshold: float = 0.95,
    n_sample_dates: int = 30,
) -> tuple[pd.DataFrame, set[str]]:
    """Compute pairwise cross-sectional correlation of factor values.

    Samples n_sample_dates evenly across the date range. On each sampled date,
    computes the cross-sectional Pearson correlation between every pair of
    factors (using stock-level values). Averages correlations across dates.

    Pairs with |avg_corr| > corr_threshold are flagged as redundant;
    the "worse" factor is determined later by resolve_redundant_flags().

    Parameters
    ----------
    factor_dir : Path
    factor_names : list[str]
    target_path : str | None
        Path to target file (used to find date range). If None, uses first factor.
    max_workers : int | None
    corr_threshold : float
    n_sample_dates : int
        Number of dates to sample for cross-sectional correlation.

    Returns
    -------
    tuple[pd.DataFrame, set[str]]
    """
    n_factors = len(factor_names)
    if n_factors < 2:
        return pd.DataFrame(), set()

    # ── Determine date range from a reference factor ──
    if target_path is not None:
        ref = pd.read_feather(target_path)
    else:
        ref = pd.read_feather(str(factor_dir / f"{factor_names[0]}.fea"))
    all_dates = sorted(ref.index.tolist())

    # Sample evenly
    step = max(1, len(all_dates) // n_sample_dates)
    sampled_dates = all_dates[::step][:n_sample_dates]
    print(f"\nSampled {len(sampled_dates)} dates for cross-sectional correlation "
          f"(from {all_dates[0]} to {all_dates[-1]})")

    if max_workers is None:
        max_workers = max(1, n_factors - 2)

    # ── Step 1: extract factor values on sampled dates ──
    print(f"Extracting factor snapshots for {n_factors} factors...")
    work_items = [
        (str(factor_dir / f"{fn}.fea"), sampled_dates, fn)
        for fn in factor_names
    ]
    factor_snapshots: dict[str, np.ndarray] = {}  # name -> (n_dates, n_codes)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_sampled_snapshot, item): item[2]
            for item in work_items
        }
        with tqdm(total=len(futures), desc="Extract snapshots", unit="factor",
                  ncols=100) as pbar:
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res is not None:
                        factor_snapshots.update(res)
                except Exception:
                    pass
                pbar.update(1)

    if len(factor_snapshots) < 2:
        return pd.DataFrame(), set()

    names = sorted(factor_snapshots.keys())
    n_eff = len(names)
    n_dates = len(sampled_dates)

    # ── Step 2: per-date cross-sectional correlation ──
    print(f"Computing {n_eff}×{n_eff} cross-sectional correlations on {n_dates} dates...")

    # Accumulate average correlation matrix
    corr_sum = np.zeros((n_eff, n_eff))
    valid_counts = np.zeros((n_eff, n_eff))

    for d_idx in tqdm(range(n_dates), desc="Per-date XS corr", unit="date", ncols=100):
        # Build (n_factors, n_stocks) matrix for this date
        rows = []
        valid_idx = []
        for i, name in enumerate(names):
            snap = factor_snapshots[name]
            if d_idx < snap.shape[0]:
                row = snap[d_idx]
                rows.append(row)
                valid_idx.append(i)

        if len(rows) < 2:
            continue

        mat = np.array(rows)  # (n_valid, n_stocks)
        # Drop columns where any factor is NaN
        valid_cols = ~np.any(np.isnan(mat), axis=0)
        if valid_cols.sum() < 10:
            continue
        mat_clean = mat[:, valid_cols]

        # Compute correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(mat_clean)  # (n_valid, n_valid)
            # Handle NaN in corr (e.g., constant factors)
            corr = np.nan_to_num(corr, nan=0.0)

        # Map back to full index
        for ii, i_full in enumerate(valid_idx):
            for jj, j_full in enumerate(valid_idx):
                if i_full < j_full:
                    corr_sum[i_full, j_full] += corr[ii, jj]
                    valid_counts[i_full, j_full] += 1

    # Average
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_corr = np.where(valid_counts > 0, corr_sum / valid_counts, 0.0)

    # ── Step 3: find highly correlated pairs ──
    pairs = []
    for i in range(n_eff):
        for j in range(i + 1, n_eff):
            if valid_counts[i, j] > 0 and abs(avg_corr[i, j]) > corr_threshold:
                pairs.append({
                    "factor_a": names[i],
                    "factor_b": names[j],
                    "correlation": float(avg_corr[i, j]),
                    "n_dates": int(valid_counts[i, j]),
                })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values("correlation", ascending=False).reset_index(drop=True)

    return pairs_df, set()


def resolve_redundant_flags(
    pairs_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    corr_threshold: float = 0.95,
) -> tuple[pd.DataFrame, set[str]]:
    """Given redundant pairs and quality metrics, flag the worse factor in each pair.

    Decision rule (per pair):
      - Flag the factor with lower |ICIR|.
      - Tie-break: higher NaN ratio.
      - Tie-break: lower xs_std_mean.

    Returns
    -------
    tuple[pd.DataFrame, set[str]]
        - quality_df with added 'flag_redundant' column
        - set of redundant factor names
    """
    quality_df = quality_df.copy()
    quality_df["flag_redundant"] = False

    if len(pairs_df) == 0:
        return quality_df, set()

    # Build lookup from quality_df
    qlookup = quality_df.set_index("factor")

    redundant: set[str] = set()
    for _, row in pairs_df.iterrows():
        a, b = row["factor_a"], row["factor_b"]
        if a not in qlookup.index or b not in qlookup.index:
            continue

        # Decide which to flag
        icir_a = abs(qlookup.loc[a, "rank_icir"])
        icir_b = abs(qlookup.loc[b, "rank_icir"])

        if pd.isna(icir_a):
            worse = a
        elif pd.isna(icir_b):
            worse = b
        elif abs(icir_a - icir_b) < 1e-6:
            # Tie-break: higher NaN
            nan_a = qlookup.loc[a, "nan_ratio"]
            nan_b = qlookup.loc[b, "nan_ratio"]
            if nan_a > nan_b:
                worse = a
            elif nan_b > nan_a:
                worse = b
            else:
                # Second tie-break: lower variance
                var_a = qlookup.loc[a, "xs_std_mean"]
                var_b = qlookup.loc[b, "xs_std_mean"]
                worse = a if var_a < var_b else b
        else:
            worse = a if icir_a < icir_b else b

        redundant.add(worse)

    if redundant:
        quality_df.loc[quality_df["factor"].isin(redundant), "flag_redundant"] = True

    return quality_df, redundant


# ── Final Bad-Factor List ────────────────────────────────────────────────────

def generate_bad_factor_list(
    quality_df: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate the final list of bad-quality factor IDs.

    A factor is considered "bad" if ANY of the following flags is True:
      - flag_high_nan
      - flag_low_ic
      - flag_low_variance
      - flag_extreme
      - flag_redundant

    Parameters
    ----------
    quality_df : pd.DataFrame
    output_path : Path | None
        If provided, save the bad-factor list to this path.

    Returns
    -------
    pd.DataFrame
        Bad factors with their reasons, sorted by severity.
    """
    flag_cols = ["flag_high_nan", "flag_low_ic", "flag_low_variance",
                 "flag_extreme", "flag_redundant"]
    available = [c for c in flag_cols if c in quality_df.columns]

    bad_mask = quality_df[available].any(axis=1)
    bad_df = quality_df.loc[bad_mask].copy()

    # Build a "reasons" string
    def _reasons(row):
        parts = []
        if row.get("flag_high_nan", False):
            parts.append(f"high_nan({row['nan_ratio']:.3f})")
        if row.get("flag_low_ic", False):
            parts.append(f"low_ic(ICIR={row['rank_icir']:+.4f})")
        if row.get("flag_low_variance", False):
            parts.append(f"low_var(std={row['xs_std_mean']:.4f})")
        if row.get("flag_extreme", False):
            parts.append(f"extreme({row['extreme_ratio']:.3f})")
        if row.get("flag_redundant", False):
            parts.append("redundant(high_corr)")
        return "; ".join(parts)

    bad_df["quality_issues"] = bad_df.apply(_reasons, axis=1)

    # Sort: flagged by more categories first, then by |ICIR| ascending
    bad_df["n_flags"] = bad_df[available].sum(axis=1)
    bad_df = bad_df.sort_values(
        ["n_flags", "rank_icir"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Select output columns
    out_cols = ["factor", "quality_issues", "n_flags",
                "nan_ratio", "rank_icir", "rank_ic_mean",
                "xs_std_mean", "extreme_ratio"]
    out_cols = [c for c in out_cols if c in bad_df.columns]

    result = bad_df[out_cols]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Bad-factor list saved to: {output_path}")

    return result


# ── Report ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the quality analysis results."""
    n = len(df)
    flag_cols = ["flag_high_nan", "flag_low_ic", "flag_low_variance", "flag_extreme"]
    if "flag_redundant" in df.columns:
        flag_cols.append("flag_redundant")
    n_bad = int(df[flag_cols].any(axis=1).sum())
    n_nan = int(df["flag_high_nan"].sum())
    n_ic = int(df["flag_low_ic"].sum())
    n_var = int(df["flag_low_variance"].sum())
    n_ext = int(df["flag_extreme"].sum())
    n_red = int(df["flag_redundant"].sum()) if "flag_redundant" in df.columns else 0

    print()
    print("=" * 72)
    print("FACTOR QUALITY SUMMARY")
    print("=" * 72)
    print(f"  Total factors analyzed: {n}")
    print(f"  Factors with ≥1 quality flag: {n_bad} ({n_bad/n*100:.1f}%)" if n > 0 else "")
    print(f"    - High NaN (>20%):         {n_nan}")
    print(f"    - Low IC (negligible):      {n_ic}")
    print(f"    - Low variance (bottom 1%): {n_var}")
    print(f"    - Extreme values (>5%):     {n_ext}")
    if "flag_redundant" in df.columns:
        print(f"    - Redundant (high corr):    {n_red}")
    print()

    # Metric distributions
    metrics = [
        ("nan_ratio", "NaN Ratio"),
        ("xs_std_mean", "XS Std (mean)"),
        ("rank_icir", "Rank ICIR"),
        ("rank_ic_mean", "Rank IC Mean"),
        ("skewness", "Skewness"),
        ("kurtosis", "Kurtosis"),
    ]
    print("─" * 72)
    print(f"{'Metric':<22s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'P25':>10s} {'P50':>10s} {'P75':>10s} {'Max':>10s}")
    print("─" * 72)
    for col, label in metrics:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"{label:<22s} {s.mean():>10.4f} {s.std():>10.4f} {s.min():>10.4f} "
                      f"{s.quantile(0.25):>10.4f} {s.median():>10.4f} {s.quantile(0.75):>10.4f} {s.max():>10.4f}")
    print("─" * 72)

    # Bottom 10 by IC
    print()
    print("Bottom 10 factors by |ICIR|:")
    bottom_ic = df.dropna(subset=["rank_icir"]).nsmallest(10, "rank_icir", keep="all")  # type: ignore[arg-type]
    for _, row in bottom_ic.iterrows():
        print(f"  {row['factor']:<45s} ICIR={row['rank_icir']:+.6f}  IC_mean={row['rank_ic_mean']:+.6f}  NaN={row['nan_ratio']:.3f}")

    # Top 10 by NaN
    print()
    print("Top 10 factors by NaN ratio:")
    top_nan = df.nlargest(10, "nan_ratio")
    for _, row in top_nan.iterrows():
        print(f"  {row['factor']:<45s} NaN={row['nan_ratio']:.4f}  ICIR={row['rank_icir']:+.6f}")

    # Bottom 10 by variance
    print()
    print("Bottom 10 factors by cross-sectional std:")
    bottom_var = df.nsmallest(10, "xs_std_mean")
    for _, row in bottom_var.iterrows():
        print(f"  {row['factor']:<45s} XS_std={row['xs_std_mean']:.6f}  ICIR={row['rank_icir']:+.6f}")

    print()
    print("=" * 72)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze factor data quality (NaN, IC, variance, extremes)"
    )
    parser.add_argument(
        "--factor-dir",
        default=str(FACTOR_DIR),
        help=f"Directory with .fea factor files (default: {FACTOR_DIR})",
    )
    parser.add_argument(
        "--target-dir",
        default=str(TARGET_DIR),
        help=f"Directory with target .fea files (default: {TARGET_DIR})",
    )
    parser.add_argument(
        "--target", default=TARGET_NAME,
        help=f"Target name without .fea extension (default: {TARGET_NAME})",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_DIR / "factor_quality_results.parquet"),
        help="Output file path for full results",
    )
    parser.add_argument(
        "--output-csv", default=str(OUTPUT_DIR / "factor_quality_results.csv"),
        help="Output CSV path for easy inspection",
    )
    parser.add_argument(
        "--factors", nargs="*",
        help="Specific factor names to analyze (omit for all 960)",
    )
    parser.add_argument(
        "--first-n", type=int, default=None,
        help="Analyze only the first N factors (for quick sampling)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count - 2)",
    )
    parser.add_argument(
        "--nan-threshold", type=float, default=0.20,
        help="NaN ratio threshold for flagging (default: 0.20)",
    )
    parser.add_argument(
        "--icir-threshold", type=float, default=0.05,
        help="|ICIR| threshold — below this flagged as low IC (default: 0.05)",
    )
    parser.add_argument(
        "--var-percentile", type=float, default=1.0,
        help="Bottom percentile for low-variance flag (default: 1.0)",
    )
    parser.add_argument(
        "--extreme-threshold", type=float, default=0.05,
        help="Extreme ratio threshold (default: 0.05)",
    )
    parser.add_argument(
        "--skip-correlation", action="store_true",
        help="Skip pairwise correlation / redundancy analysis",
    )
    parser.add_argument(
        "--corr-threshold", type=float, default=0.95,
        help="Correlation threshold for redundancy flagging (default: 0.95)",
    )

    args = parser.parse_args()

    factor_names = args.factors if args.factors else None
    if args.first_n is not None and factor_names is None:
        all_names = sorted(
            p.stem for p in Path(args.factor_dir).glob("*.fea")
            if p.suffix == ".fea"
        )
        factor_names = all_names[: args.first_n]
        print(f"[SAMPLE MODE] Analyzing first {len(factor_names)} of {len(all_names)} factors")
    elif factor_names is None:
        factor_names = sorted(
            p.stem for p in Path(args.factor_dir).glob("*.fea")
            if p.suffix == ".fea"
        )

    # Run analysis
    df = analyze_factors(
        factor_dir=Path(args.factor_dir),
        target_dir=Path(args.target_dir),
        target_name=args.target,
        factor_names=factor_names,
        max_workers=args.workers,
    )

    # Apply thresholds
    df = apply_thresholds(
        df,
        nan_threshold=args.nan_threshold,
        icir_threshold=args.icir_threshold,
        var_threshold_percentile=args.var_percentile,
        extreme_threshold=args.extreme_threshold,
    )

    # Print summary
    print_summary(df)

    output_parquet = Path(args.output)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    # ── Correlation / Redundancy Analysis ──
    if not args.skip_correlation and len(factor_names) >= 2:
        factor_dir = Path(args.factor_dir)
        target_path = str(Path(args.target_dir) / f"{args.target}.fea")
        pairs_df, _ = analyze_correlations(
            factor_dir=factor_dir,
            factor_names=factor_names,
            target_path=target_path,
            max_workers=args.workers,
            corr_threshold=args.corr_threshold,
        )
        if len(pairs_df) > 0:
            corr_pairs_path = output_parquet.parent / "factor_correlation_pairs.csv"
            pairs_df.to_csv(corr_pairs_path, index=False)
            print(f"\nHighly correlated pairs (|r| > {args.corr_threshold}): {len(pairs_df)}")
            print(f"Correlation pairs saved to: {corr_pairs_path}")
            if len(pairs_df) <= 20:
                for _, row in pairs_df.iterrows():
                    print(f"  {row['factor_a']} <-> {row['factor_b']}: {row['correlation']:.4f}")

            # Resolve which factor to flag in each pair
            df, redundant_set = resolve_redundant_flags(pairs_df, df, args.corr_threshold)
            print(f"Factors flagged as redundant: {len(redundant_set)}")
        else:
            print("\nNo highly correlated factor pairs found.")
            df["flag_redundant"] = False
    else:
        df["flag_redundant"] = False

    # ── Save full results ──
    df.to_parquet(output_parquet, index=False)
    print(f"\nResults saved to: {output_parquet}")

    output_csv = Path(args.output_csv)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    # ── Generate Bad-Factor List ──
    bad_list_path = output_parquet.parent / "bad_factor_list.csv"
    bad_df = generate_bad_factor_list(df, output_path=bad_list_path)

    # Also save a plain-text ID list (one per line)
    id_list_path = output_parquet.parent / "bad_factor_ids.txt"
    id_list_path.write_text("\n".join(bad_df["factor"].tolist()) + "\n")
    print(f"Bad factor IDs (plain text) saved to: {id_list_path}")
    print(f"\nTotal bad factors: {len(bad_df)} / {len(df)}")

    # ── Save quality summary JSON ──
    flag_fields = {
        "n_high_nan": int(df["flag_high_nan"].sum()),
        "n_low_ic": int(df["flag_low_ic"].sum()),
        "n_low_variance": int(df["flag_low_variance"].sum()),
        "n_extreme": int(df["flag_extreme"].sum()),
        "n_redundant": int(df["flag_redundant"].sum()),
    }
    summary = {
        "total_factors": len(df),
        "n_flagged_total": int(len(bad_df)),
        **flag_fields,
        "thresholds": {
            "nan_threshold": args.nan_threshold,
            "icir_threshold": args.icir_threshold,
            "var_threshold_percentile": args.var_percentile,
            "extreme_threshold": args.extreme_threshold,
            "corr_threshold": args.corr_threshold,
        },
    }
    summary_path = output_parquet.parent / "factor_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()
