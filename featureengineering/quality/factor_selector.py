#!/usr/bin/env python3
"""
Generate a prioritized factor quality ranking, then create fac_select.fea
by keeping the best K factors from fac_all.fea.

Priority order for removal (worst first):
  Tier 1: Zero variance (xs_std_mean == 0)              → must remove
  Tier 2: 100% NaN                                       → must remove
  Tier 3: High NaN (>20%)                                → strongly remove
  Tier 4: Very low |ICIR| (ascending |ICIR|)             → negligible signal
  Tier 5: Redundant (lower |ICIR| of each corr pair)    → duplicates
  Tier 6: Low variance (ascending xs_std_mean)           → weak signal

Within each tier: sorted by severity (worst first).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Resolve paths correctly (handle symlinks like /root/autodl-fs → /autodl-fs/data)
_SCRIPT_DIR = Path(__file__).resolve().parent  # .../featureengineering/quality
_FE_DIR = _SCRIPT_DIR.parent                    # .../featureengineering
_PROJECT_ROOT = _FE_DIR.parent                  # .../lingqiData

QUALITY_DIR = _SCRIPT_DIR
QUALITY_PATH = QUALITY_DIR / "factor_quality_results.csv"
CORR_PAIRS_PATH = QUALITY_DIR / "factor_correlation_pairs.csv"

OUT_DIR = _PROJECT_ROOT / "trainingdata"
FAC_ALL_PATH = OUT_DIR / "fac_all.fea"
FAC_SELECT_PATH = OUT_DIR / "fac_select.fea"
RANKING_PATH = QUALITY_DIR / "factor_priority_ranking.csv"

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 500  # Default: keep top 500 factors


def build_priority_ranking(
    quality_path: Path = QUALITY_PATH,
    corr_pairs_path: Path = CORR_PAIRS_PATH,
) -> pd.DataFrame:
    """Build a prioritized factor ranking (worst → best).

    Returns a DataFrame with columns:
      factor, tier, reason, rank, nan_ratio, rank_icir, xs_std_mean,
      flag_high_nan, flag_low_ic, flag_low_variance, flag_redundant
    sorted by removal priority (worst first, best last).
    """
    df = pd.read_csv(quality_path)

    # ── Assign tiers ──
    df["tier"] = 99  # Default: no issues
    df["reason"] = ""

    # Tier 1: Zero variance (completely constant across stocks)
    mask_t1 = df["xs_std_mean"] <= 1e-10
    df.loc[mask_t1, "tier"] = 1
    df.loc[mask_t1, "reason"] = "zero_variance"

    # Tier 2: 100% NaN (no data at all)
    mask_t2 = (df["nan_ratio"] >= 0.999) & (df["tier"] == 99)
    df.loc[mask_t2, "tier"] = 2
    df.loc[mask_t2, "reason"] = "all_nan"

    # Tier 3: High NaN (>20%)
    mask_t3 = (df["flag_high_nan"]) & (df["tier"] == 99)
    df.loc[mask_t3, "tier"] = 3
    df.loc[mask_t3, "reason"] = "high_nan"

    # Tier 4: Very low |ICIR| — no predictive power
    mask_t4 = (df["flag_low_ic"]) & (df["tier"] == 99)
    df.loc[mask_t4, "tier"] = 4
    df.loc[mask_t4, "reason"] = "low_ic"

    # Tier 5: Redundant (highly correlated with a better factor)
    mask_t5 = (df["flag_redundant"]) & (df["tier"] == 99)
    df.loc[mask_t5, "tier"] = 5
    df.loc[mask_t5, "reason"] = "redundant"

    # Tier 6: Low variance (bottom 1% that aren't already Tier 1)
    mask_t6 = (df["flag_low_variance"]) & (df["tier"] == 99)
    df.loc[mask_t6, "tier"] = 6
    df.loc[mask_t6, "reason"] = "low_variance"

    # ── Sort within each tier ──
    # Tier 1: by xs_std ascending (closest to zero first)
    # Tier 2: by nan_ratio descending (most NaN first)
    # Tier 3: by nan_ratio descending
    # Tier 4: by |ICIR| ascending (closest to zero first)
    # Tier 5: by |ICIR| ascending (worst of each pair first)
    # Tier 6: by xs_std ascending (lowest variance first)
    # Tier 99 (clean): by |ICIR| descending (best factors last, for removal ordering)

    df["_sort_icir_abs"] = df["rank_icir"].abs().fillna(0)
    df["_sort_nan"] = df["nan_ratio"].fillna(0)
    df["_sort_var"] = df["xs_std_mean"].fillna(0)

    # Build the sort key: (tier, tier_specific_metric)
    # Lower sort key = higher removal priority
    sort_values = []
    for _, row in df.iterrows():
        tier = row["tier"]
        if tier == 1:
            key = (tier, row["_sort_var"])          # ascending xs_std
        elif tier == 2:
            key = (tier, -row["_sort_nan"])         # descending NaN
        elif tier == 3:
            key = (tier, -row["_sort_nan"])         # descending NaN
        elif tier == 4:
            key = (tier, row["_sort_icir_abs"])     # ascending |ICIR| (zero first)
        elif tier == 5:
            key = (tier, row["_sort_icir_abs"])     # ascending |ICIR|
        elif tier == 6:
            key = (tier, row["_sort_var"])          # ascending xs_std
        else:
            key = (tier, row["_sort_icir_abs"])      # ascending |ICIR| (weakest first)
        sort_values.append(key)

    df["_sort_key"] = sort_values
    df = df.sort_values("_sort_key").reset_index(drop=True)

    # ── Assign final rank (1 = worst, 960 = best) ──
    df["priority_rank"] = range(1, len(df) + 1)

    # ── Clean output ──
    out_cols = [
        "priority_rank", "factor", "tier", "reason",
        "nan_ratio", "rank_icir", "rank_ic_mean", "xs_std_mean",
        "flag_high_nan", "flag_low_ic", "flag_low_variance", "flag_redundant",
    ]
    result = df[out_cols].copy()

    # Drop temporary columns from original df
    df.drop(columns=["_sort_icir_abs", "_sort_nan", "_sort_var", "_sort_key"], inplace=True, errors="ignore")

    return result


def generate_fac_select(
    ranking_df: pd.DataFrame,
    fac_all_path: Path = FAC_ALL_PATH,
    output_path: Path = FAC_SELECT_PATH,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Load fac_all.fea, keep only the best top_k factors, save as fac_select.fea.

    Parameters
    ----------
    ranking_df : pd.DataFrame
        Priority ranking from build_priority_ranking().
    fac_all_path : Path
        Path to the full factor file.
    output_path : Path
        Where to write fac_select.fea.
    top_k : int
        Number of factors to keep.

    Returns
    -------
    pd.DataFrame
        The selected factor DataFrame (long format: date, Code, factor_cols...).
    """
    print(f"Loading fac_all.fea from {fac_all_path}...")
    fac_all = pd.read_feather(fac_all_path)
    print(f"  Shape: {fac_all.shape}")

    # Get all factor columns (exclude date, Code)
    all_factor_cols = sorted(c for c in fac_all.columns if c not in ("date", "Code"))
    n_all = len(all_factor_cols)
    print(f"  Total factor columns: {n_all}")

    # Factors to remove: the worst-ranked ones (highest priority_rank = worst)
    # ranking_df is sorted worst-first. Keep only factors NOT in the first N_remove.
    n_remove = max(0, n_all - top_k)
    if n_remove == 0:
        print(f"  top_k={top_k} >= n_factors={n_all}, keeping all factors.")
        keep_factors = set(all_factor_cols)
    else:
        # Take the worst n_remove factors
        remove_factors = set(ranking_df["factor"].head(n_remove).tolist())
        keep_factors = set(all_factor_cols) - remove_factors
        print(f"  Removing {n_remove} factors, keeping {len(keep_factors)}")

    # Build the output: date, Code, then kept factors in alphabetical order
    keep_cols = ["date", "Code"] + sorted(keep_factors)
    # Only keep columns that actually exist in fac_all
    keep_cols = [c for c in keep_cols if c in fac_all.columns]

    fac_select = fac_all[keep_cols].copy()
    print(f"  Output shape: {fac_select.shape}")

    fac_select.to_feather(output_path)
    print(f"  Saved to: {output_path}")

    return fac_select


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build factor priority ranking and generate fac_select.fea"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of factors to keep (default: {DEFAULT_TOP_K})")
    parser.add_argument("--ranking-only", action="store_true",
                        help="Only build ranking, don't generate fac_select.fea")
    parser.add_argument("--select-only", action="store_true",
                        help="Only generate fac_select.fea (ranking must exist)")
    args = parser.parse_args()

    if not args.select_only:
        print("=" * 60)
        print("Building factor priority ranking...")
        print("=" * 60)
        ranking = build_priority_ranking()
        ranking.to_csv(RANKING_PATH, index=False)
        print(f"\nRanking saved to: {RANKING_PATH}")

        # Print summary
        print(f"\nTier breakdown:")
        for tier in sorted(ranking["tier"].unique()):
            n_tier = int((ranking["tier"] == tier).sum())
            label = {
                1: "Zero variance",
                2: "100% NaN",
                3: "High NaN (>20%)",
                4: "Low |ICIR|",
                5: "Redundant",
                6: "Low variance",
                99: "Clean (no issues)",
            }.get(tier, f"Unknown")
            print(f"  Tier {tier} ({label}): {n_tier}")

        # Show worst 20
        print(f"\nWorst 20 factors (highest removal priority):")
        for _, row in ranking.head(20).iterrows():
            print(f"  #{row['priority_rank']:3d}  {row['factor']:<45s}  "
                  f"T{int(row['tier']):d}-{row['reason']:<20s}  "
                  f"NaN={row['nan_ratio']:.3f}  |ICIR|={abs(row['rank_icir']):.4f}")

        print(f"\nBest 10 factors (lowest removal priority):")
        for _, row in ranking.tail(10).iterrows():
            print(f"  #{row['priority_rank']:3d}  {row['factor']:<45s}  "
                  f"|ICIR|={abs(row['rank_icir']):.4f}")

    if not args.ranking_only:
        print(f"\n{'='*60}")
        print(f"Generating fac_select.fea (top K = {args.top_k})...")
        print(f"{'='*60}")
        ranking = pd.read_csv(RANKING_PATH)
        generate_fac_select(ranking, top_k=args.top_k)
