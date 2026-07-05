#!/usr/bin/env python3
"""
Standalone correlation / redundancy analysis for the 960 factors.
Runs after factor_quality.py has already computed the quality metrics.

Uses stock sampling to keep memory manageable:
  - Samples 200 representative stocks
  - Samples 20 dates evenly
  - Computes pairwise cross-sectional Pearson correlation
  - Flags redundant pairs and generates the final bad-factor list.
"""

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

FACTOR_DIR = Path("/root/autodl-fs/lingqiData/featureengineering/data/factors")
TARGET_DIR = Path("/root/autodl-fs/lingqiData/featureengineering/data/targets")
OUTPUT_DIR = Path("/root/autodl-fs/lingqiData/featureengineering/quality")

N_SAMPLE_DATES = 20
N_SAMPLE_STOCKS = 500
CORR_THRESHOLD = 0.95


def _extract_snapshot(args: tuple[str, list[str], list[str], str]) -> dict | None:
    """Worker: extract factor values for sampled dates × sampled stocks."""
    factor_path, sampled_dates, sampled_stocks, factor_name = args
    try:
        factor = pd.read_feather(factor_path)
        factor = factor.apply(pd.to_numeric, errors="coerce")
        common_dates = factor.index.intersection(sampled_dates)
        common_stocks = factor.columns.intersection(sampled_stocks)
        if len(common_dates) == 0 or len(common_stocks) == 0:
            return None
        snap = factor.loc[common_dates, common_stocks].values.astype(np.float64)
        return {"name": factor_name, "data": snap, "dates": list(common_dates)}
    except Exception as e:
        print(f"  [WARN] Failed {factor_name}: {e}", file=sys.stderr)
        return None


def main():
    # ── Load quality results ──
    results_path = OUTPUT_DIR / "factor_quality_results.parquet"
    if not results_path.exists():
        print("ERROR: factor_quality_results.parquet not found. Run factor_quality.py first.")
        sys.exit(1)

    quality_df = pd.read_parquet(results_path)
    factor_names = quality_df["factor"].tolist()
    print(f"Loaded quality results for {len(factor_names)} factors")

    # ── Determine date range and stock pool ──
    target = pd.read_feather(str(TARGET_DIR / "label_ret_1d.fea"))
    all_dates = sorted(target.index.tolist())
    all_stocks = sorted(target.columns.tolist())

    step = max(1, len(all_dates) // N_SAMPLE_DATES)
    sampled_dates = all_dates[::step][:N_SAMPLE_DATES]
    print(f"Sampled {len(sampled_dates)} dates from {all_dates[0]} to {all_dates[-1]}")

    # Sample stocks: pick evenly to ensure coverage
    stock_step = max(1, len(all_stocks) // N_SAMPLE_STOCKS)
    sampled_stocks = all_stocks[::stock_step][:N_SAMPLE_STOCKS]
    print(f"Sampled {len(sampled_stocks)} stocks from {len(all_stocks)} total")

    # ── Extract factor snapshots ──
    print(f"Extracting snapshots for {len(factor_names)} factors...")
    work_items = [
        (str(FACTOR_DIR / f"{fn}.fea"), sampled_dates, sampled_stocks, fn)
        for fn in factor_names
    ]

    snapshots = {}  # name -> np.ndarray (n_dates, n_stocks)
    snapshot_dates = {}  # name -> list of dates

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(_extract_snapshot, item): item[3]
            for item in work_items
        }
        with tqdm(total=len(futures), desc="Extracting", unit="factor", ncols=100) as pbar:
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res is not None:
                        snapshots[res["name"]] = res["data"]
                        snapshot_dates[res["name"]] = res["dates"]
                except Exception:
                    pass
                pbar.update(1)

    print(f"Extracted snapshots for {len(snapshots)} factors")
    names = sorted(snapshots.keys())
    n_eff = len(names)
    n_dates = len(sampled_dates)

    # ── Compute average cross-sectional correlation ──
    print(f"Computing {n_eff}×{n_eff} cross-sectional correlations across {n_dates} dates...")

    corr_sum = np.zeros((n_eff, n_eff))
    valid_counts = np.zeros((n_eff, n_eff))

    for d_idx in tqdm(range(n_dates), desc="Per-date corr", unit="date", ncols=100):
        rows = []
        valid_idx = []
        for i, name in enumerate(names):
            snap = snapshots[name]
            if d_idx < snap.shape[0]:
                row = snap[d_idx]
                rows.append(row)
                valid_idx.append(i)

        if len(rows) < 2:
            continue

        mat = np.array(rows, dtype=np.float64)
        # Fill NaN with 0 (most factors are normalized around 0, so 0 is neutral)
        # This avoids the listwise-deletion problem with 960 factors where
        # every stock has NaN in at least one factor.
        mat_filled = np.nan_to_num(mat, nan=0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(mat_filled)
            corr = np.nan_to_num(corr, nan=0.0)

        for ii, i_full in enumerate(valid_idx):
            for jj, j_full in enumerate(valid_idx):
                if i_full < j_full:
                    corr_sum[i_full, j_full] += corr[ii, jj]
                    valid_counts[i_full, j_full] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_corr = np.where(valid_counts > 0, corr_sum / valid_counts, 0.0)

    # ── Find highly correlated pairs ──
    pairs = []
    # Also collect top pairs for diagnostic output
    all_corr_values = []
    for i in range(n_eff):
        for j in range(i + 1, n_eff):
            if valid_counts[i, j] >= 5:
                c = abs(avg_corr[i, j])
                all_corr_values.append(c)
                if c > CORR_THRESHOLD:
                    pairs.append({
                        "factor_a": names[i],
                        "factor_b": names[j],
                        "correlation": float(avg_corr[i, j]),
                        "n_dates": int(valid_counts[i, j]),
                    })

    # Diagnostic: show distribution of top absolute correlations
    if all_corr_values:
        all_corr_values.sort(reverse=True)
        print(f"\nTop 20 absolute correlations (avg across dates):")
        for k in range(min(20, len(all_corr_values))):
            print(f"  #{k+1}: {all_corr_values[k]:.6f}")
        print(f"Max corr: {all_corr_values[0]:.6f}")
        above_90 = sum(1 for c in all_corr_values if c > 0.90)
        above_85 = sum(1 for c in all_corr_values if c > 0.85)
        above_80 = sum(1 for c in all_corr_values if c > 0.80)
        print(f"Pairs with |avg_corr| > 0.90: {above_90}")
        print(f"Pairs with |avg_corr| > 0.85: {above_85}")
        print(f"Pairs with |avg_corr| > 0.80: {above_80}")

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values("correlation", ascending=False).reset_index(drop=True)

    pairs_path = OUTPUT_DIR / "factor_correlation_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    print(f"\nHighly correlated pairs (|r| > {CORR_THRESHOLD}): {len(pairs_df)}")
    print(f"Saved to: {pairs_path}")

    if len(pairs_df) <= 30:
        for _, row in pairs_df.iterrows():
            print(f"  {row['factor_a']} <-> {row['factor_b']}: {row['correlation']:+.4f}  (n={row['n_dates']})")

    # ── Resolve redundant flags ──
    qlookup = quality_df.set_index("factor")
    quality_df["flag_redundant"] = False
    redundant = set()

    for _, row in pairs_df.iterrows():
        a, b = row["factor_a"], row["factor_b"]
        if a not in qlookup.index or b not in qlookup.index:
            continue
        icir_a = abs(qlookup.loc[a, "rank_icir"])
        icir_b = abs(qlookup.loc[b, "rank_icir"])
        if pd.isna(icir_a):
            worse = a
        elif pd.isna(icir_b):
            worse = b
        elif abs(icir_a - icir_b) < 1e-6:
            nan_a = qlookup.loc[a, "nan_ratio"]
            nan_b = qlookup.loc[b, "nan_ratio"]
            worse = a if nan_a > nan_b else b
        else:
            worse = a if icir_a < icir_b else b
        redundant.add(worse)

    if redundant:
        quality_df.loc[quality_df["factor"].isin(redundant), "flag_redundant"] = True
    print(f"Factors flagged as redundant: {len(redundant)}")

    # ── Update results ──
    quality_df.to_parquet(results_path, index=False)
    quality_df.to_csv(OUTPUT_DIR / "factor_quality_results.csv", index=False)
    print(f"Updated results saved to: {results_path}")

    # ── Generate final bad-factor list ──
    flag_cols = ["flag_high_nan", "flag_low_ic", "flag_low_variance", "flag_extreme", "flag_redundant"]
    bad_mask = quality_df[flag_cols].any(axis=1)
    bad_df = quality_df.loc[bad_mask].copy()

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
    bad_df["n_flags"] = bad_df[flag_cols].sum(axis=1)
    bad_df = bad_df.sort_values(["n_flags", "rank_icir"], ascending=[False, True]).reset_index(drop=True)

    out_cols = ["factor", "quality_issues", "n_flags",
                "nan_ratio", "rank_icir", "rank_ic_mean",
                "xs_std_mean", "extreme_ratio"]
    out_cols = [c for c in out_cols if c in bad_df.columns]
    result = bad_df[out_cols]

    bad_list_path = OUTPUT_DIR / "bad_factor_list.csv"
    result.to_csv(bad_list_path, index=False)
    print(f"Bad-factor list saved to: {bad_list_path}")

    id_list_path = OUTPUT_DIR / "bad_factor_ids.txt"
    id_list_path.write_text("\n".join(result["factor"].tolist()) + "\n")
    print(f"Bad factor IDs saved to: {id_list_path}")

    # ── Update summary ──
    import json
    summary = {
        "total_factors": len(quality_df),
        "n_flagged_total": int(len(result)),
        "n_high_nan": int(quality_df["flag_high_nan"].sum()),
        "n_low_ic": int(quality_df["flag_low_ic"].sum()),
        "n_low_variance": int(quality_df["flag_low_variance"].sum()),
        "n_extreme": int(quality_df["flag_extreme"].sum()),
        "n_redundant": int(quality_df["flag_redundant"].sum()),
        "thresholds": {
            "nan_threshold": 0.20,
            "icir_threshold": 0.05,
            "var_threshold_percentile": 1.0,
            "extreme_threshold": 0.05,
            "corr_threshold": CORR_THRESHOLD,
        },
        "correlation_pairs": int(len(pairs_df)),
    }
    summary_path = OUTPUT_DIR / "factor_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary updated: {summary_path}")
    print(f"\nFinal: {len(result)} bad factors out of {len(quality_df)} total ({len(result)/len(quality_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
