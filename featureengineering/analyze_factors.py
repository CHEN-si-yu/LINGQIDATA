"""Comprehensive factor quality & IC analysis.

1. NaN / coverage analysis for every factor (from manifests + .fea files)
2. IC / ICIR matrix for all factors × all targets
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from featureengineering.settings import ProjectPaths
from featureengineering.ic_analysis import compute_ic, ICResult

# ── paths ──────────────────────────────────────────────────────────
paths = ProjectPaths.default()
FACTOR_DIR = paths.factor_output_dir
MANIFEST_DIR = paths.manifest_output_dir
TARGET_DIR = paths.target_output_dir

TARGETS = ["label_ret_1d", "label_ret_5d", "label_ret_10d", "label_ret_20d"]

# ── Part 1: NaN / coverage from manifests ──────────────────────────
print("=" * 90)
print("PART 1 — FACTOR COVERAGE & NaN ANALYSIS (from manifests)")
print("=" * 90)

manifest_files = sorted(MANIFEST_DIR.glob("*.json"))
# exclude target manifests
manifest_files = [f for f in manifest_files if not f.stem.startswith("label_")]

coverage_rows = []
for mf in manifest_files:
    try:
        data = json.loads(mf.read_text())
    except Exception:
        continue
    name = data.get("name", mf.stem)
    category = data.get("category", "unknown")
    rows = data.get("rows", 0)
    non_null = data.get("non_null_rows", 0)
    coverage = data.get("coverage_ratio", 0)
    dependencies = data.get("dependencies", [])
    built_at = data.get("built_at", "")

    coverage_rows.append({
        "factor": name,
        "category": category,
        "total_rows": rows,
        "non_null_rows": non_null,
        "nan_rows": rows - non_null,
        "coverage_pct": round(coverage * 100, 2),
        "dependencies": ", ".join(dependencies) if isinstance(dependencies, list) else str(dependencies),
        "built_at": built_at,
    })

df_cov = pd.DataFrame(coverage_rows)

# Summary stats
print(f"\nTotal factors with manifests: {len(df_cov)}")
print(f"\n--- Coverage distribution ---")
print(f"  Coverage == 100%:   {(df_cov['coverage_pct'] == 100).sum()} factors")
print(f"  99% <= cov < 100%:   {((df_cov['coverage_pct'] >= 99) & (df_cov['coverage_pct'] < 100)).sum()} factors")
print(f"  95% <= cov < 99%:    {((df_cov['coverage_pct'] >= 95) & (df_cov['coverage_pct'] < 99)).sum()} factors")
print(f"  90% <= cov < 95%:    {((df_cov['coverage_pct'] >= 90) & (df_cov['coverage_pct'] < 95)).sum()} factors")
print(f"  80% <= cov < 90%:    {((df_cov['coverage_pct'] >= 80) & (df_cov['coverage_pct'] < 90)).sum()} factors")
print(f"  Coverage < 80%:      {(df_cov['coverage_pct'] < 80).sum()} factors")
print(f"  Mean coverage:       {df_cov['coverage_pct'].mean():.2f}%")
print(f"  Median coverage:     {df_cov['coverage_pct'].median():.2f}%")

# Top 10 worst coverage factors
print(f"\n--- Top 10 WORST coverage factors ---")
worst = df_cov.nsmallest(10, "coverage_pct")
for _, row in worst.iterrows():
    print(f"  {row['factor']:40s} | cat={row['category']:15s} | "
          f"rows={row['total_rows']:>10,} | nan={row['nan_rows']:>10,} | "
          f"coverage={row['coverage_pct']:.2f}%")

# Coverage by category
print(f"\n--- Coverage by category ---")
for cat in sorted(df_cov["category"].unique()):
    sub = df_cov[df_cov["category"] == cat]
    print(f"  {cat:20s}: n={len(sub):>3} | mean_cov={sub['coverage_pct'].mean():.2f}% | "
          f"min={sub['coverage_pct'].min():.2f}% | max={sub['coverage_pct'].max():.2f}%")

# All factors with coverage < 100%
print(f"\n--- Factors with coverage < 99% ---")
low_cov = df_cov[df_cov["coverage_pct"] < 99].sort_values("coverage_pct")
if low_cov.empty:
    print("  (none — all factors >= 99%)")
else:
    for _, row in low_cov.iterrows():
        print(f"  {row['factor']:40s} | cat={row['category']:15s} | "
              f"coverage={row['coverage_pct']:.2f}% | nan_rows={row['nan_rows']:,}")

# ── Part 2: Deeper NaN check from actual .fea files ─────────────────
print("\n")
print("=" * 90)
print("PART 2 — DEEP NaN CHECK (from .fea files directly)")
print("=" * 90)

fea_files = sorted(FACTOR_DIR.glob("*.fea"))
fea_files = [f for f in fea_files if not f.stem.startswith("label_")]

nan_detail_rows = []
for ff in fea_files:
    try:
        df = pd.read_feather(ff)
    except Exception as e:
        nan_detail_rows.append({"factor": ff.stem, "error": str(e), "coverage_pct": 0})
        continue

    col = df.columns[0]
    total = len(df)
    non_null = df[col].notna().sum()
    nan_count = total - non_null
    coverage = non_null / total * 100 if total > 0 else 0

    # Check for inf values
    inf_count = 0
    if non_null > 0 and df[col].dtype in ("float64", "float32"):
        inf_count = int(np.isinf(df[col]).sum())

    # Dates covered
    if "Date" in df.index.names:
        dates = df.index.get_level_values("Date").unique()
    elif "Code" in df.index.names:
        dates = df.index.get_level_values("Date").unique()
    else:
        dates = []

    nan_detail_rows.append({
        "factor": ff.stem,
        "total_rows": total,
        "non_null": non_null,
        "nan_count": nan_count,
        "coverage_pct": round(coverage, 2),
        "inf_count": inf_count,
        "n_dates": len(dates),
        "dtype": str(df[col].dtype),
    })

df_nan = pd.DataFrame(nan_detail_rows)

# Factors with inf values
has_inf = df_nan[df_nan["inf_count"] > 0]
if not has_inf.empty:
    print(f"\n⚠️  Factors with Inf values ({len(has_inf)}):")
    for _, row in has_inf.iterrows():
        print(f"  {row['factor']:40s} | inf_count={row['inf_count']:,}")
else:
    print("\nNo Inf values found in any factor.")

# Factors with NaN > 5%
print(f"\n--- Factors with NaN rate > 5% ---")
high_nan = df_nan[df_nan["coverage_pct"] < 95].sort_values("coverage_pct")
if high_nan.empty:
    print("  (none)")
else:
    for _, row in high_nan.iterrows():
        print(f"  {row['factor']:40s} | coverage={row['coverage_pct']:.2f}% | "
              f"nan={row['nan_count']:>10,} / {row['total_rows']:>10,} | "
              f"dates={row.get('n_dates', '?')}")

# ── Part 3: IC / ICIR Analysis ─────────────────────────────────────
print("\n")
print("=" * 90)
print("PART 3 — IC / ICIR ANALYSIS (all factors × all targets, rank IC)")
print("=" * 90)

t0 = time.monotonic()

# Filter to factors that actually have .fea files
available_factors = sorted([f.stem for f in FACTOR_DIR.glob("*.fea")
                            if not f.stem.startswith("label_")])
available_targets = sorted([f.stem for f in TARGET_DIR.glob("*.fea")
                            if f.stem.startswith("label_")])

print(f"Factors to evaluate: {len(available_factors)}")
print(f"Targets: {available_targets}")

ic_rows = []
n = len(available_factors) * len(available_targets)
done = 0

for fn in available_factors:
    for tn in available_targets:
        try:
            r: ICResult = compute_ic(fn, tn, method="rank")
            ic_rows.append({
                "factor": fn,
                "target": tn,
                "ic_mean": r.ic_mean,
                "ic_std": r.ic_std,
                "icir": r.icir,
                "ic_positive_ratio": r.ic_positive_ratio,
                "n_dates": r.n_dates,
            })
        except Exception as e:
            ic_rows.append({
                "factor": fn, "target": tn,
                "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "ic_positive_ratio": np.nan,
                "n_dates": 0, "error": str(e)[:80],
            })
        done += 1
        if done % 40 == 0:
            elapsed = time.monotonic() - t0
            print(f"  progress: {done}/{n}  ({elapsed:.0f}s)")

elapsed = time.monotonic() - t0
print(f"IC calculation done in {elapsed:.0f}s")

df_ic = pd.DataFrame(ic_rows)

# ── Summary tables ─────────────────────────────────────────────────
print(f"\n--- IC Summary by Target ---")
for tn in available_targets:
    sub = df_ic[(df_ic["target"] == tn) & df_ic["ic_mean"].notna()]
    if sub.empty:
        continue
    print(f"\n  {tn}:")
    print(f"    Mean |IC|:        {sub['ic_mean'].abs().mean():.5f}")
    print(f"    Mean ICIR:        {sub['icir'].mean():+.4f}")
    print(f"    Max IC (abs):     {sub.loc[sub['ic_mean'].abs().idxmax(), 'factor']} "
          f"({sub['ic_mean'].abs().max():+.5f})")
    print(f"    Max ICIR (abs):   {sub.loc[sub['icir'].abs().idxmax(), 'factor']} "
          f"({sub['icir'].abs().max():+.4f})")
    sig = sub[sub["icir"].abs() > 0.3]
    print(f"    |ICIR| > 0.3:     {len(sig)} factors")

# ── Top 20 factors by |ICIR| ────────────────────────────────────────
print(f"\n--- Top 20 factors by absolute ICIR ---")
# Pivot
pivot_icir = df_ic.pivot_table(
    index="factor", columns="target", values="icir", aggfunc="first"
)
pivot_icir["max_abs_icir"] = pivot_icir.abs().max(axis=1)
pivot_icir["best_target"] = pivot_icir.abs().idxmax(axis=1)

top20 = pivot_icir.nlargest(20, "max_abs_icir")
for idx, row in top20.iterrows():
    best_t = row["best_target"]
    icir_val = row[best_t]
    ic_mean = df_ic[(df_ic["factor"] == idx) & (df_ic["target"] == best_t)]["ic_mean"].values[0]
    print(f"  {idx:38s} | {best_t:15s} | "
          f"IC={ic_mean:+.5f} | ICIR={icir_val:+.4f}")

# ── Bottom 10 factors (worst IC signal) ────────────────────────────
print(f"\n--- Bottom 10 factors by absolute ICIR (weakest signal) ---")
bot10 = pivot_icir.nsmallest(10, "max_abs_icir")
for idx, row in bot10.iterrows():
    best_t = row["best_target"]
    icir_val = row[best_t]
    ic_mean = df_ic[(df_ic["factor"] == idx) & (df_ic["target"] == best_t)]["ic_mean"].values[0]
    print(f"  {idx:38s} | {best_t:15s} | "
          f"IC={ic_mean:+.5f} | ICIR={icir_val:+.4f}")

# ── Per-category ICIR summary ──────────────────────────────────────
print(f"\n--- Average |ICIR| by factor category ---")
factor_cat_map = dict(zip(df_cov["factor"], df_cov["category"]))
pivot_icir["category"] = pivot_icir.index.map(factor_cat_map).fillna("unknown")
for cat in sorted(pivot_icir["category"].unique()):
    sub = pivot_icir[pivot_icir["category"] == cat]
    print(f"  {cat:20s}: n={len(sub):>3} | avg_max_abs_icir={sub['max_abs_icir'].mean():.4f} | "
          f"max={sub['max_abs_icir'].max():.4f}")

# ── Full IC matrix by target (pivot) ────────────────────────────────
print(f"\n--- Full IC Mean matrix (factor × target) ---")
pivot_ic = df_ic.pivot_table(
    index="factor", columns="target", values="ic_mean", aggfunc="first"
)
# Add category
pivot_ic["category"] = pivot_ic.index.map(factor_cat_map).fillna("unknown")
pivot_ic = pivot_ic.sort_values(["category", "label_ret_10d"], ascending=[True, True])
print(pivot_ic.to_string(max_rows=len(pivot_ic)))

# ── Save results ────────────────────────────────────────────────────
output_excel = paths.project_root / "data" / "factor_analysis.xlsx"
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_cov.to_excel(writer, sheet_name="coverage", index=False)
    df_nan.to_excel(writer, sheet_name="nan_detail", index=False)
    df_ic.to_excel(writer, sheet_name="ic_results", index=False)
    pivot_ic.reset_index().to_excel(writer, sheet_name="ic_pivot", index=False)

print(f"\nFull results saved to: {output_excel}")
print("Done.")
