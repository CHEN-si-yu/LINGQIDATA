#!/usr/bin/env python3
"""
Factor Quality Analysis & Priority Ranking
===========================================
完整流程:
  1. 对全部 960 个因子计算质量指标 (NaN / 方差 / IC / 极端值)
  2. 计算因子间截面相关性, 标记冗余因子
  3. 按优先级生成剔除排序 (priority_ranking.csv, 1=最差, 960=最优)

输出 (写入 featureengineering/quality/):
  - factor_quality_results.parquet  全部因子质量指标
  - factor_correlation_pairs.csv    高度相关的因子对
  - factor_priority_ranking.csv     按剔除优先级排序 (供 prepared_data.py 读取)

用法:
  python analysis_factors.py                          # 全量 960 因子
  python analysis_factors.py --first-n 50 --workers 8  # 小样本测试
  python analysis_factors.py --skip-correlation        # 跳过相关性分析
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis as kurtosis_scipy
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*correlation coefficient.*")
warnings.filterwarnings("ignore", message=".*Precision loss.*")

# ── Paths (relative to this script) ──────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent  # .../featureengineering/
FACTOR_DIR = _SCRIPT_DIR / "data" / "factors"
TARGET_DIR = _SCRIPT_DIR / "data" / "targets"
TARGET_NAME = "label_ret_1d"
OUTPUT_DIR = _SCRIPT_DIR / "quality"

# ── Config ───────────────────────────────────────────────────────────────────
N_SAMPLE_DATES = 20       # 相关性采样日期数
N_SAMPLE_STOCKS = 500     # 相关性采样股票数
CORR_THRESHOLD = 0.95     # 相关性阈值
DEFAULT_WORKERS = 16


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  Part 1: Quality Metrics (per-factor)                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class QualityResult:
    factor_name: str
    n_dates: int
    n_codes: int
    nan_ratio: float
    nan_ratio_per_date_mean: float
    nan_ratio_per_date_max: float
    xs_var_mean: float
    xs_var_std: float
    xs_std_mean: float
    ic_n_dates: int
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float
    rank_ic_pos_ratio: float
    rank_ic_tstat: float
    global_mean: float
    global_std: float
    skewness: float
    kurtosis: float
    extreme_ratio: float


def _compute_single_factor(args: tuple) -> QualityResult | None:
    """Worker: compute all quality metrics for one factor."""
    factor_path, target_path, factor_name = args
    try:
        factor = pd.read_feather(factor_path)
        target = pd.read_feather(target_path)
        factor = factor.apply(pd.to_numeric, errors="coerce")
        target = target.apply(pd.to_numeric, errors="coerce")
    except Exception:
        return None

    n_dates, n_codes = len(factor), len(factor.columns)
    total_cells = max(n_dates * n_codes, 1)

    # ── NaN ──
    nan_count = int(factor.isna().sum().sum())
    nan_ratio = nan_count / total_cells
    per_date_nan = factor.isna().sum(axis=1) / max(n_codes, 1)
    nan_ratio_per_date_mean = float(per_date_nan.mean())
    nan_ratio_per_date_max = float(per_date_nan.max())

    # ── Variance ──
    fvals = factor.values
    xs_std_arr = np.nanstd(fvals, axis=1)
    xs_var_arr = xs_std_arr ** 2
    xs_var_mean = float(np.nanmean(xs_var_arr))
    xs_var_std = float(np.nanstd(xs_var_arr))
    xs_std_mean = float(np.nanmean(xs_std_arr))

    # ── Distribution ──
    flat = fvals.ravel()
    flat_finite = flat[np.isfinite(flat)]
    global_mean = float(np.mean(flat_finite)) if len(flat_finite) > 0 else float("nan")
    global_std = float(np.std(flat_finite)) if len(flat_finite) > 0 else float("nan")

    skew_vals, kurt_vals, extreme_ratios = [], [], []
    for i in range(n_dates):
        row = fvals[i]
        row_f = row[np.isfinite(row)]
        if len(row_f) < 10:
            continue
        try:
            skew_vals.append(float(skew(row_f)))
            kurt_vals.append(float(kurtosis_scipy(row_f)))
        except (ValueError, RuntimeWarning):
            pass
        r_std = np.std(row_f)
        if r_std > 0:
            extreme_ratios.append(
                float(np.sum(np.abs(row_f - np.mean(row_f)) > 5 * r_std) / len(row_f))
            )
    skewness = float(np.mean(skew_vals)) if skew_vals else float("nan")
    kurtosis = float(np.mean(kurt_vals)) if kurt_vals else float("nan")
    extreme_ratio = float(np.mean(extreme_ratios)) if extreme_ratios else float("nan")

    # ── Rank IC vs target ──
    common_codes = factor.columns.intersection(target.columns).sort_values()
    common_dates = factor.index.intersection(target.index).sort_values()
    ic_values = []
    if len(common_codes) >= 10 and len(common_dates) > 0:
        f_a = factor.loc[common_dates, common_codes]
        t_a = target.loc[common_dates, common_codes]
        for date in common_dates:
            f_row = f_a.loc[date]; t_row = t_a.loc[date]
            mask = f_row.notna() & t_row.notna()
            if mask.sum() < 10:
                continue
            ic = f_row[mask].corr(t_row[mask], method="spearman")
            ic_values.append(float(ic))
    ic_series = pd.Series(ic_values).dropna()
    ic_n = len(ic_series)
    if ic_n > 0:
        ric_mean = float(ic_series.mean())
        ric_std = float(ic_series.std(ddof=1))
        ric_ir = ric_mean / ric_std if ric_std > 0 else float("nan")
        ric_pos = float((ic_series > 0).mean())
        ric_t = ric_mean / (ric_std / np.sqrt(ic_n)) if ric_std > 0 else float("nan")
    else:
        ric_mean = ric_std = ric_ir = ric_pos = ric_t = float("nan")

    return QualityResult(
        factor_name=factor_name, n_dates=n_dates, n_codes=n_codes,
        nan_ratio=nan_ratio, nan_ratio_per_date_mean=nan_ratio_per_date_mean,
        nan_ratio_per_date_max=nan_ratio_per_date_max,
        xs_var_mean=xs_var_mean, xs_var_std=xs_var_std, xs_std_mean=xs_std_mean,
        ic_n_dates=ic_n, rank_ic_mean=ric_mean, rank_ic_std=ric_std,
        rank_icir=ric_ir, rank_ic_pos_ratio=ric_pos, rank_ic_tstat=ric_t,
        global_mean=global_mean, global_std=global_std,
        skewness=skewness, kurtosis=kurtosis, extreme_ratio=extreme_ratio,
    )


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  Part 2: Correlation / Redundancy Analysis                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def _extract_snapshot(args: tuple) -> dict | None:
    """Worker: extract factor values on sampled dates × stocks."""
    factor_path, sampled_dates, sampled_stocks, factor_name = args
    try:
        factor = pd.read_feather(factor_path)
        factor = factor.apply(pd.to_numeric, errors="coerce")
        common_dates = factor.index.intersection(sampled_dates)
        common_stocks = factor.columns.intersection(sampled_stocks)
        if len(common_dates) == 0 or len(common_stocks) == 0:
            return None
        snap = factor.loc[common_dates, common_stocks].values.astype(np.float64)
        return {"name": factor_name, "data": snap}
    except Exception:
        return None


def analyze_correlations(
    factor_names: list[str],
    max_workers: int,
) -> pd.DataFrame:
    """Compute pairwise cross-sectional correlation, return pairs > threshold."""
    target = pd.read_feather(str(TARGET_DIR / f"{TARGET_NAME}.fea"))
    all_dates = sorted(target.index.tolist())
    all_stocks = sorted(target.columns.tolist())

    step_d = max(1, len(all_dates) // N_SAMPLE_DATES)
    sampled_dates = all_dates[::step_d][:N_SAMPLE_DATES]
    step_s = max(1, len(all_stocks) // N_SAMPLE_STOCKS)
    sampled_stocks = all_stocks[::step_s][:N_SAMPLE_STOCKS]
    print(f"  Correlation: {len(sampled_dates)} dates × {len(sampled_stocks)} stocks")

    # Extract snapshots
    work_items = [
        (str(FACTOR_DIR / f"{fn}.fea"), sampled_dates, sampled_stocks, fn)
        for fn in factor_names
    ]
    snapshots: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_extract_snapshot, item): item[3] for item in work_items}
        with tqdm(total=len(futures), desc="  Extract snapshots", unit="f", ncols=100) as p:
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    snapshots[res["name"]] = res["data"]
                p.update(1)

    names = sorted(snapshots.keys())
    n_eff = len(names)
    n_dates = len(sampled_dates)
    print(f"  Computing {n_eff}×{n_eff} cross-sectional correlations...")

    corr_sum = np.zeros((n_eff, n_eff))
    valid_counts = np.zeros((n_eff, n_eff))

    for d_idx in range(n_dates):
        rows, valid_idx = [], []
        for i, name in enumerate(names):
            snap = snapshots[name]
            if d_idx < snap.shape[0]:
                rows.append(snap[d_idx])
                valid_idx.append(i)
        if len(rows) < 2:
            continue
        mat = np.array(rows, dtype=np.float64)
        mat_filled = np.nan_to_num(mat, nan=0.0)
        corr = np.corrcoef(mat_filled)
        corr = np.nan_to_num(corr, nan=0.0)
        for ii, i_full in enumerate(valid_idx):
            for jj, j_full in enumerate(valid_idx):
                if i_full < j_full:
                    corr_sum[i_full, j_full] += corr[ii, jj]
                    valid_counts[i_full, j_full] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_corr = np.where(valid_counts > 0, corr_sum / valid_counts, 0.0)

    pairs = []
    for i in range(n_eff):
        for j in range(i + 1, n_eff):
            if valid_counts[i, j] >= 5 and abs(avg_corr[i, j]) > CORR_THRESHOLD:
                pairs.append({
                    "factor_a": names[i], "factor_b": names[j],
                    "correlation": float(avg_corr[i, j]),
                    "n_dates": int(valid_counts[i, j]),
                })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values("correlation", ascending=False).reset_index(drop=True)
    return pairs_df


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  Part 3: Priority Ranking                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def apply_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Flag factors by quality thresholds."""
    df = df.copy()
    df["flag_high_nan"] = df["nan_ratio"] > 0.20
    df["flag_low_ic"] = (
        (df["rank_icir"].abs() < 0.05) |
        ((df["rank_icir"].abs() < 0.10) &
         (df["rank_ic_pos_ratio"] > 0.48) & (df["rank_ic_pos_ratio"] < 0.52))
    )
    var_cutoff = df["xs_std_mean"].quantile(0.01)
    df["flag_low_variance"] = df["xs_std_mean"] <= var_cutoff
    df["flag_extreme"] = df["extreme_ratio"] > 0.05
    df["flag_redundant"] = False
    return df


def resolve_redundant(quality_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Flag the worse factor in each correlated pair."""
    if len(pairs_df) == 0:
        return quality_df
    q = quality_df.set_index("factor")
    redundant: set[str] = set()
    for _, row in pairs_df.iterrows():
        a, b = row["factor_a"], row["factor_b"]
        if a not in q.index or b not in q.index:
            continue
        icir_a = abs(q.loc[a, "rank_icir"]); icir_b = abs(q.loc[b, "rank_icir"])
        if pd.isna(icir_a):    worse = a
        elif pd.isna(icir_b):  worse = b
        elif abs(icir_a - icir_b) < 1e-6:
            nan_a = q.loc[a, "nan_ratio"]; nan_b = q.loc[b, "nan_ratio"]
            worse = a if nan_a > nan_b else b
        else:
            worse = a if icir_a < icir_b else b
        redundant.add(worse)
    if redundant:
        quality_df.loc[quality_df["factor"].isin(redundant), "flag_redundant"] = True
    return quality_df


def build_priority_ranking(quality_df: pd.DataFrame) -> pd.DataFrame:
    """Build the prioritized factor removal list (1 = worst, N = best)."""
    df = quality_df.copy()
    df["tier"] = 99
    df["reason"] = ""

    # T1: Zero variance
    df.loc[df["xs_std_mean"] <= 1e-10, ["tier", "reason"]] = (1, "zero_variance")
    # T2: 100% NaN
    mask2 = (df["nan_ratio"] >= 0.999) & (df["tier"] == 99)
    df.loc[mask2, ["tier", "reason"]] = (2, "all_nan")
    # T3: High NaN
    mask3 = df["flag_high_nan"] & (df["tier"] == 99)
    df.loc[mask3, ["tier", "reason"]] = (3, "high_nan")
    # T4: Low IC
    mask4 = df["flag_low_ic"] & (df["tier"] == 99)
    df.loc[mask4, ["tier", "reason"]] = (4, "low_ic")
    # T5: Redundant
    mask5 = df["flag_redundant"] & (df["tier"] == 99)
    df.loc[mask5, ["tier", "reason"]] = (5, "redundant")
    # T6: Low variance
    mask6 = df["flag_low_variance"] & (df["tier"] == 99)
    df.loc[mask6, ["tier", "reason"]] = (6, "low_variance")

    df["_icir_abs"] = df["rank_icir"].abs().fillna(0)
    df["_nan"] = df["nan_ratio"].fillna(0)
    df["_var"] = df["xs_std_mean"].fillna(0)

    sort_keys = []
    for _, row in df.iterrows():
        t = row["tier"]
        if t == 1:   key = (t, row["_var"])
        elif t == 2: key = (t, -row["_nan"])
        elif t == 3: key = (t, -row["_nan"])
        elif t == 4: key = (t, row["_icir_abs"])
        elif t == 5: key = (t, row["_icir_abs"])
        elif t == 6: key = (t, row["_var"])
        else:        key = (t, row["_icir_abs"])
        sort_keys.append(key)
    df["_sort"] = sort_keys
    df = df.sort_values("_sort").reset_index(drop=True)
    df["priority_rank"] = range(1, len(df) + 1)

    out = df[[
        "priority_rank", "factor", "tier", "reason",
        "nan_ratio", "rank_icir", "rank_ic_mean", "xs_std_mean",
        "flag_high_nan", "flag_low_ic", "flag_low_variance", "flag_redundant",
    ]].copy()
    return out


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Factor quality analysis & priority ranking")
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--skip-correlation", action="store_true")
    args = parser.parse_args()

    # ── Discover factors ──
    all_names = sorted(p.stem for p in FACTOR_DIR.glob("*.fea"))
    factor_names = all_names[:args.first_n] if args.first_n else all_names
    n = len(factor_names)
    if args.first_n:
        print(f"[SAMPLE MODE] {n} / {len(all_names)} factors")
    else:
        print(f"Analyzing all {n} factors")

    target_path = str(TARGET_DIR / f"{TARGET_NAME}.fea")
    workers = min(args.workers, n)

    # ═══ Part 1: Quality Metrics ═══
    print(f"\n{'='*60}")
    print(f"Part 1: Computing quality metrics ({n} factors, {workers} workers)")
    print(f"{'='*60}")

    work_items = [(str(FACTOR_DIR / f"{fn}.fea"), target_path, fn) for fn in factor_names]
    results: list[QualityResult] = []
    failed: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_compute_single_factor, item): item[2] for item in work_items}
        with tqdm(total=len(futures), desc="Quality metrics", unit="f", ncols=100) as p:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                    else:
                        failed.append(name)
                except Exception:
                    failed.append(name)
                p.update(1)

    if failed:
        print(f"  [WARN] {len(failed)} factors failed: {failed[:5]}...")

    # Build quality DataFrame
    records = [{
        "factor": r.factor_name, "n_dates": r.n_dates, "n_codes": r.n_codes,
        "nan_ratio": r.nan_ratio,
        "nan_ratio_per_date_mean": r.nan_ratio_per_date_mean,
        "nan_ratio_per_date_max": r.nan_ratio_per_date_max,
        "xs_var_mean": r.xs_var_mean, "xs_var_std": r.xs_var_std,
        "xs_std_mean": r.xs_std_mean,
        "ic_n_dates": r.ic_n_dates,
        "rank_ic_mean": r.rank_ic_mean, "rank_ic_std": r.rank_ic_std,
        "rank_icir": r.rank_icir, "rank_ic_pos_ratio": r.rank_ic_pos_ratio,
        "rank_ic_tstat": r.rank_ic_tstat,
        "global_mean": r.global_mean, "global_std": r.global_std,
        "skewness": r.skewness, "kurtosis": r.kurtosis,
        "extreme_ratio": r.extreme_ratio,
    } for r in results]
    quality_df = pd.DataFrame(records).sort_values("factor").reset_index(drop=True)

    # Apply thresholds
    quality_df = apply_thresholds(quality_df)

    # ═══ Part 2: Correlation ═══
    if not args.skip_correlation and n >= 2:
        print(f"\n{'='*60}")
        print(f"Part 2: Cross-sectional correlation analysis")
        print(f"{'='*60}")
        pairs_df = analyze_correlations(factor_names, workers)
        n_pairs = len(pairs_df)
        print(f"  Highly correlated pairs (|r| > {CORR_THRESHOLD}): {n_pairs}")

        # Save pairs
        pairs_path = OUTPUT_DIR / "factor_correlation_pairs.csv"
        pairs_df.to_csv(pairs_path, index=False)
        print(f"  Saved: {pairs_path}")

        # Resolve redundant flags
        quality_df = resolve_redundant(quality_df, pairs_df)
        n_red = int(quality_df["flag_redundant"].sum())
        print(f"  Flagged redundant: {n_red}")
    else:
        pairs_df = pd.DataFrame()
        n_pairs = 0

    # Save quality results
    quality_path = OUTPUT_DIR / "factor_quality_results.parquet"
    quality_df.to_parquet(quality_path, index=False)
    quality_df.to_csv(OUTPUT_DIR / "factor_quality_results.csv", index=False)
    print(f"\n  Quality results saved: {quality_path}")

    # ═══ Part 3: Priority Ranking ═══
    print(f"\n{'='*60}")
    print(f"Part 3: Building priority ranking")
    print(f"{'='*60}")

    ranking_df = build_priority_ranking(quality_df)
    ranking_path = OUTPUT_DIR / "factor_priority_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    print(f"  Saved: {ranking_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"PRIORITY RANKING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total factors:            {len(ranking_df)}")
    print(f"  Factors with issues:      {int((ranking_df['tier'] < 99).sum())}")
    for t in sorted(ranking_df["tier"].unique()):
        cnt = int((ranking_df["tier"] == t).sum())
        labels = {1: "Zero variance", 2: "100% NaN", 3: "High NaN (>20%)",
                  4: "Low |ICIR|", 5: "Redundant (|r|>0.95)", 6: "Low variance",
                  99: "Clean (no issues)"}
        print(f"    Tier {t:2d} ({labels.get(t, '?')}): {cnt}")

    print(f"\n  Worst 10 (highest removal priority):")
    for _, row in ranking_df.head(10).iterrows():
        print(f"    #{int(row['priority_rank']):3d}  {row['factor']:<45s}  "
              f"[T{int(row['tier'])}] {row['reason']}")

    print(f"\n  Best 10 (lowest removal priority):")
    for _, row in ranking_df.tail(10).iterrows():
        print(f"    #{int(row['priority_rank']):3d}  {row['factor']:<45s}  "
              f"|ICIR|={abs(row['rank_icir']):.4f}")

    # ── Write summary JSON ──
    summary = {
        "total_factors": len(ranking_df),
        "n_flagged_total": int((ranking_df["tier"] < 99).sum()),
        "n_t1_zero_variance": int((ranking_df["tier"] == 1).sum()),
        "n_t2_all_nan": int((ranking_df["tier"] == 2).sum()),
        "n_t3_high_nan": int((ranking_df["tier"] == 3).sum()),
        "n_t4_low_ic": int((ranking_df["tier"] == 4).sum()),
        "n_t5_redundant": int((ranking_df["tier"] == 5).sum()),
        "n_t6_low_variance": int((ranking_df["tier"] == 6).sum()),
        "n_corr_pairs": n_pairs,
        "thresholds": {
            "nan": 0.20, "icir": 0.05, "var_percentile": 1.0,
            "corr": CORR_THRESHOLD, "corr_n_dates": N_SAMPLE_DATES,
            "corr_n_stocks": N_SAMPLE_STOCKS,
        },
    }
    summary_path = OUTPUT_DIR / "factor_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Summary JSON: {summary_path}")
    print(f"\nDone. 供 prepared_data.py 读取的文件: {ranking_path}")


if __name__ == "__main__":
    main()
