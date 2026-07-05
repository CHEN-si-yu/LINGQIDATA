#!/usr/bin/env python3
"""
Best 4-Model Ensemble Search
=============================
从 Top-N 个候选模型中，穷举搜索所有 C(N,4) 个四人组，找出：
  - top_return 最高组
  - IC 最高组
  - 综合得分 (z_top_return + z_IC) 最高组

支持多 CPU 并行，支持断点续跑（自动跳过已有缓存）。
新模型加入后只需重新运行即可。

模型版本现已更新为 V0.1 ~ V4.8 系列（2026-06 当前）。
V0.1 ~ V3.8 已完成训练+预测（有 model_res/ALL_zscore_score.fea）；
V4.0 ~ V4.8 尚在训练中（暂未产出 model_res）。

最新结果 (2026-06-27, latest 20, 回测 20250102~20260609):
  ★ 综合最优: V2.1 + V3.2 + V3.7 + V1.2_copy  (top_return=1.2505, IC=0.0554)
  ★ IC 最高:  V2.0 + V2.4 + V2.7 + V2.8       (top_return=0.8006, IC=0.0571)

Usage:
    python search_best_ensemble.py [--top-n 20] [--workers 8] [--resume]
    python search_best_ensemble.py --top-n 20 --latest --workers 4    # 快速搜索
    python search_best_ensemble.py --top-n 30 --latest --workers 14   # 取最新 30 个模型
"""

import os, sys, itertools, warnings, json, argparse, hashlib, time, re
from functools import partial
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

import numpy as np, pandas as pd

# ============================================================
# Config
# ============================================================
MODEL_BASE = '/autodl-fs/data/lingqiData/Model'
OUTPUT_DIR = "/tmp/ensemble_search"
START, END = '20250101', '20260331'
MONEY = 1.5e9
CACHE_DIR = "/tmp/ensemble_search/cache"  # 每个 combo 的结果缓存

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _version_key(name):
    """Natural sort key for model version strings like 'V3.8', 'V0.1', 'V1.10'."""
    m = re.match(r'^V(\d+)\.(\d+)$', name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # Fallback: put non-matching names at the end
    return (999999, 0)


# ============================================================
# Global data (loaded once, shared read-only across workers)
# ============================================================
_ret_1d = None
_liquid = None
_model_cache = None
_common_dates = None
_common_codes = None
_exclude_copy = True


def load_all_data(exclude_copy=True):
    global _ret_1d, _liquid, _model_cache, _common_dates, _common_codes, _exclude_copy
    _exclude_copy = exclude_copy

    print("[1/4] Loading base data...")
    _ret_1d = pd.read_feather('/autodl-fs/data/lingqiData/trainingdata/label_ret_1d.fea').set_index("index")
    _liquid = pd.read_feather('/autodl-fs/data/lingqiData/trainingdata/trade_amt.fea').set_index("index")
    print(f"  ret_1d: {_ret_1d.shape}, liquid: {_liquid.shape}")

    # Discover models — 优先 model_res，其次 model_pred（兼容新旧目录结构）
    # 注意：默认排除 *_copy 目录（如 V1.2_copy），可通过 --include-copy 包含
    all_models_raw = [
        d for d in os.listdir(MODEL_BASE)
        if os.path.isdir(os.path.join(MODEL_BASE, d))
        and not (d.endswith('_copy') and _exclude_copy)
        and (
            os.path.exists(os.path.join(MODEL_BASE, d, 'model_res', 'ALL_zscore_score.fea'))
            or os.path.exists(os.path.join(MODEL_BASE, d, 'model_pred', 'ALL_zscore_score.fea'))
        )
    ]
    # 按版本号自然排序：V0.1 < V0.2 < ... < V3.8
    all_models = sorted(all_models_raw, key=_version_key)
    return all_models


def _find_score_path(model_name):
    """Find the score file for a model, trying model_res first, then model_pred."""
    for sub in ['model_res', 'model_pred']:
        p = os.path.join(MODEL_BASE, model_name, sub, 'ALL_zscore_score.fea')
        if os.path.exists(p):
            return p
    return None


def load_model_cache(model_list):
    global _model_cache, _common_dates, _common_codes

    print(f"[2/4] Loading {len(model_list)} model scores into cache...")
    _model_cache = {}
    for v in model_list:
        path = _find_score_path(v)
        if path is None:
            print(f"  [WARN] {v}: no model_res or model_pred found, skipping")
            continue
        df = pd.read_feather(path).set_index("date")
        _model_cache[v] = df
        print(f"  {v}: {len(df)}d x {len(df.columns)}s")

    # Filter out models that failed to load
    loaded = list(_model_cache.keys())
    if not loaded:
        raise RuntimeError("No model scores could be loaded!")

    # Common intersection
    _common_dates = _model_cache[loaded[0]].index
    _common_codes = set(_model_cache[loaded[0]].columns)
    for v in loaded[1:]:
        _common_dates = _common_dates.intersection(_model_cache[v].index)
        _common_codes = _common_codes.intersection(_model_cache[v].columns)
    _common_codes = sorted(_common_codes)
    _common_dates = sorted(_common_dates)
    print(f"  Common: {len(_common_dates)}d x {len(_common_codes)}s, "
          f"range={_common_dates[0]}~{_common_dates[-1]}")


def combo_key(combo):
    """Deterministic short key for a 4-model combo."""
    return '+'.join(combo)


def compute_ret_ic(score_df):
    """Compute liquidity-weighted top_return and IC from a score DataFrame."""
    label_ret, ic_vals = [], []
    for date in score_df.index:
        if date < START or date > END:
            continue
        try:
            code_rank = score_df.loc[date].sort_values(ascending=False)
            ret = _ret_1d.loc[date].reindex(code_rank.index).fillna(0) * 100
            liq = _liquid.loc[date].reindex(code_rank.index).fillna(0)
        except KeyError:
            continue
        total_hold = total_earned = 0.0
        for num, code in enumerate(code_rank.index):
            if num >= 500 or (MONEY - total_hold) < 1:
                break
            hold_money = min(MONEY - total_hold, liq[code])
            total_hold += hold_money
            total_earned += ret[code] * hold_money
        label_ret.append(total_earned / MONEY)
        ic_vals.append(code_rank.corr(ret))
    if not label_ret:
        return 0.0, 0.0
    ret_s = pd.Series(label_ret)
    ic_s = pd.Series(ic_vals)
    return float(ret_s.mean()), float(ic_s.mean())


def ensemble_and_evaluate(combo):
    """Ensemble 4 models and evaluate. Called by worker processes."""
    models = list(combo)
    # Check cache first
    key = combo_key(models)
    cache_path = os.path.join(CACHE_DIR, f'{key}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    # Ensemble
    parts = []
    for date in _common_dates:
        z_scores = []
        for m in models:
            s = _model_cache[m].loc[date].reindex(_common_codes)
            z = (s - s.mean()) / s.std()
            z_scores.append(z)
        avg = sum(z_scores) / len(z_scores)
        avg.name = date
        parts.append(avg)
    ensemble_df = pd.DataFrame(parts)

    top_ret, ic = compute_ret_ic(ensemble_df)
    result = {
        'models': models,
        'top_return': round(top_ret, 8),
        'IC': round(ic, 8),
    }

    # Write cache
    with open(cache_path, 'w') as f:
        json.dump(result, f)

    return result


def main():
    parser = argparse.ArgumentParser(description='Search best 4-model ensemble')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Use top N models (default: 20)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count-1)')
    parser.add_argument('--no-cache', action='store_true', default=False,
                        help='Ignore cache and recompute all')
    parser.add_argument('--latest', action='store_true', default=False,
                        help='Use the LATEST N models (by version) instead of the first N')
    parser.add_argument('--exclude-copy', action='store_true', default=True,
                        help='Exclude *_copy model directories (default: True)')
    parser.add_argument('--include-copy', action='store_true', default=False,
                        help='Include *_copy model directories (overrides --exclude-copy)')
    args = parser.parse_args()

    workers = args.workers or max(1, cpu_count() - 2)
    print(f"Workers: {workers}, CPU count: {cpu_count()}")

    # Load
    exclude_copy = args.exclude_copy and not args.include_copy
    all_models = load_all_data(exclude_copy=exclude_copy)
    print(f"[info] Discovered {len(all_models)} models with scores: "
          f"{all_models[0]} ~ {all_models[-1]}")

    if args.latest:
        candidate_models = all_models[-args.top_n:] if args.top_n < len(all_models) else all_models
        print(f"[info] Using LATEST {len(candidate_models)} models: {candidate_models[0]} ~ {candidate_models[-1]}")
    else:
        top_n = min(args.top_n, len(all_models))
        candidate_models = all_models[:top_n]
        print(f"[info] Using first {top_n} models: {candidate_models[0]} ~ {candidate_models[-1]}")

    load_model_cache(candidate_models)

    # Generate all combos
    combos = list(itertools.combinations(candidate_models, 4))
    total = len(combos)
    print(f"[3/4] Total combinations: C({len(candidate_models)},4) = {total}")

    # Clear cache if requested
    if args.no_cache:
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        print("[info] Cache cleared.")

    # Check existing cache
    cached_count = 0
    for combo in combos:
        key = combo_key(list(combo))
        if os.path.exists(os.path.join(CACHE_DIR, f'{key}.json')):
            cached_count += 1
    print(f"[info] Already cached: {cached_count}/{total}, remaining: {total - cached_count}")

    # Run
    print(f"[4/4] Evaluating {total - cached_count} combos with {workers} workers...")
    t0 = time.time()

    with Pool(processes=workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(ensemble_and_evaluate, combos)):
            results.append(result)
            if (i + 1) % 200 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1}/{total} ({100*(i+1)/total:.1f}%) | "
                      f"{rate:.1f} combo/s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/total:.2f}s per combo)")

    # Analyze
    print("\n" + "=" * 60)
    print("  RANKING")
    print("=" * 60)

    df_r = pd.DataFrame(results)
    df_r['top_return_z'] = (df_r['top_return'] - df_r['top_return'].mean()) / df_r['top_return'].std()
    df_r['IC_z'] = (df_r['IC'] - df_r['IC'].mean()) / df_r['IC'].std()
    df_r['composite_z'] = df_r['top_return_z'] + df_r['IC_z']

    best_topret = df_r.loc[df_r['top_return'].idxmax()]
    best_ic = df_r.loc[df_r['IC'].idxmax()]
    best_comp = df_r.loc[df_r['composite_z'].idxmax()]

    print(f"\n★ Best Top_Return: {list(best_topret['models'])}")
    print(f"  top_return={best_topret['top_return']:.6f}, IC={best_topret['IC']:.6f}")

    print(f"\n★ Best IC: {list(best_ic['models'])}")
    print(f"  top_return={best_ic['top_return']:.6f}, IC={best_ic['IC']:.6f}")

    print(f"\n★ Best Composite (z_sum): {list(best_comp['models'])}")
    print(f"  top_return={best_comp['top_return']:.6f}, IC={best_comp['IC']:.6f}, "
          f"composite_z={best_comp['composite_z']:.4f}")

    # Top 10 tables
    for metric, label in [('top_return', 'Top Return'), ('IC', 'IC'), ('composite_z', 'Composite')]:
        top10 = df_r.nlargest(10, metric)
        print(f"\n--- Top 10 by {label} ---")
        print(f"  {'Rank':<5} {'Models':<40} {'Top_Return':>12} {'IC':>10}")
        print(f"  {'-'*70}")
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"  {rank:<5} {'+'.join(row['models']):<40} {row['top_return']:>12.6f} {row['IC']:>10.6f}")

    # Save full results
    full_path = os.path.join(OUTPUT_DIR, 'full_results.csv')
    df_r.to_csv(full_path, index=False)
    print(f"\nFull results saved to: {full_path}")

    # Save summary JSON
    summary = {
        'candidate_models': candidate_models,
        'total_combinations': total,
        'common': f'{_common_dates[0]}~{_common_dates[-1]} ({len(_common_dates)}d x {len(_common_codes)}s)',
        'best_top_return': {
            'models': list(best_topret['models']),
            'top_return': float(best_topret['top_return']),
            'IC': float(best_topret['IC']),
        },
        'best_IC': {
            'models': list(best_ic['models']),
            'top_return': float(best_ic['top_return']),
            'IC': float(best_ic['IC']),
        },
        'best_composite': {
            'models': list(best_comp['models']),
            'top_return': float(best_comp['top_return']),
            'IC': float(best_comp['IC']),
            'composite_z': float(best_comp['composite_z']),
        },
        'top10_top_return': [
            {'models': list(r['models']), 'top_return': float(r['top_return']), 'IC': float(r['IC'])}
            for _, r in df_r.nlargest(10, 'top_return').iterrows()
        ],
        'top10_IC': [
            {'models': list(r['models']), 'top_return': float(r['top_return']), 'IC': float(r['IC'])}
            for _, r in df_r.nlargest(10, 'IC').iterrows()
        ],
        'top10_composite': [
            {'models': list(r['models']), 'top_return': float(r['top_return']), 'IC': float(r['IC']),
             'composite_z': float(r['composite_z'])}
            for _, r in df_r.nlargest(10, 'composite_z').iterrows()
        ],
    }
    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
