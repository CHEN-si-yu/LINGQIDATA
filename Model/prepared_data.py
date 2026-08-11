import argparse
import gc
import json
import os
import re
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# OpenBLAS 首次 GEMM 前设置线程数（相关性矩阵/IC 计算全核利用）
os.environ.setdefault("OMP_NUM_THREADS", str(min(64, os.cpu_count() or 1)))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Input paths
# ============================================================
code_num_path = PROJECT_ROOT / "Code_num.txt"
factor_dir = PROJECT_ROOT / "featureengineering/data/factors"

# Label source paths
label_ret_1d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_1d.fea"
label_ret_3d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_3d.fea"
label_ret_5d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_5d.fea"
label_ret_10d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_10d.fea"
daily_adj_path = PROJECT_ROOT / "data/daily_adj.parquet"
calendar_path = PROJECT_ROOT / "data/calendar.parquet"

# ============================================================
# Output paths
# ============================================================
out_dir = PROJECT_ROOT / "trainingdata"
fac_all_path = out_dir / "fac_all.fea"
fac_sample_path = out_dir / "fac_sample.fea"
fac_select_path = out_dir / "fac_select.fea"
fac_select_meta_path = out_dir / "fac_select_meta.json"
trade_path = out_dir / "trade_amt.fea"

LABEL_TARGETS = {
    "label_ret_1d":  label_ret_1d_path,
    "label_ret_3d":  label_ret_3d_path,
    "label_ret_5d":  label_ret_5d_path,
    "label_ret_10d": label_ret_10d_path,
}

# ============================================================
# Config
# ============================================================
N_WORKERS = 48
BATCH_SIZE = 320
TMP_DIR = out_dir / "tmp_batches"
SAMPLE_FACTOR_COUNT = 20
RANDOM_SEED = 42
LOOKBACK_DEFAULT = 3            # trading days to look back for factor restatements

# ── fac_select 冗余剔除配置（2026-08-11 分析定型，决策记录见文件尾部 §8.20）──
# 去冗余阈值目标 θ=0.95：与 2026-08-10 生产口径一致（745→596 @0.95）。
# 2026-08-11 全量分析（768 因子）显示 θ=0.95 可移除全部 119 对 |r|≥0.999 的
# 完全重复（变换复刻/命名变体），保留 616 个（80%）；0.92/0.90 仅多剔除
# 34/55 个中低度冗余因子，信号覆盖损失更大，故不采用。
# min_keep/max_keep 限定 [500, 620]：自适应阈值 0.98→644 超出 620 上限、
# 0.95→616 落入区间，自动选中 θ=0.95。
SELECT_MIN_KEEP = 500
SELECT_MAX_KEEP = 620


# ============================================================
# CLI argument parsing
# ============================================================
parser = argparse.ArgumentParser(
    description="Prepare training data — daily incremental or full generation."
)
parser.add_argument(
    "--incremental", "-i",
    action="store_true",
    default=None,
    help="Force incremental mode. Default: auto-detect from existing files.",
)
parser.add_argument(
    "--full", "-f",
    action="store_true",
    help="Force full regeneration: delete all existing output files and rebuild.",
)
parser.add_argument(
    "--lookback", "-l",
    type=int,
    default=None,
    help=f"Trading-day lookback buffer for factor restatements (default: {LOOKBACK_DEFAULT}, "
         "or inferred from factor names if larger).",
)
parser.add_argument(
    "--reselect",
    action="store_true",
    help="Force re-selection of fac_select factors (ignore selection cache).",
)
args = parser.parse_args()

# Resolve mode
if args.full:
    FORCE_INCREMENTAL = False
    FORCE_FULL = True
    print("--full: forcing full regeneration (will delete existing output files)")
elif args.incremental is not None:
    FORCE_INCREMENTAL = args.incremental
    FORCE_FULL = False
else:
    FORCE_INCREMENTAL = None   # auto-detect
    FORCE_FULL = False

USER_LOOKBACK = args.lookback
print(f"Config: workers={N_WORKERS}, batch={BATCH_SIZE}, default_lookback={LOOKBACK_DEFAULT}")


# ============================================================
# Calendar & trading-day helpers
# ============================================================
def _load_trading_calendar():
    """Return sorted list of trading days in YYYYMMDD (no-dash) format."""
    cal = pd.read_parquet(calendar_path)
    trading_days = sorted(
        cal.loc[cal["is_open"] == 1, "date"].str.replace("-", "", regex=False).tolist()
    )
    return trading_days


def _trading_day_offset(trading_days, base_date, offset):
    """
    Return the trading day `offset` steps from base_date.
    Negative offset = go back, positive = go forward.
    base_date and returned value are YYYYMMDD strings.
    """
    base_str = str(base_date)
    if base_str not in trading_days:
        # Find nearest trading day <= base_str
        candidates = [d for d in trading_days if d <= base_str]
        if not candidates:
            return trading_days[0]
        base_str = candidates[-1]
    idx = trading_days.index(base_str)
    target_idx = idx + offset
    target_idx = max(0, min(target_idx, len(trading_days) - 1))
    return trading_days[target_idx]


def _get_effective_lookback():
    """
    Effective lookback for factor restatement buffer.
    Uses --lookback CLI flag if provided, otherwise the LOOKBACK_DEFAULT.
    """
    if USER_LOOKBACK is not None:
        print(f"  Using user-specified lookback: {USER_LOOKBACK} trading days")
        return USER_LOOKBACK
    print(f"  Using default lookback buffer: {LOOKBACK_DEFAULT} trading days")
    return LOOKBACK_DEFAULT


def _label_horizon_days(label_name):
    """Extract the forward-return horizon from a label name like 'label_ret_5d' → 5."""
    m = re.search(r"label_ret_(\d+)d", label_name)
    return int(m.group(1)) if m else 1


# ============================================================
# Helper: load factor file, filter by min_date (inclusive)
# ============================================================
def _load_factor_stacked(path, min_date=None):
    """
    Read a factor .fea file, optionally filter to dates >= min_date, stack to Series.
    """
    df = pd.read_feather(path)
    if min_date is not None:
        df = df[df.index.astype(str) >= str(min_date)]
        if df.empty:
            return None
    return df.stack()


# ============================================================
# 因子选择模块（原 Model/factor_select.py 内联合并，2026-08-11）
# 功能：因子间相关性去冗余（主干）+ target IC 择优（辅助）的综合筛选，
# 生成 fac_select.fea 与 fac_select_meta.json。
# 综合筛选逻辑：
#   1. 主干 — 因子间相关性去冗余：
#      - pairwise-NaN Pearson 相关矩阵（掩码矩阵乘 3 次 GEMM，O(F²·N)）
#      - 重叠样本 < min_overlap 的对视为不可靠（corr=NaN，不参与剔除）
#      - 贪婪剔除：与任一已保留因子 |corr| > θ 即剔除
#   2. 辅助 — target IC 择优：
#      - 对 4 个 horizon（1/3/5/10d）计算日频截面 Spearman rank-IC（2022+ 全日期，向量化）
#      - 优先级排序键 = (-mean_abs_ic, nan_ratio, name)：高相关组内保留预测能力
#        更强、数据覆盖更完整的因子（IC 仅决定组内取舍，不设 IC 硬阈值）
#   3. 阈值自适应：候选阈值从高到低，取保留数 ∈ [min_keep, max_keep] 的最高阈值
# 已知近似（与 2026-08-10 生产口径及 2026-08-11 分析脚本完全一致）：
#   方差项用对角元素 diag(xx)（因子自身全掩码平方和）除以两两重叠样本数——
#   当两因子 NaN 掩码不一致时 |corr| 会被低估。精确重复因子（掩码一致）不受
#   影响；掩码差异越大低估越强。此为既定口径，改动会破坏与历史分析结果的可比性。
# 注意：--full 模式下 trainingdata 的 label 文件在 selection 之前被删除，
# 故 IC 择优的 label 源固定指向 featureengineering/data/targets（上游源，
# 与第 2 节 label 生成的输入一致，避免 full 重跑时退化覆盖率排序）。
# ============================================================
DEFAULT_THRESHOLDS = [0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]
SELECT_MIN_DATE = "20220101"      # 样本起点（与 skill.md NaN 口径一致）
SELECT_DATE_STEP = 3              # 相关矩阵日期抽样步长（IC 用全日期）
SELECT_MIN_OVERLAP = 500          # 最小重叠行数，低于此视为相关估计不可靠
SELECT_LABEL_HORIZONS = ["1d", "3d", "5d", "10d"]
SELECT_META_VERSION = 2
SELECT_MIN_IC_SAMPLE = 60         # IC 有效截面天数下限（低于此 IC 视为不可靠 → 退化为覆盖率排序）


def _factor_cols(df_columns: list[str]) -> list[str]:
    """因子列 = 非 (date, Code) 列（保持 fac_all 顺序）。"""
    return [c for c in df_columns if c not in ("date", "Code")]


# ────────────────────────────────────────────────────────────────
# 1. pairwise-NaN Pearson 相关矩阵（掩码矩阵乘）
# ────────────────────────────────────────────────────────────────
def _corr_matrix(x: np.ndarray, min_overlap: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows = x.shape[0]
    m = np.isfinite(x).astype(np.float64)
    x0 = np.where(m, x, 0.0)

    mm = m.T @ m
    xx = x0.T @ x0
    xm = x0.T @ m

    overlap = mm.copy()
    np.fill_diagonal(overlap, np.nan)
    ok = overlap >= min_overlap
    denom = np.where(ok, overlap, np.nan)

    sa = xm / denom
    sb = sa.T
    cov = (xx - sa * sb * overlap) / (overlap - 1)
    va = (np.diag(xx)[:, None] - sa ** 2 * overlap) / (overlap - 1)
    vb = va.T
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(va * vb)
    corr = np.where(ok, corr, np.nan)
    np.fill_diagonal(corr, np.nan)

    nan_ratio = 1.0 - np.diag(mm) / n_rows
    return corr, nan_ratio


# ────────────────────────────────────────────────────────────────
# 2. target IC（日频截面 Spearman rank-IC，向量化 + 线程并行）
# ────────────────────────────────────────────────────────────────
def _load_targets(target_dir: Path) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """读 label_ret_{h}.fea（宽表，'index' 列=日期）→ {h: (dates, T宽表)}。

    兼容两种存储格式：trainingdata 副本（'index' 列）与
    featureengineering/data/targets 源（无列名 Index，reset_index 后即 'index' 列）。
    """
    out: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for h in SELECT_LABEL_HORIZONS:
        p = Path(target_dir) / f"label_ret_{h}.fea"
        if not p.exists():
            print(f"[fac_select] target label_ret_{h}.fea not found — skipping horizon {h}")
            continue
        df = pd.read_feather(p)
        if "index" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "Date": "index"})
        df["index"] = df["index"].astype(str)
        out[h] = (df["index"].to_numpy(), df)
    return out


def _rank_ic(f: np.ndarray, t_rank: np.ndarray) -> np.ndarray:
    """逐日截面 Spearman IC（向量化）：对 f 逐行 rank，与预计算的 t_rank
    做 masked Pearson，返回逐日 IC 序列。t_rank 由调用方对每 horizon 预计算一次。"""
    f_rank = pd.DataFrame(f).rank(axis=1).to_numpy()
    m = np.isfinite(f_rank) & np.isfinite(t_rank)
    f0 = np.where(m, f_rank, 0.0)
    t0 = np.where(m, t_rank, 0.0)
    n = m.sum(axis=1)
    s_f = (f0 * m).sum(axis=1)
    s_t = (t0 * m).sum(axis=1)
    s_ft = (f0 * t0).sum(axis=1)
    s_f2 = (f0 * f0).sum(axis=1)
    s_t2 = (t0 * t0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (s_ft - s_f * s_t / n) / (n - 1)
        v_f = (s_f2 - s_f ** 2 / n) / (n - 1)
        v_t = (s_t2 - s_t ** 2 / n) / (n - 1)
        ic = cov / np.sqrt(v_f * v_t)
    return ic[np.isfinite(ic)]


def _factor_ics_aligned(x_wide: np.ndarray, aligned: dict[str, np.ndarray],
                        factor_names: list[str], threads: int) -> pd.DataFrame:
    """x_wide: (F, D, C) 与 aligned[h] (D, C) 直接对齐的版本；返回 index=因子名。"""
    rows = []
    f_dim = x_wide.shape[0]
    # 每 horizon 只对 target rank 一次（复用给全部因子）
    t_ranks = {h: pd.DataFrame(t).rank(axis=1).to_numpy() for h, t in aligned.items()}

    def work(i: int) -> dict:
        f = x_wide[i]
        rec = {"_i": i}
        n_dates_min = None
        for h, t_rank in t_ranks.items():
            ic = _rank_ic(f, t_rank)
            if ic.size < SELECT_MIN_IC_SAMPLE:
                rec[f"ic_{h}"] = float("nan")
                rec[f"icir_{h}"] = float("nan")
                rec[f"ic_abs_{h}"] = float("nan")
                continue
            ic_mean = float(np.mean(ic))
            ic_std = float(np.std(ic, ddof=1))
            rec[f"ic_{h}"] = ic_mean
            rec[f"icir_{h}"] = ic_mean / ic_std if ic_std > 0 else float("nan")
            rec[f"ic_abs_{h}"] = float(np.mean(np.abs(ic)))
            n_dates_min = ic.size if n_dates_min is None else min(n_dates_min, ic.size)
        rec["_n_dates"] = n_dates_min
        return rec

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(work, i) for i in range(f_dim)]
        for fut in as_completed(futures):
            results.append(fut.result())

    df = pd.DataFrame(results).set_index("_i").sort_index()
    df.index = pd.Index([factor_names[i] for i in df.index], name="factor")
    for h in aligned:
        df[f"ic_{h}"] = df[f"ic_{h}"].astype(float)
    ic_cols = [f"ic_{h}" for h in aligned]
    icir_cols = [f"icir_{h}" for h in aligned]
    ic_abs_cols = [f"ic_abs_{h}" for h in aligned]
    df["mean_abs_ic"] = df[ic_cols].abs().mean(axis=1)          # 方向一致性
    df["mean_abs_icir"] = df[icir_cols].abs().mean(axis=1)
    df["mean_ic_abs"] = df[ic_abs_cols].mean(axis=1)            # 单日预测强度
    df["quality"] = 0.5 * df["mean_abs_ic"] + 0.5 * df["mean_ic_abs"]  # 综合质量分
    df["n_ic_dates"] = df["_n_dates"]
    return df.drop(columns=["_n_dates"])


# ────────────────────────────────────────────────────────────────
# 3. 贪婪选择 + 阈值自适应
# ────────────────────────────────────────────────────────────────
def _greedy(corr: np.ndarray, order: list[int], threshold: float) -> list[int]:
    kept: list[int] = []
    for i in order:
        if kept and np.nanmax(np.abs(corr[i, kept])) > threshold:
            continue
        kept.append(i)
    return kept


def _pick_threshold(corr: np.ndarray, order: list[int],
                    min_keep: int, max_keep: int,
                    thresholds: list[float]) -> tuple[float, int, bool]:
    counts = [(th, len(_greedy(corr, order, th))) for th in thresholds]
    for th, k in counts:
        if min_keep <= k <= max_keep:
            return th, k, True
    th, k = min(counts, key=lambda tk: abs(tk[1] - max_keep))
    return th, k, False


def _load_meta(meta_path: Path) -> dict | None:
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def select_factors(fac_all_path: Path | str, fac_select_path: Path | str,
                   meta_path: Path | str, *,
                   force: bool = False,
                   min_keep: int = SELECT_MIN_KEEP, max_keep: int = SELECT_MAX_KEEP,
                   min_overlap: int = SELECT_MIN_OVERLAP,
                   thresholds: list[float] | None = None,
                   metric: str = "ic",
                   target_dir: Path | str | None = None,
                   ic_threads: int = 32) -> dict:
    """综合筛选：相关性去冗余（主干）+ target IC 择优（辅助）。

    metric="ic"：高相关组内保留 mean_abs_ic 更高者（IC 不可靠时退化为覆盖率）；
    metric="coverage"：纯覆盖率排序（旧行为）。
    返回结果摘要 dict（含状态 "cached" / "computed"）。
    """
    fac_all_path = Path(fac_all_path)
    fac_select_path = Path(fac_select_path)
    meta_path = Path(meta_path)
    thresholds = thresholds or DEFAULT_THRESHOLDS
    t_start = datetime.now()

    # 只读列名（feather schema 级，秒级）
    schema_df = pd.read_feather(fac_all_path, columns=[])
    all_cols = _factor_cols(list(schema_df.columns))
    del schema_df

    # ── 缓存命中：列集合未变 → 直接取列子集 ──
    if not force:
        meta = _load_meta(meta_path)
        if meta and meta.get("factor_cols") == all_cols and meta.get("metric") == metric:
            kept = [c for c in meta.get("kept", []) if c in all_cols]
            if len(kept) == len(meta.get("kept", [])):
                df = pd.read_feather(fac_all_path, columns=["date", "Code"] + kept)
                df.to_feather(fac_select_path)
                elapsed = (datetime.now() - t_start).total_seconds()
                print(f"[fac_select] cache hit: {len(kept)} factors, "
                      f"rebuilt from fac_all in {elapsed:.1f}s")
                return {"status": "cached", "n_kept": len(kept),
                        "threshold": meta.get("threshold"),
                        "kept": kept, "dropped": meta.get("dropped", []),
                        "n_factors_total": len(all_cols),
                        "elapsed": elapsed}

    # ── 全量计算 ──
    print(f"[fac_select] computing correlation matrix "
          f"({len(all_cols)} factors, {os.environ.get('OMP_NUM_THREADS', '?')} OMP threads)...")
    df = pd.read_feather(fac_all_path)
    df = df[df["date"].astype(str) >= SELECT_MIN_DATE]
    dates_all = df["date"].astype(str).to_numpy()
    codes = sorted(df["Code"].astype(str).unique())
    dates_u = np.unique(dates_all)
    print(f"[fac_select] dates: {len(dates_u)} (2022+)")

    # ---- 3a. 相关矩阵（日期抽样） ----
    sub_dates = dates_u[::SELECT_DATE_STEP]
    sub_mask = np.isin(dates_all, sub_dates)
    x_sub = df[all_cols][sub_mask].to_numpy(dtype=np.float64)
    print(f"[fac_select] corr sample: {len(sub_dates)} dates, {x_sub.shape[0]} rows (step={SELECT_DATE_STEP})")
    corr, nan_ratio = _corr_matrix(x_sub, min_overlap)
    del x_sub

    # ---- 3b. target IC（全日期） ----
    ic_stats: pd.DataFrame | None = None
    if metric == "ic":
        target_dir = Path(target_dir) if target_dir else fac_all_path.parent
        targets = _load_targets(target_dir)
        if targets:
            # 因子面板 (F, D, C)：按日期×代码网格重塑（fac_all 为全网格长表）
            d_idx = {d: i for i, d in enumerate(dates_u)}
            d_pos = np.array([d_idx[d] for d in dates_all], dtype=np.int64)
            c_idx = {c: i for i, c in enumerate(codes)}
            c_pos = np.array([c_idx[c] for c in df["Code"].astype(str).to_numpy()], dtype=np.int64)
            x_wide = np.empty((len(all_cols), len(dates_u), len(codes)), dtype=np.float32)
            for j, col in enumerate(all_cols):
                x_wide[j, d_pos, c_pos] = df[col].to_numpy(dtype=np.float32)
            del df

            aligned: dict[str, np.ndarray] = {}
            for h, (t_dates, t_df) in targets.items():
                t = t_df.set_index("index")
                common_d = [d for d in dates_u if d in t.index]
                common_c = [c for c in codes if c in t.columns]
                t_mat = t.loc[common_d, common_c].to_numpy(dtype=np.float64)
                # 映射到统一网格：缺的日期/代码列填 NaN
                full = np.full((len(dates_u), len(codes)), np.nan)
                full[np.ix_(np.isin(dates_u, common_d), np.isin(codes, common_c))] = t_mat
                aligned[h] = full
            print(f"[fac_select] computing rank-IC ({len(all_cols)} factors x "
                  f"{len(aligned)} horizons, {ic_threads} threads)...")
            ic_stats = _factor_ics_aligned(x_wide, aligned, all_cols, ic_threads)
            del x_wide
        else:
            print("[fac_select] no targets found — falling back to coverage metric")

    # ---- 3c. 优先级与贪婪选择 ----
    if ic_stats is not None and "quality" in ic_stats:
        q_map = ic_stats["quality"].to_dict()
        order = sorted(range(len(all_cols)), key=lambda i:
                       (-(q_map.get(all_cols[i]) or 0.0), nan_ratio[i], all_cols[i]))
    else:
        ic_map, icir_map = {}, {}
        order = sorted(range(len(all_cols)), key=lambda i: (nan_ratio[i], all_cols[i]))

    threshold, n_kept, ok_flag = _pick_threshold(corr, order, min_keep, max_keep, thresholds)
    kept_idx = _greedy(corr, order, threshold)
    dropped_idx = [i for i in range(len(all_cols)) if i not in kept_idx]
    if not ok_flag:
        print(f"[fac_select] WARNING: no threshold keeps {min_keep}~{max_keep} factors; "
              f"using threshold {threshold} → {n_kept} kept")
    kept = sorted(all_cols[i] for i in kept_idx)
    dropped = sorted(all_cols[i] for i in dropped_idx)

    # 写 fac_select.fea（复用已读 df）
    df_sel = pd.read_feather(fac_all_path, columns=["date", "Code"] + kept)
    df_sel.to_feather(fac_select_path)
    del df_sel

    # IC 明细（meta 记录）
    ic_detail = {}
    if ic_stats is not None:
        for col in all_cols:
            row = ic_stats.loc[col] if col in ic_stats.index else None
            ic_detail[col] = {
                "quality": float(row["quality"]) if row is not None else None,
                "mean_abs_ic": float(row["mean_abs_ic"]) if row is not None else None,
                "mean_ic_abs": float(row["mean_ic_abs"]) if row is not None else None,
                "mean_abs_icir": float(row["mean_abs_icir"]) if row is not None else None,
            }

    meta = {
        "version": SELECT_META_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metric": metric,
        "n_factors_total": len(all_cols),
        "threshold": threshold,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "kept": kept,
        "dropped": dropped,
        "factor_cols": all_cols,
        "ic": ic_detail,
        "sample": {"min_date": SELECT_MIN_DATE, "n_dates": int(len(dates_u)),
                   "n_corr_dates": int(len(sub_dates)),
                   "n_corr_rows": int(sub_mask.sum())},
        "min_overlap": min_overlap,
        "target": {"min_keep": min_keep, "max_keep": max_keep},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"[fac_select] metric={metric} threshold={threshold}  "
          f"kept={len(kept)}/{len(all_cols)}  dropped={len(dropped)}  ({elapsed:.1f}s)")
    print(f"[fac_select] saved {fac_select_path.name} ({len(kept)} factor columns) "
          f"and {meta_path.name}")
    return {"status": "computed", "n_kept": len(kept), "n_dropped": len(dropped),
            "threshold": threshold, "kept": kept, "dropped": dropped,
            "n_factors_total": len(all_cols), "elapsed": elapsed}


# ============================================================
# Setup
# ============================================================
out_dir.mkdir(parents=True, exist_ok=True)

with open(code_num_path) as f:
    allowed_codes = [line.strip() for line in f if line.strip()]
allowed_codes_set = set(allowed_codes)
print(f"Loaded {len(allowed_codes)} codes from {code_num_path}")

# ============================================================
# 1. Merge factors → fac_all.fea
# ============================================================
files = sorted(factor_dir.glob("*.fea"))
if not files:
    raise FileNotFoundError(f"No .fea files found in {factor_dir}")

# Load calendar (needed for smart window calculation)
trading_calendar = _load_trading_calendar()
print(f"Loaded {len(trading_calendar)} trading days from calendar")

# Determine upstream max date (latest date across ALL factor source files, cheap index-only scan)
factor_max_dates = []
for f in files:
    _df = pd.read_feather(f, columns=[])   # index-only
    _idx = sorted(_df.index.unique())
    factor_max_dates.append(str(_idx[-1]))
    del _df
factor_upstream_max = max(factor_max_dates)
print(f"Factor upstream max date: {factor_upstream_max}  (from {len(files)} files)")

# --- Delete / incremental detection for factors ---
fac_min_date = None
if FORCE_FULL:
    for _p in [fac_all_path, fac_sample_path, fac_select_path,
               fac_select_meta_path, trade_path] + [
        out_dir / f"{ln}.fea" for ln in LABEL_TARGETS
    ]:
        if _p.exists():
            _p.unlink()
            print(f"  Deleted {_p.name}")
    incremental_fac = False
    fac_min_date = None
elif FORCE_INCREMENTAL is True:
    incremental_fac = fac_all_path.exists()
    if not incremental_fac:
        print("--incremental specified but fac_all.fea not found; falling back to full generation.")
    fac_min_date = None
else:
    incremental_fac = fac_all_path.exists()

# Compute smart update window for factors
reprocess_start_date = None
if incremental_fac:
    _tmp = pd.read_feather(fac_all_path, columns=["date"])
    fac_existing_max = str(_tmp["date"].max())
    del _tmp; gc.collect()
    print(f"\n[Factors] existing max date: {fac_existing_max}")
    print(f"[Factors] upstream max date:  {factor_upstream_max}")

    # Compute lookback buffer
    lookback = _get_effective_lookback()
    reprocess_start_date = _trading_day_offset(trading_calendar, fac_existing_max, -lookback)
    print(f"[Factors] reprocess window: {reprocess_start_date} ~ {factor_upstream_max}"
          f"  (existing max {fac_existing_max} − {lookback} trading days)")

    # Read factor data from reprocess start onwards
    fac_min_date = reprocess_start_date

print(f"Found {len(files)} factor files, loading with {N_WORKERS} workers...")
print(f"Factor min_date filter: {fac_min_date or '(none — full load)'}")

# Clean up leftover temp files from previous failed runs
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)
TMP_DIR.mkdir(parents=True, exist_ok=True)

batches = [files[i:i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
batch_paths = []
all_factor_cols = set()

for batch_idx, batch_files in enumerate(batches):
    series_list = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_load_factor_stacked, f, fac_min_date): f.stem
                   for f in batch_files}
        for fut in tqdm(as_completed(futures), total=len(batch_files),
                        desc=f"Batch {batch_idx+1}/{len(batches)}"):
            s = fut.result()
            if s is None:
                continue
            s.name = futures[fut]
            s = s.astype(np.float32)
            series_list.append(s)

    if not series_list:
        del series_list
        gc.collect()
        continue

    batch_df = pd.concat(series_list, axis=1).copy()
    del series_list
    gc.collect()

    batch_df.reset_index(inplace=True)
    batch_df.rename(columns={"Date": "date"}, inplace=True)
    batch_df["Code"] = batch_df["Code"].astype(str).str.replace(".BJ", "", regex=False)
    batch_df["date"] = batch_df["date"].astype(str)
    batch_df = batch_df[batch_df["Code"].isin(allowed_codes_set)]

    if batch_df.empty:
        del batch_df
        gc.collect()
        continue

    all_factor_cols.update(c for c in batch_df.columns if c not in ("date", "Code"))

    batch_path = TMP_DIR / f"batch_{batch_idx:03d}.fea"
    batch_df.to_feather(batch_path)
    batch_paths.append(batch_path)
    del batch_df
    gc.collect()

# --- Merge ---
if not batch_paths:
    print("No new factor data to merge. Keeping existing file.")
    if incremental_fac:
        merged = pd.read_feather(fac_all_path)
        factor_cols = sorted(c for c in merged.columns if c not in ("date", "Code"))
    else:
        raise RuntimeError("No factor data processed and no existing file!")
else:
    print(f"\nMerging {len(batch_paths)} batches (indexed concat)...")
    dfs = []
    for bp in tqdm(batch_paths, desc="Reading & indexing"):
        dfs.append(pd.read_feather(bp).set_index(["date", "Code"]))

    new_merged = pd.concat(dfs, axis=1)
    del dfs
    gc.collect()
    print(f"  New/refreshed factor data: {new_merged.shape[0]} rows, "
          f"dates {new_merged.index.get_level_values('date').min()} ~ "
          f"{new_merged.index.get_level_values('date').max()}")

    if incremental_fac:
        existing = pd.read_feather(fac_all_path).set_index(["date", "Code"])
        # Remove rows that overlap with the reprocess window
        if reprocess_start_date is not None:
            before = existing.shape[0]
            existing = existing[existing.index.get_level_values("date").astype(str) < str(reprocess_start_date)]
            removed = before - existing.shape[0]
            if removed > 0:
                print(f"  Removed {removed} old rows (dates >= {reprocess_start_date}) to be refreshed")
        if not new_merged.empty:
            combined = pd.concat([existing, new_merged])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.copy()   # defragment after concat + boolean index
        else:
            combined = existing
        del existing, new_merged
        gc.collect()
    else:
        combined = new_merged.copy()     # defragment

    combined.reset_index(inplace=True)
    factor_cols = sorted(c for c in combined.columns if c not in ("date", "Code"))
    combined = combined[["date", "Code"] + factor_cols]
    combined.sort_values(["date", "Code"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"Merged shape: {combined.shape}")
    print(f"Factor columns: {len(factor_cols)}")

    combined.to_feather(fac_all_path)
    print(f"Saved fac_all.fea to {fac_all_path}")
    del combined
    gc.collect()

# Clean up temp files
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)
print("Cleaned up temp batch files")

# ============================================================
# 1.5 Generate fac_sample.fea — randomly sample SAMPLE_FACTOR_COUNT factors
# ============================================================
print(f"\nGenerating fac_sample.fea with {SAMPLE_FACTOR_COUNT} randomly sampled factors...")

rng = np.random.default_rng(RANDOM_SEED)
n_factors = len(factor_cols)
n_sample = min(SAMPLE_FACTOR_COUNT, n_factors)
sampled_cols = sorted(rng.choice(factor_cols, size=n_sample, replace=False).tolist())

print(f"  Sampled {n_sample}/{n_factors} factors: {sampled_cols}")

# Read fac_all.fea and select only the sampled columns
fac_all_df = pd.read_feather(fac_all_path)
fac_sample_df = fac_all_df[["date", "Code"] + sampled_cols]
fac_sample_df.to_feather(fac_sample_path)
print(f"  Saved fac_sample.fea ({fac_sample_df.shape}) to {fac_sample_path}")
del fac_all_df, fac_sample_df
gc.collect()

# ============================================================
# 1.6 Generate fac_select.fea — correlation-based redundancy filter
# ============================================================
print(f"\n{'─'*50}")
print(f"Factor selection (redundancy filter)")
print(f"{'─'*50}")

select_summary = select_factors(
    fac_all_path, fac_select_path, fac_select_meta_path,
    force=args.reselect,
    target_dir=Path(LABEL_TARGETS["label_ret_1d"]).parent,
    min_keep=SELECT_MIN_KEEP, max_keep=SELECT_MAX_KEEP,
)
print(f"  fac_select: {select_summary['n_kept']}/{select_summary['n_factors_total']} "
      f"factors kept (threshold={select_summary['threshold']}, "
      f"status={select_summary['status']})")

# ============================================================
# 2. Generate labels (1d, 3d, 5d, 10d)
# ============================================================
print(f"\n{'─'*50}")
print(f"Label generation")
print(f"{'─'*50}")

for label_name, label_src_path in LABEL_TARGETS.items():
    label_out_path = out_dir / f"{label_name}.fea"
    horizon = _label_horizon_days(label_name)

    if FORCE_FULL:
        incremental_label = False
    elif FORCE_INCREMENTAL is True:
        incremental_label = label_out_path.exists()
        if not incremental_label:
            print(f"\n[{label_name}] --incremental but no existing output; full generation.")
    else:
        incremental_label = label_out_path.exists()

    if incremental_label:
        # Read existing max date
        _tmp = pd.read_feather(label_out_path, columns=["index"])
        label_existing_max = str(_tmp["index"].max())
        del _tmp; gc.collect()

        # Read upstream label source to find its max date
        label_source_idx = pd.read_feather(label_src_path, columns=[])
        label_upstream_max = str(sorted(label_source_idx.index.unique())[-1])
        del label_source_idx

        print(f"\n[{label_name}] existing max: {label_existing_max}  |  upstream max: {label_upstream_max}")

        if label_upstream_max <= label_existing_max:
            print(f"  No new label data, skipping.")
            continue

        # Load only new dates (labels are point-in-time, no lookback needed)
        label_source = pd.read_feather(label_src_path)
        new_rows_mask = label_source.index.astype(str) > str(label_existing_max)
        label_new = label_source[new_rows_mask]

        if label_new.empty:
            print(f"  No new label data, skipping.")
        else:
            print(f"  {len(label_new)} new dates, processing...")
            existing_label = pd.read_feather(label_out_path)

            label_new = label_new.reset_index().rename(columns={"Date": "index"})
            label_new["index"] = label_new["index"].astype(str)
            label_new_cols = ["index"] + [c for c in label_new.columns
                                           if c != "index" and c in allowed_codes_set]
            label_new = label_new[label_new_cols]

            combined = pd.concat([existing_label, label_new], ignore_index=True)
            combined.drop_duplicates(subset=["index"], keep="last", inplace=True)
            combined.sort_values("index", inplace=True)
            combined.reset_index(drop=True, inplace=True)

            print(f"  [{label_name}] shape: {combined.shape}")
            combined.to_feather(label_out_path)
            print(f"  Saved to {label_out_path}")
            del existing_label, label_new, combined
            gc.collect()
        del label_source
        gc.collect()
    else:
        print(f"\n[{label_name}] No existing label, full generation...")
        print(f"  Loading source...")
        label_df = pd.read_feather(label_src_path)

        print("  Processing label...")
        label = label_df.reset_index()
        label = label.rename(columns={"Date": "index"})
        label["index"] = label["index"].astype(str)

        label_cols = ["index"] + [c for c in label.columns if c != "index" and c in allowed_codes_set]
        label = label[label_cols]

        print(f"  [{label_name}] shape: {label.shape}")
        label.to_feather(label_out_path)
        print(f"  Saved to {label_out_path}")
        del label, label_df
        gc.collect()

# ============================================================
# 3. Generate trade_amt
# ============================================================
print(f"\n{'─'*50}")
print(f"Trade amount generation")
print(f"{'─'*50}")

if FORCE_FULL:
    incremental_trade = False
elif FORCE_INCREMENTAL is True:
    incremental_trade = trade_path.exists()
    if not incremental_trade:
        print("--incremental but no existing trade_amt; full generation.")
else:
    incremental_trade = trade_path.exists()

if incremental_trade:
    _tmp = pd.read_feather(trade_path, columns=["index"])
    trade_existing_max = str(_tmp["index"].max())
    del _tmp; gc.collect()

    # Check upstream max date from daily_adj (cheap scan)
    trade_upstream = pd.read_parquet(daily_adj_path, columns=["trade_date"])
    trade_upstream_max = trade_upstream["trade_date"].str.replace("-", "", regex=False).max()
    del trade_upstream

    print(f"\n[trade_amt] existing max: {trade_existing_max}  |  upstream max: {trade_upstream_max}")

    if trade_upstream_max <= trade_existing_max:
        print("  No new trade data, skipping.")
    else:
        print(f"  Loading daily_adj.parquet (new dates > {trade_existing_max})...")
        trade_df = pd.read_parquet(daily_adj_path,
                                   columns=["stock_code", "trade_date", "amount"])

        trade_df["date_str"] = trade_df["trade_date"].str.replace("-", "", regex=False)
        trade_df = trade_df[trade_df["date_str"] > str(trade_existing_max)]

        if trade_df.empty:
            print("  No new trade data after filtering, skipping.")
        else:
            print(f"  {len(trade_df)} new rows, processing...")
            existing_trade = pd.read_feather(trade_path)

            trade_df["Code"] = trade_df["stock_code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
            trade_df = trade_df[trade_df["Code"].isin(allowed_codes_set)]
            print(f"  Filtered to {len(trade_df)} rows, pivoting...")

            trade_new = trade_df.pivot_table(index="date_str", columns="Code",
                                             values="amount", aggfunc="first")
            trade_new.reset_index(inplace=True)
            trade_new.rename(columns={"date_str": "index"}, inplace=True)

            combined = pd.concat([existing_trade, trade_new], ignore_index=True)
            combined.drop_duplicates(subset=["index"], keep="last", inplace=True)
            combined.sort_values("index", inplace=True)
            combined.reset_index(drop=True, inplace=True)

            print(f"  trade_amt shape: {combined.shape}")
            combined.to_feather(trade_path)
            print(f"  Saved trade_amt to {trade_path}")
            del existing_trade, trade_new, combined
            gc.collect()
        del trade_df
        gc.collect()
else:
    print("\nNo existing trade_amt, full generation...")
    print("Loading daily_adj.parquet...")
    trade_df = pd.read_parquet(daily_adj_path,
                               columns=["stock_code", "trade_date", "amount"])

    trade_df["Code"] = trade_df["stock_code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
    trade_df["date"] = pd.to_datetime(trade_df["trade_date"]).dt.strftime("%Y%m%d")

    trade_df = trade_df[trade_df["Code"].isin(allowed_codes_set)]
    print(f"Filtered to {len(trade_df)} rows, pivoting...")
    trade_amt = trade_df.pivot_table(index="date", columns="Code",
                                     values="amount", aggfunc="first")

    trade_amt.reset_index(inplace=True)
    trade_amt.rename(columns={"date": "index"}, inplace=True)

    print(f"trade_amt shape: {trade_amt.shape}")
    trade_amt.to_feather(trade_path)
    print(f"Saved trade_amt to {trade_path}")
    del trade_amt, trade_df
    gc.collect()

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"Data preparation complete")
print(f"{'='*60}")
print(f"  fac_all.fea:    {fac_all_path}")
print(f"  fac_sample.fea: {fac_sample_path}")
print(f"  fac_select.fea: {fac_select_path} ({select_summary['n_kept']} factors)")
for label_name in LABEL_TARGETS:
    print(f"  {label_name}.fea:     {out_dir / f'{label_name}.fea'}")
print(f"  trade_amt.fea:     {trade_path}")
print(f"\nAll done.")
