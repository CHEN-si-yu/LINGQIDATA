import argparse
import gc
import os
import re
import shutil

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
label_ret_20d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_20d.fea"
daily_adj_path = PROJECT_ROOT / "data/daily_adj.parquet"
daily_path = PROJECT_ROOT / "data/daily.parquet"   # 未复权原始价（buyable_mask 用，禁止用 daily_adj 复权价算涨停价）
calendar_path = PROJECT_ROOT / "data/calendar.parquet"

# ============================================================
# Output paths
# ============================================================
out_dir = PROJECT_ROOT / "trainingdata"
fac_all_path = out_dir / "fac_all.fea"
fac_sample_path = out_dir / "fac_sample.fea"
trade_path = out_dir / "trade_amt.fea"

LABEL_TARGETS = {
    "label_ret_1d":  label_ret_1d_path,
    "label_ret_3d":  label_ret_3d_path,
    "label_ret_5d":  label_ret_5d_path,
    "label_ret_10d": label_ret_10d_path,
    "label_ret_20d": label_ret_20d_path,
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
    for _p in [fac_all_path, fac_sample_path, trade_path] + [
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
# 4. Generate buyable_mask (T+1 开盘可买入掩码) —— 全量重建
# ============================================================
# 语义（V6.2 定型，模型侧 label 剔除依据）：mask[T][code] 表示 T+1 交易日开盘是否可买入。
#   True  — T+1 有 bar 且非一字涨停（开盘涨停但盘中打开、一字跌停均可买）
#   False — T+1 一字涨停（open==high==low==close 且 close ≥ 涨停价，无法成交）
#   None  — T+1 无 bar（停牌/退市/未上市/最后交易日；模型侧视为不可买，一并剔除）
# 涨停价按板块：主板 10% / 创业板(30x)+科创板(68x) 20% / ST 5% / 北交所(4x,8x) 30%，
# 用 T+1 的 pre_close 计算（与 T+1 收盘价同口径）。
# 注（2026-08-14 实测）：存量 buyable_mask.fea 与"一字涨停"规则一致率 99.9999%
# （3,321,572 格中仅 2 格差异）；ST 股票实测未排除（历史脚本的 ST 判定未生效），
# 为保持 V6.2~V6.9 训练口径一致照抄，如需排除 ST 需重训模型。
# 数据源：daily.parquet（未复权）；交易日历 calendar.parquet 取严格 T+1（跳停牌 → None）。
# 输出：trainingdata/buyable_mask.fea（date 列 + Code 列，True/False/None）。
print(f"\n{'─'*50}")
print(f"Buyable mask generation (T+1 一字涨停不可买)")
print(f"{'─'*50}")

mask_daily = pd.read_parquet(daily_path,
                             columns=["stock_code", "trade_date", "open", "high", "low",
                                      "close", "pre_close", "stock_name"])
mask_daily["Code"] = mask_daily["stock_code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
mask_daily["date"] = mask_daily["trade_date"].astype(str).str.replace("-", "", regex=False)
# 严格 T+1：date → 下一交易日（calendar 只追加不修改）
nxt_map = {d: n for d, n in zip(trading_calendar[:-1], trading_calendar[1:])}
mask_daily["T1"] = mask_daily["date"].map(nxt_map)

# T+1 bar 表：按 bar 自身日期标注（T1=bar 日），mask 行 date=T 通过 T1=next(T) 挂上 T+1 的 bar
t1 = mask_daily[["date", "Code", "open", "high", "low", "close", "pre_close", "stock_name"]].rename(
    columns={"date": "T1", "open": "T1_open", "high": "T1_high", "low": "T1_low",
             "close": "T1_close", "pre_close": "T1_pre", "stock_name": "T1_name"})
m = mask_daily[["date", "Code", "T1"]].merge(t1, on=["T1", "Code"], how="left")

has_t1 = m["T1_open"].notna()
t1_st = m["T1_name"].astype(str).str.contains("ST", na=False)
is_cy = m["Code"].str.startswith(("30", "68"))
is_bj = m["Code"].str.startswith(("4", "8"))
lim_pct = np.where(t1_st, 0.05, np.where(is_cy, 0.20, np.where(is_bj, 0.30, 0.10)))
t1_lim = (m["T1_pre"] * (1 + lim_pct)).round(2)
# 一字涨停：四价相同 且 收盘 ≥ 涨停价（容忍浮点舍入）
one_word_limit = (
    (m["T1_open"] == m["T1_high"]) & (m["T1_open"] == m["T1_low"]) &
    (m["T1_open"] == m["T1_close"]) & (m["T1_close"] >= t1_lim - 0.005)
)

m["mask"] = np.where(has_t1 & one_word_limit, False, np.where(has_t1, True, np.nan))
buyable = m.pivot(index="date", columns="Code", values="mask").reset_index()
buyable = buyable[buyable.iloc[:, 1:].notna().any(axis=1)]   # 去掉全 None 行（最后交易日无 T+1）
buyable.to_feather(out_dir / "buyable_mask.fea")
print(f"  Saved buyable_mask.fea {buyable.shape} ({len(buyable)} 个交易日)")
print(f"  True={int((buyable.iloc[:,1:] == True).sum().sum())} "
      f"False={int((buyable.iloc[:,1:] == False).sum().sum())} "
      f"None={int(buyable.iloc[:,1:].isna().sum().sum())}")
del mask_daily, t1, m, buyable
gc.collect()

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"Data preparation complete")
print(f"{'='*60}")
print(f"  fac_all.fea:    {fac_all_path}")
print(f"  fac_sample.fea: {fac_sample_path}")
for label_name in LABEL_TARGETS:
    print(f"  {label_name}.fea:     {out_dir / f'{label_name}.fea'}")
print(f"  trade_amt.fea:     {trade_path}")
print(f"  buyable_mask.fea:  {out_dir / 'buyable_mask.fea'}")
print(f"\nAll done.")
