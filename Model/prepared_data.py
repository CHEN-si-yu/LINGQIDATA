import gc
import shutil
import calendar

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================
# Input paths
# ============================================================
code_num_path = PROJECT_ROOT / "Code_num.txt"
factor_dir = PROJECT_ROOT / "featureengineering/data/factors"
label_ret_1d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_1d.fea"
label_ret_3d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_3d.fea"
label_ret_5d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_5d.fea"
label_ret_10d_path = PROJECT_ROOT / "featureengineering/data/targets/label_ret_10d.fea"
daily_adj_path = PROJECT_ROOT / "data/daily_adj.parquet"

# ============================================================
# Output paths (V7 data)
# ============================================================
out_dir = PROJECT_ROOT / "trainingdata"
fac_path = out_dir / "fac_all.fea"
trade_path = out_dir / "trade_amt.fea"

# Label outputs: one per target horizon
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

# --- Config for fac_sample and fac_IC_topK generation ---
SAMPLE_N_FACTORS = 50          # Number of random factors for fac_sample.fea
SAMPLE_SEED = 42               # Random seed for reproducibility
IC_TIME_RANGE = "25Q1~26Q1"    # Time range for IC computation (format: startQ~endQ)
IC_TARGET = "ret_1d"           # Target label for IC computation
IC_TOP_K = 500                 # Number of top factors by |mean(IC)|

# Label paths for IC computation (processed labels in trainingdata/)
IC_LABEL_PATHS = {
    "ret_1d":  out_dir / "label_ret_1d.fea",
    "ret_3d":  out_dir / "label_ret_3d.fea",
    "ret_5d":  out_dir / "label_ret_5d.fea",
    "ret_10d": out_dir / "label_ret_10d.fea",
}

# Output paths for subsets
fac_sample_path = out_dir / "fac_sample.fea"
fac_ic_topk_path = out_dir / rf"fac_IC_top{IC_TOP_K}.fea"
fac_ic_ranking_path = out_dir / "fac_IC_topK_ranking.csv"

# ============================================================
# Helper
# ============================================================
def _load_factor_stacked(path, cutoff=None):
    """Read a factor .fea file, optionally filter to dates > cutoff, stack to Series.
    Returns None if no data remains after filtering."""
    df = pd.read_feather(path)
    if cutoff is not None:
        df = df[df.index.astype(str) > str(cutoff)]
        if df.empty:
            return None
    return df.stack()


def _parse_quarter(q_str):
    """
    Parse a quarter string like '25Q1' or '2025Q1' into (year, quarter).

    Examples: '25Q1' -> (2025, 1), '2025Q3' -> (2025, 3)
    """
    q_str = q_str.strip().upper()
    if 'Q' not in q_str:
        raise ValueError(f"Invalid quarter format: '{q_str}', expected like '25Q1' or '2025Q1'")
    parts = q_str.split('Q')
    year_str, qtr = parts[0], int(parts[1])
    year = int(year_str)
    if year < 100:
        year += 2000
    if qtr not in (1, 2, 3, 4):
        raise ValueError(f"Invalid quarter: {qtr}, must be 1-4")
    return year, qtr


def _quarter_to_date_range(year, qtr):
    """
    Convert (year, quarter) to (start_date, end_date) as 'YYYYMMDD' strings.

    Example: (2025, 1) -> ('20250101', '20250331')
    """
    start_month = (qtr - 1) * 3 + 1
    start_date = f"{year}{start_month:02d}01"
    end_month = qtr * 3
    last_day = calendar.monthrange(year, end_month)[1]
    end_date = f"{year}{end_month:02d}{last_day}"
    return start_date, end_date


def _parse_time_range(time_range_str):
    """
    Parse a time range like '25Q1~26Q1' into (start_date, end_date) strings.

    Example: '25Q1~26Q1' -> ('20250101', '20260331')
    """
    if '~' not in time_range_str:
        raise ValueError(
            f"Invalid time range: '{time_range_str}', "
            f"expected 'startQ~endQ' (e.g. '25Q1~26Q1')"
        )
    parts = time_range_str.split('~')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid time range: '{time_range_str}', expected exactly one '~' separator"
        )
    start_year, start_qtr = _parse_quarter(parts[0])
    end_year, end_qtr = _parse_quarter(parts[1])
    start_date, _ = _quarter_to_date_range(start_year, start_qtr)
    _, end_date = _quarter_to_date_range(end_year, end_qtr)
    if start_date >= end_date:
        raise ValueError(f"Start date {start_date} is not before end date {end_date}")
    return start_date, end_date


def _generate_fac_sample(fac_df, n_factors=SAMPLE_N_FACTORS, seed=SAMPLE_SEED):
    """
    Randomly sample n_factors from fac_df columns.

    Returns (fac_sample_df, list_of_sampled_factor_names).
    """
    all_factors = sorted(c for c in fac_df.columns if c not in ("date", "Code"))
    n_available = len(all_factors)
    if n_available == 0:
        raise RuntimeError("No factor columns found in data!")
    actual_n = min(n_factors, n_available)
    rng = np.random.default_rng(seed)
    sampled = sorted(rng.choice(all_factors, size=actual_n, replace=False).tolist())
    return fac_df[["date", "Code"] + sampled].copy(), sampled


def _compute_ic_topk(fac_df, label_path, time_start, time_end, k=IC_TOP_K):
    """
    Compute the top K factors by |mean(IC)| over [time_start, time_end].

    Methodology (same as fac_filt.py's FacMetric):
      For each date: align factor values and target returns by stock code,
      compute Pearson correlation (corrwith) between each factor and the target.
      First average the signed IC across dates, then take the absolute value |mean(IC)|, rank descending, pick top K.

    Returns (result_df, ic_ranking_df, top_factor_names).
    """
    # --- Load & filter label ---
    print("  Loading label data...")
    label_df = pd.read_feather(label_path)
    label_df = label_df.rename(columns={"index": "date"})
    label_df["date"] = label_df["date"].astype(str)
    mask = (label_df["date"] >= time_start) & (label_df["date"] <= time_end)
    label_df = label_df[mask]
    if label_df.empty:
        raise RuntimeError(f"No label data found in [{time_start}, {time_end}]")

    # Melt to long format: date, Code, target
    print(f"  Melting label ({label_df.shape[0]} dates × {label_df.shape[1]-1} stocks)...")
    label_long = label_df.melt(id_vars=["date"], var_name="Code", value_name="target")
    label_long = label_long.dropna(subset=["target"])
    label_long["Code"] = label_long["Code"].astype(str)
    del label_df; gc.collect()

    # --- Filter factor data ---
    print("  Filtering factor data...")
    fac_df["date"] = fac_df["date"].astype(str)
    fac_sub = fac_df[(fac_df["date"] >= time_start) & (fac_df["date"] <= time_end)].copy()
    fac_sub["Code"] = fac_sub["Code"].astype(str)

    factor_cols = sorted(c for c in fac_df.columns if c not in ("date", "Code"))
    dates = sorted(fac_sub["date"].unique())
    print(f"  Dates in range: {len(dates)}, factors: {len(factor_cols)}")
    if len(dates) == 0:
        raise RuntimeError(f"No factor data found in [{time_start}, {time_end}]")

    # --- Compute IC per date (same corrwith method as fac_filt.py) ---
    print("  Computing daily IC...")
    ic_records = []
    skipped = 0
    for date in tqdm(dates, desc="  IC per date"):
        fac_td = fac_sub[fac_sub["date"] == date].set_index("Code")[factor_cols]
        label_td = label_long[label_long["date"] == date].set_index("Code")["target"]
        common = fac_td.index.intersection(label_td.index)
        if len(common) < 30:
            skipped += 1
            continue
        # Drop zero-variance cols (would produce NaN IC, correctly skipped in mean)
        fac_aligned = fac_td.loc[common]
        label_aligned = label_td.loc[common]
        valid_cols = fac_aligned.columns[fac_aligned.std() > 1e-15]
        with np.errstate(invalid="ignore"):
            ic = fac_aligned[valid_cols].corrwith(label_aligned)
            ic = ic.reindex(fac_aligned.columns)  # restore full index (NaN for constant cols)
        ic.name = date
        ic_records.append(ic)

    if skipped:
        print(f"  Skipped {skipped} dates with < 30 common stocks")
    if not ic_records:
        raise RuntimeError("No valid dates with sufficient common stocks found!")

    ic_df = pd.concat(ic_records, axis=1)
    print(f"  IC matrix shape (factors × dates): {ic_df.shape}")

    # First average the signed IC across dates, then take absolute: |mean(IC)|
    ic_mean_abs = ic_df.mean(axis=1).abs().sort_values(ascending=False)

    actual_k = min(k, len(ic_mean_abs))
    top_factors = ic_mean_abs.head(actual_k).index.tolist()

    print(f"  Selected {len(top_factors)} factors (K={k}, available={len(ic_mean_abs)})")
    print(f"  |mean(IC)| range of selected: [{ic_mean_abs.iloc[actual_k-1]:.6f}, {ic_mean_abs.iloc[0]:.6f}]")
    print(f"  Top 10:")
    for i, (fac, val) in enumerate(ic_mean_abs.head(10).items()):
        print(f"    {i+1:2d}. {fac:50s} |mean(IC)|={val:.6f}")

    result_df = fac_df[["date", "Code"] + top_factors].copy()

    ic_ranking = ic_mean_abs.reset_index()
    ic_ranking.columns = ["factor", "mean_ic_abs"]
    ic_ranking["rank"] = range(1, len(ic_ranking) + 1)

    return result_df, ic_ranking, top_factors


# ============================================================
# Setup
# ============================================================
out_dir.mkdir(parents=True, exist_ok=True)

with open(code_num_path) as f:
    allowed_codes = [line.strip() for line in f if line.strip()]
allowed_codes_set = set(allowed_codes)
print(f"Loaded {len(allowed_codes)} codes from {code_num_path}")

# ============================================================
# 1. Merge factors (batch-processed to avoid OOM)
# ============================================================
files = sorted(factor_dir.glob("*.fea"))
if not files:
    raise FileNotFoundError(f"No .fea files found in {factor_dir}")

# --- Incremental detection ---
incremental_fac = fac_path.exists()
cutoff_date = None

if incremental_fac:
    print(f"Found existing factors, incremental mode...")
    _tmp = pd.read_feather(fac_path, columns=["date"])
    cutoff_date = _tmp["date"].max()
    del _tmp; gc.collect()
    print(f"  Max date in existing: {cutoff_date}, will only process newer dates")
else:
    print(f"No existing factor file, full generation...")

print(f"Found {len(files)} factor files, loading with {N_WORKERS} workers...")

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
        futures = {pool.submit(_load_factor_stacked, f, cutoff_date): f.stem
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
    # Even if no new factor data, still need existing loaded for factor_cols
    if incremental_fac:
        merged = pd.read_feather(fac_path)
        factor_cols = sorted(c for c in merged.columns if c not in ("date", "Code"))
        # already saved, nothing to do
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

    if incremental_fac:
        existing = pd.read_feather(fac_path).set_index(["date", "Code"])
        combined = pd.concat([existing, new_merged])
        combined = combined[~combined.index.duplicated(keep="last")]
        del existing, new_merged
        gc.collect()
    else:
        combined = new_merged

    combined.reset_index(inplace=True)
    factor_cols = sorted(c for c in combined.columns if c not in ("date", "Code"))
    combined = combined[["date", "Code"] + factor_cols]
    combined.sort_values(["date", "Code"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"Merged shape: {combined.shape}")
    print(f"Factor columns: {len(factor_cols)}")

    combined.to_feather(fac_path)
    print(f"Saved factors to {fac_path}")
    del combined
    gc.collect()

# Clean up temp files
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)
print("Cleaned up temp batch files")

# ============================================================
# 2. Generate labels (1d, 3d, 5d, 10d)
# ============================================================
for label_name, label_src_path in LABEL_TARGETS.items():
    label_out_path = out_dir / f"{label_name}.fea"
    incremental_label = label_out_path.exists()

    if incremental_label:
        print(f"\n[{label_name}] Found existing label, incremental mode...")
        _tmp = pd.read_feather(label_out_path, columns=["index"])
        max_label_date = _tmp["index"].max()
        del _tmp; gc.collect()
        print(f"  Max date in existing: {max_label_date}")

        print(f"  Loading {label_name}.fea source...")
        label_source = pd.read_feather(label_src_path)
        new_rows_mask = label_source.index.astype(str) > str(max_label_date)
        label_new = label_source[new_rows_mask]

        if label_new.empty:
            print(f"  No new label data, skipping.")
        else:
            print(f"  {len(label_new)} new dates found, processing...")
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
        print(f"  Loading {label_name}.fea source...")
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
incremental_trade = trade_path.exists()

if incremental_trade:
    print(f"\nFound existing trade_amt, incremental mode...")
    _tmp = pd.read_feather(trade_path, columns=["index"])
    max_trade_date = _tmp["index"].max()
    del _tmp; gc.collect()
    print(f"  Max date in existing: {max_trade_date}")

    print("Loading daily_adj.parquet...")
    trade_df = pd.read_parquet(daily_adj_path,
                               columns=["stock_code", "trade_date", "amount"])

    # daily_adj trade_date is "YYYY-MM-DD", convert to "YYYYMMDD" for comparison
    trade_df["date_str"] = trade_df["trade_date"].str.replace("-", "", regex=False)
    trade_df = trade_df[trade_df["date_str"] > str(max_trade_date)]

    if trade_df.empty:
        print("No new trade data, skipping.")
        
        
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

        print(f"trade_amt shape: {combined.shape}")
        combined.to_feather(trade_path)
        print(f"Saved trade_amt to {trade_path}")
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

    # Filter to allowed codes before pivoting
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
# 4. Generate fac_sample.fea (random 50-factor subset for quick testing)
# ============================================================
print(f"\n{'='*60}")
print(f"[4] Generating fac_sample.fea ({SAMPLE_N_FACTORS} random factors, seed={SAMPLE_SEED})...")

# Load the merged factor file
fac_all = pd.read_feather(fac_path)
print(f"  Loaded fac_all: {fac_all.shape}")

fac_sample_df, sampled_factors = _generate_fac_sample(
    fac_all, n_factors=SAMPLE_N_FACTORS, seed=SAMPLE_SEED
)
fac_sample_df.to_feather(fac_sample_path)
print(f"  Saved: {fac_sample_path}")
print(f"  Shape: {fac_sample_df.shape}")
print(f"  Sampled factors ({len(sampled_factors)}):")
for i, fn in enumerate(sampled_factors):
    print(f"    {i+1:2d}. {fn}")
del fac_sample_df
gc.collect()

# # ============================================================
# # 5. Generate fac_IC_topK.fea (top K factors by |mean(IC)|, long format)
# # ============================================================
# print(f"\n{'='*60}")
# print(f"[5] Generating fac_IC_topK.fea (long format, same as fac_all.fea)")
# print(f"    Time range: {IC_TIME_RANGE}")
# print(f"    Target:     {IC_TARGET}")
# print(f"    K:          {IC_TOP_K}")

# time_start, time_end = _parse_time_range(IC_TIME_RANGE)
# print(f"    → {time_start} ~ {time_end}")

# label_path = IC_LABEL_PATHS[IC_TARGET]
# if not label_path.exists():
#     raise FileNotFoundError(f"Label file not found: {label_path}")

# fac_ic_topk_df, ic_ranking_df, top_factors = _compute_ic_topk(
#     fac_all, label_path, time_start, time_end, k=IC_TOP_K
# )

# fac_ic_topk_df.to_feather(fac_ic_topk_path)
# print(f"\n  Saved: {fac_ic_topk_path}")
# print(f"  Shape: {fac_ic_topk_df.shape}")


# del fac_all, fac_ic_topk_df
# gc.collect()

print(f"\n{'='*60}")
print("\nAll done.")
