import gc
import shutil

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
fac_path = out_dir / "fac20260614.fea"
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

print("\nAll done.")
