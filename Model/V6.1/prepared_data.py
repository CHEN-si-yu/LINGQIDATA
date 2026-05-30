import gc
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================
# Input paths
# ============================================================
code_num_path = Path("/root/shared-nvme/lingqiData/Code_num.txt")
factor_dir = Path("/root/shared-nvme/lingqiData/featureengineering/data/factors")
label_ret_path = Path("/root/shared-nvme/lingqiData/featureengineering/data/targets/label_ret_5d.fea")
daily_adj_path = Path("/root/shared-nvme/lingqiData/data/daily_adj.parquet")

# ============================================================
# Output paths (V6 data)
# ============================================================
out_dir = Path("/root/shared-nvme/lingqiData/trainingdata/V6")
fac_path = out_dir / "fac20260528.fea"
label_path = out_dir / "label.fea"
trade_path = out_dir / "trade_amt.fea"

# ============================================================
# Config
# ============================================================
N_WORKERS = 16
BATCH_SIZE = 100
TMP_DIR = out_dir / "tmp_batches"

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
        futures = {pool.submit(lambda f: pd.read_feather(f).stack(), f): f.stem for f in batch_files}
        for fut in tqdm(as_completed(futures), total=len(batch_files),
                        desc=f"Batch {batch_idx+1}/{len(batches)}"):
            s = fut.result()
            s.name = futures[fut]
            s = s.astype(np.float32)
            series_list.append(s)

    batch_df = pd.concat(series_list, axis=1)
    del series_list
    gc.collect()

    batch_df.reset_index(inplace=True)
    batch_df.rename(columns={"Date": "date"}, inplace=True)
    batch_df["Code"] = batch_df["Code"].astype(str).str.replace(".BJ", "", regex=False)
    batch_df["date"] = batch_df["date"].astype(str)
    batch_df = batch_df[batch_df["Code"].isin(allowed_codes_set)]

    all_factor_cols.update(c for c in batch_df.columns if c not in ("date", "Code"))

    batch_path = TMP_DIR / f"batch_{batch_idx:03d}.fea"
    batch_df.to_feather(batch_path)
    batch_paths.append(batch_path)
    del batch_df
    gc.collect()

print(f"\nMerging {len(batch_paths)} batches incrementally...")
merged = pd.read_feather(batch_paths[0])
for bp in tqdm(batch_paths[1:], desc="Merging batches"):
    right = pd.read_feather(bp)
    merged = merged.merge(right, on=["date", "Code"], how="outer")
    del right
    gc.collect()

factor_cols = sorted(all_factor_cols)
merged = merged[["date", "Code"] + factor_cols]
merged.sort_values(["date", "Code"], inplace=True)
merged.reset_index(drop=True, inplace=True)

print(f"Merged shape: {merged.shape}")
print(f"Factor columns: {len(factor_cols)}")

merged.to_feather(fac_path)
print(f"Saved factors to {fac_path}")

# Clean up temp files
shutil.rmtree(TMP_DIR)
print("Cleaned up temp batch files")

# ============================================================
# 2. Generate label
# ============================================================
print("\nLoading label_ret_5d.fea...")
label_df = pd.read_feather(label_ret_path)

print("Processing label...")
label = label_df.reset_index()
label = label.rename(columns={"Date": "index"})
label["index"] = label["index"].astype(str)

# Filter to allowed codes only (label has "index" + code columns)
label_cols = ["index"] + [c for c in label.columns if c != "index" and c in allowed_codes_set]
label = label[label_cols]

print(f"label shape: {label.shape}")

label.to_feather(label_path)
print(f"Saved label to {label_path}")

# ============================================================
# 3. Generate trade_amt
# ============================================================
print("\nLoading daily_adj.parquet...")
trade_df = pd.read_parquet(daily_adj_path,
                           columns=["stock_code", "trade_date", "amount"])

trade_df["Code"] = trade_df["stock_code"].str.replace(".SZ", "", regex=False).str.replace(".SH", "", regex=False).str.replace(".BJ", "", regex=False)
trade_df["date"] = pd.to_datetime(trade_df["trade_date"]).dt.strftime("%Y%m%d")

# Filter to allowed codes before pivoting
trade_df = trade_df[trade_df["Code"].isin(allowed_codes_set)]
print(f"Filtered to {len(trade_df)} rows, pivoting...")
trade_amt = trade_df.pivot_table(index="date", columns="Code", values="amount", aggfunc="first")

trade_amt.reset_index(inplace=True)
trade_amt.rename(columns={"date": "index"}, inplace=True)

print(f"trade_amt shape: {trade_amt.shape}")

trade_amt.to_feather(trade_path)
print(f"Saved trade_amt to {trade_path}")

print("\nAll done.")
