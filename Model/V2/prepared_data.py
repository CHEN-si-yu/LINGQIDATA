import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

N_WORKERS = 16

out_dir = Path("/root/shared-nvme/lingqiData/trainingdata/V2")
out_dir.mkdir(parents=True, exist_ok=True)

# Load allowed stock codes
code_num_path = Path("/root/shared-nvme/lingqiData/Code_num.txt")
with open(code_num_path) as f:
    allowed_codes = [line.strip() for line in f if line.strip()]
allowed_codes_set = set(allowed_codes)
print(f"Loaded {len(allowed_codes)} codes from {code_num_path}")

# ============================================================
# 1. Merge factors
# ============================================================
factor_dir = Path("/root/shared-nvme/lingqiData/featureengineering/data/factors")
files = sorted(factor_dir.glob("*.fea"))
if not files:
    raise FileNotFoundError(f"No .fea files found in {factor_dir}")

print(f"Found {len(files)} factor files, loading with {N_WORKERS} workers...")


def load_one(path):
    df = pd.read_feather(path)
    s = df.stack()
    s.name = path.stem
    return s


series_list = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(load_one, f): f.stem for f in files}
    for fut in tqdm(as_completed(futures), total=len(files), desc="Loading factors"):
        s = fut.result()
        series_list.append(s)

print("Concatenating all factors...")
merged = pd.concat(series_list, axis=1)
del series_list

merged.reset_index(inplace=True)
merged.rename(columns={"Date": "date"}, inplace=True)

# Strip .BJ suffix from codes (sector files have "920000.BJ" format)
merged["Code"] = merged["Code"].astype(str).str.replace(".BJ", "", regex=False)

factor_cols = sorted([c for c in merged.columns if c not in ("date", "Code")])
merged = merged[["date", "Code"] + factor_cols]
merged["date"] = merged["date"].astype(str)
merged["Code"] = merged["Code"].astype(str)
merged.sort_values(["date", "Code"], inplace=True)
merged.reset_index(drop=True, inplace=True)

# Filter to allowed codes only
merged = merged[merged["Code"].isin(allowed_codes_set)]
merged.reset_index(drop=True, inplace=True)

print(f"Merged shape: {merged.shape}")
print(f"Factor columns: {len(factor_cols)}")

fac_path = out_dir / "fac20260517.fea"
merged.to_feather(fac_path)
print(f"Saved factors to {fac_path}")

# ============================================================
# 2. Generate label
# ============================================================
print("\nLoading label_ret_1d.fea...")
label_df = pd.read_feather("/root/shared-nvme/lingqiData/featureengineering/data/targets/label_ret_1d.fea")

print("Processing label...")
label = label_df.reset_index()
label = label.rename(columns={"Date": "index"})
label["index"] = label["index"].astype(str)

# Filter to allowed codes only (label has "index" + code columns)
label_cols = ["index"] + [c for c in label.columns if c != "index" and c in allowed_codes_set]
label = label[label_cols]

print(f"label shape: {label.shape}")

label_path = out_dir / "label.fea"
label.to_feather(label_path)
print(f"Saved label to {label_path}")

# ============================================================
# 3. Generate trade_amt
# ============================================================
print("\nLoading daily_adj.parquet...")
trade_df = pd.read_parquet("/root/shared-nvme/lingqiData/data/daily_adj.parquet",
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

trade_path = out_dir / "trade_amt.fea"
trade_amt.to_feather(trade_path)
print(f"Saved trade_amt to {trade_path}")

print("\nAll done.")
