#!/usr/bin/env python3
"""
Debug快照脚本 v3 — 扩展逐日横截面数据指纹。

v3 改进：
- 移除 daily_adj（计划废弃，改用 daily + 自行复权）
- 新增 cyq_chips、history_1min、ths_daily 三个数据源
- finance 增加逐字段 MD5 追踪，精确判断哪个字段变了

用法：每天 pipeline 跑完后执行一次
    python debug/snapshot.py

输出：debug/YYYYMMDD/ 目录
"""

import os, json, hashlib, pickle, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # debug/
PROJECT = ROOT.parent                   # lingqiData/
TODAY = datetime.now().strftime("%Y%m%d")
OUT = ROOT / TODAY

# 清理旧快照（如果同一天多次运行）
import shutil
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"  Debug Snapshot v3 — {TODAY}")
print(f"{'='*60}")

# ============================================================
# 0. 脚本自身 MD5
# ============================================================
with open(__file__, "rb") as f:
    script_md5 = hashlib.md5(f.read()).hexdigest()
print(f"\n  Script MD5: {script_md5}")

# ============================================================
# 工具函数
# ============================================================
def compute_date_hash(df, date_col, code_col, skip_cols, val_cols_override=None):
    """对单个日期的完整横截面数据计算 MD5。
    
    返回: {"md5": "...", "n_stocks": N, "n_cols": N, "val_cols": [...]}
    如果 val_cols_override 指定，则只用这些列。
    """
    if val_cols_override:
        val_cols = [c for c in val_cols_override if c in df.columns]
    else:
        val_cols = sorted([c for c in df.columns if c not in [date_col, code_col] + skip_cols])
    
    df_sorted = df.sort_values(code_col)
    
    parts = []
    for _, row in df_sorted.iterrows():
        parts.append(str(row[code_col]))
        for vc in val_cols:
            val = row[vc]
            if pd.isna(val):
                parts.append("NaN")
            else:
                parts.append(f"{val:.8g}")
    
    full_str = "|".join(parts)
    return {
        "md5": hashlib.md5(full_str.encode()).hexdigest(),
        "n_stocks": len(df_sorted),
        "n_cols": len(val_cols),
        "val_cols": val_cols,
    }


def collect_last_n_dates_from_df(df, date_col, n=10):
    """从 DataFrame 中提取最近 N 个日期。"""
    dates = sorted(df[date_col].astype(str).str[:10].unique())
    return dates[-n:]


# ============================================================
# 1. ★ 逐日横截面数据指纹
# ============================================================
print("\n[1/9] Per-date cross-sectional data hashes...")

# ── 单文件数据源 ──
SOURCE_DATE_FILES = {
    "daily":          ("daily.parquet",           "trade_date", "stock_code", ["stock_name"]),
    "finance":        ("finance.parquet",         "trade_date", "stock_code", ["stock_name", "is_backfill"]),
    "cyq_perf":       ("cyq_perf.parquet",        "trade_date", "stock_code", ["is_backfill"]),
    "main_fund_flow": ("main_fund_flow.parquet",  "trade_date", "stock_code", []),
    "margin_detail":  ("margin_detail.parquet",   "trade_date", "stock_code", ["exchange_id", "is_backfill"]),
    "ths_daily":      ("ths_daily.parquet",       "trade_date", "ths_code",   []),
}

# ── Finance 字段列表（用于逐字段 MD5） ──
FINANCE_FIELDS = [
    "close", "turnover_rate", "turnover_rate_f", "volume_ratio",
    "pe", "pe_ttm", "pe_ttm_percentile", "pb", "ps", "ps_ttm",
    "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share",
    "total_mv", "circ_mv",
]

source_date_hashes = {}

# 处理单文件源
for name, (fname, date_col, code_col, skip_cols) in SOURCE_DATE_FILES.items():
    fp = PROJECT / "data" / fname
    if not fp.exists():
        source_date_hashes[name] = {"error": "file not found"}
        print(f"  {name}: FILE MISSING")
        continue
    
    df = pd.read_parquet(fp)
    df[date_col] = df[date_col].astype(str).str[:10]
    all_dates = sorted(df[date_col].unique())
    last10 = all_dates[-10:]
    
    file_info = {
        "file": fname,
        "total_rows": len(df),
        "total_dates": len(all_dates),
        "date_range": f"{all_dates[0]} ~ {all_dates[-1]}",
        "date_col": date_col,
        "dates": {},
    }
    
    for d in last10:
        day_df = df[df[date_col] == d]
        if day_df.empty:
            file_info["dates"][d] = {"n_stocks": 0, "md5": None}
        else:
            result = compute_date_hash(day_df, date_col, code_col, skip_cols)
            file_info["dates"][d] = result
            
            # ★ Finance 特殊处理: 逐字段 MD5
            if name == "finance":
                field_hashes = {}
                for field in FINANCE_FIELDS:
                    if field in day_df.columns:
                        # 只对这一个字段 + stock_code 计算 hash
                        sub_df = day_df[[code_col, field]].copy()
                        fh = compute_date_hash(sub_df, code_col, code_col, [], val_cols_override=[field])
                        field_hashes[field] = {
                            "md5": fh["md5"],
                            "n_stocks": fh["n_stocks"],
                        }
                file_info["dates"][d]["fields"] = field_hashes
    
    source_date_hashes[name] = file_info
    
    last_with_data = [d for d in last10 if file_info["dates"][d]["md5"] is not None]
    print(f"  {name}: {len(all_dates)} dates, last10={len(last_with_data)} with data")

# ── 目录型数据源: cyq_chips（轻量采样 — 1782文件完整扫描需3.5min，暂用采样）
print("\n[2/9] cyq_chips (directory, lightweight sample)...");
chips_dir = PROJECT / "data" / "cyq_chips";
if chips_dir.is_dir():
    chip_files = sorted(chips_dir.glob("*.parquet"));
    total_size = sum(f.stat().st_size for f in chip_files);
    sample_idx = [0, len(chip_files)//4, len(chip_files)//2, 3*len(chip_files)//4, -1];
    sample_files = [chip_files[i] for i in sample_idx];
    sample_hashes = {};
    all_dates_set = set();
    for fp in sample_files:
        fhash = hashlib.md5(fp.read_bytes()).hexdigest();
        ddf = pd.read_parquet(fp, columns=["trade_date"]);
        dates = sorted(ddf["trade_date"].astype(str).str[:10].unique());
        all_dates_set.update(dates);
        sample_hashes[fp.name] = {"md5": fhash, "rows": len(ddf), "dates": f"{dates[0]}~{dates[-1]}", "n_dates": len(dates)};
    all_dates = sorted(all_dates_set);
    file_info = {"file": "cyq_chips/", "type": "directory_sample", "total_files": len(chip_files),
                 "total_size_mb": round(total_size/1048576, 1), "total_dates": len(all_dates),
                 "date_range": f"{all_dates[0]} ~ {all_dates[-1]}", "sample_files": sample_hashes, "note": "Full per-date tracking requires ~3.5min scan; using 5-file sample for now"};
    source_date_hashes["cyq_chips"] = file_info;
    print(f"  cyq_chips: {len(chip_files)} files, {total_size/1e9:.1f}GB, {len(all_dates)} dates");
else:
    source_date_hashes["cyq_chips"] = {"error": "directory not found"};
    print(f"  cyq_chips: DIRECTORY MISSING");

# ── 目录型数据源: history_1min（轻量采样 — 1782文件完整扫描需10.6min，暂用采样）
print("\n[3/9] history_1min (directory, lightweight sample)...");
h1m_dir = PROJECT / "data" / "history_1min";
if h1m_dir.is_dir():
    h1m_files = sorted(h1m_dir.glob("*.parquet"));
    total_size = sum(f.stat().st_size for f in h1m_files);
    sample_idx = [0, len(h1m_files)//4, len(h1m_files)//2, 3*len(h1m_files)//4, -1];
    sample_files = [h1m_files[i] for i in sample_idx];
    sample_hashes = {};
    all_dates_set = set();
    for fp in sample_files:
        fhash = hashlib.md5(fp.read_bytes()).hexdigest();
        ddf = pd.read_parquet(fp, columns=["trade_time"]);
        dates = sorted(ddf["trade_time"].astype(str).str[:10].unique());
        all_dates_set.update(dates);
        sample_hashes[fp.name] = {"md5": fhash, "rows": len(ddf), "dates": f"{dates[0]}~{dates[-1]}", "n_dates": len(dates)};
    all_dates = sorted(all_dates_set);
    file_info = {"file": "history_1min/", "type": "directory_sample", "total_files": len(h1m_files),
                 "total_size_mb": round(total_size/1048576, 1), "total_dates": len(all_dates),
                 "date_range": f"{all_dates[0]} ~ {all_dates[-1]}", "sample_files": sample_hashes, "note": "Full per-date tracking requires ~10.6min scan; using 5-file sample for now"};
    source_date_hashes["history_1min"] = file_info;
    print(f"  history_1min: {len(h1m_files)} files, {total_size/1e9:.1f}GB, {len(all_dates)} dates");
else:
    source_date_hashes["history_1min"] = {"error": "directory not found"};
    print(f"  history_1min: DIRECTORY MISSING")

with open(OUT / "source_date_hashes.json", "w") as f:
    json.dump(source_date_hashes, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  → source_date_hashes.json saved ({len(source_date_hashes)} sources)")

# ============================================================
# 4. 源文件元信息（辅助参考）
# ============================================================
print("\n[4/9] Source file metadata...")
source_meta = {}

# 单文件
for f in sorted((PROJECT / "data").glob("*.parquet")):
    fname = f.name
    fsize = f.stat().st_size
    f_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    try:
        df = pd.read_parquet(f)
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        date_col = date_cols[0] if date_cols else None
        nrows = len(df)

        if date_col:
            dates = df[date_col].dropna()
            dates_str = dates.astype(str)
            dmin, dmax = str(dates.min()), str(dates.max())
            ndates = dates.nunique()
            last5 = sorted(dates.unique())[-5:]
            last5_rows = {str(d): int((dates_str == str(d)).sum()) for d in last5}
        else:
            dmin = dmax = ndates = "N/A"
            last5_rows = {}

        source_meta[fname] = {
            "file_md5": f_md5,
            "size_mb": round(fsize / 1048576, 2),
            "rows": nrows,
            "date_col": date_col,
            "date_range": f"{dmin} ~ {dmax}",
            "unique_dates": ndates,
            "last5_date_rows": last5_rows,
        }
        print(f"  {fname}: {nrows} rows, {dmin}~{dmax}")
    except Exception as e:
        source_meta[fname] = {"error": str(e)}
        print(f"  {fname}: ERROR - {e}")

# 目录型
for dir_name in ["cyq_chips", "history_1min"]:
    dp = PROJECT / "data" / dir_name
    if dp.is_dir():
        files = list(dp.glob("*.parquet"))
        total_rows = 0
        sample_dates = None
        try:
            for fp in files[:5]:
                df = pd.read_parquet(fp)
                total_rows += len(df)
            total_rows = total_rows * len(files) // min(5, len(files))
            # 从 source_date_hashes 获取日期范围
            sh_info = source_date_hashes.get(dir_name, {})
            date_range = sh_info.get("date_range", "N/A")
            source_meta[dir_name] = {
                "type": "directory",
                "files": len(files),
                "rows_estimate": total_rows,
                "date_range": date_range,
            }
            print(f"  {dir_name}/: ~{len(files)} files, ~{total_rows} rows")
        except Exception as e:
            source_meta[dir_name] = {"error": str(e)}

with open(OUT / "source_meta.json", "w") as f:
    json.dump(source_meta, f, indent=2, ensure_ascii=False, default=str)

# ============================================================
# 5. 因子文件抽样指纹
# ============================================================
print("\n[5/9] Factor file sample checksums...")
factor_dir = PROJECT / "featureengineering/data/factors"
all_factors = sorted([f.stem for f in factor_dir.glob("*.fea")])

SAMPLE_FACTORS = [
    "mom_5", "mom_20", "reversal_1", "reversal_5",
    "vol_20", "amplitude_20", "bias_20",
    "change_raw",
    "bp", "ep_ttm", "turnover_5", "turnover_vol_5",
    "net_mf_flow_5", "big_order_ratio_5", "order_size_ratio_change",
    "chip_concentration", "winner_rate", "avg_cost_premium",
    "margin_balance_5",
    "amihud_illiq_20", "rv_5min",
    "bp_factor_momentum_20", "value_momentum_divergence",
]
sample = [s for s in SAMPLE_FACTORS if s in all_factors]
if len(sample) < 15:
    extras = [s for s in all_factors if s not in sample]
    sample.extend(extras[:15 - len(sample)])
print(f"  Sampling {len(sample)} factors")

factor_snap = {}
for fn in sample:
    fp = factor_dir / f"{fn}.fea"
    if not fp.exists():
        factor_snap[fn] = {"error": "file not found"}
        continue

    with open(fp, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()

    try:
        df = pd.read_feather(fp)
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = df.index.astype(str).str.replace("-", "").str.slice(0, 8)

        last_dates = sorted(df.index.unique())[-3:]
        last3_values = {}
        for d in last_dates:
            row = df.loc[d].dropna()
            last3_values[str(d)] = {
                "n_stocks": len(row),
                "mean": round(float(row.mean()), 6),
                "std": round(float(row.std()), 6),
                "min": round(float(row.min()), 6),
                "max": round(float(row.max()), 6),
                "top10": {str(k): round(float(v), 6)
                          for k, v in row.nlargest(10).items()},
                "bottom10": {str(k): round(float(v), 6)
                             for k, v in row.nsmallest(10).items()},
            }

        factor_snap[fn] = {
            "md5": md5,
            "last_date": str(last_dates[-1]) if last_dates else None,
            "last3_values": last3_values,
        }
    except Exception as e:
        factor_snap[fn] = {"md5": md5, "error": str(e)}

with open(OUT / "factor_sample.json", "w") as f:
    json.dump(factor_snap, f, indent=2, ensure_ascii=False, default=str)

# ============================================================
# 6. fac_all.fea 指纹
# ============================================================
print("\n[6/9] fac_all.fea fingerprints...")
fac_path = PROJECT / "trainingdata/fac_all.fea"
fac_meta = {}
fac_last3 = {}

if fac_path.exists():
    fac_meta["size_mb"] = round(fac_path.stat().st_size / 1048576, 2)
    fac = pd.read_feather(fac_path)
    fac_meta["shape"] = list(fac.shape)
    all_factor_cols = sorted([c for c in fac.columns if c not in ("date", "Code")])
    fac_meta["factor_cols"] = len(all_factor_cols)
    fac_meta["factor_cols_list"] = all_factor_cols

    dates = sorted(fac["date"].unique())
    fac_meta["date_range"] = f"{dates[0]} ~ {dates[-1]}"
    fac_meta["unique_dates"] = len(dates)

    date_stock_counts = {}
    for d in dates[-10:]:
        date_stock_counts[str(d)] = int((fac["date"] == d).sum())
    fac_meta["last10_date_stock_counts"] = date_stock_counts

    watch_factors = [s for s in sample if s in all_factor_cols]
    fac_meta["watched_factors"] = watch_factors

    last3_dates = dates[-3:]
    for d in last3_dates:
        d_str = str(d)
        rows = fac[fac["date"] == d_str].set_index("Code")
        d_data = {}
        for wf in watch_factors:
            if wf in rows.columns:
                vals = rows[wf].dropna()
                d_data[wf] = {
                    "n": len(vals),
                    "mean": round(float(vals.mean()), 6),
                    "std": round(float(vals.std()), 6),
                }
        fac_meta[f"date_{d_str}_stats"] = d_data

    print(f"  Shape: {fac_meta['shape']}, Dates: {fac_meta['date_range']}")

    for d in last3_dates:
        d_str = str(d)
        rows = fac[fac["date"] == d_str].set_index("Code")
        d_full = {}
        for wf in watch_factors:
            if wf in rows.columns:
                vals = rows[wf].dropna()
                d_full[wf] = {str(k): round(float(v), 6)
                              for k, v in vals.items()}
        fac_last3[str(d)] = d_full
else:
    fac_meta["error"] = "fac_all.fea not found"

with open(OUT / "fac_all_meta.json", "w") as f:
    json.dump(fac_meta, f, indent=2, ensure_ascii=False, default=str)
with open(OUT / "fac_all_last3_sample.json", "w") as f:
    json.dump(fac_last3, f, indent=2, ensure_ascii=False, default=str)

# ============================================================
# 7. 模型预测快照
# ============================================================
print("\n[7/9] Model prediction snapshots...")
pred_dir = PROJECT / "Model/V1.0/model_pred"
pred_full = {}

for season_dir in sorted(pred_dir.glob("*")):
    if not season_dir.is_dir():
        continue
    season = season_dir.name
    pred_full[season] = {}

    all_pkls = sorted(season_dir.glob("*.pkl"))
    for pkl in all_pkls[-5:]:
        date_str = pkl.stem
        try:
            with open(pkl, "rb") as f:
                pred = pickle.load(f)
            if hasattr(pred, 'sort_values'):
                top10 = pred.sort_values(ascending=False).head(10)
                top20 = pred.sort_values(ascending=False).head(20)
                pred_full[season][date_str] = {
                    "n_stocks": len(pred),
                    "mean": round(float(pred.mean()), 6),
                    "std": round(float(pred.std()), 6),
                    "top10": {str(k): round(float(v), 4) for k, v in top10.items()},
                    "top20": {str(k): round(float(v), 4) for k, v in top20.items()},
                    "top50_codes": [str(k) for k in pred.nlargest(50).index],
                }
        except Exception as e:
            pred_full[season][date_str] = {"error": str(e)}

    print(f"  {season}: {len(pred_full[season])} dates captured")

with open(OUT / "model_pred_full.json", "w") as f:
    json.dump(pred_full, f, indent=2, ensure_ascii=False, default=str)

# ============================================================
# 8. 训练标签和交易额指纹
# ============================================================
print("\n[8/9] Label & trade data fingerprints...")
label_meta = {}
for label_name in ["label_ret_1d", "label_ret_3d", "label_ret_5d", "label_ret_10d"]:
    lp = PROJECT / "trainingdata" / f"{label_name}.fea"
    if lp.exists():
        df = pd.read_feather(lp)
        label_meta[label_name] = {
            "size_mb": round(lp.stat().st_size / 1048576, 2),
            "shape": list(df.shape),
        }
        if "index" in df.columns:
            label_meta[label_name]["date_range"] = f"{df['index'].min()} ~ {df['index'].max()}"

tp = PROJECT / "trainingdata" / "trade_amt.fea"
if tp.exists():
    tdf = pd.read_feather(tp)
    label_meta["trade_amt"] = {
        "size_mb": round(tp.stat().st_size / 1048576, 2),
        "shape": list(tdf.shape),
    }
    if "index" in tdf.columns:
        label_meta["trade_amt"]["date_range"] = f"{tdf['index'].min()} ~ {tdf['index'].max()}"

with open(OUT / "label_meta.json", "w") as f:
    json.dump(label_meta, f, indent=2, ensure_ascii=False, default=str)

# ============================================================
# 9. 汇总
# ============================================================
print("\n[9/9] Summary...")
code_file = PROJECT / "Code_num.txt"
if code_file.exists():
    codes = sorted(set(l.strip() for l in code_file.read_text().splitlines() if l.strip()))
    code_md5 = hashlib.md5(code_file.read_bytes()).hexdigest()
else:
    codes = []
    code_md5 = "N/A"

cal = pd.read_parquet(PROJECT / "data/calendar.parquet")
trading_days = int(cal["is_open"].sum())

summary = {
    "snapshot_date": TODAY,
    "snapshot_time": datetime.now().isoformat(),
    "snapshot_version": 3,
    "script_md5": script_md5,
    "project_root": str(PROJECT),
    "sources_tracked": sorted(source_date_hashes.keys()),
    "total_factors": len(all_factors),
    "sampled_factors": sample,
    "fac_all_shape": fac_meta.get("shape"),
    "fac_all_date_range": fac_meta.get("date_range"),
    "fac_all_factor_cols": fac_meta.get("factor_cols"),
    "stock_pool_count": len(codes),
    "stock_pool_md5": code_md5,
    "calendar_trading_days": trading_days,
    "calendar_last_date": str(cal["date"].max()),
    "finance_per_field_tracking": True,
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\n  Stock pool: {len(codes)} codes (MD5={code_md5[:12]}...)")
print(f"  Calendar: {trading_days} trading days, last={summary['calendar_last_date']}")

print(f"\n{'='*60}")
print(f"  Snapshot v3 saved to: {OUT}")
print(f"  Files: {sorted(f.name for f in OUT.iterdir())}")
print(f"  Sources tracked: {summary['sources_tracked']}")
print(f"{'='*60}")
