#!/usr/bin/env python3
"""
Debug 数据切片 v2 — 近 10 个交易日(截至锚点日期)的 raw + factors 数据切片拷贝

新思路: 不再只存哈希, 而是把原始字段/因子的最近 10 个交易日实际数据
完整拷一份到 debug/<锚点日期>/data10d/ 下, 便于后续对比找变动因子。

覆盖 (2026-08-06 起只保留 raw + factors, 不再切 targets/trainingdata):
  raw/        原始数据
    daily.parquet daily_adj.parquet finance.parquet cyq_perf.parquet
    main_fund_flow.parquet margin_detail.parquet ths_daily.parquet
    stock_list.parquet ths_constituent_stocks.parquet ths_sector_categories.parquet calendar.parquet
    cyq_chips/     (全部股票 1782)
    history_1min/  (全部股票, 日内 → 按日期过滤)
    indicator_1min/(全部股票, 日内 → 按日期过滤)
    daily_dump_1min/ (10 个日期文件)
  factors/     全量宽表 .fea (index=Date, 列=股票) — 取窗口内 10 行

用法: python debug/build_debug_data_10d.py --anchor 20260805 --jobs 24
      不传 --anchor 时锚点 = daily.parquet 最新交易日
      根目录自动探测 (/root/autodl-fs/lingqiData 或 /autodl-fs/data/lingqiData),
      root 或 claude 用户均可运行。
"""

import os, sys, json, random, shutil, argparse, hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

CANDIDATE_ROOTS = [Path("/root/autodl-fs/lingqiData"), Path("/autodl-fs/data/lingqiData")]


def _probe(p):
    try:
        return p.exists()
    except OSError:
        return False


PROJECT = next((p for p in CANDIDATE_ROOTS if _probe(p)), CANDIDATE_ROOTS[0])
DATA_DIR = PROJECT / "data"
FACTOR_DIR = PROJECT / "featureengineering" / "data" / "factors"
DEBUG_DIR = PROJECT / "debug"

# 按股票存储目录(cyq_chips/history_1min/indicator_1min)2026-08-05 起不再抽样:
# 因子宽表覆盖全部股票(1782 列),抽样 1000 只会让其余 ~782 只股票的原始数据
# 缺失,无法定位其差异(实测 0803/0804 的 cyq_chips 修订只发生在抽样外股票,
# 切片内无法直接验证)。改为全量拷贝。


# ============================================================
# 日期工具
# ============================================================
def norm_date_str(v):
    """任意日期表示 → YYYYMMDD"""
    return str(v)[:10].replace("-", "")


def get_trade_window(anchor):
    """calendar 中截至 anchor(含) 的最近 10 个交易日"""
    cal = pd.read_parquet(DATA_DIR / "calendar.parquet")
    open_days = sorted(cal.loc[cal["is_open"] == 1, "date"].astype(str).str[:10].str.replace("-", "").unique())
    anchor = anchor or None
    if anchor is None:
        d = pd.read_parquet(DATA_DIR / "daily.parquet", columns=["trade_date"])
        anchor = norm_date_str(d["trade_date"].max())
    idx = open_days.index(anchor)
    return open_days[max(0, idx - 9): idx + 1]


def get_stock_sample():
    """返回按股票存储目录中的全部股票(不再抽样), 三个目录共用同一清单"""
    dirs = [DATA_DIR / "cyq_chips", DATA_DIR / "history_1min", DATA_DIR / "indicator_1min"]
    all_stocks = set()
    for d in dirs:
        all_stocks.update(f.name for f in d.glob("*.parquet"))
    return sorted(all_stocks)


# ============================================================
# Worker: 长表单文件 (有 trade_date 列)
# ============================================================
def slice_long_single(args):
    name, src, dst, date_col, window = args
    try:
        df = pd.read_parquet(src)
        if date_col:
            mask = df[date_col].astype(str).str[:10].str.replace("-", "").isin(window)
            df = df[mask]
        df.to_parquet(dst, index=False)
        return name, "ok", len(df)
    except Exception as e:
        return name, f"ERR {e}", 0


# ============================================================
# Worker: 按股票存储的长表 (cyq_chips)
# ============================================================
def slice_perstock_long(args):
    name, src, dst, window = args
    try:
        df = pd.read_parquet(src)
        mask = df["trade_date"].astype(str).str[:10].str.replace("-", "").isin(window)
        df = df[mask]
        df.to_parquet(dst, index=False)
        return name, "ok", len(df)
    except Exception as e:
        return name, f"ERR {e}", 0


# ============================================================
# Worker: 按股票存储的日内数据 (history_1min / indicator_1min, trade_time)
# ============================================================
def slice_perstock_intraday(args):
    name, src, dst, time_col, window = args
    try:
        df = pd.read_parquet(src)
        mask = df[time_col].dt.strftime("%Y%m%d").isin(window)
        df = df[mask]
        df.to_parquet(dst, index=False)
        return name, "ok", len(df)
    except Exception as e:
        return name, f"ERR {e}", 0


# ============================================================
# Worker: 宽表 .fea (index=Date 或 'index' 列, 行=日期)
# ============================================================
def slice_wide_fea(args):
    name, src, dst, window = args
    try:
        tbl = ipc.open_file(src).read_all()
        df = tbl.to_pandas()
        if "index" in df.columns:
            dates = df["index"]
        else:
            dates = pd.Series(df.index, index=df.index)
        mask = dates.astype(str).str[:10].str.replace("-", "").isin(window).values
        out = df[mask]
        out_tbl = pa.Table.from_pandas(out, preserve_index=True)
        with pa.OSFile(dst, "wb") as w:
            writer = pa.ipc.new_file(w, schema=out_tbl.schema)
            writer.write_table(out_tbl)
            writer.close()
        return name, "ok", len(out)
    except Exception as e:
        return name, f"ERR {e}", 0


def run_pool(jobs, tasks, label):
    total = len(tasks)
    n_ok = n_err = 0
    rows = 0
    errors = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(fn, a) for fn, a in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                name, status, n = f.result()
            except Exception as e:
                status, name, n = f"EXC {e}", "?", 0
            if status == "ok":
                n_ok += 1
                rows += n
            else:
                n_err += 1
                errors.append((name, status))
            if i % 100 == 0 or i == total:
                print(f"  [{label}] {i}/{total}  ok={n_ok} err={n_err}", flush=True)
    print(f"[{label}] done: {n_ok} ok, {n_err} err, {rows} rows total", flush=True)
    for e in errors[:20]:
        print(f"  ERR: {e}", flush=True)
    return n_ok, n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=None, help="锚点日期 YYYYMMDD, 默认取 daily.parquet 最新日")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--out", default=None, help="输出目录, 默认 debug/<anchor>/data10d")
    args = ap.parse_args()

    window = get_trade_window(args.anchor)
    anchor = window[-1]
    print(f"anchor={anchor} window={window[0]}~{window[-1]} ({len(window)} 交易日)", flush=True)

    stock_sample = get_stock_sample()
    print(f"per-stock dirs: {len(stock_sample)} stocks (全量, 不再抽样), e.g. {stock_sample[:3]}", flush=True)

    out_root = Path(args.out) if args.out else DEBUG_DIR / anchor / "data10d"
    raw_dir = out_root / "raw"
    (raw_dir / "cyq_chips").mkdir(parents=True, exist_ok=True)
    (raw_dir / "history_1min").mkdir(parents=True, exist_ok=True)
    (raw_dir / "indicator_1min").mkdir(parents=True, exist_ok=True)
    (raw_dir / "daily_dump_1min").mkdir(parents=True, exist_ok=True)
    fac_dir = out_root / "factors"; fac_dir.mkdir(parents=True, exist_ok=True)
    print(f"out: {out_root}", flush=True)

    # 保存股票清单(文件名沿用 sample_1000.json 以兼容 compare_snapshots.py 等,
    # 内容为全量股票)
    with open(out_root / "sample_1000.json", "w") as f:
        json.dump({"seed": None, "n": len(stock_sample), "stocks": stock_sample,
                   "note": "全量股票(2026-08-05 起不再抽样)"}, f, ensure_ascii=False)

    tasks = []
    wset = set(window)

    # 1) 长表单文件 (切片)
    long_singles = {
        "daily.parquet": "trade_date",
        "daily_adj.parquet": "trade_date",
        "finance.parquet": "trade_date",
        "cyq_perf.parquet": "trade_date",
        "main_fund_flow.parquet": "trade_date",
        "margin_detail.parquet": "trade_date",
        "ths_daily.parquet": "trade_date",
    }
    for fn, dcol in long_singles.items():
        tasks.append((slice_long_single, (fn, DATA_DIR / fn, raw_dir / fn, dcol, wset)))

    # 2) 静态表 (整体拷贝)
    for fn in ["stock_list.parquet", "ths_constituent_stocks.parquet",
               "ths_sector_categories.parquet", "calendar.parquet"]:
        tasks.append((slice_long_single, (fn, DATA_DIR / fn, raw_dir / fn, None, None)))

    # 3) daily_dump_1min: 窗口内日期文件直接拷贝
    n_dump = 0
    for d in window:
        src = DATA_DIR / "daily_dump_1min" / f"{d[:4]}-{d[4:6]}-{d[6:8]}.parquet"
        if src.exists():
            shutil.copy2(src, raw_dir / "daily_dump_1min" / src.name)
            n_dump += 1
        else:
            print(f"  ! daily_dump_1min 缺 {src.name}", flush=True)
    print(f"daily_dump_1min: copied {n_dump}/{len(window)}", flush=True)

    # 4) per-stock: cyq_chips (长表) + history_1min/indicator_1min (日内)
    for s in stock_sample:
        tasks.append((slice_perstock_long,
                      (s, DATA_DIR / "cyq_chips" / s, raw_dir / "cyq_chips" / s, wset)))
    for s in stock_sample:
        tasks.append((slice_perstock_intraday,
                      (s, DATA_DIR / "history_1min" / s, raw_dir / "history_1min" / s, "trade_time", wset)))
    for s in stock_sample:
        tasks.append((slice_perstock_intraday,
                      (s, DATA_DIR / "indicator_1min" / s, raw_dir / "indicator_1min" / s, "trade_time", wset)))

    # 5) 因子宽表 fea
    fac_files = sorted(FACTOR_DIR.glob("*.fea"))
    for f in fac_files:
        tasks.append((slice_wide_fea, (f.name, str(f), str(fac_dir / f.name), wset)))

    print(f"total tasks: {len(tasks)} (factors={len(fac_files)})", flush=True)
    n_ok, n_err = run_pool(args.jobs, tasks, "all")

    # 汇总
    with open(__file__, "rb") as f:
        script_md5 = hashlib.md5(f.read()).hexdigest()
    summary = {
        "tool": "build_debug_data_10d",
        "snapshot_date": anchor,
        "snapshot_time": datetime.now().isoformat(),
        "window_days": len(window),
        "date_range": f"{window[0]} ~ {window[-1]}",
        "window": window,
        "stock_sample": {"seed": None, "n": len(stock_sample), "full_universe": True},
        "n_factor_files": len(fac_files),
        "tasks_total": len(tasks), "tasks_ok": n_ok, "tasks_err": n_err,
        "script_md5": script_md5,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("summary:", json.dumps(summary, ensure_ascii=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
