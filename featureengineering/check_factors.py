#!/usr/bin/env python3
"""
滚动输出：逐行打印每个因子的基本信息，一行一个因子，处理完立即输出。
每天三个指标：NaN率、当日均值、当日方差。
"""

import os
import sys
import time
import pandas as pd
import numpy as np

FACTORS_DIR = "/autodl-fs/data/lingqiData/featureengineering/data/factors"


def scan_one(filepath):
    fname = os.path.basename(filepath)
    try:
        df = pd.read_feather(filepath)
    except Exception as e:
        return {"factor": fname, "error": str(e)}

    if df.index.name != "Date":
        if "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)
            df = df.set_index("Date").sort_index()
        else:
            return {"factor": fname, "error": "no Date"}

    df = df.sort_index()
    last3 = df.index[-3:]
    row = {
        "factor": fname,
        "latest": df.index[-1],
        "n_days": len(df),
        "n_cols": len(df.columns),
    }
    for d in last3:
        r = df.loc[d]
        row[f"nan_{d}"] = r.isna().mean()
        row[f"mean_{d}"] = r.mean(skipna=True)
        row[f"var_{d}"] = r.var(skipna=True)
    return row


def fmt_row(r, date_cols):
    """格式化一行因子数据（不含换行）。"""
    if "error" in r:
        return f"{r['factor']:<40s} [错误: {r['error']}]"
    line = f"{r['factor']:<40s} {r['latest']:<10s} {r['n_days']:>8d} {r['n_cols']:>6d}"
    # NaN率 连续三列，均值 连续三列，方差 连续三列
    for d in date_cols:
        line += f" {r.get(f'nan_{d}', np.nan):>9.6f}"
    for d in date_cols:
        line += f" {r.get(f'mean_{d}', np.nan):>12.6f}"
    for d in date_cols:
        line += f" {r.get(f'var_{d}', np.nan):>12.6f}"
    return line


def main():
    files = sorted([
        f for f in os.listdir(FACTORS_DIR) if f.endswith(".fea")
    ])
    n = len(files)
    print(f"共 {n} 个因子", flush=True)

    # 先探一个文件获取日期列
    date_cols = None
    for fname in files:
        r = scan_one(os.path.join(FACTORS_DIR, fname))
        if "error" not in r:
            date_cols = sorted([k[4:] for k in r if k.startswith("nan_")])
            break

    if date_cols is None:
        print("无有效文件。")
        return

    # 日期映射行：0=最旧, 2=最新
    date_hint = "  ".join(f"{d}(_{i})" for i, d in enumerate(date_cols))
    print(f"日期索引: {date_hint}", flush=True)

    # 表头：_0 _1 _2 代替重复日期
    header = f"{'因子':<40s} {'最新日期':<10s} {'交易日数':>8s} {'列数':>6s}"
    for i in range(len(date_cols)):
        header += f"   NaN率_{i}"
    for i in range(len(date_cols)):
        header += f"    均值_{i}  "
    for i in range(len(date_cols)):
        header += f"    方差_{i}  "
    print(header)
    print("-" * len(header), flush=True)

    # 逐文件扫描，即时输出
    t0 = time.time()
    ok = err = 0
    for fname in files:
        fpath = os.path.join(FACTORS_DIR, fname)
        r = scan_one(fpath)
        if "error" in r:
            err += 1
        else:
            ok += 1
        print(fmt_row(r, date_cols), flush=True)

    elapsed = time.time() - t0
    print(f"完成: {ok} 成功, {err} 失败, 耗时 {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
