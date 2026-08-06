#!/usr/bin/env python3
"""
切片对比 v1 — 两份 debug/<日期>/data10d 快照在重叠日期上的数据差异

思路: 相邻两日切片同构, 各自是源数据的 10 交易日切片。两份切片都覆盖的重叠日期上,
同一 (日期, 代码) 行的值若不同, 说明源数据在这些行上发生了「事后改动」(回改/修订/重算)。
因子同理: 重叠日期上因子值不同 = 因子侧存在变动。

用法: python debug/compare_snapshots.py --base 20260804 --cur 20260805 --jobs 24
      --scope raw | factors | all (默认 all)
输出: debug/<cur>/compare_report_<base>_<cur>.json + .md

注: 脚本可被 root 或 claude 用户运行, 根目录自动探测 (/root/autodl-fs/lingqiData 或
/autodl-fs/data/lingqiData)。
"""

import os, sys, json, argparse, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

CANDIDATE_ROOTS = [Path("/root/autodl-fs/lingqiData"), Path("/autodl-fs/data/lingqiData")]


def _probe(p):
    try:
        return p.exists()
    except OSError:
        return False


PROJECT = next((p for p in CANDIDATE_ROOTS if _probe(p)), CANDIDATE_ROOTS[0])
DEBUG_DIR = PROJECT / "debug"

RTOL, ATOL = 1e-8, 1e-12  # 数值比较容差


def norm_ymd(v):
    """任意日期表示 → YYYYMMDD"""
    return str(v)[:10].replace("-", "")


# ============================================================
# 单元格比较
# ============================================================
def cell_diff_mask(a: pd.Series, b: pd.Series) -> np.ndarray:
    """逐单元格比较, 返回 True=有差异的布尔数组 (NaN 视为相等, 数值用容差)"""
    a = pd.Series(a).reset_index(drop=True)
    b = pd.Series(b).reset_index(drop=True)
    ak = a.astype("float64") if a.dtype.kind in "fc" else None
    bk = b.astype("float64") if b.dtype.kind in "fc" else None
    if ak is not None and bk is not None:
        # np.isclose 返回 ndarray, 不要对结果再调 to_numpy
        return ~np.isclose(ak, bk, rtol=RTOL, atol=ATOL, equal_nan=True)
    neq = (a != b)
    both_null = a.isna() & b.isna()
    return (neq & ~both_null).to_numpy()


def max_abs_diff(a: pd.Series, b: pd.Series, mask: np.ndarray):
    try:
        va = a.to_numpy(dtype="float64")[mask]
        vb = b.to_numpy(dtype="float64")[mask]
        va = np.nan_to_num(va, nan=0.0); vb = np.nan_to_num(vb, nan=0.0)
        return float(np.max(np.abs(va - vb))) if len(va) else 0.0
    except Exception:
        return 0.0


# ============================================================
# Worker: 长表单文件 (trade_date 列 + 代码列)
# ============================================================
def cmp_long(args):
    name, bp, cp, date_col, key_cols = args
    try:
        a = pd.read_parquet(bp)
        b = pd.read_parquet(cp)
        a["_d"] = a[date_col].map(norm_ymd)
        b["_d"] = b[date_col].map(norm_ymd)
        da, db = set(a["_d"]), set(b["_d"])
        overlap = sorted(da & db)
        if not overlap:
            return name, {"changed": False, "note": f"无重叠日期 (base {len(da)} 日 / cur {len(db)} 日)"}
        a = a[a["_d"].isin(overlap)].drop(columns=["_d"])
        b = b[b["_d"].isin(overlap)].drop(columns=["_d"])
        if list(a.columns) != list(b.columns):
            return name, {"changed": True, "schema": True,
                          "note": f"列集不同: base={list(a.columns)} cur={list(b.columns)}"}
        keys = key_cols + [date_col]
        a = a.sort_values(keys).reset_index(drop=True)
        b = b.sort_values(keys).reset_index(drop=True)
        ka = a[keys].astype(str).agg("|".join, axis=1).tolist()
        kb = b[keys].astype(str).agg("|".join, axis=1).tolist()
        only_a = len(set(ka) - set(kb)); only_b = len(set(kb) - set(ka))
        res = {"changed": False, "n_rows": len(a), "overlap_days": len(overlap),
               "added_rows": only_b, "deleted_rows": only_a}
        if a.shape == b.shape and list(a.columns) == list(b.columns):
            n_cells = 0
            top_cols = {}
            maxd = 0.0
            changed_row = np.zeros(len(a), dtype=bool)
            for c in a.columns:
                if c in keys:
                    continue
                mask = cell_diff_mask(a[c], b[c])
                if mask.any():
                    n = int(mask.sum())
                    n_cells += n
                    top_cols[c] = n
                    changed_row |= mask
                    maxd = max(maxd, max_abs_diff(a[c], b[c], mask))
            n_rows = int(changed_row.sum())
            if n_rows:
                by_date = a.loc[changed_row, date_col].map(norm_ymd).value_counts().to_dict()
                by_date = {str(k): int(v) for k, v in sorted(by_date.items())}
                top_cols = dict(sorted(top_cols.items(), key=lambda kv: -kv[1])[:10])
                samples = a.loc[changed_row, keys].astype(str).head(20).values.tolist()
                res.update({"changed": True, "n_changed_cells": n_cells, "n_changed_rows": n_rows,
                            "by_date": by_date, "top_cols": top_cols, "max_abs_diff": maxd,
                            "samples": samples})
        return name, res
    except Exception as e:
        return name, {"changed": None, "error": str(e)}


# ============================================================
# Worker: 静态表 (整体拷贝, 无日期过滤)
# ============================================================
def cmp_static(args):
    name, bp, cp = args
    try:
        a = pd.read_parquet(bp)
        b = pd.read_parquet(cp)
        if list(a.columns) != list(b.columns):
            return name, {"changed": True, "schema": True,
                          "note": f"列集不同: base={list(a.columns)} cur={list(b.columns)}"}
        if a.shape != b.shape:
            return name, {"changed": True, "note": f"行数不同: base={len(a)} cur={len(b)}"}
        a = a.sort_values(a.columns.tolist()).reset_index(drop=True)
        b = b.sort_values(b.columns.tolist()).reset_index(drop=True)
        n_cells = 0
        top_cols = {}
        for c in a.columns:
            m = cell_diff_mask(a[c], b[c])
            if m.any():
                n_cells += int(m.sum())
                top_cols[c] = int(m.sum())
        if n_cells:
            return name, {"changed": True, "n_changed_cells": n_cells, "top_cols": top_cols}
        return name, {"changed": False, "n_rows": len(a)}
    except Exception as e:
        return name, {"changed": None, "error": str(e)}


# ============================================================
# Worker: 按股票存储 (cyq_chips / history_1min / indicator_1min)
# ============================================================
def cmp_perstock(args):
    name, bp, cp, date_col, key_cols = args
    r = cmp_long((name, bp, cp, date_col, key_cols))[1]
    return name, r


# ============================================================
# Worker: 因子宽表 .fea (index=Date, 列=股票)
# ============================================================
def cmp_wide_fea(args):
    name, bp, cp = args
    try:
        ta = ipc.open_file(bp).read_all(); tb = ipc.open_file(cp).read_all()
        a = ta.to_pandas(); b = tb.to_pandas()
        da = (a["index"] if "index" in a.columns else pd.Series(a.index, index=a.index)).map(norm_ymd)
        db = (b["index"] if "index" in b.columns else pd.Series(b.index, index=b.index)).map(norm_ymd)
        overlap = sorted(set(da) & set(db))
        if not overlap:
            return name, {"changed": False, "note": "无重叠日期"}
        a = a[da.isin(overlap)].reset_index(drop=True)
        b = b[db.isin(overlap)].reset_index(drop=True)
        da = da[da.isin(overlap)].reset_index(drop=True)
        db = db[db.isin(overlap)].reset_index(drop=True)
        # 列集变化 (股票进出)
        ca, cb = set(a.columns), set(b.columns)
        added_cols = sorted(cb - ca); removed_cols = sorted(ca - cb)
        common = sorted(ca & cb)
        if "index" in common:
            common.remove("index")
        n_cells = 0; by_date = {}; top_stocks = {}; maxd = 0.0
        for c in common:
            m = cell_diff_mask(a[c], b[c])
            if m.any():
                n = int(m.sum())
                n_cells += n
                top_stocks[c] = n
                maxd = max(maxd, max_abs_diff(a[c], b[c], m))
        if n_cells:
            diff_df = pd.DataFrame({c: cell_diff_mask(a[c], b[c]) for c in common},
                                   index=da.astype(str))
            by_date = {str(k): int(v) for k, v in diff_df.sum(axis=1).groupby(level=0).sum().items()}
            top_stocks = dict(sorted(top_stocks.items(), key=lambda kv: -kv[1])[:10])
            return name, {"changed": True, "n_changed_cells": n_cells,
                          "by_date": by_date, "top_stocks": top_stocks, "max_abs_diff": maxd,
                          "added_cols": added_cols[:10], "n_added_cols": len(added_cols),
                          "removed_cols": removed_cols[:10], "n_removed_cols": len(removed_cols)}
        if added_cols or removed_cols:
            return name, {"changed": False, "note": f"列集变化但重叠值相同: +{len(added_cols)} -{len(removed_cols)}"}
        return name, {"changed": False, "n_days": len(overlap)}
    except Exception as e:
        return name, {"changed": None, "error": str(e)}


# ============================================================
# Worker: daily_dump_1min 字节比较
# ============================================================
def cmp_bytes(args):
    name, bp, cp = args
    try:
        with open(bp, "rb") as f: hb = hashlib.md5(f.read()).hexdigest()
        with open(cp, "rb") as f: hc = hashlib.md5(f.read()).hexdigest()
        return name, {"changed": hb != hc, "base_md5": hb[:12], "cur_md5": hc[:12]}
    except Exception as e:
        return name, {"changed": None, "error": str(e)}


def run_pool(jobs, tasks, label):
    total = len(tasks)
    results = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(fn, a) for fn, a in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            name, r = f.result()
            results[name] = r
            if i % 500 == 0 or i == total:
                print(f"  [{label}] {i}/{total}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基准快照日期 YYYYMMDD")
    ap.add_argument("--cur", required=True, help="对比快照日期 YYYYMMDD")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--scope", default="all", choices=["raw", "factors", "all"])
    args = ap.parse_args()

    base = DEBUG_DIR / args.base / "data10d"
    cur = DEBUG_DIR / args.cur / "data10d"
    out_dir = DEBUG_DIR / args.cur
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"base={base}\ncur={cur}\nout={out_dir}", flush=True)

    # 重叠窗口
    wb = json.load(open(base / "summary.json"))["window"]
    wc = json.load(open(cur / "summary.json"))["window"]
    overlap = sorted(set(wb) & set(wc))
    print(f"window base={wb[0]}~{wb[-1]} cur={wc[0]}~{wc[-1]} overlap={overlap}", flush=True)

    report = {"base": args.base, "cur": args.cur,
              "window_base": wb, "window_cur": wc, "overlap_days": overlap,
              "created": pd.Timestamp.now().isoformat()}

    if args.scope in ("raw", "all"):
        report["raw"] = compare_raw(base, cur, args.jobs)
    if args.scope in ("factors", "all"):
        report["factors"] = compare_factors(base, cur, args.jobs)

    with open(out_dir / f"compare_report_{args.base}_{args.cur}.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md = render_md(report)
    with open(out_dir / f"compare_report_{args.base}_{args.cur}.md", "w") as f:
        f.write(md)
    print(md)


# ============================================================
# raw 对比
# ============================================================
def compare_raw(base, cur, jobs):
    raw_b, raw_c = base / "raw", cur / "raw"
    out = {}
    long_singles = {
        "daily.parquet": ("trade_date", ["stock_code"]),
        "daily_adj.parquet": ("trade_date", ["stock_code"]),
        "finance.parquet": ("trade_date", ["stock_code"]),
        "cyq_perf.parquet": ("trade_date", ["stock_code"]),
        "main_fund_flow.parquet": ("trade_date", ["stock_code"]),
        "margin_detail.parquet": ("trade_date", ["stock_code"]),
        "ths_daily.parquet": ("trade_date", ["ths_code"]),
    }
    static = ["stock_list.parquet", "ths_constituent_stocks.parquet",
              "ths_sector_categories.parquet", "calendar.parquet"]
    tasks = []
    for fn, (dc, kc) in long_singles.items():
        bp, cp = raw_b / fn, raw_c / fn
        if not bp.exists() or not cp.exists():
            out[fn] = {"changed": None, "note": f"缺文件 base={bp.exists()} cur={cp.exists()}"}
            continue
        tasks.append((cmp_long, (fn, str(bp), str(cp), dc, kc)))
    for fn in static:
        bp, cp = raw_b / fn, raw_c / fn
        if not bp.exists() or not cp.exists():
            out[fn] = {"changed": None, "note": f"缺文件 base={bp.exists()} cur={cp.exists()}"}
            continue
        tasks.append((cmp_static, (fn, str(bp), str(cp))))

    # per-stock 三目录 (共用抽样清单, key 加目录前缀避免同名覆盖)
    # cyq_chips 的 key 含 price: 同日内行序可能因上游重写而变化, 按价格网格对齐后再
    # 比较, 避免「行序重排」被误报为「数据变动」(2026-08-05 实测: 08-03/08-04 行序
    # 全变但数值 0 差异)。
    sample = json.load(open(base / "sample_1000.json"))["stocks"]
    perstock = {
        "cyq_chips": ("trade_date", ["stock_code", "price"]),
        "history_1min": ("trade_time", ["stock_code", "trade_time"]),
        "indicator_1min": ("trade_time", ["stock_code", "trade_time"]),
    }
    for d, (dc, kc) in perstock.items():
        tasks += [(cmp_perstock, (f"{d}/{s}", str(raw_b / d / s), str(raw_c / d / s), dc, kc)) for s in sample]

    # daily_dump_1min 字节比较 (重叠日期文件)
    dump_b = {p.name: p for p in (raw_b / "daily_dump_1min").glob("*.parquet")}
    dump_c = {p.name: p for p in (raw_c / "daily_dump_1min").glob("*.parquet")}
    dump_names = sorted(set(dump_b) & set(dump_c))
    for n in dump_names:
        tasks.append((cmp_bytes, (f"daily_dump_1min/{n}", str(dump_b[n]), str(dump_c[n]))))

    results = run_pool(jobs, tasks, "raw")
    out.update(results)

    # 按目录聚合 per-stock 明细
    for d in perstock:
        n_changed = 0; n_err = 0; details = []
        for s in sample:
            r = out.get(f"{d}/{s}")
            if not isinstance(r, dict):
                continue
            if r.get("changed"):
                n_changed += 1
                details.append({"file": s, **{k: v for k, v in r.items() if k in (
                    "n_changed_cells", "n_changed_rows", "by_date", "top_cols", "max_abs_diff", "samples", "note", "error")}})
            elif r.get("changed") is None:
                n_err += 1
        out[f"__summary_{d}"] = {"n_files": len(sample), "n_changed": n_changed,
                                 "n_err": n_err,
                                 "err_example": next((r.get("error") for s in sample
                                                      if isinstance(out.get(f"{d}/{s}"), dict)
                                                      and out.get(f"{d}/{s}").get("changed") is None), None),
                                 "changed_files": details}
        for s in sample:
            out.pop(f"{d}/{s}", None)  # 明细并入目录汇总, 避免 3000 条噪音
    out["daily_dump_1min"] = {"n_files": len(dump_names),
                              "changed_files": [k.split("/")[-1] for k, v in out.items()
                                                if isinstance(v, dict) and v.get("changed") and k.startswith("daily_dump_1min/")]}
    for k in [f"daily_dump_1min/{n}" for n in dump_names]:
        out.pop(k, None)
    return out


# ============================================================
# factors 对比
# ============================================================
def compare_factors(base, cur, jobs):
    fb, fc = base / "factors", cur / "factors"
    fbs = sorted(p.name for p in fb.glob("*.fea"))
    fcs = sorted(p.name for p in fc.glob("*.fea"))
    missing = sorted(set(fbs) - set(fcs))
    added = sorted(set(fcs) - set(fbs))
    tasks = [(cmp_wide_fea, (n, str(fb / n), str(fc / n))) for n in fbs if n in fcs]
    results = run_pool(jobs, tasks, "factors")
    changed = {k: v for k, v in results.items() if isinstance(v, dict) and v.get("changed")}
    errs = {k: v for k, v in results.items() if isinstance(v, dict) and v.get("changed") is None}
    return {
        "n_files": len(fbs), "n_missing_in_cur": len(missing), "missing_in_cur": missing,
        "n_added_in_cur": len(added), "added_in_cur": added,
        "n_changed": len(changed),
        "changed": [{"name": k, **v} for k, v in sorted(changed.items())],
        "n_err": len(errs), "errs": errs,
    }


# ============================================================
# markdown 报告
# ============================================================
def render_md(r):
    L = []
    L.append(f"# 切片对比报告 {r['base']} vs {r['cur']}")
    L.append("")
    L.append(f"- base 窗口: {r['window_base'][0]} ~ {r['window_base'][-1]}")
    L.append(f"- cur 窗口: {r['window_cur'][0]} ~ {r['window_cur'][-1]}")
    L.append(f"- **重叠交易日 ({len(r['overlap_days'])}):** {' '.join(r['overlap_days'])}")
    L.append("")
    raw = r.get("raw", {})
    L.append("## 一、raw 原始字段")
    L.append("")
    L.append("### 长表单文件")
    L.append("")
    L.append("| 文件 | 状态 | 变更行 | 变更格 | 按日期分布 | 主要列 | 最大绝对差 |")
    L.append("|------|------|--------|--------|-----------|--------|-----------|")
    for fn in ["daily.parquet", "daily_adj.parquet", "finance.parquet", "cyq_perf.parquet",
               "main_fund_flow.parquet", "margin_detail.parquet", "ths_daily.parquet"]:
        v = raw.get(fn, {})
        if not isinstance(v, dict):
            L.append(f"| {fn} | {v} | | | | | |"); continue
        st = "❌ 变动" if v.get("changed") is True else ("✅ 无变动" if v.get("changed") is False else "⚠️ 错误")
        bd = json.dumps(v.get("by_date", {}), ensure_ascii=False)
        tc = ",".join(v.get("top_cols", {}))[:80]
        L.append(f"| {fn} | {st} | {v.get('n_changed_rows', '-')} | {v.get('n_changed_cells', '-')} | {bd} | {tc} | {v.get('max_abs_diff', '-')} |")
        if v.get("samples"):
            L.append(f"  变更样本: {v['samples'][:10]}")
    L.append("")
    L.append("### 静态表 (整体拷贝)")
    L.append("")
    for fn in ["stock_list.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet", "calendar.parquet"]:
        v = raw.get(fn, {})
        if not isinstance(v, dict):
            L.append(f"- {fn}: {v}")
            continue
        st = "❌ 变动" if v.get("changed") is True else ("✅ 无变动" if v.get("changed") is False else "⚠️ 错误")
        L.append(f"- {fn}: {st}" + (f" ({v.get('n_changed_cells')} 格, {v.get('top_cols')})" if v.get("changed") else "") + (f" [{v.get('error')}]" if v.get("error") else ""))
    L.append("")
    L.append("### 按股票存储目录 (抽样 1000 只)")
    L.append("")
    for d in ["cyq_chips", "history_1min", "indicator_1min"]:
        v = raw.get(f"__summary_{d}", {})
        n_ch = v.get("n_changed", 0)
        L.append(f"- **{d}**: {v.get('n_files', 0)} 文件, **{n_ch} 个变动**" + (" ✅" if not n_ch else ""))
        for det in v.get("changed_files", [])[:20]:
            L.append(f"  - `{det['file']}`: {det.get('n_changed_cells', '?')} 格, 日期 {det.get('by_date', {})}, max|diff|={det.get('max_abs_diff', '-')}")
    L.append("")
    v = raw.get("daily_dump_1min", {})
    L.append(f"### daily_dump_1min: {v.get('n_files', '-')} 个重叠日期文件, **{len(v.get('changed_files', []))} 个变动**")
    L.append("")
    fac = r.get("factors", {})
    L.append("## 二、factors 因子")
    L.append("")
    L.append(f"- 因子总数: {fac.get('n_files', '-')} | 变动: **{fac.get('n_changed', '-')}** | "
             f"cur 缺失: {fac.get('missing_in_cur', [])} | cur 新增: {fac.get('added_in_cur', [])}")
    L.append("")
    if fac.get("changed"):
        L.append("### 变动因子明细 (重叠日期)")
        L.append("")
        L.append("| 因子 | 变更格 | 按日期分布 | 主要股票列 | 最大绝对差 |")
        L.append("|------|--------|-----------|-----------|-----------|")
        for c in fac["changed"]:
            bd = json.dumps(c.get("by_date", {}), ensure_ascii=False)
            ts = ",".join(c.get("top_stocks", {}))[:80]
            L.append(f"| {c['name']} | {c.get('n_changed_cells')} | {bd} | {ts} | {c.get('max_abs_diff')} |")
    else:
        L.append("所有因子在重叠日期上无变动 ✅")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
