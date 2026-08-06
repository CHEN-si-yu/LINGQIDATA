#!/usr/bin/env python3
"""Repro: chip factor on 0804 — 10-day window vs full history vs both snapshots' .fea.

Question: does the factor value for 2026-08-04 differ depending on how much
history the builder loaded? And which snapshot .fea does each computation match?
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

ROOT = Path("/autodl-fs/data/lingqiData")
SNAP4 = ROOT / "debug" / "20260804" / "data10d"
SNAP5 = ROOT / "debug" / "20260805" / "data10d"
PROD = ROOT / "data"  # current production raw

DATE = "20260804"


def norm(df):
    df = df.copy()
    if "stock_code" in df.columns:
        df["stock_code"] = (
            df["stock_code"].astype(str).str.replace("-", "").str.split(".").str[0].str.zfill(6)
        )
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    return df


def load_fea(path):
    return ipc.open_file(str(path)).read_all().to_pandas()


def close_adj_series(daily: pd.DataFrame) -> pd.Series:
    """Mirror chip._close_adj_basis exactly: adj=cumprod(1+pct_chg/100), K anchored at last date."""
    d = norm(daily)
    parts = []
    for code, grp in d.groupby("stock_code"):
        grp = grp.sort_values("trade_date")
        raw = grp["close"].replace(0, np.nan)
        adj = (1.0 + grp["pct_chg"] / 100.0).cumprod()
        last_valid = raw.last_valid_index()
        k = raw.loc[last_valid] / adj.loc[last_valid] if last_valid is not None else np.nan
        s = pd.Series(adj * k, index=grp["trade_date"].values, name=code)
        s.index.name = "Date"
        parts.append(s)
    out = pd.concat(parts, axis=1)
    return out.stack().rename("close_adj").reset_index().rename(columns={0: "Date"}) if False else \
        pd.concat({c: s for c, s in zip(
            [g[0] for g in d.groupby("stock_code")],  # placeholder
            [],
        )}) if False else None


def close_adj_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-stock Series in a DataFrame indexed by date (YYYYMMDD), columns = codes."""
    d = norm(daily)
    cols = {}
    for code, grp in d.groupby("stock_code"):
        grp = grp.sort_values("trade_date")
        # build from .values arrays — constructing from a Series would re-align
        # on the (integer) index and silently produce NaN (mimics _build_close_maps)
        raw = pd.Series(grp["close"].replace(0, np.nan).values, index=grp["trade_date"].values)
        adj = pd.Series(
            (1.0 + grp["pct_chg"].values / 100.0).cumprod(),
            index=grp["trade_date"].values,
        )
        last_valid = raw.last_valid_index()
        k = raw.loc[last_valid] / adj.loc[last_valid] if last_valid is not None else np.nan
        cols[code] = adj * k
    return pd.DataFrame(cols)


def raw_ratio(daily, cyq, date):
    """chip_above_below_ratio raw: (close_adj-cost85)/(cost15-close_adj) on `date`."""
    cam = close_adj_frame(daily)
    c = norm(cyq)
    c = c[c["trade_date"] == date].set_index("stock_code")
    rows = {}
    for code in cam.columns:
        ca = cam.loc[date, code] if date in cam.index else np.nan
        if code not in c.index or not np.isfinite(ca):
            continue
        cost85 = c.loc[code, "cost_85pct"]
        cost15 = c.loc[code, "cost_15pct"]
        if not np.isfinite(cost85) or not np.isfinite(cost15):
            continue
        above = ca - cost85
        below = cost15 - ca
        rows[code] = above / below if below != 0 else np.nan
    s = pd.Series(rows, name="ratio")
    return s


def xrank(s: pd.Series) -> pd.Series:
    """Mirror cross_sectional_rank (single date): winsorize 1%/99% then pct rank."""
    s = s.replace([np.inf, -np.inf], np.nan).astype(float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lower=lo, upper=hi)
    return s.rank(pct=True)


def compare_rank(name, computed, fea_row):
    interp = computed.index.intersection(fea_row.index)
    a = computed[interp]
    b = fea_row[interp].astype(float)
    ok = a.notna() & b.notna()
    corr = a[ok].corr(b[ok])
    mad = (a[ok] - b[ok]).abs().mean()
    n = int(ok.sum())
    return corr, mad, n


def main():
    # ── inputs ────────────────────────────────────────────────
    d4 = pd.read_parquet(SNAP4 / "raw" / "daily.parquet")
    d5 = pd.read_parquet(SNAP5 / "raw" / "daily.parquet")
    dp = pd.read_parquet(PROD / "daily.parquet")
    cyq4 = pd.read_parquet(SNAP4 / "raw" / "cyq_perf.parquet")
    cyq5 = pd.read_parquet(SNAP5 / "raw" / "cyq_perf.parquet")
    cyqp = pd.read_parquet(PROD / "cyq_perf.parquet")

    # sanity: is close_adj(0804) == close(0804) as the anchor property predicts?
    cam_full = close_adj_frame(dp)
    d_full = norm(dp)
    d0804 = d_full[d_full["trade_date"] == DATE].set_index("stock_code")
    for code in ["000001", "600266", "600635", "600993"]:
        if code in cam_full.columns and code in d0804.index:
            print(f"anchor check {code}: close_adj(0804)={cam_full.loc[DATE, code]:.4f} "
                  f"close(0804)={d0804.loc[code,'close']:.4f}")

    # ── .fea rows ─────────────────────────────────────────────
    fea4 = load_fea(SNAP4 / "factors" / "chip_above_below_ratio.fea")
    fea5 = load_fea(SNAP5 / "factors" / "chip_above_below_ratio.fea")
    row4 = fea4.loc[DATE]
    row5 = fea5.loc[DATE]
    diff = (row4 != row5).sum()
    print(f"\n.fea 0804 row: {diff}/{len(row4)} stocks differ snap4 vs snap5")
    print(f"  snap4 0804 mean={row4.mean():.4f} snap5 mean={row5.mean():.4f}")

    # ── recompute ratio + rank from each window ───────────────
    r4 = raw_ratio(d4, cyq4, DATE)   # 10-day window ending 0804
    r5 = raw_ratio(d5, cyq5, DATE)   # 10-day window ending 0805
    rp = raw_ratio(dp, cyqp, DATE)   # full history (current)
    print(f"\nraw ratio(0804) stocks: snap4={len(r4)} snap5={len(r5)} prod={len(rp)}")

    common = r4.index.intersection(r5.index).intersection(rp.index)
    eq45 = np.allclose(r4[common], r5[common], equal_nan=True)
    eq4p = np.allclose(r4[common], rp[common], equal_nan=True)
    eq5p = np.allclose(r5[common], rp[common], equal_nan=True)
    print(f"ratio equality (n={len(common)}):")
    print(f"  snap4-window == snap5-window : {eq45}")
    print(f"  snap4-window == full-history : {eq4p}")
    print(f"  snap5-window == full-history : {eq5p}")
    if not eq45:
        d = (r4[common] - r5[common]).abs().dropna()
        print(f"  max|Δ| snap4 vs snap5 = {d.max():.6g}; mean {d.mean():.6g}")
    if not eq4p:
        d = (r4[common] - rp[common]).abs().dropna()
        print(f"  max|Δ| snap4-window vs full = {d.max():.6g}; mean {d.mean():.6g}")
    if not eq5p:
        d = (r5[common] - rp[common]).abs().dropna()
        print(f"  max|Δ| snap5-window vs full = {d.max():.6g}; mean {d.mean():.6g}")

    # ── rank reconstruction vs .fea ───────────────────────────
    rank4 = xrank(r4)
    rank5 = xrank(r5)
    rankp = xrank(rp)
    print("\nrank(0804) vs snapshot .fea (corr / mean|Δ| / n):")
    for tag, rk, fea in [
        ("snap4-window", rank4, row4), ("snap4-window", rank4, row5),
        ("snap5-window", rank5, row4), ("snap5-window", rank5, row5),
        ("full-history ", rankp, row4), ("full-history ", rankp, row5),
    ]:
        corr, mad, n = compare_rank(tag, rk, fea)
        print(f"  {tag} vs .fea: corr={corr:.4f} mad={mad:.5f} n={n}")

    # ── per-stock diagnostics for a few changed stocks ────────
    chg = row4.index[row4 != row5]
    print(f"\nstocks whose 0804 rank changed: {len(chg)}")
    c4 = norm(cyq4); c5 = norm(cyq5)
    c4d = c4[c4["trade_date"] == DATE].set_index("stock_code")
    c5d = c5[c5["trade_date"] == DATE].set_index("stock_code")
    d4s = norm(d4); d5s = norm(d5)
    d4d = d4s[d4s["trade_date"] == DATE].set_index("stock_code")
    d5d = d5s[d5s["trade_date"] == DATE].set_index("stock_code")
    print(f"{'code':<8}{'rank4':>8}{'rank5':>8}  {'close4':>8}{'close5':>8}  {'c85_4':>7}{'c85_5':>7}  {'c15_4':>7}{'c15_5':>7}")
    for code in chg[:12]:
        r4v = row4[code]; r5v = row5[code]
        cl4 = d4d.loc[code, "close"] if code in d4d.index else np.nan
        cl5 = d5d.loc[code, "close"] if code in d5d.index else np.nan
        c854 = c4d.loc[code, "cost_85pct"] if code in c4d.index else np.nan
        c855 = c5d.loc[code, "cost_85pct"] if code in c5d.index else np.nan
        c154 = c4d.loc[code, "cost_15pct"] if code in c4d.index else np.nan
        c155 = c5d.loc[code, "cost_15pct"] if code in c5d.index else np.nan
        print(f"{code:<8}{r4v:>8.4f}{r5v:>8.4f}  {cl4:>8.3f}{cl5:>8.3f}  {c854:>7.2f}{c855:>7.2f}  {c154:>7.2f}{c155:>7.2f}")


if __name__ == "__main__":
    main()
