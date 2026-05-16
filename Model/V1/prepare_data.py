"""
Prepare training data for V1 model — with strict point-in-time alignment.

Principle: at training date T, each factor's value must be the latest value
available at market close of T. If a factor's underlying data source hasn't
published T's value yet, the factor value is forward-filled from T-1.

Usage:
    python prepare_data.py
    python prepare_data.py --target label_ret_1d
    python prepare_data.py --target label_ret_10d --stock-pool main_board_pre2020.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Prepare V1 training data")
    p.add_argument("--source-root", type=str, default=None,
                   help="featureengineering/data directory")
    p.add_argument("--output-root", type=str, default=None,
                   help="Output root for trainingdata/v1/")
    p.add_argument("--target", type=str, default="label_ret_1d",
                   choices=["label_ret_1d", "label_ret_5d", "label_ret_10d", "label_ret_20d"],
                   help="Target label to use")
    p.add_argument("--stock-pool", type=str, default=None,
                   help="Stock pool file (one 6-digit code per line)")
    p.add_argument("--min-date", type=str, default="20200101",
                   help="Earliest date to include (YYYYMMDD)")
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel workers for loading factor files")
    return p.parse_args()


def _clean_code(code_series: pd.Series) -> pd.Series:
    """Strip exchange suffix (.SZ/.SH/.BSE) and zero-pad to 6 digits."""
    return code_series.str.replace(r"\.(SZ|SH|BSE)$", "", regex=True).str.zfill(6)


# ── Per-factor loader (runs in thread pool) ─────────────────────────────

def _load_one_factor(ff: Path, min_date: str, allowed_codes: set | None) -> dict:
    """Load and filter a factor, merging base + incremental files if present."""
    name = ff.stem
    try:
        df = pd.read_feather(ff)
    except Exception as e:
        return {"name": name, "error": str(e)}

    # Merge incremental file if present: {name}_incr.fea
    incr_path = ff.parent / f"{name}_incr.fea"
    if incr_path.exists():
        try:
            df_incr = pd.read_feather(incr_path)
            df = pd.concat([df, df_incr])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index(level=["Date", "Code"])
        except Exception:
            pass

    col = df.columns[0]
    dl = df.index.get_level_values("Date")
    df = df[dl >= min_date]

    # Clean codes in the index: strip exchange suffix, zero-pad
    raw_codes = df.index.get_level_values("Code")
    clean_codes = _clean_code(raw_codes)
    new_idx = pd.MultiIndex.from_arrays(
        [df.index.get_level_values("Date"), clean_codes],
        names=["Date", "Code"]
    )
    df.index = new_idx

    if allowed_codes is not None:
        df = df[df.index.get_level_values("Code").isin(allowed_codes)]

    df = df[df[col].notna()]
    if len(df) == 0:
        return {"name": name, "error": "all NaN"}

    # Deduplicate multi-index (keep last if duplicate Date+Code)
    dup_mask = df.index.duplicated(keep="last")
    if dup_mask.any():
        df = df[~dup_mask]

    return {
        "name": name,
        "series": df[col],
        "last_date": df.index.get_level_values("Date").max(),
        "codes": set(df.index.get_level_values("Code").unique()),
    }


def main():
    args = parse_args()
    t_start = time.monotonic()

    # ── Paths ──────────────────────────────────────────────────────────
    src = Path(args.source_root) if args.source_root else \
          Path(__file__).resolve().parents[2] / "featureengineering" / "data"
    factor_dir = src / "factors"
    target_dir = src / "targets"

    out = Path(args.output_root) if args.output_root else Path("/root/shared-nvme/lingqiData/trainingdata/v1")
    out.mkdir(parents=True, exist_ok=True)

    # ── Stock pool ─────────────────────────────────────────────────────
    allowed_codes = None
    if args.stock_pool:
        pool_path = Path(args.stock_pool)
        if not pool_path.is_absolute():
            pool_path = Path.cwd() / args.stock_pool
        with open(pool_path) as f:
            allowed_codes = set(line.strip().zfill(6) for line in f if line.strip())
        print(f"Stock pool: {len(allowed_codes)} stocks")
    else:
        print("Stock pool: ALL (no filter)")

    # ── Load label ─────────────────────────────────────────────────────
    print(f"\n=== Loading target: {args.target} ===")
    target_file = target_dir / f"{args.target}.fea"
    if not target_file.exists():
        print(f"ERROR: {target_file} not found"); sys.exit(1)

    label_df = pd.read_feather(target_file)
    label_col = label_df.columns[0]
    dl = label_df.index.get_level_values("Date")
    label_df = label_df[dl >= args.min_date]
    label_df = label_df[label_df[label_col].notna()]

    if allowed_codes is not None:
        cl = label_df.index.get_level_values("Code")
        label_df = label_df[_clean_code(cl).isin(allowed_codes)]

    label_dates = sorted(label_df.index.get_level_values("Date").unique())
    label_last, label_first = label_dates[-1], label_dates[0]
    n_label_dates = len(label_dates)
    avg_stocks = label_df.groupby(level="Date").size().mean()
    print(f"  Valid dates: {n_label_dates} ({label_first} ~ {label_last})")
    print(f"  Avg stocks/date: {avg_stocks:.0f}")

    # ── Collect codes from label ───────────────────────────────────────
    all_codes_set = set()
    for idx in label_df.index:
        all_codes_set.add(idx[1])

    # ── Parallel load all factors ──────────────────────────────────────
    fea_files = sorted(factor_dir.glob("*.fea"))
    fea_files = [f for f in fea_files
                 if not f.stem.startswith("label_")
                 and not f.stem.endswith("_incr")]
    print(f"\n=== Loading {len(fea_files)} factor files (workers={args.workers}) ===")

    factor_series = {}       # name -> Series
    factor_last_dates = {}   # name -> last_date
    skipped = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_load_one_factor, ff, args.min_date, allowed_codes): ff
            for ff in fea_files
        }
        with tqdm(total=len(futures), desc="Loading factors", unit="file") as pbar:
            for fut in as_completed(futures):
                r = fut.result()
                pbar.update(1)

                name = r["name"]
                if "error" in r:
                    skipped.append((name, r["error"]))
                    continue

                factor_series[name] = r["series"]
                factor_last_dates[name] = r["last_date"]
                all_codes_set.update(r["codes"])

    if skipped:
        print(f"  Skipped {len(skipped)} factors: {[n for n,_ in skipped[:5]]}")

    n_factors = len(factor_series)
    all_codes_sorted = sorted(all_codes_set)
    print(f"  Loaded: {n_factors} factors, {len(all_codes_sorted):,} unique codes")
    print(f"  Done in {time.monotonic() - t_start:.0f}s")

    # ── Build unified panel ────────────────────────────────────────────
    # Create a single MultiIndex, then assign each factor (with ffill if needed)
    # Most factors don't need ffill because their data covers all label dates.
    # Only factors with last_date < label_last need ffill (publication lag).

    print(f"\n=== Building panel: {n_label_dates} dates × {len(all_codes_sorted):,} codes ===")
    t_panel = time.monotonic()

    full_idx = pd.MultiIndex.from_product(
        [label_dates, all_codes_sorted], names=["date", "Code"]
    )
    print(f"  Panel index: {len(full_idx):,} rows")

    # Pre-allocate DataFrame — build column by column
    panel_data = {}
    lag_report = []
    need_ffill = 0
    skip_ffill = 0

    for name in tqdm(factor_series, desc="Aligning factors", unit="factor"):
        series = factor_series[name]
        aligned = series.reindex(full_idx)

        # Only ffill if factor's source data lags behind label
        if factor_last_dates[name] < label_last:
            before = aligned.isna().sum()
            aligned = aligned.groupby(level="Code").ffill()
            after = aligned.isna().sum()
            filled = before - after
            if filled > 0:
                lag_report.append((name, int(filled)))
            need_ffill += 1
        else:
            skip_ffill += 1

        panel_data[name] = aligned

    # Build DataFrame from dict (faster than column-by-column assignment)
    panel = pd.DataFrame(panel_data, index=full_idx)
    panel = panel.dropna(how="all").reset_index()

    print(f"  Panel built in {time.monotonic() - t_panel:.0f}s")
    print(f"  Factors needing ffill: {need_ffill}, skipped ffill: {skip_ffill}")

    if lag_report:
        lag_report.sort(key=lambda x: -x[1])
        for name, count in lag_report:
            print(f"    {name:40s} {count:>10,} values ffill'd")

    # ── Factor last-date distribution ──────────────────────────────────
    dist = Counter(factor_last_dates.values())
    print(f"\n  Factor last-source-date distribution:")
    for d in sorted(dist.keys()):
        print(f"    {d}: {dist[d]} factors")

    # ── Build label (wide format) ──────────────────────────────────────
    print(f"\n=== Building label & liquid data ===")

    label_flat = label_df.reset_index().rename(columns={"Date": "date", label_col: "label"})
    label_flat = label_flat[label_flat["Code"].isin(all_codes_sorted)]
    label_wide = label_flat.pivot(index="date", columns="Code", values="label")
    label_wide.index.name = "index"
    label_wide = label_wide.sort_index(axis=0).sort_index(axis=1)
    print(f"  Label shape: {label_wide.shape}")

    # Filter panel to dates with label
    panel_dates_set = set(str(d) for d in panel["date"].unique())
    label_dates_set = set(str(d) for d in label_wide.index)
    panel_valid_dates = panel_dates_set & label_dates_set
    if not panel_valid_dates:
        print(f"  ERROR: No date intersection! panel={len(panel_dates_set)}, label={len(label_dates_set)}")
        print(f"  Panel sample: {sorted(panel_dates_set)[:3]}")
        print(f"  Label sample: {sorted(label_dates_set)[:3]}")
        sys.exit(1)
    panel = panel[panel["date"].astype(str).isin(panel_valid_dates)]
    dates_final = sorted(panel["date"].unique())
    print(f"  Final dates: {len(dates_final)} ({dates_final[0]} ~ {dates_final[-1]})")

    # ── Build trade_amt ────────────────────────────────────────────────
    daily_path = Path("/root/lingqiData/data/daily_adj.parquet")
    if not daily_path.exists():
        daily_path = src.parent / "daily_adj.parquet"

    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
        daily["trade_date"] = daily["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
        daily = daily.rename(columns={"trade_date": "date", "stock_code": "Code"})
        # Clean codes: strip exchange suffix, zero-pad
        daily["Code"] = _clean_code(daily["Code"])
        daily = daily[daily["date"].isin(panel_valid_dates)]
        if allowed_codes is not None:
            daily = daily[daily["Code"].isin(allowed_codes)]
        daily = daily[daily["Code"].isin(all_codes_sorted)]

        amt_wide = daily.pivot(index="date", columns="Code", values="amount")
        amt_wide.index.name = "index"
        amt_wide = amt_wide.sort_index(axis=0).sort_index(axis=1)
        print(f"  Trade_amt shape: {amt_wide.shape}")
    else:
        print("  WARNING: daily_adj.parquet not found, trade_amt will be placeholder")
        amt_wide = pd.DataFrame(1.0, index=label_wide.index, columns=label_wide.columns)
        amt_wide.index.name = "index"

    # ── Save ───────────────────────────────────────────────────────────
    print(f"\n=== Saving ===")

    fac_df = panel.sort_values(["date", "Code"]).reset_index(drop=True)
    fac_path = out / "fac20260513.fea"
    fac_df.to_feather(fac_path)
    print(f"  {fac_path.name}: {fac_df.shape[0]:,} rows × {fac_df.shape[1] - 2} factors "
          f"({fac_path.stat().st_size / 1e6:.1f} MB)")

    label_path = out / "label.fea"
    label_wide.reset_index().to_feather(label_path)
    print(f"  {label_path.name}: {label_wide.shape} ({label_path.stat().st_size / 1e6:.1f} MB)")

    amt_path = out / "trade_amt.fea"
    amt_wide.reset_index().to_feather(amt_path)
    print(f"  {amt_path.name}: {amt_wide.shape} ({amt_path.stat().st_size / 1e6:.1f} MB)")

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    print(f"\n{'='*70}")
    print(f"  V1 TRAINING DATA SUMMARY  ({elapsed:.0f}s total)")
    print(f"{'='*70}")
    print(f"  Target:         {args.target}")
    print(f"  Label dates:    {label_first} ~ {label_last} ({n_label_dates})")
    print(f"  Trainable:      {dates_final[0]} ~ {dates_final[-1]} ({len(dates_final)} dates)")
    print(f"  Stocks:         {fac_df['Code'].nunique():,}")
    print(f"  Factors:        {n_factors}")
    print(f"  Rows:           {fac_df.shape[0]:,}")
    print(f"  Gap (factor - label): {max(dist.keys())} - {label_last} = {len([d for d in dist if d > label_last])} factors beyond label")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
