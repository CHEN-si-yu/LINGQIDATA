#!/usr/bin/env python3
"""Factor validation script.

Checks every generated .fea factor file for:
  1. Last-date alignment — does each factor end on the same date as daily_adj?
  2. NaN rate from 2020-01-01 onward (event factors exempt)
  3. Stock count — exactly the Code_num.txt pool
  4. Daily frequency — no gaps vs calendar
  5. Output format — wide Date×Code, 6-digit code columns, YYYYMMDD index

Usage:
    python scripts/validate_factors.py                          # default paths
    python scripts/validate_factors.py --project-root /path/to/featureengineering
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from typing import Any

import numpy as np
import pandas as pd

# ── helpers ────────────────────────────────────────────────────────────────────


def _pad_code(code: str) -> str:
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _load_stock_pool(pool_path: Path) -> set[str]:
    codes: set[str] = set()
    if pool_path.exists():
        for line in pool_path.read_text(encoding="utf-8").splitlines():
            c = line.strip()
            if c:
                codes.add(_pad_code(c))
    return codes


EVENT_CATEGORY = "event"
TARGET_CATEGORY = "target"

# Thresholds
NAN_RATE_WARN = 0.05   # warn above 5%
NAN_RATE_FAIL = 0.15   # fail above 15%
NAN_START_DATE = "20200101"


def load_calendar_dates(calendar_path: Path) -> set[str]:
    """Return all trading days (YYYYMMDD) from calendar.parquet."""
    if not calendar_path.exists():
        return set()
    cal = pd.read_parquet(calendar_path)
    if "is_open" in cal.columns:
        cal = cal[cal["is_open"].astype(bool)]
    return set(cal["date"].astype(str).str.replace("-", "").str.slice(0, 8))


def check_factor(
    fea_path: Path,
    manifest: dict[str, Any] | None,
    stock_pool: set[str],
    calendar_dates: set[str],
    reference_last_date: str,
) -> dict[str, Any]:
    """Run all checks on a single factor .fea file.

    Returns a dict of check results.
    """
    name = fea_path.stem
    result: dict[str, Any] = {
        "name": name,
        "category": manifest.get("category", "?") if manifest else "?",
        "path": str(fea_path),
        "errors": [],
        "warnings": [],
    }

    # ── Read factor ────────────────────────────────────────────────────────
    try:
        df = pd.read_feather(fea_path)
    except Exception as e:
        result["errors"].append(f"READ_ERROR: {e}")
        return result

    if df.empty:
        result["errors"].append("EMPTY_FILE")
        return result

    # ── Format: Date index (rows) × Code columns (wide) ────────────────────
    if df.index.name != "Date":
        result["errors"].append(f"Index name is '{df.index.name}', expected 'Date'")

    if df.columns.name != "Code":
        result["errors"].append(f"Columns name is '{df.columns.name}', expected 'Code'")

    dates = pd.Index(df.index)
    codes = pd.Index(df.columns)

    result["n_dates"] = len(dates)
    result["n_codes"] = len(codes)
    result["total_cells"] = len(dates) * len(codes)

    # ── Last date check ────────────────────────────────────────────────────
    last_date = str(dates.max())
    result["last_date"] = last_date
    if last_date != reference_last_date:
        delta_msg = f"last_date={last_date}, reference={reference_last_date}"
        result["warnings"].append(f"LAST_DATE_MISMATCH: {delta_msg}")

    result["last_day_stock_count"] = int(df.loc[last_date].notna().sum())

    # ── Code format check (6-digit zero-padded) ────────────────────────────
    bad_codes = [c for c in codes if not (isinstance(c, str) and len(c) == 6 and c.isdigit())]
    # Also check via first few codes:
    sample_bad = [c for c in list(codes[:10]) if not (isinstance(c, str) and len(c) == 6 and c.isdigit())]
    if sample_bad:
        result["errors"].append(f"CODE_FORMAT: codes should be 6-digit strings, got {sample_bad}")

    # ── Stock pool check ───────────────────────────────────────────────────
    extra_codes = set(codes) - stock_pool
    missing_codes = stock_pool - set(codes)
    result["extra_codes"] = len(extra_codes)
    result["missing_codes"] = len(missing_codes)
    if extra_codes:
        result["errors"].append(f"EXTRA_CODES: {len(extra_codes)} codes not in stock pool: {sorted(extra_codes)[:10]}...")
    if len(missing_codes) > len(stock_pool) * 0.1:
        result["warnings"].append(f"MISSING_CODES: {len(missing_codes)} pool stocks missing ({len(missing_codes)/len(stock_pool)*100:.1f}%)")

    # ── Date format check ──────────────────────────────────────────────────
    date_strs = [str(d) for d in dates[:5]]
    bad_dates = [d for d in date_strs if len(d) != 8 or not d.isdigit()]
    if bad_dates:
        result["errors"].append(f"DATE_FORMAT: should be YYYYMMDD, got {bad_dates}")

    # ── Daily frequency check ──────────────────────────────────────────────
    if calendar_dates:
        factor_date_set = set(str(d) for d in dates)
        # Only count gaps BETWEEN the factor's first and last date (inclusive)
        factor_first = str(dates.min())
        factor_last = str(dates.max())
        cal_in_range = {d for d in calendar_dates if factor_first <= d <= factor_last}
        missing = cal_in_range - factor_date_set
        extra_d = factor_date_set - calendar_dates
        result["missing_dates"] = len(missing)
        result["extra_dates"] = len(extra_d)
        if missing and len(missing) > 5:
            result["warnings"].append(
                f"GAPS: {len(missing)} calendar dates missing within [{factor_first}, {factor_last}]"
            )
        if extra_d:
            result["errors"].append(
                f"NON_TRADING_DAYS: {len(extra_d)} dates not in calendar: {sorted(extra_d)[:5]}..."
            )
    else:
        result["missing_dates"] = "N/A (no calendar)"

    # ── NaN rate from 2020-01-01 ───────────────────────────────────────────
    category = result["category"]
    is_event = (category == EVENT_CATEGORY)
    is_target = (category == TARGET_CATEGORY)

    nan_start_dates = dates[dates >= NAN_START_DATE]
    if len(nan_start_dates) > 0:
        # Build a boolean mask instead of generator-in-.loc
        nan_mask = df.index.isin([str(d) for d in nan_start_dates])
        nan_slice = df.loc[nan_mask]
        if len(nan_slice) > 0:
            total_cells = len(nan_slice) * len(nan_slice.columns)
            nan_cells = int(nan_slice.isna().sum().sum())
            nan_rate = nan_cells / total_cells if total_cells > 0 else 0.0
            result["nan_rate"] = round(nan_rate, 6)
            result["nan_start_date"] = NAN_START_DATE
            result["nan_slice_dates"] = len(nan_slice)
            result["nan_slice_codes"] = len(nan_slice.columns)
            result["nan_cells"] = nan_cells
            result["nan_total_cells"] = total_cells

            if not is_event and not is_target:
                if nan_rate > NAN_RATE_FAIL:
                    result["errors"].append(
                        f"NAN_RATE_HIGH: {nan_rate:.2%} > {NAN_RATE_FAIL:.0%} (threshold)"
                    )
                elif nan_rate > NAN_RATE_WARN:
                    result["warnings"].append(
                        f"NAN_RATE_ELEVATED: {nan_rate:.2%} > {NAN_RATE_WARN:.0%}"
                    )
        else:
            result["nan_rate"] = None
            result["warnings"].append("NAN_SLICE_EMPTY: no data from 2020-01-01 in factor")
    else:
        result["nan_rate"] = None
        result["warnings"].append("NO_DATA_AFTER_2020: factor starts after 2020-01-01")

    # ── Value range check (ranks should be 0-1) ────────────────────────────
    if not is_target and len(nan_start_dates) > 0:
        # Sample a few dates to avoid full memory scan
        sample_dates = nan_start_dates[::max(1, len(nan_start_dates) // 5)]
        sample_values = []
        for d in sample_dates:
            d_str = str(d)
            if d_str in df.index:
                row = df.loc[d_str]
                vals = row.dropna().values
                if len(vals) > 0:
                    sample_values.extend(vals[:500])  # cap per date
        if sample_values:
            arr = np.array(sample_values)
            vmin, vmax = arr.min(), arr.max()
            result["value_min"] = round(float(vmin), 4)
            result["value_max"] = round(float(vmax), 4)
            if vmin < 0 or vmax > 1.01:  # small tolerance for float
                result["warnings"].append(f"VALUE_RANGE: [{vmin:.4f}, {vmax:.4f}] — rank should be [0, 1]")

    return result


def print_summary(results: list[dict[str, Any]], reference_last_date: str) -> int:
    """Print results table and return exit code (0=all good, 1=errors present)."""
    # Sort: errors first, then warnings, then ok
    def _severity(r: dict[str, Any]) -> int:
        if r["errors"]:
            return 0
        if r.get("warnings"):
            return 1
        return 2

    results = sorted(results, key=lambda r: (_severity(r), r["name"]))

    error_count = sum(1 for r in results if r["errors"])
    warn_count = sum(1 for r in results if not r["errors"] and r.get("warnings"))
    ok_count = sum(1 for r in results if not r["errors"] and not r.get("warnings"))

    # ── Header ─────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  Factor Validation Report  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Reference last date: {reference_last_date}")
    print(f"  Total factors: {len(results)}")
    print(f"  Errors: {error_count}  |  Warnings: {warn_count}  |  OK: {ok_count}")
    print(f"{'='*100}")

    # ── Error rows ─────────────────────────────────────────────────────────
    def _nan_str(r: dict[str, Any]) -> str:
        nr = r.get("nan_rate")
        if nr is None:
            return "   N/A"
        return f"{nr:7.2%}"

    if error_count:
        print(f"\n  --- ERRORS ({error_count}) ---")
        print(f"  {'Factor':<35} {'Last Date':>10} {'NaN rate':>8}  Errors")
        print(f"  {'-'*35} {'-'*10} {'-'*8}  {'-'*40}")
        for r in results:
            if r["errors"]:
                last = r.get("last_date", "?")
                print(f"  {r['name']:<35} {last:>10} {_nan_str(r):>8}  {'; '.join(r['errors'])}")

    # ── Warning rows ───────────────────────────────────────────────────────
    if warn_count:
        print(f"\n  --- WARNINGS ({warn_count}) ---")
        print(f"  {'Factor':<35} {'Last Date':>10} {'NaN rate':>8}  {'Cat':<10} Warnings")
        print(f"  {'-'*35} {'-'*10} {'-'*8}  {'-'*10} {'-'*40}")
        for r in results:
            if not r["errors"] and r.get("warnings"):
                last = r.get("last_date", "?")
                cat = r.get("category", "?")
                print(f"  {r['name']:<35} {last:>10} {_nan_str(r):>8}  {cat:<10} {'; '.join(r['warnings'])}")

    # ── Summary table: all factors ─────────────────────────────────────────
    print(f"\n  --- All Factors ({len(results)}) ---")
    print(f"  {'Factor':<35} {'Last Date':>10} {'Stocks':>7} {'NaN rate':>8}  {'Cat':<10} Status")
    print(f"  {'-'*35} {'-'*10} {'-'*7} {'-'*8}  {'-'*10} {'-'*6}")

    for r in results:
        last = r.get("last_date", "?")
        stocks = r.get("n_codes", "?")
        cat = r.get("category", "?")
        if r["errors"]:
            status = "ERROR"
        elif r.get("warnings"):
            status = "WARN"
        else:
            status = "OK"
        print(f"  {r['name']:<35} {last:>10} {stocks:>7} {_nan_str(r):>8}  {cat:<10} {status}")

    # ── NaN rate distribution (non-event) ──────────────────────────────────
    non_event = [r for r in results if r.get("category") not in (EVENT_CATEGORY, TARGET_CATEGORY)]
    nan_rates = [r["nan_rate"] for r in non_event if r.get("nan_rate") is not None]
    if nan_rates:
        print(f"\n  --- NaN Rate Distribution (non-event, N={len(nan_rates)}) ---")
        arr = np.array(nan_rates)
        print(f"  min={arr.min():.4f}  p25={np.percentile(arr,25):.4f}  median={np.median(arr):.4f}  "
              f"p75={np.percentile(arr,75):.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")

    print(f"\n{'='*100}")
    print(f"  Result: {error_count} errors, {warn_count} warnings, {ok_count} clean")
    print(f"{'='*100}\n")

    return 1 if error_count > 0 else 0


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all factor .fea files")
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Feature engineering project root")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Upstream data root (for calendar.parquet, Code_num.txt)")
    parser.add_argument("--factor-dir", type=Path, default=None,
                        help="Factor output directory override")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help="Manifest directory override")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write full results as JSON to this path")
    args = parser.parse_args()

    # ── Path resolution ────────────────────────────────────────────────────
    project_root = args.project_root or Path(__file__).resolve().parents[1]
    factor_dir = args.factor_dir or (project_root / "data" / "factors")
    manifest_dir = args.manifest_dir or (project_root / "data" / "manifests")
    source_root = args.source_root or (project_root.parent / "data")
    stock_pool_path = source_root.parent / "Code_num.txt"  # /root/shared-nvme/lingqiData/Code_num.txt

    # Handle possible path resolution mismatch
    if not stock_pool_path.exists():
        stock_pool_path = project_root.parent / "Code_num.txt"

    calendar_path = source_root / "calendar.parquet"
    if not calendar_path.exists():
        # Try alternate
        calendar_path = project_root.parent / "data" / "calendar.parquet"

    print(f"Project root:  {project_root}")
    print(f"Factor dir:    {factor_dir}")
    print(f"Source root:   {source_root}")
    print(f"Stock pool:    {stock_pool_path}  (exists={stock_pool_path.exists()})")
    print(f"Calendar:      {calendar_path}  (exists={calendar_path.exists()})")

    # ── Load reference data ─────────────────────────────────────────────────
    stock_pool = _load_stock_pool(stock_pool_path)
    print(f"Stock pool size: {len(stock_pool)}")

    calendar_dates = load_calendar_dates(calendar_path)
    print(f"Trading days:    {len(calendar_dates)}")

    # Determine reference last date from daily_adj.parquet
    daily_adj_path = source_root / "daily_adj.parquet"
    if not daily_adj_path.exists():
        daily_adj_path = project_root.parent / "data" / "daily_adj.parquet"

    if daily_adj_path.exists():
        import pyarrow.parquet as pq
        schema = pq.read_schema(daily_adj_path)
        df_dates = pd.read_parquet(daily_adj_path, columns=["trade_date"])
        max_d = df_dates["trade_date"].max()
        reference_last_date = str(max_d).replace("-", "")[:8]
        print(f"Reference last date (daily_adj): {reference_last_date}")
    else:
        # Fallback: today minus 1
        from datetime import timedelta
        reference_last_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        print(f"WARNING: daily_adj not found, using yesterday as reference: {reference_last_date}")

    # ── Find factor files ──────────────────────────────────────────────────
    fea_files = sorted(factor_dir.glob("*.fea"))
    if not fea_files:
        print(f"\nERROR: No .fea files found in {factor_dir}")
        return 1

    # Load manifests for metadata
    manifests: dict[str, dict] = {}
    for mf in manifest_dir.glob("*.json"):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            manifests[m["name"]] = m
        except Exception:
            pass

    print(f"\nChecking {len(fea_files)} factor files...\n")

    # ── Run checks ─────────────────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    for fea_path in fea_files:
        name = fea_path.stem
        manifest = manifests.get(name)
        r = check_factor(
            fea_path=fea_path,
            manifest=manifest,
            stock_pool=stock_pool,
            calendar_dates=calendar_dates,
            reference_last_date=reference_last_date,
        )
        results.append(r)

    # ── Print report ───────────────────────────────────────────────────────
    exit_code = print_summary(results, reference_last_date)

    # ── JSON export ────────────────────────────────────────────────────────
    if args.json:
        out_path = args.json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Full results written to {out_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
