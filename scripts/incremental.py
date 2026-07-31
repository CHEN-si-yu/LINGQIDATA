#!/usr/bin/env python
"""
incremental.py — Daily incremental data updater for lingqiData.

Before running any updates, probes the cyq_chips and cyq_perf canary
endpoints to verify the server has published today's data.  If either
is missing, the script waits (polling every 5 min, up to 2 hours) so
that a partial update doesn't have to be re-run.

Reads existing parquet files, determines the last available date for each
dataset, fetches only new data from the diemeng.chat API, and merges it into
the existing files.  Designed to be run daily via cron / Task Scheduler.

Usage:
    python incremental.py                     # update all existing datasets
    python incremental.py --dataset daily     # update a single dataset
    python incremental.py --dataset daily finance margin_detail
    python incremental.py --dry-run           # show what needs updating
    python incremental.py --parallel -w 6     # parallel mode
    python incremental.py --overlap 5         # 5-day overlap window
    python incremental.py --no-per-stock      # skip per-stock datasets
    python incremental.py --monitor           # log rate-limit stats every 10s
    python incremental.py --no-wait           # skip canary check, run now
    python incremental.py --wait-interval 600 # poll every 10 min
    python incremental.py --max-wait 14400    # wait up to 4 hours
    python incremental.py --date 2025-07-01   # use 2025-07-01 as target day

Strategy per dataset type:
    - daily-frequency (OHLCV, fund flow, etc.): read max trade_date,
      fetch from max_date - overlap to today, merge + dedup.
    - quarterly (financial statements): read max end_date,
      fetch from that quarter start to today.
    - reference (stock_list, calendar): re-fetch entirely (small).
    - per-stock (cyq_chips, minute history): for each stock, read its file,
      fetch only new dates, append + dedup.
"""

import argparse
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure fetchers/ is on the import path so that config, fetch_* and
# incremental_core can all be imported directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetchers"))

from config import DATA_DIR, rate_limiter, log_print
import pandas as pd

# Import incremental_core as a module so we can access / mutate its globals
# (OVERLAP_DAYS, EFFECTIVE_TODAY) after wait_for_server() runs.
import incremental_core as core
from incremental_core import (
    DEFAULT_WORKERS, TODAY,
    DEFAULT_WAIT_INTERVAL, DEFAULT_MAX_WAIT,
    _monitor_loop,
    wait_for_server,
    record_state,
    add_timing, print_timing_report,
    _PER_STOCK_BATCH_SIZE,
    _call_fetch,
    _load_state,
    update_per_stock_dataset,
    update_daily_dump_dataset,
    update_consolidated,
    update_indicators_from_daily_dump,
    set_datasets,
)


def _import_fn(module_name, fn_name):
    """Lazily import a fetch function from a module in fetchers/."""
    mod = __import__(module_name, fromlist=[fn_name])
    return getattr(mod, fn_name)


# ── Dataset Registry ──────────────────────────────────────────────────────
# Each entry describes one dataset and how to incrementally update it.

DATASETS = [
    # ═══ Daily-frequency, consolidated ═══
    {
        "name": "daily",
        "module": "fetch_daily", "fn_name": "fetch_daily",
        "file": "daily.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    {
        "name": "daily_adj",
        "module": "fetch_daily", "fn_name": "fetch_daily_adj",
        "file": "daily_adj.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    {
        "name": "finance",
        "module": "fetch_finance", "fn_name": "fetch_finance",
        "file": "finance.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    {
        "name": "margin_detail",
        "module": "fetch_margin_detail", "fn_name": "fetch_margin_detail",
        "file": "margin_detail.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
        "lag_days": 1,
    },
    {
        "name": "main_fund_flow",
        "module": "fetch_main_fund_flow", "fn_name": "fetch_main_fund_flow",
        "file": "main_fund_flow.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    # ═══ Day-by-day (consolidated output) ═══
    {
        "name": "ths_daily",
        "module": "fetch_ths_daily", "fn_name": "fetch_ths_daily",
        "file": "ths_daily.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "ths_code"],
        "sort": ["ths_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    # ═══ Reference data (small, re-fetch entirely) ═══
    {
        "name": "calendar",
        "module": "fetch_calendar", "fn_name": "fetch_calendar",
        "file": "calendar.parquet",
        "date_col": "date",
        "dedup": ["date"],
        "sort": ["date"],
        "start": "2019-01-01",
        "type": "reference",
    },
    {
        "name": "stock_list",
        "module": "fetch_stock_list", "fn_name": "fetch_stock_list",
        "file": "stock_list.parquet",
        "date_col": None,
        "dedup": ["stock_code"],
        "sort": ["stock_code"],
        "start": "2019-01-01",
        "type": "reference",
    },
    {
        "name": "ths_sector_categories",
        "module": "fetch_ths_sector_categories", "fn_name": "fetch_ths_sector_categories",
        "file": "ths_sector_categories.parquet",
        "date_col": None,
        "dedup": ["index_code"],
        "sort": ["type", "index_code"],
        "start": "2019-01-01",
        "type": "reference",
        "weekly_update_day": 4,  # Friday only (Mon=0 ... Sun=6)
    },
    {
        "name": "ths_constituent_stocks",
        "module": "fetch_ths_constituent_stocks", "fn_name": "fetch_ths_constituent_stocks",
        "file": "ths_constituent_stocks.parquet",
        "date_col": None,
        "dedup": ["index_code", "stock_code"],
        "sort": ["index_code", "stock_code"],
        "start": "2019-01-01",
        "type": "reference",
        "weekly_update_day": 4,  # Friday only (Mon=0 ... Sun=6)
    },
    # ═══ Per-stock data ═══
    {
        "name": "cyq_chips",
        "module": None, "fn_name": None,
        "file": "cyq_chips/.done",
        "date_col": "trade_date",
        "dedup": ["trade_date", "price"],
        "sort": ["trade_date"],
        "start": "2019-01-01",
        "type": "per_stock",
        "out_subdir": "cyq_chips",
        "endpoint": "stock/cyq_chips",
        "backfill_module": "fetch_cyq_chips",
        "backfill_fn_name": "fetch_cyq_chips",
        "end_exclusive": True,
        "consolidate": False,
    },
    {
        "name": "cyq_perf",
        "module": "fetch_cyq_perf", "fn_name": "fetch_cyq_perf",
        "file": "cyq_perf.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
        "overlap_days": 5,
    },
    {
        "name": "history_1min",
        "module": None, "fn_name": None,
        "file": "history_1min/.done",
        "date_col": "trade_time",
        "dedup": ["trade_time", "stock_code"],
        "sort": ["trade_time"],
        "start": "2019-01-01",
        "type": "per_stock",
        "out_subdir": "history_1min",
        "use_daily_dump": True,
        "daily_dump_level": "1min",
        "backfill_module": "fetch_minute",
        "backfill_fn_name": "fetch_history",
        "backfill_kwargs": {"level": "1min"},
        "consolidate": False,
    },
    # ════════════════════ Technical indicators — 1min (combined, local compute, 0 API) ════════════════════
    {
        "name": "indicators_1min",
        "module": None, "fn_name": None,
        "file": "indicator_1min/.done",
        "date_col": "trade_time",
        "dedup": ["trade_time", "stock_code"],
        "sort": ["trade_time"],
        "start": "2019-01-01",
        "type": "per_stock",
        "out_subdir": "indicator_1min",
        "local_indicator": True,
        "indicator_name": None,
        "consolidate": False,
    },
]

# Bridge the dataset registry to incremental_core so canary checks can
# inspect local dataset state.
set_datasets(DATASETS)


def _build_registry_index():
    """Return {name: entry} for quick lookup."""
    return {d["name"]: d for d in DATASETS}


# ── Main orchestrator ─────────────────────────────────────────────────────

def _normalize_date(d):
    """Normalize a date string to YYYY-MM-DD format.

    Accepts both '2025-07-01' and '20250701'.
    Returns None if *d* is None or empty.
    """
    if not d:
        return None
    d = d.strip()
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return d


def run_updates(datasets=None, exclude=None, overlap_days=core.OVERLAP_DAYS,
                workers=DEFAULT_WORKERS, parallel=False, dry_run=False,
                skip_per_stock=False, skip_reference=False, monitor=False,
                no_wait=False, wait_interval=DEFAULT_WAIT_INTERVAL,
                max_wait=DEFAULT_MAX_WAIT, target_date=None):
    """Run incremental updates for the specified datasets.

    Parameters
    ----------
    datasets : list or None
        Specific dataset names to update.  None = all existing.
    exclude : list or None
        Dataset names to exclude.
    overlap_days : int
        Days of overlap when computing incremental ranges.
    workers : int
        Number of parallel workers per fetch task.
    parallel : bool
        Run multiple datasets concurrently.
    dry_run : bool
        Show what would be updated without making API calls.
    skip_per_stock : bool
        Skip per-stock datasets (cyq_chips, minute).
    skip_reference : bool
        Skip reference datasets (calendar, stock_list).
    monitor : bool
        Log rate-limit stats every 10s via a background thread.
    no_wait : bool
        Skip the pre-flight canary check and run immediately.
    wait_interval : int
        Seconds between canary probes (default 300).
    max_wait : int
        Maximum seconds to wait for server data (default 7200).
    target_date : str or None
        Target date in "YYYY-MM-DD" format. If provided, this date is used
        as the effective "today" for the incremental run — the script fetches
        whatever data is missing relative to this target day.
    """
    # Normalize date formats (20250701 -> 2025-07-01)
    if target_date:
        target_date = _normalize_date(target_date)
    # ── Pre-flight: wait until server has today's data ──
    if not no_wait:
        server_ready = wait_for_server(
            target_date=target_date, interval=wait_interval,
            max_wait=max_wait, dry_run=dry_run,
        )
        if not server_ready and not dry_run:
            log_print("[preflight] Server data is incomplete, "
                      "but proceeding after timeout.")
    else:
        log_print("[preflight] Skipped (--no-wait)")

    # Push overlap_days into the core module so all helpers use it
    core.OVERLAP_DAYS = overlap_days

    # EFFECTIVE_TODAY may have been updated by wait_for_server
    # If target_date is explicitly provided, use it instead
    effective = target_date if target_date else core.EFFECTIVE_TODAY

    registry = _build_registry_index()
    exclude = set(exclude or [])

    # Determine which datasets to process
    if datasets:
        selected = [d for d in DATASETS if d["name"] in datasets]
        not_found = set(datasets) - {d["name"] for d in selected}
        if not_found:
            log_print(f"[incremental] Unknown datasets: {not_found}")
    else:
        selected = []
        for d in DATASETS:
            if d["name"] in exclude:
                continue
            if d["type"] == "per_stock":
                if skip_per_stock:
                    continue
                out_dir = Path(DATA_DIR) / d.get("out_subdir", "")
                if out_dir.exists() and any(out_dir.glob("*.parquet")):
                    selected.append(d)
            else:
                if d["type"] == "reference" and skip_reference:
                    continue
                fpath = Path(DATA_DIR) / d["file"]
                if fpath.exists():
                    selected.append(d)

    if not selected:
        log_print("[incremental] No datasets to update (no existing files found)")
        return []

    log_print(f"[incremental] {'DRY-RUN: ' if dry_run else ''}"
              f"{len(selected)} datasets | overlap={overlap_days} days | "
              f"workers={workers} | parallel={parallel}")
    log_print(f"[incremental] Datasets: {[d['name'] for d in selected]}")

    # ── Defer local_indicator datasets until after all source data is updated ──
    # indicators_1min reads from history_1min/{code}.parquet, so it MUST run
    # after history_1min (per_stock / daily_dump) has been refreshed.
    local_ind_ds = [d for d in selected if d.get("local_indicator")]
    selected = [d for d in selected if not d.get("local_indicator")]

    # ── Monitor thread ──
    stop_event = threading.Event()
    monitor_thread = None
    if monitor:
        monitor_thread = threading.Thread(
            target=_monitor_loop, args=(stop_event, 10.0), daemon=True
        )
        monitor_thread.start()

    results = []

    def _process_one(d):
        """Process a single dataset entry."""
        name = d["name"]
        t0 = time.perf_counter()
        filepath = str(Path(DATA_DIR) / d["file"])
        log_print(f"\n{'='*60}")
        log_print(f"[{name}] Starting incremental update")

        try:
            if d["type"] == "per_stock":
                if skip_per_stock:
                    log_print(f"[{name}] Skipped (per-stock disabled)")
                    return {"name": name, "status": "skip", "reason": "per_stock_disabled"}

                if d.get("use_daily_dump"):
                    backfill_fn = None
                    bf_module = d.get("backfill_module")
                    bf_fn_name = d.get("backfill_fn_name")
                    if bf_module and bf_fn_name:
                        bf_mod = __import__(bf_module, fromlist=[bf_fn_name])
                        backfill_fn = getattr(bf_mod, bf_fn_name)
                    update_daily_dump_dataset(
                        name=name,
                        out_subdir=d["out_subdir"],
                        level=d["daily_dump_level"],
                        start_date=d["start"],
                        end_date=core.EFFECTIVE_TODAY,
                        date_col=d["date_col"],
                        dedup_keys=d.get("dedup"),
                        workers=16,
                        dry_run=dry_run,
                        backfill_fn=backfill_fn,
                        backfill_kwargs=d.get("backfill_kwargs"),
                        consolidate=d.get("consolidate", True),
                    )
                    return {"name": name, "status": "updated" if not dry_run else "dry_run"}

                # Resolve backfill function for stocks without existing files
                backfill_fn = None
                bf_module = d.get("backfill_module")
                bf_fn_name = d.get("backfill_fn_name")
                if bf_module and bf_fn_name:
                    bf_mod = __import__(bf_module, fromlist=[bf_fn_name])
                    backfill_fn = getattr(bf_mod, bf_fn_name)
                if d.get("end_exclusive"):
                    end_dt_excl = datetime.strptime(core.EFFECTIVE_TODAY, "%Y-%m-%d") + timedelta(days=1)
                    per_stock_end = end_dt_excl.strftime("%Y-%m-%d")
                else:
                    per_stock_end = core.EFFECTIVE_TODAY
                update_per_stock_dataset(
                    name=name,
                    out_subdir=d["out_subdir"],
                    endpoint=d["endpoint"],
                    start_date=d["start"],
                    end_date=per_stock_end,
                    date_col=d["date_col"],
                    workers=workers,
                    dry_run=dry_run,
                    extra_payload=d.get("extra_payload"),
                    backfill_fn=backfill_fn,
                    backfill_kwargs=d.get("backfill_kwargs"),
                    batch_size=d.get("batch_size", _PER_STOCK_BATCH_SIZE),
                    stock_code_as_list=d.get("stock_code_as_list", True),
                    dedup_keys=d.get("dedup"),
                    consolidate=d.get("consolidate", True),
                )
                return {"name": name, "status": "updated" if not dry_run else "dry_run"}

            if d["type"] == "reference":
                if skip_reference:
                    log_print(f"[{name}] Skipped (reference disabled)")
                    return {"name": name, "status": "skip", "reason": "reference_disabled"}
                # ── Weekly-update gate ──
                weekly_day = d.get("weekly_update_day")
                if weekly_day is not None and date.today().weekday() != weekly_day:
                    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    log_print(f"[{name}] Skipped (weekly update, next fetch on "
                              f"{day_names[weekly_day]}")
                    return {"name": name, "status": "skip",
                            "reason": f"not_{day_names[weekly_day]}"}
                if dry_run:
                    log_print(f"[{name}] DRY-RUN: would re-fetch entirely")
                    return {"name": name, "status": "dry_run"}
                fn = _import_fn(d["module"], d["fn_name"])
                try:
                    t_fetch = time.perf_counter()
                    new_df = _call_fetch(
                        fn,
                        start_date=d["start"],
                        end_date=effective,
                        workers=1,
                        cleanup=True,
                        resume=False,
                    )
                    add_timing(name, "fetch", time.perf_counter() - t_fetch)
                    if new_df is not None and not new_df.empty:
                        dest = Path(DATA_DIR) / d["file"]
                        tmp = dest.with_suffix(".parquet.tmp")
                        new_df.to_parquet(tmp, index=False)
                        tmp.replace(dest)
                        log_print(f"[{name}] Re-fetched: {len(new_df)} rows -> {dest}")
                    return {"name": name, "status": "updated",
                            "rows": len(new_df) if new_df is not None else 0}
                except Exception as e:
                    log_print(f"[{name}] FAILED: {e}")
                    return {"name": name, "status": "error", "error": str(e)}

            # Consolidated type
            fn = _import_fn(d["module"], d["fn_name"])
            return update_consolidated(
                name=name,
                fetch_fn=fn,
                filepath=filepath,
                date_col=d["date_col"],
                dedup_keys=d["dedup"],
                sort_cols=d["sort"],
                default_start=d["start"],
                workers=workers,
                dry_run=dry_run,
                extra_params=d.get("extra"),
                strict_dedup=d.get("strict_dedup", False),
                overlap_days=d.get("overlap_days"),
                backfill=d.get("backfill", False),
                lag_days=d.get("lag_days", 0),
            )
        finally:
            add_timing(name, "total", time.perf_counter() - t0)

    try:
        if parallel and len(selected) > 1:
            # Process reference datasets FIRST to avoid race conditions:
            # some datasets read calendar.parquet for trading days.
            ref_datasets = [d for d in selected if d["type"] == "reference"]
            other_datasets = [d for d in selected if d["type"] != "reference"]

            if ref_datasets:
                log_print(f"[incremental] Processing {len(ref_datasets)} reference "
                          f"dataset(s) first: {[d['name'] for d in ref_datasets]}")
                for d in ref_datasets:
                    try:
                        r = _process_one(d)
                        results.append(r)
                    except Exception as e:
                        log_print(f"[{d['name']}] UNEXPECTED ERROR: {e}")
                        results.append({"name": d["name"], "status": "error", "error": str(e)})

            if other_datasets:
                with ThreadPoolExecutor(max_workers=min(len(other_datasets), 4)) as pool:
                    futures = {pool.submit(_process_one, d): d["name"] for d in other_datasets}
                    for fut in as_completed(futures):
                        name = futures[fut]
                        try:
                            r = fut.result()
                            results.append(r)
                        except Exception as e:
                            log_print(f"[{name}] UNEXPECTED ERROR: {e}")
                            results.append({"name": name, "status": "error", "error": str(e)})
        else:
            # Always process reference datasets first
            ordered = sorted(selected, key=lambda d: 0 if d["type"] == "reference" else 1)
            for d in ordered:
                try:
                    r = _process_one(d)
                    results.append(r)
                except Exception as e:
                    log_print(f"[{d['name']}] UNEXPECTED ERROR: {e}")
                    results.append({"name": d["name"], "status": "error", "error": str(e)})

        # ── Local indicators (must run AFTER history_1min et al. are updated) ──
        if local_ind_ds:
            log_print(f"\n{'='*60}")
            log_print(f"[incremental] Batch-processing {len(local_ind_ds)} "
                      f"local indicators: {[d['name'] for d in local_ind_ds]}")
            try:
                update_indicators_from_daily_dump(
                    indicator_datasets=local_ind_ds,
                    start_date=local_ind_ds[0]["start"],
                    end_date=effective,
                    workers=workers,
                    dry_run=dry_run,
                    overlap_days=overlap_days,
                )
                for d in local_ind_ds:
                    record_state(d["name"], effective)
                    results.append({"name": d["name"], "status": "updated" if not dry_run else "dry_run"})
            except Exception as e:
                log_print(f"[incremental] Local indicator batch FAILED: {e}")
                import traceback
                traceback.print_exc()
                for d in local_ind_ds:
                    results.append({"name": d["name"], "status": "error", "error": str(e)})

        # ── Summary ──
        log_print(f"\n{'='*60}")
        log_print(f"[incremental] {'DRY-RUN ' if dry_run else ''}Summary:")
        updated = [r for r in results if r["status"] == "updated"]
        uptodate = [r for r in results if r["status"] == "uptodate"]
        skipped = [r for r in results if r["status"] == "skip"]
        errors = [r for r in results if r["status"] == "error"]
        dry = [r for r in results if r["status"] == "dry_run"]

        if dry_run:
            for r in dry:
                info = r.get("range", r.get("reason", ""))
                log_print(f"  [dry-run] {r['name']}: {info}")
        for r in updated:
            rows = r.get("rows_added", r.get("rows", "?"))
            old_m = r.get("old_max", "?")
            new_m = r.get("new_max", "?")
            log_print(f"  [updated]  {r['name']}: +{rows} rows | "
                      f"{old_m} -> {new_m}")
        for r in uptodate:
            log_print(f"  [uptodate] {r['name']}: max={r.get('max_date', '?')}")
        for r in skipped:
            log_print(f"  [skipped]  {r['name']}: {r.get('reason', '')}")
        for r in errors:
            log_print(f"  [error]    {r['name']}: {r.get('error', 'unknown')}")

        n_up = len(updated)
        n_utd = len(uptodate)
        n_sk = len(skipped)
        n_err = len(errors)
        log_print(f"[incremental] Done: {n_up} updated, {n_utd} uptodate, "
                  f"{n_sk} skipped, {n_err} errors")

        # Rate limiter stats
        stats = rate_limiter().stats
        total_calls = sum(stats["endpoint_counts"].values())
        log_print(f"[incremental] Total API calls: {total_calls}")
        for ep, n in sorted(stats["endpoint_counts"].items()):
            log_print(f"  {ep}: {n}")

        # Pending datasets (server lagging)
        state = _load_state()
        pending = {k: v for k, v in state.items() if v.get("pending")}
        if pending:
            log_print(f"\n[incremental] PENDING (server behind, "
                      f"will re-fetch with extended overlap next run):")
            for name, entry in sorted(pending.items()):
                since = entry.get("pending_since", "?")
                actual = entry.get("last_fetch_max", "?")
                log_print(f"  [pending] {name:<28} last_data={actual}  since={since}")

        # Performance timing report
        print_timing_report()

    finally:
        if monitor_thread:
            stop_event.set()
            monitor_thread.join(timeout=2)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Incremental daily data updater for lingqiData",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python incremental.py                          # Update all existing datasets
  python incremental.py --dataset daily          # Update only daily.parquet
  python incremental.py --dataset daily finance  # Update daily + finance
  python incremental.py --dry-run                # Preview what needs updating
  python incremental.py --parallel -w 6          # Run datasets in parallel
  python incremental.py --overlap 5              # 5-day overlap window
  python incremental.py --no-per-stock           # Skip cyq_chips/minute
  python incremental.py --monitor                # Log rate-limit stats every 10s
  python incremental.py --no-wait                # Skip canary check
  python incremental.py --wait-interval 600      # Poll canaries every 10 min
  python incremental.py --max-wait 14400         # Wait up to 4 hours
  python incremental.py --list                   # List all known datasets
        """,
    )
    parser.add_argument("--dataset", "-d", nargs="+",
                        help="Specific dataset(s) to update (default: all existing)")
    parser.add_argument("--exclude", "-x", nargs="+",
                        help="Dataset(s) to exclude")
    parser.add_argument("--overlap", type=int, default=core.OVERLAP_DAYS,
                        help=f"Days of overlap for incremental range "
                             f"(default: {core.OVERLAP_DAYS})")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers per task (default: {DEFAULT_WORKERS})")
    parser.add_argument("--parallel", action="store_true",
                        help="Run multiple datasets concurrently")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be updated without fetching")
    parser.add_argument("--no-per-stock", action="store_true",
                        help="Skip per-stock datasets (cyq_chips, history, min_adj)")
    parser.add_argument("--no-reference", action="store_true",
                        help="Skip reference datasets (calendar, stock_list)")
    parser.add_argument("--monitor", action="store_true",
                        help="Log rate-limit stats every 10s via a background thread")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip the pre-flight canary check (run immediately)")
    parser.add_argument("--wait-interval", type=int, default=DEFAULT_WAIT_INTERVAL,
                        help=f"Seconds between canary probes "
                             f"(default: {DEFAULT_WAIT_INTERVAL})")
    parser.add_argument("--max-wait", type=int, default=DEFAULT_MAX_WAIT,
                        help=f"Maximum seconds to wait for server data "
                             f"(default: {DEFAULT_MAX_WAIT})")
    parser.add_argument("--date", "--target-date", type=str, default=None,
                        metavar="YYYY-MM-DD",
                        help="Target date for this incremental run "
                             "(default: today / last trading day)")
    parser.add_argument("--list", action="store_true",
                        help="List all known datasets and exit")

    args = parser.parse_args()

    # Normalize date format (20250701 -> 2025-07-01)
    if args.date:
        args.date = _normalize_date(args.date)

    if args.list:
        print("\nKnown datasets:")
        print(f"{'Name':<28} {'Type':<14} {'File'}")
        print("-" * 72)
        for d in DATASETS:
            name = d["name"]
            dtype = d["type"]
            fname = d["file"]
            exists = "Y" if (Path(DATA_DIR) / d["file"]).exists() else " "
            print(f"  [{exists}] {name:<25} {dtype:<14} {fname}")
        print()
        return

    run_updates(
        datasets=args.dataset,
        exclude=args.exclude,
        overlap_days=args.overlap,
        workers=args.workers,
        parallel=args.parallel,
        dry_run=args.dry_run,
        skip_per_stock=args.no_per_stock,
        skip_reference=args.no_reference,
        monitor=args.monitor,
        no_wait=args.no_wait,
        wait_interval=args.wait_interval,
        max_wait=args.max_wait,
        target_date=args.date,
    )


if __name__ == "__main__":
    main()
