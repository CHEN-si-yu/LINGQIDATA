#!/usr/bin/env python
"""
incremental_update.py — Daily incremental data updater for lingqiData.

Reads existing parquet files, determines the last available date for each
dataset, fetches only new data from the diemeng.chat API, and merges it into
the existing files.  Designed to be run daily via cron / Task Scheduler.

Usage:
    python incremental_update.py                     # update all existing datasets
    python incremental_update.py --dataset daily     # update a single dataset
    python incremental_update.py --dataset daily finance margin_detail
    python incremental_update.py --dry-run           # show what needs updating
    python incremental_update.py --parallel -w 6     # parallel mode
    python incremental_update.py --overlap 5         # 5-day overlap window
    python incremental_update.py --no-per-stock      # skip per-stock datasets

Strategy per dataset type:
    - daily-frequency (OHLCV, fund flow, etc.): read max trade_date,
      fetch from max_date - overlap to today, merge + dedup.
    - quarterly (financial statements, holder_number): read max end_date,
      fetch from that quarter start to today.
    - reference (stock_list, calendar): re-fetch entirely (small).
    - per-stock (cyq_chips, minute history): for each stock, read its file,
      fetch only new dates, append + dedup.
"""

import argparse
import inspect
import json
import sys
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import pyarrow.parquet as pq

# Ensure the scripts directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print
)

# ── Globals ───────────────────────────────────────────────────────────────
OVERLAP_DAYS = 3  # days of overlap to catch data corrections
DEFAULT_WORKERS = 6
TODAY = date.today().strftime("%Y-%m-%d")


def _effective_today():
    """Return the date through which daily data is expected to be available.

    Before 18:00 local time, today's trading data may not have been
    published yet, so the expected latest date is yesterday (T-1).
    After 18:00, expect data through today.
    """
    now = datetime.now()
    if now.hour < 18:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d")


EFFECTIVE_TODAY = _effective_today()

# ── Incremental state tracking ──────────────────────────────────────────
STATE_FILE = Path(DATA_DIR) / ".incr_state.json"
_state_lock = threading.Lock()  # guards concurrent read-modify-write on state

# ── Per-stock date cache (avoids scanning 3000+ parquet files) ─────────
CYQ_CACHE_FILE = Path(DATA_DIR) / ".cyq_cache.json"
_cyq_cache_lock = threading.Lock()


def _load_cyq_cache():
    """Return {stock_code: max_date_str} from the per-stock date cache."""
    if CYQ_CACHE_FILE.exists():
        try:
            with open(CYQ_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cyq_cache(cache):
    """Atomically write the per-stock date cache."""
    with _cyq_cache_lock:
        tmp = CYQ_CACHE_FILE.with_suffix(f".cyq_cache.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        os.replace(tmp, CYQ_CACHE_FILE)


def _load_state():
    """Load incremental update state tracking from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_state(state):
    """Save incremental update state tracking (caller must hold _state_lock)."""
    tmp = STATE_FILE.with_suffix(f".state.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    # os.replace is atomic on Unix and won't fail if tmp was already renamed
    os.replace(tmp, STATE_FILE)


def _get_effective_overlap(name, base_overlap):
    """Return extended overlap if the last fetch was stale (server lagging).

    When a dataset was marked *pending* (server didn't reach the expected
    date last run), widen the overlap window so the stale range is
    re-fetched and replaced via the existing merge + dedup logic.
    """
    state = _load_state()
    entry = state.get(name, {})
    if entry.get("pending") and entry.get("last_fetch_max"):
        try:
            last_actual = datetime.strptime(entry["last_fetch_max"], "%Y-%m-%d")
            today_dt = datetime.strptime(TODAY, "%Y-%m-%d")
            gap = (today_dt - last_actual).days
            if gap > base_overlap:
                return gap + base_overlap
        except Exception:
            pass
    return base_overlap


def _record_state(name, fetch_max):
    """Record the result of an incremental fetch for a dataset.

    *fetch_max* is the actual max date in the data after the fetch
    (None if the fetch failed or no data was returned).

    Uses EFFECTIVE_TODAY (which respects the 18:00 cutoff) to determine
    whether the dataset is behind: before 18:00 data through yesterday
    is considered current; after 18:00 data through today is expected.
    """
    with _state_lock:
        state = _load_state()
        entry = state.get(name, {})
        entry["last_run"] = TODAY

        if fetch_max:
            entry["last_fetch_max"] = fetch_max
            try:
                max_dt = datetime.strptime(fetch_max, "%Y-%m-%d")
                effective_dt = datetime.strptime(EFFECTIVE_TODAY, "%Y-%m-%d")
                pending = max_dt < effective_dt
                entry["pending"] = pending
                if pending:
                    if not entry.get("pending_since"):
                        entry["pending_since"] = TODAY
                else:
                    entry.pop("pending_since", None)
            except Exception:
                pass
        else:
            entry["pending"] = entry.get("pending", False)

        state[name] = entry
        _save_state(state)

# ── Performance timing ─────────────────────────────────────────────────────
_timings = {}         # dataset_name -> list of {phase, elapsed, ...}
_timing_lock = threading.Lock()

def _add_timing(dataset, phase, elapsed, **kw):
    """Record a timing measurement (thread-safe)."""
    entry = {"phase": phase, "elapsed": round(elapsed, 3)}
    entry.update(kw)
    with _timing_lock:
        _timings.setdefault(dataset, []).append(entry)

def _print_timing_report():
    """Print a formatted per-dataset timing report."""
    if not _timings:
        return
    log_print(f"\n{'='*60}")
    log_print("[timing] Per-dataset timing report")
    log_print(f"{'Dataset':<28} {'Phase':<22} {'Elapsed':>9}  {'Details'}")
    log_print("-" * 85)

    ds_totals = {}
    for ds_name in sorted(_timings):
        entries = _timings[ds_name]
        ds_total = sum(e["elapsed"] for e in entries)
        ds_totals[ds_name] = ds_total
        phases = {}
        for e in entries:
            ph = e["phase"]
            phases.setdefault(ph, {"elapsed": 0, "count": 0, "rows": 0, "calls": 0})
            phases[ph]["elapsed"] += e["elapsed"]
            phases[ph]["count"] += 1
            phases[ph]["rows"] += e.get("rows", 0)
            phases[ph]["calls"] += e.get("calls", 0)
        for ph, info in sorted(phases.items()):
            detail_parts = []
            if info["calls"]:
                detail_parts.append(f"{info['calls']} calls")
            if info["rows"]:
                detail_parts.append(f"{info['rows']:,} rows")
            detail = ", ".join(detail_parts) if detail_parts else ""
            cnt = f" x{info['count']}" if info["count"] > 1 else ""
            log_print(f"  {ds_name:<26} {ph + cnt:<22} {info['elapsed']:>7.1f}s  {detail}")

    total = sum(ds_totals.values())
    log_print("-" * 85)
    log_print(f"  {'TOTAL':<26} {'':<22} {total:>7.1f}s")

    slowest = sorted(ds_totals.items(), key=lambda x: -x[1])[:5]
    if slowest:
        log_print(f"\n[timing] Top 5 slowest datasets:")
        for ds_name, ds_total in slowest:
            pct = ds_total / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 5)
            log_print(f"  {ds_name:<26} {ds_total:>7.1f}s ({pct:5.1f}%) {bar}")

# ── Utility: auto-detect date column ──────────────────────────────────────
_DATE_CANDIDATES = ["trade_date", "end_date", "date", "Date", "f_ann_date",
                    "report_date", "ann_date"]


def _detect_date_col(columns):
    """Return the first matching date column from the schema."""
    for c in _DATE_CANDIDATES:
        if c in columns:
            return c
    return None


def get_max_date(filepath):
    """Read only the date column(s) from a parquet file and return the max date.

    Returns (date_col_name, max_date_str) or (None, None) on failure.
    """
    try:
        schema = pq.read_schema(filepath)
        available = set(schema.names)
        date_col = _detect_date_col(available)
        if date_col is None:
            return None, None
        table = pq.read_table(filepath, columns=[date_col])
        series = pd.Series(table.column(0).to_pandas())
        if series.empty:
            return date_col, None
        max_val = series.max()
        if pd.isna(max_val):
            return date_col, None
        # Normalise to string
        if hasattr(max_val, "strftime"):
            return date_col, max_val.strftime("%Y-%m-%d")
        return date_col, str(max_val)[:10]
    except Exception as e:
        log_print(f"  [warn] get_max_date({filepath}): {e}")
        return None, None


def get_real_max_date(filepath):
    """Like get_max_date but excludes backfilled rows when is_backfill column exists.

    This prevents the backfill-inflated max date from suppressing the next
    incremental fetch — the state tracking must reflect the server's actual
    data, not the forward-filled placeholders.
    """
    try:
        schema = pq.read_schema(filepath)
        available = set(schema.names)
        date_col = _detect_date_col(available)
        if date_col is None:
            return None, None
        if "is_backfill" in available:
            table = pq.read_table(filepath, columns=[date_col, "is_backfill"])
            df = table.to_pandas()
            real = df[df["is_backfill"] != True][date_col]
            if real.empty:
                return date_col, None
            max_val = real.max()
        else:
            table = pq.read_table(filepath, columns=[date_col])
            series = pd.Series(table.column(0).to_pandas())
            if series.empty:
                return date_col, None
            max_val = series.max()
        if pd.isna(max_val):
            return date_col, None
        if hasattr(max_val, "strftime"):
            return date_col, max_val.strftime("%Y-%m-%d")
        return date_col, str(max_val)[:10]
    except Exception as e:
        log_print(f"  [warn] get_real_max_date({filepath}): {e}")
        return None, None


def compute_incremental_range(filepath, default_start, overlap_days=OVERLAP_DAYS):
    """Determine the date range for an incremental fetch.

    Returns (date_col, new_start_str, end_str, existing_df_or_none).
    existing_df_or_none is None if the file doesn't exist or can't be read.
    """
    path = Path(filepath)
    if not path.exists():
        return None, default_start, TODAY, None

    date_col, max_str = get_max_date(path)
    if max_str is None:
        return date_col, default_start, TODAY, None

    max_dt = datetime.strptime(max_str, "%Y-%m-%d")
    # Step back by overlap_days, but never go before default_start
    new_start_dt = max_dt - timedelta(days=overlap_days)
    default_dt = datetime.strptime(default_start, "%Y-%m-%d")
    if new_start_dt < default_dt:
        new_start_dt = default_dt

    new_start = new_start_dt.strftime("%Y-%m-%d")
    return date_col, new_start, TODAY, None


# ── Generic API helpers (for per-stock incremental) ───────────────────────
def _api_page(endpoint, start_time, end_time, page, page_size,
              api_key, extra_payload=None, retries=3):
    """Fetch a single page from the diemeng API."""
    url = f"{BASE_URL}/{endpoint}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {
        "start_time": start_time,
        "end_time": end_time,
        "page": page,
        "page_size": page_size,
    }
    if extra_payload:
        payload.update(extra_payload)

    for attempt in range(retries):
        try:
            limiter = rate_limiter()
            limiter.acquire(endpoint)
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            if result["code"] != 200:
                msg = result.get("msg", str(result))
                raise RuntimeError(f"API Error: code={result['code']}, msg={msg}")
            data = result["data"]
            return data["list"], data["total"]
        except Exception as e:
            is_429 = "429" in str(e) or "频繁" in str(e)
            if attempt < retries - 1:
                delay = (15 * (2 ** attempt)) if is_429 else (2 ** attempt)
                time.sleep(delay)
            else:
                raise


def _api_range(endpoint, start_time, end_time, api_key,
               extra_payload=None, page_size=10000):
    """Fetch all rows for a date range, splitting if total > 100K.

    Returns a list of dicts.
    """
    page0, total = _api_page(endpoint, start_time, end_time, 0,
                              page_size, api_key, extra_payload)
    if not page0:
        return []

    if total <= 100000:
        all_data = list(page0)
        page = 1
        while len(all_data) < total:
            b, _ = _api_page(endpoint, start_time, end_time, page,
                             page_size, api_key, extra_payload)
            if not b:
                break
            all_data.extend(b)
            page += 1
        return all_data

    # Range too large — split in half
    start_dt = datetime.strptime(start_time, "%Y-%m-%d")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d")
    if start_dt == end_dt:
        return list(page0)
    mid = start_dt + (end_dt - start_dt) // 2
    if mid == start_dt:
        return list(page0)
    mid_str = mid.strftime("%Y-%m-%d")
    next_str = (mid + timedelta(days=1)).strftime("%Y-%m-%d")
    left = _api_range(endpoint, start_time, mid_str, api_key,
                      extra_payload, page_size)
    right = _api_range(endpoint, next_str, end_time, api_key,
                       extra_payload, page_size)
    return left + right


# ── Safe function caller ──────────────────────────────────────────────────
def _call_fetch(fn, start_date=None, end_date=None, workers=DEFAULT_WORKERS,
                cleanup=True, resume=False, **extra):
    """Call a fetch function, adapting parameters to what it accepts."""
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = {}
    # Map date parameters — some functions use start_time/end_time
    if "start_date" in params and start_date:
        kwargs["start_date"] = start_date
    elif "start_time" in params and start_date:
        kwargs["start_time"] = start_date
    if "end_date" in params and end_date:
        kwargs["end_date"] = end_date
    elif "end_time" in params and end_date:
        kwargs["end_time"] = end_date
    if "workers" in params:
        kwargs["workers"] = workers
    if "cleanup" in params:
        kwargs["cleanup"] = cleanup
    if "resume" in params:
        kwargs["resume"] = resume
    kwargs.update(extra)
    return fn(**kwargs)


# ── Merge helpers ─────────────────────────────────────────────────────────
def _safe_read_parquet(filepath):
    """Read a parquet file, return empty DataFrame on failure."""
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        log_print(f"  [warn] Cannot read {filepath}: {e}")
        return pd.DataFrame()


def merge_and_save(existing_path, new_df, dedup_keys, sort_cols,
                   date_col=None, strict_dedup=False):
    """Merge new data into an existing parquet file.

    If *existing_path* exists, read it, concatenate with *new_df*,
    drop duplicates (keeping last = newest), sort, and save.
    Otherwise just save *new_df*.

    When *strict_dedup* is True, deduplication uses *dedup_keys* as
    the subset, so data corrections in the overlap window properly
    replace old rows.  When False, only fully identical rows are
    deduped (safe for datasets with multiple valid rows per key, e.g.
    holder_number which has several ann_date per end_date).

    Returns (rows_before, rows_after, rows_new).
    """
    existing_path = Path(existing_path)
    if existing_path.exists():
        existing = _safe_read_parquet(existing_path)
    else:
        existing = pd.DataFrame()

    if new_df is None or new_df.empty:
        return len(existing), len(existing), 0

    before = len(existing)

    if existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
        if strict_dedup:
            available_keys = [k for k in dedup_keys if k in merged.columns]
            merged = merged.drop_duplicates(subset=available_keys, keep="last")
        else:
            merged = merged.drop_duplicates(keep="last")

    # Clean up internal sort column from original fetch scripts
    if "_sort_code" in merged.columns:
        merged = merged.drop(columns=["_sort_code"])

    # Sort
    available_sort = [c for c in sort_cols if c in merged.columns]
    if available_sort:
        merged = merged.sort_values(available_sort).reset_index(drop=True)

    after = len(merged)
    rows_new = after - before

    # Atomic write via temp file
    tmp = existing_path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(existing_path)
    return before, after, rows_new


# ── Backfill helpers ────────────────────────────────────────────────────────
def _backfill_missing_dates(filepath, date_col, today_str):
    """Forward-fill missing trading days for daily-frequency datasets.

    When the server hasn't updated for the latest trading days, copy the
    last available data forward to fill the gap.  Each filled row is
    marked ``is_backfill=True`` so downstream consumers can distinguish
    real data from forward-filled placeholders.

    When the server later updates, the merge+dedup logic (keep='last')
    in update_consolidated naturally replaces backfilled rows with
    real data, since both share the same (trade_date, stock_code) key.

    Returns True if any backfill rows were added.
    """
    cal_path = Path(DATA_DIR) / "calendar.parquet"
    if not cal_path.exists():
        log_print("  [backfill] calendar.parquet not found, cannot determine trading days")
        return False

    cal = pd.read_parquet(cal_path)
    trading_days = sorted(cal[cal["is_open"] == 1]["date"].tolist())

    df = pd.read_parquet(filepath)
    if df.empty or date_col not in df.columns:
        return False

    # Ensure is_backfill column: missing or NaN means real data
    if "is_backfill" not in df.columns:
        df["is_backfill"] = False
    else:
        df["is_backfill"] = df["is_backfill"].fillna(False).astype(bool)

    # Per-stock last available date
    if "stock_code" not in df.columns:
        log_print("  [backfill] No stock_code column, cannot backfill per-stock")
        return False

    last_per_stock = df.groupby("stock_code")[date_col].max().reset_index()
    last_per_stock.columns = ["stock_code", "last_date"]

    # Which stocks are behind today?
    behind = last_per_stock[last_per_stock["last_date"] < today_str]
    if behind.empty:
        return False

    # Get full rows for each stock's last date via a single merge
    # (avoids slow per-stock boolean indexing on large DataFrames)
    last_data = df.merge(
        behind,
        left_on=["stock_code", date_col],
        right_on=["stock_code", "last_date"],
        how="inner",
    )
    last_data = last_data.drop(columns=["last_date"])

    if last_data.empty:
        return False

    # For each missing trading day, create backfill rows
    fill_parts = []
    global_max_behind = behind["last_date"].max()
    for d in trading_days:
        if d <= global_max_behind or d > today_str:
            continue
        eligible_codes = behind.loc[behind["last_date"] < d, "stock_code"]
        part = last_data[last_data["stock_code"].isin(eligible_codes)].copy()
        if part.empty:
            continue
        part[date_col] = d
        part["is_backfill"] = True
        fill_parts.append(part)

    if not fill_parts:
        return False

    fill_df = pd.concat(fill_parts, ignore_index=True)
    df = pd.concat([df, fill_df], ignore_index=True)

    if "stock_code" in df.columns:
        df = df.sort_values(["stock_code", date_col]).reset_index(drop=True)

    # Atomic save
    tmp = Path(filepath).with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(Path(filepath))

    n_stocks = fill_df["stock_code"].nunique()
    n_days = len(fill_parts)
    log_print(f"  [backfill] Filled {len(fill_df)} rows for {n_stocks} stocks "
              f"across {n_days} missing trading day(s)")
    return True


# ── Per-stock helpers ─────────────────────────────────────────────────────
_EXCLUDE_PREFIXES = ("688", "300", "301", "92", "8", "4")
_ST_PATTERN = r"ST|PT|\*ST"


def _filter_main_board_stocks():
    """Return list of main-board non-ST stock codes from stock_list.parquet."""
    sl_path = Path(DATA_DIR) / "stock_list.parquet"
    if not sl_path.exists():
        log_print("  [warn] stock_list.parquet not found, cannot filter stocks")
        return []
    df = pd.read_parquet(sl_path)
    code_num = df["stock_code"].str.extract(r"^(\d+)").iloc[:, 0]
    mask = ~code_num.str.startswith(_EXCLUDE_PREFIXES)
    mask &= ~df["name"].str.contains(_ST_PATTERN, na=False)
    mask &= df["list_status"] == "L"
    return df.loc[mask, "stock_code"].tolist()


# Batch size for per-stock multi-stock API calls.
# Each stock produces ~50-200 rows/day; for a 5-day window with 50 stocks
# that's ~12,500-50,000 rows, well under the 100K pagination limit.
_PER_STOCK_BATCH_SIZE = 500


def _merge_per_stock_batch(batch_results, out_dir, date_col):
    """Split a batch API result by stock_code and merge each into its file.

    *batch_results* is a list of dict rows from the API (all stocks mixed).
    Returns (results_list, cache_updates_dict) where cache_updates maps
    stock_code -> new_max_date_str for the cyq_cache.
    """
    if not batch_results:
        return [], {}
    df_all = pd.DataFrame(batch_results)
    if df_all.empty or "stock_code" not in df_all.columns:
        return [], {}

    results = []
    cache_updates = {}
    for code, group in df_all.groupby("stock_code"):
        out_file = Path(out_dir) / f"{code}.parquet"
        existing = _safe_read_parquet(out_file) if out_file.exists() else pd.DataFrame()
        old_rows = len(existing)

        new_chunk = group.drop(columns=["stock_code"], errors="ignore")
        if existing.empty:
            merged = new_chunk
        else:
            merged = pd.concat([existing, new_chunk], ignore_index=True)
            merged = merged.drop_duplicates(keep="last")

        if date_col and date_col in merged.columns:
            merged = merged.sort_values(date_col).reset_index(drop=True)
            # Record new max date for cache
            max_val = merged[date_col].max()
            if not pd.isna(max_val):
                max_str = str(max_val)[:10] if hasattr(max_val, "strftime") \
                          else max_val.strftime("%Y-%m-%d") if hasattr(max_val, "strftime") \
                          else str(max_val)[:10]
                cache_updates[code] = max_str

        tmp = out_file.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(out_file)
        results.append((code, old_rows, len(merged), len(merged) - old_rows))
    return results, cache_updates


def update_per_stock_dataset(name, out_subdir, endpoint, start_date,
                             end_date, date_col, workers, dry_run,
                             extra_payload=None):
    """Update per-stock parquet files using multi-stock batched API calls.

    Both new stocks (full history) and incremental stocks (recent window)
    are batched into groups of _PER_STOCK_BATCH_SIZE, leveraging the API's
    ability to accept multiple stock_codes in a single request.  This keeps
    daily updates fast: ~3000 stocks need only ~60 API calls instead of
    one per stock.
    """
    stocks = _filter_main_board_stocks()
    if not stocks:
        log_print(f"[{name}] No stocks to process")
        return

    out_dir = Path(DATA_DIR) / out_subdir
    out_dir.mkdir(exist_ok=True)
    api_key = load_api_key()

    # Categorise stocks — use cache to avoid opening 3000+ parquet files
    cache = _load_cyq_cache()
    cache_updated = False
    new_batch = []          # (code, full_start_date)  — never fetched
    incr_batch = []         # (code, inc_start_date)   — needs catch-up
    uptodate = 0

    for code in stocks:
        f = out_dir / f"{code}.parquet"
        if not f.exists():
            new_batch.append((code, start_date))
            continue
        # Use cached date if available, otherwise read from parquet
        max_s = cache.get(code)
        if max_s is None:
            dc, max_s = get_max_date(f)
            if max_s is not None:
                cache[code] = max_s
                cache_updated = True
        if max_s is None:
            new_batch.append((code, start_date))
            continue
        max_dt = datetime.strptime(max_s, "%Y-%m-%d")
        effective_overlap = _get_effective_overlap(name, OVERLAP_DAYS)
        inc_start_dt = max_dt - timedelta(days=effective_overlap)
        default_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if inc_start_dt < default_dt:
            inc_start_dt = default_dt
        inc_start = inc_start_dt.strftime("%Y-%m-%d")

        if max_s >= end_date or max_s >= TODAY:
            uptodate += 1
            continue
        incr_batch.append((code, inc_start))

    if cache_updated:
        _save_cyq_cache(cache)

    log_print(f"[{name}] {len(stocks)} stocks | "
              f"{len(incr_batch)} incremental | "
              f"{len(new_batch)} new | "
              f"{uptodate} up-to-date")

    if not new_batch and not incr_batch:
        log_print(f"[{name}] All stocks up to date")
        return

    if dry_run:
        log_print(f"[{name}] DRY-RUN: would fetch {len(new_batch)} new + "
                  f"{len(incr_batch)} incremental stocks")
        if incr_batch:
            sample = incr_batch[:3]
            for code, s in sample:
                log_print(f"  incr: {code}: {s} ~ {end_date}")
            log_print(f"  ... {len(incr_batch)} stocks in "
                      f"{(len(incr_batch) + _PER_STOCK_BATCH_SIZE - 1) // _PER_STOCK_BATCH_SIZE} batches")
        if new_batch:
            sample = new_batch[:3]
            for code, s in sample:
                log_print(f"  new:  {code}: {s} ~ {end_date}")
            log_print(f"  ... {len(new_batch)} stocks in "
                      f"{(len(new_batch) + _PER_STOCK_BATCH_SIZE - 1) // _PER_STOCK_BATCH_SIZE} batches")
        return

    limiter = rate_limiter()
    total_added = 0
    processed = 0
    lock = threading.Lock()

    def _process_batches(stock_list, phase_label):
        """Fetch and merge a list of (code, start) tuples in multi-stock batches."""
        nonlocal processed, total_added

        batches = []
        for i in range(0, len(stock_list), _PER_STOCK_BATCH_SIZE):
            batch = stock_list[i:i + _PER_STOCK_BATCH_SIZE]
            min_start = min(s for _, s in batch)
            batches.append(([c for c, _ in batch], min_start))

        n_batches = len(batches)
        log_print(f"[{name}] Phase {phase_label}: {len(stock_list)} stocks in "
                  f"{n_batches} batches (batch_size={_PER_STOCK_BATCH_SIZE})")

        completed = 0
        batch_cache_updates = {}  # collect cache updates from all batches
        if workers > 1 and n_batches > 1:
            # Helper: time an _api_range call in a pool thread
            def _timed_fetch(codes_arg, batch_start_arg):
                ep_inner = (extra_payload.copy() if extra_payload else {})
                ep_inner["stock_code"] = codes_arg
                t0 = time.perf_counter()
                rows_result = _api_range(endpoint, batch_start_arg, end_date,
                                         api_key, ep_inner, 10000)
                _add_timing(name, "fetch", time.perf_counter() - t0,
                           rows=len(rows_result), calls=1)
                return rows_result

            with ThreadPoolExecutor(max_workers=min(workers, n_batches)) as pool:
                futures = {}
                for codes, batch_start in batches:
                    fut = pool.submit(_timed_fetch, codes, batch_start)
                    futures[fut] = (codes, batch_start)

                for fut in as_completed(futures):
                    codes, batch_start = futures[fut]
                    try:
                        rows = fut.result()
                        t0 = time.perf_counter()
                        merge_results, cache_upd = _merge_per_stock_batch(rows, out_dir, date_col)
                        _add_timing(name, "merge", time.perf_counter() - t0,
                                   rows=len(rows), stocks=len(merge_results))
                        with lock:
                            completed += 1
                            batch_added = sum(r[3] for r in merge_results)
                            total_added += batch_added
                            processed += len(merge_results)
                            batch_cache_updates.update(cache_upd)
                        log_print(f"[{name}] [{phase_label} {completed}/{n_batches}] "
                                  f"{len(codes)} stocks, {len(rows)} rows, "
                                  f"+{batch_added} new | "
                                  f"tokens={limiter.stats['tokens_available']}")
                    except Exception as e:
                        with lock:
                            completed += 1
                        log_print(f"[{name}] [{phase_label} {completed}/{n_batches}] "
                                  f"{len(codes)} stocks FAILED: {e}")
        else:
            for bi, (codes, batch_start) in enumerate(batches):
                ep = (extra_payload.copy() if extra_payload else {})
                ep["stock_code"] = codes
                try:
                    t0 = time.perf_counter()
                    rows = _api_range(endpoint, batch_start, end_date,
                                      api_key, ep, 10000)
                    _add_timing(name, "fetch", time.perf_counter() - t0,
                               rows=len(rows), calls=1)
                    t0 = time.perf_counter()
                    merge_results, cache_upd = _merge_per_stock_batch(rows, out_dir, date_col)
                    _add_timing(name, "merge", time.perf_counter() - t0,
                               rows=len(rows), stocks=len(merge_results))
                    completed += 1
                    batch_added = sum(r[3] for r in merge_results)
                    total_added += batch_added
                    processed += len(merge_results)
                    batch_cache_updates.update(cache_upd)
                    log_print(f"[{name}] [{phase_label} {completed}/{n_batches}] "
                              f"{len(codes)} stocks, {len(rows)} rows, "
                              f"+{batch_added} new | "
                              f"tokens={limiter.stats['tokens_available']}")
                except Exception as e:
                    completed += 1
                    log_print(f"[{name}] [{phase_label} {completed}/{n_batches}] "
                              f"{len(codes)} stocks FAILED: {e}")
        return batch_cache_updates

    all_cache_updates = {}
    if new_batch:
        all_cache_updates.update(_process_batches(new_batch, "new"))
    if incr_batch:
        all_cache_updates.update(_process_batches(incr_batch, "incr"))

    # Persist cache updates
    if all_cache_updates:
        cache = _load_cyq_cache()
        cache.update(all_cache_updates)
        _save_cyq_cache(cache)

    log_print(f"[{name}] Done: {processed} stocks processed, "
              f"{total_added} new rows total")

    # Record state — sample a few stock files to detect the actual max date
    if not dry_run and (new_batch or incr_batch):
        sample_max = None
        for f in sorted(out_dir.glob("*.parquet"))[:200]:
            _, m = get_max_date(f)
            if m and (sample_max is None or m > sample_max):
                sample_max = m
        _record_state(name, sample_max)
    elif dry_run:
        pass  # no state change on dry run
    else:
        _record_state(name, TODAY)  # all up to date

    # Update .done marker
    done_file = out_dir / ".done"
    done_file.write_text(TODAY)


# ── Consolidated dataset updater ──────────────────────────────────────────
def update_consolidated(name, fetch_fn, filepath, date_col, dedup_keys,
                          sort_cols, default_start, workers, dry_run,
                          extra_params=None, strict_dedup=False,
                          overlap_days=None, backfill=False):
    """Incrementally update a consolidated (single-parquet) dataset.

    1. Read existing file, find max date
    2. Compute incremental start date (with overlap)
    3. Call the fetch function for the narrow range
    4. Merge new data into existing file
    """
    path = Path(filepath)
    if not path.exists():
        log_print(f"[{name}] Output file not found ({filepath}) — "
                  f"full fetch needed, run scripts/main.py first")
        return {"name": name, "status": "skip", "reason": "no existing file"}

    # Use get_real_max_date so backfill-inflated dates don't suppress
    # the next incremental fetch (state tracking must reflect server reality)
    dc, max_str = get_real_max_date(path)
    effective_dc = dc or date_col

    if max_str is None:
        log_print(f"[{name}] Cannot determine max date — "
                  f"full fetch needed, run scripts/main.py --force")
        return {"name": name, "status": "skip", "reason": "no date column"}

    # Compute incremental range (extend overlap if last fetch was stale)
    max_dt = datetime.strptime(max_str, "%Y-%m-%d")
    base_overlap = overlap_days if overlap_days is not None else OVERLAP_DAYS
    effective_overlap = _get_effective_overlap(name, base_overlap)
    inc_start_dt = max_dt - timedelta(days=effective_overlap)
    default_dt = datetime.strptime(default_start, "%Y-%m-%d")
    if inc_start_dt < default_dt:
        inc_start_dt = default_dt
    inc_start = inc_start_dt.strftime("%Y-%m-%d")

    # Check if already up to date
    end_dt = datetime.strptime(TODAY, "%Y-%m-%d")
    if max_dt >= end_dt:
        log_print(f"[{name}] Up to date (max={max_str})")
        _record_state(name, max_str)
        return {"name": name, "status": "uptodate", "max_date": max_str}

    log_print(f"[{name}] Existing max date: {max_str} | "
              f"Fetching: {inc_start} ~ {TODAY}")

    if dry_run:
        log_print(f"[{name}] DRY-RUN: would fetch {inc_start} ~ {TODAY}")
        return {"name": name, "status": "dry_run", "range": f"{inc_start}~{TODAY}"}

    # Fetch new data to a TEMP file so the original is never overwritten
    tmp_output = str(path.parent / f"_incr_{name}.tmp.parquet")
    t_fetch = time.perf_counter()
    try:
        new_df = _call_fetch(
            fetch_fn,
            start_date=inc_start,
            end_date=TODAY,
            workers=workers,
            cleanup=True,
            resume=False,
            output=tmp_output,
            **(extra_params or {})
        )
    except Exception as e:
        _add_timing(name, "fetch", time.perf_counter() - t_fetch, error=str(e))
        log_print(f"[{name}] Fetch FAILED: {e}")
        Path(tmp_output).unlink(missing_ok=True)
        _record_state(name, None)  # fetch failed, mark pending
        return {"name": name, "status": "error", "error": str(e)}
    _add_timing(name, "fetch", time.perf_counter() - t_fetch)

    # Read from temp file (more reliable than return value)
    tmp_path = Path(tmp_output)
    if tmp_path.exists():
        new_df = _safe_read_parquet(tmp_output)
        tmp_path.unlink(missing_ok=True)
    elif new_df is not None and not (hasattr(new_df, 'empty') and new_df.empty):
        pass  # use the returned DataFrame
    else:
        log_print(f"[{name}] No new data returned")
        _record_state(name, max_str)  # server may be behind
        return {"name": name, "status": "uptodate", "max_date": max_str}

    if new_df is None or (hasattr(new_df, 'empty') and new_df.empty):
        log_print(f"[{name}] No new data returned")
        _record_state(name, max_str)  # server may be behind
        return {"name": name, "status": "uptodate", "max_date": max_str}

    # Now merge: original file is intact, new_df has the incremental data
    t_merge = time.perf_counter()
    before, after, added = merge_and_save(
        filepath, new_df, dedup_keys, sort_cols, effective_dc,
        strict_dedup=strict_dedup,
    )
    _add_timing(name, "merge", time.perf_counter() - t_merge,
               rows_before=before, rows_added=added)

    # Check new max date (use get_real_max_date so state tracks server reality)
    _, new_max = get_real_max_date(path)
    log_print(f"[{name}] Merge: {before} → {after} rows (+{added}) | "
              f"max_date: {max_str} → {new_max}")

    # Record state for next run's overlap logic
    _record_state(name, new_max)

    # Backfill missing trading days for daily-frequency datasets
    # (state is recorded BEFORE backfill so pending flag reflects server reality)
    # Uses EFFECTIVE_TODAY (respects 18:00 cutoff): before 18:00 we only
    # expect data through yesterday, so don't backfill today's date yet.
    if backfill and new_max and new_max < EFFECTIVE_TODAY:
        _backfill_missing_dates(filepath, effective_dc, EFFECTIVE_TODAY)

    return {
        "name": name,
        "status": "updated",
        "rows_before": before,
        "rows_after": after,
        "rows_added": added,
        "old_max": max_str,
        "new_max": new_max,
    }


# ── Dataset Registry ──────────────────────────────────────────────────────
# Lazy imports: each entry's "module" / "fn_name" is imported on first use.
# This keeps startup fast and avoids loading all fetch scripts at once.

def _import_fn(module_name, fn_name):
    """Lazily import a fetch function from a module."""
    mod = __import__(module_name, fromlist=[fn_name])
    return getattr(mod, fn_name)


# Registry entries:
#   name        — short identifier (also used for --dataset filter)
#   module      — python module name under scripts/
#   fn_name     — function name in that module
#   file        — output parquet filename (under data/)
#   date_col    — primary date column for finding max date
#   dedup       — columns for deduplication
#   sort        — columns for final sort
#   start       — default start date (fallback if file doesn't exist)
#   type        — "consolidated" | "per_stock" | "reference"
#   extra       — extra kwargs passed to the fetch function

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
        "backfill": True,
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
        "backfill": True,
    },
    {
        "name": "finance",
        "module": "fetch_finance", "fn_name": "fetch_finance",
        "file": "finance.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2020-01-01",
        "type": "consolidated",
        "strict_dedup": True,
        "backfill": True,
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
        "backfill": True,
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
        "backfill": True,
    },
    # ═══ Day-by-day (consolidated output) ═══
    {
        "name": "dragon_tiger",
        "module": "fetch_dragon_tiger", "fn_name": "fetch_dragon_tiger",
        "file": "dragon_tiger.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
        "backfill": True,
    },
    {
        "name": "top_list",
        "module": "fetch_top_list", "fn_name": "fetch_top_list",
        "file": "top_list.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
        "backfill": True,
    },
    {
        "name": "limit_up",
        "module": "fetch_limit_up", "fn_name": "fetch_limit_up",
        "file": "limit_up.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
        "backfill": True,
    },
    {
        "name": "limit_list",
        "module": "fetch_limit_list", "fn_name": "fetch_limit_list",
        "file": "limit_list.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
        "backfill": True,
    },
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
        "backfill": True,
    },
    {
        "name": "index_weight",
        "module": "fetch_index_weight", "fn_name": "fetch_index_weight",
        "file": "index_weight.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "index_code", "stock_code"],
        "sort": ["index_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
        "overlap_days": 35,
    },
    {
        "name": "ths_hot",
        "module": "fetch_ths_hot", "fn_name": "fetch_ths_hot",
        "file": "ths_hot.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "market"],
        "sort": ["trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    # ═══ Weekly / Monthly K-line ═══
    {
        "name": "kline_weekly",
        "module": "fetch_kline", "fn_name": "fetch_kline_weekly",
        "file": "kline_weekly.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    {
        "name": "kline_monthly",
        "module": "fetch_kline", "fn_name": "fetch_kline_monthly",
        "file": "kline_monthly.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    {
        "name": "kline_adj_weekly",
        "module": "fetch_kline", "fn_name": "fetch_kline_adj_weekly",
        "file": "kline_adj_weekly.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    {
        "name": "kline_adj_monthly",
        "module": "fetch_kline", "fn_name": "fetch_kline_adj_monthly",
        "file": "kline_adj_monthly.parquet",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["stock_code", "trade_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "strict_dedup": True,
    },
    # ═══ Quarterly / report-period data ═══
    {
        "name": "holder_number",
        "module": "fetch_holder_number", "fn_name": "fetch_holder_number",
        "file": "holder_number.parquet",
        "date_col": "end_date",
        "dedup": ["end_date", "stock_code"],
        "sort": ["stock_code", "end_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "overlap_days": 95,
    },
    {
        "name": "pledge_stat",
        "module": "fetch_pledge_stat", "fn_name": "fetch_pledge_stat",
        "file": "pledge_stat.parquet",
        "date_col": "end_date",
        "dedup": ["end_date", "stock_code"],
        "sort": ["stock_code", "end_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "overlap_days": 95,
    },
    {
        "name": "financial_indicator",
        "module": "fetch_financial", "fn_name": "fetch_financial_indicator",
        "file": "financial_indicator.parquet",
        "date_col": "end_date",
        "dedup": ["end_date", "stock_code"],
        "sort": ["stock_code", "end_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "overlap_days": 95,
    },
    {
        "name": "income",
        "module": "fetch_income", "fn_name": "fetch_income",
        "file": "income.parquet",
        "date_col": "end_date",
        "dedup": ["end_date", "stock_code"],
        "sort": ["stock_code", "end_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "overlap_days": 95,
    },
    {
        "name": "balancesheet",
        "module": "fetch_balancesheet", "fn_name": "fetch_balancesheet",
        "file": "balancesheet.parquet",
        "date_col": "end_date",
        "dedup": ["end_date", "stock_code"],
        "sort": ["stock_code", "end_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "overlap_days": 95,
    },
    {
        "name": "cashflow",
        "module": "fetch_cashflow", "fn_name": "fetch_cashflow",
        "file": "cashflow.parquet",
        "date_col": "end_date",
        "dedup": ["end_date", "stock_code"],
        "sort": ["stock_code", "end_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "overlap_days": 95,
    },
    {
        "name": "report_rc",
        "module": "fetch_report_rc", "fn_name": "fetch_report_rc",
        "file": "report_rc.parquet",
        "date_col": "end_date",
        "dedup": ["end_date", "stock_code"],
        "sort": ["stock_code", "end_date"],
        "start": "2019-01-01",
        "type": "consolidated",
        "overlap_days": 95,
    },
    # ═══ Reference data (small, re-fetch entirely) ═══
    {
        "name": "calendar",
        "module": "fetch_calendar", "fn_name": "fetch_calendar",
        "file": "calendar.parquet",
        "date_col": "date",
        "dedup": ["date"],
        "sort": ["date"],
        "start": "2020-01-01",
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
    },
    # ═══ Per-stock data ═══
    {
        "name": "cyq_chips",
        "module": None, "fn_name": None,
        "file": "cyq_chips/.done",
        "date_col": "trade_date",
        "dedup": ["trade_date", "stock_code"],
        "sort": ["trade_date"],
        "start": "2019-01-01",
        "type": "per_stock",
        "out_subdir": "cyq_chips",
        "endpoint": "stock/cyq_chips",
    },
    # {
    #     "name": "history",
    #     "module": None, "fn_name": None,
    #     "file": "history/.done",
    #     "date_col": "trade_date",
    #     "dedup": ["trade_date", "stock_code"],
    #     "sort": ["trade_date"],
    #     "start": "2019-01-01",
    #     "type": "per_stock",
    #     "out_subdir": "history",
    #     "endpoint": "stock/history",
    # },
    # {
    #     "name": "min_adj",
    #     "module": None, "fn_name": None,
    #     "file": "min_adj/.done",
    #     "date_col": "trade_date",
    #     "dedup": ["trade_date", "stock_code"],
    #     "sort": ["trade_date"],
    #     "start": "2019-01-01",
    #     "type": "per_stock",
    #     "out_subdir": "min_adj",
    #     "endpoint": "stock/min_adj",
    #     "extra_payload": {"algo": "recursive"},
    # },
]


def _build_registry_index():
    """Return {name: entry} for quick lookup."""
    return {d["name"]: d for d in DATASETS}


# ── Main orchestrator ─────────────────────────────────────────────────────
def run_updates(datasets=None, exclude=None, overlap_days=OVERLAP_DAYS,
                workers=DEFAULT_WORKERS, parallel=False, dry_run=False,
                skip_per_stock=False, skip_reference=False):
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
    """
    global OVERLAP_DAYS
    OVERLAP_DAYS = overlap_days

    registry = _build_registry_index()
    exclude = set(exclude or [])

    # Determine which datasets to process
    if datasets:
        selected = [d for d in DATASETS if d["name"] in datasets]
        not_found = set(datasets) - {d["name"] for d in selected}
        if not_found:
            log_print(f"[incremental] Unknown datasets: {not_found}")
    else:
        # All datasets whose output file exists (already been fully fetched)
        selected = []
        for d in DATASETS:
            if d["name"] in exclude:
                continue
            if d["type"] == "per_stock":
                if skip_per_stock:
                    continue
                # Check if the .done marker or at least some stock files exist
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
                update_per_stock_dataset(
                    name=name,
                    out_subdir=d["out_subdir"],
                    endpoint=d["endpoint"],
                    start_date=d["start"],
                    end_date=TODAY,
                    date_col=d["date_col"],
                    workers=workers,
                    dry_run=dry_run,
                    extra_payload=d.get("extra_payload"),
                )
                return {"name": name, "status": "updated" if not dry_run else "dry_run"}

            if d["type"] == "reference":
                if skip_reference:
                    log_print(f"[{name}] Skipped (reference disabled)")
                    return {"name": name, "status": "skip", "reason": "reference_disabled"}
                # Reference data: fetch entirely, small enough to just replace
                if dry_run:
                    log_print(f"[{name}] DRY-RUN: would re-fetch entirely")
                    return {"name": name, "status": "dry_run"}
                fn = _import_fn(d["module"], d["fn_name"])
                try:
                    t_fetch = time.perf_counter()
                    new_df = _call_fetch(
                        fn,
                        start_date=d["start"],
                        end_date=TODAY,
                        workers=1,
                        cleanup=True,
                        resume=False,
                    )
                    _add_timing(name, "fetch", time.perf_counter() - t_fetch)
                    if new_df is not None and not new_df.empty:
                        dest = Path(DATA_DIR) / d["file"]
                        tmp = dest.with_suffix(".parquet.tmp")
                        new_df.to_parquet(tmp, index=False)
                        tmp.replace(dest)
                        log_print(f"[{name}] Re-fetched: {len(new_df)} rows → {dest}")
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
            )
        finally:
            _add_timing(name, "total", time.perf_counter() - t0)

    if parallel and len(selected) > 1:
        # Process reference datasets (calendar, stock_list) FIRST to avoid
        # race conditions: dragon_tiger, top_list, etc. read calendar.parquet
        # to determine trading days, so calendar must be up-to-date before
        # those datasets start fetching.
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
        for d in selected:
            try:
                r = _process_one(d)
                results.append(r)
            except Exception as e:
                log_print(f"[{d['name']}] UNEXPECTED ERROR: {e}")
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
                  f"{old_m} → {new_m}")
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
    _print_timing_report()

    return results


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Incremental daily data updater for lingqiData",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python incremental_update.py                          # Update all existing datasets
  python incremental_update.py --dataset daily          # Update only daily.parquet
  python incremental_update.py --dataset daily finance  # Update daily + finance
  python incremental_update.py --dry-run                # Preview what needs updating
  python incremental_update.py --parallel -w 6          # Run datasets in parallel
  python incremental_update.py --overlap 5              # 5-day overlap window
  python incremental_update.py --no-per-stock           # Skip cyq_chips/minute
  python incremental_update.py --list                   # List all known datasets
        """,
    )
    parser.add_argument("--dataset", "-d", nargs="+",
                        help="Specific dataset(s) to update (default: all existing)")
    parser.add_argument("--exclude", "-x", nargs="+",
                        help="Dataset(s) to exclude")
    parser.add_argument("--overlap", type=int, default=OVERLAP_DAYS,
                        help=f"Days of overlap for incremental range "
                             f"(default: {OVERLAP_DAYS})")
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
    parser.add_argument("--list", action="store_true",
                        help="List all known datasets and exit")

    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
