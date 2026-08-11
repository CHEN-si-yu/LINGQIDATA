"""
incremental_core.py — Core incremental update logic for lingqiData.

This module provides all the helper functions, API wrappers, merge logic,
and dataset-specific updaters used by incremental.py.  It is designed to
be imported by the thin entry-point script so that incremental.py stays as
simple as main.py.
"""

import inspect
import json
import os
import random
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import pyarrow.parquet as pq

from config import (
    load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print
)

# ── Globals ───────────────────────────────────────────────────────────────
OVERLAP_DAYS = 3  # days of overlap to catch data corrections
DEFAULT_WORKERS = 6
TODAY = date.today().strftime("%Y-%m-%d")

# ── Pre-flight canary check ───────────────────────────────────────────────
CANARY_ENDPOINTS = [
    "stock/daily",
    "stock/daily_adj",
    "stock/finance",
    "stock/margin_detail",
    "stock/main_fund_flow",
    "stock/cyq_chips",
    "stock/cyq_perf",
]
CANARY_NO_STOCK_FILTER = {
    "stock/margin_detail",
}
CANARY_LAG_DAYS = {
    "stock/margin_detail": 1,
}
CANARY_STOCKS = [
    "600000.SH",  # 浦发银行
    "600519.SH",  # 贵州茅台
    "000001.SZ",  # 平安银行
    "000858.SZ",  # 五粮液
    "600036.SH",  # 招商银行
    "000333.SZ",  # 美的集团
    "601318.SH",  # 中国平安
    "000651.SZ",  # 格力电器
    "600276.SH",  # 恒瑞医药
    "002415.SZ",  # 海康威视
]
CANARY_MIN_STOCKS = 5
DEFAULT_WAIT_INTERVAL = 300
DEFAULT_MAX_WAIT = 14400

# ── Concurrency gate ─────────────────────────────────────────────────────
_MAX_CONCURRENT = 3
_api_gate = threading.BoundedSemaphore(_MAX_CONCURRENT)

# ── Incremental state tracking ──────────────────────────────────────────
STATE_FILE = Path(DATA_DIR) / ".incr_state.json"
_state_lock = threading.Lock()

# ── Per-stock date cache ─────────────────────────────────────────────────
CYQ_CACHE_FILE = Path(DATA_DIR) / ".cyq_cache.json"
_cyq_cache_lock = threading.Lock()

# ── Performance timing ───────────────────────────────────────────────────
_timings = {}
_timing_lock = threading.Lock()

# ── Effective today (set by _last_trading_day) ───────────────────────────
EFFECTIVE_TODAY = date.today().strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════════
# Monitor
# ═══════════════════════════════════════════════════════════════════════════

def _monitor_loop(stop_event, interval=10.0):
    """Background thread that periodically logs rate-limiter stats."""
    limiter = rate_limiter()
    while not stop_event.wait(interval):
        s = limiter.stats
        log_print(f"[monitor] tokens={s['tokens_available']}/{s['max_rpm']} "
                  f"| calls={dict(s['endpoint_counts'])}")


# ═══════════════════════════════════════════════════════════════════════════
# Date helpers
# ═══════════════════════════════════════════════════════════════════════════

def _effective_today():
    """Return the date through which daily data is expected to be available."""
    now = datetime.now()
    if now.hour < 18:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d")


def _last_trading_day():
    """Return the last trading day for which data is expected to be available."""
    cal_path = Path(DATA_DIR) / "calendar.parquet"
    if cal_path.exists():
        try:
            cal = pd.read_parquet(cal_path)
            trading_days = cal[cal["is_open"] == 1]["date"]
            effective_str = _effective_today()
            last = trading_days[trading_days <= effective_str].max()
            if pd.notna(last):
                return str(last)[:10]
        except Exception:
            pass
    return _effective_today()


def _trading_day_offset(date_str, offset_days):
    """Return the date *offset_days* trading days before/after *date_str*."""
    cal_path = Path(DATA_DIR) / "calendar.parquet"
    if not cal_path.exists():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return (dt + timedelta(days=offset_days)).strftime("%Y-%m-%d")
    try:
        cal = pd.read_parquet(cal_path)
        trading_days = sorted(cal[cal["is_open"] == 1]["date"].tolist())
        trading_strs = [str(d)[:10] for d in trading_days]
        if offset_days == 0:
            return date_str
        if offset_days > 0:
            future = [d for d in trading_strs if d > date_str]
            if len(future) >= offset_days:
                return future[offset_days - 1]
            return future[-1] if future else date_str
        else:
            past = [d for d in trading_strs if d < date_str]
            n = abs(offset_days)
            if len(past) >= n:
                return past[-n]
            return past[0] if past else date_str
    except Exception:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return (dt + timedelta(days=offset_days)).strftime("%Y-%m-%d")


# ── Initialise EFFECTIVE_TODAY ────────────────────────────────────────────
EFFECTIVE_TODAY = _last_trading_day()


# ═══════════════════════════════════════════════════════════════════════════
# Pre-flight canary probe
# ═══════════════════════════════════════════════════════════════════════════

PROBE_RETRIES = 3


def _probe_post(url, headers, payload, endpoint=""):
    """POST a single canary probe, with retry + backoff for transient errors.

    Mirrors _fetch_page's retry behaviour: a transient server-side 500
    (e.g. while the vendor is publishing the evening data) must not be
    mistaken for "data not published yet".  Also goes through the shared
    rate limiter so probes are paced exactly like real fetches.
    """
    limiter = rate_limiter()
    last_exc = None
    for attempt in range(PROBE_RETRIES):
        try:
            limiter.acquire(endpoint)
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") != 200:
                raise RuntimeError(f"API Error: code={result.get('code')}, msg={result.get('msg')}")
            return result
        except Exception as e:
            last_exc = e
            if attempt < PROBE_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise last_exc


def _probe_canary_endpoint(endpoint, target_date, api_key):
    """Check whether *endpoint* has rows for *target_date*.

    For stock-level endpoints, probes stocks from CANARY_STOCKS one at a time
    and requires at least CANARY_MIN_STOCKS to have data.  Stops early once
    enough stocks are confirmed, avoiding false negatives from suspended stocks
    while staying compatible with endpoints that only accept a single stock_code.
    """
    yesterday = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"{BASE_URL}/{endpoint}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    base_payload = {
        "start_time": yesterday,
        "end_time": tomorrow,
        "page": 0,
        "page_size": 10000,
    }
    is_stock_level = endpoint not in CANARY_NO_STOCK_FILTER

    if not is_stock_level:
        # Non-stock endpoint (margin_detail): single probe
        try:
            result = _probe_post(url, headers, base_payload, endpoint)
            rows = result.get("data", {}).get("list", [])
            for row in rows:
                for key in ("trade_date", "date", "trade_time", "end_date"):
                    val = row.get(key)
                    if val is not None and str(val)[:10] == target_date:
                        return True, len(rows), None
            return False, len(rows), None
        except Exception as e:
            log_print(f"  [preflight] Canary probe {endpoint} FAILED: {e}")
            return False, 0, None
    else:
        # Stock-level endpoint: probe stocks one by one, stop early at CANARY_MIN_STOCKS
        stocks_with_data = set()
        total_rows = 0
        errors = 0
        first_error = None
        for stock in CANARY_STOCKS:
            payload = dict(base_payload)
            payload["stock_code"] = stock
            try:
                result = _probe_post(url, headers, payload, endpoint)
                rows = result.get("data", {}).get("list", [])
                total_rows += len(rows)
                for row in rows:
                    for key in ("trade_date", "date", "trade_time", "end_date"):
                        val = row.get(key)
                        if val is not None and str(val)[:10] == target_date:
                            stocks_with_data.add(stock)
                            break
                # Early stop: enough stocks confirmed
                if len(stocks_with_data) >= CANARY_MIN_STOCKS:
                    return True, total_rows, len(stocks_with_data)
            except Exception as e:
                errors += 1
                if first_error is None:
                    first_error = str(e)
                continue

        n_found = len(stocks_with_data)
        ready = n_found >= CANARY_MIN_STOCKS
        if errors > 0:
            log_print(f"  [preflight] Canary probe {endpoint}: "
                      f"{errors}/{len(CANARY_STOCKS)} stocks errored"
                      + (f" (e.g. {first_error})" if first_error else ""))
        return ready, total_rows, n_found


def _all_canaries_ready(target_date=None):
    """Return (all_ready, details_dict) after probing every canary endpoint."""
    if target_date is None:
        target_date = EFFECTIVE_TODAY

    api_key = load_api_key()
    details = {}
    all_ready = True

    for ep in CANARY_ENDPOINTS:
        lag = CANARY_LAG_DAYS.get(ep, 0)
        if lag:
            ep_target = _trading_day_offset(target_date, -lag)
        else:
            ep_target = target_date
        has_data, count, n_stocks = _probe_canary_endpoint(ep, ep_target, api_key)
        details[ep] = {"ready": has_data, "rows": count, "stocks": n_stocks}
        if not has_data:
            all_ready = False

    return all_ready, details


def _get_stock_expected_dates(target_date_str):
    """Return dict {stock_code: expected_date_str} for per-stock behind checks."""
    try:
        daily_path = Path(DATA_DIR) / "daily.parquet"
        if not daily_path.exists():
            return None
        table = pq.read_table(daily_path, columns=["trade_date", "stock_code"])
        df = table.to_pandas()
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        target_dt = pd.Timestamp(target_date_str)
        last_trade = df.groupby("stock_code")["trade_date"].max()
        result = {}
        for stock, last_dt in last_trade.items():
            expected = last_dt if last_dt < target_dt else target_dt
            result[stock] = expected.strftime("%Y-%m-%d")
        return result
    except Exception:
        return None


def _canary_datasets_behind(target_date_str):
    """Check whether local datasets are behind *target_date_str*.

    Checks all data-bearing datasets with appropriate strategies:
    - Per-stock (cyq_chips, indicator_1min, history_1min): sample 10 stocks
    - daily_dump_1min: check file exists on disk
    - Consolidated (cyq_perf, daily_adj, daily, finance, main_fund_flow,
      margin_detail): row count for target_date >= 1000
    """
    THRESHOLD = 1000
    SAMPLE_SIZE = 10

    any_behind = False

    # ═══ 1. cyq_chips (per-stock) ═══
    cyq_chips_dir = Path(DATA_DIR) / "cyq_chips"
    if cyq_chips_dir.exists():
        stock_files = list(cyq_chips_dir.glob("*.parquet"))
        if stock_files:
            sample = random.sample(stock_files, min(SAMPLE_SIZE, len(stock_files)))
            behind_count = 0
            for f in sample:
                _, max_s = get_max_date(f)
                if max_s is None or max_s < target_date_str:
                    behind_count += 1
            if behind_count > 0:
                log_print(f"  [preflight] Phase 1 | {target_date_str} | cyq_chips          | "
                          f"BEHIND ({behind_count}/{len(sample)} stocks)")
                any_behind = True
            else:
                log_print(f"  [preflight] Phase 1 | {target_date_str} | cyq_chips          | OK")
        else:
            log_print(f"  [preflight] Phase 1 | {target_date_str} | cyq_chips          | NO FILES")
            any_behind = True
    else:
        log_print(f"  [preflight] Phase 1 | {target_date_str} | cyq_chips          | DIR MISSING")
        any_behind = True

    # ═══ 2. daily_dump_1min (file existence) ═══
    dump_file = Path(DATA_DIR) / "daily_dump_1min" / f"{target_date_str}.parquet"
    if dump_file.exists():
        log_print(f"  [preflight] Phase 1 | {target_date_str} | daily_dump_1min    | OK")
    else:
        log_print(f"  [preflight] Phase 1 | {target_date_str} | daily_dump_1min    | MISSING")
        any_behind = True

    # ═══ 3. indicator_1min (per-stock) ═══
    indicator_dir = Path(DATA_DIR) / "indicator_1min"
    if indicator_dir.exists():
        stock_files = list(indicator_dir.glob("*.parquet"))
        if stock_files:
            sample = random.sample(stock_files, min(SAMPLE_SIZE, len(stock_files)))
            behind_count = 0
            for f in sample:
                _, max_s = get_max_date(f)
                if max_s is None or max_s < target_date_str:
                    behind_count += 1
            if behind_count > 0:
                log_print(f"  [preflight] Phase 1 | {target_date_str} | indicator_1min     | "
                          f"BEHIND ({behind_count}/{len(sample)} stocks)")
                any_behind = True
            else:
                log_print(f"  [preflight] Phase 1 | {target_date_str} | indicator_1min     | OK")
        else:
            log_print(f"  [preflight] Phase 1 | {target_date_str} | indicator_1min     | NO FILES")
            any_behind = True
    else:
        log_print(f"  [preflight] Phase 1 | {target_date_str} | indicator_1min     | DIR MISSING")
        any_behind = True

    # ═══ 4. history_1min (per-stock) ═══
    history_dir = Path(DATA_DIR) / "history_1min"
    if history_dir.exists():
        stock_files = list(history_dir.glob("*.parquet"))
        if stock_files:
            sample = random.sample(stock_files, min(SAMPLE_SIZE, len(stock_files)))
            behind_count = 0
            for f in sample:
                _, max_s = get_max_date(f)
                if max_s is None or max_s < target_date_str:
                    behind_count += 1
            if behind_count > 0:
                log_print(f"  [preflight] Phase 1 | {target_date_str} | history_1min       | "
                          f"BEHIND ({behind_count}/{len(sample)} stocks)")
                any_behind = True
            else:
                log_print(f"  [preflight] Phase 1 | {target_date_str} | history_1min       | OK")
        else:
            log_print(f"  [preflight] Phase 1 | {target_date_str} | history_1min       | NO FILES")
            any_behind = True
    else:
        log_print(f"  [preflight] Phase 1 | {target_date_str} | history_1min       | DIR MISSING")
        any_behind = True

    # ═══ 5-11. Consolidated datasets (row count >= THRESHOLD) ═══
    # (name, filename, date_col, lag_days)
    consolidated_checks = [
        ("cyq_perf",       "cyq_perf.parquet",       "trade_date", 0),
        ("daily_adj",      "daily_adj.parquet",       "trade_date", 0),
        ("daily",          "daily.parquet",           "trade_date", 0),
        ("finance",        "finance.parquet",         "trade_date", 0),
        ("main_fund_flow", "main_fund_flow.parquet",  "trade_date", 0),
        ("margin_detail",  "margin_detail.parquet",   "trade_date", 1),
    ]

    for name, filename, date_col, lag in consolidated_checks:
        fpath = Path(DATA_DIR) / filename
        if not fpath.exists():
            log_print(f"  [preflight] Phase 1 | {target_date_str} | {name:<18} | FILE MISSING")
            any_behind = True
            continue

        check_date = _trading_day_offset(target_date_str, -lag) if lag else target_date_str

        try:
            table = pq.read_table(str(fpath), columns=[date_col])
            df = table.to_pandas()
            count = (df[date_col].astype(str).str[:10] == check_date).sum()
            if count >= THRESHOLD:
                log_print(f"  [preflight] Phase 1 | {target_date_str} | {name:<18} | OK ({count} rows)")
            else:
                log_print(f"  [preflight] Phase 1 | {target_date_str} | {name:<18} | "
                          f"BEHIND ({count} rows < {THRESHOLD})")
                any_behind = True
        except Exception as e:
            log_print(f"  [preflight] Phase 1 | {target_date_str} | {name:<18} | ERROR: {e}")
            any_behind = True

    return any_behind


def wait_for_server(target_date=None, interval=DEFAULT_WAIT_INTERVAL,
                    max_wait=DEFAULT_MAX_WAIT, dry_run=False):
    """Pre-flight canary check before running incremental updates.

    Probes all canary API endpoints to verify the server has published
    data for *target_date*.  Waits (polling every *interval* seconds)
    until all endpoints are ready, up to *max_wait* seconds.
    """
    if target_date is None:
        target_date = EFFECTIVE_TODAY

    log_print(f"[preflight] target_day: {target_date}")
    log_print(f"[preflight] Canary endpoints ({len(CANARY_ENDPOINTS)}): "
              f"{', '.join(CANARY_ENDPOINTS)}")
    log_print(f"[preflight] interval={interval}s, max_wait={max_wait}s")

    if dry_run:
        ready, details = _all_canaries_ready(target_date)
        for ep in CANARY_ENDPOINTS:
            info = details.get(ep, {})
            has_data = info.get("ready", False)
            rows = info.get("rows", 0)
            lag = CANARY_LAG_DAYS.get(ep, 0)
            check_date = _trading_day_offset(target_date, -lag) if lag else target_date
            n_stocks = info.get("stocks")
            if n_stocks is not None:
                stock_info = f", {n_stocks}/{len(CANARY_STOCKS)} stocks"
            else:
                stock_info = ""
            status = "READY" if has_data else "NOT READY"
            log_print(f"  [preflight] [dry-run] {ep:<25} | {status} "
                      f"(check_date={check_date}{stock_info}, {rows} rows)")
        return ready

    start = time.time()
    attempt = 0

    while True:
        attempt += 1
        elapsed = time.time() - start

        ready, details = _all_canaries_ready(target_date)

        all_ok = True
        for ep in CANARY_ENDPOINTS:
            info = details.get(ep, {})
            has_data = info.get("ready", False)
            rows = info.get("rows", 0)
            lag = CANARY_LAG_DAYS.get(ep, 0)
            check_date = _trading_day_offset(target_date, -lag) if lag else target_date

            n_stocks = info.get("stocks")
            if n_stocks is not None:
                stock_info = f", {n_stocks}/{len(CANARY_STOCKS)} stocks"
            else:
                stock_info = ""
            if has_data:
                log_print(f"  [preflight] attempt {attempt} | {ep:<25} | OK "
                          f"(check_date={check_date}{stock_info}, {rows} rows)")
            else:
                log_print(f"  [preflight] attempt {attempt} | {ep:<25} | WAITING "
                          f"(check_date={check_date}{stock_info}, {rows} rows)")
                all_ok = False

        if all_ok:
            log_print(f"[preflight] All {len(CANARY_ENDPOINTS)} canary endpoints "
                      f"ready for {target_date}. Proceeding with download.")
            return True

        if elapsed >= max_wait:
            remaining_eps = [ep for ep in CANARY_ENDPOINTS
                           if not details.get(ep, {}).get("ready", False)]
            log_print(f"[preflight] Timed out after {elapsed:.0f}s "
                      f"(max={max_wait}s). Still waiting for: "
                      f"{', '.join(remaining_eps) if remaining_eps else 'none'}. "
                      f"Proceeding despite missing data.")
            return False

        remaining = max_wait - elapsed
        log_print(f"[preflight] Waiting {interval}s "
                  f"(elapsed: {elapsed:.0f}s, remaining: {remaining:.0f}s)...")
        time.sleep(interval)


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
    os.replace(tmp, STATE_FILE)


def _get_effective_overlap(name, base_overlap, lag_days=0):
    """Return extended overlap if the last fetch was stale (server lagging)."""
    state = _load_state()
    entry = state.get(name, {})
    if entry.get("pending") and entry.get("last_fetch_max"):
        try:
            last_actual = datetime.strptime(entry["last_fetch_max"], "%Y-%m-%d")
            today_dt = datetime.strptime(EFFECTIVE_TODAY, "%Y-%m-%d")
            if lag_days:
                today_dt = datetime.strptime(_trading_day_offset(EFFECTIVE_TODAY, -lag_days), "%Y-%m-%d")
            gap = (today_dt - last_actual).days
            if gap > base_overlap:
                return gap + base_overlap
        except Exception:
            pass
    return base_overlap


def record_state(name, fetch_max, lag_days=0):
    """Record the result of an incremental fetch for a dataset."""
    with _state_lock:
        state = _load_state()
        entry = state.get(name, {})
        entry["last_run"] = TODAY

        if fetch_max:
            entry["last_fetch_max"] = fetch_max
            try:
                max_dt = datetime.strptime(fetch_max, "%Y-%m-%d")
                effective_dt = datetime.strptime(EFFECTIVE_TODAY, "%Y-%m-%d")
                if lag_days:
                    effective_dt = datetime.strptime(_trading_day_offset(EFFECTIVE_TODAY, -lag_days), "%Y-%m-%d")
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


# ═══════════════════════════════════════════════════════════════════════════
# Performance timing
# ═══════════════════════════════════════════════════════════════════════════

def add_timing(dataset, phase, elapsed, **kw):
    """Record a timing measurement (thread-safe)."""
    entry = {"phase": phase, "elapsed": round(elapsed, 3)}
    entry.update(kw)
    with _timing_lock:
        _timings.setdefault(dataset, []).append(entry)


def print_timing_report():
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


# ═══════════════════════════════════════════════════════════════════════════
# Date-column utilities
# ═══════════════════════════════════════════════════════════════════════════

_DATE_CANDIDATES = ["trade_date", "trade_time", "end_date", "date", "Date",
                    "f_ann_date", "report_date", "ann_date"]


def _detect_date_col(columns):
    """Return the first matching date column from the schema."""
    for c in _DATE_CANDIDATES:
        if c in columns:
            return c
    return None


def get_max_date(filepath):
    """Read only the date column(s) from a parquet file and return the max date."""
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
        if hasattr(max_val, "strftime"):
            return date_col, max_val.strftime("%Y-%m-%d")
        return date_col, str(max_val)[:10]
    except Exception as e:
        log_print(f"  [warn] get_max_date({filepath}): {e}")
        return None, None


def get_real_max_date(filepath):
    """Like get_max_date but excludes backfilled rows when is_backfill column exists."""
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
    """Determine the date range for an incremental fetch."""
    path = Path(filepath)
    if not path.exists():
        return None, default_start, EFFECTIVE_TODAY, None

    date_col, max_str = get_max_date(path)
    if max_str is None:
        return date_col, default_start, EFFECTIVE_TODAY, None

    max_dt = datetime.strptime(max_str, "%Y-%m-%d")
    new_start_dt = max_dt - timedelta(days=overlap_days)
    default_dt = datetime.strptime(default_start, "%Y-%m-%d")
    if new_start_dt < default_dt:
        new_start_dt = default_dt

    new_start = new_start_dt.strftime("%Y-%m-%d")
    return date_col, new_start, EFFECTIVE_TODAY, None


# ═══════════════════════════════════════════════════════════════════════════
# Generic API helpers
# ═══════════════════════════════════════════════════════════════════════════

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
        limiter = rate_limiter()
        limiter.acquire(endpoint)
        _api_gate.acquire()
        try:
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
        finally:
            _api_gate.release()


def _api_range(endpoint, start_time, end_time, api_key,
               extra_payload=None, page_size=10000, verbose=False, log_prefix=""):
    """Fetch all rows for a date range, splitting if total > 100K."""
    t0 = time.perf_counter()
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
        if verbose:
            log_print(f"{log_prefix} {start_time} ~ {end_time}: "
                      f"{len(all_data)} rows, {page} pages, "
                      f"{time.perf_counter() - t0:.1f}s")
        return all_data

    start_dt = datetime.strptime(start_time, "%Y-%m-%d")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d")
    if start_dt == end_dt:
        return list(page0)
    mid = start_dt + (end_dt - start_dt) // 2
    if mid == start_dt:
        return list(page0)
    mid_str = mid.strftime("%Y-%m-%d")
    next_str = (mid + timedelta(days=1)).strftime("%Y-%m-%d")
    if verbose:
        log_print(f"{log_prefix} Splitting {start_time} ~ {end_time} "
                  f"(total={total}) -> [{start_time}, {mid_str}] + "
                  f"[{next_str}, {end_time}]")
    left = _api_range(endpoint, start_time, mid_str, api_key,
                      extra_payload, page_size, verbose, log_prefix)
    right = _api_range(endpoint, next_str, end_time, api_key,
                       extra_payload, page_size, verbose, log_prefix)
    return left + right


# ═══════════════════════════════════════════════════════════════════════════
# Safe function caller
# ═══════════════════════════════════════════════════════════════════════════

def _call_fetch(fn, start_date=None, end_date=None, workers=DEFAULT_WORKERS,
                cleanup=True, resume=False, **extra):
    """Call a fetch function, adapting parameters to what it accepts."""
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = {}
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


# ═══════════════════════════════════════════════════════════════════════════
# Merge helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_read_parquet(filepath):
    """Read a parquet file, return empty DataFrame on failure."""
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        log_print(f"  [warn] Cannot read {filepath}: {e}")
        return pd.DataFrame()


def merge_and_save(existing_path, new_df, dedup_keys, sort_cols,
                   date_col=None, strict_dedup=False):
    """Merge new data into an existing parquet file."""
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

    if "_sort_code" in merged.columns:
        merged = merged.drop(columns=["_sort_code"])

    available_sort = [c for c in sort_cols if c in merged.columns]
    if available_sort:
        merged = merged.sort_values(available_sort).reset_index(drop=True)

    after = len(merged)
    rows_new = after - before

    tmp = existing_path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(existing_path)
    return before, after, rows_new


# ═══════════════════════════════════════════════════════════════════════════
# Backfill helpers
# ═══════════════════════════════════════════════════════════════════════════

def _backfill_missing_dates(filepath, date_col, today_str):
    """Forward-fill missing trading days for daily-frequency datasets."""
    cal_path = Path(DATA_DIR) / "calendar.parquet"
    if not cal_path.exists():
        log_print("  [backfill] calendar.parquet not found, cannot determine trading days")
        return False

    cal = pd.read_parquet(cal_path)
    trading_days = sorted(cal[cal["is_open"] == 1]["date"].tolist())

    df = pd.read_parquet(filepath)
    if df.empty or date_col not in df.columns:
        return False

    if "is_backfill" not in df.columns:
        df["is_backfill"] = False
    else:
        df["is_backfill"] = df["is_backfill"].fillna(False).astype(bool)

    if "stock_code" not in df.columns:
        log_print("  [backfill] No stock_code column, cannot backfill per-stock")
        return False

    last_per_stock = df.groupby("stock_code")[date_col].max().reset_index()
    last_per_stock.columns = ["stock_code", "last_date"]

    behind = last_per_stock[last_per_stock["last_date"] < today_str]
    if behind.empty:
        return False

    last_data = df.merge(
        behind,
        left_on=["stock_code", date_col],
        right_on=["stock_code", "last_date"],
        how="inner",
    )
    last_data = last_data.drop(columns=["last_date"])

    if last_data.empty:
        return False

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

    tmp = Path(filepath).with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(Path(filepath))

    n_stocks = fill_df["stock_code"].nunique()
    n_days = len(fill_parts)
    log_print(f"  [backfill] Filled {len(fill_df)} rows for {n_stocks} stocks "
              f"across {n_days} missing trading day(s)")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Per-stock helpers
# ═══════════════════════════════════════════════════════════════════════════

CODE_NUM_FILE = Path(DATA_DIR).parent / "Code_num.txt"
_PER_STOCK_BATCH_SIZE = 100


def _add_exchange_suffix(code):
    """Add .SZ or .SH suffix to a 6-digit stock code."""
    return code + (".SH" if code.startswith("60") else ".SZ")


def _filter_main_board_stocks():
    """Return list of stock codes from Code_num.txt with exchange suffixes."""
    if not CODE_NUM_FILE.exists():
        log_print(f"  [warn] {CODE_NUM_FILE} not found, cannot filter stocks")
        return []
    with open(CODE_NUM_FILE) as f:
        return [_add_exchange_suffix(line.strip()) for line in f if line.strip()]


def _merge_per_stock_batch(batch_results, out_dir, date_col, cache_ns=None,
                           dedup_keys=None, workers=16):
    """Split a batch result by stock_code and merge each into its file (parallel)."""
    if isinstance(batch_results, pd.DataFrame):
        df_all = batch_results
    elif batch_results:
        df_all = pd.DataFrame(batch_results)
    else:
        return [], {}

    if df_all.empty or "stock_code" not in df_all.columns:
        return [], {}

    target_stocks = set(_filter_main_board_stocks())
    before = len(df_all)
    df_all = df_all[df_all["stock_code"].isin(target_stocks)]
    if len(df_all) < before:
        log_print(f"  [merge] Dropped {before - len(df_all):,} non-target stock rows "
                  f"({len(df_all):,} rows, {df_all['stock_code'].nunique()} stocks remain)")

    if df_all.empty:
        return [], {}

    ns_prefix = f"{cache_ns}:" if cache_ns else ""
    dk = dedup_keys if dedup_keys is not None else ([date_col] if date_col else None)

    groups = [(code, group.copy()) for code, group in df_all.groupby("stock_code")]
    del df_all

    def _merge_one(code, group):
        out_file = Path(out_dir) / f"{code}.parquet"
        existing = _safe_read_parquet(out_file) if out_file.exists() else pd.DataFrame()
        old_rows = len(existing)

        new_chunk = group.drop(columns=["stock_code"], errors="ignore")
        new_chunk["stock_code"] = code
        if existing.empty:
            merged = new_chunk
        elif date_col and date_col in existing.columns and date_col in new_chunk.columns:
            if not pd.api.types.is_datetime64_any_dtype(existing[date_col]):
                existing[date_col] = pd.to_datetime(existing[date_col])
            if not pd.api.types.is_datetime64_any_dtype(new_chunk[date_col]):
                new_chunk[date_col] = pd.to_datetime(new_chunk[date_col])

            min_new = new_chunk[date_col].min()
            tail = existing[existing[date_col] < min_new]
            overlap = existing[existing[date_col] >= min_new]

            if dk:
                available_dedup = [k for k in dk if k in overlap.columns and k in new_chunk.columns]
            else:
                available_dedup = None
            deduped = pd.concat([overlap, new_chunk], ignore_index=True) \
                        .drop_duplicates(subset=available_dedup or None, keep="last")
            deduped = deduped.sort_values(date_col).reset_index(drop=True)
            merged = pd.concat([tail, deduped], ignore_index=True)
        else:
            merged = pd.concat([existing, new_chunk], ignore_index=True)
            available_dedup = [k for k in dk if k in merged.columns] if dk else None
            merged = merged.drop_duplicates(subset=available_dedup or None, keep="last")
            if date_col and date_col in merged.columns:
                if not pd.api.types.is_datetime64_any_dtype(merged[date_col]):
                    merged[date_col] = pd.to_datetime(merged[date_col])
                merged = merged.sort_values(date_col).reset_index(drop=True)

        max_str = None
        if date_col and date_col in merged.columns:
            if not pd.api.types.is_datetime64_any_dtype(merged[date_col]):
                merged[date_col] = pd.to_datetime(merged[date_col])
            max_val = merged[date_col].max()
            if not pd.isna(max_val):
                max_str = (max_val.strftime("%Y-%m-%d") if hasattr(max_val, "strftime")
                           else str(max_val)[:10])

        tmp = out_file.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(out_file)
        return code, old_rows, len(merged), len(merged) - old_rows, max_str

    results = []
    cache_updates = {}
    lock = threading.Lock()
    n = len(groups)

    if n <= 4 or workers <= 1:
        for code, group in groups:
            res = _merge_one(code, group)
            results.append(res[:4])
            if res[4]:
                cache_updates[f"{ns_prefix}{code}"] = res[4]
    else:
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=min(workers, n)) as pool:
            futures = {pool.submit(_merge_one, code, group): code
                       for code, group in groups}
            for fut in tqdm(as_completed(futures), total=n,
                           desc="  merge", unit="stock",
                           ncols=100, smoothing=0.1):
                code = futures[fut]
                try:
                    res = fut.result()
                    with lock:
                        results.append(res[:4])
                        if res[4]:
                            cache_updates[f"{ns_prefix}{code}"] = res[4]
                except Exception as e:
                    log_print(f"  [warn] Merge failed for {code}: {e}")

    return results, cache_updates


def _consolidate_per_stock_dataset(name, out_dir, date_col, workers=8):
    """Merge per-stock .parquet files into a single consolidated file (streaming)."""
    from tqdm import tqdm
    import pyarrow as pa

    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    stock_files = sorted(out_dir.glob("*.parquet"))
    if not stock_files:
        log_print(f"  [consolidate] [{name}] No stock files found in {out_dir}")
        return

    n = len(stock_files)
    BATCH_SIZE = 300
    batches = [stock_files[i:i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    n_batches = len(batches)

    dest = Path(DATA_DIR) / f"{name}.parquet"
    tmp = dest.with_suffix(".parquet.tmp")

    log_print(f"  [consolidate] [{name}] Streaming {n} files -> {dest} "
              f"({n_batches} batches of ~{BATCH_SIZE}, {workers} workers)")

    def _read_one(f):
        try:
            return pd.read_parquet(f)
        except Exception as e:
            log_print(f"\n  [consolidate] Skipping {f.name}: {e}")
            return None

    sort_cols = ["stock_code", date_col]
    available_sort = None
    writer = None
    total_rows = 0
    skipped = 0
    batch_times = []

    try:
        for bi, batch_files in enumerate(
            tqdm(batches, desc=f"  consolidate/{name}", unit="batch",
                 ncols=100, smoothing=0.1)
        ):
            t_batch = time.perf_counter()

            parts = []
            if len(batch_files) <= 4 or workers <= 1:
                for f in batch_files:
                    df = _read_one(f)
                    if df is not None and not df.empty:
                        parts.append(df)
                    else:
                        skipped += 1
            else:
                with ThreadPoolExecutor(max_workers=min(workers, len(batch_files))) as pool:
                    futures = {pool.submit(_read_one, f): f for f in batch_files}
                    for fut in as_completed(futures):
                        df = fut.result()
                        if df is not None and not df.empty:
                            parts.append(df)
                        else:
                            skipped += 1

            if not parts:
                continue

            batch_df = pd.concat(parts, ignore_index=True)
            del parts

            if available_sort is None:
                available_sort = [c for c in sort_cols if c in batch_df.columns]
            if available_sort:
                batch_df = batch_df.sort_values(available_sort).reset_index(drop=True)

            batch_rows = len(batch_df)
            total_rows += batch_rows

            table = pa.Table.from_pandas(batch_df)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            del batch_df, table

            batch_times.append(time.perf_counter() - t_batch)

    finally:
        if writer:
            writer.close()

    tmp.replace(dest)

    elapsed = time.perf_counter() - t0
    file_gb = dest.stat().st_size / (1024 ** 3)
    avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
    log_print(f"  [consolidate] [{name}] Done: {total_rows:,} rows, "
              f"{file_gb:.1f} GB -> {dest} "
              f"(avg {avg_batch:.1f}s/batch, {skipped} skipped, "
              f"total {elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset updaters
# ═══════════════════════════════════════════════════════════════════════════

def update_per_stock_dataset(name, out_subdir, endpoint, start_date,
                             end_date, date_col, workers, dry_run,
                             extra_payload=None, backfill_fn=None,
                             backfill_kwargs=None, batch_size=_PER_STOCK_BATCH_SIZE,
                             stock_code_as_list=True, dedup_keys=None,
                             consolidate=True):
    """Update per-stock parquet files using multi-stock batched API calls."""
    stocks = _filter_main_board_stocks()
    if not stocks:
        log_print(f"[{name}] No stocks to process")
        return

    out_dir = Path(DATA_DIR) / out_subdir
    out_dir.mkdir(exist_ok=True)
    api_key = load_api_key()

    # Phase 0: backfill missing stocks via the original full-fetch function
    if not dry_run and backfill_fn:
        missing = [code for code in stocks
                   if not (out_dir / f"{code}.parquet").exists()]
        if missing:
            log_print(f"[{name}] {len(missing)} stocks without files, "
                      f"running full backfill via {backfill_fn.__name__}...")
            t0 = time.perf_counter()
            try:
                backfill_fn(
                    start_date=start_date, end_date=end_date,
                    workers=workers, cleanup=True, resume=True,
                    **(backfill_kwargs or {})
                )
                add_timing(name, "backfill", time.perf_counter() - t0,
                           stocks=len(missing))
            except Exception as e:
                log_print(f"[{name}] Backfill FAILED: {e}")
                add_timing(name, "backfill", time.perf_counter() - t0,
                           error=str(e))

    # Phase 1: classify stocks
    cache = _load_cyq_cache()
    cache_updated = False
    skipped = []
    incr_batch = []
    uptodate = 0
    cache_key_prefix = f"{name}:"

    # ── Validate cache against actual data (sample check) ──────────
    # The cyq_cache can be stale (e.g. if a previous merge failed after
    # updating the cache).  Validate a random sample; if any entry over-reports
    # relative to the actual file, invalidate the entire namespace cache.
    ns_keys_validate = [k for k in cache if k.startswith(cache_key_prefix)]
    if ns_keys_validate:
        import random as _random
        sample_keys = _random.sample(
            ns_keys_validate,
            min(30, len(ns_keys_validate))
        )
        for key in sample_keys:
            code = key[len(cache_key_prefix):]
            f_sample = out_dir / f"{code}.parquet"
            if f_sample.exists():
                _, file_max = get_max_date(f_sample)
                cached_max = cache.get(key)
                if file_max and cached_max and file_max < cached_max:
                    log_print(f"[{name}] Cache stale ({code}: cache={cached_max} "
                              f"file={file_max}), invalidating {len(ns_keys_validate)} entries")
                    for k in ns_keys_validate:
                        del cache[k]
                    cache_updated = True
                    break

    for code in stocks:
        f = out_dir / f"{code}.parquet"
        if not f.exists():
            skipped.append(code)
            continue
        max_s = cache.get(f"{cache_key_prefix}{code}")
        if max_s is None:
            dc, max_s = get_max_date(f)
            if max_s is not None:
                cache[f"{cache_key_prefix}{code}"] = max_s
                cache_updated = True
        if max_s is None:
            skipped.append(code)
            continue
        max_dt = datetime.strptime(max_s, "%Y-%m-%d")
        effective_overlap = _get_effective_overlap(name, OVERLAP_DAYS)
        inc_start_dt = max_dt - timedelta(days=effective_overlap)
        default_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if inc_start_dt < default_dt:
            inc_start_dt = default_dt
        inc_start = inc_start_dt.strftime("%Y-%m-%d")

        if max_s >= end_date or max_s >= EFFECTIVE_TODAY:
            uptodate += 1
            continue
        incr_batch.append((code, inc_start))

    if cache_updated:
        _save_cyq_cache(cache)

    log_print(f"[{name}] {len(stocks)} stocks | "
              f"{len(incr_batch)} incremental | "
              f"{len(skipped)} skipped (no file) | "
              f"{uptodate} up-to-date")

    if skipped:
        log_print(f"[{name}] {len(skipped)} stocks still missing after backfill, "
                  f"skipping: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    if not incr_batch:
        log_print(f"[{name}] All stocks up to date"
                  f"{', ' + str(len(skipped)) + ' skipped' if skipped else ''}")
        return

    if dry_run:
        effective_batch = 1 if not stock_code_as_list else batch_size
        log_print(f"[{name}] DRY-RUN: would fetch {len(incr_batch)} incremental stocks"
                  f"{' (backfill ' + str(len(skipped)) + ' new first)' if skipped else ''}")
        sample = incr_batch[:3]
        for code, s in sample:
            log_print(f"  incr: {code}: {s} ~ {end_date}")
        log_print(f"  ... {len(incr_batch)} stocks in "
                  f"{(len(incr_batch) + effective_batch - 1) // effective_batch} batches")
        return

    limiter = rate_limiter()
    total_added = 0
    processed = 0
    lock = threading.Lock()

    def _process_batches(stock_list, phase_label):
        nonlocal processed, total_added

        effective_batch = 1 if not stock_code_as_list else batch_size
        batches = []
        for i in range(0, len(stock_list), effective_batch):
            batch = stock_list[i:i + effective_batch]
            min_start = min(s for _, s in batch)
            batches.append(([c for c, _ in batch], min_start))

        n_batches = len(batches)
        phase_start = time.perf_counter()
        log_prefix = f"[{name}] [{phase_label}]"
        log_print(f"{log_prefix} {len(stock_list)} stocks in "
                  f"{n_batches} batches (batch_size={effective_batch})")

        def _resolve_stock_code(codes_arg):
            if stock_code_as_list:
                return codes_arg
            else:
                return codes_arg[0]

        completed = 0
        batch_cache_updates = {}
        if workers > 1 and n_batches > 1:
            def _timed_fetch(codes_arg, batch_start_arg):
                ep_inner = (extra_payload.copy() if extra_payload else {})
                ep_inner["stock_code"] = _resolve_stock_code(codes_arg)
                t0 = time.perf_counter()
                rows_result = _api_range(endpoint, batch_start_arg, end_date,
                                         api_key, ep_inner, 10000,
                                         verbose=True, log_prefix=log_prefix)
                add_timing(name, "fetch", time.perf_counter() - t0,
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
                        merge_results, cache_upd = _merge_per_stock_batch(
                            rows, out_dir, date_col, cache_ns=name, dedup_keys=dedup_keys)
                        add_timing(name, "merge", time.perf_counter() - t0,
                                   rows=len(rows), stocks=len(merge_results))
                        with lock:
                            completed += 1
                            batch_added = sum(r[3] for r in merge_results)
                            total_added += batch_added
                            processed += len(merge_results)
                            batch_cache_updates.update(cache_upd)
                        elapsed = time.perf_counter() - phase_start
                        avg_per_batch = elapsed / completed if completed else 0
                        eta = avg_per_batch * (n_batches - completed)
                        log_print(f"{log_prefix} {completed}/{n_batches} | "
                                  f"{len(codes)} stocks, {len(rows)} rows, "
                                  f"+{batch_added} new | "
                                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s | "
                                  f"tokens={limiter.stats['tokens_available']}")
                    except Exception as e:
                        with lock:
                            completed += 1
                        log_print(f"{log_prefix} {completed}/{n_batches} | "
                                  f"{len(codes)} stocks FAILED: {e}")
        else:
            for bi, (codes, batch_start) in enumerate(batches):
                ep = (extra_payload.copy() if extra_payload else {})
                ep["stock_code"] = _resolve_stock_code(codes)
                try:
                    t0 = time.perf_counter()
                    rows = _api_range(endpoint, batch_start, end_date,
                                      api_key, ep, 10000,
                                      verbose=True, log_prefix=log_prefix)
                    add_timing(name, "fetch", time.perf_counter() - t0,
                               rows=len(rows), calls=1)
                    t0 = time.perf_counter()
                    merge_results, cache_upd = _merge_per_stock_batch(
                        rows, out_dir, date_col, cache_ns=name, dedup_keys=dedup_keys)
                    add_timing(name, "merge", time.perf_counter() - t0,
                               rows=len(rows), stocks=len(merge_results))
                    with lock:
                        completed += 1
                        batch_added = sum(r[3] for r in merge_results)
                        total_added += batch_added
                        processed += len(merge_results)
                        batch_cache_updates.update(cache_upd)
                    elapsed = time.perf_counter() - phase_start
                    avg_per_batch = elapsed / completed if completed else 0
                    eta = avg_per_batch * (n_batches - completed)
                    log_print(f"{log_prefix} {completed}/{n_batches} | "
                              f"{len(codes)} stocks, {len(rows)} rows, "
                              f"+{batch_added} new | "
                              f"elapsed={elapsed:.0f}s ETA={eta:.0f}s | "
                              f"tokens={limiter.stats['tokens_available']}")
                except Exception as e:
                    completed += 1
                    log_print(f"{log_prefix} {completed}/{n_batches} | "
                              f"{len(codes)} stocks FAILED: {e}")
        return batch_cache_updates

    all_cache_updates = _process_batches(incr_batch, "incr")

    if all_cache_updates:
        cache = _load_cyq_cache()
        cache.update(all_cache_updates)
        _save_cyq_cache(cache)

    log_print(f"[{name}] Done: {processed} stocks processed, "
              f"{total_added} new rows total")

    if not dry_run:
        sample_max = None
        for f in sorted(out_dir.glob("*.parquet"))[:200]:
            _, m = get_max_date(f)
            if m and (sample_max is None or m > sample_max):
                sample_max = m
        record_state(name, sample_max)

    done_file = out_dir / ".done"
    done_file.write_text(TODAY)

    if not dry_run and consolidate:
        _consolidate_per_stock_dataset(
            name=name,
            out_dir=out_dir,
            date_col=date_col,
            workers=workers,
        )


def update_daily_dump_dataset(name, out_subdir, level, start_date, end_date,
                               date_col, dedup_keys, workers, dry_run,
                               backfill_fn=None, backfill_kwargs=None,
                               consolidate=True):
    """Update per-stock 1min dataset via daily_dump (local-first, one file per date)."""
    from fetch_daily_dump import fetch_daily_dump

    stocks = _filter_main_board_stocks()
    if not stocks:
        log_print(f"[{name}] No stocks to process")
        return

    target_stock_set = set(stocks)

    out_dir = Path(DATA_DIR) / out_subdir
    out_dir.mkdir(exist_ok=True)

    daily_dump_dir = Path(DATA_DIR) / f"daily_dump_{level}"

    dump_dates = set()
    if daily_dump_dir.exists():
        for f in daily_dump_dir.glob("*.parquet"):
            try:
                dump_dates.add(f.stem)
            except Exception:
                pass

    # Phase 0: backfill stocks without existing files
    if not dry_run and backfill_fn:
        missing = [code for code in stocks
                   if not (out_dir / f"{code}.parquet").exists()]
        if missing:
            log_print(f"[{name}] {len(missing)} stocks without files, "
                      f"running full backfill via {backfill_fn.__name__}...")
            t0 = time.perf_counter()
            try:
                backfill_fn(
                    start_date=start_date, end_date=end_date,
                    workers=workers, cleanup=True, resume=True,
                    **(backfill_kwargs or {})
                )
                add_timing(name, "backfill", time.perf_counter() - t0,
                           stocks=len(missing))
            except Exception as e:
                log_print(f"[{name}] Backfill FAILED: {e}")
                add_timing(name, "backfill", time.perf_counter() - t0,
                           error=str(e))

    # Phase 1: determine last fetched date
    state = _load_state()
    entry = state.get(name, {})
    last_date = entry.get("last_fetch_max")

    # Always scan actual files to verify the state file.
    # The state file can over-report (e.g. if a previous merge failed after
    # recording last_fetch_max), so we must validate against real data.
    sample_files = sorted(out_dir.glob("*.parquet"))[:200]
    files_max = None
    for f in sample_files:
        _, m = get_max_date(f)
        if m and (files_max is None or m > files_max):
            files_max = m
    if files_max is not None:
        if last_date is None or files_max > last_date:
            last_date = files_max
        elif entry.get("last_fetch_max") and entry["last_fetch_max"] > files_max:
            log_print(f"[{name}] State file max={entry['last_fetch_max']} but "
                      f"files show max={files_max}, correcting")
            last_date = files_max

    if last_date is None:
        log_print(f"[{name}] No existing data - need full backfill first")
        return

    new_dump_dates = sorted([d for d in dump_dates if d > last_date])
    if new_dump_dates:
        log_print(f"[{name}] Found {len(new_dump_dates)} unprocessed daily dump "
                  f"file(s): {new_dump_dates[0]} ~ {new_dump_dates[-1]}"
                  if len(new_dump_dates) > 1 else
                  f"[{name}] Found unprocessed daily dump file: {new_dump_dates[0]}")

    max_dt = datetime.strptime(last_date, "%Y-%m-%d")
    effective_overlap = _get_effective_overlap(name, OVERLAP_DAYS)
    inc_start_dt = max_dt - timedelta(days=effective_overlap)
    default_dt = datetime.strptime(start_date, "%Y-%m-%d")
    if inc_start_dt < default_dt:
        inc_start_dt = default_dt
    inc_start = inc_start_dt.strftime("%Y-%m-%d")

    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else \
             datetime.strptime(EFFECTIVE_TODAY, "%Y-%m-%d")

    if new_dump_dates:
        latest_dump = datetime.strptime(max(new_dump_dates), "%Y-%m-%d")
        if latest_dump > end_dt:
            end_dt = latest_dump

    effective_today_dt = datetime.strptime(EFFECTIVE_TODAY, "%Y-%m-%d")
    if max_dt >= effective_today_dt and not new_dump_dates:
        log_print(f"[{name}] Up to date (max={last_date})")
        record_state(name, last_date)
        if consolidate:
            consolidated = Path(DATA_DIR) / f"{name}.parquet"
            if not dry_run and not consolidated.exists():
                log_print(f"[{name}] Consolidated file missing, creating...")
                _consolidate_per_stock_dataset(
                    name=name, out_dir=out_dir, date_col=date_col,
                    workers=workers,
                )
        return

    log_print(f"[{name}] Last date: {last_date} | "
              f"Fetching: {inc_start} ~ {end_dt.strftime('%Y-%m-%d')}")

    if dry_run:
        log_print(f"[{name}] DRY-RUN: would process {inc_start} ~ "
                  f"{end_dt.strftime('%Y-%m-%d')} via daily_dump/{level}")
        if new_dump_dates:
            log_print(f"[{name}] DRY-RUN: {len(new_dump_dates)} local file(s) "
                      f"would be read (no API calls)")
        return

    # Phase 2: load trading calendar
    cal_path = Path(DATA_DIR) / "calendar.parquet"
    trading_days = set()
    if cal_path.exists():
        cal = pd.read_parquet(cal_path)
        trading_days = set(cal[cal["is_open"] == 1]["date"].tolist())

    dates_to_fetch = []
    current_dt = inc_start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        if not trading_days or date_str in trading_days:
            dates_to_fetch.append(date_str)
        current_dt += timedelta(days=1)

    if not dates_to_fetch:
        log_print(f"[{name}] No trading days in range")
        return

    log_print(f"[{name}] Will process {len(dates_to_fetch)} trading days "
              f"via daily_dump/{level}")

    # Phase 3: load all daily dumps, group by stock, merge each file ONCE
    limiter = rate_limiter()
    total_added = 0
    new_max_date = last_date
    local_reads = 0
    api_calls = 0

    all_parts = []
    for i, date_str in enumerate(dates_to_fetch):
        t0 = time.perf_counter()
        try:
            dump_file = daily_dump_dir / f"{date_str}.parquet"

            if dump_file.exists():
                df = pd.read_parquet(dump_file)
                local_reads += 1
                add_timing(name, "fetch", time.perf_counter() - t0,
                           rows=len(df), from_cache=True)
            else:
                df = fetch_daily_dump(date_str, level,
                                      output_dir=str(daily_dump_dir))
                api_calls += 1
                add_timing(name, "fetch", time.perf_counter() - t0,
                           rows=len(df) if df is not None else 0, from_api=True)

            if df is None or df.empty:
                log_print(f"[{name}] [{i+1}/{len(dates_to_fetch)}] {date_str}: "
                          f"no data")
                continue

            if "stock_code" in df.columns:
                before = len(df)
                df = df[df["stock_code"].isin(target_stock_set)]
                dropped = before - len(df)
                if dropped > 0:
                    log_print(f"[{name}] [{i+1}/{len(dates_to_fetch)}] "
                              f"{date_str}: filtered {dropped:,} non-target "
                              f"stocks ({len(df):,} rows remain)")

            if df.empty:
                continue

            if "trade_date" in df.columns:
                df = df.drop(columns=["trade_date"])

            all_parts.append(df)
            if date_str > new_max_date:
                new_max_date = date_str

        except Exception as e:
            log_print(f"[{name}] [{i+1}/{len(dates_to_fetch)}] {date_str} "
                      f"FAILED: {e}")

    if not all_parts:
        log_print(f"[{name}] No new data to process")
        record_state(name, new_max_date)
        done_file = out_dir / ".done"
        done_file.write_text(TODAY)
        return

    full_df = pd.concat(all_parts, ignore_index=True)
    del all_parts

    n_stocks_in_data = full_df["stock_code"].nunique()
    log_print(f"[{name}] Loaded {len(full_df):,} rows across "
              f"{n_stocks_in_data} stocks "
              f"({local_reads} local reads, {api_calls} API calls)")

    stock_groups = [(code, group.copy())
                    for code, group in full_df.groupby("stock_code")]
    del full_df

    n_stocks = len(stock_groups)
    results = []
    lock = threading.Lock()

    def _merge_one_stock(code, new_data):
        out_file = out_dir / f"{code}.parquet"
        existing = _safe_read_parquet(out_file) if out_file.exists() else pd.DataFrame()
        old_rows = len(existing)

        if existing.empty:
            merged = new_data
        elif date_col and date_col in existing.columns and date_col in new_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(existing[date_col]):
                existing[date_col] = pd.to_datetime(existing[date_col])
            if not pd.api.types.is_datetime64_any_dtype(new_data[date_col]):
                new_data[date_col] = pd.to_datetime(new_data[date_col])

            min_new = new_data[date_col].min()
            tail = existing[existing[date_col] < min_new]
            overlap = existing[existing[date_col] >= min_new]
            deduped = pd.concat([overlap, new_data], ignore_index=True) \
                        .drop_duplicates(subset=[date_col], keep="last")
            deduped = deduped.sort_values(date_col).reset_index(drop=True)
            merged = pd.concat([tail, deduped], ignore_index=True)
        else:
            merged = pd.concat([existing, new_data], ignore_index=True)
            if date_col and date_col in merged.columns:
                merged = merged.drop_duplicates(subset=[date_col], keep="last")
            else:
                merged = merged.drop_duplicates(keep="last")
            if date_col and date_col in merged.columns:
                if not pd.api.types.is_datetime64_any_dtype(merged[date_col]):
                    merged[date_col] = pd.to_datetime(merged[date_col])
                merged = merged.sort_values(date_col).reset_index(drop=True)

        if date_col and date_col in merged.columns:
            if not pd.api.types.is_datetime64_any_dtype(merged[date_col]):
                merged[date_col] = pd.to_datetime(merged[date_col])

        tmp = out_file.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(out_file)

        return code, old_rows, len(merged), len(merged) - old_rows

    t_merge = time.perf_counter()

    if n_stocks <= 4 or workers <= 1:
        for code, group in stock_groups:
            try:
                res = _merge_one_stock(code, group)
                results.append(res)
            except Exception as e:
                log_print(f"  [warn] Merge failed for {code}: {e}")
    else:
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=min(workers, n_stocks)) as pool:
            futures = {pool.submit(_merge_one_stock, code, group): code
                       for code, group in stock_groups}
            for fut in tqdm(as_completed(futures), total=n_stocks,
                           desc="  merge", unit="stock",
                           ncols=100, smoothing=0.1):
                code = futures[fut]
                try:
                    res = fut.result()
                    with lock:
                        results.append(res)
                except Exception as e:
                    log_print(f"  [warn] Merge failed for {code}: {e}")

    total_added = sum(r[3] for r in results)
    add_timing(name, "merge", time.perf_counter() - t_merge,
               rows=sum(r[2] for r in results), stocks=len(results))

    log_print(f"[{name}] Done: {total_added:,} new rows total "
              f"({local_reads} local reads, {api_calls} API calls)")

    record_state(name, new_max_date)

    done_file = out_dir / ".done"
    done_file.write_text(TODAY)


def update_consolidated(name, fetch_fn, filepath, date_col, dedup_keys,
                          sort_cols, default_start, workers, dry_run,
                          extra_params=None, strict_dedup=False,
                          overlap_days=None, backfill=False,
                          lag_days=0):
    """Incrementally update a consolidated (single-parquet) dataset."""
    path = Path(filepath)
    if not path.exists():
        log_print(f"[{name}] Output file not found ({filepath}) - "
                  f"full fetch needed, run scripts/main.py first")
        return {"name": name, "status": "skip", "reason": "no existing file"}

    dc, max_str = get_real_max_date(path)
    effective_dc = dc or date_col

    if max_str is None:
        log_print(f"[{name}] Cannot determine max date - "
                  f"full fetch needed, run scripts/main.py --force")
        return {"name": name, "status": "skip", "reason": "no date column"}

    max_dt = datetime.strptime(max_str, "%Y-%m-%d")
    base_overlap = overlap_days if overlap_days is not None else OVERLAP_DAYS
    effective_overlap = _get_effective_overlap(name, base_overlap, lag_days=lag_days)
    inc_start_dt = max_dt - timedelta(days=effective_overlap)
    default_dt = datetime.strptime(default_start, "%Y-%m-%d")
    if inc_start_dt < default_dt:
        inc_start_dt = default_dt
    inc_start = inc_start_dt.strftime("%Y-%m-%d")

    effective_end = EFFECTIVE_TODAY
    if lag_days:
        effective_end = _trading_day_offset(EFFECTIVE_TODAY, -lag_days)

    end_dt = datetime.strptime(effective_end, "%Y-%m-%d")
    if max_dt >= end_dt:
        log_print(f"[{name}] Up to date (max={max_str})")
        record_state(name, max_str, lag_days=lag_days)
        return {"name": name, "status": "uptodate", "max_date": max_str}

    log_print(f"[{name}] Existing max date: {max_str} | "
              f"Fetching: {inc_start} ~ {effective_end}")

    if dry_run:
        log_print(f"[{name}] DRY-RUN: would fetch {inc_start} ~ {effective_end}")
        return {"name": name, "status": "dry_run", "range": f"{inc_start}~{effective_end}"}

    tmp_output = str(path.parent / f"_incr_{name}.tmp.parquet")
    t_fetch = time.perf_counter()
    try:
        new_df = _call_fetch(
            fetch_fn,
            start_date=inc_start,
            end_date=effective_end,
            workers=workers,
            cleanup=True,
            resume=False,
            output=tmp_output,
            **(extra_params or {})
        )
    except Exception as e:
        add_timing(name, "fetch", time.perf_counter() - t_fetch, error=str(e))
        log_print(f"[{name}] Fetch FAILED: {e}")
        Path(tmp_output).unlink(missing_ok=True)
        record_state(name, None, lag_days=lag_days)
        return {"name": name, "status": "error", "error": str(e)}
    add_timing(name, "fetch", time.perf_counter() - t_fetch)

    tmp_path = Path(tmp_output)
    if tmp_path.exists():
        new_df = _safe_read_parquet(tmp_output)
        tmp_path.unlink(missing_ok=True)
    elif new_df is not None and not (hasattr(new_df, 'empty') and new_df.empty):
        pass
    else:
        log_print(f"[{name}] No new data returned")
        record_state(name, max_str, lag_days=lag_days)
        return {"name": name, "status": "uptodate", "max_date": max_str}

    if new_df is None or (hasattr(new_df, 'empty') and new_df.empty):
        log_print(f"[{name}] No new data returned")
        record_state(name, max_str, lag_days=lag_days)
        return {"name": name, "status": "uptodate", "max_date": max_str}

    t_merge = time.perf_counter()
    before, after, added = merge_and_save(
        filepath, new_df, dedup_keys, sort_cols, effective_dc,
        strict_dedup=strict_dedup,
    )
    add_timing(name, "merge", time.perf_counter() - t_merge,
               rows_before=before, rows_added=added)

    _, new_max = get_real_max_date(path)
    log_print(f"[{name}] Merge: {before} -> {after} rows (+{added}) | "
              f"max_date: {max_str} -> {new_max}")

    record_state(name, new_max, lag_days=lag_days)

    backfill_target = effective_end if lag_days else EFFECTIVE_TODAY
    if backfill and new_max and new_max < backfill_target:
        _backfill_missing_dates(filepath, effective_dc, backfill_target)

    return {
        "name": name,
        "status": "updated",
        "rows_before": before,
        "rows_after": after,
        "rows_added": added,
        "old_max": max_str,
        "new_max": new_max,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Local indicator incremental update
# ═══════════════════════════════════════════════════════════════════════════


def _process_indicator_stock_combined(code, history_dir_str, out_dir_str):
    """Standalone worker for ProcessPoolExecutor — combined indicator mode.

    Reads history_1min/{code}.parquet, computes all six indicators at once,
    merges with existing indicator file, and writes back.

    This is a MODULE-LEVEL function so it can be pickled by ProcessPoolExecutor.
    Returns a dict instead of mutating nonlocal variables.
    """
    import pandas as pd
    from pathlib import Path
    from fetch_indicator import _compute_indicators_for_df

    history_dir = Path(history_dir_str)
    out_dir = Path(out_dir_str)
    hist_file = history_dir / f"{code}.parquet"
    out_file = out_dir / f"{code}.parquet"

    if not hist_file.exists():
        return {"code": code, "status": "no_history"}

    # ── Fast early-skip: compare max trade_time via column projection ──
    try:
        _, hist_max = get_max_date(hist_file)
        if hist_max is None:
            return {"code": code, "status": "no_history"}
        if out_file.exists():
            _, ind_max = get_max_date(out_file)
            if ind_max is not None and ind_max >= hist_max:
                return {"code": code, "status": "uptodate"}
    except Exception:
        pass  # Fall through to safe full-processing path

    try:
        df_hist = pd.read_parquet(hist_file)
    except Exception:
        return {"code": code, "status": "error"}

    if df_hist.empty or "trade_time" not in df_hist.columns:
        return {"code": code, "status": "no_history"}

    df_hist["trade_time"] = pd.to_datetime(df_hist["trade_time"])
    df_hist = df_hist.sort_values("trade_time").reset_index(drop=True)

    needed = {"close", "high", "low", "vol"}
    if not needed.issubset(df_hist.columns):
        return {"code": code, "status": "error"}

    last_time = None
    df_existing = None
    if out_file.exists():
        try:
            df_existing = pd.read_parquet(out_file)
            if "trade_time" in df_existing.columns and not df_existing.empty:
                df_existing["trade_time"] = pd.to_datetime(df_existing["trade_time"])
                last_time = df_existing["trade_time"].max()
        except Exception:
            pass

    # Compute ALL six indicators at once
    all_indicators = _compute_indicators_for_df(df_hist, None)

    # Build combined output: trade_time, stock_code + all indicator columns
    out = df_hist[["trade_time"]].copy()
    out["stock_code"] = code
    for _name, ind_df in all_indicators.items():
        for col in ind_df.columns:
            out[col] = ind_df[col].values

    if last_time is not None:
        out = out[out["trade_time"] > last_time]

    if out.empty:
        return {"code": code, "status": "uptodate"}

    if df_existing is not None and last_time is not None:
        try:
            merged = pd.concat([df_existing, out], ignore_index=True)
            merged = merged.drop_duplicates(
                subset=["trade_time", "stock_code"], keep="last"
            )
            merged = merged.sort_values("trade_time").reset_index(drop=True)
        except Exception:
            merged = out
    else:
        merged = out

    tmp = out_file.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(out_file)

    return {"code": code, "status": "updated", "new_rows": len(out)}

def update_indicators_from_daily_dump(indicator_datasets, start_date, end_date,
                                       workers=6, dry_run=False, overlap_days=3):
    """Incrementally update technical indicators from local history_1min data.

    Pure local computation — ZERO API calls.

    Supports two modes:
      - Individual: each dataset has indicator_name (e.g. "macd"),
        saves to indicator_1min/{name}/{code}.parquet.
      - Combined: indicator_name is None (single dataset), all six
        indicators computed at once → indicator_1min/{code}.parquet.
    """
    from fetch_indicator import INDICATOR_COMPUTE, _compute_indicators_for_df

    CODE_NUM_FILE = Path(DATA_DIR).parent / "Code_num.txt"
    HISTORY_DIR = Path(DATA_DIR) / "history_1min"

    if not CODE_NUM_FILE.exists():
        log_print("[indicators] Code_num.txt not found - skip")
        return

    with open(CODE_NUM_FILE) as f:
        raw_codes = [line.strip() for line in f if line.strip()]

    def _add_suffix(code):
        return code + (".SH" if code.startswith("60") else ".SZ")

    all_stocks = [_add_suffix(c) for c in raw_codes]

    stocks = [c for c in all_stocks if (HISTORY_DIR / f"{c}.parquet").exists()]
    if not stocks:
        log_print("[indicators] No history_1min files found - skip")
        return

    log_print(f"[indicators] {len(stocks)} stocks, "
              f"{len(indicator_datasets)} indicator dataset(s): "
              f"{[d.get('indicator_name') or 'ALL' for d in indicator_datasets]}")

    if dry_run:
        log_print("[indicators] DRY-RUN - no files will be written")

        # Detect combined mode
        is_combined = any(not d.get("indicator_name") for d in indicator_datasets)
        if is_combined:
            ds = [d for d in indicator_datasets if not d.get("indicator_name")][0]
            out_dir = Path(DATA_DIR) / ds["out_subdir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            total_new = 0
            for code in stocks[:5]:
                hist_file = HISTORY_DIR / f"{code}.parquet"
                ind_file = out_dir / f"{code}.parquet"
                if hist_file.exists():
                    df_hist = pd.read_parquet(hist_file)
                    df_hist["trade_time"] = pd.to_datetime(df_hist["trade_time"])
                    hist_max = df_hist["trade_time"].max()
                    if ind_file.exists():
                        df_ind = pd.read_parquet(ind_file)
                        df_ind["trade_time"] = pd.to_datetime(df_ind["trade_time"])
                        ind_max = df_ind["trade_time"].max()
                        new_rows = len(df_hist[df_hist["trade_time"] > ind_max])
                    else:
                        new_rows = len(df_hist)
                    total_new += new_rows
            log_print(f"[indicators] DRY-RUN (combined): ~{total_new} new rows estimated (sampled)")
        else:
            total_new = 0
            for ds in indicator_datasets:
                out_dir = Path(DATA_DIR) / ds["out_subdir"]
                out_dir.mkdir(parents=True, exist_ok=True)
                for code in stocks[:5]:
                    hist_file = HISTORY_DIR / f"{code}.parquet"
                    ind_file = out_dir / f"{code}.parquet"
                    if hist_file.exists():
                        df_hist = pd.read_parquet(hist_file)
                        df_hist["trade_time"] = pd.to_datetime(df_hist["trade_time"])
                        hist_max = df_hist["trade_time"].max()
                        if ind_file.exists():
                            df_ind = pd.read_parquet(ind_file)
                            df_ind["trade_time"] = pd.to_datetime(df_ind["trade_time"])
                            ind_max = df_ind["trade_time"].max()
                            new_rows = len(df_hist[df_hist["trade_time"] > ind_max])
                        else:
                            new_rows = len(df_hist)
                        total_new += new_rows
            log_print(f"[indicators] DRY-RUN: ~{total_new} new rows estimated (sampled)")
        return

    # ── Detect combined vs individual mode ──
    is_combined = any(not d.get("indicator_name") for d in indicator_datasets)

    # Shared counters (accessible in both modes + progress logging below)
    total_new_rows = 0
    total_stocks_updated = 0
    skipped_uptodate = 0
    skipped_no_history = 0

    if is_combined:
        # ═══════════════ Combined mode ═══════════════
        ds = [d for d in indicator_datasets if not d.get("indicator_name")][0]
        out_dir = Path(DATA_DIR) / ds["out_subdir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        lock = threading.Lock()
        completed = 0
        t_start = time.perf_counter()


        # ── ProcessPoolExecutor for CPU-bound indicator computation ──
        # Uses all available CPU cores, auto-adapting to the current machine.
        import os as _os
        from concurrent.futures import ProcessPoolExecutor as _PPE
        _cpu_count = _os.cpu_count() or 4
        _max_w = min(_cpu_count, len(stocks))  # Always use all CPU cores for local computation

        with _PPE(max_workers=_max_w) as pool:
            futures = {pool.submit(_process_indicator_stock_combined, c, str(HISTORY_DIR), str(out_dir)): c
                       for c in stocks}
            for fut in as_completed(futures):
                code = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    with lock:
                        completed += 1
                    log_print(f"[indicators] {code} FAILED: {e}")
                    continue

                with lock:
                    completed += 1
                    status = result.get("status", "error")
                    if status == "updated":
                        total_new_rows += result.get("new_rows", 0)
                        total_stocks_updated += 1
                    elif status == "uptodate":
                        skipped_uptodate += 1
                    elif status == "no_history":
                        skipped_no_history += 1
                    # "error" status: already counted in completed, nothing else to do

                    if completed % 200 == 0 or completed == len(stocks):
                        elapsed = time.perf_counter() - t_start
                        rate = completed / elapsed if elapsed > 0 else 0
                        log_print(f"[indicators] {completed}/{len(stocks)} stocks "
                                  f"({rate:.0f} st/s) | new_rows={total_new_rows} "
                                  f"| skip={skipped_uptodate}")

        elapsed = time.perf_counter() - t_start
        log_print(f"[indicators] Done: {total_stocks_updated} stocks updated, "
                  f"{skipped_uptodate} skipped (uptodate), "
                  f"{skipped_no_history} skipped (no history), "
                  f"{total_new_rows} new rows | {elapsed:.1f}s "
                  f"(workers={_max_w}/{_cpu_count} CPUs)")

        done_file = out_dir / ".done"
        done_file.write_text(str(date.today()))
        log_print("[indicators] .done marker written")
        return  # Combined mode done — skip shared ThreadPoolExecutor block
    else:
        # ═══════════════ Individual mode (original) ═══════════════
        indicator_map = {}
        for ds in indicator_datasets:
            ind_name = ds["indicator_name"]
            if ind_name not in INDICATOR_COMPUTE:
                log_print(f"[indicators] Unknown indicator '{ind_name}' - skip")
                continue
            out_dir = Path(DATA_DIR) / ds["out_subdir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            compute_fn, params = INDICATOR_COMPUTE[ind_name]
            indicator_map[ind_name] = (out_dir, compute_fn, params)

        if not indicator_map:
            log_print("[indicators] No valid indicators to process")
            return

        lock = threading.Lock()
        completed = 0
        t_start = time.perf_counter()

        def _process_one_stock(code):
            nonlocal total_new_rows, total_stocks_updated, completed
            nonlocal skipped_uptodate, skipped_no_history

            hist_file = HISTORY_DIR / f"{code}.parquet"
            if not hist_file.exists():
                with lock:
                    completed += 1
                    skipped_no_history += 1
                return

            # ── Fast early-skip: check if ALL indicators are already up to date ──
            try:
                _, hist_max = get_max_date(hist_file)
                if hist_max is None:
                    with lock:
                        completed += 1
                        skipped_no_history += 1
                    return
                all_uptodate = True
                for ind_name, (out_dir, _, _) in indicator_map.items():
                    ind_file = out_dir / f"{code}.parquet"
                    if not ind_file.exists():
                        all_uptodate = False
                        break
                    _, ind_max = get_max_date(ind_file)
                    if ind_max is None or ind_max < hist_max:
                        all_uptodate = False
                        break
                if all_uptodate:
                    with lock:
                        completed += 1
                        skipped_uptodate += 1
                    return
            except Exception:
                pass  # Fall through to safe full-processing path

            try:
                df_hist = pd.read_parquet(hist_file)
            except Exception:
                with lock:
                    completed += 1
                return

            if df_hist.empty or "trade_time" not in df_hist.columns:
                with lock:
                    completed += 1
                return

            df_hist["trade_time"] = pd.to_datetime(df_hist["trade_time"])
            df_hist = df_hist.sort_values("trade_time").reset_index(drop=True)

            needed = {"close", "high", "low", "vol"}
            if not needed.issubset(df_hist.columns):
                with lock:
                    completed += 1
                return

            stock_updated = False

            for ind_name, (out_dir, compute_fn, params) in indicator_map.items():
                ind_file = out_dir / f"{code}.parquet"

                if ind_file.exists():
                    try:
                        df_existing = pd.read_parquet(ind_file)
                        if "trade_time" in df_existing.columns and not df_existing.empty:
                            df_existing["trade_time"] = pd.to_datetime(df_existing["trade_time"])
                            last_time = df_existing["trade_time"].max()
                            new_mask = df_hist["trade_time"] > last_time
                            if not new_mask.any():
                                continue
                        else:
                            last_time = None
                            new_mask = slice(None)
                    except Exception:
                        last_time = None
                        new_mask = slice(None)
                else:
                    last_time = None
                    new_mask = slice(None)

                ind_df = compute_fn(df_hist, **params)

                out_cols = ["trade_time", "stock_code"] + list(ind_df.columns)
                out = df_hist[["trade_time"]].copy()
                out["stock_code"] = code
                for col in ind_df.columns:
                    out[col] = ind_df[col].values

                if last_time is not None:
                    out = out[out["trade_time"] > last_time]

                if out.empty:
                    continue

                if ind_file.exists() and last_time is not None:
                    try:
                        df_existing = pd.read_parquet(ind_file)
                        df_existing["trade_time"] = pd.to_datetime(df_existing["trade_time"])
                        merged = pd.concat([df_existing, out], ignore_index=True)
                        merged = merged.drop_duplicates(
                            subset=["trade_time", "stock_code"], keep="last"
                        )
                        merged = merged.sort_values("trade_time").reset_index(drop=True)
                    except Exception:
                        merged = out
                else:
                    merged = out

                tmp = ind_file.with_suffix(".parquet.tmp")
                merged.to_parquet(tmp, index=False)
                tmp.replace(ind_file)
                stock_updated = True

                with lock:
                    total_new_rows += len(out)

            if stock_updated:
                with lock:
                    total_stocks_updated += 1

            with lock:
                completed += 1

    if workers > 1 and len(stocks) > 1:
        max_w = min(workers, len(stocks))
        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futures = {pool.submit(_process_one_stock, c): c for c in stocks}
            for fut in as_completed(futures):
                code = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    with lock:
                        completed += 1
                    log_print(f"[indicators] {code} FAILED: {e}")
                with lock:
                    if completed % 200 == 0 or completed == len(stocks):
                        elapsed = time.perf_counter() - t_start
                        rate = completed / elapsed if elapsed > 0 else 0
                        log_print(f"[indicators] {completed}/{len(stocks)} stocks "
                                  f"({rate:.0f} st/s) | new_rows={total_new_rows} "
                                  f"| skip={skipped_uptodate}")
    else:
        for code in stocks:
            _process_one_stock(code)
            if completed % 200 == 0:
                log_print(f"[indicators] {completed}/{len(stocks)} stocks | "
                          f"new_rows={total_new_rows} | skip={skipped_uptodate}")

    elapsed = time.perf_counter() - t_start
    log_print(f"[indicators] Done: {total_stocks_updated} stocks updated, "
              f"{skipped_uptodate} skipped (uptodate), "
              f"{skipped_no_history} skipped (no history), "
              f"{total_new_rows} new rows | {elapsed:.1f}s")

    if is_combined:
        done_file = out_dir / ".done"
        done_file.write_text(str(date.today()))
        log_print("[indicators] .done marker written")
    else:
        for ds in indicator_datasets:
            out_dir = Path(DATA_DIR) / ds["out_subdir"]
            done_file = out_dir / ".done"
            done_file.write_text(str(date.today()))
        log_print("[indicators] .done markers written")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset registry bridge — populated by incremental.py at import time
# ═══════════════════════════════════════════════════════════════════════════

_DATASETS = []


def set_datasets(datasets):
    """Set the dataset registry (called by incremental.py)."""
    global _DATASETS
    _DATASETS = datasets


def _get_datasets():
    """Return the current dataset registry."""
    return _DATASETS
