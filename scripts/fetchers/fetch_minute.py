"""
Fetch 1min K-line data from /api/stock/history.
Per-stock parallel, main-board only.  Single API key — 280 req/min.

Usage:
  python fetch_minute.py --start 2019-01-01 -w 6
"""

import requests
import argparse
import pandas as pd
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print)

ENDPOINT = "stock/history"
LEVEL = "1min"
TAG = "history_1min"

CODE_NUM_FILE = Path(DATA_DIR).parent / "Code_num.txt"


def _add_exchange_suffix(code):
    """Add .SZ or .SH suffix to a 6-digit stock code."""
    return code + (".SH" if code.startswith("60") else ".SZ")

# ── Single-key helpers ─────────────────────────────────────────────────
# One API key → 280 req/min → ~4.67 req/s sustained.
# The global rate_limiter (from config) handles long-term pacing; a small
# BoundedSemaphore prevents connection storms when many workers fire at once.

_api_key = None
_concurrency_gate = None


def _get_api_key():
    """Lazy-load and cache the single API key."""
    global _api_key
    if _api_key is None:
        _api_key = load_api_key()
        log_print(f"[minute] API key loaded: ...{_api_key[-8:]}")
    return _api_key


def _set_concurrency_gate(max_concurrent):
    """Initialise the concurrency gate (called once by _run)."""
    global _concurrency_gate
    if _concurrency_gate is None:
        _concurrency_gate = threading.BoundedSemaphore(max_concurrent)


def _filter_stocks():
    """Return list of stock codes from Code_num.txt with exchange suffixes."""
    if not CODE_NUM_FILE.exists():
        log_print(f"[minute] {CODE_NUM_FILE} not found, no filtering")
        return None
    with open(CODE_NUM_FILE) as f:
        codes = [_add_exchange_suffix(line.strip()) for line in f if line.strip()]
    log_print(f"[minute] {len(codes)} stocks loaded from Code_num.txt")
    return codes


# ── API helpers ─────────────────────────────────────────────────────────

def _fetch_page(endpoint, payload, retries=3):
    """Fetch one page using the single API key.

    Uses the global rate limiter (280 req/min) for long-term pacing and a
    concurrency gate to prevent connection storms.

    On 402 (no permission), raises immediately — there is no fallback key.
    On 429 (rate limit), retries with exponential backoff.
    """
    url = f"{BASE_URL}/{endpoint}"
    api_key = _get_api_key()

    for attempt in range(retries):
        rate_limiter().acquire(endpoint)          # global rate throttle
        _concurrency_gate.acquire()                # connection throttle
        try:
            resp = requests.post(
                url,
                headers={"apiKey": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()

            if result["code"] != 200:
                msg = result.get("msg", str(result))
                raise RuntimeError(f"API Error: code={result['code']}, msg={msg}")

            data = result["data"]
            return data.get("list", []), data.get("total", 0)

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            is_402 = "402" in str(e)
            is_429 = "429" in str(e) or "频繁" in str(e)

            if is_402:
                raise RuntimeError(
                    f"API returned 402 (no permission) for key ...{api_key[-8:]}. "
                    f"Only one key configured — cannot fall back."
                ) from e

            if attempt < retries - 1:
                delay = (15 * (2 ** attempt)) if is_429 else (2 ** attempt)
                log_print(f"  [retry {attempt+1}/{retries}] {e}")
                time.sleep(delay)
            else:
                raise
        finally:
            _concurrency_gate.release()


# ── Proactive chunk sizing ─────────────────────────────────────────────

# 1min bars/day.  API pagination caps at 100K rows (10 pages × 10K).
# We split into 180-day chunks — at 240 bars/day that's ≤43K rows, well
# under the limit even when the API `total` field is unreliable.
_DAYS_PER_CHUNK = 180
_PAGE_SIZE = 10000
_MAX_PAGE = (100000 // _PAGE_SIZE) - 1  # 9 → pages 0-9 = 10 pages = 100K


def _generate_chunks(range_start, range_end):
    """Split a date range into fixed 180-day chunks, each guaranteed
    to produce ≤43K rows — safely under the 100K pagination limit."""
    start_dt = datetime.strptime(range_start, "%Y-%m-%d")
    end_dt = datetime.strptime(range_end, "%Y-%m-%d")
    chunks = []
    cs = start_dt
    while cs <= end_dt:
        ce = min(cs + timedelta(days=_DAYS_PER_CHUNK - 1), end_dt)
        chunks.append((cs.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))
        cs = ce + timedelta(days=1)
    return chunks


# ── Chunk fetch ────────────────────────────────────────────────────────

def _fetch_chunk(stock_code, chunk_start, chunk_end):
    """Fetch all pages for one stock × date range.

    Returns (rows, capped).  Uses the last page's row count (not the API
    ``total`` field) to decide whether more pages exist, because the API
    may report an inflated ``total``.
    """
    payload = {
        "stock_code": [stock_code],
        "level": LEVEL,
        "start_time": f"{chunk_start} 00:00:00",
        "end_time":   f"{chunk_end} 23:59:59",
        "page": 0,
        "page_size": _PAGE_SIZE,
    }

    page0, total = _fetch_page(ENDPOINT, payload)
    if not page0:
        return [], False

    all_data = list(page0)
    last_page_size = len(page0)
    page = 1
    while last_page_size > 0 and page <= _MAX_PAGE:
        payload["page"] = page
        b, _ = _fetch_page(ENDPOINT, payload)
        if not b:
            break
        last_page_size = len(b)
        all_data.extend(b)
        page += 1

    # capped when we hit the page limit AND the last page had data
    # (API likely has more rows beyond what its pagination allows)
    capped = page > _MAX_PAGE and last_page_size > 0
    return all_data, capped


# ── Fallback: adaptive split for unexpectedly capped chunks ────────────

def _fetch_range_adaptive(stock_code, range_start, range_end, depth=0):
    """Fetch a date range, recursively splitting when capped by the 100K limit.

    Only reached when a proactive chunk is unexpectedly capped (e.g. stock
    with unusually dense data).  Returns (rows, failed).
    """
    rows, capped = _fetch_chunk(stock_code, range_start, range_end)
    if not capped:
        return rows, False

    start_dt = datetime.strptime(range_start, "%Y-%m-%d")
    end_dt = datetime.strptime(range_end, "%Y-%m-%d")

    if start_dt >= end_dt:
        log_print(f"  [{stock_code}] {range_start} single day still capped "
                  f"({len(rows)} rows)")
        return rows, True

    mid = start_dt + (end_dt - start_dt) // 2
    mid_str = mid.strftime("%Y-%m-%d")
    next_str = (mid + timedelta(days=1)).strftime("%Y-%m-%d")

    log_print(f"  [{stock_code}] capped {len(rows)} rows → split "
              f"{range_start}~{mid_str} | {next_str}~{range_end}")

    left_rows, left_fail = _fetch_range_adaptive(
        stock_code, range_start, mid_str, depth + 1)
    right_rows, right_fail = _fetch_range_adaptive(
        stock_code, next_str, range_end, depth + 1)

    return left_rows + right_rows, left_fail or right_fail


# ── Per-stock worker ────────────────────────────────────────────────────

def _fetch_stock(stock_code, start_date, end_date, out_dir):
    """Fetch 1min data for ``stock_code`` and **merge** with existing file.

    * Reads the existing parquet file (if any).
    * Fetches the requested date range from the API.
    * Concatenates old + new, then deduplicates by ``trade_time``.
    * Saves the merged result atomically.

    All-or-nothing per fetch: if any chunk fails, new data is discarded and
    the existing file is left untouched.
    """
    out_file = out_dir / f"{stock_code}.parquet"

    # ── Read existing file (if any) ──
    existing_df = None
    if out_file.exists():
        existing_df = pd.read_parquet(out_file)
        existing_df["trade_time"] = pd.to_datetime(existing_df["trade_time"])

    # ── Fetch new data ──
    chunks = _generate_chunks(start_date, end_date)
    all_rows = []
    t_stock = time.perf_counter()

    for i, (cs, ce) in enumerate(chunks):
        t0 = time.perf_counter()
        rows, capped = _fetch_chunk(stock_code, cs, ce)
        if capped:
            log_print(f"  [{stock_code}] {cs}~{ce} unexpectedly capped → fallback")
            rows, fb_failed = _fetch_range_adaptive(stock_code, cs, ce)
            if fb_failed:
                log_print(f"  [{stock_code}] SKIPPED — {len(rows)} rows discarded")
                return 0
        all_rows.extend(rows)
        log_print(f"  [{stock_code}] chunk {i+1}/{len(chunks)} {cs}~{ce} "
                  f"→ {len(rows)}r/{time.perf_counter()-t0:.1f}s "
                  f"(total {len(all_rows)}r/{time.perf_counter()-t_stock:.0f}s)")

    if not all_rows:
        # No new data — keep existing file unchanged
        return len(existing_df) if existing_df is not None else 0

    new_df = pd.DataFrame(all_rows)
    new_df["trade_time"] = pd.to_datetime(new_df["trade_time"])

    # ── Merge with existing data ──
    if existing_df is not None and len(existing_df) > 0:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        before_dedup = len(merged)
        merged = merged.drop_duplicates(subset=["trade_time", "stock_code"], keep="last")
        log_print(f"  [{stock_code}] merge: existing={len(existing_df)} + new={len(new_df)} "
                  f"→ {before_dedup} rows, dedup→{len(merged)} rows "
                  f"(removed {before_dedup - len(merged)} duplicates)")
    else:
        merged = new_df
        log_print(f"  [{stock_code}] new: {len(new_df)} rows (no existing file)")

    merged = merged.sort_values("trade_time").reset_index(drop=True)

    # ── Atomic write ──
    tmp = out_file.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(out_file)
    return len(merged)


# ── Orchestrator ────────────────────────────────────────────────────────

def _run(start_date, end_date, resume, workers, cleanup):
    # Initialise single-key concurrency gate
    _set_concurrency_gate(workers)

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    stocks = _filter_stocks()
    if stocks is None:
        log_print(f"[{TAG}] No stock list, abort")
        return pd.DataFrame()

    out_dir = Path(DATA_DIR) / TAG
    out_dir.mkdir(exist_ok=True)

    # ETA
    sample_chunks = _generate_chunks(start_date, end_date)
    chunks_per_stock = len(sample_chunks)
    est_calls = len(stocks) * chunks_per_stock
    rpm = 280
    est_min = est_calls / rpm

    log_print(f"[{TAG}] {len(stocks)} stocks | level={LEVEL} | "
              f"{start_date} ~ {end_date}")
    log_print(f"[{TAG}] 1 API key | rate-limit: {rpm}/min | "
              f"parallel: ×{workers} stocks")
    log_print(f"[{TAG}] ~{est_calls} API calls "
              f"({chunks_per_stock} chunks/stock × {len(stocks)} stocks) | "
              f"~{est_min:.0f} min est")

    # Determine which stocks need fetching
    todo = []
    for code in stocks:
        f = out_dir / f"{code}.parquet"
        if resume and f.exists():
            try:
                pd.read_parquet(f)
                continue
            except Exception:
                f.unlink(missing_ok=True)
        todo.append(code)

    if not todo:
        log_print(f"[{TAG}] All stocks already cached")
        return pd.DataFrame()

    total_todo = len(todo)
    log_print(f"[{TAG}] {total_todo} stocks to fetch "
              f"(parallel ×{workers})")

    # ── Parallel stock fetch ──────────────────────────────────────────
    total_rows = 0
    completed = 0
    failed = 0
    t_start = time.perf_counter()
    progress_lock = threading.Lock()
    last_log_time = [time.perf_counter()]  # mutable for closure

    def _fetch_one(code):
        nonlocal total_rows, completed, failed
        n = _fetch_stock(code, start_date, end_date, out_dir)
        with progress_lock:
            completed += 1
            total_rows += n
            if n == 0:
                failed += 1
            # throttle progress logging to ~1 line/sec
            now = time.perf_counter()
            if now - last_log_time[0] >= 1.0:
                last_log_time[0] = now
                elapsed = now - t_start
                pct = completed / total_todo * 100
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                eta_s = (total_todo - completed) / rate * 60 if rate > 0 else 0
                eta_str = f"{eta_s/60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"
                s = rate_limiter().stats
                bar_w = 20
                filled = int(bar_w * completed / total_todo)
                bar = "#" * filled + "-" * (bar_w - filled)
                log_print(f"[{TAG}] [{bar}] {pct:5.1f}%  {completed}/{total_todo}  "
                          f"{rate:.0f} st/min  ETA {eta_str}  "
                          f"rows:{total_rows}  fail:{failed}  "
                          f"tok:{s['tokens_available']:.0f}/{s['max_rpm']}")
        return n

    max_workers = min(workers, total_todo)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, code): code for code in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                code = futures[fut]
                with progress_lock:
                    completed += 1
                    failed += 1
                log_print(f"[{TAG}] {code} FAILED: {e}")

    elapsed_total = time.perf_counter() - t_start
    log_print(f"[{TAG}] Done: {total_rows} rows from {total_todo} stocks "
              f"(failed:{failed}) in {elapsed_total/60:.1f} min "
              f"({total_todo/(elapsed_total/60):.0f} st/min)")
    done_file = out_dir / ".done"
    done_file.write_text(str(date.today()))
    log_print(f"[{TAG}] Done marker -> {done_file}")
    return pd.DataFrame()


# ── Public entry point ─────────────────────────────────────────────────

def fetch_history(start_date="2019-01-01", end_date=None, output=None,
                  resume=True, workers=6, cleanup=True):
    """Fetch 1min K-line from /api/stock/history."""
    return _run(start_date, end_date, resume, workers, cleanup)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch 1min K-line data")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints")
    parser.add_argument("-w", "--workers", type=int, default=6,
                        help="Parallel stock workers (default: 6, max ~280 req/min)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-stock checkpoint files")
    args = parser.parse_args()

    fetch_history(
        start_date=args.start,
        end_date=args.end,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
    )
