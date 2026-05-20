"""
Fetch 1min K-line data from /api/stock/history.
Per-stock parallel, main-board only.

Usage:
  python fetch_minute.py --start 2019-01-01 -w 3
"""

import requests
import argparse
import pandas as pd
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (load_api_keys, BASE_URL, DATA_DIR, RateLimiter, log_print)

ENDPOINT = "stock/history"
LEVEL = "1min"
TAG = "history_1min"

CODE_NUM_FILE = Path(DATA_DIR).parent / "Code_num.txt"


def _add_exchange_suffix(code):
    """Add .SZ or .SH suffix to a 6-digit stock code."""
    return code + (".SH" if code.startswith("60") else ".SZ")

# ── Channel pool (dual-key, round-robin) ───────────────────────────────
# Each API key gets its own RateLimiter (280 req/min) and concurrency
# gate (3 in-flight).  Pages are distributed round-robin across channels,
# doubling effective throughput to ~560 req/min.

class _Channel:
    __slots__ = ('api_key', 'limiter', 'gate')
    def __init__(self, api_key, rpm=280, max_concurrent=3):
        self.api_key = api_key
        self.limiter = RateLimiter(max_rpm=rpm)
        self.gate = threading.BoundedSemaphore(max_concurrent)

_channels = None
_channel_lock = threading.Lock()
_channel_idx = 0


def _init_channels():
    global _channels, _channel_idx
    if _channels is not None:
        return
    keys = load_api_keys()
    _channels = [_Channel(k) for k in keys]
    _channel_idx = 0
    log_print(f"[minute] {len(_channels)} API key(s) loaded, "
              f"{len(_channels) * 280} req/min capacity")


def _next_channel():
    global _channel_idx
    with _channel_lock:
        ch = _channels[_channel_idx % len(_channels)]
        _channel_idx += 1
        return ch


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
    """Fetch one page via a round-robin channel.

    Each channel has its own RateLimiter (280 req/min) and concurrency
    gate (3 in-flight).  Round-robin distributes load across all keys.
    """
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        ch = _next_channel()
        headers = {"apiKey": ch.api_key, "Content-Type": "application/json"}

        ch.limiter.acquire(endpoint)        # per-key rate throttle
        ch.gate.acquire()                   # per-key burst prevention
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()

            if result["code"] != 200:
                msg = result.get("msg", str(result))
                raise RuntimeError(f"API Error: code={result['code']}, msg={msg}")

            data = result["data"]
            return data.get("list", []), data.get("total", 0)

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            is_429 = "429" in str(e) or "频繁" in str(e)
            if attempt < retries - 1:
                delay = (15 * (2 ** attempt)) if is_429 else (2 ** attempt)
                log_print(f"  [retry {attempt+1}/{retries}] {e}")
                time.sleep(delay)
            else:
                raise
        finally:
            ch.gate.release()


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


# ── API helpers ─────────────────────────────────────────────────────────

def _fetch_chunk(stock_code, chunk_start, chunk_end):
    """Fetch all pages for one stock × date range.

    Returns (rows, capped).  Uses the last page's row count (not the API
    `total` field) to decide whether more pages exist, because the API
    may report an inflated `total`.
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
    """Fetch all 1min data for one stock with proactive chunk splitting.

    The date range is divided upfront into fixed-size chunks guaranteed to
    fit within the 100K pagination limit.  The old trial-based adaptive
    splitting is kept only as a fallback for unexpected edge cases.

    All-or-nothing: if any chunk fails, the stock is NOT saved so it will
    be re-fetched from scratch on restart.
    """
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

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.sort_values("trade_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        out_file = out_dir / f"{stock_code}.parquet"
        tmp = out_file.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out_file)
        return len(df)

    pd.DataFrame(columns=["trade_time", "stock_code", "open", "high", "low",
                          "close", "vol", "amount"]).to_parquet(
        out_dir / f"{stock_code}.parquet", index=False
    )
    return 0


# ── Orchestrator ────────────────────────────────────────────────────────

def _channel_stats():
    """Aggregate token stats across all channels."""
    total_tokens = 0
    total_max = 0
    for ch in _channels:
        s = ch.limiter.stats
        total_tokens += s['tokens_available']
        total_max += s['max_rpm']
    return total_tokens, total_max


def _run(start_date, end_date, resume, workers, cleanup):
    _init_channels()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    stocks = _filter_stocks()
    if stocks is None:
        log_print(f"[{TAG}] No stock list, abort")
        return pd.DataFrame()

    out_dir = Path(DATA_DIR) / TAG
    out_dir.mkdir(exist_ok=True)

    # ETA: calculate actual chunk count from a sample date range
    sample_chunks = _generate_chunks(start_date, end_date)
    chunks_per_stock = len(sample_chunks)
    est_calls = len(stocks) * chunks_per_stock
    total_rpm = len(_channels) * 280
    total_slots = len(_channels) * 3
    est_min = est_calls / total_rpm

    log_print(f"[{TAG}] {len(stocks)} stocks | level={LEVEL} | "
              f"{start_date} ~ {end_date}")
    log_print(f"[{TAG}] {len(_channels)} key(s) | "
              f"{total_slots} in-flight max | "
              f"rate-limit: {total_rpm}/min")
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
              f"(parallel ×{total_slots} slots)")

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
                tok, max_rpm = _channel_stats()
                bar_w = 20
                filled = int(bar_w * completed / total_todo)
                bar = "#" * filled + "-" * (bar_w - filled)
                log_print(f"[{TAG}] [{bar}] {pct:5.1f}%  {completed}/{total_todo}  "
                          f"{rate:.0f} st/min  ETA {eta_str}  "
                          f"rows:{total_rows}  fail:{failed}  "
                          f"tok:{tok:.0f}/{max_rpm}")
        return n

    max_workers = min(total_slots, total_todo)
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
                  resume=True, workers=3, cleanup=True):
    """Fetch 1min K-line from /api/stock/history."""
    return _run(start_date, end_date, resume, workers, cleanup)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch 1min K-line data")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints")
    parser.add_argument("-w", "--workers", type=int, default=3,
                        help="Chunk workers per stock (default: 3)")
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
