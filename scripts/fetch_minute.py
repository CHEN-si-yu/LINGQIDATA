"""
Fetch minute-level K-line data: /api/stock/history (raw) and
/api/stock/min_adj (adjusted).  Per-stock parallel, main-board only.

Usage:
  python fetch_minute.py history --level 60min --start 2019-01-01 -w 3
  python fetch_minute.py min_adj  --level 5min  --start 2020-01-01 -w 3
"""

import requests
import argparse
import pandas as pd
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print


ENDPOINTS = {
    "history":  "stock/history",
    "min_adj":  "stock/min_adj",
}
_LEVELS = ["60min", "30min", "15min", "5min", "1min"]

# Bars-per-day estimate per level (used for ETA estimates only).
# Actual fetching uses adaptive range splitting — no fixed chunks.
_BARS_PER_DAY = {"1min": 240, "5min": 48, "15min": 16, "30min": 8, "60min": 4}

CODE_NUM_FILE = Path(DATA_DIR).parent / "Code_num.txt"


def _add_exchange_suffix(code):
    """Add .SZ or .SH suffix to a 6-digit stock code."""
    return code + (".SH" if code.startswith("60") else ".SZ")

# ── Concurrency gate ────────────────────────────────────────────────────
# The global RateLimiter controls long-term throughput (280 req/min).
# However its token bucket starts empty and the first acquire() after a
# long idle period fills it to 280 instantly — causing a burst of
# concurrent requests that trigger server-side 429.
#
# This semaphore caps the number of *simultaneous in-flight* HTTP
# requests.  Together they provide: burst prevention (semaphore) +
# sustained rate control (token bucket).
_MAX_CONCURRENT = 3
_api_gate = threading.BoundedSemaphore(_MAX_CONCURRENT)


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

def _fetch_page(endpoint, payload, api_key, retries=3):
    """Fetch one page.  Acquires both rate-limiter token *and* concurrency
    gate slot before issuing the HTTP request."""
    url = f"{BASE_URL}/{endpoint}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}

    for attempt in range(retries):
        limiter = rate_limiter()
        limiter.acquire(endpoint)          # long-term rate throttle
        _api_gate.acquire()                # burst prevention
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
            _api_gate.release()


def _fetch_chunk(endpoint, stock_code, level, chunk_start, chunk_end,
                 api_key, extra_payload=None):
    """Fetch all pages for one stock × level × date range.

    Returns (rows, capped).  capped=True when the 100K pagination limit
    prevented fetching all rows — the caller should split and retry.
    """
    payload = {
        "stock_code": stock_code if endpoint == "stock/min_adj" else [stock_code],
        "level": level,
        "start_time": f"{chunk_start} 00:00:00",
        "end_time":   f"{chunk_end} 23:59:59",
        "page": 0,
        "page_size": 10000,
    }
    if extra_payload:
        payload.update(extra_payload)

    page0, total = _fetch_page(endpoint, payload, api_key)
    if not page0:
        return [], False

    all_data = list(page0)
    max_page = 100000 // 10000  # 9 → page 0..9 = 10 pages = 100K rows
    page = 1
    while len(all_data) < total and page <= max_page:
        payload["page"] = page
        b, _ = _fetch_page(endpoint, payload, api_key)
        if not b:
            break
        all_data.extend(b)
        page += 1

    capped = len(all_data) < total
    return all_data, capped


# ── Adaptive range fetcher ─────────────────────────────────────────────

def _fetch_range_adaptive(endpoint, stock_code, level, range_start, range_end,
                          api_key, extra_payload, depth=0):
    """Fetch a date range, recursively splitting when capped by the 100K limit.

    Returns (rows, failed).
    """
    rows, capped = _fetch_chunk(endpoint, stock_code, level, range_start, range_end,
                                api_key, extra_payload)
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
        endpoint, stock_code, level, range_start, mid_str,
        api_key, extra_payload, depth + 1)
    right_rows, right_fail = _fetch_range_adaptive(
        endpoint, stock_code, level, next_str, range_end,
        api_key, extra_payload, depth + 1)

    return left_rows + right_rows, left_fail or right_fail


# ── Per-stock worker ────────────────────────────────────────────────────

def _fetch_stock(endpoint, stock_code, level, start_date, end_date,
                 api_key, out_dir, extra_payload=None, chunk_workers=3):
    """Fetch all minute data for one stock with adaptive range splitting.

    Starts with the full date range.  Only splits when the API 100K
    pagination limit prevents returning all rows (fine levels like 1min).

    All-or-nothing: if any sub-range fails, the stock is NOT saved so it
    will be re-fetched from scratch on restart.
    """
    all_rows, failed = _fetch_range_adaptive(
        endpoint, stock_code, level, start_date, end_date,
        api_key, extra_payload)

    if failed:
        log_print(f"  [{stock_code}] SKIPPED — {len(all_rows)} rows discarded")
        return 0

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

def _run(endpoint, level, start_date, end_date, output, resume, workers,
         cleanup, extra_payload=None, tag=None):
    if tag is None:
        tag = f"{endpoint.replace('stock/', '')}_{level}"

    api_key = load_api_key()
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    stocks = _filter_stocks()
    if stocks is None:
        log_print(f"[{tag}] No stock list, abort")
        return pd.DataFrame()

    out_dir = Path(DATA_DIR) / tag
    out_dir.mkdir(exist_ok=True)

    # Rough ETA estimate — coarse levels fit one call per stock,
    # fine levels may split.  Assume 1 call/stock for 60/30min,
    # 2 for 15min, 4 for 5min, 16 for 1min as a rough guide.
    splits_est = {"1min": 16, "5min": 4, "15min": 2, "30min": 1, "60min": 1}
    est_calls = len(stocks) * splits_est.get(level, 1)
    est_min = est_calls / 280.0
    bars_per_day = _BARS_PER_DAY.get(level, 4)

    log_print(f"[{tag}] {len(stocks)} stocks | level={level} | "
              f"{start_date} ~ {end_date}")
    log_print(f"[{tag}] Concurrency: {_MAX_CONCURRENT} in-flight max | "
              f"rate-limit: 280/min")
    log_print(f"[{tag}] ~{est_calls} API calls (adaptive) | "
              f"~{est_min:.0f} min est | ~{bars_per_day} bars/day")

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
        log_print(f"[{tag}] All stocks already cached")
        return pd.DataFrame()

    total_todo = len(todo)
    log_print(f"[{tag}] {total_todo} stocks to fetch")

    total_rows = 0
    t_start = time.perf_counter()
    limiter = rate_limiter()
    elapsed_smooth = None  # exponential moving average of per-stock time

    for i, code in enumerate(todo):
        t0 = time.perf_counter()
        try:
            n = _fetch_stock(endpoint, code, level, start_date, end_date,
                             api_key, out_dir, extra_payload=extra_payload,
                             chunk_workers=workers)
        except Exception as e:
            log_print(f"[{tag}] [{i+1}/{total_todo}] {code} FAILED: {e}")
            n = 0

        elapsed = time.perf_counter() - t0
        total_rows += n

        # Smooth average for ETA
        if elapsed_smooth is None:
            elapsed_smooth = elapsed
        else:
            elapsed_smooth = 0.9 * elapsed_smooth + 0.1 * elapsed

        remaining = total_todo - (i + 1)
        eta_s = elapsed_smooth * remaining
        eta_str = f"{eta_s/60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"

        pct = (i + 1) / total_todo * 100
        bar_w = 30
        filled = int(bar_w * (i + 1) / total_todo)
        bar = "#" * filled + "-" * (bar_w - filled)

        s = limiter.stats
        log_print(f"[{tag}] [{bar}] {pct:5.1f}%  {i+1}/{total_todo}  "
                  f"ETA {eta_str}  "
                  f"last:{code} {n}r/{elapsed:.1f}s  "
                  f"tok:{s['tokens_available']:.0f}/{s['max_rpm']}")

    elapsed_total = time.perf_counter() - t_start
    log_print(f"[{tag}] Done: {total_rows} rows from {total_todo} stocks "
              f"in {elapsed_total/60:.1f} min")
    done_file = out_dir / ".done"
    done_file.write_text(str(date.today()))
    log_print(f"[{tag}] Done marker -> {done_file}")
    return pd.DataFrame()


# ── Public entry points ─────────────────────────────────────────────────

def fetch_history(start_date="2019-01-01", end_date=None, output=None,
                  resume=True, workers=3, cleanup=True, level="5min"):
    """Fetch raw minute K-line (/api/stock/history)."""
    return _run("stock/history", level, start_date, end_date, output,
                resume, workers, cleanup, tag=f"history_{level}")


def fetch_min_adj(start_date="2019-01-01", end_date=None, output=None,
                  resume=True, workers=3, cleanup=True, level="5min"):
    """Fetch adjusted minute K-line (/api/stock/min_adj)."""
    return _run("stock/min_adj", level, start_date, end_date, output,
                resume, workers, cleanup,
                extra_payload={"algo": "recursive"},
                tag=f"min_adj_{level}")


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch minute K-line data")
    parser.add_argument("endpoint", choices=["history", "min_adj"],
                        help="Which endpoint")
    parser.add_argument("--level", default="30min", choices=_LEVELS,
                        help="Bar level (default: 30min)")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints")
    parser.add_argument("-w", "--workers", type=int, default=3,
                        help="Chunk workers per stock (default: 3)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-stock checkpoint files")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()

    fn = fetch_history if args.endpoint == "history" else fetch_min_adj
    fn(
        start_date=args.start,
        end_date=args.end,
        output=args.output,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
        level=args.level,
    )
