"""
Fetch minute-level K-line data: /api/stock/history (raw) and
/api/stock/min_adj (adjusted).  Per-stock parallel, main-board only.

Usage:
  python fetch_minute.py history --level 1min --start 2019-01-01 -w 8
  python fetch_minute.py min_adj  --level 5min --start 2020-01-01 -w 4
"""

import requests
import argparse
import pandas as pd
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print


ENDPOINTS = {
    "history":  "stock/history",
    "min_adj":  "stock/min_adj",
}
_LEVELS = ["1min", "5min", "15min", "30min", "60min"]

_EXCLUDE_PREFIXES = ("688", "300", "301", "8", "4")
_ST_PATTERN = r"ST|PT|\*ST"


def _filter_stocks():
    sl_path = Path(DATA_DIR) / "stock_list.parquet"
    if not sl_path.exists():
        log_print("[minute] stock_list.parquet not found")
        return None
    df = pd.read_parquet(sl_path)
    code_num = df["stock_code"].str.extract(r"^(\d+)").iloc[:, 0]
    mask = ~code_num.str.startswith(_EXCLUDE_PREFIXES)
    mask &= ~df["name"].str.contains(_ST_PATTERN, na=False)
    mask &= df["list_status"] == "L"
    return df.loc[mask, "stock_code"].tolist()


# ── Date chunking (monthly) ──────────────────────────────────────────

def _month_chunks(start_str, end_str):
    """Split a wide date range into month-sized (start, end) pairs
    that stay well under the 100K pagination limit.

    start_str/end_str are 'YYYY-MM-DD' (date only); the caller appends
    HH:MM:SS when building the API payload.
    """
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    chunks = []
    y, m = start.year, start.month
    while True:
        ms = datetime(y, m, 1)
        if m == 12:
            me = datetime(y, m, 31)
        else:
            me = datetime(y, m + 1, 1) - timedelta(days=1)
        r_start = max(ms, start)
        r_end = min(me, end)
        if r_start <= r_end:
            chunks.append((
                r_start.strftime("%Y-%m-%d"),
                r_end.strftime("%Y-%m-%d"),
            ))
        if me >= end:
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
    return chunks


# ── API helpers ───────────────────────────────────────────────────────

def _fetch_page(endpoint, payload, api_key, retries=3):
    url = f"{BASE_URL}/{endpoint}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}

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
            return data.get("list", []), data.get("total", 0)

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            if attempt < retries - 1:
                import time
                # 429 rate-limit: use long backoff to let the server recover
                is_429 = (
                    (hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429)
                    or "429" in str(e) or "频繁" in str(e)
                )
                delay = (15 * (2 ** attempt)) if is_429 else (2 ** attempt)
                time.sleep(delay)
                log_print(f"  [retry {attempt+1}/{retries}] {e}")
            else:
                raise


def _fetch_chunk(endpoint, stock_code, level, chunk_start, chunk_end,
                 api_key, extra_payload=None):
    """Fetch all pages for one stock × level × month chunk."""
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
        return []

    all_data = list(page0)
    page = 1
    while len(all_data) < total:
        payload["page"] = page
        b, _ = _fetch_page(endpoint, payload, api_key)
        if not b:
            break
        all_data.extend(b)
        if page * 10000 > 100000:  # safety cap
            log_print(f"  [{endpoint}] {stock_code} {chunk_start} capped at 100K")
            break
        page += 1
    return all_data


# ── Per-stock worker ──────────────────────────────────────────────────

def _fetch_stock(endpoint, stock_code, level, start_date, end_date,
                 api_key, out_dir, extra_payload=None):
    """Fetch all minute data for one stock, save per-stock parquet.

    Each monthly chunk is independently retried (via _fetch_chunk →
    _fetch_page).  If a chunk exhausts its 3 retries the chunk is
    skipped but previous chunks are preserved."""
    chunks = _month_chunks(start_date, end_date)
    all_rows = []
    for cs, ce in chunks:
        try:
            batch = _fetch_chunk(endpoint, stock_code, level, cs, ce,
                                 api_key, extra_payload)
            if batch:
                all_rows.extend(batch)
        except Exception as e:
            log_print(f"  [{endpoint}] {stock_code} chunk {cs}~{ce} FAILED: {e}")

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
    log_print(f"  [{endpoint}] {stock_code} -> 0 rows (empty marker)")
    return 0


# ── Orchestrator ──────────────────────────────────────────────────────

def _run(endpoint, level, start_date, end_date, output, resume, workers,
          cleanup, extra_payload=None, tag=None):
    if tag is None:
        tag = endpoint.replace("stock/", "")

    api_key = load_api_key()
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    stocks = _filter_stocks()
    if stocks is None:
        log_print(f"[{tag}] No stock list, abort")
        return pd.DataFrame()

    out_dir = Path(DATA_DIR) / tag
    out_dir.mkdir(exist_ok=True)

    log_print(f"[{tag}] {len(stocks)} stocks | level={level} | "
              f"{start_date} ~ {end_date} | workers={workers}")

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

    log_print(f"[{tag}] {len(todo)} stocks to fetch")
    completed = 0
    total_rows = 0
    lock = threading.Lock()
    limiter = rate_limiter()

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for code in todo:
                fut = pool.submit(_fetch_stock, endpoint, code, level,
                                  start_date, end_date, api_key, out_dir,
                                  extra_payload)
                futures[fut] = code
            for fut in as_completed(futures):
                code = futures[fut]
                try:
                    n = fut.result()
                    with lock:
                        completed += 1
                        total_rows += n
                    log_print(f"[{tag}] [{completed}/{len(todo)}] {code} done "
                              f"({n} rows) | tokens={limiter.stats['tokens_available']}")
                except Exception as e:
                    with lock:
                        completed += 1
                    log_print(f"[{tag}] [{completed}/{len(todo)}] {code} FAILED: {e}")
    else:
        for i, code in enumerate(todo):
            try:
                n = _fetch_stock(endpoint, code, level, start_date, end_date,
                                 api_key, out_dir, extra_payload)
                completed += 1
                total_rows += n
                log_print(f"[{tag}] [{completed}/{len(todo)}] {code} done "
                          f"({n} rows) | tokens={limiter.stats['tokens_available']}")
            except Exception as e:
                completed += 1
                log_print(f"[{tag}] [{completed}/{len(todo)}] {code} FAILED: {e}")

    log_print(f"[{tag}] Fetched {total_rows} total rows")
    done_file = out_dir / ".done"
    done_file.write_text(str(date.today()))
    log_print(f"[{tag}] Done marker -> {done_file}")
    return pd.DataFrame()


# ── Public entry points ───────────────────────────────────────────────

def fetch_history(start_date="2019-01-01", end_date=None, output=None,
                  resume=True, workers=6, cleanup=True, level="1min"):
    """Fetch raw minute K-line (/api/stock/history)."""
    return _run("stock/history", level, start_date, end_date, output,
                resume, workers, cleanup, tag="history")


def fetch_min_adj(start_date="2019-01-01", end_date=None, output=None,
                  resume=True, workers=6, cleanup=True, level="1min"):
    """Fetch adjusted minute K-line (/api/stock/min_adj)."""
    return _run("stock/min_adj", level, start_date, end_date, output,
                resume, workers, cleanup,
                extra_payload={"algo": "recursive"},
                tag="min_adj")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch minute K-line data")
    parser.add_argument("endpoint", choices=["history", "min_adj"],
                        help="Which endpoint")
    parser.add_argument("--level", default="1min", choices=_LEVELS,
                        help="Bar level (default: 1min)")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints")
    parser.add_argument("-w", "--workers", type=int, default=6,
                        help="Parallel workers (default: 6)")
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
