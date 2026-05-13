import requests
import argparse
import pandas as pd
import threading
from datetime import date, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print


ENDPOINT = "stock/financial_indicator"


def _quarter_ends(start_str, end_str):
    """All quarter-end dates (03-31, 06-30, 09-30, 12-31) in [start, end]."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    dates = []
    for y in range(start.year, end.year + 1):
        for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            dt = datetime(y, m, d)
            if start <= dt <= end:
                dates.append(dt.strftime("%Y-%m-%d"))
    return dates


def _fetch_page(end_date, page, page_size, api_key, retries=3):
    """Fetch a single page. Returns (list_of_rows, total)."""
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {"end_date": end_date, "page": page, "page_size": page_size}

    for attempt in range(retries):
        try:
            limiter = rate_limiter()
            limiter.acquire(ENDPOINT)
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()

            if result["code"] != 200:
                msg = result.get("msg", str(result))
                raise RuntimeError(f"API Error: code={result['code']}, msg={msg}")

            data = result["data"]
            return data["list"], data["total"]

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            if attempt < retries - 1:
                wait_s = 2 ** attempt
                log_print(f"  [retry {attempt+1}/{retries}] {e}")
                import time
                time.sleep(wait_s)
            else:
                raise


def _fetch_quarter(end_date, api_key):
    """Fetch all rows for one quarter-end date. Returns list of dicts."""
    all_data = []
    page = 0
    page_size = 10000

    while True:
        batch, total = _fetch_page(end_date, page, page_size, api_key)
        if not batch:
            break
        all_data.extend(batch)
        if len(all_data) >= total:
            break
        page += 1

    return all_data


def _fetch_and_save(end_date, api_key, checkpoints_dir):
    """Fetch one quarter and atomically save checkpoint. Returns list of dicts."""
    batch = _fetch_quarter(end_date, api_key)
    if batch:
        df_q = pd.DataFrame(batch)
        checkpoint_file = checkpoints_dir / f"{end_date}.parquet"
        tmp = checkpoint_file.with_suffix(".parquet.tmp")
        df_q.to_parquet(tmp, index=False)
        tmp.replace(checkpoint_file)
    return batch


def fetch_financial_indicator(start_date="2019-01-01", end_date=None,
                               output=None, resume=True, workers=6,
                               cleanup=True):
    """Fetch financial indicators for all quarters from *start_date* to *end_date*.

    Parameters
    ----------
    workers:
        Number of parallel workers for quarter fetching (1 = serial).
        Default 6 stays in the 4-8 range.  All workers share the global
        rate limiter so total calls stay under 280/min.
    cleanup:
        If True, remove per-quarter checkpoint files after the consolidated
        output is successfully written.
    """
    api_key = load_api_key()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    quarters = _quarter_ends(start_date, end_date)
    if not quarters:
        log_print("[financial] No quarters in range — nothing to fetch")
        return pd.DataFrame()

    log_print(f"[financial] {len(quarters)} quarters: {quarters[0]} ~ {quarters[-1]}")

    checkpoints_dir = Path(DATA_DIR) / "financial"
    checkpoints_dir.mkdir(exist_ok=True)

    # ── Phase 1: load cached checkpoints (serial, fast) ──
    all_data = []
    todo = []
    for q_end in quarters:
        checkpoint_file = checkpoints_dir / f"{q_end}.parquet"
        if resume and checkpoint_file.exists():
            try:
                df_q = pd.read_parquet(checkpoint_file)
                all_data.extend(df_q.to_dict(orient="records"))
                log_print(f"[financial] {q_end} → checkpoint ({len(df_q)} rows)")
                continue
            except Exception:
                log_print(f"[financial] {q_end} → checkpoint corrupt, re-fetching")
                checkpoint_file.unlink(missing_ok=True)
        todo.append(q_end)

    if not todo:
        log_print(f"[financial] All {len(quarters)} quarters already cached")
    else:
        limiter = rate_limiter()
        log_print(f"[financial] {len(todo)} to fetch, workers={workers}, "
                  f"tokens={limiter.stats['tokens_available']}")

        if workers > 1:
            lock = threading.Lock()
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_fetch_and_save, q, api_key, checkpoints_dir): q
                    for q in todo
                }
                for fut in as_completed(futures):
                    q = futures[fut]
                    try:
                        batch = fut.result()
                        with lock:
                            all_data.extend(batch)
                            completed += 1
                        log_print(f"[financial] [{completed}/{len(todo)}] {q} → "
                                  f"{len(batch)} rows | tokens={limiter.stats['tokens_available']}")
                    except Exception as e:
                        with lock:
                            completed += 1
                        log_print(f"[financial] [{completed}/{len(todo)}] {q} FAILED: {e}")
        else:
            for i, q_end in enumerate(todo):
                batch = _fetch_and_save(q_end, api_key, checkpoints_dir)
                if batch:
                    all_data.extend(batch)
                log_print(f"[financial] [{i+1}/{len(todo)}] {q_end} → "
                          f"{len(batch)} rows | tokens={limiter.stats['tokens_available']}")

    df = pd.DataFrame(all_data)

    if not df.empty and 'stock_code' in df.columns:
        before = len(df)
        df = df[df['stock_code'].str.match(r'^\d{6}\.(SH|SZ|BJ)$')]
        if len(df) < before:
            log_print(f"[financial] Dropped {before - len(df)} rows with malformed stock_code")

        _key = df['stock_code'].str.extract(r'(\d{6})').iloc[:, 0]
        df['_sort_code'] = pd.to_numeric(_key, errors='coerce').fillna(0).astype(int)
        df = df.sort_values(['_sort_code', 'ann_date']).drop(columns=['_sort_code']).reset_index(drop=True)

    log_print(f"[financial] Total: {len(df)} rows, {len(df.columns)} columns")

    if output is None:
        output = f"{DATA_DIR}/financial_indicator.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[financial] Saved → {output}")

    if cleanup:
        removed = 0
        for q_end in quarters:
            cf = checkpoints_dir / f"{q_end}.parquet"
            if cf.exists():
                cf.unlink()
                removed += 1
        # Remove directory if empty
        try:
            checkpoints_dir.rmdir()
        except OSError:
            pass  # not empty, leave it
        log_print(f"[financial] Cleaned up {removed} checkpoint files")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock financial indicator data")
    parser.add_argument("--start", default="2019-01-01", help="Start date for end_date filter")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints, re-fetch all")
    parser.add_argument("-w", "--workers", type=int, default=6,
                        help="Parallel workers for quarter fetching (1=serial, default: 6)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-quarter checkpoint files after consolidation")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()

    fetch_financial_indicator(
        start_date=args.start,
        end_date=args.end,
        output=args.output,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
    )
