import requests
import argparse
import pandas as pd
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print


ENDPOINT = "stock/daily_dump"


def _date_range(start_str, end_str):
    """Generate every calendar date in [start, end] as 'YYYY-MM-DD'."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def _month_label(date_str):
    return date_str[:7]


def _fetch_one_date(date_str, api_key, retries=2):
    """Fetch daily dump for a single date. Returns list of dicts (may be empty)."""
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {"date": date_str, "level": "daily"}

    for attempt in range(retries):
        try:
            limiter = rate_limiter()
            limiter.acquire(ENDPOINT)
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()

            if result["code"] != 200:
                msg = result.get("msg", str(result))
                raise RuntimeError(f"API Error: code={result['code']}, msg={msg}")

            data = result["data"]
            if isinstance(data, list):
                return data
            # Some responses wrap in an object with 'list'
            if isinstance(data, dict) and "list" in data:
                return data["list"]
            return []

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)
                log_print(f"  [retry {attempt+1}/{retries}] {e}")
            else:
                raise


def _fetch_month_dates(month_label, dates, api_key, checkpoints_dir, workers):
    """Fetch all dates in one month, save checkpoint. Returns list of dicts."""
    limiter = rate_limiter()
    all_rows = []

    if workers > 1:
        lock = threading.Lock()
        completed = 0
        with ThreadPoolExecutor(max_workers=min(workers, len(dates))) as pool:
            futures = {pool.submit(_fetch_one_date, d, api_key): d for d in dates}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    batch = fut.result()
                    with lock:
                        if batch:
                            all_rows.extend(batch)
                        completed += 1
                    if batch:
                        log_print(f"  [daily_dump/{month_label}] {d} -> {len(batch)} rows "
                                  f"| tokens={limiter.stats['tokens_available']}")
                except Exception as e:
                    with lock:
                        completed += 1
                    log_print(f"  [daily_dump/{month_label}] {d} FAILED: {e}")
    else:
        for i, d in enumerate(dates):
            try:
                batch = _fetch_one_date(d, api_key)
                if batch:
                    all_rows.extend(batch)
                log_print(f"  [daily_dump/{month_label}] [{i+1}/{len(dates)}] {d} -> "
                          f"{len(batch)} rows | tokens={limiter.stats['tokens_available']}")
            except Exception as e:
                log_print(f"  [daily_dump/{month_label}] [{i+1}/{len(dates)}] {d} FAILED: {e}")

    if all_rows:
        df_m = pd.DataFrame(all_rows)
        checkpoint_file = checkpoints_dir / f"{month_label}.parquet"
        tmp = checkpoint_file.with_suffix(".parquet.tmp")
        df_m.to_parquet(tmp, index=False)
        tmp.replace(checkpoint_file)

    return all_rows


def fetch_daily_dump(start_date="2019-01-01", end_date=None, output=None,
                     resume=True, workers=6, cleanup=True):
    """Fetch full-market daily OHLCV via the daily_dump endpoint.

    Unlike the paginated /api/stock/daily, this endpoint returns all
    stocks for a single date in one call.  We iterate date-by-date
    and aggregate by month for checkpointing.

    Parameters
    ----------
    workers: Parallel workers for date fetching within a month.
    """
    api_key = load_api_key()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    dates = _date_range(start_date, end_date)
    log_print(f"[daily_dump] {len(dates)} calendar days: {dates[0]} ~ {dates[-1]}")

    # Group dates by month
    month_dates = defaultdict(list)
    for d in dates:
        month_dates[_month_label(d)].append(d)

    log_print(f"[daily_dump] {len(month_dates)} months")

    checkpoints_dir = Path(DATA_DIR) / "daily_dump"
    checkpoints_dir.mkdir(exist_ok=True)

    all_data = []
    todo = []
    for label in sorted(month_dates.keys()):
        checkpoint_file = checkpoints_dir / f"{label}.parquet"
        if resume and checkpoint_file.exists():
            try:
                df_m = pd.read_parquet(checkpoint_file)
                all_data.extend(df_m.to_dict(orient="records"))
                log_print(f"[daily_dump] {label} -> checkpoint ({len(df_m)} rows)")
                continue
            except Exception:
                log_print(f"[daily_dump] {label} -> checkpoint corrupt, re-fetching")
                checkpoint_file.unlink(missing_ok=True)
        todo.append(label)

    if not todo:
        log_print("[daily_dump] All months already cached")
    else:
        limiter = rate_limiter()
        log_print(f"[daily_dump] {len(todo)} months to fetch, workers={workers}, "
                  f"tokens={limiter.stats['tokens_available']}")

        for i, label in enumerate(todo):
            m_dates = month_dates[label]
            log_print(f"[daily_dump] [{i+1}/{len(todo)}] {label}: {len(m_dates)} days")
            batch = _fetch_month_dates(label, m_dates, api_key, checkpoints_dir, workers)
            if batch:
                all_data.extend(batch)
            log_print(f"[daily_dump] [{i+1}/{len(todo)}] {label} -> "
                      f"{len(batch)} rows | tokens={limiter.stats['tokens_available']}")

    df = pd.DataFrame(all_data)

    if not df.empty and 'stock_code' in df.columns:
        before = len(df)
        df = df[df['stock_code'].str.match(r'^\d{6}\.(SH|SZ|BJ)$')]
        if len(df) < before:
            log_print(f"[daily_dump] Dropped {before - len(df)} rows with malformed stock_code")

        _key = df['stock_code'].str.extract(r'(\d{6})').iloc[:, 0]
        df['_sort_code'] = pd.to_numeric(_key, errors='coerce').fillna(0).astype(int)
        df = df.sort_values(['_sort_code', 'trade_date']).drop(columns=['_sort_code']).reset_index(drop=True)

    log_print(f"[daily_dump] Total: {len(df)} rows, {len(df.columns)} columns")

    if output is None:
        output = f"{DATA_DIR}/daily_dump.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[daily_dump] Saved -> {output}")

    if cleanup:
        removed = 0
        for label in month_dates:
            cf = checkpoints_dir / f"{label}.parquet"
            if cf.exists():
                cf.unlink()
                removed += 1
        try:
            checkpoints_dir.rmdir()
        except OSError:
            pass
        log_print(f"[daily_dump] Cleaned up {removed} checkpoint files")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch full-market daily dump (OHLCV)")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints, re-fetch all")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Parallel workers per month (1=serial, default: 4)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-month checkpoint files")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()

    fetch_daily_dump(
        start_date=args.start,
        end_date=args.end,
        output=args.output,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
    )
