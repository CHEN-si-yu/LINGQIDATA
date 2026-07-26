import requests
import argparse
import pandas as pd
import threading
from datetime import date, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print


def _month_ranges(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    ranges = []
    y, m = start.year, start.month
    while True:
        month_start = datetime(y, m, 1)
        if m == 12:
            month_end = datetime(y, m, 31)
        else:
            month_end = datetime(y, m + 1, 1) - pd.Timedelta(days=1)
        r_start = max(month_start, start)
        r_end = min(month_end, end)
        if r_start <= r_end:
            ranges.append((r_start.strftime("%Y-%m-%d"), r_end.strftime("%Y-%m-%d")))
        if month_end >= end:
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
    return ranges


def _month_label(start_str, end_str):
    return start_str[:7]


def _fetch_page(endpoint, period, start_time, end_time, page, page_size,
                api_key, extra_payload=None, retries=3):
    url = f"{BASE_URL}/{endpoint}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {
        "period": period,
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

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)
                log_print(f"  [retry {attempt+1}/{retries}] {e}")
            else:
                raise


def _fetch_range(endpoint, period, start_time, end_time, api_key,
                 extra_payload=None):
    """Fetch all rows for a date range, splitting if total exceeds 100K."""
    page0, total = _fetch_page(endpoint, period, start_time, end_time,
                               0, 10000, api_key, extra_payload)
    if not page0:
        return []

    if total <= 100000:
        all_data = list(page0)
        page = 1
        while len(all_data) < total:
            b, _ = _fetch_page(endpoint, period, start_time, end_time,
                               page, 10000, api_key, extra_payload)
            if not b:
                break
            all_data.extend(b)
            page += 1
        return all_data

    start_dt = datetime.strptime(start_time, "%Y-%m-%d")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d")
    mid = start_dt + (end_dt - start_dt) // 2
    mid_str = mid.strftime("%Y-%m-%d")
    next_str = (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    log_print(f"  [{endpoint}/{period}] splitting {start_time}~{end_time} "
              f"(total={total}) -> {start_time}~{mid_str} + {next_str}~{end_time}")

    left = _fetch_range(endpoint, period, start_time, mid_str, api_key, extra_payload)
    right = _fetch_range(endpoint, period, next_str, end_time, api_key, extra_payload)
    return left + right


def _fetch_and_save(endpoint, period, start_time, end_time, label, api_key,
                    checkpoints_dir, extra_payload=None):
    batch = _fetch_range(endpoint, period, start_time, end_time, api_key, extra_payload)
    if batch:
        df_q = pd.DataFrame(batch)
        checkpoint_file = checkpoints_dir / f"{label}.parquet"
        tmp = checkpoint_file.with_suffix(".parquet.tmp")
        df_q.to_parquet(tmp, index=False)
        tmp.replace(checkpoint_file)
    return batch


def _fetch_kline(endpoint, period, start_date, end_date, output, resume,
                 workers, cleanup, extra_payload=None, tag=None):
    """Shared implementation for kline / kline_adj (weekly / monthly)."""
    if tag is None:
        tag = f"{endpoint}/{period}"

    api_key = load_api_key()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    months = _month_ranges(start_date, end_date)
    if not months:
        log_print(f"[{tag}] No months in range")
        return pd.DataFrame()

    log_print(f"[{tag}] {len(months)} months: "
              f"{months[0][0]} ~ {months[-1][1]}")

    checkpoints_dir = Path(DATA_DIR) / f"kline_{period}" if endpoint == "stock/kline" else Path(DATA_DIR) / f"kline_adj_{period}"
    checkpoints_dir.mkdir(exist_ok=True)

    all_data = []
    todo = []
    for (r_start, r_end) in months:
        label = _month_label(r_start, r_end)
        checkpoint_file = checkpoints_dir / f"{label}.parquet"
        if resume and checkpoint_file.exists():
            try:
                df_q = pd.read_parquet(checkpoint_file)
                all_data.extend(df_q.to_dict(orient="records"))
                log_print(f"[{tag}] {label} -> checkpoint ({len(df_q)} rows)")
                continue
            except Exception:
                log_print(f"[{tag}] {label} -> checkpoint corrupt, re-fetching")
                checkpoint_file.unlink(missing_ok=True)
        todo.append((r_start, r_end, label))

    if not todo:
        log_print(f"[{tag}] All {len(months)} months already cached")
    else:
        limiter = rate_limiter()
        log_print(f"[{tag}] {len(todo)} to fetch, workers={workers}, "
                  f"tokens={limiter.stats['tokens_available']}")

        if workers > 1:
            lock = threading.Lock()
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_fetch_and_save, endpoint, period, rs, re, label,
                                api_key, checkpoints_dir, extra_payload): label
                    for (rs, re, label) in todo
                }
                for fut in as_completed(futures):
                    label = futures[fut]
                    try:
                        batch = fut.result()
                        with lock:
                            all_data.extend(batch)
                            completed += 1
                        log_print(f"[{tag}] [{completed}/{len(todo)}] {label} -> "
                                  f"{len(batch)} rows | tokens={limiter.stats['tokens_available']}")
                    except Exception as e:
                        with lock:
                            completed += 1
                        log_print(f"[{tag}] [{completed}/{len(todo)}] {label} FAILED: {e}")
        else:
            for i, (rs, re, label) in enumerate(todo):
                batch = _fetch_and_save(endpoint, period, rs, re, label, api_key,
                                        checkpoints_dir, extra_payload)
                if batch:
                    all_data.extend(batch)
                log_print(f"[{tag}] [{i+1}/{len(todo)}] {label} -> "
                          f"{len(batch)} rows | tokens={limiter.stats['tokens_available']}")

    df = pd.DataFrame(all_data)

    if not df.empty and 'stock_code' in df.columns:
        before = len(df)
        df = df[df['stock_code'].str.match(r'^\d{6}\.(SH|SZ|BJ)$')]
        if len(df) < before:
            log_print(f"[{tag}] Dropped {before - len(df)} rows with malformed stock_code")

        _key = df['stock_code'].str.extract(r'(\d{6})').iloc[:, 0]
        df['_sort_code'] = pd.to_numeric(_key, errors='coerce').fillna(0).astype(int)
        df = df.sort_values(['_sort_code', 'trade_date']).drop(columns=['_sort_code']).reset_index(drop=True)

    log_print(f"[{tag}] Total: {len(df)} rows, {len(df.columns)} columns")

    if output is None:
        suffix = f"kline_{period}" if endpoint == "stock/kline" else f"kline_adj_{period}"
        output = f"{DATA_DIR}/{suffix}.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[{tag}] Saved -> {output}")

    if cleanup:
        removed = 0
        for (r_start, r_end) in months:
            label = _month_label(r_start, r_end)
            cf = checkpoints_dir / f"{label}.parquet"
            if cf.exists():
                cf.unlink()
                removed += 1
        try:
            checkpoints_dir.rmdir()
        except OSError:
            pass
        log_print(f"[{tag}] Cleaned up {removed} checkpoint files")

    return df


def fetch_kline_weekly(start_date="2019-01-01", end_date=None, output=None,
                       resume=True, workers=6, cleanup=True):
    """Fetch weekly K-line data for all stocks."""
    return _fetch_kline("stock/kline", "weekly", start_date, end_date, output,
                        resume, workers, cleanup, tag="kline/weekly")


def fetch_kline_monthly(start_date="2019-01-01", end_date=None, output=None,
                        resume=True, workers=6, cleanup=True):
    """Fetch monthly K-line data for all stocks."""
    return _fetch_kline("stock/kline", "monthly", start_date, end_date, output,
                        resume, workers, cleanup, tag="kline/monthly")


def fetch_kline_adj_weekly(start_date="2019-01-01", end_date=None, output=None,
                           resume=True, workers=6, cleanup=True):
    """Fetch adjusted (前复权) weekly K-line data for all stocks."""
    return _fetch_kline("stock/kline_adj", "weekly", start_date, end_date, output,
                        resume, workers, cleanup,
                        extra_payload={"algo": "recursive", "volType": "share"},
                        tag="kline_adj/weekly")


def fetch_kline_adj_monthly(start_date="2019-01-01", end_date=None, output=None,
                            resume=True, workers=6, cleanup=True):
    """Fetch adjusted (前复权) monthly K-line data for all stocks."""
    return _fetch_kline("stock/kline_adj", "monthly", start_date, end_date, output,
                        resume, workers, cleanup,
                        extra_payload={"algo": "recursive", "volType": "share"},
                        tag="kline_adj/monthly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock weekly/monthly K-line (raw or adj)")
    parser.add_argument("period", choices=["weekly", "monthly"],
                        help="K-line period")
    parser.add_argument("--adj", action="store_true",
                        help="Fetch adjusted (前复权) K-line instead of raw")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints, re-fetch all")
    parser.add_argument("-w", "--workers", type=int, default=6,
                        help="Parallel workers (1=serial, default: 6)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-month checkpoint files")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()

    if args.adj:
        fn = fetch_kline_adj_weekly if args.period == "weekly" else fetch_kline_adj_monthly
    else:
        fn = fetch_kline_weekly if args.period == "weekly" else fetch_kline_monthly
    fn(
        start_date=args.start,
        end_date=args.end,
        output=args.output,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
    )
