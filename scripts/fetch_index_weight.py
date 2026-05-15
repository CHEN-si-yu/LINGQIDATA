import requests
import argparse
import pandas as pd
import threading
from datetime import date, datetime
from datetime import timedelta as td
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print

ENDPOINT = "index/weight"

# This API is designed for **market indices** (000300.SH, 399006.SZ, etc.),
# NOT THS sector indices (882001.TI).  Iterating 1200+ sector codes produces
# 100K+ empty/429 responses.  Keep the default list small and practical.
_DEFAULT_INDICES = [
    "000001.SH",  # 上证指数
    "000016.SH",  # 上证50
    "000300.SH",  # 沪深300
    "000688.SH",  # 科创50
    "000852.SH",  # 中证1000
    "000905.SH",  # 中证500
    "399001.SZ",  # 深证成指
    "399006.SZ",  # 创业板指
]

# When full_history=False, cap the look-back to this many months.
# Avoids accidentally scheduling thousands of calls for data that
# most strategies don't need beyond a year or two.
_DEFAULT_MONTHS = 24


def _month_ranges(start_str, end_str):
    """Generate (month_start, month_end) pairs for every month in [start, end]."""
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


def _fetch_page(index_code, trade_date, page, page_size, api_key, retries=3):
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {
        "index_code": index_code,
        "trade_date": trade_date,
        "page": page,
        "page_size": page_size,
    }

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
            is_429 = "429" in str(e) or "频繁" in str(e)
            if attempt < retries - 1:
                import time
                delay = (15 * (2 ** attempt)) if is_429 else (2 ** attempt)
                time.sleep(delay)
                log_print(f"  [retry {attempt+1}/{retries}] {e}")
            else:
                raise


def _fetch_for_index_month(index_code, trade_date, api_key):
    """Fetch all pages for one index×month combination."""
    all_data = []
    page = 0
    page_size = 10000

    while True:
        batch, total = _fetch_page(index_code, trade_date, page, page_size, api_key)
        if not batch:
            break
        all_data.extend(batch)
        if len(all_data) >= total:
            break
        page += 1

    return all_data


def _fetch_and_save(index_code, trade_date, api_key, checkpoints_dir):
    """Fetch one index×month and atomically save checkpoint."""
    batch = _fetch_for_index_month(index_code, trade_date, api_key)
    if batch:
        df_q = pd.DataFrame(batch)
        safe_code = index_code.replace(".", "_")
        checkpoint_file = checkpoints_dir / f"{safe_code}_{trade_date[:7]}.parquet"
        tmp = checkpoint_file.with_suffix(".parquet.tmp")
        df_q.to_parquet(tmp, index=False)
        tmp.replace(checkpoint_file)
    return batch


def _resolve_index_codes(codes_arg):
    """Resolve index codes: explicit list > defaults.

    Unlike the previous version, this does NOT derive codes from
    ths_sector_categories — those are THS sector codes (882001.TI)
    which this market-index API does not serve.
    """
    if codes_arg:
        return codes_arg
    log_print(f"[index_weight] Using {len(_DEFAULT_INDICES)} default major indices")
    return list(_DEFAULT_INDICES)


def fetch_index_weight(start_date="2019-01-01", end_date=None, output=None,
                       index_codes=None, resume=True, workers=4, cleanup=True,
                       full_history=False):
    """Fetch index constituent weights (monthly) for the given index codes.

    Parameters
    ----------
    index_codes : list or None
        Market index codes.  None defaults to 8 major A-share indices.
    full_history : bool
        When False (default), *start_date* is capped to *_DEFAULT_MONTHS*
        ago.  Pass True to fetch the entire requested range.
    """
    api_key = load_api_key()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    # Cap look-back unless explicitly asked for full history
    effective_start = start_date
    if not full_history:
        cap_dt = datetime.now() - td(days=_DEFAULT_MONTHS * 31)
        cap_str = cap_dt.strftime("%Y-%m-%d")
        if start_date < cap_str:
            log_print(f"[index_weight] Capping start_date: {start_date} -> {cap_str} "
                      f"({_DEFAULT_MONTHS}-month window).  Use --full-history to override.")
            effective_start = cap_str

    codes = _resolve_index_codes(index_codes)
    if not codes:
        log_print("[index_weight] No index codes to process")
        return pd.DataFrame()

    months = _month_ranges(effective_start, end_date)
    if not months:
        log_print("[index_weight] No months in range")
        return pd.DataFrame()

    total_combos = len(codes) * len(months)
    est_minutes = total_combos / 280.0
    log_print(f"[index_weight] {len(codes)} indices × {len(months)} months "
              f"= {total_combos} API calls | ~{est_minutes:.0f} min | "
              f"{months[0][0]} ~ {months[-1][1]} | workers={workers}")

    checkpoints_dir = Path(DATA_DIR) / "index_weight_chk"
    checkpoints_dir.mkdir(exist_ok=True)

    all_data = []
    todo = []
    for code in codes:
        for (r_start, _r_end) in months:
            label = r_start[:7]
            safe_code = code.replace(".", "_")
            checkpoint_file = checkpoints_dir / f"{safe_code}_{label}.parquet"
            if resume and checkpoint_file.exists():
                try:
                    df_q = pd.read_parquet(checkpoint_file)
                    all_data.extend(df_q.to_dict(orient="records"))
                    continue
                except Exception:
                    checkpoint_file.unlink(missing_ok=True)
            todo.append((code, r_start))

    if not todo:
        log_print(f"[index_weight] All checkpoints cached")
    else:
        limiter = rate_limiter()
        total_items = len(todo)
        log_print(f"[index_weight] {total_items} to fetch | "
                  f"tokens={limiter.stats['tokens_available']}")

        if workers > 1:
            lock = threading.Lock()
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_fetch_and_save, code, ts, api_key, checkpoints_dir): (code, ts)
                    for (code, ts) in todo
                }
                for fut in as_completed(futures):
                    code, ts = futures[fut]
                    try:
                        batch = fut.result()
                        with lock:
                            all_data.extend(batch)
                            completed += 1
                        if completed % 20 == 0 or completed == total_items:
                            s = limiter.stats
                            log_print(f"[index_weight] [{completed}/{total_items}] "
                                      f"tokens={s['tokens_available']:.0f}/{s['max_rpm']}")
                    except Exception as e:
                        with lock:
                            completed += 1
                        log_print(f"[index_weight] [{completed}/{total_items}] "
                                  f"{code}/{ts} FAILED: {e}")
        else:
            for i, (code, ts) in enumerate(todo):
                batch = _fetch_and_save(code, ts, api_key, checkpoints_dir)
                if batch:
                    all_data.extend(batch)
                if (i + 1) % 20 == 0 or (i + 1) == total_items:
                    s = rate_limiter().stats
                    log_print(f"[index_weight] [{i+1}/{total_items}] "
                              f"tokens={s['tokens_available']:.0f}/{s['max_rpm']}")

    df = pd.DataFrame(all_data)

    if not df.empty:
        if 'index_code' in df.columns and 'trade_date' in df.columns:
            df = df.sort_values(['index_code', 'trade_date']).reset_index(drop=True)

    n_idx = df['index_code'].nunique() if 'index_code' in df.columns else 0
    log_print(f"[index_weight] Total: {len(df)} rows | {n_idx} indices")

    if output is None:
        output = f"{DATA_DIR}/index_weight.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[index_weight] Saved -> {output}")

    if cleanup:
        removed = 0
        for code in codes:
            for (r_start, _r_end) in months:
                label = r_start[:7]
                safe_code = code.replace(".", "_")
                cf = checkpoints_dir / f"{safe_code}_{label}.parquet"
                if cf.exists():
                    cf.unlink()
                    removed += 1
        try:
            checkpoints_dir.rmdir()
        except OSError:
            pass
        log_print(f"[index_weight] Cleaned up {removed} checkpoint files")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch index constituent weights (monthly)"
    )
    parser.add_argument("--start", default="2019-01-01",
                        help="Start date (default: 2019-01-01, capped to 24-month window unless --full-history)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--index-codes", nargs="*",
                        help="Market index codes (e.g. 000300.SH). Default: 8 major indices")
    parser.add_argument("--full-history", action="store_true",
                        help="Fetch the full date range (no 24-month cap)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints, re-fetch all")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-index×month checkpoint files")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()

    fetch_index_weight(
        start_date=args.start,
        end_date=args.end,
        output=args.output,
        index_codes=args.index_codes,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
        full_history=args.full_history,
    )
