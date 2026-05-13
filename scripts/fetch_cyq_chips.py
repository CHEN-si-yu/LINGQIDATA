import requests
import argparse
import pandas as pd
import threading
from datetime import date, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print


ENDPOINT = "stock/cyq_chips"

# Stock codes to exclude: 科创板 688xxx, 创业板 300/301xxx, 北交所 8xxxxx
_EXCLUDE_PREFIXES = ("688", "300", "301", "8", "4")
_ST_PATTERN = r"ST|PT|\*ST"


def _filter_stocks():
    """Return list of main-board non-ST stock codes."""
    sl_path = Path(DATA_DIR) / "stock_list.parquet"
    if not sl_path.exists():
        log_print("[cyq_chips] WARNING: stock_list.parquet not found, no filtering")
        return None
    df = pd.read_parquet(sl_path)
    # Exclude by code prefix
    code_num = df["stock_code"].str.extract(r"^(\d+)").iloc[:, 0]
    mask = ~code_num.str.startswith(_EXCLUDE_PREFIXES)
    # Exclude ST/PT
    mask &= ~df["name"].str.contains(_ST_PATTERN, na=False)
    # Only listed
    mask &= df["list_status"] == "L"
    codes = df.loc[mask, "stock_code"].tolist()
    log_print(f"[cyq_chips] {len(codes)} stocks after filter (excl 科创/创业/北交/ST)")
    return codes


def _fetch_page(stock_code, start_time, end_time, page, page_size, api_key, retries=3):
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {
        "stock_code": [stock_code],
        "start_time": start_time,
        "end_time": end_time,
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


def _fetch_range(stock_code, start_time, end_time, api_key):
    """Fetch all rows for one stock+range, splitting date range if total > 100K."""
    page0, total = _fetch_page(stock_code, start_time, end_time, 0, 10000, api_key)
    if not page0:
        return []

    if total <= 100000:
        all_data = list(page0)
        page = 1
        while len(all_data) < total:
            b, _ = _fetch_page(stock_code, start_time, end_time, page, 10000, api_key)
            if not b:
                break
            all_data.extend(b)
            page += 1
        return all_data

    if start_time == end_time:
        log_print(f"  [cyq_chips/{stock_code}] {start_time} total={total} > 100K, capped")
        all_data = list(page0)
        page = 1
        while page * 10000 <= 100000:
            b, _ = _fetch_page(stock_code, start_time, end_time, page, 10000, api_key)
            if not b:
                break
            all_data.extend(b)
            page += 1
        return all_data

    start_dt = datetime.strptime(start_time, "%Y-%m-%d")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d")
    mid = start_dt + (end_dt - start_dt) // 2
    if mid == start_dt:
        all_data = list(page0)
        page = 1
        while page * 10000 <= 100000:
            b, _ = _fetch_page(stock_code, start_time, end_time, page, 10000, api_key)
            if not b:
                break
            all_data.extend(b)
            page += 1
        return all_data

    mid_str = mid.strftime("%Y-%m-%d")
    next_str = (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    left = _fetch_range(stock_code, start_time, mid_str, api_key)
    right = _fetch_range(stock_code, next_str, end_time, api_key)
    return left + right


def _fetch_stock(stock_code, start_date, end_date, api_key, out_dir):
    """Fetch all cyq_chips for one stock, save to per-stock parquet.

    _fetch_page retries up to 3 times per page before giving up.
    If the entire range fails after retries, the stock is skipped."""
    log_print(f"  [cyq_chips] {stock_code} starting")
    try:
        rows = _fetch_range(stock_code, start_date, end_date, api_key)
    except Exception as e:
        log_print(f"  [cyq_chips] {stock_code} FAILED after retries: {e}")
        return 0
    if rows:
        df = pd.DataFrame(rows)
        df.sort_values("trade_date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        out_file = out_dir / f"{stock_code}.parquet"
        tmp = out_file.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out_file)
        log_print(f"  [cyq_chips] {stock_code} -> {len(df)} rows")
        return len(df)
    # Save empty marker so resume logic skips this stock on re-runs.
    # The API genuinely has no chips data for some stocks; re-fetching
    # them every run wastes calls and tokens.
    pd.DataFrame(columns=["trade_date", "stock_code", "price", "percent"]).to_parquet(
        out_dir / f"{stock_code}.parquet", index=False
    )
    log_print(f"  [cyq_chips] {stock_code} -> 0 rows (empty marker)")
    return 0


def fetch_cyq_chips(start_date="2019-01-01", end_date=None, output=None,
                    resume=True, workers=6, cleanup=True):
    """Fetch CYQ chips distribution — per-stock parallel, main board only."""
    api_key = load_api_key()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    stocks = _filter_stocks()
    if stocks is None:
        log_print("[cyq_chips] No stock list available, abort")
        return pd.DataFrame()

    log_print(f"[cyq_chips] {len(stocks)} stocks | {start_date} ~ {end_date} "
              f"| workers={workers}")

    out_dir = Path(DATA_DIR) / "cyq_chips"
    out_dir.mkdir(exist_ok=True)

    # Determine which stocks need fetching
    todo = []
    for code in stocks:
        f = out_dir / f"{code}.parquet"
        if resume and f.exists():
            try:
                pd.read_parquet(f)
                continue  # already done
            except Exception:
                f.unlink(missing_ok=True)
        todo.append(code)

    if not todo:
        log_print("[cyq_chips] All stocks already cached")
        return pd.DataFrame()

    log_print(f"[cyq_chips] {len(todo)} stocks to fetch")

    completed = 0
    total_rows = 0
    lock = threading.Lock()
    limiter = rate_limiter()

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_stock, code, start_date, end_date, api_key, out_dir): code
                for code in todo
            }
            for fut in as_completed(futures):
                code = futures[fut]
                try:
                    n = fut.result()
                    with lock:
                        completed += 1
                        total_rows += n
                    log_print(f"[cyq_chips] [{completed}/{len(todo)}] {code} done "
                              f"({n} rows) | tokens={limiter.stats['tokens_available']}")
                except Exception as e:
                    with lock:
                        completed += 1
                    log_print(f"[cyq_chips] [{completed}/{len(todo)}] {code} FAILED: {e}")
    else:
        for i, code in enumerate(todo):
            try:
                n = _fetch_stock(code, start_date, end_date, api_key, out_dir)
                completed += 1
                total_rows += n
                log_print(f"[cyq_chips] [{completed}/{len(todo)}] {code} done "
                          f"({n} rows) | tokens={limiter.stats['tokens_available']}")
            except Exception as e:
                completed += 1
                log_print(f"[cyq_chips] [{completed}/{len(todo)}] {code} FAILED: {e}")

    log_print(f"[cyq_chips] Fetched {total_rows} total rows from {len(todo)} stocks")
    # Write completion marker so main.py knows this task is done.
    # Data lives as per-stock parquets under out_dir/.
    done_file = out_dir / ".done"
    done_file.write_text(str(date.today()))
    log_print(f"[cyq_chips] Done marker -> {done_file}")
    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch CYQ chips per stock")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints, re-fetch all")
    parser.add_argument("-w", "--workers", type=int, default=6,
                        help="Parallel workers (1=serial, default: 6)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-stock checkpoint files")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()

    fetch_cyq_chips(
        start_date=args.start,
        end_date=args.end,
        output=args.output,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
    )
