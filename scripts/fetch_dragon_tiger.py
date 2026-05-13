import requests
import argparse
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print


ENDPOINT = "stock/dragon_tiger"
PAGE_SIZE = 1000  # max for this endpoint


def _trading_days(start_str, end_str):
    cal_path = Path(DATA_DIR) / "calendar.parquet"
    if not cal_path.exists():
        log_print("[dragon_tiger] WARNING: calendar.parquet not found, using all dates")
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        dates = []
        d = start
        while d <= end:
            dates.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        return dates
    cal = pd.read_parquet(cal_path)
    cal = cal[(cal["is_open"] == 1) & (cal["date"] >= start_str) & (cal["date"] <= end_str)]
    return sorted(cal["date"].tolist())


def _month_label(date_str):
    return date_str[:7]


def _fetch_date(date_str, api_key):
    """Fetch all pages for a single date. Page starts at 1, page_size max 1000."""
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    all_rows = []
    page = 0

    while True:
        payload = {"date": date_str, "page": page, "page_size": PAGE_SIZE}

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
            if isinstance(data, list):
                if not data:
                    break
                all_rows.extend(data)
                # Keep paginating while the page is full (may have more)
                if len(data) < PAGE_SIZE:
                    break
                page += 1
            elif isinstance(data, dict):
                batch = data.get("list", [])
                total = data.get("total", 0)
                if not batch:
                    break
                all_rows.extend(batch)
                if len(all_rows) >= total:
                    break
                page += 1
            else:
                break

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            log_print(f"  [dragon_tiger] {date_str} page={page} FAILED: {e}")
            break

    return all_rows


def _fetch_month(label, dates, api_key, checkpoints_dir):
    all_rows = []

    for i, d in enumerate(dates):
        batch = _fetch_date(d, api_key)
        if batch:
            all_rows.extend(batch)
            log_print(f"  [dragon_tiger/{label}] [{i+1}/{len(dates)}] {d} -> {len(batch)} rows")
        else:
            log_print(f"  [dragon_tiger/{label}] [{i+1}/{len(dates)}] {d} -> 0 rows")

    if all_rows:
        df_m = pd.DataFrame(all_rows)
        checkpoint_file = checkpoints_dir / f"{label}.parquet"
        tmp = checkpoint_file.with_suffix(".parquet.tmp")
        df_m.to_parquet(tmp, index=False)
        tmp.replace(checkpoint_file)

    return all_rows


def fetch_dragon_tiger(start_date="2019-01-01", end_date=None, output=None,
                       resume=True, workers=6, cleanup=True):
    """Fetch dragon-tiger board institution details."""
    api_key = load_api_key()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    dates = _trading_days(start_date, end_date)
    if not dates:
        log_print("[dragon_tiger] No trading days in range")
        return pd.DataFrame()

    log_print(f"[dragon_tiger] {len(dates)} trading days: {dates[0]} ~ {dates[-1]}")

    month_dates = defaultdict(list)
    for d in dates:
        month_dates[_month_label(d)].append(d)

    log_print(f"[dragon_tiger] {len(month_dates)} months")

    checkpoints_dir = Path(DATA_DIR) / "dragon_tiger"
    checkpoints_dir.mkdir(exist_ok=True)

    all_data = []
    todo = []
    for label in sorted(month_dates.keys()):
        checkpoint_file = checkpoints_dir / f"{label}.parquet"
        if resume and checkpoint_file.exists():
            try:
                df_m = pd.read_parquet(checkpoint_file)
                all_data.extend(df_m.to_dict(orient="records"))
                log_print(f"[dragon_tiger] {label} -> checkpoint ({len(df_m)} rows)")
                continue
            except Exception:
                log_print(f"[dragon_tiger] {label} -> checkpoint corrupt, re-fetching")
                checkpoint_file.unlink(missing_ok=True)
        todo.append(label)

    if not todo:
        log_print("[dragon_tiger] All months already cached")
    else:
        log_print(f"[dragon_tiger] {len(todo)} months to fetch")

        for i, label in enumerate(todo):
            m_dates = month_dates[label]
            log_print(f"[dragon_tiger] [{i+1}/{len(todo)}] {label}: {len(m_dates)} days")
            batch = _fetch_month(label, m_dates, api_key, checkpoints_dir)
            all_data.extend(batch)
            log_print(f"[dragon_tiger] [{i+1}/{len(todo)}] {label} -> {len(batch)} rows")

    df = pd.DataFrame(all_data)

    if not df.empty and 'stock_code' in df.columns:
        before = len(df)
        df = df[df['stock_code'].str.match(r'^\d{6}\.(SH|SZ|BJ)$')]
        if len(df) < before:
            log_print(f"[dragon_tiger] Dropped {before - len(df)} rows with malformed stock_code")

        _key = df['stock_code'].str.extract(r'(\d{6})').iloc[:, 0]
        df['_sort_code'] = pd.to_numeric(_key, errors='coerce').fillna(0).astype(int)
        df = df.sort_values(['_sort_code', 'trade_date']).drop(columns=['_sort_code']).reset_index(drop=True)

    log_print(f"[dragon_tiger] Total: {len(df)} rows, {len(df.columns)} columns")

    if output is None:
        output = f"{DATA_DIR}/dragon_tiger.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[dragon_tiger] Saved -> {output}")

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
        log_print(f"[dragon_tiger] Cleaned up {removed} checkpoint files")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch dragon-tiger board details")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints, re-fetch all")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="(unused, kept for compatibility)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-month checkpoint files")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()

    fetch_dragon_tiger(
        start_date=args.start,
        end_date=args.end,
        output=args.output,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
    )
