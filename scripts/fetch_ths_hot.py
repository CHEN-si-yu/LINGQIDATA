import requests
import argparse
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict
from config import load_api_key, BASE_URL, DATA_DIR, RateLimiter, log_print


ENDPOINT = "ths/hot"
MARKETS = ["热股", "ETF", "可转债", "行业板块", "概念板块", "期货"]

# Per-endpoint rate limiter — this endpoint tolerates at most 200 req/min
_limiter = RateLimiter(max_rpm=200)

# Retry delays in seconds: 1st retry 1s, 2nd 5s, 3rd 10s
_RETRY_DELAYS = [1, 5, 10]


def _trading_days(start_str, end_str):
    """Return only trading days (is_open==1) from the calendar parquet."""
    cal_path = Path(DATA_DIR) / "calendar.parquet"
    if not cal_path.exists():
        log_print("[ths_hot] WARNING: calendar.parquet not found, falling back to all dates")
        from datetime import timedelta
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


def _fetch_one(date_str, market, api_key):
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {"trade_date": date_str, "market": market}
    retries = len(_RETRY_DELAYS) + 1

    for attempt in range(retries):
        try:
            _limiter.acquire(ENDPOINT)
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()

            if result["code"] != 200:
                msg = result.get("msg", str(result))
                raise RuntimeError(f"API Error: code={result['code']}, msg={msg}")

            data = result["data"]
            rows = data.get("list", [])
            for r in rows:
                r["market"] = market
            return rows

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            if attempt < retries - 1:
                import time
                delay = _RETRY_DELAYS[attempt]
                time.sleep(delay)
                log_print(f"  [retry {attempt+1}/{retries-1} wait={delay}s] {e}")
            else:
                raise


def _fetch_date_markets(date_str, api_key):
    """Fetch all markets for a single date — **sequentially** to avoid
    bursting the per-endpoint rate limiter."""
    all_rows = []
    for m in MARKETS:
        try:
            batch = _fetch_one(date_str, m, api_key)
            if batch:
                all_rows.extend(batch)
        except Exception as e:
            log_print(f"  [ths_hot] {date_str}/{m} FAILED: {e}")
    return all_rows


def _fetch_month(label, dates, api_key, checkpoints_dir):
    """Fetch all dates in one month, save checkpoint."""
    all_rows = []

    for i, d in enumerate(dates):
        try:
            batch = _fetch_date_markets(d, api_key)
            if batch:
                all_rows.extend(batch)
            log_print(f"  [ths_hot/{label}] [{i+1}/{len(dates)}] {d} -> "
                      f"{len(batch)} rows | tokens={_limiter.stats['tokens_available']}")
        except Exception as e:
            log_print(f"  [ths_hot/{label}] [{i+1}/{len(dates)}] {d} FAILED: {e}")

    if all_rows:
        df_m = pd.DataFrame(all_rows)
        checkpoint_file = checkpoints_dir / f"{label}.parquet"
        tmp = checkpoint_file.with_suffix(".parquet.tmp")
        df_m.to_parquet(tmp, index=False)
        tmp.replace(checkpoint_file)

    return all_rows


def fetch_ths_hot(start_date="2019-01-01", end_date=None, output=None,
                  resume=True, workers=6, cleanup=True):
    """Fetch THS (同花顺) hot ranking data for all markets.

    Only queries trading days (from calendar.parquet) to avoid wasting
    API calls on weekends/holidays.  Markets are fetched sequentially
    per date to respect the 200 req/min rate limit.

    Parameters
    ----------
    workers: Ignored (kept for main.py compatibility).  Markets are
             always fetched sequentially.
    """
    api_key = load_api_key()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    dates = _trading_days(start_date, end_date)
    log_print(f"[ths_hot] {len(dates)} trading days: {dates[0] if dates else 'none'} "
              f"~ {dates[-1] if dates else 'none'}")

    if not dates:
        log_print("[ths_hot] No trading days in range")
        return pd.DataFrame()

    month_dates = defaultdict(list)
    for d in dates:
        month_dates[_month_label(d)].append(d)

    log_print(f"[ths_hot] {len(month_dates)} months | markets={MARKETS}")

    checkpoints_dir = Path(DATA_DIR) / "ths_hot"
    checkpoints_dir.mkdir(exist_ok=True)

    all_data = []
    todo = []
    for label in sorted(month_dates.keys()):
        checkpoint_file = checkpoints_dir / f"{label}.parquet"
        if resume and checkpoint_file.exists():
            try:
                df_m = pd.read_parquet(checkpoint_file)
                all_data.extend(df_m.to_dict(orient="records"))
                log_print(f"[ths_hot] {label} -> checkpoint ({len(df_m)} rows)")
                continue
            except Exception:
                log_print(f"[ths_hot] {label} -> checkpoint corrupt, re-fetching")
                checkpoint_file.unlink(missing_ok=True)
        todo.append(label)

    if not todo:
        log_print("[ths_hot] All months already cached")
    else:
        log_print(f"[ths_hot] {len(todo)} months to fetch, "
                  f"tokens={_limiter.stats['tokens_available']}")

        for i, label in enumerate(todo):
            m_dates = month_dates[label]
            log_print(f"[ths_hot] [{i+1}/{len(todo)}] {label}: {len(m_dates)} days")
            batch = _fetch_month(label, m_dates, api_key, checkpoints_dir)
            all_data.extend(batch)
            log_print(f"[ths_hot] [{i+1}/{len(todo)}] {label} -> "
                      f"{len(batch)} rows | tokens={_limiter.stats['tokens_available']}")

    df = pd.DataFrame(all_data)
    log_print(f"[ths_hot] Total: {len(df)} rows, {len(df.columns)} columns")

    if output is None:
        output = f"{DATA_DIR}/ths_hot.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[ths_hot] Saved -> {output}")

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
        log_print(f"[ths_hot] Cleaned up {removed} checkpoint files")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch THS hot ranking data")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoints, re-fetch all")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="(unused, kept for compatibility)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-month checkpoint files")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()

    fetch_ths_hot(
        start_date=args.start,
        end_date=args.end,
        output=args.output,
        resume=not args.no_resume,
        workers=args.workers,
        cleanup=not args.no_cleanup,
    )
