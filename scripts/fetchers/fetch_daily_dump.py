"""
Fetch full-market daily dump data from /api/stock/daily_dump.

Supports 1min/5min/15min/30min/60min minute data and daily data.
API is limited to 10 calls per date per day — use sparingly.

Usage:
  python fetch_daily_dump.py --date 2026-05-20 --level 1min
"""

import argparse
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print

ENDPOINT = "stock/daily_dump"

COLUMNS = ["trade_date", "trade_time", "stock_code",
           "open", "high", "low", "close", "vol", "amount"]


def _flatten_minute(data: dict, date_str: str) -> pd.DataFrame:
    """Flatten Map<StockCode, List<[time, o, h, l, c, vol, amount]>> into a DataFrame."""
    rows = []
    for stock_code, entries in data.items():
        for entry in entries:
            rows.append({
                "trade_date": date_str,
                "trade_time": f"{date_str} {entry[0]}:00",
                "stock_code": stock_code,
                "open": entry[1],
                "high": entry[2],
                "low": entry[3],
                "close": entry[4],
                "vol": entry[5],
                "amount": entry[6],
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_daily_dump(date_str: str, level: str = "1min",
                     output_dir: str = None, skip_if_exists: bool = True) -> pd.DataFrame:
    """Fetch full-market daily dump and save to parquet.

    Args:
        date_str: Trading date (YYYY-MM-DD).
        level: Data granularity — daily/1min/5min/15min/30min/60min.
        output_dir: Directory to save parquet files (default: data/daily_dump_{level}/).
        skip_if_exists: If True and the output file already exists, read from disk
            instead of calling the API (default: True).

    Returns:
        DataFrame with the fetched data.
    """
    if output_dir is None:
        output_dir = Path(DATA_DIR) / f"daily_dump_{level}"
    output_dir = Path(output_dir)
    out_file = output_dir / f"{date_str}.parquet"

    # ── Cache hit: file already exists ──
    if skip_if_exists and out_file.exists():
        log_print(f"[daily_dump] Cache hit → {out_file}")
        return pd.read_parquet(out_file)

    api_key = load_api_key()
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {"date": date_str, "level": level}

    limiter = rate_limiter()
    limiter.acquire(ENDPOINT)

    log_print(f"[daily_dump] POST {url}  date={date_str}  level={level}")
    t0 = datetime.now()

    resp = requests.post(url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    result = resp.json()

    if result["code"] != 200:
        raise RuntimeError(
            f"API Error: code={result['code']}, msg={result.get('msg', result)}"
        )

    data = result["data"]
    elapsed = (datetime.now() - t0).total_seconds()

    if level == "daily":
        df = pd.DataFrame(data)
        log_print(f"[daily_dump] {len(df)} daily records in {elapsed:.1f}s")
    else:
        df = _flatten_minute(data, date_str)
        log_print(f"[daily_dump] {len(df)} rows | {df['stock_code'].nunique()} stocks "
                  f"in {elapsed:.1f}s")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)
    file_mb = out_file.stat().st_size / (1024 * 1024)
    log_print(f"[daily_dump] Saved → {out_file}  ({len(df):,} rows, {file_mb:.1f} MB)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch full-market daily dump (daily limited to 10 calls per date)"
    )
    parser.add_argument("--date", required=True, help="Trading date (YYYY-MM-DD)")
    parser.add_argument("--level", default="1min",
                        choices=["daily", "1min", "5min", "15min", "30min", "60min"],
                        help="Data granularity (default: 1min)")
    parser.add_argument("-o", "--output-dir", help="Output directory for parquet files")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file already exists locally")
    args = parser.parse_args()

    fetch_daily_dump(args.date, args.level, args.output_dir,
                     skip_if_exists=not args.force)
