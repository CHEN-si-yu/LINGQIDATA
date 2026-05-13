import requests
import argparse
import pandas as pd
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print

ENDPOINT = "basic/calendar"


def fetch_calendar(start_time, end_time, output=None):
    api_key = load_api_key()
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key}
    params = {"start_time": start_time, "end_time": end_time}

    limiter = rate_limiter()
    limiter.acquire(ENDPOINT)

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    result = resp.json()

    if result["code"] != 200:
        raise RuntimeError(f"API Error: code={result['code']}, msg={result['msg']}")

    df = pd.DataFrame(result["data"])
    open_days = int((df["is_open"] == 1).sum())
    close_days = int((df["is_open"] == 0).sum())

    log_print(f"[calendar] {start_time} ~ {end_time} | {len(df)} days | Trading: {open_days} | Non-trading: {close_days}")

    if output is None:
        output = f"{DATA_DIR}/calendar.parquet"

    df.to_parquet(output, index=False)
    log_print(f"[calendar] Saved → {output}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch trading calendar")
    parser.add_argument("start_time", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_time", help="End date (YYYY-MM-DD)")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()
    fetch_calendar(args.start_time, args.end_time, args.output)
