import requests
import argparse
import pandas as pd
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print

ENDPOINT = "stock/list"


def fetch_stock_list(output=None):
    """Fetch the full stock list (basic info for all listed stocks).

    Returns a DataFrame with stock_code, name, area, industry, list_date,
    symbol, act_name, act_ent_type, list_status, delist_date, is_hs.
    """
    api_key = load_api_key()
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key}

    limiter = rate_limiter()
    limiter.acquire(ENDPOINT)

    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    if result["code"] != 200:
        raise RuntimeError(f"API Error: code={result['code']}, msg={result['msg']}")

    data = result["data"]
    all_rows = list(data["list"])
    total = data["total"]
    log_print(f"[stock_list] page 0: {len(all_rows)} rows | total={total}")

    df = pd.DataFrame(all_rows)

    if not df.empty and 'stock_code' in df.columns:
        _key = df['stock_code'].str.extract(r'(\d{6})').iloc[:, 0]
        df['_sort_code'] = pd.to_numeric(_key, errors='coerce').fillna(0).astype(int)
        df = df.sort_values('_sort_code').drop(columns=['_sort_code']).reset_index(drop=True)

    log_print(f"[stock_list] Total: {len(df)} stocks, {len(df.columns)} columns")

    if output is None:
        output = f"{DATA_DIR}/stock_list.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[stock_list] Saved -> {output}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch full stock list")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()
    fetch_stock_list(output=args.output)
