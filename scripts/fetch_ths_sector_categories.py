import requests
import argparse
import pandas as pd
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print

ENDPOINT = "index/ths_sector_categories"

# All known sector types — used to split requests if total > 100K
_ALL_TYPES = ["N", "I", "R", "S", "ST", "TH", "BB"]


def _fetch_page(sector_type, page, page_size, api_key, retries=3):
    """Fetch a single page, optionally filtered by type."""
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {"page": page, "page_size": page_size}
    if sector_type:
        payload["type"] = sector_type

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
                time.sleep(2 ** attempt)
                log_print(f"  [retry {attempt+1}/{retries}] {e}")
            else:
                raise


def _fetch_all(sector_type, api_key):
    """Fetch all categories (optionally filtered by type)."""
    all_data = []
    page = 0
    page_size = 10000
    while True:
        batch, total = _fetch_page(sector_type, page, page_size, api_key)
        if not batch:
            break
        all_data.extend(batch)
        if len(all_data) >= total:
            break
        page += 1
    return all_data


def fetch_ths_sector_categories(output=None):
    """Fetch all THS sector categories (概念/行业/地域/风格/主题/宽基).

    The API limits page*page_size to 100K.  If the full dataset exceeds
    that we split by *type* — each type has far fewer than 100K entries.
    """
    api_key = load_api_key()

    # Check total first
    page0, total = _fetch_page(None, 0, 10000, api_key)
    if not page0:
        log_print("[ths_sector_categories] No data returned")
        return pd.DataFrame()

    if total <= 100000:
        all_data = list(page0)
        page = 1
        while len(all_data) < total:
            batch, _ = _fetch_page(None, page, 10000, api_key)
            if not batch:
                break
            all_data.extend(batch)
            page += 1
    else:
        log_print(f"[ths_sector_categories] total={total} > 100K, "
                  f"splitting by type ({len(_ALL_TYPES)} types)")
        all_data = []
        for t in _ALL_TYPES:
            try:
                rows = _fetch_all(t, api_key)
                all_data.extend(rows)
                log_print(f"  [ths_sector_categories] type={t}: {len(rows)} rows")
            except Exception as e:
                log_print(f"  [ths_sector_categories] type={t} FAILED: {e}")

    df = pd.DataFrame(all_data)

    if not df.empty and 'index_code' in df.columns:
        df = df.sort_values(['type', 'index_code']).reset_index(drop=True)

    type_counts = df['type'].value_counts().to_dict() if 'type' in df.columns else {}
    log_print(f"[ths_sector_categories] Total: {len(df)} categories | {type_counts}")

    if output is None:
        output = f"{DATA_DIR}/ths_sector_categories.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[ths_sector_categories] Saved -> {output}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch THS sector categories")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()
    fetch_ths_sector_categories(output=args.output)
