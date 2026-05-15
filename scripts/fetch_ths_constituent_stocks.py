import requests
import argparse
import pandas as pd
from pathlib import Path
from config import load_api_key, BASE_URL, DATA_DIR, rate_limiter, log_print

ENDPOINT = "index/ths_constituent_stocks"


def _fetch_page(index_code, page, page_size, api_key, retries=3):
    """Fetch a single page, optionally filtered by index_code."""
    url = f"{BASE_URL}/{ENDPOINT}"
    headers = {"apiKey": api_key, "Content-Type": "application/json"}
    payload = {"page": page, "page_size": page_size}
    if index_code:
        payload["index_code"] = index_code

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


def _fetch_index(index_code, api_key):
    """Fetch all constituent rows for one index_code (well under 100K each)."""
    all_data = []
    page = 0
    page_size = 10000
    while True:
        batch, total = _fetch_page(index_code, page, page_size, api_key)
        if not batch:
            break
        all_data.extend(batch)
        if len(all_data) >= total:
            break
        page += 1
    return all_data


def _resolve_index_codes():
    """Return index_codes from ths_sector_categories.parquet, or None."""
    cat_path = Path(DATA_DIR) / "ths_sector_categories.parquet"
    if cat_path.exists():
        try:
            cat = pd.read_parquet(cat_path)
            if "index_code" in cat.columns:
                codes = cat["index_code"].dropna().unique().tolist()
                log_print(f"[ths_constituent_stocks] Derived {len(codes)} index codes "
                          f"from ths_sector_categories.parquet")
                return codes
        except Exception:
            pass
    return None


def _collect_index_codes_from_api(api_key):
    """Fetch the first 100K rows without index_code filter and extract unique codes."""
    page0, total = _fetch_page(None, 0, 10000, api_key)
    if not page0:
        return [], []

    all_data = list(page0)
    page = 1
    while len(all_data) < total and len(all_data) < 100000:
        batch, _ = _fetch_page(None, page, 10000, api_key)
        if not batch:
            break
        all_data.extend(batch)
        page += 1

    df = pd.DataFrame(all_data)
    codes = df["index_code"].dropna().unique().tolist() if "index_code" in df.columns else []
    log_print(f"[ths_constituent_stocks] Extracted {len(codes)} index codes "
              f"from first {len(all_data)} rows (total={total})")
    return codes, all_data


def fetch_ths_constituent_stocks(output=None):
    """Fetch all THS constituent stock mappings (index_code -> stock_code).

    The API limits page*page_size to 100K.  We work around this by fetching
    per-index_code — each index has far fewer than 100K constituents.
    """
    api_key = load_api_key()

    # Try to get index codes from existing sector categories data
    index_codes = _resolve_index_codes()
    seed_data = []

    if not index_codes:
        # Gather index codes from the first 100K rows, also keep that data
        index_codes, seed_data = _collect_index_codes_from_api(api_key)
        if not index_codes:
            log_print("[ths_constituent_stocks] No index codes found, abort")
            return pd.DataFrame()

    # Fetch per-index_code (each well under 100K)
    all_data = list(seed_data)  # reuse seed data to avoid redundant calls
    seen_keys = set()
    if seed_data:
        for row in seed_data:
            key = (row.get("index_code", ""), row.get("stock_code", ""))
            seen_keys.add(key)

    total_indices = len(index_codes)
    new_rows = 0
    for i, code in enumerate(index_codes):
        try:
            rows = _fetch_index(code, api_key)
            added = 0
            for row in rows:
                key = (row.get("index_code", ""), row.get("stock_code", ""))
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_data.append(row)
                    added += 1
            new_rows += added
        except Exception as e:
            log_print(f"  [ths_constituent_stocks] index {code} FAILED: {e}")

        if (i + 1) % 200 == 0 or (i + 1) == total_indices:
            log_print(f"[ths_constituent_stocks] [{i+1}/{total_indices}] indices done | "
                      f"{len(all_data)} rows accumulated")

    df = pd.DataFrame(all_data)

    if not df.empty and 'index_code' in df.columns:
        df = df.sort_values(['index_code', 'stock_code']).reset_index(drop=True)

    n_indices = df['index_code'].nunique() if 'index_code' in df.columns else 0
    log_print(f"[ths_constituent_stocks] Total: {len(df)} rows | "
              f"{n_indices} unique indices | {new_rows} new from per-index fetch")

    if output is None:
        output = f"{DATA_DIR}/ths_constituent_stocks.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[ths_constituent_stocks] Saved -> {output}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch THS constituent stock mappings")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()
    fetch_ths_constituent_stocks(output=args.output)
