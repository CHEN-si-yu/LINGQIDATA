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


# ── Checkpoint helpers ──────────────────────────────────────────────────

_CHECKPOINT_INTERVAL = 50  # Save progress every N indices
_CHECKPOINT_DIR = Path(DATA_DIR) / ".checkpoints" / "ths_constituent_stocks"


def _load_checkpoint():
    """Load checkpoint data. Returns (all_data, seen_keys, processed_indices)."""
    ckpt_parquet = _CHECKPOINT_DIR / "progress.parquet"
    ckpt_json = _CHECKPOINT_DIR / "progress.json"
    if ckpt_parquet.exists() and ckpt_json.exists():
        try:
            import json
            df = pd.read_parquet(ckpt_parquet)
            all_data = df.to_dict("records")
            with open(ckpt_json, "r") as f:
                meta = json.load(f)
            seen_keys = {tuple(k) for k in meta.get("seen_keys", [])}
            processed = set(meta.get("processed_indices", []))
            log_print(
                f"[ths_constituent_stocks] Resuming from checkpoint: "
                f"{len(all_data)} rows, {len(processed)} indices done, "
                f"{len(seen_keys)} unique keys"
            )
            return all_data, seen_keys, processed
        except Exception as e:
            log_print(f"  [checkpoint] Load failed: {e}, starting fresh")
    return [], set(), set()


def _save_checkpoint(all_data, seen_keys, processed_indices, force=False, total=0):
    """Save intermediate progress to disk. *force* writes even if interval not met."""
    import json
    try:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_data)
        df.to_parquet(_CHECKPOINT_DIR / "progress.parquet", index=False)
        # Store seen_keys as list-of-lists for JSON serialization
        sk_list = [list(k) for k in seen_keys]
        with open(_CHECKPOINT_DIR / "progress.json", "w") as f:
            json.dump({
                "seen_keys": sk_list,
                "processed_indices": list(processed_indices),
                "rows": len(all_data),
                "total_indices": total,
            }, f)
    except Exception as e:
        log_print(f"  [checkpoint] Save warning: {e}")


def _clear_checkpoint():
    """Remove checkpoint files after successful completion."""
    import shutil
    try:
        if _CHECKPOINT_DIR.exists():
            shutil.rmtree(_CHECKPOINT_DIR)
    except Exception:
        pass


def fetch_ths_constituent_stocks(output=None):
    """Fetch all THS constituent stock mappings (index_code -> stock_code).

    The API limits page*page_size to 100K.  We work around this by fetching
    per-index_code — each index has far fewer than 100K constituents.

    Checkpoints are saved every ``_CHECKPOINT_INTERVAL`` indices so that
    partial progress is never lost across restarts.  The checkpoint is
    automatically cleaned up on successful completion.
    """
    api_key = load_api_key()

    # Try to get index codes from existing sector categories data
    index_codes = _resolve_index_codes()
    seed_data = []

    # ── Load checkpoint if resuming ──
    all_data, seen_keys, processed = _load_checkpoint()
    if processed:
        # Filter out already-processed indices
        index_codes = [c for c in index_codes if c not in processed]
        seed_data = []  # seed data already in checkpoint

    if not index_codes and not processed:
        # Gather index codes from the first 100K rows, also keep that data
        index_codes, seed_data = _collect_index_codes_from_api(api_key)
        if not index_codes:
            log_print("[ths_constituent_stocks] No index codes found, abort")
            return pd.DataFrame()

    # Merge seed data into checkpoint data
    if seed_data:
        for row in seed_data:
            key = (row.get("index_code", ""), row.get("stock_code", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                all_data.append(row)

    total_indices = len(index_codes) + len(processed)
    if not processed:
        log_print(f"[ths_constituent_stocks] Fetching {total_indices} indices "
                  f"(checkpoint every {_CHECKPOINT_INTERVAL})")

    failed_indices: list[str] = []
    new_rows = 0
    start_idx = len(processed)

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
            processed.add(code)
        except Exception as e:
            log_print(f"  [ths_constituent_stocks] index {code} FAILED: {e}")
            failed_indices.append(code)
            # Still mark as processed so checkpoint can skip it on resume
            processed.add(code)

        global_i = start_idx + i + 1
        if (i + 1) % _CHECKPOINT_INTERVAL == 0 or global_i == total_indices:
            log_print(
                f"[ths_constituent_stocks] [{global_i}/{total_indices}] "
                f"indices done | {len(all_data)} rows | "
                f"{len(failed_indices)} failed so far"
            )
            _save_checkpoint(all_data, seen_keys, processed, total=total_indices)

    # ── Final assembly ──
    df = pd.DataFrame(all_data)

    if not df.empty and "index_code" in df.columns:
        df = df.sort_values(["index_code", "stock_code"]).reset_index(drop=True)

    n_indices = df["index_code"].nunique() if "index_code" in df.columns else 0
    log_print(
        f"[ths_constituent_stocks] Total: {len(df)} rows | "
        f"{n_indices} unique indices | {new_rows} new from per-index fetch"
    )
    if failed_indices:
        log_print(
            f"[ths_constituent_stocks] WARNING: {len(failed_indices)} indices FAILED: "
            f"{failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}"
        )

    if output is None:
        output = f"{DATA_DIR}/ths_constituent_stocks.parquet"
    df.to_parquet(output, index=False)
    log_print(f"[ths_constituent_stocks] Saved -> {output}")

    # Clean up checkpoint on success
    _clear_checkpoint()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch THS constituent stock mappings")
    parser.add_argument("-o", "--output", help="Output parquet path")
    args = parser.parse_args()
    fetch_ths_constituent_stocks(output=args.output)
