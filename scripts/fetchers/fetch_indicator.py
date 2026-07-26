"""
Local technical indicator computation from raw 1min K-line data.

Reads OHLCV data from history_1min/{code}.parquet (downloaded by the
history task), computes MACD/KDJ/RSI/BOLL/MA locally with pandas
vectorized operations, and saves results as per-stock parquet files
under data/{indicator}_1min/.

Zero API calls — all computation is local.

Indicators computed:
  macd_1min  — DIF, DEA, MACD histogram  (12, 26, 9)
  kdj_1min   — K, D, J                    (9, 3, 3)
  rsi_1min   — RSI                        (6)
  boll_1min  — BOLL mid, upper, lower     (20, 2)
  ma_1min    — MA5, MA10, MA20, MA30, MA60

Usage:
  python fetch_indicator.py --indicator macd --start 2019-01-01 -w 6
  python fetch_indicator.py --all --start 2019-01-01 -w 6

Imported by main.py for full download and by incremental_update.py for
the indicator computation functions.
"""

import argparse
import pandas as pd
import numpy as np
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DATA_DIR, log_print

# ═══════════════════════════════════════════════════════════════════════
# Indicator computation functions (pure pandas, no API)
# ═══════════════════════════════════════════════════════════════════════

def compute_macd(df, fast=12, slow=26, signal=9):
    """Compute MACD on a DataFrame with 'close' column.
    Returns DataFrame with columns: dif, dea, macd (rounded to 4 decimals).
    """
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, min_periods=signal, adjust=False).mean()
    macd_bar = 2.0 * (dif - dea)
    return pd.DataFrame({
        "dif": dif.round(4),
        "dea": dea.round(4),
        "macd": macd_bar.round(4),
    })


def compute_kdj(df, period=9, k_period=3, d_period=3):
    """Compute KDJ. Returns DataFrame with columns: k, d, j (rounded to 2)."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    low_n = low.rolling(window=period, min_periods=period).min()
    high_n = high.rolling(window=period, min_periods=period).max()
    rsv = ((close - low_n) / (high_n - low_n + 1e-12) * 100.0)

    k = rsv.ewm(com=k_period - 1, min_periods=k_period, adjust=False).mean()
    d = k.ewm(com=d_period - 1, min_periods=d_period, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({
        "k": k.round(2),
        "d": d.round(2),
        "j": j.round(2),
    })


def compute_rsi(df, period=6):
    """Compute RSI. Returns DataFrame with column: rsi (rounded to 2)."""
    close = df["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.DataFrame({"rsi": rsi.round(2)})


def compute_boll(df, period=20, std=2.0):
    """Compute BOLL. Returns DataFrame: boll_mid, boll_upper, boll_lower (rounded to 2)."""
    close = df["close"].astype(float)
    mid = close.rolling(window=period, min_periods=period).mean()
    std_dev = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    return pd.DataFrame({
        "boll_mid": mid.round(2),
        "boll_upper": upper.round(2),
        "boll_lower": lower.round(2),
    })


def compute_ma(df, periods=(5, 10, 20, 30, 60)):
    """Compute moving averages. Returns DataFrame with columns: ma5, ma10, … (rounded to 2)."""
    close = df["close"].astype(float)
    cols = {}
    for p in periods:
        ma = close.rolling(window=p, min_periods=p).mean()
        cols[f"ma{p}"] = ma.round(2)
    return pd.DataFrame(cols)



def compute_mavol(df, periods=(5, 10)):
    """Compute volume moving averages. Returns DataFrame with columns: mavol5, mavol10 (rounded to 2)."""
    vol = df["vol"].astype(float)
    cols = {}
    for p in periods:
        mav = vol.rolling(window=p, min_periods=p).mean()
        cols[f"mavol{p}"] = mav.round(2)
    return pd.DataFrame(cols)

# Map indicator name → compute function + params
INDICATOR_COMPUTE = {
    "macd": (compute_macd, {"fast": 12, "slow": 26, "signal": 9}),
    "kdj":  (compute_kdj,  {"period": 9, "k_period": 3, "d_period": 3}),
    "rsi":  (compute_rsi,  {"period": 6}),
    "boll": (compute_boll, {"period": 20, "std": 2.0}),
    "ma":   (compute_ma,   {"periods": (5, 10, 20, 30, 60)}),
    "mavol": (compute_mavol, {"periods": (5, 10)}),
}

# Maximum lookback bars needed (for incremental computation)
MAX_LOOKBACK = 60  # ma60

# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

CODE_NUM_FILE = Path(DATA_DIR).parent / "Code_num.txt"
HISTORY_DIR = Path(DATA_DIR) / "history_1min"


def _add_exchange_suffix(code):
    return code + (".SH" if code.startswith("60") else ".SZ")


def _filter_stocks():
    if not CODE_NUM_FILE.exists():
        log_print("[indicator] Code_num.txt not found")
        return None
    with open(CODE_NUM_FILE) as f:
        codes = [_add_exchange_suffix(line.strip()) for line in f if line.strip()]
    log_print(f"[indicator] {len(codes)} stocks loaded")
    return codes


def _compute_indicators_for_df(df, indicator_names=None):
    """Compute one or more indicators on a raw OHLCV DataFrame.

    Args:
        df: DataFrame with columns [close, high, low] (plus trade_time, stock_code, etc.)
        indicator_names: list of indicator names, or None for all.

    Returns:
        dict: {indicator_name: DataFrame with indicator columns + same index as df}
    """
    if indicator_names is None:
        indicator_names = list(INDICATOR_COMPUTE.keys())

    results = {}
    for name in indicator_names:
        fn, params = INDICATOR_COMPUTE[name]
        ind_df = fn(df, **params)
        results[name] = ind_df
    return results


# ═══════════════════════════════════════════════════════════════════════
# Per-stock full download (reads history_1min → saves indicators)
# ═══════════════════════════════════════════════════════════════════════

def _compute_stock_indicators(stock_code, start_date, end_date,
                               indicator_names, output_dirs, resume):
    """Compute indicators for one stock from its history_1min file,
    then save to one or more output directories.

    Returns dict: {ind_name: row_count} for successfully saved indicators.
    """
    hist_file = HISTORY_DIR / f"{stock_code}.parquet"
    if not hist_file.exists():
        log_print(f"  [{stock_code}] No history_1min file — skipped")
        return {}

    # ── Read raw K-line ──
    df = pd.read_parquet(hist_file)
    if "trade_time" not in df.columns or df.empty:
        return {}

    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df.sort_values("trade_time").reset_index(drop=True)

    # ── Filter date range ──
    if start_date:
        start_dt = pd.Timestamp(start_date)
        df = df[df["trade_time"] >= start_dt]
    if end_date:
        end_dt = pd.Timestamp(end_date)
        df = df[df["trade_time"] <= end_dt]

    if df.empty:
        return {}

    # ── Compute indicators ──
    all_indicators = _compute_indicators_for_df(df, indicator_names)

    # ── Save each indicator ──
    results = {}
    for name, ind_df in all_indicators.items():
        out_dir = output_dirs.get(name)
        if out_dir is None:
            continue

        out_file = out_dir / f"{stock_code}.parquet"

        # Build output: keep trade_time + stock_code + indicator columns
        out_cols = ["trade_time", "stock_code"] + list(ind_df.columns)
        out = pd.concat([df[["trade_time", "stock_code"]].reset_index(drop=True),
                         ind_df.reset_index(drop=True)], axis=1)

        # ── Merge with existing ──
        if resume and out_file.exists():
            existing = pd.read_parquet(out_file)
            existing["trade_time"] = pd.to_datetime(existing["trade_time"])
            merged = pd.concat([existing, out], ignore_index=True)
            merged = merged.drop_duplicates(subset=["trade_time", "stock_code"], keep="last")
            merged = merged.sort_values("trade_time").reset_index(drop=True)
        else:
            merged = out

        tmp = out_file.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(out_file)
        results[name] = len(merged)

    return results


def _run_full(indicator_names, start_date, end_date, resume, workers):
    """Full download: read history_1min, compute indicators, save to per-stock files.

    If *indicator_names* is None, computes all five indicators.
    """
    if indicator_names is None:
        indicator_names = list(INDICATOR_COMPUTE.keys())

    stocks = _filter_stocks()
    if stocks is None:
        return

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    # Create output directories
    output_dirs = {}
    tags = []
    for name in indicator_names:
        tag = f"indicator_1min/{name}"
        tags.append(tag)
        out_dir = Path(DATA_DIR) / "indicator_1min" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[name] = out_dir

    tag_str = ",".join(tags)
    log_print(f"[{tag_str}] Full download from history_1min | "
              f"{len(stocks)} stocks | {start_date} ~ {end_date}")

    # Determine which stocks need computation
    todo = []
    for code in stocks:
        hist_file = HISTORY_DIR / f"{code}.parquet"
        if not hist_file.exists():
            continue
        # Check if any output file is missing
        need_compute = False
        if not resume:
            need_compute = True
        else:
            for name in indicator_names:
                out_file = output_dirs[name] / f"{code}.parquet"
                if not out_file.exists():
                    need_compute = True
                    break
        if need_compute:
            todo.append(code)

    if not todo:
        log_print(f"[{tag_str}] All stocks already cached")
        for out_dir in output_dirs.values():
            done_file = out_dir / ".done"
            done_file.write_text(str(date.today()))
        return

    total_todo = len(todo)
    log_print(f"[{tag_str}] {total_todo} stocks to compute (parallel x{workers})")

    total_rows = 0
    completed = 0
    failed = 0
    t_start = time.perf_counter()
    progress_lock = threading.Lock()
    last_log_time = [time.perf_counter()]

    def _compute_one(code):
        nonlocal total_rows, completed, failed
        results = _compute_stock_indicators(
            code, start_date, end_date, indicator_names, output_dirs, resume)
        with progress_lock:
            completed += 1
            rows = sum(results.values())
            total_rows += rows
            if not results:
                failed += 1
            now = time.perf_counter()
            if now - last_log_time[0] >= 1.0:
                last_log_time[0] = now
                elapsed = now - t_start
                pct = completed / total_todo * 100
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                eta_s = (total_todo - completed) / rate * 60 if rate > 0 else 0
                eta_str = f"{eta_s/60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"
                bar_w = 20
                filled = int(bar_w * completed / total_todo)
                bar = "#" * filled + "-" * (bar_w - filled)
                log_print(f"[{tag_str}] [{bar}] {pct:5.1f}%  {completed}/{total_todo}  "
                          f"{rate:.0f} st/min  ETA {eta_str}  "
                          f"rows:{total_rows}  fail:{failed}")
        return results

    max_workers = min(workers, total_todo)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compute_one, code): code for code in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                code = futures[fut]
                with progress_lock:
                    completed += 1
                    failed += 1
                log_print(f"[{tag_str}] {code} FAILED: {e}")

    elapsed_total = time.perf_counter() - t_start
    log_print(f"[{tag_str}] Done: {total_rows} rows from {total_todo} stocks "
              f"(failed:{failed}) in {elapsed_total/60:.1f} min")
    for out_dir in output_dirs.values():
        done_file = out_dir / ".done"
        done_file.write_text(str(date.today()))
    log_print(f"[{tag_str}] Done markers written")


# ═══════════════════════════════════════════════════════════════════════
# Public entry points (for main.py TASKS)
# ═══════════════════════════════════════════════════════════════════════

def _make_fetch_fn(indicator_name):
    """Return a callable with the standard TASKS signature for one indicator."""
    def _fetch(start_date="2019-01-01", end_date=None, output=None,
               resume=True, workers=6, cleanup=True, **kwargs):
        _run_full(
            indicator_names=[indicator_name],
            start_date=start_date,
            end_date=end_date,
            resume=resume,
            workers=workers,
        )
    _fetch.__name__ = f"fetch_{indicator_name}_1min"
    return _fetch


def fetch_all_indicators(start_date="2019-01-01", end_date=None, output=None,
                          resume=True, workers=6, cleanup=True, **kwargs):
    """Compute all five indicators from history_1min in a single pass."""
    _run_full(
        indicator_names=None,  # all
        start_date=start_date,
        end_date=end_date,
        resume=resume,
        workers=workers,
    )


# ═══════════════════════════════════════════════════════════════════════
# Combined indicators — one pass, one file per stock
# ═══════════════════════════════════════════════════════════════════════

def _compute_stock_indicators_combined(stock_code, start_date, end_date,
                                        output_dir, resume):
    """Compute ALL six indicators for one stock from history_1min,
    then save into a SINGLE combined parquet file.

    Returns row count, or 0 if skipped/failed.
    """
    hist_file = HISTORY_DIR / f"{stock_code}.parquet"
    if not hist_file.exists():
        log_print(f"  [{stock_code}] No history_1min file — skipped")
        return 0

    df = pd.read_parquet(hist_file)
    if "trade_time" not in df.columns or df.empty:
        return 0

    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df.sort_values("trade_time").reset_index(drop=True)

    if start_date:
        df = df[df["trade_time"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["trade_time"] <= pd.Timestamp(end_date)]

    if df.empty:
        return 0

    # Compute all six indicators at once
    all_indicators = _compute_indicators_for_df(df, None)  # None = all

    # Build combined output: trade_time, stock_code + all indicator columns
    out = df[["trade_time", "stock_code"]].copy()
    for _name, ind_df in all_indicators.items():
        for col in ind_df.columns:
            out[col] = ind_df[col].values

    out_file = output_dir / f"{stock_code}.parquet"

    if resume and out_file.exists():
        existing = pd.read_parquet(out_file)
        existing["trade_time"] = pd.to_datetime(existing["trade_time"])
        merged = pd.concat([existing, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=["trade_time", "stock_code"], keep="last")
        merged = merged.sort_values("trade_time").reset_index(drop=True)
    else:
        merged = out

    tmp = out_file.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(out_file)
    return len(merged)


def fetch_indicators_combined(start_date="2019-01-01", end_date=None, output=None,
                               resume=True, workers=6, cleanup=True, **kwargs):
    """Compute ALL six technical indicators from history_1min in a single pass.

    Saves one combined parquet file per stock under DATA_DIR/indicator_1min/
    containing all indicator columns (dif, dea, macd, k, d, j, rsi,
    boll_mid, boll_upper, boll_lower, ma5, ma10, ma20, ma30, ma60,
    mavol5, mavol10).
    """
    stocks = _filter_stocks()
    if stocks is None:
        return

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    output_dir = Path(DATA_DIR) / "indicator_1min"
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_str = "indicator_1min"
    log_print(f"[{tag_str}] Combined indicators from history_1min | "
              f"{len(stocks)} stocks | {start_date} ~ {end_date}")

    # Determine which stocks need computation
    todo = []
    for code in stocks:
        hist_file = HISTORY_DIR / f"{code}.parquet"
        if not hist_file.exists():
            continue
        if not resume:
            todo.append(code)
        else:
            out_file = output_dir / f"{code}.parquet"
            if not out_file.exists():
                todo.append(code)

    if not todo:
        log_print(f"[{tag_str}] All stocks already cached")
        done_file = output_dir / ".done"
        done_file.write_text(str(date.today()))
        return

    total_todo = len(todo)
    log_print(f"[{tag_str}] {total_todo} stocks to compute (parallel x{workers})")

    total_rows = 0
    completed = 0
    failed = 0
    t_start = time.perf_counter()
    progress_lock = threading.Lock()
    last_log_time = [time.perf_counter()]

    def _compute_one(code):
        nonlocal total_rows, completed, failed
        rows = _compute_stock_indicators_combined(
            code, start_date, end_date, output_dir, resume)
        with progress_lock:
            completed += 1
            total_rows += rows
            if rows == 0:
                failed += 1
            now = time.perf_counter()
            if now - last_log_time[0] >= 1.0:
                last_log_time[0] = now
                elapsed = now - t_start
                pct = completed / total_todo * 100
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                eta_s = (total_todo - completed) / rate * 60 if rate > 0 else 0
                eta_str = f"{eta_s/60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"
                bar_w = 20
                filled = int(bar_w * completed / total_todo)
                bar = "#" * filled + "-" * (bar_w - filled)
                log_print(f"[{tag_str}] [{bar}] {pct:5.1f}%  {completed}/{total_todo}  "
                          f"{rate:.0f} st/min  ETA {eta_str}  "
                          f"rows:{total_rows}  fail:{failed}")
        return rows

    max_workers = min(workers, total_todo)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compute_one, code): code for code in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                code = futures[fut]
                with progress_lock:
                    completed += 1
                    failed += 1
                log_print(f"[{tag_str}] {code} FAILED: {e}")

    elapsed_total = time.perf_counter() - t_start
    log_print(f"[{tag_str}] Done: {total_rows} rows from {total_todo} stocks "
              f"(failed:{failed}) in {elapsed_total/60:.1f} min")
    done_file = output_dir / ".done"
    done_file.write_text(str(date.today()))
    log_print(f"[{tag_str}] Done marker written")


# Individual wrapper functions (imported by main.py)
fetch_macd_1min = _make_fetch_fn("macd")
fetch_kdj_1min  = _make_fetch_fn("kdj")
fetch_rsi_1min  = _make_fetch_fn("rsi")
fetch_boll_1min = _make_fetch_fn("boll")
fetch_ma_1min   = _make_fetch_fn("ma")
fetch_mavol_1min = _make_fetch_fn("mavol")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute technical indicators locally from history_1min data"
    )
    parser.add_argument("--indicator", choices=list(INDICATOR_COMPUTE.keys()),
                        help="Which indicator to compute")
    parser.add_argument("--all", action="store_true",
                        help="Compute all five indicators in one pass")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", help="End date (default: today)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("-w", "--workers", type=int, default=6)
    args = parser.parse_args()

    if args.all:
        names = None
    elif args.indicator:
        names = [args.indicator]
    else:
        parser.error("Must specify --indicator or --all")

    _run_full(
        indicator_names=names,
        start_date=args.start,
        end_date=args.end,
        resume=not args.no_resume,
        workers=args.workers,
    )
