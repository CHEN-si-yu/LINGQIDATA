import argparse
import os
import sys
import threading
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure fetchers/ is on the import path so config and fetch_* modules
# can be imported directly after the directory restructure.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetchers"))

from config import load_api_key, rate_limiter, log_print, DATA_DIR
from fetch_calendar import fetch_calendar
from fetch_finance import fetch_finance
from fetch_stock_list import fetch_stock_list
from fetch_daily import fetch_daily, fetch_daily_adj
from fetch_main_fund_flow import fetch_main_fund_flow

from fetch_cyq_chips import fetch_cyq_chips
from fetch_cyq_perf import fetch_cyq_perf
from fetch_minute import fetch_history

from fetch_margin_detail import fetch_margin_detail
from fetch_ths_daily import fetch_ths_daily
from fetch_ths_sector_categories import fetch_ths_sector_categories
from fetch_ths_constituent_stocks import fetch_ths_constituent_stocks

# Technical indicators (1min)
from fetch_indicator import fetch_indicators_combined

CODE_NUM_FILE = Path(DATA_DIR).parent / "Code_num.txt"


def load_stock_codes():
    """Load the 6-digit stock-code list from Code_num.txt (no suffix).

    Used as stock_filter for the date-based fetchers so that the final
    parquet files contain exactly this universe (Code_num stocks only).
    """
    if not CODE_NUM_FILE.exists():
        log_print(f"[main] WARNING: {CODE_NUM_FILE} not found, no stock filter")
        return None
    with open(CODE_NUM_FILE) as f:
        codes = [line.strip() for line in f if line.strip()]
    log_print(f"[main] {len(codes)} stocks loaded from Code_num.txt")
    return codes


# ── Task registry ──────────────────────────────────────────────────
# Comment out any line to skip that endpoint.
# Each task's final output file is checked: if it already exists the
# task is skipped (pass --force to override).
# Tasks with "stock_filter": True only keep Code_num.txt stocks in the
# final parquet (daily/finance/margin_detail/cyq_perf). cyq_chips is
# per-stock and already iterates Code_num.txt itself.

TASKS = [
    # {
    #     "name": "indicators_1min",
    #     "fn": fetch_indicators_combined,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/indicator_1min/.done",
    # },
    # {
    #     "name": "stock_list",
    #     "fn": fetch_stock_list,
    #     "output": f"{DATA_DIR}/stock_list.parquet",
    # },
    # {
    #     "name": "calendar",
    #     "fn": fetch_calendar,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/calendar.parquet",
    # },
    # {
    #     "name": "ths_sector_categories",
    #     "fn": fetch_ths_sector_categories,
    #     "output": f"{DATA_DIR}/ths_sector_categories.parquet",
    # },
    # {
    #     "name": "ths_constituent_stocks",
    #     "fn": fetch_ths_constituent_stocks,
    #     "output": f"{DATA_DIR}/ths_constituent_stocks.parquet",
    # },
    # {
    #     "name": "ths_daily",
    #     "fn": fetch_ths_daily,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/ths_daily.parquet",
    # },
    # ════════════════════ 有差异数据源: 全量重拉 (2026-08-01 一致性检查后启用) ════════════════════
    # 运行: python scripts/main.py --force [-w 8]   (全量覆盖本地, 2019-01-01 ~ 今天; 并发≤8)
    # refetch_recovery=True: --force 时仍复用已抓取的月度 checkpoint (断点续跑);
    # 如需彻底重抓, 先删除 data/daily data/finance data/margin_detail
    # data/cyq_perf_checkpoints 目录及 data/cyq_chips/*.parquet。
    #
    # 状态 (2026-08-01 17:xx):
    # - daily / finance / margin_detail / cyq_perf: 已重拉完成并验证无误
    #   (1782 只全在 Code_num 内, 91 个月完整), 暂时注释。需要重拉时取消注释。
    # - cyq_chips: 待重拉 (数据量大, 逐股拉取 ~2.5h)。
    # {
    #     "name": "daily",
    #     "fn": fetch_daily,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/daily.parquet",
    #     "stock_filter": True,
    #     "refetch_recovery": True,
    # },
    # {
    #     "name": "finance",
    #     "fn": fetch_finance,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/finance.parquet",
    #     "stock_filter": True,
    #     "refetch_recovery": True,
    # },
    # {
    #     "name": "margin_detail",
    #     "fn": fetch_margin_detail,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/margin_detail.parquet",
    #     "stock_filter": True,
    #     "refetch_recovery": True,
    # },
    # {
    #     "name": "cyq_perf",
    #     "fn": fetch_cyq_perf,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/cyq_perf.parquet",
    #     "stock_filter": True,
    #     "refetch_recovery": True,
    # },
    # ════════════════════ 天级 (per-stock, 散文件落盘) ════════════════════
    {
        "name": "cyq_chips",
        "fn": fetch_cyq_chips,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/cyq_chips/.done",
        "refetch_recovery": True,
    },
    # {
    #     "name": "daily_adj",
    #     "fn": fetch_daily_adj,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/daily_adj.parquet",
    # },
    # {
    #     "name": "main_fund_flow",
    #     "fn": fetch_main_fund_flow,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/main_fund_flow.parquet",
    # },
    # {
    #     "name": "history",
    #     "fn": fetch_history,
    #     "start": "2019-01-01",
    #     "output": f"{DATA_DIR}/history_1min/.done",
    # },
]



def _check_completeness(name, output_path):
    """Return True if the existing output file is complete enough to skip re-fetch.

    For ``ths_constituent_stocks.parquet``, a file with < 50 unique indices is
    treated as incomplete — it means the previous fetch was interrupted or the
    per-index loop failed for most sectors.
    """
    if name == "ths_constituent_stocks":
        try:
            import pandas as pd
            df = pd.read_parquet(output_path)
            n_indices = df["index_code"].nunique() if "index_code" in df.columns else 0
            if n_indices < 50:
                log_print(
                    f"[main] {name} -> INCOMPLETE ({n_indices} unique indices, "
                    f"expected 1600+), forcing re-fetch"
                )
                return False
        except Exception as e:
            log_print(f"[main] {name} -> cannot verify completeness ({e}), "
                      f"forcing re-fetch")
            return False
    return True


def _monitor_loop(stop_event, interval=10.0):
    limiter = rate_limiter()
    while not stop_event.wait(interval):
        s = limiter.stats
        log_print(f"[monitor] tokens={s['tokens_available']}/{s['max_rpm']} "
                  f"| calls={dict(s['endpoint_counts'])}")


def run_all(stock_codes=None, start_date="2019-01-01", end_date=None,
            parallel=False, monitor=False, workers=6, cleanup=True,
            force=False):
    """Orchestrate data fetching based on the TASKS registry.

    Tasks whose final output file already exist are skipped unless
    *force* is True.  Comment out entries in TASKS to permanently
    disable an endpoint.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    limiter = rate_limiter()

    # Decide which tasks actually need to run
    active = []
    for t in TASKS:
        output_path = Path(t["output"])
        if not force and output_path.exists():
            if _check_completeness(t["name"], output_path):
                log_print(f"[main] {t['name']} -> skip (output exists: {output_path})")
                continue
            # File exists but is incomplete — force re-fetch
            log_print(f"[main] {t['name']} -> INCOMPLETE, will re-fetch")
        active.append(t)

    if not active:
        log_print("[main] All tasks up to date — nothing to do")
        return

    log_print(f"[main] start | rate_limit={limiter.max_rpm} req/min | "
              f"parallel={parallel} | workers={workers} | "
              f"tasks={[t['name'] for t in active]}")

    stop_event = threading.Event()
    monitor_thread = None
    if monitor:
        monitor_thread = threading.Thread(
            target=_monitor_loop, args=(stop_event, 10.0), daemon=True
        )
        monitor_thread.start()

    # Stock universe for the date-based fetchers (Code_num.txt)
    stock_codes = None
    if any(t.get("stock_filter") for t in active):
        stock_codes = load_stock_codes()

    def _task_kwargs(t):
        # refetch_recovery: --force 全量重拉时仍启用 checkpoint 断点续跑
        kwargs = {
            "start_date": t["start"],
            "end_date": end_date,
            "workers": workers,
            "cleanup": cleanup,
            "resume": (not force) or t.get("refetch_recovery", False),
        }
        if t.get("stock_filter") and stock_codes is not None:
            kwargs["stock_filter"] = stock_codes
        return kwargs

    try:
        if parallel:
            with ThreadPoolExecutor(max_workers=len(active)) as pool:
                futures = {}
                for t in active:
                    if t["name"] == "calendar":
                        fut = pool.submit(t["fn"], t["start"], end_date)
                    elif t["name"] in ("stock_list", "ths_sector_categories",
                                       "ths_constituent_stocks"):
                        fut = pool.submit(t["fn"])
                    else:
                        fut = pool.submit(t["fn"], **_task_kwargs(t))
                    futures[fut] = t["name"]

                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        fut.result()
                        log_print(f"[main] {name} completed")
                    except Exception as e:
                        log_print(f"[main] {name} FAILED: {e}")
        else:
            for t in active:
                date_info = f" ({t['start']} ~ {end_date})" if "start" in t else ""
                log_print(f"[main] === {t['name']}{date_info} ===")
                if t["name"] == "calendar":
                    t["fn"](t["start"], end_date)
                elif t["name"] in ("stock_list", "ths_sector_categories",
                                   "ths_constituent_stocks"):
                    t["fn"]()
                else:
                    t["fn"](**_task_kwargs(t))

        stats = limiter.stats
        log_print(f"[main] Done.  Total API calls: {sum(stats['endpoint_counts'].values())}")
        for ep, n in stats["endpoint_counts"].items():
            log_print(f"  {ep}: {n} calls")

    finally:
        if monitor_thread:
            stop_event.set()
            monitor_thread.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch diemeng data (calendar + financial statements)"
    )
    parser.add_argument("-s", "--stock_codes", nargs="*",
                        help="Stock codes for financial data")
    parser.add_argument("--start", default="2019-01-01",
                        help="Start date (default: 2019-01-01)")
    parser.add_argument("--end", default=date.today().isoformat(),
                        help="End date (default: today)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run all tasks concurrently")
    parser.add_argument("--monitor", action="store_true",
                        help="Log rate-limit stats every 10s")
    parser.add_argument("-w", "--workers", type=int, default=6,
                        help="Parallel workers per task (1-64, default: 6; "
                             "请求速率由限流器 280/min 统一控制)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-quarter checkpoint files")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if output file already exists")
    args = parser.parse_args()

    # 并发上限 64: 限流器(config.py)已修复为严格匀速 280/min 放行
    # (acquire 循环重查 token + 保留部分 token), 高并发 worker 只是排队
    # 等 token, 不会产生请求突刺触发服务器 429。实际请求速率仍由
    # 280/min 上限决定, 提高 worker 数不会超过该吞吐上限。
    workers = max(1, min(64, args.workers))

    run_all(
        stock_codes=args.stock_codes,
        start_date=args.start,
        end_date=args.end,
        parallel=args.parallel,
        monitor=args.monitor,
        workers=workers,
        cleanup=not args.no_cleanup,
        force=args.force,
    )
