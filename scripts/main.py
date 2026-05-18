import argparse
import sys
import threading
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import load_api_key, rate_limiter, log_print, DATA_DIR
from fetch_calendar import fetch_calendar
from fetch_financial import fetch_financial_indicator
from fetch_income import fetch_income
from fetch_balancesheet import fetch_balancesheet
from fetch_cashflow import fetch_cashflow
from fetch_finance import fetch_finance
from fetch_stock_list import fetch_stock_list
from fetch_daily import fetch_daily, fetch_daily_adj
from fetch_limit_up import fetch_limit_up
from fetch_limit_list import fetch_limit_list
from fetch_dragon_tiger import fetch_dragon_tiger
from fetch_main_fund_flow import fetch_main_fund_flow

from fetch_top_list import fetch_top_list
from fetch_cyq_chips import fetch_cyq_chips
from fetch_minute import fetch_history, fetch_min_adj
from fetch_kline import fetch_kline_weekly, fetch_kline_monthly
from fetch_kline import fetch_kline_adj_weekly, fetch_kline_adj_monthly
from fetch_holder_number import fetch_holder_number
from fetch_pledge_stat import fetch_pledge_stat
from fetch_margin_detail import fetch_margin_detail
from fetch_ths_daily import fetch_ths_daily
from fetch_ths_sector_categories import fetch_ths_sector_categories
from fetch_ths_constituent_stocks import fetch_ths_constituent_stocks
from fetch_index_weight import fetch_index_weight

# ── Task registry ──────────────────────────────────────────────────
# Comment out any line to skip that endpoint.
# Each task's final output file is checked: if it already exists the
# task is skipped (pass --force to override).

TASKS = [
    # ════════════════════ 秒级 ════════════════════
    {
        "name": "stock_list",
        "fn": fetch_stock_list,
        "output": f"{DATA_DIR}/stock_list.parquet",
    },
    {
        "name": "calendar",
        "fn": fetch_calendar,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/calendar.parquet",
    },
    {
        "name": "ths_sector_categories",
        "fn": fetch_ths_sector_categories,
        "output": f"{DATA_DIR}/ths_sector_categories.parquet",
    },
    {
        "name": "ths_constituent_stocks",
        "fn": fetch_ths_constituent_stocks,
        "output": f"{DATA_DIR}/ths_constituent_stocks.parquet",
    },
    # ════════════════════ 分钟级 ════════════════════
    {
        "name": "holder_number",
        "fn": fetch_holder_number,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/holder_number.parquet",
    },
    {
        "name": "pledge_stat",
        "fn": fetch_pledge_stat,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/pledge_stat.parquet",
    },
    {
        "name": "financial",
        "fn": fetch_financial_indicator,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/financial_indicator.parquet",
    },
    {
        "name": "income",
        "fn": fetch_income,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/income.parquet",
    },
    {
        "name": "balancesheet",
        "fn": fetch_balancesheet,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/balancesheet.parquet",
    },
    {
        "name": "cashflow",
        "fn": fetch_cashflow,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/cashflow.parquet",
    },
    {
        "name": "limit_list",
        "fn": fetch_limit_list,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/limit_list.parquet",
    },
    {
        "name": "limit_up",
        "fn": fetch_limit_up,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/limit_up.parquet",
    },
    {
        "name": "kline_monthly",
        "fn": fetch_kline_monthly,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/kline_monthly.parquet",
    },
    {
        "name": "kline_adj_monthly",
        "fn": fetch_kline_adj_monthly,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/kline_adj_monthly.parquet",
    },
    # # ════════════════════ 十分钟级 ════════════════════
    {
        "name": "top_list",
        "fn": fetch_top_list,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/top_list.parquet",
    },
    {
        "name": "dragon_tiger",
        "fn": fetch_dragon_tiger,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/dragon_tiger.parquet",
    },
    {
        "name": "kline_weekly",
        "fn": fetch_kline_weekly,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/kline_weekly.parquet",
    },
    {
        "name": "kline_adj_weekly",
        "fn": fetch_kline_adj_weekly,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/kline_adj_weekly.parquet",
    },
    # # ════════════════════ 小时级 ════════════════════
    {
        "name": "finance",
        "fn": fetch_finance,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/finance.parquet",
    },
    {
        "name": "daily",
        "fn": fetch_daily,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/daily.parquet",
    },
    {
        "name": "daily_adj",
        "fn": fetch_daily_adj,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/daily_adj.parquet",
    },
    {
        "name": "main_fund_flow",
        "fn": fetch_main_fund_flow,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/main_fund_flow.parquet",
    },
    {
        "name": "margin_detail",
        "fn": fetch_margin_detail,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/margin_detail.parquet",
    },
    {
        "name": "ths_daily",
        "fn": fetch_ths_daily,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/ths_daily.parquet",
    },
    {
        "name": "index_weight",
        "fn": fetch_index_weight,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/index_weight.parquet",
    },
    # ════════════════════ 天级 (per-stock, 散文件落盘) ════════════════════
    {
        "name": "cyq_chips",
        "fn": fetch_cyq_chips,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/cyq_chips/.done",
    },
    {
        "name": "history",
        "fn": fetch_history,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/history/.done",
    },
    {
        "name": "min_adj",
        "fn": fetch_min_adj,
        "start": "2019-01-01",
        "output": f"{DATA_DIR}/min_adj/.done",
    },
]


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
            log_print(f"[main] {t['name']} -> skip (output exists: {output_path})")
            continue
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
                        fut = pool.submit(
                            t["fn"], t["start"], end_date, None, True, workers, cleanup
                        )
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
                    t["fn"](
                        start_date=t["start"], end_date=end_date,
                        workers=workers, cleanup=cleanup,
                    )

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
                        help="Parallel workers per task (1-8, default: 6)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep per-quarter checkpoint files")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if output file already exists")
    args = parser.parse_args()

    workers = max(1, min(8, args.workers))

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
