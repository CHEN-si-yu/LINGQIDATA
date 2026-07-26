#!/usr/bin/env python3
"""
删除指定 parquet 数据目录/文件中指定日期的全部数据（支持多 CPU 并行）。

用法:
    # 处理所有数据集（默认使用所有 CPU）
    sudo python3 /root/autodl-fs/3.py 2024-01-15

    # 支持多个日期（逗号分隔或列表格式）
    sudo python3 /root/autodl-fs/3.py 20260720,20260721,20260722
    sudo python3 /root/autodl-fs/3.py [20260720,20260721,20260722,20260723,20260724]

    # 仅预览（推荐先执行）
    sudo python3 /root/autodl-fs/3.py 2024-01-15 --dry-run

    # 指定并行进程数
    sudo python3 /root/autodl-fs/3.py 2024-01-15 --workers 16

    # 仅处理指定数据集
    sudo python3 /root/autodl-fs/3.py 2024-01-15 --datasets indicator_1min,daily

    # 列出所有可用数据集
    sudo python3 /root/autodl-fs/3.py --list-datasets
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd


# ============================================================
# 数据集配置
# ============================================================

# 日历文件路径（用于计算上一交易日）
CALENDAR_PATH = "/root/autodl-fs/lingqiData/data/calendar.parquet"

# 目录型数据集（每个目录下有多个 parquet 文件）
# 日期列候选名会按优先级自动检测
DIR_DATASETS = {
    "indicator_1min": "/root/autodl-fs/lingqiData/data/indicator_1min",
    "cyq_chips":      "/root/autodl-fs/lingqiData/data/cyq_chips",
    "history_1min":   "/root/autodl-fs/lingqiData/data/history_1min",
}

# 文件型数据集（单个 parquet 文件）
# date_col: 日期列名
# t_minus_1: True 表示该数据为 T-1 类型，需用上一交易日
FILE_DATASETS = {
    "cyq_perf": {
        "path": "/root/autodl-fs/lingqiData/data/cyq_perf.parquet",
        "date_col": "trade_date",
        "t_minus_1": False,
    },
    "daily_adj": {
        "path": "/root/autodl-fs/lingqiData/data/daily_adj.parquet",
        "date_col": "trade_date",
        "t_minus_1": False,
    },
    "daily": {
        "path": "/root/autodl-fs/lingqiData/data/daily.parquet",
        "date_col": "trade_date",
        "t_minus_1": False,
    },
    "finance": {
        "path": "/root/autodl-fs/lingqiData/data/finance.parquet",
        "date_col": "trade_date",
        "t_minus_1": False,
    },
    "main_fund_flow": {
        "path": "/root/autodl-fs/lingqiData/data/main_fund_flow.parquet",
        "date_col": "trade_date",
        "t_minus_1": False,
    },
    "ths_daily": {
        "path": "/root/autodl-fs/lingqiData/data/ths_daily.parquet",
        "date_col": "trade_date",
        "t_minus_1": False,
    },
    "margin_detail": {
        "path": "/root/autodl-fs/lingqiData/data/margin_detail.parquet",
        "date_col": "trade_date",
        "t_minus_1": True,   # T-1 数据：删除时用上一交易日
    },
}

DATE_COLUMN_CANDIDATES = ["trade_time", "trade_date"]


# ============================================================
# 工具函数
# ============================================================

def parse_date(date_str: str) -> datetime:
    """解析单个日期字符串，支持多种格式。"""
    formats = ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y.%m.%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}，支持的格式: {formats}")


def parse_dates(date_arg: str) -> list[datetime]:
    """
    解析日期参数，支持以下格式:
        - 单个日期: 20260720
        - 逗号分隔: 20260720,20260721,20260722
        - 列表格式: [20260720,20260721,20260722,20260723,20260724]

    Returns:
        解析后的 datetime 列表（已去重排序）
    """
    # 去除首尾空白
    date_arg = date_arg.strip()

    # 处理列表格式: [20260720, 20260721, ...]
    if date_arg.startswith("[") and date_arg.endswith("]"):
        # 去掉方括号，按逗号分割
        inner = date_arg[1:-1].strip()
        if not inner:
            return []
        date_strs = [s.strip().strip("'\"") for s in inner.split(",")]
    else:
        # 按逗号分割（支持逗号分隔和单个日期）
        date_strs = [s.strip() for s in date_arg.split(",")]

    # 解析每个日期
    dates = []
    for ds in date_strs:
        if ds:  # 跳过空字符串
            dates.append(parse_date(ds))

    # 去重并排序
    seen = set()
    unique_dates = []
    for d in dates:
        key = d.strftime("%Y-%m-%d")
        if key not in seen:
            seen.add(key)
            unique_dates.append(d)
    unique_dates.sort()

    return unique_dates


def load_calendar() -> pd.DataFrame:
    """加载交易日历。"""
    return pd.read_parquet(CALENDAR_PATH)


def get_previous_trading_day(date_str: str, calendar: pd.DataFrame) -> str:
    """
    根据交易日历获取指定日期的上一个交易日。

    Args:
        date_str: 日期字符串 "YYYY-MM-DD"
        calendar: 包含 date, is_open 列的 DataFrame

    Returns:
        上一个交易日字符串
    """
    open_days = calendar[calendar["is_open"] == 1]["date"].sort_values().tolist()
    # 确保 date_str 在列表中（可能不是交易日）
    try:
        idx = open_days.index(date_str)
    except ValueError:
        # 如果传入的不是交易日，找到它之前最近的交易日
        idx = len([d for d in open_days if d < date_str]) - 1
    if idx <= 0:
        return open_days[0]
    return open_days[idx - 1]


def list_parquet_files(directory: str) -> list[str]:
    """返回目录下所有 parquet 文件（按文件名排序）。"""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".parquet")
    )


# ============================================================
# 并行 worker 函数（模块级，供 ProcessPoolExecutor 使用）
# ============================================================

def _process_dir_file(args: tuple) -> dict:
    """
    处理目录中的一个 parquet 文件（在子进程中执行）。

    Args:
        args: (filepath, target_date_str, dry_run)

    Returns:
        {"filepath": str, "original": int, "deleted": int, "error": str|None}
    """
    filepath, target_date_str, dry_run = args
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

    try:
        df = pd.read_parquet(filepath)
        original = len(df)

        date_col = None
        for col in DATE_COLUMN_CANDIDATES:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return {"filepath": filepath, "original": original, "deleted": 0,
                    "error": f"未找到日期列，可用: {df.columns.tolist()}"}

        mask = df[date_col].dt.date == target_date.date()
        deleted = int(mask.sum())

        if deleted > 0 and not dry_run:
            df_filtered = df[~mask]
            df_filtered.to_parquet(filepath, index=False)

        return {"filepath": filepath, "original": original, "deleted": deleted, "error": None}

    except Exception as e:
        return {"filepath": filepath, "original": 0, "deleted": 0, "error": str(e)}


def _process_standalone_file(args: tuple) -> dict:
    """
    处理单个独立的 parquet 文件（在子进程中执行）。

    Args:
        args: (filepath, date_col, target_date_str, dry_run, is_t_minus_1, prev_trading_day)

    Returns:
        {"filepath": str, "original": int, "deleted": int, "error": str|None}
    """
    filepath, date_col, target_date_str, dry_run = args

    try:
        df = pd.read_parquet(filepath)
        original = len(df)

        # 日期列可能是 string 或 datetime，统一转为字符串比较
        if pd.api.types.is_string_dtype(df[date_col]):
            # 字符串类型
            mask = df[date_col] == target_date_str
        else:
            # datetime 类型
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            mask = df[date_col].dt.date == target_date.date()

        deleted = int(mask.sum())

        if deleted > 0 and not dry_run:
            df_filtered = df[~mask]
            df_filtered.to_parquet(filepath, index=False)

        return {"filepath": filepath, "original": original, "deleted": deleted, "error": None}

    except Exception as e:
        return {"filepath": filepath, "original": 0, "deleted": 0, "error": str(e)}


# ============================================================
# 处理函数
# ============================================================

def process_directory(dir_path: str, target_date: datetime, dry_run: bool = False,
                      workers: int = None) -> dict:
    """
    并行处理单个目录下所有 parquet 文件。
    """
    parquet_files = list_parquet_files(dir_path)
    if not parquet_files:
        return {"files_total": 0, "files_affected": 0, "rows_total": 0, "rows_deleted": 0}

    if workers is None:
        workers = cpu_count()

    target_date_str = target_date.strftime("%Y-%m-%d")
    tasks = [(f, target_date_str, dry_run) for f in parquet_files]

    total_original = 0
    total_deleted = 0
    files_affected = 0
    completed = 0
    dir_name = os.path.basename(dir_path)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_dir_file, t): t[0] for t in tasks}

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            total_original += result["original"]
            total_deleted += result["deleted"]
            if result["deleted"] > 0:
                files_affected += 1
            if result["error"]:
                fname = os.path.basename(result["filepath"])
                print(f"    ✗ {fname}: {result['error']}")

            if completed % 200 == 0 or completed == len(parquet_files):
                print(f"    [{dir_name}] {completed}/{len(parquet_files)} "
                      f"文件, 已删除 {total_deleted:,} 行")

    return {
        "files_total": len(parquet_files),
        "files_affected": files_affected,
        "rows_total": total_original,
        "rows_deleted": total_deleted,
    }


def process_standalone_file(name: str, config: dict, target_date_str: str,
                            dry_run: bool = False, workers: int = None) -> dict:
    """
    处理单个独立 parquet 文件。

    对于大文件，利用多进程分块处理以加速。
    """
    filepath = config["path"]

    if workers is None:
        workers = cpu_count()

    try:
        df = pd.read_parquet(filepath)
        original = len(df)

        date_col = config["date_col"]

        # 统一字符串比较
        if pd.api.types.is_string_dtype(df[date_col]):
            mask = df[date_col] == target_date_str
        else:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            mask = df[date_col].dt.date == target_date.date()

        deleted = int(mask.sum())

        if deleted > 0:
            if not dry_run:
                df_filtered = df[~mask]
                df_filtered.to_parquet(filepath, index=False)

        return {
            "files_total": 1,
            "files_affected": 1 if deleted > 0 else 0,
            "rows_total": original,
            "rows_deleted": deleted,
            "error": None,
        }

    except Exception as e:
        return {
            "files_total": 1,
            "files_affected": 0,
            "rows_total": 0,
            "rows_deleted": 0,
            "error": str(e),
        }


# ============================================================
# 单日处理逻辑（被 main 中循环调用）
# ============================================================

def process_single_date(target_date_str: str, target_date: datetime,
                        calendar: pd.DataFrame, prev_trading_day: str,
                        selected_dirs: list[str], selected_files: list[str],
                        dry_run: bool, workers: int) -> dict:
    """
    处理单个日期的所有数据集删除操作。

    Returns:
        统计字典
    """
    import time

    grand_total_original = 0
    grand_total_deleted = 0
    grand_files_affected = 0
    grand_files_total = 0

    # ---- 处理目录型数据集 ----
    for name in selected_dirs:
        dir_path = DIR_DATASETS[name]

        if not os.path.isdir(dir_path):
            print(f"  ✗ 跳过 {name}: 目录不存在 ({dir_path})")
            continue

        n_files = len([f for f in os.listdir(dir_path) if f.endswith(".parquet")])
        print(f"  ▶ [目录] {name} ({n_files} 个文件)...")

        t0 = time.time()
        result = process_directory(dir_path, target_date, dry_run=dry_run,
                                   workers=workers)
        elapsed = time.time() - t0

        grand_total_original += result["rows_total"]
        grand_total_deleted += result["rows_deleted"]
        grand_files_affected += result["files_affected"]
        grand_files_total += result["files_total"]

        print(f"    文件: {result['files_total']}, "
              f"受影响: {result['files_affected']}, "
              f"行数: {result['rows_total']:,}, "
              f"删除: {result['rows_deleted']:,}, "
              f"耗时: {elapsed:.1f}s")
        print()

    # ---- 处理文件型数据集 ----
    for name in selected_files:
        config = FILE_DATASETS[name]
        filepath = config["path"]

        if not os.path.isfile(filepath):
            print(f"  ✗ 跳过 {name}: 文件不存在 ({filepath})")
            continue

        # T-1 数据使用上一交易日
        actual_date_str = prev_trading_day if config["t_minus_1"] else target_date_str
        t_label = f" [T-1 → {actual_date_str}]" if config["t_minus_1"] else ""

        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  ▶ [文件] {name}{t_label} ({size_mb:.0f} MB)...")

        t0 = time.time()
        result = process_standalone_file(name, config, actual_date_str,
                                         dry_run=dry_run, workers=workers)
        elapsed = time.time() - t0

        if result.get("error"):
            print(f"    ✗ 错误: {result['error']}")
        else:
            grand_total_original += result["rows_total"]
            grand_total_deleted += result["rows_deleted"]
            grand_files_affected += result["files_affected"]
            grand_files_total += result["files_total"]

            print(f"    文件: {result['files_total']}, "
                  f"受影响: {result['files_affected']}, "
                  f"行数: {result['rows_total']:,}, "
                  f"删除: {result['rows_deleted']:,}, "
                  f"耗时: {elapsed:.1f}s")
        print()

    return {
        "rows_total": grand_total_original,
        "rows_deleted": grand_total_deleted,
        "files_affected": grand_files_affected,
        "files_total": grand_files_total,
    }


# ============================================================
# 主入口
# ============================================================

def main():
    default_workers = cpu_count() // 2

    # 聚合所有数据集名称
    all_dataset_names = list(DIR_DATASETS.keys()) + list(FILE_DATASETS.keys())

    parser = argparse.ArgumentParser(
        description="删除 parquet 数据中指定日期的所有数据（多 CPU 并行）\n"
                    "包含目录型数据集和独立 parquet 文件。\n"
                    "margin_detail 为 T-1 数据，自动使用上一交易日。\n\n"
                    "支持单个日期或批量多个日期:\n"
                    "  单个:   20260720\n"
                    "  逗号分隔: 20260720,20260721,20260722\n"
                    "  列表格式: [20260720,20260721,20260722,20260723,20260724]"
    )
    parser.add_argument(
        "date", type=str, nargs="?",
        help="要删除的日期，支持单个或逗号/列表分隔的多个日期。\n"
             "格式: 2024-01-15, 20240115, 2024/01/15",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览将要删除的数据，不实际执行删除",
    )
    parser.add_argument(
        "--datasets", "-d", type=str, default=None,
        help="要处理的数据集（逗号分隔），默认处理所有。\n"
             "目录型: " + ", ".join(DIR_DATASETS.keys()) + "\n"
             "文件型: " + ", ".join(FILE_DATASETS.keys()),
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=default_workers,
        help=f"并行进程数（默认: {default_workers}）",
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="列出所有可用数据集及其信息",
    )
    args = parser.parse_args()

    # ---- 列出数据集 ----
    if args.list_datasets:
        print("=== 目录型数据集（每个含多个 parquet 文件）===")
        for name, path in DIR_DATASETS.items():
            if os.path.isdir(path):
                n = len([f for f in os.listdir(path) if f.endswith(".parquet")])
                print(f"  {name}: {path}/  ({n} 个文件)")
            else:
                print(f"  {name}: {path}/  (不存在)")
        print()
        print("=== 文件型数据集（单个 parquet 文件）===")
        for name, cfg in FILE_DATASETS.items():
            exists = os.path.isfile(cfg["path"])
            size = ""
            if exists:
                size_mb = os.path.getsize(cfg["path"]) / 1024 / 1024
                size = f", {size_mb:.0f} MB"
            t_flag = " [T-1]" if cfg["t_minus_1"] else ""
            print(f"  {name}{t_flag}: {cfg['path']}  "
                  f"({'存在' if exists else '不存在'}{size})")
        return

    # ---- 必须提供日期 ----
    if args.date is None:
        parser.print_help()
        print("\n错误: 必须提供日期参数")
        sys.exit(1)

    # ---- 解析日期（支持多日期） ----
    target_dates = parse_dates(args.date)
    if not target_dates:
        print("错误: 未能解析任何有效日期")
        sys.exit(1)

    target_date_strs = [d.strftime("%Y-%m-%d") for d in target_dates]

    print(f"✓ 解析到 {len(target_dates)} 个日期: {target_date_strs}")

    # ---- 加载交易日历 ----
    try:
        calendar = load_calendar()
    except Exception as e:
        print(f"错误: 无法加载交易日历 {CALENDAR_PATH}: {e}")
        sys.exit(1)

    # ---- 确定要处理的数据集 ----
    if args.datasets:
        selected = [n.strip() for n in args.datasets.split(",")]
        invalid = [n for n in selected if n not in all_dataset_names]
        if invalid:
            print(f"错误: 未知数据集: {invalid}")
            print(f"可用: {all_dataset_names}")
            sys.exit(1)
    else:
        selected = all_dataset_names

    selected_dirs = [n for n in selected if n in DIR_DATASETS]
    selected_files = [n for n in selected if n in FILE_DATASETS]

    print(f"{'='*60}")
    print(f"目标日期（{len(target_dates)} 个）: {target_date_strs}")
    print(f"并行进程数: {args.workers}")
    if args.dry_run:
        print("⚠ DRY RUN 模式 — 不会实际修改文件")
    print(f"目录型数据集: {selected_dirs if selected_dirs else '无'}")
    print(f"文件型数据集: {selected_files if selected_files else '无'}")
    print(f"{'='*60}\n")

    import time

    # 全局累计统计
    global_rows_total = 0
    global_rows_deleted = 0
    global_files_affected = 0
    global_files_total = 0

    # ---- 逐日处理 ----
    for i, (target_date_str, target_date) in enumerate(zip(target_date_strs, target_dates)):
        # 计算该日期的上一交易日
        prev_trading_day = get_previous_trading_day(target_date_str, calendar)

        if len(target_dates) > 1:
            print(f"{'─'*60}")
            print(f"📅 [{i+1}/{len(target_dates)}] 处理日期: {target_date_str}")
            if any(FILE_DATASETS[n]["t_minus_1"] for n in selected_files):
                print(f"   上一交易日 (T-1): {prev_trading_day}")
            print(f"{'─'*60}\n")

        t_start = time.time()
        result = process_single_date(
            target_date_str=target_date_str,
            target_date=target_date,
            calendar=calendar,
            prev_trading_day=prev_trading_day,
            selected_dirs=selected_dirs,
            selected_files=selected_files,
            dry_run=args.dry_run,
            workers=args.workers,
        )
        elapsed = time.time() - t_start

        global_rows_total += result["rows_total"]
        global_rows_deleted += result["rows_deleted"]
        global_files_affected += result["files_affected"]
        global_files_total += result["files_total"]

        if len(target_dates) > 1:
            print(f"  [{target_date_str}] 扫描: {result['files_total']} 文件, "
                  f"删除: {result['rows_deleted']:,} 行, "
                  f"耗时: {elapsed:.1f}s\n")

    # ---- 总结 ----
    mode = "DRY RUN - " if args.dry_run else ""
    print(f"{'='*60}")
    print(f"总计 ({mode}{len(target_dates)} 个日期):")
    print(f"  日期: {target_date_strs}")
    if any(FILE_DATASETS[n]["t_minus_1"] for n in selected_files):
        print(f"  margin_detail 为 T-1 数据，使用各日期的上一交易日")
    print(f"  处理数据集数: {len(selected)}")
    print(f"  扫描文件数: {global_files_total}")
    print(f"  受影响文件数: {global_files_affected}")
    print(f"  总行数: {global_rows_total:,}")
    print(f"  删除行数: {global_rows_deleted:,}")
    if global_rows_total > 0:
        print(f"  剩余行数: {global_rows_total - global_rows_deleted:,}")
    print(f"{'='*60}")

    if args.dry_run and global_rows_deleted > 0:
        print("\n提示: 去掉 --dry-run 参数重新运行以实际执行删除。")


if __name__ == "__main__":
    main()
