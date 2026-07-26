#!/usr/bin/env python3
"""
删除 /root/autodl-fs/lingqiData/featureengineering/data 下所有 .fea 文件中的指定日期。

用法:
    python delete_factors.py 20200102 20200103 20200106          # 删除多个日期
    python delete_factors.py --file dates.txt                     # 从文件读取日期列表
    python delete_factors.py --dry-run 20200102                   # 仅预览，不实际删除
    python delete_factors.py --workers 64 20200102 20200103       # 指定并行进程数
"""

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import pandas as pd

# ============================================================
# 配置
# ============================================================
DATA_ROOT = "/root/autodl-fs/lingqiData/featureengineering/data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def find_all_fea_files(data_root: str) -> list:
    """递归查找所有 .fea 文件，返回绝对路径列表。"""
    fea_files = []
    for dirpath, _, filenames in os.walk(data_root):
        for fname in filenames:
            if fname.endswith(".fea"):
                fea_files.append(os.path.join(dirpath, fname))
    return sorted(fea_files)


def load_dates_from_file(filepath: str) -> list:
    """从文本文件中读取日期列表（每行一个日期或空格/逗号分隔）。"""
    dates = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 支持空格/逗号/换行分隔
            for token in line.replace(",", " ").split():
                dates.append(token.strip())
    return dates


def validate_date_format(date_str: str) -> bool:
    """验证日期格式为 YYYYMMDD。"""
    if len(date_str) != 8:
        return False
    if not date_str.isdigit():
        return False
    try:
        y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
        if not (2020 <= y <= 2030):
            return False
        if not (1 <= m <= 12):
            return False
        if not (1 <= d <= 31):
            return False
        return True
    except ValueError:
        return False


# ============================================================
# 核心逻辑：单文件处理
# ============================================================

def process_single_file(args: tuple) -> dict:
    """
    处理单个 .fea 文件：读取 -> 删除指定日期 -> 写回。

    参数:
        args: (file_path, dates_to_delete, dry_run)

    返回:
        dict: 处理结果统计
    """
    file_path, dates_to_delete, dry_run = args
    fname = os.path.basename(file_path)

    try:
        # 读取 feather 文件
        df = pd.read_feather(file_path)
        rows_before = len(df)

        # 查找实际存在的日期
        dates_in_file = set(df.index)
        dates_to_drop = [d for d in dates_to_delete if d in dates_in_file]

        if not dates_to_drop:
            return {
                "file": fname,
                "status": "skipped",
                "msg": "没有匹配的日期",
                "rows_before": rows_before,
                "rows_after": rows_before,
                "deleted": 0,
            }

        df = df.drop(index=dates_to_drop)
        rows_after = len(df)
        deleted = rows_before - rows_after

        if not dry_run:
            df.to_feather(file_path)

        return {
            "file": fname,
            "status": "ok",
            "msg": f"删除 {deleted} 行",
            "rows_before": rows_before,
            "rows_after": rows_after,
            "deleted": deleted,
        }

    except Exception as e:
        return {
            "file": fname,
            "status": "error",
            "msg": str(e),
            "rows_before": 0,
            "rows_after": 0,
            "deleted": 0,
        }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="批量删除 .fea 文件中的指定日期",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python delete_factors.py 20200102 20200103
    python delete_factors.py --file dates.txt
    python delete_factors.py --dry-run 20200102 20200103
    python delete_factors.py --workers 64 20200102
        """,
    )
    parser.add_argument(
        "dates",
        nargs="*",
        help="要删除的日期，格式 YYYYMMDD，可多个",
    )
    parser.add_argument(
        "--file", "-f",
        default=None,
        help="从文件读取日期列表（每行一个，支持 # 注释）",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="仅预览，不实际修改文件",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help=f"并行进程数（默认: CPU 核数的一半，当前系统 {cpu_count()} 核）",
    )
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help=f"数据目录（默认: {DATA_ROOT}）",
    )

    args = parser.parse_args()

    # ---- 收集日期 ----
    dates_to_delete = list(args.dates)

    if args.file:
        if not os.path.isfile(args.file):
            logger.error(f"日期文件不存在: {args.file}")
            sys.exit(1)
        dates_to_delete.extend(load_dates_from_file(args.file))

    # 去重并排序
    dates_to_delete = sorted(set(dates_to_delete))

    if not dates_to_delete:
        logger.error("未提供任何日期。请通过命令行参数或 --file 指定。")
        sys.exit(1)

    # 验证日期格式
    invalid = [d for d in dates_to_delete if not validate_date_format(d)]
    if invalid:
        logger.error(f"日期格式无效 (需为 YYYYMMDD): {invalid}")
        sys.exit(1)

    logger.info(f"待删除日期 ({len(dates_to_delete)}): {dates_to_delete}")

    # ---- 查找文件 ----
    if not os.path.isdir(args.data_root):
        logger.error(f"数据目录不存在: {args.data_root}")
        sys.exit(1)

    fea_files = find_all_fea_files(args.data_root)
    logger.info(f"找到 {len(fea_files)} 个 .fea 文件")

    if not fea_files:
        logger.warning("没有找到任何 .fea 文件，退出。")
        sys.exit(0)

    # ---- 并行处理 ----
    workers = args.workers
    if workers is None:
        # 默认使用一半核心，避免 IO 过载
        workers = max(1, cpu_count() // 2)
    workers = min(workers, len(fea_files))

    if args.dry_run:
        logger.info(f"模式: DRY-RUN（不会修改文件）")
    logger.info(f"使用 {workers} 个进程并行处理")

    task_args = [(f, dates_to_delete, args.dry_run) for f in fea_files]

    t_start = time.time()

    with Pool(processes=workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(process_single_file, task_args, chunksize=10)):
            results.append(result)
            if result["status"] == "ok":
                logger.info(
                    f"[{i + 1}/{len(fea_files)}] {result['file']}: "
                    f"删除 {result['deleted']} 行 "
                    f"({result['rows_before']} -> {result['rows_after']})"
                )
            elif result["status"] == "error":
                logger.error(f"[{i + 1}/{len(fea_files)}] {result['file']}: 错误 - {result['msg']}")

    elapsed = time.time() - t_start

    # ---- 汇总 ----
    ok_count = sum(1 for r in results if r["status"] == "ok")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")
    total_deleted_rows = sum(r["deleted"] for r in results)

    logger.info("=" * 60)
    logger.info(f"完成! 耗时: {elapsed:.1f} 秒")
    logger.info(f"  成功: {ok_count} 个文件")
    logger.info(f"  跳过: {skipped_count} 个文件（无匹配日期）")
    logger.info(f"  失败: {error_count} 个文件")
    logger.info(f"  总共删除: {total_deleted_rows} 行")
    if args.dry_run:
        logger.warning(">>> 这是 DRY-RUN 模式，未实际修改任何文件 <<<")

    if error_count > 0:
        logger.warning("以下文件处理出错:")
        for r in results:
            if r["status"] == "error":
                logger.warning(f"  {r['file']}: {r['msg']}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
