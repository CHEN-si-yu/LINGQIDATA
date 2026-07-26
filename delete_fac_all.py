#!/usr/bin/env python3
"""
delete_fac_all.py — 删除 fac_all.fea 中指定日期的数据。

用法:
    python delete_fac_all.py <日期>                    从文件中删除指定日期，结果写回原文件
    python delete_fac_all.py <日期> --dry-run          仅打印匹配行数，不修改文件
    python delete_fac_all.py <日期> --output out.fea   删除后输出到指定文件（原文件不变）

日期格式示例: 20240101 或 2024-01-01（脚本会自动统一为 YYYYMMDD 比对）

数据格式假设:
    fac_all.fea 为 CSV 文件，第一列（或名为 trade_date/date 的列）是日期。
    若实际列名不同，可通过 --date-col 参数指定。
"""

import argparse
import csv
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
DEFAULT_INPUT = "/root/autodl-fs/lingqiData/trainingdata/fac_all.fea"

# 尝试引入 pandas（更快更鲁棒），没有则回退到 csv 模块
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def normalize_date(raw: str) -> str:
    """将用户输入的日期统一为 YYYYMMDD（纯数字，无分隔符）。"""
    cleaned = raw.strip().replace("-", "").replace("/", "").replace(".", "")
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise ValueError(f"无法解析日期: '{raw}'，期望格式如 20240101 或 2024-01-01")
    return cleaned


def sniff_delimiter(filepath: str) -> str:
    """尝试检测 CSV 分隔符（逗号或制表符）。"""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        sample = f.readline()
    if sample.count("\t") > sample.count(","):
        return "\t"
    return ","


# ---------------------------------------------------------------------------
# pandas 实现（推荐路径）
# ---------------------------------------------------------------------------
def delete_with_pandas(
    input_path: str,
    target_date: str,
    output_path: str,
    date_col: str,
    dry_run: bool,
) -> None:
    """使用 pandas 读取、过滤、写出。"""
    try:
        df = pd.read_csv(input_path, dtype=str)
    except Exception:
        # 可能是无表头文件 —— 尝试不给列名，手动赋予
        df = pd.read_csv(input_path, header=None, dtype=str)
        if date_col.isdigit():
            date_col = int(date_col)
        print(f"[信息] 文件无表头，共 {df.shape[1]} 列，将第 {date_col} 列视为日期列。")

    print(f"[信息] 读取 {len(df)} 行，列: {list(df.columns)[:10]}...")

    # ---- 定位日期列 ----
    if isinstance(date_col, str) and date_col in df.columns:
        col = date_col
    elif isinstance(date_col, int) and date_col < len(df.columns):
        col = df.columns[date_col]
    else:
        # 自动猜测：找列名包含 trade_date / date / time 的
        candidates = [
            c
            for c in df.columns
            if any(k in str(c).lower() for k in ("trade_date", "date", "time"))
        ]
        if candidates:
            col = candidates[0]
            print(f"[信息] 自动识别日期列: '{col}'")
        else:
            print(f"[错误] 找不到日期列。可用列: {list(df.columns)[:15]}")
            print(f"       请用 --date-col 指定列名或列索引（从 0 开始）。")
            sys.exit(1)

    # ---- 统一日期格式 ----
    date_series = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace("-", "")
        .str.replace("/", "")
        .str.replace(".", "")
    )
    # 处理可能的时间戳 —— 只取前 8 位
    date_series = date_series.str[:8]

    mask = date_series == target_date
    match_count = mask.sum()

    print(
        f"[信息] 日期列 '{col}' 中匹配 '{target_date}' 的共 {match_count} 行。"
    )

    if match_count == 0:
        print("[警告] 未找到匹配日期的数据，文件将保持不变。")
        if dry_run:
            return
        if output_path != input_path:
            shutil.copy2(input_path, output_path)
            print(f"[完成] 数据已复制到 {output_path}（未做删减）")
        return

    if dry_run:
        print(
            f"[DRY-RUN] 将删除 {match_count} 行，剩余 {len(df) - match_count} 行。"
        )
        preview = df[mask].head(5)
        print("[DRY-RUN] 将被删除的行预览:")
        print(preview.to_string(max_colwidth=40))
        return

    # ---- 删除并写出 ----
    filtered = df[~mask]
    print(f"[信息] 删除后剩余 {len(filtered)} 行。")

    filtered.to_csv(output_path, index=False)
    print(f"[完成] 已写入 {output_path}")


# ---------------------------------------------------------------------------
# csv 模块实现（纯标准库回退）
# ---------------------------------------------------------------------------
def delete_with_csv(
    input_path: str,
    target_date: str,
    output_path: str,
    date_col: str,
    dry_run: bool,
) -> None:
    """使用标准库 csv 逐行处理（适合超大文件，内存友好）。"""
    delimiter = sniff_delimiter(input_path)

    with open(input_path, "r", encoding="utf-8-sig") as f_in:
        reader = csv.reader(f_in, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        print("[错误] 文件为空。")
        sys.exit(1)

    header = rows[0]
    print(f"[信息] 读取 {len(rows) - 1} 行数据，{len(header)} 列。")
    print(f"[信息] 列名: {header[:10]}...")

    # ---- 定位日期列索引 ----
    if isinstance(date_col, int):
        col_idx = date_col
    elif date_col.isdigit():
        col_idx = int(date_col)
    elif date_col in header:
        col_idx = header.index(date_col)
    else:
        # 自动猜测
        candidates = [
            i
            for i, h in enumerate(header)
            if any(k in h.lower() for k in ("trade_date", "date", "time"))
        ]
        if candidates:
            col_idx = candidates[0]
            print(f"[信息] 自动识别日期列: '{header[col_idx]}' (索引 {col_idx})")
        else:
            print(f"[错误] 找不到日期列。可用列: {header[:15]}")
            print(f"       请用 --date-col 指定列名或列索引（从 0 开始）。")
            sys.exit(1)

    # ---- 过滤 ----
    keep = [header]
    deleted = []

    for row in rows[1:]:
        if col_idx >= len(row):
            keep.append(row)
            continue
        raw_date = (
            row[col_idx]
            .strip()
            .replace("-", "")
            .replace("/", "")
            .replace(".", "")[:8]
        )
        if raw_date == target_date:
            deleted.append(row)
        else:
            keep.append(row)

    print(
        f"[信息] 日期列 '{header[col_idx]}' 中匹配 '{target_date}' 的共 {len(deleted)} 行。"
    )

    if len(deleted) == 0:
        print("[警告] 未找到匹配日期的数据，文件将保持不变。")
        if dry_run:
            return
        if output_path != input_path:
            shutil.copy2(input_path, output_path)
            print(f"[完成] 数据已复制到 {output_path}（未做删减）")
        return

    if dry_run:
        print(f"[DRY-RUN] 将删除 {len(deleted)} 行，剩余 {len(keep) - 1} 行。")
        print("[DRY-RUN] 将被删除的前 5 行:")
        for row in deleted[:5]:
            print(f"  {row}")
        return

    # ---- 写出 ----
    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter=delimiter)
        writer.writerows(keep)

    print(f"[完成] 已写入 {output_path}，保留 {len(keep) - 1} 行。")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="删除 fac_all.fea 中指定日期的数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python delete_fac_all.py 20240101
  python delete_fac_all.py 2024-01-01 --dry-run
  python delete_fac_all.py 20240101 -o /tmp/fac_clean.fea""",
    )
    parser.add_argument(
        "date",
        help="要删除的日期，如 20240101 或 2024-01-01",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT,
        help=f"输入文件路径（默认: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出文件路径（默认: 覆盖输入文件）",
    )
    parser.add_argument(
        "--date-col",
        default=None,
        help="日期列名或列索引（从 0 开始）。不指定则自动识别。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不实际写入。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="覆盖时不创建 .bak 备份。",
    )

    args = parser.parse_args()

    # ---- 校验输入 ----
    if not os.path.exists(args.input):
        print(f"[错误] 输入文件不存在: {args.input}")
        print(f"       请确认 autodl-fs 已挂载且路径正确。")
        sys.exit(1)

    target_date = normalize_date(args.date)
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else input_path

    # 解析 date_col
    date_col = args.date_col
    if date_col is not None and date_col.isdigit():
        date_col = int(date_col)

    # ---- 备份 ----
    if output_path == input_path and not args.dry_run and not args.no_backup:
        backup = input_path + ".bak"
        print(f"[信息] 创建备份: {backup}")
        shutil.copy2(input_path, backup)

    # ---- 执行 ----
    if HAS_PANDAS:
        delete_with_pandas(
            input_path, target_date, output_path, date_col, args.dry_run
        )
    else:
        delete_with_csv(
            input_path, target_date, output_path, date_col, args.dry_run
        )


if __name__ == "__main__":
    main()
