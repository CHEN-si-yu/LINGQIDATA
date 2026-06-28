#!/usr/bin/env python3
"""
因子全量评估与回测脚本

对 data/factors/ 下所有因子，按 data/targets/ 下所有 target 标签
进行截面 IC 评估，生成评估报告。

特性:
- 多进程并行 (根据 CPU 核数自动配置)
- 实时 Rich 仪表盘进度可视化
- 断点续跑 (自动跳过已完成的因子)
- 内存批量处理 (每批处理 N 个因子，避免 OOM)
- 输出: CSV 详细结果 + Markdown 报告

用法::

    # 全量评估 (使用默认 target)
    sudo python evaluate_all_factors.py

    # 指定 target 和 worker 数
    sudo python evaluate_all_factors.py --targets label_ret_5d,label_ret_10d --workers 32

    # 只评估特定类别因子
    sudo python evaluate_all_factors.py --category price,valuation,quality

    # 单因子快速排查
    sudo python evaluate_all_factors.py --factor-name adx_14

    # 生成因子筛选 (按季度)
    sudo python evaluate_all_factors.py --screen

参考: fac_filt.py 的筛选逻辑
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── 将 src/ 加入 sys.path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from featureengineering.factor_evaluation import (
    FactorEvalResult,
    FactorScreener,
    evaluate_factors,
    evaluate_single_factor,
    evaluate_by_season,
    generate_eval_report,
    save_eval_report,
    save_eval_csv,
    find_redundant_factors,
    quick_check_factors,
)
from featureengineering.eval_dashboard import EvalDashboard

logger = logging.getLogger("evaluate_all_factors")

# ── Paths ────────────────────────────────────────────────────────────────────

FACTOR_DIR = PROJECT_ROOT / "data" / "factors"
TARGET_DIR = PROJECT_ROOT / "data" / "targets"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
REPORT_DIR = PROJECT_ROOT / "reports"

# ── Helpers ──────────────────────────────────────────────────────────────────


def list_factor_files() -> list[str]:
    """列出所有可用的因子名称 (不含 .fea 后缀)。"""
    if not FACTOR_DIR.exists():
        return []
    names = sorted(
        p.stem for p in FACTOR_DIR.glob("*.fea")
        if not p.name.startswith(".")
    )
    return names


def list_target_files() -> list[str]:
    """列出所有可用的 target 名称 (不含 .fea 后缀)。"""
    if not TARGET_DIR.exists():
        return []
    return sorted(
        p.stem for p in TARGET_DIR.glob("*.fea")
        if p.name.startswith("label_")
    )


def get_factor_metadata(factor_name: str) -> dict:
    """读取因子的 manifest JSON 元数据。"""
    manifest_path = MANIFEST_DIR / f"{factor_name}.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except Exception:
        return {}


def filter_by_category(factor_names: list[str], categories: set[str]) -> list[str]:
    """按因子类别筛选."""
    result = []
    for name in factor_names:
        meta = get_factor_metadata(name)
        cat = meta.get("category", "other")
        if cat in categories:
            result.append(name)
    return result


# ── Batch evaluation with progress ───────────────────────────────────────────


def evaluate_factors_batched(
    factor_names: list[str],
    target_names: list[str],
    max_workers: int = 16,
    batch_size: int = 100,
    use_dashboard: bool = True,
    checkpoint_file: Path | None = None,
) -> pd.DataFrame:
    """分批评估因子，支持进度可视化和断点续跑。

    Parameters
    ----------
    factor_names : list[str]
    target_names : list[str]
    max_workers : int
    batch_size : int
        每批处理的因子数。较小值减少内存压力。
    use_dashboard : bool
    checkpoint_file : Path or None
        中间结果保存路径 (用于断点续跑)。

    Returns
    -------
    pd.DataFrame
    """
    total_combinations = len(factor_names) * len(target_names)

    # Check for checkpoint
    results_so_far: list[dict] = []
    completed_factors: set[str] = set()
    if checkpoint_file and checkpoint_file.exists():
        try:
            existing = pd.read_csv(checkpoint_file)
            results_so_far = existing.to_dict("records")
            completed_factors = set(existing["factor_name"].unique())
            logger.info(
                "Resuming from checkpoint: %d results, %d factors done",
                len(results_so_far), len(completed_factors),
            )
        except Exception:
            pass

    pending = [f for f in factor_names if f not in completed_factors]
    logger.info(
        "Factors: %d total, %d done, %d pending", 
        len(factor_names), len(completed_factors), len(pending),
    )

    dashboard = EvalDashboard(
        total_combinations=total_combinations,
        max_workers=max_workers,
    ) if use_dashboard else None

    try:
        if dashboard:
            dashboard.__enter__()

        # Process in batches
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(pending) + batch_size - 1) // batch_size

            print(f"\n  Batch {batch_num}/{total_batches}: "
                  f"evaluating {len(batch)} factors × {len(target_names)} targets "
                  f"({len(batch) * len(target_names)} combinations)")

            df = evaluate_factors(
                factor_names=batch,
                target_names=target_names,
                max_workers=max_workers,
                progress=True,
            )

            if not df.empty:
                results_so_far.extend(df.to_dict("records"))

            # Save checkpoint
            if checkpoint_file:
                ckpt_df = pd.DataFrame(results_so_far)
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                ckpt_df.to_csv(checkpoint_file, index=False)

            # Update dashboard
            if dashboard:
                for _, row in df.iterrows():
                    icir = row.get("rank_icir", None)
                    ok = not (pd.isna(icir) if icir is not None else True)
                    dashboard.mark_done(
                        f"{row['factor_name']}×{row['target_name']}",
                        icir=icir,
                        ok=ok,
                    )
                dashboard.refresh()

    finally:
        if dashboard:
            dashboard.__exit__(None, None, None)

    result_df = pd.DataFrame(results_so_far)

    # Print summary
    if dashboard:
        dashboard.print_summary()

    if not result_df.empty:
        print(f"\n  Evaluation complete: {len(result_df)} combinations")
        if "abs_rank_icir" in result_df.columns:
            good = (result_df["abs_rank_icir"] >= 0.5).sum()
            warn = ((result_df["abs_rank_icir"] >= 0.1) & (result_df["abs_rank_icir"] < 0.5)).sum()
            bad = (result_df["abs_rank_icir"] < 0.1).sum()
            print(f"  |ICIR| ≥ 0.5: {good}  |  0.1 ≤ |ICIR| < 0.5: {warn}  |  < 0.1: {bad}")

    return result_df


# ── Season-based screening ────────────────────────────────────────────────────


def run_season_screening(
    factor_names: list[str],
    target_name: str = "label_ret_5d",
    all_dates: list[str] | None = None,
) -> dict[str, list[str]]:
    """按季度筛选因子 (复刻 fac_filt.py 的 sel_fac_season)。"""
    season_list = [
        "2023q1", "2023q2", "2023q3", "2023q4",
        "2024q1", "2024q2", "2024q3", "2024q4",
    ]

    # Get all dates from one factor file
    if all_dates is None:
        sample = factor_names[0] if factor_names else None
        if sample:
            fp = FACTOR_DIR / f"{sample}.fea"
            if fp.exists():
                f = pd.read_feather(fp)
                for idx_col in ["Date", "date", "index"]:
                    if idx_col in f.columns:
                        f = f.set_index(idx_col)
                        break
                all_dates = sorted(f.index.unique().tolist())

    if not all_dates:
        logger.error("Cannot determine date list")
        return {}

    results_by_season = evaluate_by_season(
        factor_names=factor_names,
        target_name=target_name,
        season_list=season_list,
        all_dates=[str(d) for d in all_dates],
    )

    screener = FactorScreener(fac_num_target=1400)
    sel_fac_season: dict[str, list[str]] = {}

    for season, df in results_by_season.items():
        results = [
            FactorEvalResult(
                factor_name=row["factor_name"],
                target_name=target_name,
                rank_icir=row.get("rank_icir", float("nan")),
                rank_ic_mean=row.get("rank_ic_mean", float("nan")),
                long_short_spread_amt=row.get("long_short_spread_amt", float("nan")),
                long_ret_head_amt=row.get("long_ret_head_amt", float("nan")),
                short_ret_tail_amt=row.get("short_ret_tail_amt", float("nan")),
                coverage_ratio=row.get("coverage_ratio", float("nan")),
                n_dates=int(row.get("n_dates", 0)),
            )
            for _, row in df.iterrows()
        ]
        sel_fac_season[season] = screener.screen(results)

    return sel_fac_season


# ── Main ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Factor Evaluation & Backtesting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--targets", type=str, default=None,
        help="Comma-separated target names (default: all label_ret_*).",
    )
    p.add_argument(
        "--workers", "-j", type=int, default=None,
        help="Number of parallel workers (default: auto-detect, 4-32).",
    )
    p.add_argument(
        "--batch-size", type=int, default=100,
        help="Factors per batch for memory control (default: 100).",
    )
    p.add_argument(
        "--category", type=str, default=None,
        help="Only evaluate factors of given category (comma-separated).",
    )
    p.add_argument(
        "--factor-name", "-f", type=str, default=None,
        help="Evaluate a single factor and print detailed results.",
    )
    p.add_argument(
        "--top-n", type=int, default=50,
        help="Top N factors to show in report (default: 50).",
    )
    p.add_argument(
        "--screen", action="store_true",
        help="Run season-based factor screening (fac_filt.py style).",
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable Rich live dashboard.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=REPORT_DIR,
        help="Output directory for reports (default: reports/).",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint CSV path for resuming interrupted runs.",
    )
    p.add_argument(
        "--quick-check", action="store_true",
        help="Run quick pre-check on all factors for NaN/constant/sparse issues.",
    )
    p.add_argument(
        "--output-screened-factors", action="store_true",
        help="After evaluation, run screening and write screened factor list as JSON.",
    )
    p.add_argument(
        "--corr-threshold", type=float, default=0.85,
        help="Correlation threshold for redundancy removal (default: 0.85).",
    )
    p.add_argument(
        "--screen-target", type=str, default="label_ret_20d",
        help="Target to use for screening output (default: label_ret_20d).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    # Configure
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Auto-detect workers ──
    cpu_count = os.cpu_count() or 8
    if args.workers is None:
        args.workers = max(4, min(cpu_count - 2, 32))

    print(f"{'='*64}")
    print(f"  Factor Evaluation & Backtesting")
    print(f"  Workers: {args.workers}  |  Batch size: {args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*64}")

    # ── Factor / Target selection ──
    factor_names = list_factor_files()
    print(f"\n  Available factors:  {len(factor_names)}")
    print(f"  Available targets:  {len(list_target_files())}")

    if args.category:
        categories = {c.strip() for c in args.category.split(",")}
        factor_names = filter_by_category(factor_names, categories)
        print(f"  After category filter ({args.category}): {len(factor_names)}")

    if args.factor_name:
        factor_names = [args.factor_name]
        print(f"  Single factor mode: {args.factor_name}")

    if not factor_names:
        print("  No factors to evaluate!")
        return 1

    target_names = list_target_files()
    if args.targets:
        target_names = [t.strip() for t in args.targets.split(",")]
        # Validate
        available = list_target_files()
        target_names = [t for t in target_names if t in available]
        if not target_names:
            print(f"  Invalid targets: {args.targets}")
            return 1

    # ── Quick check mode ──
    if args.quick_check:
        print(f"\n{'='*64}")
        print(f"  Quick Check: Scanning {len(factor_names)} factors for issues...")
        print(f"{'='*64}")
        qc_df = quick_check_factors(factor_names)
        summary = qc_df["status"].value_counts()
        print(f"\n  Status distribution:")
        for st, cnt in summary.items():
            print(f"    {st:<20} {cnt:>5}")
        bad = qc_df[qc_df["status"] != "ok"]
        if not bad.empty:
            print(f"\n  Factors needing attention ({len(bad)}):")
            for _, row in bad.iterrows():
                print(f"    {row['factor_name']:<45} [{row['status']:<12}] {row['recommendation']}")
        qc_path = args.output_dir / "quick_check.csv"
        qc_df.to_csv(qc_path, index=False)
        print(f"\n  Quick check saved: {qc_path}")
        if not args.factor_name and args.quick_check:
            # If only quick-check (no evaluation requested implicitly), return
            # But we still proceed with evaluation by default
            pass

    # ── Checkpoint path ──
    checkpoint = args.checkpoint or (args.output_dir / "eval_checkpoint.csv")

    # ── Run evaluation ──
    t_start = time.perf_counter()

    if args.factor_name and len(factor_names) == 1:
        # Single-factor detailed mode
        print(f"\n  Evaluating single factor: {args.factor_name}")
        for t in target_names:
            r = evaluate_single_factor(args.factor_name, t)
            print(f"\n{r.summary()}")

        eval_df = pd.DataFrame([
            evaluate_single_factor(args.factor_name, t).to_dict()
            for t in target_names
        ])
    else:
        eval_df = evaluate_factors_batched(
            factor_names=factor_names,
            target_names=target_names,
            max_workers=args.workers,
            batch_size=args.batch_size,
            use_dashboard=not args.no_dashboard,
            checkpoint_file=checkpoint,
        )

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Save results ──
    if not eval_df.empty:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = args.output_dir / f"factor_evaluation_{timestamp}.csv"
        save_eval_csv(eval_df, csv_path)
        print(f"  CSV saved: {csv_path}")

        # Markdown report
        md_path = args.output_dir / f"factor_evaluation_report_{timestamp}.md"
        save_eval_report(eval_df, md_path, top_n=args.top_n)
        print(f"  Report saved: {md_path}")

        # Latest symlinks
        latest_csv = args.output_dir / "factor_evaluation_latest.csv"
        latest_md = args.output_dir / "factor_evaluation_latest.md"
        for src, dst in [(csv_path, latest_csv), (md_path, latest_md)]:
            try:
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src.name)
            except OSError:
                pass

    # ── Print top factors ──
    if not eval_df.empty and "abs_rank_icir" in eval_df.columns:
        print(f"\n{'='*64}")
        print(f"  Top 20 Factors by |Rank ICIR| (avg across targets)")
        print(f"{'='*64}")

        top_by_icir = (
            eval_df.groupby("factor_name")["abs_rank_icir"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )
        for i, (name, val) in enumerate(top_by_icir.items(), 1):
            meta = get_factor_metadata(name)
            cat = meta.get("category", "?")
            print(f"  {i:2d}. {name:<40} |ICIR|={val:.4f}  [{cat}]")

    # ── Optional: output screened factors ──
    if args.output_screened_factors and not eval_df.empty:
        print(f"\n{'='*64}")
        print(f"  Factor Screening & Redundancy Removal")
        print(f"  Target: {args.screen_target}  |  Corr threshold: {args.corr_threshold}")
        print(f"{'='*64}")

        # Use the specified target for screening
        target_results = eval_df[eval_df["target_name"] == args.screen_target]
        if target_results.empty:
            # Fallback: use all targets and pick best per factor
            target_results = (
                eval_df.groupby("factor_name")
                .apply(lambda g: g.loc[g["abs_rank_icir"].idxmax()])
                .reset_index(drop=True)
            )

        results_objects = [
            FactorEvalResult(
                factor_name=row["factor_name"],
                target_name=row.get("target_name", args.screen_target),
                rank_icir=row.get("rank_icir", float("nan")),
                rank_ic_mean=row.get("rank_ic_mean", float("nan")),
                long_short_spread_amt=row.get("long_short_spread_amt", float("nan")),
                long_ret_head_amt=row.get("long_ret_head_amt", float("nan")),
                short_ret_tail_amt=row.get("short_ret_tail_amt", float("nan")),
                coverage_ratio=row.get("coverage_ratio", float("nan")),
                n_dates=int(row.get("n_dates", 0)),
                ic_stability=row.get("ic_stability", float("nan")),
                ic_decay_5d=row.get("ic_decay_5d", float("nan")),
            )
            for _, row in target_results.iterrows()
        ]

        # Import paths for redundancy removal
        from featureengineering.settings import configure_paths
        proj_paths = configure_paths()

        screener = FactorScreener(fac_num_target=1400)
        screened = screener.screen(
            results_objects,
            sort_metric="abs_long_short_amt",
            remove_redundant=True,
            corr_threshold=args.corr_threshold,
            paths=proj_paths,
            keep_metric="abs_rank_icir",
        )

        screen_path = args.output_dir / f"screened_factors_{timestamp}.json"
        output = {
            "target": args.screen_target,
            "corr_threshold": args.corr_threshold,
            "n_total": len(factor_names),
            "n_evaluated": len(target_results),
            "n_screened": len(screened),
            "factors": screened,
            "generated_at": pd.Timestamp.now().isoformat(),
        }
        with open(screen_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  Screened factors saved: {screen_path} ({len(screened)} factors)")

    # ── Optional: season screening ──
    if args.screen:
        print(f"\n{'='*64}")
        print(f"  Season-based Factor Screening (fac_filt.py style)")
        print(f"{'='*64}")
        sel = run_season_screening(
            factor_names=list(eval_df["factor_name"].unique()) if not eval_df.empty else factor_names,
            target_name="label_ret_5d",
        )
        screen_path = args.output_dir / f"season_screening_{timestamp}.json"
        with open(screen_path, "w") as f:
            json.dump({k: list(v) for k, v in sel.items()}, f, indent=2, ensure_ascii=False)
        print(f"  Screening saved: {screen_path}")
        for season, facs in sel.items():
            print(f"  {season}: {len(facs)} factors selected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
