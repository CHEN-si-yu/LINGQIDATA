from __future__ import annotations

import argparse
from pathlib import Path

from .builder import build_all, list_factors, recommend_worker_count
from .factor_loader import ensure_builtin_factors_loaded
from .settings import configure_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build single-factor .fea files and target labels.")
    parser.add_argument("--list", action="store_true", help="List available factors.")
    parser.add_argument("--all", action="store_true", help="Build all registered factors.")
    parser.add_argument("--targets-only", action="store_true", help="Build only target label factors.")
    parser.add_argument("--skip-targets", action="store_true", help="Exclude target factors from build.")
    parser.add_argument(
        "--factor", action="append", default=[],
        help="Build one specific factor. Repeatable.",
    )
    parser.add_argument(
        "--jobs", type=int, default=recommend_worker_count(),
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Disable multiprocessing and run sequentially.",
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="Project root. Defaults to the current package root.",
    )
    parser.add_argument(
        "--source-root", type=Path, default=None,
        help="Input data root. Defaults to ../data under the project parent directory.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild all factors (ignore date-based skip/incremental logic).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_paths(project_root=args.project_root, source_root=args.source_root)
    ensure_builtin_factors_loaded()
    specs = list_factors()

    if args.list:
        target_specs = [s for s in specs if s.category == "target"]
        feature_specs = [s for s in specs if s.category != "target"]
        print(f"\n{'='*60}")
        print(f"Target labels ({len(target_specs)}):")
        for spec in target_specs:
            print(f"  {spec.name:<32} {spec.description}")
        print(f"\nFeature factors ({len(feature_specs)}):")
        for spec in feature_specs:
            print(f"  {spec.name:<32} {spec.category:<14} {spec.description}")
        return 0

    if args.targets_only:
        factor_names = [s.name for s in specs if s.category == "target"]
    elif args.all:
        factor_names = [s.name for s in specs]
    else:
        factor_names = args.factor

    if args.skip_targets:
        factor_names = [n for n in factor_names if not n.startswith("label_ret_")]

    if not factor_names:
        parser.error("Please provide --all, --targets-only, or at least one --factor.")

    factor_names = sorted(set(factor_names))
    results = build_all(
        factor_names=factor_names,
        max_workers=args.jobs,
        sequential=args.sequential,
        project_root=args.project_root,
        source_root=args.source_root,
        force=args.force,
    )
    for result in results:
        print(f"[OK] {result.factor_name} -> {result.factor_path}")
    return 0
