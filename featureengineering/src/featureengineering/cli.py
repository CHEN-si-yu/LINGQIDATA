from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import (
    build_all,
    build_many,
    check_factor_dates,
    decide_build_action,
    get_factor,
    list_factors,
    recommend_worker_count,
)
from .factor_loader import ensure_builtin_factors_loaded
from .settings import configure_paths, ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build single-factor .fea files and target labels."
    )
    # ── Factor selection ─────────────────────────────────────────────────
    p.add_argument("--list", action="store_true", help="List available factors.")
    p.add_argument("--all", action="store_true", help="Build all registered factors.")
    p.add_argument("--targets-only", action="store_true",
                   help="Build only target label factors.")
    p.add_argument("--skip-targets", action="store_true",
                   help="Exclude target factors from build.")
    p.add_argument("--factor", action="append", default=[],
                   help="Build one specific factor. Repeatable.")
    p.add_argument("--only-factor", type=str, default=None,
                   help="Build exactly one factor and exit "
                        "(shortcut for --factor NAME).")

    # ── Build control ────────────────────────────────────────────────────
    p.add_argument("--jobs", type=int, default=recommend_worker_count(),
                   help="Parallel worker count.")
    p.add_argument("--sequential", action="store_true",
                   help="Disable multiprocessing and run sequentially.")
    p.add_argument("--force", action="store_true",
                   help="Force rebuild all factors (ignore date-based skip/incremental logic).")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Resume from interrupted build via state file (default: on).")
    p.add_argument("--no-resume", action="store_false", dest="resume",
                   help="Start a fresh build, ignoring any prior state file.")

    # ── Inspection (no build) ────────────────────────────────────────────
    p.add_argument("--check-dates", action="store_true",
                   help="Scan all factor files and report their last date.")
    p.add_argument("--status", action="store_true",
                   help="Show factor health summary (alias for --check-dates).")
    p.add_argument("--plan", action="store_true",
                   help="Print what would be built and exit (dry-run).")

    # ── Paths ────────────────────────────────────────────────────────────
    p.add_argument("--project-root", type=Path, default=None,
                   help="Project root. Defaults to the current package root.")
    p.add_argument("--source-root", type=Path, default=None,
                   help="Input data root. Defaults to ../data under the project parent directory.")
    p.add_argument("--factor-output-dir", type=Path, default=None,
                   help="Override output directory for factor .fea files.")
    p.add_argument("--manifest-output-dir", type=Path, default=None,
                   help="Override output directory for factor .json manifest files.")
    p.add_argument("--state-file", type=Path, default=None,
                   help="Override path to build-state JSON file.")

    return p


def _print_status_table(
    date_info: dict[str, dict[str, str | None]],
    title: str = "Factor status",
) -> None:
    if not date_info:
        print("No factor files found.")
        return
    effective_end = next(iter(date_info.values()))["effective_end"]
    print(f"\n{'='*72}")
    print(f"  {title}  (effective end: {effective_end})")
    print(f"{'='*72}")
    print(f"  {'Factor':<38} {'Last date':>10}  Status")
    print(f"  {'-'*38} {'-'*10}  {'-'*6}")
    counts: dict[str, int] = {}
    for name, info in date_info.items():
        last = info["last_date"] or "---"
        status = info["status"]
        marker = {"ok": "✓", "stale": "△", "future": "▶",
                  "error": "✗", "empty": "○"}.get(status, "?")
        print(f"  {name:<38} {last:>10}  {marker}  {status}")
        counts[status] = counts.get(status, 0) + 1
    print(f"{'='*72}")
    total = sum(counts.values())
    detail = "  ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"  Total: {total}  |  {detail}")


def _print_plan(selected: list[str], paths: ProjectPaths, force: bool) -> None:
    """Print build plan: what each factor would do."""
    plan: dict[str, tuple[str, str | None]] = {}
    for name in selected:
        spec = get_factor(name)
        fp = paths.factor_output_dir / f"{spec.name}.fea"
        plan[name] = decide_build_action(
            name, fp, spec.dependencies, paths.source_root, force=force,
        )

    counts: dict[str, int] = {}
    for action, _ in plan.values():
        counts[action] = counts.get(action, 0) + 1
    parts = "  ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"\nBuild plan ({len(plan)} factors, force={force}):  {parts}\n")

    action_order = {"rebuild": 0, "incremental": 1, "skip": 2}
    for name in sorted(plan, key=lambda n: (action_order.get(plan[n][0], 9), n)):
        action, reason = plan[name]
        print(f"  {action:<13} {name:<35}  ({reason})")
    print()


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    paths = configure_paths(
        project_root=args.project_root,
        source_root=args.source_root,
        factor_output_dir=args.factor_output_dir,
        manifest_output_dir=args.manifest_output_dir,
    )
    ensure_builtin_factors_loaded()

    # ── Status / check-dates (no build) ──────────────────────────────────
    if args.status or args.check_dates:
        _print_status_table(check_factor_dates(paths=paths))
        return 0

    # ── List (no build) ──────────────────────────────────────────────────
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

    # ── Factor selection ─────────────────────────────────────────────────
    if args.only_factor:
        factor_names = [args.only_factor]
    elif args.targets_only:
        factor_names = [s.name for s in specs if s.category == "target"]
    elif args.all:
        factor_names = [s.name for s in specs]
    else:
        factor_names = args.factor

    if args.skip_targets:
        factor_names = [n for n in factor_names
                        if not n.startswith("label_ret_")]

    if not factor_names:
        p.error("Please provide --all, --targets-only, --only-factor, "
                "or at least one --factor.")

    factor_names = sorted(set(factor_names))

    # ── Plan mode (no build) ─────────────────────────────────────────────
    if args.plan:
        _print_plan(factor_names, paths, args.force)
        return 0

    # ── Build ────────────────────────────────────────────────────────────
    # Single factor or sequential → use build_many directly (sub-progress bar)
    if len(factor_names) == 1 or args.sequential:
        results = build_many(factor_names, paths=paths, force=args.force)
    else:
        results = build_all(
            factor_names=factor_names,
            max_workers=args.jobs,
            sequential=False,
            force=args.force,
        )

    for result in results:
        status = "OK" if result.action not in ("error",) else "ERR"
        print(f"[{status}] {result.factor_name} -> {result.factor_path}")
    return 0
