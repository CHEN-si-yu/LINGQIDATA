from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import (
    build_many,
    build_many_parallel,
    check_factor_dates,
    classify_factor,
    decide_build_action,
    get_factor,
    list_factors,
)
from .factor_loader import ensure_builtin_factors_loaded
from .report import action_colored, print_status_table
from .settings import configure_paths, log_environment_info, ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build single-factor .fea files and target labels."
    )
    # ── Factor selection ─────────────────────────────────────────────────
    p.add_argument("--list", action="store_true", help="List available factors.")
    p.add_argument("--all", action="store_true", help="Build all registered factors.")
    p.add_argument("--fac-name", "--only-factor", "-f", type=str, default=None, dest="only_factor",
                   help="Build exactly one factor and exit.")

    # ── Class selection ───────────────────────────────────────────────────
    p.add_argument("--only-class", type=str, default=None,
                   help="Only build factors of the given class(es), "
                        "comma-separated: 1,2,3,4  (e.g. --only-class 1,2)")

    # ── Build control ────────────────────────────────────────────────────
    p.add_argument("--jobs", type=int, default=None,
                   help="Parallel worker count (default: auto-detected).")
    p.add_argument("--force", action="store_true",
                   help="Force rebuild all factors (ignore date-based skip/incremental logic).")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Resume from interrupted build via state file (default: on).")
    p.add_argument("--no-resume", action="store_false", dest="resume",
                   help="Start a fresh build, ignoring any prior state file.")
    p.add_argument("--new", action="store_true", dest="use_new",
                   help="(Now the default for Class 2 & 3 — kept for compatibility.)")

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
        print(f"  {action_colored(action):<22} {name:<35}  ({reason})")
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
    log_environment_info()
    ensure_builtin_factors_loaded()

    # ── Status / check-dates (no build) ──────────────────────────────────
    if args.status or args.check_dates:
        print_status_table(check_factor_dates(paths=paths))
        return 0

    # ── List (no build) ──────────────────────────────────────────────────
    if args.list:
        specs = list_factors()
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
    specs = list_factors()

    if args.only_factor:
        factor_names = [args.only_factor]
    elif args.all or args.only_class is not None:
        factor_names = [s.name for s in specs]
    else:
        p.error("Please provide --all, --only-class, or --only-factor.")

    factor_names = sorted(set(factor_names))

    # ── Filter by class (--only-class 1,2,3) ────────────────────────────
    if args.only_class is not None:
        selected = {int(c.strip()) for c in args.only_class.split(",") if c.strip()}
        invalid = selected - {1, 2, 3, 4}
        if invalid:
            p.error(f"Invalid class number(s): {sorted(invalid)}.  Must be 1, 2, 3, or 4.")

        before = len(factor_names)
        factor_names = [n for n in factor_names if classify_factor(n) in selected]
        print(f"  --only-class {selected}: {before} -> {len(factor_names)} factors")
        if not factor_names:
            p.error(f"No factors match class filter {selected}.")

    # ── Plan mode (no build) ─────────────────────────────────────────────
    if args.plan:
        _print_plan(factor_names, paths, args.force)
        return 0

    # ── Build ────────────────────────────────────────────────────────────
    # Class 2 & 3 always use unified single-pass builders.
    # Separate factors by class so each data source is opened once.
    c2_names = [n for n in factor_names if classify_factor(n) == 2]
    c3_names = [n for n in factor_names if classify_factor(n) == 3]
    other_names = [n for n in factor_names if classify_factor(n) not in (2, 3)]

    all_results = []

    if c2_names:
        from .factors.chip_deep import build_cyq_chips_new
        all_results.extend(build_cyq_chips_new(
            factor_names=c2_names,
            paths=paths,
            force=args.force,
            max_workers=args.jobs,
        ))

    if c3_names:
        from .factors.intraday import build_intraday_new
        all_results.extend(build_intraday_new(
            factor_names=c3_names,
            paths=paths,
            force=args.force,
            max_workers=args.jobs,
        ))

    if other_names:
        if len(other_names) == 1:
            all_results.extend(build_many(other_names, paths=paths, force=args.force))
        else:
            all_results.extend(build_many_parallel(
                other_names, max_workers=args.jobs, paths=paths, force=args.force,
            ))

    for result in all_results:
        status = "OK" if result.action not in ("error",) else "ERR"
        print(f"[{status}] {result.factor_name} -> {result.factor_path}")
    return 0
