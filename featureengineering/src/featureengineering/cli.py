from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import (
    set_skip_existing,
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
    p.add_argument("--skip-existing", action="store_true", default=False,
                   help="Skip factors whose .fea file already exists (simple file check).")

    p.add_argument("--new", action="store_true", dest="use_new",
                   help="(Now the default for Class 2 & 3 — kept for compatibility.)")

    # ── Quality & visualisation ──────────────────────────────────────────
    p.add_argument("--dashboard", action="store_true",
                   help="Enable Rich-based live dashboard for build progress.")
    p.add_argument("--quality-check-days", type=int, default=0,
                   metavar="N",
                   help="Check last N trading days of existing .fea files. "
                        "If all-NaN, rebuild from scratch (default: 0 = skip check). "
                        "Recommended: 5 for daily builds.")

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
    from .builder import _resolve_effective_end_date
    _effective_end = _resolve_effective_end_date(paths.source_root)
    plan: dict[str, tuple[str, str | None]] = {}
    for name in selected:
        spec = get_factor(name)
        fp = paths.factor_output_dir / f"{spec.name}.fea"
        plan[name] = decide_build_action(
            name, fp, spec.dependencies, paths.source_root, force=force,
            effective_end=_effective_end,
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
    # Windows may expose a GBK console that cannot encode mathematical
    # symbols used in factor descriptions.  Keep the native encoding (so
    # Chinese remains readable) but never let reporting abort a build.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (AttributeError, OSError):
                pass

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

    # ── Skip-existing: if set, skip factors with existing .fea files ──
    if args.skip_existing:
        set_skip_existing(True)

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

    # ── Filter by class (--only-class 1,2,3,4,5) ────────────────────────
    if args.only_class is not None:
        selected = {int(c.strip()) for c in args.only_class.split(",") if c.strip()}
        invalid = selected - {1, 2, 3, 4, 5}
        if invalid:
            p.error(f"Invalid class number(s): {sorted(invalid)}.  Must be 1, 2, 3, 4, or 5.")

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
    # Preserve the dependency order Class 1 -> ... -> Class 5.  In particular,
    # Class 5 consumes Class 1-4 .fea outputs and must never share a parallel
    # worker pool with Class 1 when --all or a multi-class selection is used.
    c1_names = [n for n in factor_names if classify_factor(n) == 1]
    c2_names = [n for n in factor_names if classify_factor(n) == 2]
    c3_names = [n for n in factor_names if classify_factor(n) == 3]
    c4_names = [n for n in factor_names if classify_factor(n) == 4]
    c5_names = [n for n in factor_names if classify_factor(n) == 5]

    all_results = []

    def _build_regular(names: list[str]):
        if not names:
            return []
        if len(names) == 1:
            return build_many(names, paths=paths, force=args.force)
        return build_many_parallel(
            names, max_workers=args.jobs, paths=paths, force=args.force,
            quality_check_days=args.quality_check_days,
            use_dashboard=args.dashboard,
        )

    if c1_names:
        all_results.extend(_build_regular(c1_names))

    if c2_names:
        from .factors.chip_deep import build_cyq_chips_new
        all_results.extend(build_cyq_chips_new(
            factor_names=c2_names,
            paths=paths,
            force=args.force,
            max_workers=args.jobs,
            quality_check_days=args.quality_check_days,
        ))

    if c3_names:
        from .factors.intraday import build_intraday_new
        all_results.extend(build_intraday_new(
            factor_names=c3_names,
            paths=paths,
            force=args.force,
            max_workers=args.jobs,
            quality_check_days=args.quality_check_days,
        ))

    if c4_names:
        from .factors.indicator_minute import build_indicator_1min_new
        all_results.extend(build_indicator_1min_new(
            factor_names=c4_names,
            paths=paths,
            force=args.force,
            max_workers=args.jobs,
            quality_check_days=args.quality_check_days,
        ))

    if c5_names:
        all_results.extend(_build_regular(c5_names))

    for result in all_results:
        status = "OK" if result.action not in ("error",) else "ERR"
        path_str = str(result.factor_path)
        if not path_str or path_str == ".":
            print(f"[{status}] {result.factor_name} -> N/A (build failed)")
        else:
            print(f"[{status}] {result.factor_name} -> {path_str}")
    return 0
