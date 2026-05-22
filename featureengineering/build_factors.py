from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from featureengineering.builder import (
    build_all,
    build_many,
    decide_build_action,
    get_factor,
    list_factors,
)
from featureengineering.factor_loader import ensure_builtin_factors_loaded
from featureengineering.report import action_colored, print_plan_summary
from featureengineering.settings import ProjectPaths, configure_paths
from featureengineering.state import BuildState, resolve_state_path

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
#   Edit these to control behaviour when launched without CLI arguments.
#   With arguments the standard CLI handles everything instead.
# ═══════════════════════════════════════════════════════════════════════════════

#: Force rebuild even when factor data is already up to date.
FORCE = False

#: Parallelism for regular (single-panel) factors — mostly CPU-bound.
JOBS_REGULAR = 8

#: Run factors sequentially (no ProcessPoolExecutor).
SEQUENTIAL = False

# ── New feature toggles ───────────────────────────────────────────────────────

#: Build only this single factor (by name).  When set, all other factors
#: are ignored.
#: Example:  ONLY_FACTOR = "mom_20"
ONLY_FACTOR: str | None = None

#: Print factor status (last date / health) and exit without building.
SHOW_STATUS = False

#: Print the build plan (which factors will be rebuilt / skipped / etc.)
#: and exit without building.
DRY_RUN = False

#: Resume from an interrupted build.  Factors recorded as completed in the
#: state file are skipped instantly without re-scanning source dates.
RESUME = True

#: Path to the build-state file (relative to project root).
STATE_FILE = "data/build_state.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_factor(name: str, paths: ProjectPaths, force: bool) -> tuple[str, str | None]:
    """Return (action, reason) for a single factor using the full date logic."""
    spec = get_factor(name)
    factor_path = paths.factor_output_dir / f"{spec.name}.fea"
    return decide_build_action(
        name, factor_path, spec.dependencies, paths.source_root, force=force,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Build (with state tracking & fast resume)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_all() -> int:
    """Build all factors, tracking progress in a state file so
    interrupted runs can resume quickly.

    Returns 0 on success, 1 if any factor errored.
    """
    paths = configure_paths()
    ensure_builtin_factors_loaded()

    specs = list_factors()
    names = sorted(s.name for s in specs)

    # ── State file setup ──────────────────────────────────────────────────
    state = BuildState(resolve_state_path(PROJECT_ROOT, STATE_FILE), force=FORCE)
    state.load()

    # Resume from interrupted run?
    if RESUME and state.is_active():
        print(f"Resuming from previous build [{state._data.get('build_id', '?')}] — "
              f"{state.completed_count} factors already completed.\n")
    elif not state.is_active() or not RESUME:
        state.start(len(names))

    # Filter out already-completed factors (fast-resume path)
    remaining = [n for n in names if not state.is_completed(n)]
    skipped_by_state = len(names) - len(remaining)

    if remaining:
        print(f"=== Building factors ({len(names)})  "
              f"workers={JOBS_REGULAR}  force={FORCE} ===")
        if skipped_by_state:
            print(f"  ({skipped_by_state} already completed — resuming)")

        state.mark_phase("phase1", "running")

        results = build_all(
            factor_names=remaining,
            max_workers=JOBS_REGULAR if not SEQUENTIAL else 1,
            sequential=SEQUENTIAL or len(remaining) == 1,
            force=FORCE,
        )
        built = 0
        skipped = 0
        errors: list[str] = []
        for r in results:
            state.mark_factor(r.factor_name, r.action, "phase1")
            if r.action == "error":
                errors.append(r.factor_name)
            elif r.action == "skip":
                skipped += 1
            else:
                built += 1

        state.mark_phase("phase1", "completed")
        print(f"Build done: {built} built, {skipped} skipped, "
              f"{len(errors)} errors\n")
    else:
        state.mark_phase("phase1", "completed")
        print(f"All {len(names)} factors already completed (state file)\n")

    # ── Final summary ─────────────────────────────────────────────────────
    all_errors = state._data.get("phase1", {}).get("errors", [])
    total_completed = state.completed_count

    state.finish("completed" if not all_errors else "completed_with_errors")
    print(f"All done: {total_completed} factors processed, "
          f"{len(all_errors)} errors")
    if all_errors:
        print(f"Errors: {', '.join(all_errors)}")
        return 1
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from featureengineering.cli import main
        raise SystemExit(main())

    # ── Direct-run dispatch ──────────────────────────────────────────────────
    if SHOW_STATUS:
        from featureengineering.builder import check_factor_dates
        from featureengineering.report import print_status_table
        configure_paths()
        ensure_builtin_factors_loaded()
        print_status_table(check_factor_dates())
        raise SystemExit(0)

    if ONLY_FACTOR:
        configure_paths()
        ensure_builtin_factors_loaded()
        specify = get_factor(ONLY_FACTOR)
        print(f"Building single factor: {ONLY_FACTOR}  (category={specify.category})")
        result = build_many(names=[ONLY_FACTOR], force=FORCE)
        if result:
            r = result[0]
            action_str = action_colored(r.action)
            print(f"[{action_str}] {r.factor_name}  elapsed={r.elapsed:.1f}s  "
                  f"rows={r.rows}  path={r.factor_path}")
        raise SystemExit(0 if result and result[0].action != "error" else 1)

    if DRY_RUN:
        configure_paths()
        ensure_builtin_factors_loaded()
        specs = list_factors()
        names = sorted(s.name for s in specs)
        state = BuildState(resolve_state_path(PROJECT_ROOT, STATE_FILE), force=FORCE)
        state.load()

        plan: dict[str, tuple[str, str | None]] = {}
        for name in names:
            if state.is_completed(name):
                plan[name] = ("skip", "state: already completed")
            else:
                plan[name] = _classify_factor(name, configure_paths(), FORCE)
        print_plan_summary(plan, "Build plan")
        print()
        raise SystemExit(0)

    raise SystemExit(_build_all())
