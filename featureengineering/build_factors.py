from __future__ import annotations

import sys
import time
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
from featureengineering.report import action_colored, print_post_build_report
from featureengineering.settings import ProjectPaths, configure_paths
from featureengineering.state import BuildState, resolve_state_path
from featureengineering.builder import check_factor_dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

#: Force rebuild even when factor data is already up to date.
FORCE = False

#: Parallelism for Phase 1 (regular daily-level factors).
JOBS_REGULAR = 8

#: Run factors sequentially (no ProcessPoolExecutor).
SEQUENTIAL = False

# ── New feature toggles ───────────────────────────────────────────────────────

#: Build only this single factor (by name).  When set, all other factors
#: are ignored.  Works for Phase 2 factors as well.
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

#: Skip Phase 2 (heavy data: cyq_chips, history_1min).
SKIP_PHASE2 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 identification
# ═══════════════════════════════════════════════════════════════════════════════

# Data dependencies that mark a factor as Phase 2 (heavy data).
PHASE2_DEPS: tuple[str, ...] = ("cyq_chips.parquet", "history_1min.parquet")


def _is_phase2(name: str) -> bool:
    """Return True if the factor uses heavy data sources."""
    spec = get_factor(name)
    return any(dep in spec.dependencies for dep in PHASE2_DEPS)


def _split_phases(names: list[str]) -> tuple[list[str], list[str]]:
    """Split factor names into (phase1, phase2) lists."""
    p1, p2 = [], []
    for n in names:
        if _is_phase2(n):
            p2.append(n)
        else:
            p1.append(n)
    return p1, p2


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
# Phase 1 — regular factors (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_phase1(
    names: list[str],
    paths: ProjectPaths,
    state: BuildState,
) -> tuple[list[str], list[str]]:
    """Build Phase 1 factors in parallel.

    Returns (errors, completed_names).
    """
    if not names:
        print("Phase 1: no factors to build.\n")
        return [], []

    print(f"\n{'='*64}")
    print(f"  PHASE 1 — Regular Factors  ({len(names)} factors, {JOBS_REGULAR} workers)")
    print(f"{'='*64}\n")

    results = build_all(
        factor_names=names,
        max_workers=JOBS_REGULAR if not SEQUENTIAL else 1,
        sequential=SEQUENTIAL or len(names) == 1,
        force=FORCE,
    )

    built, skipped, errors = 0, 0, []
    for r in results:
        state.mark_factor(r.factor_name, r.action, "phase1")
        if r.action == "error":
            errors.append(r.factor_name)
        elif r.action == "skip":
            skipped += 1
        else:
            built += 1

    state.mark_phase("phase1", "completed")
    print(f"\nPhase 1 done: {built} built, {skipped} skipped, {len(errors)} errors\n")
    return errors, [r.factor_name for r in results]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — heavy data factors (sequential, detailed progress)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_phase2(
    names: list[str],
    paths: ProjectPaths,
    state: BuildState,
) -> list[str]:
    """Build Phase 2 factors sequentially with per-chunk progress.

    Returns list of error factor names.
    """
    if not names:
        print("Phase 2: no heavy-data factors to build.\n")
        return []

    print(f"\n{'='*64}")
    print(f"  PHASE 2 — Heavy Data Factors  ({len(names)} factors, sequential)")
    print(f"  Sources: cyq_chips.parquet (461M rows), history_1min.parquet")
    print(f"{'='*64}\n")

    state.mark_phase("phase2", "running")
    errors: list[str] = []

    for name in names:
        spec = get_factor(name)
        factor_path = paths.factor_output_dir / f"{spec.name}.fea"
        action, reason = decide_build_action(
            name, factor_path, spec.dependencies, paths.source_root, force=FORCE,
        )

        if action == "skip":
            print(f"  [{action_colored('skip'):<22}] {name}  ({reason})")
            state.mark_factor(name, "skip", "phase2")
            continue

        print(f"  [{action_colored(action):<22}] {name}  ...")
        t0 = time.perf_counter()

        try:
            result = build_many(
                [name],
                paths=paths,
                force=FORCE,
            )[0]
            elapsed = time.perf_counter() - t0

            if result.action == "error":
                errors.append(name)
                state.mark_factor(name, "error", "phase2")
            else:
                state.mark_factor(name, result.action, "phase2")

            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"  [{action_colored(result.action):<22}] {name}  "
                  f"{result.rows} dates  {mins}m{secs}s\n")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            errors.append(name)
            state.mark_factor(name, "error", "phase2")
            print(f"  [{action_colored('error'):<22}] {name}  {e}\n")

    state.mark_phase("phase2", "completed")
    print(f"Phase 2 done: {len(names) - len(errors)}/{len(names)} succeeded, "
          f"{len(errors)} errors\n")
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# Main build
# ═══════════════════════════════════════════════════════════════════════════════

def _build_all() -> int:
    """Build all factors in two phases, tracking progress in a state file.

    Phase 1: all daily-level+ factors (parallel)
    Phase 2: cyq_chips / history_1min factors (sequential, detailed progress)

    Returns 0 on success, 1 if any factor errored.
    """
    paths = configure_paths()
    ensure_builtin_factors_loaded()

    specs = list_factors()
    all_names = sorted(s.name for s in specs)

    # ── State file setup ──────────────────────────────────────────────────
    state = BuildState(resolve_state_path(PROJECT_ROOT, STATE_FILE), force=FORCE)
    state.load()

    # Resume from interrupted run?
    if RESUME and state.is_active():
        prev_phase2 = state._data.get("phase2", {})
        if prev_phase2.get("status") == "completed":
            print(f"Resuming from previous build [{state._data.get('build_id', '?')}] — "
                  f"{state.completed_count} factors already completed (both phases done).\n")
            state.finish("completed")
            return 0
        print(f"Resuming from previous build [{state._data.get('build_id', '?')}] — "
              f"{state.completed_count} factors already completed.\n")
    elif not state.is_active() or not RESUME:
        state.start(len(all_names))

    # ── Split phases ──────────────────────────────────────────────────────
    p1_names, p2_names = _split_phases(all_names)

    # Filter already-completed factors (fast-resume)
    p1_remaining = [n for n in p1_names if not state.is_completed(n)]
    p2_remaining = [n for n in p2_names if not state.is_completed(n)]

    skipped_p1 = len(p1_names) - len(p1_remaining)
    skipped_p2 = len(p2_names) - len(p2_remaining)

    # Print plan
    print(f"\nFactors: {len(p1_names)} Phase 1 + {len(p2_names)} Phase 2 = {len(all_names)} total")
    if skipped_p1 or skipped_p2:
        print(f"  Already completed: {skipped_p1} P1 + {skipped_p2} P2")
    print(f"  To build:          {len(p1_remaining)} P1 + {len(p2_remaining)} P2")
    print(f"  Force rebuild:     {FORCE}")
    print(f"  Phase 1 workers:   {JOBS_REGULAR}")
    print(f"  Phase 2 workers:   1 (sequential)")

    all_errors: list[str] = []

    # ── Phase 1 ───────────────────────────────────────────────────────────
    p1_errors, _ = _run_phase1(p1_remaining, paths, state)
    all_errors.extend(p1_errors)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    if SKIP_PHASE2:
        print("Phase 2 skipped (SKIP_PHASE2=True).\n")
        state.mark_phase("phase2", "skipped")
    else:
        p2_errors = _run_phase2(p2_remaining, paths, state)
        all_errors.extend(p2_errors)

    # ── Final summary ─────────────────────────────────────────────────────
    total_err = len(all_errors)
    total_completed = state.completed_count

    if total_err:
        state.finish("completed_with_errors")
    else:
        state.finish("completed")

    print(f"{'='*64}")
    print(f"  BUILD COMPLETE")
    print(f"  Total factors: {len(all_names)}")
    print(f"  Completed:     {total_completed}")
    print(f"  Errors:        {total_err}")
    if all_errors:
        print(f"  Error list:    {', '.join(all_errors)}")
    print(f"{'='*64}\n")

    # Print factor date health report
    print_post_build_report(check_factor_dates(paths=paths))

    return 1 if total_err else 0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from featureengineering.cli import main
        raise SystemExit(main())

    # ── Direct-run dispatch ──────────────────────────────────────────────────
    if SHOW_STATUS:
        configure_paths()
        ensure_builtin_factors_loaded()
        from featureengineering.report import print_status_table
        print_status_table(check_factor_dates())
        raise SystemExit(0)

    if ONLY_FACTOR:
        configure_paths()
        ensure_builtin_factors_loaded()
        specify = get_factor(ONLY_FACTOR)
        is_p2 = _is_phase2(ONLY_FACTOR)
        phase_tag = "Phase 2 (heavy data)" if is_p2 else "Phase 1"
        print(f"Building single factor: {ONLY_FACTOR}  "
              f"(category={specify.category}, {phase_tag})")
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
        p1_names, p2_names = _split_phases(names)
        state = BuildState(resolve_state_path(PROJECT_ROOT, STATE_FILE), force=FORCE)
        state.load()

        from featureengineering.report import print_plan_detail

        # Phase 1 plan
        p1_plan: dict[str, tuple[str, str | None]] = {}
        for name in p1_names:
            if state.is_completed(name):
                p1_plan[name] = ("skip", "state: already completed")
            else:
                p1_plan[name] = _classify_factor(name, configure_paths(), FORCE)
        print(f"\n--- Phase 1 ({len(p1_names)} factors) ---")
        print_plan_detail(p1_plan, force=FORCE)

        # Phase 2 plan
        p2_plan: dict[str, tuple[str, str | None]] = {}
        for name in p2_names:
            if state.is_completed(name):
                p2_plan[name] = ("skip", "state: already completed")
            else:
                p2_plan[name] = _classify_factor(name, configure_paths(), FORCE)
        print(f"\n--- Phase 2 ({len(p2_names)} factors) ---")
        print_plan_detail(p2_plan, force=FORCE)

        raise SystemExit(0)

    raise SystemExit(_build_all())
