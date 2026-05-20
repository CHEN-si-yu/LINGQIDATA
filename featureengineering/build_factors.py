from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
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
from featureengineering.settings import ProjectPaths, configure_paths

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
# BuildState — lightweight JSON-backed state for crash recovery & fast restart
# ═══════════════════════════════════════════════════════════════════════════════

class BuildState:
    """Tracks build progress so interrupted runs can resume near-instantly.

    On every factor completion the state is flushed to disk.  On restart
    *completed* factors are skipped without touching source parquet files,
    which makes resume effectively free for large factor sets.
    """

    def __init__(self, path: Path, force: bool = False) -> None:
        self._path = path
        self._force = force
        self._data: dict = {}
        self._dirty = False

    # ── load / save ──────────────────────────────────────────────────────────

    def load(self) -> dict:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

        # Invalidate if force mode changed
        if self._data.get("force") != self._force:
            self._data = {}

        return self._data

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self._path)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, total: int) -> None:
        """Initialise a fresh build record."""
        self._data = {
            "version": 1,
            "build_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": "",
            "force": self._force,
            "status": "running",
            "phase1": {"status": "pending", "total": total, "done": 0,
                       "errors": []},
            "completed": {},
        }
        self.save()

    def mark_phase(self, phase: str, status: str) -> None:
        if phase in self._data:
            self._data[phase]["status"] = status
        self.save()

    def finish(self, status: str = "completed") -> None:
        self._data["status"] = status
        self._data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    # ── per-factor tracking ──────────────────────────────────────────────────

    def is_completed(self, factor_name: str) -> bool:
        return factor_name in self._data.get("completed", {})

    def mark_factor(self, factor_name: str, action: str, phase: str) -> None:
        self._data["completed"][factor_name] = {
            "action": action,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        phase_data = self._data.get(phase)
        if phase_data:
            phase_data["done"] = phase_data.get("done", 0) + 1
            if action == "error":
                phase_data.setdefault("errors", []).append(factor_name)
        self.save()

    def completed_in_phase(self, phase: str) -> set[str]:
        """Return factor names completed during *phase* in the current run."""
        phase_data = self._data.get(phase)
        if not phase_data:
            return set()
        # All completed factors belong to the active build — we trust
        # that factors completed in prior phases won't be re-submitted.
        return set(self._data.get("completed", {}).keys())

    # ── query ────────────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        return self._data.get("status") == "running"

    def stale_build_id(self) -> str | None:
        if self.is_active():
            return self._data.get("build_id")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_state_path() -> Path:
    return (PROJECT_ROOT / STATE_FILE).resolve()


def _classify_factor(name: str, paths: ProjectPaths, force: bool) -> tuple[str, str | None]:
    """Return (action, reason) for a single factor using the full date logic."""
    spec = get_factor(name)
    factor_path = paths.factor_output_dir / f"{spec.name}.fea"
    return decide_build_action(
        name, factor_path, spec.dependencies, paths.source_root, force=force,
    )


def _print_plan_summary(plan: dict[str, tuple[str, str | None]],
                         title: str = "Build plan") -> None:
    """Print a compact build-plan summary."""
    counts: dict[str, int] = {}
    for action, _ in plan.values():
        counts[action] = counts.get(action, 0) + 1
    parts = "  ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"{title}:  {parts}  (total: {len(plan)})")


_STATUS_MARKERS = {
    "ok": "✓", "stale": "△", "future": "▶", "error": "✗", "empty": "○",
}


def _print_status_table(
    date_info: dict[str, dict[str, str | None]],
    title: str = "Factor status",
) -> None:
    """Print a multi-column status table with aligned columns."""
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
        marker = _STATUS_MARKERS.get(status, "?")
        print(f"  {name:<38} {last:>10}  {marker}  {status}")
        counts[status] = counts.get(status, 0) + 1
    print(f"{'='*72}")
    total = sum(counts.values())
    detail = "  ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"  Total: {total}  |  {detail}")


_COLOR_MAP = {"rebuild": "\033[33m", "incremental": "\033[36m",
              "skip": "\033[32m", "error": "\033[31m"}
_RESET = "\033[0m"


def _action_colored(action: str) -> str:
    c = _COLOR_MAP.get(action, "")
    return f"{c}{action}{_RESET}"


# ═══════════════════════════════════════════════════════════════════════════════
# Status mode
# ═══════════════════════════════════════════════════════════════════════════════

def _show_status() -> None:
    """Print every factor's last date and health, then exit."""
    from featureengineering.builder import check_factor_dates

    configure_paths()
    ensure_builtin_factors_loaded()
    _print_status_table(check_factor_dates())


# ═══════════════════════════════════════════════════════════════════════════════
# Dry-run / plan mode
# ═══════════════════════════════════════════════════════════════════════════════

def _plan_build(names: list[str], state: BuildState, paths: ProjectPaths) -> None:
    """Print what *would* be built without touching any data."""
    configure_paths()
    ensure_builtin_factors_loaded()

    plan: dict[str, tuple[str, str | None]] = {}
    for name in names:
        if state.is_completed(name):
            plan[name] = ("skip", "state: already completed")
        else:
            plan[name] = _classify_factor(name, paths, FORCE)
    _print_plan_summary(plan, "Build plan")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Single-factor mode
# ═══════════════════════════════════════════════════════════════════════════════

def _build_single_factor(name: str) -> int:
    """Build exactly one factor and print the result."""
    configure_paths()
    ensure_builtin_factors_loaded()

    spec = get_factor(name)  # validates the factor exists
    print(f"Building single factor: {name}  (category={spec.category})")

    result = build_many(names=[name], force=FORCE)
    if not result:
        return 1

    r = result[0]
    action_str = _action_colored(r.action)
    print(f"[{action_str}] {r.factor_name}  elapsed={r.elapsed:.1f}s  "
          f"rows={r.rows}  path={r.factor_path}")
    return 0 if r.action != "error" else 1


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
    state_path = _resolve_state_path()
    state = BuildState(state_path, force=FORCE)
    state.load()

    # Resume from interrupted run?
    if RESUME and state.is_active():
        prev_id = state.stale_build_id()
        prev_completed = set(state._data.get("completed", {}).keys())
        print(f"Found interrupted build [{prev_id}] — "
              f"{len(prev_completed)} factors already completed, will resume.\n")
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
    total_completed = len(state._data.get("completed", {}))

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
        # CLI passthrough — forward all arguments to the standard CLI entry point.
        from featureengineering.cli import main

        raise SystemExit(main())

    # ── Direct-run dispatch ──────────────────────────────────────────────────
    if SHOW_STATUS:
        _show_status()
        raise SystemExit(0)

    if ONLY_FACTOR:
        raise SystemExit(_build_single_factor(ONLY_FACTOR))

    if DRY_RUN:
        configure_paths()
        ensure_builtin_factors_loaded()
        specs = list_factors()
        names = sorted(s.name for s in specs)
        state = BuildState(_resolve_state_path(), force=FORCE)
        state.load()
        _plan_build(names, state, configure_paths())
        raise SystemExit(0)

    raise SystemExit(_build_all())
