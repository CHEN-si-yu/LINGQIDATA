from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)

from .dataset import DataRepository
from .factor_loader import ensure_builtin_factors_loaded
from .registry import FACTOR_REGISTRY, FactorContext, FactorSpec, get_factor
from .settings import ProjectPaths, configure_paths
from .storage import (
    ensure_single_factor_frame,
    write_factor,
    write_factor_incremental,
    write_target,
)


@dataclass(frozen=True)
class BuildResult:
    factor_name: str
    factor_path: Path
    manifest_path: Path
    elapsed: float
    action: str  # "rebuild", "incremental", "skip"


CATEGORY_ORDER = [
    "price", "valuation", "quality", "event", "fund_flow",
    "financial", "index", "target", "other",
]


# ── Date helpers ────────────────────────────────────────────────────────────

_DATE_CANDIDATES = ["trade_date", "end_date", "date", "ann_date", "f_ann_date"]

#: Daily pipeline cutoff hour (24h). Before this hour, todayʼs trading data is
#: considered unavailable; after this hour the ETL has finished and todayʼs
#: data may be used as the effective end date.
_EOD_CUTOFF_HOUR = 18


def _detect_date_col(columns: list[str]) -> str | None:
    for c in _DATE_CANDIDATES:
        if c in columns:
            return c
    return None


def _normalize_date(date_str: str) -> str:
    """Normalize a date string to YYYYMMDD format for consistent comparison."""
    return date_str.replace("-", "").strip()[:8]


def _read_source_max_date(filepath: Path) -> str | None:
    """Read the maximum date from a parquet source file (returns YYYYMMDD)."""
    if not filepath.exists():
        return None
    try:
        schema = pq.read_schema(filepath)
        date_col = _detect_date_col(schema.names)
        if date_col is None:
            return None
        table = pq.read_table(filepath, columns=[date_col])
        series = table.column(0).to_pandas()
        if series.empty:
            return None
        max_val = series.max()
        if hasattr(max_val, "strftime"):
            return max_val.strftime("%Y%m%d")
        return _normalize_date(str(max_val))
    except Exception:
        return None


def _resolve_effective_end_date(source_root: Path) -> str:
    """Return the latest date (YYYYMMDD) for which factors should produce data.

    Uses *daily_adj.parquet* as the canonical reference for available trading
    data, then applies the :data:`_EOD_CUTOFF_HOUR` (18:00) rule:

    - Before 18:00 — todayʼs market data is not yet available; cap at
      ``daily_adj.parquet`` max (typically the previous trading day).
    - At or after 18:00 — the daily ETL has completed; ``daily_adj.parquet``
      already reflects today and its max is used directly.

    The 18:00 cutoff is a belt-and-suspenders safeguard: the ETL pipeline
    that refreshes ``daily_adj.parquet`` only runs after market close, so
    the fileʼs max date is already correct.  This function adds an explicit
    time check so that even if the file were updated earlier the pipeline
    would not accidentally forward-fill into a trading day that has not
    concluded.
    """
    from datetime import datetime

    daily_max = _read_source_max_date(source_root / "daily_adj.parquet")
    if daily_max is None:
        # Fallback: if daily_adj is missing, trust the clock alone
        today = datetime.now().strftime("%Y%m%d")
        if datetime.now().hour < _EOD_CUTOFF_HOUR:
            from datetime import timedelta
            return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        return today

    effective = _normalize_date(daily_max)

    now = datetime.now()
    if now.hour < _EOD_CUTOFF_HOUR:
        today_str = now.strftime("%Y%m%d")
        if effective >= today_str:
            from datetime import timedelta
            effective = (now - timedelta(days=1)).strftime("%Y%m%d")

    return effective


def _read_single_file_max_date(filepath: Path) -> str | None:
    """Read the maximum Date index value from a single .fea file."""
    if not filepath.exists():
        return None
    try:
        df = pq.read_table(filepath).to_pandas()
    except Exception:
        import pandas as pd
        try:
            df = pd.read_feather(filepath)
        except Exception:
            return None
    if df.empty:
        return None
    idx = df.columns[0] if not df.index.names or df.index.names[0] is None else None
    if idx is not None and ("Date" in df.columns or "date" in df.columns):
        date_col = "Date" if "Date" in df.columns else "date"
        max_val = df[date_col].max()
        return _normalize_date(str(max_val)) if max_val is not None else None
    if isinstance(df.index, pd.MultiIndex):
        for name in df.index.names:
            if name and name.lower() == "date":
                vals = df.index.get_level_values(name)
                max_val = vals.max()
                return _normalize_date(str(max_val)) if max_val is not None else None
    return None


def _read_factor_max_date(factor_base_path: Path) -> str | None:
    """Read the maximum Date across base and incremental factor files."""
    from .storage import INCR_SUFFIX

    best = _read_single_file_max_date(factor_base_path)

    incr_path = factor_base_path.parent / f"{factor_base_path.stem}{INCR_SUFFIX}.fea"
    incr_max = _read_single_file_max_date(incr_path)
    if incr_max is not None and (best is None or incr_max > best):
        best = incr_max

    return best


def _check_source_dates(dependencies: tuple[str, ...], source_root: Path) -> dict[str, str | None]:
    """Return {dep_filename: max_date_or_None} for each dependency."""
    result: dict[str, str | None] = {}
    for dep in dependencies:
        fpath = source_root / dep
        result[dep] = _read_source_max_date(fpath)
    return result


def decide_build_action(
    factor_name: str,
    factor_path: Path,
    deps: tuple[str, ...],
    source_root: Path,
    force: bool = False,
) -> tuple[str, str | None]:
    """Decide whether to skip, incremental-build, or rebuild a factor.

    Returns (action, reason_or_new_start_date).
    - "skip": factor is up to date
    - "rebuild": need full rebuild
    - "incremental": rebuild only from a new start date
    """
    if force:
        return "rebuild", "forced"

    if not factor_path.exists():
        return "rebuild", "no existing factor file"

    # Effective end date: cap at the latest legitimately-available trading day
    effective_end = _resolve_effective_end_date(source_root)

    # Get source data max dates — use MIN so all dependencies must have data
    source_dates = _check_source_dates(deps, source_root)
    source_max: str | None = None
    for dep, max_d in source_dates.items():
        if max_d is None:
            source_max = None
            break
        if source_max is None or max_d < source_max:
            source_max = max_d

    if source_max is None:
        return "skip", "one or more source dependencies have no detectable max date"

    # Never treat source data as available beyond the effective end date
    if source_max > effective_end:
        source_max = effective_end

    factor_max = _read_factor_max_date(factor_path)

    if factor_max is None:
        return "rebuild", "cannot determine factor max date"

    # If factor has future data (from a buggy previous build), force rebuild
    if factor_max > effective_end:
        return "rebuild", f"factor has future date {factor_max} > effective end {effective_end}"

    if factor_max >= source_max:
        return "skip", f"factor max {factor_max} >= safe source date {source_max}"

    # Factor is behind — need incremental rebuild
    return "incremental", factor_max


# ── Listing ─────────────────────────────────────────────────────────────────

def list_factors() -> list[FactorSpec]:
    return [FACTOR_REGISTRY[name] for name in sorted(FACTOR_REGISTRY)]


def check_factor_dates(paths: ProjectPaths | None = None) -> dict[str, dict[str, str | None]]:
    """Scan all .fea files and return each factor's last date and status.

    Returns a dict keyed by factor name, each value is a dict with:
      - "last_date": str (YYYYMMDD) or None if unreadable
      - "effective_end": str (YYYYMMDD), the reference end date
      - "status": "ok" | "stale" | "future" | "error" | "empty"
    """
    configured = paths or configure_paths()
    factor_dir = configured.factor_output_dir
    effective_end = _resolve_effective_end_date(configured.source_root)

    result: dict[str, dict[str, str | None]] = {}
    for fpath in sorted(factor_dir.glob("*.fea")):
        name = fpath.stem
        max_date = _read_factor_max_date(fpath)
        if max_date is None:
            status = "error"
        elif max_date > effective_end:
            status = "future"
        elif max_date < effective_end:
            status = "stale"
        else:
            status = "ok"
        result[name] = {
            "last_date": max_date,
            "effective_end": effective_end,
            "status": status,
        }
    return result


def recommend_worker_count() -> int:
    """Return the recommended number of factor-level parallel workers.

    Fixed at 3 to keep overall memory and I/O pressure controlled.
    Individual factors may use internal threading for their own
    I/O parallelism — the factor pool size controls only how many factors
    are computed concurrently, not how each factor uses CPU internally.
    """
    return 3


def _category_rank(category: str) -> tuple[int, str]:
    try:
        return (CATEGORY_ORDER.index(category), category)
    except ValueError:
        return (len(CATEGORY_ORDER), category)


def factor_names_by_category(names: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        spec = get_factor(name)
        grouped.setdefault(spec.category, []).append(name)
    return {
        category: sorted(category_names)
        for category, category_names in sorted(
            grouped.items(), key=lambda item: _category_rank(item[0])
        )
    }


def _write_done_marker(
    name: str, action: str, manifest_dir: Path,
) -> None:
    """Write a .done marker so restarted runs can skip completed factors."""
    import json as _json
    from datetime import datetime as _dt
    marker = manifest_dir / f"{name}.done"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_json.dumps({
        "name": name,
        "action": action,
        "completed_at": _dt.now().isoformat(timespec="seconds"),
    }))


# ── Single factor build ─────────────────────────────────────────────────────

def build_factor(
    name: str,
    repo: DataRepository | None = None,
    paths: ProjectPaths | None = None,
    factor_start_date: str | None = None,
) -> BuildResult:
    """Build a single factor. Returns timing + result info.

    If *factor_start_date* is provided, only rows after that date are kept
    (incremental mode). The caller handles merging with existing data.
    """
    spec = get_factor(name)
    repo = repo or DataRepository(paths=paths)

    t0 = time.perf_counter()
    factor_frame = None

    # Compute context start date for date-aware loading
    _LOOKBACK = 252  # one calendar year, covers all rolling-window factors
    context_start: str | None = None
    if factor_start_date:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(factor_start_date, "%Y%m%d") - timedelta(days=_LOOKBACK)
        context_start = start_dt.strftime("%Y%m%d")

    # Determine the effective end date (6 PM cutoff + daily_adj max)
    _effective_end = _resolve_effective_end_date(repo.paths.source_root)
    # For incremental builds, never cap BEFORE the factor's own max date
    if factor_start_date and _effective_end < factor_start_date:
        _effective_end = factor_start_date

    try:
        context = FactorContext(repo=repo, start_date=context_start, end_date=_effective_end)
        raw_output = spec.compute(context)
        factor_frame = ensure_single_factor_frame(raw_output, spec.name)

        # If incremental, slice to only new dates
        if factor_start_date:
            date_level = factor_frame.index.get_level_values("Date")
            factor_frame = factor_frame.loc[date_level > factor_start_date]

        if spec.category == "target":
            factor_path, manifest_path = write_target(spec, factor_frame, paths=repo.paths)
        elif factor_start_date:
            factor_path, manifest_path = write_factor_incremental(
                spec, factor_frame, paths=repo.paths
            )
        else:
            factor_path, manifest_path = write_factor(spec, factor_frame, paths=repo.paths)

        action = "incremental" if factor_start_date else "rebuild"
        _write_done_marker(name, action, repo.paths.manifest_output_dir)

    except Exception:
        factor_path = repo.paths.factor_output_dir / f"{spec.name}.fea"
        manifest_path = repo.paths.manifest_output_dir / f"{spec.name}.json"
        raise

    finally:
        elapsed = time.perf_counter() - t0

    action = "incremental" if factor_start_date else "rebuild"
    return BuildResult(
        factor_name=spec.name,
        factor_path=factor_path,
        manifest_path=manifest_path,
        elapsed=elapsed,
        action=action,
    )


# ── Batch build (sequential) ────────────────────────────────────────────────

def _flatten_build_plan(names: list[str]) -> list[tuple[str, str]]:
    """Return a flat list of (category, factor_name) ordered by category rank."""
    grouped = factor_names_by_category(names)
    plan: list[tuple[str, str]] = []
    for category, factor_names in grouped.items():
        for name in factor_names:
            plan.append((category, name))
    return plan


def _sub_bar_label(name: str, stage: str, current: int, total: int, elapsed: float) -> str:
    """Format the sub-progress bar label."""
    if total == 0:
        return f"  {name} [{stage}] {elapsed:.0f}s"
    pct = current / total * 100 if total else 0
    return f"  {name} [{stage} {current}/{total} {pct:.0f}%] {elapsed:.0f}s"


def build_many(
    names: list[str],
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> list[BuildResult]:
    """Build factors sequentially, with skip/incremental/rebuild logic."""
    if not names:
        return []

    ensure_builtin_factors_loaded()
    plan = _flatten_build_plan(names)
    results: list[BuildResult] = []

    # First pass: classify actions
    repo_temp = DataRepository(paths=paths)
    action_map: dict[str, tuple[str, str | None]] = {}
    for _, name in plan:
        spec = get_factor(name)
        factor_path = repo_temp.paths.factor_output_dir / f"{spec.name}.fea"
        action, reason = decide_build_action(
            name, factor_path, spec.dependencies, repo_temp.paths.source_root, force=force,
        )
        action_map[name] = (action, reason)

    skipped = sum(1 for a, _ in action_map.values() if a == "skip")
    incr = sum(1 for a, _ in action_map.values() if a == "incremental")
    rebuild = sum(1 for a, _ in action_map.values() if a == "rebuild")
    logger.info(f"Build plan: {rebuild} rebuild | {incr} incremental | {skipped} skip")

    # Sub-progress bar (position 1) for the current factor's internal steps
    sub_bar = tqdm(total=1, position=1, desc="  Sub-progress", unit="step", leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}")

    # Track sub-progress state from callback
    sub_state: dict[str, Any] = {"name": "", "stage": "", "current": 0, "total": 0, "t0": 0.0}

    def _on_sub_progress(stage: str, current: int, total: int) -> None:
        if sub_state["stage"] != stage:
            sub_state["t0"] = time.perf_counter()
        sub_state["stage"] = stage
        sub_state["current"] = current
        sub_state["total"] = total
        elapsed = time.perf_counter() - sub_state["t0"]
        sub_bar.total = total if total > 0 else 1
        sub_bar.n = current if total > 0 else 0
        sub_bar.set_description(_sub_bar_label(
            sub_state["name"], stage, current, total, elapsed,
        ), refresh=True)
        if total > 0 and current >= total:
            sub_bar.n = sub_bar.total
            sub_bar.refresh()
        sub_bar.refresh()

    with tqdm(total=len(plan), desc="Building factors", unit="factor", position=0) as bar:
        for category, name in plan:
            action, reason = action_map[name]

            if action == "skip":
                bar.set_postfix_str(f"{name} (skip)")
                results.append(BuildResult(
                    factor_name=name,
                    factor_path=repo_temp.paths.factor_output_dir / f"{get_factor(name).name}.fea",
                    manifest_path=repo_temp.paths.manifest_output_dir / f"{get_factor(name).name}.json",
                    elapsed=0.0,
                    action="skip",
                ))
                bar.update(1)
                continue

            # Reset sub-bar for this factor
            sub_state["name"] = name
            sub_state["stage"] = "init"
            sub_state["current"] = 0
            sub_state["total"] = 0
            sub_state["t0"] = time.perf_counter()
            sub_bar.reset(total=1)
            sub_bar.set_description(
                _sub_bar_label(name, "init", 0, 1, 0), refresh=True
            )
            sub_bar.refresh()

            bar.set_postfix_str(f"{category}/{name} ({action})")
            repo = DataRepository(paths=paths, on_progress=_on_sub_progress)

            try:
                t0 = time.perf_counter()
                result = build_factor(
                    name,
                    repo=repo,
                    factor_start_date=reason if action == "incremental" else None,
                )
                results.append(result)
                elapsed = time.perf_counter() - t0
                sub_bar.n = sub_bar.total
                sub_bar.set_description(
                    _sub_bar_label(name, "done", sub_bar.total, sub_bar.total, elapsed),
                    refresh=True,
                )
                sub_bar.refresh()
            except Exception as e:
                logger.error(f"{name}: {e}")
                results.append(BuildResult(
                    factor_name=name,
                    factor_path=repo_temp.paths.factor_output_dir / f"{get_factor(name).name}.fea",
                    manifest_path=repo_temp.paths.manifest_output_dir / f"{get_factor(name).name}.json",
                    elapsed=0.0,
                    action="error",
                ))
                sub_bar.reset(total=0)
            bar.update(1)

    sub_bar.close()
    results.sort(key=lambda item: item.factor_name)
    return results


# ── Parallel build ──────────────────────────────────────────────────────────

def _build_factor_worker(
    name: str,
    project_root: str,
    source_root: str,
    shared_progress_state: Any | None,
    force: bool,
) -> dict[str, Any]:
    """Worker function for ProcessPoolExecutor.

    Returns a dict of results because BuildResult may not be picklable
    across processes if paths differ.
    """
    configure_paths(project_root=project_root, source_root=source_root)
    ensure_builtin_factors_loaded()

    paths = configure_paths()
    spec = get_factor(name)
    factor_path = paths.factor_output_dir / f"{spec.name}.fea"

    # Decide action
    action, reason = decide_build_action(
        name, factor_path, spec.dependencies, paths.source_root, force=force,
    )

    if action == "skip":
        return {
            "factor_name": name,
            "factor_path": str(factor_path),
            "manifest_path": str(paths.manifest_output_dir / f"{spec.name}.json"),
            "elapsed": 0.0,
            "action": "skip",
        }

    # Set up progress callback for financial panel building.
    # Each worker writes to its own named slot so the main process can
    # display one sub-bar per active worker.
    if shared_progress_state is not None:
        try:
            shared_progress_state[name] = {
                "stage": "init",
                "current": 0,
                "total": 0,
                "start": time.time(),
            }
        except Exception:
            pass

        def _on_progress(stage: str, current: int, total: int) -> None:
            try:
                entry = shared_progress_state.get(name, {})
                entry["stage"] = stage
                entry["current"] = current
                entry["total"] = total
                shared_progress_state[name] = entry
            except Exception:
                pass
    else:
        _on_progress = None

    repo = DataRepository(paths=paths, on_progress=_on_progress)

    # Compute context start date for date-aware loading
    _LOOKBACK = 252  # one calendar year, covers all rolling-window factors
    context_start: str | None = None
    if action == "incremental" and reason:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(reason, "%Y%m%d") - timedelta(days=_LOOKBACK)
        context_start = start_dt.strftime("%Y%m%d")

    # Determine the effective end date (6 PM cutoff + daily_adj max)
    _effective_end = _resolve_effective_end_date(paths.source_root)
    if action == "incremental" and reason and _effective_end < reason:
        _effective_end = reason

    t0 = time.perf_counter()
    error_msg = None
    factor_frame = None

    def _set_stage(stage: str) -> None:
        if shared_progress_state is not None:
            try:
                entry = shared_progress_state.get(name, {})
                entry["stage"] = stage
                entry["current"] = 0
                entry["total"] = 0
                shared_progress_state[name] = entry
            except Exception:
                pass

    try:
        context = FactorContext(repo=repo, start_date=context_start, end_date=_effective_end)
        _set_stage("computing")
        raw_output = spec.compute(context)
        factor_frame = ensure_single_factor_frame(raw_output, spec.name)

        if action == "incremental" and reason:
            date_level = factor_frame.index.get_level_values("Date")
            factor_frame = factor_frame.loc[date_level > reason]

        _set_stage("writing")
        if spec.category == "target":
            fp, mp = write_target(spec, factor_frame, paths=paths)
        elif action == "incremental":
            fp, mp = write_factor_incremental(spec, factor_frame, paths=paths)
        else:
            fp, mp = write_factor(spec, factor_frame, paths=paths)

        _write_done_marker(name, action, paths.manifest_output_dir)

    except Exception as e:
        error_msg = str(e)
        fp = paths.factor_output_dir / f"{spec.name}.fea"
        mp = paths.manifest_output_dir / f"{spec.name}.json"

    elapsed = time.perf_counter() - t0
    rows = len(factor_frame) if factor_frame is not None else 0
    nn_rows = int(factor_frame[spec.name].notna().sum()) if factor_frame is not None else 0

    if shared_progress_state is not None:
        try:
            entry = shared_progress_state.get(name, {})
            entry["stage"] = "done" if not error_msg else "error"
            entry["current"] = 1
            entry["total"] = 1
            shared_progress_state[name] = entry
        except Exception:
            pass

    # Write timing locally, will be collected by main process
    return {
        "factor_name": name,
        "factor_path": str(fp),
        "manifest_path": str(mp),
        "elapsed": elapsed,
        "action": action if not error_msg else "error",
        "category": spec.category,
        "rows": rows,
        "non_null_rows": nn_rows,
        "error": error_msg,
    }


def build_many_parallel(
    names: list[str],
    max_workers: int | None = None,
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> list[BuildResult]:
    """Build factors in parallel with sub-progress display."""
    if not names:
        return []

    ensure_builtin_factors_loaded()
    configured_paths = paths or configure_paths()
    max_workers = max_workers or recommend_worker_count()
    plan = _flatten_build_plan(names)

    # Pre-classify actions
    action_map: dict[str, tuple[str, str | None]] = {}
    for _, name in plan:
        spec = get_factor(name)
        factor_path = configured_paths.factor_output_dir / f"{spec.name}.fea"
        action, reason = decide_build_action(
            name, factor_path, spec.dependencies, configured_paths.source_root, force=force,
        )
        action_map[name] = (action, reason)

    skipped = sum(1 for a, _ in action_map.values() if a == "skip")
    incr = sum(1 for a, _ in action_map.values() if a == "incremental")
    rebuild = sum(1 for a, _ in action_map.values() if a == "rebuild")
    active = sum(1 for a, _ in action_map.values() if a != "skip")
    logger.info(f"Build plan: {rebuild} rebuild | {incr} incremental | {skipped} skip")

    if active == 0:
        logger.info("All factors up to date, nothing to build.")
        return [
            BuildResult(
                factor_name=name,
                factor_path=configured_paths.factor_output_dir / f"{get_factor(name).name}.fea",
                manifest_path=configured_paths.manifest_output_dir / f"{get_factor(name).name}.json",
                elapsed=0.0,
                action="skip",
            )
            for _, name in plan
        ]

    # Shared state for per-worker sub-progress.
    # Each worker writes to shared_state[factor_name] = {stage, current, total, start}.
    import multiprocessing
    manager = multiprocessing.Manager()
    shared_state = manager.dict()

    active_names = [name for _, name in plan if action_map[name][0] != "skip"]

    results: list[BuildResult] = []

    if active_names:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _build_factor_worker,
                    name,
                    str(configured_paths.project_root),
                    str(configured_paths.source_root),
                    shared_state,
                    force,
                ): name
                for name in active_names
            }

            # One sub-bar per worker slot so every concurrent worker is visible.
            num_slots = min(max_workers, len(active_names))
            sub_bars: list[tqdm] = []
            for i in range(num_slots):
                sub_bars.append(tqdm(
                    total=1, position=i + 1, desc=f"  [{i + 1}] ---",
                    unit="step", leave=False,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}",
                ))

            slot_factor: dict[int, str | None] = {i: None for i in range(num_slots)}
            factor_slot: dict[str, int] = {}
            available_slots: set[int] = set(range(num_slots))

            with tqdm(total=len(futures), desc="Building factors", unit="factor", position=0) as bar:
                pending = set(futures.keys())
                last_sub_display = time.time()

                while pending:
                    now = time.time()
                    if now - last_sub_display > 0.5:
                        # ── Assign slots to newly-seen factors ──
                        try:
                            st = dict(shared_state)
                        except Exception:
                            st = {}

                        for key, info in st.items():
                            if not isinstance(info, dict):
                                continue
                            f_name = key
                            if f_name in factor_slot:
                                continue
                            stage = info.get("stage", "")
                            if stage in ("done", "error"):
                                continue
                            if available_slots:
                                slot = min(available_slots)
                                available_slots.discard(slot)
                                slot_factor[slot] = f_name
                                factor_slot[f_name] = slot

                        # ── Update each sub-bar ──
                        for slot, sub_bar in enumerate(sub_bars):
                            f_name = slot_factor[slot]
                            if f_name is None:
                                sub_bar.set_description(f"  [{slot + 1}] ---", refresh=True)
                                sub_bar.n = 0
                                sub_bar.total = 1
                                sub_bar.refresh()
                                continue

                            info = st.get(f_name, {})
                            stage = info.get("stage", "---")
                            cur = info.get("current", 0)
                            tot = info.get("total", 0)
                            t_start = info.get("start", 0)
                            elapsed = (now - t_start) if t_start else 0

                            if stage in ("done", "error"):
                                sub_bar.n = sub_bar.total
                            elif tot > 0:
                                sub_bar.total = tot
                                sub_bar.n = min(cur, tot)
                            else:
                                sub_bar.total = 1
                                sub_bar.n = 0

                            sub_bar.set_description(
                                _sub_bar_label(f_name, stage, cur, tot, elapsed),
                                refresh=True,
                            )
                            sub_bar.refresh()

                        n_pending = len(pending)
                        bar.set_postfix_str(f"remaining: {n_pending}")
                        last_sub_display = now

                    # ── Collect completed futures ──
                    done = {f for f in pending if f.done()}
                    for fut in done:
                        pending.discard(fut)
                        name = futures[fut]
                        try:
                            worker_result = fut.result()
                        except Exception as e:
                            worker_result = {
                                "factor_name": name,
                                "factor_path": "",
                                "manifest_path": "",
                                "elapsed": 0.0,
                                "action": "error",
                                "category": "",
                                "rows": 0,
                                "non_null_rows": 0,
                                "error": str(e),
                            }

                        wr = worker_result
                        elapsed = wr.get("elapsed", 0.0)

                        results.append(BuildResult(
                            factor_name=wr["factor_name"],
                            factor_path=Path(wr.get("factor_path", "")),
                            manifest_path=Path(wr.get("manifest_path", "")),
                            elapsed=elapsed,
                            action=wr.get("action", "error"),
                        ))
                        bar.update(1)

                        # Free the worker's sub-bar slot
                        if name in factor_slot:
                            slot = factor_slot.pop(name)
                            slot_factor[slot] = None
                            available_slots.add(slot)
                            sub_bars[slot].set_description(
                                _sub_bar_label(name, "done", 1, 1, elapsed),
                                refresh=True,
                            )
                            sub_bars[slot].n = sub_bars[slot].total
                            sub_bars[slot].refresh()

                        # Mark done in shared state so slot assignment skips it
                        try:
                            shared_state[name] = {"stage": "done", "current": 1, "total": 1, "start": 0}
                        except Exception:
                            pass

                    if pending:
                        import time as _time
                        _time.sleep(0.1)

                for sub_bar in sub_bars:
                    sub_bar.close()

    # Add skipped results
    for _, name in plan:
        if action_map[name][0] == "skip":
            spec = get_factor(name)
            results.append(BuildResult(
                factor_name=name,
                factor_path=configured_paths.factor_output_dir / f"{spec.name}.fea",
                manifest_path=configured_paths.manifest_output_dir / f"{spec.name}.json",
                elapsed=0.0,
                action="skip",
            ))

    results.sort(key=lambda item: item.factor_name)
    return results


# ── Main entry ──────────────────────────────────────────────────────────────

def build_all(
    *,
    factor_names: list[str] | None = None,
    max_workers: int | None = None,
    sequential: bool = False,
    project_root: str | Path | None = None,
    source_root: str | Path | None = None,
    force: bool = False,
) -> list[BuildResult]:
    configured_paths = configure_paths(
        project_root=project_root, source_root=source_root,
    )
    ensure_builtin_factors_loaded()

    selected_factor_names = factor_names or [spec.name for spec in list_factors()]
    selected_factor_names = sorted(set(selected_factor_names))

    results: list[BuildResult]
    if sequential or len(selected_factor_names) == 1:
        results = build_many(selected_factor_names, paths=configured_paths, force=force)
    else:
        results = build_many_parallel(
            selected_factor_names, max_workers=max_workers, paths=configured_paths, force=force,
        )

    # Summary
    actions = {}
    for r in results:
        actions[r.action] = actions.get(r.action, 0) + 1
    logger.info(f"Build summary: {actions}")

    slowest = sorted(
        [r for r in results if r.elapsed > 0],
        key=lambda r: r.elapsed, reverse=True,
    )[:5]
    if slowest:
        logger.info("Slowest factors:")
        for r in slowest:
            logger.info(f"  {r.factor_name:<35} {r.elapsed:>8.1f}s  ({r.action})")

    # Print factor date report after every build
    date_info = check_factor_dates(paths=configured_paths)
    if date_info:
        counts = {"ok": 0, "stale": 0, "future": 0, "error": 0}
        effective_end = next(iter(date_info.values()))["effective_end"]
        print(f"\n{'='*64}")
        print(f"Factor last-date report  (effective end: {effective_end})")
        print(f"{'='*64}")
        print(f"{'Factor':<35} {'Last date':>10}  Status")
        print(f"{'-'*35} {'-'*10}  {'-'*6}")
        for name, info in date_info.items():
            last = info["last_date"] or "---"
            status = info["status"]
            marker = {"ok": "", "stale": "!", "future": ">>", "error": "ERR"}.get(status, "?")
            print(f"{name:<35} {last:>10}  {marker:<4} {status}")
            counts[status] = counts.get(status, 0) + 1
        print(f"{'='*64}")
        total = sum(counts.values())
        print(f"Total: {total}  |  ok: {counts['ok']}  stale: {counts['stale']}  "
              f"future: {counts['future']}  error: {counts['error']}")

    return results
