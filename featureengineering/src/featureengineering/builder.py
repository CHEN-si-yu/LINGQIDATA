from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from tqdm import tqdm

from .dataset import DataRepository
from .factor_loader import ensure_builtin_factors_loaded
from .progress import (
    SubProgress,
    TimingCollector,
    get_timing_collector,
    FactorTiming,
)
from .registry import FACTOR_REGISTRY, FactorContext, FactorSpec, get_factor
from .settings import ProjectPaths, configure_paths
from .storage import (
    ensure_single_factor_frame,
    write_factor,
    write_factor_incremental,
    write_target,
)

TODAY = date.today().strftime("%Y-%m-%d")


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


def _detect_date_col(columns: list[str]) -> str | None:
    for c in _DATE_CANDIDATES:
        if c in columns:
            return c
    return None


def _read_source_max_date(filepath: Path) -> str | None:
    """Read the maximum date from a parquet source file."""
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
            return max_val.strftime("%Y-%m-%d")
        return str(max_val)[:10]
    except Exception:
        return None


def _read_factor_max_date(factor_path: Path) -> str | None:
    """Read the maximum Date index value from a .fea factor file."""
    if not factor_path.exists():
        return None
    try:
        df = pq.read_table(factor_path).to_pandas()
    except Exception:
        # feather format fallback
        import pandas as pd
        try:
            df = pd.read_feather(factor_path)
        except Exception:
            return None
    if df.empty:
        return None
    idx = df.columns[0] if not df.index.names or df.index.names[0] is None else None
    if idx is not None and ("Date" in df.columns or "date" in df.columns):
        date_col = "Date" if "Date" in df.columns else "date"
        max_val = df[date_col].max()
        return str(max_val)[:10] if max_val is not None else None
    # MultiIndex — get max from Date level
    if isinstance(df.index, pd.core.indexes.multi.MultiIndex):
        for name in df.index.names:
            if name and name.lower() == "date":
                vals = df.index.get_level_values(name)
                max_val = vals.max()
                return str(max_val)[:10] if max_val is not None else None
    return None


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

    # Get source data max dates
    source_dates = _check_source_dates(deps, source_root)
    source_max = "0000-00-00"
    for dep, max_d in source_dates.items():
        if max_d is None:
            continue
        if max_d > source_max:
            source_max = max_d

    factor_max = _read_factor_max_date(factor_path)

    if factor_max is None:
        return "rebuild", "cannot determine factor max date"

    if factor_max >= source_max and factor_max >= TODAY:
        return "skip", f"factor max date {factor_max} >= source max {source_max}"

    if factor_max >= source_max:
        return "skip", f"factor max {factor_max} >= source max {source_max}"

    # Factor is behind — need incremental rebuild
    return "incremental", factor_max


# ── Listing ─────────────────────────────────────────────────────────────────

def list_factors() -> list[FactorSpec]:
    return [FACTOR_REGISTRY[name] for name in sorted(FACTOR_REGISTRY)]


def recommend_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count >= 8:
        return min(4, cpu_count)
    return max(1, cpu_count // 2)


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
    timing = get_timing_collector()

    t0 = time.perf_counter()
    error_msg: str | None = None
    factor_frame = None

    try:
        context = FactorContext(repo=repo)
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

    except Exception as e:
        error_msg = str(e)
        factor_path = repo.paths.factor_output_dir / f"{spec.name}.fea"
        manifest_path = repo.paths.manifest_output_dir / f"{spec.name}.json"
        raise

    finally:
        elapsed = time.perf_counter() - t0
        rows = len(factor_frame) if factor_frame is not None else 0
        nn_rows = int(factor_frame[spec.name].notna().sum()) if factor_frame is not None else 0
        timing.add(FactorTiming(
            name=spec.name,
            category=spec.category,
            start=t0,
            end=t0 + elapsed,
            elapsed=elapsed,
            rows=rows,
            non_null_rows=nn_rows,
            error=error_msg,
        ))

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
        return f"  {name} [{stage}]"
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
    print(f"Build plan: {rebuild} rebuild | {incr} incremental | {skipped} skip")

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
                print(f"\n[ERROR] {name}: {e}")
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

    # Set up progress callback for financial panel building
    if shared_progress_state is not None:
        try:
            shared_progress_state["factor"] = name
            shared_progress_state["stage"] = "init"
            shared_progress_state["current"] = 0
            shared_progress_state["total"] = 0
            shared_progress_state["start"] = time.time()
        except Exception:
            pass

        def _on_progress(stage: str, current: int, total: int) -> None:
            try:
                shared_progress_state["factor"] = name
                shared_progress_state["stage"] = stage
                shared_progress_state["current"] = current
                shared_progress_state["total"] = total
            except Exception:
                pass
    else:
        _on_progress = None

    repo = DataRepository(paths=paths, on_progress=_on_progress)

    t0 = time.perf_counter()
    error_msg = None
    factor_frame = None

    try:
        context = FactorContext(repo=repo)
        raw_output = spec.compute(context)
        factor_frame = ensure_single_factor_frame(raw_output, spec.name)

        if action == "incremental" and reason:
            date_level = factor_frame.index.get_level_values("Date")
            factor_frame = factor_frame.loc[date_level > reason]

        if spec.category == "target":
            fp, mp = write_target(spec, factor_frame, paths=paths)
        elif action == "incremental":
            fp, mp = write_factor_incremental(spec, factor_frame, paths=paths)
        else:
            fp, mp = write_factor(spec, factor_frame, paths=paths)

    except Exception as e:
        error_msg = str(e)
        fp = paths.factor_output_dir / f"{spec.name}.fea"
        mp = paths.manifest_output_dir / f"{spec.name}.json"

    elapsed = time.perf_counter() - t0
    rows = len(factor_frame) if factor_frame is not None else 0
    nn_rows = int(factor_frame[spec.name].notna().sum()) if factor_frame is not None else 0

    if shared_progress_state is not None:
        try:
            shared_progress_state["stage"] = "done" if not error_msg else "error"
            shared_progress_state["current"] = 1
            shared_progress_state["total"] = 1
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
    print(f"Build plan: {rebuild} rebuild | {incr} incremental | {skipped} skip")

    if active == 0:
        print("All factors up to date, nothing to build.")
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

    # Shared state for sub-progress
    import multiprocessing
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    shared_state["factor"] = ""
    shared_state["stage"] = ""
    shared_state["current"] = 0
    shared_state["total"] = 0

    active_names = [name for _, name in plan if action_map[name][0] != "skip"]

    results: list[BuildResult] = []
    timing = get_timing_collector()

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

        # Dual-bar: main bar at position 0, sub-progress at position 1
        sub_bar = tqdm(
            total=1, position=1, desc="  Sub-progress",
            unit="step", leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}",
        )

        with tqdm(total=len(futures), desc="Building factors", unit="factor", position=0) as bar:
            pending = set(futures.keys())
            last_sub_display = time.time()
            prev_factor = ""
            sub_t0 = time.perf_counter()

            while pending:
                # Poll sub-progress every 0.5s
                now = time.time()
                if now - last_sub_display > 0.5:
                    try:
                        st = dict(shared_state)
                        f_name = st.get("factor", "")
                        stage = st.get("stage", "")
                        cur = st.get("current", 0)
                        tot = st.get("total", 0)
                        worker_start = st.get("start", 0)

                        if f_name:
                            elapsed = (now - worker_start) if worker_start else (now - sub_t0)

                            if stage in ("done", "error"):
                                sub_bar.total = 1
                                sub_bar.n = 1
                            elif tot > 0:
                                sub_bar.total = tot
                                sub_bar.n = min(cur, tot)
                            else:
                                # No detailed progress yet: show indeterminate pulse
                                sub_bar.total = 1
                                sub_bar.n = 0

                            sub_bar.set_description(
                                _sub_bar_label(f_name, stage, cur, tot, elapsed),
                                refresh=True,
                            )
                            n_pending = len(pending)
                            bar.set_postfix_str(f"active: {f_name} | remaining: {n_pending}")
                            sub_bar.refresh()
                    except Exception:
                        pass
                    last_sub_display = now

                # Check for completed futures
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

                    # Record timing
                    timing.add(FactorTiming(
                        name=wr["factor_name"],
                        category=wr.get("category", ""),
                        start=0, end=elapsed,
                        elapsed=elapsed,
                        rows=wr.get("rows", 0),
                        non_null_rows=wr.get("non_null_rows", 0),
                        error=wr.get("error"),
                    ))

                    results.append(BuildResult(
                        factor_name=wr["factor_name"],
                        factor_path=Path(wr.get("factor_path", "")),
                        manifest_path=Path(wr.get("manifest_path", "")),
                        elapsed=elapsed,
                        action=wr.get("action", "error"),
                    ))
                    bar.update(1)

                    # Mark sub-bar complete for this factor
                    if prev_factor == name:
                        sub_bar.n = sub_bar.total
                        sub_bar.set_description(
                            _sub_bar_label(name, "done", sub_bar.total, sub_bar.total, elapsed),
                            refresh=True,
                        )
                        sub_bar.refresh()

                if pending:
                    import time as _time
                    _time.sleep(0.1)

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

    # Save timing report
    timing_path = configured_paths.project_root / "data" / "factor_timing.json"
    get_timing_collector().save(timing_path)
    print(f"Timing report saved to {timing_path}")

    # Summary
    actions = {}
    for r in results:
        actions[r.action] = actions.get(r.action, 0) + 1
    print(f"Build summary: {actions}")

    slowest = sorted(
        [r for r in results if r.elapsed > 0],
        key=lambda r: r.elapsed, reverse=True,
    )[:5]
    if slowest:
        print("Slowest factors:")
        for r in slowest:
            print(f"  {r.factor_name:<35} {r.elapsed:>8.1f}s  ({r.action})")

    return results
