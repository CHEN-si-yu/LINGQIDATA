"""
Batch factor builder: groups factors by shared source-file dependencies,
loads each source file once, and computes all dependent factors from memory.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

from .builder import (
    BuildResult,
    _resolve_effective_end_date,
    _write_done_marker,
    decide_build_action,
    factor_names_by_category,
    recommend_worker_count,
)
from .dataset import DataRepository, _filter_by_date_range
from .factor_loader import ensure_builtin_factors_loaded
from .registry import FactorSpec, get_factor
from .settings import ProjectPaths, configure_paths
from .storage import (
    ensure_single_factor_frame,
    write_factor,
    write_factor_incremental,
    write_target,
)

logger = logging.getLogger(__name__)

_LOOKBACK = 252

# ── Metadata column exclusion for financial file schema discovery ─────────────

_FINANCIAL_EXCLUDE_COLS = {
    "stock_code", "ann_date", "f_ann_date", "end_date",
    "report_type", "comp_type", "end_type", "update_time",
    "name", "stock_name",
}

# Relative paths known to be report-frequency (loaded via load_financial).
_REPORT_FREQUENCY_FILES: set[str] = {
    "financial_indicator.parquet",
    "balancesheet.parquet",
    "income.parquet",
    "cashflow.parquet",
    "holder_number.parquet",
    "pledge_stat.parquet",
}

# Files that appear in dependencies but are loaded through DataRepository
# methods (load_industry_map, load_stock_pool) rather than context.load().
# They should not be pre-loaded as daily panels.
_REPO_METHOD_FILES: set[str] = {
    "stock_list.parquet",
    "ths_constituent_stocks.parquet",
    "ths_sector_categories.parquet",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency classification
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_factor_dependencies(spec: FactorSpec) -> tuple[set[str], set[tuple[str, str]]]:
    """Classify a factor's dependencies into daily vs financial loads.

    Returns:
        daily_paths: paths loaded via ``context.load()``
        financial_paths: (path, date_col) pairs loaded via ``context.load_financial()``
    """
    daily_paths: set[str] = set()
    financial_paths: set[tuple[str, str]] = set()

    # Fast path: classify by known file type
    for dep in spec.dependencies:
        if dep in _REPORT_FREQUENCY_FILES:
            financial_paths.add((dep, "ann_date"))
        elif dep == "calendar.parquet" or dep in _REPO_METHOD_FILES:
            continue
        else:
            daily_paths.add(dep)

    # Refine with source inspection for date_col overrides and mixed patterns
    try:
        source = inspect.getsource(spec.compute)
    except (OSError, TypeError):
        return daily_paths, financial_paths

    # Extract context.load("path") calls
    source_daily = set(re.findall(r'context\.load\s*\(\s*"([^"]+)"', source))
    daily_paths.update(source_daily)

    # Extract context.load_financial(...) calls with optional date_col
    for m in re.finditer(r'context\.load_financial\s*\(([^)]+)\)', source):
        args = m.group(1)
        path_m = re.search(r'"([^"]+)"', args)
        if not path_m:
            continue
        path = path_m.group(1)
        dc_m = re.search(r'date_col\s*=\s*"([^"]+)"', args)
        date_col = dc_m.group(1) if dc_m else "ann_date"
        financial_paths.add((path, date_col))
        daily_paths.discard(path)

    # Only keep paths declared in the factor's dependencies
    dep_set = set(spec.dependencies)
    daily_paths &= dep_set
    financial_paths = {(p, dc) for (p, dc) in financial_paths if p in dep_set}

    # If source inspection found an explicit date_col, remove the generic
    # ann_date default (the fast-path may have added it before inspection)
    override_paths = {p for p, dc in financial_paths if dc != "ann_date"}
    financial_paths = {
        (p, dc) for (p, dc) in financial_paths
        if not (dc == "ann_date" and p in override_paths)
    }

    return daily_paths, financial_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Grouping
# ═══════════════════════════════════════════════════════════════════════════════

def _group_by_dependencies(names: list[str]) -> list[list[FactorSpec]]:
    """Group factor names by their dependencies tuple, sorted largest-first."""
    groups: dict[tuple[str, ...], list[FactorSpec]] = {}
    for name in names:
        spec = get_factor(name)
        groups.setdefault(spec.dependencies, []).append(spec)

    for specs in groups.values():
        specs.sort(key=lambda s: s.name)

    return [
        specs
        for _deps, specs in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0]),
        )
    ]


def _group_by_source_file(names: list[str]) -> list[list[FactorSpec]]:
    """Group factor names by their source .py file, sorted largest-first."""
    groups: dict[str, list[FactorSpec]] = {}
    for name in names:
        spec = get_factor(name)
        src_file = inspect.getfile(spec.compute)
        groups.setdefault(src_file, []).append(spec)

    for specs in groups.values():
        specs.sort(key=lambda s: s.name)

    return [
        specs
        for _file, specs in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0]),
        )
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared data preloading
# ═══════════════════════════════════════════════════════════════════════════════

def _preload_group_data(
    specs: list[FactorSpec],
    repo: DataRepository,
    context_start: str | None,
    effective_end: str,
    source_root: Path,
) -> dict[str, pd.DataFrame]:
    """Pre-load all shared data for a group of factors.

    Returns a dict mapping cache keys to DataFrames:
      - Daily panels keyed by relative path (e.g. ``"daily_adj.parquet"``)
      - Financial panels keyed by ``"__financial__{path}_{date_col}"``
    """
    all_daily: set[str] = set()
    all_financial: set[tuple[str, str]] = set()

    for spec in specs:
        daily_paths, fin_paths = _classify_factor_dependencies(spec)
        all_daily.update(daily_paths)
        all_financial.update(fin_paths)

    shared: dict[str, pd.DataFrame] = {}

    # Daily panels
    for path in sorted(all_daily):
        logger.debug("Pre-loading daily: %s", path)
        shared[path] = repo.load_panel(
            path,
            min_date=context_start,
            max_date=effective_end,
        )

    # Financial panels — load all value columns in one forward-fill pass
    for path, date_col in sorted(all_financial):
        fpath = source_root / path
        try:
            schema = pq.read_schema(fpath)
            value_cols = [
                c for c in schema.names
                if c not in _FINANCIAL_EXCLUDE_COLS
            ]
        except Exception:
            value_cols = None

        cache_key = f"__financial__{path}_{date_col}"
        logger.debug(
            "Pre-loading financial: %s date_col=%s %d cols",
            path, date_col, len(value_cols) if value_cols else 0,
        )
        shared[cache_key] = repo.load_financial_panel(
            path,
            value_cols=value_cols,
            date_col=date_col,
            min_date=context_start,
            max_date=effective_end,
        )

    return shared


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# BatchFactorContext
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchFactorContext:
    """A FactorContext-compatible object that serves pre-loaded data.

    Implements the same ``.load()`` / ``.load_financial()`` interface so
    all existing factor functions work without modification.

    Also exposes ``.repo`` for factors that access DataRepository methods
    directly (e.g. load_industry_map, load_stock_pool).
    """
    _shared_data: dict[str, pd.DataFrame]
    repo: DataRepository | None = None
    start_date: str | None = None
    end_date: str | None = None
    _lookback_days: int = _LOOKBACK

    def load(self, relative_path: str) -> pd.DataFrame:
        df = self._shared_data.get(relative_path)
        if df is not None:
            return _filter_by_date_range(
                df, self.start_date, self.end_date, self._lookback_days,
            )
        raise KeyError(
            f"BatchFactorContext: '{relative_path}' not pre-loaded. "
            f"Available: {list(self._shared_data.keys())}"
        )

    def load_financial(
        self,
        relative_path: str,
        value_cols: list[str] | None = None,
        date_col: str = "ann_date",
    ) -> pd.DataFrame:
        cache_key = f"__financial__{relative_path}_{date_col}"
        df = self._shared_data.get(cache_key)
        if df is not None:
            df = _filter_by_date_range(
                df, self.start_date, self.end_date, self._lookback_days,
            )
            if value_cols is not None:
                available = [c for c in value_cols if c in df.columns]
                if len(available) != len(value_cols):
                    missing = set(value_cols) - set(df.columns)
                    logger.warning(
                        "Columns not found in pre-loaded %s: %s",
                        relative_path, missing,
                    )
                if not available:
                    return pd.DataFrame(index=df.index)
                return df[available]
            return df
        raise KeyError(
            f"BatchFactorContext: financial '{relative_path}' (date_col={date_col}) "
            f"not pre-loaded. Available: {list(self._shared_data.keys())}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Group build
# ═══════════════════════════════════════════════════════════════════════════════

def build_factor_group(
    specs: list[FactorSpec],
    paths: ProjectPaths,
    force: bool = False,
    on_progress: Callable[[str, int, int], None] | None = None,
    parallel: bool = False,
    max_workers: int = 8,
) -> list[BuildResult]:
    """Build a group of factors with shared data loaded once.

    Shared source files are loaded once and all factors in the group are
    computed from the in-memory data. When *parallel* is True, factors are
    computed concurrently via ThreadPoolExecutor.
    """
    if not specs:
        return []

    results: list[BuildResult] = []
    effective_end = _resolve_effective_end_date(paths.source_root)

    # ── Decide action per factor ──
    action_map: dict[str, tuple[str, str | None]] = {}
    needs_data = False
    for spec in specs:
        factor_path = paths.factor_output_dir / f"{spec.name}.fea"
        action, reason = decide_build_action(
            spec.name, factor_path, spec.dependencies, paths.source_root,
            force=force,
        )
        action_map[spec.name] = (action, reason)
        if action != "skip":
            needs_data = True

    if not needs_data:
        logger.info(
            "Group %s: all %d factors up to date",
            specs[0].dependencies, len(specs),
        )
        for spec in specs:
            results.append(BuildResult(
                factor_name=spec.name,
                factor_path=paths.factor_output_dir / f"{spec.name}.fea",
                manifest_path=paths.manifest_output_dir / f"{spec.name}.json",
                elapsed=0.0,
                action="skip",
            ))
        return results

    active_specs = [s for s in specs if action_map[s.name][0] != "skip"]
    logger.info(
        "Group %s: %d/%d factors need build",
        specs[0].dependencies, len(active_specs), len(specs),
    )

    # ── Determine date range for data loading ──
    min_start_date: str | None = None
    for spec in active_specs:
        _action, reason = action_map[spec.name]
        if _action == "incremental" and reason:
            if min_start_date is None or reason < min_start_date:
                min_start_date = reason

    context_start: str | None = None
    if min_start_date:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(min_start_date, "%Y%m%d") - timedelta(days=_LOOKBACK)
        context_start = start_dt.strftime("%Y%m%d")

    # ── Pre-load shared data ──
    repo = DataRepository(paths=paths, on_progress=on_progress)
    shared_data = _preload_group_data(
        specs, repo, context_start, effective_end, paths.source_root,
    )

    # ── Compute each factor ──
    if parallel:
        _compute_group_parallel(
            specs, action_map, shared_data, effective_end, context_start,
            paths, results, max_workers=max_workers, repo=repo,
        )
    else:
        _compute_group_sequential(
            specs, action_map, shared_data, effective_end, context_start,
            paths, results, repo=repo,
        )

    return results


def _compute_group_sequential(
    specs: list[FactorSpec],
    action_map: dict[str, tuple[str, str | None]],
    shared_data: dict[str, pd.DataFrame],
    effective_end: str,
    context_start: str | None,
    paths: ProjectPaths,
    results: list[BuildResult],
    repo: DataRepository | None = None,
) -> None:
    """Compute factors one by one using shared pre-loaded data."""
    for spec in specs:
        action, reason = action_map[spec.name]
        t0 = time.perf_counter()

        if action == "skip":
            results.append(BuildResult(
                factor_name=spec.name,
                factor_path=paths.factor_output_dir / f"{spec.name}.fea",
                manifest_path=paths.manifest_output_dir / f"{spec.name}.json",
                elapsed=0.0,
                action="skip",
            ))
            continue

        try:
            context = BatchFactorContext(
                _shared_data=shared_data,
                repo=repo,
                start_date=context_start,
                end_date=effective_end,
            )
            raw_output = spec.compute(context)
            factor_frame = ensure_single_factor_frame(raw_output, spec.name)

            if action == "incremental" and reason:
                factor_frame = factor_frame.loc[factor_frame.index > reason]

            rows = len(factor_frame)
            nn_rows = int(factor_frame.notna().sum().sum())

            if spec.category == "target":
                factor_path, manifest_path = write_target(spec, factor_frame, paths=paths)
            elif action == "incremental":
                factor_path, manifest_path = write_factor_incremental(spec, factor_frame, paths=paths)
            else:
                factor_path, manifest_path = write_factor(spec, factor_frame, paths=paths)

            _write_done_marker(spec.name, action, paths.manifest_output_dir)

            elapsed = time.perf_counter() - t0
            results.append(BuildResult(
                factor_name=spec.name,
                factor_path=factor_path,
                manifest_path=manifest_path,
                elapsed=elapsed,
                action=action,
                rows=rows,
                non_null_rows=nn_rows,
            ))
            logger.info("  %-35s %-15s %8.1fs %5d rows", spec.name, action, elapsed, rows)

        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception("%s: failed", spec.name)
            results.append(BuildResult(
                factor_name=spec.name,
                factor_path=paths.factor_output_dir / f"{spec.name}.fea",
                manifest_path=paths.manifest_output_dir / f"{spec.name}.json",
                elapsed=elapsed,
                action="error",
            ))


def _compute_group_parallel(
    specs: list[FactorSpec],
    action_map: dict[str, tuple[str, str | None]],
    shared_data: dict[str, pd.DataFrame],
    effective_end: str,
    context_start: str | None,
    paths: ProjectPaths,
    results: list[BuildResult],
    max_workers: int = 8,
    repo: DataRepository | None = None,
) -> None:
    """Compute factors in parallel using shared pre-loaded data.

    Uses ThreadPoolExecutor so all threads can access the same in-memory
    shared_data dict without serialization overhead.
    """
    active_items: list[tuple[FactorSpec, str, str | None]] = []
    for spec in specs:
        action, reason = action_map[spec.name]
        if action == "skip":
            results.append(BuildResult(
                factor_name=spec.name,
                factor_path=paths.factor_output_dir / f"{spec.name}.fea",
                manifest_path=paths.manifest_output_dir / f"{spec.name}.json",
                elapsed=0.0,
                action="skip",
            ))
        else:
            active_items.append((spec, action, reason))

    if not active_items:
        return

    def _compute_one(spec: FactorSpec, action: str, reason: str | None) -> BuildResult:
        t0 = time.perf_counter()
        try:
            context = BatchFactorContext(
                _shared_data=shared_data,
                repo=repo,
                start_date=context_start,
                end_date=effective_end,
            )
            raw_output = spec.compute(context)
            factor_frame = ensure_single_factor_frame(raw_output, spec.name)

            if action == "incremental" and reason:
                factor_frame = factor_frame.loc[factor_frame.index > reason]

            rows = len(factor_frame)
            nn_rows = int(factor_frame.notna().sum().sum())

            if spec.category == "target":
                factor_path, manifest_path = write_target(spec, factor_frame, paths=paths)
            elif action == "incremental":
                factor_path, manifest_path = write_factor_incremental(spec, factor_frame, paths=paths)
            else:
                factor_path, manifest_path = write_factor(spec, factor_frame, paths=paths)

            _write_done_marker(spec.name, action, paths.manifest_output_dir)

            elapsed = time.perf_counter() - t0
            logger.info("  %-35s %-15s %8.1fs %5d rows", spec.name, action, elapsed, rows)
            return BuildResult(
                factor_name=spec.name,
                factor_path=factor_path,
                manifest_path=manifest_path,
                elapsed=elapsed,
                action=action,
                rows=rows,
                non_null_rows=nn_rows,
            )
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception("%s: failed", spec.name)
            return BuildResult(
                factor_name=spec.name,
                factor_path=paths.factor_output_dir / f"{spec.name}.fea",
                manifest_path=paths.manifest_output_dir / f"{spec.name}.json",
                elapsed=elapsed,
                action="error",
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_one, spec, action, reason): spec
            for spec, action, reason in active_items
        }
        for future in as_completed(futures):
            results.append(future.result())


# ═══════════════════════════════════════════════════════════════════════════════
# Sequential batch entry
# ═══════════════════════════════════════════════════════════════════════════════

def build_many_batched(
    names: list[str],
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> list[BuildResult]:
    """Build factors in dependency-group batches, sequentially.

    Groups are processed largest-first.  Shared data is loaded once per
    group and released before moving to the next group.
    """
    if not names:
        return []

    ensure_builtin_factors_loaded()
    paths = paths or configure_paths()
    groups = _group_by_dependencies(names)

    total_factors = sum(len(g) for g in groups)
    logger.info(
        "Batch build: %d factors in %d groups (sequential)",
        total_factors, len(groups),
    )

    all_results: list[BuildResult] = []
    for i, group_specs in enumerate(groups):
        deps = group_specs[0].dependencies
        logger.info(
            "[%d/%d] Group %s: %d factors",
            i + 1, len(groups), deps, len(group_specs),
        )
        group_results = build_factor_group(group_specs, paths, force=force)
        all_results.extend(group_results)

    all_results.sort(key=lambda r: r.factor_name)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# File-based batch entry (serial across files, parallel within)
# ═══════════════════════════════════════════════════════════════════════════════

def build_many_by_file(
    names: list[str],
    paths: ProjectPaths | None = None,
    force: bool = False,
    max_workers: int = 8,
    sequential: bool = False,
) -> list[BuildResult]:
    """Build factors grouped by source .py file.

    Files are processed serially — one file's factors are finished before
    the next file starts.  Factors within the same file are computed in
    parallel via ThreadPoolExecutor using shared pre-loaded data.

    When *sequential* is True, factors within each file are also computed
    one by one.
    """
    if not names:
        return []

    ensure_builtin_factors_loaded()
    paths = paths or configure_paths()
    groups = _group_by_source_file(names)

    total_factors = sum(len(g) for g in groups)
    logger.info(
        "File build: %d factors in %d source files (serial across files, %s within)",
        total_factors, len(groups),
        "sequential" if sequential else f"parallel x{max_workers}",
    )

    all_results: list[BuildResult] = []
    for i, group_specs in enumerate(groups):
        src_file = inspect.getfile(group_specs[0].compute)
        fname = Path(src_file).name
        print(f"\n  [{i + 1}/{len(groups)}] {fname} ({len(group_specs)} factors)")
        logger.info(
            "[%d/%d] File %s: %d factors",
            i + 1, len(groups), fname, len(group_specs),
        )
        group_results = build_factor_group(
            group_specs, paths, force=force,
            parallel=not sequential,
            max_workers=max_workers,
        )
        all_results.extend(group_results)

    all_results.sort(key=lambda r: r.factor_name)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel batch entry (ProcessPoolExecutor across groups)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_factor_group_worker(
    group_names: list[str],
    project_root_str: str,
    source_root_str: str,
    force: bool,
) -> dict:
    """Worker for ProcessPoolExecutor (module-level, picklable)."""
    configure_paths(project_root=project_root_str, source_root=source_root_str)
    ensure_builtin_factors_loaded()
    paths = configure_paths()

    specs = [get_factor(name) for name in group_names]
    results = build_factor_group(specs, paths, force=force)

    return {
        "results": [
            {
                "factor_name": r.factor_name,
                "factor_path": str(r.factor_path),
                "manifest_path": str(r.manifest_path),
                "elapsed": r.elapsed,
                "action": r.action,
                "rows": r.rows,
                "non_null_rows": r.non_null_rows,
            }
            for r in results
        ],
        "error": None,
    }


def build_many_batched_parallel(
    names: list[str],
    max_workers: int | None = None,
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> list[BuildResult]:
    """Build factors in dependency-group batches, dispatching groups to a process pool.

    Each group loads its shared data once within its worker process, then
    computes all factors sequentially within that process.
    """
    if not names:
        return []

    ensure_builtin_factors_loaded()
    paths = paths or configure_paths()
    max_workers = max_workers or recommend_worker_count()
    groups = _group_by_dependencies(names)

    # Pre-classify all actions
    action_map: dict[str, tuple[str, str | None]] = {}
    for group_specs in groups:
        for spec in group_specs:
            factor_path = paths.factor_output_dir / f"{spec.name}.fea"
            action, reason = decide_build_action(
                spec.name, factor_path, spec.dependencies, paths.source_root,
                force=force,
            )
            action_map[spec.name] = (action, reason)

    # Separate active groups from all-skip groups
    active_groups: list[list[FactorSpec]] = []
    skipped_results: list[BuildResult] = []

    for group_specs in groups:
        if any(action_map[s.name][0] != "skip" for s in group_specs):
            active_groups.append(group_specs)
        else:
            for spec in group_specs:
                skipped_results.append(BuildResult(
                    factor_name=spec.name,
                    factor_path=paths.factor_output_dir / f"{spec.name}.fea",
                    manifest_path=paths.manifest_output_dir / f"{spec.name}.json",
                    elapsed=0.0,
                    action="skip",
                ))

    total = sum(len(g) for g in groups)
    active_count = sum(len(g) for g in active_groups)
    skip_count = len(skipped_results)
    logger.info(
        "Batch build: %d factors in %d groups (%d active, %d skip, %d workers)",
        total, len(groups), active_count, skip_count, max_workers,
    )

    if not active_groups:
        logger.info("All factors up to date, nothing to build.")
        return skipped_results

    results: list[BuildResult] = list(skipped_results)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _build_factor_group_worker,
                [s.name for s in group_specs],
                str(paths.project_root),
                str(paths.source_root),
                force,
            ): group_specs[0].dependencies
            for group_specs in active_groups
        }

        for future in as_completed(futures):
            deps = futures[future]
            try:
                worker_result = future.result()
                if worker_result.get("error"):
                    logger.error("Group %s failed: %s", deps, worker_result["error"])
                    continue
                for rd in worker_result.get("results", []):
                    results.append(BuildResult(
                        factor_name=rd["factor_name"],
                        factor_path=Path(rd["factor_path"]),
                        manifest_path=Path(rd["manifest_path"]),
                        elapsed=rd["elapsed"],
                        action=rd["action"],
                        rows=rd.get("rows", 0),
                        non_null_rows=rd.get("non_null_rows", 0),
                    ))
            except BrokenProcessPool:
                logger.error("Group %s: process pool broken (OOM kill likely) — "
                             "cancelling remaining groups", deps)
                for f in futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass
                break
            except Exception:
                logger.exception("Group %s: worker failed", deps)

    results.sort(key=lambda r: r.factor_name)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry (compatible with builder.build_all)
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_batched(
    *,
    factor_names: list[str] | None = None,
    max_workers: int | None = None,
    sequential: bool = False,
    project_root: str | Path | None = None,
    source_root: str | Path | None = None,
    force: bool = False,
) -> list[BuildResult]:
    """Build all factors grouped by source file.

    Files are processed serially; factors within each file are computed
    in parallel via ThreadPoolExecutor using shared pre-loaded data.

    Signature is compatible with ``builder.build_all()``.
    """
    configured_paths = configure_paths(
        project_root=project_root, source_root=source_root,
    )
    ensure_builtin_factors_loaded()

    from .registry import FACTOR_REGISTRY as _REG
    selected = factor_names or sorted(_REG.keys())
    selected = sorted(set(selected))

    return build_many_by_file(
        selected, paths=configured_paths, force=force,
        max_workers=max_workers or 8,
        sequential=sequential,
    )
