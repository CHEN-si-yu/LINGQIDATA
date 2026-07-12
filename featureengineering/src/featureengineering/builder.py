from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)

from .dataset import DataRepository
from .factor_loader import ensure_builtin_factors_loaded
from .report import print_post_build_report

from .registry import FACTOR_REGISTRY, FactorContext, FactorSpec, get_factor
from .settings import ProjectPaths, configure_paths
from .storage import (
    write_target_incremental,
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
    rows: int = 0
    non_null_rows: int = 0


CATEGORY_ORDER = [
    "price", "timeseries", "valuation", "quality", "event", "fund_flow",
    "financial", "sector", "neutral", "index", "target", "other",
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
    """Read the maximum Date index value from a single .fea file.

    Uses ``columns=[]`` to read only the index (no data columns),
    avoiding a full-file load for wide factor matrices.
    """
    if not filepath.exists():
        return None
    import pandas as _pd
    try:
        df = _pd.read_feather(filepath, columns=[])
    except Exception:
        try:
            df = _pd.read_feather(filepath)
        except Exception:
            return None
    if df.empty:
        return None
    # Wide format: Date index (single-level)
    if df.index.name is not None and df.index.name.lower() == "date":
        max_val = df.index.max()
        return _normalize_date(str(max_val))
    # Legacy MultiIndex format (Date, Code) or (Code, Date)
    if isinstance(df.index, pd.MultiIndex):
        for name in df.index.names:
            if name and name.lower() == "date":
                vals = df.index.get_level_values(name)
                max_val = vals.max()
                return _normalize_date(str(max_val)) if max_val is not None else None
    # Legacy flat format: Date in columns
    if "Date" in df.columns or "date" in df.columns:
        date_col = "Date" if "Date" in df.columns else "date"
        max_val = df[date_col].max()
        return _normalize_date(str(max_val)) if max_val is not None else None
    return None


def _read_factor_max_date(factor_base_path: Path) -> str | None:
    """Read the maximum Date from the factor file."""
    return _read_single_file_max_date(factor_base_path)


def _check_source_dates(dependencies: tuple[str, ...], source_root: Path) -> dict[str, str | None]:
    """Return {dep_filename: max_date_or_None} for each dependency."""
    result: dict[str, str | None] = {}
    for dep in dependencies:
        fpath = source_root / dep
        result[dep] = _read_source_max_date(fpath)
    return result


def _get_recent_trading_days(source_root: Path, n_days: int = 5, effective_end: str | None = None) -> list[str]:
    """Return the last *n_days* trading days (YYYYMMDD) before *effective_end*.

    Reads ``calendar.parquet`` and returns the most recent dates where
    ``is_open == 1``, sorted ascending.

    When *effective_end* is provided, only dates strictly before it are
    returned — *effective_end* itself is excluded because it is the target
    date for the upcoming incremental build, not a "past" trading day.
    """
    import pandas as pd

    cal_path = source_root / "calendar.parquet"
    if not cal_path.exists():
        return []

    try:
        cal = pd.read_parquet(cal_path, columns=["date", "is_open"])
    except Exception:
        return []

    trading = cal.loc[cal["is_open"].astype(bool), "date"]
    if trading.empty:
        return []

    # Normalise to YYYYMMDD and take the last N
    dates = (
        trading.astype(str)
        .str.replace("-", "", regex=False)
        .str.slice(0, 8)
        .sort_values()
        .unique()
        .tolist()
    )
    # 取 effective_end *之前* 的过去 N 个交易日（不包含 effective_end 本身）。
    # effective_end 是本次构建即将增量补充的目标日期，因子文件理所当然
    # 还不包含它 —— 若将其纳入质量检查的范围，会误判为 missing_dates
    # 从而触发不必要的全量重建。
    if effective_end is not None:
        dates = [d for d in dates if d < effective_end]
    return dates[-n_days:]


def _check_factor_recent_quality(
    factor_path: Path,
    source_root: Path,
    n_days: int = 5,
    effective_end: str | None = None,
) -> str:
    """Check whether recent trading days in a .fea file have valid data.

    Only trading days **before** *effective_end* are checked — the factor
    is not expected to contain *effective_end* itself because that is the
    very date the upcoming incremental build will add.

    Returns one of:

    - ``"ok"`` — all *n_days* past trading days are present and have at
      least some non-NaN values.
    - ``"all_nan"`` — the past trading days exist in the file but every
      value across all stocks is NaN (upstream data gap).
    - ``"missing_dates"`` — at least one of the past trading days is
      absent from the file entirely.
    - ``"no_calendar"`` — the calendar file is unavailable; cannot check.
    - ``"unreadable"`` — the .fea file exists but could not be read.
    """
    if not factor_path.exists():
        return "missing"

    recent = _get_recent_trading_days(source_root, n_days, effective_end=effective_end)
    if not recent:
        return "no_calendar"

    import pandas as pd

    try:
        df = pd.read_feather(factor_path)
    except Exception:
        return "unreadable"

    # Normalise the index to YYYYMMDD strings for comparison
    idx = df.index.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

    present = [d for d in recent if d in idx.values]
    if len(present) < len(recent):
        return "missing_dates"

    # For the dates that are present, check whether ALL values are NaN.
    # A factor where every stock is NaN on every recent day signals an
    # upstream data gap (e.g. q_* fields before 2022) and should be
    # rebuilt so the gap is visible rather than silently carried forward.
    try:
        recent_slice = df.loc[df.index.isin(present)]
    except Exception:
        # Index mismatch edge case — treat as unreadable
        return "unreadable"

    if recent_slice.empty:
        return "missing_dates"

    if recent_slice.notna().any(axis=None):
        return "ok"

    return "all_nan"


def decide_build_action(
    factor_name: str,
    factor_path: Path,
    deps: tuple[str, ...],
    source_root: Path,
    force: bool = False,
    quality_check_days: int = 0,
    effective_end: str | None = None,
) -> tuple[str, str | None]:
    """Decide whether to skip, incrementally update, or rebuild a factor.

    Returns ``(action, reason)``:

    - ``("skip", reason)`` — factor is already up to date; no work needed.
    - ``("incremental", max_date)`` — factor exists but is behind; *max_date*
      (YYYYMMDD) is passed as *factor_start_date* so only new rows are
      computed and appended via :func:`write_factor_incremental`.
    - ``("rebuild", reason)`` — no usable existing file, or *force* is set.

    Parameters
    ----------
    effective_end:
        Pre-computed effective end date (YYYYMMDD).  When None it is
        resolved from *source_root* — pass a cached value when calling
        in a tight loop to avoid repeated parquet reads.
    """
    if force:
        return "rebuild", "forced"

    # No existing file → full rebuild
    if not factor_path.exists():
        return "rebuild", "missing"

    # Read the factor's current maximum date
    factor_max_date = _read_factor_max_date(factor_path)
    if factor_max_date is None:
        return "rebuild", "unreadable"

    # Latest date for which source data is available.
    # Use the pre-computed value when calling in a tight loop, otherwise
    # resolve from source data (reads daily_adj.parquet — expensive).
    if effective_end is None:
        effective_end = _resolve_effective_end_date(source_root)

    # ── Quality gate: check recent trading days for NaN gaps ──────────
    # A factor whose recent rows are all-NaN (e.g. upstream data not yet
    # populated for q_* fields) must be rebuilt so the gap is visible
    # rather than silently carried forward as stale data.
    if quality_check_days > 0:
        quality = _check_factor_recent_quality(
            factor_path, source_root, n_days=quality_check_days,
            effective_end=effective_end,
        )
        if quality == "all_nan":
            return "rebuild", f"last_{quality_check_days}_trading_days_all_nan"
        if quality == "unreadable":
            return "rebuild", "unreadable"
        if quality == "missing_dates":
            # Recent calendar dates are absent from the factor — force rebuild
            # so the factor is brought up to date (covers stale factors and
            # factors whose quality check was previously fooled by future
            # calendar dates).
            return "rebuild", f"last_{quality_check_days}_trading_days_missing"

    # Factor is already current
    if factor_max_date >= effective_end:
        return "skip", f"up to date ({factor_max_date} >= {effective_end})"

    # Factor exists but is behind — only compute the new tail
    return "incremental", factor_max_date


# ── Factor classification ────────────────────────────────────────────────────

#: Dependency keys that determine which build strategy a factor uses.
_CLASS_3_DEP = "history_1min"   # per-stock directory → unified intraday pass
_CLASS_2_DEP = "cyq_chips"      # per-stock directory → unified cyq_chips pass
_CLASS_4_DEP = "__factors__"    # factor-coupling: loads existing .fea files


def classify_factor(name: str) -> int:
    """Return 1, 2, 3, or 4 based on the factor's declared data dependencies.

    - Class 1 — Panel:            loads from a single .parquet file, vectorised
    - Class 2 — cyq_chips:        loads from cyq_chips/ per-stock directory
    - Class 3 — history_1min:     loads from history_1min/ per-stock directory
    - Class 4 — Coupling:         loads existing .fea factor files, combines them
    """
    spec = get_factor(name)
    deps = spec.dependencies
    if _CLASS_3_DEP in deps:
        return 3
    if _CLASS_2_DEP in deps:
        return 2
    if _CLASS_4_DEP in deps:
        return 4
    return 1


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
    n = 3
    try:
        # Linux /proc/meminfo is the most reliable across Python versions
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        import re
        avail = re.search(r"MemAvailable:\s+(\d+)", meminfo)
        if avail:
            avail_kb = int(avail.group(1))
            avail_gb = avail_kb / (1024 * 1024)
            if avail_gb < 8:
                n = 1
            elif avail_gb < 16:
                n = 2
    except Exception:
        pass
    return n


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
    import time as _time_module

    spec = get_factor(name)
    repo = repo or DataRepository(paths=paths)

    t0 = time.perf_counter()
    factor_frame = None
    rows = 0
    nn_rows = 0
    error_msg = None

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

    logger.debug("%s [%s]: start%s",
                 name, spec.category,
                 f" (incremental from {factor_start_date})" if factor_start_date else "")

    try:
        context = FactorContext(repo=repo, start_date=context_start, end_date=_effective_end)
        raw_output = spec.compute(context)
        factor_frame = ensure_single_factor_frame(raw_output, spec.name, skip_ffill=(spec.category == "target"))

        # If incremental, slice to only new dates
        if factor_start_date:
            factor_frame = factor_frame.loc[factor_frame.index > factor_start_date]

        rows = len(factor_frame)
        nn_rows = int(factor_frame.notna().sum().sum())

        # Quick stats for the log
        mem_mb = factor_frame.memory_usage(deep=True).sum() / (1024 * 1024) if rows else 0

        if spec.category == "target":
            if factor_start_date:
                factor_path, manifest_path = write_target_incremental(spec, factor_frame, paths=repo.paths)
            else:
                factor_path, manifest_path = write_target(spec, factor_frame, paths=repo.paths)
        elif factor_start_date:
            factor_path, manifest_path = write_factor_incremental(
                spec, factor_frame, paths=repo.paths
            )
        else:
            factor_path, manifest_path = write_factor(spec, factor_frame, paths=repo.paths)

        action = "incremental" if factor_start_date else "rebuild"
        _write_done_marker(name, action, repo.paths.manifest_output_dir)

    except Exception as exc:
        error_msg = str(exc)
        logger.error("%s [%s]: FAILED — %s", name, spec.category, error_msg, exc_info=True)
        factor_path = repo.paths.factor_output_dir / f"{spec.name}.fea"
        manifest_path = repo.paths.manifest_output_dir / f"{spec.name}.json"
        raise

    finally:
        elapsed = time.perf_counter() - t0
        if error_msg is None:
            logger.debug("%s [%s]: done — %d rows, %d non-null, %.1f s, %.1f MB",
                         name, spec.category, rows, nn_rows, elapsed, mem_mb)

    action = "incremental" if factor_start_date else "rebuild"
    return BuildResult(
        factor_name=spec.name,
        factor_path=factor_path,
        manifest_path=manifest_path,
        elapsed=elapsed,
        action=action,
        rows=rows,
        non_null_rows=nn_rows,
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

    # First pass: classify actions (effective_end cached to avoid
    # re-reading daily_adj.parquet for every factor).
    repo_temp = DataRepository(paths=paths)
    _effective_end = _resolve_effective_end_date(repo_temp.paths.source_root)
    action_map: dict[str, tuple[str, str | None]] = {}
    for i, (_, name) in enumerate(plan):
        spec = get_factor(name)
        factor_path = repo_temp.paths.factor_output_dir / f"{spec.name}.fea"
        action, reason = decide_build_action(
            name, factor_path, spec.dependencies, repo_temp.paths.source_root,
            force=force, effective_end=_effective_end,
        )
        action_map[name] = (action, reason)
        if (i + 1) % 50 == 0:
            print(f"  [plan] scanned {i + 1}/{len(plan)} factors...", flush=True)

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
        completed = 0
        total = len(plan)
        for category, name in plan:
            action, reason = action_map[name]

            if action == "skip":
                completed += 1
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
                completed += 1
                # Simple line-based progress — always visible, even through pipes
                print(
                    f"  [{completed}/{total}] {name} {result.action} ({elapsed:.1f}s)",
                    flush=True,
                )
            except Exception:
                logger.exception("%s: build failed", name)
                results.append(BuildResult(
                    factor_name=name,
                    factor_path=repo_temp.paths.factor_output_dir / f"{get_factor(name).name}.fea",
                    manifest_path=repo_temp.paths.manifest_output_dir / f"{get_factor(name).name}.json",
                    elapsed=0.0,
                    action="error",
                ))
                completed += 1
                print(
                    f"  [{completed}/{total}] {name} ERROR",
                    flush=True,
                )
                sub_bar.reset(total=0)
            bar.update(1)

    sub_bar.close()
    results.sort(key=lambda item: item.factor_name)

    # Summary log
    ok = sum(1 for r in results if r.action not in ("skip", "error"))
    skipped = sum(1 for r in results if r.action == "skip")
    errors = sum(1 for r in results if r.action == "error")
    total_elapsed = sum(r.elapsed for r in results)
    total_rows = sum(r.rows for r in results)
    logger.info("Build complete — %d ok, %d skip, %d error | %.0f s wall | %d total rows",
                ok, skipped, errors, total_elapsed, total_rows)

    return results


# ── Parallel build ──────────────────────────────────────────────────────────

def _build_factor_worker(
    name: str,
    project_root: str,
    source_root: str,
    shared_progress_state: Any | None,
    force: bool,
    quality_check_days: int = 0,
    effective_end: str | None = None,
) -> dict[str, Any]:
    """Worker function for ProcessPoolExecutor.

    Returns a dict of results because BuildResult may not be picklable
    across processes if paths differ.

    When *quality_check_days* > 0 the worker inspects the last N trading
    days of the existing .fea file **before** computing.  If every value
    in those recent rows is NaN the worker switches from ``"skip"`` /
    ``"incremental"`` to a full ``"rebuild"`` so upstream data gaps are
    not silently carried forward.

    When *effective_end* is provided it is used directly instead of
    re-reading ``daily_adj.parquet`` (saves one parquet read per worker).
    """
    configure_paths(project_root=project_root, source_root=source_root)
    ensure_builtin_factors_loaded()

    paths = configure_paths()
    spec = get_factor(name)
    factor_path = paths.factor_output_dir / f"{spec.name}.fea"

    # Decide action — include the quality gate in the worker so each
    # core independently verifies local .fea integrity before computing.
    action, reason = decide_build_action(
        name, factor_path, spec.dependencies, paths.source_root,
        force=force, quality_check_days=quality_check_days,
        effective_end=effective_end,
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

    # Determine the effective end date — use the cached value when
    # provided, otherwise resolve from source data.
    if effective_end is not None:
        _effective_end = effective_end
    else:
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
        factor_frame = ensure_single_factor_frame(raw_output, spec.name, skip_ffill=(spec.category == "target"))

        if action == "incremental" and reason:
            factor_frame = factor_frame.loc[factor_frame.index > reason]

        _set_stage("writing")
        if spec.category == "target":
            if action == "incremental":
                fp, mp = write_target_incremental(spec, factor_frame, paths=paths)
            else:
                fp, mp = write_target(spec, factor_frame, paths=paths)
        elif action == "incremental":
            fp, mp = write_factor_incremental(spec, factor_frame, paths=paths)
        else:
            fp, mp = write_factor(spec, factor_frame, paths=paths)

        _write_done_marker(name, action, paths.manifest_output_dir)

    except Exception as e:
        logger.exception("%s: failed", name)
        error_msg = str(e)
        fp = paths.factor_output_dir / f"{spec.name}.fea"
        mp = paths.manifest_output_dir / f"{spec.name}.json"

    elapsed = time.perf_counter() - t0
    rows = len(factor_frame) if factor_frame is not None else 0
    nn_rows = int(factor_frame.notna().sum().sum()) if factor_frame is not None else 0

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
    quality_check_days: int = 0,
    use_dashboard: bool = False,
) -> list[BuildResult]:
    """Build factors in parallel with sub-progress display.

    Parameters
    ----------
    quality_check_days:
        If > 0, each worker inspects the last *quality_check_days*
        trading days of the existing .fea file.  When every value in
        those rows is NaN the worker rebuilds from scratch instead of
        skipping / incrementally updating.
    use_dashboard:
        When True, use the Rich-based live dashboard instead of tqdm
        progress bars (requires the ``rich`` package).
    """
    if not names:
        return []

    ensure_builtin_factors_loaded()
    configured_paths = paths or configure_paths()
    max_workers = max_workers or recommend_worker_count()
    plan = _flatten_build_plan(names)

    # ── Resolve effective end date ONCE for the whole batch ──────────
    # _resolve_effective_end_date reads daily_adj.parquet — calling it
    # per-factor (580×) would waste minutes with no output.  Cache it
    # and pass to every worker so they don't recompute it either.
    _effective_end = _resolve_effective_end_date(configured_paths.source_root)
    logger.info("Effective end date: %s  |  %d factors to check",
                _effective_end, len(plan))
    print(f"  Effective end date: {_effective_end}", flush=True)
    print(f"  Dispatching {len(plan)} factors to {max_workers} workers...", flush=True)

    # ── Shared state for per-worker progress ─────────────────────────
    import multiprocessing
    manager = multiprocessing.Manager()
    shared_state = manager.dict()

    # ── Dashboard (rich-based live progress) ──────────────────────────
    # Safety: when stdout is piped (e.g. parent streaming subprocess
    # output line-by-line), the Rich Live ANSI cursor codes would block
    # the parent's readline().  Force-disable in that case.
    import sys as _sys2
    if use_dashboard and not _sys2.stdout.isatty():
        logger.info("Dashboard disabled — stdout is not a TTY (likely piped to parent)")
        use_dashboard = False

    dashboard = None
    dashboard_slots: dict[str, int] = {}
    _next_dash_slot = 0
    if use_dashboard:
        try:
            from featureengineering.dashboard import FactorBuildDashboard
            dashboard = FactorBuildDashboard(
                total_factors=len(plan),
                max_workers=max_workers,
                title=f"Class 1 Panel — {len(plan)} factors",
            )
            dashboard.__enter__()
        except Exception:
            dashboard = None

    # ── Submit ALL factors directly to workers ───────────────────────
    # No sequential pre-classification — each worker independently
    # checks its .fea file, decides skip/incremental/rebuild, and
    # either returns immediately (skip) or computes the factor.
    # This gives instant visibility: workers start within seconds and
    # the user sees progress right away.
    all_names = [name for _, name in plan]
    results: list[BuildResult] = []
    remaining_names: list[str] = []  # captured names when pool breaks

    if all_names:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _build_factor_worker,
                    name,
                    str(configured_paths.project_root),
                    str(configured_paths.source_root),
                    shared_state,
                    force,
                    quality_check_days,
                    _effective_end,
                ): name
                for name in all_names
            }

            # ── Progress display ───────────────────────────────────
            # Simple, clean output that works equally well through
            # pipes and in a real terminal:
            #  • One main tqdm bar for overall progress
            #  • Per-factor line only for "interesting" actions
            #    (rebuild, incremental, error) that take real time
            #  • Periodic compact status heartbeat
            #  • Rich dashboard (when TTY + --dashboard)

            _total = len(futures)
            _counts: dict[str, int] = {"rebuild": 0, "incremental": 0,
                                         "skip": 0, "error": 0}
            _last_heartbeat = time.perf_counter()

            with tqdm(total=_total, desc="Building", unit="f",
                      position=0, leave=True,
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                                 "{n_fmt}/{total_fmt} "
                                 "[{elapsed}<{remaining}]") as bar:

                pending = set(futures.keys())
                _next_heartbeat = 50  # first heartbeat after 50 completions

                while pending:
                    # Collect any completed futures (non-blocking poll)
                    done = {f for f in pending if f.done()}
                    pool_broken = False

                    for fut in done:
                        pending.discard(fut)
                        name = futures[fut]
                        try:
                            worker_result = fut.result()
                        except BrokenProcessPool:
                            logger.error("%s: pool broken (OOM kill likely) — "
                                         "%d remaining failed", name, len(pending) + 1)
                            pool_broken = True
                            worker_result = {
                                "factor_name": name, "factor_path": "",
                                "manifest_path": "", "elapsed": 0.0,
                                "action": "error", "category": "",
                                "rows": 0, "non_null_rows": 0,
                                "error": "Process pool terminated abruptly",
                            }
                        except Exception as e:
                            logger.exception("%s: worker failed", name)
                            worker_result = {
                                "factor_name": name, "factor_path": "",
                                "manifest_path": "", "elapsed": 0.0,
                                "action": "error", "category": "",
                                "rows": 0, "non_null_rows": 0,
                                "error": str(e),
                            }

                        wr = worker_result
                        action = wr.get("action", "error")
                        elapsed_w = wr.get("elapsed", 0.0)

                        results.append(BuildResult(
                            factor_name=wr["factor_name"],
                            factor_path=Path(wr.get("factor_path", "")),
                            manifest_path=Path(wr.get("manifest_path", "")),
                            elapsed=elapsed_w,
                            action=action,
                        ))
                        bar.update(1)
                        _counts[action] = _counts.get(action, 0) + 1

                        # ── Dashboard update ──
                        if dashboard is not None:
                            dash_slot = dashboard_slots.pop(
                                wr["factor_name"], None,
                            )
                            dashboard.mark_done(
                                wr["factor_name"], action, elapsed_w,
                                slot=dash_slot,
                            )

                        # ── Print interesting actions individually ──
                        # Skip factors are instant — printing 500+
                        # one-liners creates visual noise.  Only print
                        # rebuild / incremental / error which are
                        # actionable and take real time.
                        if action != "skip":
                            _n = bar.n
                            tag = "ERR" if action == "error" else action[:4].upper()
                            print(
                                f"  [{_n}/{_total}] {wr['factor_name']} "
                                f"{tag} ({elapsed_w:.1f}s)",
                                flush=True,
                            )
                        elif wr.get("error"):
                            logger.error("%s: %s", wr["factor_name"], wr["error"])

                        # ── Periodic heartbeat ──
                        if bar.n >= _next_heartbeat:
                            _next_heartbeat = bar.n + 50
                            print(
                                f"  [{bar.n}/{_total}] "
                                f"◆{_counts['rebuild']} rebuild  "
                                f"▲{_counts['incremental']} incr  "
                                f"✓{_counts['skip']} skip  "
                                f"✗{_counts['error']} err",
                                flush=True,
                            )

                        # ── Dashboard: refresh worker slots ──
                        if dashboard is not None:
                            try:
                                st = dict(shared_state)
                            except Exception:
                                st = {}
                            for f_name, info in st.items():
                                if not isinstance(info, dict):
                                    continue
                                stage = info.get("stage", "")
                                if stage in ("done", "error", ""):
                                    continue
                                cur = info.get("current", 0)
                                tot = info.get("total", 0)
                                t_start = info.get("start", 0)
                                pct = cur / max(tot, 1) if tot > 0 else 0.0
                                if f_name not in dashboard_slots:
                                    dashboard_slots[f_name] = _next_dash_slot % max_workers
                                    _next_dash_slot += 1
                                dashboard.update_worker(
                                    dashboard_slots[f_name],
                                    name=f_name, stage=stage,
                                    pct=pct,
                                    elapsed=(time.perf_counter() - t_start) if t_start else 0,
                                )

                    # ── Handle broken pool ──
                    if pool_broken:
                        for remaining in list(pending):
                            pending.discard(remaining)
                            rname = futures[remaining]
                            remaining_names.append(rname)
                            try:
                                remaining.cancel()
                            except Exception:
                                pass
                        logger.warning("Pool broken — %d factors will retry sequentially", len(remaining_names))
                        break

                    if pending:
                        import time as _time
                        _time.sleep(0.05)  # short sleep to avoid busy-wait

    # ── Dashboard: finalise and print summary ────────────────────────
    if dashboard is not None:
        try:
            dashboard.__exit__(None, None, None)
        except Exception:
            pass
        dashboard.print_summary()

    # ── Pool-broken fallback: sequential retry ─────────────────────────
    if remaining_names:
        import time as _time2
        print(f"\n  ⚠ Pool broken (likely OOM) — retrying {len(remaining_names)} factors sequentially...\n", flush=True)
        _time2.sleep(2)  # let the OS reclaim memory from killed workers
        sequential_results = build_many(remaining_names, paths=configured_paths, force=force)
        results.extend(sequential_results)
        ok_seq = sum(1 for r in sequential_results if r.action not in ("error",))
        err_seq = sum(1 for r in sequential_results if r.action == "error")
        logger.info("Sequential fallback complete: %d ok, %d error", ok_seq, err_seq)

    results.sort(key=lambda item: item.factor_name)

    # Summary log
    ok_count = sum(1 for r in results if r.action not in ("skip", "error"))
    skip_count = sum(1 for r in results if r.action == "skip")
    err_count = sum(1 for r in results if r.action == "error")
    total_elapsed = sum(r.elapsed for r in results)
    total_rows = sum(r.rows for r in results)
    logger.info("Build complete — %d ok, %d skip, %d error | %.0f s wall | %d total rows",
                ok_count, skip_count, err_count, total_elapsed, total_rows)

    return results

