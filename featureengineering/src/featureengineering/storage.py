from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import FactorSpec
from .settings import ProjectPaths, configure_paths

INCR_SUFFIX = "_incr"

# ── Reference frame cache ────────────────────────────────────────────────────
_ref_cache: tuple[list[str], list[str]] | None = None


def _get_reference_frame() -> tuple[list[str], list[str]]:
    """Return (ref_dates, ref_codes) — canonical index from 2020-01-01."""
    global _ref_cache
    if _ref_cache is not None:
        return _ref_cache

    paths = configure_paths()
    daily = pd.read_parquet(
        paths.source_root / "daily_adj.parquet",
        columns=["stock_code", "trade_date"],
    )
    daily["d"] = daily["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    ref_dates = sorted(d for d in daily["d"].unique() if d >= "20200101")

    stock_pool_file = paths.project_root.parent / "Code_num.txt"
    with open(stock_pool_file) as f:
        ref_codes = sorted(set(line.strip() for line in f if line.strip()))

    _ref_cache = (ref_dates, ref_codes)
    return _ref_cache


def align_to_reference(wide: pd.DataFrame, skip_ffill: bool = False) -> pd.DataFrame:
    """Align a factor wide-format DataFrame to the canonical reference frame.

    - Dates: reindex to reference trading days, forward-fill gaps (source outages).
    - Codes: reindex to reference stock pool, NaN for stocks not in factor universe.

    This ensures every factor has uniform shape: len(ref_dates) x len(ref_codes).

    When *skip_ffill* is True (e.g. for target labels), missing dates are
    deliberately NOT added — targets use future data (e.g. open[t+N]) and
    dates for which that future data is unavailable should not appear in
    the output at all.  The date index is left as-is from the factor
    computation; only the column (stock) axis is aligned to the reference
    code pool.
    """
    ref_dates, ref_codes = _get_reference_frame()

    # Keep only dates >= 2020-01-01
    wide = wide[wide.index >= "20200101"]

    # Drop extra dates (e.g. THS calendar days not in daily_adj)
    extra_dates = sorted(set(wide.index) - set(ref_dates))
    if extra_dates:
        wide = wide.drop(extra_dates, errors="ignore")

    if skip_ffill:
        # Targets: do NOT add dates that the factor could not compute.
        # Missing dates mean the factor legitimately had no output
        # (e.g. future open prices unavailable for label_ret_*).
        pass
    else:
        # Add missing dates and forward-fill (source data outages)
        missing_dates = sorted(set(ref_dates) - set(wide.index))
        if missing_dates:
            missing_df = pd.DataFrame(index=missing_dates, columns=wide.columns)
            wide = pd.concat([wide, missing_df])
            wide = wide.sort_index()
            wide = wide.ffill()

    # Drop extra codes (e.g. THS stocks not in pool)
    extra_codes = sorted(set(wide.columns) - set(ref_codes))
    if extra_codes:
        wide = wide.drop(columns=extra_codes, errors="ignore")

    # Add missing codes as NaN (e.g. margin-limited universe)
    missing_codes = sorted(set(ref_codes) - set(wide.columns))
    if missing_codes:
        wide = wide.join(
            pd.DataFrame(np.nan, index=wide.index, columns=missing_codes)
        )

    # Final reindex: targets align only the column axis;
    # regular factors align both axes to the reference frame.
    if skip_ffill:
        wide = wide.reindex(columns=ref_codes)
    else:
        wide = wide.reindex(index=ref_dates, columns=ref_codes)
    wide.index.name = "Date"
    wide.columns.name = "Code"
    return wide


# ── Original functions below (with alignment inserted) ──────────────────────


def factor_base_path(spec_or_name: FactorSpec | str, paths: ProjectPaths | None = None) -> Path:
    """Return the base (historical) factor file path: {name}.fea."""
    paths = paths or configure_paths()
    name = spec_or_name if isinstance(spec_or_name, str) else spec_or_name.name
    return paths.factor_output_dir / f"{name}.fea"


def factor_incr_path(spec_or_name: FactorSpec | str, paths: ProjectPaths | None = None) -> Path:
    """Return the incremental factor file path: {name}_incr.fea."""
    paths = paths or configure_paths()
    name = spec_or_name if isinstance(spec_or_name, str) else spec_or_name.name
    return paths.factor_output_dir / f"{name}{INCR_SUFFIX}.fea"


def load_factor(name: str, paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Load a factor from {name}.fea."""
    paths = paths or configure_paths()
    base_path = factor_base_path(name, paths)

    if not base_path.exists():
        raise FileNotFoundError(
            f"No factor file found for '{name}' in {paths.factor_output_dir}"
        )

    return pd.read_feather(base_path)


def ensure_single_factor_frame(
    data: pd.Series | pd.DataFrame,
    factor_name: str,
    skip_ffill: bool = False,
) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        frame = data.to_frame(name=factor_name)
    else:
        frame = data.copy()

    if len(frame.columns) != 1:
        raise ValueError(f"{factor_name} must output exactly one column, got {len(frame.columns)}.")

    frame.columns = [factor_name]

    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError(f"{factor_name} must output a MultiIndex with Date and Code.")

    index_names = list(frame.index.names)
    lowered = [str(name).lower() if name is not None else "" for name in index_names]
    if lowered not in (["date", "code"], ["code", "date"]):
        raise ValueError(
            f"{factor_name} index names must be Date/Code or date/code, got: {index_names}"
        )

    if lowered == ["date", "code"]:
        frame.index = frame.index.set_names(["Date", "Code"])
    else:
        frame.index = frame.index.set_names(["Code", "Date"])
        frame = frame.reorder_levels(["Date", "Code"])

    frame = frame.sort_index()

    # Pivot to Date (rows) x Code (columns) wide format
    wide = frame.reset_index().pivot(index="Date", columns="Code", values=factor_name)
    wide = wide.sort_index()
    wide.columns.name = "Code"
    wide.index.name = "Date"

    # ── Align to reference frame (uniform shape) ──
    wide = align_to_reference(wide, skip_ffill=skip_ffill)

    return wide


def _write_manifest(path: Path, spec: FactorSpec, frame: pd.DataFrame) -> None:
    """Write the JSON manifest for a factor file."""
    dates = frame.index
    last_date = str(dates.max()) if len(dates) > 0 else "N/A"
    last_day_stock_count = int(frame.loc[last_date].notna().sum()) if len(dates) > 0 else 0

    total_cells = len(frame) * len(frame.columns) if len(frame) > 0 else 0
    non_null_cells = int(frame.notna().sum().sum())

    manifest = {
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "thesis": spec.thesis,
        "dependencies": list(spec.dependencies),
        "rows": int(len(frame)),
        "cols": int(len(frame.columns)),
        "non_null_cells": non_null_cells,
        "coverage_ratio": float(non_null_cells / total_cells) if total_cells else 0.0,
        "index_names": list(frame.index.names),
        "column": spec.name,
        "last_date": last_date,
        "last_day_stock_count": last_day_stock_count,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_factor(
    spec: FactorSpec,
    factor_frame: pd.DataFrame,
    *,
    paths: ProjectPaths | None = None,
) -> tuple[Path, Path]:
    paths = paths or configure_paths()
    paths.factor_output_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_output_dir.mkdir(parents=True, exist_ok=True)

    factor_path = factor_base_path(spec, paths)
    manifest_path = paths.manifest_output_dir / f"{spec.name}.json"

    # Atomic write: write to temp, then replace
    tmp = factor_path.with_suffix(".fea.tmp")
    factor_frame.to_feather(tmp)
    tmp.replace(factor_path)

    # Purge stale incremental file — only after the base write succeeded
    incr_path = factor_incr_path(spec, paths)
    if incr_path.exists():
        incr_path.unlink()

    _write_manifest(manifest_path, spec, factor_frame)
    return factor_path, manifest_path


def write_factor_incremental(
    spec: FactorSpec,
    new_frame: pd.DataFrame,
    *,
    paths: ProjectPaths | None = None,
) -> tuple[Path, Path]:
    """Merge incremental factor data directly into the base {name}.fea file.

    Reads the existing base, concatenates the new dates, deduplicates,
    and writes back atomically.  No separate _incr.fea file is created.
    """
    paths = paths or configure_paths()
    base_path = factor_base_path(spec, paths)
    manifest_path = paths.manifest_output_dir / f"{spec.name}.json"

    if not base_path.exists():
        raise FileNotFoundError(
            f"Base factor file not found for '{spec.name}' — run a full rebuild first"
        )

    existing = pd.read_feather(base_path)

    # Normalise index format only — the existing file was already
    # aligned via write_factor (or a prior incremental merge).
    # Calling align_to_reference here would forward-fill every
    # NaN gap in the *existing* data when there are any missing dates,
    # producing a different result from a full rebuild.  We normalise
    # the index to YYYYMMDD strings for safe concatenation with
    # *new_frame* (which was aligned by ensure_single_factor_frame).
    if "Date" in existing.columns:
        existing = existing.set_index("Date")
    existing.index = existing.index.astype(str).str.replace("-", "").str.slice(0, 8)

    merged = pd.concat([existing, new_frame], ignore_index=False)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()

    # Re-align the merged result — a safety measure that normalises
    # the column set to the reference stock pool.  Because the merge
    # fills all missing dates (existing had them except the tail;
    # new_frame supplies the tail), *missing_dates* will be empty and
    # forward-fill will not be triggered.
    merged = align_to_reference(merged)

    # Atomic write back to base
    tmp = base_path.with_suffix(".fea.tmp")
    merged.to_feather(tmp)
    tmp.replace(base_path)

    _write_manifest(manifest_path, spec, merged)
    return base_path, manifest_path


def write_target(
    spec: FactorSpec,
    target_frame: pd.DataFrame,
    *,
    paths: ProjectPaths | None = None,
) -> tuple[Path, Path]:
    """Write a target label .fea and manifest to the targets output directory."""
    paths = paths or configure_paths()
    paths.target_output_dir.mkdir(parents=True, exist_ok=True)

    target_path = paths.target_output_dir / f"{spec.name}.fea"
    manifest_path = paths.target_output_dir / f"{spec.name}.json"

    # Atomic write
    tmp = target_path.with_suffix(".fea.tmp")
    target_frame.to_feather(tmp)
    tmp.replace(target_path)

    _write_manifest(manifest_path, spec, target_frame)
    return target_path, manifest_path


def write_target_incremental(
    spec: FactorSpec,
    new_frame: pd.DataFrame,
    *,
    paths: ProjectPaths | None = None,
) -> tuple[Path, Path]:
    """Merge incremental target data into the existing target .fea file.

    Reads the existing target, concatenates the new dates, deduplicates
    (keeping new values), and writes back atomically.  Uses the same
    ``skip_ffill=True`` alignment as :func:`write_target` so that dates
    the factor cannot compute (missing future data) are never added.
    """
    paths = paths or configure_paths()
    target_path = paths.target_output_dir / f"{spec.name}.fea"
    manifest_path = paths.target_output_dir / f"{spec.name}.json"

    if not target_path.exists():
        raise FileNotFoundError(
            f"Target file not found for '{spec.name}' — run a full rebuild first"
        )

    existing = pd.read_feather(target_path)

    # Normalise index format only — the existing file was already
    # aligned via write_target (or a prior incremental merge).
    # Calling align_to_reference here would risk altering the
    # existing date/code set even with skip_ffill=True (reindex to
    # ref_dates drops dates and ref_codes adds columns).
    if "Date" in existing.columns:
        existing = existing.set_index("Date")
    existing.index = existing.index.astype(str).str.replace("-", "").str.slice(0, 8)

    # Merge: new dates override old, existing dates preserved
    merged = pd.concat([existing, new_frame], ignore_index=False)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()

    # Re-align after merge (target behaviour: no date padding)
    merged = align_to_reference(merged, skip_ffill=True)

    # Atomic write back to target
    tmp = target_path.with_suffix(".fea.tmp")
    merged.to_feather(tmp)
    tmp.replace(target_path)

    _write_manifest(manifest_path, spec, merged)
    return target_path, manifest_path
