from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .registry import FactorSpec
from .settings import ProjectPaths, configure_paths

INCR_SUFFIX = "_incr"


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
    """Load a factor by merging base ({name}.fea) and incremental ({name}_incr.fea) files.

    The base file stores historical data and is never modified after full build.
    The incremental file stores only the latest dates from daily updates.
    Duplicates are resolved by keeping the incremental version (keep='last').
    """
    paths = paths or configure_paths()
    base_path = factor_base_path(name, paths)
    incr_path = factor_incr_path(name, paths)

    frames: list[pd.DataFrame] = []
    if base_path.exists():
        frames.append(pd.read_feather(base_path))
    if incr_path.exists():
        frames.append(pd.read_feather(incr_path))

    if not frames:
        raise FileNotFoundError(
            f"No factor files found for '{name}' in {paths.factor_output_dir}"
        )

    if len(frames) == 1:
        return frames[0]

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    return merged


def ensure_single_factor_frame(data: pd.Series | pd.DataFrame, factor_name: str) -> pd.DataFrame:
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
    return wide


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

    # Purge stale incremental file — base file now contains full data.
    incr_path = factor_incr_path(spec, paths)
    if incr_path.exists():
        incr_path.unlink()

    factor_frame.to_feather(factor_path)

    # Last date and stock count (wide format: Date index, Code columns)
    dates = factor_frame.index
    last_date = str(dates.max())
    last_day_stock_count = int(factor_frame.loc[last_date].notna().sum())

    total_cells = len(factor_frame) * len(factor_frame.columns)
    non_null_cells = int(factor_frame.notna().sum().sum())

    manifest = {
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "thesis": spec.thesis,
        "dependencies": list(spec.dependencies),
        "rows": int(len(factor_frame)),
        "cols": int(len(factor_frame.columns)),
        "non_null_cells": non_null_cells,
        "coverage_ratio": float(non_null_cells / total_cells) if total_cells else 0.0,
        "index_names": list(factor_frame.index.names),
        "column": spec.name,
        "last_date": last_date,
        "last_day_stock_count": last_day_stock_count,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return factor_path, manifest_path


def write_factor_incremental(
    spec: FactorSpec,
    new_frame: pd.DataFrame,
    *,
    paths: ProjectPaths | None = None,
) -> tuple[Path, Path]:
    """Write incremental factor data to {name}_incr.fea.

    Merges with existing incremental data (if any) so the incr file
    accumulates new dates across multiple daily updates.  The base
    {name}.fea file is never touched.
    """
    paths = paths or configure_paths()
    incr_path = factor_incr_path(spec, paths)
    manifest_path = paths.manifest_output_dir / f"{spec.name}.json"

    if incr_path.exists():
        try:
            existing = pd.read_feather(incr_path)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if existing.empty:
        merged = new_frame
    else:
        merged = pd.concat([existing, new_frame], ignore_index=False)
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()

    # Atomic write
    tmp = incr_path.with_suffix(".fea.tmp")
    merged.to_feather(tmp)
    tmp.replace(incr_path)

    # Update manifest: reflect the full combined (base + incr) state
    base_path = factor_base_path(spec, paths)
    base_cells = 0
    base_nn = 0
    if base_path.exists():
        try:
            base_df = pd.read_feather(base_path)
            base_cells = len(base_df) * len(base_df.columns)
            base_nn = int(base_df.notna().sum().sum())
        except Exception:
            pass

    incr_cells = len(merged) * len(merged.columns)
    incr_nn = int(merged.notna().sum().sum())
    total_cells = base_cells + incr_cells
    total_nn = base_nn + incr_nn

    # Last date and stock count come from the incremental data (newest)
    dates = merged.index
    last_date = str(dates.max())
    last_day_stock_count = int(merged.loc[last_date].notna().sum())

    manifest = {
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "thesis": spec.thesis,
        "dependencies": list(spec.dependencies),
        "rows": total_rows,
        "cols": int(len(merged.columns)),
        "non_null_cells": total_nn,
        "coverage_ratio": float(total_nn / total_cells) if total_cells else 0.0,
        "index_names": list(merged.index.names),
        "column": spec.name,
        "last_date": last_date,
        "last_day_stock_count": last_day_stock_count,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return incr_path, manifest_path


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

    target_frame.to_feather(target_path)

    dates = target_frame.index
    last_date = str(dates.max())
    last_day_stock_count = int(target_frame.loc[last_date].notna().sum())

    total_cells = len(target_frame) * len(target_frame.columns)
    non_null_cells = int(target_frame.notna().sum().sum())

    manifest = {
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "thesis": spec.thesis,
        "dependencies": list(spec.dependencies),
        "rows": int(len(target_frame)),
        "cols": int(len(target_frame.columns)),
        "non_null_cells": non_null_cells,
        "coverage_ratio": float(non_null_cells / total_cells) if total_cells else 0.0,
        "index_names": list(target_frame.index.names),
        "column": spec.name,
        "last_date": last_date,
        "last_day_stock_count": last_day_stock_count,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target_path, manifest_path
