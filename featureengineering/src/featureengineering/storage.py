from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .registry import FactorSpec
from .settings import ProjectPaths, configure_paths


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
        frame = frame.reorder_levels(["Code", "Date"])
    else:
        frame.index = frame.index.set_names(["Code", "Date"])

    frame = frame.sort_index(level=["Date", "Code"])
    return frame


def write_factor(
    spec: FactorSpec,
    factor_frame: pd.DataFrame,
    *,
    paths: ProjectPaths | None = None,
) -> tuple[Path, Path]:
    paths = paths or configure_paths()
    paths.factor_output_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_output_dir.mkdir(parents=True, exist_ok=True)

    factor_path = paths.factor_output_dir / f"{spec.name}.fea"
    manifest_path = paths.manifest_output_dir / f"{spec.name}.json"

    factor_frame.to_feather(factor_path)

    manifest = {
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "thesis": spec.thesis,
        "dependencies": list(spec.dependencies),
        "rows": int(len(factor_frame)),
        "non_null_rows": int(factor_frame[spec.name].notna().sum()),
        "coverage_ratio": float(factor_frame[spec.name].notna().mean()) if len(factor_frame) else 0.0,
        "index_names": list(factor_frame.index.names),
        "column": spec.name,
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
    """Merge new incremental factor data into the existing .fea file.

    Reads the existing file, concatenates with *new_frame*,
    drops exact duplicates (keep last), sorts, and writes back.
    """
    paths = paths or configure_paths()
    factor_path = paths.factor_output_dir / f"{spec.name}.fea"
    manifest_path = paths.manifest_output_dir / f"{spec.name}.json"

    if factor_path.exists():
        try:
            existing = pd.read_feather(factor_path)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if existing.empty:
        merged = new_frame
    else:
        merged = pd.concat([existing, new_frame], ignore_index=False)
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index(level=["Date", "Code"])

    # Atomic write
    tmp = factor_path.with_suffix(".fea.tmp")
    merged.to_feather(tmp)
    tmp.replace(factor_path)

    # Update manifest
    manifest = {
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "thesis": spec.thesis,
        "dependencies": list(spec.dependencies),
        "rows": int(len(merged)),
        "non_null_rows": int(merged[spec.name].notna().sum()),
        "coverage_ratio": float(merged[spec.name].notna().mean()) if len(merged) else 0.0,
        "index_names": list(merged.index.names),
        "column": spec.name,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return factor_path, manifest_path


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

    manifest = {
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "thesis": spec.thesis,
        "dependencies": list(spec.dependencies),
        "rows": int(len(target_frame)),
        "non_null_rows": int(target_frame[spec.name].notna().sum()),
        "coverage_ratio": float(target_frame[spec.name].notna().mean()) if len(target_frame) else 0.0,
        "index_names": list(target_frame.index.names),
        "column": spec.name,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target_path, manifest_path
