from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    source_root: Path
    factor_output_dir: Path
    manifest_output_dir: Path
    target_output_dir: Path
    stock_pool_file: Path

    @classmethod
    def default(cls) -> "ProjectPaths":
        package_root = Path(__file__).resolve().parents[2]
        project_root = Path(os.environ.get("FEATURE_ENGINEERING_PROJECT_ROOT", package_root))
        source_root = Path(
            os.environ.get(
                "FEATURE_ENGINEERING_SOURCE_ROOT",
                project_root.parent / "data",
            )
        )
        return cls(
            project_root=project_root.resolve(),
            source_root=source_root.resolve(),
            factor_output_dir=project_root / "data" / "factors",
            manifest_output_dir=project_root / "data" / "manifests",
            target_output_dir=project_root / "data" / "targets",
            stock_pool_file=project_root.parent / "Code_num.txt",
        )


PATHS = ProjectPaths.default()


def configure_paths(
    *,
    project_root: str | Path | None = None,
    source_root: str | Path | None = None,
) -> ProjectPaths:
    global PATHS
    resolved_project_root = Path(project_root).resolve() if project_root is not None else PATHS.project_root
    resolved_source_root = Path(source_root).resolve() if source_root is not None else PATHS.source_root
    PATHS = ProjectPaths(
        project_root=resolved_project_root,
        source_root=resolved_source_root,
        factor_output_dir=resolved_project_root / "data" / "factors",
        manifest_output_dir=resolved_project_root / "data" / "manifests",
        target_output_dir=resolved_project_root / "data" / "targets",
        stock_pool_file=resolved_project_root.parent / "Code_num.txt",
    )
    return PATHS
