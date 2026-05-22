from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

_LOGGING_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging once — stdout INFO+, file DEBUG+ in data/logs/."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler — INFO and above
    stdout = logging.StreamHandler()
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(stdout)


def _setup_file_logging(log_dir: Path) -> None:
    """Add a file handler once the log directory is known."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        return  # silently skip file logging if the directory is not writable
    try:
        file_handler = logging.FileHandler(log_dir / "featureengineering.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            root.addHandler(file_handler)
    except (OSError, PermissionError):
        return


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
            factor_output_dir=Path(os.environ.get(
                "FEATURE_ENGINEERING_FACTOR_OUTPUT_DIR",
                project_root / "data" / "factors",
            )).resolve(),
            manifest_output_dir=Path(os.environ.get(
                "FEATURE_ENGINEERING_MANIFEST_OUTPUT_DIR",
                project_root / "data" / "manifests",
            )).resolve(),
            target_output_dir=project_root / "data" / "targets",
            stock_pool_file=project_root.parent / "Code_num.txt",
        )


PATHS = ProjectPaths.default()


def configure_paths(
    *,
    project_root: str | Path | None = None,
    source_root: str | Path | None = None,
    factor_output_dir: str | Path | None = None,
    manifest_output_dir: str | Path | None = None,
) -> ProjectPaths:
    global PATHS
    resolved_project_root = Path(project_root).resolve() if project_root is not None else PATHS.project_root
    resolved_source_root = Path(source_root).resolve() if source_root is not None else PATHS.source_root
    resolved_factor_output_dir = (
        Path(factor_output_dir).resolve() if factor_output_dir is not None
        else Path(os.environ.get(
            "FEATURE_ENGINEERING_FACTOR_OUTPUT_DIR",
            str(PATHS.factor_output_dir),
        )).resolve()
    )
    resolved_manifest_output_dir = (
        Path(manifest_output_dir).resolve() if manifest_output_dir is not None
        else Path(os.environ.get(
            "FEATURE_ENGINEERING_MANIFEST_OUTPUT_DIR",
            str(PATHS.manifest_output_dir),
        )).resolve()
    )
    setup_logging()
    PATHS = ProjectPaths(
        project_root=resolved_project_root,
        source_root=resolved_source_root,
        factor_output_dir=resolved_factor_output_dir,
        manifest_output_dir=resolved_manifest_output_dir,
        target_output_dir=resolved_project_root / "data" / "targets",
        stock_pool_file=resolved_project_root.parent / "Code_num.txt",
    )
    _setup_file_logging(PATHS.project_root / "data" / "logs")
    return PATHS
