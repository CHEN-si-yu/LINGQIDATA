from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

_LOGGING_CONFIGURED = False
_FILE_LOGGING_CONFIGURED = False


def log_environment_info() -> None:
    """Log Python / package versions and log-file path once per run.

    Idempotent across process trees via an environment variable so it is
    safe to call from *both* ``build_factors.py`` and ``cli.main()`` —
    only the first caller across all processes actually emits the lines.
    """
    if os.environ.get("FEATURE_ENGINEERING_ENV_LOGGED"):
        return
    os.environ["FEATURE_ENGINEERING_ENV_LOGGED"] = "1"

    root = logging.getLogger()
    log_path = os.environ.get("FEATURE_ENGINEERING_LOG_FILE", "")
    if log_path:
        root.info("Log file  : %s", log_path)
    try:
        import pandas as _pd
        pd_ver = _pd.__version__
    except Exception:
        pd_ver = "N/A"
    root.info("Python    : %s", sys.version.split()[0])
    root.info("pandas    : %s", pd_ver)
    root.info("numpy     : %s", np.__version__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging once — stdout INFO+, file DEBUG+ in data/logs/."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler — INFO and above, keeps console output
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(stdout)

    # — numpy floating-point warnings → logger —
    # We redirect them so the tqdm progress bars are not corrupted by
    # stderr spam.  Factors that produce divide-by-zero / invalid-value
    # are still visible in the log file for triage.
    np.seterr(all="call")
    np.seterr(under="ignore")  # underflow is harmless in financial calcs

    def _np_err_handler(err, flag):
        logging.getLogger("numpy").warning("NumPy %s: %s", flag, err)

    np.seterrcall(_np_err_handler)

    # — Python warnings (pandas FutureWarning, RuntimeWarning, etc.) → logger —
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")

    # Show each unique warning once per run (not per site).
    warnings.filterwarnings("default")
    # Demote the noisy pct_change fill_method FutureWarning so it does not
    # drown the log — it is already tracked in the code and doesn't block
    # correct results.
    warnings.filterwarnings(
        "ignore",
        message=".*fill_method.*pct_change.*",
        category=FutureWarning,
    )


def _setup_file_logging(log_dir: Path) -> None:
    """Add a timestamped file handler once the log directory is known.

    Log files are named ``featureengineering_YYYYMMDD_HHMMSS.log`` so
    each run produces a separate, chronologically sortable file.
    A symlink at ``featureengineering_latest.log`` always points
    to the most recent file for quick access.

    When running inside parallel workers the log-file path is read from
    the ``FEATURE_ENGINEERING_LOG_FILE`` environment variable so every
    worker writes to the same file as the main process.
    """
    global _FILE_LOGGING_CONFIGURED
    if _FILE_LOGGING_CONFIGURED:
        return
    _FILE_LOGGING_CONFIGURED = True

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        return

    # Use the parent-provided log path when running inside a worker,
    # otherwise create a new timestamped file.
    shared_log = os.environ.get("FEATURE_ENGINEERING_LOG_FILE")
    if shared_log:
        log_file = Path(shared_log)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"featureengineering_{timestamp}.log"
        # Expose the log path so child processes (parallel workers) reuse it.
        os.environ["FEATURE_ENGINEERING_LOG_FILE"] = str(log_file)

    latest_link = log_dir / "featureengineering_latest.log"

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root = logging.getLogger()
        root.addHandler(file_handler)
    except (OSError, PermissionError):
        return

    # Update / create the "latest" symlink for convenience (main process only)
    if not shared_log:
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(log_file.name)
        except OSError:
            pass


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
