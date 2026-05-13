from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from featureengineering.cli import main


# PyCharm direct-run defaults:
# - If this file is launched without parameters, the configuration below is used.
# - If parameters are provided, they take precedence.
RUN_MODE = "all"  # "all", "list", "targets-only", or "custom"
FACTORS: list[str] = []
JOBS: int | None = 2
SEQUENTIAL = False
SKIP_TARGETS = False
FORCE = False
SOURCE_ROOT: str | None = None


def build_default_argv() -> list[str] | None:
    if len(sys.argv) > 1:
        return None

    argv: list[str] = []
    if RUN_MODE == "list":
        argv.append("--list")
    elif RUN_MODE == "all":
        argv.append("--all")
    elif RUN_MODE == "targets-only":
        argv.append("--targets-only")
    elif RUN_MODE == "custom":
        for factor_name in FACTORS:
            argv.extend(["--factor", factor_name])
    else:
        raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")

    if JOBS is not None:
        argv.extend(["--jobs", str(JOBS)])
    if SEQUENTIAL:
        argv.append("--sequential")
    if SKIP_TARGETS:
        argv.append("--skip-targets")
    if FORCE:
        argv.append("--force")
    if SOURCE_ROOT:
        argv.extend(["--source-root", SOURCE_ROOT])
    return argv


if __name__ == "__main__":
    raise SystemExit(main(build_default_argv()))
