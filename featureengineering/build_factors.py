#!/usr/bin/env python3
"""Factor build entry point.

Without arguments: runs each class sequentially as a subprocess with tuned job counts.
With arguments: delegates to the full CLI (see featureengineering.cli).
"""

from __future__ import annotations

import multiprocessing as _mp

try:
    _mp.set_start_method("spawn")
except RuntimeError:
    pass

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger("build_factors")


def _log_banner(title: str) -> None:
    """Print a banner to both the log and stdout."""
    line = "#" * 64
    msg = f"\n{line}\n#  {title}\n{line}"
    logger.info(msg)
    print(msg, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from featureengineering.cli import main

        raise SystemExit(main())

    # Default (no args): run each class sequentially with tuned job counts
    import subprocess

    # Configure logging early so we get a timestamped log file and
    # numpy / pandas warnings are captured.
    from featureengineering.settings import configure_paths, log_environment_info

    paths = configure_paths()
    log_environment_info()

    script = Path(__file__).resolve()
    stages = [
        ("Class 1 (Panel)",     ["--only-class", "1", "--jobs", "8"]),
        ("Class 2 (cyq_chips)", ["--only-class", "2", "--jobs", "8"]),
        ("Class 3 (intraday)",  ["--only-class", "3", "--jobs", "8"]),
        ("Class 4 (Coupling)",  ["--only-class", "4", "--jobs", "8"]),
    ]

    t_start = time.perf_counter()
    logger.info("Factor build started — %d stage(s)", len(stages))
    _log_banner("BUILD START")

    overall_rc = 0
    for i, (label, args) in enumerate(stages, 1):
        stage_t0 = time.perf_counter()
        _log_banner(f"STAGE {i}/{len(stages)}: {label}")
        logger.info("Stage %d/%d: %s — launching subprocess", i, len(stages), label)

        rc = subprocess.run([sys.executable, str(script)] + args).returncode
        stage_elapsed = time.perf_counter() - stage_t0

        if rc != 0:
            msg = f"[WARN] {label} exited with code {rc} (elapsed: {stage_elapsed:.0f}s)"
            logger.warning(msg)
            print(f"\n{msg}", flush=True)
            overall_rc = 1
        else:
            logger.info("Stage %d/%d: %s — OK (%.0fs)", i, len(stages), label, stage_elapsed)

    total_elapsed = time.perf_counter() - t_start
    status = "OK" if overall_rc == 0 else "WARNINGS"
    logger.info("Factor build finished — %s, total elapsed: %.0f s", status, total_elapsed)
    _log_banner(f"BUILD DONE — {status} (total: {total_elapsed:.0f}s)")

    raise SystemExit(overall_rc)
