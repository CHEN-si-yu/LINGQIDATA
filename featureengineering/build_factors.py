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
import os
import subprocess
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


def _stream_subprocess(cmd: list[str], label: str) -> int:
    """Run a subprocess, streaming its stdout/stderr in real time.

    Prints every line the child emits as-is, plus a heartbeat line every
    30 seconds of silence so the user knows the process hasn't hung.

    Returns the process exit code.
    """
    # Ensure the child Python runs unbuffered so output appears in real time.
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        universal_newlines=True,
    )

    t0 = time.perf_counter()
    last_output = t0

    def _elapsed() -> float:
        return time.perf_counter() - t0

    try:
        while True:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if line:
                # Print child output as-is (it already ends with newline)
                print(line, end="", flush=True)
                last_output = time.perf_counter()
            else:
                # No output yet — show a heartbeat every 30 s of silence
                now = time.perf_counter()
                if now - last_output >= 30:
                    print(
                        f"  [{label}] waiting for output ... {_elapsed():.0f}s elapsed",
                        flush=True,
                    )
                    last_output = now
                time.sleep(0.1)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    finally:
        rc = proc.wait()
        # Clear any partial spinner residue
        print(flush=True)

    return rc


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from featureengineering.cli import main

        raise SystemExit(main())

    # Default (no args): run each class sequentially with tuned job counts

    # Configure logging early so we get a timestamped log file and
    # numpy / pandas warnings are captured.
    from featureengineering.settings import configure_paths, log_environment_info

    paths = configure_paths()
    log_environment_info()

    script = Path(__file__).resolve()
    import torch

    # 获取可用 GPU 数量（若没有 GPU 则返回 0）
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # 根据 GPU 数量决定并行任务数
    jobs_value = "33" if gpu_count >= 2 else "16"

    stages = [
        ("Class 1 (Panel)",     ["--only-class", "1", "--jobs", jobs_value,
                                "--quality-check-days", "5", "--dashboard"]),
        ("Class 2 (cyq_chips)", ["--only-class", "2", "--jobs", jobs_value]),
        ("Class 3 (intraday)",  ["--only-class", "3", "--jobs", jobs_value]),
        ("Class 4 (Coupling)",  ["--only-class", "4", "--jobs", jobs_value]),
    ]

    t_start = time.perf_counter()
    logger.info("Factor build started — %d stage(s)", len(stages))
    _log_banner("BUILD START")

    overall_rc = 0
    for i, (label, args) in enumerate(stages, 1):
        stage_t0 = time.perf_counter()
        _log_banner(f"STAGE {i}/{len(stages)}: {label}")
        logger.info("Stage %d/%d: %s — launching subprocess", i, len(stages), label)

        cmd = [sys.executable, str(script)] + args
        rc = _stream_subprocess(cmd, label)
        stage_elapsed = time.perf_counter() - stage_t0

        if rc != 0:
            msg = f"[WARN] {label} exited with code {rc} (elapsed: {stage_elapsed:.0f}s)"
            logger.warning(msg)
            print(f"\n{msg}", flush=True)
            overall_rc = 1
        else:
            logger.info(
                "Stage %d/%d: %s — OK (%.0fs)",
                i, len(stages), label, stage_elapsed,
            )

    total_elapsed = time.perf_counter() - t_start
    status = "OK" if overall_rc == 0 else "WARNINGS"
    logger.info(
        "Factor build finished — %s, total elapsed: %.0f s",
        status, total_elapsed,
    )
    _log_banner(f"BUILD DONE — {status} (total: {total_elapsed:.0f}s)")

    raise SystemExit(overall_rc)
