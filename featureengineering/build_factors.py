#!/usr/bin/env python3
"""Factor build entry point.

Default (no args): runs Class 1 → Class 5 sequentially.
    python build_factors.py

With args: delegates to featureengineering.cli.
    python build_factors.py --only-class 3
    python build_factors.py --only-class 1 --jobs 8 --dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    from featureengineering.cli import main

    if len(sys.argv) > 1:
        raise SystemExit(main())

    stages = [
        ["--only-class", "1", "--jobs", "28", "--quality-check-days", "5"],
        ["--only-class", "2", "--jobs", "50"],
        ["--only-class", "3", "--jobs", "50"],
        ["--only-class", "4", "--jobs", "50"],
        ["--only-class", "5", "--jobs", "50"],
    ]

    for args in stages:
        try:
            main(args)
        except SystemExit as e:
            if e.code and e.code != 0:
                raise
