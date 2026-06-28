#!/usr/bin/env python3
"""Training runner for V5.X series.

This is a thin wrapper around model.py's train() function.
Each V5.X version has an identical run.py — only model.py differs.

Usage:
    python run.py <fold> [--season SEASON]
    python run.py 1                    # Fold 1, all 6 seasons
    python run.py 1 --season 2026q2    # Fold 1, single season
"""

import sys
import os

# Ensure the current directory is on the path so `from model import ...` works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_basic_name, parse_args, train

MARKET = "ALL"
SEASONS = ["2026q2", "2026q1", "2025q4", "2025q3", "2025q2", "2025q1"]

if __name__ == '__main__':
    args = parse_args()
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    # Override fold and season from command line if provided
    for i, arg in enumerate(sys.argv):
        if arg == '--season' and i + 1 < len(sys.argv):
            SEASONS = [sys.argv[i + 1]]
        if arg == '--fold' and i + 1 < len(sys.argv):
            fold = int(sys.argv[i + 1])

    name = get_basic_name()

    for season in SEASONS:
        train(args, name=name, market=MARKET, season=season, fold=fold, state="train")
