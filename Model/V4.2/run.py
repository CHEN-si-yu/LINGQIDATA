#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import parse_args, train

SEASONS = ["2026q3","2026q2","2026q1","2025q4","2025q3","2025q2","2025q1"]
if __name__ == '__main__':
    args = parse_args()
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    for i, arg in enumerate(sys.argv):
        if arg == '--season' and i + 1 < len(sys.argv):
            SEASONS = [sys.argv[i + 1]]
        if arg == '--fold' and i + 1 < len(sys.argv):
            fold = int(sys.argv[i + 1])

    for season in SEASONS:
        train(args, season=season, fold=fold, state="train")
