"""V2 training runner.

Usage:
    python run.py <fold>                # train one fold across all seasons
    python run.py <fold> --season 2025q1  # train one fold for one season
    python run.py --season 2025q1 --fold 1  # same, keyword args
"""

import sys

from model import get_basic_name, parse_args, train

name = get_basic_name()
args = parse_args()

# ===== 10 quarters for cross-sectional training =====
SEASONS = ["2024q1", "2024q2", "2024q3", "2024q4",
           "2025q1", "2025q2", "2025q3", "2025q4",
           "2026q1", "2026q2"]
MARKET = "ALL"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", type=int, nargs="?", default=None)
    ap.add_argument("--season", type=str, default=None)
    ap.add_argument("--fold", dest="fold_kw", type=int, default=None)
    opts = ap.parse_args()

    fold = opts.fold or opts.fold_kw
    if fold is None:
        print("Usage: python run.py <fold>  (fold=1,2,3,4)")
        sys.exit(1)

    seasons = [opts.season] if opts.season else SEASONS

    for season in seasons:
        train(args, name=name, market=MARKET, season=season, fold=fold, state="train")
