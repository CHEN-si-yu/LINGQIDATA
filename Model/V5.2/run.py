"""V5.2 training runner — quarterly retrain + weekly online fine-tuning.

Usage:
    python run.py <fold>                              # train one fold across all seasons
    python run.py <fold> --season 2026q2 --online     # train + weekly online update
    python run.py --season 2026q2 --fold 1 --online   # same, keyword args
"""

import sys

from model import get_basic_name, parse_args, train

name = get_basic_name()
args = parse_args()

# ===== Resume from 2025Q2 (2024Q1-2025Q1 already completed) =====
SEASONS = ["2025q2", "2025q3", "2025q4", "2026q1", "2026q2"]
MARKET = "ALL"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", type=int, nargs="?", default=None)
    ap.add_argument("--season", type=str, default=None)
    ap.add_argument("--fold", dest="fold_kw", type=int, default=None)
    ap.add_argument("--online", action="store_true", default=False,
                    help="Enable weekly online fine-tuning after quarterly training")
    opts = ap.parse_args()

    fold = opts.fold or opts.fold_kw
    if fold is None:
        print("Usage: python run.py <fold>  (fold=1,2,3,4)")
        sys.exit(1)

    seasons = [opts.season] if opts.season else SEASONS

    for season in seasons:
        train(args, name=name, market=MARKET, season=season, fold=fold, state="train")

    # ── V5.2: weekly online fine-tuning for the prediction season ────
    if opts.online and opts.season:
        pred_season = opts.season
        print(f"\n{'='*60}")
        print(f"[V5.2] Starting weekly online updates for {pred_season}")
        print(f"{'='*60}")
        train(args, name=name, market=MARKET, season=pred_season, fold=fold, state="online")
