import sys

from model import get_basic_name, parse_args, train

name = get_basic_name()
args = parse_args()

# ===== 在这里控制季度 =====
SEASONS = ["2024q1", "2024q2", "2024q3", "2024q4",
           "2025q1", "2025q2", "2025q3", "2025q4",
           "2026q1", "2026q2"]
MARKET = "ALL"

if __name__ == "__main__":
    fold = int(sys.argv[1])  # fold 由 train.sh 传入
    for season in SEASONS:
        train(args, name=name, market=MARKET, season=season, fold=fold, state="train")
