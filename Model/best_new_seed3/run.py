#!/usr/bin/env python3
"""V2 基线模型训练入口。

用法:
  python run.py                                   # 全数据训练 (fac_all, 默认 30 epochs)
  python run.py --data fac_sample --epochs 3      # 小样本验证
  python run.py --season 2026q3 --fold 1          # 指定 checkpoint 目录名 (默认即可)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import parse_args, train

# season 仅用于 checkpoint 目录命名 (model_train/{season}/foldN); 4-Fold 由 train.sh 依次启动
SEASON = "2026q3"

if __name__ == '__main__':
    args = parse_args()
    fold = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1

    for i, arg in enumerate(sys.argv):
        if arg == '--season' and i + 1 < len(sys.argv):
            SEASON = sys.argv[i + 1]
        if arg == '--fold' and i + 1 < len(sys.argv):
            fold = int(sys.argv[i + 1])

    train(args, season=SEASON, fold=fold, state="train")
