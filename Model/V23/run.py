#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — V23 训练入口 (每折一个进程; train.sh 会以全局 fold 1..16 按 §2 内存规则分批调用, 120GB→6 折/批)

用法:
  python run.py <fold>                          # 训练全局折 1..16 (族/风格/划分自动映射)
  python run.py <fold> --epochs 3 --data fac_sample   # 小样本冒烟验证
  python run.py --fold 3 --season 2026q3

全局折 → (族, 族内折) 映射 (见 model.fold_spec):
  fold 1..8  = 族 a (V11 多目标: 顶分支 1d 主)  , k=1..8 (k≤4 保守 / k>4 激进)
  fold 9..16 = 族 c (V13 顶部3d: 顶分支 3d 主)  , k=1..8 (k≤4 保守 / k>4 激进)
训练产物: model_train/{season}/fold{全局折}/...  (checkpoint = 最优 val_rankic)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import parse_args, train, fold_spec, TOTAL_FOLDS  # noqa: E402

SEASON = '2026q3'


if __name__ == '__main__':
    args = parse_args()
    fold = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
    for i, arg in enumerate(sys.argv):
        if arg == '--season' and i + 1 < len(sys.argv):
            SEASON = sys.argv[i + 1]
        if arg == '--fold' and i + 1 < len(sys.argv):
            fold = int(sys.argv[i + 1])

    fam, k = fold_spec(fold)
    print(f'[V23] run.py 启动: 全局 fold {fold}/{TOTAL_FOLDS} → '
          f'族 {fam} ({"trail主/10d辅" if fam == "a" else "trail主/20d辅"}), '
          f'族内折 k={k}, season={SEASON}', flush=True)
    train(args, season=SEASON, fold=fold, state='train')
