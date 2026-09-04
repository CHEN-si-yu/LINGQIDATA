#!/usr/bin/env bash
# run_daily_pipeline.sh — 冠军系统每日流水线 (ens_w2 + hold5s8-收盘卖)
# 1. V11 / V13 各自增量推演 (重推最新 10 个交易日)
# 2. 构建 ens_w2 打分 = z(mixA) + 2·z(v11c_top)
# 3. daily_runner/paper_trader 输出决策并滚动记录纸面交易 (Top1 持有5天 + -8%收盘止损)
set -e
cd "$(dirname "$0")"

echo "=== [1/3] V11 (=Model/V11) 增量推演 ==="
cd "$(dirname "$0")"
python3 analysis.py > logs/analysis.log 2>&1 || true
cd ../V13
echo "=== [2/3] V13 增量推演 ==="
python3 analysis.py > logs/analysis.log 2>&1 || true

echo "=== [2.5/3] 构建 ens_w2 ==="
cd ../V11
python3 - << 'EOF'
import os
import pandas as pd

def zn(df):
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]

def build(heads_dir, w1, w5, w3, wt, heads=('r1', 'r5', 'r3', 'top'), nf=8):
    perfold = {h: {} for h in heads}
    for h in heads:
        for f in range(1, nf + 1):
            p = os.path.join(heads_dir, f'{h}_f{f}.fea')
            if os.path.exists(p):
                perfold[h][f] = pd.read_feather(p).set_index('date')
    score = None
    for f in sorted(perfold['r1'].keys()):
        fs = w1 * perfold['r1'][f]
        if w5 and 'r5' in perfold:
            fs = fs + w5 * perfold['r5'][f]
        if w3 and 'r3' in perfold:
            fs = fs + w3 * perfold['r3'][f]
        fs = fs + wt * perfold['top'][f]
        fs = zn(fs)
        score = fs if score is None else score.add(fs, fill_value=0.0)
    return score

mixA = build('model_pred/2026q3/heads', 1.0, 0.0, 0.25, 2.0)
v11c = build('../V13/model_pred/2026q3/heads', 1.0, 0.0, 0.0, 1.0,
             heads=('r1', 'r5', 'r3', 'top'))
ens = zn(mixA).add(2.0 * zn(v11c), fill_value=0.0)
ens.index.name = 'date'
ens.reset_index().to_feather('model_pred/2026q3/score_ens_w2.fea')
print(f'ens_w2 更新完成: {ens.shape[0]} 天 × {ens.shape[1]} 股票, '
      f'最新日期 {ens.index.max()}')
EOF

echo "=== [3/3] 每日决策 + 纸面交易追踪 ==="
python3 daily_runner.py model_pred/2026q3/score_ens_w2.fea
python3 paper_trader.py model_pred/2026q3/score_ens_w2.fea
