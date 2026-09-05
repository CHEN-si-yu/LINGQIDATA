#!/usr/bin/env bash
# run_daily_pipeline_combo.sh — 组合系统每日流水线 (模型×策略组合层)
#
# 组合形态 (iter5/iter6, 稳定性优先):
#   默认 0.5/0.5: [V20_ensw2 + h20tr20_t2] × [V11_ensw2 + D01]
#     → 真实净值 +301%, Sharpe 3.75, MaxDD -16.0%, 滚动60日正率 97.8%
#   可选三系统 (--triple): + [V22_20d + S2_re300]
#     → +225~246%, Sharpe 3.8, MaxDD -14%, 滚动正率 100%
#
# 每晚收盘后 (数据更新完成后) 运行:
#   1) V20 analysis.py 增量推演 (heads → score_ens_w2)
#   2) V11 + V13 analysis.py 增量推演 (heads → score_ens_w2)
#   3) Trading/daily_runner_combo.py 各子系统两阶段纸面决策 (合并指令)
set -e
cd "$(dirname "$0")"

echo "=== [1/3] V20 增量推演 ==="
cd /autodl-fs/data/lingqiData/Model/V20
python3 analysis.py > logs/analysis_daily.log 2>&1 || true

echo "=== [2/3] V11 + V13 增量推演 ==="
cd /autodl-fs/data/lingqiData/Model/V11
python3 analysis.py > logs/analysis_daily.log 2>&1 || true
cd /autodl-fs/data/lingqiData/Model/V13
python3 analysis.py > logs/analysis_daily.log 2>&1 || true
# 重建 V11 ens_w2 (V11 heads + V13 heads, 冠军配方)
cd /autodl-fs/data/lingqiData/Model/V11
python3 - << 'EOF' > logs/ens_build.log 2>&1 || true
import pandas as pd
def zn(df):
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]
def build(heads_dir, w1, w3, w5, wt, nf=8):
    perfold = {h: {} for h in ('r1', 'r3', 'r5', 'top')}
    for h in perfold:
        for f in range(1, nf + 1):
            import os
            p = os.path.join(heads_dir, f'{h}_f{f}.fea')
            if os.path.exists(p):
                perfold[h][f] = pd.read_feather(p).set_index('date')
    score = None
    for f in sorted(perfold['r1'].keys()):
        fs = w1 * perfold['r1'][f] + w3 * perfold['r3'][f] + w5 * perfold['r5'][f] + wt * perfold['top'][f]
        fs = zn(fs)
        score = fs if score is None else score.add(fs, fill_value=0.0)
    return score
mixA = build('model_pred/2026q3/heads', 1.0, 0.25, 0.0, 2.0)
v13c = build('../V13/model_pred/2026q3/heads', 1.0, 0.0, 0.0, 1.0)
ens = zn(mixA).add(2.0 * zn(v13c), fill_value=0.0)
ens.index.name = 'date'
ens.reset_index().to_feather('model_pred/2026q3/score_ens_w2.fea')
print(f'ens_w2 更新: {ens.shape[0]} 天, 最新 {ens.index.max()}')
EOF

echo "=== [3/3] 组合决策 ==="
cd /autodl-fs/data/lingqiData/Model/Trading
MODE="${1:---pair}"
if [ "$MODE" = "--triple" ]; then
  python3 daily_runner_combo.py --triple
else
  python3 daily_runner_combo.py
fi
echo "=== 组合流水线完成 (状态 Trading/holdings_combo.json) ==="
