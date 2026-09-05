#!/usr/bin/env bash
# run_daily_pipeline_v2.sh — h20tr20_t2 协议每日流水线 (V20 ens_w2 打分 + Top2 双仓)
#
# 每晚收盘后 (数据更新完成后) 运行:
#   1) V20 analysis.py 增量推演 (重推最新因子日, 组装 score_ens_w2)
#   2) Trading/daily_runner_v2.py 纸面决策 (结算今日开盘指令 + 产出明日开盘指令)
#
# 协议口径: Top2 等权, 持有 ≤20 交易日, 自峰值移动止损 20%, 开盘先卖后买,
#           利润再投资等权再平衡; 与 Trading/engine.py 逐笔一致 (回放校验 24/24)。
set -e
cd "$(dirname "$0")"

echo "=== [1/2] V20 增量推演 + ens_w2 组装 ==="
cd /autodl-fs/data/lingqiData/Model/V20
python3 analysis.py > logs/analysis_daily.log 2>&1 || true
SCORE=/autodl-fs/data/lingqiData/Model/V20/model_pred/2026q3/score_ens_w2.fea
if [ ! -s "$SCORE" ]; then
  echo "!! ens_w2 打分缺失: $SCORE"
  exit 1
fi

echo "=== [2/2] 每日决策 (h20tr20_t2) ==="
cd /autodl-fs/data/lingqiData/Model/Trading
python3 daily_runner_v2.py "$SCORE"
echo "=== 流水线完成 (决策见上; 状态 Trading/holdings_v2.json) ==="
