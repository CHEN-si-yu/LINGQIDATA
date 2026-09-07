#!/usr/bin/env bash
# flex_best.sh — best(V8配方) 复现 + 3种子稳定性 灵活控制器
# 用法:
#   CONC=6 bash flex_best.sh                     # 全流程 (默认并发6)
#   CONC=6 bash flex_best.sh best best_new_seed3 # 只跑指定单元
#   bash flex_best.sh --stop                      # 优雅停止 (完成当前批次后退出)
#   bash flex_best.sh --status                    # 查看各单元进度
# 续跑: 已 EXIT:0 的折自动跳过; 已有 score 且新于训练日志则跳过 analysis。
set -u
cd "$(autodl-fs 2>/dev/null; echo /autodl-fs/data/lingqiData/Model)"
STOP_FILE="$PWD/.flex_stop"
CONC="${CONC:-6}"
UNITS="${@:-best best_new_seed1 best_new_seed2 best_new_seed3}"

if [ "${1:-}" = "--stop" ]; then touch "$STOP_FILE"; echo "[flex] 停止信号已发 (完成当前批次后退出)"; exit 0; fi
if [ "${1:-}" = "--status" ]; then
  for d in best best_new_seed1 best_new_seed2 best_new_seed3; do
    n=$(grep -c "EXIT:0" $d/logs/fold*.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
    sc=$([ -f $d/model_pred/2026q3/all_zscore_score.fea ] && echo yes || echo no)
    echo "  $d: 训练 $n/8, score=$sc"
  done
  exit 0
fi
rm -f "$STOP_FILE"

for d in $UNITS; do
  [ -f "$STOP_FILE" ] && { echo "[flex] 收到停止信号 — 在 $d 前退出"; break; }
  echo "[flex] ====== $d 开始 $(date +%H:%M) (CONC=$CONC) ======"
  cd "$PWD/$d"
  # 训练 (续跑跳过已完成折)
  bash train.sh 2>&1 | tee -a train_master.log
  n=$(grep -c "EXIT:0" logs/fold*.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
  if [ "$n" -ne 8 ]; then echo "[flex] $d 训练未齐 (EXIT:0=$n/8) — 中止链"; exit 1; fi
  echo "[flex] $d 训练完成 8/8"
  # 推演 (若已有 score 且新于最近 fold 日志则跳过 → 续跑友好)
  newest_fold_log=$(ls -t logs/fold*.log | head -1)
  if [ -f model_pred/2026q3/all_zscore_score.fea ] \
     && [ "$newest_fold_log" -nt model_pred/2026q3/all_zscore_score.fea ]; then
    echo "[flex] $d score 已最新, 跳过 analysis"
  else
    python3 -u analysis.py > analysis_run.log 2>&1
    echo "[flex] $d analysis exit $? (score: $(ls model_pred/2026q3/all_zscore_score.fea 2>/dev/null | wc -l))"
  fi
  cd "$PWD"
done
echo "[flex] ALL DONE $(date +%H:%M) (units: $UNITS)"
