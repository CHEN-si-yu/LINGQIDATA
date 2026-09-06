#!/usr/bin/env bash
# best 复现 + 3 新种子稳定性: 依次 train → analysis (每单元)
set -u
cd /autodl-fs/data/lingqiData/Model
for d in best best_new_seed1 best_new_seed2 best_new_seed3; do
  echo "[orch] ==== $d 训练开始 $(date +%H:%M) ===="
  cd $d
  bash train.sh > train_master.log 2>&1
  n=$(grep -c "EXIT:0" logs/fold*.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
  if [ "$n" -ne 16 ]; then echo "[orch] $d 训练异常 EXIT:0=$n — 中止"; exit 1; fi
  echo "[orch] $d 训练完成 (16/16)"
  python3 -u analysis.py > analysis_run.log 2>&1
  echo "[orch] $d analysis exit $?"
  cd ..
done
echo "[orch] ALL BEST+SEEDS DONE $(date +%H:%M)"
