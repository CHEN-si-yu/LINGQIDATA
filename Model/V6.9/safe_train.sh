#!/bin/bash
# safe_train.sh <fold1> [fold2] — 内存安全的折并行训练
# 约束: 本容器 cgroup 内存上限 120G, 单折峰值 ~28-30G → 并发最多 2 折 (峰值 55-60G)。
# 用法: bash safe_train.sh 3 4     # 并行训练 fold3+fold4
#       bash safe_train.sh 1       # 只训练 fold1
# 每次运行以 setsid 完全脱离会话, 会话断开不影响训练; 完成后写入 logs/run_status.txt。
set -u
cd /autodl-fs/data/lingqiData/Model/V6.9
mkdir -p logs
export FORCE_TQDM_PROGRESS=1

f1="${1:?用法: safe_train.sh <fold1> [fold2]}"
f2="${2:-}"

echo "[$(date '+%F %T')] safe_train 启动 fold=$f1 $f2 (pid=$$)" >> logs/run_status.txt

CUDA_VISIBLE_DEVICES=0 nohup python run.py "$f1" > "logs/fold$f1.log" 2>&1 &
p1=$!
if [ -n "$f2" ]; then
  CUDA_VISIBLE_DEVICES=0 nohup python run.py "$f2" > "logs/fold$f2.log" 2>&1 &
  p2=$!
  wait "$p1" "$p2"
else
  wait "$p1"
fi
rc=$?
echo "[$(date '+%F %T')] safe_train 结束 fold=$f1 $f2 rc=$rc" >> logs/run_status.txt
exit $rc
