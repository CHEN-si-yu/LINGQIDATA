#!/bin/bash
# wait_and_eval.sh — 等待 safe_train.sh 完成, 然后跑 4 折完整集成评估
cd /autodl-fs/data/lingqiData/Model/V6.9 || exit 1

# 等待训练结束标记 (最多 3 小时)
for i in $(seq 1 180); do
    if grep -q "safe_train 结束" logs/run_status.txt 2>/dev/null; then
        break
    fi
    sleep 60
done

echo "[$(date '+%F %T')] 训练结束, 等待文件落盘..." >> logs/run_status.txt
sleep 30

# 各折 best 汇总 (从 checkpoint 文件名)
echo "=== 各折 best checkpoint ==="
for f in 1 2 3 4; do
    best=$(ls model_train/2026q3/fold$f/*/checkpoints/*.ckpt 2>/dev/null | sed 's/.*epoch=//; s/-val_rankic_avg=/ /; s/\.ckpt//' | sort -t' ' -k2,2rn | head -1)
    echo "fold$f: $best" | tee -a logs/run_status.txt
done

# 完整 4 折集成评估
echo "[$(date '+%F %T')] 启动 4 折集成评估" >> logs/run_status.txt
python eval_ensemble.py 1 2 3 4 > logs/eval_full.log 2>&1
rc=$?
echo "[$(date '+%F %T')] 集成评估完成 rc=$rc" >> logs/run_status.txt
