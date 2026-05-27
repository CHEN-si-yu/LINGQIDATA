#!/bin/bash
# V5.2 training — 2-fold batched for single-GPU stability
export FORCE_TQDM_PROGRESS=1
mkdir -p ./logs

echo "=== Batch 1: Fold 1 + Fold 2 ==="
CUDA_VISIBLE_DEVICES=0 nohup python run.py 1 > ./logs/fold1.log 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=0 nohup python run.py 2 > ./logs/fold2.log 2>&1 &
PID2=$!
echo "Fold1 PID=$PID1  Fold2 PID=$PID2"
wait $PID1 $PID2
echo "=== Batch 1 done ==="

sleep 5

echo "=== Batch 2: Fold 3 + Fold 4 ==="
CUDA_VISIBLE_DEVICES=0 nohup python run.py 3 > ./logs/fold3.log 2>&1 &
PID3=$!
CUDA_VISIBLE_DEVICES=0 nohup python run.py 4 > ./logs/fold4.log 2>&1 &
PID4=$!
echo "Fold3 PID=$PID3  Fold4 PID=$PID4"
wait $PID3 $PID4
echo "=== All four folds done. ==="
