#!/usr/bin/env bash
# train.sh (动态并发版): 并发折数 = $CONC (默认 6), 分块串行; 断点续跑(跳过已 EXIT:0 折)
# 用法: CONC=6 bash train.sh            # 全量 8 折
#       CONC=6 bash train.sh --folds "3 4"   # 只跑指定折 (续跑)
#       bash train.sh --clean           # 清空产物后全量 (并发默认6)
set -u
cd "$(dirname "$0")"
CONC="${CONC:-6}"
FOLDS_LIST="1 2 3 4 5 6 7 8"
if [ "${1:-}" = "--folds" ]; then FOLDS_LIST="${2:-$FOLDS_LIST}"; shift 2; fi
if [ "${1:-}" = "--clean" ]; then
  rm -rf model_train model_test model_pred model_pic logs __pycache__ train_master.log
  mkdir -p logs
  FOLDS_LIST="1 2 3 4 5 6 7 8"
fi
mkdir -p logs
export FORCE_TQDM_PROGRESS=1
run_batch() {
  local items="$1"
  local pids=""
  for x in $items; do
    if grep -q "fold$x EXIT:0" logs/fold$x.log 2>/dev/null; then
      echo "[train.sh] fold$x 已 EXIT:0, 跳过 (续跑)"
      continue
    fi
    rm -rf model_train/2026q3/fold$x          # 清残留防 version 冲突
    CUDA_VISIBLE_DEVICES=0 nohup bash -c "python run.py $x; echo \"fold$x EXIT:\$?\"" \
      > ./logs/fold$x.log 2>&1 &
    pids="$pids $!"
    sleep 8
  done
  [ -n "$pids" ] && wait $pids
}
i=0; B=""
for f in $FOLDS_LIST; do
  B="$B $f"; i=$((i+1))
  if [ $i -ge $CONC ]; then echo "[train.sh] 批次:{ $B } 并发=$CONC"; run_batch "$B"; B=""; i=0; fi
done
if [ -n "$B" ]; then echo "[train.sh] 批次:{ $B } 并发=$CONC"; run_batch "$B"; fi
echo "TRAIN DONE (folds: $FOLDS_LIST)"
