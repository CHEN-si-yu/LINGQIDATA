#!/usr/bin/env bash
# ============================================================
# train.sh — V20c 训练调度: 16 折 = 族a(8) + 族c(8), 分 3 批 × 6/6/4 折并发启动 (120GB→6, playbook §2)
#
# 全局折 1..16:
#   fold 1..8  = 族 a (V11 多目标: 顶分支 1d 主),  k1-4 保守 / k5-8 激进
#   fold 9..16 = 族 c (V13 顶部3d: 顶分支 3d 主),  k1-4 保守 / k5-8 激进
#
# 内存/GPU 铁律 (TRAINING_PLAYBOOK): 同一时刻最多 6 折并发 (cgroup 120GB → 6, playbook §2)。
#
# 用法:
#   bash train.sh              # 全量训练 16 折 (若已存在训练产物则拒绝, 防 checkpoint 混淆)
#   bash train.sh --clean      # 先清理旧 model_train/model_pred/model_test/logs 再全量训练
#
# 训练完成 → python3 analysis.py  (全窗口推演 + 冠军集成 + 排行榜/回测)
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " V20c 全量训练: 16 折 = 族 a (V11 多目标, fold1-8) + 族 c (V13 顶部3d, fold9-16)"
echo " 每族 k1-4 保守(IC主导) / k5-8 激进(顶部主导); 3 批 × 6/6/4 折并发"
echo "============================================================"
mkdir -p ./logs

if [ "$1" = "--clean" ]; then
  echo "[train.sh] --clean: 清理旧训练/推演产物 (model_train model_test model_pred model_pic logs) ..."
  rm -rf model_train model_test model_pred model_pic logs __pycache__
  mkdir -p logs
  echo "[train.sh] 清理完成, 开始全新训练"
fi

if [ -d "model_train/2026q3" ] && [ "$(ls -A model_train/2026q3 2>/dev/null)" ]; then
  echo "[train.sh] 检测到已有训练产物 model_train/2026q3 — 为避免旧 checkpoint 混淆"
  echo "          (训练手册铁律), 拒绝直接重跑。如需重训: bash train.sh --clean"
  exit 1
fi

export FORCE_TQDM_PROGRESS=1

# 3 批: 6/6/4 折并发 (120GB→6, playbook §2; nohup 且不带 timeout)
for BATCH in "1 2 3 4 5 6" "7 8 9 10 11 12" "13 14 15 16"; do
  echo "[train.sh] 启动批次: fold ${BATCH}"
  for f in $BATCH; do
    CUDA_VISIBLE_DEVICES=0 nohup bash -c "python run.py $f; echo \"fold$f EXIT:\$?\"" \
      > ./logs/fold$f.log 2>&1 &
    sleep 10
  done
  wait
  echo "[train.sh] 批次完成: fold ${BATCH}"
done

echo "============================================================"
echo " ALL 16 FOLDS DONE — 训练产物: model_train/2026q3/fold{1..16}"
echo " 下一步: python3 analysis.py   (全窗口推演 + ens_w2 出榜)"
echo "         python3 analysis.py --backtest  (冠军协议回测)"
echo "============================================================"
