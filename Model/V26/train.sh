#!/usr/bin/env bash
# ============================================================
# train.sh — V26 训练调度: 16 折 = 族a(8) + 族c(8)
# V26 假设: 顶部分支"下行不对称惩罚" (TOP_DOWN) — 抑制顶部误选深亏股,
# 适配冻结慢腿策略 (u_h20_re300_tr15, 见 Trading/FIXED_STRATEGY.md)。
# 分批: 6 折/批 ×2 + 4 折/批 (用户规格: 120GB 内存可跑 6 折并发;
# 实测 4 折峰值 ~25GB → 6 折 ~40GB, 远低于 100GB 红线)。
#
# 用法:
#   bash train.sh              # 全量训练 16 折 (已存在产物则拒绝)
#   bash train.sh --clean      # 清理 model_train/test/pred/pic/logs 后全量训练
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " V26 全量训练: 16 折 = 族 a (V11 多目标, fold1-8) + 族 c (V13 顶部3d, fold9-16)"
echo " 每族 k1-4 保守(IC主导) / k5-8 激进(顶部主导); 批次 6+6+4 折并发"
echo " 唯一结构增量: 选择层策略适配 (val_combo = val_rankic + W·Top2-5d 代理; 不改损失)"
echo "============================================================"
mkdir -p ./logs

if [ "$1" = "--clean" ]; then
  echo "[train.sh] --clean: 清理旧训练/推演产物 ..."
  rm -rf model_train model_test model_pred model_pic logs __pycache__
  mkdir -p logs
fi

if [ -d "model_train/2026q3" ] && [ "$(ls -A model_train/2026q3 2>/dev/null)" ]; then
  echo "[train.sh] 检测到已有训练产物 — 拒绝直接重跑。如需重训: bash train.sh --clean"
  exit 1
fi

export FORCE_TQDM_PROGRESS=1

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
echo " 下一步: python3 analysis.py  (全窗口推演 + ens_w2 出榜)"
echo " 官方评估: Trading/FIXED_STRATEGY.md 慢腿 u_h20_re300_tr15 + 双腿"
echo "============================================================"
