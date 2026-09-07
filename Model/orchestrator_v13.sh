#!/usr/bin/env bash
# orchestrator_v13.sh — V13 种子敏感性全流程 (V13b/V13c 训练 → 推演 → iter8)
set -u
cd /autodl-fs/data/lingqiData/Model

check_folds() {
  local unit="$1"
  local ok=0
  for f in 1 2 3 4 5 6 7 8; do
    if grep -q "fold$f EXIT:0" "$unit/logs/fold$f.log" 2>/dev/null; then ok=$((ok+1)); fi
  done
  echo "$ok"
}

# --- V13b: 等训练完成 (train_master.log 由链式任务写入) ---
echo "[orch] 等待 V13b 训练完成..."
while ! grep -q "ALL 8 FOLDS DONE" V13b/train_master.log 2>/dev/null; do sleep 20; done
V13B_OK=$(check_folds V13b)
echo "[orch] V13b 训练结束, EXIT:0 = ${V13B_OK}/8"
if [ "$V13B_OK" -ne 8 ]; then
  echo "[orch] V13b 训练异常 — 中止"; exit 1
fi

# --- V13b analysis ---
cd V13b && python3 analysis.py > analysis_run.log 2>&1
echo "[orch] V13b analysis exit $?"
ls model_pred/2026q3/ 2>/dev/null | head -3
cd ..

# --- V13c 训练 ---
echo "[orch] 启动 V13c 训练"
cd V13c && nohup bash train.sh > train_master.log 2>&1
echo "[orch] V13c 训练完成"
V13C_OK=$(check_folds V13c)
echo "[orch] V13c EXIT:0 = ${V13C_OK}/8"
if [ "$V13C_OK" -ne 8 ]; then
  echo "[orch] V13c 训练异常 — 中止"; exit 1
fi

# --- V13c analysis ---
python3 analysis.py > analysis_run.log 2>&1
echo "[orch] V13c analysis exit $?"
ls model_pred/2026q3/ 2>/dev/null | head -3
cd ..

# --- iter8 种子稳定性 ---
cd Trading && python3 iter8_seed_stability_v13.py > iter8_run.log 2>&1
echo "[orch] iter8 exit $?"
echo "[orch] ALL V13 SEED WORK DONE"
