#!/usr/bin/env bash
# ============================================================
# train.sh — V20 "训练" 入口 (冻结冠军校验)
#
# V20 不训练任何新参数: 模型 = V11(多目标族) + V13(顶部3d族) 冻结集成 (ens_w2)。
# 本脚本负责"训练期"应做的事 — 校验冠军组件齐备/数据覆盖, 并给出启动指引。
#
# 用法:
#   bash train.sh               # 健康检查 (exit 0 = 就绪)
#   bash train.sh --full        # 同检查 + 打印组件/打分概要
#
# 若需真正重训基座模型, 请分别进入 Model/V11、Model/V13 执行各自的 train.sh
# (V20 冠军集成公式见 model.py 顶部, 训练完成后用 analysis.py 重新出榜)。
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " V20: 冻结冠军系统 (ens_w2 = z(V11 mixA) + 2·z(V13 r1+top))"
echo " 本版无可训练参数; 以下为组件健康检查。"
echo "============================================================"
python3 -c "
import sys
sys.path.insert(0, '.')
import model
ok, checks = model.validate_frozen(verbose=True)
if not ok:
    print('存在 FAIL 项: 请检查 V11/V13 是否已训练并完成 analysis.py 全量推演')
    sys.exit(1)
print('就绪。每日出结果: python3 analysis.py ; 回测: python3 analysis.py --backtest')
"

if [ "$1" = "--full" ]; then
  echo ""
  echo "----- 打分概要 (由现有 heads 组装) -----"
  python3 -c "
import sys
sys.path.insert(0, '.')
import model
sc = model.score_ens_w2()
print(f'ens_w2: {sc.shape[0]} 天 x {sc.shape[1]} 股票  {sc.index.min()} ~ {sc.index.max()}')
print(f'因子最新日: {model.latest_reportable_date()}')
"
fi
