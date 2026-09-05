# V22 — 长周期 GBDT 排序器单元 (10d/20d 标签, 模型×策略耦合)

**V22 = 耦合迭代第 2 号单元** (2026-09-05): 用修复后的 V18 LightGBM 管线
(numpy 链式索引 bug 已修) 训练 **10d/20d 标签** 的 GBDT 排序器, 检验非线性树模型
在 h20tr20_t2 协议 (平均持有 ~19 交易日) 上是否提供正交 alpha。

## 一、口径 (与 V18/V20 完全同口径)

- 特征: 848 因子, 逐日截面 rank → 标准化 → 缺失填 0 (float32)
- 标签: **10d / 20d 收益** winsor(MAD5) → 逐日截面秩高斯化 (V18 为 1d)
- 划分: 4 折 (seed 3253), train < 20250809, valid = 最后 120 天, purge 5
- 权重: 时间衰减 hl=600d; LGBMRegressor(l2, 3000 轮, lr 0.05, leaves 63,
  min_child 100, colsample 0.6, early_stopping 100)
- 输出: `model_pred/v22_{label_ret_10d,label_ret_20d}_f{1..4}.fea` (逐折日截面 z)
  + `v22_{label_ret_10d,label_ret_20d}.fea` (跨折 z 求和)

## 二、用法

```bash
cd /autodl-fs/data/lingqiData/Model/V22
V22_NJOBS=10 python3 train_gbdt.py    # 预处理 ~15min + 8 次 fit (resume 安全)
```

## 三、初步结果 (2026-09-05)

- v22_10d 与 V20_ensw2 逐日相关 ~0.49 (正交性好), 但策略层弱:
  h20tr20_t2 真实净值 +9% (V20_ensw2 +352%); 任何权重 (0.25~2.0) 混入
  V20_ensw2 均稀释收益 → **GBDT 组件 (1d/10d) 在冠军策略层无增量**, 与
  V18 1d 结论一致 (FINAL_REPORT: 信号近似线性, 树模型无正交 alpha)。
- 待办: 20d 结果同样检验; 若同样为负 → 记录死胡同 (GBDT 组件线) 并封存。

## 四、结论纪律

- 负结论也是结论: 若 10d/20d 均无增量, 将写入 TRAINING_PLAYBOOK 死胡同清单,
  明确 "GBDT 排序器组件 (1d/10d/20d 标签) 在 Top 选择策略层无正交 alpha"。
