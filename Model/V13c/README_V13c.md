# V13c — V13 种子变体单元 2 (唯一变量: base seed = 12345)

> **本单元不是新架构版本**: 它是 **V13 顶部分支 3d 主目标单元** 的独立重训复现,
> 唯一差别是 `model.py` 的 `--seed` 默认值 (3253 → **12345**) 与 `root_path` (Model/V13c)。
> 用途: 回答「V13_allz + S7 (h10_tr12_t1) 冠军组合 (+538.7% 真实净值) 对训练随机
> 种子是否敏感」。配套: Model/V13b (seed=7777); 综合结论见
> Model/Trading/SEED_STABILITY_V13_REPORT.md。

## 一、与 V13 的差异 (干净因果对照)

| 项 | V13 | V13c |
|---|---|---|
| base seed (`model.py --seed`) | 3253 | **12345** |
| 每折种子 | 3253 + k*1000 (k=1..8) | 12345 + k*1000 → 13345..20345 |
| 数据划分 | seed 3253 | **固定 SPLIT_SEED=3253 (与 V13 完全一致)** |
| root_path | Model/V13 | Model/V13c |
| 架构/损失/ridge 热启动/analysis 逻辑 | — | **逐字节相同** (仅 banner 改 V13c) |

> 注: 划分种子固定为 3253 (V20b/V20c 的划分曾随 seed 变化); 本单元仅变化
> **训练随机性** (初始化/dropout/批序), 是比 V20 测试更干净的因果对照。

## 二、标准流程 (与 V13 完全一致)

```bash
cd /autodl-fs/data/lingqiData/Model/V13c
bash train.sh            # 8 折 = fold1-4 保守 + fold5-8 激进, 2 批 × 4 折并发 (~30min)
python3 analysis.py      # 全窗口推演 → all_zscore_score.fea + model_pic 报告
```

## 三、评估 (官方口径在 Trading 层)

- 主协议: S7 = Top1 + 持有≤10日 + 移动止损12% (开盘先卖后买, 1 份资金, 含成本)。
- 对照协议: S6 (h20tr20_t2) / S4 (re300) / S5 (re300_sl8_tr15)。
- 种子稳定性综合结论: Model/Trading/SEED_STABILITY_V13_REPORT.md。
