# V29 — 已采纳组件叠加: hl=300 (V27✓) × val_combo 选择层 (V26✓)

> 模型发展向迭代。V25~V28 单轴实验后, 两个方向为正贡献: 时间衰减 hl=300
> (V27: 同种子 +5.6pp, dd -3pp) 与 Valid 选择标尺 val_combo = val_rankic +
> W·Top2-5d 代理 (V26: 同种子 +13pp)。V29 把两者叠加 (V26 的 monitor 改动 +
> V27 的 hl=300), 种子 7777 延续同种子对照链 → 干净归因于"组件叠加"。
> 参照: V27 +61.4% / 通过线 慢腿≥+80% 且 H1/H2 双正。

## 一、与 V27 的差异
仅 monitor 变为 val_combo (cons W=2 / aggr W=6), 其余逐字节同 V27 (hl=300, seed 7777)。

## 二、流程/评估
```bash
bash train.sh && python3 analysis.py
```
慢腿 (参照 V27 +61.4%) / 双腿 ≥+150% (V27 +163.4%) / RankIC ≥0.035。
