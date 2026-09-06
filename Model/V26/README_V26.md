# V26 — 冻结策略适配单元 2: 选择层策略适配 (Valid 监控混合 Top2-5d 代理)

> 背景: V25 (损失层顶部惩罚) ❌ 证伪 — 教训是"顶部监督只奖不罚, 损失层不动"。
> 本单元采纳用户建议: **模型仍以隔日(1d)收益为训练主目标, 但 Valid 上的
> checkpoint 选择/早停标尺从纯 1d RankIC 改为混入"策略代理指标"**, 不改损失、
> 不改架构 — 让"挑哪个 epoch 的模型"这件事对齐冻结慢腿策略
> (Trading/FIXED_STRATEGY.md v1.0)。

## 一、与 V20 的差异 (唯一增量 = 选择标尺)

| 项 | V20 | V26 |
|---|---|---|
| base seed | 3253 | **24681** (与 V25 相同 → 可对照归因) |
| 训练损失/架构/数据划分/ridge | — | **逐字节相同 (无任何损失改动)** |
| Valid 监控 | val_rankic (1d RankIC of mixed) | **val_combo = val_rankic + W·val_strat5** |
| val_strat5 | — | 每 Valid 日取 mixed 预测 **Top2**, 其 **5d 标签均值** 的全窗平均 (慢腿买 Top2、中位持有 ~6 日; Valid 30 天/折, 选择发生在 Test 前 → 不泄漏) |
| W | — | 预注册: cons 折 2.0 (IC 主导) / **aggr 折 6.0** (顶部主导, 让激进折承担策略适配) |
| ckpt 文件名/选择 | val_rankic | val_combo (analysis 同步优先解析) |

**风险预注册**: Valid 仅 30 天 × Top2 ≈ 60 观测, val_strat5 噪声大 → W 不过大
(2/6); 若选择层适配无效或有害 (在噪声上选模型), 如实记录负结论。

## 二、标准流程

```bash
cd /autodl-fs/data/lingqiData/Model/V26
bash train.sh            # 16 折, 6+6+4 分批 (~50min)
python3 analysis.py      # 全窗口推演 → heads + score_ens_w2 → model_pic 自检
```

## 三、评估 (官方口径 = Trading/FIXED_STRATEGY.md §二, 预注册通过线)

- 主判定: 慢腿 u_h20_re300_tr15 在 V26 score_ens_w2 上 cum ≥ **+80%** 且
  H1/H2 双正 (参照: V11_ensw2 +100.8% / V20b +55.8%);
- 次判定: 双腿 (快腿 V11_ensw2+D01) cum ≥ +150% 且 MaxDD ≥ -20%;
- 健康检查: Test RankIC ≥ 0.035; 若达标须第二种子复刻 (V26b) 确认。

## 四、产物 (均不入库)

`model_train|test/2026q3/`、`model_pred/2026q3/heads_{a,c}/*.fea + score_ens_w2.fea`、
`model_pic/`、`logs/`。

---

## 五、V26 迭代结果 (2026-09-06): ❌ 负结论 — 选择层适配有增益但未达线

| 标尺 | V26_ensw2 | 通过线 | V25 (同种子) | V20b 参照 | V11_ensw2 参照 |
|---|---|---|---|---|---|
| Test RankIC / IR | +0.0367 / 0.2287 | ≥0.035 | 0.0423 | 0.0406 | 0.0422 |
| 慢腿 cum | **+13.6%** | ≥+80% | +0.2% | +55.8% | +100.8% |
| 慢腿 H1/H2 | +20.6% / -5.8% | 双正 | — | — | — |
| 慢腿 MaxDD | -34.8% | — | — | -17.9% | -18.3% |
| 双腿 cum | +140.2% | ≥+150% | +132.9% | +159% | +181.7% |

**判定: 不通过 → 负结论存档。** 归因: (a) 同种子 24681 对照下, val_combo 选择
比纯 val_rankic 选择好 +13pp (V25 +0.2% → V26 +13.6%), 选择层适配方向**有真实
增益** (rank 笔均 +1.0 → +1.94pp), 但幅度不足以达线; (b) trail 亏损簇 6 笔 ×
-13.3pp 拖累 ~80pp — 选择层无法解决 trail 簇; (c) 24681 种子本身是弱抽取
(IC 0.0367 为谱系低位)。**经验: Valid 代理选择有效但边际 ~+13pp; 弱种子 +
trail 簇是当前主要损失来源。**
