# V20c — V20 种子变体单元 2(唯一变量:base seed = 12345)

> **本单元不是新架构版本**:它是冠军单元 **V20 的独立重训复现**,唯一差别是
> `model.py` 的 `--seed` 默认值(3253 → **12345**)与 `root_path`(Model/V20c)。
> 用途:与 V20b 一起回答「V20 ens_w2 + h20tr20_t2 冠军 (+352.1%) 对训练随机种子是否敏感」。
> 配套:Model/V20b(seed=7777);综合结论见 Model/Trading/SEED_STABILITY_REPORT.md。

## 一、与 V20 的差异(干净因果对照)

| 项 | V20 | V20c |
|---|---|---|
| base seed (`model.py --seed`) | 3253 | **12345** |
| 每折种子 | 3253 + k*1000 (k=1..8) → 4253..11253 | 12345 + k*1000 → **13345..20345** |
| root_path | Model/V20 | Model/V20c |
| 架构/损失/数据划分/ridge 热启动/ens_w2 配方/analysis 逻辑 | — | **逐字节相同**(仅文案 banner 改 V20c) |

三个单元 (V20/V20b/V20c) 48 个折种子互不重合(mod 1000 余数分别为 253/777/345)。
数据划分按日期确定性切分 → 单元间差异 = **纯训练随机性**。

## 二、标准流程(与 V20 完全一致)

```bash
cd /autodl-fs/data/lingqiData/Model/V20c
bash train.sh            # 16 折 = 族a(fold1-8) + 族c(fold9-16), 4 批 × 4 折并发 (~1h)
python3 analysis.py      # 全窗口推演 → 组装 ens_w2 → score_ens_w2.fea + model_pic 报告
```

## 三、评估(官方口径在 Trading 层,见上)

- 本单元 `analysis.py` 自报告沿用 V20 原版文案(hold5s8-收盘卖 旧引擎口径),仅作
  单元自检/对照;**种子稳定性判读一律以 Trading 引擎 h20tr20_t2 真实净值口径为准**
  (iter7_seed_stability.py → results/seed_stability.csv → SEED_STABILITY_REPORT.md)。
- 训练健康参照:V20 fold1-4 best val_rankic = 0.0987/0.0661/0.0688/0.0740
  (种子不同,数值应同量级而非相同;如大幅偏离或 EXIT:1 按 TRAINING_PLAYBOOK 排查)。

## 四、产物(均不入库)

`model_train|test/2026q3/`、`model_pred/2026q3/heads_{a,c}/*.fea + score_ens_w2.fea`、
`model_pic/output.md + figure_01.png`、`logs/`。
