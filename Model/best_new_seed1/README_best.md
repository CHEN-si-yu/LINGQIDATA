# Model/best — 最佳模型独立单元 (可单独训练/推演/测试)

> 2026-09-06 建立。回答"目前最佳的模型是谁"并给一个可独立测试的单元。

## 一、最佳是谁 (冻结双腿策略口径, Test 20250901~20260901)

| 口径 | 打分源 | 慢腿 cum | 双腿 cum |
|---|---|---|---|
| **全场最优打分实例** | **V11_ensw2** (Model/V11/model_pred/2026q3/score_ens_w2.fea) | **+100.8%** | **+181.7%** |
| 可自训配方最优单实例 | V29 配方 (seed 7777) | +73.7% | +169.4% |
| ens_w2 谱系稳健参照 | V20b (seed 7777) | +55.8% | +159% |

说明: V11_ensw2 的 +100.8% 是 V11 时代管线 (V11 8 折 + V13 8 折 heads 组装) 的
单实例成绩 — 换种子重训无法再生该实例 (种子彩票, 期望 ~+40~60%)。**本单元 =
"最佳配方"的可自训封装**: ens_w2 双族 16 折统一配方 + 已采纳改进 (hl=300,
val_combo 选择层), seed 默认 7777 (即 V29 单元配方, 慢腿 +73.7% 为可复现最佳)。

## 二、本单元 = 独立 VX 同构单元

- model.py: V20 统一 ens_w2 双族 (族a V11多目标 + 族c V13顶部3d) + hl=300
  + val_combo 选择层 (cons W2 / aggr W6), ridge 热启动, 16 折 = 8+8;
- run.py / train.sh (6+6+4 分批) / analysis.py (推演 → heads_a/c →
  score_ens_w2.fea → 冻结双腿回测 + 当日纸面操作);
- 产物不入库 (model_train/test/pred/pic/logs), 脚本与 README 入库。

## 三、用法

```bash
cd /autodl-fs/data/lingqiData/Model/best
bash train.sh            # 全量训练 16 折 (~50 min; 换种子: 改 model.py --seed 默认)
python3 analysis.py      # 推演 + score_ens_w2.fea + 冻结双腿回测 + 当日操作输出
```

每日收盘后数据更新完, 运行 `python3 analysis.py` 即得当日操作 (状态机推进,
纸面状态 Trading/holdings_frozen.json)。回放校验: `python3 ../Trading/daily_frozen.py --replay`。

## 四、独立测试建议

1. 复现 V29 单实例: 用默认 seed 7777 训练 → 慢腿预期 ~+74% (训练随机性 ±);
2. 种子分布: 改 seed 再训 (如 13579/24681) → 观察慢腿 +2.7~+74% 的种子彩票区间,
   与 V11_ensw2 (+100.8%) 对照 — 结论见 FINAL_REPORT §0.6。
