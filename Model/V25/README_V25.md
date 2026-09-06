# V25 — 冻结策略适配单元 1: 顶部分支下行不对称惩罚 (TOP_DOWN)

> 目标: 针对**冻结慢腿策略** `u_h20_re300_tr15` (Trading/FIXED_STRATEGY.md v1.0)
> 提升模型适配。假设基于 iter12 诊断: 30 打分集 × 慢腿的盈亏分解显示
> (a) 排名退出单笔质量是收益主引擎 (与 cum 相关 +0.53);
> (b) trail 移动止盈退出是**稳定亏损簇** (每笔 -10~-18%, 每年 2~8 笔 ≈ -20~-60pp 拖累);
> (c) 高分组 vs 低分组 mean_rank = +1.86% vs +1.13%/笔。
> → 顶部误选的"将深亏股"同时伤害 (a)(b)。V25 给顶部分支加**下行不对称惩罚**:
> 被当前模型预测进 Top-k 的股票, 若未来标签为负, 按亏损幅度惩罚 (把它们的打分
> 从顶部拉下来)。这是不对称的: 只惩罚"预测高+实际亏", 不惩罚"预测低+实际涨"
> (后者由 rank/IC 损失负责), 理论上不钝化上行锐度。

## 一、与 V20 的差异 (唯一结构增量)

| 项 | V20 | V25 |
|---|---|---|
| base seed | 3253 | **24681** (全新随机抽取) |
| 顶部分支损失 | soft-top + listnet + aux | + **下行惩罚**: `w_down·mean(max(0,-y)[pred_top_k])` |
| TOP_DOWN_WEIGHT | — | cons 0.2 (k_frac 0.05) / **aggr 1.0** (k_frac 0.03) |
| 惩罚标签周期 | — | 与顶部分支主目标同周期 (族a:1d / 族c:3d winsor 标签) |
| root_path / 架构 / 划分 / ridge 热启动 | — | 逐字节相同 (Model/V25) |

已知死胡同 (不重复): 长周期点标签 (V21/22)、轨迹标签 (V23)、弱长周期辅助
(V24)、新种子扩展 (V16)、打分平滑、打分层跨模型 z-均 (iter12 先导: ens 均钝化,
最佳 ens3 仅 +34.6% < 单模型 V11_ensw2 +100.8%)。

## 二、标准流程

```bash
cd /autodl-fs/data/lingqiData/Model/V25
bash train.sh            # 16 折 = 族a(fold1-8) + 族c(fold9-16), 6+6+4 折分批 (~50min)
python3 analysis.py      # 全窗口推演 → heads + score_ens_w2 → model_pic 自检报告
```

## 三、评估 (官方口径 = Trading/FIXED_STRATEGY.md §二, 预注册通过线)

主判定: 慢腿 u_h20_re300_tr15 在 V25 score_ens_w2 上:
- 通过线: cum ≥ **+80%** 且 H1/H2 双正 (基线: V11_ensw2 +100.8% / V20b +55.8%);
- 次判定: 双腿组合 (快腿 V11_ensw2+D01) cum ≥ +150% 且 MaxDD ≥ -20%
  (基线 V11_ensw2 双腿 +181.7%);
- 健康检查: Test RankIC ≥ 0.035, 16 折 best val_rankic 与 V20 同量级。
- 通过后须第二种子复刻 (V25b) 确认非种子中奖, 才宣布升级。

## 四、产物 (均不入库)

`model_train|test/2026q3/`、`model_pred/2026q3/heads_{a,c}/*.fea + score_ens_w2.fea`、
`model_pic/`、`logs/`。训练/推演日志在版本目录内。
