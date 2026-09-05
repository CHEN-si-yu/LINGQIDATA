# V23 — 移动止损轨迹标签耦合模型单元 (双族 16 折, 从零训练)

**V23 = 耦合迭代第 3 号单元** (2026-09-05): 模型×策略的**最紧耦合** —
把 h20tr20_t2 协议的**实际实现收益轨迹**直接作为训练主标签, 让打分直接排序
「该股按协议执行能赚多少」。

## 一、轨迹标签 (build_trail_label.py)

与 Trading/engine.py h20tr20_t2 逐日一致 (已逐笔验证 24/24):
```
label_trail20d[d,s] = 次日(d+1)开盘买入 open[E]
  peak = max(open[E], close[E]);  d'∈[E+1, E+19]:
    peak = max(peak, close[d']);  close[d'] ≤ peak×0.80 → 卖出价 open[d'+1] (触发)
  20 个交易日无触发 → 卖出价 open[E+20]
  收益 = 卖出价/买入价 - 1
不可买 (buyable_mask 外 / 买入日一字涨停 / 缺价) → NaN
```
输出: `trainingdata/label_trail20d.fea` (1859 天 × 5794 代码, 覆盖至 20260831)。

## 二、V23 模型 (16 折双族)

| 全局 fold | 族 | 线性排序头 | 顶部分支主/辅 | 风格 |
|---|---|---|---|---|
| 1..8 | a | lin_trail / lin_10d / lin_20d | trail 软Top 主 + 10d 辅 | k1-4 保守 / k5-8 激进 |
| 9..16 | c | lin_trail / lin_10d / lin_20d | trail 软Top 主 + 20d 辅 | k1-4 保守 / k5-8 激进 |

- 其余口径与 V21 一致: 848 因子秩高斯化、ridge 热启动、-(IC+RankIC)+软Top+
  ListNet+R-Drop+时间衰减(hl=600d)、4 折划分×2 风格×种子 (seed+k×1000)。
- checkpoint 选择 = trail 标签 val_rankic (两族同口径)。
- 10d/20d 点标签为辅助头, 防单一轨迹标签过拟合。

## 三、打分组装

```
ens23 = z(族a mix) + wc·z(族c mix)
族 mix = Σ_{8折} z( wt·rtrail + w10·r10 + w20·r20 + wtop·top )
```
analysis.py 内置权重搜索 (同 V21), 在 h20tr20_t2 上以真实净值排序。

## 四、结果 (2026-09-05, 负结论 — V23 不升级)

16 折训练完成 (trail val_rankic 0.01~0.14, 训练健康)。权重搜索最优组合在
h20tr20_t2 上真实净值 **+113.4%** (Sharpe 2.36, H1/H2 +67%/+28%):

| 打分 (h20tr20_t2) | 真实净值 | 结论 |
|---|---|---|
| V20_ensw2 | +352.1% | 冠军 |
| V21 (点标签 5d/10d/20d) | +208.3% (搜索最优) | 弱 |
| **V23 (trail 轨迹标签)** | **+113.4%** (搜索最优) | **最弱** |

结论: **「点标签 vs 轨迹标签」对照实验定论 — 协议收益轨迹标签 (最紧耦合)
反而最钝化 Top 选择**; 轨迹收益的截面噪声远大于点标签, 软Top 训练无法恢复
冠军头部集成的锐度。模型×策略耦合轴 (V21/V22/V23) 全线负结论,
冠军模型与打分 (V20 ens_w2) 保持不变; V23 与 V21/V22 一并保留为
「IC ≠ Top 收益」的负结论证据 (model_pic/output.md 有完整记录)。

## 五、流程与纪律

```bash
cd /autodl-fs/data/lingqiData/Model/V23
python3 build_trail_label.py     # 已生成 trainingdata/label_trail20d.fea (幂等)
bash train.sh                    # 16 折全量训练 (6/6/4 三批)
python3 analysis.py              # 推演 → 权重搜索 → ens23 出榜+回测
```

- 与 V21 的对照实验: V21 = 点标签(5d/10d/20d) 耦合, V23 = 轨迹标签耦合;
  两者在 h20tr20_t2 上真实净值对比决定「轨迹标签是否优于点标签」。
- 若 V23 显著更优 → 升级为新冠军模型; 否则记录结论, 冠军模型回退 V21/V20 择优。
- 训练 GPU 与 V21 串行 (同一时刻只跑一个单元的 16 折, 单实例纪律)。
