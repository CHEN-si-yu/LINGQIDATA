# V20 — 冠军系统独立可训练单元 (双族 16 折, 从零训练)

**V20 不是对旧版本的封装调用, 而是一个完整的可训练模型版本**: 用版本四文件惯例
(model.py / run.py / train.sh / analysis.py) 从零训练最终冠军的全部组成模型,
并对自训 checkpoint 推演、组装、回测、出榜。配方与冠军评估 (Model/FINAL_REPORT.md)
逐项一致, 训练数据/划分/损失/风格与 V11、V13 完全相同。

## 一、V20 在训练什么 (双族 16 折)

冠军集成需要两族模型, 故 V20 = 16 个全局折:

| 全局 fold | 族 | 对应原型 | 顶部分支训练目标 | 族内 k |
|---|---|---|---|---|
| 1..8 | **a** | V11 多目标 | **1d 主** + 5d 辅 | k1-4 保守(IC 主导) / k5-8 激进(顶部主导) |
| 9..16 | **c** | V13 顶部3d | **3d 主** + 1d 辅 | 同上 |

- 每族 k1..8: 4 折划分 × 2 风格, 各自独立种子 (`seed + k*1000`), 划分口径与 V11/V13 相同
  (TEST 20250901~20260901 / TRAIN_END 20250809 / VALID 120 天 / PURGE 5)。
- 结构 (V9 以来不变): lin_1d/lin_3d/lin_5d 线性排序头 (ridge 热启动) + 独立小 MLP 顶部分支;
  848 因子秩高斯化; 损失 = -(IC+RankIC) + 软Top(主/辅) + ListNet + R-Drop + 时间衰减(hl=600d)。
- 冠军打分 (analysis.py 组装):
  ```
  ens_w2 = z(族a mixA) + 2·z(族c r1+top)
    mixA     = Σ_{a族 k=1..8} z( 1.0·r1 + 0.25·r3 + 2.0·top )
    r1+top   = Σ_{c族 k=1..8} z( 1.0·r1 + 1.0·top )
  ```

## 二、四个基本文件的使命 (不要混用)

| 文件 | 使命 | 用法 |
|---|---|---|
| `model.py` | 架构/损失/数据划分/训练函数 + 装配工具 (唯一权威定义) | 被其余文件 import; `python3 model.py` = 健康检查 |
| `run.py` | **训练入口 (每折一进程)**: 全局 fold 1..16 → (族,风格,划分) 自动映射 | `python run.py 1` … `python run.py 16` |
| `train.sh` | **训练调度**: 16 折分 **4 批 × 每批 4 折并发** (GPU 铁律) | `bash train.sh` / `bash train.sh --clean` |
| `analysis.py` | **推演+回测+最终结果**: 对自训 16 checkpoint 全窗口推演各头 → 组装 ens_w2 → 排行榜/策略指令/冠军回测 | `python3 analysis.py [--backtest]` |

## 三、标准流程

```bash
cd /autodl-fs/data/lingqiData/Model/V20

# 1) 训练 (16 折, 4 批 × 4 并发, 单 GPU; 每折约 1.5~2.5h → 全程 6~10h+)
bash train.sh                 # 已有训练产物会拒绝; 重训: bash train.sh --clean

# 2) 全窗口推演 + 出榜 (首次约 245 交易日 × 16 模型, GPU 30~40 分钟; 之后增量数分钟)
python3 analysis.py           # 自动推演 → 最新因子日 Top10 + Top1 策略指令
python3 analysis.py --backtest   # 冠军协议回测 (hold5+stop8%+收盘卖, 含成本, 附分半/IC)
python3 analysis.py --days 3  # 最近 3 个因子日排行榜
```

产物: `model_train/2026q3/fold{1..16}` (checkpoint) → `analysis.py` 产出
`model_pred/2026q3/heads_{a,c}/{r1,r3,r5,top}_f{1..8}.fea` 与 `score_ens_w2.fea`。

## 四、关键纪律 (与 TRAINING_PLAYBOOK 一致)

- 16 折 **分 4 批, 每批最多 4 折并发** (train.sh 已内置); 勿 8 折并发。
- 重训前必须清理旧产物 (`bash train.sh --clean`), 否则 checkpoint 混淆 (train.sh 默认拒绝)。
- 训练 wrapper 不带 timeout (SIGTERM 传染 DataLoader workers)。
- 换 checkpoint/重训后 `analysis.py --update` (重推最近 10 日自愈)。
- 结构已收敛 (FINAL_REPORT): 16 折配方是冠军全局最优, 勿再叠加第三组件/种子/平滑。

## 五、与旧版本的关系

- V11/V13: 本 V20 配方的训练源 (V20 复刻其 model.py 训练口径并统一为双族 16 折);
  V20 训练后即自给自足, 不再读 V11/V13 的打分产物。
- 冠军回测引擎读 `Model/V11/strat_backtest.py` (全项目共用评估代码, 收盘卖口径), 仅此一项外部依赖。
- ridge_init.csv (线性头热启动) 已复制进 V20。
