# 训练流程与经验手册 (V2→V18 全历程沉淀, 2026-09-04)

> 用途: 后续任何新版本的训练/评估/实盘启动, 先读本手册。配套: FINAL_REPORT.md
> (研究结论 + 全历程履历附录A), Model/V11/README_V11.md (实盘协议)。
> 最后修订: 2026-09-04 — V18 卡住 (0样本bug) 后按用户指示停止新方向探索, 结论封存;
> V18/V19 目录保留, 重启清单见 §7b 与坑 11/12。

## 0.5 核心方法论: VX = 独立模型单元 (研究跨版本 / 落地成单元) ⭐

**版本目录 `Model/V{N}` 是一个"独立模型单元"**, 必须自带从零可训练的全部能力,
对齐四个核心文件 (使命固定, 不要混用):

| 文件 | 使命 |
|---|---|
| `model.py` | 模型架构/损失/数据划分/训练函数的**唯一权威定义** (被其余文件 import) |
| `run.py` | **训练入口 (每折一进程)**: fold N → (变体/风格/划分/种子) 自动映射 |
| `train.sh` | **训练调度**: 该单元全部折数**分批启动** (铁律: 每批 ≤4 折并发) |
| `analysis.py` | **推演/回测/最终结果入口**: 对**本单元自训**的 checkpoint 全窗口推演 → 打分 → 排行榜/策略指令/回测 |

辅件: `ridge_init.csv` (热启动, 复制自上一版) + 产物目录 `model_train / model_test /
model_pred / model_pic` (均 gitignore, 不入库)。

**Agent 研究范式 (允许 & 鼓励):**
1. 研究/搜索阶段可以**自由跨不同版本的结果进行匹配** — 交叉验证、跨版本 heads 集成、
   打分网格、混合搜索 (历史案例: 冠军 ens_w2 = V11 家族 × V13 家族 heads 集成,
   超加性 75.8+65.1 → +365%; 这类发现正是"搜索"阶段的产物, 不要求每次都在单版本内完成)。
2. **一旦确定一个更优方案 (冠军组合/新结构), 必须落地为一个新的独立单元 V{N}**:
   - 复制上一版 `model.py` 骨架, 把新发现参数化/结构化为该单元的模型定义
     (例: V20 把 V11/V13 双族统一为 `fold_spec(1..16)` 的双族训练配方);
   - `run.py / train.sh / analysis.py` 对齐新单元: 从零训练全部折数 →
     对自训 checkpoint 推演出结果 → 该单元**自给自足**, 不再把被集成旧版本的
     运行时产物 (heads/score) 当作日常输入;
   - 旧版本保留为"模型动物园 / 研究历史", 只读引用。
3. **反例 (禁止)**: 只写一个"集成脚本"读取旧版本 heads 即宣称新版本 —— 那是**封装**,
   不是单元; 研究探索可以这样临时做, 但正式落版本必须重构成可训练单元
   (V20 即从"封装版"重构为独立训练单元的实例, 见 Model/V20/README_V20.md §五)。
4. 收尾: 版本结论/履历写入 `FINAL_REPORT.md`; 验证脚本通过后清理生成物再入库。

## 0. 目录与命名

- 版本目录: `Model/V{N}` (数字递增, V11/V12/...); 每版含 model.py / run.py / train.sh /
  analysis.py / ridge_init.csv / model_train / model_test / model_pred / model_pic
- 数据: `trainingdata/` (fac_all.fea 848因子, label_ret_{1,3,5,10,20}d.fea,
  buyable_mask.fea, trade_amt.fea); 因子库 `featureengineering/data/factors/` (848 .fea)
- 原始数据: `data/` (daily_adj.parquet 价格, calendar.parquet 交易日历, stock_list 名称)
- 每个新版本从上一版 copy 4 个脚本 + ridge_init.csv, 然后 `sed` 改 root_path 与标签

## 1. 实验规模纪律 (用户 2026-09-03 指定, 严格执行)

| 实验类型 | 折数 | 说明 |
|---|---|---|
| 单模型 | 4 或 8 折 | 8 折 = 2 风格 × 4 折 (V9 起惯例) |
| 2 模型集成 | 各 4 折 | 共 8 fold-run |
| 4 模型集成 | 各 2 折 | 共 8 fold-run |
| 单次实验总折数 | ≈ 8 | 一次实验约 1.5~2.5h (单 GPU) |

## 2. 内存/并发铁律 (2026-09-04 用户指定, 血泪教训)

- **运行时 cgroup 内存上限 (曾 90GB, 现 120GB), 超限会被系统 KILL — 无警告**。
  `cat /sys/fs/cgroup/memory.max` 确认当前上限; `memory.current` 看用量。
- **GPU 训练同一时刻最多 4 折并发** (train.sh 两波 × 4 的模式), 绝不 8 折并发。
- **不同时跑两个重内存任务** (如全表 feather 预处理 + 抓取/因子构建并行) — 曾因此 OOM。
- 重内存 Python 脚本: float32 化、逐日/分批处理、预分配 numpy 矩阵、避免 groupby 全表
  临时副本 (pandas groupby rank/transform 会产生数倍于源数据的临时对象)。
- 检查命令: `ps aux --sort=-%mem | head`、`cat /sys/fs/cgroup/memory.current`。
- 训练 wrapper 用 nohup 且**绝不带 timeout** (SIGTERM 会传染杀死 DataLoader workers)。

## 3. 数据划分 (所有版本一致, 不要动)

- TEST: 20250901 ~ 20260901 (243 天, 严格样本外); TRAIN_END=20250809;
  VALID_DAYS=120 (最后120天4折轮换); PURGE_DAYS=5
- 8 折时 fold5-8 复用划分 1-4, 换 seed (fold_seed = seed + fold*1000)
- 变体映射 (fold → variant/split) 写在 model.py train() 里, 改了要检查 split 公式
  (教训: V12 曾用错 split 公式 → folds 5-8 全 EXIT:1)

## 4. 模型骨架 (V9 以来不变的组件)

- 结构: lin_1d/lin_3d/lin_5d 线性排序头 (ridge_init.csv 热启动) + 独立小 MLP 顶部分支
- 特征: 848 因子逐日截面 rank → 标准化 → 缺失填 0 (train/inference 必须一致)
- 标签: 1d 收益 winsor(MAD=5) → 截面秩高斯化; 损失 = -(IC+RankIC) + 软Top收益 + ListNet
  (除以 log n) + R-Drop(0.1) + 时间衰减(hl=600d) + AdamW(warmup+cosine) + 早停 patience=10
- 打分: mixed = z(lin_1d) + z(top) (checkpoint 选择 val_rankic 用此口径)
- 风格: cons (RANK_W1=4/TOPRET=0.3, IC 主导) vs aggr (RANK_W1=0.5/TOPRET=2.0, 顶部主导)
- 集成: 每折逐日 z-score → 跨折求和 (官方外部-z: 每折先加权求和再 z)

## 5. 训练→评估流水线

1. `bash train.sh` (nohup 两波×4折; 日志 logs/foldN.log)
2. 重启训练前**必须清理** `model_train/<season>/*/version_*`、`logs/*`、
   `model_pred/<season>` (否则 analysis 会 glob 到旧 checkpoint 错位)
3. `python3 analysis.py` (GPU 推演 245 天 × 8 模型 ~20min, 落盘 all_zscore_score.fea +
   各头逐折矩阵 heads/{h}_f{f}.fea + model_pic/output.md)
4. 策略层: `python3 strat_backtest.py <score.fea> [--battery]` / mix_grid2.py 网格
5. 更新 FINAL_REPORT.md (履历附录A/结论) → git commit

## 6. 评估口径 (三维, 缺一不可)

- **IC 层**: RankIC / RankICIR / top_return (可交易池, Test 243 天)
- **策略层 (5W 账户)**: 含成本净收益 (佣金万2.5 min5 + 印花万5卖 + 过户万0.1, 整手100股,
  T+1); 冠军协议 hold5s8-收盘卖 (买入次日开盘, 持有5交易日收盘卖, -8%收盘止损)
- **稳定性层**: H1/H2 分半 (20250901-0227 / 0302-0901) 必须双正;
  bootstrap Sharpe 95%CI (P(Sharpe<0)); 去最大1-2笔敏感性; 权重/参数平台区检查
- 诚实话语: 所有 mix 权重与策略参数都在 Test 上挑选, 存在选择偏差 → 分半+平台区缓解

## 7. 已探明的死胡同 (不要再试)

| 方向 | 结果 |
|---|---|
| 因子剪枝 500 (V7) | IC 升但 Top1 崩 |
| 三风格 12 模型 (V10) | 平衡风格稀释 |
| 顶部分支扩容 256→128→64 (V12/V14) | 稀释 |
| ultra 锐化 (TOPRET 3.0, τ0.02) | 无增益 |
| 联合共享躯干双头 1d+3d (V15) | 信号互相破坏, IC 崩 |
| 新种子扩展 24 折 (V16) | 稀释 Top1 锐度 |
| 5d 专属族作第三组件 (V17) | H2 转负 |
| 打分平滑 (2/3日均, EWM) | Top1 尖锐性即 alpha |
| 移动止损/止盈 | 切掉赢家 |
| 收盘买入 (捕捉隔夜) | 隔夜对 Top1 不利 + 涨停不可成交 |
| 阈值择时 (expanding 分位) | 低分日反而好, 高分日差 |
| Top2/Top3 分散 | 信号集中在第 1 名 |
| 止损 <6% 或 >10% | 鞭打或无效 |

## 7b. 暂停/未完成方向 (2026-09-04 停止探索后封存, 重启请先读此)

| 方向 | 现状 | 重启前置条件 |
|---|---|---|
| V18 LightGBM 第三组件 | fold1/2 已训 (初版), 重构版 0样本 bug 卡在 fold3, **无有效结果** | ① 修 train_gbdt.py:150 链式索引 (见坑11); ② 删净 model_pred/v18_f*.fea 全量重跑; ③ 首跑核对 "有效行 N" (~百万级) 再放长跑 |
| V19 堆叠元特征 | valid 期 (20250501~20250808, 68天) 元特征已落盘 meta_feats/; **未训练元模型、未推 Test、未评估** | ① 先改 OOF 口径: 该窗口与各折 valid 池重叠 → 每折只预测自己的 valid 折日期 (120天池并集), 否则早停泄漏; ② 再训元模型 + 策略层三维评估 |

## 8. 已验证有效的组件 (冠军配方)

- 线性主干 + 岭热启动 (信号近似线性, 深 MLP 稀释)
- 秩高斯特征/标签 (有界损失)
- 多目标 lin_3d 头 (RANK_W3=2/0.25) — regime 稳健性关键
- 顶部分支 1d 主目标 (V11) 与 3d 主目标 (V13) **分模型族**训练
- 跨版本集成 ens_w2 = z(V11 mixA) + 2·z(V13 r1+top) — 超加性 (75.8+65.1→365)
- 策略: hold5 + stop8% + 收盘卖 (开盘买) — 各轴全局最优
- 头部混合: mixA = Σ_f z(1·zr1 + 0.25·zr3 + 2·ztop); V13 组件 = Σ_f z(zr1 + ztop)

## 9. 常见坑清单

1. feature_map.fea 是单行 `\n` 转义文件, 解析要 replace('\\n','\n')
2. feather 往返 MultiIndex 列不可靠 → 每 (head,fold) 单独存文件
3. pd.DataFrame(list_of_Series) 用 Series.name 作索引, name=None 会变 RangeIndex
4. 各日有效股票数不同 (dropna thresh), 逐折矩阵要按行独立切分
5. analysis 增量模式只重推最后 10 天; 换 checkpoint 必须删 all_zscore_score.fea
6. daily_adj 无 prev_close 列, 要 close.shift(1) 自算
7. 卖出时点: 收盘卖 > 开盘卖 (+92pp); 买入: 开盘 > 收盘
8. 每日决策以最新打分日 Top1 为准, 勿与回测最后一笔混淆 (曾写错 README 信号)
9. paper_state.json / holdings.json 是纸面交易状态, 每日流水线会更新
10. GPU 检查: nvidia-smi; 训练 epoch 日志用 `grep -oE "epoch [0-9]+] val_rankic=..."` 提取
11. **numpy 链式布尔索引赋值 `arr[m1][m2] = v` 静默丢弃写入** (高级索引返回副本) —
    必须用单掩码 `pos = np.flatnonzero(m1 & m2); arr[pos] = v`。V18 (2026-09-04) 因此
    秩高斯标签全 NaN ("有效行 0/1556516"), fold3 0 样本崩溃。排查法: 中间量打印计数
    (如 "有效行 N/总数"), 0 或异常立即停。
12. **"文件已存在即跳过" 的 resume 会掩盖重构回归** — 改训练脚本后首次运行必须删净
    model_pred 全量重跑; 切勿带旧产物 resume, 否则 bug 直到第一个缺失 fold 才爆
    (V18 教训: fold1/2 被跳过, fold3 才崩溃)。

## 10. 每日实盘流水线 (数据更新完成后)

```
run_daily_pipeline.sh:
  1) V11/V13 各自 analysis.py 增量推演 (重推最新10日)
  2) 由 heads 构建 ens_w2 = z(mixA) + 2·z(V13 r1+top)
  3) daily_runner.py (决策) + paper_trader.py (纸面交易结算/挂单)
```
注意: 数据抓取/因子重建/prepared_data 由既有每日任务负责 (用户指定, 模型侧不要越界)。
