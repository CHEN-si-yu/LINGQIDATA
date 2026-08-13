# V6.1 改动说明 (2026-08-13, 最终版: 学 V5.2 多周期联合训练 + 1d 口径回测)

## 背景诊断结论 (V6.0 2026Q3 回测 -4.25% 的根因)
1. 模型对 1d 收益的真实信号极弱: 回测窗 rank IC +0.034, 验证池 OOS +0.018。
2. 旧 val_wei = top10均值收益*100 + 0.7*IC, 约 97% 权重是市场 beta → checkpoint 选择按牛市 beta 选模型。
3. 验证池(240天)与回测窗重叠, 选择流程偷看了回测窗(评估不诚实)。
4. 4-fold 同种子同训练集 = 同一条训练轨迹的快照, 伪集成。
5. Top1 单票是彩票: 分数极端尾部 39% 在追涨停股, 2026-07-01~15 小盘崩盘(-0.44%/天)期间全部 16 个 checkpoint 的 Top1 无一为正。
6. epoch 与 7 月表现显著负相关 (r=-0.59, p=0.017): 模型 epoch 13 后过拟合 → IC 早停正好选更早的 epoch。

## V6.1 设计 (学 V5.2)
model.py:
- 多头联合训练: 共享主干 512→256→64 → 3 个头 (1d/5d/10d), 各头独立 WPCC 损失,
  权重 softmax 参数化但保持初始值 {1d:1.0, 5d:0.5, 10d:0.3} (与 V5.2 相同)。
- 评估/checkpoint 选择/推理/回测统一 1d 口径: 只用 pred_1d 对 label_ret_1d 算 IC/top收益。
- 标签平滑噪声 label_smooth_noise=0.05 (V1.7, IC +2.6%)。
- checkpoint/早停按 val_icmean 选择, 文件名 {epoch}-{val_icmean:.4f} (兼容旧 val_wei)。
- 每个 Fold 独立种子 (args.seed + fold*1000) → 真正集成。
- mixup_alpha=0 (截面 mixup 破坏排序结构)。
- data_end_date: 设 20260630 则 20260701+ 完全不参与训练/验证/选择, 回测才是真实样本外。
  (代价: 训练数据止于 20250618。评估用设截止; 实盘定期重训可不设。)

analysis.py:
- 回测 P&L 永远用正确的 1d 收益 (open[t+2]/open[t+1]-1, 来自 daily_adj 实际价格), 与训练 target 解耦。
- LABEL_HORIZON_DAYS>1 时: 组合日度 P&L 按重叠持仓的 1d 实际收益计算 (build_daily_pnl_series,
  已与独立暴力持仓模拟零误差交叉验证); 逐笔 H 日收益直接相加会重复计算重叠区间。
- 执行可行性过滤(默认开): EXCLUDE_ST / BUY_GAP_LIMIT=9.5%(一字板) / MIN_AMOUNT=5e7。
- 分散组合对比表: Top5/10/20 + 前1%/5%/10% 分数带等权 (Top1 是彩票, 判断信号质量以该表为准)。

## 用法
训练: bash train.sh (或 python run.py <fold> --season 2026q3)
推演+回测: python analysis.py  → 输出 model_pic/output.md
