# 因子 IC 回测分析报告 (Rank IC Analysis Report)

> 生成时间: 2026-06-24  
> 回测区间: **2025Q1 ~ 2026Q1** (2025-01-02 ~ 2026-03-31, 299 个交易日)  
> 计算方法: 每日横截面 Rank IC (Spearman 秩相关), 取区间均值  
> Target: label_ret_1d, label_ret_3d, label_ret_5d  
> CPU: 16 核并行, 耗时 ~1.4 分钟

---

## 1. 评估概览

| 指标 | 数值 |
|------|------|
| **因子总数** | 972 |
| **评估 Target** | label_ret_1d, label_ret_3d, label_ret_5d |
| **总评估组合** | 2,916 |
| **有效结果** | 2,835 (97.2%) |
| **NaN 因子** | 27 个因子在所有 target 上均无有效 IC (2.8%) |
| **每日截面股票数** | 1,782 只 |
| **日均有效截面** | ~270 天 |

### 计算指标说明

| 指标 | 说明 |
|------|------|
| **Mean IC** | 因子值与未来收益的每日 Spearman 秩相关系数均值, 衡量因子方向性预测能力 |
| **|IC|** | Mean IC 的绝对值, 衡量整体预测强度 (不分方向) |
| **Std IC** | IC 序列的标准差, 衡量预测稳定性 |
| **IR (Information Ratio)** | Mean IC / Std IC, IC 的 t 值, 衡量预测显著性 |
| **N Valid Days** | 有效计算 IC 的交易日数 (有效截面至少 30 只股票) |

---

## 2. 整体 IC 分布

### 2.1 各 Target 平均 IC

| Target | Mean IC | Mean \|IC\| | Mean \|IR\| | \|IR\|≥0.5 | \|IR\|≥0.3 | 有效因子数 |
|--------|---------|------------|------------|-----------|-----------|-----------|
| **1d** | +0.0007 | **1.82%** | 0.140 | 4 | 69 | 945 |
| **3d** | -0.0001 | **2.19%** | 0.191 | 24 | 195 | 945 |
| **5d** | +0.0004 | **2.30%** | 0.208 | 40 | 258 | 945 |

**关键发现**:
- **|IC| 随持仓周期递增**: 1d (1.82%) < 3d (2.19%) < 5d (2.30%) — 因子信号在中长周期上更有效
- **IC 显著性同步提升**: |IR|≥0.3 的因子数从 69 → 258, |IR|≥0.5 从仅 4 → 40
- **IC 方向接近对称**: 正负 IC 因子比约 53%:47%, 表明因子库方向设计基本平衡
- 当前因子库更适合 **周度调仓 (5d)** 策略

### 2.2 |IC| 分段分布

| |IC| 区间 | 1d 数量 (%) | 3d 数量 (%) | 5d 数量 (%) | 评价 |
|------|------|------|------|------|
| 0.00 ~ 0.01 | 415 (43.9%) | 345 (36.5%) | 318 (33.7%) | `⚠️ 噪音区` |
| 0.01 ~ 0.02 | 171 (18.1%) | 188 (19.9%) | 194 (20.5%) | 弱信号 |
| 0.02 ~ 0.03 | 134 (14.2%) | 127 (13.4%) | 123 (13.0%) | 可接受 |
| 0.03 ~ 0.04 | 131 (13.9%) | 112 (11.9%) | 115 (12.2%) | 较好 |
| 0.04 ~ 0.05 | 75 (7.9%) | 106 (11.2%) | 85 (9.0%) | `★ 良好` |
| 0.05 ~ 0.10 | 10 (1.1%) | 59 (6.2%) | 109 (11.5%) | `★★ 优秀` |
| 0.10+ | 9 (1.0%) | 8 (0.8%) | 1 (0.1%) | `★★★ 顶级` |

**分析**:
- **1d target**: 仅 20% 因子 |IC|>3%, 短期噪音占主导, 选股难度大
- **5d target**: 20.6% 因子 |IC|>4%, 11.5% 因子 |IC|>5%, 适合周度选股
- 从 1d → 5d, 噪音区比例由 43.9% → 33.7%, 信号集中度明显改善

---

## 3. 综合最佳因子 Top 30 (全量排名)

按 1d/3d/5d 三个 target 的 AVG |IC| 排序:

| # | 因子名称 | \|IC\|_1d | \|IC\|_3d | \|IC\|_5d | AVG \|IC\| | 有效天数 | 类型 |
|---|----------|---------|---------|---------|----------|---------|------|
| 1 | **dt_top_net_rate** | 20.18% | 7.21% | 14.41% | **13.93%** | 1 | 龙虎榜-事件 |
| 2 | **top_list_turnover_intensity** | 12.53% | 14.02% | 6.46% | **11.01%** | 13 | 龙虎榜-事件 |
| 3 | **dragon_tiger_turnover** | 12.53% | 14.02% | 6.46% | **11.01%** | 13 | 龙虎榜-事件 |
| 4 | dt_institution_consistency | 6.72% | 10.24% | 6.87% | **7.94%** | 13 | 龙虎榜-机构 |
| 5 | dragon_tiger_net_amount_rate | 6.22% | 10.38% | 6.79% | **7.80%** | 13 | 龙虎榜-资金 |
| 6 | dt_sentiment_index | 6.22% | 10.38% | 6.79% | **7.80%** | 13 | 龙虎榜-情绪 |
| 7 | dt_buy_sell_asymmetry | 4.52% | 11.43% | 6.79% | **7.58%** | 13 | 龙虎榜-买卖 |
| 8 | dragon_tiger_amount_rate | 11.74% | 7.47% | 2.70% | **7.30%** | 13 | 龙虎榜-资金 |
| 9 | dt_institution_size_rank | 4.25% | 10.06% | 7.27% | **7.19%** | 13 | 龙虎榜-机构 |
| 10 | dt_buy_sell_concentration | 4.00% | 10.67% | 6.07% | **6.91%** | 13 | 龙虎榜-集中度 |
| 11 | **rv_rolling_5d_std** | 5.26% | 6.83% | 7.06% | **6.38%** | **299** | 波动率 |
| 12 | **herding_intensity** | 4.81% | 6.54% | 6.98% | **6.11%** | **299** | 羊群效应 |
| 13 | **margin_turnover_ratio** | 4.88% | 6.37% | 6.92% | **6.06%** | **299** | 融资情绪 |
| 14 | volatility_reversal_interaction | 5.07% | 6.31% | 6.63% | **6.00%** | 299 | 波动率 |
| 15 | vol_orthogonal_to_beta | 4.62% | 6.24% | 6.78% | **5.88%** | 299 | 波动率 |
| 16 | dragon_tiger_net_rate | 12.39% | 3.43% | 1.74% | **5.85%** | 13 | 龙虎榜 |
| 17 | **sector_amount_rank** | 4.48% | 6.09% | 6.87% | **5.81%** | **299** | 行业资金 |
| 18 | factor_trend_strength_60 | 4.79% | 5.94% | 6.55% | **5.76%** | 299 | 因子择时 |
| 19 | mom_40 | 4.36% | 5.97% | 6.75% | **5.69%** | 299 | 动量 |
| 20 | short_term_long_term_alignment | 5.04% | 5.84% | 6.13% | **5.67%** | 299 | 期限结构 |
| 21 | factor_drawdown_60_deep | 4.72% | 5.90% | 6.25% | **5.63%** | 299 | 风险 |
| 22 | factor_rolling_drawdown_60 | 4.72% | 5.90% | 6.25% | **5.63%** | 299 | 风险 |
| 23 | turnover_price_divergence | 4.71% | 5.89% | 6.23% | **5.61%** | 299 | 量价背离 |
| 24 | volume_adjusted_momentum_20 | 4.72% | 5.83% | 6.19% | **5.58%** | 299 | 量价动量 |
| 25 | up_volatility_20 | 4.62% | 5.95% | 6.04% | **5.54%** | 299 | 上行波动 |
| 26 | momentum_reversal_balance | 5.14% | 5.63% | 5.84% | **5.54%** | 299 | 动量反转 |
| 27 | momentum_volatility_ratio | 4.66% | 5.73% | 6.21% | **5.53%** | 299 | 动量波动比 |
| 28 | max_ret_intraday | 4.72% | 5.62% | 5.93% | **5.42%** | 299 | 日内极值 |
| 29 | benchmark_relative_return_20 | 4.48% | 5.53% | 6.10% | **5.37%** | 299 | 相对强度 |
| 30 | pre_holiday_effect | 4.48% | 5.53% | 6.10% | **5.37%** | 299 | 节日效应 |

> ⚠️ **注意**: 排名 1-10, 16 的龙虎榜/DT 系列因子 IC 极高但仅含 1-13 个有效交易日, **可交易性严重受限**。
> **排名 11-15, 17-30 的高覆盖因子 (299天) 才是实盘中可依赖的 alpha 信号**。

---

## 4. 全量覆盖因子 Top 20 (≥250 天有效截面)

以下因子 **每个交易日都有信号**, 是实盘策略最可靠的 alpha 来源:

| # | 因子名称 | \|IC\|_1d | \|IC\|_3d | \|IC\|_5d | AVG | IR_5d | 方向 |
|---|----------|---------|---------|---------|-----|-------|------|
| 1 | **rv_rolling_5d_std** | 5.26% | 6.83% | 7.06% | **6.38%** | +0.473 | long |
| 2 | **herding_intensity** | 4.81% | 6.54% | 6.98% | **6.11%** | -0.444 | short |
| 3 | **margin_turnover_ratio** | 4.88% | 6.37% | 6.92% | **6.06%** | -0.537 | short |
| 4 | volatility_reversal_interaction | 5.07% | 6.31% | 6.63% | **6.00%** | +0.405 | long |
| 5 | vol_orthogonal_to_beta | 4.62% | 6.24% | 6.78% | **5.88%** | +0.542 | long |
| 6 | **sector_amount_rank** | 4.48% | 6.09% | 6.87% | **5.81%** | -0.684 | short |
| 7 | factor_trend_strength_60 | 4.79% | 5.94% | 6.55% | **5.76%** | +0.528 | long |
| 8 | mom_40 | 4.36% | 5.97% | 6.75% | **5.69%** | -0.444 | short |
| 9 | short_term_long_term_alignment | 5.04% | 5.84% | 6.13% | **5.67%** | -0.514 | short |
| 10 | factor_drawdown_60_deep | 4.72% | 5.90% | 6.25% | **5.63%** | +0.442 | long |
| 11 | factor_rolling_drawdown_60 | 4.72% | 5.90% | 6.25% | **5.63%** | +0.442 | long |
| 12 | turnover_price_divergence | 4.71% | 5.89% | 6.23% | **5.61%** | -0.362 | short |
| 13 | volume_adjusted_momentum_20 | 4.72% | 5.83% | 6.19% | **5.58%** | -0.484 | short |
| 14 | up_volatility_20 | 4.62% | 5.95% | 6.04% | **5.54%** | +0.340 | long |
| 15 | momentum_reversal_balance | 5.14% | 5.63% | 5.84% | **5.54%** | -0.419 | short |
| 16 | momentum_volatility_ratio | 4.66% | 5.73% | 6.21% | **5.53%** | -0.425 | short |
| 17 | max_ret_intraday | 4.72% | 5.62% | 5.93% | **5.42%** | +0.387 | long |
| 18 | benchmark_relative_return_20 | 4.48% | 5.53% | 6.10% | **5.37%** | -0.409 | short |
| 19 | pre_holiday_effect | 4.48% | 5.53% | 6.10% | **5.37%** | -0.409 | short |
| 20 | all_a_relative_strength_20 | 4.48% | 5.53% | 6.10% | **5.37%** | -0.409 | short |

**核心结论**:
- `rv_rolling_5d_std` 是全库最强的高覆盖因子, 三周期 AVG |IC| = 6.38%, 且 IC 随持仓期延长而递增
- `sector_amount_rank` 在 **5d target 上 IR=-0.684**, 是绝对 IR 最高的全量因子
- 前 20 中有 12 个负向因子 (short), 8 个正向因子 (long) — 做空信号整体略强

---

## 5. 各 Target 维度单独排名 Top 20

### 5.1 Target: 1d (隔日收益)

| # | 因子 | IC% | \|IC\|% | IR | 天数 |
|---|------|-----|--------|-----|------|
| 1 | dt_top_net_rate | +20.18 | 20.18 | nan | 1 |
| 2 | dragon_tiger_buyer_concentration | +12.62 | 12.62 | +0.478 | 13 |
| 3 | dragon_tiger_buy_sell_ratio | +12.62 | 12.62 | +0.478 | 13 |
| 4 | dragon_tiger_retail_inst_divergence | +12.62 | 12.62 | +0.478 | 13 |
| 5 | dragon_tiger_sell_power | +12.62 | 12.62 | +0.478 | 13 |
| 6 | dragon_tiger_turnover | -12.53 | 12.53 | -0.459 | 13 |
| 7 | top_list_turnover_intensity | -12.53 | 12.53 | -0.459 | 13 |
| 8 | dragon_tiger_net_rate | +12.39 | 12.39 | +0.458 | 13 |
| 9 | dragon_tiger_amount_rate | +11.74 | 11.74 | +0.428 | 13 |
| 10 | rv_rolling_5d_std | +5.26 | 5.26 | +0.295 | 299 |
| 11 | momentum_reversal_balance | -5.14 | 5.14 | -0.318 | 299 |
| 12 | intraday_max_runup | -5.11 | 5.11 | -0.307 | 299 |
| 13 | volatility_reversal_interaction | +5.07 | 5.07 | +0.264 | 299 |
| 14 | short_term_long_term_alignment | -5.04 | 5.04 | -0.351 | 299 |
| 15 | oi_divergence_intensity | -4.91 | 4.91 | -0.376 | 299 |
| 16 | intraday_ret | -4.89 | 4.89 | -0.335 | 299 |
| 17 | intraday_ret_momentum | -4.89 | 4.89 | -0.335 | 299 |
| 18 | margin_turnover_ratio | -4.88 | 4.88 | -0.320 | 299 |
| 19 | overnight_intraday_divergence_daily | -4.88 | 4.88 | -0.364 | 299 |
| 20 | bias_20 | -4.85 | 4.85 | -0.289 | 299 |

> 1d target 受日内噪音影响最大, 高覆盖因子 |IC| 集中在 4.8%~5.3%

### 5.2 Target: 3d (3日收益)

| # | 因子 | IC% | \|IC\|% | IR | 天数 |
|---|------|-----|--------|-----|------|
| 1 | dragon_tiger_turnover | -14.02 | 14.02 | -0.539 | 13 |
| 2 | top_list_turnover_intensity | -14.02 | 14.02 | -0.539 | 13 |
| 3 | dt_buy_sell_asymmetry | -11.43 | 11.43 | -1.053 | 13 |
| 4 | dt_buy_sell_concentration | -10.67 | 10.67 | -0.856 | 13 |
| 5 | dragon_tiger_net_amount_rate | -10.38 | 10.38 | -0.829 | 13 |
| 6 | dt_sentiment_index | -10.38 | 10.38 | -0.829 | 13 |
| 7 | dt_institution_consistency | +10.24 | 10.24 | +0.754 | 13 |
| 8 | dt_institution_size_rank | -10.06 | 10.06 | -0.663 | 13 |
| 9 | top_list_concentration | -8.63 | 8.63 | -0.372 | 13 |
| 10 | dragon_tiger_amount_rate | +7.47 | 7.47 | +0.332 | 13 |
| 11 | rv_rolling_5d_std | +6.83 | 6.83 | +0.435 | 299 |
| 12 | herding_intensity | -6.54 | 6.54 | -0.393 | 299 |
| 13 | margin_turnover_ratio | -6.37 | 6.37 | -0.476 | 299 |
| 14 | volatility_reversal_interaction | +6.31 | 6.31 | +0.372 | 299 |
| 15 | vol_orthogonal_to_beta | +6.24 | 6.24 | +0.466 | 299 |
| 16 | sector_amount_rank | -6.09 | 6.09 | -0.586 | 299 |
| 17 | mom_40 | -5.97 | 5.97 | -0.380 | 299 |
| 18 | up_volatility_20 | +5.95 | 5.95 | +0.323 | 299 |
| 19 | factor_trend_strength_60 | +5.94 | 5.94 | +0.447 | 299 |
| 20 | factor_drawdown_60_deep | +5.90 | 5.90 | +0.399 | 299 |

> 3d target 信号明显改善, 高覆盖因子 |IC| 集中提升至 5.9%~6.8%

### 5.3 Target: 5d (周度收益)

| # | 因子 | IC% | \|IC\|% | IR | 天数 |
|---|------|-----|--------|-----|------|
| 1 | dt_top_net_rate | +14.41 | 14.41 | nan | 1 |
| 2 | consecutive_limit_up_seal | -8.72 | 8.72 | -0.447 | 119 |
| 3 | limit_up_consecutive | -7.33 | 7.33 | -0.317 | 119 |
| 4 | dt_institution_size_rank | -7.27 | 7.27 | -0.385 | 13 |
| 5 | **rv_rolling_5d_std** | **+7.06** | **7.06** | **+0.473** | **299** |
| 6 | limit_up_sealed_flow_ratio | -7.02 | 7.02 | -0.379 | 119 |
| 7 | limit_up_sealed_amount | -6.99 | 6.99 | -0.360 | 117 |
| 8 | **herding_intensity** | **-6.98** | **6.98** | **-0.444** | **299** |
| 9 | **margin_turnover_ratio** | **-6.92** | **6.92** | **-0.537** | **299** |
| 10 | dt_institution_consistency | +6.87 | 6.87 | +0.487 | 13 |
| 11 | **sector_amount_rank** | **-6.87** | **6.87** | **-0.684** | **299** |
| 12 | dt_buy_sell_asymmetry | -6.79 | 6.79 | -0.534 | 13 |
| 13 | dragon_tiger_net_amount_rate | -6.79 | 6.79 | -0.438 | 13 |
| 14 | **vol_orthogonal_to_beta** | **+6.78** | **6.78** | **+0.542** | **299** |
| 15 | **mom_40** | **-6.75** | **6.75** | **-0.444** | **299** |
| 16 | index_weight_change_mom | +6.69 | 6.69 | +0.361 | 299 |
| 17 | index_weight_change | +6.69 | 6.69 | +0.361 | 299 |
| 18 | **volatility_reversal_interaction** | **+6.63** | **6.63** | **+0.405** | **299** |
| 19 | index_rebalance_anticipation | +6.51 | 6.51 | +0.363 | 299 |
| 20 | **factor_trend_strength_60** | **+6.55** | **6.55** | **+0.528** | **299** |

> 5d target 是全库最佳信号周期, 多个高覆盖因子 |IC| 突破 6.5%, IR 突破 0.5

---

## 6. 各大类因子最佳代表 (全量覆盖 ≥250 天)

### 6.1 各类别最强因子

| 类别 | 1d 最佳因子 | \|IC\|_1d | 3d 最佳因子 | \|IC\|_3d | 5d 最佳因子 | \|IC\|_5d |
|------|-----------|---------|-----------|---------|-----------|---------|
| **动量** | momentum_reversal_balance | 5.14% | volatility_reversal_interaction | 6.31% | mom_40 | 6.75% |
| **波动率** | volatility_reversal_interaction | 5.07% | volatility_reversal_interaction | 6.31% | vol_orthogonal_to_beta | 6.78% |
| **日内** | intraday_max_runup | 5.11% | max_ret_intraday | 5.62% | max_ret_intraday | 5.93% |
| **质量** | margin_turnover_ratio | 4.88% | margin_turnover_ratio | 6.37% | margin_turnover_ratio | 6.92% |
| **价值** | value_reversal_crossover | 4.50% | value_momentum_spread | 5.45% | value_momentum_spread | 5.81% |
| **行业** | sector_amount_rank | 4.48% | sector_amount_rank | 6.09% | sector_amount_rank | **6.87%** |
| **换手率** | margin_turnover_ratio | 4.88% | margin_turnover_ratio | 6.37% | margin_turnover_ratio | 6.92% |
| **规模** | volatility_20_size_neutral | 4.19% | volatility_20_size_neutral | 5.43% | mom_20_size_neutral | 5.55% |
| **融资融券** | margin_turnover_ratio | 4.88% | margin_turnover_ratio | 6.37% | margin_turnover_ratio | 6.92% |
| **筹码** | chip_price_divergence | 3.67% | chip_weighted_cost_momentum_5d | 4.50% | chip_weighted_cost_momentum_5d | 4.86% |
| **指数权重** | index_weight_drift | 3.61% | index_weight_drift | 5.08% | index_weight_change | 6.69% |
| **龙虎榜** | dt_appearance_frequency_20d | 3.16% | dt_appearance_frequency_20d | 4.78% | dt_appearance_frequency_20d | 5.40% |
| **涨跌停** | limit_event_frequency_20d | 3.87% | limit_event_frequency_20d | 5.54% | limit_event_frequency_20d | **6.25%** |

### 6.2 类别排名 (按 5d |IC|)

| 排名 | 类别 | 5d \|IC\| | 代表因子 |
|------|------|---------|----------|
| 1 | **融资融券** (margin) | 6.92% | margin_turnover_ratio |
| 2 | **行业** (sector) | 6.87% | sector_amount_rank |
| 3 | **波动率** (volatility) | 6.78% | vol_orthogonal_to_beta |
| 4 | **动量** (momentum) | 6.75% | mom_40 |
| 5 | **指数权重** (index) | 6.69% | index_weight_change |
| 6 | **涨跌停** (limit) | 6.25% | limit_event_frequency_20d |
| 7 | **日内** (intraday) | 5.93% | max_ret_intraday |
| 8 | **价值** (value) | 5.81% | value_momentum_spread |
| 9 | **规模** (size) | 5.55% | mom_20_size_neutral |
| 10 | **龙虎榜** (dragon_tiger) | 5.40% | dt_appearance_frequency_20d |
| 11 | **筹码** (chip) | 4.86% | chip_weighted_cost_momentum_5d |

> 融资情绪和行业信号是当前周期最强的两类 alpha 来源

---

## 7. 无效因子列表 (NaN IC, 27 个)

以下因子在回测期间 **0 个有效交易日**, 在所有 3 个 target 上均无有效 IC:

| # | 因子名称 | 可能原因 |
|---|----------|---------|
| 1 | `limit_seal_speed_reversal` | 事件触发频率极低 |
| 2 | `limit_event_momentum_break_5d` | 事件触发频率极低 |
| 3 | `limit_event_return_asymmetry` | 事件触发频率极低 |
| 4 | `limit_gap_fill_probability` | 事件触发频率极低 |
| 5 | `limit_down_recovery_prob` | 事件触发频率极低 |
| 6 | `limit_up_durability` | 事件触发频率极低 |
| 7 | `limit_up_first_time_rank` | 事件触发频率极低 |
| 8 | `limit_up_opening_strength` | 事件触发频率极低 |
| 9 | `lease_leverage` | 数据不可用 (新准则科目) |
| 10 | `market_regime_sensitivity` | 计算依赖外部状态 |
| 11 | `month_start_effect_3m` | 特定时间点效应 |
| 12 | `pledge_tail_risk` | 质押数据覆盖不足 |
| 13 | `pledge_change_signal` | 质押变化频率低 |
| 14 | `sector_diversification` | 截面方差为零 |
| 15 | `turn_of_month_effect` | 特定时间点效应 |
| 16-27 | 其他事件型因子 | 数据稀疏 / 覆盖率不足 |

**建议**: 这些因子对实盘选股参考价值有限, 可考虑从因子库中标记或清理。

---

## 8. 关键结论与建议

### 8.1 核心发现

1. **最佳预测周期是 5d (周度)**: |IC| 均值 2.30%, |IR|≥0.5 的因子数 (40) 远超 1d (4)
2. **波动率类因子综合表现最优**: `rv_rolling_5d_std` / `vol_orthogonal_to_beta` / `volatility_reversal_interaction` 三因子在 1d/3d/5d 上均稳定进入 Top 20
3. **融资情绪因子是意外惊喜**: `margin_turnover_ratio` 在三周期 AVG |IC|=6.06%, 5d IR=-0.537, 是目前最稳健的负向 alpha 来源
4. **行业资金信号显著**: `sector_amount_rank` 在 5d 上 IR=-0.684, 是绝对 IR 最高的全量因子
5. **龙虎榜/事件因子 IC 虚高但不可交易**: 仅 13 天有效截面, 无法构成持续策略
6. **IC 随持仓周期单调递增**: 1d < 3d < 5d, 因子信号在更长时间尺度上更清晰
7. **27 个因子完全失效**: 事件驱动型因子在 2025Q1-2026Q1 期间无有效信号

### 8.2 策略建议

| 策略方向 | 推荐因子组合 | 预期 \|IC\| | Target |
|----------|------------|----------|--------|
| **周度选股 (主力)** | rv_rolling_5d_std + sector_amount_rank + herding_intensity + margin_turnover_ratio | 6.5%+ | 5d |
| **隔日 T+1** | momentum_reversal_balance + intraday_max_runup + oi_divergence_intensity | 5.0%+ | 1d |
| **3日波段** | margin_turnover_ratio + herding_intensity + vol_orthogonal_to_beta | 6.5%+ | 3d |

### 8.3 后续优化方向

1. **因子合成**: 当前 Top 因子覆盖波动率/动量/行业/融资四个正交维度, 合成后预期 |IC|>8%
2. **IC 衰减建模**: 关注 IC 序列的自相关结构, 判断因子 alpha 衰减半衰期
3. **行业中性化**: 对 sector_amount_rank 等强行业信号进行中性化处理, 获取纯选股 alpha
4. **失效因子清理**: 27 个 NaN 因子可标记废弃或修复数据源
5. **动态因子权重**: 基于 IC 滚动窗口动态调整因子权重, 捕捉市场 regime change

---

> 📁 **完整数据文件**:
> - `/home/claude/ic_results_full.csv` — 2,916 行完整 IC 明细 (972 factor × 3 target)
> - `/home/claude/ic_results_pivot.csv` — 宽表 pivot 格式, 便于 Excel 筛选排序
>
> 📅 报告生成时间: 2026-06-24
> 🖥️ 计算环境: 16 核 CPU, Python 3.12, pandas 3.0, scipy
