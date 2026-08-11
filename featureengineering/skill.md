# 因子开发规范与指南

## 一、核心原则

### 1.1 日频约束（必须）

因子**必须**使用日频原始字段，且数据源仅限**白名单 8 源**（2026-08-08 debug 实证定型，依据见 §1.3）。
仅依赖静态属性（如 `stock_list.parquet` 的公司属性）而不使用任何日频数据的因子，一律不合格。

| 许可数据源（白名单） | 频率 | 说明 |
|---|---|---|
| `daily.parquet` | 日频 | 未复权日线 OHLCV（**唯一日线价格源**，pre_close / pct_chg 口径） |
| `finance.parquet` | 日频 | 日频财务/市场数据（pb、pe_ttm、turnover_rate、total_mv 等；**pe_ttm_percentile 字段禁止使用**） |
| `cyq_perf.parquet` | 日频 | 筹码分布（**his_low / his_high 字段禁止使用**，上游回溯改写） |
| `main_fund_flow.parquet` | 日频 | 主力资金流向（大/中/小单分类） |
| `margin_detail.parquet` | 日频 | 融资融券明细（T+1 到达，数据层已统一 shift(1) 对齐；只追加不修订） |
| `cyq_chips/` | 日频 | 筹码分布面板（逐股目录，Class 2） |
| `history_1min/` | 日频 | 1 分钟线（逐股目录，Class 3） |
| `indicator_1min/` | 日频 | 分钟指标（逐股目录，Class 4） |
| `calendar.parquet` | 辅助 | 交易日历，只追加不修改；可用于交易日对齐/时间编码，**不承载因子数值** |

| 禁止依赖 | 原因（debug 实证，详见 §1.3） |
|---|---|
| `daily_adj.parquet` | 前复权，除权事件回溯改写历史 OHLC（仅 target 标签可用） |
| `finance.parquet.pe_ttm_percentile` | T 日 0 占位 → T+1 回填真实分位并全市场重算（字段级禁用） |
| `cyq_perf.parquet.his_low / his_high` | 上游回溯改写、复权口径与未复权 close 混比（字段级禁用） |
| `ths_daily.parquet` | T 日数据在 T+1 被修订（0805 vs 0806 对比 355 行改写先例） |
| `ths_constituent_stocks.parquet` | 当前快照、无成员历史 → 无法 point-in-time 归属 |
| `ths_sector_categories.parquet` | 指数元数据，随 ths 系不可用 |
| `stock_list.parquet` | 静态快照（含上游重复行缺陷），构建时自动禁用依赖因子 |
| `short_selling.parquet` | 非白名单（无因子依赖，历史已清理） |
| 仅靠静态属性 broadcast 到每日 | 产出常量因子，无截面信息 |

### 1.2 时间编码型因子（例外）

日历效应类因子（如 `month_start_effect`、`quarter_end_proximity`、`day_of_week_effect`）虽然截面方差极低，但使用日频 calendar 数据，属于**时间编码**，合格保留。

### 1.3 point-in-time（PIT）契约 —— 白名单依据（2026-08-08 debug 实证定型）

因子 T 日值必须满足两点：**只依赖 ≤ T 的数据**（无未来函数）；且上游对 ≤ T 的历史数据
**只追加、不修改**（未来数据更新不回溯改写历史）。

实证依据（debug 20260806 / 20260807 两次 10 日全量切片对比，9 个重叠交易日逐格 diff）：

- **白名单 8 源数值零变动** → 满足"只追加不修改"：daily / finance / cyq_perf /
  main_fund_flow / margin_detail / cyq_chips / history_1min / indicator_1min；
- **daily_adj**：前复权，除权事件回溯改写历史 OHLC（0806 vs 0807 对比 57 行 / 274 格，
  14 只股票 08-03~08-06 被改写）→ 禁用，仅 target 可用；
- **finance.pe_ttm_percentile**：T 日 0 占位、T+1 回填真实分位并全市场重算
  （0806 vs 0807 对比 3916 格；0805 vs 0806 对比 2073 格）→ 字段级禁用；
- **ths_daily**：T 日数据在 T+1 被修订（0805 vs 0806 对比 355 行 / 393 格全列改写）→ 禁用；
- **stock_list / ths_constituent_stocks / ths_sector_categories**：静态快照或成分无历史，
  用当前数据回填历史 = 前视偏差 → 禁用（构建时由 factor_loader 自动拦截）。

收敛结果：全部因子（运行时注册表 − 5 标签 = 686，与 debug 切片 .fea 数量一致）的
上游依赖已收敛于白名单 8 源（20 种文件组合，无第 9 类来源），两次对比中 686 个 .fea
在 9 个重叠交易日**零变动**。**新因子引入白名单外的数据源 = 违约**，构建评审不通过。

---

## 二、因子质量标准

### 2.1 NaN 控制

- **NaN 占比 > 20% → 剔除**
- 需要解释的是，NAN的计算范围是所有股票在2022年之后的所有数据
- 典型不合格场景（NaN > 20%）：
  - 融资融券因子覆盖约 85% 股票 → NaN ≈ 15%，✅ 合格（< 20%）
  - PE 类因子覆盖约 83% 股票（负利润无 PE）→ NaN ≈ 17%，✅ 合格（< 20%）
  - 分红类因子覆盖约 78% 股票 → NaN ≈ 22%，❌ 不合格（> 20%）
- 解决方法：要么确保数据全覆盖，要么不要开发

### 2.2 截面方差

- 因子在每期截面上必须有足够区分度
- **全市场所有股票值相同（如全 0.5）→ 剔除**
- 静态 0/1 dummy 不做截面排名（排名后所有人都得到同一个百分位值）

### 2.3 日频变化

- 因子值必须随交易日变化（连续型因子）
- **0 次变化（1589 个交易日同一个值）→ 剔除**
- 时间编码型因子可例外
- 日变化率 < 1% 的需要人工审查

---

## 三、性能约束

### 3.1 严禁模式

以下模式在 5000+ 股票 × 1500+ 交易日的规模上不可行：

```python
# ❌ 禁止：rolling().apply(lambda, raw=False) 对宽 DataFrame
#    每个窗口将 DataFrame 传给 Python lambda → O(N²) 级
vol_panel.rolling(20).apply(lambda x: ..., raw=False)

# ❌ 禁止：日期 × 日期 双重循环
for i, (d, m) in enumerate(zip(days, months)):
    for h in holiday_dates:
        ...

# ❌ 禁止：逐股逐窗口 Python for-loop
for i in range(59, len(values)):
    window = values[i-59:i+1]
    result[i] = heavy_computation(window)
```

### 3.2 推荐模式

```python
# ✅ pandas 内置向量化操作
df.rolling(20).mean()        # C 级别优化
df.rolling(20).corr(other)   # 内置 corr

# ✅ groupby transform（单股操作，天然并行）
series.groupby(level="Code").transform(lambda s: s.pct_change(20))

# ✅ 截面向量化（numpy 原生）
np.where(condition, a, b)
```

### 3.3 耗时目标

- 单个因子 rebuild **< 60 秒**
- 超过 100 秒的因子原则上不保留
- 慢因子提交顺序优化（`_PRIORITY_FACTORS` 在 `builder.py` 中配置）

---

## 四、pandas 3.0 兼容性

代码运行在 pandas 3.0.3 环境，以下模式已废弃/行为变更：

### 4.1 必须使用封装的工具函数

```python
# ✅ stack_date_code(df) 替代 df.stack().reorder_levels(["Date", "Code"]).sort_index()
#    pandas 3.0 下 stack() 后 level names 丢失
from ..utils import stack_date_code
result = stack_date_code(combined)

# ✅ safe_divide(a, b) 替代 a / b.replace(0, np.nan)
#    处理 numpy array 的只读属性问题
from ..utils import safe_divide
share = safe_divide(amount, denominator)  # 不要传 .values

# ✅ np.asarray(x) 替代 x.values
#    DatetimeIndex.month/day 在 pandas 3.0 返回 numpy array，无 .values 属性
months = np.asarray(dt_panel.month)
```

### 4.2 DataFrame groupby 陷阱

```python
# ❌ pandas 3.0: groupby grouper 必须是一维的
#    (up.iloc[::-1] == False).cumsum() 返回 DataFrame
consec = df.groupby(df_condition).cumsum()

# ✅ 使用逐列 apply
consec = up.apply(lambda col: col.astype(int).iloc[::-1]
    .groupby((~col).iloc[::-1].cumsum()).cumsum().iloc[::-1].where(col, 0))
```

### 4.3 numpy array 只读问题

```python
# ❌ copy=False 可能导致只读 array 赋值失败
right = right.astype(float, copy=False)
right[right == 0] = np.nan  # ValueError: assignment destination is read-only

# ✅ 确保可写
right = right.astype(float, copy=True)
right[right == 0] = np.nan
```

---

## 五、因子注册与命名

### 5.1 注册格式

```python
@register_factor(
    name="my_factor_name",          # 唯一，snake_case
    description="因子描述（中文）",
    category="sector",              # price/timeseries/valuation/fund_flow/...
    thesis=("投资逻辑..."),          # 三行以内
    dependencies=("daily.parquet",),  # 必须声明所有数据源；daily_adj.parquet 禁止用于因子
)
def factor_my_factor_name(context: FactorContext):
    ...
```

### 5.2 禁止的做法

- 禁止在 `@register_factor` 中声明虚假的 dependencies（写了但实际不用）
- 禁止引用数据源不存在的字段
- 因子名必须与实际函数名无歧义（可以在 decorator 中用不同的 `name=`）

---

## 六、开发前检查清单

在新建因子前，确认以下各项：

- [ ] 依赖的数据源 ∈ 白名单 8 源：daily / finance / cyq_perf / main_fund_flow / margin_detail / cyq_chips / history_1min / indicator_1min（calendar 仅辅助，不承载因子数值）
- [ ] 不依赖 daily_adj.parquet（target 例外）—— 所有因子统一使用 daily.parquet
- [ ] 不使用 finance.parquet 中的 pe_ttm_percentile 字段（T+1 回填改写）
- [ ] 不使用 cyq_perf.parquet 中的 his_low / his_high 字段（上游回溯改写）
- [ ] 不依赖 ths_daily / ths_constituent_stocks / ths_sector_categories / stock_list / short_selling
- [ ] 不使用 `rolling().apply(lambda, raw=False)` 对宽 DataFrame
- [ ] 不使用日期 × 日期双重循环
- [ ] 使用 `stack_date_code()`、`safe_divide()`、`np.asarray()` 等 pandas 3.0 兼容工具
- [ ] 预期 NaN 覆盖率 > 80%（即缺失率 < 20%）
- [ ] 预期单因子 rebuild 耗时 < 60s

---

## 七、数据源状态

| 数据源 | 状态 | 说明 |
|---|---|---|
| `daily_adj.parquet` | ❌ 因子禁用 | **仅 target 可用**，前复权会回溯改写历史 |
| `daily.parquet` | ✅ 白名单 | 核心日线数据源（未复权 OHLCV，pre_close / pct_chg 口径） |
| `finance.parquet` | ✅ 白名单 | pb/pe_ttm/turnover_rate/total_mv 等；**pe_ttm_percentile 字段禁用**（T+1 回填） |
| `cyq_perf.parquet` | ✅ 白名单 | 筹码分布；**his_low / his_high 字段禁用**（上游回溯改写） |
| `main_fund_flow.parquet` | ✅ 白名单 | 大单/中单/小单分类 |
| `margin_detail.parquet` | ✅ 白名单 | T+1 到达，数据层已 shift(1) 对齐；仅约 85% 股票覆盖，做因子需接受高 NaN |
| `cyq_chips/` | ✅ 白名单 | 逐股筹码面板（Class 2） |
| `history_1min/` | ✅ 白名单 | 逐股 1 分钟线（Class 3） |
| `indicator_1min/` | ✅ 白名单 | 逐股分钟指标（Class 4） |
| `calendar.parquet` | ✅ 辅助 | 只追加不修改；交易日对齐/时间编码用，不承载因子数值 |
| `ths_daily.parquet` | ❌ 禁用 | 纯指数日线；T 日数据 T+1 被修订（0805 vs 0806 对比 355 行改写先例） |
| `ths_constituent_stocks.parquet` | ❌ 禁用 | 当前快照、无成员历史，无法 point-in-time 归属 |
| `ths_sector_categories.parquet` | ❌ 禁用 | 指数元数据，随 ths 系不可用 |
| `stock_list.parquet` | ❌ 禁用 | 静态快照，构建时自动禁用依赖因子（factor_loader） |
| `short_selling.parquet` | ❌ 非白名单 | 无因子依赖，历史已清理 |

---

## 八、2026-07-26 清理记录（累计）

本次两轮清理共删除 **187** 个因子，分四类：

### 8.1 死因子 — 100% NaN 或 THS 依赖（73个）

全部为 THS 板块依赖因子（`sector_*`，依赖 `ths_daily.parquet` / `ths_constituent_stocks.parquet`）+ `margin_exchange_flow_diff`。

### 8.2 状态耦合因子 — NaN > 20%（32个）

由 `regime_conditional.py` 动态生成的高低波动率/牛熊市分片因子，NaN 范围 29%-88%。已删除整个 `regime_conditional.py` 文件。

| 分类 | 因子 |
|---|---|
| Vol regime (highvol/lowvol) | `rv_5min_*`, `gk_vol_*`, `parkinson_vol_*`, `volatility_20_*`, `amihud_intraday_*`, `turnover_20_*`, `mom_20_*`, `herding_intensity_*`, `max_ret_20_*`, `idiosyncratic_vol_60_*` |
| Trend regime (bull/bear) | `rv_5min_*`, `gk_vol_*`, `parkinson_vol_*`, `volatility_20_*`, `amihud_intraday_*`, `turnover_20_*` |

### 8.3 零截面方差因子（非时间编码）（4个）

所有股票在每期截面上值相同，且不属于日历效应：

| 因子 | 来源文件 |
|---|---|
| `boll_bandwalk_lower` | `indicator_minute.py` |
| `boll_bandwalk_upper` | `indicator_minute.py` |
| `boll_touch_ratio` | `indicator_minute.py` |
| `price_vs_ma5_deviation` | `indicator_minute.py` |

### 8.4 静态身份 / 零日频变动（6个）

| 因子 | 来源文件 | 原因 |
|---|---|---|
| `delist_young_stock` | `stock_identity_extended.py` | 静态 0/1 哑变量 |
| `new_listing_momentum` | `stock_identity.py` | 排名后退化为哑变量 |
| `factor_anti_crowding` | `coupling_deep.py` | 计算异常，无变动 |
| `factor_outlier_frequency` | `coupling_deep.py` | 计算异常，无变动 |
| `liquidity_resilience` | `intraday.py` | NaN 25.56% |

### 8.5 时间编码因子（7个）

日历效应类因子，虽截面方差为零但使用日频 calendar 数据。原属 skill.md §1.2 豁免范围，在第二轮清理中一并删除。

| 因子 | 来源文件 |
|---|---|
| `december_effect` | `calendar_extended.py` |
| `half_year_effect` | `calendar_extended.py` |
| `month_mid_effect` | `calendar_extended.py` |
| `quarterly_seasonality` | `calendar_extended.py` |
| `week_of_month_first` | `calendar_extended.py` |
| `week_of_month_last` | `calendar_extended.py` |
| `week_of_month_position` | `calendar_extended.py` |

### 8.6 历史已删除（之前清理）

| 类别 | 数量 | 原因 |
|---|---|---|
| 融资融券因子 (`margin_*`, `short_*`) | ~50 | 部分 NaN > 20% |
| PE/分红因子 NaN>20% | 1 | `dv_yield_growth_proxy` |
| 算法过慢因子 | 3 | `sector_vol_trend_20d`、`holiday_gap_effect`、`hurst_exponent_60` |
| `cs_spread_20` | 1 | 全零方差 |
| `area_industry_bp_rank` | 1 | 51.4% NaN |
| `margin_exchange_flow_diff` | 1 | 100% NaN |

### 8.7 修改文件清单

| 操作 | 文件 |
|---|---|
| 删除 | `factors/regime_conditional.py`（整文件） |
| 删减 | `factors/calendar_extended.py`（7 个注册块） |
| 删减 | `factors/intraday.py`（`liquidity_resilience`） |
| 删减 | `factors/coupling_deep.py`（`factor_outlier_frequency`、`factor_anti_crowding`） |
| 删减 | `factors/stock_identity.py`（`new_listing_momentum`） |
| 删减 | `factors/stock_identity_extended.py`（`delist_young_stock`） |
| 删减 | `factors/indicator_minute.py`（4 个注册块） |
| 删除 | `data/factors/*.fea`（48 个） |
| 删除 | `data/manifests/*.json` + `*.done`（48+ 个） |

### 8.8 注册表状态

- 本轮清理前：695 个注册因子
- 本轮清理后：**647** 个注册因子
- .fea 数据文件：689 → **641** 个

---


### 8.9 2026-07-26 第三轮清理 — 自动化体检（9个因子）

本轮的因子通过多核 CPU 自动化体检脚本对全部 734 个 .fea 数据文件逐一检查（排除 target 类），
检查标准：
1. NaN 占比 > 20%（2022年之后）
2. 截面方差为零（全市场所有股票值相同）
3. 日频变化率为零（因子不随交易日变化）

共发现 **9** 个不合格因子，分三类：

#### NaN > 20%（3个）

| 因子 | NaN 率 | 来源文件 | 原因 |
|---|---|---|---|
| `leverage_bet_ratio` | 30.9% | `margin_advanced.py` | 融资融券数据仅覆盖约 85% 股票，叠加融券卖出数据更稀疏 |
| `dv_ttm_stability_20d` | 24.4% | `valuation_deep_extended.py` | 股息率 TTM 覆盖约 78% 股票，叠加 20 日滚动窗口 → NaN 超标 |
| `dv_ratio_change_20d` | 23.8% | `valuation_deep_extended.py` | dv_ratio 覆盖约 78% 股票，20 日变化率进一步放大缺失 |

#### 零截面方差 — 时间编码因子（6个）

这些因子在每期截面上所有股票值相同（由日期决定的全局属性），
不符合因子需有截面区分度的要求。根据 §8.5 已确定的时间编码因子删除政策，全部剔除。

| 因子 | 来源文件 |
|---|---|
| `non_trading_day_gap` | `calendar_gap.py` |
| `holiday_proximity` | `calendar_gap.py` |
| `month_start_gap_effect` | `calendar_gap.py` |
| `weekend_proximity_effect` | `calendar_gap.py` |
| `trading_session_position` | `calendar_extended.py` |
| `month_progress` | `calendar_extended.py` |

#### 修改文件清单

| 操作 | 文件 |
|---|---|
| 删减 | `factors/margin_advanced.py`（移除 `leverage_bet_ratio` 注册块） |
| 删减 | `factors/valuation_deep_extended.py`（移除 `dv_ratio_change_20d`、`dv_ttm_stability_20d`） |
| 删除 | `factors/calendar_gap.py`（整文件，4 个因子全部剔除） |
| 删除 | `factors/calendar_extended.py`（整文件，2 个因子全部剔除） |
| 删除 | `data/factors/*.fea`（9 个） |
| 删除 | `data/manifests/*.json` + `*.done`（各 9 个） |

#### 注册表状态更新

- 本轮清理前：702 个注册因子（skill.md §8.8 记录为 647，期间新增 55 个）
- 本轮清理后：**693** 个注册因子
- .fea 数据文件：734 → **725** 个

---

*最后更新：2026-07-26*

### 3.4 禁止 `.rolling().corr()` 在 `groupby().transform()` 中

```python
# ❌ 禁止：rolling corr 在 groupby transform 中极慢（5000+ 股票 × 1500+ 日）
#    每个 group 串行执行 → 总耗时可达数小时，永远跑不完
overnight.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).corr(intraday)
)

# ✅ 替代方案：unstack 到宽表 → 向量化 rolling corr → stack 回 Series
wide_a = series_a.unstack("Code")     # Date × Code
wide_b = series_b.unstack("Code")
corr_wide = wide_a.rolling(20).corr(wide_b)  # C 级别向量化
result = corr_wide.stack(future_stack=True)   # 回到 (Date, Code) MultiIndex
```

---

### 8.10 2026-07-31 除权/未来函数专项清理(9 删 6 增)

本轮基于「因子只使用历史数据(无未来函数)」与「不使用 daily_adj(前复权会被未来事件回溯改写)」两大原则,对全部因子做除权影响与未来函数审计后修改:

#### 删除因子(9 个)

| 因子 | 来源文件 | 原因 |
|---|---|---|
| `price_to_52w_high` / `price_to_52w_high_dist` / `price_to_52w_low_dist` | `behavioral.py` | 未复权 close 的 252 日滚动高低点,除权后信号被污染最长一整年 |
| `disposition_effect` | `behavioral.py` | 同上(叠加 winner_rate) |
| `anchoring_to_52w_high` / `anchoring_to_52w_low` | `behavioral.py` | 同上 |
| `max_drawdown_120` / `ulcer_index_20` | `risk.py`(整文件删除) | 基于未复权 close 的回撤计算,cummax 滞留 120 日 |
| `chip_historical_position` | `chip.py` | **未来函数**:cyq_perf 的 his_low/his_high 是全生命周期极值,T 日看到未来价位 |

#### 除权失真修复(3 处 close.shift(1) → pre_close)

pre_close 在除权日已复权,close.shift(1) 是未复权昨收——两者混算复刻除权失真:

| 文件 | 影响因子 |
|---|---|
| `price.py` factor_overnight_gap_vol_20 | `overnight_gap_vol_20` |
| `trend_pattern.py` `_true_range`(+ 两处调用点,含 `_directional_movement` 错传 high 的既有 bug) | `adx_14` / `di_plus_minus_ratio_14` / `keltner_position_20` |
| `technical_pattern.py` 两处 ATR | `donchian_breakout_strength` / `atr_ratio_20` |

#### 其他清理

- `trend_pattern.py`:12 个因子移除虚假的 `calendar.parquet` 依赖声明(未使用,违反 §5.2)
- `ths_daily_factors.py`:删除死代码 `_stock_returns`(无调用方,且为 close.pct_change 模式)
- `dataset.py` `load_panel`:新增 `_drop_backfill_rows`,统一过滤 is_backfill==True 的行(回填/重下载数据,集中在 2026-05 之后;用 ne(True) 保留 False/None,无该列的表自动跳过)

#### 新增因子(6 个,`factors/momentum_rebuilt.py`)

基于**自建后复权基座** adj = cumprod(1 + pct_chg/100)(pct_chg 为数据商复权口径日收益,来自调整后的 pre_close):
- 该基座 point-in-time 稳定:新分红只影响新一天,不回溯改写历史(区别于 daily_adj 的前复权)
- 除权日无跳变、无未来函数、仅依赖 daily.parquet

| 因子 | 说明 |
|---|---|
| `momentum_5` / `momentum_10` / `momentum_20` / `momentum_60` | 5/10/20/60 日动量 |
| `short_term_reversal_5` | 5 日反转(负向 5 日动量) |
| `momentum_stability_20_60` | 20-60 日动量差(趋势加速度) |

#### 注册表状态

- 修改后注册因子:**518** 个(Class 1 面板 270 + Class 2/3/4 分钟级 + Class 5 耦合 + target 5)
- 本文件 §8.9 记录的 693 为 7-26 状态,7-31 已先期清理 ~200 个依赖 close.pct_change() 的因子,本轮在此基础上再 -9 +6

#### 数据基座原则(重申)

- 因子一律使用未复权 `daily.parquet`;**禁止** `daily_adj.parquet`(仅 target/ret.py 的 label_ret_* 可用)
- 收益率计算统一走 `pct_chg` / `pre_close` 口径,禁止 `close.pct_change()` / `close.shift(1)` 冒充昨收
- 需要连续价格序列的因子,用 cumprod(1 + pct_chg/100) 自建后复权基座(见 momentum_rebuilt.py)
- 新因子必须通过 is_backfill 过滤后的面板构建(数据层已统一处理,因子代码无需感知)

*最后更新:2026-07-31*

---

### 8.11 2026-08-05 全面审计修复 + 76 个新因子开发

#### 一、审计发现并修复的问题(全部已改代码)

| 问题 | 位置 | 修复 |
|---|---|---|
| **margin 跨源 1 日错配**(margin 已 shift(1) 而分母未 shift) | `margin_advanced.py` total_leverage_ratio;`margin_short_deep.py` short_sell_volume_ratio | 分母 total_mv / vol 同步 `groupby(level="Code").shift(1)` 对齐 |
| **耦合因子二次取反**(输入 .fea 均为"高=好方向"编码,再用 (1-X) 或直接用 X 造成选向相反) | `coupling_volume_price.py` 6 处(size/lowvol/idio/liquidity/turnover×2) | 改为直接乘 X 或 (1.0-X) 翻回;`coupling.py` winner_rate_factor_momentum_20 去掉双重取负 |
| **ma_cross_count 构建方向与描述相反** | `indicator_minute.py` spec "pos" | 改为 "neg" 与回退函数/描述一致 |
| **his_high/his_low 上游回溯改写**(实测 2026-06-12 无交易事件跳变 21.0→20.8;复权口径与未复权 close 混比) | `chip_deep.py` chip_support_resistance_20d;`chip_cost_extended.py` distance_from_all_time_high/low | 三个因子删除(与已删 chip_historical_position 同源),.fea 同步清除 |
| **未复权 rolling 极值除权污染**(跨日窗口,与 7-31 已修口径不一致) | `price_deep.py` price_position_20d/high_low_amplitude_20;`trend_pattern.py` _directional_movement(ADX/DM)/donchian_breakout_20;`technical_pattern.py` donchian_position_20/donchian_breakout_strength;`fund_flow.py` large_order_timing_signal | 全部改用 `_adjusted_close` 基座或 `scale=adj/close` 折算 high/low;DM 部分与 TR 同口径 |
| **intraday_deep.py 伪分钟 + 除权** | `intraday_deep.py` 13 因子 | 确认是 daily 代理(Class 1),docstring/描述改为与实现一致;open_auction/gap 的 close.shift(1) 改 pre_close |
| **spec 孤儿条目**(无注册函数或方向无分支,Phase 3 落 else 产出空帧) | `indicator_minute.py` 4 个已删因子残留 + 10 个 momentum 孤儿;`intraday.py` 10 个 momentum 孤儿 + liquidity_resilience 残留 | 全部移除 spec 条目 + 底层无引用指标计算;注意:已删因子不进 FACTOR_REGISTRY,CLI 默认不选中,spec 残留仅影响显式 -f 调用 |
| **死代码/死文件** | price_deep.py 不可达 return(注意:函数内首个 return 后的同缩进语句是合法但不可达代码,不报 SyntaxError,需人工识别);fund_flow_deep.py big_net 未使用;margin_extended.py/ths_daily_factors.py/stock_identity_extended.py/short_selling_factors.py 零注册 | 清理删除 |
| **描述与实现脱节** | mf_open_close_divergence_10d(开盘/收盘背离→净流入偏离波动);mf_big_order_turnover_ratio 与 small_order_crowding(描述"成交量"实为成交额→改用 *_vol 字段);dv_stability_4q("4季"实为 60 交易日);short_sell_velocity_5d 改名 short_sell_volume_ratio(单日占比);close_auction_pressure/volume_distribution_skew 描述统一 | 改实现或改描述 |
| **漏声明依赖** | valuation_deep.py pb_industry_adjusted 用 load_industry_map 未声明 stock_list.parquet | 补声明 |
| **静态快照时点回溯**(当前 stock_list 回填历史) | sector.py(已声明)/stock_identity.py/neutral.py | 补 ⚠️ 声明;属轻微时点回溯,行业归属变化缓慢可接受 |
| **daily_adj 变量名误导**(实际 load daily.parquet) | price.py/chip.py/chip_deep.py/fund_flow.py 多处 | 改名 daily_panel/daily_path |

#### 二、⚠️ 注意事项(新因子开发必读)

1. **margin 跨源混算**:margin 面板已被数据层统一 shift(1)(Date=T 上是原始 T-1 值)。凡 margin 因子与非 margin 数据(如 daily.vol、finance.total_mv)混算,分母必须同步 `groupby(level="Code").shift(1)`,否则 1 日错配。
2. **耦合因子方向契约**:Class 5 输入 .fea 均为"高=好方向"编码(log_circ_mv 高=小盘、turnover_20 高=低换手、amihud_intraday 高=高流动性、parkinson_vol 高=低波、idio_vol_60 高=低特质波、winner_rate 高=低获利盘)。需要"反向"时用 `(1.0 - X)` 翻回;再取反=二次取反。开新耦合因子前先核对源因子注册方向。
3. **pandas 3.0 陷阱(2026-08-05 新增)**:
   - `Rolling.prod()` 已移除 → 用 `(1+r).cumprod().pct_change(N, fill_method=None)` 求 N 日累计收益;
   - `SeriesGroupBy.where()` 不存在 → 先 `df[col].where(mask)` 再 `groupby(Series)`(groupby 需传 Series grouper,不能传列名);
   - DataFrame/Series 除法必须显式 `axis=0`(默认 axis="columns" 按列名对齐会全 NaN)。
4. **his_high/his_low 禁用**:cyq_perf 的 his_low/his_high 为第三方复权口径且会被上游回溯改写,与未复权 close 混比口径不一致。任何"距历史高低点"类因子一律用 `_adjusted_close` 基座自建(见 momentum_rebuilt.py)。
5. **单位陷阱**:daily.vol 单位是**手**,amount 是**元**;`close×vol` 不是元。VWAP 类计算统一 `amount/vol` 口径。
6. **事件因子无未来函数**:所有"事件后表现"类因子用**滞后事件**(`event.shift(N)` 按 Code 分组)与当期收益对齐,严禁把事件日当日的未来窗口收益算进当日因子。
7. **聪明钱 zdf 需过滤**:`|ret|/√vol` 计算前必须过滤 `vol>0 & |ret|>0` 的分钟,否则无成交分钟 zdf=0 混入 top 分位。
8. **涨跌停判定**:用 `pct_chg >= 9.8%` 近似主板涨停(ST 5%/双创 20% 边界不同,不区分板块,属覆盖范围问题)。

#### 三、2026-08-05 新开发 76 个因子(五类框架)

| 模块 | 类 | 数量 | 因子 |
|---|---|---|---|
| `technical_daily.py` | 1 | 26 | kama_efficiency_20、trix_12_20、trix_signal_gap、bias_60、td_setup_count、rs_60/120/250、macd_daily_hist_5d、rsi_spread_6_14、kdj_daily_j、cci_20、roc_12、aroon_up_25、aroon_down_25、dpo_20、williams_r_14、stoch_slow_k、force_index_13、eom_14、mfi_14、obv_slope_20、obv_divergence_20、residual_momentum_20、tail_corr_60、fear_index_20 |
| `event_dynamics.py` | 1 | 13 | limit_up_event_5、limit_down_event_5、consecutive_limit_up、limit_up_fade_10、gap_event_decay_5、volume_spike_event、new_high_60_event、new_low_60_event、extreme_move_event、three_black_crows、limit_alternation_20、new_high_frequency_60、big_gap_reversal_5 |
| `risk_metrics.py` | 1 | 13 | downside_frequency_60、var_95_20、sortino_ratio_60、loss_probability_20、vol_regime_switch_20、vol_decay_ratio_20、vol_clustering_20、tail_risk_pct_60、downside_upside_vol_60、drawdown_duration_120、drawdown_recovery_60、max_consecutive_loss_20、range_vol_ratio_20 |
| `volume_price_dynamics.py` | 1 | 12 | vwap_daily_deviation、avg_price_trend_20、up_day_volume_ratio_20、volume_skew_5d、zero_return_fraction_20、liquidity_shock_20、turnover_ret_corr_20、accumulation_distribution_20、volume_price_divergence_score、ah_volume_ratio、ah_amount_ratio、ah_activity_change_20d(AH 因子覆盖仅 A+H 股,预计高 NaN) |
| `intraday.py`(聪明钱) | 3 | 4 | smart_money_vwap_ratio、smart_money_share、smart_money_net_bias、large_trade_intensity(spec 新增 metric,方向 pos) |
| `coupling_daily.py` | 5 | 8 | momentum_rs_resonance_20、lowvol_momentum_rs_20、value_event_combo_20、rs_value_divergence_20、event_momentum_divergence_20、turnover_event_confirmation_20、vol_liquidity_resonance_20、momentum_stability_combo_60 |

开发素材:99 个聚宽策略(技术指标/动量/事件/聪明钱)+ 63 份券商研报(残差动量/尾部相关/恐惧指标)。
构建顺序依赖:coupling_daily.py 的 6 个因子依赖 rs_60/limit_up_event_5 的 .fea,须先构建 Class 1 新因子。

#### 注册表状态

- 修改后注册因子:**609** 个(Class 1 344 + Class 2 26 + Class 3 80 + Class 4 66 + Class 5 93)
- 已删除 .fea:chip_support_resistance_20d / distance_from_all_time_high / distance_from_all_time_low / short_sell_velocity_5d(改名 short_sell_volume_ratio)

*最后更新:2026-08-05*

### 8.12 2026-08-05 新开发 42 个因子(第二轮,六模块)

在 8.11 的 76 个新因子基础上补充 42 个(Class 1 × 34 + Class 5 × 8),全部通过
全量数据冒烟测试(NaN < 20%、截面有区分度、日频变动),部分实测指标见下表。

| 模块 | 类 | 数量 | 因子 |
|---|---|---|---|
| `structure_patterns.py` | 1 | 13 | consecutive_limit_down、one_word_limit_up_freq_20、one_word_limit_down_freq_20、limit_up_open_fail_freq_20(炸板)、limit_down_rebound_10、inside_bar_count_20(孕线)、outside_bar_count_20(吞没)、doji_frequency_20、big_range_day_freq_20、current_up_streak、current_down_streak、current_vol_shrink_streak、gap_open_follow_ratio_20 |
| `momentum_structure.py` | 1 | 10 | price_distance_from_52w_low、momentum_accel_60_120、rebound_from_low_20、volume_autocorr_5/20、ret_vol_lead_corr_20(量领先价)、ret_autocorr_1d_20、amihud_trend_20_60、amihud_asymmetry_20、overnight_return_share_20 |
| `risk_structure.py` | 1 | 6 | cvar_95_120、panic_selling_ratio_60(放量×下跌)、vol_cycle_position_120、gain_loss_asymmetry_60、market_regime_sensitivity_60、extreme_gain_freq_20 |
| `margin_extra.py` | 1 | 2 | margin_leverage_change_20d、short_balance_ratio_change_20d |
| `finance_extra.py` | 1 | 3 | pb_change_20d、volume_ratio_momentum_5d、free_float_expansion_20d(解禁代理) |
| `coupling_daily_extra.py` | 5 | 8 | value_reversal_combo_60、momentum_high_proximity_combo_20、lowvol_trend_efficiency_combo_20、reversal_oversold_combo_5、winner_rs_combo_60、volume_event_momentum_combo_10、margin_price_resonance_20、smart_money_momentum_combo_20 |

#### 开发要点与坑(本批实测发现)

1. **同日比较无需复权折算**:high/open/close 与当日 pre_close 同尺度
   (pre_close 在除权日已复权到当日价格尺度),直接比较正确;只有跨日比较
   (孕线/吞没的昨日高低)才需 scale=adj/close 折算。曾误用折算导致 high_pct
   全为负(adj 为相对量级 ≈1.0,与真实价格混算)。
2. **margin 跨源混算的索引并集陷阱**:margin 面板(164 万行)与 finance
   面板(219 万行)直接相除会做索引并集,把 25% 无融资数据股票的缺行以 NaN
   拉入结果 → 必须 `total_mv.reindex(m.index)` 后再除(既有 margin 因子全为
   纯 margin 字段运算故无此问题)。
3. **尾部/条件因子的样本量**:CVaR 类(5% 尾部日呈泊松分布)与低放量频率
   股票的条件占比,短窗口会大面积 NaN——估计窗需拉长到 120/60 日
   (cvar_95_120 NaN 18.6%、panic_selling_ratio_60 NaN 1.8%,20 日版本实测
   93%/22% 不合格)。
4. **稀疏事件因子**:consecutive_limit_down(86 日有截面区分)与
   one_word_limit_down_freq_20(46 日)为跌停侧镜像,2023-2024 跌停连板
   显著少于涨停连板(consecutive_limit_up 733 日),与既有因子同构,保留待体检。
5. Class 5 中 5 个因子依赖 8.11 批次未构建 .fea(kama_efficiency_20 /
   williams_r_14 / rs_60 / volume_spike_event / smart_money_net_bias),
   构建顺序:先 `--only-class 1`(及 Class 3 的 smart_money)再 Class 5。

#### 注册表状态

- 修改后注册因子:**651** 个(609 + 42,Class 1 378 + Class 2 26 + Class 3 80 + Class 4 66 + Class 5 101)

*最后更新:2026-08-05*

### 8.13 2026-08-05 新开发 49 个因子(第三轮,Class 2/3/4 深入 + Class 5 多因子耦合)

在 8.11/8.12 基础上深入 Class 2/3/4(逐股目录数据源)并升级 Class 5
(双因子→三因子耦合 + A股特色量价起飞)。全部经真实数据验证:
Class 2 统一构建器 Phase 2/3 全量 1782 只股票零错误;Class 3/4 扇出模拟
23 因子 1840 交易日全有截面区分;Class 5 实算 17 因子 993 交易日全有区分。

| 模块 | 类 | 数量 | 说明 |
|---|---|---|---|
| `chip_deep.py`(改) | 2 | 10 | 新增4个原始指标 + 10条 CHIP_FACTOR_SPEC |
| `chip_deep_extra.py`(新) | 2 | 10 | 回退注册函数(与 spec 方向一一对应) |
| `intraday.py`(改) | 3 | 10 | 新增11个时段结构指标 + 10条 INTRADAY_FACTOR_SPEC |
| `intraday_extra.py`(新) | 3 | 10 | 回退注册函数 |
| `indicator_minute.py`(改) | 4 | 13 | 新增10个指标 + 13条 INDICATOR_FACTOR_SPEC |
| `indicator_minute_extra.py`(新) | 4 | 13 | 回退注册函数 |
| `coupling_daily_extra2.py`(新) | 5 | 16 | 三因子耦合(量价起飞族6 + 左侧/质量/防御8 + margin 2) |

#### Class 2 新因子(筹码分布,10)

新原始指标:`chip_below_90`(深套盘<0.9×close)、`chip_upper_110`(高位浮筹>1.1×close)、
`chip_mode_median_gap`(双峰性|峰−中位|/std)、`chip_range_skew`(分位数偏斜)。
因子:chip_deep_trap_ratio / chip_high_float_ratio / chip_bimodality /
chip_range_skew_factor(形态族);chip_median_momentum / chip_weighted_mean_momentum
(成本重心上移);chip_iqr_momentum_20d / chip_semi_std_momentum /
chip_width_ratio_momentum / chip_percentile_20d(离散与风险动量)。
实测:深套盘均值14%、高位浮筹29%,2022+ NaN 与既有 chip 因子同水平。

#### Class 3 新因子(1分钟线时段结构,10)

新原始指标:`am_momentum`(上午动量,对称于 pm_momentum)、`open_30_mom`/
`close_30_mom`(首尾30分钟动量,补 open_30_open)、`open_close_30_mom_ratio`(首尾
强度差=冲高回落风险)、`open_30_vol_share`/`close_30_vol_share`(首尾量占比)、
`am_pm_hl_range_ratio`(上下午振幅比)、`vwap_am_pm_gap`、`open_30_range_pct`/
`close_30_range_pct`(首尾振幅占全天)、`am_hl_position`(上午收盘位置)。
因子:am_momentum_intraday / open_30_momentum / close_30_momentum /
open_close_momentum_gap / am_pm_range_ratio / tail_volume_share /
open_volume_share / vwap_am_pm_gap_factor / open_30_range_share / am_close_position。

#### Class 4 新因子(分钟指标结构,13)

未映射指标补位:`price_vs_ma10_deviation` / `price_vs_ma30_deviation` /
`kdj_dead_cross_count`。新指标:`rsi_day_position`/`j_day_position`(收盘在当日
区间位置)、`macd_bar_energy`(|macd| 均值)、`ma10_slope`/`ma30_slope`/
`ma5_ma10_gap`(均线系统斜率谱)、跨日:`boll_width_5d_chg`、`rsi_excess_ma5`、
`kdj_bull_frac_5d_chg`、`j_close_5d_gap`。

#### Class 5 三因子耦合(A股特色,16)

- **量价起飞族(6)**:volume_price_liftoff_20(momentum×放量突破×浅回撤)、
  smallcap_liftoff_combo_60(小盘×动量×量增)、value_liftoff_combo_20、
  lowvol_liftoff_combo_20、liftoff_pulse_combo_20(资金脉冲)、
  smart_capital_liftoff_20(主力净流入)。
- **左侧/质量/防御(8)**:deep_value_reversal_combo_60(低估×深回撤×超跌三重左侧)、
  lowvol_quality_momentum_60、chip_price_resonance_20(筹码×动量×52周高)、
  defensive_momentum_combo_60(低beta×动量×低溃疡)、multi_horizon_momentum_combo_20
  (5/10/20日动量共振)、fund_flow_alpha_combo_60、quality_liquidity_combo_20
  (高流动性×低波×低特质波)、reversal_liquidity_combo_5。
- **margin(2)**:margin_trend_combo_20、margin_value_combo_20。

#### 坑与修复(本批实测)

1. **margin 耦合的索引并集**:margin .fea 为全网格(1596×1782)存储,
   自身 2022+ NaN 9.2%(全量14.9%),耦合时以 margin 因子索引 reindex 其余
   因子防并集膨胀;已同步修复 8.12 的 margin_price_resonance_20。
   ⚠️ 注意:2022+ 口径合格,但含 2019-2021 的窗口(早期两融覆盖差)NaN 达 21%,
   评估时需用 2022+ 口径。
2. **SeriesGroupBy 无 .abs()**:`grouped["macd"].abs().mean()` 报 AttributeError,
   改为 `df["macd"].abs().groupby(df["trade_date"]).mean()` 向量化。
3. **回退函数与 spec 方向必须一一对应**:新增 spec 条目时,回退注册函数
   (chip_deep_extra/intraday_extra/indicator_minute_extra)方向需与 Phase 3
   的 pos/neg/neg_abs/momentum/momentum_rev/momentum_20d 完全一致。
4. **写盘权限**:data/factors 为 root 所有,非 root 运行构建会 PermissionError
   (环境问题,与代码无关;Phase 2/3 已全量验证)。

#### 注册表状态

- 修改后注册因子:**700** 个(Class 1 378 + Class 2 36 + Class 3 90 + Class 4 79 + Class 5 117)

*最后更新:2026-08-05*

---

### 8.14 2026-08-05 因子数据自动体检(第4轮,6 删)

对全部 700 个 `.fea` 做多核并行体检(28 进程),检查标准:
1. **NaN 占比 > 20%**(2022-01-01 之后、全市场所有股票)
2. **截面方差为零**(2022+ 日期中 ≥98% 的截面 std==0)
3. **日频变化率为零**(全区间无任何取值变化)
4. **非连续数据**(2022+ 去重值 ≤ 2,哑变量/常数)

#### 不合格因子(6 个)

| 因子 | 类 | NaN 率 | 来源文件 | 原因 |
|---|---|---|---|---|
| `ah_volume_ratio` | 1 | 99.5% | `volume_price_dynamics.py` | A+H 股仅百余只,非 AH 全 NaN |
| `ah_amount_ratio` | 1 | 99.5% | `volume_price_dynamics.py` | 同上 |
| `ah_activity_change_20d` | 1 | 100% | `volume_price_dynamics.py` | 全 NaN,0 日频变化,0 去重值 |
| `updown_volume_ratio_20` | 1 | 82.6% | `volume_price_coupling.py` | 量能方向性计算大面积缺失 |
| `volume_spike_event` | 1 | 24.0% | `event_dynamics.py` | NaN > 20% |
| `volume_event_momentum_combo_10` | 5 | 24.0% | `coupling_daily_extra.py` | 自身 NaN > 20%,且依赖 `volume_spike_event`(级联删除) |

#### 边界审查保留(未达阈值)

- **NaN 10-20%**:`pe_ttm_change_20d`(17.4%)、`short_interest_volatility_20d`(14.9%)、`cvar_95_120`(10.2%)——均 < 20%,方差与日频变动正常,保留。
- **稀疏事件因子**:`consecutive_limit_up/down`、`one_word_limit_down_freq_20`、`big_gap_reversal_5` 零方差日占比最高 47%,但日频变动 807-1390/1596 日、去重值 54+,符合 §2.2/2.3,保留(与 §8.12"保留待体检"结论一致)。

#### 修改文件清单

| 操作 | 文件 |
|---|---|
| 删减 | `factors/volume_price_dynamics.py`(3 个 A-H 注册块) |
| 删减 | `factors/volume_price_coupling.py`(1 个) |
| 删减 | `factors/event_dynamics.py`(1 个) |
| 删减 | `factors/coupling_daily_extra.py`(1 个,Class 5 级联) |
| 删除 | `data/factors/*.fea`(6 个) |
| 删除 | `data/manifests/*.json` + `*.done`(各 6 个) |

#### 注册表状态

- 修改后注册因子:**694** 个(Class 1 372 + Class 2 36 + Class 3 90 + Class 4 79 + Class 5 116)
- .fea 数据文件:700 → **694**;manifests json 700→694、done 500→494

*最后更新:2026-08-05*

---

### 8.15 2026-08-08 debug 实证：数据源白名单定型（8 源）

基于 debug/20260806、debug/20260807 两次 10 交易日全量切片（1782 只股票 × 10 日，
逐格 diff，9 个重叠交易日）总结，数据源准入规则定型。

**上游修订实证（谁在变）**：

| 数据 | 0805 vs 0806 | 0806 vs 0807 | 判定 |
|---|---|---|---|
| `daily_adj.parquet` | 25 行 / 98 格 | 57 行 / 274 格（14 只除权，08-03~08-06，最大差 2.06） | ❌ 前复权改写历史 |
| `finance.parquet` pe_ttm_percentile | 2073 格（08-05 回填 0→真实分位） | 3916 格（08-06 同模式） | ❌ T+1 回填重算；pe_ttm 本身零变动 |
| `ths_daily.parquet` | 355 行 / 393 格（08-05 全列修订） | 0 | ❌ T 日数据 T+1 被改写 |
| `daily.parquet` | 仅 stock_name 7 格 | 仅 stock_name 1 格 | ✅ 数值零变动（stock_name 为元数据） |
| `cyq_perf` / `main_fund_flow` / `margin_detail` / `cyq_chips` / `history_1min` / `indicator_1min` | 0 | 0 | ✅ 只追加不修改 |
| 686 个 factors/*.fea | 0 | 0 | ✅ 9 重叠日零变动 |

（daily_dump_1min 同样零变动，但无任何因子依赖，按口径不列入白名单。）

**结论 1 — 白名单 8 源**：因子开发仅允许使用
`daily` / `finance` / `cyq_perf` / `main_fund_flow` / `margin_detail` / `cyq_chips` /
`history_1min` / `indicator_1min`。
这是实证"数值零变动（只追加不修改）"的完整集合；全部因子（运行时注册表 − 5 标签 = 686，
与切片 .fea 数量一致）的上游依赖普查也恰好落在这 8 个源（20 种文件组合，无第 9 类来源）。

**结论 2 — 字段级禁用**：
- `finance.pe_ttm_percentile`：T 日 0 占位 → T+1 全市场回填真实分位，T 日用到它的因子值会被改写；
- `cyq_perf.his_low / his_high`：上游回溯改写 + 复权口径与未复权 close 混比（§8.11 已删同源因子）。

**结论 3 — THS 三件套永久不可用**（ths_daily / ths_constituent_stocks / ths_sector_categories）：
- ths_daily 为**纯指数日线**（2571 个指数代码，无任何个股行），落地到个股必须经成分映射；
- ths_constituent_stocks **无日期列**（当前快照）：700001.TI 一周内成员 5000→5537
  （新增 537 只、0 移除），无成员历史 → 任何历史行业归属都是前视偏差；上游另有
  875 只代码重复 ×2 的生成缺陷；
- 时间覆盖不足：69% 的指数无 2019~2022 数据（700001.TI 全A指数仅 2023-01 起）；
- 三文件指数口径不一致（2571 vs 1666 vs 1664）。
- 工程侧已于 2026-07-31 将 sector 因子 THS 路径替换为 stock_list 兜底，与本节一致。

规范落点：白名单 → §1.1；PIT 契约 → §1.3；开发前检查 → §6；数据源状态 → §7。

### 8.16 2026-08-08 新开发 40 个因子(第四轮,策略素材挖掘,五类全覆盖)

基于 99 个聚宽策略 + 券商研报(净换手率/龙虎榜替代/RSRS 量能加权)挖掘开发,
全部通过真实数据冒烟测试(2024 上半年窗口,NaN < 20%、截面有区分度、日频变动)。

| 模块 | 类 | 数量 | 因子 |
|---|---|---|---|
| `strategy_daily.py`(新) | 1 | 26 | psy_12、up_down_count_ratio_20、cmo_20、ppo_signal_12_26_9、bias_signal_29_19、kama_position_20、rsrs_volume_right_deviation、rsrs_beta_momentum_5、fib_retracement_proximity、atr_position_250、chandelier_position、gap_intraday_corr_20、high_open_low_close_frac_20、limit_board_streak_mean_60、limit_up_vol_shrink_60、limit_streak_volume_ratio、listing_age_heat、net_turnover_rate_20、lhb_proxy_score_60、margin_buyer_avg_cost_premium、margin_chip_cost_gap、margin_proxy_ttm、mf_tier_net_spread_20、mf_big_order_net_kurt_20、big_order_net_accel_10、elg_net_60d_to_mv |
| `intraday.py`/`intraday_extra.py`(改) | 3 | 4 | intraday_high_time、min_limit_touch_frac_20、min_bar_gap_freq_20、min_vwap_dev_std |
| `indicator_minute.py`/`indicator_minute_extra.py`(改) | 4 | 5 | min_macd_hist_area_20、min_j_overbought_frac_20、min_boll_width_std_20、min_ma_alignment_frac_20、min_rsi_extreme_frac_20 |
| `coupling_daily_extra3.py`(新) | 5 | 5 | coupling_bigflow_margin_buy_20、coupling_winner_bigflow_20、coupling_margin_chip_cost_20、coupling_net_turnover_momentum_20、coupling_lhb_reversal_20 |

#### 数据单位实证(2026-08-08,与既有注释不一致处以此为准)

| 字段 | 实证结果 | 说明 |
|---|---|---|
| daily.vol | **股** | close×vol ≈ amount(中位比 0.9999),skill.md §8.11 的「手」注记有误 |
| history_1min.vol | **手** | amount/(close×vol) ≈ 100 |
| main_fund_flow 各档 vol | **手** | 四档买卖量合计/日 vol ≈ 2(买卖各计一次) |
| main_fund_flow 各档 amount | **万元** | (买+卖)8 档金额合计/daily.amount ≈ 2e-4 |
| finance.free_share / circ_mv | 股 / 元 | 平安银行 2019 free_share≈76 亿股、circ_mv≈1578 亿元 |

**注意**:库内既有 intraday vwap = amount/vol(vol 手),实为 100× 价格(实测 vwap_dev≈−0.99 近常量);
本批次 min_vwap_dev_std 用 amount/(vol×100) 真 VWAP 口径,未复刻该缺陷。

#### 开发要点与坑(本批实测)

1. **pandas 3.0 + numpy 2:ufunc 作用于 Series 返回 ndarray**——`np.maximum.reduce`/
   `np.minimum.reduce`/`np.abs` 结果必须 `pd.Series(..., index=...)` 包回,
   否则 `cross_sectional_rank` 内部 `.replace()` 报 AttributeError(3 处)。
2. **运算符优先级**:`sealed & ~sealed.astype(bool).groupby(...)` 中 `~` 绑定整条
   链(属性访问优先于一元运算符),须先 `prev = ...shift(1).fillna(False)` 再 `& ~prev`。
3. **Index.groupby 无 level 参数**:`DatetimeIndex.groupby(level=...)` 报错,先转
   `pd.Series(dates, index=...)` 再 groupby。
4. **稀疏事件窗口 min_periods**:涨停类事件 min_periods 必须 1(60 日内 ≥1 个事件即
   估),10 会导致 NaN 67% 不合格;无事件股票 fillna(0/1.0) 中性填充(limit_board_
   streak_mean_60 NaN 67%→0%、limit_up_vol_shrink_60 99.5%→0%)。
5. **250/400 日长窗因子的 NaN 假象**:冒烟窗口(800 日 lookback 截断历史)前 ~20%
   日期全 NaN,限定 2023+ 日期均为 0.00%——真实全量构建(2019 起)无此问题。
6. **margin 配价**:margin 面板已 shift(1),加权成本必须用 close 逐股 shift(1)
   (T-1 收盘)配对,否则 1 日错配(margin_buyer_avg_cost_premium/margin_chip_cost_gap)。
7. **margin 耦合 reindex**:含 margin 输入的三因子耦合以 margin 因子索引 reindex
   其余因子防并集(coupling_margin_chip_cost_20,8.13 先例)。
8. **Class 5 构建顺序**:coupling_daily_extra3.py 依赖本批次 Class 1 新因子 .fea,
   先 `--only-class 1` 再构建。

#### 注册表状态

- 修改后注册因子:**733**(691 + 42:Class 1 404 + Class 2 38 + Class 3 94 + Class 4 84 + Class 5 113)

*最后更新:2026-08-08*
*最后更新:2026-08-08*

### 8.17 2026-08-08 全量因子自动体检(第5轮,1 删)

对全部 726 个 `.fea` 做多核并行体检(56 进程),检查标准:
1. **NaN 占比 > 20%**(2022-01-01 之后、全市场所有股票)
2. **截面方差为零**(2022+ 日期中 ≥98% 的截面 std==0)
3. **日频变化率为零**(全区间无任何取值变化)
4. **非连续数据**(2022+ 去重值 ≤ 2,哑变量/常数)
另附复核指标:日变化率、有限值逐日不变占比(staleFrac,NaN 打断)、最长连续同值天数。

#### 不合格因子(1 个)

| 因子 | NaN 率 | 来源文件 | 原因 |
|---|---|---|---|
| `gap_fill_5d` | 83.4% | `price_deep.py` | 5 日缺口回补事件过于稀疏,大部分股票大部分日期无有效值 |

#### 边界审查保留(未达阈值)

- **NaN 10-20%**:`margin_proxy_ttm`(18.3%)、`pe_ttm_change_20d`(17.4%)、`short_interest_volatility_20d`(14.9%)——均 < 20%,方差与日频变动正常,保留(与 §8.14 边界审查结论一致)。
- **稀疏事件因子**:`one_word_limit_down_freq_20`(staleFrac 54.6%)、`consecutive_limit_down`(49.6%)、`big_gap_reversal_5`(31.6%)、`limit_down_event_5`(30.2%)等——无事件日计数自然不变,但日频变动 506-1112/1113 日、去重值 55+、std0 日期占比最高 47%,符合 §2.2/2.3,保留(§8.14 先例)。
- **低频股本结构因子**:`float_mv_ratio`(staleFrac 36.2%)、`circ_mv_to_total_mv`/`float_share_ratio`(31.9%)——股本结构本身变动缓慢,去重值 1.2 万+,非病态,保留。
- 所有因子最长"连续同值"伪影(1113 日)均来自 2022+ 无数据的稀疏列,非真实停滞;真实 staleFrac 最高 54.6%(稀疏事件类)。

#### 修改文件清单

| 操作 | 文件 |
|---|---|
| 删减 | `factors/price_deep.py`(移除 `gap_fill_5d` 注册块) |
| 删除 | `data/factors/gap_fill_5d.fea` |
| 删除 | `data/manifests/gap_fill_5d.json` + `gap_fill_5d.done` |

#### 注册表状态

- 修改后注册因子:**732**(733 − 1)
- .fea 数据文件:726 → **725**;无孤儿 .fea、无注册缺失(除 5 个 label 与 2 个未构建 Class 2 因子)

*最后更新:2026-08-08*

### 8.18 2026-08-09 新开发 20 个因子(第五轮,Class 5 时间维度耦合 + 新基因多因子耦合)

在 8.13/8.16 基础上升级 Class 5:**时间维度耦合**(跨期自共振/领先-滞后/因子时间
加速度,全部滞后 k ∈ {5,10,20,60} ≤ 60 天,shift 样本期外 NaN 按规范保留)
+ 前四类**从未入耦合的基因**(Class 2 筹码支撑/成本峰、Class 3 尾盘量/聪明钱、
Class 4 分钟均线排列)。全部经真实数据冒烟测试与全量构建。

| 模式 | 数量 | 因子 |
|---|---|---|
| 跨期自共振 X_t×X_t-k | 2 | coupling_flow_persistence_20(融资净流入20日持续)、coupling_margin_buy_persist_10(买入意愿10日持续) |
| 领先-滞后 X_t-k×Y_t | 4 | coupling_moneyflow_lead_momentum_10(资金领先10日)、coupling_volume_lead_momentum_5(量领先5日)、coupling_smartmoney_lead_momentum_10(聪明钱领先10日)、coupling_margin_lead_trend_60(杠杆领先60日=滞后上限) |
| 因子时间加速度 X_t−X_t-k | 3 | coupling_momentum_drift_20(动量加速)、coupling_fundflow_accel_10(资金流加速)、coupling_chip_cost_accel_20(筹码成本加速) |
| 多因子耦合(新基因) | 11 | coupling_chip_support_reversal_5、coupling_chip_trend_confirm_20、coupling_intraday_tail_momentum_20、coupling_min_align_momentum_20、coupling_lowrisk_momentum_60、coupling_quality_trend_60、coupling_stableflow_momentum_20、coupling_flowaccel_breakout_60、coupling_margin_buy_trend_20、coupling_limitup_momentum_20、coupling_volterm_momentum_60 |

#### 开发要点与坑(本批实测)

1. **时间耦合滞后上限**:全批滞后 ∈ {5,10,20,60} 天,60 天为规范上限
   (coupling_margin_lead_trend_60 恰好压线);shift(k) 每股前 k 个交易日
   自然 NaN,不填充,2022+ 评估窗口无影响(数据 2019 起)。
2. **margin 耦合 reindex**:margin_buy_pressure / margin_net_flow_ratio 输入
   的耦合以 margin 因子索引 reindex 其余因子防并集(8.13 先例);纯单基因
   时间耦合(margin 自身 shift)无并集风险,不需 reindex。
3. **避免与既有因子重复**:设计时核对 coupling_indicator_cross.py(coupling_
   indicator_consensus_value 已覆盖 共识×估值,本批不再开发 consensus_value)、
   coupling_extended.py(因子自身 20 日动量已覆盖 value/quality/liquidity,
   本批时间加速度扩展到 momentum/mf/chip 基因)。
4. **冒烟实测**:margin 类 NaN ≈ 17%(85% 覆盖×两时点),其余 < 10%;
   单因子构建 2-3s,全批零依赖(全部输入 .fea 已在 8.16 批次前构建)。

#### 注册表状态

- 修改后注册因子:**752**(732 + 20)
- .fea 数据文件:725 → **745**

*最后更新:2026-08-09*

---

### 8.19 2026-08-11 新开发 23 个因子(第六轮,分钟级量价四象限 + 筹码峰集中 + 量价筹码耦合)

在既有量价因子(全部为日频口径:up_day_volume_ratio_20 / turnover_ret_corr_20 /
volume_price_divergence_score 等)基础上,开发**分钟级(时分)量价耦合体系**:
放量/缩量 × 上涨/下跌 四象限、诱多/诱空、吸筹/洗筹的多角度刻画。
代码已完成并通过注册验证;**本轮未执行全量构建**,构建与评估由项目侧后续执行。

| 模块 | 类 | 数量 | 因子 |
|---|---|---|---|
| `intraday.py`+`intraday_extra.py`(改) | 3 | 11 | vp_expand_up_share、vp_expand_down_share、vp_shrink_up_share、vp_shrink_down_share、vp_consistency_score、vp_consistency_20、up_minute_vol_share、minute_ret_vol_corr、vp_expand_ret_gap、vp_expand_price_pos、vp_expand_down_am_share |
| `indicator_minute.py`+`indicator_minute_extra.py`(改) | 4 | 3 | min_expand_bull_frac_20、min_shrink_bull_frac_20、min_rsi_overbought_expand_20 |
| `chip_deep.py`+`chip_deep_extra.py`(改) | 2 | 3 | chip_win_peak_frac、chip_loss_peak_frac、chip_win_peak_growth |
| `coupling_vp_chips.py`(新) | 5 | 6 | coupling_vp_chip_consistency_20、coupling_vp_expand_up_mom_20、coupling_vp_shrink_down_rev_5、coupling_vp_lowpos_accumulate_20、coupling_vp_retvol_mom_20、coupling_vp_amfade_rev_20 |

#### 核心设计:放量基准与四象限

- **放量/缩量基准** = 分钟量 vs **过去20日同时段(同一 minute)均量**
  (`groupby("minute").rolling(20).mean().shift(1)`,shift 避免当日自引用),
  时段基准校正日内量能 U 型曲线(早尾盘天然放量不误判),仅依赖 history_1min 自身。
- **分钟涨跌** = `ret_1min`(close/前分钟close−1,库内既有口径,首分钟 NaN 不计)。
- **四象限量占比**:放量上涨=资金真实做多(pos);缩量下跌=抛压轻/洗盘(pos);
  缩量上涨=无量反弹/诱多嫌疑(neg);放量下跌=恐慌抛售(neg)。
- **信号映射**:诱多 → vp_shrink_up_share + vp_expand_ret_gap(放量不推价=对倒)
  + min_shrink_bull_frac_20(指标侧无量上涨)+ min_rsi_overbought_expand_20(超买放量追高);
  诱空 → vp_expand_down_am_share(早盘恐慌释放)+ coupling_vp_amfade_rev_20;
  吸筹 → vp_expand_price_pos(低位放量)+ chip_win_peak_frac(获利筹码集中)+
  coupling_vp_lowpos_accumulate_20(低位放量×筹码集中×主力净流入三因子);
  洗筹 → vp_shrink_down_share + coupling_vp_shrink_down_rev_5(缩量回调×超跌)。

#### 开发要点与坑(本批)

1. **Class 4 无量源**:indicator_1min 无原始 vol 列,量能状态用 mavol5/mavol10
   关系代理(放量=mavol5>mavol10),分母限定"两量均线均非 NaN"的有效分钟,
   避免指标未就绪分钟拉低占比。
2. **Class 3 指标全部向量化**:四象限量占比用 `df["vol"].where(mask).groupby(gdate).sum()`
   (SeriesGroupBy.where 不存在,用 df 级 where);分钟量价相关用
   协方差/方差积开方全向量化(避免逐日 apply corr)。
3. **类路由与构建顺序**:Class 5 新文件依赖 Class 3(vp_*)与 Class 2
   (chip_win_peak_frac)新 .fea,构建须先 `--only-class 3,2,4` 再 `--only-class 5`。
   Class 5 全部输入为"高=好"编码,无需 (1.0−X) 翻回,无 margin 输入
   (无 reindex 需求)。
4. **方向契约**:负向因子(vp_expand_down_share / vp_shrink_up_share /
   vp_expand_price_pos / chip_loss_peak_frac 等)已在 spec 中取负,耦合时直接相乘。
5. **NaN 控制**:四象限无稀疏条件(任意有涨跌分钟即有效,2022+ 前 20 日基准
   NaN 属正常滚动窗);Class 2 新指标全股票有 close 即有值。

#### 注册表状态

- 修改后注册因子:**773**(含 5 个 label;非 label 768;本轮 +23,代码层验证
  全部注册成功、spec 无孤儿)
- 本轮未构建 .fea(构建由项目侧后续执行)

*最后更新:2026-08-11*
