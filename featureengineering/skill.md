# 因子开发规范与指南

## 一、核心原则

### 1.1 日频约束（必须）

因子**必须**使用日频原始字段。仅依赖静态属性（如 `stock_list.parquet` 的公司属性）而不使用任何日频数据的因子，一律不合格。

| 许可数据源 | 频率 | 说明 |
|---|---|---|
| `daily_adj.parquet` | 日频 | 复权日线 OHLCV |
| `daily.parquet` | 日频 | 未复权日线 |
| `finance.parquet` | 日频 | 日频财务/市场数据（pb, pe_ttm, turnover_rate, dividend_yield 等） |
| `ths_daily.parquet` | 日频 | 同花顺板块指数日频 ⚠️ 注意：THS 板块→个股映射数据缺失，该数据源当前**不可用** |
| `calendar.parquet` | 日频 | 交易日历 |
| `main_fund_flow.parquet` | 日频 | 主力资金流向 |
| `margin_detail.parquet` | 日频 | 融资融券明细 |
| `cyq_perf.parquet` | 日频 | 筹码分布 |

| 禁止依赖 | 原因 |
|---|---|
| `stock_list.parquet` **单独使用** | 静态属性，无日频变动 |
| `ths_constituent_stocks.parquet` | 映射数据缺失（type="I" 行业板块无数据） |
| 仅靠静态属性 broadcast 到每日 | 产出常量因子，无截面信息 |

### 1.2 时间编码型因子（例外）

日历效应类因子（如 `month_start_effect`、`quarter_end_proximity`、`day_of_week_effect`）虽然截面方差极低，但使用日频 calendar 数据，属于**时间编码**，合格保留。

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
    dependencies=("daily_adj.parquet",),  # 必须声明所有数据源
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

- [ ] 依赖的数据源是日频的（daily_adj / finance / calendar / 或同级别）
- [ ] 不依赖 stock_list.parquet 作为唯一数据源
- [ ] 不依赖 ths_daily.parquet（当前 THS 板块→个股映射不可用）
- [ ] 不使用 `rolling().apply(lambda, raw=False)` 对宽 DataFrame
- [ ] 不使用日期 × 日期双重循环
- [ ] 使用 `stack_date_code()`、`safe_divide()`、`np.asarray()` 等 pandas 3.0 兼容工具
- [ ] 预期 NaN 覆盖率 > 80%（即缺失率 < 20%）
- [ ] 预期单因子 rebuild 耗时 < 60s

---

## 七、数据源状态

| 数据源 | 状态 | 说明 |
|---|---|---|
| `daily_adj.parquet` | ✅ 可用 | 核心数据源 |
| `daily.parquet` | ✅ 可用 | |
| `finance.parquet` | ✅ 可用 | pb/pe/turnover_rate/dividend 等日频字段 |
| `calendar.parquet` | ✅ 可用 | |
| `main_fund_flow.parquet` | ✅ 可用 | 大单/中单/小单分类 |
| `margin_detail.parquet` | ⚠️ 部分覆盖 | 仅约 85% 股票有融资融券，做因子需接受高 NaN |
| `short_selling.parquet` | ⚠️ 部分覆盖 | 仅约 85% 股票有融券，做因子需接受高 NaN |
| `ths_daily.parquet` | ❌ 不可用 | THS 板块→个股映射缺失，所有 sector_* 因子产出 100% NaN |
| `ths_constituent_stocks.parquet` | ❌ 不可用 | type="I" 行业板块无匹配数据 |
| `stock_list.parquet` | ⚠️ 辅助 | 仅作辅助（如行业名称补充），不可作为唯一数据源 |

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
