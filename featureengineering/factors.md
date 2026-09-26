# 灵启因子库文档（featureengineering）

> 本文档由因子注册表（`FACTOR_REGISTRY`）与因子源码自动提取生成，
> 每个因子为一个独立小节，包含**定义**、**公式（计算逻辑）**、**意义**三项。
> 生成时间：2026-09-13 ｜ 数据构建截至：20260911。

## 目录

- [一、总览](#一总览)
- [二、因子 Class 体系](#二因子-class-体系)
- [三、通用计算约定](#三通用计算约定)
- [四、数据源一览](#四数据源一览)
- [五、因子详解](#五因子详解)
  - [Class 1 — Panel 面板类（466）](#class-1)
    - [价格 / 量价（201）](#cat-price-c1)
    - [成交量（3）](#cat-volume-c1)
    - [估值（40）](#cat-valuation-c1)
    - [资金流（97）](#cat-fund_flow-c1)
    - [日内 / 微观结构（11）](#cat-intraday-c1)
    - [行业 / 板块（15）](#cat-sector-c1)
    - [事件（22）](#cat-event-c1)
    - [风险（38）](#cat-risk-c1)
    - [时间序列（24）](#cat-timeseries-c1)
    - [因子耦合（8）](#cat-coupling-c1)
    - [中性化（2）](#cat-neutral-c1)
    - [标签（Label）（5）](#cat-target-c1)
  - [Class 2 — cyq_chips 筹码类（39）](#class-2)
    - [价格 / 量价（39）](#cat-price-c2)
  - [Class 3 — history_1min 分钟行情类（114）](#class-3)
    - [日内 / 微观结构（114）](#cat-intraday-c3)
  - [Class 4 — indicator_1min 分钟指标类（87）](#class-4)
    - [日内 / 微观结构（87）](#cat-intraday-c4)
  - [Class 5 — 因子耦合类（147）](#class-5)
    - [风险（28）](#cat-risk-c5)
    - [因子耦合（117）](#cat-coupling-c5)
    - [增强（2）](#cat-enhanced-c5)
- [六、已禁用因子（非 PIT）](#六已禁用因子非-pit)
- [附录 A：全因子速查索引（字母序）](#附录-a全因子速查索引字母序)

## 一、总览

- **注册因子总数**：859 个（启用 853 / 禁用 6）
- **非标签因子**：848 个，均已生成 `.fea` 数据文件（`data/factors/`）
- **标签**：5 个（`data/targets/`），仅作监督学习目标，**严禁作为选股特征使用**
- **禁用因子**：6 个，因依赖非 point-in-time 数据源（`stock_list.parquet`）在加载时被自动移除，详见[第六章](#六已禁用因子非-pit)

按 Class 分布：

| Class | 类型 | 因子数 |
|---|---|---|
| 1 | Panel 面板类 | 466 |
| 2 | cyq_chips 筹码类 | 39 |
| 3 | history_1min 分钟行情类 | 114 |
| 4 | indicator_1min 分钟指标类 | 87 |
| 5 | 因子耦合类 | 147 |

按类别（category）分布：

| 类别 | 因子数 |
|---|---|
| price（价格 / 量价） | 240 |
| intraday（日内 / 微观结构） | 212 |
| coupling（因子耦合） | 125 |
| fund_flow（资金流） | 97 |
| risk（风险） | 66 |
| valuation（估值） | 40 |
| timeseries（时间序列） | 24 |
| event（事件） | 22 |
| sector（行业 / 板块） | 15 |
| target（标签（Label）） | 5 |
| volume（成交量） | 3 |
| neutral（中性化） | 2 |
| enhanced（增强） | 2 |

## 二、因子 Class 体系

Class 由因子的**数据依赖**决定（`classify_factor` 规则）：

| Class | 判定依据 | 说明 |
|---|---|---|
| 1 — Panel | 其余依赖 | 从单个 `.parquet` 面板文件加载，全向量化 |
| 2 — cyq_chips | 依赖 `cyq_chips` | 逐股票筹码分布数据 |
| 3 — history_1min | 依赖 `history_1min` | 逐股票 1 分钟历史行情 |
| 4 — indicator_1min | 依赖 `indicator_1min` | 逐股票分钟级技术指标 |
| 5 — Coupling | 依赖 `__factors__` | 加载已有因子 `.fea`，因子间耦合 |

判定优先级：`history_1min` → `cyq_chips` → `indicator_1min` → `__factors__` → 默认 Class 1。

## 三、通用计算约定

1. **截面排名 `cross_sectional_rank`**：绝大多数因子最终输出为**每日截面内百分位排名**
   （`rank(pct=True)`，默认先做 1%/99% 分位 winsorize 截尾）。排名值域 [0, 1]，
   值越大表示该股当日在此因子上的截面位置越靠前。方向语义见各因子的“定义”：
   描述中含“低振幅排前”等字样时，表示因子对原始值取负后再排名。
2. **因子方向**：因子值本身是排序后的相对位置，不做归一化到固定分布；
   耦合类（Class 5）因子对多个已有因子直接相乘/加权后再排名。
3. **数据对齐**：所有因子按 (Date, Code) 双索引对齐；日内因子在日截面内聚合。
4. **PIT（point-in-time）契约**：因子只允许使用截至当日的已披露数据；
   依赖 `stock_list.parquet`（当前快照）的因子因违反该契约被禁用。
5. **标签语义**：`label_ret_Nd` 统一交易逻辑为 —— T 日收盘后计算信号 →
   T+1 日**开盘买入** → 持有 N 个交易日 → T+1+N 日**开盘卖出**，
   `label = open[t+1+N] / open[t+1] - 1`（复权开盘价），尾段因前视不足截断。

## 四、数据源一览

全部因子引用的上游数据依赖（含禁用因子）及其引用次数：

| 数据依赖 | 引用次数 | 说明 |
|---|---|---|
| `daily.parquet` | 380 | 日线行情（未复权） |
| `__factors__` | 147 | 已有因子 .fea 文件（Class 5 耦合） |
| `history_1min` | 114 | 1 分钟历史行情（逐股票） |
| `indicator_1min` | 87 | 分钟级技术指标（逐股票） |
| `main_fund_flow.parquet` | 76 |  |
| `finance.parquet` | 67 |  |
| `cyq_perf.parquet` | 44 |  |
| `momentum_20` | 40 |  |
| `cyq_chips` | 39 | 筹码分布（逐股票） |
| `bp` | 37 |  |
| `margin_detail.parquet` | 22 |  |
| `turnover_20` | 13 |  |
| `mf_net_inflow_ratio` | 12 |  |
| `momentum_60` | 11 |  |
| `parkinson_vol` | 9 |  |
| `short_term_reversal_5` | 9 |  |
| `amihud_intraday` | 6 |  |
| `margin_net_flow_ratio` | 6 |  |
| `stock_list.parquet` | 6 | 股票列表快照（非 PIT，被禁用） |
| `winner_rate` | 5 |  |
| `drawdown_60` | 5 |  |
| `daily_adj.parquet` | 5 | 日线行情（前复权） |
| `rv_5min` | 5 |  |
| `volume_momentum_5` | 4 |  |
| `rs_60` | 4 |  |
| `log_circ_mv` | 4 |  |
| `chip_cr3_factor` | 3 |  |
| `limit_up_event_5` | 3 |  |
| `idio_vol_60` | 3 |  |
| `volume_breakout_confirm_20` | 3 |  |
| `mf_big_order_ratio` | 2 |  |
| `price_to_52w_high` | 2 |  |
| `macd_trend_strength` | 2 |  |
| `margin_buy_pressure` | 2 |  |
| `sp_ttm` | 2 |  |
| `rsi_14_excess` | 2 |  |
| `chip_win_peak_frac` | 2 |  |
| `mf_net_inflow_5d` | 2 |  |
| `momentum_10` | 2 |  |
| `rv_60min` | 2 |  |
| `momentum_stability_20_60` | 2 |  |
| `boll_squeeze` | 1 |  |
| `chip_median_momentum` | 1 |  |
| `chip_support_strength` | 1 |  |
| `chip_peak_shift` | 1 |  |
| `big_order_net_accel_10` | 1 |  |
| `new_high_60_event` | 1 |  |
| `indicator_consensus` | 1 |  |
| `tail_volume_share` | 1 |  |
| `kdj_cross_net` | 1 |  |
| `lhb_proxy_score_60` | 1 |  |
| `limit_up_fade_10` | 1 |  |
| `downside_frequency_60` | 1 |  |
| `ma_alignment_score` | 1 |  |
| `margin_chip_cost_gap` | 1 |  |
| `min_ma_alignment_frac_20` | 1 |  |
| `net_turnover_rate_20` | 1 |  |
| `sortino_ratio_60` | 1 |  |
| `rsi_intraday_trend` | 1 |  |
| `log_total_mv` | 1 |  |
| `smart_money_share` | 1 |  |
| `mf_flow_stability_20d` | 1 |  |
| `rv_term_structure_slope` | 1 |  |
| `vp_expand_down_am_share` | 1 |  |
| `vp_consistency_20` | 1 |  |
| `vp_expand_up_share` | 1 |  |
| `vp_expand_price_pos` | 1 |  |
| `minute_ret_vol_corr` | 1 |  |
| `vp_shrink_down_share` | 1 |  |
| `drawdown_120` | 1 |  |
| `beta_60` | 1 |  |
| `ulcer_index_20` | 1 |  |
| `dv_ttm_rank` | 1 |  |
| `corr_market_60` | 1 |  |
| `mf_small_order_ratio` | 1 |  |
| `amount_surge_count_20` | 1 |  |
| `amplitude_20` | 1 |  |
| `gk_vol` | 1 |  |
| `high_low_volatility_20` | 1 |  |
| `hl_range_intraday` | 1 |  |
| `intraday_high_low_volatility` | 1 |  |
| `ret_std_intraday` | 1 |  |
| `rv_10min` | 1 |  |
| `rv_15min` | 1 |  |
| `rv_30min` | 1 |  |
| `rv_rolling_5d_std` | 1 |  |
| `kama_efficiency_20` | 1 |  |
| `momentum_5` | 1 |  |
| `williams_r_14` | 1 |  |
| `smart_money_net_bias` | 1 |  |
| `am_hl_range_intraday` | 1 |  |
| `max_ret_intraday` | 1 |  |
| `relative_spread` | 1 |  |
| `turnover_std_20` | 1 |  |
| `turnover_vol_20` | 1 |  |
| `realized_spread_5min` | 1 |  |
| `turnover_f_20` | 1 |  |
| `volume_rv_ratio` | 1 |  |

## 五、因子详解

共 853 个启用因子，按 Class → 类别 → 因子名字母序组织。

### <a name="class-1"></a>Class 1 — Panel 面板类（466 个）

从单个 `.parquet` 面板文件加载数据（如 `daily.parquet`、`daily_adj.parquet` 等），全向量化计算。

#### <a name="cat-price-c1"></a>类别 price — 价格 / 量价（201 个）

##### accumulation_distribution_20

**定义**：量积累线斜率因子：AD线20日变化截面排名（资金净积累排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
high = daily["high"]
low = daily["low"]
close = daily["close"]
hl = (high - low).replace(0, np.nan)
clv = ((close - low) - (high - close)) / hl
ad = (clv * daily["vol"]).groupby(level="Code").cumsum()
slope = ad.groupby(level="Code").diff(20)
avg_vol = daily["vol"].groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
norm = safe_divide(slope, avg_vol + 1e-10)
return cross_sectional_rank(norm)
```

**意义**：Accumulation/Distribution线=Σ CLV×vol，CLV=(收盘位置)衡量每根K线的资金进出——AD线20日变化=近一月资金净积累方向。CLV用当日高低区间比值(除权日局部失真可接受)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

##### adx_14

**定义**：14日趋势强度因子 (ADX)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
di_plus, di_minus = _directional_movement(daily["high"], daily["low"], daily["pre_close"], scale, 14)
dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-8)
adx = dx.groupby(level="Code").transform(
    lambda s: s.rolling(14, min_periods=7).mean()
)
return cross_sectional_rank(adx)
```

**意义**：ADX衡量趋势强度(非方向)，趋势明确的股票动量策略更有效

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### amihud_amt_20d

**定义**：Amihud 非流动性（|日收益|/成交额×1e8 的20日均值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["amihud"] = df["absret"] / df["amount"].replace(0, np.nan) * 1e8
return _out(df, "amihud_amt_20d", roll(df, "amihud", 20, "mean", min_periods=5))
```

**意义**：单位成交额推动的价格变动越大=流动性越差，非流动性溢价在 A 股显著，高 amihud 小盘股未来收益更高。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### amihud_daily_20

**定义**：Amihud日频非流动性因子（20日平均|收益|/成交额，正向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret_abs = daily["pct_chg"].abs() / 100.0
amihud = safe_divide(ret_abs, daily["amount"])
amihud20 = amihud.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(amihud20)
```

**意义**：Amihud 非流动性=单位成交额对应的价格冲击，是流动性的经典度量。高非流动性股票存在流动性溢价补偿，且往往被市场忽视。与日内 amihud_intraday（分钟级口径）互补——此为日频标准口径。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/liquidity_factors.py`

##### amihud_daily_5

**定义**：Amihud 非流动性（|日收益|/成交额×1e8 的 5 日均值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["amihud"] = df["absret"] / df["amount"].replace(0, np.nan) * 1e8
return _out(df, "amihud_daily_5", roll(df, "amihud", 5, "mean", min_periods=2))
```

**意义**：amihud_daily_20 的短周期版：5 日非流动性对资金流入流出更敏感，短期流动性冲击与 1d 收益反转相关。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### amount_ratio_20

**定义**：20日相对成交额因子，amount/avg_amount_20 - 1 截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
amount = daily_panel["amount"]
avg_amount = amount.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
a_ratio = amount / avg_amount.replace(0, np.nan) - 1.0
return cross_sectional_rank(-a_ratio)
```

**意义**：成交额比成交量更能反映资金参与度，异常放量常伴随趋势转折。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### amount_surge_count_20

**定义**：资金脉冲频率：20日内成交额>1.5倍20日均额的交易日占比截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
amount = daily["amount"]
amt_ma20 = amount.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
surge = (amount > 1.5 * amt_ma20).astype(float)
frac = surge.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(frac)
```

**意义**：成交额异常放大(>1.5×均额)的频率衡量资金进出的脉冲性：频繁脉冲=有大资金在反复进出，关注度与博弈程度高(情绪票特征)；无脉冲=交投冷清、被市场遗忘。频率而非幅度，避免被单日巨量主导。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_coupling.py`

##### amplitude_20

**定义**：20日均振幅因子，(high-low)/close 的20日均值截面排名（低振幅排前）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
amp = (daily_panel["high"] - daily_panel["low"]) / daily_panel["close"].replace(0, np.nan)
avg_amp = amp.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-avg_amp)
```

**意义**：振幅是流动性与不确定性的综合指标，低振幅反映筹码稳定性。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### amt_rel_ind

**定义**：行业内相对量能（amount 行业内 z-score）。

**公式（计算逻辑）**：

```python
df = _daily(context)
vals = (df["amount"] - ind_mean(df, "amount")) / (ind_std(df, "amount") + 1e-8)
return _out(df, "amt_rel_ind", vals)
```

**意义**：行业内成交额异常放大=资金关注度提升，z-score 剥离板块整体交投水平。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### amt_rel_ind_ma5

**定义**：量能相对自身均值（amount/个股全期均值 的5日均值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
vals = df.groupby("Code")["amount"].transform(
    lambda s: (s / s.mean()).rolling(5, min_periods=1).mean()
)
return _out(df, "amt_rel_ind_ma5", vals)
```

**意义**：成交额相对自身历史水平的短期抬升，反映个股层面的量能异动（注：沿袭原脚本语义，实为个股自身归一，非行业相对）。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### aroon_down_25

**定义**：Aroon下行因子：25日窗口内距最近新低的位置取负排名（近期破位排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _adjusted_close(daily).unstack("Code")
days_since = _rolling_extreme_age(wide, window=25, find_max=False)
aroon = safe_divide(25.0 - days_since, 25.0) * 100.0
return cross_sectional_rank(-stack_date_code(aroon))
```

**意义**：Aroon下行衡量距离上次创新低的时间——越接近新低=趋势越弱。与aroon_up_25互补，反映空头主导程度。基于复权基座，除权日不产生假新低。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### aroon_up_25

**定义**：Aroon上行因子：25日窗口内距最近新高的天数位置截面排名（强势突破排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _adjusted_close(daily).unstack("Code")
days_since = _rolling_extreme_age(wide, window=25, find_max=True)
aroon = safe_divide(25.0 - days_since, 25.0) * 100.0
return cross_sectional_rank(stack_date_code(aroon))
```

**意义**：Aroon(阿隆)衡量距离上次创新高的时间——越接近新高=趋势越强。基于复权基座判断新高(除权日不产生假突破)，25日窗口为经典参数。新高持续出现=多头主导。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### atr_20

**定义**：20日平均真实波幅(ATR)因子，低ATR排前。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
high = daily["high"]
low = daily["low"]
pre_close = daily["pre_close"]

# True Range = max(high-low, |high-prev_close|, |low-prev_close|)
# Row-wise operation — no per-stock groupby needed
tr = np.maximum(
    high - low,
    np.maximum((high - pre_close).abs(), (low - pre_close).abs()),
)

atr = tr.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-atr)
```

**意义**：低ATR股票波动平稳、筹码稳定，高ATR意味着剧烈波动风险，低波异象支持低ATR溢价。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### avg_cost_premium

**定义**：平均成本溢价因子，(close-weight_avg)/weight_avg截面排名（现价高于均价=多数人盈利排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
daily = context.load("daily.parquet")
# cyq 成本价为复权口径,把 close 折算到同一空间后再比较(见 chip._close_adj_basis)
close_adj = _close_adj_basis(daily)
weight_avg = cyq["weight_avg"]
common = weight_avg.index.intersection(close_adj.index)
premium = safe_divide(
    close_adj.loc[common] - weight_avg.loc[common],
    weight_avg.loc[common] + 1e-10
)
premium = premium.clip(-1, 5)
return cross_sectional_rank(premium)
```

**意义**：Premium of current price over volume-weighted average cost. Positive = average holder in profit = bullish. Negative = average holder underwater = selling pressure on rallies.

**依赖数据**：`cyq_perf.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_cost_extended.py`

##### avg_price_trend_20

**定义**：均价趋势因子：当日VWAP的20日变化率截面排名（成交均价抬升排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vwap = safe_divide(daily["amount"], daily["vol"])
# VWAP 为当日实际成交均价(未复权),跨日 pct_change 在除权日会产生伪位移;
# 乘 scale=adj/close 折算到复权空间后再比较(2026-08-05 修复)
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
trend = (vwap * scale).groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
return cross_sectional_rank(trend)
```

**意义**：成交均价(amount/vol)的20日变化率衡量持仓成本的迁移方向——均价持续抬升=增量资金以更高价位进场(换手充分、筹码上移)，均价下移=筹码下移(套牢加深)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

##### bias_20

**定义**：20日均线乖离率，close/ma_20 - 1 的截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
adj = _adjusted_close(daily_panel)
ma_20 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
bias = adj / ma_20.replace(0, np.nan) - 1.0
return cross_sectional_rank(bias)
```

**意义**：均线乖离反映价格对中期成本的偏离程度，极端乖离预示均值回归。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### bias_60

**定义**：60日乖离率因子：(adj-MA60)/MA60截面排名（偏离中期均线排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
ma60 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
bias = safe_divide(adj - ma60, ma60)
return cross_sectional_rank(bias)
```

**意义**：乖离率衡量价格对中期均线的偏离程度——策略21用乖离及其均线金叉做择时。60日乖离极端正=短期超买(均值回归风险)，但中期强势延续性也强；与20日乖离(bias_20)互补，捕捉更慢的回归周期。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### bias_signal_29_19

**定义**：BIAS金叉信号因子：29日乖离率减去其19日均线截面排名（乖离加速扩张排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
g = adj.groupby(level="Code")
ma29 = g.transform(lambda s: s.rolling(29, min_periods=15).mean())
bias = safe_divide(adj - ma29, ma29)
bias_ma19 = bias.groupby(level="Code").transform(
    lambda s: s.rolling(19, min_periods=10).mean()
)
return cross_sectional_rank(bias - bias_ma19)
```

**意义**：策略21(BIAS_QL乖离率择时)用 N=29 日乖离与其 M=19 日均线的金叉/死叉判断乖离的趋势方向:乖离减去其均线>0=乖离仍在加速扩张(主升/超跌反弹进行中),<0=乖离回归(趋势衰竭)。与既有 bias_20/bias_60(乖离绝对值)互补:本因子是「乖离的加速度」维度。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### big_range_day_freq_20

**定义**：大振幅日频率因子：20日振幅(复权)≥5%的天数占比截面排名（剧烈波动排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 同日比较:(high−low) 与当日 pre_close 同尺度,无需复权折算;
# pre_close 在除权日已调整到当日价格尺度,振幅口径与涨跌停判定一致。
amp = safe_divide(daily["high"] - daily["low"], daily["pre_close"]) * 100.0
is_big = amp.ge(5.0).astype(float)
freq = _roll_mean(is_big, 20, 5)
return cross_sectional_rank(freq)
```

**意义**：单日振幅≥5%是多空激烈博弈的异动日——振幅日的频率衡量股票的波动性格(题材股高频、银行股低频)。高振幅频率=情绪驱动与信息冲击频繁,与波动率水平正交但捕捉其分布形态。高低点折算复权口径,除权日不失真。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### bollinger_position

**定义**：布林带位置因子，(close-下轨)/(上轨-下轨)截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
# 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染均线与带宽
adj = _adjusted_close(daily_panel)
ma_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
std_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
upper = ma_20 + 2 * std_20
lower = ma_20 - 2 * std_20
position = (adj - lower) / (upper - lower).replace(0, np.nan)
return cross_sectional_rank(position)
```

**意义**：布林带位置衡量价格在波动区间中的相对位置——接近上轨=强势/超买，接近下轨=弱势/超卖。与传统布林带%B等价，是均值回复策略的基础信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### bollinger_position_20

**定义**：20日布林带位置因子 (%B, 高位排后, 负向)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
# 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染均线与带宽
adj = _adjusted_close(daily)
ma = rolling_group_mean(adj, 20)
std = rolling_group_std(adj, 20)
b_upper = ma + 2 * std
b_lower = ma - 2 * std
pct_b = (adj - b_lower) / (b_upper - b_lower + 1e-8)
return cross_sectional_rank(-pct_b)  # lower position = more mean-reversion upside
```

**意义**：%B>1意味着突破上轨(短期超买)，<0意味着突破下轨(超卖)，均值回归视角

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### bollinger_squeeze

**定义**：布林带收缩因子，-(上轨-下轨)/均价截面排名（带宽窄=挤压突破前兆排前）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
# 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染带宽
adj = _adjusted_close(daily_panel)
ma_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
std_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
bandwidth = 4 * std_20 / ma_20.replace(0, np.nan)
# Rank negative: narrow band = squeeze = ranked high
return cross_sectional_rank(-bandwidth)
```

**意义**：布林带收缩(带宽变小)是波动率压缩的信号——低波动后往往伴随着剧烈的方向性突破。带宽处于历史低位时预示着即将出现趋势性行情。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### bollinger_width_20

**定义**：20日布林带宽度因子 (窄幅排前, 负向)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
# 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染带宽
adj = _adjusted_close(daily)
ma = rolling_group_mean(adj, 20)
std = rolling_group_std(adj, 20)
width = safe_divide(4 * std, ma)
return cross_sectional_rank(-width)  # narrow bandwidth = potential breakout
```

**意义**：布林带宽缩减(bandwidth squeeze)预示突破即将到来，窄幅是低波动蓄力

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### breakout_60

**定义**：60日价格突破强度因子，close/max(high,60)-1截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# Raw high has no point-in-time adjusted counterpart; use the adjusted
# close base's rolling maximum (George-Hwang style approximation), which
# is immune to ex-dividend jumps in the 60-day window.
adj = _adjusted_close(daily)
previous = adj.groupby(level="Code").shift(1)
adj_max = previous.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).max()
)
breakout = adj / adj_max.replace(0, np.nan) - 1.0
return cross_sectional_rank(breakout)
```

**意义**：价格突破近期高点反映上涨动能强劲，突破强度越高趋势延续性越强。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### cci_20

**定义**：20日CCI因子：(TP−MA20)/(0.015×MD)截面排名（超买排后，负向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 20日均值/平均偏离为跨日窗口,TP 需走复权口径(scale=adj/close 折算 high/low)
adj = _adjusted_close(daily)
scale = adj / daily["close"].replace(0, np.nan)
tp = (daily["high"] * scale + daily["low"] * scale + adj) / 3.0
tp_w = tp.unstack("Code")
ma = tp_w.rolling(20, min_periods=10).mean()
md = tp_w.sub(ma).abs().rolling(20, min_periods=10).mean()
cci = safe_divide(tp_w - ma, 0.015 * md + 1e-10)
return cross_sectional_rank(-stack_date_code(cci))
```

**意义**：CCI(顺势指标)衡量典型价格对20日均价的标准化偏离——>100超买、<−100超卖。策略51的技术指标开关含CCI。极端正CCI=短期涨幅透支，负向排名做均值回归信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### chandelier_position

**定义**：吊灯止损距离因子：现价距(20日高点−3×ATR)止损线的ATR单位数截面排名（趋势健康度排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
scale = adj / daily["close"].replace(0, np.nan)
h_adj = daily["high"] * scale
l_adj = daily["low"] * scale
prev_adj = adj.groupby(level="Code").shift(1)
tr = pd.Series(
    np.maximum.reduce([h_adj - l_adj, (h_adj - prev_adj).abs(), (l_adj - prev_adj).abs()]),
    index=h_adj.index,
)
atr20 = tr.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
high20 = h_adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).max()
)
stop = high20 - 3.0 * atr20
pos = safe_divide(adj - stop, atr20)
return cross_sectional_rank(pos)
```

**意义**：海龟系变体(策略01用4×ATR吊灯止损):止损线=20日最高价−3×ATR,价格距止损线的 ATR 单位数度量趋势的「余量」——距离远=趋势完整、回调尚未触及风控位;距离近=趋势濒临破坏。ATR 归一使高波与低波趋势股可比。全部在复权空间计算(high 折算、TR 用复权基座差分),无除权污染。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### chip_concentration

**定义**：筹码集中度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（取负向=窄区间排前）。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
spread = (perf["cost_95pct"] - perf["cost_5pct"]) / perf["cost_50pct"].replace(0, np.nan)
return cross_sectional_rank(-spread)
```

**意义**：筹码分布区间窄=筹码密集，突破后趋势性强。密集区间突破方向的持续性优于分散区间。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_concentration_change_20d

**定义**：筹码集中度20日变化因子，-(cost_95pct-cost_5pct)/cost_50pct的20日变化截面排名（凝聚=正向排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
concentration = -(cyq["cost_95pct"] - cyq["cost_5pct"]) / cyq["cost_50pct"].replace(0, np.nan)
chg = concentration.groupby(level="Code").transform(lambda s: s.diff(20))
return cross_sectional_rank(chg)
```

**意义**：筹码集中度的边际变化比绝对集中度更重要——筹码正在凝聚(区间收窄)意味着主力正在控制筹码，后续突破概率增大。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_deep.py`

##### chip_concentration_change_5d

**定义**：筹码集中度5日变化因子，concentration的5日差分截面排名。集中度收窄=吸筹，放宽=派发。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
conc = (perf["cost_95pct"] - perf["cost_5pct"]) / perf["cost_50pct"].replace(0, np.nan)
conc_chg = conc.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(conc_chg)
```

**意义**：筹码集中度的变化方向揭示了主力资金的动向：集中度快速收窄（区间收缩）=资金吸筹，集中度放宽（区间扩散）=派发或恐慌。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_concentration_ma5

**定义**：筹码成本宽度5日均值因子：(cost95−cost5)/cost50取负排名，区间越窄（越集中）排前。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
concentration = safe_divide(
    perf["cost_95pct"] - perf["cost_5pct"],
    perf["cost_50pct"] + 1e-8
)
ma5 = rolling_group_mean(concentration, 5)
return cross_sectional_rank(-ma5)
```

**意义**：筹码集中度均值化后稳定性提升，高集中度意味着筹码被少数人掌控=拉升易但出货难

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_concentration_streak

**定义**：筹码凝聚持续性因子，筹码集中度连续改善天数截面排名。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
concentration = -(cyq["cost_95pct"] - cyq["cost_5pct"]) / cyq["cost_50pct"].replace(0, np.nan)
improving = (concentration.groupby(level="Code").transform(lambda s: s.diff(1)) > 0).astype(int)

def _count_streak(x):
    x = x.values
    streak = np.zeros_like(x, dtype=float)
    cnt = 0
    for i in range(len(x)):
        if x[i] == 1:
            cnt += 1
        else:
            cnt = 0
        streak[i] = cnt
    return streak

streak = improving.groupby(level="Code").transform(_count_streak)
return cross_sectional_rank(streak)
```

**意义**：筹码连续凝聚是主力收集筹码过程的表现——凝聚趋势的持续性比单日凝聚程度更重要，连凝天数多=主力持续在收集。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_deep.py`

##### chip_cost_asymmetry

**定义**：成本分布不对称因子，(cost_95pct-cost_50pct)/(cost_50pct-cost_5pct)截面排名（取负向=上重下轻排后）。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
upper = (perf["cost_95pct"] - perf["cost_50pct"]).replace(0, np.nan)
lower = (perf["cost_50pct"] - perf["cost_5pct"]).replace(0, np.nan)
asym = upper / lower.replace(0, np.nan)
return cross_sectional_rank(-asym)
```

**意义**：成本分布不对称度>1意味着上方套牢盘筹码量多于下方获利盘，上方抛压更重，上行阻力大。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_cost_convergence_20d

**定义**：筹码成本收敛=(cost_85pct-cost_5pct)/cost_50pct的20日变化取反。成本收敛=方向选择在即。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
width = safe_divide(cyq["cost_85pct"] - cyq["cost_5pct"], cyq["cost_50pct"])
chg = width.groupby(level="Code").transform(lambda s: s.diff(20))
return cross_sectional_rank(-chg)
```

**意义**：筹码分布宽度(cost_85pct-cost_5pct的相对宽度)的20日变化。宽度收敛=筹码在集中,多空成本趋于一致——通常是大行情前的蓄力阶段;宽度发散=筹码在分散,多空分歧加大。收敛后往往出现方向性突破。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### chip_cost_kurtosis_20d

**定义**：成本分布尖峰度因子 (85%-15%价差/95%-5%价差, 低值=分布尖峰排前)。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
# Approximate kurtosis proxy using cost percentiles
spread_90_10 = perf["cost_95pct"] - perf["cost_5pct"]
spread_70_30 = perf["cost_85pct"] - perf["cost_15pct"]
kurtosis_proxy = safe_divide(spread_70_30, spread_90_10 + 1e-8)
# Lower ratio = more peaked distribution (tighter middle relative to tails)
return cross_sectional_rank(-kurtosis_proxy)
```

**意义**：成本分布的尖峰形态意味着筹码高度集中在窄区间，支撑/阻力更明确。2026-08-05 命名修正:实现为当日截面代理(非20日滚动),'20d'后缀保留仅为兼容既有 .fea 文件名。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_cost_premium_change_20

**定义**：成本溢价动量：(close-cost_50pct)/cost_50pct的20日变化截面排名。溢价率抬升=资金持续高于成本线买入。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
d = context.load("daily.parquet")
close_adj = _close_adj_basis(d)
cost50 = cyq["cost_50pct"]
premium = safe_divide(close_adj - cost50, cost50)
chg = premium.groupby(level="Code").transform(lambda s: s.diff(20))
return cross_sectional_rank(chg)
```

**意义**：cost_support_strength 是现价相对50%成本位的溢价水平，本因子捕捉其20日变化——溢价率持续抬升=资金不断以高于筹码成本的价格承接，成本线正在被夯实为支撑；溢价率走低=价格向成本线靠拢，支撑在失守。与 chip_cost_momentum_20d 的weight_avg 口径互补(本因子用成本中位数位)。

**依赖数据**：`cyq_perf.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### chip_cost_skew

**定义**：成本分布偏度因子，(cost_50pct-cost_15pct)/(cost_85pct-cost_50pct)截面排名。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
lower = (perf["cost_50pct"] - perf["cost_15pct"]).replace(0, np.nan)
upper = (perf["cost_85pct"] - perf["cost_50pct"]).replace(0, np.nan)
skew = lower / upper.replace(0, np.nan)
return cross_sectional_rank(skew)
```

**意义**：偏度>1=上方套牢盘重于下方获利盘（负向），偏度<1=下方支撑强于上方压力（正向）。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_position

**定义**：筹码位置因子，(close-cost_5pct)/(cost_95pct-cost_5pct)截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
perf = context.load("cyq_perf.parquet")
# cyq 成本价为复权口径,把 close 折算到同一空间后再比较(见 _close_adj_basis)
close_adj = _close_adj_basis(daily_panel)
lo = perf["cost_5pct"]
hi = perf["cost_95pct"]
common = close_adj.index.intersection(lo.index).intersection(hi.index)
position = (close_adj.loc[common] - lo.loc[common]) / (hi.loc[common] - lo.loc[common]).replace(0, np.nan)
return cross_sectional_rank(-position)
```

**意义**：当前价在筹码分布中的相对位置反映获利盘大小，高位=接近套牢区上沿，上行阻力增大。

**依赖数据**：`daily.parquet`、`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_profit_loss_ratio

**定义**：盈亏筹码比因子，(close-cost_95pct)/(cost_5pct-close)截面排名。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
daily_panel = context.load("daily.parquet")
# cyq 成本价为复权口径,把 close 折算到同一空间后再比较(见 chip._close_adj_basis)
close_adj = _close_adj_basis(daily_panel)
common = close_adj.index.intersection(cyq["cost_95pct"].index)
upper = close_adj.loc[common] - cyq["cost_95pct"].loc[common]
lower = cyq["cost_5pct"].loc[common] - close_adj.loc[common]
ratio = lower / (upper.abs() + 0.01)
return cross_sectional_rank(ratio)
```

**意义**：上方筹码(套牢盘)与下方筹码(获利盘)的相对比例——比率高=上方套牢盘重(负向)、上升阻力大；比率低=下方获利盘多(正向)、有支撑。

**依赖数据**：`cyq_perf.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_deep.py`

##### chip_range_normalized

**定义**：归一化筹码区间因子，(cost_85pct-cost_15pct)/cost_50pct截面排名（取负向=窄区间排前）。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
spread = (perf["cost_85pct"] - perf["cost_15pct"]) / perf["cost_50pct"].replace(0, np.nan)
return cross_sectional_rank(-spread)
```

**意义**：归一化后的筹码分布区间宽度反映了筹码的相对密集程度。区间窄=筹码密集，突破后趋势性强；区间宽=筹码分散，有效性降低。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_resistance_distance

**定义**：筹码阻力距离因子，cost_85pct/close-1截面排名（取负向=接近阻力排后）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
perf = context.load("cyq_perf.parquet")
close_adj = _close_adj_basis(daily_panel)
resistance = perf["cost_85pct"]
common = close_adj.index.intersection(resistance.index)
result = resistance.loc[common] / close_adj.loc[common].replace(0, np.nan) - 1
return cross_sectional_rank(-result)
```

**意义**：收盘价接近85%分位成本线意味着上方套牢盘压力近在咫尺，短期上行阻力增大，存在回落风险。

**依赖数据**：`daily.parquet`、`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_support_distance

**定义**：筹码支撑距离因子，close/cost_15pct-1截面排名。距离支撑位越近=反弹潜力越大。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
perf = context.load("cyq_perf.parquet")
close_adj = _close_adj_basis(daily_panel)
support = perf["cost_15pct"]
common = close_adj.index.intersection(support.index)
result = close_adj.loc[common] / support.loc[common].replace(0, np.nan) - 1
return cross_sectional_rank(result)
```

**意义**：收盘价接近15%分位成本线意味着当前价格处于筹码密集支撑区附近，技术性反弹概率增大。

**依赖数据**：`daily.parquet`、`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_support_strength_20d

**定义**：20日筹码支撑强度因子 (成本密集区数量/支撑距离)。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
# Support strength: how concentrated + how close the cost distribution is below current price
cost_15 = perf["cost_15pct"]
cost_50 = perf["cost_50pct"]
cost_85 = perf["cost_85pct"]
# Lower cost_50 relative to cost_85 = support band is narrow and close
support = safe_divide(cost_50 - cost_15, cost_85 - cost_50 + 1e-8)
# Normalize: more mass below = stronger support
return cross_sectional_rank(support)
```

**意义**：筹码密集区离当前价格越近且越集中，支撑越强，下跌空间越小

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_weighted_cost_momentum_5d

**定义**：加权成本5日动量因子，(weight_avg-weight_avg.shift(5))/weight_avg截面排名。衡量平均成本迁移速度。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
wavg = perf["weight_avg"]
mom = wavg.groupby(level="Code").transform(lambda s: s.diff(5)) / wavg.replace(0, np.nan)
return cross_sectional_rank(mom)
```

**意义**：加权平均成本的快速变化反映了筹码在投资者之间的快速转移。成本快速上移=增量资金入场，成本快速下移=恐慌性抛售。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_weighted_cost_volatility_20d

**定义**：20日加权成本波动率因子 (成本稳定排前, 负向)。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
wa = perf["weight_avg"]
wa_std = wa.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
wa_mean = wa.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
cv = safe_divide(wa_std, wa_mean + 1e-8)
return cross_sectional_rank(-cv)
```

**意义**：加权成本波动大意味着筹码大幅换手、持仓成本不稳定，持仓者面临更大不确定性

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_winner_rate_acceleration

**定义**：获利盘比例加速度因子，winner_rate_change_5d的5日差分（二阶导数）截面排名（取负向=加速获利排后）。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
wr = perf["winner_rate"]
wr_chg = wr.groupby(level="Code").transform(lambda s: s.diff(5))
accel = wr_chg.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(-accel)
```

**意义**：获利盘比例增长的速度本身在加快=上涨进入加速阶段，这通常是短期赶顶的强烈信号。正加速度意味着越来越多人快速进入盈利状态。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_winner_rate_ma5

**定义**：获利盘比例5日均值因子 (获利盘多排后, 负向)。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
wr = perf["winner_rate"]
ma5 = rolling_group_mean(wr, 5)
return cross_sectional_rank(-ma5)
```

**意义**：获利盘比例均值平滑后过滤单日噪音，高获利盘=潜在抛压

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### chip_winner_rate_stability_20d

**定义**：获利盘比例20日稳定性因子，winner_rate的20日滚动标准差截面排名（取负向=高波动排后）。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
stable = perf["winner_rate"].groupby(level="Code").transform(lambda s: s.rolling(20).std())
return cross_sectional_rank(-stable)
```

**意义**：获利盘比例的剧烈波动意味着筹码结构不稳定，多空分歧大，趋势难以持续。稳定的获利盘比例是健康趋势的特征。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### close_location_20d

**定义**：收盘位置（20日收益 ÷ 20日振幅）。

**公式（计算逻辑）**：

```python
df = _daily(context)
h20 = roll(df, "high", 20, "max")
l20 = roll(df, "low", 20, "min")
chg20 = df.groupby("Code")["close"].shift(20)
vals = (df["close"] - chg20) / (h20 - l20 + 1e-8)
return _out(df, "close_location_20d", vals)
```

**意义**：同振幅下收益越高=上攻效率越高，刻画 20 日上涨质量而非单纯涨幅。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### close_location_5d

**定义**：5 日收盘位置（(close − 5日最低价)/(5日最高价 − 5日最低价)）。

**公式（计算逻辑）**：

```python
df = _daily(context)
lo = df.groupby("Code")["low"].transform(lambda s: s.rolling(5, min_periods=1).min())
hi = df.groupby("Code")["high"].transform(lambda s: s.rolling(5, min_periods=1).max())
vals = (df["close"] - lo) / (hi - lo + 1e-8)
return _out(df, "close_location_5d", vals)
```

**意义**：close_location_20d 的短周期版：5 日区间内的收盘位置刻画超短期供需平衡点，高位钝化/低位启动对次日收益有区分度。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### close_position_intraday_20

**定义**：收盘价日内位置因子，20日均(close-low)/(high-low)截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
denom = (daily_panel["high"] - daily_panel["low"]).replace(0, np.nan)
position = (daily_panel["close"] - daily_panel["low"]) / denom
avg_pos = position.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(avg_pos)
```

**意义**：收盘价在日内区间的相对位置揭示买卖压力——连续在区间高位收盘代表买盘主导（正向），在低位收盘代表卖盘主导。均线平滑后过滤单日噪音。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### close_position_ratio

**定义**：收盘价在日内高低点区间中的位置：(close-low)/(high-low)，强势收盘=排名高。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
range_hl = daily["high"] - daily["low"]
position = safe_divide(daily["close"] - daily["low"], range_hl)
position = position.clip(0, 1)
return cross_sectional_rank(position)
```

**意义**：收盘价在日内高低点区间中的相对位置。接近1=收于日内高点附近(强势收盘、多头主导)，接近0=收于低点附近(弱势收盘、空头主导)。该指标比单看涨跌幅更能反映日内多空博弈结果——同样的涨幅，收于高点vs收于低点代表完全不同的日内走势质量。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### close_position_vol_weighted_20

**定义**：量能加权收盘位置：20日均值(量比×日内收盘位置)截面排名。量能集中于高位收盘=强承接。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vol = daily["vol"]
pos = safe_divide(daily["close"] - daily["low"], daily["high"] - daily["low"])
vol_ma20 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
vol_norm = safe_divide(vol, vol_ma20)
vp = (vol_norm * pos).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(vp)
```

**意义**：日内收盘位置((close-low)/(high-low))反映当日多空结果，乘上相对量比后，量能大的日子权重更高——若放量日都收在日内高位，说明大资金在承接，后续看涨；放量日收在低位则是出货特征。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_coupling.py`

##### close_to_high_20d

**定义**：收盘距20日最高价距离（close/max20(high)−1）。

**公式（计算逻辑）**：

```python
df = _daily(context)
h20 = roll(df, "high", 20, "max")
return _out(df, "close_to_high_20d", df["close"] / h20 - 1)
```

**意义**：越接近 20 日高点=突破形态越完整，创新高动量在 A 股有正溢价。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### close_to_high_5d

**定义**：收盘相对 5 日最高价（close / 5日最高价）。

**公式（计算逻辑）**：

```python
df = _daily(context)
hi = df.groupby("Code")["high"].transform(lambda s: s.rolling(5, min_periods=1).max())
return _out(df, "close_to_high_5d", df["close"] / (hi + 1e-8))
```

**意义**：距离 5 日高点的位置区分「创短期新高」与「深度回调」状态；贴近新高后的次日行为（突破延续 vs 获利回吐）是 1d 有效信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### close_to_high_ratio_20

**定义**：20日均收盘/最高价比率因子 (收盘强势)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
ratio = (daily["close"] - daily["low"]) / (daily["high"] - daily["low"] + 1e-8)
avg_ratio = rolling_group_mean(ratio, 20)
return cross_sectional_rank(avg_ratio)
```

**意义**：收盘价接近最高价意味着持续的日内外买盘力量，收盘位置高预示次日强势

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### cost_convergence_signal

**定义**：成本收敛信号因子，(cost_95pct-cost_5pct)的20日变化率截面排名（分布收窄=筹码集中排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
width = cyq["cost_95pct"] - cyq["cost_5pct"]
chg = width.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
chg = chg.clip(-1, 1)
return cross_sectional_rank(-chg)
```

**意义**：Cost distribution convergence: narrowing = holders agree on value (accumulation). Widening = new holders at diverse prices (distribution). Rate of convergence is the second derivative.

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_cost_extended.py`

##### cost_displacement

**定义**：成本偏离因子，(close-weight_avg)/weight_avg截面排名（取负向=大幅偏离排后）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
perf = context.load("cyq_perf.parquet")
close_adj = _close_adj_basis(daily_panel)
wavg = perf["weight_avg"]
common = close_adj.index.intersection(wavg.index)
displacement = (close_adj.loc[common] - wavg.loc[common]) / wavg.loc[common].replace(0, np.nan)
return cross_sectional_rank(-displacement)
```

**意义**：现价偏离加权平均成本越大，获利回吐/抄底反弹的均值回归动力越强。

**依赖数据**：`daily.parquet`、`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### cost_displacement_extreme

**定义**：成本偏离极端度因子，(close-weight_avg)/weight_avg的绝对值截面排名（极度偏离=均值回归压力大排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
daily_panel = context.load("daily.parquet")
# cyq 成本价为复权口径,把 close 折算到同一空间后再比较(见 chip._close_adj_basis)
close_adj = _close_adj_basis(daily_panel)
cost_avg = cyq["weight_avg"]
common = close_adj.index.intersection(cost_avg.index)
displacement = (close_adj.loc[common] - cost_avg.loc[common]).abs() / cost_avg.loc[common].replace(0, np.nan)
# Positive rank for high displacement = high reversion probability (reversal signal)
return cross_sectional_rank(displacement)
```

**意义**：价格大幅偏离加权平均成本后存在均值回归倾向——无论是大幅盈利(上方偏离)还是大幅亏损(下方偏离)，都有筹码驱动的回归压力。逆向排名：偏离越大越倾向于反转。

**依赖数据**：`cyq_perf.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_deep.py`

##### cost_distribution_skew

**定义**：成本分布偏度因子，(cost_50pct-cost_15pct)/(cost_85pct-cost_50pct)截面排名。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
left_tail = cyq["cost_50pct"] - cyq["cost_15pct"]
right_tail = cyq["cost_85pct"] - cyq["cost_50pct"]
skew = left_tail / right_tail.replace(0, np.nan)
return cross_sectional_rank(skew)
```

**意义**：成本分布左右偏度反映筹码的'重心'偏向——右偏(上方筹码多)=套牢盘重、上升压力大；左偏(下方筹码多)=获利盘多、上升有支撑。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_deep.py`

##### cost_distribution_width

**定义**：筹码成本分布宽度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（窄分布=筹码集中排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
width = cyq["cost_95pct"] - cyq["cost_5pct"]
median = cyq["cost_50pct"]
rel_width = safe_divide(width, median + 1e-10)
rel_width = rel_width.clip(0, 5)
return cross_sectional_rank(-rel_width)
```

**意义**：Width of chip cost distribution (95th - 5th percentile) relative to median cost measures holder concentration. Narrow = holders agree on value = accumulation. Wide = holders have diverse costs = potential volatility.

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_cost_extended.py`

##### cost_skew_momentum_5d

**定义**：成本偏度5日变化因子，chip_cost_skew的5日差分截面排名。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
lower = (perf["cost_50pct"] - perf["cost_15pct"]).replace(0, np.nan)
upper = (perf["cost_85pct"] - perf["cost_50pct"]).replace(0, np.nan)
skew = lower / upper.replace(0, np.nan)
skew_chg = skew.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(-skew_chg)
```

**意义**：成本分布偏度从'下方支撑>上方压力'转变为'上方压力>下方支撑'是趋势衰竭的领先信号。持续上涨理应将筹码从下方搬运到上方（偏度自然下降），但如果偏度快速逆转（5日变化转正），说明上方套牢盘在快速增厚——这是多空力量对比变化的微观信号。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### cost_skew_ratio

**定义**：筹码成本偏度比率因子，(cost_50pct-cost_5pct)/(cost_95pct-cost_50pct)截面排名（右偏=获利盘主导排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
lower_range = cyq["cost_50pct"] - cyq["cost_5pct"]
upper_range = cyq["cost_95pct"] - cyq["cost_50pct"]
skew = safe_divide(lower_range, upper_range + 1e-10)
skew = skew.clip(0.1, 10)
return cross_sectional_rank(skew)
```

**意义**：Cost distribution skew: right-skew means more chips below median (most holders in profit = bullish). Left-skew means more chips above median (most holders underwater = selling pressure).

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_cost_extended.py`

##### cost_support_strength

**定义**：筹码支撑强度=(收盘价-cost_50pct)/cost_50pct取反。价格在成本线下方=超跌,支撑强,排名高。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
d = context.load("daily.parquet")
close_adj = _close_adj_basis(d)
common = cyq.index.intersection(close_adj.index)
cost50 = cyq.loc[common, "cost_50pct"]
deviation = safe_divide(close_adj.loc[common] - cost50, cost50)
return cross_sectional_rank(-deviation)
```

**意义**：收盘价相对筹码中位成本(cost_50pct)的偏离取反。价格远低于中位成本=多数持仓者亏损,抛售意愿降低+抄底意愿增强=强支撑区域。价格远高于中位成本=获利盘充裕,存在获利回吐压力。该因子做多超跌(远离成本线下方)的股票。

**依赖数据**：`cyq_perf.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### current_down_streak

**定义**：当前连跌天数因子：截至今日连续收跌天数截面排名（连跌超卖排前，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_down = (daily["close"] < daily["pre_close"]).astype(int)
code = is_down.index.get_level_values("Code")
seg = (~is_down.astype(bool)).groupby(level="Code").cumsum()
streak = is_down.groupby([code, seg]).cumsum()
return cross_sectional_rank(-streak)
```

**意义**：连续下跌天数反映下跌的持续性——连跌越深,恐慌释放越充分、超卖越极端(反转候选),但连跌本身也说明空头完全掌控(趋势未止)。与 current_up_streak 镜像,作为左侧反转的时间维度信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### current_up_streak

**定义**：当前连涨天数因子：截至今日连续收涨天数截面排名（连涨动能排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_up = (daily["close"] > daily["pre_close"]).astype(int)
code = is_up.index.get_level_values("Code")
seg = (~is_up.astype(bool)).groupby(level="Code").cumsum()
streak = is_up.groupby([code, seg]).cumsum()
return cross_sectional_rank(streak)
```

**意义**：当前连续上涨天数衡量趋势的即时动能状态——连涨天数长=买盘连续性强,但过长也积累短期超买风险。与滚动窗口动量不同,本因子是状态变量:反映「今天正在发生什么」而非「过去发生了什么」。close 与数据商复权昨收 pre_close 比较,除权日判定正确。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### current_vol_shrink_streak

**定义**：当前缩量连天数因子：截至今日成交量连续递减天数截面排名（持续缩量排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
prev_vol = daily["vol"].groupby(level="Code").shift(1)
is_shrink = (daily["vol"] < prev_vol).astype(int)
code = is_shrink.index.get_level_values("Code")
seg = (~is_shrink.astype(bool)).groupby(level="Code").cumsum()
streak = is_shrink.groupby([code, seg]).cumsum()
return cross_sectional_rank(streak)
```

**意义**：成交量连续递减=交投热度持续退潮(无人问津)或惜售锁仓(缩量洗盘)——缩量连天在底部区域是筑底特征,在顶部是上涨动力衰竭的先行信号。与价格连涨/连跌正交,补充量能的节奏状态维度。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### di_plus_minus_ratio_14

**定义**：14日DI+/DI-比率因子 (多头趋势强度)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
di_plus, di_minus = _directional_movement(daily["high"], daily["low"], daily["pre_close"], scale, 14)
ratio = safe_divide(di_plus, di_minus + 1e-8)
return cross_sectional_rank(ratio)
```

**意义**：DI+>DI-意味着上升趋势，DI+/DI-比率衡量多头相对空头的优势

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### distance_from_ma_120

**定义**：收盘价/120日均线-1因子截面排名 (负向：远离均线=回归压力)。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
adj = _adjusted_close(daily_panel)
ma_120 = rolling_group_mean(adj, 120)
return cross_sectional_rank(-safe_divide(adj - ma_120, ma_120 + 1e-8).abs())
```

**意义**：价格大幅偏离半年线后均值回复力量增强，低偏离股更安全

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### distance_from_ma_5

**定义**：收盘价/5日均线-1因子截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
adj = _adjusted_close(daily_panel)
ma_5 = rolling_group_mean(adj, 5)
return cross_sectional_rank(safe_divide(adj - ma_5, ma_5 + 1e-8))
```

**意义**：短期偏离均线过大存在回归压力，但强势股可维持正偏离

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### doji_frequency_20

**定义**：十字星频率因子：20日十字星(|实体|≤10%振幅)天数占比截面排名（分歧整理排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
rng = daily["high"] - daily["low"]
body = (daily["close"] - daily["open"]).abs()
doji = (body.le(0.1 * rng) & rng.gt(0)).astype(float)
freq = _roll_mean(doji, 20, 5)
return cross_sectional_rank(freq)
```

**意义**：十字星=开盘价与收盘价几乎重合,多空当日势均力敌——连续十字星=高位滞涨或底部吸筹的分歧期。十字星密集股短线方向不明,作为低趋势确认度的反指标与趋势效率因子互补。单日内比较,无需复权。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### donchian_breakout_20

**定义**：20日Donchian突破强度因子

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
# 与 donchian_position_60 同口径:用后复权基座,除权日不产生假突破;
# shift(1) 须按 Code 分组,排除当日(T-20..T-1 窗口),避免全局 shift 跨股错位。
adj = _adjusted_close(daily)
hh = rolling_group_max(adj.groupby(level="Code").shift(1), 20)
breakout = safe_divide(adj - hh, hh)
return cross_sectional_rank(breakout)
```

**意义**：突破20日最高价是趋势启动的信号，突破幅度越大趋势越强

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### donchian_breakout_strength

**定义**：Donchian突破强度因子，(close-20日最高)/ATR截面排名（突破幅度相对波动率排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
high = daily["high"]
low = daily["low"]
# 通道上轨用复权折算后的 high (除权日不产生假突破), 2026-08-05
scale = _adjusted_close(daily) / close.replace(0, np.nan)
adj = _adjusted_close(daily)
adj_high = high * scale

highest = adj_high.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).max()
)

# ATR: average true range (pre_close is dividend-adjusted at ex-dividend dates)
tr1 = high - low
tr2 = (high - daily["pre_close"]).abs()
tr3 = (low - daily["pre_close"]).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
# ATR 折算到复权空间(×scale)与分子 adj-highest 同口径——未复权 ATR 与
# 复权基座分子混除会造成单位/量级错配(2026-08-05 修复)
atr = (tr * scale).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)

strength = safe_divide(adj - highest, atr + 1e-10)
return cross_sectional_rank(strength)
```

**意义**：Donchian breakout strength normalizes the breakout distance by ATR. A breakout 2 ATR above the channel high is more significant than a breakout 0.1 ATR above. ATR-normalization makes breakouts comparable across stocks with different volatility. Strong breakouts are more likely to develop into sustained trends.

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_pattern.py`

##### donchian_position_20

**定义**：Donchian通道位置因子，(close-20日最低)/(20日最高-20日最低)截面排名（突破高位=强势排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
# 用每日复权系数 (后复权基座/close) 折算 high/low 后再取 20 日极值,
# 避免除权日污染通道上下轨 (与 trend_pattern.donchian_position_60 同口径, 2026-08-05)
scale = _adjusted_close(daily) / close.replace(0, np.nan)
adj_high = daily["high"] * scale
adj_low = daily["low"] * scale
adj = _adjusted_close(daily)

previous_high = adj_high.groupby(level="Code").shift(1)
highest = previous_high.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).max()
)
lowest = adj_low.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).min()
)

position = safe_divide(adj - lowest, highest - lowest + 1e-10)
position = position.clip(0, 1)
return cross_sectional_rank(position)
```

**意义**：Donchian Channel position measures where price sits within its 20-day range. Values near 1.0 = price at 20-day high (breakout), near 0.0 = price at 20-day low (breakdown). Turtle traders buy breakouts from 20-day highs. This factor captures the same signal in cross-sectional form. Channel position is comparable across stocks regardless of price level.

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_pattern.py`

##### donchian_position_60

**定义**：60日Donchian通道位置因子 (低位置排前, 负向)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
# Adjusted base (cumprod(1+pct_chg/100)) — raw close 60d extrema are
# ex-dividend polluted for up to 60 days (see price.py price_position_60).
adj = _adjusted_close(daily)
hh = rolling_group_max(adj, 60)
ll = rolling_group_min(adj, 60)
position = (adj - ll) / (hh - ll + 1e-8)
return cross_sectional_rank(-position)
```

**意义**：价格在60日高低区间内的位置，低位意味着潜在反转上行

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### dpo_20

**定义**：20日DPO去趋势因子：(11日前复权价−当前20日均价)/20日均价截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
ma20 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
lagged = adj.groupby(level="Code").shift(11)
dpo = safe_divide(lagged - ma20, ma20)
return cross_sectional_rank(dpo)
```

**意义**：DPO(Detrended Price Oscillator)用11日前价格减去当前20日均价，以移除中期趋势后观察价格周期位置；shift(11)为20日周期的一半加一。再除以均价以保证股票间可比。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### eom_14

**定义**：14日EOM简易波动因子：价格中点位移÷(量/区间)的14日均值截面排名（上涨越省量排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 跨日中点位移走复权口径,避免除权日伪位移(scale=adj/close 折算 high/low)
adj = _adjusted_close(daily)
scale = adj / daily["close"].replace(0, np.nan)
adj_h = daily["high"] * scale
adj_l = daily["low"] * scale
mid = (adj_h + adj_l) / 2.0
box_ratio = daily["vol"] / (adj_h - adj_l).replace(0, np.nan)
distance = mid - mid.groupby(level="Code").shift(1)
# Ease of Movement divides by the box ratio.  Multiplication would reward
# high-volume/narrow-range days and is the inverse of the named indicator.
eom = safe_divide(distance, box_ratio)
eom_w = eom.unstack("Code")
eom_avg = eom_w.rolling(14, min_periods=7).mean()
return cross_sectional_rank(stack_date_code(eom_avg))
```

**意义**：Ease of Movement(简易波动指标)=价格中点位移÷(成交量/价格区间)——衡量价格在较少成交量下移动的容易程度。EOM高=较宽区间、较低成交量下向上移动；EOM低=价格向下移动或上行需要较大成交量。14日均值平滑。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### fib_retracement_proximity

**定义**：黄金分割位邻近度因子：250日波段的斐波那契回撤位距离截面排名（靠近关键位排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
g = adj.groupby(level="Code")
hh = g.transform(lambda s: s.rolling(250, min_periods=120).max())
ll = g.transform(lambda s: s.rolling(250, min_periods=120).min())
r = safe_divide(adj - ll, hh - ll)
# pandas 3.0 + numpy 2:ufunc 作用于 Series 返回 ndarray,须包回 Series
# (保留 (Date, Code) 索引供 cross_sectional_rank 使用)
dists = [np.abs(r - level) for level in _FIB_LEVELS]
prox = pd.Series(np.minimum.reduce(dists), index=r.index)
return cross_sectional_rank(-prox)
```

**意义**：A股技术派的共识支撑/阻力位:回调到 0.382/0.5/0.618 分位时承接与抛压结构发生切换(策略 12 均值回归同源)。因子取现价在 250 日波段中的位置距最近斐波那契位的最小距离,越近=越处于关键博弈价位。波段高低点基于后复权基座,无除权假高低。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### force_index_13

**定义**：13日力指数因子：(close−pre_close)×vol的13日EWMA截面排名（量价合力排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
price_move = daily["close"] - daily["pre_close"]
fi = price_move * daily["vol"]
fi_w = fi.unstack("Code")
fi_ema = fi_w.ewm(span=13, adjust=False).mean()
return cross_sectional_rank(stack_date_code(fi_ema))
```

**意义**：Force Index(力指数)=价格变动×成交量，衡量价格变动的资金合力——大涨配巨量=强多头力，小涨配巨量=分歧。用close−pre_close(复权口径涨跌额)替代close diff，除权日不失真。13日EWMA平滑捕捉中期合力。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### gap_abs_ma_20d

**定义**：缺口幅度均值（|open/pre_close−1| 的20日均值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["gap_abs"] = df["overnight"].abs()
return _out(df, "gap_abs_ma_20d", roll(df, "gap_abs", 20, "mean"))
```

**意义**：平均跳空幅度反映信息冲击强度与交易拥挤度，高跳空股波动与成本更高。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### gap_abs_ma_5d

**定义**：5 日平均绝对跳空（|open/pre_close − 1| 的 5 日均值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "gap_abs_ma_5d", roll(df, "overnight", 5, "mean"))
```

**意义**：gap_abs_ma_20d 的短周期版：近期跳空频率与幅度反映消息面活跃度，高跳空股票 1d 波动与可交易机会更大，与 1d 收益的尾部相关。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### gap_down_recover_freq_20d

**定义**：低开高走频率（20日内 overnight<0 且 intraday>0 的天数占比）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["gap_down_rec"] = ((df["overnight"] < 0) & (df["intraday"] > 0)).astype(float)
return _out(df, "gap_down_recover_freq_20d", roll(df, "gap_down_rec", 20, "mean"))
```

**意义**：低开高走=承接有力、恐慌被消化，高频出现说明买盘韧性好，短期偏强。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### gap_fill_5d_reversal

**定义**：缺口回补反转因子，5日内出现向下跳空后的回补倾向截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = daily["open"] / daily["pre_close"].replace(0, np.nan) - 1.0
# 5-day rolling minimum gap (most negative gap in recent 5 days)
min_gap_5d = gap.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).min()
)
return cross_sectional_rank(-min_gap_5d)
```

**意义**：A股'缺口必补'的民间规律有一定统计基础——向下跳空缺口在短期内面临均值回归压力。因子计算：(open-pre_close)/pre_close在5日内的min，取负向（即缺口越深=回补概率越大=正向预期）。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### gap_fill_tendency_10d

**定义**：缺口回补倾向因子，近10日缺口天数/(缺口天数+0.01)截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
pre_close = daily["pre_close"]
# A gap exists if open != pre_close significantly
gap = (daily["open"] - pre_close) / pre_close.replace(0, np.nan)
gap_abs = gap.abs()
is_gap = (gap_abs > 0.01).astype(float)  # >1% gap
# Gap fills if close crosses back toward pre_close
gap_filled = ((gap > 0.01) & (close < pre_close)) | ((gap < -0.01) & (close > pre_close))
gap_filled = gap_filled.astype(float)
gap_count_10 = is_gap.groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=5).sum()
)
fill_count_10 = gap_filled.groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=5).sum()
)
fill_rate = fill_count_10 / (gap_count_10 + 0.01)
return cross_sectional_rank(fill_rate)
```

**意义**：A股有'缺口必补'的民间说法——统计上，向上跳空缺口在短期内被回补的概率较高。该因子度量跳空后回补的频率：高频回补的股票可能在缺口后趋势反而较弱。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### gap_intraday_corr_20

**定义**：缺口-日内收益相关性因子：20日(高开幅度,日内收益)滚动相关截面排名（高开常延续排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
intra = safe_divide(daily["close"], daily["open"]) - 1.0
wide_gap = gap.unstack("Code")
wide_intra = intra.unstack("Code")
corr = wide_gap.rolling(20, min_periods=10).corr(wide_intra)
return cross_sectional_rank(stack_date_code(corr))
```

**意义**：每只股票「高开惯性」vs「高开回落」的固有模式:corr 高=高开常在日内延续(强势股承接强);corr 低/负=高开必被砸(出货股)。滚动 corr 用宽表向量化(Date×Code 面板 rolling().corr()),C 级实现。缺口用 open/pre_close、日内用 close/open,均为同日比较,除权日无假缺口。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### gap_open_follow_ratio_20

**定义**：跳空方向延续率因子：20日跳空(>0.5%)日中跳空方向与日内方向一致占比截面排名（顺延结构排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
intra = safe_divide(daily["close"], daily["open"]) - 1.0
has_gap = gap.abs().gt(0.005)
align = ((np.sign(gap) == np.sign(intra)) & has_gap).astype(float)
n_align = _roll_sum(align, 20, 3)
n_gap = _roll_sum(has_gap.astype(float), 20, 3)
ratio = safe_divide(n_align, n_gap)
return cross_sectional_rank(ratio)
```

**意义**：跳空后日内继续同向=隔夜信息被市场认可(趋势延续);跳空后反向回补=缺口是情绪陷阱(反转结构)。方向延续率高=跳空质量高、缺口具支撑/压力意义,是海龟/缺口交易逻辑的统计刻画。open/pre_close 均为复权口径。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### gap_ratio

**定义**：跳空比率因子，(open-pre_close)/pre_close截面排名（正=高开幅度大排前）。

**公式（计算逻辑）**：

```python
"""Compute (open - pre_close) / pre_close = gap up/down magnitude."""
daily = context.load("daily.parquet")
gap = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
return cross_sectional_rank(gap)
```

**意义**：跳空幅度反映隔夜信息冲击强度——大幅跳空高开意味着利好集中释放，但A股存在跳空回补效应，极端跳空方向可能面临反转。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### gap_ratio_20

**定义**：20日均跳空比率因子，open/pre_close - 1 截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = daily["open"] / daily["pre_close"].replace(0, np.nan) - 1.0
avg_gap = gap.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-avg_gap)
```

**意义**：向上跳空缺口反映隔夜利好信息，跳空后短期存在反转压力。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### gap_reversal_5d

**定义**：跳空反转信号——跳空方向与日内走势方向相反时标记强度取反。高开低走/低开高走=趋势陷阱。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = safe_divide(daily["open"] - daily["pre_close"], daily["pre_close"])
intraday = safe_divide(daily["close"] - daily["open"], daily["open"])
# Trapped: gap direction != intraday direction
trapped = ((np.sign(gap) * np.sign(intraday)) < 0).astype(float)
reversal_5d = trapped.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=1).mean()
)
return cross_sectional_rank(reversal_5d)
```

**意义**：识别跳空方向与日内走势方向相反的趋势陷阱。高开低走=多头陷阱(开盘诱多后出货)；低开高走=空头陷阱(开盘诱空后吸筹)。趋势陷阱是强烈的反转信号，该因子对陷阱日给予高排名(预期发生反转)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### gap_up_fade_freq_20d

**定义**：高开低走频率（20日内 overnight>0 且 intraday<0 的天数占比）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["gap_up_fade"] = ((df["overnight"] > 0) & (df["intraday"] < 0)).astype(float)
return _out(df, "gap_up_fade_freq_20d", roll(df, "gap_up_fade", 20, "mean"))
```

**意义**：高开低走=冲高抛压、情绪透支，高频出现说明上方套牢盘重，短期承压。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### gap_up_ratio_20d

**定义**：20日高开概率因子，高开(open>pre_close)天数/20截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap_up = (daily["open"] > daily["pre_close"]).astype(float)
ratio = gap_up.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(ratio)
```

**意义**：高开频率反映市场对股票的持续正面预期——频繁高开的股票往往是机构持续买入或利好信息持续释放的标的。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### gap_volume_interaction_20

**定义**：跳空×量能交互：20日均值(跳空幅度×量比)截面排名。高开且放量=资金抢筹。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vol = daily["vol"]
gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
vol_ma20 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
vol_ratio = safe_divide(vol, vol_ma20)
gi = (gap * vol_ratio).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(gi)
```

**意义**：跳空高开(open/pre_close)反映隔夜信息冲击，量比反映当日参与度——高开+放量=资金抢筹、方向确认；高开+缩量=高开低走风险；低开+放量=恐慌抛售。交互项同时捕捉方向与确认度。pre_close 已复权，跳空口径无除权失真。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_coupling.py`

##### hammer_ratio_20d

**定义**：锤子线频率因子，近20日下影线>实体2倍且实体小的天数截面排名（反转信号排前）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
body = (daily_panel["close"] - daily_panel["open"]).abs()
lower_shadow = daily_panel[["open", "close"]].min(axis=1) - daily_panel["low"]
upper_shadow = daily_panel["high"] - daily_panel[["open", "close"]].max(axis=1)
total_range = daily_panel["high"] - daily_panel["low"]
is_hammer = ((lower_shadow > 2 * body) & (upper_shadow < 0.3 * total_range) & (body > 0)).astype(float)
ratio = is_hammer.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
return cross_sectional_rank(ratio)
```

**意义**：锤子线(长下影+小实体)是经典的技术反转形态——出现在下跌趋势中往往预示底部反转，长下影代表空方在盘中打压失败后被多方强势收复。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### high_low_amplitude_20

**定义**：振幅因子，(20日最高-20日最低)/20日均价截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
close = daily_panel["close"]
# 用每日复权系数 (adj/close) 把未复权 high/low 折算到后复权空间,
# 避免除权日跨日 rolling 极值被污染(与 price_position_20d 同口径)
adj = _adjusted_close(daily_panel)
scale = adj / close.replace(0, np.nan)
adj_high = daily_panel["high"] * scale
adj_low = daily_panel["low"] * scale
high_20 = adj_high.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max())
low_20 = adj_low.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).min())
mean_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
hl_vol = (high_20 - low_20) / mean_20.replace(0, np.nan)
return cross_sectional_rank(hl_vol)
```

**意义**：振幅是波动性的另一个维度——与收益率标准差互补，振幅衡量的是日内极端价格范围而非收益率离散度。高振幅=投机性强、多空分歧大。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### high_low_expansion

**定义**：日内振幅相对20日均值的扩张程度。振幅扩大=分歧加剧，振幅收缩=方向选择在即。

**公式（计算逻辑）**：

```python
d = context.load("daily.parquet")
hl_range = d["high"] - d["low"]
mean_range = hl_range.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
expansion = safe_divide(hl_range, mean_range)
expansion = expansion.clip(0, 5)
return cross_sectional_rank(expansion)
```

**意义**：日内振幅(high-low)相对20日均值的扩张倍数。振幅突然扩大意味着多空分歧加剧、波动率突变——通常是重大信息冲击(利好或利空)或主力洗盘/出货的信号。振幅持续收缩意味着市场关注度下降或方向即将选择(暴风雨前的平静)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### high_low_volatility_20

**定义**：Parkinson波动率因子，20日基于最高最低价的波动率估计截面排名（高波排后）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
high = daily_panel["high"]
low = daily_panel["low"]

# Parkinson estimator: sqrt(1/(4*ln(2)*n) * sum(ln(H/L)^2))
hl_ratio = high / low.replace(0, np.nan)
hl_ratio = hl_ratio.where(hl_ratio > 0, np.nan)  # guard against log(<=0)
hl_ratio_log = np.log(hl_ratio)
hl_sq = hl_ratio_log ** 2
parkinson_raw = np.sqrt(hl_sq.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
) / (4.0 * np.log(2)))
return cross_sectional_rank(-parkinson_raw)
```

**意义**：Parkinson(1980)波动率使用日内高低价范围，比收盘价波动率效率高5.2倍——在同窗口下能更精确地捕捉真实波动。高HL波动率=价格振幅大=不确定性高，预期收益为负。与volatility_20互补：一个用极差估计波动，一个用收盘收益率估计。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### higher_highs_20

**定义**：20日不断抬高的高点因子 (趋势延续)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
# 跨20日分段比较 high 极值,需折算到复权空间,未复权 high 在除权日阶跃会伪造假高点结构
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
high = daily["high"] * scale

def _hh_count(s, window):
    half = max(1, window // 2)
    s1 = s.rolling(half, min_periods=half // 2).max()
    s2 = s.shift(half).rolling(half, min_periods=half // 2).max()
    return (s1 > s2).astype(float)

hh = high.groupby(level="Code").transform(lambda s: _hh_count(s, 20))
return cross_sectional_rank(hh)
```

**意义**：不断创出更高高点是强势上升趋势的特征

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### inside_bar_count_20

**定义**：孕线频率因子：20日孕线(当日高低点完全落入昨日区间)天数截面排名（收敛蓄势排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
high_adj, low_adj = _scale_hl(daily, adj)
prev_high = high_adj.groupby(level="Code").shift(1)
prev_low = low_adj.groupby(level="Code").shift(1)
inside = (high_adj.le(prev_high) & low_adj.ge(prev_low)).astype(float)
count = _roll_sum(inside, 20, 5)
return cross_sectional_rank(count)
```

**意义**：孕线=当日波动完全被昨日区间包裹,是多空分歧收敛、变盘前的压缩形态(突破前的能量积蓄)。孕线密集=波动压缩到位,后续突破方向一旦确立往往伴随主升/主跌段。跨日比较用复权口径高低点。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### intraday_cum_20d

**定义**：日内收益20日复利累计。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "intraday_cum_20d", _cum_prod(df, "intraday", 20))
```

**意义**：累计日内收益=资金行为溢价的复利视角，捕捉持续被资金承接的标的。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### intraday_ma_20d

**定义**：日内收益均值（close/open−1 的20日均值）。

**公式（计算逻辑）**：

```python
return _intraday_ma(context, 20, "intraday_ma_20d", "日内收益均值（20日）", "日内动量月度")
```

**意义**：月度日内收益方向，捕捉资金日内承接能力的持续性。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### intraday_ma_5d

**定义**：日内收益均值（close/open−1 的5日均值）。

**公式（计算逻辑）**：

```python
return _intraday_ma(context, 5, "intraday_ma_5d", "日内收益均值（5日）", "日内动量")
```

**意义**：日内收益承载交易行为溢价（承接/抛压），短窗口均值反映日内动量。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### intraday_ma_60d

**定义**：日内收益均值（close/open−1 的60日均值）。

**公式（计算逻辑）**：

```python
return _intraday_ma(context, 60, "intraday_ma_60d", "日内收益均值（60日）", "日内动量季度")
```

**意义**：季度日内收益倾向，识别交易行为长期稳定的标的。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### intraday_ret

**定义**：日内收益率因子，(close-open)/open截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = (daily["close"] - daily["open"]) / daily["open"].replace(0, np.nan)
return cross_sectional_rank(ret)
```

**意义**：日内收益与隔夜收益相关性低，提供独立于传统动量的alpha维度。强势股日内持续走高。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/alternative.py`

##### intraday_ret_momentum

**定义**：日内收益因子，(close-open)/open截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
intraday = (daily["close"] - daily["open"]) / daily["open"].replace(0, np.nan)
return cross_sectional_rank(intraday)
```

**意义**：日内收益与隔夜收益相关性低，提供独立的alpha维度。日内强势(收盘远离开盘价)反映日内买方主导和价格发现效率高。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### intraday_vol_ratio_5d

**定义**：日内/隔夜波动比（5 日 |日内收益| 均值 ÷ 5 日 |隔夜收益| 均值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["_ia"] = df["intraday"].abs()
df["_oa"] = df["overnight"].abs()
m_ia = roll(df, "_ia", 5, "mean")
m_oa = roll(df, "_oa", 5, "mean")
vals = m_ia / (m_oa + 1e-8)
return _out(df, "intraday_vol_ratio_5d", vals)
```

**意义**：日内波动占比高 = 盘中博弈主导（换手型），隔夜占比高 = 信息驱动；两者主导机制不同，比值识别 1d 收益的驱动类型。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### kama_efficiency_20

**定义**：KAMA效率比率因子：|20日净变动|/20日累计波动（趋势效率，高效率=单边趋势排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
net_move = adj.groupby(level="Code").diff(20).abs()
gross_move = adj.groupby(level="Code").diff().abs()
gross_20 = gross_move.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
er = safe_divide(net_move, gross_20)
return cross_sectional_rank(er)
```

**意义**：KAMA自适应均线的核心是效率比率ER——净变动占累计波动的比例。ER高=价格单边运行(强趋势，KAMA快速跟随)，ER低=震荡(均值回归)。是向量化的KAMA代理，既捕捉趋势强度又规避递归实现的性能问题。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### kdj_daily_j

**定义**：日频KDJ的J值因子：J=3K−2D取负截面排名（极端超买排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 9日极值跨日窗口必须走复权基座,未复权 close 在除权日跳变会污染RSV
wide = _adjusted_close(daily).unstack("Code")
ll9 = wide.rolling(9, min_periods=5).min()
hh9 = wide.rolling(9, min_periods=5).max()
rsv = safe_divide(wide - ll9, hh9 - ll9 + 1e-10) * 100.0
k = rsv.ewm(alpha=1.0 / 3.0, adjust=False).mean()
d = k.ewm(alpha=1.0 / 3.0, adjust=False).mean()
j = 3.0 * k - 2.0 * d
return cross_sectional_rank(-stack_date_code(j))
```

**意义**：策略29/40用KDJ金叉做信号——J值最灵敏(3K−2D)，>100超买、<0超卖。J值衡量短期随机动能，极端J=均值回归风险高，故负向排名。与分钟级kdj_*区分(日频口径)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### keltner_position_20

**定义**：20日Keltner通道位置因子 (高位排后, 负向)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
high, low = daily["high"], daily["low"]
# 主轨 MA 走复权基座,与已复权的 ATR 保持同口径(未复权 close 跨日窗口受除权污染)
adj = _adjusted_close(daily)
scale = adj / daily["close"].replace(0, np.nan)
tr = _true_range(high, low, daily["pre_close"])
atr = (tr * scale).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
ma = rolling_group_mean(adj, 20)
k_upper = ma + 1.5 * atr
k_lower = ma - 1.5 * atr
position = (adj - k_lower) / (k_upper - k_lower + 1e-8)
return cross_sectional_rank(-position)
```

**意义**：Keltner通道使用ATR估计带宽，对波动率变化更敏感，位置过高意味着短期过度上涨

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### liquidity_shock_20

**定义**：流动性冲击因子：Amihud(20日均值)相对前20日的变化截面排名（负向，冲击放大排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
amihud = safe_divide((daily["pct_chg"] / 100.0).abs(), daily["amount"])
log_amihud = np.log(amihud + 1e-12)
ma_now = log_amihud.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
ma_prev = ma_now.groupby(level="Code").shift(20)
shock = ma_now - ma_prev
return cross_sectional_rank(-shock)
```

**意义**：Amihud=|收益|/成交额衡量单位成交额的价格冲击——其20日均值相对更早20日的放大=流动性恶化(冲击成本上升，常伴随恐慌抛售)；收敛=流动性修复。先log再求比，消除量纲差异。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

##### listing_age_heat

**定义**：次新热度衰减因子：换手率×exp(−上市天数/500)截面排名（次新且换手高排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
fin = context.load("finance.parquet")
dates_s = pd.Series(
    pd.to_datetime(daily.index.get_level_values("Date"), format="%Y%m%d"),
    index=daily.index,
)
first_dt = dates_s.groupby(level="Code").transform("min")
age_days = (dates_s - first_dt).dt.days
turnover = fin["turnover_rate"]
heat = turnover * np.exp(-age_days / 500.0)
return cross_sectional_rank(heat)
```

**意义**：策略47/82(次新小盘)的核心是次新股的高弹性与高热度——热度随上市时间指数衰减:上市 250 天内换手仍高的次新=资金未撤离。换手率项保证日频变化与老股间的正常分层,年龄项放大次新效应。上市年龄取该股在daily 面板的首条记录(数据始于 2019,老股年龄统一截断,截面可比)。

**依赖数据**：`daily.parquet`、`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### lower_shadow_ratio

**定义**：下影线比率因子，(min(open,close)-low)/(high-low+1e-9)截面排名（长下影=承接力强排前）。

**公式（计算逻辑）**：

```python
"""Compute lower shadow ratio: (min(open, close) - low) / (high - low)."""
daily = context.load("daily.parquet")
body_range = daily["high"] - daily["low"]
lower_shadow = np.minimum(daily["open"], daily["close"]) - daily["low"]
ratio = lower_shadow / (body_range + 1e-9)
return cross_sectional_rank(ratio)
```

**意义**：长下影线反映日内探底回升过程中买盘积极承接，是重要支撑信号，低位下影线往往预示短期底部确立。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### ma_convergence_20_60

**定义**：均线收敛因子，20日均线与60日均线的距离比率截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会污染均线距离
adj = _adjusted_close(daily_panel)
ma_20 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
ma_60 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
convergence = ma_20 / ma_60.replace(0, np.nan) - 1.0
return cross_sectional_rank(convergence)
```

**意义**：短均线相对长均线的偏离程度反映趋势加速/减速，极端收敛后常伴随趋势突破。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### ma_distance_20_60

**定义**：20日-60日均线距离因子 (趋势一致性)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
adj = _adjusted_close(daily)
ma_20 = rolling_group_mean(adj, 20)
ma_60 = rolling_group_mean(adj, 60)
distance = safe_divide(ma_20 - ma_60, ma_60)
return cross_sectional_rank(distance)
```

**意义**：20日均线>60日均线=上升趋势，距离扩大=趋势加速

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### ma_distance_5_20

**定义**：5日-20日均线距离因子 (乖离)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
adj = _adjusted_close(daily)
ma_5 = rolling_group_mean(adj, 5)
ma_20 = rolling_group_mean(adj, 20)
distance = safe_divide(ma_5 - ma_20, ma_20)
return cross_sectional_rank(distance)
```

**意义**：短期均线高于长期均线=上升趋势，乖离大小反映趋势强度

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### ma_distance_5_60

**定义**：5日-60日均线距离因子 (中期乖离)

**公式（计算逻辑）**：

```python
daily = ctx.load("daily.parquet")
# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
adj = _adjusted_close(daily)
ma_5 = rolling_group_mean(adj, 5)
ma_60 = rolling_group_mean(adj, 60)
distance = safe_divide(ma_5 - ma_60, ma_60)
return cross_sectional_rank(distance)
```

**意义**：短期均线与中期均线的距离衡量中期趋势强度

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### macd_daily_hist_5d

**定义**：日频MACD柱5日变化因子：MACD(12,26,9)柱状图的5日变化截面排名（动能增强排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _adjusted_close(daily).unstack("Code")
ema12 = wide.ewm(span=12, adjust=False).mean()
ema26 = wide.ewm(span=26, adjust=False).mean()
# Normalise by the slow EMA before cross-sectional comparison.  Raw MACD
# is denominated in price units and would otherwise mostly reflect the
# arbitrary base level of each self-built adjusted-price index.
dif = safe_divide(ema12 - ema26, ema26)
dea = dif.ewm(span=9, adjust=False).mean()
hist = dif - dea
chg = hist.diff(5)
return cross_sectional_rank(stack_date_code(chg))
```

**意义**：策略58用MACD金叉死叉做选股——柱状图由负转正/持续放大=多头动能增强。5日柱变化捕捉动能二阶导。注意与分钟级macd_*区分：本因子为日频口径。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### marubozu_ratio_10d

**定义**：光头光脚阳线频率因子，近10日实体阳线(上下影极短)天数截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
body = (daily_panel["close"] - daily_panel["open"]).abs()
upper_shadow = daily_panel["high"] - daily_panel[["open", "close"]].max(axis=1)
lower_shadow = daily_panel[["open", "close"]].min(axis=1) - daily_panel["low"]
total_range = daily_panel["high"] - daily_panel["low"]
is_marubozu = ((upper_shadow + lower_shadow) / total_range.replace(0, np.nan) < 0.1).astype(float)
is_green = (daily_panel["close"] > daily_panel["open"]).astype(float)
signal = is_marubozu * is_green
ratio = signal.groupby(level="Code").transform(lambda s: s.rolling(10, min_periods=5).mean())
return cross_sectional_rank(ratio)
```

**意义**：光头光脚阳线(没有上下影或极短的实体大阳线)代表日内空方被完全压制——出现频率高意味着买方压倒性强势。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### max_vol_day_contribution_20

**定义**：单日脉冲主导度：20日内最大量日的|收益|占20日|收益|总和比例截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vol = daily["vol"]
abs_pct = daily["pct_chg"].abs()
# 近似取 rolling 窗口内 vol 最大的 1-2 个交易日(同值并列均计入)
max_vol = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).max()
)
is_max_day = (vol == max_vol) & (vol > 0)
max_day_abs = (abs_pct * is_max_day.astype(float)).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
total_abs = abs_pct.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
contrib = safe_divide(max_day_abs, total_abs)
return cross_sectional_rank(-contrib)
```

**意义**：若一段行情的收益集中在单日脉冲(占比高)，说明行情靠事件驱动、不可持续，后续波动加大；收益均匀分布=筹码换手充分、趋势扎实。该因子度量行情的时间结构。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_coupling.py`

##### mfi_14

**定义**：14日MFI资金流量指标：正负量流比截面排名（资金流入推动排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# TP 方向判定为跨日比较,走复权口径避免除权日误判方向
# (scale=adj/close 折算 high/low,close 直接用复权基座)
adj = _adjusted_close(daily)
scale = adj / daily["close"].replace(0, np.nan)
tp = (daily["high"] * scale + daily["low"] * scale + adj) / 3.0
raw_flow = tp * daily["vol"]
tp_chg = tp.groupby(level="Code").diff()
pos_flow = raw_flow.where(tp_chg > 0, 0.0)
neg_flow = raw_flow.where(tp_chg < 0, 0.0)
pos_w = pos_flow.unstack("Code").rolling(14, min_periods=7).sum()
neg_w = neg_flow.unstack("Code").rolling(14, min_periods=7).sum()
ratio = safe_divide(pos_w, neg_w + 1e-10)
mfi = 100.0 - 100.0 / (1.0 + ratio)
return cross_sectional_rank(stack_date_code(mfi))
```

**意义**：Money Flow Index(资金流量指标)用典型价格×量区分买卖压力——MFI>80超买、<20超卖。与RSI互补(量加权)。TP=(H+L+C)/3,量流方向由TP相对昨日判定——TP 走复权口径,除权日无伪方向(2026-08-05 修复)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### momentum_3

**定义**：3 日收益（收盘价 3 日变化率）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "momentum_3", df.groupby("Code")["close"].pct_change(3))
```

**意义**：3 日介于超短反转（2 日）与周度动量（5 日）之间，是 1d 预测的敏感窗口，补充 momentum_5 未覆盖的粒度。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### obv_divergence_20

**定义**：OBV量价背离因子：价格动量排名−OBV动量排名的背离截面排名（价涨量缩背离排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
price_mom = adj.groupby(level="Code").pct_change(20, fill_method=None)
obv = _obv_series(daily)
obv_mom = obv.groupby(level="Code").diff(20)
avg_vol = daily["vol"].groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
obv_norm = safe_divide(obv_mom, avg_vol + 1e-10)
# 截面标准化后求背离:价动量强 且 OBV 弱 = 正背离
pr = price_mom.groupby(level="Date").rank(pct=True)
ob = obv_norm.groupby(level="Date").rank(pct=True)
return cross_sectional_rank(pr - ob)
```

**意义**：量价背离是经典顶部/底部信号——价格创新高而OBV未能跟进=上涨缺乏量能确认(顶部背离，风险信号)；价格新低而OBV企稳=抛压衰竭(底部背离)。本因子排名高=价强量弱背离，作为趋势谨慎信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### obv_slope_20

**定义**：20日OBV斜率因子：OBV的20日变化截面排名（量能净流入加速排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
obv = _obv_series(daily)
slope = obv.groupby(level="Code").diff(20)
# 量纲归一:除以20日平均成交量的对数量级,避免大市值股天然占优
avg_vol = daily["vol"].groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
norm = safe_divide(slope, avg_vol + 1e-10)
return cross_sectional_rank(norm)
```

**意义**：OBV(能量潮)累计价涨量正、价跌量负的成交——20日变化=近一月量能净方向。OBV上升=放量上涨主导(资金主动买入)，下降=放量下跌主导。方向用pct_chg符号判定，除权日不失真。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### oi_divergence_intensity

**定义**：隔夜日内背离强度因子，(close-open)-(open-pre_close)截面排名（正=日内强化跳空方向排前）。

**公式（计算逻辑）**：

```python
"""Signed intraday return in the direction of the opening gap."""
daily = context.load("daily.parquet")
overnight_gap = safe_divide(
    daily["open"] - daily["pre_close"], daily["pre_close"]
)
intraday_ret = safe_divide(daily["close"] - daily["open"], daily["open"])
reinforcement = np.sign(overnight_gap) * intraday_ret
return cross_sectional_rank(reinforcement)
```

**意义**：隔夜跳空与日内走势的背离度是判断跳空质量的关键——跳空后日内继续同向运行（正背离）说明跳空方向得到市场认可，反向运行（负背离）说明跳空是情绪过度反应。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### open_price_shock

**定义**：开盘跳空幅度取正。大幅跳空=隔夜信息冲击强→短期反转概率高，排名高。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap_abs = safe_divide(
    (daily["open"] - daily["pre_close"]).abs(),
    daily["pre_close"],
)
return cross_sectional_rank(gap_abs)
```

**意义**：开盘跳空幅度(绝对值)的截面排名。大幅跳空(无论正负)意味着隔夜信息冲击强烈——集合竞价阶段出现极端不平衡。这种情况往往导致开盘后短期反转(跳空回补效应)。高幅度排名意味着反转交易机会更大。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### outside_bar_count_20

**定义**：吞没/突破频率因子：20日收盘突破昨日全天区间(吞没)天数截面排名（强吞没走势排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
high_adj, low_adj = _scale_hl(daily, adj)
prev_high = high_adj.groupby(level="Code").shift(1)
prev_low = low_adj.groupby(level="Code").shift(1)
outside = (adj.gt(prev_high) | adj.lt(prev_low)).astype(float)
count = _roll_sum(outside, 20, 5)
return cross_sectional_rank(count)
```

**意义**：吞没形态(收盘突破昨日全区间)是多空力量的单边碾压——向上吞没=多方完全收复昨日空方阵地(强势确认),向下吞没=空方碾压。频率高=趋势性股票持续单边运行,与震荡股形成截面区分。用复权 close 与复权昨日高低比较。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### overnight_cum_20d

**定义**：隔夜收益20日复利累计。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "overnight_cum_20d", _cum_prod(df, "overnight", 20))
```

**意义**：累计隔夜收益=消息面溢价的复利视角，比均值更能反映趋势型隔夜强势股。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### overnight_gap

**定义**：隔夜跳空因子，-(open-pre_close)/pre_close截面排名（跳空高开=反转信号排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
return cross_sectional_rank(-gap)
```

**意义**：A股隔夜跳空存在均值回归特征，大幅高开/低开后倾向于日内反转。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/alternative.py`

##### overnight_gap_momentum

**定义**：隔夜跳空因子，(open-pre_close)/pre_close截面排名（高开排前=利好消化未完）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
return cross_sectional_rank(gap)
```

**意义**：A股隔夜跳空后短期存在动量效应——正跳空(高开)反映隔夜利好信息未完全消化、开盘后仍有追涨动力；负跳空(低开)反映利空。与intraday_ret互补：一个捕捉隔夜信息冲击，一个捕捉日内价格发现。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### overnight_gap_vol_20

**定义**：20日隔夜跳空波动率因子 (高波动排后, 负向)。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
close = daily_panel["close"]
open_p = daily_panel["open"]
# pre_close is dividend-adjusted at ex-dividend dates, unlike close.shift(1)
prev_close = daily_panel["pre_close"]
overnight_ret = safe_divide(open_p - prev_close, prev_close + 1e-8)
gap_vol = overnight_ret.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
return cross_sectional_rank(-gap_vol)
```

**意义**：隔夜跳空波动大意味着信息不确定性高，可能存在信息不对称风险

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### overnight_intraday_divergence_daily

**定义**：隔夜-日内背离因子，overnight_gap - intraday_ret截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
overnight = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
intraday = (daily["close"] - daily["open"]) / daily["open"].replace(0, np.nan)
divergence = overnight - intraday
return cross_sectional_rank(-divergence)
```

**意义**：隔夜和日内方向相反时反映信息不对称——高开低走=隔夜乐观情绪被日内交易否定(短期反转信号)，低开高走=隔夜恐慌被纠正(短期反弹信号)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### overnight_intraday_ratio_20d

**定义**：隔夜/日内收益强度比（20日均值之比，分母取绝对值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
oma = roll(df, "overnight", 20, "mean")
ima = roll(df, "intraday", 20, "mean")
return _out(df, "overnight_intraday_ratio_20d", oma / (ima.abs() + 1e-6))
```

**意义**：比值高=收益主要来自隔夜（消息/情绪驱动），低=日内交易行为主导；刻画两类收益来源的相对强弱。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### overnight_ma5

**定义**：隔夜收益5日均值因子：open/pre_close-1 的5日滚动均值（隔夜动能排前）。

**公式（计算逻辑）**：

```python
df = _daily(context)
vals = df.groupby("Code")["overnight"].transform(
    lambda s: s.rolling(5, min_periods=1).mean()
)
return _out(df, "overnight_ma5", vals)
```

**意义**：隔夜收益衡量集合竞价定价的信息冲击，其5日均值过滤单日噪音后捕捉隔夜动能的持续性——连续隔夜高开=市场对该股的信息定价持续上修。V8 筛查 meanIC 0.019/ICIR 0.17。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_cand_daily.py`

##### overnight_ma_20d

**定义**：隔夜收益均值（open/pre_close−1 的20日均值）。

**公式（计算逻辑）**：

```python
return _overnight_ma(context, 20, "overnight_ma_20d", "隔夜收益均值（20日）", "隔夜动量月度")
```

**意义**：月度隔夜收益方向，过滤单日噪音，刻画消息驱动的持续性溢价。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### overnight_ma_5d

**定义**：隔夜收益均值（open/pre_close−1 的5日均值）。

**公式（计算逻辑）**：

```python
return _overnight_ma(context, 5, "overnight_ma_5d", "隔夜收益均值（5日）", "隔夜动量")
```

**意义**：隔夜收益承载消息面与情绪溢价，短窗口均值反映近期隔夜动量方向。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### overnight_ma_60d

**定义**：隔夜收益均值（open/pre_close−1 的60日均值）。

**公式（计算逻辑）**：

```python
return _overnight_ma(context, 60, "overnight_ma_60d", "隔夜收益均值（60日）", "隔夜动量季度")
```

**意义**：季度隔夜收益倾向，识别长期存在稳定隔夜溢价/折价的标的。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### overnight_minus_intraday

**定义**：隔夜−日内收益差因子：open/pre_close-1 与 close/open-1 之差（隔夜强于日内排前）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "overnight_minus_intraday", df["overnight"] - df["intraday"])
```

**意义**：隔夜与日内收益的差值衡量消息定价结构——隔夜涨而日内跌=开盘消化利好后日内反转压力大；隔夜跌而日内涨=利空出尽、资金盘中承接。V8 筛查 meanIC 0.030/ICIR 0.20，为 27 个日频候选中最强。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_cand_daily.py`

##### overnight_return_share_20

**定义**：隔夜收益占比因子：20日|隔夜跳空|占(跳空+日内)总波动的比例截面排名（隔夜驱动排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
intra = safe_divide(daily["close"], daily["open"]) - 1.0
abs_gap = gap.abs()
abs_intra = intra.abs()
num = _roll_sum(abs_gap, 20, 10)
den = _roll_sum(abs_gap + abs_intra, 20, 10)
share = safe_divide(num, den)
return cross_sectional_rank(share)
```

**意义**：每日收益可分解为隔夜(open/pre_close−1)与日内(close/open−1)两部分——隔夜占比高=股票由隔夜信息驱动(公告/海外/政策敏感,跳空定价);日内占比高=盘中博弈驱动(题材/情绪)。隔夜驱动型股票的信息传导更快,与 am_pm_return_ratio(分钟级)区分:本因子为日频分解口径。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### overnight_sign_consistency_20d

**定义**：隔夜方向一致性（20日内 overnight>0 的天数占比）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["overnight_pos"] = (df["overnight"] > 0).astype(float)
return _out(df, "overnight_sign_consistency_20d", roll(df, "overnight_pos", 20, "mean"))
```

**意义**：隔夜收益正占比高=消息面持续偏多，方向一致性强于幅度，稳健捕捉情绪溢价。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### overnight_skewness_20d

**定义**：20日隔夜收益偏度的绝对值取反。极端偏度=信息冲击不稳定，排名低。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
overnight = safe_divide(daily["open"] - daily["pre_close"], daily["pre_close"])
skew = overnight.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).skew()
)
skew = skew.clip(-5, 5)
return cross_sectional_rank(-skew.abs())
```

**意义**：20日隔夜收益(open/pre_close-1)的偏度。正偏度=偶尔大幅高开(利好集中释放)，负偏度=偶尔大幅低开(利空突袭)。极端偏度(无论正负)反映信息冲击的不稳定性——公司基本面存在不确定性或信息不对称严重。取绝对值后排名取反。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### overnight_std_5d

**定义**：隔夜收益标准差（5 日）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "overnight_std_5d", roll(df, "overnight", 5, "std"))
```

**意义**：overnight_std_20d 的短周期版：隔夜波动反映盘后信息流强度，5 日窗口对信息事件更敏感，是 1d 波动与方向的先行指标。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### price_position_20d

**定义**：20日价格位置因子，(close-20日最低)/(20日最高-20日最低)截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
# 用自建后复权基座(pct_chg 累乘)替代未复权 close,避免除权日 20 日区间高低点被污染
adj = _adjusted_close(daily_panel)
high_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max())
low_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).min())
position = (adj - low_20) / (high_20 - low_20).replace(0, np.nan)
return cross_sectional_rank(position)
```

**意义**：价格在近期高低点区间中的位置反映短期趋势强度——接近区间上沿=强势趋势中(正动量)，接近区间下沿=弱势中(负动量)。与单纯看涨跌幅不同，位置因子在震荡市中也能提供有效区分度。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### price_position_60

**定义**：60日价格位置，(close-60d_low)/(60d_high-60d_low) 截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# Use the point-in-time adjusted base (cumprod(1+pct_chg/100)): raw close
# rolling extrema stay polluted for up to 60 days after an ex-dividend
# event (same class of bug the 7-31 audit fixed for the 252-day factors).
adj = _adjusted_close(daily)
high_60 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).max()
)
low_60 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).min()
)
position = (adj - low_60) / (high_60 - low_60).replace(0, np.nan)
return cross_sectional_rank(position)
```

**意义**：价格在近60日区间内的相对位置反映短期趋势强度。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### psy_12

**定义**：心理线因子：12日上涨天数占比截面排名（多头情绪浓度排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
up = (daily["pct_chg"] > 0).astype(float)
psy = up.groupby(level="Code").transform(
    lambda s: s.rolling(12, min_periods=6).mean()
)
return cross_sectional_rank(psy)
```

**意义**：心理线(PSY)是经典情绪指标——12日内上涨天数占比度量散户情绪的持续性:占比高=多头氛围浓(趋势延续的群众基础);占比低=空头氛围。与偏动量/价格类因子正交:只计数涨跌方向、不看幅度,对温和上涨(幅度小但天数多)的股票给予高排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### range_position_20d

**定义**：20日区间位置（close 在20日 high-low 区间内的位置）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "range_position_20d", _range_position(df, 20))
```

**意义**：接近区间上沿=短期强势/突破在即，接近下沿=超跌；区间位置是经典的趋势位置信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### range_position_60d

**定义**：60日区间位置（close 在60日 high-low 区间内的位置）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "range_position_60d", _range_position(df, 60))
```

**意义**：季度区间位置过滤短期噪音，识别中期趋势所处阶段。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### residual_momentum_20

**定义**：残差动量因子：剔除市场暴露后的残差20日累计截面排名（特质动量排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
mkt = _market_proxy(wide)
beta = _rolling_beta(wide, mkt, 60, 30)
resid = wide - beta.multiply(mkt, axis=0)
resid_mom = resid.rolling(20, min_periods=10).sum()
return cross_sectional_rank(stack_date_code(resid_mom))
```

**意义**：残差动量(研报:基于残差动量的相对收益动量策略)——个股对全市场回归后的残差剔除了系统性beta暴露，纯特质部分的中期动量更稳定、拥挤度更低。基于beta_60回归残差的20日累计，与绝对动量正交。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### ret_autocorr_1d_20

**定义**：日收益一阶自相关因子：ret与昨日ret的20日滚动相关截面排名（趋势性排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret_w = _ret_wide(daily)
ac = ret_w.rolling(20, min_periods=10).corr(ret_w.shift(1))
return cross_sectional_rank(stack_date_code(ac))
```

**意义**：日收益的一阶自相关直接度量「涨后跟涨/跌后跟跌」的延续性——正自相关=趋势性股票(动量内部结构稳固);负自相关=均值回归股票(高低切频繁)。与成交量自相关正交(收益维度),与分钟级 ret_autocorr_5min 区分(日频口径)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### ret_efficiency_20

**定义**：价格效率：20日累计收益/20日累计成交额截面排名。单位成交额推动的价格变动效率。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
pct_sum = daily["pct_chg"].groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
amt_sum = daily["amount"].groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
eff = safe_divide(pct_sum, amt_sum)
return cross_sectional_rank(eff)
```

**意义**：同样的成交额推动更大的价格变动=筹码锁定好、抛压小(amihud 的收益侧有符号版本)；价格效率高的股票上涨更'省力'，趋势阻力小。该因子与 amihud 互补：amihud 看成本的绝对水平，本因子看收益的相对效率。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_coupling.py`

##### ret_kurt_5d

**定义**：5 日收益峰度。

**公式（计算逻辑）**：

```python
df = _daily(context)
vals = df.groupby("Code")["ret"].transform(
    lambda s: s.rolling(5, min_periods=3).kurt())
return _out(df, "ret_kurt_5d", vals)
```

**意义**：5 日峰度识别极端行情聚集：高尖峰 = 单日极端波动主导，随后 1d 波动收缩与均值回归概率上升。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### ret_skew_5d

**定义**：5 日收益偏度。

**公式（计算逻辑）**：

```python
df = _daily(context)
vals = df.groupby("Code")["ret"].transform(
    lambda s: s.rolling(5, min_periods=3).skew())
return _out(df, "ret_skew_5d", vals)
```

**意义**：ret_skew_20/60 的短周期版：5 日偏度识别脉冲式拉升/跳水形态，正偏度（脉冲上涨）后 1d 倾向反转，负偏度后倾向修复。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### ret_vol_lead_corr_20

**定义**：量领先价相关性因子：昨日vol与今日收益的20日滚动相关截面排名（量能前瞻有效排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret_w = _ret_wide(daily)
lag_vol_w = daily["vol"].unstack("Code").shift(1)
corr = ret_w.rolling(20, min_periods=10).corr(lag_vol_w)
return cross_sectional_rank(stack_date_code(corr))
```

**意义**：量领先价相关(corr(vol_t−1, ret_t))衡量成交量的前瞻信息含量——高=昨日放量的股票次日倾向上涨(量是价格的先行指标,资金提前布局);低/负=放量次日回落(放量出货)。是「量在价先」逻辑的统计检验,与同期相关(volume_price_corr_20)互为先后手。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### retail_attention

**定义**：散户关注度代理因子，异常高换手率(当日换手/20日均换手-1)与大单净流出交乘截面排名（取负向）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
ff = context.load("main_fund_flow.parquet")
turnover = fin["turnover_rate"]
to_20_mean = rolling_group_mean(turnover, 20)
abnormal_to = turnover / to_20_mean.replace(0, np.nan) - 1.0

# Total amount and big order net
total_amount = (
    ff["buy_sm_amount"] + ff["sell_sm_amount"]
    + ff["buy_md_amount"] + ff["sell_md_amount"]
    + ff["buy_lg_amount"] + ff["sell_lg_amount"]
    + ff["buy_elg_amount"] + ff["sell_elg_amount"]
).replace(0, np.nan)
big_net = (ff["buy_lg_amount"] + ff["buy_elg_amount"]
           - ff["sell_lg_amount"] - ff["sell_elg_amount"])
big_net_ratio = big_net / total_amount

common = abnormal_to.index.intersection(big_net_ratio.index)
# Retail attention = high abnormal turnover + big order selling
signal = abnormal_to.loc[common] * (-big_net_ratio.loc[common])
return cross_sectional_rank(-signal)
```

**意义**：高换手+大单流出=散户接盘信号。机构通过大单出货、散户通过中小单接盘，这种成交量结构预示后续下跌。

**依赖数据**：`finance.parquet`、`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/behavioral.py`

##### reversal_2d

**定义**：2 日收益（收盘价 2 日变化率）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "reversal_2d", df.groupby("Code")["close"].pct_change(2))
```

**意义**：A 股超短周期（1-5 日）以反转为主：2 日大涨的股票次日倾向回吐，2 日大跌倾向反弹，1d 持有期内信号最直接。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### roc_12

**定义**：12日ROC变动率因子：(adj−adj.shift(12))/adj.shift(12)截面排名（中期加速排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
roc = adj.groupby(level="Code").transform(
    lambda s: s.pct_change(12, fill_method=None)
)
return cross_sectional_rank(roc)
```

**意义**：ROC(Price Rate of Change)是经典动量指标——12日变化率衡量中期速度，正ROC=价格加速上行。基于复权基座计算，除权日无跳变失真。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### rs_120

**定义**：120日相对强度因子：中期相对全市场的RS截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
return cross_sectional_rank(_relative_strength(daily, 120))
```

**意义**：120日(半年)RS捕捉中期风格——策略23要求 RS_year>RS_month>0 的渐进强势结构，120日RS介于30日与365日之间，衡量半年趋势质量。半年持续跑赢=机构持仓换手充分。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### rs_250

**定义**：250日相对强度因子：年度相对全市场的RS截面排名（年度强势排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
return cross_sectional_rank(_relative_strength(daily, 250))
```

**意义**：250日(年)RS是祖鲁法则的最终确认——年度持续跑赢的股票基本面与资金面双重强劲，是机构「抱团」股票的量化刻画。年RS过滤了短期噪音，信号最稳定但响应最慢。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### rs_60

**定义**：60日相对强度因子：个股60日收益相对全市场等权收益的RS截面排名（跑赢市场排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
return cross_sectional_rank(_relative_strength(daily, 60))
```

**意义**：祖鲁法则(策略23)的核心确认信号：RS=(r_s−r_m)/(1+r_m)。60日RS衡量中期相对强弱——持续跑赢市场的股票处于资金聚集区，动量延续概率高。与绝对动量正交(剔市场贝塔)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### rsi_spread_6_14

**定义**：日频RSI(6)−RSI(14)因子：短中期超买超卖差截面排名（短期强于中期排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# pct_chg itself is the one-day price change used by RSI.  Differencing it
# again would calculate an RSI of return acceleration, not a price RSI.
delta = (daily["pct_chg"] / 100.0).unstack("Code")

def _rsi(delta, n):
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing: alpha=1/n.  min_periods avoids presenting a
    # partially initialised oscillator as a fully formed RSI value.
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = safe_divide(avg_gain, avg_loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)

spread = _rsi(delta, 6) - _rsi(delta, 14)
return cross_sectional_rank(stack_date_code(spread))
```

**意义**：策略42用RSI(6)在55-80区间做超买判断——RSI6−RSI14的正差=短期动能强于中期(新一波拉升启动)，负差=短期动能衰竭。捕捉RSI的动量而非水平，与分钟级rsi_*区分(日频口径)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### rsrs_beta_18

**定义**：RSRS Beta因子，18日high~low回归斜率截面排名（高beta=阻力上升快于支撑=看涨排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 折算到复权空间再回归,避免除权日 high/low 阶跃污染斜率(见 _compute_rsrs_beta)
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
high = daily["high"] * scale
low = daily["low"] * scale
beta, _ = _compute_rsrs_beta(high, low, window=18)
return cross_sectional_rank(beta)
```

**意义**：RSRS (Resistance Support Relative Strength) measures the dynamic relationship between daily highs and lows via OLS regression. When beta is high, resistance levels are rising faster than support levels -- a bullish regime shift. When beta is low or falling, support is weakening -- a bearish signal. Originally developed for index timing, applied cross-sectionally this identifies stocks undergoing support/resistance regime changes.

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_pattern.py`

##### rsrs_beta_momentum_5

**定义**：RSRS斜率动量因子：18日high~low回归斜率5日变化截面排名（斜率转升排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
beta, _ = _rsrs_beta_r2(daily, window=18)
mom = beta.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(mom)
```

**意义**：RSRS β 的 5 日变化度量支撑阻力关系的边际方向——斜率刚转升=阻力相对支撑抬升的初期(阻力转支撑的早期信号),斜率转降=支撑走弱。与 β 绝对值(水平)互补:在 β 接近的股票中区分斜率加速与减速者。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### rsrs_r2_18

**定义**：RSRS R-squared因子，18日high~low回归拟合优度截面排名（高R2=支撑阻力关系清晰排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
high = daily["high"] * scale
low = daily["low"] * scale
_, r2 = _compute_rsrs_beta(high, low, window=18)
return cross_sectional_rank(r2)
```

**意义**：RSRS R-squared measures the quality of the support/resistance relationship. High R2 means the high-low relationship is tight and predictable -- clear support/resistance structure. Low R2 means the relationship is breaking down -- uncertainty, potential regime change. Low R2 precedes volatility expansion.

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_pattern.py`

##### rsrs_right_deviation

**定义**：RSRS右偏离因子，(zscore * beta * r2)截面排名（量价验证=信号可靠排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
high = daily["high"] * scale
low = daily["low"] * scale
beta, r2 = _compute_rsrs_beta(high, low, window=18)

roll_mean = rolling_group_mean(beta, 400, min_periods=100)
roll_std = rolling_group_std(beta, 400, min_periods=100)
zscore = safe_divide(beta - roll_mean, roll_std + 1e-8)

# Combined score: right-deviation = zscore * beta * r2
score = zscore * beta * r2.fillna(0)
return cross_sectional_rank(score)
```

**意义**：RSRS Right Deviation combines three dimensions: 1) Z-score: how extreme is the regime shift? 2) Beta: what direction is the shift? 3) R-squared: how reliable is the measured relationship? This is the composite signal from the original RSRS strategy, applied cross-sectionally to rank stocks by support/resistance quality.

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_pattern.py`

##### rsrs_volume_right_deviation

**定义**：RSRS量能加权右偏离因子：β的400日Z分数×β×R²×近期量能占比截面排名（趋势信号+量能确认排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
beta, r2 = _rsrs_beta_r2(daily, window=18)
g = beta.groupby(level="Code")
beta_mean = g.transform(lambda s: s.rolling(400, min_periods=120).mean())
beta_std = g.transform(lambda s: s.rolling(400, min_periods=120).std())
z = safe_divide(beta - beta_mean, beta_std)
vol = daily["vol"]
vol9 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(9, min_periods=5).sum()
)
vol18 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(18, min_periods=10).sum()
)
vol_ratio = safe_divide(2.0 * vol9, vol18)
raw = z * beta * r2 * vol_ratio
return cross_sectional_rank(raw)
```

**意义**：策略77的独有变形:RSRS 右偏离(β 相对自身 400 日历史的标准化偏离)叠加近 9 日/18 日量能占比加权——量能集中在近期=趋势信号获得资金确认,放量突破的右偏离比缩量阴跌的右偏离更可信。与既有 rsrs_right_deviation的区别仅在量能加权项,捕捉「信号+量」的共振。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### rsrs_zscore_18

**定义**：RSRS Z-score因子，beta相对自身历史400日的标准化偏离截面排名（极端偏离=支撑阻力位重塑排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
high = daily["high"] * scale
low = daily["low"] * scale
beta, _ = _compute_rsrs_beta(high, low, window=18)

# Z-score relative to 400-day rolling window
roll_mean = rolling_group_mean(beta, 400, min_periods=100)
roll_std = rolling_group_std(beta, 400, min_periods=100)
zscore = safe_divide(beta - roll_mean, roll_std + 1e-8)
return cross_sectional_rank(zscore)
```

**意义**：RSRS standardized score measures how extreme the current beta is relative to its own history. A large positive z-score means the support/resistance structure is undergoing a significant bullish shift. This is the signal used in the original RSRS timing strategy.

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_pattern.py`

##### shadow_asymmetry

**定义**：影线不对称性因子，-(upper_shadow_ratio-lower_shadow_ratio)*(high-low)/close截面排名（上影主导=看空排后）。

**公式（计算逻辑）**：

```python
"""Compute weighted shadow asymmetry: (upper_shadow_ratio - lower_shadow_ratio) * (high-low) / close.
Rank negative (more upper shadow = bearish).
"""
daily = context.load("daily.parquet")
body_range = daily["high"] - daily["low"]
close = daily["close"]

upper_shadow = daily["high"] - np.maximum(daily["open"], daily["close"])
lower_shadow = np.minimum(daily["open"], daily["close"]) - daily["low"]

upper_shadow_ratio = upper_shadow / (body_range + 1e-9)
lower_shadow_ratio = lower_shadow / (body_range + 1e-9)

asymmetry = (upper_shadow_ratio - lower_shadow_ratio) * body_range / close.replace(0, np.nan)
return cross_sectional_rank(-asymmetry)
```

**意义**：加权影线不对称性将上下影线的相对长度与蜡烛实体大小结合——大实体+上影主导是最强烈的日内见顶形态。因子值越负（上影明显长于下影且实体大）越看空。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### shadow_lower_20

**定义**：20日均下影线比例，下影线/(high-low) 截面排名（高值=强支撑）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
lower_shadow = daily[["open", "close"]].min(axis=1) - daily["low"]
body_range = daily["high"] - daily["low"]
ratio = lower_shadow / body_range.replace(0, np.nan)
avg_ratio = ratio.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(avg_ratio)
```

**意义**：下影线反映低位承接力，高下影线比例是底部支撑信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### shadow_upper_20

**定义**：20日均上影线比例，上影线/(high-low) 截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
upper_shadow = daily["high"] - daily[["open", "close"]].max(axis=1)
body_range = daily["high"] - daily["low"]
ratio = upper_shadow / body_range.replace(0, np.nan)
avg_ratio = ratio.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-avg_ratio)
```

**意义**：上影线反映高位抛压，长期高上影线比例是上涨阻力信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### stoch_slow_k

**定义**：慢速随机%K因子：RSV(9)的三日平滑截面排名（随机动能排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 9日极值跨日窗口必须走复权基座,未复权 close 在除权日跳变会污染RSV
wide = _adjusted_close(daily).unstack("Code")
ll9 = wide.rolling(9, min_periods=5).min()
hh9 = wide.rolling(9, min_periods=5).max()
rsv = safe_divide(wide - ll9, hh9 - ll9 + 1e-10) * 100.0
slow_k = rsv.rolling(3, min_periods=2).mean()
return cross_sectional_rank(stack_date_code(slow_k))
```

**意义**：慢速随机指标(策略29 KD)通过平滑RSV过滤噪音——%K上穿%D是经典金叉。慢速%K衡量短期超买超卖动能，取正向排名(动能强排前)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### td_setup_count

**定义**：TD序列setup计数因子：连续close≤4日前close的天数截面排名（连续下跌setup排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 跨日比较必须走复权基座,未复权 close 在除权日跳变会伪造连跌setup
adj = _adjusted_close(daily)
cond = (adj <= adj.groupby(level="Code").shift(4)).astype(int)
# 每股连续段计数:段号 = (cond==0) 的累计,段内 = 组内 cumsum
code = cond.index.get_level_values("Code")
seg = (~cond.astype(bool)).groupby(level="Code").cumsum()
count = cond.groupby([code, seg]).cumsum()
return cross_sectional_rank(count)
```

**意义**：TD Sequential(策略78 GFTD)的买入setup=连续N日收盘价≤4日前收盘价，计数达到9通常意味着耗尽性下跌。连续setup天数反映超卖衰竭的程度——setup计数高=持续阴跌后反转概率积累，作为左侧反转信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### trix_12_20

**定义**：TRIX趋势因子：三重指数平滑(12)的20日变化率截面排名（趋势加速排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _adjusted_close(daily).unstack("Code")
trix = _trix_wide(wide)
return cross_sectional_rank(stack_date_code(trix))
```

**意义**：TRIX通过三重平滑滤除短期噪音，其20日变化率反映中长期趋势的加速/减速——策略42验证 TRIX 在低回撤组合中的有效性。正变化率=中长期趋势向上。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### trix_signal_gap

**定义**：TRIX信号乖离因子：TRIX与其20日信号线的乖离截面排名（强于自身趋势线排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _adjusted_close(daily).unstack("Code")
trix = _trix_wide(wide)
signal = trix.rolling(20, min_periods=10).mean()
gap = trix.sub(signal).div(signal.abs() + 1e-10)
return cross_sectional_rank(stack_date_code(gap))
```

**意义**：TRIX上穿/下穿其信号线(自身20日均值)是经典买卖信号——乖离扩大=趋势加速中，乖离收敛=趋势衰竭。捕捉策略42中 TRIX×MATRIX 的交叉逻辑的横截面形态。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### turnover_anomaly_20

**定义**：换手率异常因子，20日均换手/60日均换手-1截面排名（取负向=异常高换手排后）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
turnover = fin["turnover_rate"]
to_20 = rolling_group_mean(turnover, 20)
to_60 = rolling_group_mean(turnover, 60)
anomaly = to_20 / to_60.replace(0, np.nan) - 1.0
return cross_sectional_rank(-anomaly)
```

**意义**：换手率短期飙升往往伴随投机性交易或信息事件，高异常换手率预示短期反转。中长期低换手率则与低波动溢价相关。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/behavioral.py`

##### turnover_anomaly_mean_20d

**定义**：换手率异象因子，-(近20日平均换手率)截面排名（高换手=投机性强排后）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate"]
to_20 = turnover.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-to_20)
```

**意义**：A股换手率异象是全球最显著的——高换手率股票后续收益显著偏低，原因在于散户过度交易和投机炒作后的均值回复。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/behavioral.py`

##### turnover_ret_corr_20

**定义**：换手率-收益相关因子：20日换手率与收益的相关性截面排名（量价同步排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
fin = context.load("finance.parquet")
ret = (daily["pct_chg"] / 100.0).unstack("Code")
to = fin["turnover_rate"].unstack("Code")
common_cols = ret.columns.intersection(to.columns)
ret = ret[common_cols]
to = to[common_cols]
corr = ret.rolling(20, min_periods=10).corr(to)
return cross_sectional_rank(stack_date_code(corr))
```

**意义**：换手率与收益的正相关=上涨伴随放量(量价同步、趋势健康)，负相关=上涨缩量/下跌放量(背离)。用 finance.turnover_rate 与 pct_chg 的20日滚动相关(宽表向量化)，捕捉资金与价格的同步性。

**依赖数据**：`daily.parquet`、`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

##### turnover_std_20

**定义**：换手率波动率因子，20日换手率标准差截面排名（高换手波动排后=流动性风险）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate"]
to_vol = turnover.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
return cross_sectional_rank(-to_vol)
```

**意义**：换手率剧烈波动意味着流动性不稳定——要么是资金突击进出、要么是筹码松动。与turnover_20互补：一个看换手水平，一个看换手稳定性。高换手波动代表流动性风险溢价。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### turnover_zscore_20

**定义**：换手率20日Z-score因子（当日换手偏离自身20日均值的标准差数，反向排名）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate"].clip(0, 100)
z = _zscore_group(turnover, 20, 10)
return cross_sectional_rank(-z)
```

**意义**：换手率异象：个股换手相对自身历史水平异常放大=交易过热、筹码换手频繁，未来收益倾向偏低（A股高换手负溢价）。z-score 消除了换手率的水平差异（低换手大盘股与高换手小盘股不可直接比较），只保留异常度。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/liquidity_factors.py`

##### up_day_freq_20d

**定义**：上涨天数频率（20日内 close>前收盘 的天数占比）。

**公式（计算逻辑）**：

```python
df = _daily(context)
up = (df.groupby("Code")["close"].diff() > 0).astype(float)
df["up"] = up
return _out(df, "up_day_freq_20d", roll(df, "up", 20, "mean"))
```

**意义**：上涨频率比累计涨幅更稳健：高频率=胜率高、路径平稳，规避暴涨暴跌型标的。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### up_day_volume_ratio_20

**定义**：上涨日量占比因子：20日上涨日成交量占总量的比例截面排名（上涨放量排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vol = daily["vol"]
up = (daily["pct_chg"] > 0).astype(float)
up_vol = (up * vol).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
tot_vol = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
ratio = safe_divide(up_vol, tot_vol)
return cross_sectional_rank(ratio)
```

**意义**：上涨日的成交量占比衡量「量能的方向质量」——涨时放量(占比>50%)=资金主动做多；涨时缩量=无量反弹(不可持续)。是量价共振的日频简化版。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

##### upper_shadow_ratio

**定义**：上影线比率因子，-(high-max(open,close))/(high-low+1e-9)截面排名（长上影=抛压重排后）。

**公式（计算逻辑）**：

```python
"""Compute upper shadow ratio: (high - max(open, close)) / (high - low). Rank negative."""
daily = context.load("daily.parquet")
body_range = daily["high"] - daily["low"]
upper_shadow = daily["high"] - np.maximum(daily["open"], daily["close"])
ratio = upper_shadow / (body_range + 1e-9)
return cross_sectional_rank(-ratio)
```

**意义**：长上影线是典型的日内冲高回落形态，反映空头在高位阻击，大量卖盘在上涨过程中涌现，是短期上涨阻力信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price_deep.py`

##### vol_of_vol_5d

**定义**：波动率之波动（5 日 |日收益| 的标准差）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "vol_of_vol_5d", roll(df, "absret", 5, "std"))
```

**意义**：vol_of_vol_20d/60d 证明波动率二阶矩是强信号；5 日窗口刻画波动率刚启动/收敛的拐点，与 1d 短期反转效应高度相关。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### volume_autocorr_20

**定义**：成交量20日自相关因子：vol与昨日vol的20日滚动相关截面排名（量能惯性排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vol_w = daily["vol"].unstack("Code")
ac = vol_w.rolling(20, min_periods=10).corr(vol_w.shift(1))
return cross_sectional_rank(stack_date_code(ac))
```

**意义**：20日窗口的量能自相关衡量中期资金行为一致性——高=机构建仓/出货的持续性放量结构;低=散户化的随机博弈。与5日版本互补:5日看短节奏、20日看中期行为模式。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### volume_autocorr_5

**定义**：成交量5日自相关因子：vol与昨日vol的5日滚动相关截面排名（量能节奏规律排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vol_w = daily["vol"].unstack("Code")
ac = vol_w.rolling(5, min_periods=3).corr(vol_w.shift(1))
return cross_sectional_rank(stack_date_code(ac))
```

**意义**：成交量的一阶自相关衡量放缩量的节奏规律性——高自相关=量能变化有持续惯性(持续放量或持续缩量,资金行为一致);低/负自相关=量能忽大忽小(消息驱动的离散博弈)。5日窗口捕捉短节奏。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### volume_breakout_confirm_20

**定义**：突破量能确认：20日内突破前20日高点的量比累计截面排名。突破+放量=强确认。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vol = daily["vol"]
adj = _adjusted_close(daily)
prior_high = adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).max().shift(1)
)
breakout = (adj > prior_high) & prior_high.notna()
vol_ma20 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
vol_ratio = safe_divide(vol, vol_ma20)
confirm = (breakout.astype(float) * vol_ratio).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
return cross_sectional_rank(confirm)
```

**意义**：价格突破关键高点后，量能是否配合决定突破的有效性：放量突破=新资金入场接力，缩量突破=大概率假突破。以自建后复权基座判定突破(避免除权日伪突破)，对突破日的 vol/vol_ma20 量比做20日累计，无突破则记0。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_coupling.py`

##### volume_climax

**定义**：放量异动因子，今日成交量/20日均量截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
vol = daily_panel["vol"]
vol_ma_20 = vol.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
vol_ratio = vol / vol_ma_20.replace(0, np.nan)
return cross_sectional_rank(vol_ratio)
```

**意义**：成交量突然放大数倍于平均水平是异动信号——或为机构建仓/出货、或为消息驱动的大规模换手。高量比往往意味着趋势变盘的前兆。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### volume_dry_up

**定义**：缩量因子，-(20日最低成交量/20日均量)截面排名（极度缩量=变盘前兆排前）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
vol = daily_panel["vol"]
vol_min_20 = vol.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).min())
vol_ma_20 = vol.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
dryness = vol_min_20 / vol_ma_20.replace(0, np.nan)
return cross_sectional_rank(-dryness)
```

**意义**：成交量极度萎缩(地量)代表市场交投意愿降至冰点——往往是趋势即将反转的前兆。在下跌趋势中地量=抛压枯竭=可能见底，上涨趋势中地量=追涨意愿不足=可能见顶。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/trend_pattern.py`

##### volume_momentum_5

**定义**：5日成交量动量因子 (量增排前)。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
vol = daily_panel["vol"]
ma_5 = rolling_group_mean(vol, 5)
ma_20 = rolling_group_mean(vol, 20)
ratio = safe_divide(ma_5, ma_20 + 1e-8)
return cross_sectional_rank(ratio)
```

**意义**：成交量短期增长意味着关注度提升，量先于价是A股常见规律

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### volume_price_corr_20

**定义**：量价相关因子（20日滚动收益与对数成交量的相关性，正向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = daily["pct_chg"] / 100.0
log_vol = np.log(daily["vol"].where(daily["vol"] > 0, np.nan))
wide_r = ret.unstack("Code")
wide_v = log_vol.unstack("Code")
corr = wide_r.rolling(20, min_periods=10).corr(wide_v)
return cross_sectional_rank(stack_date_code(corr))
```

**意义**：量价齐升（正相关）表示放量上涨的健康趋势、资金持续参与；量价背离（负相关）=缩量上涨（动能不足）或放量下跌（派发）。采用 skill.md §3.4 的宽表 rolling corr 方案。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/liquidity_factors.py`

##### volume_price_divergence_score

**定义**：量价背离得分因子：价动量排名−量动量排名的背离截面排名（价强量弱背离排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
price_mom = adj.groupby(level="Code").pct_change(20, fill_method=None)
vol_mom = daily["vol"].groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
pr = price_mom.groupby(level="Date").rank(pct=True)
vr = vol_mom.groupby(level="Date").rank(pct=True)
return cross_sectional_rank(pr - vr)
```

**意义**：价格动量与成交量动量的方向背离——价升量缩=上涨缺乏确认(顶部背离预警)；价跌量增=恐慌放量(底部临近)。排名高=价强量弱的可疑上涨，作谨慎信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

##### volume_ratio_20

**定义**：20日相对成交量因子，vol/avg_vol_20 - 1 截面排名。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
vol = daily_panel["vol"]
avg_vol = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
v_ratio = vol / avg_vol.replace(0, np.nan) - 1.0
return cross_sectional_rank(-v_ratio)
```

**意义**：放量上涨和缩量下跌都是技术面确认信号，成交量异常值得关注。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### volume_ratio_zscore_20

**定义**：量比20日Z-score因子（当日量比偏离自身20日均值的标准差数，反向排名）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
volume_ratio = finance["volume_ratio"].clip(0, 20)
z = _zscore_group(volume_ratio, 20, 10)
return cross_sectional_rank(-z)
```

**意义**：finance.volume_ratio 为当日成交量/近5日均量；其自身 z-score 度量量比异常的持续性放大。量比连续异常高企=放量滞涨/派发风险，与 turnover_zscore_20 互为补充（一个是成交量口径、一个是换手率口径）。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/liquidity_factors.py`

##### volume_skew_5d

**定义**：5日量能偏度因子：5日成交量的偏度截面排名（放量脉冲排前，负向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
skew = daily["vol"].groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).skew()
)
return cross_sectional_rank(-skew)
```

**意义**：5日成交量偏度正=量能脉冲式放大(单日巨量主导)，负=均匀放量。正偏度常对应事件驱动的一次性放量，脉冲放量后量能难以持续，负向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

##### volume_surge_3d

**定义**：成交量脉冲因子，3日最大(vol/60日中位数vol)截面排名（负向：脉冲后反转）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
vol = daily_panel["vol"]
vol_median_60 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).median()
)
vol_ratio = vol / vol_median_60.replace(0, np.nan)
surge = vol_ratio.groupby(level="Code").transform(
    lambda s: s.rolling(3, min_periods=1).max()
)
return cross_sectional_rank(-surge)
```

**意义**：成交量短期急剧放大往往是信息冲击或情绪顶点的标志。极端放量后A股存在显著的反转效应——放量脉冲越大，后续回调压力越强。使用中位数而非均值避免极端日污染基线。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### volume_tilt_20

**定义**：量能倾斜：20日量加权收益与等权收益之差截面排名。正倾斜=收益主要来自放量日。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
pct = daily["pct_chg"]
vol = daily["vol"]
wsum = (pct * vol).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
vsum = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
vw = safe_divide(wsum, vsum)
ew = pct.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
tilt = vw - ew
return cross_sectional_rank(tilt)
```

**意义**：若20日收益主要来自放量日(量加权收益>等权收益)，说明上涨由真实资金推动而非小量偷袭，趋势可信度高；若上涨全靠缩量日(倾斜为负)，参与者稀少、反转风险大。该因子量化了'收益的资金确认度'。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_coupling.py`

##### vwap_daily_deviation

**定义**：日频VWAP偏离因子：close/(amount/vol)−1截面排名（收盘高于日均价=尾盘强势排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
vwap = safe_divide(daily["amount"], daily["vol"])
deviation = safe_divide(daily["close"] - vwap, vwap)
return cross_sectional_rank(deviation)
```

**意义**：收盘价相对当日VWAP(成交量加权均价)的偏离——收盘高于VWAP=尾盘买方主导(策略38聪明钱语境中的价格位置)，低于VWAP=尾盘抛压。日频口径的 VWAP 偏离，与分钟级 vwap_deviation 互补。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

##### vwap_dev_1d

**定义**：日频VWAP偏离因子：close/(amount/vol)-1（收盘价高于日均价排前）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "vwap_dev_1d", df["close"] / df["vwap"] - 1.0)
```

**意义**：收盘价相对全日成交均价的位置度量尾盘定价方向——收盘显著高于 VWAP=尾盘买盘占优、资金愿意以高于均价的成本拿货；低于 VWAP=尾盘承压。V8 筛查 meanIC -0.034/|IC|0.13（该池内方向为负，截面排名可逆）。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_cand_daily.py`

##### williams_r_14

**定义**：14日威廉%R因子：(HH14−C)/(HH14−LL14)×(−100)截面排名（超卖排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
wide = adj.unstack("Code")
hh = wide.rolling(14, min_periods=7).max()
ll = wide.rolling(14, min_periods=7).min()
wr = safe_divide(hh - wide, hh - ll + 1e-10) * (-100.0)
return cross_sectional_rank(-stack_date_code(wr))
```

**意义**：威廉%R是KDJ前身的反向随机指标——接近−100=超卖(反弹概率高)，接近0=超买。用复权基座计算14日高低区间，除权日不产生假极值。负向取值=超卖排前。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### winner_rate

**定义**：获利盘比例因子，winner_rate截面排名（取负向=高获利盘为反转信号）。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
return cross_sectional_rank(-perf["winner_rate"])
```

**意义**：高获利盘比例意味着多数持仓者处于盈利状态，短期存在获利了结压力，是反转信号。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### winner_rate_acceleration

**定义**：获利盘比例加速度因子，winner_rate的5日变化截面排名。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
wr = cyq["winner_rate"]
chg_5 = wr.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(chg_5)
```

**意义**：获利盘比例的快速变化反映筹码结构的急剧变化——winner_rate快速上升=大量持仓者从亏损转为盈利，可能触发获利了结；快速下降=恐慌抛售导致深套，可能形成支撑位。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_deep.py`

##### winner_rate_change_20d

**定义**：获利盘比例20日变化。获利盘增加=上涨趋势中筹码逐步盈利，排名高。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
wr = cyq["winner_rate"]
chg = wr.groupby(level="Code").transform(lambda s: s.diff(20))
return cross_sectional_rank(chg)
```

**意义**：获利盘比例(winner_rate)的20日变化捕捉了筹码盈亏结构的边际变化。获利盘比例上升=越来越多的持仓者处于盈利状态——上涨趋势确认；获利盘比例下降=盈利持仓减少——趋势可能转弱。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### winner_rate_change_5d

**定义**：获利盘5日变化因子，winner_rate - winner_rate.shift(5)截面排名（取负向）。

**公式（计算逻辑）**：

```python
perf = context.load("cyq_perf.parquet")
wr = perf["winner_rate"]
change = wr.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(-change)
```

**意义**：获利盘比例快速扩大=上涨过程中筹码加速获利，往往是短期赶顶信号。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip.py`

##### winner_rate_reversal_signal

**定义**：获利盘极端反转因子，-|winner_rate-0.5|截面排名（50%附近=方向不确定=不确定溢价排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
distance = -(cyq["winner_rate"] - 0.5).abs()
return cross_sectional_rank(distance)
```

**意义**：获利盘比例在50%附近时筹码结构最为平衡——多空双方势均力敌，任何方向突破都可能是大行情的起点。极端获利(>90%)或极端亏损(<10%)则方向确定但反转压力也最大。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_deep.py`

##### zero_return_fraction_20

**定义**：零收益占比因子：20日|pct_chg|<0.1%的天数占比截面排名（负向，交投冷淡排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
zero = daily["pct_chg"].abs().lt(0.1).astype(float)
freq = zero.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-freq)
```

**意义**：Amivest流动性视角：零收益(涨跌幅<0.1%)占比高=成交稀疏、价格惰性(流动性差，交易成本高)；占比低=价格活跃、定价充分。流动性差的股票难以交易，负向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/volume_price_dynamics.py`

#### <a name="cat-volume-c1"></a>类别 volume — 成交量（3 个）

##### amt_mom_accel

**定义**：成交额动量加速（3 日变化率 − 10 日变化率）。

**公式（计算逻辑）**：

```python
df = _daily(context)
g = df.groupby("Code")["amount"]
p3 = g.pct_change(3)
p10 = g.pct_change(10)
vals = p3 - p10
return _out(df, "amt_mom_accel", vals)
```

**意义**：量能一阶动量反映资金流入方向，二阶（加速）反映资金行为切换；加速放量与 1d 动量/反转切换点相关。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### amt_surge_3d

**定义**：3 日成交额冲击（当日成交额 / 3 日均额）。

**公式（计算逻辑）**：

```python
df = _daily(context)
ma = df.groupby("Code")["amount"].transform(
    lambda s: s.rolling(3, min_periods=1).mean())
vals = df["amount"] / (ma + 1e-8)
return _out(df, "amt_surge_3d", vals)
```

**意义**：成交额冲击比成交量更贴近资金规模：3 日窗口的突然放量常对应主力进出，次日有方向性延续或反转。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### vol_ratio_ma3

**定义**：3 日量比（当日成交量 / 3 日均量）。

**公式（计算逻辑）**：

```python
df = _daily(context)
mv = df.groupby("Code")["vol"].transform(
    lambda s: s.rolling(3, min_periods=1).mean())
vals = df["vol"] / (mv + 1e-8)
return _out(df, "vol_ratio_ma3", vals)
```

**意义**：vol_ratio_ma5/ma20 的短周期版：3 日量比捕捉当日异动最锐利，放量+位置组合是 1d 短期信号经典来源。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

#### <a name="cat-valuation-c1"></a>类别 valuation — 估值（40 个）

##### bp

**定义**：账面市值比(BP)因子，1/PB截面排名，高值代表价值股。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
bp = 1.0 / finance["pb"].replace(0, np.nan)
return cross_sectional_rank(bp)
```

**意义**：价值因子是Fama-French三因子之一，高BP股票长期有超额收益。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### bp_momentum_20

**定义**：BP动量因子（BP的20日变化率截面排名，BP上升=价值增强排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
pb = finance["pb"].replace(0, np.nan).clip(lower=0.1, upper=100)
bp = 1.0 / pb

delta = bp.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
delta = delta.clip(-0.5, 1.0)
return cross_sectional_rank(delta)  # BP rising = more value = ranks high
```

**意义**：BP的变化来自两方面：价格变化（分母）和净资产变化（分子）。BP短期上升（价格下跌快于净资产或净资产增长）意味着价值属性增强，可能预示着价值回归机会。BP动量捕捉了价值因子的动态变化，比静态BP水平更能反映边际变化。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_deep.py`

##### circ_mv_share_change_20d

**定义**：流通市值占比20日变化。占比增加=限售解禁/减持压力，排名低；占比减少=回购/增持，排名高。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
circ_share = safe_divide(f["circ_mv"], f["total_mv"])
chg = circ_share.groupby(level="Code").transform(lambda s: s.diff(20))
chg = chg.clip(-0.1, 0.1)
return cross_sectional_rank(-chg)
```

**意义**：流通市值占总市值比重的20日变化捕捉了股本结构变化对价格的影响。流通占比增加=限售股解禁或大股东减持(可能带来卖压)；流通占比减少=大股东增持或回购注销(利好信号)。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_deep_extended.py`

##### circ_mv_to_total_mv

**定义**：流通市值/总市值截面排名（高流通比排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
ratio = finance["circ_mv"] / finance["total_mv"].replace(0, np.nan)
return cross_sectional_rank(ratio)
```

**意义**：流通市值占总市值的比例——高流通比意味着限售股压力小、流通性好，低流通比意味着未来解禁后有大量潜在抛压。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/market_structure.py`

##### dp_ttm

**定义**：滚动股息率因子，截面排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
dp = finance["dv_ttm"]
return cross_sectional_rank(dp)
```

**意义**：高股息率股票在低利率环境中具备配置价值，且在下跌市中具有防御属性。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### dv_composite

**定义**：股息率综合因子，dv_ratio与dv_ttm的等权平均截面排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
with np.errstate(invalid="ignore"):
    rank_ratio = finance["dv_ratio"].groupby(level="Date").rank(pct=True)
    rank_ttm = finance["dv_ttm"].groupby(level="Date").rank(pct=True)
composite = (rank_ratio + rank_ttm) / 2.0
return composite.rename("dv_composite")
```

**意义**：dv_ratio（报告期股息率）与dv_ttm（滚动股息率）从不同维度度量股息回报，综合后信号更稳定。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### dv_stability_4q

**定义**：股息稳定性因子，约1季度(60交易日)dv_ratio变异系数截面排名（股息稳定排前）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
dv = fin["dv_ratio"].clip(0, 20)
# Per-stock rolling std/mean over ~60 trading days (approx 1 quarter)
roll_std = dv.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=20).std()
)
roll_mean = dv.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=20).mean()
)
cv = safe_divide(roll_std, roll_mean + 1e-10)
return cross_sectional_rank(-cv)  # stable = low CV
```

**意义**：Dividend stability is as important as dividend level. Companies that maintain stable dividends signal confidence in future cash flows. Erratic dividends signal uncertainty. 2026-08-05 描述修正:实现为 60 交易日(约1季)滚动 CV,原描述误写'4季'。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_extended.py`

##### dv_ttm_rank

**定义**：股息率TTM因子，dv_ttm截面排名（高TTM股息排前）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
dv = fin["dv_ttm"].clip(0, 20)
return cross_sectional_rank(dv)
```

**意义**：TTM dividend yield is slightly more current than dv_ratio. Both measures are highly correlated but dv_ttm updates faster when companies announce new dividend policies. Using both provides a more robust dividend signal.

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_extended.py`

##### dv_yield_rank

**定义**：股息率因子，dv_ratio截面排名（高股息排前）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
dv = fin["dv_ratio"].clip(0, 20)  # winsorize extreme yields
return cross_sectional_rank(dv)
```

**意义**：Dividend yield is a classic value and quality factor. High-dividend stocks in A-shares have become increasingly important under regulatory guidance encouraging dividend payouts. Unlike BP (book-to-price), dividend yield provides a direct cash return to shareholders and is harder to manipulate than earnings. The dv_ratio field in finance.parquet represents trailing 12-month dividend yield and has 0% NaN coverage.

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_extended.py`

##### float_mv_ratio

**定义**：自由流通市值占比因子，free_share/total_share截面排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
ratio = finance["free_share"] / finance["total_share"].replace(0, np.nan)
return cross_sectional_rank(-ratio)
```

**意义**：自由流通盘占比低代表筹码锁定度高、实际流通盘小，可能伴随更高的波动弹性。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### float_share_ratio

**定义**：自由流通股占比取反排名。流通盘小=筹码稀缺+弹性大，排名高。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
ratio = safe_divide(f["float_share"], f["total_share"])
ratio = ratio.clip(0, 1)
return cross_sectional_rank(-ratio)
```

**意义**：流通股/总股本比率取反排名。流通股占比低意味着大量股份锁定期未满或大股东高比例控股，实际可交易盘小——筹码稀缺效应在上涨时放大弹性。该字段(float_share)在现有因子中零引用。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/direct_fields.py`

##### free_float_expansion_20d

**定义**：自由流通股扩张因子：自由流通股/总股本占比的20日变化截面排名（负向，解禁抛压排后）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
ffr = safe_divide(fin["free_share"], fin["total_share"])
change = ffr - ffr.groupby(level="Code").shift(20)
return cross_sectional_rank(-change)
```

**意义**：自由流通股占比的抬升主要来自限售股解禁/转流通(事件驱动研报:解禁后的抛售压力与筹码扩容)——占比上升=潜在供给增加(承压);占比稳定=筹码结构不变。finance.parquet 的 free_share/total_share 为日频字段,20日变化即解禁事件的近似代理,与事件驱动因子族互补。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/finance_extra.py`

##### free_share_turnover_ratio

**定义**：自由流通换手/总换手比。高比值=实际可交易筹码在充分换手,排名高。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
d = context.load("daily.parquet")
common = f.index.intersection(d.index)
# 单位对齐(2026-08-05):daily.vol 单位是手(1手=100股),free_share 单位是股,
# 直接相除整体偏小 100 倍(截面 rank 不受影响,但值与定义不符),故 ×100。
free_to = safe_divide(d.loc[common, "vol"] * 100, f.loc[common, "free_share"])
free_to = free_to.clip(0, 1e10)
return cross_sectional_rank(free_to)
```

**意义**：用自由流通股本(free_share)调整后的换手率: volume / free_share。相比总换手率(turnover_rate=vol/total_share),自由流通换手率更真实反映实际可交易筹码的周转速度——剔除了大股东锁定股份的干扰。free_share字段仅1处引用(valuation.py),该因子是其第二个使用者。

**依赖数据**：`finance.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_deep_extended.py`

##### log_circ_mv

**定义**：对数流通市值因子，负对数流通市值截面排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
log_cmv = np.log(finance["circ_mv"].replace(0, np.nan))
return cross_sectional_rank(-log_cmv)
```

**意义**：流通市值比总市值更精确反映可交易盘规模，对小盘效应捕捉更纯。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### log_mv

**定义**：对数市值（ln(total_mv)，0值置NaN）。

**公式（计算逻辑）**：

```python
df = _fin(context)
return _out(df, "log_mv", df["log_mv"])
```

**意义**：市值是 A 股最经典的风险因子：小市值长期有溢价，对数化压缩右尾，作为规模基准参与截面定价。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### log_mv_chg_20d

**定义**：市值20日变化（log_mv 的20日diff）。

**公式（计算逻辑）**：

```python
df = _fin(context)
vals = df.groupby("Code")["log_mv"].diff(20)
return _out(df, "log_mv_chg_20d", vals)
```

**意义**：对数市值变化近似 20 日累计收益率（市值维度），与价格动量互补，捕捉股本变动（增发/解禁）带来的规模跳变。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### log_total_mv

**定义**：对数总市值因子，负对数总市值（小市值排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
log_mv = np.log(finance["total_mv"].replace(0, np.nan))
return cross_sectional_rank(-log_mv)
```

**意义**：规模因子是A股最显著的单因子之一，小市值效应长期存在。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### margin_proxy_ttm

**定义**：净利率代理因子：ps_ttm/pe_ttm(市值恒等式=净利/营收)截面排名（高净利率排前）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
pe = fin["pe_ttm"]
ps = fin["ps_ttm"]
raw = safe_divide(ps, pe.where(pe > 0))
return cross_sectional_rank(raw)
```

**意义**：ps_ttm/pe_ttm = (市值/营收)/(市值/净利) = 净利率——无财报白名单下用市值恒等式构造盈利质量维度:银行白酒(净利率 30%+)与低毛利制造业(<5%)在此分离,是 pe_ttm/pb 之外的独立质量维度。负 PE 置 NaN(与既有 PE 类因子同口径,NaN≈17% 合格)。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### market_cap_concentration_20d

**定义**：市值集中度因子，log_total_mv的20日波动率截面排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
mv = np.log(finance["total_mv"].replace(0, np.nan))
mv_vol_20 = mv.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
return cross_sectional_rank(-mv_vol_20)
```

**意义**：市值在短期内的剧烈波动反映公司基本面的不确定性——市值频繁大幅波动意味着市场对公司价值的共识度低、信息不对称高。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/market_structure.py`

##### pb_change_20d

**定义**：市净率变化因子：pb的20日变化率截面排名（负向，估值抬升排后）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
pb_lag = fin["pb"].groupby(level="Code").shift(20)
change = safe_divide(fin["pb"] - pb_lag, pb_lag)
return cross_sectional_rank(-change)
```

**意义**：pb 的 20 日变化率衡量估值扩张/收缩的速度——pb 快速抬升=股价涨幅超过净资产增长(估值透支,未来回报被摊薄);pb 收缩=股价相对净资产走低(估值回归价值区)。与 pe_ttm_change_20d(盈利口径)互补,构成估值动量维度。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/finance_extra.py`

##### pb_turnover_regime

**定义**：PB-换手率状态因子（低PB+高换手=价值重估排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")

pb = finance["pb"].replace(0, np.nan).clip(lower=0.1, upper=100)
turnover = finance["turnover_rate"].clip(0, 50)

pb_rank = pb.groupby(level="Date").rank(pct=True)  # high = expensive
to_rank = turnover.groupby(level="Date").rank(pct=True)

value_score = (1.0 - pb_rank)
activity_bonus = 0.5 + 0.5 * to_rank
regime_score = value_score * activity_bonus

return cross_sectional_rank(regime_score)
```

**意义**：PB与换手率的交互与PE逻辑类似但侧重资产价值维度。低PB+高换手=资产价值被重新发现（排前）；低PB+低换手=资产价值被长期忽视（中性）；高PB+低换手=高ROE的合理溢价（中性偏正）；高PB+高换手=高估值下的筹码博弈（排后）。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_deep.py`

##### pe_pb_divergence

**定义**：PB与PE百分位排名差。正偏离=隐含ROE较高，负偏离=隐含ROE较低。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
pe_rank = f["pe_ttm"].groupby(level="Date").rank(pct=True)
pb_rank = f["pb"].groupby(level="Date").rank(pct=True)
divergence = pb_rank - pe_rank
return cross_sectional_rank(divergence)
```

**意义**：PB与PE截面百分位排名的差值。由ROE≈PB/PE可知，PB相对PE越高通常对应更高的隐含净资产收益率；反之则可能对应重资产、低盈利或周期高点。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/direct_fields.py`

##### pe_ttm_absolute

**定义**：PE_TTM原始值截面排名取负，低PE=价值信号（当期所有股票的截面比较）。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
pe = f["pe_ttm"].clip(0, 500)
return cross_sectional_rank(-pe)
```

**意义**：PE_TTM原始值截面排名取负。低PE是经典的价值信号——低PE股票在A股中长期有超额收益。本因子做当期截面比较(当期所有股票PE排序)。注:finance.parquet 的pe_ttm_percentile 字段(历史分位)属禁用数据源,不用于任何因子。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/direct_fields.py`

##### pe_ttm_change_20d

**定义**：PE_TTM 20日变化率取反，估值收缩=价值改善，估值扩张=均值回归风险。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
pe = f["pe_ttm"]
chg = pe.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
chg = chg.clip(-0.5, 1.0)
return cross_sectional_rank(-chg)
```

**意义**：PE_TTM的20日变化率取反排名。PE下降(盈利增长或价格下跌使估值趋于合理)意味着估值回归，是价值改善的信号；PE上升(估值扩张)意味着股价上涨快于盈利增长，可能面临均值回归压力。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/direct_fields.py`

##### ps_ttm_momentum_20d

**定义**：PS_TTM 20日变化率取反，PS下降=变便宜，是价值改善信号。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
ps = f["ps_ttm"]
chg = ps.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
chg = chg.clip(-0.5, 1.0)
return cross_sectional_rank(-chg)
```

**意义**：PS_TTM的20日变化率取反排名。PS_TTM下降意味着股价下跌快于收入增长——公司变得更便宜，是价值回归的潜在信号。PS_TTM上升意味着估值扩张。PS不受利润波动影响，比PE更稳定。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/direct_fields.py`

##### ps_ttm_rank

**定义**：市销率TTM因子，ps_ttm截面排名（低市销率排前）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
ps = fin["ps_ttm"].clip(0, 500)
return cross_sectional_rank(-ps)  # low P/S ranks high
```

**意义**：Price-to-Sales TTM is a fundamental valuation metric that works for companies with negative earnings (where PE fails). Low P/S stocks tend to be overlooked by earnings-focused investors and can offer significant upside when profitability improves. TTM variant is more responsive than annual P/S. 0% NaN -- excellent coverage.

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_extended.py`

##### rel_log_mv_ind

**定义**：行业内相对市值（log_mv − 行业等权均值）。

**公式（计算逻辑）**：

```python
df = _fin(context)
vals = df["log_mv"] - ind_mean(df, "log_mv")
return _out(df, "rel_log_mv_ind", vals)
```

**意义**：行业内相对规模：同一行业中偏小的标的弹性更大，剥离板块整体市值水平后规模效应更纯粹。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### rel_pb_ind

**定义**：行业内相对市净率（pb − 行业等权均值，0值置NaN）。

**公式（计算逻辑）**：

```python
df = _fin(context)
vals = df["pb_n"] - ind_mean(df, "pb_n")
return _out(df, "rel_pb_ind", vals)
```

**意义**：行业内相对 PB 刻画净资产定价差异，低相对 PB 标的具备价值修复弹性。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### rel_pe_ind

**定义**：行业内相对估值（pe_ttm − 行业等权均值，0值置NaN）。

**公式（计算逻辑）**：

```python
df = _fin(context)
vals = df["pe_ttm_n"] - ind_mean(df, "pe_ttm_n")
return _out(df, "rel_pe_ind", vals)
```

**意义**：同一行业内相对便宜的股票（负偏离大）估值修复空间更大，绝对 PE 受行业属性干扰大，行业内相对估值更可比。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### sp_raw

**定义**：未调整市销率因子（低值排前），ps = 总市值/最近报告期营收。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
return cross_sectional_rank(-finance["ps"])
```

**意义**：ps字段提供了一种不同于ps_ttm的市销率计算口径（使用最近一期报告而非TTM），两者之间的差异本身包含了信息（TTM调整的方向和幅度）。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/unused_fields_factors.py`

##### sp_ttm

**定义**：市销率倒数(SP_TTM)因子，1/PS_TTM截面排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
sp = 1.0 / finance["ps_ttm"].replace(0, np.nan)
return cross_sectional_rank(sp)
```

**意义**：PS估值对净利润为负的公司仍有定义域，在A股覆盖面优于PE类因子。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### sp_ttm_momentum_20

**定义**：SP_TTM动量因子（1/PS_TTM的20日变化率截面排名，SP上升排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
ps = finance["ps_ttm"].replace(0, np.nan).clip(lower=0.1, upper=500)
sp = 1.0 / ps

delta = sp.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
delta = delta.clip(-0.5, 1.0)
return cross_sectional_rank(delta)
```

**意义**：SP_TTM（市销率倒数）的变化反映了销售额相对价格的变化。SP上升可能源于：收入增长（基本面改善）或价格下跌（变得便宜）。与BP动量互补，SP动量对轻资产、高PB公司（如科技、消费）更为有效。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_deep.py`

##### turnover_20

**定义**：20日平均换手率因子（总股本换手率），低换手排前。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate"]
avg_turnover = turnover.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-avg_turnover)
```

**意义**：低换手率反映筹码稳定、投机度低，在A股中具有正向截面预测力。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### turnover_f_20

**定义**：20日平均自由流通换手率因子，低换手排前。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate_f"]
avg_turnover = turnover.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-avg_turnover)
```

**意义**：自由流通换手率剔除大股东锁定股份，更精确反映真实交易活跃度。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### turnover_f_delta_5

**定义**：自由流通换手率5日变化因子（换手率下降=浮筹减少排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate_f"]
delta = turnover.groupby(level="Code").transform(
    lambda s: s.pct_change(5, fill_method=None)
)
delta = delta.clip(-1, 3)
return cross_sectional_rank(-delta)
```

**意义**：自由流通换手率的短期下降意味着浮动筹码被逐步吸收，筹码趋于集中。换手率持续下降往往伴随价格筑底，是筹码锁定度改善的领先信号。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/direct_fields.py`

##### turnover_f_divergence

**定义**：自由流通换手率/总换手率。>1=交易集中于自由流通盘，存量筹码活跃，排名高。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
ratio = safe_divide(f["turnover_rate_f"], f["turnover_rate"])
ratio = ratio.clip(0, 2)
return cross_sectional_rank(ratio)
```

**意义**：自由流通换手率/总股本换手率反映交易在自由流通盘中的集中度。比率>1意味着交易高度集中于自由流通盘、存量筹码充分交换——筹码活跃度高；比率<1意味着大股东也在参与交易(异常信号)。该比率捕捉了交易结构的质量差异。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_deep_extended.py`

##### turnover_f_raw

**定义**：自由流通换手率原始值因子（低换手排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate_f"]
turnover = turnover.clip(0, 50)
return cross_sectional_rank(-turnover)
```

**意义**：turnover_rate_f剔除大股东锁定股份，更精确反映真实可交易盘的换手活跃度。现有因子中仅turnover_f_20使用了20日均值，该因子使用每日原始值以更及时地捕捉自由流通盘的交易强度变化。低换手代表筹码稳定、投机度低，在A股截面中具有正向预测力。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/direct_fields.py`

##### turnover_vol_20

**定义**：20日换手率波动因子，换手率标准差截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate"]
vol = turnover.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
return cross_sectional_rank(-vol)
```

**意义**：换手率剧烈波动常反映资金博弈激烈，是风险信号。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### volume_ratio

**定义**：量比因子（当日成交量相对5日均量），截面排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
vol_ratio = finance["volume_ratio"]
return cross_sectional_rank(-vol_ratio)
```

**意义**：量比是盘中常用指标，极端量比常伴随短期反转。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation.py`

##### volume_ratio_extreme

**定义**：量比极端值因子取反。异常放量(vr>3)=量能衰竭→反转，异常缩量(vr<0.3)=无人问津→可能启动。

**公式（计算逻辑）**：

```python
f = context.load("finance.parquet")
vr = f["volume_ratio"]
extreme = (vr - 1.0).abs()
# Only flag when vr is truly extreme (>2.5 or <0.4)
is_extreme = (vr > 2.5) | (vr < 0.4)
signal = extreme * is_extreme.astype(float)
return cross_sectional_rank(-signal)
```

**意义**：量比极端偏离1的程度取反排名。量比>3(异常放量)往往伴随短期量能衰竭后的反转——放量见顶；量比<0.3(异常缩量)意味着无人问津但地量见地价——可能反转启动。该因子捕捉极端成交量的均值回归特性。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_deep_extended.py`

#### <a name="cat-fund_flow-c1"></a>类别 fund_flow — 资金流（97 个）

##### big_order_net_accel_10

**定义**：大单净额加速度因子：近5日大单净额−前5日大单净额(占成交额比)截面排名（资金加速流入排前）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
daily = context.load("daily.parquet")
x = (
    mf["buy_lg_amount"] + mf["buy_elg_amount"]
    - mf["sell_lg_amount"] - mf["sell_elg_amount"]
)
net5 = x.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).sum()
)
net10 = x.groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=5).sum()
)
prev5 = net10 - net5
amount10 = daily["amount"].groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=5).sum()
).reindex(x.index)
accel = safe_divide((net5 - prev5) * 1e4, amount10)
return cross_sectional_rank(accel)
```

**意义**：大单净流入的边际变化比水平更有信息:加速流入=机构刚进场(最佳跟随点),减速=流入尾声(追高风险)。与 mf_flow_acceleration_5d(全层级 net_mf)区分:本因子专看 lg+elg 档,剔除散户/中单噪声。以 10 日成交额归一,跨市值可比。单位:万元×1e4 转元后与 amount 同尺度。

**依赖数据**：`main_fund_flow.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### big_vs_small_divergence_5d

**定义**：大小单背离5日因子，(大单净流入率-小单净流入率)的5日变化截面排名。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
big_net = (mf["buy_lg_amount"] + mf["buy_elg_amount"] - mf["sell_lg_amount"] - mf["sell_elg_amount"]) / _total_amount(mf)
small_net = (mf["buy_sm_amount"] - mf["sell_sm_amount"]) / _total_amount(mf)
divergence = big_net - small_net
div_5d = divergence.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(div_5d)
```

**意义**：大单资金与小单资金方向的背离变化是聪明的信号——大单资金相对小单的流入加速意味着机构/大户正在加速建仓，而散户可能还在犹豫或减持。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### elg_net_60d_to_mv

**定义**：超大单60日净买入占流通市值比因子：Σ(超大单买−卖,60日)×1e4/流通市值截面排名（长线吸筹强度排前）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
fin = context.load("finance.parquet")
net = (mf["buy_elg_amount"] - mf["sell_elg_amount"]) * 1e4
net60 = net.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=20).sum()
)
circ_mv = fin["circ_mv"].reindex(net60.index)
raw = safe_divide(net60, circ_mv)
return cross_sectional_rank(raw)
```

**意义**：无北向持股数据白名单下,超大单(elg)是外资/产业资本的最佳代理——60 日累计净买入占流通市值比衡量长线吸筹强度(策略73跟随北向资金的日频替代)。与 mf_cumulative_flow_20d(20日全口径)区分:60 日 elg 专属口径更长更纯,过滤短线噪音。单位:金额万元×1e4 转元,circ_mv=元。

**依赖数据**：`main_fund_flow.parquet`、`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### ext_mf_amount_concentration

**定义**：成交额集中度因子（低值排前），各订单规模成交额占比的Herfindahl指数。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total = _total_amount(ff)

sm_share = (ff["buy_sm_amount"] + ff["sell_sm_amount"]) / total
md_share = (ff["buy_md_amount"] + ff["sell_md_amount"]) / total
lg_share = (ff["buy_lg_amount"] + ff["sell_lg_amount"]) / total
elg_share = (ff["buy_elg_amount"] + ff["sell_elg_amount"]) / total

# Herfindahl-Hirschman Index: sum of squared shares
hhi = sm_share**2 + md_share**2 + lg_share**2 + elg_share**2
return cross_sectional_rank(-hhi)  # lower concentration = higher rank
```

**意义**：成交额在各订单规模间的分布集中度反映了市场参与者的多样性。高度集中（如大单占比>70%）意味着单一资金类型主导，行情持续性存疑。适度分散的成交结构更健康（各类资金共同参与）。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_extended.py`

##### ext_mf_big_order_net_amount_ratio

**定义**：大单净买入金额占比因子，大单+特大单净买入额/总成交额截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
big_net_amt = (
    ff["buy_lg_amount"] - ff["sell_lg_amount"]
    + ff["buy_elg_amount"] - ff["sell_elg_amount"]
)
ratio = safe_divide(big_net_amt, _total_amount(ff))
return cross_sectional_rank(ratio)
```

**意义**：成交额口径的大单净买入占比与成交量口径（mf_big_order_vol_ratio）互补：成交额口径考虑了价格信息，反映了机构资金的'质量'（高价买入=更强的信心）。两个口径的差异本身就是重要信号（参考mf_amount_vol_divergence）。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_extended.py`

##### ext_mf_large_order_avg_price

**定义**：大单均价因子，大单+特大单成交均价/VWAP截面排名（高值排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total = _total_amount(ff)
total_vol = (
    ff["buy_sm_vol"] + ff["sell_sm_vol"]
    + ff["buy_md_vol"] + ff["sell_md_vol"]
    + ff["buy_lg_vol"] + ff["sell_lg_vol"]
    + ff["buy_elg_vol"] + ff["sell_elg_vol"]
).replace(0, np.nan)

vwap = total / total_vol

lg_total_vol = (
    ff["buy_lg_vol"] + ff["sell_lg_vol"]
    + ff["buy_elg_vol"] + ff["sell_elg_vol"]
)
lg_total_amt = (
    ff["buy_lg_amount"] + ff["sell_lg_amount"]
    + ff["buy_elg_amount"] + ff["sell_elg_amount"]
)
lg_avg_price = safe_divide(lg_total_amt, lg_total_vol)

ratio = safe_divide(lg_avg_price, vwap)
return cross_sectional_rank(ratio)
```

**意义**：大单成交均价高于VWAP意味着机构愿意以高于市场均价的价格买入——这是机构做多意愿强烈的信号。机构低价买入是正常行为，高价买入是超常行为。大单均价/VWAP的比率是机构买卖紧迫度的代理变量。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_extended.py`

##### ext_mf_medium_order_amount_ratio

**定义**：中单成交额占比因子，中单（4-20万）成交额/总成交额截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
md_amount = ff["buy_md_amount"] + ff["sell_md_amount"]
ratio = safe_divide(md_amount, _total_amount(ff))
return cross_sectional_rank(ratio)
```

**意义**：中单成交额占比反映了中等资金（游资/大户）的参与程度。与散户和机构不同，中等资金的行为模式更加灵活，是市场活跃度的中间层指标。中单占比异常波动往往预示着资金结构的变化。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_extended.py`

##### ext_mf_small_order_amount_ratio

**定义**：小额订单成交额占比因子，散户参与度度量（低值排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
sm_amount = ff["buy_sm_amount"] + ff["sell_sm_amount"]
ratio = safe_divide(sm_amount, _total_amount(ff))
return cross_sectional_rank(-ratio)
```

**意义**：小额订单（<4万元）成交额占比高意味着散户交易活跃。散户占比高的股票往往存在行为偏差（过度交易、追涨杀跌），机构化的股票小额订单占比低。低散户占比是A股机构化趋势下的质量信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_extended.py`

##### ext_mf_small_order_avg_price

**定义**：小额订单均价因子（低值排前），小额成交均价/VWAP截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total = _total_amount(ff)
total_vol = (
    ff["buy_sm_vol"] + ff["sell_sm_vol"]
    + ff["buy_md_vol"] + ff["sell_md_vol"]
    + ff["buy_lg_vol"] + ff["sell_lg_vol"]
    + ff["buy_elg_vol"] + ff["sell_elg_vol"]
).replace(0, np.nan)

vwap = total / total_vol  # volume-weighted average price

sm_total_vol = ff["buy_sm_vol"] + ff["sell_sm_vol"]
sm_total_amt = ff["buy_sm_amount"] + ff["sell_sm_amount"]
sm_avg_price = safe_divide(sm_total_amt, sm_total_vol)

ratio = safe_divide(sm_avg_price, vwap)
# Deviation from 1 (overpaying or underpaying relative to VWAP)
deviation = (ratio - 1.0).abs()
return cross_sectional_rank(-deviation)
```

**意义**：小额订单的成交均价与VWAP的比较反映散户交易的执行质量。散户成交均价偏高（>VWAP）可能意味着追高买入行为，散户成交均价偏低（<VWAP）可能是恐慌性抛售。该比率偏离1的程度是散户情绪的温度计。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_extended.py`

##### fund_flow_volatility_20

**定义**：资金流波动性因子，-(net_mf_amount/vol的20日标准差)截面排名（资金流不稳定排后）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
net_ratio = mf["net_mf_amount"] / _total_amount(mf)
vol_20 = net_ratio.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
return cross_sectional_rank(-vol_20)
```

**意义**：主力资金净流向频繁变换方向意味着多空分歧大、主力意图不明确——资金流方向稳定是趋势确定性高的信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### large_order_timing_signal

**定义**：大单时机信号因子，大单净买入/成交量×20日价格位置截面排名（低位大单流入=最佳买点排前）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
daily_panel = context.load("daily.parquet")
big_net = (mf["buy_lg_amount"] + mf["buy_elg_amount"] - mf["sell_lg_amount"] - mf["sell_elg_amount"]) / _total_amount(mf)
# 20 日价格位置用后复权基座计算,除权日未复权 close 的区间高低点不产生假低位 (2026-08-05)
adj = _adjusted_close(daily_panel)
high_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max())
low_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).min())
position = (adj - low_20) / (high_20 - low_20).replace(0, np.nan)
# Low position (close to low) + positive big net = strong buy signal
signal = big_net * (1 - position)
return cross_sectional_rank(signal)
```

**意义**：大单流入配合价格在低位是最优的信号组合——机构在低位大额建仓意味着他们对当前价格水平认可，且预期未来上涨空间大。

**依赖数据**：`main_fund_flow.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### lg_sm_divergence

**定义**：大小单背离因子，(大单净买-小单净卖)/总成交额截面排名（机构买+散户卖=最佳组合排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
big = _big_net(ff)
small = _small_net(ff)
total = _total_amount(ff)
divergence = (big - small) / total
return cross_sectional_rank(divergence)
```

**意义**：大单净买入同时小单净卖出是'聪明钱在吸、散户在抛'的背离信号——这种组合表明筹码正在从弱手（散户）转移到强手（机构），是经典的底部吸筹特征。反向（大单卖+小单买）则是顶部出货信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep.py`

##### margin_balance_20d

**定义**：融资余额20日变化率，反映中期杠杆资金趋势。稳步增长=持续看多共识，排名高。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
chg = m["rzye"].groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
chg = chg.clip(-0.5, 1.0)
return cross_sectional_rank(chg)
```

**意义**：融资余额20日变化率过滤了短期噪音，反映中期杠杆资金趋势方向。稳步增长的融资余额代表持续的看多共识，比5日变化更可靠地捕捉中线资金态度。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_balance_5d

**定义**：融资余额5日变化率，反映杠杆资金短期流入/流出速度。余额增长=杠杆做多，排名高。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
chg = m["rzye"].groupby(level="Code").transform(
    lambda s: s.pct_change(5, fill_method=None)
)
chg = chg.clip(-0.5, 1.0)
return cross_sectional_rank(chg)
```

**意义**：融资余额5日变化率捕捉杠杆资金的短期边际变化。余额快速增长意味着融资盘积极做多，是短期看多信号；余额下降意味着融资盘去杠杆，预示短期承压。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_balance_ma_divergence

**定义**：融资余额偏离20日均线幅度，极端偏离预示均值回归。正向偏离=可能超买，排名居中。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
rzye = m["rzye"]
ma20 = rzye.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
div = safe_divide(rzye - ma20, ma20)
div = div.clip(-0.1, 0.1)
return cross_sectional_rank(div)
```

**意义**：融资余额偏离20日均线的幅度反映杠杆资金的极端情绪。正向偏离(余额在均线上方)意味着融资盘快速涌入、可能超买；负向偏离意味着融资盘撤离、可能超卖。这是一个均值回归信号——极端偏离后倾向于回归。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_balance_volatility_20d

**定义**：融资余额20日波动率(变异系数)，反映杠杆资金稳定性。低波动=方向一致，排名高(取反)。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
rzye = m["rzye"]
roll_std = rzye.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
roll_mean = rzye.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
cv = safe_divide(roll_std, roll_mean)
cv = cv.clip(0, 0.5)
return cross_sectional_rank(-cv)
```

**意义**：融资余额20日波动率(变异系数=std/mean)衡量杠杆资金的稳定性。高波动反映融资盘多空观点频繁变化、持仓不稳定；低波动意味着融资盘方向一致、持仓信心稳定。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_buy_momentum_5d

**定义**：融资买入5日均值相对余额动量。持续买入=信号可靠，排名高。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
buy_ma5 = m["rzmre"].groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).mean()
)
momentum = safe_divide(buy_ma5, m["rzye"])
momentum = momentum.clip(0, 0.5)
return cross_sectional_rank(momentum)
```

**意义**：融资买入5日均值相对余额反映买入行为的持续性。持续的高买入动量意味着杠杆资金在一段时间内保持积极进场态势，而非单日脉冲，信号更可靠。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_buy_pressure

**定义**：融资买入压力比=融资买入/(融资买入+融资偿还)。>0.5=买入意愿强于偿还，排名高。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
total = m["rzmre"] + m["rzche"]
pressure = safe_divide(m["rzmre"], total)
return cross_sectional_rank(pressure)
```

**意义**：融资买入压力比率衡量杠杆资金的日内买卖倾向。比率>0.5意味着买入意愿强于偿还意愿，杠杆资金在主动加仓；比率<0.5意味着偿还压力大于买入意愿，杠杆资金在减仓或被动偿还。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_buyer_avg_cost_premium

**定义**：融资盘成本溢价因子：现价相对近20日融资买入加权平均成本截面排名（融资盘浮盈排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
margin = context.load("margin_detail.parquet")
cost = _margin_weighted_cost(margin, daily)
close = daily["close"].reindex(cost.index)
raw = safe_divide(close, cost) - 1.0
return cross_sectional_rank(raw)
```

**意义**：现价低于融资盘平均建仓成本=融资盘被套(强平/止损压力随时释放);高于=融资盘浮盈(惜售)。融资盘是 A 股最活跃的杠杆资金,其盈亏状态直接影响后续抛压。margin 数据已 shift(1),配对价格用 close 逐股 shift(1)(T-1 收盘),严格对齐无未来函数。与既有 margin_* 量额类因子区分:本因子是价格×金额的加权成本维度。

**依赖数据**：`margin_detail.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### margin_chg_abs_5d

**定义**：融资余额5日变化率（rzye 5日pct_change，PIT 平移后）。

**公式（计算逻辑）**：

```python
df = _margin(context)
return _out(df, "margin_chg_abs_5d", df["rz_chg"])
```

**意义**：融资余额变化捕捉杠杆资金边际动向：余额增长=杠杆做多升温，下降=去杠杆承压。本因子为个股自身变化率。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_margin.py`

##### margin_chg_rel_ind_5d

**定义**：融资余额5日变化率行业相对（减去行业等权均值，PIT 平移后）。

**公式（计算逻辑）**：

```python
df = _margin(context)
vals = df["rz_chg"] - ind_mean(df, "rz_chg")
return _out(df, "margin_chg_rel_ind_5d", vals)
```

**意义**：行业内相对杠杆变化：同一板块中融资余额加速增长的个股=资金更看好，剥离板块整体杠杆环境。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_margin.py`

##### margin_chip_cost_gap

**定义**：融资盘-筹码成本差因子：融资盘平均成本/市场筹码平均成本−1截面排名（负向，杠杆盘高位接盘排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
margin = context.load("margin_detail.parquet")
cyq = context.load("cyq_perf.parquet")
cost = _margin_weighted_cost(margin, daily)
weight_avg = cyq["weight_avg"].reindex(cost.index)
gap = safe_divide(cost, weight_avg) - 1.0
return cross_sectional_rank(-gap)
```

**意义**：融资盘平均建仓成本相对 cyq_perf 筹码平均成本(weight_avg)的位置:融资成本显著高于市场平均=杠杆盘在高位接盘(后续易引发止损踩踏);低于市场平均=杠杆盘低位建仓(安全边际高)。同一只股票筹码结构不变时,本因子给出融资盘的增量位置信息,与 margin_buyer_avg_cost_premium(现价视角)互补为成本结构双视角。

**依赖数据**：`margin_detail.parquet`、`daily.parquet`、`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### margin_flow_asymmetry_10d

**定义**：10日累计融资净买入/累计总交易额，资金流向方向持续性。持续净买入=排名高。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
net = m["rzmre"] - m["rzche"]
total = m["rzmre"] + m["rzche"]
net_10d = net.groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=5).sum()
)
total_10d = total.groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=5).sum()
)
asymmetry = safe_divide(net_10d, total_10d)
asymmetry = asymmetry.clip(-1, 1)
return cross_sectional_rank(asymmetry)
```

**意义**：过去10日累计融资净买入/(累计买入+累计偿还)衡量融资净流入的方向持续性。接近1意味着持续10日几乎全是净买入——杠杆资金方向极其一致；接近-1意味着持续净卖出。资金流向的持续性比单日方向更能预测未来走势。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin_deep.py`

##### margin_leverage_change_20d

**定义**：融资杠杆变化因子：融资余额/流通市值占比的20日变化截面排名（杠杆资金加仓排前）。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
fin = context.load("finance.parquet")
# ⚠️ 必须 reindex 到 margin 面板索引:直接相除会做索引并集,
# 把无融资数据股票的 25% 缺行以 NaN 拉进结果(与既有 margin 因子同口径,
# .fea 仅覆盖融资标的,NaN 由覆盖范围决定而非对齐方式)。
total_mv_m = fin["total_mv"].reindex(m.index)
leverage = safe_divide(m["rzye"], total_mv_m)
change = leverage.groupby(level="Code").diff(20)
return cross_sectional_rank(change)
```

**意义**：融资余额相对流通市值的占比是杠杆资金参与度的标准度量——占比20日上升=杠杆资金持续加仓(增量资金确认,常伴随行情启动);下降=杠杆资金撤退。与总杠杆水平(total_leverage_ratio)互补:本因子捕捉方向而非水平。Date=T 使用可获得的 T-1 融资数据与 T 日市值。

**依赖数据**：`margin_detail.parquet`、`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin_extra.py`

##### margin_leverage_trend_10d

**定义**：融资融券总余额10日变化率。总杠杆增加=风险偏好提升，排名高。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
chg = m["rzrqye"].groupby(level="Code").transform(
    lambda s: s.pct_change(10, fill_method=None)
)
chg = chg.clip(-0.3, 0.3)
return cross_sectional_rank(chg)
```

**意义**：融资融券总余额(rzrqye=rzye+rqye)10日变化率反映整体杠杆水平变化趋势。总杠杆增加意味着市场风险偏好提升；总杠杆减少意味着避险情绪升温。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_net_flow_ratio

**定义**：融资净流入相对余额比=(融资买入-融资偿还)/融资余额，标准化净流量强度。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
net_flow = m["rzmre"] - m["rzche"]
ratio = safe_divide(net_flow, m["rzye"])
ratio = ratio.clip(-0.1, 0.1)
return cross_sectional_rank(ratio)
```

**意义**：融资净流入相对余额的比率标准化了净流量的强度，使得不同融资余额规模的股票可比。高比率说明新的杠杆资金在加速涌入，是增量资金信号；负值意味着杠杆资金净流出。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_repay_deceleration

**定义**：融资偿还额5日变化率取反。偿还减速=空方力量减弱，排名高。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
chg = m["rzche"].groupby(level="Code").transform(
    lambda s: s.pct_change(5, fill_method=None)
)
chg = chg.clip(-0.5, 0.5)
return cross_sectional_rank(-chg)
```

**意义**：融资偿还额5日变化率的反向排名。偿还额下降(正排名)意味着杠杆资金不再急于平仓——看空力量减弱、持筹信心恢复。偿还额加速上升意味着恐慌性平仓、信心崩溃。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### margin_repay_shock

**定义**：融资偿还冲击=当日偿还额/20日均偿还额取反。突然放大=恐慌平仓，排名低。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
rzche = m["rzche"]
repay_ma20 = rzche.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
shock = safe_divide(rzche, repay_ma20)
shock = shock.clip(0, 5)
return cross_sectional_rank(-shock)
```

**意义**：单日融资偿还额相对20日均值的倍数取反。偿还突然放大(偿还冲击)意味着融资盘集中平仓——可能是股价触发平仓线或投资者主动止损。偿还冲击越大的股票越看空(排名取反)。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin_deep.py`

##### margin_velocity

**定义**：融资周转速度=(融资买入+融资偿还)/融资余额。高速度=投机性强，排名高。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
total_flow = m["rzmre"] + m["rzche"]
velocity = safe_divide(total_flow, m["rzye"])
velocity = velocity.clip(0, 2)
return cross_sectional_rank(velocity)
```

**意义**：融资周转速度反映融资盘的交易活跃度。高速度意味着融资盘换手频繁——投机性强、资金快进快出；低速度意味着融资盘锁定不动——筹码稳定性高、持仓信心强。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin.py`

##### medium_order_flow

**定义**：中单资金流因子，中单净买入/总成交额截面排名。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
md_net = (mf["buy_md_amount"] - mf["sell_md_amount"]) / _total_amount(mf)
return cross_sectional_rank(md_net)
```

**意义**：中单资金流往往被忽视——中单代表中等资金量级的投资者行为，在大单和小单之间提供了额外信息维度。中单净流入可能是机构隐藏建仓意图的手段。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_amount_vol_divergence

**定义**：资金流量价背离因子，（大单净买入额占比 - 大单净买入量占比）截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
# Big order net amount ratio
big_net_amt = (
    ff["buy_lg_amount"] - ff["sell_lg_amount"]
    + ff["buy_elg_amount"] - ff["sell_elg_amount"]
)
total_amt = (
    ff["buy_sm_amount"] + ff["sell_sm_amount"]
    + ff["buy_md_amount"] + ff["sell_md_amount"]
    + ff["buy_lg_amount"] + ff["sell_lg_amount"]
    + ff["buy_elg_amount"] + ff["sell_elg_amount"]
).replace(0, np.nan)
amt_ratio = big_net_amt / total_amt

# Big order net volume ratio
big_net_vol = (
    ff["buy_lg_vol"] - ff["sell_lg_vol"]
    + ff["buy_elg_vol"] - ff["sell_elg_vol"]
)
vol_ratio = big_net_vol / _total_vol(ff)

divergence = amt_ratio - vol_ratio
return cross_sectional_rank(divergence)
```

**意义**：成交额占比与成交量占比的差异反映了不同价位上的交易分布。大单净买入额占比 > 大单净买入量占比意味着大资金偏好高价买入（更强的买入意愿）。额量背离是聪明钱强度的重要信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/unused_fields_factors.py`

##### mf_amount_weighted_direction

**定义**：金额加权方向复合因子，四档订单方向信号按金额占比加权求和截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_amt = _total_amount(ff)

# Direction sign for each tier: +1 if buy > sell, -1 if sell > buy
sm_dir = np.sign(ff["buy_sm_amount"] - ff["sell_sm_amount"])
md_dir = np.sign(ff["buy_md_amount"] - ff["sell_md_amount"])
lg_dir = np.sign(ff["buy_lg_amount"] - ff["sell_lg_amount"])
elg_dir = np.sign(ff["buy_elg_amount"] - ff["sell_elg_amount"])

# Amount proportion weight for each tier
sm_w = (ff["buy_sm_amount"] + ff["sell_sm_amount"]) / total_amt
md_w = (ff["buy_md_amount"] + ff["sell_md_amount"]) / total_amt
lg_w = (ff["buy_lg_amount"] + ff["sell_lg_amount"]) / total_amt
elg_w = (ff["buy_elg_amount"] + ff["sell_elg_amount"]) / total_amt

composite = sm_dir * sm_w + md_dir * md_w + lg_dir * lg_w + elg_dir * elg_w
return cross_sectional_rank(composite)
```

**意义**：不同规模订单的资金方向信号强度不同，按金额占比加权综合各档位方向信号，比单一净流入更全面反映市场多空结构。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_avg_trade_price_momentum

**定义**：成交均价动量因子（VWAP(大单)/VWAP(小单)的5日变化率截面排名，比率上升=机构买入紧迫度升排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")

# Big order (LG + ELG) VWAP
big_vol = ff["buy_lg_vol"] + ff["sell_lg_vol"] + ff["buy_elg_vol"] + ff["sell_elg_vol"]
big_amt = ff["buy_lg_amount"] + ff["sell_lg_amount"] + ff["buy_elg_amount"] + ff["sell_elg_amount"]
big_vwap = safe_divide(big_amt, big_vol)

# Small order (SM) VWAP
sm_vol = ff["buy_sm_vol"] + ff["sell_sm_vol"]
sm_amt = ff["buy_sm_amount"] + ff["sell_sm_amount"]
sm_vwap = safe_divide(sm_amt, sm_vol)

# Ratio: big/small VWAP
ratio = safe_divide(big_vwap, sm_vwap)
ratio = ratio.clip(0.5, 3.0)

# 5-day momentum of ratio
mom = ratio.groupby(level="Code").transform(
    lambda s: s.pct_change(5, fill_method=None)
)
mom = mom.clip(-0.3, 0.5)

return cross_sectional_rank(mom)
```

**意义**：大单VWAP与小单VWAP的比率反映了机构相对于散户的交易价格水平。比率上升=机构愿意以越来越高的价格买入（买入紧迫度增加）；比率下降=机构在降价出货。该比率的变化方向比绝对值更有信息量。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol.py`

##### mf_big_order_net_kurt_20

**定义**：大单净流入峰度因子：20日(大单+超大单净流入)峰度截面排名（脉冲式建仓排前）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
x = (
    mf["buy_lg_amount"] + mf["buy_elg_amount"]
    - mf["sell_lg_amount"] - mf["sell_elg_amount"]
)
kurt = x.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).kurt()
)
return cross_sectional_rank(kurt)
```

**意义**：大单净流入的分布形态区分资金行为:峰度高=集中脉冲式建仓(1-2 天巨额大单砸下,事件驱动);峰度低=每天均匀流入(长期吸筹)。与mf_flow_volatility_20d/mf_flow_stability_20d(二阶矩)互补,峰度直达三阶形态。单位一致(万元)内部比较,无需换算。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### mf_big_order_ratio

**定义**：大单+特大单净买入率因子，(特大+大净买入)/总成交额截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
big_net = (
    ff["buy_lg_amount"] - ff["sell_lg_amount"]
    + ff["buy_elg_amount"] - ff["sell_elg_amount"]
)
ratio = big_net / _total_amount(ff)
return cross_sectional_rank(ratio)
```

**意义**：特大单和大单通常代表机构行为，净买入占比高是专业资金看多的信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_big_order_stability_20d

**定义**：20日大单净买入率稳定性因子 (高稳定排前)。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_amt = _total_amount(ff)
big_net = ff["buy_lg_amount"] + ff["buy_elg_amount"] - ff["sell_lg_amount"] - ff["sell_elg_amount"]
big_rate = safe_divide(big_net, total_amt)
rate_std = rolling_group_std(big_rate, 20)
rate_mean = rolling_group_mean(big_rate, 20)
stability = safe_divide(rate_mean.abs() + 1e-8, rate_std + 1e-8)
return cross_sectional_rank(stability)
```

**意义**：大单行为一致性反映机构意图明确，频繁方向切换意味着不确定性和噪音交易

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_big_order_turnover_ratio

**定义**：大额订单成交占比因子，(大单+特大单成交量)/总成交量截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
# 2026-08-05:改用 *_vol(成交量)口径,与因子名/描述"成交量"一致(原误用 *_amount)
big_vol = (
    ff["buy_lg_vol"] + ff["buy_elg_vol"]
    + ff["sell_lg_vol"] + ff["sell_elg_vol"]
)
total_vol = (
    big_vol
    + ff["buy_sm_vol"] + ff["sell_sm_vol"]
    + ff["buy_md_vol"] + ff["sell_md_vol"]
).replace(0, np.nan)
ratio = big_vol / total_vol
return cross_sectional_rank(ratio)
```

**意义**：大额订单成交量占比高说明市场由机构主导，机构参与度高的股票信息传递效率更高，定价更有效。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_big_order_vol_ratio

**定义**：大单+特大单成交量占比因子，基于成交量口径的机构行为度量。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
big_vol = (
    ff["buy_lg_vol"] + ff["sell_lg_vol"]
    + ff["buy_elg_vol"] + ff["sell_elg_vol"]
)
ratio = big_vol / _total_vol(ff)
return cross_sectional_rank(ratio)
```

**意义**：成交量口径的大单占比与成交额口径互补：成交量剔除了高价股偏差，更能反映真实的交易笔数集中度。两个口径的信号差异本身也是信息。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/unused_fields_factors.py`

##### mf_big_small_convergence_20d

**定义**：20日大单/小单收敛因子 (大单趋势-小单趋势)。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_amt = _total_amount(ff)
big_net = ff["buy_lg_amount"] + ff["buy_elg_amount"] - ff["sell_lg_amount"] - ff["sell_elg_amount"]
small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
big_rate = safe_divide(big_net, total_amt)
small_rate = safe_divide(small_net, total_amt)
big_trend = rolling_group_mean(big_rate, 5)
small_trend = rolling_group_mean(small_rate, 5)
convergence = big_trend - small_trend
return cross_sectional_rank(convergence)
```

**意义**：大单趋势与小单趋势的背离收敛包含信息——大单领先小单转向是机构先行的信号

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_big_small_divergence

**定义**：大小单背离因子，(大单净买-小单净买)/总成交额截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
big_net = (
    ff["buy_lg_amount"] - ff["sell_lg_amount"]
    + ff["buy_elg_amount"] - ff["sell_elg_amount"]
)
small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
divergence = (big_net - small_net) / _total_amount(ff)
return cross_sectional_rank(divergence)
```

**意义**：大小单背离度越大，说明机构与散户行为分歧越大，分歧顶点常伴随趋势转折。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_cumulative_flow_20d

**定义**：20日累计主力净流入率因子。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_amt = _total_amount(ff)
net_amt = ff["net_mf_amount"]
net_rate = safe_divide(net_amt, total_amt)
cum_rate = net_rate.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
return cross_sectional_rank(cum_rate)
```

**意义**：累计净流入/流出反映中期资金态度，持续的净流入比单日信号更可靠

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_elg_order_ratio

**定义**：特大单净买入率因子，(特大单净买入额/总成交额)截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
elg_net = ff["buy_elg_amount"] - ff["sell_elg_amount"]
ratio = elg_net / _total_amount(ff)
return cross_sectional_rank(ratio)
```

**意义**：特大单代表机构大额交易，净买入占比高反映专业性资金对后市的明确看多态度。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_elg_small_divergence

**定义**：特大单与小单背离因子，(特大单净买入-小单净买入)/总成交额截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
elg_net = ff["buy_elg_amount"] - ff["sell_elg_amount"]
small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
divergence = (elg_net - small_net) / _total_amount(ff)
return cross_sectional_rank(divergence)
```

**意义**：特大单（机构）与小单（散户）行为背离度越大，说明精英资金与散户分歧越严重，背离极端值往往预示趋势转折。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_extra_large_sell_pressure

**定义**：超大单卖出占比取反。超大单卖出集中=机构出货，排名低。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total = (ff["buy_sm_amount"] + ff["sell_sm_amount"] +
         ff["buy_md_amount"] + ff["sell_md_amount"] +
         ff["buy_lg_amount"] + ff["sell_lg_amount"] +
         ff["buy_elg_amount"] + ff["sell_elg_amount"])
elg_sell_ratio = safe_divide(ff["sell_elg_amount"], total)
return cross_sectional_rank(-elg_sell_ratio)
```

**意义**：超大单卖出占总成交额的比例取反排名。超大单(>100万元/笔)几乎全是机构交易——超大单卖出集中意味着机构在主动减持或清仓,是强烈的看空信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_extended.py`

##### mf_flow_acceleration_5d

**定义**：主力资金净流入加速度=5日净流入变化率。加速流入=增量资金积极，排名高。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
net = ff["net_mf_amount"]
velocity = safe_divide(net, _total_amount(ff))
accel = velocity.groupby(level="Code").transform(lambda s: s.diff(5))
accel = accel.clip(-0.5, 0.5)
return cross_sectional_rank(accel)
```

**意义**：主力资金净流入的5日二阶变化(加速度)。净流入本身是速度概念，加速度则捕捉了资金态度的边际变化——加速流入意味着资金越来越积极(乐观信号)；减速(从流入转为流出)意味着资金态度转变(谨慎信号)。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep_extended.py`

##### mf_flow_acceleration_ext

**定义**：资金流加速度因子，主力净流入率的5日变化截面排名（流入在加速=趋势加强排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
net = ff["net_mf_amount"]
total = _total_amount(ff)
net_ratio = net / total
# 5-day change in net_ratio (acceleration)
accel = net_ratio.groupby(level="Code").transform(
    lambda s: s.diff(5)
)
return cross_sectional_rank(accel)
```

**意义**：资金流的二阶导(加速度)比一阶导(方向)更具前瞻性——净流入从正到更正向=买盘在加速(最强信号)；净流入从负到正(转正)=趋势可能反转(注意跟进)；净流入从正到负(减弱)=主力在撤退(预警信号)。加速度为正意味着资金的边际态度在改善。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep.py`

##### mf_flow_continuity

**定义**：主力资金连续流入天数因子，统计各股票连续净流入天数截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
positive = ff["net_mf_amount"] > 0

def _streak_series(s):
    """Compute consecutive True streak for a single stock."""
    groups = (s != s.shift(1)).cumsum()
    streak = s.groupby(groups).cumcount() + 1
    return streak.where(s, 0)

streak = positive.groupby(level="Code").transform(_streak_series)
return cross_sectional_rank(streak)
```

**意义**：连续净流入天数越长，说明主力资金对该股的持续看好态度越坚定，短期股价支撑越强。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_flow_reversal_20d

**定义**：20日主力资金反转因子 (从流出的流出反转为流入)。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_amt = _total_amount(ff)
net_rate = safe_divide(ff["net_mf_amount"], total_amt)
cum_10 = rolling_group_mean(net_rate, 10)
cum_20 = rolling_group_mean(net_rate, 20)
# Reversal: recent 10d positive while longer 20d negative
reversal = cum_10 - cum_20
return cross_sectional_rank(reversal)
```

**意义**：主力从净流出转为净流入是重要的拐点信号，捕捉资金态度的边际变化

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_flow_stability_20d

**定义**：主力资金流向20日稳定性=连续同向天数占比。频繁转向=信号不可靠，排名低。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
net = ff["net_mf_amount"]
# 修正(2026-08-05):
# 1) NaN 处理:np.sign(NaN)=NaN,NaN!=x 恒 True——每股首行(prev 为 NaN)及
#    任何 NaN 日都会被误计为一次"方向变化"。须同时要求 net 与 prev 均有效。
# 2) 方向:原 rank(-stability) 使不稳定者排名反而靠前;描述要求"稳定者排前、
#    不稳定者排后",应 rank(stability)。
prev = net.groupby(level="Code").shift(1)
sign_changes = (
    (np.sign(net) != np.sign(prev)) & net.notna() & prev.notna()
).astype(float)
change_count = sign_changes.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
stability = 1.0 - change_count / 20.0
return cross_sectional_rank(stability)
```

**意义**：主力资金净流入方向在20日内的稳定性。持续同向(一直流入或一直流出)意味着主力态度明确——方向稳定。频繁转向(今天进明天出)意味着主力态度摇摆——信号噪音大、可靠性低。排名取反使得不稳定者排后。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep_extended.py`

##### mf_flow_streak_5d

**定义**：主力净流入天数频率（5日内 net_mf_amount>0 的天数占比）。

**公式（计算逻辑）**：

```python
df = _mf(context)
df["pos"] = (df["net_mf_amount"] > 0).astype(float)
return _out(df, "mf_flow_streak_5d", roll(df, "pos", 5, "mean"))
```

**意义**：净流入天数占比比净额更稳健，高频净流入=主力持续吸筹，（注：名义 streak，实为频率，照抄原脚本语义）。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_fundflow.py`

##### mf_flow_volatility_20d

**定义**：资金流波动率因子，-(主力净流入20日标准差)截面排名（资金流稳定=有序建仓排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
net = ff["net_mf_amount"]
total = _total_amount(ff)
net_ratio = net / total
vol = net_ratio.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
return cross_sectional_rank(-vol)
```

**意义**：主力资金流的波动率区分'有序建仓'和'游资短炒'——机构建仓通常表现为持续、稳定的净流入（低波动），而游资操作则呈现大进大出（高波动）。低波动+正流入是最优信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep.py`

##### mf_large_order_avg_price

**定义**：大单+超大单成交均价相对总成交均价。高比值=机构高价成交(拉升建仓),排名高。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
large_vol = ff["buy_lg_vol"] + ff["sell_lg_vol"] + ff["buy_elg_vol"] + ff["sell_elg_vol"]
large_amt = ff["buy_lg_amount"] + ff["sell_lg_amount"] + ff["buy_elg_amount"] + ff["sell_elg_amount"]
total_vol = _total_vol(ff)
total_amt = _total_amount(ff)
large_avg = safe_divide(large_amt, large_vol)
total_avg = safe_divide(total_amt, total_vol)
ratio = safe_divide(large_avg, total_avg)
ratio = ratio.clip(0.5, 2.0)
return cross_sectional_rank(ratio)
```

**意义**：大单和超大单的成交均价(amount/vol)相对总成交均价的比值。大单均价高于总均价=机构在较高价位积极买入(拉升式建仓)——看多信号；大单均价低于总均价=机构在压低价格吸筹或高位出货——需结合方向判断。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol_extended.py`

##### mf_large_order_net_5d

**定义**：大单净流入5日均值/总成交额。持续大单净流入=机构持续吸筹，排名高。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
large_net = ff["buy_lg_amount"] - ff["sell_lg_amount"]
total = (ff["buy_sm_amount"] + ff["sell_sm_amount"] +
         ff["buy_md_amount"] + ff["sell_md_amount"] +
         ff["buy_lg_amount"] + ff["sell_lg_amount"] +
         ff["buy_elg_amount"] + ff["sell_elg_amount"])
ratio = safe_divide(large_net, total)
ratio_ma5 = ratio.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).mean()
)
ratio_ma5 = ratio_ma5.clip(-0.5, 0.5)
return cross_sectional_rank(ratio_ma5)
```

**意义**：大单净买入5日均值占总成交额的比例。持续的大单净流入比单日大单方向更可靠——机构建仓是逐步完成的,5日均值过滤了单日波动,捕捉了持续性的机构行为。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_extended.py`

##### mf_large_vol_net_5d

**定义**：大单净买入量占比5日均值。持续的大单量净流入=机构持续建仓(量能角度)。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
large_net_vol = (ff["buy_lg_vol"] + ff["buy_elg_vol"]
                 - ff["sell_lg_vol"] - ff["sell_elg_vol"])
ratio = safe_divide(large_net_vol, _total_vol(ff))
ratio_ma5 = ratio.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).mean()
)
ratio_ma5 = ratio_ma5.clip(-0.5, 0.5)
return cross_sectional_rank(ratio_ma5)
```

**意义**：大单和超大单的净买入量占总成交量比例的5日均值。与金额维度的大单净流入互补——成交量维度的大单净流入更能反映机构的手数行为(买入多少股),而不仅仅是金额行为(花了多少钱)。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol_extended.py`

##### mf_md_order_vol_ratio

**定义**：中单成交量占比因子（(buy_md_vol+sell_md_vol)/总成交量截面排名）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
md_vol = ff["buy_md_vol"] + ff["sell_md_vol"]
ratio = safe_divide(md_vol, _total_vol(ff))
return cross_sectional_rank(ratio)
```

**意义**：中单（4-20万元）成交量占总成交量的比例反映了'聪明散户'或小型机构的参与度。中单资金在A股具有独特的信息优势：资金量足够影响盘面但又不是太大以至于暴露意图。成交额口径已有类似因子（ext_mf_medium_order_amount_ratio），成交量口径的中单占比提供了无价格偏差的参与度度量。两者差异反映中单资金的'单价'变化方向。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol.py`

##### mf_mid_order_ratio

**定义**：中单净买入率因子，(中单净买入额/总成交额)截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
md_net = ff["buy_md_amount"] - ff["sell_md_amount"]
ratio = md_net / _total_amount(ff)
return cross_sectional_rank(ratio)
```

**意义**：中单代表专业但非机构级别的交易者（如大户/游资），其净流向反映中等规模资金的短期态度。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_net_amount_intensity

**定义**：主力资金净额强度因子，net_mf_amount/流通市值截面排名。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
finance = context.load("finance.parquet")
net_amount = mf["net_mf_amount"]
circ_mv = finance["circ_mv"]
common = net_amount.index.intersection(circ_mv.index)
intensity = net_amount.loc[common] / circ_mv.loc[common].replace(0, np.nan)
return cross_sectional_rank(intensity)
```

**意义**：主力资金净买入相对流通市值的比例——消除规模效应后，小市值股票的主力建仓信号更灵敏。

**依赖数据**：`main_fund_flow.parquet`、`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_net_inflow_5d

**定义**：5日累计主力净流入率因子截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
daily_ratio = ff["net_mf_amount"] / _total_amount(ff)
cum_ratio = daily_ratio.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).sum()
)
return cross_sectional_rank(cum_ratio)
```

**意义**：短期累计主力资金行为比单日更具稳定性，过滤噪音。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_net_inflow_ratio

**定义**：主力资金净流入率因子，主力净流入额/成交额截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
ratio = ff["net_mf_amount"] / _total_amount(ff)
return cross_sectional_rank(ratio)
```

**意义**：主力净流入率是日内聪明钱行为的直接度量，持续净流入预示后续上涨动力。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_net_inflow_trend_5d

**定义**：5日主力净流入率趋势因子，近5日净流入率线性回归斜率截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
daily_ratio = ff["net_mf_amount"] / _total_amount(ff)

def _trend_slope(y):
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x * x).sum()
    if denom == 0:
        return np.nan
    return (x * y).sum() / denom

slope = daily_ratio.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).apply(_trend_slope, raw=True)
)
return cross_sectional_rank(slope)
```

**意义**：主力资金流的趋势方向比单日流向量更具信息量，持续流入斜率反映资金态度的一致性强弱。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_net_inflow_volatility_20d

**定义**：20日主力净流入率波动率因子，(负向排名)净流入率波动越大排名越低。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
daily_ratio = ff["net_mf_amount"] / _total_amount(ff)
vol_20d = daily_ratio.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
return cross_sectional_rank(-vol_20d)
```

**意义**：主力资金流向波动率高说明资金态度分歧大、缺乏一致方向，高波动区间后市走势不确定性增加，应给予低评分。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_net_persistent_5d

**定义**：主力净流入持续性因子，5日主力净流入为正的天数截面排名（持续净流入=坚定看多排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
net = ff["net_mf_amount"]
is_positive = (net > 0).astype(float)
persist = is_positive.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).sum()
)
return cross_sectional_rank(persist)
```

**意义**：主力资金的持续性比单日力度更重要——连续5天净流入说明机构在系统性建仓而非短线博弈。持续净流入的股票中期趋势延续概率显著高于脉冲式流入的股票。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep.py`

##### mf_net_vol_intensity

**定义**：主力净流入量/总成交量因子，成交量口径的资金净流向强度。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
ratio = ff["net_mf_vol"] / _total_vol(ff)
return cross_sectional_rank(ratio)
```

**意义**：净流入量占比为正意味着买方在数量上占优，配合净流入额占比使用可以识别量价配合vs量价背离的信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/unused_fields_factors.py`

##### mf_net_vol_ma_divergence

**定义**：净成交量均线偏离因子（净成交量/(净成交量20日均值)-1截面排名，当前>均值=加速排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
net_vol = ff["net_mf_vol"]

ma_20 = net_vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
divergence = safe_divide(net_vol, ma_20.abs() + 1e-10) - 1.0
divergence = divergence.clip(-3, 5)

return cross_sectional_rank(divergence)
```

**意义**：净成交量相对于其20日均值的偏离反映了资金流向的异常强度。当前净流入远超均值=资金正在'抢筹'；当前净流出远超均值=资金正在'逃离'。与趋势因子(net_vol_trend_3d)不同，该因子关注的是'当前是否异常'而非'方向是否改变'。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol.py`

##### mf_net_vol_ratio_5d

**定义**：净买入成交量占比5日变化。量能角度净流入的边际改善,排名高。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
net_vol_ratio = safe_divide(ff["net_mf_vol"], _total_vol(ff))
ratio_ma5 = net_vol_ratio.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).mean()
)
ratio_ma5 = ratio_ma5.clip(-0.5, 0.5)
return cross_sectional_rank(ratio_ma5)
```

**意义**：净买入成交量占比(net_mf_vol/total_vol)的5日均值。成交量维度的净流入捕捉了交易量的方向性——净流入量持续增加=更多成交量在买方,即使金额净流入不大,量能的方向性也很重要。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol_extended.py`

##### mf_net_vol_trend_3d

**定义**：3日净成交量趋势因子（net_mf_vol的3日变化率截面排名，净流入加速排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
net_vol = ff["net_mf_vol"]

# 3-day change
delta = net_vol.groupby(level="Code").transform(
    lambda s: s.diff(3)
)
# Normalize by total volume for cross-sectional comparability
normalized = safe_divide(delta, _total_vol(ff))
normalized = normalized.clip(-0.5, 0.5)

return cross_sectional_rank(normalized)
```

**意义**：净成交量（net_mf_vol = 总买量 - 总卖量）的短期趋势捕捉了资金流向的加速度。净流入加速=买方力量在增强；净流出减速=卖方力量在衰竭。3日窗口足够短以捕捉转折点，又足够长以过滤单日噪音。成交量口径的净流趋势比成交额口径更能反映交易行为的广泛性变化。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol.py`

##### mf_open_close_divergence_10d

**定义**：10日资金流偏离波动因子：净流入率相对其5日均线的偏离的10日波动率截面排名（偏离剧烈=资金态度摇摆排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_amt = _total_amount(ff)
net_rate = safe_divide(ff["net_mf_amount"], total_amt)
# Proxy: difference between daily net rate and its 5d trend
trend_5 = rolling_group_mean(net_rate, 5)
divergence = safe_divide(net_rate - trend_5, trend_5.abs() + 1e-8)
div_std = rolling_group_std(divergence, 10)
return cross_sectional_rank(div_std)
```

**意义**：2026-08-05 描述与实现统一(原描述声称开盘/收盘背离,实现为日频代理):净流入率相对5日均线的偏离反映当日资金态度与近期趋势的落差,其10日波动率衡量资金行为的摇摆程度——偏离频繁放大=主力态度反复、方向不确定。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_order_concentration

**定义**：主力资金订单集中度=大单+超大单占比，Herfindahl指数式度量。高集中度=机构主导，排名高。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
big = ff["buy_lg_amount"] + ff["sell_lg_amount"] + ff["buy_elg_amount"] + ff["sell_elg_amount"]
total = _total_amount(ff)
concentration = safe_divide(big, total)
return cross_sectional_rank(concentration)
```

**意义**：大单和超大单成交额占总成交额的比例。高比例意味着机构和主力资金主导了当日交易——这是专业资金的足迹。低比例意味着散户交易占主导，价格发现效率低。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep_extended.py`

##### mf_order_size_entropy

**定义**：订单规模分布的熵因子（高值排前），度量资金参与结构的多样性。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_vol = _total_vol(ff)

# Volume share per order size category
sm_share = (ff["buy_sm_vol"] + ff["sell_sm_vol"]) / total_vol
md_share = (ff["buy_md_vol"] + ff["sell_md_vol"]) / total_vol
lg_share = (ff["buy_lg_vol"] + ff["sell_lg_vol"]) / total_vol
elg_share = (ff["buy_elg_vol"] + ff["sell_elg_vol"]) / total_vol

# Entropy: -sum(p * ln(p)), higher = more diverse participation
eps = 1e-10
entropy = -(
    sm_share * np.log(sm_share + eps)
    + md_share * np.log(md_share + eps)
    + lg_share * np.log(lg_share + eps)
    + elg_share * np.log(elg_share + eps)
)
return cross_sectional_rank(entropy)
```

**意义**：当各规模订单（小/中/大/特大）的参与比例较为均匀时，市场参与结构健康。当某一类订单占比畸高时（例如全靠大单拉动），行情的持续性存疑。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/unused_fields_factors.py`

##### mf_rel_ind_ma20d

**定义**：主力净流入行业相对强度（行业内z-score的20日均值）。

**公式（计算逻辑）**：

```python
df = _mf(context)
return _out(df, "mf_rel_ind_ma20d", roll(df, "net_rel_ind", 20, "mean"))
```

**意义**：月度主力资金相对强度，过滤短期对倒噪音，刻画机构资金的持续布局方向。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_fundflow.py`

##### mf_rel_ind_ma5d

**定义**：主力净流入行业相对强度（行业内z-score的5日均值）。

**公式（计算逻辑）**：

```python
df = _mf(context)
return _out(df, "mf_rel_ind_ma5d", roll(df, "net_rel_ind", 5, "mean"))
```

**意义**：主力资金（大单+超大单净流入）行业内相对强弱，5日均值平滑单日噪音，反映短期聪明钱动向。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_fundflow.py`

##### mf_retail_dominance

**定义**：散户交易占比=(小单买入+小单卖出)/总成交额取反。高散户占比=噪音交易多，排名低。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
retail = ff["buy_sm_amount"] + ff["sell_sm_amount"]
ratio = safe_divide(retail, _total_amount(ff))
return cross_sectional_rank(-ratio)
```

**意义**：小单成交额占总成交额的比例取反排名。小单占比高意味着散户交易主导——噪音交易多、价格发现效率低。小单占比低意味着机构主导——定价更有效。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep_extended.py`

##### mf_sm_order_vol_ratio

**定义**：小单成交量占比因子（散户成交量占比截面排名，低占比=机构化排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
sm_vol = ff["buy_sm_vol"] + ff["sell_sm_vol"]
ratio = safe_divide(sm_vol, _total_vol(ff))
return cross_sectional_rank(-ratio)  # low retail = institutionalized = good
```

**意义**：小单（<4万元）成交量占比是散户参与度的纯粹度量。高散户占比的股票更容易出现行为偏差驱动的错误定价。低散户占比意味着机构化程度高，定价更有效。成交量口径避免了散户偏好低价股的偏差（成交额口径会低估散户在低价股中的真实参与度）。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol.py`

##### mf_small_order_ratio

**定义**：小单净买入率因子（负值=散户净卖出，排名高=散户流出多）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
ratio = small_net / _total_amount(ff)
return cross_sectional_rank(-ratio)
```

**意义**：散户净卖出+机构净买入的组合是较强的看多信号，反向使用小单数据更有效。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### mf_smart_dumb_divergence

**定义**：聪明钱vs散户分歧=(大单净买/大单总额)-(小单净买/小单总额)。正值=机构买散户卖，排名高。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
big_net = (ff["buy_lg_amount"] + ff["buy_elg_amount"]
           - ff["sell_lg_amount"] - ff["sell_elg_amount"])
big_total = (ff["buy_lg_amount"] + ff["sell_lg_amount"]
             + ff["buy_elg_amount"] + ff["sell_elg_amount"])
small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
small_total = ff["buy_sm_amount"] + ff["sell_sm_amount"]
big_ratio = safe_divide(big_net, big_total)
small_ratio = safe_divide(small_net, small_total)
divergence = big_ratio - small_ratio
divergence = divergence.clip(-1, 1)
return cross_sectional_rank(divergence)
```

**意义**：大单净流向与小单净流向的标准化差值。正值=大单净买入但小单净卖出——机构在收集筹码而散户在卖出(聪明钱信号)；负值=机构卖出但散户买入——机构在派发筹码(危险信号)。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep_extended.py`

##### mf_tier_net_spread_20

**定义**：大中小单分歧度因子：20日四档(小/中/大/超大)净占比极差截面排名（负向，多空分歧排后）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
daily = context.load("daily.parquet")
amount20 = daily["amount"].groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
).reindex(mf.index)
shares = []
for tier in ("sm", "md", "lg", "elg"):
    net = (mf[f"buy_{tier}_amount"] - mf[f"sell_{tier}_amount"]) * 1e4
    net20 = net.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    shares.append(safe_divide(net20, amount20))
spread = pd.Series(
    np.maximum.reduce(shares) - np.minimum.reduce(shares),
    index=shares[0].index,
)
return cross_sectional_rank(-spread)
```

**意义**：四个层级 20 日净占比(单档净流入/成交额)的极差度量资金方向的分歧度:极差大=机构与散户方向严重对立(变盘前兆);极差小=各层级方向一致(趋势健康)。与 mf_big_small_divergence(仅大小两档)区分:四档极差覆盖中单(游资/大户)的独立信号。单位:main_fund_flow 金额=万元(×1e4 转元),daily.amount=元。

**依赖数据**：`main_fund_flow.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### mf_vol_amount_corr_20

**定义**：量额相关性因子（20日各档vol/amount日变化率相关性截面排名，低相关=价格偏差大排后）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_vol = _total_vol(ff)
total_amt = _total_amount(ff)

# Daily change rates
vol_chg = total_vol.groupby(level="Code").transform(
    lambda s: s.pct_change(1, fill_method=None)
)
amt_chg = total_amt.groupby(level="Code").transform(
    lambda s: s.pct_change(1, fill_method=None)
)

# 20-day rolling correlation of daily changes
def _rolling_corr(vol_s, amt_s, window=20):
    """Vectorized rolling correlation."""
    vol_roll = vol_s.rolling(window, min_periods=10)
    amt_roll = amt_s.rolling(window, min_periods=10)
    cov = (vol_s * amt_s).rolling(window, min_periods=10).mean() - vol_roll.mean() * amt_roll.mean()
    denom = vol_roll.std() * amt_roll.std()
    return safe_divide(cov, denom)

corr = vol_chg.groupby(level="Code").transform(
    lambda s: _rolling_corr(s, amt_chg.loc[s.index], 20)
)
corr = corr.clip(-1, 1)

return cross_sectional_rank(corr)  # high correlation = stable pricing = good
```

**意义**：成交量和成交额应当高度正相关。当两者背离时，意味着成交均价发生了变化——同样的股数在更高的价格成交（均价上升：买方急迫）或在更低的价格成交（均价下降：卖方急迫）。低vol-amount相关性意味着成交均价不稳定，可能是信息不对称的标志。该因子捕捉了'成交背后的价格故事'。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol.py`

##### mf_vol_amount_divergence

**定义**：资金流量的量-额背离：净买入量占比-净买入额占比。正=量大但额小(低价成交),负=量小但额大(高价成交)。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total_vol = _total_vol(ff)
total_amt = _total_amount(ff)
net_vol_ratio = safe_divide(ff["net_mf_vol"], total_vol)
net_amt_ratio = safe_divide(ff["net_mf_amount"], total_amt)
divergence = net_vol_ratio - net_amt_ratio
divergence = divergence.clip(-0.5, 0.5)
return cross_sectional_rank(divergence)
```

**意义**：净买入的成交量占比与成交额占比的差值。正值=成交量净买入多于成交额净买入(低价位大量成交,散户主导,机构可能在压价吸筹)；负值=成交额净买入多于成交量净买入(高价位少量成交,机构拉升中买入)。量-额背离揭示了资金行为的质量差异——同样的净流入,高价vs低价含义完全不同。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol_extended.py`

##### mf_vol_concentration_large

**定义**：大单+超大单成交量占比。高占比=机构交易量集中,排名高。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
large_vol = ff["buy_lg_vol"] + ff["sell_lg_vol"] + ff["buy_elg_vol"] + ff["sell_elg_vol"]
ratio = safe_divide(large_vol, _total_vol(ff))
return cross_sectional_rank(ratio)
```

**意义**：大单和超大单成交量占总成交量的比例。高比例意味着机构和主力资金主导了当日交易量——这是专业资金活跃度的直接证据。与金额集中度配合使用,量+额双高=最可靠的机构信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol_extended.py`

##### mf_vol_retail_ratio

**定义**：小单成交量占比取反。小单量占比高=散户活跃,噪音大,排名低。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
retail_vol = ff["buy_sm_vol"] + ff["sell_sm_vol"]
ratio = safe_divide(retail_vol, _total_vol(ff))
return cross_sectional_rank(-ratio)
```

**意义**：小单成交量占总成交量的比例取反排名。小单量占比高意味着散户交易活跃——噪音交易者主导、价格发现效率低。低小单量占比意味着机构主导——价格信号更有效。这是vol维度对散户参与度的度量,与amount维度的mf_retail_dominance互补。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol_extended.py`

##### mf_vol_tier_balance

**定义**：各规模成交量平衡度因子（各档vol占比两两差异绝对值之和截面排名，均衡=健康排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
total = _total_vol(ff)

sm_share = (ff["buy_sm_vol"] + ff["sell_sm_vol"]) / total
md_share = (ff["buy_md_vol"] + ff["sell_md_vol"]) / total
lg_share = (ff["buy_lg_vol"] + ff["sell_lg_vol"]) / total
elg_share = (ff["buy_elg_vol"] + ff["sell_elg_vol"]) / total

# Pairwise absolute differences — lower = more balanced
imbalance = (
    (sm_share - md_share).abs()
    + (sm_share - lg_share).abs()
    + (sm_share - elg_share).abs()
    + (md_share - lg_share).abs()
    + (md_share - elg_share).abs()
    + (lg_share - elg_share).abs()
)

return cross_sectional_rank(-imbalance)  # low imbalance = balanced = good
```

**意义**：当四个订单规模档（小/中/大/特大）的成交量占比趋于均衡时，市场参与结构健康——各类资金共同参与、互相制衡。当某一档占比严重偏高时（如全靠大单），行情持续性存疑。成交量口径的平衡度比成交额口径更强调'参与广度'而非'资金强度'。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_vol.py`

##### net_mf_amount_intensity

**定义**：成交量基础主力净流入因子，(主力净流入量/总成交量)截面排名。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
ratio = ff["net_mf_amount"] / _total_amount(ff)
return cross_sectional_rank(ratio)
```

**意义**：所有现有主力资金流因子均基于成交金额，但金额受股价高低影响大——高价股在金额排名中天然占优。成交量基础的主力净流入率从'股数'维度衡量主力行为，消除价格偏差后更公平地跨股票比较主力参与度。与mf_net_inflow_ratio互补：一个看金额权重，一个看量权重。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### net_mf_amount_momentum_5d

**定义**：主力资金净额5日动量因子，net_mf_amount的5日变化截面排名。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
net_amt = mf["net_mf_amount"]
mom = net_amt.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(mom)
```

**意义**：主力资金净买入额的边际变化——净买入正在加速(无论正负)意味着主力行为在改变，是趋势转变的早期信号。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### net_mf_flow_persistence

**定义**：主力资金净流入持续性因子，近5日净流入为正的天数截面排名。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
net_pos = (mf["net_mf_amount"] > 0).astype(float)
persistence = net_pos.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).sum()
)
return cross_sectional_rank(persistence)
```

**意义**：主力资金持续净流入比单日净流入更有意义——持续流入代表机构系统性建仓而非一日游。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### net_turnover_rate_20

**定义**：净换手率因子：20日(主动买量−主动卖量)/自由流通股本截面排名（净买入比例高排前）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
fin = context.load("finance.parquet")
buy = (
    mf["buy_sm_vol"] + mf["buy_md_vol"] + mf["buy_lg_vol"] + mf["buy_elg_vol"]
)
sell = (
    mf["sell_sm_vol"] + mf["sell_md_vol"] + mf["sell_lg_vol"] + mf["sell_elg_vol"]
)
net_shares = (buy - sell) * 100.0  # 手 → 股
net20 = net_shares.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
free_share = fin["free_share"].reindex(net20.index)
pct = safe_divide(net20, free_share) * 100.0
return cross_sectional_rank(pct)
```

**意义**：研报《净换手率》+策略95:净换手率=(主动买量−主动卖量)/流通股本,是短线动量维度的高效因子。资金流 buy/sell 即主动买卖代理,自由流通股本归一给出「多少比例的流通盘被净买入」的直观含义。与 mf_net_vol_ratio_5d(总成交量归一)区分:本因子用股本归一。单位:main_fund_flow vol=手(×100转股),free_share=股。

**依赖数据**：`main_fund_flow.parquet`、`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### order_concentration

**定义**：订单集中度因子，-(中单+小单)/总成交截面排名（大单+超大单占比高=机构主导排前）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
small_mid = (mf["buy_sm_amount"] + mf["sell_sm_amount"] +
              mf["buy_md_amount"] + mf["sell_md_amount"])
total = _total_amount(mf)
concentration = small_mid / total.replace(0, np.nan)
return cross_sectional_rank(-concentration)
```

**意义**：成交量的订单结构反映市场参与者的构成——大单占比越高意味着机构/大户参与度越高，信息的alpha质量越高。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### order_size_concentration

**定义**：订单规模集中度因子，四个规模档的成交额HHI截面排名（集中度高=机构交易主导排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
sm_total = ff["buy_sm_amount"] + ff["sell_sm_amount"]
md_total = ff["buy_md_amount"] + ff["sell_md_amount"]
lg_total = ff["buy_lg_amount"] + ff["sell_lg_amount"]
elg_total = ff["buy_elg_amount"] + ff["sell_elg_amount"]
total_amount = sm_total + md_total + lg_total + elg_total

# HHI = sum of squared market shares
hhi = (
    (sm_total / total_amount) ** 2
    + (md_total / total_amount) ** 2
    + (lg_total / total_amount) ** 2
    + (elg_total / total_amount) ** 2
)
# 注:旧 big_net 死代码已于 2026-08-05 移除(未使用且单位混乱)
# Simplified: if ELG+LG net is positive, HHI is positive; if negative, HHI is negative
smart_direction = (
    ff["buy_elg_amount"] - ff["sell_elg_amount"]
    + ff["buy_lg_amount"] - ff["sell_lg_amount"]
)
signed_hhi = hhi * np.sign(smart_direction)
return cross_sectional_rank(signed_hhi)
```

**意义**：四个订单规模档位(sm/md/lg/elg)的成交额赫芬达尔指数(HHI)——HHI高=交易集中在某个规模档(通常是特大单或小单)=交易者类型单一、方向明确；HHI低=交易分散在四个档=多空分歧大、方向不明。特大单主导的高HHI=机构行动一致(排前)；小单主导的高HHI=散户情绪集中(排后)。结合方向判断后区分两种高HHI情形。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep.py`

##### order_size_ratio_change

**定义**：订单规模比变化因子，(大单+特大)/总成交的5日变化截面排名（大单占比提升=机构参与加深排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
big_total = (
    ff["buy_lg_amount"] + ff["sell_lg_amount"]
    + ff["buy_elg_amount"] + ff["sell_elg_amount"]
)
total = _total_amount(ff)
big_ratio = big_total / total
chg = big_ratio.groupby(level="Code").transform(
    lambda s: s.diff(5)
)
return cross_sectional_rank(chg)
```

**意义**：大单占比的边际变化比绝对水平更有信息量——大单占比从10%升到20%意味着机构刚开始介入（最佳买点），而占比已经在40%高位意味着机构可能已经开始出货。二阶变化捕捉拐点。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep.py`

##### short_balance_ratio_change_20d

**定义**：融券余额占比变化因子：融券余额/流通市值占比的20日变化截面排名（负向，空头加仓排后）。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
fin = context.load("finance.parquet")
# 与 margin_leverage_change_20d 同口径:reindex 到 margin 面板索引,
# 避免索引并集引入 25% 缺行 NaN。
total_mv_m = fin["total_mv"].reindex(m.index)
short_ratio = safe_divide(m["rqye"], total_mv_m)
change = short_ratio.groupby(level="Code").diff(20)
return cross_sectional_rank(-change)
```

**意义**：融券余额相对流通市值的占比上升=看空力量持续加码(空头筹码累积),其后若轧空则涨幅剧烈;占比下降=空头回补。与融券成交类因子(short_sell_volume_ratio)互补:本因子刻画存量而非流量。占比为0的股票变化也为0(有效信息),仅覆盖约85%有融资融券资格的股票。

**依赖数据**：`margin_detail.parquet`、`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin_extra.py`

##### short_interest_volatility_20d

**定义**：融券余量20日波动率取反。余量剧烈波动=空头态度摇摆/不稳定,排名低。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
rqyl = m["rqyl"]
roll_std = rqyl.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
roll_mean = rqyl.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
cv = safe_divide(roll_std, roll_mean)
cv = cv.clip(0, 2)
return cross_sectional_rank(-cv)
```

**意义**：融券余量20日变异系数(std/mean)。高波动意味着空头在频繁开仓平仓——做空方向不坚定、观点摇摆(噪音空头)；低波动意味着空头仓位稳定——做空信念坚定(信息型空头)。低波动的持续做空比高波动的频繁进出更有预测力。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin_short_deep.py`

##### short_sell_volume_ratio

**定义**：融券卖出占比：rqmcl/vol。融券卖出量相对总成交量的占比,高速=活跃做空。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
d = context.load("daily.parquet")
# Date=T 使用当时可获得的 T-1 融券卖出量和 T 日总成交量。
vol_t = d["vol"].reindex(m.index)
ratio = safe_divide(m["rqmcl"], vol_t)
ratio = ratio.clip(0, 1)
return cross_sectional_rank(ratio)
```

**意义**：融券卖出量相对总成交量的占比。高占比=当日做空交易在总交易中占比大——做空意愿强烈、空头攻击火力集中。该比率提供了做空行为的市场份额视角。需要daily.parquet的成交量做分母。

**依赖数据**：`margin_detail.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin_short_deep.py`

##### short_squeeze_risk

**定义**：逼空风险=融券余量/融资余额。高比值=大量做空仓位vs低做多杠杆,逼空风险大,排名高(反转做多信号)。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
squeeze = safe_divide(m["rqyl"], m["rzye"])
squeeze = squeeze.clip(0, 10)
return cross_sectional_rank(squeeze)
```

**意义**：融券余量(rqyl)/融资余额(rzye)衡量做空仓位相对做多杠杆的规模。比值异常高意味着大量资金在做空该股票——一旦股价反弹,空头被迫回补(逼空),可能引发剧烈上涨。该因子是经典的short-squeeze预警信号——做空过度拥挤的股票存在逼空风险,是潜在的反转做多机会。

**依赖数据**：`margin_detail.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin_short_deep.py`

##### small_order_crowding

**定义**：小单拥挤度因子，-(小单买入量/总成交量)截面排名（高小单占比=散户追涨排后）。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
# 2026-08-05:改用 *_vol(成交量)口径,与描述"小单买入量/总成交量"一致
small_vol = mf["buy_sm_vol"] + mf["sell_sm_vol"]
total_vol = (
    mf["buy_sm_vol"] + mf["sell_sm_vol"]
    + mf["buy_md_vol"] + mf["sell_md_vol"]
    + mf["buy_lg_vol"] + mf["sell_lg_vol"]
    + mf["buy_elg_vol"] + mf["sell_elg_vol"]
).replace(0, np.nan)
small_pct = small_vol / total_vol
return cross_sectional_rank(-small_pct)
```

**意义**：小单成交量占比过高意味着散户主导交易——散户追涨是经典的短期见顶信号，机构通常在小单占比极端时反向操作。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### smart_money_concentration

**定义**：聪明钱集中度因子，(特大单+大单净买-小单-中单净卖)/总成交额截面排名（聪明钱相对噪音交易者越集中排前）。

**公式（计算逻辑）**：

```python
ff = context.load("main_fund_flow.parquet")
smart = (
    ff["buy_elg_amount"] - ff["sell_elg_amount"]
    + ff["buy_lg_amount"] - ff["sell_lg_amount"]
)
noise = (
    ff["buy_sm_amount"] - ff["sell_sm_amount"]
    + ff["buy_md_amount"] - ff["sell_md_amount"]
)
total = _total_amount(ff)
# Smart money net minus noise net, scaled by total turnover
score = (smart - noise) / total
return cross_sectional_rank(score)
```

**意义**：将订单按规模分为聪明钱(特大+大单)和噪音(小+中单)——聪明钱净买入远大于噪音交易者净买入时=机构主导定价权、方向可靠；噪音交易者主导时=散户情绪驱动、方向不确定。该比率度量的是谁的边际定价权更强。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow_deep.py`

##### super_large_order_intensity

**定义**：超大单强度因子，超大单净买入/总成交额截面排名。

**公式（计算逻辑）**：

```python
mf = context.load("main_fund_flow.parquet")
elg_net = (mf["buy_elg_amount"] - mf["sell_elg_amount"]) / _total_amount(mf)
return cross_sectional_rank(elg_net)
```

**意义**：超大单(每单>500万)是机构定制化交易和主力大额博弈的直接体现——超大单净流入持续为正意味着主力在持续收集筹码。

**依赖数据**：`main_fund_flow.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fund_flow.py`

##### total_leverage_ratio

**定义**：融资融券总余额/总市值，衡量杠杆化程度。高杠杆=波动风险大，排名取反。

**公式（计算逻辑）**：

```python
m = context.load("margin_detail.parquet")
f = context.load("finance.parquet")
# Date=T: margin 是可获得的原始 T-1 值，市值按用户约束使用 T 日值。
mv_t = f["total_mv"].reindex(m.index)
ratio = safe_divide(
    m["rzrqye"],
    mv_t,
)
ratio = ratio.clip(0, 0.5)
return cross_sectional_rank(-ratio)
```

**意义**：融资融券总余额(rzrqye)/总市值(total_mv)衡量股票的杠杆化程度。高杠杆意味着一部分市值被杠杆资金锁定——这些资金在市场下跌时面临强制平仓风险，可能放大跌幅。低杠杆意味着更健康的持仓结构。

**依赖数据**：`margin_detail.parquet`、`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/margin_advanced.py`

##### volume_ratio_momentum_5d

**定义**：量比动量因子：量比(volume_ratio)的5日变化截面排名（量能扩张排前）。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
vr = fin["volume_ratio"]
change = vr - vr.groupby(level="Code").shift(5)
return cross_sectional_rank(change)
```

**意义**：volume_ratio(当日成交量/过去5日均量)是量能热度的即时度量——其5日变化捕捉量能扩张/收缩的速度:量比持续抬升=关注度快速升温(资金进场确认);量比持续走低=热度退潮。与成交量水平类因子正交,是量能的一阶导。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/finance_extra.py`

#### <a name="cat-intraday-c1"></a>类别 intraday — 日内 / 微观结构（11 个）

##### am_large_order_ratio

**定义**：上午大单占比因子，上午成交额/全天成交额截面排名（上午集中放量=机构主导排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 注:本因子为 daily 代理(非真实分钟数据,2026-08-05 声明)——使用 daily.parquet
# 的未复权 OHLCV 近似上午/全天成交结构,不使用 daily_adj.parquet。
close = daily["close"]
open_ = daily["open"]
high = daily["high"]
low = daily["low"]

# Morning proxy: AM range relative to full-day range
am_range = (high - open_).abs()
pm_range = (close - low).abs()
total_range = (high - low).replace(0, np.nan)

am_ratio = safe_divide(am_range, total_range)
return cross_sectional_rank(am_ratio)
```

**意义**：上午（尤其是开盘后1小时）是机构交易最密集的时段——上午成交占比高说明机构在主动参与，而不是尾盘被动调整。上午占比>55%通常意味着机构在积极建仓或调仓。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### close_auction_pressure

**定义**：收盘位置压力因子，收盘价在日内(high-low)区间的位置截面排名（收盘接近低点=尾盘卖压排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]
low = daily["low"]

# Proxy: how close is the close to the low of day
# Close near low = selling pressure at end of day
total_range = (daily["high"] - low).replace(0, np.nan)
close_position = safe_divide(close - low, total_range)

return cross_sectional_rank(-close_position)
```

**意义**：日频代理(2026-08-05 改为与实现一致):收盘价在日内区间的位置衡量收盘时点的多空力量——收盘接近日低说明尾盘卖压沉重、买方未能收复失地,次日承压概率大;收盘接近日高则相反(抢筹信号)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### gap_momentum_5d

**定义**：跳空动量因子，5日跳空缺口累计截面排名（持续跳空=强势延续排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]

# pre_close 为除权调整后的昨收,替代 close.groupby(Code).shift(1)(2026-08-05)
prev_close = daily["pre_close"]
gap = safe_divide(open_ - prev_close, prev_close)

gap_5d = gap.groupby(level="Code").transform(
    lambda s: s.rolling(5, min_periods=3).sum()
)
return cross_sectional_rank(gap_5d)
```

**意义**：跳空缺口（开盘价≠昨日收盘价）的累计方向反映短线趋势的加速度——连续向上跳空是短线最强势形态（连续高开），连续向下跳空则是恐慌蔓延。跳空方向×持续性=趋势加速信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### intraday_high_low_volatility

**定义**：日内高低波幅因子，(日内最高-日内最低)/开盘价截面排名（取负向=剧烈波动=不确定性高排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
range_pct = (daily["high"] - daily["low"]) / daily["open"].replace(0, np.nan)
return cross_sectional_rank(-range_pct)
```

**意义**：日内高低波幅是日内不确定性的综合度量——波幅大意味着多空在日内激烈博弈、方向不确定。低波幅+明确方向的交易日后续趋势延续性最好。波幅配合方向使用：高波幅+涨=强多，高波幅+跌=恐慌。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### intraday_lower_shadow

**定义**：下影线比例因子，(下影线/实体)截面排名（长下影=支撑强=探底回升排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]
low = daily["low"]

# Lower shadow: min(open, close) - low
lower_shadow = np.minimum(open_, close) - low
body = (close - open_).abs().replace(0, np.nan)
ratio = safe_divide(lower_shadow, body)

return cross_sectional_rank(ratio)
```

**意义**：下影线（开盘价-最低价，或收盘在最低下方时为收盘-最低）反映下跌过程中的抄底力量——长下影线说明价格被砸下去后买方强力接回，是下方支撑的量化表达。连续长下影线是'底部'形态。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### intraday_reversal_intensity

**定义**：日内反转强度因子，-(|收益|/最高最低波幅)截面排名（高反转=方向不确定排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]
high = daily["high"]
low = daily["low"]

ret = (close - open_) / open_.replace(0, np.nan)
range_ = (high - low) / open_.replace(0, np.nan)

# Reversal = range consumed but little net movement
reversal = safe_divide(range_ - ret.abs(), range_ + 0.001)
return cross_sectional_rank(-reversal)
```

**意义**：日内反转强度衡量价格在日内'走回头路'的程度——开盘上涨但收跌（或相反）意味着日内方向被逆转。高反转交易日后续方向不确定，低反转（单边）交易日趋势更可靠。与trend_strength互补：趋势强度看'直线度'，反转强度看'回头度'。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### intraday_trend_strength

**定义**：日内趋势强度因子，|close-open|/(high-low)截面排名（单边趋势强=方向确定排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]
high = daily["high"]
low = daily["low"]

net_move = (close - open_).abs()
total_range = (high - low).replace(0, np.nan)

strength = safe_divide(net_move, total_range)
return cross_sectional_rank(strength)
```

**意义**：日内价格趋势的'直线度'反映方向的确定性——|收盘-开盘|/(最高-最低)接近1意味着价格在单边运行（高确定性），接近0意味着大幅震荡后回到起点（高不确定性）。趋势强度高时跟随方向更可靠。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### intraday_upper_shadow

**定义**：上影线比例因子，-(上影线/实体)截面排名（长上影=抛压重排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]
high = daily["high"]

upper_shadow = high - close
body = (close - open_).abs().replace(0, np.nan)
ratio = safe_divide(upper_shadow, body)

return cross_sectional_rank(-ratio)
```

**意义**：上影线（最高价-收盘价）反映上涨过程中遭遇的抛压——长上影线说明价格冲高后被卖盘打压回来，是上方阻力的直接体现。连续长上影线是'顶部'形态的量化刻画。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### open_auction_intensity

**定义**：开盘强度因子，(开盘价-昨收)/昨收 × 开盘量/20日均量截面排名（跳空+放量=强信号排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]
vol = daily["vol"]

# pre_close 为除权调整后的昨收(除权日不失真),替代 close.shift(1)(2026-08-05)
prev_close = daily["pre_close"]
gap = safe_divide(open_ - prev_close, prev_close)

avg_vol_20 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
vol_ratio = safe_divide(vol, avg_vol_20)

intensity = gap * vol_ratio
return cross_sectional_rank(intensity)
```

**意义**：集合竞价的价格跳空和成交量组合是开盘最强信号——跳空高开+竞价放量=隔夜重大利好+机构抢筹，是当日大概率走强的最可靠开盘信号。跳空但不放量则可能是假突破。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### pm_reversal_signal

**定义**：下午反转信号因子，-(下午收益/上午收益)截面排名（上午涨+下午跌=盘尾反转排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]
high = daily["high"]
low = daily["low"]

# AM return proxy: (high - open) / open
am_ret = safe_divide(high - open_, open_)
# PM return proxy: (close - high) / high
pm_ret = safe_divide(close - high, high.replace(0, np.nan))

# Reversal = PM opposite direction of AM
reversal = safe_divide(-pm_ret, am_ret.abs() + 0.001)
return cross_sectional_rank(-reversal)
```

**意义**：上午涨但下午回吐是'冲高回落'的典型形态——说明早盘买入力量不足、午盘被卖盘压制。下午反转信号强的股票次日大概率继续走弱。反向（上午跌+下午涨）则是'探底回升'的积极信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

##### volume_distribution_skew

**定义**：日内价格移动偏度代理因子，(开盘至最高涨幅)-(最高至收盘涨幅)截面排名（早盘冲高=机构抢筹排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# Proxy using daily OHLC: morning intensity vs afternoon intensity
open_ = daily["open"]
high = daily["high"]
low = daily["low"]
close = daily["close"]

# Morning movement ratio
morning_move = (high - open_) / open_.replace(0, np.nan)
afternoon_move = (close - high) / high.replace(0, np.nan)

# Skew: positive = more volume/movement in morning
skew = morning_move - afternoon_move
return cross_sectional_rank(skew)
```

**意义**：日频代理(2026-08-05 改为与实现一致):用'开盘至最高'与'最高至收盘'的价格移动差近似日内量能分布的早盘/尾盘集中度——早盘冲高(正偏)通常对应开盘放量、机构集中执行;尾盘回落(负偏)对应日终抛压。真实小时级量分布需 history_1min 数据,本因子为 daily 代理。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/intraday_deep.py`

#### <a name="cat-sector-c1"></a>类别 sector — 行业 / 板块（15 个）

##### ind_disp_ma_20d

**定义**：行业离散度（行业日收益横截面std的20日均值）。

**公式（计算逻辑）**：

```python
return _ind_disp_ma(context, 20, "ind_disp_ma_20d", "行业离散度（20日）", "行业中周期分化度")
```

**意义**：中周期行业分化度，反映板块选股空间与风格切换特征。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### ind_disp_ma_5d

**定义**：行业离散度（行业日收益横截面std的5日均值）。

**公式（计算逻辑）**：

```python
return _ind_disp_ma(context, 5, "ind_disp_ma_5d", "行业离散度（5日）", "行业内分化度")
```

**意义**：行业内个股分化程度：离散度高=行业内机会多但选股难度大，低=板块共振明显，适合行业 β 交易。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### ind_disp_ma_60d

**定义**：行业离散度（行业日收益横截面std的60日均值）。

**公式（计算逻辑）**：

```python
return _ind_disp_ma(context, 60, "ind_disp_ma_60d", "行业离散度（60日）", "行业季度分化度")
```

**意义**：季度级别的行业分化度，标识行业进入分化/共振状态的持续性。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### ind_ret_ma_20d

**定义**：行业收益动能（行业等权日收益的20日均值）。

**公式（计算逻辑）**：

```python
return _ind_ret_ma(context, 20, "ind_ret_ma_20d", "行业收益动能（20日）", "行业中周期动能")
```

**意义**：行业层面的中周期动能，识别处于趋势中的板块。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### ind_ret_ma_3d

**定义**：行业收益动能（行业等权日收益的3日均值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["ind_ret"] = ind_mean(df, "ret")
vals = roll(df, "ind_ret", 3, "mean")
return _out(df, "ind_ret_ma_3d", vals)
```

**意义**：3 日行业动能捕捉板块超短期情绪脉冲，与 ind_ret_ma_5d/20d/60d 构成完整周期谱系，1d 预测中近期板块共振是重要背景项。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### ind_ret_ma_5d

**定义**：行业收益动能（行业等权日收益的5日均值）。

**公式（计算逻辑）**：

```python
return _ind_ret_ma(context, 5, "ind_ret_ma_5d", "行业收益动能（5日）", "行业短周期动能")
```

**意义**：行业层面的短周期动能，反映板块整体资金情绪，作为行业内选股的背景项。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### ind_ret_ma_60d

**定义**：行业收益动能（行业等权日收益的60日均值）。

**公式（计算逻辑）**：

```python
return _ind_ret_ma(context, 60, "ind_ret_ma_60d", "行业收益动能（60日）", "行业季度动能")
```

**意义**：季度级别的行业趋势，过滤短期噪音，捕捉中期板块轮动方向。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### rel_mom_ind_10d

**定义**：行业相对动量（个股10日收益 − 行业等权10日收益）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "rel_mom_ind_10d", _rel_mom(df, 10))
```

**意义**：10 日窗口介于 5d 与 20d 之间，捕捉中短期行业内相对趋势，兼顾响应速度与噪音过滤。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### rel_mom_ind_20d

**定义**：行业相对动量（个股20日收益 − 行业等权20日收益）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "rel_mom_ind_20d", _rel_mom(df, 20))
```

**意义**：月度行业相对动量，趋势更稳、换手更低，是行业内轮动的主信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### rel_mom_ind_3d

**定义**：行业相对动量（个股3日收益 − 行业等权3日收益）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "rel_mom_ind_3d", _rel_mom(df, 3))
```

**意义**：3 日行业相对动量为 rel_mom_ind_5d 的短周期补充：行业内超短期领先-滞后关系在 1d 持有期内衰减最快，需要更短的窗口捕捉。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_short_term.py`

##### rel_mom_ind_5d

**定义**：行业相对动量（个股5日收益 − 行业等权5日收益）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "rel_mom_ind_5d", _rel_mom(df, 5))
```

**意义**：行业内相对动量剥离板块 β：同一行业里跑赢同伴的股票延续性更强，绝对动量受行业轮动干扰大，行业相对更稳定。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### rel_turnover_ind

**定义**：行业内相对换手（换手率 − 行业等权均值）。

**公式（计算逻辑）**：

```python
df = _fin(context)
vals = df["turnover_rate"] - ind_mean(df, "turnover_rate")
return _out(df, "rel_turnover_ind", vals)
```

**意义**：行业内换手相对水平：高换手=资金关注度高但筹码交换剧烈，低换手=惜售/锁筹；剥离板块整体交投活跃度。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### rel_turnover_ind_ma20

**定义**：换手率相对自身均值20日均值（turnover_rate − 个股全期均值）。

**公式（计算逻辑）**：

```python
df = _fin(context)
vals = df.groupby("Code")["turnover_rate"].transform(
    lambda s: (s - s.mean()).rolling(20, min_periods=1).mean()
)
return _out(df, "rel_turnover_ind_ma20", vals)
```

**意义**：换手率相对自身历史水平的月度趋势，识别个股交投活跃度中枢上移/下移（注：沿袭原脚本语义，实为个股自身 demean，非行业相对）。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### rel_vol_ind_20d

**定义**：行业内相对波动（个股20日收益std − 行业等权均值）。

**公式（计算逻辑）**：

```python
df = _daily(context)
df["vol20"] = roll(df, "ret", 20, "std", min_periods=5)
vals = df["vol20"] - ind_mean(df, "vol20")
return _out(df, "rel_vol_ind_20d", vals)
```

**意义**：同一行业内波动相对更低的股票风险调整后更优，剥离板块整体波动水平。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### ret_ind_rel_1d

**定义**：行业相对1日收益因子：个股pct_chg/100 − 行业等权日收益（跑赢行业排前）。

**公式（计算逻辑）**：

```python
df = _daily(context)
ind_ret = df.groupby(["Date", "industry"])["ret"].transform("mean")
return _out(df, "ret_ind_rel_1d", df["ret"] - ind_ret)
```

**意义**：剥离行业 β 的当日个股超额收益——同日行业普涨普跌时，个股相对行业的超额部分更能反映个股层面的资金选择。与 _fac_new_common 的行业相对族同口径（行业映射用当前快照，变化缓慢，属可接受轻微时点回溯）。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_cand_daily.py`

#### <a name="cat-event-c1"></a>类别 event — 事件（22 个）

##### big_gap_reversal_5

**定义**：高开回补因子：5日前高开(>3%)事件的5日累计收益截面排名（高开后走强排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
gap_event = gap.gt(0.03).astype(float)
# 5日前的跳空事件: shift 需按 Code 分组
event_lag5 = gap_event.groupby(level="Code").shift(5)
ret = daily["pct_chg"] / 100.0
ret_5 = ret.groupby(level="Code").transform(
    lambda s: (1.0 + s).cumprod().pct_change(5, fill_method=None)
)
reversal = event_lag5 * ret_5
return cross_sectional_rank(reversal)
```

**意义**：大幅高开(>3%)后是否回补是判断跳空性质的试金石——高开后5日继续上涨=真实突破(缺口为支撑)，高开后回落=假突破(缺口被回补)。用5日前的高开事件与5日累计收益对齐，滞后视角无未来函数。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### consecutive_limit_down

**定义**：连续跌停天数因子：连续跌停(近似)的连跌计数截面排名（连跌高度排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_ld = daily["pct_chg"].le(-9.8).astype(int)
code = is_ld.index.get_level_values("Code")
seg = (~is_ld.astype(bool)).groupby(level="Code").cumsum()
count = is_ld.groupby([code, seg]).cumsum()
return cross_sectional_rank(count)
```

**意义**：连续跌停是流动性枯竭与恐慌加速的标志(与连续涨停镜像)——连跌越深,恐慌盘释放越充分,超跌反弹的概率与幅度也越大(研报《如何捕捉短线反弹机会》)。计数采用连续段逻辑:断板即归零。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### consecutive_limit_up

**定义**：连续涨停天数因子：连续涨停(近似)的连板计数截面排名（连板高度排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_lu = daily["pct_chg"].ge(9.8).astype(int)
code = is_lu.index.get_level_values("Code")
seg = (~is_lu.astype(bool)).groupby(level="Code").cumsum()
count = is_lu.groupby([code, seg]).cumsum()
return cross_sectional_rank(count)
```

**意义**：策略10追三板的核心变量是连板高度——连板天数越长，市场关注度与情绪溢价越高，但连板越高断板风险也越大。计数采用连续段逻辑：断板即归零。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### extreme_move_event

**定义**：极端波动事件衰减因子：|pct_chg|>7%事件后指数衰减（半衰期10日，异动后惯性排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_ext = daily["pct_chg"].abs().gt(7.0)
event = is_ext.astype(float).where(is_ext, np.nan)
decayed = event_decay(event, half_life=10)
return cross_sectional_rank(decayed)
```

**意义**：单日|涨跌|>7%是极端波动事件——大涨(利好兑现)或大跌(利空冲击)后通常有数日的方向惯性或反转(研报《如何捕捉短线反弹机会》)。半衰期10日捕捉中期效应。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### gap_event_decay_5

**定义**：跳空事件衰减因子：|open/pre_close−1|>5%跳空事件后指数衰减（半衰期5日）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
is_gap = gap.abs().gt(0.05)
event = is_gap.astype(float).where(is_gap, np.nan)
decayed = event_decay(event, half_life=5)
return cross_sectional_rank(decayed)
```

**意义**：大幅跳空(>5%)是隔夜信息冲击的直接体现——跳空方向与幅度携带信息。衰减信号区分近期跳空(信息新鲜)与远期跳空(已被消化)。open/pre_close 为复权口径，除权日不产生假跳空。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### high_open_low_close_frac_20

**定义**：高开低走频率因子：20日(高开≥2%且收阴)天数占比截面排名（负向，高开低走惯犯排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
gapup = safe_divide(daily["open"], daily["pre_close"]) - 1.0
fade = (gapup >= 0.02) & (daily["close"] < daily["open"])
freq = fade.astype(float).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=5).mean()
)
return cross_sectional_rank(-freq)
```

**意义**：高开 2% 以上却收阴是主力借高开出货的典型形态——「高开低走惯犯」=上方抛压沉重、承接乏力。与 gap_open_follow_ratio_20(高开次日跟随)不同:本因子聚焦当日高开→收阴的日内反转频率,是出货特征的直接证据。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### lhb_proxy_score_60

**定义**：龙虎榜替代活跃度因子：60日(大波动×高换手×大单高参与)事件次数截面排名（博弈票活跃度排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
mf = context.load("main_fund_flow.parquet")
fin = context.load("finance.parquet")
big_buy = mf["buy_lg_amount"] + mf["buy_elg_amount"]
big_sell = mf["sell_lg_amount"] + mf["sell_elg_amount"]
big_net_yuan = (big_buy - big_sell) * 1e4  # 万元 → 元
amount = daily["amount"].reindex(big_net_yuan.index)
big_share = safe_divide(big_net_yuan.abs(), amount)
hit = (
    (daily["pct_chg"].abs().reindex(big_net_yuan.index) >= 7.0)
    & (fin["turnover_rate"].reindex(big_net_yuan.index) >= 5.0)
    & (amount >= 1e8)
    & (big_share >= 0.10)
)
score = hit.astype(float).groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=10).sum()
)
return cross_sectional_rank(score)
```

**意义**：无龙虎榜数据白名单下,用「|涨跌|≥7% × 换手率≥5% × 成交额≥1亿 × 大单净参与≥10%」四条件近似上榜事件——上榜惯犯=游资博弈票,其短期波动与题材弹性显著高于均值。与 extreme_move_count(仅价格幅度)区分:本因子叠加换手与大单参与度,筛出真正的资金博弈而非单纯暴涨暴跌。单位:main_fund_flow 金额=万元(×1e4 转元),amount=元。

**依赖数据**：`main_fund_flow.parquet`、`finance.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### limit_alternation_20

**定义**：涨跌停交替因子：20日内涨停与跌停事件数量的乘积截面排名（情绪极端反转排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_lu = daily["pct_chg"].ge(9.8).astype(float)
is_ld = daily["pct_chg"].le(-9.8).astype(float)
lu_20 = is_lu.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=5).sum()
)
ld_20 = is_ld.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=5).sum()
)
alternation = lu_20 * ld_20
return cross_sectional_rank(alternation)
```

**意义**：同一股票20日内既出现涨停又出现跌停=情绪剧烈反转(多空激烈博弈)——研报《基于板块效应动量反转特征》指出情绪反转点常伴随大级别变盘。涨停数与跌停数的乘积大=两种极端同时频繁出现。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### limit_board_streak_mean_60

**定义**：平均连板高度因子：60日封板天数/连板启动次数截面排名（历史连板惯性排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
sealed = (daily["pct_chg"] >= _LIMIT_UP).astype(float)
# 连板启动日=当日封板且前一日未封板;shift 后首日 NaN 视为未启动(fillna False)。
# shift 会把 numpy bool 列提升为 object dtype(无法表示 NaN),必须再转回 bool,
# 否则对 object 列做位取反会触发 numpy 2.x 的 DeprecationWarning。
prev_sealed = sealed.astype(bool).groupby(level="Code").shift(1).fillna(False).astype(bool)
start = (sealed.astype(bool) & ~prev_sealed).astype(float)
# 涨停事件稀疏:两窗口均 min_periods=1(60 日内 ≥1 个封板日/启动日即估,
# panic_selling_ratio_60 先例);无任何封板日的股票用 0 中性填充
days60 = sealed.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=1).sum()
)
starts60 = start.astype(float).groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=1).sum()
)
avg = safe_divide(days60, starts60).fillna(0.0)
return cross_sectional_rank(avg)
```

**意义**：策略10追三板的核心是连板高度——60日内封板天数除以连板启动次数=该股被资金连续拉板的惯性禀赋:反复 2-3 板被砸的票与单次 5 板以上的历史妖股在此分离。与 consecutive_limit_up(当前连板数)互补:本因子是历史均值,刻画「拉板基因」而非当下状态。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### limit_down_event_5

**定义**：跌停事件衰减因子：pct_chg≤−9.8%近似跌停事件后指数衰减（半衰期3日，恐慌延续排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
down = daily["pct_chg"].le(-9.8)
event = down.astype(float).where(down, np.nan)
decayed = event_decay(event, half_life=3)
return cross_sectional_rank(decayed)
```

**意义**：跌停是极端恐慌事件——跌停后惯性下杀与超跌反弹并存。衰减信号捕捉恐慌的时间结构：刚跌停的股票承压最强。作为风险警示因子，排名高=近期恐慌。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### limit_down_rebound_10

**定义**：跌停后表现因子：10日内有跌停事件的股票其10日累计收益（跌停后反弹排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_ld = daily["pct_chg"].le(-9.8)
has_ld_10 = is_ld.astype(float).groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=1).max()
)
ret = daily["pct_chg"] / 100.0
ret_10 = ret.groupby(level="Code").transform(
    lambda s: (1.0 + s).cumprod().pct_change(10, fill_method=None)
)
rebound = has_ld_10 * ret_10
return cross_sectional_rank(rebound)
```

**意义**：跌停后的中期表现检验恐慌是否过度——过去10日跌停过的股票,其10日累计收益衡量超跌反弹的兑现程度(研报《如何捕捉短线反弹机会》:跌停次日反弹胜率高)。滞后视角无未来函数,与 limit_up_fade_10 镜像。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### limit_streak_volume_ratio

**定义**：连板放量结构因子：当前连板数(≤5)×当日量/近3日最大量截面排名（连板且量创新高排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_lu = daily["pct_chg"].ge(_LIMIT_UP).astype(int)
code = is_lu.index.get_level_values("Code")
seg = (~is_lu.astype(bool)).groupby(level="Code").cumsum()
streak = is_lu.groupby([code, seg]).cumsum().clip(upper=5.0)
vol = daily["vol"]
max3 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(3, min_periods=1).max().shift(1)
)
ratio = safe_divide(vol, max3)
raw = streak * ratio
return cross_sectional_rank(raw)
```

**意义**：策略10的量能条件:追三板要求当日量是近 3 日最大——连板且当日量创新高=接力资金进场(换手板);连板但量萎缩=一字板(无法上车,断板风险由封单决定)。连板数与放量比的乘积在连板梯队内部再分层,与既有consecutive_limit_up 只度量高度区分。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### limit_up_event_5

**定义**：涨停事件衰减因子：pct_chg≥9.8%近似涨停事件后指数衰减（半衰期3日，强势延续排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
event = daily["pct_chg"].ge(9.8).astype(float).where(
    daily["pct_chg"].ge(9.8), np.nan
)
decayed = event_decay(event, half_life=3)
return cross_sectional_rank(decayed)
```

**意义**：涨停是A股最强的情绪事件——涨停后通常有2-3日的惯性溢价(策略10追三板逻辑)。用event_decay(半衰期3日)把稀疏涨停事件转为连续衰减信号：刚涨停的股票信号最强，随后衰减。涨停频发+近期=连板强势，排前。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### limit_up_fade_10

**定义**：涨停后表现因子：10日内有涨停事件的股票其10日累计收益（涨停后延续排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
is_lu = daily["pct_chg"].ge(9.8)
has_lu_10 = is_lu.groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=1).max()
)
ret = daily["pct_chg"] / 100.0
ret_10 = ret.groupby(level="Code").transform(
    lambda s: (1.0 + s).cumprod().pct_change(10, fill_method=None)
)
fade = has_lu_10.astype(float) * ret_10
return cross_sectional_rank(fade)
```

**意义**：涨停后的中期表现是情绪溢价的检验——过去10日内涨停过的股票，其10日累计收益衡量「追涨停资金」的盈亏状态：收益为正=涨停后行情延续(强者恒强)，收益为负=涨停后回落(情绪退潮)。滞后视角无未来函数。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### limit_up_open_fail_freq_20

**定义**：炸板频率因子：20日盘中触板(高点≥9.8%)但收盘未封板的天数占比截面排名（炸板频发排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 同日比较:high 与当日 pre_close 天然同尺度(pre_close 在除权日已复权到
# 当日价格尺度),无需 scale 折算——折算反而会把相对量级(adj≈1.0)混入真实价格。
high_pct = safe_divide(daily["high"] - daily["pre_close"], daily["pre_close"]) * 100.0
touched = high_pct.ge(9.8)
sealed = daily["pct_chg"].ge(9.8)
fail = (touched & ~sealed).astype(float)
freq = _roll_mean(fail, 20, 5)
return cross_sectional_rank(freq)
```

**意义**：炸板(触板未封)是情绪由强转弱的直接证据——涨停封单被抛压击穿,说明该价位承接不足。炸板频繁=情绪票、主力出货特征,是追高策略的反向指标。high 与当日 pre_close 同尺度比较,除权日不产生假触板。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### limit_up_vol_shrink_60

**定义**：缩量涨停率因子：60日涨停日均量/非涨停日均量截面排名（负向，涨停放量分歧排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
sealed = daily["pct_chg"] >= _LIMIT_UP
vol = daily["vol"]
# 涨停日稀疏:涨停侧 min_periods=1(60 日内 ≥1 个涨停日即估均值,panic_selling
# 先例),非涨停侧稠密用常规 min_periods=30;无涨停日的股票比值为 NaN,
# 用 1.0 中性填充(与一字板频率类 0 填充同思路,保证全市场覆盖)
sealed_vol = vol.where(sealed).groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=1).mean()
)
plain_vol = vol.where(~sealed).groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
ratio = safe_divide(sealed_vol, plain_vol).fillna(1.0)
return cross_sectional_rank(-ratio)
```

**意义**：缩量涨停=惜售=筹码锁定良好(一字板特征),放量涨停=分歧大(换手充分、随时可能被砸)。涨停日量均相对非涨停日量均的比值连续度量封板形态,覆盖一字板以外的全部封板方式,与 one_word_limit_up_freq_20(仅数一字板)互补。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### new_high_60_event

**定义**：60日新高事件衰减因子：复权价创60日新高事件后指数衰减（半衰期5日，突破强势排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
is_high = adj.eq(adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).max()
))
event = is_high.astype(float).where(is_high, np.nan)
decayed = event_decay(event, half_life=5)
return cross_sectional_rank(decayed)
```

**意义**：创60日新高是趋势突破的事件化表达——突破后惯性延续(海龟/Donchian逻辑)。基于复权基座判定新高，除权日不产生假突破(与7-31 52周修复同口径)。衰减信号衡量突破的新鲜度。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### new_high_frequency_60

**定义**：60日新高频率因子：60日内创新高天数占比截面排名（趋势强势频率排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
is_high = adj.eq(adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).max()
))
freq = is_high.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
return cross_sectional_rank(freq)
```

**意义**：创新高的频率比单次新高更稳健——60日内多次创新高=持续趋势(强势股票)；仅一次新高=脉冲行情。基于复权基座，除权日不产生假新高。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### new_low_60_event

**定义**：60日新低事件衰减因子：复权价创60日新低事件后指数衰减（半衰期5日，破位风险排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
is_low = adj.eq(adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).min()
))
event = is_low.astype(float).where(is_low, np.nan)
decayed = event_decay(event, half_life=5)
return cross_sectional_rank(decayed)
```

**意义**：创60日新低是趋势破位的事件化表达——新低后惯性下杀与超跌反弹并存。基于复权基座，除权日不产生假新低。作为风险因子与 new_high_60_event 镜像。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

##### one_word_limit_down_freq_20

**定义**：一字跌停频率因子：20日一字跌停天数占比截面排名（恐慌锁死排前，负向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
no_range = (daily["high"] - daily["low"]).abs().lt(1e-6)
one_word = (no_range & daily["pct_chg"].le(-9.8)).astype(float)
freq = _roll_mean(one_word, 20, 5)
return cross_sectional_rank(-freq)
```

**意义**：一字跌停=卖盘封死、无法出逃的流动性冻结——是极端恐慌的信号,一字跌停频发股票的流动性风险与后续踩踏风险高。与一字涨停镜像,负向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### one_word_limit_up_freq_20

**定义**：一字涨停频率因子：20日一字板(全天无波动且涨停)天数占比截面排名（一字连板强势排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
no_range = (daily["high"] - daily["low"]).abs().lt(1e-6)
one_word = (no_range & daily["pct_chg"].ge(9.8)).astype(float)
freq = _roll_mean(one_word, 20, 5)
return cross_sectional_rank(freq)
```

**意义**：一字涨停(open=high=low=close)意味着买盘封死、筹码完全锁仓——是A股最强情绪形态(策略10追三板的高阶形态)。一字板频繁出现=强庄控盘,但也伴随流动性风险。20日频率平滑稀疏事件。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/structure_patterns.py`

##### three_black_crows

**定义**：三只黑鸦形态因子：连续阴线(收盘<开盘且低于前收)天数计数截面排名（连续阴跌排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
open_ = daily["open"]
# pre_close is the ex-date reference close supplied by daily.parquet;
# raw close.shift(1) fabricates a bearish step on corporate-action days.
cond = (close < open_) & (close < daily["pre_close"])
cond_i = cond.astype(int)
code = cond_i.index.get_level_values("Code")
seg = (~cond_i.astype(bool)).groupby(level="Code").cumsum()
count = cond_i.groupby([code, seg]).cumsum()
return cross_sectional_rank(count)
```

**意义**：三只黑鸦是经典顶部形态(策略51)：连续三根阴线收盘逐步走低=空头主导。实现为连续阴跌天数计数(不限于3天)，连续阴跌越长空头动能越强。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/event_dynamics.py`

#### <a name="cat-risk-c1"></a>类别 risk — 风险（38 个）

##### amihud_asymmetry_20

**定义**：涨跌日流动性不对称因子：下跌日Amihud/上涨日Amihud截面排名（负向，恐慌性难出货排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret_w = _ret_wide(daily)
illiq_w = _amihud(daily).unstack("Code")
up_illiq = illiq_w.where(ret_w > 0).rolling(20, min_periods=5).mean()
down_illiq = illiq_w.where(ret_w < 0).rolling(20, min_periods=5).mean()
asym = safe_divide(down_illiq, up_illiq + 1e-12)
return cross_sectional_rank(-stack_date_code(asym))
```

**意义**：下跌日与上涨日冲击成本的差异揭示抛售的性质——下跌日流动性显著更差=恐慌性单边出货(无人接盘),上涨日更差=缩量阴跌(观望情绪)。不对称度高=流动性风险在坏消息时集中爆发,负向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### amihud_trend_20_60

**定义**：Amihud流动性趋势因子：20日Amihud/60日Amihud截面排名（负向，流动性恶化排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
illiq_w = _amihud(daily).unstack("Code")
a20 = illiq_w.rolling(20, min_periods=10).mean()
a60 = illiq_w.rolling(60, min_periods=30).mean()
ratio = safe_divide(a20, a60 + 1e-12)
return cross_sectional_rank(-stack_date_code(ratio))
```

**意义**：近期非流动性相对中期水平抬升=流动性正在恶化(接盘变少、冲击成本上升)——流动性恶化的股票在下跌市中更易踩踏,是隐性的下行风险放大因子。与 amihud_daily_20(水平)互补:本因子捕捉趋势方向而非绝对水平。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### atr_position_250

**定义**：ATR历史位置因子：ATR20相对自身250日分布的标准化偏离截面排名（负向，波动分位高排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# 同日比较:high/low 与当日 pre_close 天然同尺度,除权日无假 TR
h = daily["high"]
l = daily["low"]
pc = daily["pre_close"]
tr = pd.Series(
    np.maximum.reduce([h - l, (h - pc).abs(), (l - pc).abs()]),
    index=h.index,
)
atr20 = tr.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
g = atr20.groupby(level="Code")
mean250 = g.transform(lambda s: s.rolling(250, min_periods=120).mean())
std250 = g.transform(lambda s: s.rolling(250, min_periods=120).std())
z = safe_divide(atr20 - mean250, std250)
return cross_sectional_rank(-z)
```

**意义**：当前波动率在自身一年历史中的相对位置:低分位=波动压缩末端(变盘前夜),高分位=波动宣泄中(风险释放,追涨胜率低)。与 vol_cycle_position_120(成交量周期)区分:本因子是 ATR 价格波动维度,用 z-score 度量(向量化,滚动分位在 5000 股票规模不可行)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### atr_ratio_20

**定义**：ATR比率因子，ATR_20/close截面排名（高相对波幅=高风险排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
high = daily["high"]
low = daily["low"]
# ATR 折算到复权空间(×scale)后与复权基座 adj 同口径,避免除权日 close
# 跳变造成相对波幅虚高(2026-08-05 修复)
scale = _adjusted_close(daily) / close.replace(0, np.nan)
adj = _adjusted_close(daily)

tr1 = high - low
tr2 = (high - daily["pre_close"]).abs()
tr3 = (low - daily["pre_close"]).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = (tr * scale).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)

atr_pct = safe_divide(atr, adj + 1e-10)
return cross_sectional_rank(-atr_pct)  # low relative ATR = stable
```

**意义**：ATR as a percentage of price measures relative volatility. Unlike standard deviation which treats all deviations equally, ATR focuses on the true range (including overnight gaps). This captures gap risk that standard deviation misses. ATR/Price is directly comparable across stocks.

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_pattern.py`

##### beta_60

**定义**：60日市场贝塔因子（对全池等权市场收益的60日滚动beta，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
mkt = _market_proxy(wide)
beta = _rolling_beta(wide, mkt, 60, 30)
return cross_sectional_rank(-stack_date_code(beta))
```

**意义**：低贝塔异象（A股同样成立）：高贝塔股票承担更多系统性风险却未被充分补偿。低贝塔股票排名靠前。与日内微观结构风险因子互补，是日频系统性风险暴露的基础衡量。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/market_relative.py`

##### corr_market_60

**定义**：60日市场相关性因子（个股日收益与全池等权市场收益的60日滚动相关，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
mkt = _market_proxy(wide)
corr = wide.rolling(60, min_periods=30).corr(mkt)
return cross_sectional_rank(-stack_date_code(corr))
```

**意义**：与市场高度同步的股票缺乏独立alpha来源、且在系统性下跌时无分散价值；低相关股票更可能由自身基本面驱动。低相关性排名靠前，与 beta_60 互补（相关性与贝塔的差异在于特质波动成分）。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/market_relative.py`

##### cvar_95_120

**定义**：120日CVaR因子：1%至5%滚动分位的积分近似（条件尾部损失），排名高=尾部风险小。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret_w = _ret_wide(daily)
# CVaR_5% = (1/0.05) * integral_0^0.05 VaR_q dq.  Averaging five
# contemporaneous rolling quantiles is a vectorised Riemann approximation
# and, unlike filtering each old return by its old threshold, describes
# the *current* 120-day distribution at every date.
quantiles = [
    ret_w.rolling(120, min_periods=60).quantile(q)
    for q in (0.01, 0.02, 0.03, 0.04, 0.05)
]
cvar_w = sum(quantiles) / len(quantiles)
return cross_sectional_rank(stack_date_code(cvar_w))
```

**意义**：CVaR(条件VaR)取最坏5%日收益的均值,比 var_95_20 的单点分位数更稳健地度量尾部损失的期望深度——两只股票 VaR 相同但 CVaR 更负者尾部更肥(厚尾风险,研报《A股市场特征研究》尾部相关性语境)。120日窗口保证尾部样本充足(期望6个极端日),估计稳定、覆盖率高;排名方向与 var_95_20 一致。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_structure.py`

##### downside_frequency_60

**定义**：下行频率因子：60日负收益天数占比截面排名（负向，频繁下跌排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
neg = _ret(daily).lt(0)
freq = neg.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
return cross_sectional_rank(-freq)
```

**意义**：60日负收益天数占比衡量「钝刀子割肉」式的持续阴跌风险——单日大跌可由事件解释，但高频负收益=股票缺乏上行弹性。与波动率正交：低波动+高频下行=阴跌股，风险高。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### downside_upside_vol_60

**定义**：下行/上行波动比因子：60日负收益波动/正收益波动截面排名（负向，下行风险主导排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = _ret(daily)
down = ret.where(ret < 0)
up = ret.where(ret > 0)
# 正/负样本各自稀疏:60日窗口内 min_periods=5 即可估计
down_std = down.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=5).std()
)
up_std = up.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=5).std()
)
ratio = safe_divide(down_std, up_std + 1e-10)
return cross_sectional_rank(-ratio)
```

**意义**：下行波动大于上行波动=下跌比上涨更剧烈(空头主导、接盘意愿弱)；比值<1=上涨比下跌更有力度。衡量收益分布的非对称风险，与偏度互补。(命名加60与已删除的 downside_vol_ratio_20 区分)

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### downside_vol_ratio_20

**定义**：下行波动占比因子（20日半波动/总波动比，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
neg = wide.clip(upper=0.0)
var_all = (wide ** 2).rolling(20, min_periods=15).mean()
var_neg = (neg ** 2).rolling(20, min_periods=15).mean()
ratio = var_neg.pow(0.5) / var_all.pow(0.5).replace(0, np.nan)
return cross_sectional_rank(-stack_date_code(ratio))
```

**意义**：下行波动占比衡量收益波动的方向不对称性：比值高=下跌贡献了大部分波动（负向不对称），风险尚未充分定价。比值低=波动主要由上涨驱动，持有体验好。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/higher_moments.py`

##### drawdown_duration_120

**定义**：120日回撤持续期因子：当前价格低于120日滚动前高的连续天数（上限120日，负向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
rolling_high = adj.groupby(level="Code").transform(
    lambda s: s.rolling(120, min_periods=1).max()
)
in_dd = adj.lt(rolling_high * 0.999)
# 回撤段:从回到前高起计数,回撤中每天+1
duration = _consecutive_count(in_dd).clip(upper=120)
return cross_sectional_rank(-duration)
```

**意义**：回撤持续期=从前期高点回落至今的天数——持续期长=趋势破坏久、套牢盘积压重。基于复权基座的120日滚动高点判定前高，除权日无假回撤。负向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### drawdown_recovery_60

**定义**：回撤修复率因子：复权价/60日新高截面排名（正向，接近新高排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
high60 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).max()
)
recovery = safe_divide(adj, high60 + 1e-10)
return cross_sectional_rank(recovery)
```

**意义**：当前价相对60日新高的距离=回撤修复程度——接近新高=趋势修复完成(强势)，远离新高=回撤未修复(弱势)。与 drawdown_duration 互补(深度 vs 时间)。基于复权基座，除权日无假修复。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### extreme_gain_freq_20

**定义**：上行极端收益频率因子：20日收益高于均值+1.5σ的天数占比截面排名（负向，彩票偏好排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = daily["pct_chg"] / 100.0
mean20 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
std20 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
extreme_up = ret.gt(mean20 + 1.5 * std20)
freq = extreme_up.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-freq)
```

**意义**：彩票偏好异象的上行侧刻画:上行极端收益频发=收益分布呈彩票形态,散户高估其概率而推高价格,未来收益系统性偏低。与 fear_index_20(下行极端,正向反转逻辑)镜像,本因子取负向排名——上行极端频繁排后。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_structure.py`

##### fear_index_20

**定义**：恐惧指数因子：20日窗口内下行极端收益(低于均值1.5σ)的频率截面排名（恐慌频发排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = daily["pct_chg"] / 100.0
mean20 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
std20 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
extreme_down = ret.lt(mean20 - 1.5 * std20)
freq = extreme_down.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(freq)
```

**意义**：研报《度量市场恐惧与贪婪的量化择时指标》的个股化：下行极端收益频率衡量个股的恐慌状态——恐慌频发=情绪释放充分(反转做多机会)但也是高风险标签。取正向排名：极端下行频率高=超卖反转候选。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### gain_loss_asymmetry_60

**定义**：涨跌幅度不对称因子：60日平均涨幅/|平均跌幅|截面排名（涨多跌少排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret_w = _ret_wide(daily)
mean_up = ret_w.where(ret_w > 0).rolling(60, min_periods=10).mean()
mean_down = ret_w.where(ret_w < 0).rolling(60, min_periods=10).mean()
asym = safe_divide(mean_up, mean_down.abs() + 1e-10)
return cross_sectional_rank(stack_date_code(asym))
```

**意义**：平均涨幅与平均跌幅的比值捕捉收益分布的第一阶不对称——涨多跌少=多头占据主动(买盘强于卖盘,趋势质量好);涨少跌多=空头主导。与downside_upside_vol_60(二阶矩比)互补:一只股票可以波动对称但幅度不对称。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_structure.py`

##### idio_vol_60

**定义**：60日特质波动率因子（剔除市场暴露后的残差波动，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
mkt = _market_proxy(wide)
beta = _rolling_beta(wide, mkt, 60, 30)
resid = wide - beta.multiply(mkt, axis=0)
idio = resid.rolling(60, min_periods=30).std()
return cross_sectional_rank(-stack_date_code(idio))
```

**意义**：特质波动率异象（Ang et al.）：特质波动率高的股票未来收益系统性偏低（套利限制+彩票偏好），A股广泛验证。残差法：ret - beta*mkt 的60日滚动标准差。8.2 曾删除了 regime 分片的 idiosyncratic_vol_60_*，此为未分片完整版。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/market_relative.py`

##### loss_probability_20

**定义**：损失概率因子：20日负收益频率截面排名（负向，高损失概率排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
neg = _ret(daily).lt(0)
freq = neg.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
return cross_sectional_rank(-freq)
```

**意义**：20日损失概率衡量短期亏钱体验——概率高=高频小幅下跌(负期望暴露)。与 downside_frequency_60 的区别在于窗口更短、更贴近近期状态。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### market_beta_change_20

**定义**：贝塔变化因子（20日贝塔-60日贝塔=系统性风险暴露的短期变化）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
mkt = _market_proxy(wide)
beta20 = _rolling_beta(wide, mkt, 20, 10)
beta60 = _rolling_beta(wide, mkt, 60, 30)
change = beta20 - beta60
return cross_sectional_rank(stack_date_code(change))
```

**意义**：贝塔加速上升=资金正系统性涌入（风险偏好抬升），常伴随行情启动；贝塔快速下降=防御性调仓。贝塔动量捕捉市场风格切换的先行信号，与绝对贝塔水平正交。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/market_relative.py`

##### market_regime_sensitivity_60

**定义**：市场状态敏感度因子：60日下跌市均收益/|上涨市均收益|（下跌市抗跌排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
mkt = _market_proxy(wide)
stock_on_down = wide.where(mkt < 0).rolling(60, min_periods=10).mean()
stock_on_up = wide.where(mkt > 0).rolling(60, min_periods=10).mean()
sens = safe_divide(stock_on_down, stock_on_up.abs() + 1e-10)
return cross_sectional_rank(stack_date_code(sens))
```

**意义**：把收益按市场涨/跌状态条件化:下跌市中的均收益相对上涨市中的均收益之比,衡量股票对市场状态的对称性——下跌市跌得比上涨市涨得多=高beta且不对称(危机敏感股);下跌市抗跌=防御属性(低系统性下行暴露)。是无条件 beta_60 的状态条件化版本,对尾部市况的刻画更直接。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_structure.py`

##### max_consecutive_loss_20

**定义**：最长连亏因子：20日内最长连续亏损天数截面排名（负向，长连亏排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
neg = _ret(daily).lt(0)
consec = _consecutive_count(neg)
max_loss = consec.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=5).max()
)
return cross_sectional_rank(-max_loss)
```

**意义**：20日内最长连续亏损天数衡量下跌的「韧性」——连亏7天比断续亏7天伤害更大(持仓者更容易在底部割肉)。连续段计数后取20日窗口最大值。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### overnight_std_20d

**定义**：隔夜收益波动（overnight 的20日std）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "overnight_std_20d", roll(df, "overnight", 20, "std", min_periods=5))
```

**意义**：隔夜波动反映消息面不确定性：高隔夜波动=定价信息冲击频繁，风险溢价要求更高。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### panic_selling_ratio_60

**定义**：放量下跌占比因子：60日放量(vol>1.5×20日均量)且下跌天数/放量天数（恐慌抛售排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = daily["pct_chg"] / 100.0
vol = daily["vol"]
ma20 = vol.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
big = vol.gt(1.5 * ma20)
down = ret.lt(0)
panic = (big & down).astype(float)
# min_periods=1:只要有 ≥1 个放量日即可估计占比
n_big = big.astype(float).groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=1).sum()
)
n_panic = panic.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=1).sum()
)
ratio = safe_divide(n_panic, n_big)
return cross_sectional_rank(ratio)
```

**意义**：放量下跌是恐慌性抛售的直接证据(量大且方向向下)——放量日下跌占比高=筹码在下跌中充分换手、情绪集中释放,超跌反转候选(与 fear_index_20 同逻辑);放量日多为上涨=资金进场推动。60日估计窗保证低放量频率股票也有足够样本;联合量价条件,区分于纯量(volume_spike_event)与纯价(downside_frequency_60)因子。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_structure.py`

##### range_vol_ratio_20

**定义**：区间波动比因子：20日高低价区间/20日收益波动截面排名（负向，极端区间排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
close = daily["close"]
scale = _adjusted_close(daily) / close.replace(0, np.nan)
adj = _adjusted_close(daily)
adj_high = daily["high"] * scale
adj_low = daily["low"] * scale
hi20 = adj_high.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).max()
)
lo20 = adj_low.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).min()
)
vol20 = _ret(daily).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
# 分母用复权基座 adj 与分子(复权空间区间)同口径——未复权 close 在除权日
# 跳变会瞬时改变分母标尺(2026-08-05 修复)
ratio = safe_divide(hi20 - lo20, vol20 * adj + 1e-10)
return cross_sectional_rank(-ratio)
```

**意义**：20日(高-低)区间相对收益波动的比值——比值高=价格在区间内大幅摆动(游资博弈、方向不明)，比值低=单边趋势(方向确定)。区间用复权折算后的high/low(除权日不失真)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### ret_autocorr_20d

**定义**：收益自相关（日收益20日滚动 lag-1 自相关）。

**公式（计算逻辑）**：

```python
df = _daily(context)
vals = df.groupby("Code")["ret"].transform(
    lambda s: s.rolling(20, min_periods=10).corr(s.shift(1))
)
return _out(df, "ret_autocorr_20d", vals)
```

**意义**：正自相关=动量延续，负自相关=日内反转；刻画收益序列的持续性/均值回归特征。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### ret_kurt_20

**定义**：20日收益峰度因子（日收益20日滚动峰度，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
kurt = _ret_wide(daily).rolling(20, min_periods=15).kurt()
return cross_sectional_rank(-stack_date_code(kurt))
```

**意义**：高峰度=收益分布尾部厚重、极端行情频繁，风险溢价要求更高而实际收益往往更差（彩票型收益特征）。低峰度股票收益路径平稳，排名靠前。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/higher_moments.py`

##### ret_skew_20

**定义**：20日收益偏度因子（日收益20日滚动偏度，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
skew = _ret_wide(daily).rolling(20, min_periods=15).skew()
return cross_sectional_rank(-stack_date_code(skew))
```

**意义**：彩票偏好：高偏度股票被散户高估、未来收益更低，A股负偏度溢价显著。低（负）偏度股票收益分布稳健，排名靠前。与日内 rv_skew_intraday 互补——此处为日频收益率的三阶矩，捕捉跨日分布形态。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/higher_moments.py`

##### ret_skew_60

**定义**：60日收益偏度因子（日收益60日滚动偏度，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
skew = _ret_wide(daily).rolling(60, min_periods=40).skew()
return cross_sectional_rank(-stack_date_code(skew))
```

**意义**：中期收益偏度刻画季度级别的收益分布形态：持续正偏（偶发大涨）的股票常被市场高估，负偏度则隐含风险已被释放。60日窗口更稳健，与20日版本互补。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/higher_moments.py`

##### sortino_ratio_60

**定义**：60日Sortino比率因子：均收益/下行标准差截面排名（正向，风险调整收益高排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = _ret(daily)
mean60 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
down = ret.where(ret < 0)
# 下行样本稀疏:60日窗口内 min_periods=5 个负收益日即可估计
down_std = down.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=5).std()
)
sortino = safe_divide(mean60, down_std + 1e-10)
return cross_sectional_rank(sortino)
```

**意义**：Sortino比率只惩罚下行波动(优于Sharpe)——同收益下下行波动小的股票持有体验更好、回撤更浅。60日均收益/60日下行标准差，正向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### tail_corr_60

**定义**：60日尾部相关性因子：与市场同向极端收益(|z|>1.5)的频率截面排名（尾部同步排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
mkt = _market_proxy(wide)
z = wide.sub(wide.rolling(60, min_periods=30).mean()).div(
    wide.rolling(60, min_periods=30).std() + 1e-10
)
mkt_z = mkt.sub(mkt.rolling(60, min_periods=30).mean()).div(
    mkt.rolling(60, min_periods=30).std() + 1e-10
)
same_sign = np.sign(z).eq(np.sign(mkt_z), axis=0)
extreme = z.abs().gt(1.5)
tail_freq = (same_sign & extreme).rolling(60, min_periods=30).mean()
return cross_sectional_rank(-stack_date_code(tail_freq))
```

**意义**：研报《沪深300样本股尾部相关性》：尾部同向波动反映系统性风险的传导——与市场同时出现极端收益的股票在危机中无分散价值。60日窗口内个股|z|>1.5与市场同号的频率越高=尾部相关性越强，负向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/technical_daily.py`

##### tail_risk_pct_60

**定义**：尾风险频率因子：60日内|z|>2极端收益占比截面排名（负向，极端波动频发排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = _ret(daily)
mean60 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
std60 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).std()
)
z = safe_divide(ret - mean60, std60 + 1e-10)
freq = z.abs().gt(2).groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
return cross_sectional_rank(-freq)
```

**意义**：60日窗口内超过2倍标准差的极端收益频率=厚尾风险(研报《A股市场特征研究》尾部相关性)。极端波动频发=信息冲击不稳定、定价效率低，负向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### var_95_20

**定义**：20日VaR(95%)因子：收益5%分位数截面排名（负向，尾部损失大排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
var = _ret(daily).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).quantile(0.05)
)
return cross_sectional_rank(var)
```

**意义**：VaR(95%, 20日)衡量20日窗口内的最坏单日损失——5%分位数越负=尾部风险越大。是比标准差更贴合损失视角的风险度量(研报《A股市场特征研究》尾部相关性语境)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### vol_clustering_20

**定义**：波动聚集因子：|收益|的20日自相关截面排名（负向，波动持续聚集排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
abs_ret = _ret(daily).abs().unstack("Code")
lag = abs_ret.shift(1)
ac = abs_ret.rolling(20, min_periods=10).corr(lag)
return cross_sectional_rank(-stack_date_code(ac))
```

**意义**：波动聚集(GARCH特征)：|收益|自相关高=大波动日扎堆出现(风险持续时间长)，自相关低/负=波动独立分散(风险快速释放)。用宽表 rolling corr 向量化。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### vol_cycle_position_120

**定义**：波动周期位置因子：20日波动/120日内最低20日波动截面排名（负向，波动扩张排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = daily["pct_chg"] / 100.0
vol20 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
vol20_min120 = vol20.groupby(level="Code").transform(
    lambda s: s.rolling(120, min_periods=30).min()
)
pos = safe_divide(vol20, vol20_min120 + 1e-10)
return cross_sectional_rank(-pos)
```

**意义**：当前20日波动相对自身120日滚动最低水平的比值=波动周期中的绝对位置——接近1=波动压缩到极致(变盘前兆,方向未知但爆发在即);远高于1=波动扩张进行中(风险释放未完成)。与 vol_regime_switch_20(相对均值)互补,本因子以自身极值为基准,对波动率的「地量地价」更敏感。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_structure.py`

##### vol_decay_ratio_20

**定义**：波动衰减因子：20日波动/10日波动截面排名（负向，波动持续放大排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = _ret(daily)
vol20 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
vol10 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(10, min_periods=5).std()
)
ratio = safe_divide(vol20, vol10 + 1e-10)
return cross_sectional_rank(-ratio)
```

**意义**：20日波动相对10日波动的比值衡量波动的时间结构——>1=近期波动仍处高位(波动未衰减)，<1=波动正在收敛(风险释放)。与 vol_regime_switch_20 互补(更长基期)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

##### vol_of_vol_20d

**定义**：波动之波动（|日收益|的20日std）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "vol_of_vol_20d", roll(df, "absret", 20, "std", min_periods=5))
```

**意义**：波动状态不稳定（vol-of-vol 高）的股票交易成本高、regime 切换频繁，低 vol-of-vol 标的波动环境一致、更可预测。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### vol_of_vol_60

**定义**：波动率的波动因子（60日收益std的20日std，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
wide = _ret_wide(daily)
vol60 = wide.rolling(60, min_periods=30).std()
vov = vol60.rolling(20, min_periods=10).std()
return cross_sectional_rank(-stack_date_code(vov))
```

**意义**：波动率自身的不稳定性：vol-of-vol 高的股票波动状态频繁切换、regime 不稳定，预测难度与交易成本高；vol-of-vol 低=波动环境一致，低波异象更可靠。波动率二阶矩是常见风险因子缺项。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/higher_moments.py`

##### vol_of_vol_60d

**定义**：波动之波动（|日收益|的60日std）。

**公式（计算逻辑）**：

```python
df = _daily(context)
return _out(df, "vol_of_vol_60d", roll(df, "absret", 60, "std", min_periods=10))
```

**意义**：季度级别的波动状态稳定性，比 20d 更稳健地刻画波动 regime 切换频率。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_daily.py`

##### vol_regime_switch_20

**定义**：波动状态切换因子：20日波动/60日波动截面排名（负向，波动骤升排后）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = _ret(daily)
vol20 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
vol60 = ret.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).std()
)
ratio = safe_divide(vol20, vol60 + 1e-10)
return cross_sectional_rank(-ratio)
```

**意义**：20日波动相对60日波动的比值>1=近期波动放大(状态切换至高风险区)，<1=波动收敛(蓄势)。波动放大常伴随方向不明的剧烈博弈，负向排名。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/risk_metrics.py`

#### <a name="cat-timeseries-c1"></a>类别 timeseries — 时间序列（24 个）

##### cmo_20

**定义**：Chande动量振荡器因子：20日(涨额和−跌额和)/(涨额和+跌额和)截面排名（多空动能净额排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
ret = daily["pct_chg"] / 100.0
up = ret.clip(lower=0.0)
dn = (-ret).clip(lower=0.0)
g = up.groupby(level="Code")
up_sum = g.transform(lambda s: s.rolling(20, min_periods=10).sum())
dn_sum = dn.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
cmo = safe_divide(up_sum - dn_sum, up_sum + dn_sum)
return cross_sectional_rank(cmo)
```

**意义**：CMO(Chande Momentum Oscillator)度量动量与反向动量的对称净值——与 RSI 同族但窗口对称、无钝化:涨跌额旗鼓相当=0,单边=±100。与 gain_loss_asymmetry_60(60日)窗口互补,20日口径更贴近短周期轮动。用 pct_chg/100 做涨跌额,除权日无失真。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### drawdown_120

**定义**：120日最大回撤因子（当前价格相对120日窗口峰值回撤，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
peak = adj.groupby(level="Code").transform(
    lambda s: s.rolling(120, min_periods=60).max()
)
drawdown = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0
return cross_sectional_rank(drawdown)
```

**意义**：中期（半年）回撤深度衡量趋势的完整性与牛熊状态：回撤浅=仍处上行趋势，回撤深=趋势已破坏。与 drawdown_60 互补短中期风险维度。基于自建后复权基座，无除权假回撤。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### drawdown_60

**定义**：60日最大回撤因子（当前价格相对60日窗口峰值回撤，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
peak = adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).max()
)
drawdown = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0
# drawdown is non-positive, so ranking it directly puts values closest to
# zero (the shallowest drawdowns) first, as required by the factor thesis.
return cross_sectional_rank(drawdown)
```

**意义**：回撤深度反映近期下行风险与筹码套牢程度：回撤浅的股票趋势完整、上行斜率健康，回撤深的股票面临解套抛压。基于自建后复权基座，无未复权数据除权日的假回撤（8.10 删除的 max_drawdown_120 的合规重建）。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### kama_position_20

**定义**：KAMA位置因子：后复权收盘价相对20期自适应均线的偏离截面排名（价格站上KAMA排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
kama = adj.groupby(level="Code").transform(_kama)
pos = safe_divide(adj - kama, kama)
return cross_sectional_rank(pos)
```

**意义**：KAMA(自适应均线)在震荡市收紧、趋势市放宽,能过滤假突破——价格站上KAMA=自适应确认的趋势方向向上(策略01/82的自适应均线过滤器)。与 kama_efficiency_20(效率比 ER)互补:本因子是价格相对 KAMA 线的位置,直接度量趋势的方位。基于后复权基座计算,无除权失真。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### momentum_10

**定义**：10日后复权动量因子（自建后复权基座，无除权失真），截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
mom = adj.groupby(level="Code").transform(
    lambda s: s.pct_change(10, fill_method=None)
)
return cross_sectional_rank(mom)
```

**意义**：基于 pct_chg 累乘的自建后复权基座的短中期动量，捕捉 2 周趋势动能。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### momentum_20

**定义**：20日后复权动量因子（自建后复权基座，无除权失真），截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
mom = adj.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
return cross_sectional_rank(mom)
```

**意义**：经典 Jegadeesh-Titman 一个月动量区间。基于 pct_chg 累乘的自建后复权基座，A 股月度动量效应显著，且无未复权数据在除权日的跳空污染。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### momentum_5

**定义**：5日后复权动量因子（自建后复权基座，无除权失真），截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
mom = adj.groupby(level="Code").transform(
    lambda s: s.pct_change(5, fill_method=None)
)
return cross_sectional_rank(mom)
```

**意义**：基于 pct_chg 累乘的自建后复权基座的短周期动量。消除除权日跳变失真，point-in-time 稳定（新分红不回溯改写历史）。短期动量捕捉 1 周趋势动能。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### momentum_60

**定义**：60日后复权动量因子（自建后复权基座，无除权失真），截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
mom = adj.groupby(level="Code").transform(
    lambda s: s.pct_change(60, fill_method=None)
)
return cross_sectional_rank(mom)
```

**意义**：中周期（季度）动量，捕捉中期趋势延续。基于 pct_chg 累乘的自建后复权基座，与短周期动量互补，衡量趋势的持续性而非短期动能。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### momentum_accel_60_120

**定义**：60/120日动量加速度因子：(60日动量−120日动量)截面排名（中期趋势加速排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
g = adj.groupby(level="Code")
mom60 = g.transform(lambda s: s.pct_change(60, fill_method=None))
mom120 = g.transform(lambda s: s.pct_change(120, fill_method=None))
accel = mom60 - mom120
return cross_sectional_rank(accel)
```

**意义**：60日动量相对120日动量的差衡量季度尺度的趋势加速度——60日动量强于120日=近三个月正在加速上行,弱于120日=中期动能衰竭。与 momentum_stability_20_60(月尺度)互补,构成动量的二阶结构:水平(60)+加速度(60-120)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### momentum_stability_20_60

**定义**：20/60日动量趋势差因子（短期动量-中期动量=趋势加速度），截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
g = adj.groupby(level="Code")
mom20 = g.transform(lambda s: s.pct_change(20, fill_method=None))
mom60 = g.transform(lambda s: s.pct_change(60, fill_method=None))
accel = mom20 - mom60
return cross_sectional_rank(accel)
```

**意义**：短周期动量与中周期动量之差视为趋势加速度：差值为正=短线强于中线（加速上行），差值转负=短线动能衰竭（减速）。区分动量水平与动量变化率，在趋势延续与反转之间提供先行信号。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### ppo_signal_12_26_9

**定义**：PPO信号差因子：百分比价格振荡器(EMA12−EMA26)/EMA26减去其9日EMA截面排名（动能反转确认排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
g = adj.groupby(level="Code")
ema12 = g.transform(
    lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean()
)
ema26 = g.transform(
    lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean()
)
ppo = safe_divide(ema12 - ema26, ema26) * 100.0
signal = ppo.groupby(level="Code").transform(
    lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean()
)
return cross_sectional_rank(ppo - signal)
```

**意义**：PPO 是 MACD 的价格水平归一化变形——消除高价股/低价股的绝对量纲,使横盘低价股与高价股可比。PPO 与其 9 日 EMA 之差=柱状图的百分比版,突破 0 轴=动能反转确认。与 macd_* 绝对口径互补。基于自建后复权基座,无除权失真。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### price_distance_from_52w_low

**定义**：距52周低点距离因子：(adj−252日最低)/252日最低截面排名（远离年内低点排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
low_252 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(252, min_periods=120).min()
)
dist = safe_divide(adj - low_252, low_252)
return cross_sectional_rank(dist)
```

**意义**：George-Hwang 52周效应在低点侧:远离一年低点=趋势处于健康区间,贴着52周低点运行=持续弱势阴跌(研报《上市公司动量反转》:极值附近的反转与动量并存)。基于复权基座,除权日不产生假新低(与 price_to_52w_high 同族)。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### price_to_52w_high

**定义**：52周高点接近度因子（close/252日最高收盘价-1），截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
high_252 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(252, min_periods=120).max()
)
proximity = adj / high_252.replace(0, np.nan) - 1.0
return cross_sectional_rank(proximity)
```

**意义**：George-Hwang 52周高点效应：接近一年高点的股票在A股同样呈现动量延续，突破/接近高点时套牢盘释放完毕、上方阻力最小。基于 pct_chg 累乘的自建后复权基座，不受除权污染（8.10 曾因未复权口径删除 price_to_52w_high，此为合规重建）。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### rebound_from_low_20

**定义**：20日低点反弹幅度因子：当前价相对20日低点的涨幅截面排名（强劲反弹排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
low_20 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).min()
)
rebound = safe_divide(adj - low_20, low_20)
return cross_sectional_rank(rebound)
```

**意义**：当前价相对20日低点的反弹幅度=下跌后的修复弹性——反弹幅度大=买盘承接有力,趋势由弱转强;反弹幅度小=趴在低点(阴跌未止)。与 drawdown_60(相对峰值回撤)从两端刻画同一趋势:回撤管顶部、反弹管底部。复权基座,无除权假反弹。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_structure.py`

##### short_term_reversal_5

**定义**：5日短周期反转因子（负向5日动量），截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
mom5 = adj.groupby(level="Code").transform(
    lambda s: s.pct_change(5, fill_method=None)
)
return cross_sectional_rank(-mom5)
```

**意义**：A 股短周期（1-2 周）存在显著反转效应：近 5 日涨幅过大的股票随后回调概率高。取自建后复权基座 5 日动量的负向，排名靠前=短期超买待回落。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### time_since_52w_high

**定义**：距252日新高天数因子（最近一次创年内新高距今的天数，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
high_252 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(252, min_periods=120).max()
)
at_high = adj >= high_252
dates_dt = pd.to_datetime(
    daily.index.get_level_values("Date"), format="%Y%m%d"
)
# Keep only True rows, forward-fill the last new-high timestamp per stock.
hit_ts = at_high.where(at_high).mul(dates_dt.astype("int64"))
last_hit_ts = hit_ts.groupby(level="Code").ffill()
days_since = (dates_dt.astype("int64") - last_hit_ts) / 86_400_000_000_000
return cross_sectional_rank(-days_since)
```

**意义**：距创新高的时间越短，动量状态越新鲜；长期未能创新高说明趋势持续走弱。与 price_to_52w_high（接近度）互补：接近度衡量空间、时间衡量趋势新鲜度。自建后复权基座上判断新高，无除权假突破。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### ts_price_self_rank_60

**定义**：价格自身60日位置因子，收盘价在自身60日高低区间的相对位置截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
# Adjusted base (cumprod(1+pct_chg/100)) — raw close extrema are
# ex-dividend polluted for up to 60 days (see price_position_60).
adj = _adjusted_close(daily)

high_60 = rolling_group_max(adj, 60)
low_60 = rolling_group_min(adj, 60)

position = (adj - low_60) / (high_60 - low_60).replace(0, np.nan)
return cross_sectional_rank(position)
```

**意义**：现有price_position_60是截面对比，ts_price_self_rank_60是个股价格在自身60日范围内的位置，捕捉个股自身的超买超卖状态，与截面因子互补。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### ts_volume_zscore_20

**定义**：成交量20日Z-score因子，当日成交量偏离20日均值的标准差数截面排名（高放量排后）。

**公式（计算逻辑）**：

```python
daily_panel = context.load("daily.parquet")
vol = daily_panel["vol"]

mean_20 = rolling_group_mean(vol, 20)
std_20 = rolling_group_std(vol, 20)

zscore = (vol - mean_20) / std_20.replace(0, np.nan)
return cross_sectional_rank(-zscore)
```

**意义**：成交量异常放大（高Z-score）往往伴随信息冲击、主力进出或市场过度关注，后续可能面临反转压力。低Z-score（缩量）则可能处于蓄势阶段。Z-score标准化使不同股票的成交量更具可比性。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/price.py`

##### turnover_chg_20d

**定义**：换手率20日变化率（turnover_rate 20日pct_change）。

**公式（计算逻辑）**：

```python
df = _fin(context)
vals = df.groupby("Code")["turnover_rate"].pct_change(20)
return _out(df, "turnover_chg_20d", vals)
```

**意义**：月度换手变化反映交投活跃度的中期趋势切换，过滤短期脉冲。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### turnover_chg_5d

**定义**：换手率5日变化率（turnover_rate 5日pct_change）。

**公式（计算逻辑）**：

```python
df = _fin(context)
vals = df.groupby("Code")["turnover_rate"].pct_change(5)
return _out(df, "turnover_chg_5d", vals)
```

**意义**：换手率短期骤升=资金异动/事件催化，骤降=交投萎缩；变化率捕捉边际拐点。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### ulcer_index_20

**定义**：溃疡指数因子（20日窗口内回撤平方均值的平方根，反向排名）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
adj = _adjusted_close(daily)
peak = adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).max()
)
dd = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0
dd_sq = (dd ** 2).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
ulcer = dd_sq.pow(0.5)
return cross_sectional_rank(-ulcer)
```

**意义**：溃疡指数度量回撤的深度与持续性的综合风险（回撤面积）：它同时惩罚大回撤和长时间不回本，比单一最大回撤更平滑稳健。低溃疡指数=持有体验好、趋势平稳，A股稳健溢价支持其正alpha。基于自建后复权基座，无除权失真（8.10 删除的 ulcer_index_20 的合规重建）。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/momentum_rebuilt.py`

##### up_down_count_ratio_20

**定义**：20日涨跌天数比因子：(涨天数+1)/(跌天数+1)截面排名（涨多跌少排前）。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
up = (daily["pct_chg"] > 0).astype(float)
dn = (daily["pct_chg"] < 0).astype(float)
g = up.groupby(level="Code")
up20 = g.transform(lambda s: s.rolling(20, min_periods=10).sum())
dn20 = dn.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).sum()
)
ratio = safe_divide(up20 + 1.0, dn20 + 1.0)
return cross_sectional_rank(ratio)
```

**意义**：涨跌天数比刻画动量质量:涨多跌少=稳健多头(上涨有持续性),涨少跌多=阴跌。与 20 日动量(幅度维)互补——本因子只看方向频率,可识别「小步慢涨」(动量温和但天数比极高)与「大涨大跌」的区别。

**依赖数据**：`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/strategy_daily.py`

##### vol_ratio_ma20

**定义**：量比20日均值（volume_ratio 的20日滚动均值）。

**公式（计算逻辑）**：

```python
df = _fin(context)
return _out(df, "vol_ratio_ma20", roll(df, "volume_ratio", 20, "mean"))
```

**意义**：月度量能水平，识别持续放量/缩量阶段的切换，作为趋势确认的辅助信号。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

##### vol_ratio_ma5

**定义**：量比5日均值（volume_ratio 的5日滚动均值）。

**公式（计算逻辑）**：

```python
df = _fin(context)
return _out(df, "vol_ratio_ma5", roll(df, "volume_ratio", 5, "mean"))
```

**意义**：量比=当日成交量/过去5日均量，其均值的抬升反映近期持续放量，温和放量上行比单日爆量更可持续。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/fac_new_finance.py`

#### <a name="cat-coupling-c1"></a>类别 coupling — 因子耦合（8 个）

##### chip_above_below_ratio

**定义**：筹码压力比因子，(price-cost_85pct)/(cost_15pct-price)截面排名（上方套牢>下方获利=压力大排后）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
daily = context.load("daily.parquet")

# cyq 成本价为复权口径,把 close 折算到同一空间后再比较(见 chip._close_adj_basis)
close_adj = _close_adj_basis(daily)
cost_85 = cyq["cost_85pct"]
cost_15 = cyq["cost_15pct"]

common = close_adj.index.intersection(cost_85.index).intersection(cost_15.index)
above = close_adj.loc[common] - cost_85.loc[common]  # distance above 85th pct
below = cost_15.loc[common] - close_adj.loc[common]  # distance below 15th pct

ratio = safe_divide(above, below)
return cross_sectional_rank(-ratio)
```

**意义**：上方套牢盘与下方获利盘的比例关系衡量筹码的'重力方向'——上方套牢盘越重（高cost_85pct），股价上涨阻力越大；下方获利盘越多（低cost_15pct），股价下跌支撑越强。比值>2=压力远大于支撑。

**依赖数据**：`cyq_perf.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### chip_concentration_zone

**定义**：筹码集中区位因子，-(|winner_rate-0.5|)截面排名（获利盘50%=多空平衡=方向将出排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
wr = cyq["winner_rate"]
# Distance from 0.5, negative = closer to balance
zone = -np.abs(wr - 0.5)
return cross_sectional_rank(zone)
```

**意义**：获利盘比例在50%附近是多空力量最平衡的状态——平衡即将被打破，方向选择在即。获利盘>80%（超买）或<20%（超卖）则趋势可能耗尽。50%±15%是最佳'突破前夜'区间。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### chip_cost_momentum_20d

**定义**：筹码成本重心趋势因子，weight_avg的20日变化率截面排名（成本上移=资金抬轿排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
wa = cyq["weight_avg"]
mom = wa.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
return cross_sectional_rank(mom)
```

**意义**：筹码加权平均成本的变化方向反映资金流向——成本重心持续上移说明新增资金愿意以更高价格买入（看涨），成本重心下移则说明持仓者在不断降低预期。成本趋势是持仓者集体智慧的表达。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### chip_dispersion_width

**定义**：筹码成本离散度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（取负向=宽散=分歧大排后）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
width = (cyq["cost_95pct"] - cyq["cost_5pct"]) / cyq["cost_50pct"].replace(0, np.nan)
return cross_sectional_rank(-width)
```

**意义**：筹码成本分布的宽度反映市场参与者的分歧程度——成本高度集中（窄）意味着持仓者成本接近、形成共识，容易形成趋势；成本分散（宽）意味着多空分歧大、方向不明。窄幅集中的股票突破方向更确定。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### chip_peak_shift

**定义**：筹码峰移动因子，cost_50pct的20日变化率截面排名（中位数成本上移=看涨排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
c50 = cyq["cost_50pct"]
shift = c50.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
return cross_sectional_rank(shift)
```

**意义**：筹码中位数成本（50分位）的位置变化反映筹码峰在向哪个方向移动——中位数成本上升意味着市场整体成本在抬升、接盘力量充足；中位数成本下降则说明支撑在走弱。与cost_momentum互补：一个看均值移动，一个看中位数移动。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### chip_skewness_ratio

**定义**：筹码成本偏度因子，(cost_50pct-cost_5pct)/(cost_95pct-cost_50pct)截面排名（>1=低位密集看涨排前）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
lower_half = cyq["cost_50pct"] - cyq["cost_5pct"]
upper_half = cyq["cost_95pct"] - cyq["cost_50pct"]
skew = safe_divide(lower_half, upper_half)
return cross_sectional_rank(skew)
```

**意义**：筹码偏度是判断筹码在'低位密集'还是'高位密集'的核心指标——>1意味着中位数更靠近下沿（低位密集，看涨），<1意味着中位数更靠近上沿（高位密集，看跌）。这是筹码分布中最有预测力的单一形态指标。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### chip_support_strength

**定义**：筹码支撑强度因子，cost_15pct处的筹码密度×1/(价格-成本15pct距离)截面排名。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
daily = context.load("daily.parquet")

close_adj = _close_adj_basis(daily)
cost_15 = cyq["cost_15pct"]
cost_5 = cyq["cost_5pct"]

common = close_adj.index.intersection(cost_15.index)
price = close_adj.loc[common]
c15 = cost_15.loc[common]
c5 = cost_5.loc[common]

# Density proxy: how tight the lower cost range is (tighter = denser)
density = 1.0 / (c15 - c5).replace(0, np.nan).abs().clip(lower=0.01)
# Distance from price to support (c15), closer = stronger support
distance = (price - c15).abs().clip(lower=0.01)
strength = density / distance

return cross_sectional_rank(strength)
```

**意义**：底部的筹码密集区形成支撑——筹码支撑强度=底部筹码密度×距离倒数。底部筹码越多、价格离支撑越近，支撑力越强。这是技术分析中'筹码密集区是强支撑'的量化表达。

**依赖数据**：`cyq_perf.parquet`、`daily.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

##### winner_rate_momentum_5d

**定义**：获利盘变化率因子，winner_rate的5日变化截面排名（获利盘快速增加=短期过热排后）。

**公式（计算逻辑）**：

```python
cyq = context.load("cyq_perf.parquet")
wr = cyq["winner_rate"]
chg = wr.groupby(level="Code").transform(
    lambda s: s.diff(5)
)
return cross_sectional_rank(-chg)
```

**意义**：获利盘比例的短期变化速度是超买超卖的灵敏指标——5日内获利盘从30%飙升至70%意味着大量持仓者快速获利，短期获利了结压力上升（负向信号）。反之获利盘从70%跌至30%则恐慌盘出清（正向信号）。

**依赖数据**：`cyq_perf.parquet` ｜ **Class**：1 ｜ **源码**：`factors/chip_extended.py`

#### <a name="cat-neutral-c1"></a>类别 neutral — 中性化（2 个）

##### bp_size_neutral

**定义**：规模中性化BP因子，市值分桶内截面排名。消除市值与估值相关性。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
bp = 1.0 / finance["pb"].replace(0, np.nan)
neutral = _size_neutral_rank(bp, context)
return cross_sectional_rank(neutral)
```

**意义**：BP与市值存在系统性相关性（大盘股普遍PB较低），规模中性化后提取更纯粹的估值信号，避免选到全是银行股的'价值陷阱'。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/neutral.py`

##### turnover_20_size_neutral

**定义**：规模中性化换手率因子，市值分桶内低换手排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
turnover = finance["turnover_rate"]
to_20 = rolling_group_mean(turnover, 20)
neutral = _size_neutral_rank(-to_20, context)
return cross_sectional_rank(neutral)
```

**意义**：小盘股换手率天然高，规模中性化后找到同市值级别中换手率偏低（筹码稳定）的股票。

**依赖数据**：`finance.parquet` ｜ **Class**：1 ｜ **源码**：`factors/neutral.py`

#### <a name="cat-target-c1"></a>类别 target — 标签（Label）（5 个）

##### label_ret_10d

**定义**：T+1开盘买入、T+11开盘卖出，10日目标收益。

**公式（计算逻辑）**：

```python
daily_adj = context.load("daily_adj.parquet")
return _compute_label_ret(daily_adj, 10, "label_ret_10d")
```

**意义**：两周持股周期，平衡信号衰减与换手成本。

**依赖数据**：`daily_adj.parquet` ｜ **Class**：1 ｜ **源码**：`factors/target/ret.py`

##### label_ret_1d

**定义**：T+1开盘买入、T+2开盘卖出，1日目标收益。

**公式（计算逻辑）**：

```python
daily_adj = context.load("daily_adj.parquet")
return _compute_label_ret(daily_adj, 1, "label_ret_1d")
```

**意义**：隔日开盘买入次日开盘卖出，最短持股周期，适合超短线截面策略。

**依赖数据**：`daily_adj.parquet` ｜ **Class**：1 ｜ **源码**：`factors/target/ret.py`

##### label_ret_20d

**定义**：T+1开盘买入、T+21开盘卖出，20日目标收益。

**公式（计算逻辑）**：

```python
daily_adj = context.load("daily_adj.parquet")
return _compute_label_ret(daily_adj, 20, "label_ret_20d")
```

**意义**：月度持股周期，IC衰减但换手成本大幅降低，适合低频截面策略。

**依赖数据**：`daily_adj.parquet` ｜ **Class**：1 ｜ **源码**：`factors/target/ret.py`

##### label_ret_3d

**定义**：T+1开盘买入、T+4开盘卖出，3日目标收益。

**公式（计算逻辑）**：

```python
daily_adj = context.load("daily_adj.parquet")
return _compute_label_ret(daily_adj, 3, "label_ret_3d")
```

**意义**：三日持股周期，介于超短线和周度之间，适合中高频截面策略。

**依赖数据**：`daily_adj.parquet` ｜ **Class**：1 ｜ **源码**：`factors/target/ret.py`

##### label_ret_5d

**定义**：T+1开盘买入、T+6开盘卖出，5日目标收益。

**公式（计算逻辑）**：

```python
daily_adj = context.load("daily_adj.parquet")
return _compute_label_ret(daily_adj, 5, "label_ret_5d")
```

**意义**：一周持股周期，是A股中频截面策略最常用的目标标签。

**依赖数据**：`daily_adj.parquet` ｜ **Class**：1 ｜ **源码**：`factors/target/ret.py`

---

### <a name="class-2"></a>Class 2 — cyq_chips 筹码类（39 个）

从 `cyq_chips/` 逐股票目录加载筹码分布数据，按股票单独处理（unified 单股 pass）。

#### <a name="cat-price-c2"></a>类别 price — 价格 / 量价（39 个）

##### chip_below_momentum

**定义**：下方筹码动量因子，chip_peak_ratio的5日变化截面排名。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress

close_map, close_adj_map = _load_close_maps(source_root, allowed)
ratio_series = _compute_chip_factor(
    source_root, allowed, "chip_below_ratio",
    close_map=close_map, close_adj_map=close_adj_map, on_progress=on_progress,
)
if isinstance(ratio_series, pd.DataFrame):
    ratio_series = ratio_series.squeeze(axis=1)
chg = ratio_series.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(chg)
```

**意义**：下方筹码占比的快速增加意味着价格在向上突破——越来越多的筹码从上方套牢转为下方盈利。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_bimodality

**定义**：筹码双峰因子：|主峰价−中位价|/标准差截面排名（负向，双峰分歧排后）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_mode_median_gap")
return cross_sectional_rank(-s)
```

**意义**：双峰分布=筹码聚集在两个分离的价格带(多空两大阵营成本分离),是分歧加剧、变盘前的典型结构;单峰分布=筹码共识度高。以模式价与中位价的分离度衡量双峰性,对分布形态比对偏度(三阶矩)更直接。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_cr3_factor

**定义**：筹码CR3集中度因子，前三峰筹码占比截面排名。CR3高=多个价格区间筹码集中=多级支撑结构。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_cr3", on_progress=on_progress)
return cross_sectional_rank(s)
```

**意义**：CR3衡量前三高筹码峰的总占比。与单峰纯度(chip_peak_purity)不同，CR3能识别多峰分布中的集中度——即使分布是多峰的，只要前三个峰合计占比高，说明筹码仍集中。CR3>0.6通常意味着三峰已覆盖大部分持仓成本。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_cr3_momentum

**定义**：筹码CR3动量因子，前三峰占比的5日变化截面排名。CR3上升=筹码向多个核心锚点集中=结构性收集。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_cr3", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(chg)
```

**意义**：CR3上升意味着分布在多个价位形成支撑点——不是单一主力在吸筹，而是多路资金在多个价格区间同时收集。CR3快速上升往往预示更为稳健的多级支撑结构。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_cv_factor

**定义**：筹码变异系数因子，chip_cv=std/mean截面排名（取负=低CV排前）。低CV=分布相对均值集中=风险可控。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_cv", on_progress=on_progress)
return cross_sectional_rank(-s)
```

**意义**：变异系数将标准差按均价归一化，消除股价水平的影响。低CV=筹码在均价周围的相对离散度小、持仓成本一致性强。与chip_dispersion互补：CV跨股票可比、chip_dispersion保留价格量纲。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_cv_momentum

**定义**：筹码CV动量因子，变异系数的5日变化截面排名（取负=CV降排前）。CV收窄=相对离散度降低=筹码趋于集中。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_cv", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(-chg)
```

**意义**：变异系数的变化消除了股价波动的影响。CV快速收窄=筹码在均价周围的相对分散度缩小=持仓成本趋同=主力吸筹信号。比绝对宽度的变化更干净，特别是对于股价波动较大的标的。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_deep_trap_ratio

**定义**：深套筹码占比因子：成本价>1.1×close的筹码比例截面排名（负向，深套盘多排后）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_upper_110", need_close=True)
return cross_sectional_rank(-s)
```

**意义**：深套盘(成本在现价10%以上)是反弹的抛压来源——深套盘越重,上方解套卖出的意愿越强,反弹持续性越差;深套盘轻=筹码干净、拉升阻力小。与 chip_above_ratio(全部上方筹码)区分:本因子聚焦深度套牢区。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_dispersion

**定义**：筹码分布宽度因子，chip_weighted_std截面排名（取负=窄分布排前）。窄分布=筹码集中=一致预期强。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_weighted_std", on_progress=on_progress)
return cross_sectional_rank(-s)
```

**意义**：分布宽度（加权标准差）衡量芯片在价格轴上的分散程度。标准差小=所有持仓成本集中在窄区间=筹码结构紧凑、主力控盘能力强。与(cost_95pct-cost_5pct)/cost_50pct不同，标准差使用全部分布计算，对离群价格敏感。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_dispersion_momentum

**定义**：筹码宽度动量因子，chip_weighted_std的5日变化截面排名（取负=收窄排前）。宽度收窄=筹码凝聚=突破前兆。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_weighted_std", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(-chg)
```

**意义**：分布宽度的变化方向比绝对宽度更有前瞻性。宽度快速收窄=资金集中+筹码被持续吸收，是典型的主力收集特征。宽度扩张=筹码分散+出货信号。变化的边际意义大于静态状态。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_entropy_convergence

**定义**：筹码熵收敛因子，分布熵的5日变化截面排名（取负=熵降排前）。熵下降=信息收敛=预期正在趋于一致。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_entropy", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(-chg)
```

**意义**：熵值下降=筹码分布从无序到有序——市场对合理价格区间正在形成共识，这是价格突破前的一个重要信号。连续多日熵降往往预示趋势性行情即将启动。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_entropy_signal

**定义**：筹码熵信号因子，分布信息熵截面排名（取负=低熵排前）。低熵=结构有序=主力控盘明显。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_entropy", on_progress=on_progress)
return cross_sectional_rank(-s)
```

**意义**：信息熵衡量筹码分布的无序程度。低熵=筹码集中在少数价格水平（有清晰主峰）=市场参与者对估值区间形成共识。高熵=筹码均匀分散在各价位=多空意见分歧大、缺乏明确方向。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_gini_factor

**定义**：筹码基尼系数因子，分布Gini系数截面排名。高Gini=筹码集中在少数价位=价格锚定清晰。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_gini", on_progress=on_progress)
return cross_sectional_rank(s)
```

**意义**：基尼系数是经典的不平等度量。高Gini=少数价位占有大部分筹码=市场对核心成本区形成强共识。低Gini=筹码均匀分布在各价位=多空分歧大。与熵信息互补：Gini对均匀分布更敏感。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_gini_momentum

**定义**：筹码基尼动量因子，Gini系数的5日变化截面排名。Gini上升=筹码向少数价位集中=均衡被打破、新共识形成中。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_gini", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(chg)
```

**意义**：Gini系数上升（不平等度增加）=筹码从均匀分布转向集中分布=市场在淘汰'非共识'价位上的筹码。Gini的边际变化通常比绝对水平更有信号意义——快速的Gini上升是个强烈的趋势信号。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_high_float_ratio

**定义**：高浮盈筹码占比因子：成本价<0.9×close的筹码比例截面排名（负向，深度获利盘多排后）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_below_90", need_close=True)
return cross_sectional_rank(-s)
```

**意义**：高浮盈筹码(成本在现价10%以下)是获利兑现的来源——浮筹占比高=大量持仓者盈利丰厚、随时可能卖出锁定利润(上涨持续性存疑);浮筹少=上方无阻力或获利盘已充分换手。是获利盘因子的分布细粒度版本。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_iqr_factor

**定义**：筹码四分位距因子，分布IQR（P75-P25）截面排名（取负=窄IQR排前）。窄IQR=核心50%筹码高度集中。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_iqr", on_progress=on_progress)
return cross_sectional_rank(-s)
```

**意义**：四分位距仅衡量中间50%筹码的分布宽度，剔除上下25%极端值的影响。IQR窄=主力持仓成本高度集中在一个小区间=突破时的合力和方向确定性更强。与chip_dispersion互补：IQR对尾部异常值不敏感。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_iqr_momentum_20d

**定义**：筹码四分位距20日变化因子：IQR的20日变化截面排名（收窄排前）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_iqr")
return cross_sectional_rank(_momentum(s, window=20))
```

**意义**：四分位距的20日变化捕捉中期筹码收敛/发散——IQR 持续收窄=筹码向中枢凝聚(主力吸筹特征),发散=筹码松动(派发特征)。比5日变化更平滑,适合中期筹码结构判断。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_loss_peak_frac

**定义**：套牢筹码集中度因子：现价上方筹码中最大成本峰占比截面排名（负向，单点套牢压力排后）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_loss_peak_frac", need_close=True)
return cross_sectional_rank(-s)
```

**意义**：现价上方(套牢)筹码中最大单箱的占比——套牢盘集中成峰=该价位是明确的解套抛压点(突破时压力集中,反复受阻);分散=套牢盘零散(压力绵长但单点薄弱)。方向 neg:集中套牢排后。与 chip_deep_trap_ratio(深套盘总量)区分:本因子是套牢盘的分布形态维度。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_mean_distance

**定义**：筹码均价距离因子，(close-chip_weighted_mean)/close截面排名。价格在加权平均成本上方=多数盈利+强支撑。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
mean_series = _compute_chip_factor(
    source_root, allowed, "chip_weighted_mean", on_progress=on_progress,
)
daily_panel = context.load("daily.parquet")
close = daily_panel["close"]
common = close.index.intersection(mean_series.index)
distance = (close.loc[common] - mean_series.loc[common]) / close.loc[common].replace(0, np.nan)
return cross_sectional_rank(distance)
```

**意义**：与chip_peak_distance（模态成本）类似，但使用全部分布的加权平均成本——这是比模态峰价更稳健的平均持仓成本估计。价格高于均价=多数持仓者盈利=下方支撑可靠。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_median_distance

**定义**：筹码中位数距离因子，(close-chip_median_price)/close截面排名。价格在持仓成本中位数上方=多数盈利+支撑。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
median_series = _compute_chip_factor(
    source_root, allowed, "chip_median_price", on_progress=on_progress,
)
daily_panel = context.load("daily.parquet")
close = daily_panel["close"]
common = close.index.intersection(median_series.index)
distance = (close.loc[common] - median_series.loc[common]) / close.loc[common].replace(0, np.nan)
return cross_sectional_rank(distance)
```

**意义**：使用中位数成本（而非众数峰价或加权均值）来度量价格偏离。中位数对极端值不敏感，是更稳健的'典型成本'估计。价格高于中位数成本意味着超过一半的持仓者盈利。与chip_peak_distance和chip_mean_distance形成估计方法上的三角互补。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_median_momentum

**定义**：筹码中位成本动量因子：中位成本价5日变化截面排名（成本上移排前）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_median_price")
return cross_sectional_rank(_momentum(s))
```

**意义**：中位成本价上移=新进筹码成本抬升(资金在更高位接筹,趋势获确认);下移=成本重心下降(接盘乏力)。与价格动量正交:价格可因缩量上涨而动量高,但筹码成本不动=虚涨。成本动量为动量的筹码结构确认。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_mode_mean_convergence

**定义**：筹码众均收敛因子，|peak-mean|/mean的5日变化截面排名（取负=缺口收窄排前）。众数均值趋于一致=分布从多峰向单峰收敛。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_mode_mean_gap", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(-chg)
```

**意义**：众数-均值缺口收窄=分布结构从复杂（多峰、非对称）走向简单（单峰、对称）——这是筹码结构从混乱走向有序的过程。缺口的快速收窄意味着市场分歧正在消除，主力筹码正在形成统一的价格锚。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_mode_mean_gap

**定义**：筹码众数均值偏离因子，|peak_mean|/mean截面排名（取负=缺口小排前）。缺口小=分布对称+单峰结构健康。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_mode_mean_gap", on_progress=on_progress)
return cross_sectional_rank(-s)
```

**意义**：众数（筹码峰价格）与加权均值的差距衡量分布的对称性。缺口大=分布非对称/多峰——筹码结构不统一，可能有两组投资者在博弈。缺口小=单峰对称的分布、价格锚定清晰。与众数→右偏不同，此指标捕捉的是局部的结构不一致。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_p90_p10_factor

**定义**：筹码P90-P10范围因子，分布90%筹码价格区间宽度截面排名（取负=窄区间排前）。窄区间=核心筹码高度重叠。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_p90_p10", on_progress=on_progress)
return cross_sectional_rank(-s)
```

**意义**：P90-P10衡量90%筹码的价格宽度（剔除上下5%极端值）。比IQR更宽地覆盖分布、比全距更稳健。窄P90-P10=绝大多数持仓成本集中于小区间=共识基础扎实。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_p90_p10_momentum

**定义**：筹码P90-P10动量因子，90%筹码范围的5日变化截面排名（取负=收窄排前）。核心区间收窄=筹码进一步浓缩。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_p90_p10", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(-chg)
```

**意义**：P90-P10的收窄是筹码浓缩最直观的体现——90%的持仓成本区间缩小，说明高成本的套牢盘和低成本的获利盘正在被消化，筹码向中心价位收敛。范围快速缩小往往是强烈趋势的前奏。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_peak_distance

**定义**：筹码峰距离因子，(close-chip_peak_price)/close截面排名。价格在最大筹码峰上方=支撑排前。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress

peak_series = _compute_chip_factor(
    source_root, allowed, "chip_peak_price",
    on_progress=on_progress,
)

daily_panel = context.load("daily.parquet")
# cyq_chips 成本价为复权口径,把 close 折算到同一空间后再比较(见 chip._close_adj_basis)
close_adj = _close_adj_basis(daily_panel)
common = close_adj.index.intersection(peak_series.index)
distance = (close_adj.loc[common] - peak_series.loc[common]) / close_adj.loc[common].replace(0, np.nan)
return cross_sectional_rank(distance)
```

**意义**：最大筹码峰是最密集的持仓成本区，价格在峰上方时有强支撑，在峰下方时变为强阻力。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_peak_growing

**定义**：筹码主峰增强因子，主峰占比的5日变化截面排名。主峰增强=资金持续在核心价位收集筹码。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_peak_dominance", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(chg)
```

**意义**：主峰占比上升=越来越多的筹码集中到同一价格区间——主力资金在固定的核心价位持续吸筹。主峰强度连续上升往往是拉升前的重要信号。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_peak_purity

**定义**：筹码主峰纯度因子，最大峰占总筹码比例截面排名。主峰突出=清晰的价格锚点=有效的支撑/阻力位。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_peak_dominance", on_progress=on_progress)
return cross_sectional_rank(s)
```

**意义**：最大筹码峰占比越高，意味着最多的投资者在同一个价格附近持有筹码。这个价格因此具备最强的支撑/阻力含义。高峰占比=明确的价格锚定=突破该价位后的趋势确定性更强。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_peak_ratio

**定义**：下方筹码占比因子，当前价格以下的筹码面积占总筹码面积比例截面排名。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress

close_map, close_adj_map = _load_close_maps(source_root, allowed)
ratio_series = _compute_chip_factor(
    source_root, allowed, "chip_below_ratio",
    close_map=close_map, close_adj_map=close_adj_map, on_progress=on_progress,
)
return cross_sectional_rank(ratio_series)
```

**意义**：价格下方的筹码面积占比越高，意味着越多的持仓者处于盈利状态、且下方支撑越密集。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_percentile_20d

**定义**：筹码价格分位20日变化因子：当前价在筹码分布中的分位20日变化截面排名（分位抬升排前）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_percentile", need_close=True)
return cross_sectional_rank(_momentum(s, window=20))
```

**意义**：当前价格在筹码分布中的分位=「筹码被套比例」的连续版本——分位20日抬升=价格正穿越筹码密集区上行(套牢盘被解放,趋势获得筹码确认);分位下降=价格跌破成本中枢(抛压涌现)。比分位水平更具动态信息。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_range_skew_factor

**定义**：筹码分位数偏斜因子：(p90−p50)/(p50−p10)截面排名（负向，右偏上方筹码厚排后）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_range_skew")
return cross_sectional_rank(-s)
```

**意义**：分位数偏斜衡量筹码分布的不对称性:>1=上方(高价侧)筹码带更宽(套牢/浮筹堆积,上方压力大);<1=下方筹码带更宽(支撑厚)。比三阶矩偏度对离群价格更稳健,是 chip_skew_factor 的互补口径。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_semi_std_momentum

**定义**：筹码下行半方差动量因子：下行半标准差的5日变化截面排名（负向，下行离散扩大排后）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_semi_std")
return cross_sectional_rank(-_momentum(s))
```

**意义**：下行半方差(均值下方筹码的离散度)扩大=低位筹码带正在拉宽,承接力量分散、下方支撑弱化(风险信号);收窄=下方筹码集中、支撑增强。与 chip_downside_risk(水平)互补,捕捉其方向。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_skew_factor

**定义**：筹码偏度因子，分布偏度截面排名。右偏（正偏）=上方筹码多=牛市中换手充分、趋势惯性。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_skewness", on_progress=on_progress)
return cross_sectional_rank(s)
```

**意义**：偏度衡量筹码分布的对称性。右偏=均价上方有更多筹码分布（上升趋势中换手活跃），通常出现在上升趋势中。左偏=均价下方筹码堆积=套牢盘重。与cost_distribution_skew（仅用3个分位点估计）互补——此因子使用全部分布精确计算。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_skew_momentum

**定义**：筹码偏度动量因子，分布偏度的5日变化截面排名。偏度右移=筹码重心上移+趋势延续。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_skewness", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(chg)
```

**意义**：偏度的变化比静态偏度更有信号价值。偏度从负转正（或正向加大）=筹码重心向高价区移动=上升趋势有内在惯性。偏度从正转负=趋势可能转向——分布正在向低价区倾斜。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_tail_risk

**定义**：筹码尾部风险因子，分布超额峰度截面排名（取负=低峰度排前）。高峰度=肥尾=远离均价的极端筹码堆积、异动风险高。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_kurtosis", on_progress=on_progress)
return cross_sectional_rank(-s)
```

**意义**：超额峰度为正（尖峰+肥尾）时，筹码在远离均价的位置有额外堆积。正峰度=大量套牢盘或获利盘堆积在极端价位，当价格靠近时会有剧烈涌出。负峰度（低峰）=筹码分布较为均匀。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_tail_risk_change

**定义**：筹码尾部风险变化因子，分布峰度的5日变化截面排名（取负=尾部收窄排前）。尾部收窄=极端筹码被消化=结构改善。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress
s = _compute_chip_factor(source_root, allowed, "chip_kurtosis", on_progress=on_progress)
chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
return cross_sectional_rank(-chg)
```

**意义**：峰度变化反映尾部风险的边际变化。峰度下降（超额峰度减小）=极端价位的筹码正在被消化=尾部风险降低+筹码结构改善。峰度上升=需警惕极端价位附近的筹码压力积累。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep.py`

##### chip_weighted_mean_momentum

**定义**：筹码平均成本动量因子：加权平均成本价5日变化截面排名（成本上移排前）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_weighted_mean")
return cross_sectional_rank(_momentum(s))
```

**意义**：平均成本比中位成本对高价筹码更敏感(右偏分布下更高)——平均成本上移=大量资金在高位换手、整体持仓成本抬升,是趋势推进的筹码证据;与 chip_median_momentum 互补(均值 vs 中位数)。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_width_ratio_momentum

**定义**：筹码相对宽度动量因子：宽度比(std/close)的5日变化截面排名（负向，变宽排后）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_width_ratio")
return cross_sectional_rank(-_momentum(s))
```

**意义**：筹码相对宽度(std/close)收窄=分布向现价收敛(价格与筹码结构同步凝聚,突破能量积蓄);变宽=分布发散(换手分散)。宽度比已对价格水平归一,其变化衡量筹码结构相对价格的动态松紧。

**依赖数据**：`cyq_chips` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_win_peak_frac

**定义**：获利筹码集中度因子：现价下方筹码中最大成本峰占比截面排名（获利筹码集中锁筹排前）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_win_peak_frac", need_close=True)
return cross_sectional_rank(s)
```

**意义**：现价下方(获利)筹码中最大单箱的占比——衡量获利筹码是否集中成峰:集中=主力成本密集单一(吸筹完成、锁筹特征,回踩有支撑、上方抛压轻);分散=获利盘散落(浮筹多,涨时兑现压力大)。与 chip_peak_ratio(获利盘总量占比)区分:本因子是获利盘的分布形态维度,「吸筹 vs 出货」的关键

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

##### chip_win_peak_growth

**定义**：获利筹码集中度动量因子：获利筹码集中度的5日变化截面排名（吸筹中成本峰强化排前）。

**公式（计算逻辑）**：

```python
s = _chip_series(context, "chip_win_peak_frac", need_close=True)
return cross_sectional_rank(_momentum(s))
```

**意义**：获利筹码集中度(chip_win_peak_frac)的5日变化——吸筹过程中,主力吸筹导致获利筹码向成本峰凝聚(集中度抬升);派发阶段,获利盘扩散(集中度下降)。集中度上升=主力成本结构在强化,是「吸筹进行时」的动态信号,与 chip_win_peak_frac(静态水平)互补。

**依赖数据**：`cyq_chips`、`daily.parquet` ｜ **Class**：2 ｜ **源码**：`factors/chip_deep_extra.py`

---

### <a name="class-3"></a>Class 3 — history_1min 分钟行情类（114 个）

从 `history_1min/` 逐股票目录加载 1 分钟历史行情，构建日内/分钟级因子。

#### <a name="cat-intraday-c3"></a>类别 intraday — 日内 / 微观结构（114 个）

##### am_close_position

**定义**：上午收盘位置因子：上午收盘在上午振幅区间中的相对位置截面排名（上午收高排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "am_hl_position"))
```

**意义**：上午收盘位于上午区间上沿=上午多方完全控制(下午大概率延续);位于下沿=上午空方主导(下午承压)。与全日 close_position 区分:本因子以午间为分界,捕捉日内方向的时段归属与下午的延续概率。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### am_hl_range_intraday

**定义**：上午振幅因子，上午最高/上午最低-1截面排名（取负向=上午高振幅=分歧大排后）。

**公式（计算逻辑）**：

```python
ah = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "am_hl_range",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-ah)
```

**意义**：上午的振幅反映了隔夜信息消化过程中的多空分歧——上午振幅大意味着市场对新信息的价格发现还处于激烈博弈阶段，定价尚未收敛。低上午振幅意味着市场对信息的解读较为一致。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### am_momentum_intraday

**定义**：上午动量因子：上午收盘相对上午开盘的涨跌幅截面排名（上午走强排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "am_momentum"))
```

**意义**：上午动量(am_close/am_open−1)刻画上午半场的多空胜负——上午强=早盘信息被消化后多方持续占优;与已有 pm_momentum_intraday 对称,两者之差即日内方向的时段归属。分钟级口径,与日频 am_pm_return_ratio 互补。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### am_pm_range_ratio

**定义**：上午/下午振幅比因子：上午振幅/下午振幅截面排名（负向，早盘波动主导排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "am_pm_hl_range_ratio"))
```

**意义**：上午振幅占优=价格波动集中在早盘(隔夜信息冲击大、方向快速博弈);下午振幅占优=午后多空拉锯(方向不明)。早盘主导波动=日内方向早定,交易效率高;下午主导=全天反复,趋势质量差。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### am_pm_return_ratio

**定义**：上午/下午收益比因子，上午收益/下午收益截面排名（上午领涨=主动买入排前）。

**公式（计算逻辑）**：

```python
ar = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "am_pm_return_ratio",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(ar)
```

**意义**：上午和下午的收益分配反映不同类型资金的行为——上午收益主要由隔夜信息和机构调仓驱动，下午收益更多受短线资金和情绪影响。上午领涨的股票信息优势更强。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### am_pm_rv_ratio

**定义**：上午/下午波动率比因子，上午已实现方差/下午已实现方差截面排名。波动率的日内分布不对称性。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ar = _compute_intraday_factor(source_root, context.repo.allowed_codes, "am_pm_rv_ratio",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(ar)
```

**意义**：上午和下午的波动率分布反映不同类型的信息：上午波动主要由隔夜信息消化驱动，下午波动更多由盘中事件和尾盘博弈驱动。高am_pm_rv_ratio意味着信息冲击集中在上午（高效率定价），低比值则暗示不确定性延续到下午（定价效率低）。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### am_pm_vol_ratio

**定义**：上午/下午成交量比因子截面排名。上午放量=信息消化积极，下午放量=尾盘博弈。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ratio = _compute_intraday_factor(source_root, context.repo.allowed_codes, "am_pm_vol_ratio",
                                  on_progress=context.repo.on_progress)
return cross_sectional_rank(ratio)
```

**意义**：上午成交量占比高说明市场对隔夜信息反应积极，通常与正收益相关。下午成交量异常放大（尤其是最后半小时）往往是短期博弈行为，质量较差。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### am_vol_share

**定义**：上午成交量占比因子，上午成交量/全日成交量截面排名。上午占比高=信息消化积极。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
avs = _compute_intraday_factor(source_root, context.repo.allowed_codes, "am_vol_share",
                                on_progress=context.repo.on_progress)
return cross_sectional_rank(avs)
```

**意义**：上午成交量占比反映了隔夜信息和开盘信息的消化强度。上午成交占比高的股票说明市场对信息的反应积极且集中，而非拖到尾盘博弈。与am_pm_vol_ratio相比，am_vol_share更直观地衡量上午的相对活跃度。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### amihud_intraday

**定义**：高频Amihud非流动性因子，5分钟|ret|/amount均值截面排名（取负向=高流动性排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
amihud = _compute_intraday_factor(source_root, context.repo.allowed_codes, "amihud_5min",
                                   on_progress=context.repo.on_progress)
return cross_sectional_rank(-amihud)
```

**意义**：基于5分钟的高频Amihud比日度版本精确一个数量级，避免了日度数据的Epps效应。高非流动性=高交易成本=需要更高收益补偿。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### bv_daily

**定义**：日度双幂变差因子，1分钟数据双幂变差截面排名（低波排前）。连续价格变动的稳健波动估计。

**公式（计算逻辑）**：

```python
bv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "bv_daily",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-bv)
```

**意义**：双幂变差(BV)是对跳跃稳健的波动率估计量(Barndorff-Nielsen & Shephard 2004)——通过相邻收益的乘积而非平方来降低跳跃的影响。BV与RV的差异(RV²-BV²)可分离出跳跃成分。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### close_30_momentum

**定义**：尾盘30分钟动量因子：收盘半小时(14:30-15:00)涨跌幅截面排名（尾盘拉升排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "close_30_mom"))
```

**意义**：尾盘半小时动量反映收盘定调——A股尾盘拉升常是主力做收盘价/次日溢价的行为(次日惯性),尾盘砸盘=次日低开风险。尾盘方向对次日开盘有预测力,与 close_auction_impact(最后3分钟)区分:本因子为30分钟窗口。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### close_5min_momentum

**定义**：尾盘5分钟动量因子，最后5分钟收益截面排名（取负向=尾盘拉升=次日易低开排后）。

**公式（计算逻辑）**：

```python
cm = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "close_5min_momentum",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-cm)
```

**意义**：尾盘最后5分钟的收益率是A股T+1制度下最具争议的信号——尾盘拉升往往是为了做高收盘价(无法当日卖出)，次日大概率低开。尾盘打压则可能是为了次日低价吸筹。尾盘动量对次日开盘有显著的负向预测能力。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### close_auction_impact

**定义**：收盘竞价影响因子，最后3分钟收益截面排名（正拉升=尾盘抢筹排前）。

**公式（计算逻辑）**：

```python
ca = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "close_auction_impact",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(ca)
```

**意义**：A股收盘前3分钟(14:57-15:00)是集合竞价阶段——这3分钟的涨跌反映了资金对收盘价的态度。收盘竞价拉升通常是机构为了提高净值或技术面做收盘价，次日高开概率较大。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### close_auction_ret

**定义**：收盘集合竞价收益因子：14:57开盘→15:00收盘涨跌幅截面排名（负向）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "close_auction_impact"))
```

**意义**：集合竞价是收盘定价的最终博弈——竞价段急拉=尾盘抢筹但次日兑现压力大,竞价段平稳/回落=定价充分。与既有 close_auction_impact 逐值等价。V8 筛查 meanIC -0.023/ICIR -0.50（方向最稳）。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### close_auction_vol_share

**定义**：收盘集合竞价量占比因子：14:57-14:59成交额占全天比例截面排名（负向）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "close_auction_vol_share"))
```

**意义**：竞价量占比=大资金借收盘定价窗口调仓的强度——竞价放量=尾盘博弈激烈、次日兑现压力大；竞价平静=定价充分。V8 筛查 meanIC -0.0143/ICIR -0.21，方向与既有 open_volume_share(neg) 一致。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### close_position

**定义**：收盘位置因子，(收盘-最低)/(最高-最低)截面排名。收盘在高位=买方主导全日。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
cp = _compute_intraday_factor(source_root, context.repo.allowed_codes, "close_position",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(cp)
```

**意义**：收盘价在日内价格区间的位置反映了多空力量博弈的最终结果。收盘在区间上沿（close_position≈1）意味着买方在尾盘占据主导，次日延续概率较高。收盘在低位（≈0）则卖方压力较大。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### cumulative_ret_path

**定义**：累积收益路径效率因子，|总收益|/Σ|1分钟收益|截面排名（高效率=趋势明确排前）。

**公式（计算逻辑）**：

```python
cr = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "cumulative_ret_path",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(cr)
```

**意义**：累积收益路径效率是趋势强度的度量——如果股票全天持续上涨(|总收益|/Σ|每分钟收益|接近1)，说明买方持续主导。如果比值小(0.1-0.3)，说明日内涨跌互现、无明确方向。高效率的日内路径预示着更强的短期趋势。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### extreme_move_count

**定义**：极端波动次数因子，日内|ret_5min|>3σ的次数截面排名（取负向=频繁极端波动排后）。

**公式（计算逻辑）**：

```python
em = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "extreme_move_count",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-em)
```

**意义**：日内极端5分钟波动的次数直接度量了价格的'不平静'程度——即使总体RV不高，频繁的极端波动也意味着价格过程远非高斯，跳跃风险高。极端波动次数是对标准波动率的重要补充。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### flash_crash_risk

**定义**：闪崩风险因子，日内5分钟累计收益最大回撤截面排名（取负向=闪崩风险高排后）。

**公式（计算逻辑）**：

```python
fc = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "flash_crash_risk",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-fc)
```

**意义**：闪崩(Flash Crash)是短时间内价格剧烈下跌的现象——即使最终收盘涨回来，盘中闪崩也会触发大量止损单，对持仓者造成实质性伤害。闪崩风险高的股票需要额外的尾部风险溢价。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### gap_abs_intraday

**定义**：隔夜跳空绝对值因子，|开盘/前收-1|截面排名（取负向=大跳空排后）。隔夜信息冲击的绝对程度。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ga = _compute_intraday_factor(source_root, context.repo.allowed_codes, "gap_abs",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-ga)
```

**意义**：隔夜跳空的绝对值（不论方向）衡量信息冲击的强度。大幅跳空（无论高开低开）意味着信息不确定性高、定价分歧大。持续的隔夜大幅波动往往伴随更高的未来波动率和更差的风险调整收益。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### gk_vol

**定义**：Garman-Klass波动率估计因子，基于OHLC四价的高效波动率截面排名（低波排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
gk = _compute_intraday_factor(source_root, context.repo.allowed_codes, "gk_vol",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-gk)
```

**意义**：Garman-Klass(1980)利用OHLC四价信息的波动率估计量效率是收盘价波动率的7.4倍。相比Parkinson仅用HL，GK额外利用OC信息区分趋势日和震荡日，对开盘跳空更稳健。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### hl_range_intraday

**定义**：日内振幅因子，1分钟数据得到的日度(high/low-1)截面排名（低振幅=筹码稳定排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
hl = _compute_intraday_factor(source_root, context.repo.allowed_codes, "hl_range",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-hl)
```

**意义**：1分钟数据计算的日内振幅比日线OHLC更精确（排除集合竞价失真）。低日内振幅代表交易结构健康、多空力量均衡。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### intra_trend

**定义**：日内趋势强度因子，正收益5分钟区间占比截面排名。日内方向一致性度量。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
it_ = _compute_intraday_factor(source_root, context.repo.allowed_codes, "intra_trend",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(it_)
```

**意义**：日内收益方向的一致性（正收益5分钟区间的占比）反映了买方力量的持续性。高intra_trend意味着买家在全天持续主导，而非仅在个别时段发力。日内趋势的延续性与短期动量效应正相关。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### intraday_high_time

**定义**：日内最高价时点因子：日内最高价出现的归一化时段截面排名（尾盘创新高排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "intraday_high_time"))
```

**意义**：日内最高价出现时点是多空掌控力的指纹:强势股尾盘创新高(时点→1),弱势股早盘冲高回落(时点→0.1)。与 volume_peak_time(量能时点)正交:早盘放量+尾盘新高=有效放量,早盘放量+早盘见顶=出货特征。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### intraday_max_drawdown

**定义**：日内最大回撤因子，日内从最高点到后续最低点的最大跌幅截面排名（取负向=深回撤排后）。

**公式（计算逻辑）**：

```python
dd = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "intraday_max_drawdown",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-dd.abs())
```

**意义**：日内最大回撤衡量盘中持仓可能遭遇的最差情景——大回撤意味着即使收盘涨了，盘中也有大量资金被套。日内回撤小的股票持有人体验好、止损盘少、趋势更稳定。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### intraday_max_runup

**定义**：日内最大拉升因子，日内从最低点到后续最高点的最大涨幅截面排名（强拉升=买方力量排前）。

**公式（计算逻辑）**：

```python
ru = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "intraday_max_runup",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(ru)
```

**意义**：日内最大拉升反映买方在盘中的反击力度——在经历低点后能快速大幅拉升说明下方支撑坚实、买方力量强。大拉升后的股票短期动量和信心都更强。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### intraday_momentum

**定义**：日内动量因子，开盘30分钟收益截面排名。衡量隔夜信息消化后的早盘方向。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
# Compute open_30 return: (open_30 close / open) - 1
import os
min_dir = source_root / "history_1min"
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress

files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
parts = []
total = len(files)

for i, fname in enumerate(files):
    code = fname.replace(".parquet", "").split(".")[0].zfill(6)
    if allowed and code not in allowed:
        if on_progress: on_progress("intra_mom", i + 1, total)
        continue

    filepath = min_dir / fname
    try:
        stock_df = pd.read_parquet(filepath)
    except Exception:
        if on_progress: on_progress("intra_mom", i + 1, total)
        continue

    if stock_df.empty:
        if on_progress: on_progress("intra_mom", i + 1, total)
        continue

    stock_df["trade_date"] = pd.to_datetime(stock_df["trade_time"]).dt.strftime("%Y%m%d")
    stock_df["minute"] = pd.to_datetime(stock_df["trade_time"]).dt.hour * 60 + \
                         pd.to_datetime(stock_df["trade_time"]).dt.minute

    grouped = stock_df.groupby("trade_date")
    daily_open = grouped["open"].first()

    open_30 = stock_df[(stock_df["minute"] >= 570) & (stock_df["minute"] <= 600)]
    if open_30.empty:
        if on_progress: on_progress("intra_mom", i + 1, total)
        continue

    open_30_close = open_30.groupby("trade_date")["close"].last()
    ret = open_30_close / daily_open.replace(0, np.nan) - 1.0
    ret = ret.dropna()
    if ret.empty:
        if on_progress: on_progress("intra_mom", i + 1, total)
        continue

    s = pd.DataFrame({"ret": ret.values},
                     index=pd.MultiIndex.from_arrays(
                         [ret.index, [code] * len(ret)], names=["Date", "Code"]
                     ))["ret"]
    parts.append(s)
    if on_progress: on_progress("intra_mom", i + 1, total)

if not parts:
    return cross_sectional_rank(pd.Series(dtype=float))

result = pd.concat(parts)
result.index = result.index.set_names(["Date", "Code"])
result = result.reorder_levels(["Date", "Code"]).sort_index()
result = result.groupby(list(result.index.names)).last()
if isinstance(result.index, pd.MultiIndex):
    result.index = result.index.set_names(["Date", "Code"])
return cross_sectional_rank(result)
```

**意义**：开盘30分钟的方向往往由隔夜信息和集合竞价阶段的市场情绪决定，延续性较强。'开盘定方向'在A股中具有显著的日内动量效应。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### intraday_reversal

**定义**：尾盘反转因子，收盘30分钟收益截面排名（取负向=尾盘拉升排后，易次日低开）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
import os
min_dir = source_root / "history_1min"
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress

files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
parts = []
total = len(files)

for i, fname in enumerate(files):
    code = fname.replace(".parquet", "").split(".")[0].zfill(6)
    if allowed and code not in allowed:
        if on_progress: on_progress("intra_rev", i + 1, total)
        continue

    filepath = min_dir / fname
    try:
        stock_df = pd.read_parquet(filepath)
    except Exception:
        if on_progress: on_progress("intra_rev", i + 1, total)
        continue

    if stock_df.empty:
        if on_progress: on_progress("intra_rev", i + 1, total)
        continue

    stock_df["trade_date"] = pd.to_datetime(stock_df["trade_time"]).dt.strftime("%Y%m%d")
    stock_df["minute"] = pd.to_datetime(stock_df["trade_time"]).dt.hour * 60 + \
                         pd.to_datetime(stock_df["trade_time"]).dt.minute

    close_30 = stock_df[(stock_df["minute"] >= 870) & (stock_df["minute"] <= 900)]
    if close_30.empty:
        if on_progress: on_progress("intra_rev", i + 1, total)
        continue

    cg = close_30.groupby("trade_date")
    ret = cg["close"].last() / cg["open"].first().replace(0, np.nan) - 1.0
    ret = ret.dropna()
    if ret.empty:
        if on_progress: on_progress("intra_rev", i + 1, total)
        continue

    s = pd.DataFrame({"ret": ret.values},
                     index=pd.MultiIndex.from_arrays(
                         [ret.index, [code] * len(ret)], names=["Date", "Code"]
                     ))["ret"]
    parts.append(s)
    if on_progress: on_progress("intra_rev", i + 1, total)

if not parts:
    return cross_sectional_rank(pd.Series(dtype=float))

result = pd.concat(parts)
result.index = result.index.set_names(["Date", "Code"])
result = result.reorder_levels(["Date", "Code"]).sort_index()
result = result.groupby(list(result.index.names)).last()
if isinstance(result.index, pd.MultiIndex):
    result.index = result.index.set_names(["Date", "Code"])
return cross_sectional_rank(-result)
```

**意义**：尾盘拉升（尤其是最后5-10分钟）往往是主力做收盘价的刻意行为，次日经常低开。尾盘反转效应是A股T+1制度下的特殊alpha来源。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### jump_ratio_intraday

**定义**：跳跃占比因子，日度已实现跳跃/日度RV截面排名（取负向=跳跃主导=不稳定排后）。

**公式（计算逻辑）**：

```python
jr = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "jump_ratio",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-jr)
```

**意义**：跳跃成分占总波动的比例反映了价格过程的'光滑度'——跳跃占比高意味着价格变动主要由不连续的冲击驱动(而非连续的扩散)，预测难度更大、套利风险更高。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### large_trade_intensity

**定义**：大单分钟集中度因子，前5%分钟成交额占全天比例截面排名（机构大额交易集中排前）。

**公式（计算逻辑）**：

```python
li = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "large_trade_intensity",
                              on_progress=context.repo.on_progress)
return cross_sectional_rank(li)
```

**意义**：分钟成交额的集中度——前5%分钟占据的成交额比例高=交易由少数大单主导(机构/主力行为，方向明确)；比例低=交易分散(散户主导、噪音多)。大单集中度是资金结构的日内度量。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### last30_ret

**定义**：尾盘30分钟收益因子：14:30开盘→收盘涨跌幅截面排名（负向）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "intra_rev_ret"))
```

**意义**：尾盘30分钟是全天定价的收官段——尾盘急拉=短线资金尾盘偷袭(次日惯性存疑),尾盘平稳=全天定价充分。与既有 intra_rev_ret 逐值等价。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### last30_vol_share

**定义**：尾盘30分钟量占比因子：14:30-14:59成交额占全天比例截面排名（正向）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "last30_vol_share"))
```

**意义**：尾盘量占比=资金在定价收官段的参与深度——尾盘放量(尤其伴随价格平稳)=机构调仓/建仓痕迹,次日延续性更强。与既有 tail_volume_share 的差异仅在 15:00 竞价条是否计入。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### lunch_break_effect

**定义**：午间休市效应因子，下午开盘价/上午收盘价-1截面排名（高值=午间利好堆积）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
lb = _compute_intraday_factor(source_root, context.repo.allowed_codes, "lunch_break_ret",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(lb)
```

**意义**：A股特有的11:30-13:00午间休市期间信息持续累积，下午开盘跳空幅度反映午间信息的冲击强度。持续的午间正跳空代表信息面偏积极。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### max_ret_intraday

**定义**：最大5分钟收益因子，日内最大|ret_5min|截面排名（取负向=极端波动排后）。捕获日内最剧烈的价格冲击。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
mr = _compute_intraday_factor(source_root, context.repo.allowed_codes, "max_abs_ret_5min",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-mr)
```

**意义**：日内最大绝对收益（最小最大值）捕获了传统矩（方差、偏度、峰度）无法完全概括的极端事件信息。单次极端5分钟波动往往对应着大单冲击、乌龙指或突发公告。极端值的大小与未来短期波动率正相关，对流动性较差的股票尤为显著。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### min_bar_gap_freq_20

**定义**：分钟跳空频率因子：20日(|分钟开盘/前分钟收盘−1|>0.2%)占比均值截面排名（负向，盘口断档排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "min_bar_gap_freq_20"))
```

**意义**：1 分钟级频繁跳空=连续竞价断档=流动性薄/大单砸单;低频率=盘口连续=流动性好(机构单边托单)。与 rjump_5min(跳变幅度)区分:本因子是跳空发生的频率(结构维),流动性差的题材小票频率数倍于蓝筹。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### min_limit_touch_frac_20

**定义**：分钟触板密度因子：20日(分钟涨幅≥9.8%占比)均值截面排名（封板维持时间长排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "min_limit_touch_frac_20"))
```

**意义**：日频炸板率(limit_up_open_fail_freq_20)只有「封没封上」两个状态,分钟线给出「在板上的时间」——封板时间长=封单牢固(秒板/回封与临收盘偷袭板在此分离)。用 pre_close×1.098 近似涨停价,与全库 9.8% 口径一致,除权日无假触板。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### min_ret_max

**定义**：1分钟收益最大值因子：日内最大分钟涨幅的截面排名（负向，急拉排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "ret_1min_max"))
```

**意义**：单分钟最大涨幅=瞬时脉冲式拉升(游资点火特征)——急拉分钟频繁出现的股票短期筹码结构不稳、追高风险大。V8 筛查 meanIC -0.038/ICIR -0.23。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### min_ret_min

**定义**：1分钟收益最小值因子：日内最大分钟跌幅的截面排名（负向绝对值，深跌排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "ret_1min_min"))
```

**意义**：日内单分钟最大跌幅刻画瞬时恐慌强度——深跌分钟=盘中瞬时抛压极端释放(闪崩风险),与 min_ret_max 构成分钟收益的尾部谱系。V8 筛查 meanIC +0.034/ICIR 0.20（原值方向为正）。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### min_ret_std

**定义**：1分钟收益标准差因子：日内分钟收益波动的截面排名（负向，波动大排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "ret_1min_std"))
```

**意义**：1 分钟颗粒度的收益波动捕捉 5 分钟口径（ret_5min_std）无法刻画的高频噪音结构——分钟波动大=日内定价分歧剧烈、短期持有风险高。V8 筛查 meanIC -0.038/ICIR -0.22，为分钟候选最强档。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### min_vwap_dev_std

**定义**：VWAP贴合度因子：20日(分钟价对当日VWAP偏离的标准差)均值截面排名（负向，价格围绕VWAP剧烈摆动排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "min_vwap_dev_std"))
```

**意义**：价格围绕当日 VWAP 的日内摆动幅度:稳定贴合=机构按 VWAP 单边建仓/控盘,剧烈摆动=多空拉锯(趋势质量差)。与 vwap_daily_deviation(仅收盘一个点)区分:本因子用全日内分布,趋势票与对倒票分离。口径修正:1min vol=手(实证),×100 后 amount/vol=真 VWAP。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### minute_ret_vol_corr

**定义**：分钟量价相关因子：日内分钟收益与分钟量的皮尔逊相关截面排名（量价同步排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "minute_ret_vol_corr"))
```

**意义**：分钟级收益-量相关是量价同步的微观度量:正相关=涨放量跌缩量(健康);负相关=涨缩量跌放量(背离)。与日频 turnover_ret_corr_20/volume_price_corr_20区分:本因子用日内分钟样本,不受跨日窗口长度影响,捕捉盘中即时响应。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### open_30_momentum

**定义**：开盘30分钟动量因子：开盘半小时(09:30-10:00)涨跌幅截面排名（开盘强势排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "open_30_mom"))
```

**意义**：开盘半小时是隔夜信息定价最集中的时段——开盘30分钟动量强=资金抢筹坚决(集合竞价+早盘首波买盘的真实意愿);弱=高开低走或低开无承接。与 open_auction_ret(开盘跳空)区分:本因子度量跳空之后的半小时走势。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### open_30_range_share

**定义**：开盘振幅占比因子：开盘30分钟振幅占全天振幅比例截面排名（负向，早盘大幅博弈排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "open_30_range_pct"))
```

**意义**：开盘30分钟振幅占全天比例高=全天的价格区间在早盘就已确定(方向快速)或早盘多空剧烈拉锯(方向未定);占比低=全天逐步展开。早盘定区间=日内趋势延续性强;早盘反复=方向博弈激烈,负向排名。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### open_5min_momentum

**定义**：开盘5分钟动量因子，前5分钟收益截面排名（开盘强势=隔夜利好排前）。

**公式（计算逻辑）**：

```python
om = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "open_5min_momentum",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(om)
```

**意义**：开盘前5分钟是集合竞价后的第一个连续交易时段——这5分钟的走势是市场对集合竞价定价的'确认'或'否定'。开盘5分钟继续上涨意味着隔夜利好被确认，有延续性。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### open_auction_ret

**定义**：集合竞价收益率因子，(开盘价/前日收盘-1)截面排名。隔夜信息冲击的直接度量。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
oa = _compute_intraday_factor(source_root, context.repo.allowed_codes, "open_auction_ret",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(oa)
```

**意义**：A股集合竞价(9:15-9:25)产生的开盘价包含了隔夜全部信息的综合定价。大幅高开往往延续（动量），大幅低开如果盘中回升则构成反转信号。此因子与overnight_gap互补——使用1分钟首笔数据更精准。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### open_close_momentum_gap

**定义**：首尾动量差因子：开盘30分钟动量−尾盘30分钟动量截面排名（负向，冲高回落排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "open_close_30_mom_ratio"))
```

**意义**：开盘强而尾盘弱=早盘透支、全天动能衰竭(冲高回落风险,次日低开概率大);开盘弱而尾盘强=尾盘修复(资金尾盘进场,次日惯性延续)。首尾强弱差捕捉日内动能的迁移方向,是单时段动量不具备的结构信息。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### open_volume_share

**定义**：开盘量能占比因子：开盘30分钟成交量占全天比例截面排名（负向，早盘情绪交易排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "open_30_vol_share"))
```

**意义**：开盘30分钟量能占比高=交易集中于早盘(散户情绪化参与高峰、信息冲击日);占比低=全天交易节奏均匀(机构化、策略化交易)。早盘集中放量常伴随日内冲高回落,负向排名。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### parkinson_vol

**定义**：Parkinson波动率估计因子，基于日内最高最低价的ln(H/L)/sqrt(4ln2)截面排名（低波排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
pv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "parkinson_vol",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-pv)
```

**意义**：Parkinson(1980)证明基于日内最高最低价的波动率估计量比收盘价波动率效率高5倍。HL范围包含了日内全部价格路径信息，而不仅仅是收盘时点。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### path_efficiency

**定义**：价格路径效率因子，|收盘-开盘|/(最高-最低)截面排名（高效=趋势性强排前）。

**公式（计算逻辑）**：

```python
pe = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "path_efficiency",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(pe)
```

**意义**：价格路径效率衡量日内价格是趋势性运动还是震荡运动——高效率(接近1)意味着价格从开盘到收盘基本单向运动，趋势明确。低效率(接近0)意味着日内大幅双向波动后回到原点，多空分歧大。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### pm_hl_range_intraday

**定义**：下午振幅因子，下午最高/下午最低-1截面排名（取负向=下午高振幅=尾盘博弈排后）。

**公式（计算逻辑）**：

```python
ph = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "pm_hl_range",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-ph)
```

**意义**：下午的振幅更多反映盘中新增信息和T+1博弈——下午振幅异常放大往往与短线资金的短线操作和尾盘避险/抢筹有关。下午振幅大的股票隔夜风险更高。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### pm_momentum_intraday

**定义**：下午动量因子，下午收盘/下午开盘-1截面排名（下午走强=买盘持续排前）。

**公式（计算逻辑）**：

```python
pm = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "pm_momentum",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(pm)
```

**意义**：下午的走势(尤其是下午开盘后)是A股短线交易的重要参考——经过午间休市的信息消化后，下午开盘的方向往往代表了机构资金的最终判断。下午持续走强比上午冲高下午回落更可靠。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### pos_rv_ratio

**定义**：正收益波动占比因子，正5分钟收益平方和/总RV²截面排名。上涨驱动的波动占比。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
pr = _compute_intraday_factor(source_root, context.repo.allowed_codes, "pos_rv_ratio",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(pr)
```

**意义**：将已实现方差分解为正收益贡献和负收益贡献(Barndorff-Nielsen et al. 2010)，捕捉波动的非对称性。高pos_rv_ratio意味着日内波动主要由上涨推动——买家主导价格发现过程。持续的高正波动占比是bullish信号。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### price_impact_asymmetry

**定义**：价格冲击不对称因子，-(上涨冲击/下跌冲击)截面排名（取负向=不对称=上涨费劲排后）。

**公式（计算逻辑）**：

```python
pa = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "price_impact_asymmetry",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-np.abs(pa - 1.0))
```

**意义**：上涨和下跌的价格冲击比反映了买卖方流动性供给的不对称——上涨比下跌需要更大成交量（高比值）意味着卖方挂单稀疏、上涨阻力大。低比值则意味着买方承接力弱。接近1的对称性是最优的交易结构。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### price_impact_intraday

**定义**：价格冲击因子（Kyle's Lambda），|收盘-开盘|/成交量截面排名（取负向=高冲击排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
pi = _compute_intraday_factor(source_root, context.repo.allowed_codes, "price_impact",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-pi)
```

**意义**：Kyle(1985)的价格冲击系数衡量单位成交量引起的价格变动。高价格冲击意味着市场深度浅、流动性差，交易成本高。1分钟数据计算的日内价格冲击比日度版本更能捕捉微观层面的流动性枯竭。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### range_rv_ratio

**定义**：振幅波动比因子，(high/low-1)/rv_5min截面排名（取负向=高比值排后）。跳成分相对于连续波动的比例。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rr = _compute_intraday_factor(source_root, context.repo.allowed_codes, "range_rv_ratio",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rr)
```

**意义**：日内振幅相对已实现波动的比值越高，说明价格变动中有更大的跳跃成分（而非连续扩散）。跳跃主导的价格过程意味着更大的尾部风险和更低的夏普比率。此因子与rjump_5min互补——前者是比值视角，后者是绝对跳跃强度。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### realized_spread_5min

**定义**：已实现价差因子，5分钟|ret|均值截面排名（取负向=高价差=高交易成本排后）。

**公式（计算逻辑）**：

```python
rs = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "realized_spread_5min",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rs)
```

**意义**：5分钟绝对收益的均值是bid-ask spread的代理变量——在有效市场中，价格在两个方向间弹跳产生小的绝对收益。绝对收益均值大意味着实际买卖价差大、交易成本高。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rel_vol_first_hour

**定义**：首小时量比因子，开盘首小时成交量/全日成交量截面排名（早盘活跃=信息驱动排前）。

**公式（计算逻辑）**：

```python
rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rel_vol_first_hour",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(rv)
```

**意义**：开盘首小时是A股全天成交最活跃的时段——首小时成交占比高意味着市场对隔夜信息的反应积极高效，价格发现质量高。首小时占比持续高的股票流动性更好。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rel_vol_last_hour

**定义**：尾小时量比因子，收盘前1小时成交量/全日成交量截面排名（取负向=尾盘博弈=不可靠排后）。

**公式（计算逻辑）**：

```python
rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rel_vol_last_hour",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：尾盘最后1小时成交占比过高是A股T+1制度下的特殊风险信号——尾盘拉抬或打压无法当日了结，次日走势往往反向。尾盘量比高意味着短线博弈而非价值投资。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rel_vol_midday

**定义**：午间量比因子，11:00-13:30成交量/全日成交量截面排名（取负向=午间异常放量排后）。

**公式（计算逻辑）**：

```python
rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rel_vol_midday",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：午间时段通常是A股成交最清淡的时段——午间异常放量往往意味着有特定消息或主力在这个'低关注窗口'进行操作。午间量比高是信息不对称的信号。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### relative_spread

**定义**：相对价差因子，(最高-最低)/VWAP截面排名（取负向=高振幅排后）。VWAP标准化后的日内波动幅度。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rs = _compute_intraday_factor(source_root, context.repo.allowed_codes, "relative_spread",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rs)
```

**意义**：以VWAP标准化的日内振幅剔除了价格水平的影响，使得高低价差在不同价格区间的股票之间可比。高相对价差代表日内价格波动剧烈、多空分歧大，通常伴随更高的未来波动和不确定性。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### ret_autocorr_5min

**定义**：5分钟收益自相关因子，日内5分钟收益一阶自相关系数截面排名。正自相关=日内动量，负自相关=均值回复。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ac = _compute_intraday_factor(source_root, context.repo.allowed_codes, "ret_autocorr",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(ac)
```

**意义**：高频收益的自相关结构反映市场微观结构特征：负自相关通常源于bid-ask bounce和存货管理（做市商行为），正自相关则暗示信息渐进扩散或趋势交易。A股市场中，负自相关较强的股票往往是做市商活跃的标的，流动性更好；正自相关则反映散户追涨杀跌。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### ret_autocorr_abs

**定义**：绝对收益自相关因子，|ret_5min|的一阶自相关系数截面排名（正自相关=波动聚集排前=预测性好）。

**公式（计算逻辑）**：

```python
ra = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "ret_autocorr_abs",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(ra)
```

**意义**：绝对收益的自相关(而非收益本身的自相关)是波动率聚集的微观体现——正自相关意味着大波动后继续大波动(波动聚集)，这是金融市场普遍存在的特征。较高的正自相关意味着波动率具有一定的可预测性(可使用GARCH类模型)。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### ret_kurt_intraday

**定义**：5分钟收益峰度因子，日内5分钟收益分布的峰度截面排名（取负向=厚尾排后）。衡量日内极端波动的集中度。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rk = _compute_intraday_factor(source_root, context.repo.allowed_codes, "ret_5min_kurt",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rk)
```

**意义**：金融时间序列的尖峰厚尾特征在日内高频数据中更加明显。高峰度意味着收益分布更集中（多数小幅波动）但尾部更厚（少数极端波动）。高峰度股票的日内价格过程包含更多跳跃成分，未来波动率更不稳定。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### ret_std_intraday

**定义**：5分钟收益标准差因子，日内5分钟收益截面标准差排名（取负向=高离散排后）。不同于RV用平方和，std衡量收益围绕均值的离散度。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rs = _compute_intraday_factor(source_root, context.repo.allowed_codes, "ret_5min_std",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rs)
```

**意义**：5分钟收益的标准差与RV高度相关但存在差异：趋势日（均值≠0）的std < sqrt(RV^2/n)，震荡日的std ≈ sqrt(RV^2/n)。std/RV的差异隐含日内方向性信息。高std代表价格在短时间内剧烈双向波动，信息不确定性大。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### return_asymmetry_intraday

**定义**：日内收益不对称因子，-(5分钟收益均值-中位数)截面排名（取负向=不对称=偏度大排后）。

**公式（计算逻辑）**：

```python
ra = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "return_asymmetry",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-ra.abs())
```

**意义**：收益均值与中位数的差异反映分布的偏度——均值>中位数意味着少数极端正收益拉高了均值（正偏），均值<中位数则是少数极端负收益（负偏）。不对称性大意味着价格过程包含跳跃。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rjump_5min

**定义**：已实现跳跃因子，RV_5min² - BV_5min²的正部开根截面排名（高跳跃排后=风险信号）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rj = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rj_5min",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rj)
```

**意义**：价格跳跃代表信息冲击或流动性断裂(Barndorff-Nielsen&Shephard 2004)。高跳跃股票面临更大的尾部风险，未来收益的波动率更高且偏负。跳跃强度是传统波动率无法捕捉的风险维度。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rjump_daily

**定义**：日度已实现跳跃因子，sqrt(max(RV²-BV²,0))截面排名（高跳跃=风险信号排后）。

**公式（计算逻辑）**：

```python
rj = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rjump_daily",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rj)
```

**意义**：日度已实现跳跃分离了价格过程中的不连续成分——跳跃代表信息冲击或流动性断裂。高跳跃股票面临更大的尾部风险和更不稳定的波动率。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rq_intraday

**定义**：已实现四次变差因子（Realized Quarticity），5分钟收益四次方和截面排名（取负向=高方差波动排后）。度量波动的波动。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rq = _compute_intraday_factor(source_root, context.repo.allowed_codes, "realized_quarticity",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rq)
```

**意义**：已实现四次变差(RQ)是已实现方差(RV²)的方差的估计量(Barndorff-Nielsen & Shephard 2002)。RQ衡量波动率的不确定性——即使两只股票的RV相同，RQ更高的股票未来波动率更不稳定，跳跃风险更大。RQ是波动率预测精度的关键变量。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rsv_5min

**定义**：已实现半方差因子（下行风险），仅5分钟负收益的平方和开根截面排名（高下行波排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rsv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rsv_5min",
                                on_progress=context.repo.on_progress)
return cross_sectional_rank(-rsv)
```

**意义**：下行已实现半方差(Barndorff-Nielsen et al. 2010)将波动率分解为非对称成分。投资者只厌恶下行波动，上行波动反映正信息冲击。RSV比RV更能捕捉真正的'坏波动'，对尾部风险定价更精准。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_10min

**定义**：10分钟已实现波动率因子，基于1分钟数据的10分钟收益平方和开根截面排名（低波排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_10min",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：10分钟频率的RV在5分钟（高噪声）和15分钟（响应慢）之间取得平衡，捕捉中等频率的波动动态。多频RV联合使用可提取波动率期限结构信息。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_15min

**定义**：15分钟已实现波动率因子，基于1分钟数据的15分钟收益平方和开根截面排名（低波排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_15min",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：15分钟已实现波动率比5分钟更平滑，减少 microstructure noise，同时比日度波动率更及时。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_30min

**定义**：30分钟已实现波动率因子，基于1分钟数据的30分钟收益平方和开根截面排名（低波排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_30min",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：30分钟频率的RV滤除了高频 microstructure noise，更接近'持久波动率'成分。低频RV对跳跃更稳健，与高频RV的差异可反映噪声大小。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_5min

**定义**：5分钟已实现波动率因子，基于1分钟数据的5分钟收益平方和开根截面排名（低波排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_5min",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：高频已实现波动率比日度波动率精确得多(Andersen&Bollerslev 1998)，捕捉日内信息流驱动的真实波动。低RV股票经风险调整后收益更高。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_60min

**定义**：60分钟已实现波动率因子，1分钟数据60分钟收益平方和开根截面排名（低波排前）。

**公式（计算逻辑）**：

```python
rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_60min",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：60分钟频率的RV捕捉更长周期的波动动态——相比5/15/30分钟RV，60分钟RV对 microstructure noise 更稳健，接近'长期波动率'成分。多频RV联合提供波动率期限结构信息。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_daily

**定义**：日度已实现波动率因子，1分钟数据全日收益平方和开根截面排名（低波排前）。

**公式（计算逻辑）**：

```python
rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_daily",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：日度RV使用全部240个1分钟收益，是最精确的日度波动率估计量。相比收盘价波动率，日度RV利用了全天的价格路径信息，精确度高一个数量级(Andersen et al. 2001)。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_hourly_1

**定义**：第一小时波动率因子，9:30-10:30已实现波动率截面排名（取负向=开盘高波排后）。

**公式（计算逻辑）**：

```python
rh = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_hourly_1",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rh)
```

**意义**：开盘第一小时的波动率是隔夜信息冲击的集中释放——第一小时波动率异常高意味着隔夜发生了重大事件，市场需要时间消化。持续的高开盘波动往往伴随更高的后续波动。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_hourly_4

**定义**：第四小时波动率因子，14:00-15:00已实现波动率截面排名（取负向=尾盘高波排后）。

**公式（计算逻辑）**：

```python
rh = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_hourly_4",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rh)
```

**意义**：尾盘最后一小时的波动率包含了T+1制度下的特殊博弈——尾盘波动率异常高往往与短线资金的日内了结和隔夜避险行为有关，对次日开盘有预测意义。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_hourly_dispersion

**定义**：小时波动率离散度因子，四个小时RV的标准差/均值截面排名（取负向=波动集中=不稳定排后）。

**公式（计算逻辑）**：

```python
rd = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_hourly_dispersion",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rd)
```

**意义**：四个小时的波动率离散度反映日内波动的'均匀性'——波动集中在某一个小时(高离散)意味着信息冲击集中在该时段，其余时间缺乏定价活动。低离散(均匀分布)代表全天持续的信息流和稳定的交易节奏。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_rolling_5d_std

**定义**：波动率波动因子，rv_5min的5日标准差截面排名（取负向=波动率不稳定排后）。

**公式（计算逻辑）**：

```python
vs = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_rolling_5d_std",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-vs)
```

**意义**：波动率本身的波动率(vol-of-vol)是二阶风险——两只股票当前波动率相同，但波动率更不稳定的那只未来风险更大。Vol-of-vol是预测波动率突变的重要先行指标。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_semi_down

**定义**：下行已实现半方差因子，仅负1分钟收益平方和开根截面排名（高下行波=风险排后）。

**公式（计算逻辑）**：

```python
rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_semi_down",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rv)
```

**意义**：下行半方差是投资者真正厌恶的'坏波动'——下行RV高的股票在下跌行情中跌幅更大，需要更高的风险溢价。下行RV对尾部风险的预测能力优于对称RV。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_semi_up

**定义**：上行已实现半方差因子，仅正1分钟收益平方和开根截面排名（上涨波动=正面信号排前）。

**公式（计算逻辑）**：

```python
rv = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_semi_up",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(rv)
```

**意义**：上行半方差(Barndorff-Nielsen et al. 2010)将波动率分解为'好波动'(上涨驱动)和'坏波动'(下跌驱动)——上涨波动率高的股票处于资金主动买入阶段，是积极的价格发现过程。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_skew_intraday

**定义**：日内已实现偏度因子，5分钟收益的截面偏度排名。正偏=上涨跳跃多，负偏=下跌跳跃多。

**公式（计算逻辑）**：

```python
"""Compute 5-min return skewness from per-stock files."""
import os
source_root = context.repo.paths.source_root
min_dir = source_root / "history_1min"
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress

files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
parts = []
total = len(files)

for i, fname in enumerate(files):
    code = fname.replace(".parquet", "").split(".")[0].zfill(6)
    if allowed and code not in allowed:
        if on_progress: on_progress("rv_skew", i + 1, total)
        continue

    filepath = min_dir / fname
    try:
        stock_df = pd.read_parquet(filepath)
    except Exception:
        if on_progress: on_progress("rv_skew", i + 1, total)
        continue

    if stock_df.empty:
        if on_progress: on_progress("rv_skew", i + 1, total)
        continue

    stock_df["trade_date"] = pd.to_datetime(stock_df["trade_time"]).dt.strftime("%Y%m%d")
    stock_df["close_5min_ago"] = stock_df.groupby("trade_date")["close"].shift(5)
    stock_df["ret_5min"] = stock_df["close"] / stock_df["close_5min_ago"].replace(0, np.nan) - 1.0
    skew = stock_df.groupby("trade_date")["ret_5min"].skew().dropna()
    if skew.empty:
        if on_progress: on_progress("rv_skew", i + 1, total)
        continue

    s = pd.DataFrame({"skew": skew.values},
                     index=pd.MultiIndex.from_arrays(
                         [skew.index, [code] * len(skew)], names=["Date", "Code"]
                     ))["skew"]
    parts.append(s)

    if on_progress: on_progress("rv_skew", i + 1, total)

if not parts:
    return cross_sectional_rank(pd.Series(dtype=float))

result = pd.concat(parts)
result.index = result.index.set_names(["Date", "Code"])
result = result.reorder_levels(["Date", "Code"]).sort_index()
result = result.groupby(list(result.index.names)).last()
if isinstance(result.index, pd.MultiIndex):
    result.index = result.index.set_names(["Date", "Code"])
return cross_sectional_rank(result)
```

**意义**：日内收益偏度反映日内价格路径的不对称性。正偏（多数上涨在下午发生）通常比负偏（恐慌性下跌）的后续表现更好。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_term_structure_slope

**定义**：波动率期限结构斜率因子，rv_5min/rv_60min-1截面排名（取负向=陡峭=短期波动高排后）。

**公式（计算逻辑）**：

```python
ts = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "rv_term_structure_slope",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-ts)
```

**意义**：短期与长期RV的比值是波动率期限结构的斜率——比值>1意味着短期波动率高于长期(波动率期限结构向下倾斜)，往往对应着短期事件冲击或市场压力。比值<1(正常状态)意味着波动率期限结构向上。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### rv_trend_5d

**定义**：波动率趋势因子，rv_5min的5日均值/20日均值-1截面排名（取负向=波动加速排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rt = _compute_intraday_factor(source_root, context.repo.allowed_codes, "rv_trend_5d",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-rt)
```

**意义**：波动率本身的趋势（波动率动量）包含了独立于波动率水平的信息。波动率处于上升趋势（高rv_trend_5d）意味着不确定性在加剧，即使当前波动率水平不高，也预示着风险上升。波动率上升期的股票未来收益分布更加左偏。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### smart_money_net_bias

**定义**：聪明钱净方向因子，异动分钟量加权涨跌方向截面排名（聪明钱净买入排前）。

**公式（计算逻辑）**：

```python
bias = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "smart_money_net_bias",
                                on_progress=context.repo.on_progress)
return cross_sectional_rank(bias)
```

**意义**：聪明钱成交的方向(量加权)是信息型资金的真实态度——异动分钟净买入=知情资金在低位吸筹；净卖出=知情资金在撤退。比价格动量更早的信号。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### smart_money_share

**定义**：聪明钱成交量占比因子，异动分钟成交量占全天比例截面排名（信息型交易活跃排前）。

**公式（计算逻辑）**：

```python
share = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "smart_money_share",
                                 on_progress=context.repo.on_progress)
return cross_sectional_rank(share)
```

**意义**：聪明钱(异动分钟)成交量占比衡量信息型交易的市场参与度——占比高=日内存在集中的信息驱动交易(可能有未公开利好/利空)；占比低=交易均匀、无信息冲击。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### smart_money_vwap_ratio

**定义**：聪明钱VWAP比因子，异动分钟(|收益|/√量前20%)的VWAP/全天VWAP截面排名（聪明钱成交价高于均价=抢筹排前）。

**公式（计算逻辑）**：

```python
sm = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "smart_money_vwap_ratio",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(sm)
```

**意义**：策略38聪明钱逻辑：|分钟收益|/√量 最高的20%分钟=信息型交易(小量大幅)。这些分钟成交的VWAP相对全天VWAP的比值衡量聪明钱的成交价位——聪明钱以高于全天均价的价格成交=主动买入抢筹(看多)；低于均价=主动砸盘。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### tail_ret_3d

**定义**：近3日尾盘收益累计因子：last30_ret 滚动3日和的截面排名（负向）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "tail_ret_3d"))
```

**意义**：连续3日尾盘收益的累计值捕捉尾盘资金的持续性——连续尾盘拉升=短线资金抱团(拥挤度上升),均值回归压力累积。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### tail_volume_share

**定义**：尾盘量能占比因子：尾盘30分钟成交量占全天比例截面排名（尾盘放量排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "close_30_vol_share"))
```

**意义**：尾盘量能占比高=资金在收盘前集中进场/对倒(尾盘定调行为活跃)——A股尾盘放量常与次日高开正相关;尾盘占比低=量能集中在早盘(消息驱动型)。与 rel_vol_last_hour(最后1小时)区分:30分钟窗口更聚焦。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### turnover_concentration_intraday

**定义**：换手率集中度因子，最大5分钟成交量/全日成交量截面排名（取负向=集中度过高排后）。

**公式（计算逻辑）**：

```python
tc = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "turnover_concentration",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-tc)
```

**意义**：换手率在单一5分钟区间过度集中往往是大单冲击或主力对倒——集中度>30%意味着三分之一的日成交在5分钟内完成，不利于中小投资者的交易执行。适度分散的成交是健康流动性的标志。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### up_minute_vol_share

**定义**：上涨分钟量占比因子：当日上涨分钟成交量占全天量比例截面排名（涨时整体放量排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "up_minute_vol_share"))
```

**意义**：当日上涨分钟的量占比——分钟级版 up_day_volume_ratio_20(日频按天判涨跌)。与 up_minutes_ratio(上涨分钟数占比)区分:本因子是量维度,资金重仓参与的上涨与零星反弹在量占比上分离。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### up_minutes_ratio

**定义**：上涨分钟占比因子，1分钟正收益分钟数/总分钟数截面排名（买盘主导排前）。

**公式（计算逻辑）**：

```python
um = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "up_minutes_ratio",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(um)
```

**意义**：上涨分钟的占比是日内买方力量的直接度量——>50%的分钟在上涨意味着买方在全天持续主导，而非仅在个别时段发力。高上涨占比的股票日内定价效率高、趋势可靠。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### vol_concentration

**定义**：成交量集中度因子，(开盘30分+收盘30分)成交量/全日成交量截面排名（取负向=过于集中排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
vc = _compute_intraday_factor(source_root, context.repo.allowed_codes, "vol_concentration",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-vc)
```

**意义**：成交量过度集中在开盘和收盘时段（A股经典的U型分布极端化）往往意味着信息不对称严重或主力刻意为之。适中的成交量分布代表自然的交易节奏。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### vol_of_vol_intraday

**定义**：波动率的波动率因子，|ret_5min|的std/mean截面排名（取负向=波动不稳定排后）。波动率自身的变异系数。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
vv = _compute_intraday_factor(source_root, context.repo.allowed_codes, "vol_of_vol",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-vv)
```

**意义**：波动率的波动率（vol-of-vol）衡量日内波动率的聚集和分散程度(Corsi et al. 2008)。高vol-of-vol意味着波动率在日内剧烈变化（如早盘高波、尾盘低波），价格过程远非平稳。波动率的不确定性本身是第二阶风险，需要额外的风险溢价。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### vol_stability

**定义**：成交量稳定性因子，5分钟成交量std/均值截面排名（取负向=不稳定排后）。成交量日内分布的规律性。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
vs = _compute_intraday_factor(source_root, context.repo.allowed_codes, "vol_stability",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-vs)
```

**意义**：成交量日内分布的稳定性反映了交易结构的健康程度。稳定的成交量模式（低vol_stability）意味着流动性供给可预测，交易成本可控。成交量忽大忽小往往与信息不对称和主力操纵行为相关。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### volume_peak_time

**定义**：成交量峰值时间因子，日内最大5分钟成交量所在分钟截面排名（取负向=尾盘放量=异常排后）。

**公式（计算逻辑）**：

```python
vpt = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_peak_time",
                                on_progress=context.repo.on_progress)
return cross_sectional_rank(-vpt)
```

**意义**：成交量峰值出现的时间包含信息——早盘峰值(正常)+午盘峰值(可接受)+尾盘峰值(警惕)。尾盘突然放量往往是主力做收盘价或短线资金集中进出的标志，次日走势不确定。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### volume_profile_kurt

**定义**：成交量分布峰度因子，5分钟成交量日内分布的峰度截面排名（取负向=高峰度=不均排后）。

**公式（计算逻辑）**：

```python
vk = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_profile_kurt",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-vk)
```

**意义**：成交量峰度高意味着交易集中在少数几个5分钟区间——极端的成交量集中往往对应着大单冲击或主力行为。均匀分布的成交量代表自然的、健康的交易节奏。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### volume_profile_skew

**定义**：成交量分布偏度因子，5分钟成交量日内分布的偏度截面排名（取负向=偏度极端排后）。

**公式（计算逻辑）**：

```python
vs = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_profile_skew",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-vs.abs())
```

**意义**：成交量分布的偏度反映成交活跃度的日内倾斜——正偏(集中在早盘)意味着信息消化高效，负偏(集中在尾盘)可能是被动交易或操纵行为。极端的成交量偏度是不健康的交易结构。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### volume_rv_ratio

**定义**：成交量波动比因子，log(成交量)/rv_5min截面排名。单位波动对应的交易活跃度。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
vr = _compute_intraday_factor(source_root, context.repo.allowed_codes, "volume_rv_ratio",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(vr)
```

**意义**：每单位已实现波动率对应的成交量衡量了'信息的定价效率'。高volume_rv_ratio意味着大量交易产生了相对较小的价格波动，说明市场吸收信息的能力强、流动性充足。低比值可能暗示流动性脆弱——少量交易即引发较大波动。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### volume_u_shape_score

**定义**：U型分布评分因子，成交量日内分布与U型模板的相关性截面排名（取负向=极端U型=操纵风险排后）。

**公式（计算逻辑）**：

```python
us = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_u_shape_score",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(-us.abs())
```

**意义**：A股成交量呈U型分布(早盘和尾盘量大、午间量小)是正常现象——但极端的U型(过于集中首尾)往往与操纵行为有关。适度的U型是健康的，极端的U型可能暗示主力在特定时段操作。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### volume_weighted_ret

**定义**：成交量加权收益因子，Σ(ret_i×vol_i)/Σvol_i截面排名（量价配合=真实涨跌排前）。

**公式（计算逻辑）**：

```python
vw = _compute_intraday_factor(context.repo.paths.source_root, context.repo.allowed_codes, "volume_weighted_ret",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(vw)
```

**意义**：成交量加权收益比简单收益更能反映'真实'的价格变动——在成交量大的价位上的价格变动比成交量小的价位上的变动更有信息含量。VW收益排除了无量空涨/空跌的噪音。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### vp_consistency_20

**定义**：量价一致性20日均值因子：四象限一致性得分的20日均值截面排名（持续量价健康排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "vp_consistency_20"))
```

**意义**：单日一致性噪声大,20日均值度量「量价配合的持续性」——持续涨有量跌无量=资金长期驻留(吸筹/控盘特征);持续背离=出货周期。与 vp_consistency_score区分:本因子过滤日内噪声,信号更稳,适合与动量/筹码因子做 Class 5 耦合。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vp_consistency_score

**定义**：量价一致性得分因子：四象限一致性(放量涨+缩量跌−缩量涨−放量跌)/全天量截面排名（量价健康排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "vp_consistency_score"))
```

**意义**：四象限联合得分:涨有量+跌无量(量价健康)为正,涨无量+跌有量(背离/出货)为负。该得分把「诱多(缩量涨)」「诱空(放量跌)」两类陷阱统一到一条量纲一致的轴上,是量价关系的综合度量,与各象限占比互补。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vp_expand_down_am_share

**定义**：早盘放量下跌占比因子：早盘(09:30-11:30)放量下跌量占全天放量下跌量比例截面排名（早盘恐慌释放排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "vp_expand_down_am_share"))
```

**意义**：放量下跌发生在早盘=恐慌在开盘集中释放(洗筹,午后修复概率大,A股常见「早盘杀跌午后V」);发生在尾盘=出货延续(次日低开风险)。方向 pos:早盘集中排前。与 vp_expand_down_share(全天放量下跌总量)区分:本因子度量恐慌的时段归属,是诱空识别的重要维度。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vp_expand_down_share

**定义**：放量下跌量占比因子：日内放量下跌分钟量占全天量比例截面排名（负向，恐慌抛售排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "vp_expand_down_share"))
```

**意义**：放量下跌=恐慌抛售/主力出货(跌有量,承接不足);占比高=当日筹码在下跌中被大量换手,抛压沉重。与 panic_selling_ratio_60(日频放量×下跌联合)区分:本因子为分钟粒度,放量基准为同时段均量,能捕捉日内局部的放量砸盘。方向 neg:放量下跌占比低(下跌无量)排前。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vp_expand_price_pos

**定义**：放量价格位置因子：放量分钟的量加权日内位置均值截面排名（负向，低位放量=吸筹排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "vp_expand_price_pos"))
```

**意义**：放量发生在日内什么价位:低位(接近当日低点)放量=承接吸筹(主力低位吃货);高位放量=追高/出货(拉高出货特征)。方向 neg:高位放量排后。与 close_position(收盘单点位)区分:本因子用全部放量分钟的分布,是「吸筹 vs 出货」的日内位置刻画。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vp_expand_ret_gap

**定义**：放缩量收益差因子：放量分钟均收益−缩量分钟均收益截面排名（放量推动价格排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "vp_expand_ret_gap"))
```

**意义**：放量分钟的涨幅相对缩量分钟的涨幅之差——度量「放量是否推得动价格」。放量上涨而价格不动(差值≈0/负)=对倒/出货(大单对敲吸引跟风);放量即涨=增量资金真实进场。是识别诱多(放量不涨)的关键信号,与 volume_price_divergence_score(日频价量动量差)区分:本因子为日内分钟口径。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vp_expand_up_share

**定义**：放量上涨量占比因子：日内放量(超20日同时段均量)上涨分钟量占全天量比例截面排名（涨有量排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "vp_expand_up_share"))
```

**意义**：上涨分钟中放量的量占比衡量「上涨的量能质量」——放量上涨=资金真实做多(增量承接,趋势可持续);缩量上涨=无量反弹(诱多嫌疑,见 vp_shrink_up_share)。与日频 up_day_volume_ratio_20 区分:本因子为分钟粒度,同一天内即可区分「涨时放量/跌时缩量」与「跌时放量/涨时缩量」的微观差异。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vp_shrink_down_share

**定义**：缩量下跌量占比因子：日内缩量下跌分钟量占全天量比例截面排名（跌无量排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "vp_shrink_down_share"))
```

**意义**：下跌但缩量=无量阴跌/洗盘回调——卖盘枯竭、抛压轻(浮筹锁定),常是主力洗盘而非出逃(出逃必然放量)。占比高=当日下跌质量好,方向 pos。与 vp_expand_down_share 互补:跌时量能结构刻画恐慌 vs 洗盘。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vp_shrink_up_share

**定义**：缩量上涨量占比因子：日内缩量上涨分钟量占全天量比例截面排名（负向，无量反弹排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "vp_shrink_up_share"))
```

**意义**：上涨但缩量(量低于同时段均量)=无量反弹/对倒拉升——价格上涨缺乏成交确认,典型诱多形态(拉高无人跟风,后续易回落)。占比高=当日上涨质量差,方向 neg。与 vp_expand_up_share 互补,二者合计=上涨分钟的量占比。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vwap_am_pm_gap_factor

**定义**：上午/下午VWAP差因子：上午VWAP相对下午VWAP偏离截面排名（上午价格水平高排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "vwap_am_pm_gap"))
```

**意义**：上午VWAP高于下午=价格重心逐段下移(早盘高买盘午后撤退);上午低于下午=全天重心抬升(早盘低吸午后拉升)。VWAP是时段内真实成交均价,对时段强弱比收盘价比较更抗尾盘扰动。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/intraday_extra.py`

##### vwap_close_ratio

**定义**：收盘价/分钟VWAP偏离因子：close/(amount/vol)-1 截面排名（负向）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "vwap_dev"))
```

**意义**：与既有 vwap_deviation 同秩（V8 原脚本 vol 单位×100 仅为常数缩放，截面排名不变），V8 池内方向为负：收盘显著高于分钟 VWAP=尾盘透支,低于 VWAP=尾盘修复空间。V8 筛查 meanIC -0.034/ICIR -0.20。

**依赖数据**：`history_1min` ｜ **Class**：3 ｜ **源码**：`factors/fac_cand_min.py`

##### vwap_deviation

**定义**：VWAP偏离因子，(收盘-VWAP)/VWAP截面排名。收盘价相对日均价的位置。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
import os
min_dir = source_root / "history_1min"
allowed = context.repo.allowed_codes
on_progress = context.repo.on_progress

files = sorted([f for f in os.listdir(min_dir) if f.endswith(".parquet")])
parts = []
total = len(files)

for i, fname in enumerate(files):
    code = fname.replace(".parquet", "").split(".")[0].zfill(6)
    if allowed and code not in allowed:
        if on_progress: on_progress("vwap", i + 1, total)
        continue

    filepath = min_dir / fname
    try:
        stock_df = pd.read_parquet(filepath)
    except Exception:
        if on_progress: on_progress("vwap", i + 1, total)
        continue

    if stock_df.empty:
        if on_progress: on_progress("vwap", i + 1, total)
        continue

    stock_df["trade_date"] = pd.to_datetime(stock_df["trade_time"]).dt.strftime("%Y%m%d")
    grouped = stock_df.groupby("trade_date")
    daily_vol = grouped["vol"].sum()
    daily_amount = grouped["amount"].sum()
    daily_close = grouped["close"].last()
    vwap = daily_amount / daily_vol.replace(0, np.nan)
    dev = (daily_close - vwap) / vwap.replace(0, np.nan)
    dev = dev.dropna()
    if dev.empty:
        if on_progress: on_progress("vwap", i + 1, total)
        continue

    s = pd.DataFrame({"dev": dev.values},
                     index=pd.MultiIndex.from_arrays(
                         [dev.index, [code] * len(dev)], names=["Date", "Code"]
                     ))["dev"]
    parts.append(s)
    if on_progress: on_progress("vwap", i + 1, total)

if not parts:
    return cross_sectional_rank(pd.Series(dtype=float))

result = pd.concat(parts)
result.index = result.index.set_names(["Date", "Code"])
result = result.reorder_levels(["Date", "Code"]).sort_index()
result = result.groupby(list(result.index.names)).last()
if isinstance(result.index, pd.MultiIndex):
    result.index = result.index.set_names(["Date", "Code"])
return cross_sectional_rank(result)
```

**意义**：收盘价高于VWAP说明尾盘有资金主动推升，买方主导日内的均价水平。持续的VWAP正偏离（收盘>VWAP）是机构主动建仓的信号，尤其在连续多日正偏离后趋势延续性强。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

##### vwap_momentum_5d

**定义**：VWAP动量因子，(5日VWAP均值/20日VWAP均值-1)截面排名。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
vw = _compute_intraday_factor(source_root, context.repo.allowed_codes, "vwap_mom_5d",
                               on_progress=context.repo.on_progress)
return cross_sectional_rank(vw)
```

**意义**：VWAP比收盘价更能代表'真实成交价格'，VWAP动量排除了尾盘操纵的影响，比收盘价动量更纯净地反映资金的平均建仓成本变化方向。

**依赖数据**：`history_1min`、`daily.parquet` ｜ **Class**：3 ｜ **源码**：`factors/intraday.py`

---

### <a name="class-4"></a>Class 4 — indicator_1min 分钟指标类（87 个）

从 `indicator_1min/` 逐股票目录加载分钟级技术指标，构建日内指标因子。

#### <a name="cat-intraday-c4"></a>类别 intraday — 日内 / 微观结构（87 个）

##### am_macd_trend

**定义**：早盘MACD趋势因子（上午MACD与时间的相关性截面排名，正相关=早盘动能积聚排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
at = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "am_macd_trend",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(at)
```

**意义**：早盘(9:30-11:30)MACD的日内趋势反映了上午交易时段的多空力量演变。早盘MACD持续上升=多头在上午逐步取得优势；早盘MACD持续下降=空头在上午逐步施压。A股有'上午定调'的说法，早盘的MACD趋势往往决定了全天的基调。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### am_pm_macd_ratio

**定义**：早午盘MACD比率因子（上午MACD均值/下午MACD均值截面排名，比率>1=早盘强于午盘=动能衰减排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ap = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "am_pm_macd_ratio",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ap)
```

**意义**：上午MACD与下午MACD的比率反映了动能的日内分布。比率>1=上午MACD强于下午，动能可能在日内衰减；比率<1=下午MACD强于上午，动能可能在下午加速。'上午冲高下午回落'是A股的经典模式，该因子量化了这个现象。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### am_pm_rsi_ratio

**定义**：早午盘RSI比率因子（上午RSI均值/下午RSI均值截面排名，比率>1=早盘动量强于午盘排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ap = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "am_pm_rsi_ratio",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ap)
```

**意义**：上午RSI与下午RSI的比率反映了动量的日内迁移。比率>1=早盘RSI更高，动量在上午集中释放，下午可能衰减；比率<1=午盘RSI更高，动量在下午增强。'上午拉高出货'的模式可以通过高比率+低绝对RSI来识别。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### am_rsi_trend

**定义**：早盘RSI趋势因子（上午RSI与时间的相关性截面排名，正相关=早盘动量积聚排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
at = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "am_rsi_trend",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(at)
```

**意义**：早盘RSI的日内趋势是上午动量方向的独立度量（与MACD正交）。RSI持续走高=买盘在整个上午持续占优；RSI持续走低=卖盘在上午主导。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### boll_band_deviation

**定义**：布林带偏离因子（(ma5-boll_mid)/(upper-lower)截面排名，偏离大=脱离均衡排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
bd = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "boll_band_deviation",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-bd.abs())
```

**意义**：价格(ma5)与布林带中心(中轨)的距离，以带宽为单位标准化。该因子与boll_position类似，但使用中轨作为基准（而非相对上下轨的位置）。偏离越大=价格越远离均衡位置，均值回归压力越大。取绝对值排名，双向极端偏离均排后。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### boll_mid_slope

**定义**：布林带中轨斜率因子（(boll_mid_close-boll_mid_open)/|boll_mid_open|截面排名，中轨上移=趋势向上排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ms = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "boll_mid_slope",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ms)
```

**意义**：布林带中轨(20期均线)的日内斜率反映了短期趋势的方向和强度。中轨是20期均线——其斜率为正意味着均线正在上移，趋势向上；斜率为负意味着均线下移，趋势向下。中轨斜率是布林带体系中的'方向性'信息，与带宽(波动率)正交。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### boll_position

**定义**：布林带位置因子（(ma5_close-boll_mid)/(boll_upper-boll_lower)截面排名，上轨附近=强势排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
bp = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "boll_position",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(bp)
```

**意义**：收盘价(ma5 proxy)在布林带中的相对位置是一个标准化的动量信号。接近上轨(+0.5~+1.0)=强势，往往有继续向上突破的动能；接近下轨(-1.0~-0.5)=弱势，但也是潜在的反弹位置。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### boll_squeeze

**定义**：布林带挤压因子（1/布林带宽截面排名，带宽极窄=挤压程度高=变盘迫近排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
sq = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "boll_squeeze",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-sq)
```

**意义**：布林带挤压(带宽的倒数)是波动率压缩到极致时的信号。挤压程度高=波动率被极度压缩，价格在极窄的区间内运行——这是经典的技术性变盘前兆(Bollinger Squeeze)。挤压后的突破方向由其他因素决定，但挤压本身提供了'即将行动'的时间窗口。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### boll_width_20

**定义**：布林带宽度因子（(upper-lower)/mid截面排名，窄带=变盘前兆排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
bw = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "boll_width",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-bw)
```

**意义**：布林带宽度是波动率的直接度量。带宽窄=波动率低='暴风雨前的宁静'，往往预示着即将出现大行情。带宽宽=高波动环境，行情已经在进行中。该因子捕捉了'波动率收缩-扩张'周期的收缩端。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### boll_width_5d_change

**定义**：布林带宽5日变化因子：分钟布林带宽的5日变化截面排名（带宽扩张排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "boll_width_5d_chg"))
```

**意义**：布林带宽5日扩张=波动率周期启动(趋势行情开启的前兆);持续收窄=横盘蓄势。与 boll_width_change(日内变化)、boll_squeeze(绝对宽度)互补,捕捉带宽的中期方向——A股波动率具有明显状态切换特征。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### boll_width_change

**定义**：布林带宽变化因子（收盘带宽-开盘带宽截面排名，带宽扩张=波动率上升排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
wc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "boll_width_change",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(wc)
```

**意义**：布林带宽度在一天内的变化方向反映了波动率的日内趋势。带宽扩张=波动率在上升，往往伴随趋势启动或加速；带宽收缩=波动率在下降，往往伴随趋势衰竭或盘整。带宽变化方向比带宽的绝对水平更有预测价值——扩张中的带宽意味着'波动率正在从低点回升'。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### indicator_consensus

**定义**：指标共识度因子（6个指标中同向占比截面排名，高共识=多指标共振排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ic = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "indicator_consensus",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ic)
```

**意义**：MACD、KDJ(K-D)、RSI(>50)、布林带位置(>0)、MA排列(≥3)、MAVol比率(>1)六个独立技术指标中看多方向的比例。6/6=所有指标一致看多，趋势确认度最高；0/6=所有指标一致看空；3/6=指标之间互相矛盾，方向不确定。多指标共振是技术分析中的'圣杯'——信号可靠性随共识度指数级提升。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### indicator_dispersion

**定义**：指标离散度因子（5个标准化指标信号的std截面排名，离散度高=指标矛盾排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
id_val = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "indicator_dispersion",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-id_val)
```

**意义**：五个标准化指标信号(MACD/KDJ/RSI/布林带/MA排列)的截面标准差。标准差大=不同指标给出的信号相互矛盾——有些指标看多，有些看空，综合方向不明确；标准差小=所有指标给出的信号一致——方向明确。该因子度量了'技术分析的噪音水平'——噪音越低越好。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### j_day_position

**定义**：KDJ J值日内区间位置因子：(J收盘−J最低)/(J最高−J最低)截面排名（J收盘靠上沿排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "j_day_position"))
```

**意义**：J 值对价格变动最敏感,其收盘在当日区间的位置是日内短线动能方向的终态判断——靠上沿=尾盘动能向上(次日惯性概率大);靠下沿=尾盘走弱。与 j_range(波动宽度)、kdj_j_value_close(绝对水平)互补。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### kdj_boll_combo

**定义**：KDJ-布林带组合因子（(j-50)/50+boll_position截面排名，双重极端=强烈信号排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
kc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "kdj_boll_combo",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(kc)
```

**意义**：KDJ的J值(标准化的-1到+1)与布林带位置的组合捕捉了超买超卖+价格位置的共振。J极端超买+价格在上轨=双重超买，回调概率极高；J极端超卖+价格在下轨=双重超卖，反弹概率极高。两个指标从不同角度（动量+波动率）给出了相同的极端判断，信号可靠性高。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_bull_frac

**定义**：KDJ多头时间占比因子（K>D的分钟占比截面排名，高占比=日内持续多头排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
bf = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "kdj_bull_frac",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(bf)
```

**意义**：K线在D线之上的时间占比是KDJ多头排列的持续性指标。接近100%=全天K都在D之上，多头控制全局；接近0%=全天K都在D之下，空头控制全局；接近50%=K-D反复交叉，方向不明确。该因子提供了一个连续的多头强度度量，比离散的'金叉次数'更平滑。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_bull_frac_5d_change

**定义**：KDJ多头占比5日变化因子：日内K>D占比的5日变化截面排名（多头增强排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "kdj_bull_frac_5d_chg"))
```

**意义**：K>D 的日内占比衡量全天金叉状态的时间份额——其5日变化捕捉多头结构的增强/衰减:占比持续上升=KDJ 系统的多头状态正在建立;下降=金叉质量恶化。与 kdj_bull_frac(水平)互补,捕捉方向。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### kdj_cross_net

**定义**：KDJ金叉净数量因子（金叉次数-死叉次数截面排名，净多头=强势排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
cn = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "kdj_cross_net",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(cn)
```

**意义**：金叉与死叉的净差值综合了多头和空头信号的博弈结果。正数且大=日内多头信号远多于空头，强势确认；负数且小=空头信号主导，弱势。净金叉数量比单独的金叉次数更有信息量——它同时惩罚了空头信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_cross_signal

**定义**：KDJ金叉信号因子（日内K上穿D的次数截面排名，金叉次数多=强势排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
kc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "kdj_cross_count",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(kc)
```

**意义**：K线上穿D线（金叉）是KDJ的经典做多信号。在日内1分钟频率上，金叉可以发生多次。金叉次数多意味着K线多次试图并成功突破D线——多头反复确认。单次金叉可能是噪音，多次金叉信号可靠性更高。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_d_stability

**定义**：KDJ-D线稳定性因子（std(D)/mean(D)截面排名，D线不稳定=信号质量差排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ds = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "d_stability",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-ds)
```

**意义**：D线(慢线)的变异系数反映了KDJ慢线的稳定性。D线是K线的3日平滑，理论上应该比较稳定。如果D线在一天内也有较大波动（高CV），说明即使是慢线也在被剧烈拉扯，这是一个强烈的趋势不确定信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_dead_cross_count

**定义**：KDJ死叉次数因子：日内K下穿D的次数截面排名（负向，死叉频繁排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "kdj_dead_cross_count"))
```

**意义**：日内 K 下穿 D 的次数衡量空头信号的反复强度——死叉频繁=多方反击乏力、空头持续压制;死叉稀少=趋势结构稳定。与 kdj_cross_signal(净金叉)互补:本因子单看空头侧。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### kdj_j_5d_acceleration

**定义**：KDJ J值5日加速度因子：J收盘的5日变化截面排名（J值加速排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "j_close_5d_gap"))
```

**意义**：J 值5日变化=KDJ 动能的加速度——J 持续抬升=短线动能加速(主升段特征);J 回落=动能衰竭。与 j_day_position(日内位置)互补:本因子为跨日动量维度,捕捉KDJ 系统的趋势性变化。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### kdj_j_range

**定义**：KDJ-J值振幅因子（j_max-j_min截面排名，J值波动大=多空分歧大排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
jr = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "j_range",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-jr)
```

**意义**：J值在一天内的波动范围反映了多空分歧的激烈程度。J值振幅大=日内经历从超买到超卖的剧烈摆动，行情极不稳定；J值振幅小=日内多空力量相对均衡或单边温和运行。高振幅往往预示着后续的均值回归。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_j_reversal_risk

**定义**：KDJ-J值反转风险因子（|j_close-50|截面排名，极端J值=高反转概率排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
jr = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "j_reversal_risk",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-jr)
```

**意义**：J值偏离50的绝对值度量了J值的极端程度。J值越极端（远高100或远低于0），反转概率越高。该因子独立于J的方向，纯粹度量了'距离中性的偏离'——无论是极度超买还是极度超卖，都意味着短期方向不可持续。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_j_value_close

**定义**：KDJ-J值因子（日内收盘J值截面排名，高J=超买排后，低J=超卖排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
jv = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "j_close",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-jv)
```

**意义**：J值(3K-2D)是KDJ中最敏感的线，率先反应动量的极端状态。J>100：超买区域，短期内大概率回调；J<0：超卖区域，短期内大概率反弹。日末收盘J值包含了全天的多空博弈结果，比盘中J值更稳定。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_j_volatility

**定义**：KDJ-J值波动因子（std(J)截面排名，J值波动率高=不稳定排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
jv = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "j_volatility",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-jv)
```

**意义**：J值的日内标准差度量了J值的不稳定性。高波动=J值频繁上下跳动，技术信号不可靠；低波动=J值变化平滑，技术信号可信度高。该因子评估了KDJ信号的质量而非方向。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_k_d_distance

**定义**：KDJ-K-D距离因子（(K-D)/|D|截面排名，正=多头动能排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
kd = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "k_d_distance",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(kd)
```

**意义**：K线与D线的距离(归一化)度量了KDJ的多空强度。(K-D)/|D|>0：K在D之上，多头排列；数值越大，多头动能越强。(K-D)/|D|<0：K在D之下，空头排列；数值越小，空头动能越强。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_overbought_frac

**定义**：KDJ超买时间占比因子（K>80分钟占比截面排名，高占比=强势超买排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ob = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "kdj_overbought_frac",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-ob)
```

**意义**：K值在超买区域(>80)的停留时间占比反映了买盘的持续性。高占比=全天大部分时间处于超买状态，买盘极其强劲但也透支了购买力；低占比=未能进入超买或仅在超买区域短暂停留。持续的超买状态既可能是强势上涨的确认，也可能意味着顶部即将到来。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### kdj_oversold_frac

**定义**：KDJ超卖时间占比因子（K<20分钟占比截面排名，高占比=深度超卖排前=抄底信号）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
os = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "kdj_oversold_frac",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(os)
```

**意义**：K值在超卖区域(<20)的停留时间占比反映了卖盘的持续性。高占比=全天大部分时间处于超卖状态，恐慌性抛售充分释放；低占比=未进入超卖或仅在超卖短暂停留。持续超卖后的反弹概率显著升高，是逆向买入信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma10_slope

**定义**：MA10斜率因子：分钟MA10收盘相对开盘的斜率截面排名（MA10上倾排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "ma10_slope"))
```

**意义**：MA10 斜率是短周期趋势的方向计——上倾=短线均线系统多头结构;与已有ma5_slope/ma20_slope 构成完整斜率谱系,MA10 的斜率差异可用于判断短线趋势的传导层级(MA5 上穿 MA10 的加速阶段)。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### ma20_slope

**定义**：MA20斜率因子（(ma20_close-ma20_open)/|ma20_open|截面排名，中期均线上移排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ms = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma20_slope",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ms)
```

**意义**：MA20(布林带中轨)的日内斜率反映了中期趋势的日内变化。MA20日内上升=中期均线正在上移，趋势健康向上；MA20日内下降=中期均线正在下移，趋势走弱。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma30_slope

**定义**：MA30斜率因子：分钟MA30收盘相对开盘的斜率截面排名（MA30上倾排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "ma30_slope"))
```

**意义**：MA30 斜率捕捉6日尺度的均线方向——MA30 上倾=中期均线结构转多(趋势级别提升);下倾=中期结构走弱。与 ma20_slope 的背离可识别均线系统的内部换档(20日线斜率与30日线斜率的交叉区)。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### ma5_ma10_gap

**定义**：MA5/MA10乖离因子：分钟MA5相对MA10的偏离截面排名（短均线在上排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "ma5_ma10_gap"))
```

**意义**：MA5 在 MA10 上方=短周期动能强于短中期(短线攻击结构);MA5 跌破 MA10=短线动能衰减(均线死叉前兆)。是分钟级均线系统的内部结构度量,与日频 ma_distance_5_10 区分(分钟口径)。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### ma5_slope

**定义**：MA5斜率因子（(ma5_close-ma5_open)/|ma5_open|截面排名，均线上移=趋势向上排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ms = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma5_slope",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ms)
```

**意义**：MA5在一天内的变化率反映了最短周期均线的日内趋势。MA5日内上升=价格在5分钟尺度上持续走高，短期动能向上；MA5日内下降=价格在5分钟尺度上持续走低，短期动能向下。MA5斜率是微观趋势的最直接度量。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma_alignment_score

**定义**：MA均线排列因子（ma5>ma10>ma20>ma30>ma60的满足数量截面排名，多头排列排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ma = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma_alignment",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ma)
```

**意义**：多周期均线的排列顺序是多时间框架趋势一致性的直接度量。5条均线全部多头排列(score=4)=最强多头趋势，趋势的持续性最高；全部空头排列(score=0)=最强空头趋势；均线交织(score=2左右)=方向不明确，震荡行情。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma_bull_bear_ratio

**定义**：MA多空比率因子（所有MA对中多头排列对数/总对数截面排名，高比率=全面多头排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
br = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma_bull_bear_ratio",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(br)
```

**意义**：在6个MA对组合(5-10,5-20,5-60,10-20,10-60,20-60)中，短MA > 长MA的占比。6/6=全面多头，0/6=全面空头。比ma_alignment_score(0-4)更细粒度，覆盖了更多的MA对组合。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma_convergence

**定义**：MA均线收敛因子（|ma5-ma60|/|ma60|截面排名，短长均线距离近=收敛排前=变盘信号）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
mc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma_convergence",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-mc)
```

**意义**：最短周期均线(MA5)与最长周期均线(MA60)的距离度量了短期和中期趋势的离散程度。距离小=短中期均线收敛，多时间框架交易者趋于一致，即将变盘；距离大=均线发散，趋势明确。该因子与ma_dispersion互补——ma_dispersion衡量5条均线的离散度，而ma_convergence关注最短与最长均线的距离。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma_cross_count

**定义**：MA均线交叉因子（ma5上穿/下穿ma20次数截面排名，交叉多=方向切换频繁排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
cc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma_cross_count",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-cc)
```

**意义**：MA5与MA20在一天内的交叉次数反映了短期-中期趋势的一致性。零次交叉=MA5全天保持在MA20的同一侧，短中期方向一致；多次交叉=MA5频繁穿越MA20，短中期方向反复切换。高频交叉意味着技术信号极度混乱，方向不可预测。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma_curvature

**定义**：MA曲率因子（ma5_slope-ma20_slope截面排名，正曲率=短周期加速快于中期排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
mc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma_curvature",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(mc)
```

**意义**：MA5斜率与MA20斜率的差值度量了均线的'曲率'——即加速度。正曲率=短周期均线比中期均线上移得更快，趋势正在加速；负曲率=短周期均线的上升速度不及中期，或在更快地下跌——趋势减速。曲率是趋势的二阶导数，比斜率更早地发出趋势变化信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma_dispersion

**定义**：MA离散度因子（各均线间距的变异系数截面排名，低离散=均线收敛=变盘前兆排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
md = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma_dispersion",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-md)
```

**意义**：五条均线(MA5/10/20/30/60)的离散度反映了不同时间框架交易者的共识程度。离散度低=均线收敛在一起=不同时间框架的交易者成本趋于一致='暴风雨前的宁静'，即将出现方向性突破。离散度高=均线发散=趋势已经在进行中，顺势而为。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### ma_rsi_divergence

**定义**：MA-RSI背离因子（ma5与rsi的日内相关性截面排名，负相关=量价背离排前=反转信号）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rd = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "ma_rsi_divergence",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-rd)
```

**意义**：MA5(价格代理)与RSI在一天内的相关性度量了价格和动量的配合程度。正相关=价格上涨+RSI上升（健康），或价格下跌+RSI下降（健康）；负相关=价格上涨但RSI在下降（顶背离），或价格下跌但RSI在上升（底背离）。负相关=经典的技术背离，是强烈的反转预警。该因子将日线级别的'RSI背离'概念应用到了1分钟级别。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_acceleration

**定义**：MACD柱加速度因子（(macd_close-macd_open)/|macd_open|截面排名，加速扩张排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
acc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_acceleration",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(acc)
```

**意义**：MACD柱(即dif-dea)在一天内的变化率衡量了趋势是否在加速。MACD柱正在扩大=动能加速度为正，趋势还有延续空间；MACD柱正在缩小=动能在衰竭，趋势可能即将反转。这是'二阶导数'信号，比单纯的MACD方向更灵敏。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_bar_energy

**定义**：MACD柱能量因子：日内|macd|均值截面排名（负向，柱体活跃排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "macd_bar_energy"))
```

**意义**：MACD 柱的绝对均值衡量多空动能释放的强度——柱体能量大=趋势推动力强但波动剧烈(方向切换频繁);能量小=动能枯竭(变盘前兆)。与macd_daily_range(柱极差)互补:能量看均值、极差看极端。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### macd_bar_sign_change

**定义**：MACD柱翻红翻绿因子（日内macd柱正负切换次数截面排名，频繁切换=趋势不稳排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
sc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_bar_sign_change",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-sc)
```

**意义**：MACD柱在一天内的正负切换次数反映了多空力量的拉锯程度。切换次数少=全天单边趋势明确；切换次数多=多空反复拉锯，方向不明。频繁的柱状图正负切换意味着量化/程序化交易的来回博弈，趋势不可靠。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_daily_consistency

**定义**：MACD日内一致性因子（日内macd柱>0的分钟占比截面排名，高一致性=趋势明确排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
cons = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_consistency",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(cons)
```

**意义**：MACD柱方向在全天各分钟的占比反映了日内趋势的一致性。一致性接近100%或0%意味着全天单边走势，趋势十分明确；一致性接近50%意味着macd柱频繁翻红翻绿，日内多空拉锯。高一致性的交易日提供了更可靠的趋势信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_daily_range

**定义**：MACD日内振幅因子（macd_max-macd_min截面排名，振幅大=波动剧烈排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rng = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_range",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-rng)
```

**意义**：MACD柱在一天内的最大-最小值差反映了MACD的振幅，即多空力量的波动范围。振幅大=多空博弈激烈，趋势不稳定；振幅小=MACD柱变化平缓，趋势温和但稳定。MACD振幅与价格波动率高度相关，但提供了纯技术指标层面的正交信息。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_dif_slope

**定义**：MACD-DIF日内斜率因子（(dif_close-dif_open)/|dif_open|截面排名，DIF上升=强势排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
slope = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_dif_slope",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(slope)
```

**意义**：DIF日内斜率反映了MACD快线在当天的运动方向和速度。DIF日内持续上升意味着短期动能持续增强，属于强势信号；DIF日内持续下降意味着动能衰竭，即使是macd柱为正也在减速。DIF斜率比收盘DIF绝对值更早地发出趋势变化信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_extreme_ratio

**定义**：MACD极端值占比因子（|macd|>2σ的分钟占比截面排名，高极端值=异常行情排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
er = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_extreme_ratio",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-er)
```

**意义**：MACD柱绝对值异常放大（超过2倍标准差）的分钟占比反映了行情的极端程度。高比例=全天MACD柱频繁出现极端值，可能是资金博弈激烈或恐慌；低比例=MACD柱在正常范围内波动，行情平稳。极端占比异常升高往往是变盘或行情衰竭的前兆。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_kdj_alignment

**定义**：MACD-KDJ一致性因子（sign(macd)*sign(K-D)截面排名，双指标同向=趋势确认排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ma = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_kdj_alignment",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ma)
```

**意义**：MACD方向和KDJ方向的一致性度量了两个最常用的趋势指标是否互相确认。+1=MACD多头+KDJ多头（双重确认，信号最强）；-1=MACD空头+KDJ空头（双重确认下跌）；0=两个指标方向矛盾（需要更多信息判断）。双指标同向确认的信号远优于单一指标。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_price_divergence

**定义**：MACD价格背离因子（MA5趋势与macd柱趋势的日内相关性截面排名，负相关=背离信号排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
corr = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_price_corr",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-corr)
```

**意义**：日内价格运动方向与MACD柱运动方向的相关系数度量了量价配合程度。正相关=价格上涨+MACD柱增大（健康上涨），或价格下跌+MACD柱减小（健康下跌）；负相关=价格上涨但MACD柱在减小（顶背离），或价格下跌但MACD柱在增大（底背离）。负相关=MACD背离，是经典的反转预警信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_rsi_combo

**定义**：MACD-RSI组合因子（sign(macd)*(rsi-50)截面排名，双动量指标同向排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
mc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_rsi_combo",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(mc)
```

**意义**：MACD方向(正/负)与RSI偏离中性的程度(RSI-50)的乘积综合了两个动量指标。MACD为正+RSI>50=双动量向上，趋势强劲；MACD为负+RSI<50=双动量向下，趋势疲弱。MACD和RSI是技术分析中最常用的两个动量指标——它们的组合信号比单独使用任何一个都更可靠。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_signal_cross

**定义**：MACD金叉死叉信号因子（日内收盘macd柱>0且dif>dea截面排名，金叉状态排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
sig = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_signal_raw",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(sig)
```

**意义**：日内收盘时的MACD柱(macd = dif - dea)方向是日内趋势的终态判断。macd>0且dif>dea：全天MACD处于多头状态，收盘确认金叉有效性；macd<0且dif<dea：全天MACD处于空头状态，收盘确认死叉。收盘MACD状态比盘中任何单分钟的MACD状态更有信息量——因为它包含了全天的多空博弈结果。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_trend_strength

**定义**：MACD趋势强度因子（(dif-dea)/|dea|截面排名，趋势强度高排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ts = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_trend_strength",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ts)
```

**意义**：DIF与DEA之间的距离(归一化)反映了MACD趋势的强度。|DIF-DEA|/|DEA|大的股票处于强趋势中（无论方向），应有趋势持续性；该比率小的股票处于盘整中，趋势信号不可靠。该因子评估了'MACD信号的可信度'而非方向。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### macd_zero_cross

**定义**：MACD零轴穿越因子（DIF穿越零轴的频率截面排名，频繁穿越=趋势混乱排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
zc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "macd_zero_cross",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-zc)
```

**意义**：DIF在一天内穿越零轴的次数反映了趋势的稳定性。零次穿越=全天DIF维持在同一侧（多或空），趋势稳固；多次穿越=DIF在正负间反复，方向不明确，趋势混乱。频繁零轴穿越的股票后续走势难以预测，应避免。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### mavol5_slope

**定义**：成交量MA5斜率因子（(mavol5_close-mavol5_open)/|mavol5_open|截面排名，成交量均线上移=放量排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ms = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "mavol5_slope",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ms)
```

**意义**：短期成交量均线(MAVol5)的日内斜率反映了成交活跃度的变化方向。MAVol5上升=成交量在5分钟尺度上持续放大，资金参与度提升；MAVol5下降=成交量萎缩，资金参与度降低。放量往往伴随趋势启动或加速，缩量往往预示趋势衰竭。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### mavol_expansion

**定义**：成交量扩张因子（mavol5_close/mavol5_open-1截面排名，成交量扩张=活跃度升排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
me = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "mavol_expansion",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(me)
```

**意义**：MAVol5在一天内的膨胀率度量了成交活跃度的日内变化。正值且大=成交量在日内显著放大，资金正在涌入/涌出；负值=成交量在日内萎缩，交易兴趣降低。成交量扩张是趋势可靠性的重要确认——价格上涨+成交量扩张=健康上涨；价格上涨+成交量萎缩=上涨乏力。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### mavol_ratio_signal

**定义**：成交量均线比率因子（mavol5/mavol10截面排名，比率>1=短期放量排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
mr = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "mavol_ratio",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(mr)
```

**意义**：短期(5分钟)成交量均线与中期(10分钟)成交量均线的比率是经典的放量/缩量信号。比率>1=短期成交活跃度高于中期，正在放量（可能伴随突破）；比率<1=短期成交萎缩，正在缩量（变盘前的沉寂）。与日线的量比不同，1分钟级别的vol均线比更能捕捉日内微观放量。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### mavol_ratio_std

**定义**：成交量比率波动因子（std(mavol5/mavol10)截面排名，比率波动大=成交量不稳定排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rs = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "mavol_ratio_std",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-rs)
```

**意义**：MAVol比率(5/10)在一天内的波动性反映了成交量节奏的稳定性。高波动=成交量忽大忽小，资金进出无序；低波动=成交量节奏稳定，交易行为有规律。成交量的不稳定性往往伴随着价格的不确定性。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### mavol_ratio_trend

**定义**：成交量比率趋势因子（mavol比率的日内时间相关性截面排名，比率上升=持续放量排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rt = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "mavol_ratio_trend",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(rt)
```

**意义**：MAVol比率(5/10)与时间的相关性反映了成交量变化的持续性。正相关=成交量比率在日内持续走高——放量趋势具有持续性；负相关=成交量比率在日内走低——缩量趋势持续。持续放量比间歇性放量更能确认趋势的有效性。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### mavol_stability

**定义**：成交量稳定性因子（std(mavol5)/mean(mavol5)截面排名，成交量不稳定排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ms = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "mavol_stability",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-ms)
```

**意义**：MAVol5的变异系数度量了短期成交量均线的稳定性。高CV=成交量忽大忽小，波动剧烈，资金行为不稳定；低CV=成交量平稳，交易节奏健康。成交量的稳定性本身就是一个信号——稳定的成交量意味着成熟的交易结构，不稳定的成交量意味着情绪化交易。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### min_boll_width_std_20

**定义**：分钟布林带宽波动因子：20日(分钟带宽日内标准差)均值截面排名（负向，带宽反复扩张挤压排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "min_boll_width_std_20"))
```

**意义**：分钟布林带宽的日内波动=挤压-扩张的日内反复次数——反复挤压扩张=变盘酝酿(方向未定);带宽日内平稳=趋势节奏稳定。与 boll_squeeze/boll_width_5d_change(日频带宽水平与变化)区分:本因子是带宽的日内节奏维度。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### min_expand_bull_frac_20

**定义**：分钟放量多头占比因子：20日(放量且多头排列分钟占比)均值截面排名（量价趋势三线确认排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "min_expand_bull_frac_20"))
```

**意义**：mavol5>mavol10(量能扩张)且 ma5>ma10(价格多头)的分钟占比——量价+趋势三重确认的上涨质量。与 min_ma_alignment_frac_20(纯价格排列)区分:本因子叠加量能维度,「多头但无量」的诱多形态在此被排除。indicator_1min 无原始 vol 列,mavol 关系代理量能状态。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### min_j_overbought_frac_20

**定义**：分钟KDJ超买占比因子：20日(J>100分钟占比)均值截面排名（负向，盘中反复冲顶排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "min_j_overbought_frac_20"))
```

**意义**：J>100 的分钟占比=盘中超买状态的持续程度——持续超买=情绪票(买盘透支,回调风险累积)。与日线 kdj_overbought_frac 区分:本因子是分钟粒度,把「盘中反复冲顶」与「日线级别超买」分开,方向与既有超买类一致。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### min_ma_alignment_frac_20

**定义**：分钟均线多头排列占比因子：20日(ma5>ma10>ma20>ma30分钟占比)均值截面排名（日内趋势稳固排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "min_ma_alignment_frac_20"))
```

**意义**：盘中分钟均线多头排列的时间占比=日内趋势的稳固度:全天保持多头排列=趋势在日内持续(多方掌控);尾盘才翻多=日内反复。与日频ma_alignment_score 区分:本因子是分钟粒度,尾盘偷袭与全天多头分离。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### min_macd_hist_area_20

**定义**：分钟MACD柱面积因子：20日(Σ分钟MACD/当日收盘价)均值截面排名（日内动能净值排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "min_macd_hist_area_20"))
```

**意义**：日内 MACD 柱净面积=红柱面积−绿柱面积=全天动能净值,按当日收盘价归一(消除股价量纲)。区分「尾盘翻红但日内整体空头」与「全天单边红柱」。与 am_macd_trend/pm_macd_trend(上下半场斜率)互补:本因子是全天积分。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### min_rsi_extreme_frac_20

**定义**：分钟RSI极值占比因子：20日(RSI>80或<20分钟占比)均值截面排名（负向，情绪烈度高排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "min_rsi_extreme_frac_20"))
```

**意义**：分钟 RSI 触及极端区(>80 或 <20)的时长占比=情绪化交易的主导程度,不分多空方向。与日线 rsi_extreme_fraction/rsi_overbought_frac 区分:本因子是分钟粒度的情绪烈度,度量盘中反复过热的持续时长。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### min_rsi_overbought_expand_20

**定义**：分钟超买放量占比因子：20日(RSI>70且放量分钟占比)均值截面排名（负向，追高放量排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "min_rsi_overbought_expand_20"))
```

**意义**：RSI>70 且量能扩张同时发生的分钟占比——情绪过热叠加放量=追高/拉高出货特征(高位放量换手)。与 min_j_overbought_frac_20(KDJ 超买,无量能维度)区分:本因子把「超买」与「放量」联合,是诱多识别的指标侧信号。方向与既有超买类一致(neg)。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### min_shrink_bull_frac_20

**定义**：分钟缩量多头占比因子：20日(缩量但多头排列分钟占比)均值截面排名（负向，无量上涨排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "min_shrink_bull_frac_20"))
```

**意义**：多头排列但量能收缩(mavol5<mavol10)的分钟占比——价格上涨缺乏量能确认=无量反弹/诱多嫌疑(拉高无人跟风)。与 min_expand_bull_frac_20 互补:把「多头行情中的量能质量」拆成放量/缩量两轴,方向 neg。与 Class 3 的 vp_shrink_up_share(真实分钟量四象限)区分:本因子用mavol 代理与指标状态,粒度更粗但跨日更稳定。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### multi_indicator_extreme

**定义**：多指标极端值因子（5个标准化指标的|z-score|均值截面排名，多指标同时极端=变盘排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
me = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "multi_indicator_extreme",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-me)
```

**意义**：五个标准化指标信号的绝对Z-score均值。均值高=多个指标同时处于极端区域——强烈的超买或超卖；均值低=所有指标都在正常范围内。多指标同时极端是'过度延伸'的量化表达——当所有技术指标都指向极端时，反转的概率急剧升高。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### pm_macd_trend

**定义**：午盘MACD趋势因子（下午MACD与时间的相关性截面排名，正相关=午盘动能持续排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
pt = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "pm_macd_trend",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(pt)
```

**意义**：午盘(13:00-15:00)MACD的日内趋势反映了下午交易时段的多空演变。午盘MACD上升=多头在下午持续发力，全日强势；午盘MACD下降=多头在下午衰竭或空头反攻。A股的'下午反转'现象可以通过该因子捕捉——早盘强势但午盘MACD趋势转负的股票容易出现下午跳水。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### pm_rsi_trend

**定义**：午盘RSI趋势因子（下午RSI与时间的相关性截面排名，正相关=午盘动量持续排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
pt = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "pm_rsi_trend",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(pt)
```

**意义**：午盘RSI的日内趋势是下午动量方向的独立度量。RSI持续走高=买盘在下午持续发力；RSI持续走低=下午卖压加重。该因子与pm_macd_trend互补——MACD偏趋势，RSI偏动量。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### price_vs_ma10_deviation

**定义**：价格对MA10偏离因子：close相对分钟MA10的偏离截面排名（负向绝对值，大幅偏离排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "price_vs_ma10").abs())
```

**意义**：分钟级 MA10 是短周期均价中枢——价格大幅偏离 MA10=短期过热/超卖(均值回归风险),贴均线运行=健康趋势。与 price_vs_ma20/ma60 构成不同周期的偏离谱系,MA10 对短线择时更敏感。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### price_vs_ma20_deviation

**定义**：价格偏离MA20因子（(close-ma20)/|ma20|截面排名，偏离大=均值回归压力大排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
pv = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "price_vs_ma20",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-pv.abs())
```

**意义**：价格对20日均线(MA20)的偏离百分比是一个经典的均值回归信号。大幅高于MA20=短期内涨幅过大，存在获利了结压力；大幅低于MA20=短期内跌幅过大，存在技术性反弹需求。使用1分钟MA20（而非日线MA20）的优势在于更及时地反映日内趋势。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### price_vs_ma30_deviation

**定义**：价格对MA30偏离因子：close相对分钟MA30的偏离截面排名（负向绝对值，大幅偏离排后）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(-_metric(context, "price_vs_ma30").abs())
```

**意义**：MA30(6个交易日约)介于短中周期之间——价格显著高于 MA30=短期涨幅透支,显著低于=超跌待修复。负向绝对值排名:偏离越大越靠后(极端偏离回归概率高)。与 MA20/MA60 版本互补,细化偏离周期。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### price_vs_ma60_deviation

**定义**：价格偏离MA60因子（(close-ma60)/|ma60|截面排名，偏离大=长期趋势偏离排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
pv = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "price_vs_ma60",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-pv.abs())
```

**意义**：价格对60日均线(MA60)的偏离反映了中长期趋势的偏离程度。MA60是经典的中期趋势线——价格大幅偏离MA60意味着与中期趋势的背离，回归概率较高。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### rsi_14_excess

**定义**：RSI超买超卖因子（(rsi_close-50)截面排名，高RSI=超买排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rsi = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "rsi_excess",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-rsi)
```

**意义**：RSI偏离50的程度直接度量了短期超买/超卖状态。使用1分钟RSI的收盘值（最后一分钟）作为当日RSI的终态判断。RSI>70=超买（排后），RSI<30=超卖（排前），中间区域线性过渡。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### rsi_boll_combo

**定义**：RSI-布林带组合因子（rsi_zscore*boll_position截面排名，双双极端=强烈信号排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "rsi_boll_combo",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(rc)
```

**意义**：RSI的Z-score与布林带位置的乘积捕捉了两个指标同时发出极端信号的时刻。RSI极端超买+价格在上轨附近=强烈的超买信号（可能反转）；RSI极端超卖+价格在下轨附近=强烈的超卖信号（可能反弹）。同向极端=两个独立的指标体系互相验证，信号可靠性显著提高。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### rsi_day_position

**定义**：RSI日内区间位置因子：(RSI收盘−RSI最低)/(RSI最高−RSI最低)截面排名（收盘靠上沿排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "rsi_day_position"))
```

**意义**：RSI 收盘在当日区间中的位置衡量日内动能的方向终态——收盘靠上沿=日内多方逐步占据优势(收盘确认强势);靠下沿=空方压制。与 rsi_range(区间宽度)正交:宽度管振幅、位置管方向。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### rsi_extreme_fraction

**定义**：RSI极端时间占比因子（RSI>70或<30的分钟占比截面排名，高占比=极端行情排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ef = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "rsi_extreme_frac",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ef)
```

**意义**：RSI在极端区域(>70或<30)停留的时间占比反映了行情的极端程度。高占比=全天大部分时间处于超买或超卖状态，单边行情特征明显；低占比=RSI在中性区域波动，典型的震荡行情。极端时间占比异常升高往往是变盘的前兆。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### rsi_intraday_trend

**定义**：RSI日内趋势因子（RSI与时间的相关性截面排名，正相关=日内动量积聚排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
tc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "rsi_time_corr",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(tc)
```

**意义**：RSI在一天中与分钟序号的相关系数反映了日内动量的积聚方向。正相关=RSI随交易推进而上升（盘中资金持续流入，动量积聚）；负相关=RSI随交易推进而下降（盘中资金持续流出，动能衰竭）。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### rsi_overbought_frac

**定义**：RSI超买时间占比因子（RSI>70分钟占比截面排名，高占比=强势超买排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ro = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "rsi_overbought_frac",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-ro)
```

**意义**：RSI在超买区域(>70)停留的时间占比是独立的超买强度度量。与rsi_extreme_fraction不同（它混合了超买和超卖），该因子只捕获单边的超买压力。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### rsi_oversold_frac

**定义**：RSI超卖时间占比因子（RSI<30分钟占比截面排名，高占比=深度超卖排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
ro = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "rsi_oversold_frac",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(ro)
```

**意义**：RSI在超卖区域(<30)停留的时间占比是独立的超卖强度度量。持续超卖后的反弹概率显著升高，是逆向买入信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### rsi_range

**定义**：RSI日内振幅因子（rsi_max-rsi_min截面排名，振幅大=剧烈波动排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rr = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "rsi_range",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-rr)
```

**意义**：RSI在一天内的波动范围反映了日内情绪的摇摆程度。振幅大=RSI经历了从超买到超卖的剧烈摆动（或反之），行情极不稳定；振幅小=RSI在狭窄范围内波动，情绪稳定。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### rsi_trend_ma5

**定义**：RSI超额5日均值因子：(RSI−50)的5日移动平均截面排名（中期动能排前）。

**公式（计算逻辑）**：

```python
return cross_sectional_rank(_metric(context, "rsi_excess_ma5"))
```

**意义**：RSI 相对50中线的超额部分5日均值=中期动能水平(过滤单日噪音)——持续正值=中期多方占优;持续负值=中期空头主导。与 rsi_14_excess(当日)互补:本因子为5日平滑版本,更贴近趋势状态。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute_extra.py`

##### rsi_volatility

**定义**：RSI波动率因子（std(RSI)/mean(RSI)截面排名，RSI不稳定=信号质量差排后）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
rv = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "rsi_volatility",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(-rv)
```

**意义**：RSI的变异系数度量了RSI本身的稳定性。高CV=RSI频繁上下跳动，技术信号不可靠；低CV=RSI变化平滑，信号质量高。该因子评估了RSI信号的信噪比。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### trend_confirmation

**定义**：趋势确认因子（ma_alignment*sign(macd_close)截面排名，均线+MACD双确认排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
tc = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "trend_confirmation",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(tc)
```

**意义**：MA排列分数(0-4)与MACD方向(+/-)的乘积综合了两个最重要的趋势指标。正值大=MA多头排列+MACD为正——双重确认上升趋势；负值大=MA空头排列+MACD为负——双重确认下降趋势。MA排列和MACD各自独立地度量趋势，它们的乘积提供了更强的趋势确认。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

##### volume_price_confirmation

**定义**：量价配合确认因子（mavol5_slope符号==ma5_slope符号截面排名，量价同步=趋势确认排前）。

**公式（计算逻辑）**：

```python
source_root = context.repo.paths.source_root
vp = _compute_indicator_factor(
    source_root, context.repo.allowed_codes, "volume_price_confirmation",
    on_progress=context.repo.on_progress,
)
return cross_sectional_rank(vp)
```

**意义**：成交量MA5斜率与价格MA5斜率的方向一致性是经典的量价配合确认。方向一致=量价同步（价涨量增/价跌量减），趋势有成交量支撑；方向不一致=量价背离（价涨量缩/价跌量增），趋势不可靠。该因子将'量价配合'概念量化为1分钟级别的微观信号。

**依赖数据**：`indicator_1min` ｜ **Class**：4 ｜ **源码**：`factors/indicator_minute.py`

---

### <a name="class-5"></a>Class 5 — 因子耦合类（147 个）

加载已有因子 `.fea` 文件（依赖 `__factors__`），做因子间耦合、共振与二次组合。

#### <a name="cat-risk-c5"></a>类别 risk — 风险（28 个）

##### amihud_parkinson_ratio

**定义**：非流动性-波动率比因子，amihud_intraday排名/parkinson_vol排名截面排名（高冲击成本相对低波动的异常排后）。

**公式（计算逻辑）**：

```python
df = ctx.load_factors(["amihud_intraday", "parkinson_vol"])
amihud_r = _rank(df["amihud_intraday"])
park_r = _rank(df["parkinson_vol"])
ratio = safe_divide(amihud_r, park_r + 1e-8)
return cross_sectional_rank(-ratio)  # low illiquidity per unit vol = good
```

**意义**：Amihud非流动性与Parkinson波动率的比率度量单位波动率的流动性成本——同样的价格波动下，流动性成本越高的股票交易执行质量越差。低比率=高效率交易、低比率=低隐性成本，alpha来自于交易效率的差异。

**依赖数据**：`__factors__`、`amihud_intraday`、`parkinson_vol` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_amplitude_20

**定义**：对数变换20日振幅因子，log(amplitude_20+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始20日振幅因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`amplitude_20` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_gk_vol

**定义**：对数变换Garman-Klass波动率因子，log(gk_vol+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始Garman-Klass波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`gk_vol` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_high_low_volatility_20

**定义**：对数变换20日高低波动率因子，log(high_low_volatility_20+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始20日高低波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`high_low_volatility_20` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_hl_range_intraday

**定义**：对数变换日内高低价差因子，log(hl_range_intraday+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始日内高低价差因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`hl_range_intraday` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_intraday_high_low_volatility

**定义**：对数变换日内高低波动率因子，log(intraday_high_low_volatility+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始日内高低波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`intraday_high_low_volatility` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_parkinson_vol

**定义**：对数变换Parkinson波动率因子，log(parkinson_vol+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始Parkinson波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`parkinson_vol` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_ret_std_intraday

**定义**：对数变换日内收益标准差因子，log(ret_std_intraday+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始日内收益标准差因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`ret_std_intraday` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_rv_10min

**定义**：对数变换10分钟已实现波动率因子，log(rv_10min+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始10分钟已实现波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`rv_10min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_rv_15min

**定义**：对数变换15分钟已实现波动率因子，log(rv_15min+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始15分钟已实现波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`rv_15min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_rv_30min

**定义**：对数变换30分钟已实现波动率因子，log(rv_30min+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始30分钟已实现波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`rv_30min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_rv_5min

**定义**：对数变换5分钟已实现波动率因子，log(rv_5min+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始5分钟已实现波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`rv_5min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_rv_60min

**定义**：对数变换60分钟已实现波动率因子，log(rv_60min+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始60分钟已实现波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`rv_60min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### log_rv_rolling_5d_std

**定义**：对数变换5日RV波动率因子，log(rv_rolling_5d_std+ε)截面排名（低波动排前）。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
log_val = np.log(raw.clip(lower=1e-10))
return cross_sectional_rank(-log_val)  # low vol ranks higher
```

**意义**：原始5日RV波动率因子分布极度右偏（少数高波动日极端值主导），对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。低波动股票在A股市场长期具有alpha溢价。

**依赖数据**：`__factors__`、`rv_rolling_5d_std` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### microstructure_efficiency

**定义**：微观结构效率因子，(rv_5min/parkinson_vol排名)截面排名（日内效率高=信息消化快排前）。

**公式（计算逻辑）**：

```python
df = ctx.load_factors(["rv_5min", "parkinson_vol"])
efficiency = safe_divide(df["rv_5min"], df["parkinson_vol"] + 1e-10)
return cross_sectional_rank(-efficiency)  # closer to 1 = more efficient
```

**意义**：RV(5min)与Parkinson波动率(日高低价)的比率反映市场微观结构效率——比值接近1=日内价格发现效率高、信息被均匀消化；比值远大于1=日内波动远大于日间波动=价格发现效率低、信息在日内被过度反应后修正。高效率股票的未来收益可预测性更强。

**依赖数据**：`__factors__`、`rv_5min`、`parkinson_vol` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### rv_term_structure

**定义**：RV期限结构因子，rv_5min排名/rv_60min排名截面排名（短期波动相对长期波动高=波动正在集聚排后）。

**公式（计算逻辑）**：

```python
df = ctx.load_factors(["rv_5min", "rv_60min"])
term = safe_divide(df["rv_5min"], df["rv_60min"] + 1e-10)
return cross_sectional_rank(-term)  # low short-term RV relative to long-term = good
```

**意义**：已实现波动率的期限结构反映波动率的短期vs长期动态——短期RV远高于长期RV=波动率正在急剧上升(通常是负面事件驱动)；短期RV远低于长期RV=波动率正在消退(不确定性解除)。波动率消退期的股票回报率更高。

**依赖数据**：`__factors__`、`rv_5min`、`rv_60min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### sqrt_am_hl_range_intraday

**定义**：平方根变换上午高低价差因子，sign×√(|am_hl_range_intraday|)截面排名。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
sqrt_val = np.sign(raw) * np.sqrt(np.abs(raw))
return cross_sectional_rank(sqrt_val)
```

**意义**：原始上午高低价差因子极端值被少数异常交易日主导。平方根变换压缩极端值的影响同时保留符号方向，使截面排名更稳定、更少受离群值扰动。

**依赖数据**：`__factors__`、`am_hl_range_intraday` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### sqrt_max_ret_intraday

**定义**：平方根变换日内最大收益因子，sign×√(|max_ret_intraday|)截面排名。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
sqrt_val = np.sign(raw) * np.sqrt(np.abs(raw))
return cross_sectional_rank(sqrt_val)
```

**意义**：原始日内最大收益因子极端值被少数异常交易日主导。平方根变换压缩极端值的影响同时保留符号方向，使截面排名更稳定、更少受离群值扰动。

**依赖数据**：`__factors__`、`max_ret_intraday` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### sqrt_relative_spread

**定义**：平方根变换相对价差因子，sign×√(|relative_spread|)截面排名。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
sqrt_val = np.sign(raw) * np.sqrt(np.abs(raw))
return cross_sectional_rank(sqrt_val)
```

**意义**：原始相对价差因子极端值被少数异常交易日主导。平方根变换压缩极端值的影响同时保留符号方向，使截面排名更稳定、更少受离群值扰动。

**依赖数据**：`__factors__`、`relative_spread` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### sqrt_turnover_std_20

**定义**：平方根变换20日换手标准差因子，sign×√(|turnover_std_20|)截面排名。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
sqrt_val = np.sign(raw) * np.sqrt(np.abs(raw))
return cross_sectional_rank(sqrt_val)
```

**意义**：原始20日换手标准差因子极端值被少数异常交易日主导。平方根变换压缩极端值的影响同时保留符号方向，使截面排名更稳定、更少受离群值扰动。

**依赖数据**：`__factors__`、`turnover_std_20` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### sqrt_turnover_vol_20

**定义**：平方根变换20日换手波动率因子，sign×√(|turnover_vol_20|)截面排名。

**公式（计算逻辑）**：

```python
raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
sqrt_val = np.sign(raw) * np.sqrt(np.abs(raw))
return cross_sectional_rank(sqrt_val)
```

**意义**：原始20日换手波动率因子极端值被少数异常交易日主导。平方根变换压缩极端值的影响同时保留符号方向，使截面排名更稳定、更少受离群值扰动。

**依赖数据**：`__factors__`、`turnover_vol_20` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### turnover_rv_interaction

**定义**：换手-波动耦合因子，turnover_20排名×rv_5min排名截面排名（量价共振强度排前）。

**公式（计算逻辑）**：

```python
df = ctx.load_factors(["turnover_20", "rv_5min"])
to_r = _rank(df["turnover_20"])
rv_r = _rank(df["rv_5min"])
# High turnover + low RV = accumulation signal (rank high)
interaction = to_r * (1 - rv_r)
return cross_sectional_rank(interaction)
```

**意义**：换手率和已实现波动率的乘积捕获量价共振——高换手×高波动=市场分歧大、交易活跃但方向不明(排后)；低换手×低波动=市场共识强、价格稳定(排前)；高换手×低波动=资金在稳定吸筹(积极信号)；低换手×高波动=流动性枯竭+价格剧烈波动(危险信号)。

**依赖数据**：`__factors__`、`turnover_20`、`rv_5min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### vol_of_rv

**定义**：波动率的波动率因子，rv_5min的20日滚动标准差截面排名（波动率不稳定排后）。

**公式（计算逻辑）**：

```python
rv = ctx.load_factor("rv_5min")
rv_std = rolling_group_std(rv, 20, min_periods=10)
return cross_sectional_rank(-rv_std)  # stable vol ranks higher
```

**意义**：已实现波动率本身也有波动率——波动率不稳定=市场对股票的定价不确定性高。波动率稳定的股票信息环境清晰、定价效率高，未来收益的可预测性更强。vol-of-vol是波动率维度的二阶风险度量。

**依赖数据**：`__factors__`、`rv_5min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### zscore_amihud_intraday

**定义**：时序Z-score变换日内Amihud非流动性因子，(原始值-252日均值)/252日标准差截面排名。

**公式（计算逻辑）**：

```python
raw = ctx.load_factor(base_name)
roll_mean = rolling_group_mean(raw, 252, min_periods=60)
roll_std = rolling_group_std(raw, 252, min_periods=60)
zscore = safe_divide(raw - roll_mean, roll_std + 1e-8)
return cross_sectional_rank(zscore)
```

**意义**：原始日内Amihud非流动性因子的绝对水平受股票自身特征(市值、行业、流动性)严重影响。时序Z-score标准化到股票自身历史分布后，捕捉的是该因子在当前时点相对于其自身历史的异常程度——这比绝对水平更具预测价值。

**依赖数据**：`__factors__`、`amihud_intraday` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### zscore_realized_spread_5min

**定义**：时序Z-score变换5分钟已实现价差因子，(原始值-252日均值)/252日标准差截面排名。

**公式（计算逻辑）**：

```python
raw = ctx.load_factor(base_name)
roll_mean = rolling_group_mean(raw, 252, min_periods=60)
roll_std = rolling_group_std(raw, 252, min_periods=60)
zscore = safe_divide(raw - roll_mean, roll_std + 1e-8)
return cross_sectional_rank(zscore)
```

**意义**：原始5分钟已实现价差因子的绝对水平受股票自身特征(市值、行业、流动性)严重影响。时序Z-score标准化到股票自身历史分布后，捕捉的是该因子在当前时点相对于其自身历史的异常程度——这比绝对水平更具预测价值。

**依赖数据**：`__factors__`、`realized_spread_5min` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### zscore_turnover_20

**定义**：时序Z-score变换20日换手率因子，(原始值-252日均值)/252日标准差截面排名。

**公式（计算逻辑）**：

```python
raw = ctx.load_factor(base_name)
roll_mean = rolling_group_mean(raw, 252, min_periods=60)
roll_std = rolling_group_std(raw, 252, min_periods=60)
zscore = safe_divide(raw - roll_mean, roll_std + 1e-8)
return cross_sectional_rank(zscore)
```

**意义**：原始20日换手率因子的绝对水平受股票自身特征(市值、行业、流动性)严重影响。时序Z-score标准化到股票自身历史分布后，捕捉的是该因子在当前时点相对于其自身历史的异常程度——这比绝对水平更具预测价值。

**依赖数据**：`__factors__`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### zscore_turnover_f_20

**定义**：时序Z-score变换20日自由流通换手率因子，(原始值-252日均值)/252日标准差截面排名。

**公式（计算逻辑）**：

```python
raw = ctx.load_factor(base_name)
roll_mean = rolling_group_mean(raw, 252, min_periods=60)
roll_std = rolling_group_std(raw, 252, min_periods=60)
zscore = safe_divide(raw - roll_mean, roll_std + 1e-8)
return cross_sectional_rank(zscore)
```

**意义**：原始20日自由流通换手率因子的绝对水平受股票自身特征(市值、行业、流动性)严重影响。时序Z-score标准化到股票自身历史分布后，捕捉的是该因子在当前时点相对于其自身历史的异常程度——这比绝对水平更具预测价值。

**依赖数据**：`__factors__`、`turnover_f_20` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

##### zscore_volume_rv_ratio

**定义**：时序Z-score变换成交量-RV比率因子，(原始值-252日均值)/252日标准差截面排名。

**公式（计算逻辑）**：

```python
raw = ctx.load_factor(base_name)
roll_mean = rolling_group_mean(raw, 252, min_periods=60)
roll_std = rolling_group_std(raw, 252, min_periods=60)
zscore = safe_divide(raw - roll_mean, roll_std + 1e-8)
return cross_sectional_rank(zscore)
```

**意义**：原始成交量-RV比率因子的绝对水平受股票自身特征(市值、行业、流动性)严重影响。时序Z-score标准化到股票自身历史分布后，捕捉的是该因子在当前时点相对于其自身历史的异常程度——这比绝对水平更具预测价值。

**依赖数据**：`__factors__`、`volume_rv_ratio` ｜ **Class**：5 ｜ **源码**：`factors/microstructure_transform.py`

#### <a name="cat-coupling-c5"></a>类别 coupling — 因子耦合（117 个）

##### bigorder_momentum_resonance_20

**定义**：大单×动量：mf_big_order_ratio×momentum_20。大单占比高+上涨=机构主导的行情。

**公式（计算逻辑）**：

```python
big = ctx.load_factor("mf_big_order_ratio")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(big * mom)
```

**意义**：大单占比反映机构/大户参与度——上涨行情中若大单占比持续高,说明是机构主导的拉升(散户小单无法推动大行情),行情的级别和持续性更高。

**依赖数据**：`__factors__`、`mf_big_order_ratio`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### bp_factor_momentum_20

**定义**：BP因子20日动量 (BP因子值的变化率)。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
mom = _momentum(bp, 20)
return cross_sectional_rank(mom)
```

**意义**：BP自身的动量捕捉估值修复的启动——BP快速上升的股票正在被市场重估

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### bp_momentum_combo_20

**定义**：价值+动量复合：(bp+momentum_20)/2。低估且走强的股票双因子确认。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank((bp + mom) / 2.0)
```

**意义**：经典 HML×MOM 的 A股实现:既便宜(bp高)又在上涨(momentum_20高)的股票同时获得价值与趋势资金的支撑,是价值-动量复合中最常见的alpha组合。

**依赖数据**：`__factors__`、`bp`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### bp_momentum_divergence_20

**定义**：估值-趋势背离：momentum_20−bp。趋势强但估值贵的股票(估值透支)排名高。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(mom - bp)
```

**意义**：趋势与估值背离的两种情形:①趋势强+估值贵=涨幅透支基本面,回调风险大(排名高=谨慎);②趋势弱+估值便宜=超跌价值股,可能被错杀。该因子作为估值约束的动量信号。

**依赖数据**：`__factors__`、`bp`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### chip_momentum_resonance_20

**定义**：筹码集中×动量：chip_cr3_factor×momentum_20。筹码集中且走强=主力控盘的趋势。

**公式（计算逻辑）**：

```python
chip = ctx.load_factor("chip_cr3_factor")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(chip * mom)
```

**意义**：筹码集中度(前3大成本区占比)高+价格走强=主力吸筹完成后的拉升阶段;筹码分散+走强=跟风盘推动、随时可能抛压出逃。筹码结构确认的趋势更可靠。

**依赖数据**：`__factors__`、`chip_cr3_factor`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### chip_price_resonance_20

**定义**：筹码×价格共振因子：winner_rate×momentum_20×price_to_52w_high截面排名。

**公式（计算逻辑）**：

```python
wr = ctx.load_factor("winner_rate")
mom = ctx.load_factor("momentum_20")
prox = ctx.load_factor("price_to_52w_high")
return cross_sectional_rank(wr * mom * prox)
```

**意义**：筹码压力小(获利盘占比低、上方套牢出清)+动量向上+接近52周新高=趋势的筹码结构与价格结构双重确认——筹码干净让上涨无解套抛压,接近新高确认空间打开。筹码(winner_rate)、动量(momentum)、位置(52w高)三个正交维度共振。

**依赖数据**：`__factors__`、`winner_rate`、`momentum_20`、`price_to_52w_high` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### coupling_bigflow_margin_buy_20

**定义**：大单融资双确认因子：mf_net_inflow_ratio×margin_net_flow_ratio截面排名（两类聪明钱同向流入排前）。

**公式（计算逻辑）**：

```python
mg = ctx.load_factor("margin_net_flow_ratio")
mf = ctx.load_factor("mf_net_inflow_ratio").reindex(mg.index)
return cross_sectional_rank(mg * mf)
```

**意义**：场内大单(主力资金)与场外杠杆资金(融资净流入)同时净流入=两类聪明钱互相印证,信号质量最高;单边流入但另一边撤退(分歧)则被乘法稀释。与 margin_price_resonance_20(融资×价格)区分:本因子是资金×资金的双确认,不依赖价格方向。⚠️ 以 margin 因子索引 reindex 防并集。

**依赖数据**：`__factors__`、`mf_net_inflow_ratio`、`margin_net_flow_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra3.py`

##### coupling_boll_squeeze_value

**定义**：布林挤压(boll_squeeze) × 低估值(bp)。布林收窄(蓄力)+低估=突破前的最优买点。

**公式（计算逻辑）**：

```python
boll = ctx.load_factor("boll_squeeze")
bp = ctx.load_factor("bp")
signal = (1.0 - boll) * bp
return cross_sectional_rank(signal)
```

**意义**：布林带收窄=波动率压缩，价格在狭窄区间整理——是突破前的蓄力阶段。收窄+低估=一只便宜的股票正在筑底蓄力。

**依赖数据**：`__factors__`、`boll_squeeze`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_chip_cost_accel_20

**定义**：筹码成本上移加速度：chip_median_momentum_t − chip_median_momentum_t-20 截面排名（成本上移20日加速排前）。

**公式（计算逻辑）**：

```python
cm = ctx.load_factor("chip_median_momentum")
return cross_sectional_rank(cm - _lag(cm, 20))
```

**意义**：筹码中位成本动量排名的 20 日漂移:排名持续抬升=吸筹成本在加速上移,主力在更高价位继续收集筹码,成本重心跟随价格上行=健康拉升;排名回落=成本重心滞涨,价格与筹码成本背离,警惕派发。

**依赖数据**：`__factors__`、`chip_median_momentum` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_chip_support_reversal_5

**定义**：筹码支撑超跌因子：short_term_reversal_5 × chip_support_strength 截面排名（超跌且有筹码支撑排前）。

**公式（计算逻辑）**：

```python
rev = ctx.load_factor("short_term_reversal_5")
sup = ctx.load_factor("chip_support_strength").reindex(rev.index)
return cross_sectional_rank(rev * sup)
```

**意义**：超跌反弹的质量过滤:单纯超跌(short_term_reversal_5 高)可能继续阴跌,但在筹码成本 15pct 支撑位上的超跌=下方有真实承接盘,反弹安全边际高。Class 2 筹码基因首次与反转因子耦合。

**依赖数据**：`__factors__`、`short_term_reversal_5`、`chip_support_strength` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_chip_trend_confirm_20

**定义**：筹码成本确认趋势：momentum_20 × chip_peak_shift 截面排名（趋势+成本峰20日上移排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
cps = ctx.load_factor("chip_peak_shift").reindex(mom.index)
return cross_sectional_rank(mom * cps)
```

**意义**：价格趋势与筹码成本重心同向:20 日动量上涨的同时,筹码峰(成本中位数)也在20 日上移=上涨由真实换手成本抬升支撑,而非缩量虚涨。与 chip_momentum_resonance_20(cr3 集中度)区分:本因子用成本重心位移,捕捉'换手推升'。

**依赖数据**：`__factors__`、`momentum_20`、`chip_peak_shift` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_flow_persistence_20

**定义**：融资净流入跨期自共振：margin_net_flow_ratio_t × margin_net_flow_ratio_t-20 截面排名（20日持续净流入排前）。

**公式（计算逻辑）**：

```python
mg = ctx.load_factor("margin_net_flow_ratio")
return cross_sectional_rank(mg * _lag(mg, 20))
```

**意义**：融资净流入的跨时点持续性:今天与 20 天前都在净流入=杠杆资金的中期建仓行为,区别于单日脉冲式流入(游资一日游)。持续性资金比脉冲资金对趋势的支撑更可靠。纯单基因时间耦合,无并集风险。

**依赖数据**：`__factors__`、`margin_net_flow_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_flowaccel_breakout_60

**定义**：资金加速突破因子：big_order_net_accel_10 × new_high_60_event 截面排名（大单加速+60日新高排前）。

**公式（计算逻辑）**：

```python
acc = ctx.load_factor("big_order_net_accel_10")
nh = ctx.load_factor("new_high_60_event").reindex(acc.index)
return cross_sectional_rank(acc * nh)
```

**意义**：有效突破的资金验证:大单净额 5 日相对前 5 日加速流入(资金在突破前吸筹)+60 日新高事件=资金驱动的真突破;新高但大单净流出=拉高出货的假突破。与 liftoff 族(动量×放量×回撤)区分:本因子用事件衰减形态的新高+大单加速度。

**依赖数据**：`__factors__`、`big_order_net_accel_10`、`new_high_60_event` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_fundflow_accel_10

**定义**：主力资金时间加速度：mf_net_inflow_ratio_t − mf_net_inflow_ratio_t-10 截面排名（净流入排名10日抬升排前）。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_ratio")
return cross_sectional_rank(mf - _lag(mf, 10))
```

**意义**：主力净流入截面排名的 10 日漂移:排名抬升=资金在转强(流出一致性减弱→流入一致性增强),排名回落=资金在转弱。与基于原始字段的 mf_flow_acceleration_5d区分:本因子作用于 rank 序列,天然截面可比、剔除个股量纲差异。

**依赖数据**：`__factors__`、`mf_net_inflow_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_indicator_consensus_value

**定义**：技术指标共识度(indicator_consensus) × 低估值(bp)。多指标一致看多+低估=最强信号。

**公式（计算逻辑）**：

```python
consensus = ctx.load_factor("indicator_consensus")
bp = ctx.load_factor("bp")
signal = consensus * bp
return cross_sectional_rank(signal)
```

**意义**：当MACD/KDJ/RSI/Bollinger等多个指标一致发出看多信号，且股票处于低估状态时——技术面和基本面达成了罕见的全票通过。

**依赖数据**：`__factors__`、`indicator_consensus`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_intraday_tail_momentum_20

**定义**：尾盘资金确认趋势：momentum_20 × tail_volume_share 截面排名（趋势+尾盘放量排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
tvs = ctx.load_factor("tail_volume_share").reindex(mom.index)
return cross_sectional_rank(mom * tvs)
```

**意义**：尾盘 30 分钟量能是当日资金态度的浓缩:上涨趋势中尾盘放量=资金当日尾段继续买入(次日延续性强),尾盘缩量=拉高无力承接。Class 3 盘中基因(tail_volume_share)首次与日线动量耦合。

**依赖数据**：`__factors__`、`momentum_20`、`tail_volume_share` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_kdj_moneyflow_divergence

**定义**：KDJ金叉净数(kdj_cross_net) vs 主力资金背离。KDJ金叉+主力流出=技术虚涨，KDJ死叉+主力流入=洗盘。

**公式（计算逻辑）**：

```python
kdj = ctx.load_factor("kdj_cross_net")
mf = ctx.load_factor("mf_net_inflow_ratio")
divergence = kdj - mf
return cross_sectional_rank(divergence)
```

**意义**：KDJ频繁金叉但主力持续流出=散户推动的技术反弹(虚涨)；KDJ频繁死叉但主力持续流入=主力在吸筹(洗盘)。

**依赖数据**：`__factors__`、`kdj_cross_net`、`mf_net_inflow_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_lhb_reversal_20

**定义**：博弈票超跌反弹因子：lhb_proxy_score_60×short_term_reversal_5截面排名（游资惯犯且超跌排前）。

**公式（计算逻辑）**：

```python
lhb = ctx.load_factor("lhb_proxy_score_60")
rev = ctx.load_factor("short_term_reversal_5").reindex(lhb.index)
return cross_sectional_rank(lhb * rev)
```

**意义**：龙虎榜替代活跃度(游资博弈票)×5日超跌=游资关注的超跌反弹标的:博弈票弹性大、超跌后反弹兑现快;单纯超跌但无人问津的票(低活跃度)被过滤。与 reversal_oversold_combo_5(通用超跌)区分:本因子限定游资博弈人群,聚焦题材票的反弹窗口。

**依赖数据**：`__factors__`、`lhb_proxy_score_60`、`short_term_reversal_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra3.py`

##### coupling_limitup_momentum_20

**定义**：涨停延续动量：limit_up_fade_10 × momentum_20 截面排名（涨停后延续+动量排前）。

**公式（计算逻辑）**：

```python
fade = ctx.load_factor("limit_up_fade_10")
mom = ctx.load_factor("momentum_20").reindex(fade.index)
return cross_sectional_rank(fade * mom)
```

**意义**：事件后的趋势延续:10 日内有涨停事件且其后累计收益为正(涨停后延续强)叠加 20 日动量=强势事件驱动的趋势;涨停后即回落(limit_up_fade_10 低)的动量=情绪顶点出货。用事件条件过滤动量中的'最后一段'。

**依赖数据**：`__factors__`、`limit_up_fade_10`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_liquidity_momentum_20d

**定义**：流动性因子(turnover_20)自身20日变化。换手率上升=关注度提升，排名高。

**公式（计算逻辑）**：

```python
# turnover_20 的 .fea 为 rank(-换手),高=低换手;描述要求"换手率上升=关注度
# 提升,排名高",故对 (1.0 - to) 取差分。修复前对 to 取 diff = 换手率下降排前,
# 与描述相反(2026-08-05)。
to = ctx.load_factor("turnover_20")
mom = (1.0 - to).groupby(level="Code").transform(lambda s: s.diff(20))
return cross_sectional_rank(mom)
```

**意义**：换手率因子的时序变化捕捉了市场关注度的边际变化。

**依赖数据**：`__factors__`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_extended.py`

##### coupling_lowrisk_momentum_60

**定义**：低风险趋势因子：momentum_60 × downside_frequency_60 截面排名（中期动量+下行频率低排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_60")
ds = ctx.load_factor("downside_frequency_60").reindex(mom.index)
return cross_sectional_rank(mom * ds)
```

**意义**：趋势质量的另一个维度:60 日动量相同的股票,负收益日占比低者=上涨由连续的正收益构成(台阶式上行),而非大涨大跌的脉冲。与 defensive_momentum_combo_60(低beta×动量×低溃疡)区分:本因子直接用下行频率度量趋势的'流畅度'。

**依赖数据**：`__factors__`、`momentum_60`、`downside_frequency_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_ma_chip_resonance

**定义**：均线多头排列 × 筹码集中(cr3)。均线完美+筹码集中=主力高度控盘的上升趋势。

**公式（计算逻辑）**：

```python
ma = ctx.load_factor("ma_alignment_score")
cr3 = ctx.load_factor("chip_cr3_factor")
resonance = ma * cr3
return cross_sectional_rank(resonance)
```

**意义**：均线多头排列=趋势完美，筹码高度集中=主力控盘——两者叠加是主力控盘拉升的最强技术形态。

**依赖数据**：`__factors__`、`ma_alignment_score`、`chip_cr3_factor` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_macd_chip_divergence

**定义**：MACD趋势强度 vs 筹码集中度(cr3)背离。MACD强+筹码散=虚涨，MACD弱+筹码集中=蓄力。

**公式（计算逻辑）**：

```python
macd = ctx.load_factor("macd_trend_strength")
cr3 = ctx.load_factor("chip_cr3_factor")
divergence = macd - cr3
return cross_sectional_rank(divergence)
```

**意义**：MACD走强但筹码分散=技术面虚涨、缺乏主力支撑；MACD走弱但筹码高度集中=主力控盘洗盘。

**依赖数据**：`__factors__`、`macd_trend_strength`、`chip_cr3_factor` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_macd_value_resonance

**定义**：MACD趋势强度 × 价值(bp)共振。MACD走强+低估值=技术面与基本面双重确认。

**公式（计算逻辑）**：

```python
macd = ctx.load_factor("macd_trend_strength")
bp = ctx.load_factor("bp")
resonance = macd * bp
return cross_sectional_rank(resonance)
```

**意义**：MACD走强=技术面趋势向上，bp高=基本面低估——两者同时成立时，是便宜且开始涨的经典共振信号。

**依赖数据**：`__factors__`、`macd_trend_strength`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_margin_buy_persist_10

**定义**：融资买入意愿跨期自共振：margin_buy_pressure_t × margin_buy_pressure_t-10 截面排名（买入意愿持续强于偿还排前）。

**公式（计算逻辑）**：

```python
mbp = ctx.load_factor("margin_buy_pressure")
return cross_sectional_rank(mbp * _lag(mbp, 10))
```

**意义**：融资买入意愿(买入/偿还比)跨 10 日持续 > 0.5 = 多头杠杆资金在稳定加仓,而非一日冲高后的回落。持续性买入意愿的确认度高于单日读数。

**依赖数据**：`__factors__`、`margin_buy_pressure` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_margin_buy_trend_20

**定义**：融资买入确认趋势：margin_buy_pressure × momentum_20 截面排名（买入意愿强+动量排前）。

**公式（计算逻辑）**：

```python
mbp = ctx.load_factor("margin_buy_pressure")
mom = ctx.load_factor("momentum_20").reindex(mbp.index)
return cross_sectional_rank(mbp * mom)
```

**意义**：杠杆资金参与度确认的趋势:融资买入/偿还比高=多头杠杆资金在主动加仓,动量叠加杠杆买入=两类资金合力;动量高但杠杆买入意愿弱=散户行情。⚠️ 以 margin 因子索引 reindex 防并集。

**依赖数据**：`__factors__`、`margin_buy_pressure`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_margin_chip_cost_20

**定义**：融资筹码成本复合因子：margin_chip_cost_gap×momentum_20×drawdown_60截面排名（杠杆结构健康的中期趋势排前）。

**公式（计算逻辑）**：

```python
gap = ctx.load_factor("margin_chip_cost_gap")
mom = ctx.load_factor("momentum_20").reindex(gap.index)
dd = ctx.load_factor("drawdown_60").reindex(gap.index)
return cross_sectional_rank(gap * mom * dd)
```

**意义**：融资盘建仓成本不高于市场筹码平均成本(未高位接盘,止损踩踏风险低)×中期动量×趋势完整(回撤浅)=杠杆结构健康的中期上行趋势。三因子共振过滤「杠杆资金高位接盘后的动量陷阱」。⚠️ 以 margin 因子索引reindex 其余因子防索引并集。

**依赖数据**：`__factors__`、`margin_chip_cost_gap`、`momentum_20`、`drawdown_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra3.py`

##### coupling_margin_lead_trend_60

**定义**：杠杆资金领先趋势：margin_net_flow_ratio_t-60 × momentum_60_t 截面排名（60日前融资净流入+中期动量排前）。

**公式（计算逻辑）**：

```python
mg = ctx.load_factor("margin_net_flow_ratio")
mom = ctx.load_factor("momentum_60").reindex(mg.index)
return cross_sectional_rank(_lag(mg, 60) * mom)
```

**意义**：最大滞后(60 天)的领先-滞后耦合:融资资金在 60 天前即开始净流入的中期趋势=杠杆资金的长线建仓行为,趋势级别高于短期资金推动。滞后 60 天恰好落在时间窗口平移上限。⚠️ 以 margin 因子索引 reindex 防并集。

**依赖数据**：`__factors__`、`margin_net_flow_ratio`、`momentum_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_min_align_momentum_20

**定义**：分钟均线确认趋势：momentum_20 × min_ma_alignment_frac_20 截面排名（趋势+分钟均线多头排列占比高排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
align = ctx.load_factor("min_ma_alignment_frac_20").reindex(mom.index)
return cross_sectional_rank(mom * align)
```

**意义**：日内趋势结构稳固的日线趋势:20 日分钟均线多头排列占比高(ma5>ma10>ma20>ma30 的分钟占比均值高)=日内买盘持续占优,日线动量由日内结构支撑;日线动量高但分钟排列差=尾盘偷袭/脉冲行情。Class 4 分钟指标基因首次与日线动量耦合。

**依赖数据**：`__factors__`、`momentum_20`、`min_ma_alignment_frac_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_momentum_drift_20

**定义**：动量时间加速度：momentum_20_t − momentum_20_t-20 截面排名（动量排名20日抬升=趋势加速排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(mom - _lag(mom, 20))
```

**意义**：动量的动量:20 日动量截面排名相对 20 天前的漂移量。排名抬升=趋势在加速(新资金推动短周期走强),排名回落=动能衰竭的前兆。水平动量(动量_20 本身)与加速度解耦——加速度能更早捕捉拐点,避免在动量顶点追高。

**依赖数据**：`__factors__`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_moneyflow_lead_momentum_10

**定义**：资金领先动量：mf_net_inflow_ratio_t-10 × momentum_20_t 截面排名（10日前主力净流入+当前动量排前）。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_ratio")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(_lag(mf, 10) * mom)
```

**意义**：领先-滞后耦合:10 天前的主力净流入是机构提前布局的证据,当前动量是布局后的价格兑现。资金先于价格(吸筹→拉升),故用 t-10 的资金确认 t 的价格趋势,比同日耦合(资金与价格同向共振)多一层因果时序,过滤'拉升中才追入'的跟风盘。

**依赖数据**：`__factors__`、`mf_net_inflow_ratio`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_net_turnover_momentum_20

**定义**：净换手动量共振因子：net_turnover_rate_20×momentum_20截面排名（净买入驱动的动量排前）。

**公式（计算逻辑）**：

```python
ntr = ctx.load_factor("net_turnover_rate_20")
mom = ctx.load_factor("momentum_20").reindex(ntr.index)
return cross_sectional_rank(ntr * mom)
```

**意义**：净换手率(主动净买量/自由流通股本)与动量共振:净买入驱动的上涨=资金真实承接(区别于对倒/缩量假涨);净卖出中的上涨=出货嫌疑。与 momentum_volume_resonance_20(总量能)区分:本因子用主动买卖的净量口径,剔除被动成交噪声。

**依赖数据**：`__factors__`、`net_turnover_rate_20`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra3.py`

##### coupling_quality_momentum_20d

**定义**：盈利质量因子(sp_ttm+bp rank)20日动量。质量改善=排名高。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
sp = ctx.load_factor("sp_ttm")
quality = sp.groupby(level="Date").rank(pct=True) + bp.groupby(level="Date").rank(pct=True)
mom = quality.groupby(level="Code").transform(lambda s: s.diff(20))
return cross_sectional_rank(mom)
```

**意义**：用bp和sp_ttm合成质量代理，然后做20日动量。质量因子持续改善=盈利能力提升。

**依赖数据**：`__factors__`、`bp`、`sp_ttm` ｜ **Class**：5 ｜ **源码**：`factors/coupling_extended.py`

##### coupling_quality_trend_60

**定义**：高质量趋势因子：momentum_60 × sortino_ratio_60 截面排名（中期动量+Sortino高排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_60")
so = ctx.load_factor("sortino_ratio_60").reindex(mom.index)
return cross_sectional_rank(mom * so)
```

**意义**：风险调整后的趋势:Sortino 比率(均收益/下行标准差)高的 60 日动量=上涨质量高(下行风险小)。与 coupling_lowrisk_momentum_60(下行频率)互补:本因子加权下行幅度,对'少而大的下跌'更敏感。

**依赖数据**：`__factors__`、`momentum_60`、`sortino_ratio_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_rsi_moneyflow_resonance

**定义**：RSI趋势 × 主力资金净流入。RSI走强+主力流入=技术和资金双重看多。

**公式（计算逻辑）**：

```python
rsi_trend = ctx.load_factor("rsi_intraday_trend")
mf = ctx.load_factor("mf_net_inflow_ratio")
resonance = rsi_trend * mf
return cross_sectional_rank(resonance)
```

**意义**：RSI日内走高=当日买盘持续强于卖盘，主力净流入=大资金在买入——技术面和资金面双重确认。

**依赖数据**：`__factors__`、`rsi_intraday_trend`、`mf_net_inflow_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_rsi_turnover_divergence

**定义**：RSI超卖 × 换手率(turnover_20)。超卖+高换手=恐慌抛售中的抄底机会。

**公式（计算逻辑）**：

```python
rsi = ctx.load_factor("rsi_14_excess")
to = ctx.load_factor("turnover_20")
high_to = 1.0 - to
signal = rsi * high_to
return cross_sectional_rank(signal)
```

**意义**：超卖+高换手=恐慌性抛售——散户在恐慌中割肉，机构在低位接盘。

**依赖数据**：`__factors__`、`rsi_14_excess`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_rsi_value_combo

**定义**：RSI超卖(低RSI=排前) × 低估值(bp)。超卖+低估=抄底双信号。

**公式（计算逻辑）**：

```python
rsi = ctx.load_factor("rsi_14_excess")
bp = ctx.load_factor("bp")
combo = rsi * bp
return cross_sectional_rank(combo)
```

**意义**：RSI超卖=短期技术面超跌，bp高=基本面低估——两者同时出现时，是又便宜又超跌的双重抄底信号。

**依赖数据**：`__factors__`、`rsi_14_excess`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_indicator_cross.py`

##### coupling_smallcap_value_combo

**定义**：小市值(log_total_mv取反=小盘高排)+低估值(bp)。小盘价值股效应。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
size = ctx.load_factor("log_total_mv")
combo = bp * size
return cross_sectional_rank(combo)
```

**意义**：小盘价值股在全球市场均有显著溢价。在A股，小盘+低估值组合在牛市和震荡市中表现突出。

**依赖数据**：`__factors__`、`bp`、`log_total_mv` ｜ **Class**：5 ｜ **源码**：`factors/coupling_extended.py`

##### coupling_smartmoney_lead_momentum_10

**定义**：聪明钱领先动量：smart_money_share_t-10 × momentum_20_t 截面排名（10日前信息型交易活跃+当前动量排前）。

**公式（计算逻辑）**：

```python
sms = ctx.load_factor("smart_money_share")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(_lag(sms, 10) * mom)
```

**意义**：信息型交易(聪明钱)的活跃度领先于价格趋势:10 天前异动分钟成交占比高的股票,当前动量更可能由知情资金驱动而非散户跟风。Class 3 盘中基因首次以滞后形式进入耦合层。

**依赖数据**：`__factors__`、`smart_money_share`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_stableflow_momentum_20

**定义**：稳定资金流趋势：momentum_20 × mf_flow_stability_20d 截面排名（趋势+主力资金连续同向排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
st = ctx.load_factor("mf_flow_stability_20d").reindex(mom.index)
return cross_sectional_rank(mom * st)
```

**意义**：资金流方向一致性的确认:主力资金 20 日连续同向占比高=资金态度稳定(持续吸筹或持续撤退),叠加动量=稳定吸筹中的趋势;资金流频繁转向(不稳定)下的动量=博弈盘推动,持续性存疑。与 moneyflow_momentum_resonance_20(净流入水平)区分:本因子用方向稳定性而非水平。

**依赖数据**：`__factors__`、`momentum_20`、`mf_flow_stability_20d` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_valuation_sentiment_divergence

**定义**：估值(bp)与情绪(turnover_20)的背离。低估值+高换手=价值发现，排名高。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
to = ctx.load_factor("turnover_20")
signal = bp * to
return cross_sectional_rank(signal)
```

**意义**：低估值(bp高)同时高换手意味着市场在积极交易这只低估股票——可能是价值发现过程。

**依赖数据**：`__factors__`、`bp`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_extended.py`

##### coupling_value_momentum_20d

**定义**：价值因子(bp)自身20日动量。bp值持续上升=估值优势在扩大，排名高。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
mom = bp.groupby(level="Code").transform(lambda s: s.diff(20))
return cross_sectional_rank(mom)
```

**意义**：因子值的时序动量捕捉了因子暴露的边际变化。bp持续上升=股价下跌快于账面价值——估值优势在扩大。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_extended.py`

##### coupling_value_quality_resonance

**定义**：价值(bp)+质量(sp_ttm)共振。两者同时高排名=优质低估，超级信号。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
sp = ctx.load_factor("sp_ttm")
resonance = bp * sp
return cross_sectional_rank(resonance)
```

**意义**：当价值因子和质量因子同时给出高排名时(低估+高盈利)，是A股最可靠的alpha组合。

**依赖数据**：`__factors__`、`bp`、`sp_ttm` ｜ **Class**：5 ｜ **源码**：`factors/coupling_extended.py`

##### coupling_volterm_momentum_60

**定义**：平缓期限结构趋势：momentum_60 × rv_term_structure_slope 截面排名（中期动量+短期波动不陡峭排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_60")
slope = ctx.load_factor("rv_term_structure_slope").reindex(mom.index)
return cross_sectional_rank(mom * slope)
```

**意义**：波动率期限结构(5min RV/60min RV)平缓=短期波动未放大=趋势未被噪声扰动,中期动量更'干净';期限结构陡峭(短期波动高)=筹码高度分歧,趋势随时被日内噪声打断。高 rv_term_structure_slope 已是低陡峭编码,直接相乘。

**依赖数据**：`__factors__`、`momentum_60`、`rv_term_structure_slope` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_volume_lead_momentum_5

**定义**：量能领先动量：volume_momentum_5_t-5 × momentum_20_t 截面排名（5日前量能扩张+当前动量排前）。

**公式（计算逻辑）**：

```python
vm = ctx.load_factor("volume_momentum_5")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(_lag(vm, 5) * mom)
```

**意义**：量在价先:5 天前的量能扩张先行,当前 20 日动量是量能推动的价格表现。与 momentum_volume_resonance_20(同日量价共振)区分:本因子要求量能提前确认,排除'当日才放量'的脉冲行情,捕捉量能持续推动的趋势。

**依赖数据**：`__factors__`、`volume_momentum_5`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_time.py`

##### coupling_vp_amfade_rev_20

**定义**：早盘恐慌反转因子：vp_expand_down_am_share×short_term_reversal_5截面排名（早盘恐慌释放的超跌排前）。

**公式（计算逻辑）**：

```python
am = ctx.load_factor("vp_expand_down_am_share")
rev = ctx.load_factor("short_term_reversal_5").reindex(am.index)
return cross_sectional_rank(am * rev)
```

**意义**：放量下跌集中在早盘(恐慌开盘释放)+5日超跌的共振:早盘恐慌杀跌是A股经典「诱空」形态(洗盘式砸盘),叠加超跌=恐慌抛压接近枯竭,午后/次日修复概率大;尾盘放量下跌(份额低)则相反=出货延续。与 vp_expand_down_share(恐慌总量)区分:本因子锚定恐慌的时段归属与超跌状态,是诱空识别的完整买侧信号。

**依赖数据**：`__factors__`、`vp_expand_down_am_share`、`short_term_reversal_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_vp_chips.py`

##### coupling_vp_chip_consistency_20

**定义**：量价筹码共振因子：vp_consistency_20×chip_win_peak_frac截面排名（量价健康且筹码锁筹排前）。

**公式（计算逻辑）**：

```python
vp = ctx.load_factor("vp_consistency_20")
chip = ctx.load_factor("chip_win_peak_frac").reindex(vp.index)
return cross_sectional_rank(vp * chip)
```

**意义**：分钟级量价一致性(涨有量跌无量)与获利筹码集中度(主力成本密集)的共振:两者同高=主力控盘+筹码锁定的健康趋势股(拉抬无抛压);量价一致但筹码分散=浮筹多,涨时兑现压力大。筹码维度把「量价健康」从市场行为升级为筹码结构确认,是吸筹完成后的典型状态。

**依赖数据**：`__factors__`、`vp_consistency_20`、`chip_win_peak_frac` ｜ **Class**：5 ｜ **源码**：`factors/coupling_vp_chips.py`

##### coupling_vp_expand_up_mom_20

**定义**：放量上涨动量确认因子：vp_expand_up_share×momentum_20截面排名（放量且动量确认排前）。

**公式（计算逻辑）**：

```python
vp = ctx.load_factor("vp_expand_up_share")
mom = ctx.load_factor("momentum_20").reindex(vp.index)
return cross_sectional_rank(vp * mom)
```

**意义**：日内放量上涨的量占比与20日动量的共振:放量上涨+动量向上=上涨有量能与趋势双重确认(有效上涨,可持续);放量上涨但动量停滞=放量滞涨(对倒/出货嫌疑)。与 momentum_volume_resonance_20(日频量比×动量)区分:本因子用分钟级四象限的放量上涨占比,日内结构更细。

**依赖数据**：`__factors__`、`vp_expand_up_share`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_vp_chips.py`

##### coupling_vp_lowpos_accumulate_20

**定义**：低位放量吸筹确认因子：vp_expand_price_pos×chip_win_peak_frac×mf_net_inflow_ratio截面排名（三因子吸筹确认排前）。

**公式（计算逻辑）**：

```python
pos = ctx.load_factor("vp_expand_price_pos")
chip = ctx.load_factor("chip_win_peak_frac").reindex(pos.index)
mf = ctx.load_factor("mf_net_inflow_ratio").reindex(pos.index)
return cross_sectional_rank(pos * chip * mf)
```

**意义**：放量发生在日内低位(吸筹承接)×获利筹码集中(成本峰形成)×主力资金净流入的三重吸筹确认:放量位置、筹码结构、资金方向三个独立信号同时指向「主力在低位吸筹」,任一单信号都易被洗盘/对倒混淆,三者共振才是吸筹完成的高置信信号。与 smart_capital_liftoff_20(资金×动量×突破)区分:本因子锚定日内低位,是左侧吸筹视角。

**依赖数据**：`__factors__`、`vp_expand_price_pos`、`chip_win_peak_frac`、`mf_net_inflow_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling_vp_chips.py`

##### coupling_vp_retvol_mom_20

**定义**：量价同步动量因子：minute_ret_vol_corr×momentum_20截面排名（微观量价同步且趋势向上排前）。

**公式（计算逻辑）**：

```python
corr = ctx.load_factor("minute_ret_vol_corr")
mom = ctx.load_factor("momentum_20").reindex(corr.index)
return cross_sectional_rank(corr * mom)
```

**意义**：日内分钟收益-量相关(微观量价同步度)与20日动量的共振:分钟级涨放量跌缩量+动量向上=趋势的微观基础健康(资金在每个价位都真实承接);动量向上但分钟量价脱钩(相关≈0/负)=对倒拉升,趋势脆弱。与 turnover_ret_corr_20(日频换手×收益)区分:本因子用日内分钟样本,捕捉盘中即时响应而非跨日窗口。

**依赖数据**：`__factors__`、`minute_ret_vol_corr`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_vp_chips.py`

##### coupling_vp_shrink_down_rev_5

**定义**：缩量下跌反转确认因子：vp_shrink_down_share×short_term_reversal_5截面排名（洗盘缩量回调的超跌排前）。

**公式（计算逻辑）**：

```python
vp = ctx.load_factor("vp_shrink_down_share")
rev = ctx.load_factor("short_term_reversal_5").reindex(vp.index)
return cross_sectional_rank(vp * rev)
```

**意义**：缩量下跌(抛压轻/洗盘特征)与5日超跌的共振:两者同高=洗盘式回调后的超跌(浮筹清洗完毕、卖盘枯竭,反转概率大);缩量下跌但不超跌=正常回调未到买点。与 panic_selling_ratio_60(放量恐慌,负向逻辑)互为镜像——本因子捕捉「跌无量」的洗盘侧信号,是诱空识别的买侧视角。

**依赖数据**：`__factors__`、`vp_shrink_down_share`、`short_term_reversal_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_vp_chips.py`

##### coupling_winner_bigflow_20

**定义**：获利盘大单共振因子：(1−winner_rate)×mf_net_inflow_ratio截面排名（获利盘多且主力净买入排前）。

**公式（计算逻辑）**：

```python
win = ctx.load_factor("winner_rate")
mf = ctx.load_factor("mf_net_inflow_ratio").reindex(win.index)
return cross_sectional_rank((1.0 - win) * mf)
```

**意义**：获利盘占比抬升(筹码在涨,散户跟风)+大单持续净买(主力拉升)=健康上行的正反馈;大单流出但获利盘上升(诱多)与两者同步(健康上行)在此分离。winner_rate .fea 高=低获利盘,须 (1.0−X) 翻回(8.11 二次取反先例)。

**依赖数据**：`__factors__`、`winner_rate`、`mf_net_inflow_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra3.py`

##### deep_value_reversal_combo_60

**定义**：三重左侧复合因子：bp×(1−drawdown_120)×short_term_reversal_5截面排名。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
dd = ctx.load_factor("drawdown_120")
rev = ctx.load_factor("short_term_reversal_5")
return cross_sectional_rank(bp * (1.0 - dd) * rev)
```

**意义**：低估(bp高)+深度回撤(半年尺度)+短期超跌=价值/回撤/反转三个维度的左侧共振——深度回撤压制情绪、低估提供安全边际、短期超跌提供弹性,三条件同时满足的股票是超跌价值修复的最佳候选。

**依赖数据**：`__factors__`、`bp`、`drawdown_120`、`short_term_reversal_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### defensive_momentum_combo_60

**定义**：防御动量复合因子：beta_60×momentum_60×ulcer_index_20截面排名。

**公式（计算逻辑）**：

```python
beta = ctx.load_factor("beta_60")
mom = ctx.load_factor("momentum_60")
ulcer = ctx.load_factor("ulcer_index_20")
return cross_sectional_rank(beta * mom * ulcer)
```

**意义**：低beta(市场敏感度低)+中期动量+低溃疡(回撤浅而短)=「不靠市场也能涨」的防御型趋势——低beta过滤系统性行情依赖,溃疡指数约束回撤体验,动量确认趋势。是低beta异象与动量异象的稳健交集。

**依赖数据**：`__factors__`、`beta_60`、`momentum_60`、`ulcer_index_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### dv_momentum_combo_20

**定义**：高股息×动量：dv_ttm_rank×momentum_20。高股息+走强=股息资金与趋势资金共振。

**公式（计算逻辑）**：

```python
dv = ctx.load_factor("dv_ttm_rank")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(dv * mom)
```

**意义**：高股息股票提供下行保护(股息现金流+估值锚),叠加动量确认后攻守兼备——股息资金(险资/固收替代)与趋势资金形成双买盘,回撤更小、持有体验更好。

**依赖数据**：`__factors__`、`dv_ttm_rank`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### event_momentum_divergence_20

**定义**：事件-动量背离因子：limit_up_event_5−momentum_20截面排名（事件强动量弱排前）。

**公式（计算逻辑）**：

```python
ev = ctx.load_factor("limit_up_event_5")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(ev - mom)
```

**意义**：涨停事件刚发生但20日动量尚未跟上=行情刚启动(事件领先于趋势)；动量已高但事件衰减=趋势中后段。背离项捕捉事件驱动的早期阶段。

**依赖数据**：`__factors__`、`limit_up_event_5`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily.py`

##### factor_consistency_ratio

**定义**：因子一致性比率因子，BP因子60日变化方向一致的天数占比（持续方向=高信度排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
bp_delta = bp.groupby(level="Code").diff(1)

def _consistency(s: pd.Series) -> pd.Series:
    direction = np.sign(s)
    net_direction = direction.rolling(60, min_periods=30).sum()
    count = direction.abs().rolling(60, min_periods=30).sum()
    return safe_divide(net_direction.abs(), count + 1e-8)

consistency = bp_delta.groupby(level="Code").transform(_consistency)
return cross_sectional_rank(consistency)
```

**意义**：因子变化方向的持续性是因子信度的度量

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### factor_consistency_score

**定义**：因子一致性因子，bp 60日在极端分位(>0.8或<0.2)的占比截面排名（持续极端=信号强烈排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
rank = _rank(bp)
is_extreme = ((rank > 0.8) | (rank < 0.2)).astype(float)
consistency = is_extreme.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
return cross_sectional_rank(consistency)
```

**意义**：因子持续处于极端分位意味着该股票在该因子维度上有稳定的特征——而非偶尔极端。持续在极端分位的股票最具因子特征的代表性，是因子策略最核心的标的。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_crowding_warning

**定义**：因子拥挤预警因子，-(bp排名60日中同方向占比>80%)截面排名（拥挤=一致预期风险排后）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
rank = _rank(bp)
is_high = (rank > 0.8).astype(float)
is_low = (rank < 0.2).astype(float)

high_pct = is_high.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
low_pct = is_low.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)

crowding = np.maximum(high_pct, low_pct)
return cross_sectional_rank(-crowding)
```

**意义**：当因子排名持续处于同一方向(>80%的时间在顶部或底部)，意味着该因子的拥挤度极高——过多资金在追逐同一个因子信号，反转风险在累积。因子拥挤是量化策略最大的尾部风险。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_cycle_position

**定义**：因子周期位置因子，bp偏离2年均值的符号×(偏离持续的月数)截面排名（正偏离+持续长=周期高位排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
ma_500 = bp.groupby(level="Code").transform(
    lambda s: s.rolling(500, min_periods=120).mean()
)
deviation = bp - ma_500

# Count consecutive periods of same-sign deviation
sign = np.sign(deviation)
def _consecutive(s):
    result = pd.Series(0, index=s.index)
    cnt = 0
    prev = 0
    for i, v in enumerate(s.values):
        if np.isnan(v):
            result.iloc[i] = np.nan
            continue
        if v == prev and v != 0:
            cnt += 1
        else:
            cnt = 1
        prev = v if v != 0 else prev
        result.iloc[i] = cnt
    return result

consecutive = sign.groupby(level="Code").transform(_consecutive)
cycle = sign * consecutive

return cross_sectional_rank(cycle)
```

**意义**：因子值在自身历史周期中的位置是因子择时的核心——偏离历史均值的符号和持续时间共同决定了当前处于周期的什么阶段。周期分析可以帮助避免在因子周期顶部买入。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_drawdown_60_deep

**定义**：因子滚动回撤因子，-(bp 60日滚动最大回撤)截面排名（深度回撤=因子失效风险排后）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
peak = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).max())
dd = (bp / peak.replace(0, np.nan)) - 1.0
return cross_sectional_rank(dd)
```

**意义**：因子值的滚动回撤衡量因子本身的'表现'——因子值持续下跌意味着该股票在持续失去该因子特征。深度的因子回撤可能意味着基本面发生了转折性变化。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_ic_ir_proxy_60

**定义**：因子IC IR代理因子，bp 60日(均值/std)×sqrt(60)截面排名（高信息比率=因子有效性强排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
ma60 = bp.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean()
)
std60 = bp.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).std()
)
ir = (ma60.abs() / std60.replace(0, np.nan)) * np.sqrt(60)
return cross_sectional_rank(ir)
```

**意义**：因子自身的信噪比(均值/std)乘以sqrt(N)是IC IR(信息比率)的代理——衡量因子信号相对于噪音的强度。高IC IR的因子信号更可靠，是因子权重配置的核心依据。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_mean_reversion_20

**定义**：因子均值回复因子，-(bp偏离20日均值的标准差倍数)截面排名（取负向=过度偏离=回复压力排后）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
ma20 = bp.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
std20 = bp.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
z = (bp - ma20) / std20.replace(0, np.nan)
return cross_sectional_rank(-z.abs())
```

**意义**：因子值对短期均值的偏离具有均值回复特征——偏离过大的股票在因子维度上'超买'或'超卖'。因子均值回复是因子择时的重要信号。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_momentum_decay

**定义**：因子动量衰减因子，bp的5日动量/20日动量截面排名（衰减=短期弱于长期=动能减弱排后）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
mom5 = _momentum(bp, 5)
mom20 = _momentum(bp, 20)
common = mom5.index.intersection(mom20.index)
decay = mom5.loc[common] / mom20.loc[common].replace(0, np.nan)
return cross_sectional_rank(decay)
```

**意义**：因子短期动量与长期动量的比值反映因子趋势的'健康度'——短期动量<长期动量意味着趋势在减速(衰减)，可能即将反转。比值稳定在1附近则趋势健康。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_multi_horizon_momentum

**定义**：多周期因子动量因子，(bp 5日动量排名+bp 20日动量排名+bp 60日动量排名)/3截面排名。多周期共振。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
mom5 = _momentum(bp, 5)
mom20 = _momentum(bp, 20)
mom60 = _momentum(bp, 60)

common = mom5.index.intersection(mom20.index).intersection(mom60.index)
composite = (
    _rank(mom5.loc[common]) + _rank(mom20.loc[common]) + _rank(mom60.loc[common])
) / 3.0

return cross_sectional_rank(composite)
```

**意义**：多周期因子动量的共振比单一周期更可靠——短中长三个周期的动量方向一致时，因子趋势最为确定。多周期共振可以过滤掉短期噪音和虚假反转。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_profile_shift_deep

**定义**：因子轮廓位移因子，bp 20日前排名与当前排名的均方差截面排名（位移大=剧烈变化排后）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
rank = _rank(bp)
rank_20 = rank.groupby(level="Code").shift(20)
shift = (rank - rank_20).abs()
return cross_sectional_rank(-shift)
```

**意义**：因子截面排名的位移(profile shift)反映因子结构是否在发生根本性变化——突然的大幅位移意味着因子与股票的关系在重构，历史规律可能不再适用。稳定的因子轮廓意味着稳定的alpha预期。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_rolling_drawdown_60

**定义**：因子滚动回撤因子，BP因子从60日高点的回撤程度（回撤大=因子超跌排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")

def _rolling_dd(s: pd.Series) -> pd.Series:
    rolling_max = s.rolling(60, min_periods=30).max()
    dd = (s - rolling_max) / rolling_max.replace(0, np.nan)
    return dd

bp_dd = bp.groupby(level="Code").transform(_rolling_dd)
return cross_sectional_rank(bp_dd)
```

**意义**：因子值从自身近期高点的回撤可能意味着因子被过度抛售

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### factor_signal_to_noise_60

**定义**：因子信噪比因子，bp的60日均值/std截面排名（高信噪比=因子信号清晰排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
ma60 = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).mean())
std60 = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).std())
snr = ma60.abs() / std60.replace(0, np.nan)
return cross_sectional_rank(snr)
```

**意义**：因子的信噪比(均值/标准差)是因子质量的度量——高信噪比意味着因子的信号稳定、噪音小，低信噪比的因子信号可能只是随机波动。选择高信噪比因子是量化投资的基础原则。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_trend_strength_60

**定义**：因子趋势强度因子，bp因子60日均值偏移/60日std截面排名（强趋势=因子方向可靠排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
ma60 = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).mean())
std60 = bp.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).std())
strength = (bp - ma60) / std60.replace(0, np.nan)
return cross_sectional_rank(strength)
```

**意义**：因子值的趋势强度衡量因子信号的'可信度'——因子在持续改善(如BP持续上升)比因子在某一时点的水平更具信息量。趋势强度高的因子信号更可能持续而非反转。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_turnover_ratio_20

**定义**：因子换手率因子，bp截面排名20日变化绝对值截面排名（取负向=高换手=不稳定排后）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
rank = _rank(bp)
chg = rank.groupby(level="Code").diff(20).abs()
return cross_sectional_rank(-chg)
```

**意义**：因子截面排名的变化(因子换手率)反映了因子信号的不稳定性——因子换手率过高意味着因子信号每天都在变，难以形成稳定的alpha。低因子换手率意味着信号的持续性。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### factor_volatility_regime_shift

**定义**：因子波动率状态转换因子，bp 20日std/60日std截面排名（短期波动>长期波动=状态转入高波排后）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
std20 = rolling_group_std(bp, 20)
std60 = rolling_group_std(bp, 60)
ratio = std20 / std60.replace(0, np.nan)
return cross_sectional_rank(-ratio)
```

**意义**：因子波动率的结构性变化(regime shift)对因子策略有重大影响——波动率突然放大意味着因子可能进入了新的状态，历史参数不再适用。识别波动率状态转换是动态因子配置的前提。

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### fund_flow_alpha_combo_60

**定义**：主力资金独立alpha因子：mf_net_inflow_5d×momentum_60×corr_market_60截面排名。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_5d")
mom = ctx.load_factor("momentum_60")
corr = ctx.load_factor("corr_market_60")
return cross_sectional_rank(mf * mom * corr)
```

**意义**：主力净流入+中期动量+低市场相关=「资金推动的独立行情」——低相关排除市场beta贡献(独立alpha),主力流入提供机构证据,动量确认趋势。与 beta_60(无条件暴露)互补:本因子显式要求独立性。

**依赖数据**：`__factors__`、`mf_net_inflow_5d`、`momentum_60`、`corr_market_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### fundflow_retail_inst_divergence

**定义**：主力-散户资金背离因子 (大单净买+小单净卖=机构吸筹)。

**公式（计算逻辑）**：

```python
big = ctx.load_factor("mf_big_order_ratio")
small = ctx.load_factor("mf_small_order_ratio")
big_r = _rank(big)
small_r = _rank(-small)
divergence = big_r * small_r
return cross_sectional_rank(divergence)
```

**意义**：大单(机构)买入而小单(散户)卖出是机构吸筹的清晰信号

**依赖数据**：`__factors__`、`mf_big_order_ratio`、`mf_small_order_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### fundflow_value_interaction

**定义**：资金流-价值交互因子，主力净流入排名×bp排名截面排名（资金流入+低估=戴维斯双击前兆排前）。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_ratio")
bp = ctx.load_factor("bp")

common = mf.index.intersection(bp.index)
interaction = _rank(mf.loc[common]) * _rank(bp.loc[common])
return cross_sectional_rank(interaction)
```

**意义**：主力资金的流入方向与价值的结合是最强的'聪明钱'信号——主力资金流入低估值股票意味着机构在系统性布局价值洼地，是戴维斯双击(估值修复+盈利增长)的前兆。

**依赖数据**：`__factors__`、`mf_net_inflow_ratio`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### idio_vol_momentum_combo_20

**定义**：低特质波动×动量：momentum_20×idio_vol_60。低特质波动的动量更稳。

**公式（计算逻辑）**：

```python
ivol = ctx.load_factor("idio_vol_60")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(mom * ivol)
```

**意义**：特质波动率高的股票噪音大、动量信号被干扰,且高特质波动与低收益相关(低波动异象);在低特质波动股上保留动量暴露,风险调整后收益更高。idio_vol_60 的 .fea 高=低特质波,直接作权重。

**依赖数据**：`__factors__`、`idio_vol_60`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### liftoff_pulse_combo_20

**定义**：资金脉冲起飞因子：amount_surge_count_20×momentum_10×drawdown_60截面排名。

**公式（计算逻辑）**：

```python
pulse = ctx.load_factor("amount_surge_count_20")
mom = ctx.load_factor("momentum_10")
dd = ctx.load_factor("drawdown_60")
return cross_sectional_rank(pulse * mom * dd)
```

**意义**：资金脉冲频繁(成交额异常放大反复出现)+短期动量+趋势完整=大资金反复进出的活跃票正处于启动段——脉冲是资金行为痕迹,动量确认方向,浅回撤排除高位派发。与 volume_price_liftoff_20 的突破口径互补(脉冲口径不要求突破事件)。

**依赖数据**：`__factors__`、`amount_surge_count_20`、`momentum_10`、`drawdown_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### lowvol_liftoff_combo_20

**定义**：低波量价起飞因子：parkinson_vol×volume_breakout_confirm_20×momentum_20截面排名。

**公式（计算逻辑）**：

```python
vol = ctx.load_factor("parkinson_vol")
brk = ctx.load_factor("volume_breakout_confirm_20")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(vol * brk * mom)
```

**意义**：低波动股票的放量突破=低波异象与突破信号的叠加——低波股突破的成功率高于高波股(噪音少、突破真实),放量确认+动量支持下的低波突破是最干净的趋势启动形态,与 high_quality_liquidity_combo 互补。

**依赖数据**：`__factors__`、`parkinson_vol`、`volume_breakout_confirm_20`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### lowvol_momentum_rs_20

**定义**：低波×动量×RS复合因子：idio_vol_60×momentum_20×rs_60截面排名。

**公式（计算逻辑）**：

```python
ivol = ctx.load_factor("idio_vol_60")
mom = ctx.load_factor("momentum_20")
rs = ctx.load_factor("rs_60")
return cross_sectional_rank(ivol * mom * rs)
```

**意义**：低波动异象+动量+相对强度的三重叠加——低特质波动的股票动量信号更干净(噪音少)，叠加RS确认相对优势，是风险调整后最强的趋势暴露。

**依赖数据**：`__factors__`、`idio_vol_60`、`momentum_20`、`rs_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily.py`

##### lowvol_quality_momentum_60

**定义**：低波质量动量因子：parkinson_vol×momentum_60×momentum_stability_20_60截面排名。

**公式（计算逻辑）**：

```python
vol = ctx.load_factor("parkinson_vol")
mom = ctx.load_factor("momentum_60")
acc = ctx.load_factor("momentum_stability_20_60")
return cross_sectional_rank(vol * mom * acc)
```

**意义**：低波动+中期动量+趋势加速的三重确认——低波过滤噪音、60日动量确认趋势级别、20-60日动量差确认加速方向。三重共振=「低波+加速趋势」的最强形态,是低波异象与动量异象的乘积结构。

**依赖数据**：`__factors__`、`parkinson_vol`、`momentum_60`、`momentum_stability_20_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### lowvol_trend_efficiency_combo_20

**定义**：低波×趋势效率复合因子：parkinson_vol×kama_efficiency_20截面排名（干净趋势排前）。

**公式（计算逻辑）**：

```python
vol = ctx.load_factor("parkinson_vol")
eff = ctx.load_factor("kama_efficiency_20")
return cross_sectional_rank(vol * eff)
```

**意义**：低波动(可预测、回撤浅)与高趋势效率(单边运行)共振=最干净的趋势行情——KAMA效率高但波动大=波动剧烈方向不稳定;低波但效率低=横盘整理。双条件共振过滤噪音,是低波异象与趋势跟踪的交集。

**依赖数据**：`__factors__`、`parkinson_vol`、`kama_efficiency_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra.py`

##### margin_price_resonance_20

**定义**：融资流入×动量复合因子：margin_net_flow_ratio×momentum_20截面排名（杠杆加仓且上涨排前）。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("margin_net_flow_ratio")
# ⚠️ margin .fea 仅覆盖融资标的(约1.64M行),直接相乘会做索引并集,
# 把25%缺行以NaN拉入结果 → 以 margin 索引为基准 reindex(2026-08-05修复)。
mom = ctx.load_factor("momentum_20").reindex(mf.index)
return cross_sectional_rank(mf * mom)
```

**意义**：杠杆资金净流入(融资买入>偿还)与价格动量共振=增量资金推动的趋势,真实性高于单纯动量(杠杆资金成本敏感、行为更谨慎)。融资加仓+上涨=确认行情;融资流出+上涨=存量博弈。NaN≈15%(融资融券覆盖范围)。

**依赖数据**：`__factors__`、`margin_net_flow_ratio`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra.py`

##### margin_trend_combo_20

**定义**：杠杆趋势复合因子：margin_net_flow_ratio×momentum_20×drawdown_60截面排名。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("margin_net_flow_ratio")
mom = ctx.load_factor("momentum_20").reindex(mf.index)
dd = ctx.load_factor("drawdown_60").reindex(mf.index)
return cross_sectional_rank(mf * mom * dd)
```

**意义**：融资净流入+价格动量+趋势完整的杠杆资金确认——融资加仓是杠杆资金的真金白银表态,动量确认方向,浅回撤确认趋势健康。三因子共振=杠杆推动的趋势中段最强形态。⚠️ 以 margin 因子索引为基准 reindex,避免索引并集把 25% 缺行以 NaN 拉入(.fea 仅覆盖融资标的,与既有 margin 因子同口径)。

**依赖数据**：`__factors__`、`margin_net_flow_ratio`、`momentum_20`、`drawdown_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### margin_value_combo_20

**定义**：杠杆价值复合因子：margin_net_flow_ratio×bp×log_circ_mv截面排名。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("margin_net_flow_ratio")
bp = ctx.load_factor("bp").reindex(mf.index)
size = ctx.load_factor("log_circ_mv").reindex(mf.index)
return cross_sectional_rank(mf * bp * size)
```

**意义**：融资净流入+低估+小盘=「杠杆资金抄底小盘价值」——融资盘在小盘价值股上的净流入往往是短线资金博弈与价值修复的混合信号,小盘放大弹性,低估提供安全边际。以 margin 索引为基准,与 margin_trend_combo_20 同口径。

**依赖数据**：`__factors__`、`margin_net_flow_ratio`、`bp`、`log_circ_mv` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### mf_flow_factor_momentum_20

**定义**：主力资金流因子20日动量 (资金态度变化)。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_ratio")
delta = _delta(mf, 20)
return cross_sectional_rank(delta)
```

**意义**：主力资金因子的趋势上升意味着机构态度在边际改善

**依赖数据**：`__factors__`、`mf_net_inflow_ratio` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### momentum_accel_20_60

**定义**：动量加速度：momentum_20−momentum_60。短中期动量差=趋势加速/减速。

**公式（计算逻辑）**：

```python
mom20 = ctx.load_factor("momentum_20")
mom60 = ctx.load_factor("momentum_60")
return cross_sectional_rank(mom20 - mom60)
```

**意义**：20日动量强于60日动量=趋势在加速(新资金入场推动短周期走强);20日动量弱于60日=趋势在衰减(动能枯竭的前兆)。加速度比水平更能捕捉拐点。

**依赖数据**：`__factors__`、`momentum_20`、`momentum_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### momentum_high_proximity_combo_20

**定义**：动量×52周高接近度复合因子：momentum_60×price_to_52w_high截面排名。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_60")
prox = ctx.load_factor("price_to_52w_high")
return cross_sectional_rank(mom * prox)
```

**意义**：60日动量与52周高点接近度双确认=趋势既有动能(涨幅)又有空间状态(接近新高、套牢盘出清)——动量强但离高点远=反弹未到压力位;动量强且接近新高=突破在即的最强形态(George-Hwang 52周效应)。

**依赖数据**：`__factors__`、`momentum_60`、`price_to_52w_high` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra.py`

##### momentum_liquidity_resonance_20

**定义**：动量×高流动性：momentum_20×amihud_intraday。流动性好的股票动量更可靠。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
amihud = ctx.load_factor("amihud_intraday")
return cross_sectional_rank(mom * amihud)
```

**意义**：高流动性股票的动量来自真实成交、可交易性强；低流动性股票的动量易被少量资金扭曲(虚假拉升),且交易成本侵蚀收益。amihud_intraday 的 .fea 高=高流动性,直接作权重。

**依赖数据**：`__factors__`、`momentum_20`、`amihud_intraday` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### momentum_lowvol_combo_20

**定义**：动量×低波动：momentum_20×parkinson_vol。低波动趋势股的风险调整后动量。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
vol = ctx.load_factor("parkinson_vol")
return cross_sectional_rank(mom * vol)
```

**意义**：低波动异象(低波股经风险调整后收益更高)与动量结合:低波动+强趋势=风险调整后动量最强,回撤可控。parkinson_vol 的 .fea 高=低波,直接作权重。

**依赖数据**：`__factors__`、`momentum_20`、`parkinson_vol` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### momentum_rs_resonance_20

**定义**：动量×RS共振因子：momentum_20×rs_60截面排名（绝对与相对动量双强排前）。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
rs = ctx.load_factor("rs_60")
return cross_sectional_rank(mom * rs)
```

**意义**：绝对动量(自身涨幅)与相对动量(跑赢市场)双确认=趋势既有内生动力又有相对优势——RS强但动量弱=刚启动(左侧)；动量强但RS弱=市场beta贡献(虚胖)。双强共振最可靠。

**依赖数据**：`__factors__`、`momentum_20`、`rs_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily.py`

##### momentum_stability_combo_60

**定义**：动量×稳定性复合因子：momentum_60×momentum_stability_20_60截面排名。

**公式（计算逻辑）**：

```python
mom60 = ctx.load_factor("momentum_60")
accel = ctx.load_factor("momentum_stability_20_60")
return cross_sectional_rank(mom60 * accel)
```

**意义**：60日动量叠加趋势加速度(20-60动量差)——动量强且正在加速=趋势中段最强形态；动量强但减速=趋势衰竭前兆。加速度确认后的动量暴露更安全。

**依赖数据**：`__factors__`、`momentum_60`、`momentum_stability_20_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily.py`

##### momentum_volume_divergence_20

**定义**：量价背离：momentum_20−volume_momentum_5。价升量缩(背离)排名高=缺乏资金确认。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
vol_mom = ctx.load_factor("volume_momentum_5")
return cross_sectional_rank(mom - vol_mom)
```

**意义**：价格趋势与成交量趋势背离(价升量缩/价跌量增)意味着走势缺乏真实资金支持，持续性弱、反转风险大。该因子排名高=背离显著,作为趋势的谨慎信号。

**依赖数据**：`__factors__`、`momentum_20`、`volume_momentum_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### momentum_volume_resonance_20

**定义**：动量×换手共振：momentum_20与turnover_20双高排名。放量上涨=资金确认的趋势。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
# turnover_20 的 .fea 为 rank(-换手),高=低换手,故 (1.0 - to) 翻回"高换手"
to = ctx.load_factor("turnover_20")
return cross_sectional_rank(mom * (1.0 - to))
```

**意义**：价格动量与换手率同时高=上涨由持续的交易参与推动(量价共振)；动量高但换手低=缩量上涨,趋势未获资金确认。共振项捕捉'有人气的趋势'。

**依赖数据**：`__factors__`、`momentum_20`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### moneyflow_momentum_resonance_20

**定义**：资金确认动量：mf_net_inflow_ratio×momentum_20。主力净流入+上涨=趋势有资金背书。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_ratio")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(mf * mom)
```

**意义**：上涨趋势若同时有主力资金净流入背书,则趋势由机构资金驱动而非散户跟风,延续性更强。资金与价格双确认是趋势交易的核心过滤条件。

**依赖数据**：`__factors__`、`mf_net_inflow_ratio`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### moneyflow_reversal_divergence_5

**定义**：资金×反转背离：mf_net_inflow_ratio−short_term_reversal_5。主力流入但股价超跌=潜在反转。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_ratio")
rev = ctx.load_factor("short_term_reversal_5")
return cross_sectional_rank(mf - rev)
```

**意义**：主力资金持续流入(净流入排名高)但股价仍在超跌区(反转因子低)——机构在低位吸筹而散户情绪仍悲观,是经典的左侧反转买点。背离越大信号越强。

**依赖数据**：`__factors__`、`mf_net_inflow_ratio`、`short_term_reversal_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### multi_horizon_momentum_combo_20

**定义**：多周期动量共振因子：momentum_5×momentum_10×momentum_20截面排名。

**公式（计算逻辑）**：

```python
m5 = ctx.load_factor("momentum_5")
m10 = ctx.load_factor("momentum_10")
m20 = ctx.load_factor("momentum_20")
return cross_sectional_rank(m5 * m10 * m20)
```

**意义**：5/10/20日三个周期的动量同时为正且都强=短中周期趋势方向一致(多周期共振,信号最可靠)——单一周期动量可能只是噪音,三周期同向=趋势的内外结构同步,过滤了周期错位的伪信号。

**依赖数据**：`__factors__`、`momentum_5`、`momentum_10`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### quality_liquidity_combo_20

**定义**：质量流动性三合因子：amihud_intraday×parkinson_vol×idio_vol_60截面排名。

**公式（计算逻辑）**：

```python
liq = ctx.load_factor("amihud_intraday")
vol = ctx.load_factor("parkinson_vol")
ivol = ctx.load_factor("idio_vol_60")
return cross_sectional_rank(liq * vol * ivol)
```

**意义**：高流动性×低波动×低特质波动的三重质量筛选——高流动性保证策略可交易(冲击成本低),低波动保证可预测,低特质波动排除噪音型股票。三合=最干净的可交易标的池,是 vol_liquidity_resonance_20 的特质波升级版。

**依赖数据**：`__factors__`、`amihud_intraday`、`parkinson_vol`、`idio_vol_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### reversal_liquidity_combo_5

**定义**：超跌流动性复合因子：short_term_reversal_5×amihud_intraday×turnover_20截面排名。

**公式（计算逻辑）**：

```python
rev = ctx.load_factor("short_term_reversal_5")
liq = ctx.load_factor("amihud_intraday")
to = ctx.load_factor("turnover_20")
return cross_sectional_rank(rev * liq * to)
```

**意义**：短期超跌+高流动性+低换手=「可交易的缩量超跌」——超跌提供反弹弹性,高流动性保证实际可买入(超跌但流动性枯竭的股票无法交易),低换手说明抛压衰竭(缩量超跌=跌无可跌)。三个条件过滤出最具操作性的反弹候选。

**依赖数据**：`__factors__`、`short_term_reversal_5`、`amihud_intraday`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### reversal_oversold_combo_5

**定义**：超跌×超卖复合因子：short_term_reversal_5×williams_r_14截面排名（超跌且超卖排前）。

**公式（计算逻辑）**：

```python
rev = ctx.load_factor("short_term_reversal_5")
wr = ctx.load_factor("williams_r_14")
return cross_sectional_rank(rev * wr)
```

**意义**：短期超跌(近5日回落)与威廉%R超卖(价格贴近14日低区)双条件=时间维度与空间维度的超卖共振——单一条件可能只是普通回调,双条件同时满足时反弹的概率与弹性最大(短线反弹捕捉研报逻辑)。

**依赖数据**：`__factors__`、`short_term_reversal_5`、`williams_r_14` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra.py`

##### reversal_turnover_resonance_5

**定义**：高换手反转：short_term_reversal_5×turnover_20。高换手的超跌股反转概率更高。

**公式（计算逻辑）**：

```python
rev = ctx.load_factor("short_term_reversal_5")
# turnover_20 的 .fea 高=低换手, (1.0 - to) 翻回"高换手"
to = ctx.load_factor("turnover_20")
return cross_sectional_rank(rev * (1.0 - to))
```

**意义**：短期反转效应在高换手股票上显著更强(高换手=投资者情绪驱动、过度反应更极端)。超跌(反转因子高)×高换手=博弈资金开始回补,反转行情启动概率高。

**依赖数据**：`__factors__`、`short_term_reversal_5`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### rs_value_divergence_20

**定义**：RS-估值背离因子：rs_60−bp截面排名（强RS但高估值排前，谨慎信号）。

**公式（计算逻辑）**：

```python
rs = ctx.load_factor("rs_60")
bp = ctx.load_factor("bp")
return cross_sectional_rank(rs - bp)
```

**意义**：相对强度与估值的背离——RS强但bp低(估值贵)=趋势透支基本面的风险暴露；RS弱但bp高(便宜)=超跌价值股的潜在修复。排名高=趋势与估值脱节(谨慎)。

**依赖数据**：`__factors__`、`rs_60`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily.py`

##### sentiment_value_gap

**定义**：情绪-价值差因子，-(资金流因子排名-bp排名)截面排名（取负向=情绪脱离价值=风险排后）。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_ratio")
bp = ctx.load_factor("bp")

common = mf.index.intersection(bp.index)
gap = _rank(mf.loc[common]) - _rank(bp.loc[common])
return cross_sectional_rank(-gap.abs())
```

**意义**：资金情绪与基本价值的差距是'泡沫/恐慌'的度量——资金大幅流入但价值排名很低=情绪脱离基本面(泡沫风险)。资金大幅流出但价值排名很高=恐慌超卖(价值机会)。

**依赖数据**：`__factors__`、`mf_net_inflow_ratio`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling_deep.py`

##### size_momentum_combo_20

**定义**：小盘×动量：log_circ_mv×momentum_20。小盘股的动量效应更强。

**公式（计算逻辑）**：

```python
mv = ctx.load_factor("log_circ_mv")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(mv * mom)
```

**意义**：小盘股动量效应显著强于大盘股(交易者异质性更高、价格调整更慢、羊群效应更强)。log_circ_mv 的 .fea 高=小盘(已取反),直接作权重,优先在小盘上暴露动量。

**依赖数据**：`__factors__`、`log_circ_mv`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### smallcap_liftoff_combo_60

**定义**：小盘量价起飞因子：log_circ_mv×momentum_60×volume_momentum_5截面排名。

**公式（计算逻辑）**：

```python
size = ctx.load_factor("log_circ_mv")
mom = ctx.load_factor("momentum_60")
vm = ctx.load_factor("volume_momentum_5")
return cross_sectional_rank(size * mom * vm)
```

**意义**：A股小市值+量增+中期动量的经典起飞组合——小盘股弹性大,量能放大是资金进场的先行信号,60日动量确认趋势级别。小盘×量增×动量三重共振捕捉「小盘股放量启动」行情(小市值策略的动量增强版)。

**依赖数据**：`__factors__`、`log_circ_mv`、`momentum_60`、`volume_momentum_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### smart_capital_liftoff_20

**定义**：主力资金起飞因子：mf_net_inflow_5d×momentum_20×volume_momentum_5截面排名。

**公式（计算逻辑）**：

```python
mf = ctx.load_factor("mf_net_inflow_5d")
mom = ctx.load_factor("momentum_20")
vm = ctx.load_factor("volume_momentum_5")
return cross_sectional_rank(mf * mom * vm)
```

**意义**：主力资金持续净流入(5日)+动量向上+成交量放大=「聪明钱推动的起飞」——主力净流入是机构行为证据,量增提供流动性配合,动量确认方向。与smart_money_momentum_combo_20(分钟级聪明钱)区分:本因子为日频主力资金口径。

**依赖数据**：`__factors__`、`mf_net_inflow_5d`、`momentum_20`、`volume_momentum_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### smart_money_momentum_combo_20

**定义**：聪明钱×动量复合因子：smart_money_net_bias×momentum_20截面排名（聪明钱净买且上涨排前）。

**公式（计算逻辑）**：

```python
bias = ctx.load_factor("smart_money_net_bias")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(bias * mom)
```

**意义**：聪明钱净方向与价格动量的共振=知情资金与市场趋势的方向一致——聪明钱净买入且动量向上=信息型资金推动的上涨(最可信);聪明钱净卖出但动量向上=散户接盘推动(危险)。用机构行为确认价格信号。

**依赖数据**：`__factors__`、`smart_money_net_bias`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra.py`

##### turnover_event_confirmation_20

**定义**：高换手×事件确认因子：(1−turnover_20)×limit_up_event_5截面排名。

**公式（计算逻辑）**：

```python
to = ctx.load_factor("turnover_20")
ev = ctx.load_factor("limit_up_event_5")
return cross_sectional_rank((1.0 - to) * ev)
```

**意义**：涨停事件配合高换手=筹码充分交换、参与者进场充分(事件有效性高)；涨停但低换手=惜售一字板(事件可持续性存疑)。turnover_20的.fea高=低换手，故用(1−to)翻回高换手。

**依赖数据**：`__factors__`、`turnover_20`、`limit_up_event_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily.py`

##### turnover_factor_momentum_20

**定义**：换手率因子20日动量 (关注度变化)。

**公式（计算逻辑）**：

```python
# turnover_20 的 .fea 为 rank(-换手),高=低换手;描述要求"关注度提升(换手率
# 上升)排前",故对 (1.0 - to) 取动量(与 coupling_liquidity_momentum_20d 同口径)。
# 修复前直接对 to 取动量 = 低换手程度增强排前,与描述相反(2026-08-05)。
to = ctx.load_factor("turnover_20")
mom = _momentum(1.0 - to, 20)
return cross_sectional_rank(mom)
```

**意义**：换手率因子趋势上升=市场关注度在持续提升，是正面信号

**依赖数据**：`__factors__`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### turnover_orthogonal_to_mv

**定义**：换手率对市值正交化因子 (剔除规模效应的纯流动性)。

**公式（计算逻辑）**：

```python
to = ctx.load_factor("turnover_20")
size = ctx.load_factor("log_circ_mv")

def _zscore(s):
    mu = s.groupby(level="Date").transform("mean")
    sg = s.groupby(level="Date").transform("std")
    return safe_divide(s - mu, sg + 1e-8)

to_z = _zscore(to)
size_z = _zscore(size)
b = (to_z * size_z).groupby(level="Date").transform("mean") / (
    (size_z ** 2).groupby(level="Date").transform("mean") + 1e-8
)
residual = to_z - b * size_z
return cross_sectional_rank(residual)
```

**意义**：小盘股天然换手率高——剔除市值影响后的换手率才是真正的流动性偏好度量

**依赖数据**：`__factors__`、`turnover_20`、`log_circ_mv` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### value_event_combo_20

**定义**：价值×事件复合因子：bp×limit_up_event_5截面排名（低估且刚涨停排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
ev = ctx.load_factor("limit_up_event_5")
return cross_sectional_rank(bp * ev)
```

**意义**：低估股票(bp高)出现涨停事件=价值发现启动(事件驱动研报的逻辑)——涨停带来关注度重估，低估提供安全边际，事件确认了催化剂。组合捕捉价值股的「戴维斯双击」启动点。

**依赖数据**：`__factors__`、`bp`、`limit_up_event_5` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily.py`

##### value_factor_zscore_252

**定义**：价值因子(BP)252日历史Z-score (相对自身历史的高估/低估)。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")

def _rolling_zscore(s, window):
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    return safe_divide(s - mean, std + 1e-8)

z = bp.groupby(level="Code").transform(lambda s: _rolling_zscore(s, 252))
return cross_sectional_rank(z)
```

**意义**：BP相对于自身历史水平处于高位时价值因子更有效——估值回复的引力更强

**依赖数据**：`__factors__`、`bp` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### value_liftoff_combo_20

**定义**：价值量价起飞因子：bp×volume_breakout_confirm_20×momentum_20截面排名。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
brk = ctx.load_factor("volume_breakout_confirm_20")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank(bp * brk * mom)
```

**意义**：低估价值股出现放量突破且动量转正=价值发现的启动点(戴维斯双击的量价版本)——bp 提供安全边际,放量突破确认资金进场,动量确认趋势方向。比双因子 value_event_combo_20(仅bp×事件)多一重动量确认。

**依赖数据**：`__factors__`、`bp`、`volume_breakout_confirm_20`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### value_reversal_combo_60

**定义**：价值×深回撤复合因子：bp×(1−drawdown_60)截面排名（低估且深度回撤排前）。

**公式（计算逻辑）**：

```python
bp = ctx.load_factor("bp")
dd = ctx.load_factor("drawdown_60")
return cross_sectional_rank(bp * (1.0 - dd))
```

**意义**：低估(bp高)叠加深度回撤(回撤深=左侧)是价值投资的经典买点组合——回撤压制了短期情绪,低估提供了安全边际,两者共振=超跌价值股的修复空间最大(极值视角选股研报:估值与回撤双极端)。

**依赖数据**：`__factors__`、`bp`、`drawdown_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra.py`

##### vol_liquidity_resonance_20

**定义**：低波×高流动性共振因子：parkinson_vol×amihud_intraday截面排名。

**公式（计算逻辑）**：

```python
vol = ctx.load_factor("parkinson_vol")
amihud = ctx.load_factor("amihud_intraday")
return cross_sectional_rank(vol * amihud)
```

**意义**：低波动(可预测)与高流动性(可交易)共振=最干净的可交易标的——低波异象的收益在高流动性股票上可实际捕获(低流动性的低波股交易成本侵蚀收益)。

**依赖数据**：`__factors__`、`parkinson_vol`、`amihud_intraday` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily.py`

##### volume_price_liftoff_20

**定义**：量价起飞因子：momentum_20×volume_breakout_confirm_20×drawdown_60截面排名。

**公式（计算逻辑）**：

```python
mom = ctx.load_factor("momentum_20")
brk = ctx.load_factor("volume_breakout_confirm_20")
dd = ctx.load_factor("drawdown_60")
return cross_sectional_rank(mom * brk * dd)
```

**意义**：A股「量价起飞」的量化刻画:价格突破(动量+接近突破位)必须有量能确认(放量突破)且趋势完整(回撤浅)——三者共振=资金推动的实质行情启动,区别于缩量假突破与深回撤后的弱反弹。三重条件过滤最严格的起飞信号。

**依赖数据**：`__factors__`、`momentum_20`、`volume_breakout_confirm_20`、`drawdown_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra2.py`

##### winner_momentum_combo_20

**定义**：获利盘×动量：winner_rate×momentum_20。获利盘多+上涨=浮盈筹码形成支撑。

**公式（计算逻辑）**：

```python
# winner_rate 的 .fea 为 rank(-获利盘),高=低获利盘;描述要求"获利盘多+上涨排前",
# 故翻回 (1.0 - win)。修复前 win*mom = 低获利盘×动量,与描述相反(2026-08-05)。
win = ctx.load_factor("winner_rate")
mom = ctx.load_factor("momentum_20")
return cross_sectional_rank((1.0 - win) * mom)
```

**意义**：获利盘占比高说明多数持仓者浮盈——上涨趋势中浮盈筹码锁仓意愿强、抛压小,形成'惜售-上涨'的正反馈;叠加动量确认后趋势的自我强化特征更明确。

**依赖数据**：`__factors__`、`winner_rate`、`momentum_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling_volume_price.py`

##### winner_rate_factor_momentum_20

**定义**：获利盘因子20日动量 (筹码结构变化方向)。

**公式（计算逻辑）**：

```python
# winner_rate 的 .fea 为 rank(-获利盘),高=低获利盘;
# delta 高 = winner_rate 因子上升 = 原始获利盘下降。描述要求"获利盘下降排前",故 rank(delta)。
wr = ctx.load_factor("winner_rate")
delta = _delta(wr, 20)
return cross_sectional_rank(delta)
```

**意义**：获利盘比例趋势下降意味着筹码在从分散到集中——主力收集筹码的迹象

**依赖数据**：`__factors__`、`winner_rate` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

##### winner_rs_combo_60

**定义**：获利盘×相对强度复合因子：winner_rate×rs_60截面排名（筹码压力小且强势排前）。

**公式（计算逻辑）**：

```python
wr = ctx.load_factor("winner_rate")
rs = ctx.load_factor("rs_60")
return cross_sectional_rank(wr * rs)
```

**意义**：获利盘占比低(上方套牢筹码出清、抛压小)且相对市场强势(资金持续流入)=反弹阻力最小与上涨动力最强的组合——套牢盘少让涨势无解套抛压,RS强确认资金认可。筹码结构与相对强弱两个正交维度共振。

**依赖数据**：`__factors__`、`winner_rate`、`rs_60` ｜ **Class**：5 ｜ **源码**：`factors/coupling_daily_extra.py`

#### <a name="cat-enhanced-c5"></a>类别 enhanced — 增强（2 个）

##### liquidity_discount_factor

**定义**：流动性折价因子，(1-bp排名)×(1-turnover_20排名)截面排名（低估值+低流动性=流动性折价排前）。

**公式（计算逻辑）**：

```python
bp = context.load_factor("bp")
turnover = context.load_factor("turnover_20")
common = bp.index.intersection(turnover.index)
bp_r = bp.loc[common].groupby(level="Date").rank(pct=True)
low_liq = (1 - turnover.loc[common].groupby(level="Date").rank(pct=True))
return cross_sectional_rank(bp_r * low_liq)
```

**意义**：低估值的低流动性股票有双重折价——估值折价+流动性折价。当流动性改善时(如被纳入指数)，流动性折价修复会带来显著的alpha。

**依赖数据**：`__factors__`、`bp`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/enhanced.py`

##### turnover_shock_20

**定义**：换手率异动因子，换手率20日Z-score截面排名（异常高换手排后）。

**公式（计算逻辑）**：

```python
df = context.load_factors(["turnover_20"])
to = df["turnover_20"]
to_mean = to.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=20).mean())
to_std = to.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=20).std())
zscore = (to - to_mean) / to_std.replace(0, np.nan)
return cross_sectional_rank(-zscore.abs())
```

**意义**：换手率突然飙升(偏离正常水平多个标准差)通常伴随事件驱动

**依赖数据**：`__factors__`、`turnover_20` ｜ **Class**：5 ｜ **源码**：`factors/coupling.py`

---

## 六、已禁用因子（非 PIT）

以下因子依赖 `stock_list.parquet`（当前快照型数据，违反 point-in-time 契约），
在 `factor_loader.ensure_builtin_factors_loaded()` 加载时被 `_filter_non_point_in_time_factors`
自动移除，**不参与任何构建与建模**。

##### industry_relative_momentum_20

**定义**：行业中性20日动量因子（个股20日动量减所属行业均值动量），截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
stock_map = _load_industry_map(context)
sector_stocks = _build_sector_stocks(stock_map)

adj = _adjusted_close(daily)
mom20 = adj.groupby(level="Code").transform(
    lambda s: s.pct_change(20, fill_method=None)
)
mom_frame = mom20.unstack("Code")

# Industry mean 20d momentum (Date × industry), then broadcast back to
# stocks and subtract: residual = stock momentum minus industry momentum.
industry_mean: dict[str, pd.Series] = {}
for ind, codes in sector_stocks.items():
    available = [c for c in codes if c in mom_frame.columns]
    if len(available) < 3:
        continue
    industry_mean[ind] = mom_frame[available].mean(axis=1)

if not industry_mean:
    return cross_sectional_rank(mom20)

ind_mean_df = pd.DataFrame(industry_mean)
stock_ind_mean = _map_sector_metric_to_stocks(ind_mean_df, sector_stocks)

if stock_ind_mean.empty:
    return cross_sectional_rank(mom20)

residual = mom20 - stock_ind_mean.reindex(mom20.index)
return cross_sectional_rank(residual)
```

**意义**：将个股动量剥离行业系统性成分后，剩余的是行业内相对强弱：同行业中动量显著强于均值的个股，在资金抱团与板块轮动中更具持续性。行业中性化同时消除了动量因子对行业风格的暴露，与 momentum_20（绝对动量）形成互补。

**依赖数据**：`daily.parquet`、`stock_list.parquet` ｜ **Class**：1 ｜ **源码**：`factors/sector_momentum.py` ｜ **已禁用（非 PIT 数据源）**

##### pb_industry_adjusted

**定义**：行业调整市净率因子，1/PB在同THS行业内截面排名（低PB行业内排前）。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
pb = finance["pb"].replace(0, np.nan).clip(lower=0.1, upper=100)
bp = 1.0 / pb

# Load industry mapping
industry_map = context.repo.load_industry_map()

codes = bp.index.get_level_values("Code")
industries = codes.map(industry_map)
df = pd.DataFrame({"bp": bp.values, "industry": industries.values}, index=bp.index)
df = df.dropna(subset=["industry"])

# Rank BP within Date + industry groups
bp_rank = df.groupby(["Date", "industry"])["bp"].rank(pct=True)
return cross_sectional_rank(bp_rank)
```

**意义**：不同行业的PB水平存在系统性差异：金融行业PB通常<1，科技行业PB>5。不做行业调整的BP因子会将金融股系统性排前，科技股系统性排后——这并非alpha信号，而是行业偏差。行业调整后的PB消除了这种偏差，提取了真正的行业内相对价值信号。

**依赖数据**：`finance.parquet`、`stock_list.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_deep.py` ｜ **已禁用（非 PIT 数据源）**

##### ps_ttm_sector_neutral

**定义**：行业中性市销率因子，(ps_ttm排名-行业均值排名)截面排名。

**公式（计算逻辑）**：

```python
fin = context.load("finance.parquet")
ps = fin["ps_ttm"].clip(0, 500)
industry_map = context.repo.load_industry_map()

codes = ps.index.get_level_values("Code")
industries = codes.map(industry_map)
df = pd.DataFrame({"ps": ps.values, "industry": industries.values}, index=ps.index)
df = df.dropna(subset=["industry"])

# Rank within industry
ps_rank = df.groupby(["Date", "industry"])["ps"].rank(pct=True)
return cross_sectional_rank(-ps_rank)
```

**意义**：P/S ratios vary dramatically by industry (tech vs utilities). Industry-neutral P/S captures within-industry relative cheapness, which is more predictive than absolute P/S level. Uses stock_list.parquet industry classification (load_industry_map).

**依赖数据**：`finance.parquet`、`stock_list.parquet` ｜ **Class**：1 ｜ **源码**：`factors/valuation_extended.py` ｜ **已禁用（非 PIT 数据源）**

##### sector_amount_momentum_5d

**定义**：板块成交额动量因子：个股所属行业平均成交额5日变化率，映射到个股后截面排名。反映资金在板块层面的流入/流出动能。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
stock_map = _load_industry_map(context)
sector_stocks = _build_sector_stocks(stock_map)

amount = daily["amount"].where(daily["amount"] > 0, np.nan)
amount_frame = amount.unstack("Code")

sector_avg_amount: dict[str, pd.Series] = {}
for ind, codes in sector_stocks.items():
    available = [c for c in codes if c in amount_frame.columns]
    if not available:
        continue
    sector_avg_amount[ind] = amount_frame[available].mean(axis=1)

if not sector_avg_amount:
    return cross_sectional_rank(
        amount.groupby(level="Code").transform(
            lambda s: s.pct_change(5, fill_method=None)
        )
    )

sector_avg_df = pd.DataFrame(sector_avg_amount)
sector_amount_mom_5 = sector_avg_df.pct_change(5, fill_method=None)
sector_amount_mom_5 = sector_amount_mom_5.replace([np.inf, -np.inf], np.nan)

stock_metric = _map_sector_metric_to_stocks(sector_amount_mom_5, sector_stocks)

if stock_metric.empty:
    return pd.Series(
        index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Code"]),
        dtype=float,
    )

return cross_sectional_rank(stock_metric)
```

**意义**：行业板块成交额的变化是机构资金调仓的代理变量。板块成交额持续放大意味着资金正在系统性流入该板块，是板块级别行情启动的重要先行指标。

**依赖数据**：`daily.parquet`、`stock_list.parquet` ｜ **Class**：1 ｜ **源码**：`factors/sector_momentum.py` ｜ **已禁用（非 PIT 数据源）**

##### sector_amount_rank

**定义**：行业内成交额占比因子，个股成交额在所属行业内的截面排名。

**公式（计算逻辑）**：

```python
daily = context.load("daily.parquet")
amount = daily["amount"].where(daily["amount"] > 0, np.nan)

stock_map = _load_industry_map(context)
sector_stocks = _build_sector_stocks(stock_map)

amt_frame = amount.unstack("Code")
rank_parts: list[pd.DataFrame] = []

for ind, codes in sector_stocks.items():
    available = [c for c in codes if c in amt_frame.columns]
    if len(available) < 3:
        continue
    sector_amt = amt_frame[available]
    sector_rank = sector_amt.rank(axis=1, pct=True)
    rank_parts.append(sector_rank)

if not rank_parts:
    return cross_sectional_rank(amount)

combined = pd.concat(rank_parts, axis=1)
combined = combined.T.groupby(level=0).mean().T
combined = stack_date_code(combined)
combined.name = "sector_amount_rank"
return cross_sectional_rank(combined)
```

**意义**：行业内成交额占比高的个股是资金关注的焦点，具有更好的流动性和价格发现效率。成交额占比持续领先的个股往往是行业的情绪龙头或机构重仓标的。

**依赖数据**：`daily.parquet`、`stock_list.parquet` ｜ **Class**：1 ｜ **源码**：`factors/sector.py` ｜ **已禁用（非 PIT 数据源）**

##### sector_mv_rank

**定义**：行业内市值占比因子，个股总市值在所属行业内的截面排名。

**公式（计算逻辑）**：

```python
finance = context.load("finance.parquet")
total_mv = finance["total_mv"].where(finance["total_mv"] > 0, np.nan)

stock_map = _load_industry_map(context)
sector_stocks = _build_sector_stocks(stock_map)

# For each industry on each date, rank stocks by market value within it
mv_frame = total_mv.unstack("Code")  # Date × Code
rank_parts: list[pd.Series] = []

for ind, codes in sector_stocks.items():
    available = [c for c in codes if c in mv_frame.columns]
    if len(available) < 3:
        continue
    sector_mv = mv_frame[available]
    sector_rank = sector_mv.rank(axis=1, pct=True)  # within-industry rank
    rank_parts.append(sector_rank)

if not rank_parts:
    return cross_sectional_rank(total_mv)

# Average within-industry rank across all industries each stock belongs to
combined = pd.concat(rank_parts, axis=1)
combined = combined.T.groupby(level=0).mean().T
combined = stack_date_code(combined)
combined.name = "sector_mv_rank"
return cross_sectional_rank(combined)
```

**意义**：行业内市值最大的公司通常是行业龙头，享有流动性溢价、机构关注度和定价权优势。行业内排名比绝对市值排名更能反映公司在细分赛道中的竞争地位。

**依赖数据**：`finance.parquet`、`stock_list.parquet` ｜ **Class**：1 ｜ **源码**：`factors/sector.py` ｜ **已禁用（非 PIT 数据源）**

---

## 附录 A：全因子速查索引（字母序）

全部 859 个注册因子，每个因子一行（“一个因子一个点”）：

- [accumulation_distribution_20](#accumulation_distribution_20)  — 量积累线斜率因子：AD线20日变化截面排名（资金净积累排前）。
- [adx_14](#adx_14)  — 14日趋势强度因子 (ADX)
- [am_close_position](#am_close_position)  — 上午收盘位置因子：上午收盘在上午振幅区间中的相对位置截面排名（上午收高排前）。
- [am_hl_range_intraday](#am_hl_range_intraday)  — 上午振幅因子，上午最高/上午最低-1截面排名（取负向=上午高振幅=分歧大排后）。
- [am_large_order_ratio](#am_large_order_ratio)  — 上午大单占比因子，上午成交额/全天成交额截面排名（上午集中放量=机构主导排前）。
- [am_macd_trend](#am_macd_trend)  — 早盘MACD趋势因子（上午MACD与时间的相关性截面排名，正相关=早盘动能积聚排前）。
- [am_momentum_intraday](#am_momentum_intraday)  — 上午动量因子：上午收盘相对上午开盘的涨跌幅截面排名（上午走强排前）。
- [am_pm_macd_ratio](#am_pm_macd_ratio)  — 早午盘MACD比率因子（上午MACD均值/下午MACD均值截面排名，比率>1=早盘强于午盘=动能衰减排后）。
- [am_pm_range_ratio](#am_pm_range_ratio)  — 上午/下午振幅比因子：上午振幅/下午振幅截面排名（负向，早盘波动主导排后）。
- [am_pm_return_ratio](#am_pm_return_ratio)  — 上午/下午收益比因子，上午收益/下午收益截面排名（上午领涨=主动买入排前）。
- [am_pm_rsi_ratio](#am_pm_rsi_ratio)  — 早午盘RSI比率因子（上午RSI均值/下午RSI均值截面排名，比率>1=早盘动量强于午盘排后）。
- [am_pm_rv_ratio](#am_pm_rv_ratio)  — 上午/下午波动率比因子，上午已实现方差/下午已实现方差截面排名。波动率的日内分布不对称性。
- [am_pm_vol_ratio](#am_pm_vol_ratio)  — 上午/下午成交量比因子截面排名。上午放量=信息消化积极，下午放量=尾盘博弈。
- [am_rsi_trend](#am_rsi_trend)  — 早盘RSI趋势因子（上午RSI与时间的相关性截面排名，正相关=早盘动量积聚排前）。
- [am_vol_share](#am_vol_share)  — 上午成交量占比因子，上午成交量/全日成交量截面排名。上午占比高=信息消化积极。
- [amihud_amt_20d](#amihud_amt_20d)  — Amihud 非流动性（|日收益|/成交额×1e8 的20日均值）。
- [amihud_asymmetry_20](#amihud_asymmetry_20)  — 涨跌日流动性不对称因子：下跌日Amihud/上涨日Amihud截面排名（负向，恐慌性难出货排后）。
- [amihud_daily_20](#amihud_daily_20)  — Amihud日频非流动性因子（20日平均|收益|/成交额，正向排名）。
- [amihud_daily_5](#amihud_daily_5)  — Amihud 非流动性（|日收益|/成交额×1e8 的 5 日均值）。
- [amihud_intraday](#amihud_intraday)  — 高频Amihud非流动性因子，5分钟|ret|/amount均值截面排名（取负向=高流动性排前）。
- [amihud_parkinson_ratio](#amihud_parkinson_ratio)  — 非流动性-波动率比因子，amihud_intraday排名/parkinson_vol排名截面排名（高冲击成本相对低波动的异常排后）。
- [amihud_trend_20_60](#amihud_trend_20_60)  — Amihud流动性趋势因子：20日Amihud/60日Amihud截面排名（负向，流动性恶化排后）。
- [amount_ratio_20](#amount_ratio_20)  — 20日相对成交额因子，amount/avg_amount_20 - 1 截面排名。
- [amount_surge_count_20](#amount_surge_count_20)  — 资金脉冲频率：20日内成交额>1.5倍20日均额的交易日占比截面排名。
- [amplitude_20](#amplitude_20)  — 20日均振幅因子，(high-low)/close 的20日均值截面排名（低振幅排前）。
- [amt_mom_accel](#amt_mom_accel)  — 成交额动量加速（3 日变化率 − 10 日变化率）。
- [amt_rel_ind](#amt_rel_ind)  — 行业内相对量能（amount 行业内 z-score）。
- [amt_rel_ind_ma5](#amt_rel_ind_ma5)  — 量能相对自身均值（amount/个股全期均值 的5日均值）。
- [amt_surge_3d](#amt_surge_3d)  — 3 日成交额冲击（当日成交额 / 3 日均额）。
- [aroon_down_25](#aroon_down_25)  — Aroon下行因子：25日窗口内距最近新低的位置取负排名（近期破位排后）。
- [aroon_up_25](#aroon_up_25)  — Aroon上行因子：25日窗口内距最近新高的天数位置截面排名（强势突破排前）。
- [atr_20](#atr_20)  — 20日平均真实波幅(ATR)因子，低ATR排前。
- [atr_position_250](#atr_position_250)  — ATR历史位置因子：ATR20相对自身250日分布的标准化偏离截面排名（负向，波动分位高排后）。
- [atr_ratio_20](#atr_ratio_20)  — ATR比率因子，ATR_20/close截面排名（高相对波幅=高风险排后）。
- [avg_cost_premium](#avg_cost_premium)  — 平均成本溢价因子，(close-weight_avg)/weight_avg截面排名（现价高于均价=多数人盈利排前）。
- [avg_price_trend_20](#avg_price_trend_20)  — 均价趋势因子：当日VWAP的20日变化率截面排名（成交均价抬升排前）。
- [beta_60](#beta_60)  — 60日市场贝塔因子（对全池等权市场收益的60日滚动beta，反向排名）。
- [bias_20](#bias_20)  — 20日均线乖离率，close/ma_20 - 1 的截面排名。
- [bias_60](#bias_60)  — 60日乖离率因子：(adj-MA60)/MA60截面排名（偏离中期均线排前）。
- [bias_signal_29_19](#bias_signal_29_19)  — BIAS金叉信号因子：29日乖离率减去其19日均线截面排名（乖离加速扩张排前）。
- [big_gap_reversal_5](#big_gap_reversal_5)  — 高开回补因子：5日前高开(>3%)事件的5日累计收益截面排名（高开后走强排前）。
- [big_order_net_accel_10](#big_order_net_accel_10)  — 大单净额加速度因子：近5日大单净额−前5日大单净额(占成交额比)截面排名（资金加速流入排前）。
- [big_range_day_freq_20](#big_range_day_freq_20)  — 大振幅日频率因子：20日振幅(复权)≥5%的天数占比截面排名（剧烈波动排前）。
- [big_vs_small_divergence_5d](#big_vs_small_divergence_5d)  — 大小单背离5日因子，(大单净流入率-小单净流入率)的5日变化截面排名。
- [bigorder_momentum_resonance_20](#bigorder_momentum_resonance_20)  — 大单×动量：mf_big_order_ratio×momentum_20。大单占比高+上涨=机构主导的行情。
- [boll_band_deviation](#boll_band_deviation)  — 布林带偏离因子（(ma5-boll_mid)/(upper-lower)截面排名，偏离大=脱离均衡排后）。
- [boll_mid_slope](#boll_mid_slope)  — 布林带中轨斜率因子（(boll_mid_close-boll_mid_open)/|boll_mid_open|截面排名，中轨上移=趋势向上排前）。
- [boll_position](#boll_position)  — 布林带位置因子（(ma5_close-boll_mid)/(boll_upper-boll_lower)截面排名，上轨附近=强势排前）。
- [boll_squeeze](#boll_squeeze)  — 布林带挤压因子（1/布林带宽截面排名，带宽极窄=挤压程度高=变盘迫近排前）。
- [boll_width_20](#boll_width_20)  — 布林带宽度因子（(upper-lower)/mid截面排名，窄带=变盘前兆排前）。
- [boll_width_5d_change](#boll_width_5d_change)  — 布林带宽5日变化因子：分钟布林带宽的5日变化截面排名（带宽扩张排前）。
- [boll_width_change](#boll_width_change)  — 布林带宽变化因子（收盘带宽-开盘带宽截面排名，带宽扩张=波动率上升排前）。
- [bollinger_position](#bollinger_position)  — 布林带位置因子，(close-下轨)/(上轨-下轨)截面排名。
- [bollinger_position_20](#bollinger_position_20)  — 20日布林带位置因子 (%B, 高位排后, 负向)
- [bollinger_squeeze](#bollinger_squeeze)  — 布林带收缩因子，-(上轨-下轨)/均价截面排名（带宽窄=挤压突破前兆排前）。
- [bollinger_width_20](#bollinger_width_20)  — 20日布林带宽度因子 (窄幅排前, 负向)
- [bp](#bp)  — 账面市值比(BP)因子，1/PB截面排名，高值代表价值股。
- [bp_factor_momentum_20](#bp_factor_momentum_20)  — BP因子20日动量 (BP因子值的变化率)。
- [bp_momentum_20](#bp_momentum_20)  — BP动量因子（BP的20日变化率截面排名，BP上升=价值增强排前）。
- [bp_momentum_combo_20](#bp_momentum_combo_20)  — 价值+动量复合：(bp+momentum_20)/2。低估且走强的股票双因子确认。
- [bp_momentum_divergence_20](#bp_momentum_divergence_20)  — 估值-趋势背离：momentum_20−bp。趋势强但估值贵的股票(估值透支)排名高。
- [bp_size_neutral](#bp_size_neutral)  — 规模中性化BP因子，市值分桶内截面排名。消除市值与估值相关性。
- [breakout_60](#breakout_60)  — 60日价格突破强度因子，close/max(high,60)-1截面排名。
- [bv_daily](#bv_daily)  — 日度双幂变差因子，1分钟数据双幂变差截面排名（低波排前）。连续价格变动的稳健波动估计。
- [cci_20](#cci_20)  — 20日CCI因子：(TP−MA20)/(0.015×MD)截面排名（超买排后，负向排名）。
- [chandelier_position](#chandelier_position)  — 吊灯止损距离因子：现价距(20日高点−3×ATR)止损线的ATR单位数截面排名（趋势健康度排前）。
- [chip_above_below_ratio](#chip_above_below_ratio)  — 筹码压力比因子，(price-cost_85pct)/(cost_15pct-price)截面排名（上方套牢>下方获利=压力大排后）。
- [chip_below_momentum](#chip_below_momentum)  — 下方筹码动量因子，chip_peak_ratio的5日变化截面排名。
- [chip_bimodality](#chip_bimodality)  — 筹码双峰因子：|主峰价−中位价|/标准差截面排名（负向，双峰分歧排后）。
- [chip_concentration](#chip_concentration)  — 筹码集中度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（取负向=窄区间排前）。
- [chip_concentration_change_20d](#chip_concentration_change_20d)  — 筹码集中度20日变化因子，-(cost_95pct-cost_5pct)/cost_50pct的20日变化截面排名（凝聚=正向排前）。
- [chip_concentration_change_5d](#chip_concentration_change_5d)  — 筹码集中度5日变化因子，concentration的5日差分截面排名。集中度收窄=吸筹，放宽=派发。
- [chip_concentration_ma5](#chip_concentration_ma5)  — 筹码成本宽度5日均值因子：(cost95−cost5)/cost50取负排名，区间越窄（越集中）排前。
- [chip_concentration_streak](#chip_concentration_streak)  — 筹码凝聚持续性因子，筹码集中度连续改善天数截面排名。
- [chip_concentration_zone](#chip_concentration_zone)  — 筹码集中区位因子，-(|winner_rate-0.5|)截面排名（获利盘50%=多空平衡=方向将出排前）。
- [chip_cost_asymmetry](#chip_cost_asymmetry)  — 成本分布不对称因子，(cost_95pct-cost_50pct)/(cost_50pct-cost_5pct)截面排名（取负向=上重下轻排后）。
- [chip_cost_convergence_20d](#chip_cost_convergence_20d)  — 筹码成本收敛=(cost_85pct-cost_5pct)/cost_50pct的20日变化取反。成本收敛=方向选择在即。
- [chip_cost_kurtosis_20d](#chip_cost_kurtosis_20d)  — 成本分布尖峰度因子 (85%-15%价差/95%-5%价差, 低值=分布尖峰排前)。
- [chip_cost_momentum_20d](#chip_cost_momentum_20d)  — 筹码成本重心趋势因子，weight_avg的20日变化率截面排名（成本上移=资金抬轿排前）。
- [chip_cost_premium_change_20](#chip_cost_premium_change_20)  — 成本溢价动量：(close-cost_50pct)/cost_50pct的20日变化截面排名。溢价率抬升=资金持续高于成本线买入。
- [chip_cost_skew](#chip_cost_skew)  — 成本分布偏度因子，(cost_50pct-cost_15pct)/(cost_85pct-cost_50pct)截面排名。
- [chip_cr3_factor](#chip_cr3_factor)  — 筹码CR3集中度因子，前三峰筹码占比截面排名。CR3高=多个价格区间筹码集中=多级支撑结构。
- [chip_cr3_momentum](#chip_cr3_momentum)  — 筹码CR3动量因子，前三峰占比的5日变化截面排名。CR3上升=筹码向多个核心锚点集中=结构性收集。
- [chip_cv_factor](#chip_cv_factor)  — 筹码变异系数因子，chip_cv=std/mean截面排名（取负=低CV排前）。低CV=分布相对均值集中=风险可控。
- [chip_cv_momentum](#chip_cv_momentum)  — 筹码CV动量因子，变异系数的5日变化截面排名（取负=CV降排前）。CV收窄=相对离散度降低=筹码趋于集中。
- [chip_deep_trap_ratio](#chip_deep_trap_ratio)  — 深套筹码占比因子：成本价>1.1×close的筹码比例截面排名（负向，深套盘多排后）。
- [chip_dispersion](#chip_dispersion)  — 筹码分布宽度因子，chip_weighted_std截面排名（取负=窄分布排前）。窄分布=筹码集中=一致预期强。
- [chip_dispersion_momentum](#chip_dispersion_momentum)  — 筹码宽度动量因子，chip_weighted_std的5日变化截面排名（取负=收窄排前）。宽度收窄=筹码凝聚=突破前兆。
- [chip_dispersion_width](#chip_dispersion_width)  — 筹码成本离散度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（取负向=宽散=分歧大排后）。
- [chip_entropy_convergence](#chip_entropy_convergence)  — 筹码熵收敛因子，分布熵的5日变化截面排名（取负=熵降排前）。熵下降=信息收敛=预期正在趋于一致。
- [chip_entropy_signal](#chip_entropy_signal)  — 筹码熵信号因子，分布信息熵截面排名（取负=低熵排前）。低熵=结构有序=主力控盘明显。
- [chip_gini_factor](#chip_gini_factor)  — 筹码基尼系数因子，分布Gini系数截面排名。高Gini=筹码集中在少数价位=价格锚定清晰。
- [chip_gini_momentum](#chip_gini_momentum)  — 筹码基尼动量因子，Gini系数的5日变化截面排名。Gini上升=筹码向少数价位集中=均衡被打破、新共识形成中。
- [chip_high_float_ratio](#chip_high_float_ratio)  — 高浮盈筹码占比因子：成本价<0.9×close的筹码比例截面排名（负向，深度获利盘多排后）。
- [chip_iqr_factor](#chip_iqr_factor)  — 筹码四分位距因子，分布IQR（P75-P25）截面排名（取负=窄IQR排前）。窄IQR=核心50%筹码高度集中。
- [chip_iqr_momentum_20d](#chip_iqr_momentum_20d)  — 筹码四分位距20日变化因子：IQR的20日变化截面排名（收窄排前）。
- [chip_loss_peak_frac](#chip_loss_peak_frac)  — 套牢筹码集中度因子：现价上方筹码中最大成本峰占比截面排名（负向，单点套牢压力排后）。
- [chip_mean_distance](#chip_mean_distance)  — 筹码均价距离因子，(close-chip_weighted_mean)/close截面排名。价格在加权平均成本上方=多数盈利+强支撑。
- [chip_median_distance](#chip_median_distance)  — 筹码中位数距离因子，(close-chip_median_price)/close截面排名。价格在持仓成本中位数上方=多数盈利+支撑。
- [chip_median_momentum](#chip_median_momentum)  — 筹码中位成本动量因子：中位成本价5日变化截面排名（成本上移排前）。
- [chip_mode_mean_convergence](#chip_mode_mean_convergence)  — 筹码众均收敛因子，|peak-mean|/mean的5日变化截面排名（取负=缺口收窄排前）。众数均值趋于一致=分布从多峰向单峰收敛。
- [chip_mode_mean_gap](#chip_mode_mean_gap)  — 筹码众数均值偏离因子，|peak_mean|/mean截面排名（取负=缺口小排前）。缺口小=分布对称+单峰结构健康。
- [chip_momentum_resonance_20](#chip_momentum_resonance_20)  — 筹码集中×动量：chip_cr3_factor×momentum_20。筹码集中且走强=主力控盘的趋势。
- [chip_p90_p10_factor](#chip_p90_p10_factor)  — 筹码P90-P10范围因子，分布90%筹码价格区间宽度截面排名（取负=窄区间排前）。窄区间=核心筹码高度重叠。
- [chip_p90_p10_momentum](#chip_p90_p10_momentum)  — 筹码P90-P10动量因子，90%筹码范围的5日变化截面排名（取负=收窄排前）。核心区间收窄=筹码进一步浓缩。
- [chip_peak_distance](#chip_peak_distance)  — 筹码峰距离因子，(close-chip_peak_price)/close截面排名。价格在最大筹码峰上方=支撑排前。
- [chip_peak_growing](#chip_peak_growing)  — 筹码主峰增强因子，主峰占比的5日变化截面排名。主峰增强=资金持续在核心价位收集筹码。
- [chip_peak_purity](#chip_peak_purity)  — 筹码主峰纯度因子，最大峰占总筹码比例截面排名。主峰突出=清晰的价格锚点=有效的支撑/阻力位。
- [chip_peak_ratio](#chip_peak_ratio)  — 下方筹码占比因子，当前价格以下的筹码面积占总筹码面积比例截面排名。
- [chip_peak_shift](#chip_peak_shift)  — 筹码峰移动因子，cost_50pct的20日变化率截面排名（中位数成本上移=看涨排前）。
- [chip_percentile_20d](#chip_percentile_20d)  — 筹码价格分位20日变化因子：当前价在筹码分布中的分位20日变化截面排名（分位抬升排前）。
- [chip_position](#chip_position)  — 筹码位置因子，(close-cost_5pct)/(cost_95pct-cost_5pct)截面排名。
- [chip_price_resonance_20](#chip_price_resonance_20)  — 筹码×价格共振因子：winner_rate×momentum_20×price_to_52w_high截面排名。
- [chip_profit_loss_ratio](#chip_profit_loss_ratio)  — 盈亏筹码比因子，(close-cost_95pct)/(cost_5pct-close)截面排名。
- [chip_range_normalized](#chip_range_normalized)  — 归一化筹码区间因子，(cost_85pct-cost_15pct)/cost_50pct截面排名（取负向=窄区间排前）。
- [chip_range_skew_factor](#chip_range_skew_factor)  — 筹码分位数偏斜因子：(p90−p50)/(p50−p10)截面排名（负向，右偏上方筹码厚排后）。
- [chip_resistance_distance](#chip_resistance_distance)  — 筹码阻力距离因子，cost_85pct/close-1截面排名（取负向=接近阻力排后）。
- [chip_semi_std_momentum](#chip_semi_std_momentum)  — 筹码下行半方差动量因子：下行半标准差的5日变化截面排名（负向，下行离散扩大排后）。
- [chip_skew_factor](#chip_skew_factor)  — 筹码偏度因子，分布偏度截面排名。右偏（正偏）=上方筹码多=牛市中换手充分、趋势惯性。
- [chip_skew_momentum](#chip_skew_momentum)  — 筹码偏度动量因子，分布偏度的5日变化截面排名。偏度右移=筹码重心上移+趋势延续。
- [chip_skewness_ratio](#chip_skewness_ratio)  — 筹码成本偏度因子，(cost_50pct-cost_5pct)/(cost_95pct-cost_50pct)截面排名（>1=低位密集看涨排前）。
- [chip_support_distance](#chip_support_distance)  — 筹码支撑距离因子，close/cost_15pct-1截面排名。距离支撑位越近=反弹潜力越大。
- [chip_support_strength](#chip_support_strength)  — 筹码支撑强度因子，cost_15pct处的筹码密度×1/(价格-成本15pct距离)截面排名。
- [chip_support_strength_20d](#chip_support_strength_20d)  — 20日筹码支撑强度因子 (成本密集区数量/支撑距离)。
- [chip_tail_risk](#chip_tail_risk)  — 筹码尾部风险因子，分布超额峰度截面排名（取负=低峰度排前）。高峰度=肥尾=远离均价的极端筹码堆积、异动风险高。
- [chip_tail_risk_change](#chip_tail_risk_change)  — 筹码尾部风险变化因子，分布峰度的5日变化截面排名（取负=尾部收窄排前）。尾部收窄=极端筹码被消化=结构改善。
- [chip_weighted_cost_momentum_5d](#chip_weighted_cost_momentum_5d)  — 加权成本5日动量因子，(weight_avg-weight_avg.shift(5))/weight_avg截面排名。衡量平均成本迁移速度。
- [chip_weighted_cost_volatility_20d](#chip_weighted_cost_volatility_20d)  — 20日加权成本波动率因子 (成本稳定排前, 负向)。
- [chip_weighted_mean_momentum](#chip_weighted_mean_momentum)  — 筹码平均成本动量因子：加权平均成本价5日变化截面排名（成本上移排前）。
- [chip_width_ratio_momentum](#chip_width_ratio_momentum)  — 筹码相对宽度动量因子：宽度比(std/close)的5日变化截面排名（负向，变宽排后）。
- [chip_win_peak_frac](#chip_win_peak_frac)  — 获利筹码集中度因子：现价下方筹码中最大成本峰占比截面排名（获利筹码集中锁筹排前）。
- [chip_win_peak_growth](#chip_win_peak_growth)  — 获利筹码集中度动量因子：获利筹码集中度的5日变化截面排名（吸筹中成本峰强化排前）。
- [chip_winner_rate_acceleration](#chip_winner_rate_acceleration)  — 获利盘比例加速度因子，winner_rate_change_5d的5日差分（二阶导数）截面排名（取负向=加速获利排后）。
- [chip_winner_rate_ma5](#chip_winner_rate_ma5)  — 获利盘比例5日均值因子 (获利盘多排后, 负向)。
- [chip_winner_rate_stability_20d](#chip_winner_rate_stability_20d)  — 获利盘比例20日稳定性因子，winner_rate的20日滚动标准差截面排名（取负向=高波动排后）。
- [circ_mv_share_change_20d](#circ_mv_share_change_20d)  — 流通市值占比20日变化。占比增加=限售解禁/减持压力，排名低；占比减少=回购/增持，排名高。
- [circ_mv_to_total_mv](#circ_mv_to_total_mv)  — 流通市值/总市值截面排名（高流通比排前）。
- [close_30_momentum](#close_30_momentum)  — 尾盘30分钟动量因子：收盘半小时(14:30-15:00)涨跌幅截面排名（尾盘拉升排前）。
- [close_5min_momentum](#close_5min_momentum)  — 尾盘5分钟动量因子，最后5分钟收益截面排名（取负向=尾盘拉升=次日易低开排后）。
- [close_auction_impact](#close_auction_impact)  — 收盘竞价影响因子，最后3分钟收益截面排名（正拉升=尾盘抢筹排前）。
- [close_auction_pressure](#close_auction_pressure)  — 收盘位置压力因子，收盘价在日内(high-low)区间的位置截面排名（收盘接近低点=尾盘卖压排后）。
- [close_auction_ret](#close_auction_ret)  — 收盘集合竞价收益因子：14:57开盘→15:00收盘涨跌幅截面排名（负向）。
- [close_auction_vol_share](#close_auction_vol_share)  — 收盘集合竞价量占比因子：14:57-14:59成交额占全天比例截面排名（负向）。
- [close_location_20d](#close_location_20d)  — 收盘位置（20日收益 ÷ 20日振幅）。
- [close_location_5d](#close_location_5d)  — 5 日收盘位置（(close − 5日最低价)/(5日最高价 − 5日最低价)）。
- [close_position](#close_position)  — 收盘位置因子，(收盘-最低)/(最高-最低)截面排名。收盘在高位=买方主导全日。
- [close_position_intraday_20](#close_position_intraday_20)  — 收盘价日内位置因子，20日均(close-low)/(high-low)截面排名。
- [close_position_ratio](#close_position_ratio)  — 收盘价在日内高低点区间中的位置：(close-low)/(high-low)，强势收盘=排名高。
- [close_position_vol_weighted_20](#close_position_vol_weighted_20)  — 量能加权收盘位置：20日均值(量比×日内收盘位置)截面排名。量能集中于高位收盘=强承接。
- [close_to_high_20d](#close_to_high_20d)  — 收盘距20日最高价距离（close/max20(high)−1）。
- [close_to_high_5d](#close_to_high_5d)  — 收盘相对 5 日最高价（close / 5日最高价）。
- [close_to_high_ratio_20](#close_to_high_ratio_20)  — 20日均收盘/最高价比率因子 (收盘强势)
- [cmo_20](#cmo_20)  — Chande动量振荡器因子：20日(涨额和−跌额和)/(涨额和+跌额和)截面排名（多空动能净额排前）。
- [consecutive_limit_down](#consecutive_limit_down)  — 连续跌停天数因子：连续跌停(近似)的连跌计数截面排名（连跌高度排前）。
- [consecutive_limit_up](#consecutive_limit_up)  — 连续涨停天数因子：连续涨停(近似)的连板计数截面排名（连板高度排前）。
- [corr_market_60](#corr_market_60)  — 60日市场相关性因子（个股日收益与全池等权市场收益的60日滚动相关，反向排名）。
- [cost_convergence_signal](#cost_convergence_signal)  — 成本收敛信号因子，(cost_95pct-cost_5pct)的20日变化率截面排名（分布收窄=筹码集中排前）。
- [cost_displacement](#cost_displacement)  — 成本偏离因子，(close-weight_avg)/weight_avg截面排名（取负向=大幅偏离排后）。
- [cost_displacement_extreme](#cost_displacement_extreme)  — 成本偏离极端度因子，(close-weight_avg)/weight_avg的绝对值截面排名（极度偏离=均值回归压力大排前）。
- [cost_distribution_skew](#cost_distribution_skew)  — 成本分布偏度因子，(cost_50pct-cost_15pct)/(cost_85pct-cost_50pct)截面排名。
- [cost_distribution_width](#cost_distribution_width)  — 筹码成本分布宽度因子，(cost_95pct-cost_5pct)/cost_50pct截面排名（窄分布=筹码集中排前）。
- [cost_skew_momentum_5d](#cost_skew_momentum_5d)  — 成本偏度5日变化因子，chip_cost_skew的5日差分截面排名。
- [cost_skew_ratio](#cost_skew_ratio)  — 筹码成本偏度比率因子，(cost_50pct-cost_5pct)/(cost_95pct-cost_50pct)截面排名（右偏=获利盘主导排前）。
- [cost_support_strength](#cost_support_strength)  — 筹码支撑强度=(收盘价-cost_50pct)/cost_50pct取反。价格在成本线下方=超跌,支撑强,排名高。
- [coupling_bigflow_margin_buy_20](#coupling_bigflow_margin_buy_20)  — 大单融资双确认因子：mf_net_inflow_ratio×margin_net_flow_ratio截面排名（两类聪明钱同向流入排前）。
- [coupling_boll_squeeze_value](#coupling_boll_squeeze_value)  — 布林挤压(boll_squeeze) × 低估值(bp)。布林收窄(蓄力)+低估=突破前的最优买点。
- [coupling_chip_cost_accel_20](#coupling_chip_cost_accel_20)  — 筹码成本上移加速度：chip_median_momentum_t − chip_median_momentum_t-20 截面排名（成本上移20日加速排前）。
- [coupling_chip_support_reversal_5](#coupling_chip_support_reversal_5)  — 筹码支撑超跌因子：short_term_reversal_5 × chip_support_strength 截面排名（超跌且有筹码支撑排前）。
- [coupling_chip_trend_confirm_20](#coupling_chip_trend_confirm_20)  — 筹码成本确认趋势：momentum_20 × chip_peak_shift 截面排名（趋势+成本峰20日上移排前）。
- [coupling_flow_persistence_20](#coupling_flow_persistence_20)  — 融资净流入跨期自共振：margin_net_flow_ratio_t × margin_net_flow_ratio_t-20 截面排名（20日持续净流入排前）。
- [coupling_flowaccel_breakout_60](#coupling_flowaccel_breakout_60)  — 资金加速突破因子：big_order_net_accel_10 × new_high_60_event 截面排名（大单加速+60日新高排前）。
- [coupling_fundflow_accel_10](#coupling_fundflow_accel_10)  — 主力资金时间加速度：mf_net_inflow_ratio_t − mf_net_inflow_ratio_t-10 截面排名（净流入排名10日抬升排前）。
- [coupling_indicator_consensus_value](#coupling_indicator_consensus_value)  — 技术指标共识度(indicator_consensus) × 低估值(bp)。多指标一致看多+低估=最强信号。
- [coupling_intraday_tail_momentum_20](#coupling_intraday_tail_momentum_20)  — 尾盘资金确认趋势：momentum_20 × tail_volume_share 截面排名（趋势+尾盘放量排前）。
- [coupling_kdj_moneyflow_divergence](#coupling_kdj_moneyflow_divergence)  — KDJ金叉净数(kdj_cross_net) vs 主力资金背离。KDJ金叉+主力流出=技术虚涨，KDJ死叉+主力流入=洗盘。
- [coupling_lhb_reversal_20](#coupling_lhb_reversal_20)  — 博弈票超跌反弹因子：lhb_proxy_score_60×short_term_reversal_5截面排名（游资惯犯且超跌排前）。
- [coupling_limitup_momentum_20](#coupling_limitup_momentum_20)  — 涨停延续动量：limit_up_fade_10 × momentum_20 截面排名（涨停后延续+动量排前）。
- [coupling_liquidity_momentum_20d](#coupling_liquidity_momentum_20d)  — 流动性因子(turnover_20)自身20日变化。换手率上升=关注度提升，排名高。
- [coupling_lowrisk_momentum_60](#coupling_lowrisk_momentum_60)  — 低风险趋势因子：momentum_60 × downside_frequency_60 截面排名（中期动量+下行频率低排前）。
- [coupling_ma_chip_resonance](#coupling_ma_chip_resonance)  — 均线多头排列 × 筹码集中(cr3)。均线完美+筹码集中=主力高度控盘的上升趋势。
- [coupling_macd_chip_divergence](#coupling_macd_chip_divergence)  — MACD趋势强度 vs 筹码集中度(cr3)背离。MACD强+筹码散=虚涨，MACD弱+筹码集中=蓄力。
- [coupling_macd_value_resonance](#coupling_macd_value_resonance)  — MACD趋势强度 × 价值(bp)共振。MACD走强+低估值=技术面与基本面双重确认。
- [coupling_margin_buy_persist_10](#coupling_margin_buy_persist_10)  — 融资买入意愿跨期自共振：margin_buy_pressure_t × margin_buy_pressure_t-10 截面排名（买入意愿持续强于偿还排前）。
- [coupling_margin_buy_trend_20](#coupling_margin_buy_trend_20)  — 融资买入确认趋势：margin_buy_pressure × momentum_20 截面排名（买入意愿强+动量排前）。
- [coupling_margin_chip_cost_20](#coupling_margin_chip_cost_20)  — 融资筹码成本复合因子：margin_chip_cost_gap×momentum_20×drawdown_60截面排名（杠杆结构健康的中期趋势排前）。
- [coupling_margin_lead_trend_60](#coupling_margin_lead_trend_60)  — 杠杆资金领先趋势：margin_net_flow_ratio_t-60 × momentum_60_t 截面排名（60日前融资净流入+中期动量排前）。
- [coupling_min_align_momentum_20](#coupling_min_align_momentum_20)  — 分钟均线确认趋势：momentum_20 × min_ma_alignment_frac_20 截面排名（趋势+分钟均线多头排列占比高排前）。
- [coupling_momentum_drift_20](#coupling_momentum_drift_20)  — 动量时间加速度：momentum_20_t − momentum_20_t-20 截面排名（动量排名20日抬升=趋势加速排前）。
- [coupling_moneyflow_lead_momentum_10](#coupling_moneyflow_lead_momentum_10)  — 资金领先动量：mf_net_inflow_ratio_t-10 × momentum_20_t 截面排名（10日前主力净流入+当前动量排前）。
- [coupling_net_turnover_momentum_20](#coupling_net_turnover_momentum_20)  — 净换手动量共振因子：net_turnover_rate_20×momentum_20截面排名（净买入驱动的动量排前）。
- [coupling_quality_momentum_20d](#coupling_quality_momentum_20d)  — 盈利质量因子(sp_ttm+bp rank)20日动量。质量改善=排名高。
- [coupling_quality_trend_60](#coupling_quality_trend_60)  — 高质量趋势因子：momentum_60 × sortino_ratio_60 截面排名（中期动量+Sortino高排前）。
- [coupling_rsi_moneyflow_resonance](#coupling_rsi_moneyflow_resonance)  — RSI趋势 × 主力资金净流入。RSI走强+主力流入=技术和资金双重看多。
- [coupling_rsi_turnover_divergence](#coupling_rsi_turnover_divergence)  — RSI超卖 × 换手率(turnover_20)。超卖+高换手=恐慌抛售中的抄底机会。
- [coupling_rsi_value_combo](#coupling_rsi_value_combo)  — RSI超卖(低RSI=排前) × 低估值(bp)。超卖+低估=抄底双信号。
- [coupling_smallcap_value_combo](#coupling_smallcap_value_combo)  — 小市值(log_total_mv取反=小盘高排)+低估值(bp)。小盘价值股效应。
- [coupling_smartmoney_lead_momentum_10](#coupling_smartmoney_lead_momentum_10)  — 聪明钱领先动量：smart_money_share_t-10 × momentum_20_t 截面排名（10日前信息型交易活跃+当前动量排前）。
- [coupling_stableflow_momentum_20](#coupling_stableflow_momentum_20)  — 稳定资金流趋势：momentum_20 × mf_flow_stability_20d 截面排名（趋势+主力资金连续同向排前）。
- [coupling_valuation_sentiment_divergence](#coupling_valuation_sentiment_divergence)  — 估值(bp)与情绪(turnover_20)的背离。低估值+高换手=价值发现，排名高。
- [coupling_value_momentum_20d](#coupling_value_momentum_20d)  — 价值因子(bp)自身20日动量。bp值持续上升=估值优势在扩大，排名高。
- [coupling_value_quality_resonance](#coupling_value_quality_resonance)  — 价值(bp)+质量(sp_ttm)共振。两者同时高排名=优质低估，超级信号。
- [coupling_volterm_momentum_60](#coupling_volterm_momentum_60)  — 平缓期限结构趋势：momentum_60 × rv_term_structure_slope 截面排名（中期动量+短期波动不陡峭排前）。
- [coupling_volume_lead_momentum_5](#coupling_volume_lead_momentum_5)  — 量能领先动量：volume_momentum_5_t-5 × momentum_20_t 截面排名（5日前量能扩张+当前动量排前）。
- [coupling_vp_amfade_rev_20](#coupling_vp_amfade_rev_20)  — 早盘恐慌反转因子：vp_expand_down_am_share×short_term_reversal_5截面排名（早盘恐慌释放的超跌排前）。
- [coupling_vp_chip_consistency_20](#coupling_vp_chip_consistency_20)  — 量价筹码共振因子：vp_consistency_20×chip_win_peak_frac截面排名（量价健康且筹码锁筹排前）。
- [coupling_vp_expand_up_mom_20](#coupling_vp_expand_up_mom_20)  — 放量上涨动量确认因子：vp_expand_up_share×momentum_20截面排名（放量且动量确认排前）。
- [coupling_vp_lowpos_accumulate_20](#coupling_vp_lowpos_accumulate_20)  — 低位放量吸筹确认因子：vp_expand_price_pos×chip_win_peak_frac×mf_net_inflow_ratio截面排名（三因子吸筹确认排前）。
- [coupling_vp_retvol_mom_20](#coupling_vp_retvol_mom_20)  — 量价同步动量因子：minute_ret_vol_corr×momentum_20截面排名（微观量价同步且趋势向上排前）。
- [coupling_vp_shrink_down_rev_5](#coupling_vp_shrink_down_rev_5)  — 缩量下跌反转确认因子：vp_shrink_down_share×short_term_reversal_5截面排名（洗盘缩量回调的超跌排前）。
- [coupling_winner_bigflow_20](#coupling_winner_bigflow_20)  — 获利盘大单共振因子：(1−winner_rate)×mf_net_inflow_ratio截面排名（获利盘多且主力净买入排前）。
- [cumulative_ret_path](#cumulative_ret_path)  — 累积收益路径效率因子，|总收益|/Σ|1分钟收益|截面排名（高效率=趋势明确排前）。
- [current_down_streak](#current_down_streak)  — 当前连跌天数因子：截至今日连续收跌天数截面排名（连跌超卖排前，反向排名）。
- [current_up_streak](#current_up_streak)  — 当前连涨天数因子：截至今日连续收涨天数截面排名（连涨动能排前）。
- [current_vol_shrink_streak](#current_vol_shrink_streak)  — 当前缩量连天数因子：截至今日成交量连续递减天数截面排名（持续缩量排前）。
- [cvar_95_120](#cvar_95_120)  — 120日CVaR因子：1%至5%滚动分位的积分近似（条件尾部损失），排名高=尾部风险小。
- [deep_value_reversal_combo_60](#deep_value_reversal_combo_60)  — 三重左侧复合因子：bp×(1−drawdown_120)×short_term_reversal_5截面排名。
- [defensive_momentum_combo_60](#defensive_momentum_combo_60)  — 防御动量复合因子：beta_60×momentum_60×ulcer_index_20截面排名。
- [di_plus_minus_ratio_14](#di_plus_minus_ratio_14)  — 14日DI+/DI-比率因子 (多头趋势强度)
- [distance_from_ma_120](#distance_from_ma_120)  — 收盘价/120日均线-1因子截面排名 (负向：远离均线=回归压力)。
- [distance_from_ma_5](#distance_from_ma_5)  — 收盘价/5日均线-1因子截面排名。
- [doji_frequency_20](#doji_frequency_20)  — 十字星频率因子：20日十字星(|实体|≤10%振幅)天数占比截面排名（分歧整理排前）。
- [donchian_breakout_20](#donchian_breakout_20)  — 20日Donchian突破强度因子
- [donchian_breakout_strength](#donchian_breakout_strength)  — Donchian突破强度因子，(close-20日最高)/ATR截面排名（突破幅度相对波动率排前）。
- [donchian_position_20](#donchian_position_20)  — Donchian通道位置因子，(close-20日最低)/(20日最高-20日最低)截面排名（突破高位=强势排前）。
- [donchian_position_60](#donchian_position_60)  — 60日Donchian通道位置因子 (低位置排前, 负向)
- [downside_frequency_60](#downside_frequency_60)  — 下行频率因子：60日负收益天数占比截面排名（负向，频繁下跌排后）。
- [downside_upside_vol_60](#downside_upside_vol_60)  — 下行/上行波动比因子：60日负收益波动/正收益波动截面排名（负向，下行风险主导排后）。
- [downside_vol_ratio_20](#downside_vol_ratio_20)  — 下行波动占比因子（20日半波动/总波动比，反向排名）。
- [dp_ttm](#dp_ttm)  — 滚动股息率因子，截面排名。
- [dpo_20](#dpo_20)  — 20日DPO去趋势因子：(11日前复权价−当前20日均价)/20日均价截面排名。
- [drawdown_120](#drawdown_120)  — 120日最大回撤因子（当前价格相对120日窗口峰值回撤，反向排名）。
- [drawdown_60](#drawdown_60)  — 60日最大回撤因子（当前价格相对60日窗口峰值回撤，反向排名）。
- [drawdown_duration_120](#drawdown_duration_120)  — 120日回撤持续期因子：当前价格低于120日滚动前高的连续天数（上限120日，负向排名）。
- [drawdown_recovery_60](#drawdown_recovery_60)  — 回撤修复率因子：复权价/60日新高截面排名（正向，接近新高排前）。
- [dv_composite](#dv_composite)  — 股息率综合因子，dv_ratio与dv_ttm的等权平均截面排名。
- [dv_momentum_combo_20](#dv_momentum_combo_20)  — 高股息×动量：dv_ttm_rank×momentum_20。高股息+走强=股息资金与趋势资金共振。
- [dv_stability_4q](#dv_stability_4q)  — 股息稳定性因子，约1季度(60交易日)dv_ratio变异系数截面排名（股息稳定排前）。
- [dv_ttm_rank](#dv_ttm_rank)  — 股息率TTM因子，dv_ttm截面排名（高TTM股息排前）。
- [dv_yield_rank](#dv_yield_rank)  — 股息率因子，dv_ratio截面排名（高股息排前）。
- [elg_net_60d_to_mv](#elg_net_60d_to_mv)  — 超大单60日净买入占流通市值比因子：Σ(超大单买−卖,60日)×1e4/流通市值截面排名（长线吸筹强度排前）。
- [eom_14](#eom_14)  — 14日EOM简易波动因子：价格中点位移÷(量/区间)的14日均值截面排名（上涨越省量排前）。
- [event_momentum_divergence_20](#event_momentum_divergence_20)  — 事件-动量背离因子：limit_up_event_5−momentum_20截面排名（事件强动量弱排前）。
- [ext_mf_amount_concentration](#ext_mf_amount_concentration)  — 成交额集中度因子（低值排前），各订单规模成交额占比的Herfindahl指数。
- [ext_mf_big_order_net_amount_ratio](#ext_mf_big_order_net_amount_ratio)  — 大单净买入金额占比因子，大单+特大单净买入额/总成交额截面排名。
- [ext_mf_large_order_avg_price](#ext_mf_large_order_avg_price)  — 大单均价因子，大单+特大单成交均价/VWAP截面排名（高值排前）。
- [ext_mf_medium_order_amount_ratio](#ext_mf_medium_order_amount_ratio)  — 中单成交额占比因子，中单（4-20万）成交额/总成交额截面排名。
- [ext_mf_small_order_amount_ratio](#ext_mf_small_order_amount_ratio)  — 小额订单成交额占比因子，散户参与度度量（低值排前）。
- [ext_mf_small_order_avg_price](#ext_mf_small_order_avg_price)  — 小额订单均价因子（低值排前），小额成交均价/VWAP截面排名。
- [extreme_gain_freq_20](#extreme_gain_freq_20)  — 上行极端收益频率因子：20日收益高于均值+1.5σ的天数占比截面排名（负向，彩票偏好排后）。
- [extreme_move_count](#extreme_move_count)  — 极端波动次数因子，日内|ret_5min|>3σ的次数截面排名（取负向=频繁极端波动排后）。
- [extreme_move_event](#extreme_move_event)  — 极端波动事件衰减因子：|pct_chg|>7%事件后指数衰减（半衰期10日，异动后惯性排前）。
- [factor_consistency_ratio](#factor_consistency_ratio)  — 因子一致性比率因子，BP因子60日变化方向一致的天数占比（持续方向=高信度排前）。
- [factor_consistency_score](#factor_consistency_score)  — 因子一致性因子，bp 60日在极端分位(>0.8或<0.2)的占比截面排名（持续极端=信号强烈排前）。
- [factor_crowding_warning](#factor_crowding_warning)  — 因子拥挤预警因子，-(bp排名60日中同方向占比>80%)截面排名（拥挤=一致预期风险排后）。
- [factor_cycle_position](#factor_cycle_position)  — 因子周期位置因子，bp偏离2年均值的符号×(偏离持续的月数)截面排名（正偏离+持续长=周期高位排前）。
- [factor_drawdown_60_deep](#factor_drawdown_60_deep)  — 因子滚动回撤因子，-(bp 60日滚动最大回撤)截面排名（深度回撤=因子失效风险排后）。
- [factor_ic_ir_proxy_60](#factor_ic_ir_proxy_60)  — 因子IC IR代理因子，bp 60日(均值/std)×sqrt(60)截面排名（高信息比率=因子有效性强排前）。
- [factor_mean_reversion_20](#factor_mean_reversion_20)  — 因子均值回复因子，-(bp偏离20日均值的标准差倍数)截面排名（取负向=过度偏离=回复压力排后）。
- [factor_momentum_decay](#factor_momentum_decay)  — 因子动量衰减因子，bp的5日动量/20日动量截面排名（衰减=短期弱于长期=动能减弱排后）。
- [factor_multi_horizon_momentum](#factor_multi_horizon_momentum)  — 多周期因子动量因子，(bp 5日动量排名+bp 20日动量排名+bp 60日动量排名)/3截面排名。多周期共振。
- [factor_profile_shift_deep](#factor_profile_shift_deep)  — 因子轮廓位移因子，bp 20日前排名与当前排名的均方差截面排名（位移大=剧烈变化排后）。
- [factor_rolling_drawdown_60](#factor_rolling_drawdown_60)  — 因子滚动回撤因子，BP因子从60日高点的回撤程度（回撤大=因子超跌排前）。
- [factor_signal_to_noise_60](#factor_signal_to_noise_60)  — 因子信噪比因子，bp的60日均值/std截面排名（高信噪比=因子信号清晰排前）。
- [factor_trend_strength_60](#factor_trend_strength_60)  — 因子趋势强度因子，bp因子60日均值偏移/60日std截面排名（强趋势=因子方向可靠排前）。
- [factor_turnover_ratio_20](#factor_turnover_ratio_20)  — 因子换手率因子，bp截面排名20日变化绝对值截面排名（取负向=高换手=不稳定排后）。
- [factor_volatility_regime_shift](#factor_volatility_regime_shift)  — 因子波动率状态转换因子，bp 20日std/60日std截面排名（短期波动>长期波动=状态转入高波排后）。
- [fear_index_20](#fear_index_20)  — 恐惧指数因子：20日窗口内下行极端收益(低于均值1.5σ)的频率截面排名（恐慌频发排前）。
- [fib_retracement_proximity](#fib_retracement_proximity)  — 黄金分割位邻近度因子：250日波段的斐波那契回撤位距离截面排名（靠近关键位排前）。
- [flash_crash_risk](#flash_crash_risk)  — 闪崩风险因子，日内5分钟累计收益最大回撤截面排名（取负向=闪崩风险高排后）。
- [float_mv_ratio](#float_mv_ratio)  — 自由流通市值占比因子，free_share/total_share截面排名。
- [float_share_ratio](#float_share_ratio)  — 自由流通股占比取反排名。流通盘小=筹码稀缺+弹性大，排名高。
- [force_index_13](#force_index_13)  — 13日力指数因子：(close−pre_close)×vol的13日EWMA截面排名（量价合力排前）。
- [free_float_expansion_20d](#free_float_expansion_20d)  — 自由流通股扩张因子：自由流通股/总股本占比的20日变化截面排名（负向，解禁抛压排后）。
- [free_share_turnover_ratio](#free_share_turnover_ratio)  — 自由流通换手/总换手比。高比值=实际可交易筹码在充分换手,排名高。
- [fund_flow_alpha_combo_60](#fund_flow_alpha_combo_60)  — 主力资金独立alpha因子：mf_net_inflow_5d×momentum_60×corr_market_60截面排名。
- [fund_flow_volatility_20](#fund_flow_volatility_20)  — 资金流波动性因子，-(net_mf_amount/vol的20日标准差)截面排名（资金流不稳定排后）。
- [fundflow_retail_inst_divergence](#fundflow_retail_inst_divergence)  — 主力-散户资金背离因子 (大单净买+小单净卖=机构吸筹)。
- [fundflow_value_interaction](#fundflow_value_interaction)  — 资金流-价值交互因子，主力净流入排名×bp排名截面排名（资金流入+低估=戴维斯双击前兆排前）。
- [gain_loss_asymmetry_60](#gain_loss_asymmetry_60)  — 涨跌幅度不对称因子：60日平均涨幅/|平均跌幅|截面排名（涨多跌少排前）。
- [gap_abs_intraday](#gap_abs_intraday)  — 隔夜跳空绝对值因子，|开盘/前收-1|截面排名（取负向=大跳空排后）。隔夜信息冲击的绝对程度。
- [gap_abs_ma_20d](#gap_abs_ma_20d)  — 缺口幅度均值（|open/pre_close−1| 的20日均值）。
- [gap_abs_ma_5d](#gap_abs_ma_5d)  — 5 日平均绝对跳空（|open/pre_close − 1| 的 5 日均值）。
- [gap_down_recover_freq_20d](#gap_down_recover_freq_20d)  — 低开高走频率（20日内 overnight<0 且 intraday>0 的天数占比）。
- [gap_event_decay_5](#gap_event_decay_5)  — 跳空事件衰减因子：|open/pre_close−1|>5%跳空事件后指数衰减（半衰期5日）。
- [gap_fill_5d_reversal](#gap_fill_5d_reversal)  — 缺口回补反转因子，5日内出现向下跳空后的回补倾向截面排名。
- [gap_fill_tendency_10d](#gap_fill_tendency_10d)  — 缺口回补倾向因子，近10日缺口天数/(缺口天数+0.01)截面排名。
- [gap_intraday_corr_20](#gap_intraday_corr_20)  — 缺口-日内收益相关性因子：20日(高开幅度,日内收益)滚动相关截面排名（高开常延续排前）。
- [gap_momentum_5d](#gap_momentum_5d)  — 跳空动量因子，5日跳空缺口累计截面排名（持续跳空=强势延续排前）。
- [gap_open_follow_ratio_20](#gap_open_follow_ratio_20)  — 跳空方向延续率因子：20日跳空(>0.5%)日中跳空方向与日内方向一致占比截面排名（顺延结构排前）。
- [gap_ratio](#gap_ratio)  — 跳空比率因子，(open-pre_close)/pre_close截面排名（正=高开幅度大排前）。
- [gap_ratio_20](#gap_ratio_20)  — 20日均跳空比率因子，open/pre_close - 1 截面排名。
- [gap_reversal_5d](#gap_reversal_5d)  — 跳空反转信号——跳空方向与日内走势方向相反时标记强度取反。高开低走/低开高走=趋势陷阱。
- [gap_up_fade_freq_20d](#gap_up_fade_freq_20d)  — 高开低走频率（20日内 overnight>0 且 intraday<0 的天数占比）。
- [gap_up_ratio_20d](#gap_up_ratio_20d)  — 20日高开概率因子，高开(open>pre_close)天数/20截面排名。
- [gap_volume_interaction_20](#gap_volume_interaction_20)  — 跳空×量能交互：20日均值(跳空幅度×量比)截面排名。高开且放量=资金抢筹。
- [gk_vol](#gk_vol)  — Garman-Klass波动率估计因子，基于OHLC四价的高效波动率截面排名（低波排前）。
- [hammer_ratio_20d](#hammer_ratio_20d)  — 锤子线频率因子，近20日下影线>实体2倍且实体小的天数截面排名（反转信号排前）。
- [high_low_amplitude_20](#high_low_amplitude_20)  — 振幅因子，(20日最高-20日最低)/20日均价截面排名。
- [high_low_expansion](#high_low_expansion)  — 日内振幅相对20日均值的扩张程度。振幅扩大=分歧加剧，振幅收缩=方向选择在即。
- [high_low_volatility_20](#high_low_volatility_20)  — Parkinson波动率因子，20日基于最高最低价的波动率估计截面排名（高波排后）。
- [high_open_low_close_frac_20](#high_open_low_close_frac_20)  — 高开低走频率因子：20日(高开≥2%且收阴)天数占比截面排名（负向，高开低走惯犯排后）。
- [higher_highs_20](#higher_highs_20)  — 20日不断抬高的高点因子 (趋势延续)
- [hl_range_intraday](#hl_range_intraday)  — 日内振幅因子，1分钟数据得到的日度(high/low-1)截面排名（低振幅=筹码稳定排前）。
- [idio_vol_60](#idio_vol_60)  — 60日特质波动率因子（剔除市场暴露后的残差波动，反向排名）。
- [idio_vol_momentum_combo_20](#idio_vol_momentum_combo_20)  — 低特质波动×动量：momentum_20×idio_vol_60。低特质波动的动量更稳。
- [ind_disp_ma_20d](#ind_disp_ma_20d)  — 行业离散度（行业日收益横截面std的20日均值）。
- [ind_disp_ma_5d](#ind_disp_ma_5d)  — 行业离散度（行业日收益横截面std的5日均值）。
- [ind_disp_ma_60d](#ind_disp_ma_60d)  — 行业离散度（行业日收益横截面std的60日均值）。
- [ind_ret_ma_20d](#ind_ret_ma_20d)  — 行业收益动能（行业等权日收益的20日均值）。
- [ind_ret_ma_3d](#ind_ret_ma_3d)  — 行业收益动能（行业等权日收益的3日均值）。
- [ind_ret_ma_5d](#ind_ret_ma_5d)  — 行业收益动能（行业等权日收益的5日均值）。
- [ind_ret_ma_60d](#ind_ret_ma_60d)  — 行业收益动能（行业等权日收益的60日均值）。
- [indicator_consensus](#indicator_consensus)  — 指标共识度因子（6个指标中同向占比截面排名，高共识=多指标共振排前）。
- [indicator_dispersion](#indicator_dispersion)  — 指标离散度因子（5个标准化指标信号的std截面排名，离散度高=指标矛盾排后）。
- [industry_relative_momentum_20](#industry_relative_momentum_20) 【已禁用】 — 行业中性20日动量因子（个股20日动量减所属行业均值动量），截面排名。
- [inside_bar_count_20](#inside_bar_count_20)  — 孕线频率因子：20日孕线(当日高低点完全落入昨日区间)天数截面排名（收敛蓄势排前）。
- [intra_trend](#intra_trend)  — 日内趋势强度因子，正收益5分钟区间占比截面排名。日内方向一致性度量。
- [intraday_cum_20d](#intraday_cum_20d)  — 日内收益20日复利累计。
- [intraday_high_low_volatility](#intraday_high_low_volatility)  — 日内高低波幅因子，(日内最高-日内最低)/开盘价截面排名（取负向=剧烈波动=不确定性高排后）。
- [intraday_high_time](#intraday_high_time)  — 日内最高价时点因子：日内最高价出现的归一化时段截面排名（尾盘创新高排前）。
- [intraday_lower_shadow](#intraday_lower_shadow)  — 下影线比例因子，(下影线/实体)截面排名（长下影=支撑强=探底回升排前）。
- [intraday_ma_20d](#intraday_ma_20d)  — 日内收益均值（close/open−1 的20日均值）。
- [intraday_ma_5d](#intraday_ma_5d)  — 日内收益均值（close/open−1 的5日均值）。
- [intraday_ma_60d](#intraday_ma_60d)  — 日内收益均值（close/open−1 的60日均值）。
- [intraday_max_drawdown](#intraday_max_drawdown)  — 日内最大回撤因子，日内从最高点到后续最低点的最大跌幅截面排名（取负向=深回撤排后）。
- [intraday_max_runup](#intraday_max_runup)  — 日内最大拉升因子，日内从最低点到后续最高点的最大涨幅截面排名（强拉升=买方力量排前）。
- [intraday_momentum](#intraday_momentum)  — 日内动量因子，开盘30分钟收益截面排名。衡量隔夜信息消化后的早盘方向。
- [intraday_ret](#intraday_ret)  — 日内收益率因子，(close-open)/open截面排名。
- [intraday_ret_momentum](#intraday_ret_momentum)  — 日内收益因子，(close-open)/open截面排名。
- [intraday_reversal](#intraday_reversal)  — 尾盘反转因子，收盘30分钟收益截面排名（取负向=尾盘拉升排后，易次日低开）。
- [intraday_reversal_intensity](#intraday_reversal_intensity)  — 日内反转强度因子，-(|收益|/最高最低波幅)截面排名（高反转=方向不确定排后）。
- [intraday_trend_strength](#intraday_trend_strength)  — 日内趋势强度因子，|close-open|/(high-low)截面排名（单边趋势强=方向确定排前）。
- [intraday_upper_shadow](#intraday_upper_shadow)  — 上影线比例因子，-(上影线/实体)截面排名（长上影=抛压重排后）。
- [intraday_vol_ratio_5d](#intraday_vol_ratio_5d)  — 日内/隔夜波动比（5 日 |日内收益| 均值 ÷ 5 日 |隔夜收益| 均值）。
- [j_day_position](#j_day_position)  — KDJ J值日内区间位置因子：(J收盘−J最低)/(J最高−J最低)截面排名（J收盘靠上沿排前）。
- [jump_ratio_intraday](#jump_ratio_intraday)  — 跳跃占比因子，日度已实现跳跃/日度RV截面排名（取负向=跳跃主导=不稳定排后）。
- [kama_efficiency_20](#kama_efficiency_20)  — KAMA效率比率因子：|20日净变动|/20日累计波动（趋势效率，高效率=单边趋势排前）。
- [kama_position_20](#kama_position_20)  — KAMA位置因子：后复权收盘价相对20期自适应均线的偏离截面排名（价格站上KAMA排前）。
- [kdj_boll_combo](#kdj_boll_combo)  — KDJ-布林带组合因子（(j-50)/50+boll_position截面排名，双重极端=强烈信号排前）。
- [kdj_bull_frac](#kdj_bull_frac)  — KDJ多头时间占比因子（K>D的分钟占比截面排名，高占比=日内持续多头排前）。
- [kdj_bull_frac_5d_change](#kdj_bull_frac_5d_change)  — KDJ多头占比5日变化因子：日内K>D占比的5日变化截面排名（多头增强排前）。
- [kdj_cross_net](#kdj_cross_net)  — KDJ金叉净数量因子（金叉次数-死叉次数截面排名，净多头=强势排前）。
- [kdj_cross_signal](#kdj_cross_signal)  — KDJ金叉信号因子（日内K上穿D的次数截面排名，金叉次数多=强势排前）。
- [kdj_d_stability](#kdj_d_stability)  — KDJ-D线稳定性因子（std(D)/mean(D)截面排名，D线不稳定=信号质量差排后）。
- [kdj_daily_j](#kdj_daily_j)  — 日频KDJ的J值因子：J=3K−2D取负截面排名（极端超买排后）。
- [kdj_dead_cross_count](#kdj_dead_cross_count)  — KDJ死叉次数因子：日内K下穿D的次数截面排名（负向，死叉频繁排后）。
- [kdj_j_5d_acceleration](#kdj_j_5d_acceleration)  — KDJ J值5日加速度因子：J收盘的5日变化截面排名（J值加速排前）。
- [kdj_j_range](#kdj_j_range)  — KDJ-J值振幅因子（j_max-j_min截面排名，J值波动大=多空分歧大排后）。
- [kdj_j_reversal_risk](#kdj_j_reversal_risk)  — KDJ-J值反转风险因子（|j_close-50|截面排名，极端J值=高反转概率排后）。
- [kdj_j_value_close](#kdj_j_value_close)  — KDJ-J值因子（日内收盘J值截面排名，高J=超买排后，低J=超卖排前）。
- [kdj_j_volatility](#kdj_j_volatility)  — KDJ-J值波动因子（std(J)截面排名，J值波动率高=不稳定排后）。
- [kdj_k_d_distance](#kdj_k_d_distance)  — KDJ-K-D距离因子（(K-D)/|D|截面排名，正=多头动能排前）。
- [kdj_overbought_frac](#kdj_overbought_frac)  — KDJ超买时间占比因子（K>80分钟占比截面排名，高占比=强势超买排后）。
- [kdj_oversold_frac](#kdj_oversold_frac)  — KDJ超卖时间占比因子（K<20分钟占比截面排名，高占比=深度超卖排前=抄底信号）。
- [keltner_position_20](#keltner_position_20)  — 20日Keltner通道位置因子 (高位排后, 负向)
- [label_ret_10d](#label_ret_10d) 【标签】 — T+1开盘买入、T+11开盘卖出，10日目标收益。
- [label_ret_1d](#label_ret_1d) 【标签】 — T+1开盘买入、T+2开盘卖出，1日目标收益。
- [label_ret_20d](#label_ret_20d) 【标签】 — T+1开盘买入、T+21开盘卖出，20日目标收益。
- [label_ret_3d](#label_ret_3d) 【标签】 — T+1开盘买入、T+4开盘卖出，3日目标收益。
- [label_ret_5d](#label_ret_5d) 【标签】 — T+1开盘买入、T+6开盘卖出，5日目标收益。
- [large_order_timing_signal](#large_order_timing_signal)  — 大单时机信号因子，大单净买入/成交量×20日价格位置截面排名（低位大单流入=最佳买点排前）。
- [large_trade_intensity](#large_trade_intensity)  — 大单分钟集中度因子，前5%分钟成交额占全天比例截面排名（机构大额交易集中排前）。
- [last30_ret](#last30_ret)  — 尾盘30分钟收益因子：14:30开盘→收盘涨跌幅截面排名（负向）。
- [last30_vol_share](#last30_vol_share)  — 尾盘30分钟量占比因子：14:30-14:59成交额占全天比例截面排名（正向）。
- [lg_sm_divergence](#lg_sm_divergence)  — 大小单背离因子，(大单净买-小单净卖)/总成交额截面排名（机构买+散户卖=最佳组合排前）。
- [lhb_proxy_score_60](#lhb_proxy_score_60)  — 龙虎榜替代活跃度因子：60日(大波动×高换手×大单高参与)事件次数截面排名（博弈票活跃度排前）。
- [liftoff_pulse_combo_20](#liftoff_pulse_combo_20)  — 资金脉冲起飞因子：amount_surge_count_20×momentum_10×drawdown_60截面排名。
- [limit_alternation_20](#limit_alternation_20)  — 涨跌停交替因子：20日内涨停与跌停事件数量的乘积截面排名（情绪极端反转排前）。
- [limit_board_streak_mean_60](#limit_board_streak_mean_60)  — 平均连板高度因子：60日封板天数/连板启动次数截面排名（历史连板惯性排前）。
- [limit_down_event_5](#limit_down_event_5)  — 跌停事件衰减因子：pct_chg≤−9.8%近似跌停事件后指数衰减（半衰期3日，恐慌延续排前）。
- [limit_down_rebound_10](#limit_down_rebound_10)  — 跌停后表现因子：10日内有跌停事件的股票其10日累计收益（跌停后反弹排前）。
- [limit_streak_volume_ratio](#limit_streak_volume_ratio)  — 连板放量结构因子：当前连板数(≤5)×当日量/近3日最大量截面排名（连板且量创新高排前）。
- [limit_up_event_5](#limit_up_event_5)  — 涨停事件衰减因子：pct_chg≥9.8%近似涨停事件后指数衰减（半衰期3日，强势延续排前）。
- [limit_up_fade_10](#limit_up_fade_10)  — 涨停后表现因子：10日内有涨停事件的股票其10日累计收益（涨停后延续排前）。
- [limit_up_open_fail_freq_20](#limit_up_open_fail_freq_20)  — 炸板频率因子：20日盘中触板(高点≥9.8%)但收盘未封板的天数占比截面排名（炸板频发排前）。
- [limit_up_vol_shrink_60](#limit_up_vol_shrink_60)  — 缩量涨停率因子：60日涨停日均量/非涨停日均量截面排名（负向，涨停放量分歧排后）。
- [liquidity_discount_factor](#liquidity_discount_factor)  — 流动性折价因子，(1-bp排名)×(1-turnover_20排名)截面排名（低估值+低流动性=流动性折价排前）。
- [liquidity_shock_20](#liquidity_shock_20)  — 流动性冲击因子：Amihud(20日均值)相对前20日的变化截面排名（负向，冲击放大排后）。
- [listing_age_heat](#listing_age_heat)  — 次新热度衰减因子：换手率×exp(−上市天数/500)截面排名（次新且换手高排前）。
- [log_amplitude_20](#log_amplitude_20)  — 对数变换20日振幅因子，log(amplitude_20+ε)截面排名（低波动排前）。
- [log_circ_mv](#log_circ_mv)  — 对数流通市值因子，负对数流通市值截面排名。
- [log_gk_vol](#log_gk_vol)  — 对数变换Garman-Klass波动率因子，log(gk_vol+ε)截面排名（低波动排前）。
- [log_high_low_volatility_20](#log_high_low_volatility_20)  — 对数变换20日高低波动率因子，log(high_low_volatility_20+ε)截面排名（低波动排前）。
- [log_hl_range_intraday](#log_hl_range_intraday)  — 对数变换日内高低价差因子，log(hl_range_intraday+ε)截面排名（低波动排前）。
- [log_intraday_high_low_volatility](#log_intraday_high_low_volatility)  — 对数变换日内高低波动率因子，log(intraday_high_low_volatility+ε)截面排名（低波动排前）。
- [log_mv](#log_mv)  — 对数市值（ln(total_mv)，0值置NaN）。
- [log_mv_chg_20d](#log_mv_chg_20d)  — 市值20日变化（log_mv 的20日diff）。
- [log_parkinson_vol](#log_parkinson_vol)  — 对数变换Parkinson波动率因子，log(parkinson_vol+ε)截面排名（低波动排前）。
- [log_ret_std_intraday](#log_ret_std_intraday)  — 对数变换日内收益标准差因子，log(ret_std_intraday+ε)截面排名（低波动排前）。
- [log_rv_10min](#log_rv_10min)  — 对数变换10分钟已实现波动率因子，log(rv_10min+ε)截面排名（低波动排前）。
- [log_rv_15min](#log_rv_15min)  — 对数变换15分钟已实现波动率因子，log(rv_15min+ε)截面排名（低波动排前）。
- [log_rv_30min](#log_rv_30min)  — 对数变换30分钟已实现波动率因子，log(rv_30min+ε)截面排名（低波动排前）。
- [log_rv_5min](#log_rv_5min)  — 对数变换5分钟已实现波动率因子，log(rv_5min+ε)截面排名（低波动排前）。
- [log_rv_60min](#log_rv_60min)  — 对数变换60分钟已实现波动率因子，log(rv_60min+ε)截面排名（低波动排前）。
- [log_rv_rolling_5d_std](#log_rv_rolling_5d_std)  — 对数变换5日RV波动率因子，log(rv_rolling_5d_std+ε)截面排名（低波动排前）。
- [log_total_mv](#log_total_mv)  — 对数总市值因子，负对数总市值（小市值排前）。
- [loss_probability_20](#loss_probability_20)  — 损失概率因子：20日负收益频率截面排名（负向，高损失概率排后）。
- [lower_shadow_ratio](#lower_shadow_ratio)  — 下影线比率因子，(min(open,close)-low)/(high-low+1e-9)截面排名（长下影=承接力强排前）。
- [lowvol_liftoff_combo_20](#lowvol_liftoff_combo_20)  — 低波量价起飞因子：parkinson_vol×volume_breakout_confirm_20×momentum_20截面排名。
- [lowvol_momentum_rs_20](#lowvol_momentum_rs_20)  — 低波×动量×RS复合因子：idio_vol_60×momentum_20×rs_60截面排名。
- [lowvol_quality_momentum_60](#lowvol_quality_momentum_60)  — 低波质量动量因子：parkinson_vol×momentum_60×momentum_stability_20_60截面排名。
- [lowvol_trend_efficiency_combo_20](#lowvol_trend_efficiency_combo_20)  — 低波×趋势效率复合因子：parkinson_vol×kama_efficiency_20截面排名（干净趋势排前）。
- [lunch_break_effect](#lunch_break_effect)  — 午间休市效应因子，下午开盘价/上午收盘价-1截面排名（高值=午间利好堆积）。
- [ma10_slope](#ma10_slope)  — MA10斜率因子：分钟MA10收盘相对开盘的斜率截面排名（MA10上倾排前）。
- [ma20_slope](#ma20_slope)  — MA20斜率因子（(ma20_close-ma20_open)/|ma20_open|截面排名，中期均线上移排前）。
- [ma30_slope](#ma30_slope)  — MA30斜率因子：分钟MA30收盘相对开盘的斜率截面排名（MA30上倾排前）。
- [ma5_ma10_gap](#ma5_ma10_gap)  — MA5/MA10乖离因子：分钟MA5相对MA10的偏离截面排名（短均线在上排前）。
- [ma5_slope](#ma5_slope)  — MA5斜率因子（(ma5_close-ma5_open)/|ma5_open|截面排名，均线上移=趋势向上排前）。
- [ma_alignment_score](#ma_alignment_score)  — MA均线排列因子（ma5>ma10>ma20>ma30>ma60的满足数量截面排名，多头排列排前）。
- [ma_bull_bear_ratio](#ma_bull_bear_ratio)  — MA多空比率因子（所有MA对中多头排列对数/总对数截面排名，高比率=全面多头排前）。
- [ma_convergence](#ma_convergence)  — MA均线收敛因子（|ma5-ma60|/|ma60|截面排名，短长均线距离近=收敛排前=变盘信号）。
- [ma_convergence_20_60](#ma_convergence_20_60)  — 均线收敛因子，20日均线与60日均线的距离比率截面排名。
- [ma_cross_count](#ma_cross_count)  — MA均线交叉因子（ma5上穿/下穿ma20次数截面排名，交叉多=方向切换频繁排后）。
- [ma_curvature](#ma_curvature)  — MA曲率因子（ma5_slope-ma20_slope截面排名，正曲率=短周期加速快于中期排前）。
- [ma_dispersion](#ma_dispersion)  — MA离散度因子（各均线间距的变异系数截面排名，低离散=均线收敛=变盘前兆排前）。
- [ma_distance_20_60](#ma_distance_20_60)  — 20日-60日均线距离因子 (趋势一致性)
- [ma_distance_5_20](#ma_distance_5_20)  — 5日-20日均线距离因子 (乖离)
- [ma_distance_5_60](#ma_distance_5_60)  — 5日-60日均线距离因子 (中期乖离)
- [ma_rsi_divergence](#ma_rsi_divergence)  — MA-RSI背离因子（ma5与rsi的日内相关性截面排名，负相关=量价背离排前=反转信号）。
- [macd_acceleration](#macd_acceleration)  — MACD柱加速度因子（(macd_close-macd_open)/|macd_open|截面排名，加速扩张排前）。
- [macd_bar_energy](#macd_bar_energy)  — MACD柱能量因子：日内|macd|均值截面排名（负向，柱体活跃排后）。
- [macd_bar_sign_change](#macd_bar_sign_change)  — MACD柱翻红翻绿因子（日内macd柱正负切换次数截面排名，频繁切换=趋势不稳排后）。
- [macd_daily_consistency](#macd_daily_consistency)  — MACD日内一致性因子（日内macd柱>0的分钟占比截面排名，高一致性=趋势明确排前）。
- [macd_daily_hist_5d](#macd_daily_hist_5d)  — 日频MACD柱5日变化因子：MACD(12,26,9)柱状图的5日变化截面排名（动能增强排前）。
- [macd_daily_range](#macd_daily_range)  — MACD日内振幅因子（macd_max-macd_min截面排名，振幅大=波动剧烈排后）。
- [macd_dif_slope](#macd_dif_slope)  — MACD-DIF日内斜率因子（(dif_close-dif_open)/|dif_open|截面排名，DIF上升=强势排前）。
- [macd_extreme_ratio](#macd_extreme_ratio)  — MACD极端值占比因子（|macd|>2σ的分钟占比截面排名，高极端值=异常行情排后）。
- [macd_kdj_alignment](#macd_kdj_alignment)  — MACD-KDJ一致性因子（sign(macd)*sign(K-D)截面排名，双指标同向=趋势确认排前）。
- [macd_price_divergence](#macd_price_divergence)  — MACD价格背离因子（MA5趋势与macd柱趋势的日内相关性截面排名，负相关=背离信号排前）。
- [macd_rsi_combo](#macd_rsi_combo)  — MACD-RSI组合因子（sign(macd)*(rsi-50)截面排名，双动量指标同向排前）。
- [macd_signal_cross](#macd_signal_cross)  — MACD金叉死叉信号因子（日内收盘macd柱>0且dif>dea截面排名，金叉状态排前）。
- [macd_trend_strength](#macd_trend_strength)  — MACD趋势强度因子（(dif-dea)/|dea|截面排名，趋势强度高排前）。
- [macd_zero_cross](#macd_zero_cross)  — MACD零轴穿越因子（DIF穿越零轴的频率截面排名，频繁穿越=趋势混乱排后）。
- [margin_balance_20d](#margin_balance_20d)  — 融资余额20日变化率，反映中期杠杆资金趋势。稳步增长=持续看多共识，排名高。
- [margin_balance_5d](#margin_balance_5d)  — 融资余额5日变化率，反映杠杆资金短期流入/流出速度。余额增长=杠杆做多，排名高。
- [margin_balance_ma_divergence](#margin_balance_ma_divergence)  — 融资余额偏离20日均线幅度，极端偏离预示均值回归。正向偏离=可能超买，排名居中。
- [margin_balance_volatility_20d](#margin_balance_volatility_20d)  — 融资余额20日波动率(变异系数)，反映杠杆资金稳定性。低波动=方向一致，排名高(取反)。
- [margin_buy_momentum_5d](#margin_buy_momentum_5d)  — 融资买入5日均值相对余额动量。持续买入=信号可靠，排名高。
- [margin_buy_pressure](#margin_buy_pressure)  — 融资买入压力比=融资买入/(融资买入+融资偿还)。>0.5=买入意愿强于偿还，排名高。
- [margin_buyer_avg_cost_premium](#margin_buyer_avg_cost_premium)  — 融资盘成本溢价因子：现价相对近20日融资买入加权平均成本截面排名（融资盘浮盈排前）。
- [margin_chg_abs_5d](#margin_chg_abs_5d)  — 融资余额5日变化率（rzye 5日pct_change，PIT 平移后）。
- [margin_chg_rel_ind_5d](#margin_chg_rel_ind_5d)  — 融资余额5日变化率行业相对（减去行业等权均值，PIT 平移后）。
- [margin_chip_cost_gap](#margin_chip_cost_gap)  — 融资盘-筹码成本差因子：融资盘平均成本/市场筹码平均成本−1截面排名（负向，杠杆盘高位接盘排后）。
- [margin_flow_asymmetry_10d](#margin_flow_asymmetry_10d)  — 10日累计融资净买入/累计总交易额，资金流向方向持续性。持续净买入=排名高。
- [margin_leverage_change_20d](#margin_leverage_change_20d)  — 融资杠杆变化因子：融资余额/流通市值占比的20日变化截面排名（杠杆资金加仓排前）。
- [margin_leverage_trend_10d](#margin_leverage_trend_10d)  — 融资融券总余额10日变化率。总杠杆增加=风险偏好提升，排名高。
- [margin_net_flow_ratio](#margin_net_flow_ratio)  — 融资净流入相对余额比=(融资买入-融资偿还)/融资余额，标准化净流量强度。
- [margin_price_resonance_20](#margin_price_resonance_20)  — 融资流入×动量复合因子：margin_net_flow_ratio×momentum_20截面排名（杠杆加仓且上涨排前）。
- [margin_proxy_ttm](#margin_proxy_ttm)  — 净利率代理因子：ps_ttm/pe_ttm(市值恒等式=净利/营收)截面排名（高净利率排前）。
- [margin_repay_deceleration](#margin_repay_deceleration)  — 融资偿还额5日变化率取反。偿还减速=空方力量减弱，排名高。
- [margin_repay_shock](#margin_repay_shock)  — 融资偿还冲击=当日偿还额/20日均偿还额取反。突然放大=恐慌平仓，排名低。
- [margin_trend_combo_20](#margin_trend_combo_20)  — 杠杆趋势复合因子：margin_net_flow_ratio×momentum_20×drawdown_60截面排名。
- [margin_value_combo_20](#margin_value_combo_20)  — 杠杆价值复合因子：margin_net_flow_ratio×bp×log_circ_mv截面排名。
- [margin_velocity](#margin_velocity)  — 融资周转速度=(融资买入+融资偿还)/融资余额。高速度=投机性强，排名高。
- [market_beta_change_20](#market_beta_change_20)  — 贝塔变化因子（20日贝塔-60日贝塔=系统性风险暴露的短期变化）。
- [market_cap_concentration_20d](#market_cap_concentration_20d)  — 市值集中度因子，log_total_mv的20日波动率截面排名。
- [market_regime_sensitivity_60](#market_regime_sensitivity_60)  — 市场状态敏感度因子：60日下跌市均收益/|上涨市均收益|（下跌市抗跌排前）。
- [marubozu_ratio_10d](#marubozu_ratio_10d)  — 光头光脚阳线频率因子，近10日实体阳线(上下影极短)天数截面排名。
- [mavol5_slope](#mavol5_slope)  — 成交量MA5斜率因子（(mavol5_close-mavol5_open)/|mavol5_open|截面排名，成交量均线上移=放量排前）。
- [mavol_expansion](#mavol_expansion)  — 成交量扩张因子（mavol5_close/mavol5_open-1截面排名，成交量扩张=活跃度升排前）。
- [mavol_ratio_signal](#mavol_ratio_signal)  — 成交量均线比率因子（mavol5/mavol10截面排名，比率>1=短期放量排前）。
- [mavol_ratio_std](#mavol_ratio_std)  — 成交量比率波动因子（std(mavol5/mavol10)截面排名，比率波动大=成交量不稳定排后）。
- [mavol_ratio_trend](#mavol_ratio_trend)  — 成交量比率趋势因子（mavol比率的日内时间相关性截面排名，比率上升=持续放量排前）。
- [mavol_stability](#mavol_stability)  — 成交量稳定性因子（std(mavol5)/mean(mavol5)截面排名，成交量不稳定排后）。
- [max_consecutive_loss_20](#max_consecutive_loss_20)  — 最长连亏因子：20日内最长连续亏损天数截面排名（负向，长连亏排后）。
- [max_ret_intraday](#max_ret_intraday)  — 最大5分钟收益因子，日内最大|ret_5min|截面排名（取负向=极端波动排后）。捕获日内最剧烈的价格冲击。
- [max_vol_day_contribution_20](#max_vol_day_contribution_20)  — 单日脉冲主导度：20日内最大量日的|收益|占20日|收益|总和比例截面排名。
- [medium_order_flow](#medium_order_flow)  — 中单资金流因子，中单净买入/总成交额截面排名。
- [mf_amount_vol_divergence](#mf_amount_vol_divergence)  — 资金流量价背离因子，（大单净买入额占比 - 大单净买入量占比）截面排名。
- [mf_amount_weighted_direction](#mf_amount_weighted_direction)  — 金额加权方向复合因子，四档订单方向信号按金额占比加权求和截面排名。
- [mf_avg_trade_price_momentum](#mf_avg_trade_price_momentum)  — 成交均价动量因子（VWAP(大单)/VWAP(小单)的5日变化率截面排名，比率上升=机构买入紧迫度升排前）。
- [mf_big_order_net_kurt_20](#mf_big_order_net_kurt_20)  — 大单净流入峰度因子：20日(大单+超大单净流入)峰度截面排名（脉冲式建仓排前）。
- [mf_big_order_ratio](#mf_big_order_ratio)  — 大单+特大单净买入率因子，(特大+大净买入)/总成交额截面排名。
- [mf_big_order_stability_20d](#mf_big_order_stability_20d)  — 20日大单净买入率稳定性因子 (高稳定排前)。
- [mf_big_order_turnover_ratio](#mf_big_order_turnover_ratio)  — 大额订单成交占比因子，(大单+特大单成交量)/总成交量截面排名。
- [mf_big_order_vol_ratio](#mf_big_order_vol_ratio)  — 大单+特大单成交量占比因子，基于成交量口径的机构行为度量。
- [mf_big_small_convergence_20d](#mf_big_small_convergence_20d)  — 20日大单/小单收敛因子 (大单趋势-小单趋势)。
- [mf_big_small_divergence](#mf_big_small_divergence)  — 大小单背离因子，(大单净买-小单净买)/总成交额截面排名。
- [mf_cumulative_flow_20d](#mf_cumulative_flow_20d)  — 20日累计主力净流入率因子。
- [mf_elg_order_ratio](#mf_elg_order_ratio)  — 特大单净买入率因子，(特大单净买入额/总成交额)截面排名。
- [mf_elg_small_divergence](#mf_elg_small_divergence)  — 特大单与小单背离因子，(特大单净买入-小单净买入)/总成交额截面排名。
- [mf_extra_large_sell_pressure](#mf_extra_large_sell_pressure)  — 超大单卖出占比取反。超大单卖出集中=机构出货，排名低。
- [mf_flow_acceleration_5d](#mf_flow_acceleration_5d)  — 主力资金净流入加速度=5日净流入变化率。加速流入=增量资金积极，排名高。
- [mf_flow_acceleration_ext](#mf_flow_acceleration_ext)  — 资金流加速度因子，主力净流入率的5日变化截面排名（流入在加速=趋势加强排前）。
- [mf_flow_continuity](#mf_flow_continuity)  — 主力资金连续流入天数因子，统计各股票连续净流入天数截面排名。
- [mf_flow_factor_momentum_20](#mf_flow_factor_momentum_20)  — 主力资金流因子20日动量 (资金态度变化)。
- [mf_flow_reversal_20d](#mf_flow_reversal_20d)  — 20日主力资金反转因子 (从流出的流出反转为流入)。
- [mf_flow_stability_20d](#mf_flow_stability_20d)  — 主力资金流向20日稳定性=连续同向天数占比。频繁转向=信号不可靠，排名低。
- [mf_flow_streak_5d](#mf_flow_streak_5d)  — 主力净流入天数频率（5日内 net_mf_amount>0 的天数占比）。
- [mf_flow_volatility_20d](#mf_flow_volatility_20d)  — 资金流波动率因子，-(主力净流入20日标准差)截面排名（资金流稳定=有序建仓排前）。
- [mf_large_order_avg_price](#mf_large_order_avg_price)  — 大单+超大单成交均价相对总成交均价。高比值=机构高价成交(拉升建仓),排名高。
- [mf_large_order_net_5d](#mf_large_order_net_5d)  — 大单净流入5日均值/总成交额。持续大单净流入=机构持续吸筹，排名高。
- [mf_large_vol_net_5d](#mf_large_vol_net_5d)  — 大单净买入量占比5日均值。持续的大单量净流入=机构持续建仓(量能角度)。
- [mf_md_order_vol_ratio](#mf_md_order_vol_ratio)  — 中单成交量占比因子（(buy_md_vol+sell_md_vol)/总成交量截面排名）。
- [mf_mid_order_ratio](#mf_mid_order_ratio)  — 中单净买入率因子，(中单净买入额/总成交额)截面排名。
- [mf_net_amount_intensity](#mf_net_amount_intensity)  — 主力资金净额强度因子，net_mf_amount/流通市值截面排名。
- [mf_net_inflow_5d](#mf_net_inflow_5d)  — 5日累计主力净流入率因子截面排名。
- [mf_net_inflow_ratio](#mf_net_inflow_ratio)  — 主力资金净流入率因子，主力净流入额/成交额截面排名。
- [mf_net_inflow_trend_5d](#mf_net_inflow_trend_5d)  — 5日主力净流入率趋势因子，近5日净流入率线性回归斜率截面排名。
- [mf_net_inflow_volatility_20d](#mf_net_inflow_volatility_20d)  — 20日主力净流入率波动率因子，(负向排名)净流入率波动越大排名越低。
- [mf_net_persistent_5d](#mf_net_persistent_5d)  — 主力净流入持续性因子，5日主力净流入为正的天数截面排名（持续净流入=坚定看多排前）。
- [mf_net_vol_intensity](#mf_net_vol_intensity)  — 主力净流入量/总成交量因子，成交量口径的资金净流向强度。
- [mf_net_vol_ma_divergence](#mf_net_vol_ma_divergence)  — 净成交量均线偏离因子（净成交量/(净成交量20日均值)-1截面排名，当前>均值=加速排前）。
- [mf_net_vol_ratio_5d](#mf_net_vol_ratio_5d)  — 净买入成交量占比5日变化。量能角度净流入的边际改善,排名高。
- [mf_net_vol_trend_3d](#mf_net_vol_trend_3d)  — 3日净成交量趋势因子（net_mf_vol的3日变化率截面排名，净流入加速排前）。
- [mf_open_close_divergence_10d](#mf_open_close_divergence_10d)  — 10日资金流偏离波动因子：净流入率相对其5日均线的偏离的10日波动率截面排名（偏离剧烈=资金态度摇摆排前）。
- [mf_order_concentration](#mf_order_concentration)  — 主力资金订单集中度=大单+超大单占比，Herfindahl指数式度量。高集中度=机构主导，排名高。
- [mf_order_size_entropy](#mf_order_size_entropy)  — 订单规模分布的熵因子（高值排前），度量资金参与结构的多样性。
- [mf_rel_ind_ma20d](#mf_rel_ind_ma20d)  — 主力净流入行业相对强度（行业内z-score的20日均值）。
- [mf_rel_ind_ma5d](#mf_rel_ind_ma5d)  — 主力净流入行业相对强度（行业内z-score的5日均值）。
- [mf_retail_dominance](#mf_retail_dominance)  — 散户交易占比=(小单买入+小单卖出)/总成交额取反。高散户占比=噪音交易多，排名低。
- [mf_sm_order_vol_ratio](#mf_sm_order_vol_ratio)  — 小单成交量占比因子（散户成交量占比截面排名，低占比=机构化排前）。
- [mf_small_order_ratio](#mf_small_order_ratio)  — 小单净买入率因子（负值=散户净卖出，排名高=散户流出多）。
- [mf_smart_dumb_divergence](#mf_smart_dumb_divergence)  — 聪明钱vs散户分歧=(大单净买/大单总额)-(小单净买/小单总额)。正值=机构买散户卖，排名高。
- [mf_tier_net_spread_20](#mf_tier_net_spread_20)  — 大中小单分歧度因子：20日四档(小/中/大/超大)净占比极差截面排名（负向，多空分歧排后）。
- [mf_vol_amount_corr_20](#mf_vol_amount_corr_20)  — 量额相关性因子（20日各档vol/amount日变化率相关性截面排名，低相关=价格偏差大排后）。
- [mf_vol_amount_divergence](#mf_vol_amount_divergence)  — 资金流量的量-额背离：净买入量占比-净买入额占比。正=量大但额小(低价成交),负=量小但额大(高价成交)。
- [mf_vol_concentration_large](#mf_vol_concentration_large)  — 大单+超大单成交量占比。高占比=机构交易量集中,排名高。
- [mf_vol_retail_ratio](#mf_vol_retail_ratio)  — 小单成交量占比取反。小单量占比高=散户活跃,噪音大,排名低。
- [mf_vol_tier_balance](#mf_vol_tier_balance)  — 各规模成交量平衡度因子（各档vol占比两两差异绝对值之和截面排名，均衡=健康排前）。
- [mfi_14](#mfi_14)  — 14日MFI资金流量指标：正负量流比截面排名（资金流入推动排前）。
- [microstructure_efficiency](#microstructure_efficiency)  — 微观结构效率因子，(rv_5min/parkinson_vol排名)截面排名（日内效率高=信息消化快排前）。
- [min_bar_gap_freq_20](#min_bar_gap_freq_20)  — 分钟跳空频率因子：20日(|分钟开盘/前分钟收盘−1|>0.2%)占比均值截面排名（负向，盘口断档排后）。
- [min_boll_width_std_20](#min_boll_width_std_20)  — 分钟布林带宽波动因子：20日(分钟带宽日内标准差)均值截面排名（负向，带宽反复扩张挤压排后）。
- [min_expand_bull_frac_20](#min_expand_bull_frac_20)  — 分钟放量多头占比因子：20日(放量且多头排列分钟占比)均值截面排名（量价趋势三线确认排前）。
- [min_j_overbought_frac_20](#min_j_overbought_frac_20)  — 分钟KDJ超买占比因子：20日(J>100分钟占比)均值截面排名（负向，盘中反复冲顶排后）。
- [min_limit_touch_frac_20](#min_limit_touch_frac_20)  — 分钟触板密度因子：20日(分钟涨幅≥9.8%占比)均值截面排名（封板维持时间长排前）。
- [min_ma_alignment_frac_20](#min_ma_alignment_frac_20)  — 分钟均线多头排列占比因子：20日(ma5>ma10>ma20>ma30分钟占比)均值截面排名（日内趋势稳固排前）。
- [min_macd_hist_area_20](#min_macd_hist_area_20)  — 分钟MACD柱面积因子：20日(Σ分钟MACD/当日收盘价)均值截面排名（日内动能净值排前）。
- [min_ret_max](#min_ret_max)  — 1分钟收益最大值因子：日内最大分钟涨幅的截面排名（负向，急拉排后）。
- [min_ret_min](#min_ret_min)  — 1分钟收益最小值因子：日内最大分钟跌幅的截面排名（负向绝对值，深跌排后）。
- [min_ret_std](#min_ret_std)  — 1分钟收益标准差因子：日内分钟收益波动的截面排名（负向，波动大排后）。
- [min_rsi_extreme_frac_20](#min_rsi_extreme_frac_20)  — 分钟RSI极值占比因子：20日(RSI>80或<20分钟占比)均值截面排名（负向，情绪烈度高排后）。
- [min_rsi_overbought_expand_20](#min_rsi_overbought_expand_20)  — 分钟超买放量占比因子：20日(RSI>70且放量分钟占比)均值截面排名（负向，追高放量排后）。
- [min_shrink_bull_frac_20](#min_shrink_bull_frac_20)  — 分钟缩量多头占比因子：20日(缩量但多头排列分钟占比)均值截面排名（负向，无量上涨排后）。
- [min_vwap_dev_std](#min_vwap_dev_std)  — VWAP贴合度因子：20日(分钟价对当日VWAP偏离的标准差)均值截面排名（负向，价格围绕VWAP剧烈摆动排后）。
- [minute_ret_vol_corr](#minute_ret_vol_corr)  — 分钟量价相关因子：日内分钟收益与分钟量的皮尔逊相关截面排名（量价同步排前）。
- [momentum_10](#momentum_10)  — 10日后复权动量因子（自建后复权基座，无除权失真），截面排名。
- [momentum_20](#momentum_20)  — 20日后复权动量因子（自建后复权基座，无除权失真），截面排名。
- [momentum_3](#momentum_3)  — 3 日收益（收盘价 3 日变化率）。
- [momentum_5](#momentum_5)  — 5日后复权动量因子（自建后复权基座，无除权失真），截面排名。
- [momentum_60](#momentum_60)  — 60日后复权动量因子（自建后复权基座，无除权失真），截面排名。
- [momentum_accel_20_60](#momentum_accel_20_60)  — 动量加速度：momentum_20−momentum_60。短中期动量差=趋势加速/减速。
- [momentum_accel_60_120](#momentum_accel_60_120)  — 60/120日动量加速度因子：(60日动量−120日动量)截面排名（中期趋势加速排前）。
- [momentum_high_proximity_combo_20](#momentum_high_proximity_combo_20)  — 动量×52周高接近度复合因子：momentum_60×price_to_52w_high截面排名。
- [momentum_liquidity_resonance_20](#momentum_liquidity_resonance_20)  — 动量×高流动性：momentum_20×amihud_intraday。流动性好的股票动量更可靠。
- [momentum_lowvol_combo_20](#momentum_lowvol_combo_20)  — 动量×低波动：momentum_20×parkinson_vol。低波动趋势股的风险调整后动量。
- [momentum_rs_resonance_20](#momentum_rs_resonance_20)  — 动量×RS共振因子：momentum_20×rs_60截面排名（绝对与相对动量双强排前）。
- [momentum_stability_20_60](#momentum_stability_20_60)  — 20/60日动量趋势差因子（短期动量-中期动量=趋势加速度），截面排名。
- [momentum_stability_combo_60](#momentum_stability_combo_60)  — 动量×稳定性复合因子：momentum_60×momentum_stability_20_60截面排名。
- [momentum_volume_divergence_20](#momentum_volume_divergence_20)  — 量价背离：momentum_20−volume_momentum_5。价升量缩(背离)排名高=缺乏资金确认。
- [momentum_volume_resonance_20](#momentum_volume_resonance_20)  — 动量×换手共振：momentum_20与turnover_20双高排名。放量上涨=资金确认的趋势。
- [moneyflow_momentum_resonance_20](#moneyflow_momentum_resonance_20)  — 资金确认动量：mf_net_inflow_ratio×momentum_20。主力净流入+上涨=趋势有资金背书。
- [moneyflow_reversal_divergence_5](#moneyflow_reversal_divergence_5)  — 资金×反转背离：mf_net_inflow_ratio−short_term_reversal_5。主力流入但股价超跌=潜在反转。
- [multi_horizon_momentum_combo_20](#multi_horizon_momentum_combo_20)  — 多周期动量共振因子：momentum_5×momentum_10×momentum_20截面排名。
- [multi_indicator_extreme](#multi_indicator_extreme)  — 多指标极端值因子（5个标准化指标的|z-score|均值截面排名，多指标同时极端=变盘排后）。
- [net_mf_amount_intensity](#net_mf_amount_intensity)  — 成交量基础主力净流入因子，(主力净流入量/总成交量)截面排名。
- [net_mf_amount_momentum_5d](#net_mf_amount_momentum_5d)  — 主力资金净额5日动量因子，net_mf_amount的5日变化截面排名。
- [net_mf_flow_persistence](#net_mf_flow_persistence)  — 主力资金净流入持续性因子，近5日净流入为正的天数截面排名。
- [net_turnover_rate_20](#net_turnover_rate_20)  — 净换手率因子：20日(主动买量−主动卖量)/自由流通股本截面排名（净买入比例高排前）。
- [new_high_60_event](#new_high_60_event)  — 60日新高事件衰减因子：复权价创60日新高事件后指数衰减（半衰期5日，突破强势排前）。
- [new_high_frequency_60](#new_high_frequency_60)  — 60日新高频率因子：60日内创新高天数占比截面排名（趋势强势频率排前）。
- [new_low_60_event](#new_low_60_event)  — 60日新低事件衰减因子：复权价创60日新低事件后指数衰减（半衰期5日，破位风险排前）。
- [obv_divergence_20](#obv_divergence_20)  — OBV量价背离因子：价格动量排名−OBV动量排名的背离截面排名（价涨量缩背离排前）。
- [obv_slope_20](#obv_slope_20)  — 20日OBV斜率因子：OBV的20日变化截面排名（量能净流入加速排前）。
- [oi_divergence_intensity](#oi_divergence_intensity)  — 隔夜日内背离强度因子，(close-open)-(open-pre_close)截面排名（正=日内强化跳空方向排前）。
- [one_word_limit_down_freq_20](#one_word_limit_down_freq_20)  — 一字跌停频率因子：20日一字跌停天数占比截面排名（恐慌锁死排前，负向排名）。
- [one_word_limit_up_freq_20](#one_word_limit_up_freq_20)  — 一字涨停频率因子：20日一字板(全天无波动且涨停)天数占比截面排名（一字连板强势排前）。
- [open_30_momentum](#open_30_momentum)  — 开盘30分钟动量因子：开盘半小时(09:30-10:00)涨跌幅截面排名（开盘强势排前）。
- [open_30_range_share](#open_30_range_share)  — 开盘振幅占比因子：开盘30分钟振幅占全天振幅比例截面排名（负向，早盘大幅博弈排后）。
- [open_5min_momentum](#open_5min_momentum)  — 开盘5分钟动量因子，前5分钟收益截面排名（开盘强势=隔夜利好排前）。
- [open_auction_intensity](#open_auction_intensity)  — 开盘强度因子，(开盘价-昨收)/昨收 × 开盘量/20日均量截面排名（跳空+放量=强信号排前）。
- [open_auction_ret](#open_auction_ret)  — 集合竞价收益率因子，(开盘价/前日收盘-1)截面排名。隔夜信息冲击的直接度量。
- [open_close_momentum_gap](#open_close_momentum_gap)  — 首尾动量差因子：开盘30分钟动量−尾盘30分钟动量截面排名（负向，冲高回落排后）。
- [open_price_shock](#open_price_shock)  — 开盘跳空幅度取正。大幅跳空=隔夜信息冲击强→短期反转概率高，排名高。
- [open_volume_share](#open_volume_share)  — 开盘量能占比因子：开盘30分钟成交量占全天比例截面排名（负向，早盘情绪交易排后）。
- [order_concentration](#order_concentration)  — 订单集中度因子，-(中单+小单)/总成交截面排名（大单+超大单占比高=机构主导排前）。
- [order_size_concentration](#order_size_concentration)  — 订单规模集中度因子，四个规模档的成交额HHI截面排名（集中度高=机构交易主导排前）。
- [order_size_ratio_change](#order_size_ratio_change)  — 订单规模比变化因子，(大单+特大)/总成交的5日变化截面排名（大单占比提升=机构参与加深排前）。
- [outside_bar_count_20](#outside_bar_count_20)  — 吞没/突破频率因子：20日收盘突破昨日全天区间(吞没)天数截面排名（强吞没走势排前）。
- [overnight_cum_20d](#overnight_cum_20d)  — 隔夜收益20日复利累计。
- [overnight_gap](#overnight_gap)  — 隔夜跳空因子，-(open-pre_close)/pre_close截面排名（跳空高开=反转信号排后）。
- [overnight_gap_momentum](#overnight_gap_momentum)  — 隔夜跳空因子，(open-pre_close)/pre_close截面排名（高开排前=利好消化未完）。
- [overnight_gap_vol_20](#overnight_gap_vol_20)  — 20日隔夜跳空波动率因子 (高波动排后, 负向)。
- [overnight_intraday_divergence_daily](#overnight_intraday_divergence_daily)  — 隔夜-日内背离因子，overnight_gap - intraday_ret截面排名。
- [overnight_intraday_ratio_20d](#overnight_intraday_ratio_20d)  — 隔夜/日内收益强度比（20日均值之比，分母取绝对值）。
- [overnight_ma5](#overnight_ma5)  — 隔夜收益5日均值因子：open/pre_close-1 的5日滚动均值（隔夜动能排前）。
- [overnight_ma_20d](#overnight_ma_20d)  — 隔夜收益均值（open/pre_close−1 的20日均值）。
- [overnight_ma_5d](#overnight_ma_5d)  — 隔夜收益均值（open/pre_close−1 的5日均值）。
- [overnight_ma_60d](#overnight_ma_60d)  — 隔夜收益均值（open/pre_close−1 的60日均值）。
- [overnight_minus_intraday](#overnight_minus_intraday)  — 隔夜−日内收益差因子：open/pre_close-1 与 close/open-1 之差（隔夜强于日内排前）。
- [overnight_return_share_20](#overnight_return_share_20)  — 隔夜收益占比因子：20日|隔夜跳空|占(跳空+日内)总波动的比例截面排名（隔夜驱动排前）。
- [overnight_sign_consistency_20d](#overnight_sign_consistency_20d)  — 隔夜方向一致性（20日内 overnight>0 的天数占比）。
- [overnight_skewness_20d](#overnight_skewness_20d)  — 20日隔夜收益偏度的绝对值取反。极端偏度=信息冲击不稳定，排名低。
- [overnight_std_20d](#overnight_std_20d)  — 隔夜收益波动（overnight 的20日std）。
- [overnight_std_5d](#overnight_std_5d)  — 隔夜收益标准差（5 日）。
- [panic_selling_ratio_60](#panic_selling_ratio_60)  — 放量下跌占比因子：60日放量(vol>1.5×20日均量)且下跌天数/放量天数（恐慌抛售排前）。
- [parkinson_vol](#parkinson_vol)  — Parkinson波动率估计因子，基于日内最高最低价的ln(H/L)/sqrt(4ln2)截面排名（低波排前）。
- [path_efficiency](#path_efficiency)  — 价格路径效率因子，|收盘-开盘|/(最高-最低)截面排名（高效=趋势性强排前）。
- [pb_change_20d](#pb_change_20d)  — 市净率变化因子：pb的20日变化率截面排名（负向，估值抬升排后）。
- [pb_industry_adjusted](#pb_industry_adjusted) 【已禁用】 — 行业调整市净率因子，1/PB在同THS行业内截面排名（低PB行业内排前）。
- [pb_turnover_regime](#pb_turnover_regime)  — PB-换手率状态因子（低PB+高换手=价值重估排前）。
- [pe_pb_divergence](#pe_pb_divergence)  — PB与PE百分位排名差。正偏离=隐含ROE较高，负偏离=隐含ROE较低。
- [pe_ttm_absolute](#pe_ttm_absolute)  — PE_TTM原始值截面排名取负，低PE=价值信号（当期所有股票的截面比较）。
- [pe_ttm_change_20d](#pe_ttm_change_20d)  — PE_TTM 20日变化率取反，估值收缩=价值改善，估值扩张=均值回归风险。
- [pm_hl_range_intraday](#pm_hl_range_intraday)  — 下午振幅因子，下午最高/下午最低-1截面排名（取负向=下午高振幅=尾盘博弈排后）。
- [pm_macd_trend](#pm_macd_trend)  — 午盘MACD趋势因子（下午MACD与时间的相关性截面排名，正相关=午盘动能持续排前）。
- [pm_momentum_intraday](#pm_momentum_intraday)  — 下午动量因子，下午收盘/下午开盘-1截面排名（下午走强=买盘持续排前）。
- [pm_reversal_signal](#pm_reversal_signal)  — 下午反转信号因子，-(下午收益/上午收益)截面排名（上午涨+下午跌=盘尾反转排后）。
- [pm_rsi_trend](#pm_rsi_trend)  — 午盘RSI趋势因子（下午RSI与时间的相关性截面排名，正相关=午盘动量持续排前）。
- [pos_rv_ratio](#pos_rv_ratio)  — 正收益波动占比因子，正5分钟收益平方和/总RV²截面排名。上涨驱动的波动占比。
- [ppo_signal_12_26_9](#ppo_signal_12_26_9)  — PPO信号差因子：百分比价格振荡器(EMA12−EMA26)/EMA26减去其9日EMA截面排名（动能反转确认排前）。
- [price_distance_from_52w_low](#price_distance_from_52w_low)  — 距52周低点距离因子：(adj−252日最低)/252日最低截面排名（远离年内低点排前）。
- [price_impact_asymmetry](#price_impact_asymmetry)  — 价格冲击不对称因子，-(上涨冲击/下跌冲击)截面排名（取负向=不对称=上涨费劲排后）。
- [price_impact_intraday](#price_impact_intraday)  — 价格冲击因子（Kyle's Lambda），|收盘-开盘|/成交量截面排名（取负向=高冲击排后）。
- [price_position_20d](#price_position_20d)  — 20日价格位置因子，(close-20日最低)/(20日最高-20日最低)截面排名。
- [price_position_60](#price_position_60)  — 60日价格位置，(close-60d_low)/(60d_high-60d_low) 截面排名。
- [price_to_52w_high](#price_to_52w_high)  — 52周高点接近度因子（close/252日最高收盘价-1），截面排名。
- [price_vs_ma10_deviation](#price_vs_ma10_deviation)  — 价格对MA10偏离因子：close相对分钟MA10的偏离截面排名（负向绝对值，大幅偏离排后）。
- [price_vs_ma20_deviation](#price_vs_ma20_deviation)  — 价格偏离MA20因子（(close-ma20)/|ma20|截面排名，偏离大=均值回归压力大排后）。
- [price_vs_ma30_deviation](#price_vs_ma30_deviation)  — 价格对MA30偏离因子：close相对分钟MA30的偏离截面排名（负向绝对值，大幅偏离排后）。
- [price_vs_ma60_deviation](#price_vs_ma60_deviation)  — 价格偏离MA60因子（(close-ma60)/|ma60|截面排名，偏离大=长期趋势偏离排后）。
- [ps_ttm_momentum_20d](#ps_ttm_momentum_20d)  — PS_TTM 20日变化率取反，PS下降=变便宜，是价值改善信号。
- [ps_ttm_rank](#ps_ttm_rank)  — 市销率TTM因子，ps_ttm截面排名（低市销率排前）。
- [ps_ttm_sector_neutral](#ps_ttm_sector_neutral) 【已禁用】 — 行业中性市销率因子，(ps_ttm排名-行业均值排名)截面排名。
- [psy_12](#psy_12)  — 心理线因子：12日上涨天数占比截面排名（多头情绪浓度排前）。
- [quality_liquidity_combo_20](#quality_liquidity_combo_20)  — 质量流动性三合因子：amihud_intraday×parkinson_vol×idio_vol_60截面排名。
- [range_position_20d](#range_position_20d)  — 20日区间位置（close 在20日 high-low 区间内的位置）。
- [range_position_60d](#range_position_60d)  — 60日区间位置（close 在60日 high-low 区间内的位置）。
- [range_rv_ratio](#range_rv_ratio)  — 振幅波动比因子，(high/low-1)/rv_5min截面排名（取负向=高比值排后）。跳成分相对于连续波动的比例。
- [range_vol_ratio_20](#range_vol_ratio_20)  — 区间波动比因子：20日高低价区间/20日收益波动截面排名（负向，极端区间排后）。
- [realized_spread_5min](#realized_spread_5min)  — 已实现价差因子，5分钟|ret|均值截面排名（取负向=高价差=高交易成本排后）。
- [rebound_from_low_20](#rebound_from_low_20)  — 20日低点反弹幅度因子：当前价相对20日低点的涨幅截面排名（强劲反弹排前）。
- [rel_log_mv_ind](#rel_log_mv_ind)  — 行业内相对市值（log_mv − 行业等权均值）。
- [rel_mom_ind_10d](#rel_mom_ind_10d)  — 行业相对动量（个股10日收益 − 行业等权10日收益）。
- [rel_mom_ind_20d](#rel_mom_ind_20d)  — 行业相对动量（个股20日收益 − 行业等权20日收益）。
- [rel_mom_ind_3d](#rel_mom_ind_3d)  — 行业相对动量（个股3日收益 − 行业等权3日收益）。
- [rel_mom_ind_5d](#rel_mom_ind_5d)  — 行业相对动量（个股5日收益 − 行业等权5日收益）。
- [rel_pb_ind](#rel_pb_ind)  — 行业内相对市净率（pb − 行业等权均值，0值置NaN）。
- [rel_pe_ind](#rel_pe_ind)  — 行业内相对估值（pe_ttm − 行业等权均值，0值置NaN）。
- [rel_turnover_ind](#rel_turnover_ind)  — 行业内相对换手（换手率 − 行业等权均值）。
- [rel_turnover_ind_ma20](#rel_turnover_ind_ma20)  — 换手率相对自身均值20日均值（turnover_rate − 个股全期均值）。
- [rel_vol_first_hour](#rel_vol_first_hour)  — 首小时量比因子，开盘首小时成交量/全日成交量截面排名（早盘活跃=信息驱动排前）。
- [rel_vol_ind_20d](#rel_vol_ind_20d)  — 行业内相对波动（个股20日收益std − 行业等权均值）。
- [rel_vol_last_hour](#rel_vol_last_hour)  — 尾小时量比因子，收盘前1小时成交量/全日成交量截面排名（取负向=尾盘博弈=不可靠排后）。
- [rel_vol_midday](#rel_vol_midday)  — 午间量比因子，11:00-13:30成交量/全日成交量截面排名（取负向=午间异常放量排后）。
- [relative_spread](#relative_spread)  — 相对价差因子，(最高-最低)/VWAP截面排名（取负向=高振幅排后）。VWAP标准化后的日内波动幅度。
- [residual_momentum_20](#residual_momentum_20)  — 残差动量因子：剔除市场暴露后的残差20日累计截面排名（特质动量排前）。
- [ret_autocorr_1d_20](#ret_autocorr_1d_20)  — 日收益一阶自相关因子：ret与昨日ret的20日滚动相关截面排名（趋势性排前）。
- [ret_autocorr_20d](#ret_autocorr_20d)  — 收益自相关（日收益20日滚动 lag-1 自相关）。
- [ret_autocorr_5min](#ret_autocorr_5min)  — 5分钟收益自相关因子，日内5分钟收益一阶自相关系数截面排名。正自相关=日内动量，负自相关=均值回复。
- [ret_autocorr_abs](#ret_autocorr_abs)  — 绝对收益自相关因子，|ret_5min|的一阶自相关系数截面排名（正自相关=波动聚集排前=预测性好）。
- [ret_efficiency_20](#ret_efficiency_20)  — 价格效率：20日累计收益/20日累计成交额截面排名。单位成交额推动的价格变动效率。
- [ret_ind_rel_1d](#ret_ind_rel_1d)  — 行业相对1日收益因子：个股pct_chg/100 − 行业等权日收益（跑赢行业排前）。
- [ret_kurt_20](#ret_kurt_20)  — 20日收益峰度因子（日收益20日滚动峰度，反向排名）。
- [ret_kurt_5d](#ret_kurt_5d)  — 5 日收益峰度。
- [ret_kurt_intraday](#ret_kurt_intraday)  — 5分钟收益峰度因子，日内5分钟收益分布的峰度截面排名（取负向=厚尾排后）。衡量日内极端波动的集中度。
- [ret_skew_20](#ret_skew_20)  — 20日收益偏度因子（日收益20日滚动偏度，反向排名）。
- [ret_skew_5d](#ret_skew_5d)  — 5 日收益偏度。
- [ret_skew_60](#ret_skew_60)  — 60日收益偏度因子（日收益60日滚动偏度，反向排名）。
- [ret_std_intraday](#ret_std_intraday)  — 5分钟收益标准差因子，日内5分钟收益截面标准差排名（取负向=高离散排后）。不同于RV用平方和，std衡量收益围绕均值的离散度。
- [ret_vol_lead_corr_20](#ret_vol_lead_corr_20)  — 量领先价相关性因子：昨日vol与今日收益的20日滚动相关截面排名（量能前瞻有效排前）。
- [retail_attention](#retail_attention)  — 散户关注度代理因子，异常高换手率(当日换手/20日均换手-1)与大单净流出交乘截面排名（取负向）。
- [return_asymmetry_intraday](#return_asymmetry_intraday)  — 日内收益不对称因子，-(5分钟收益均值-中位数)截面排名（取负向=不对称=偏度大排后）。
- [reversal_2d](#reversal_2d)  — 2 日收益（收盘价 2 日变化率）。
- [reversal_liquidity_combo_5](#reversal_liquidity_combo_5)  — 超跌流动性复合因子：short_term_reversal_5×amihud_intraday×turnover_20截面排名。
- [reversal_oversold_combo_5](#reversal_oversold_combo_5)  — 超跌×超卖复合因子：short_term_reversal_5×williams_r_14截面排名（超跌且超卖排前）。
- [reversal_turnover_resonance_5](#reversal_turnover_resonance_5)  — 高换手反转：short_term_reversal_5×turnover_20。高换手的超跌股反转概率更高。
- [rjump_5min](#rjump_5min)  — 已实现跳跃因子，RV_5min² - BV_5min²的正部开根截面排名（高跳跃排后=风险信号）。
- [rjump_daily](#rjump_daily)  — 日度已实现跳跃因子，sqrt(max(RV²-BV²,0))截面排名（高跳跃=风险信号排后）。
- [roc_12](#roc_12)  — 12日ROC变动率因子：(adj−adj.shift(12))/adj.shift(12)截面排名（中期加速排前）。
- [rq_intraday](#rq_intraday)  — 已实现四次变差因子（Realized Quarticity），5分钟收益四次方和截面排名（取负向=高方差波动排后）。度量波动的波动。
- [rs_120](#rs_120)  — 120日相对强度因子：中期相对全市场的RS截面排名。
- [rs_250](#rs_250)  — 250日相对强度因子：年度相对全市场的RS截面排名（年度强势排前）。
- [rs_60](#rs_60)  — 60日相对强度因子：个股60日收益相对全市场等权收益的RS截面排名（跑赢市场排前）。
- [rs_value_divergence_20](#rs_value_divergence_20)  — RS-估值背离因子：rs_60−bp截面排名（强RS但高估值排前，谨慎信号）。
- [rsi_14_excess](#rsi_14_excess)  — RSI超买超卖因子（(rsi_close-50)截面排名，高RSI=超买排后）。
- [rsi_boll_combo](#rsi_boll_combo)  — RSI-布林带组合因子（rsi_zscore*boll_position截面排名，双双极端=强烈信号排前）。
- [rsi_day_position](#rsi_day_position)  — RSI日内区间位置因子：(RSI收盘−RSI最低)/(RSI最高−RSI最低)截面排名（收盘靠上沿排前）。
- [rsi_extreme_fraction](#rsi_extreme_fraction)  — RSI极端时间占比因子（RSI>70或<30的分钟占比截面排名，高占比=极端行情排前）。
- [rsi_intraday_trend](#rsi_intraday_trend)  — RSI日内趋势因子（RSI与时间的相关性截面排名，正相关=日内动量积聚排前）。
- [rsi_overbought_frac](#rsi_overbought_frac)  — RSI超买时间占比因子（RSI>70分钟占比截面排名，高占比=强势超买排后）。
- [rsi_oversold_frac](#rsi_oversold_frac)  — RSI超卖时间占比因子（RSI<30分钟占比截面排名，高占比=深度超卖排前）。
- [rsi_range](#rsi_range)  — RSI日内振幅因子（rsi_max-rsi_min截面排名，振幅大=剧烈波动排后）。
- [rsi_spread_6_14](#rsi_spread_6_14)  — 日频RSI(6)−RSI(14)因子：短中期超买超卖差截面排名（短期强于中期排前）。
- [rsi_trend_ma5](#rsi_trend_ma5)  — RSI超额5日均值因子：(RSI−50)的5日移动平均截面排名（中期动能排前）。
- [rsi_volatility](#rsi_volatility)  — RSI波动率因子（std(RSI)/mean(RSI)截面排名，RSI不稳定=信号质量差排后）。
- [rsrs_beta_18](#rsrs_beta_18)  — RSRS Beta因子，18日high~low回归斜率截面排名（高beta=阻力上升快于支撑=看涨排前）。
- [rsrs_beta_momentum_5](#rsrs_beta_momentum_5)  — RSRS斜率动量因子：18日high~low回归斜率5日变化截面排名（斜率转升排前）。
- [rsrs_r2_18](#rsrs_r2_18)  — RSRS R-squared因子，18日high~low回归拟合优度截面排名（高R2=支撑阻力关系清晰排前）。
- [rsrs_right_deviation](#rsrs_right_deviation)  — RSRS右偏离因子，(zscore * beta * r2)截面排名（量价验证=信号可靠排前）。
- [rsrs_volume_right_deviation](#rsrs_volume_right_deviation)  — RSRS量能加权右偏离因子：β的400日Z分数×β×R²×近期量能占比截面排名（趋势信号+量能确认排前）。
- [rsrs_zscore_18](#rsrs_zscore_18)  — RSRS Z-score因子，beta相对自身历史400日的标准化偏离截面排名（极端偏离=支撑阻力位重塑排前）。
- [rsv_5min](#rsv_5min)  — 已实现半方差因子（下行风险），仅5分钟负收益的平方和开根截面排名（高下行波排后）。
- [rv_10min](#rv_10min)  — 10分钟已实现波动率因子，基于1分钟数据的10分钟收益平方和开根截面排名（低波排前）。
- [rv_15min](#rv_15min)  — 15分钟已实现波动率因子，基于1分钟数据的15分钟收益平方和开根截面排名（低波排前）。
- [rv_30min](#rv_30min)  — 30分钟已实现波动率因子，基于1分钟数据的30分钟收益平方和开根截面排名（低波排前）。
- [rv_5min](#rv_5min)  — 5分钟已实现波动率因子，基于1分钟数据的5分钟收益平方和开根截面排名（低波排前）。
- [rv_60min](#rv_60min)  — 60分钟已实现波动率因子，1分钟数据60分钟收益平方和开根截面排名（低波排前）。
- [rv_daily](#rv_daily)  — 日度已实现波动率因子，1分钟数据全日收益平方和开根截面排名（低波排前）。
- [rv_hourly_1](#rv_hourly_1)  — 第一小时波动率因子，9:30-10:30已实现波动率截面排名（取负向=开盘高波排后）。
- [rv_hourly_4](#rv_hourly_4)  — 第四小时波动率因子，14:00-15:00已实现波动率截面排名（取负向=尾盘高波排后）。
- [rv_hourly_dispersion](#rv_hourly_dispersion)  — 小时波动率离散度因子，四个小时RV的标准差/均值截面排名（取负向=波动集中=不稳定排后）。
- [rv_rolling_5d_std](#rv_rolling_5d_std)  — 波动率波动因子，rv_5min的5日标准差截面排名（取负向=波动率不稳定排后）。
- [rv_semi_down](#rv_semi_down)  — 下行已实现半方差因子，仅负1分钟收益平方和开根截面排名（高下行波=风险排后）。
- [rv_semi_up](#rv_semi_up)  — 上行已实现半方差因子，仅正1分钟收益平方和开根截面排名（上涨波动=正面信号排前）。
- [rv_skew_intraday](#rv_skew_intraday)  — 日内已实现偏度因子，5分钟收益的截面偏度排名。正偏=上涨跳跃多，负偏=下跌跳跃多。
- [rv_term_structure](#rv_term_structure)  — RV期限结构因子，rv_5min排名/rv_60min排名截面排名（短期波动相对长期波动高=波动正在集聚排后）。
- [rv_term_structure_slope](#rv_term_structure_slope)  — 波动率期限结构斜率因子，rv_5min/rv_60min-1截面排名（取负向=陡峭=短期波动高排后）。
- [rv_trend_5d](#rv_trend_5d)  — 波动率趋势因子，rv_5min的5日均值/20日均值-1截面排名（取负向=波动加速排后）。
- [sector_amount_momentum_5d](#sector_amount_momentum_5d) 【已禁用】 — 板块成交额动量因子：个股所属行业平均成交额5日变化率，映射到个股后截面排名。反映资金在板块层面的流入/流出动能。
- [sector_amount_rank](#sector_amount_rank) 【已禁用】 — 行业内成交额占比因子，个股成交额在所属行业内的截面排名。
- [sector_mv_rank](#sector_mv_rank) 【已禁用】 — 行业内市值占比因子，个股总市值在所属行业内的截面排名。
- [sentiment_value_gap](#sentiment_value_gap)  — 情绪-价值差因子，-(资金流因子排名-bp排名)截面排名（取负向=情绪脱离价值=风险排后）。
- [shadow_asymmetry](#shadow_asymmetry)  — 影线不对称性因子，-(upper_shadow_ratio-lower_shadow_ratio)*(high-low)/close截面排名（上影主导=看空排后）。
- [shadow_lower_20](#shadow_lower_20)  — 20日均下影线比例，下影线/(high-low) 截面排名（高值=强支撑）。
- [shadow_upper_20](#shadow_upper_20)  — 20日均上影线比例，上影线/(high-low) 截面排名。
- [short_balance_ratio_change_20d](#short_balance_ratio_change_20d)  — 融券余额占比变化因子：融券余额/流通市值占比的20日变化截面排名（负向，空头加仓排后）。
- [short_interest_volatility_20d](#short_interest_volatility_20d)  — 融券余量20日波动率取反。余量剧烈波动=空头态度摇摆/不稳定,排名低。
- [short_sell_volume_ratio](#short_sell_volume_ratio)  — 融券卖出占比：rqmcl/vol。融券卖出量相对总成交量的占比,高速=活跃做空。
- [short_squeeze_risk](#short_squeeze_risk)  — 逼空风险=融券余量/融资余额。高比值=大量做空仓位vs低做多杠杆,逼空风险大,排名高(反转做多信号)。
- [short_term_reversal_5](#short_term_reversal_5)  — 5日短周期反转因子（负向5日动量），截面排名。
- [size_momentum_combo_20](#size_momentum_combo_20)  — 小盘×动量：log_circ_mv×momentum_20。小盘股的动量效应更强。
- [small_order_crowding](#small_order_crowding)  — 小单拥挤度因子，-(小单买入量/总成交量)截面排名（高小单占比=散户追涨排后）。
- [smallcap_liftoff_combo_60](#smallcap_liftoff_combo_60)  — 小盘量价起飞因子：log_circ_mv×momentum_60×volume_momentum_5截面排名。
- [smart_capital_liftoff_20](#smart_capital_liftoff_20)  — 主力资金起飞因子：mf_net_inflow_5d×momentum_20×volume_momentum_5截面排名。
- [smart_money_concentration](#smart_money_concentration)  — 聪明钱集中度因子，(特大单+大单净买-小单-中单净卖)/总成交额截面排名（聪明钱相对噪音交易者越集中排前）。
- [smart_money_momentum_combo_20](#smart_money_momentum_combo_20)  — 聪明钱×动量复合因子：smart_money_net_bias×momentum_20截面排名（聪明钱净买且上涨排前）。
- [smart_money_net_bias](#smart_money_net_bias)  — 聪明钱净方向因子，异动分钟量加权涨跌方向截面排名（聪明钱净买入排前）。
- [smart_money_share](#smart_money_share)  — 聪明钱成交量占比因子，异动分钟成交量占全天比例截面排名（信息型交易活跃排前）。
- [smart_money_vwap_ratio](#smart_money_vwap_ratio)  — 聪明钱VWAP比因子，异动分钟(|收益|/√量前20%)的VWAP/全天VWAP截面排名（聪明钱成交价高于均价=抢筹排前）。
- [sortino_ratio_60](#sortino_ratio_60)  — 60日Sortino比率因子：均收益/下行标准差截面排名（正向，风险调整收益高排前）。
- [sp_raw](#sp_raw)  — 未调整市销率因子（低值排前），ps = 总市值/最近报告期营收。
- [sp_ttm](#sp_ttm)  — 市销率倒数(SP_TTM)因子，1/PS_TTM截面排名。
- [sp_ttm_momentum_20](#sp_ttm_momentum_20)  — SP_TTM动量因子（1/PS_TTM的20日变化率截面排名，SP上升排前）。
- [sqrt_am_hl_range_intraday](#sqrt_am_hl_range_intraday)  — 平方根变换上午高低价差因子，sign×√(|am_hl_range_intraday|)截面排名。
- [sqrt_max_ret_intraday](#sqrt_max_ret_intraday)  — 平方根变换日内最大收益因子，sign×√(|max_ret_intraday|)截面排名。
- [sqrt_relative_spread](#sqrt_relative_spread)  — 平方根变换相对价差因子，sign×√(|relative_spread|)截面排名。
- [sqrt_turnover_std_20](#sqrt_turnover_std_20)  — 平方根变换20日换手标准差因子，sign×√(|turnover_std_20|)截面排名。
- [sqrt_turnover_vol_20](#sqrt_turnover_vol_20)  — 平方根变换20日换手波动率因子，sign×√(|turnover_vol_20|)截面排名。
- [stoch_slow_k](#stoch_slow_k)  — 慢速随机%K因子：RSV(9)的三日平滑截面排名（随机动能排前）。
- [super_large_order_intensity](#super_large_order_intensity)  — 超大单强度因子，超大单净买入/总成交额截面排名。
- [tail_corr_60](#tail_corr_60)  — 60日尾部相关性因子：与市场同向极端收益(|z|>1.5)的频率截面排名（尾部同步排前）。
- [tail_ret_3d](#tail_ret_3d)  — 近3日尾盘收益累计因子：last30_ret 滚动3日和的截面排名（负向）。
- [tail_risk_pct_60](#tail_risk_pct_60)  — 尾风险频率因子：60日内|z|>2极端收益占比截面排名（负向，极端波动频发排后）。
- [tail_volume_share](#tail_volume_share)  — 尾盘量能占比因子：尾盘30分钟成交量占全天比例截面排名（尾盘放量排前）。
- [td_setup_count](#td_setup_count)  — TD序列setup计数因子：连续close≤4日前close的天数截面排名（连续下跌setup排前）。
- [three_black_crows](#three_black_crows)  — 三只黑鸦形态因子：连续阴线(收盘<开盘且低于前收)天数计数截面排名（连续阴跌排前）。
- [time_since_52w_high](#time_since_52w_high)  — 距252日新高天数因子（最近一次创年内新高距今的天数，反向排名）。
- [total_leverage_ratio](#total_leverage_ratio)  — 融资融券总余额/总市值，衡量杠杆化程度。高杠杆=波动风险大，排名取反。
- [trend_confirmation](#trend_confirmation)  — 趋势确认因子（ma_alignment*sign(macd_close)截面排名，均线+MACD双确认排前）。
- [trix_12_20](#trix_12_20)  — TRIX趋势因子：三重指数平滑(12)的20日变化率截面排名（趋势加速排前）。
- [trix_signal_gap](#trix_signal_gap)  — TRIX信号乖离因子：TRIX与其20日信号线的乖离截面排名（强于自身趋势线排前）。
- [ts_price_self_rank_60](#ts_price_self_rank_60)  — 价格自身60日位置因子，收盘价在自身60日高低区间的相对位置截面排名。
- [ts_volume_zscore_20](#ts_volume_zscore_20)  — 成交量20日Z-score因子，当日成交量偏离20日均值的标准差数截面排名（高放量排后）。
- [turnover_20](#turnover_20)  — 20日平均换手率因子（总股本换手率），低换手排前。
- [turnover_20_size_neutral](#turnover_20_size_neutral)  — 规模中性化换手率因子，市值分桶内低换手排名。
- [turnover_anomaly_20](#turnover_anomaly_20)  — 换手率异常因子，20日均换手/60日均换手-1截面排名（取负向=异常高换手排后）。
- [turnover_anomaly_mean_20d](#turnover_anomaly_mean_20d)  — 换手率异象因子，-(近20日平均换手率)截面排名（高换手=投机性强排后）。
- [turnover_chg_20d](#turnover_chg_20d)  — 换手率20日变化率（turnover_rate 20日pct_change）。
- [turnover_chg_5d](#turnover_chg_5d)  — 换手率5日变化率（turnover_rate 5日pct_change）。
- [turnover_concentration_intraday](#turnover_concentration_intraday)  — 换手率集中度因子，最大5分钟成交量/全日成交量截面排名（取负向=集中度过高排后）。
- [turnover_event_confirmation_20](#turnover_event_confirmation_20)  — 高换手×事件确认因子：(1−turnover_20)×limit_up_event_5截面排名。
- [turnover_f_20](#turnover_f_20)  — 20日平均自由流通换手率因子，低换手排前。
- [turnover_f_delta_5](#turnover_f_delta_5)  — 自由流通换手率5日变化因子（换手率下降=浮筹减少排前）。
- [turnover_f_divergence](#turnover_f_divergence)  — 自由流通换手率/总换手率。>1=交易集中于自由流通盘，存量筹码活跃，排名高。
- [turnover_f_raw](#turnover_f_raw)  — 自由流通换手率原始值因子（低换手排前）。
- [turnover_factor_momentum_20](#turnover_factor_momentum_20)  — 换手率因子20日动量 (关注度变化)。
- [turnover_orthogonal_to_mv](#turnover_orthogonal_to_mv)  — 换手率对市值正交化因子 (剔除规模效应的纯流动性)。
- [turnover_ret_corr_20](#turnover_ret_corr_20)  — 换手率-收益相关因子：20日换手率与收益的相关性截面排名（量价同步排前）。
- [turnover_rv_interaction](#turnover_rv_interaction)  — 换手-波动耦合因子，turnover_20排名×rv_5min排名截面排名（量价共振强度排前）。
- [turnover_shock_20](#turnover_shock_20)  — 换手率异动因子，换手率20日Z-score截面排名（异常高换手排后）。
- [turnover_std_20](#turnover_std_20)  — 换手率波动率因子，20日换手率标准差截面排名（高换手波动排后=流动性风险）。
- [turnover_vol_20](#turnover_vol_20)  — 20日换手率波动因子，换手率标准差截面排名（低波动排前）。
- [turnover_zscore_20](#turnover_zscore_20)  — 换手率20日Z-score因子（当日换手偏离自身20日均值的标准差数，反向排名）。
- [ulcer_index_20](#ulcer_index_20)  — 溃疡指数因子（20日窗口内回撤平方均值的平方根，反向排名）。
- [up_day_freq_20d](#up_day_freq_20d)  — 上涨天数频率（20日内 close>前收盘 的天数占比）。
- [up_day_volume_ratio_20](#up_day_volume_ratio_20)  — 上涨日量占比因子：20日上涨日成交量占总量的比例截面排名（上涨放量排前）。
- [up_down_count_ratio_20](#up_down_count_ratio_20)  — 20日涨跌天数比因子：(涨天数+1)/(跌天数+1)截面排名（涨多跌少排前）。
- [up_minute_vol_share](#up_minute_vol_share)  — 上涨分钟量占比因子：当日上涨分钟成交量占全天量比例截面排名（涨时整体放量排前）。
- [up_minutes_ratio](#up_minutes_ratio)  — 上涨分钟占比因子，1分钟正收益分钟数/总分钟数截面排名（买盘主导排前）。
- [upper_shadow_ratio](#upper_shadow_ratio)  — 上影线比率因子，-(high-max(open,close))/(high-low+1e-9)截面排名（长上影=抛压重排后）。
- [value_event_combo_20](#value_event_combo_20)  — 价值×事件复合因子：bp×limit_up_event_5截面排名（低估且刚涨停排前）。
- [value_factor_zscore_252](#value_factor_zscore_252)  — 价值因子(BP)252日历史Z-score (相对自身历史的高估/低估)。
- [value_liftoff_combo_20](#value_liftoff_combo_20)  — 价值量价起飞因子：bp×volume_breakout_confirm_20×momentum_20截面排名。
- [value_reversal_combo_60](#value_reversal_combo_60)  — 价值×深回撤复合因子：bp×(1−drawdown_60)截面排名（低估且深度回撤排前）。
- [var_95_20](#var_95_20)  — 20日VaR(95%)因子：收益5%分位数截面排名（负向，尾部损失大排后）。
- [vol_clustering_20](#vol_clustering_20)  — 波动聚集因子：|收益|的20日自相关截面排名（负向，波动持续聚集排后）。
- [vol_concentration](#vol_concentration)  — 成交量集中度因子，(开盘30分+收盘30分)成交量/全日成交量截面排名（取负向=过于集中排后）。
- [vol_cycle_position_120](#vol_cycle_position_120)  — 波动周期位置因子：20日波动/120日内最低20日波动截面排名（负向，波动扩张排后）。
- [vol_decay_ratio_20](#vol_decay_ratio_20)  — 波动衰减因子：20日波动/10日波动截面排名（负向，波动持续放大排后）。
- [vol_liquidity_resonance_20](#vol_liquidity_resonance_20)  — 低波×高流动性共振因子：parkinson_vol×amihud_intraday截面排名。
- [vol_of_rv](#vol_of_rv)  — 波动率的波动率因子，rv_5min的20日滚动标准差截面排名（波动率不稳定排后）。
- [vol_of_vol_20d](#vol_of_vol_20d)  — 波动之波动（|日收益|的20日std）。
- [vol_of_vol_5d](#vol_of_vol_5d)  — 波动率之波动（5 日 |日收益| 的标准差）。
- [vol_of_vol_60](#vol_of_vol_60)  — 波动率的波动因子（60日收益std的20日std，反向排名）。
- [vol_of_vol_60d](#vol_of_vol_60d)  — 波动之波动（|日收益|的60日std）。
- [vol_of_vol_intraday](#vol_of_vol_intraday)  — 波动率的波动率因子，|ret_5min|的std/mean截面排名（取负向=波动不稳定排后）。波动率自身的变异系数。
- [vol_ratio_ma20](#vol_ratio_ma20)  — 量比20日均值（volume_ratio 的20日滚动均值）。
- [vol_ratio_ma3](#vol_ratio_ma3)  — 3 日量比（当日成交量 / 3 日均量）。
- [vol_ratio_ma5](#vol_ratio_ma5)  — 量比5日均值（volume_ratio 的5日滚动均值）。
- [vol_regime_switch_20](#vol_regime_switch_20)  — 波动状态切换因子：20日波动/60日波动截面排名（负向，波动骤升排后）。
- [vol_stability](#vol_stability)  — 成交量稳定性因子，5分钟成交量std/均值截面排名（取负向=不稳定排后）。成交量日内分布的规律性。
- [volume_autocorr_20](#volume_autocorr_20)  — 成交量20日自相关因子：vol与昨日vol的20日滚动相关截面排名（量能惯性排前）。
- [volume_autocorr_5](#volume_autocorr_5)  — 成交量5日自相关因子：vol与昨日vol的5日滚动相关截面排名（量能节奏规律排前）。
- [volume_breakout_confirm_20](#volume_breakout_confirm_20)  — 突破量能确认：20日内突破前20日高点的量比累计截面排名。突破+放量=强确认。
- [volume_climax](#volume_climax)  — 放量异动因子，今日成交量/20日均量截面排名。
- [volume_distribution_skew](#volume_distribution_skew)  — 日内价格移动偏度代理因子，(开盘至最高涨幅)-(最高至收盘涨幅)截面排名（早盘冲高=机构抢筹排前）。
- [volume_dry_up](#volume_dry_up)  — 缩量因子，-(20日最低成交量/20日均量)截面排名（极度缩量=变盘前兆排前）。
- [volume_momentum_5](#volume_momentum_5)  — 5日成交量动量因子 (量增排前)。
- [volume_peak_time](#volume_peak_time)  — 成交量峰值时间因子，日内最大5分钟成交量所在分钟截面排名（取负向=尾盘放量=异常排后）。
- [volume_price_confirmation](#volume_price_confirmation)  — 量价配合确认因子（mavol5_slope符号==ma5_slope符号截面排名，量价同步=趋势确认排前）。
- [volume_price_corr_20](#volume_price_corr_20)  — 量价相关因子（20日滚动收益与对数成交量的相关性，正向排名）。
- [volume_price_divergence_score](#volume_price_divergence_score)  — 量价背离得分因子：价动量排名−量动量排名的背离截面排名（价强量弱背离排前）。
- [volume_price_liftoff_20](#volume_price_liftoff_20)  — 量价起飞因子：momentum_20×volume_breakout_confirm_20×drawdown_60截面排名。
- [volume_profile_kurt](#volume_profile_kurt)  — 成交量分布峰度因子，5分钟成交量日内分布的峰度截面排名（取负向=高峰度=不均排后）。
- [volume_profile_skew](#volume_profile_skew)  — 成交量分布偏度因子，5分钟成交量日内分布的偏度截面排名（取负向=偏度极端排后）。
- [volume_ratio](#volume_ratio)  — 量比因子（当日成交量相对5日均量），截面排名。
- [volume_ratio_20](#volume_ratio_20)  — 20日相对成交量因子，vol/avg_vol_20 - 1 截面排名。
- [volume_ratio_extreme](#volume_ratio_extreme)  — 量比极端值因子取反。异常放量(vr>3)=量能衰竭→反转，异常缩量(vr<0.3)=无人问津→可能启动。
- [volume_ratio_momentum_5d](#volume_ratio_momentum_5d)  — 量比动量因子：量比(volume_ratio)的5日变化截面排名（量能扩张排前）。
- [volume_ratio_zscore_20](#volume_ratio_zscore_20)  — 量比20日Z-score因子（当日量比偏离自身20日均值的标准差数，反向排名）。
- [volume_rv_ratio](#volume_rv_ratio)  — 成交量波动比因子，log(成交量)/rv_5min截面排名。单位波动对应的交易活跃度。
- [volume_skew_5d](#volume_skew_5d)  — 5日量能偏度因子：5日成交量的偏度截面排名（放量脉冲排前，负向排名）。
- [volume_surge_3d](#volume_surge_3d)  — 成交量脉冲因子，3日最大(vol/60日中位数vol)截面排名（负向：脉冲后反转）。
- [volume_tilt_20](#volume_tilt_20)  — 量能倾斜：20日量加权收益与等权收益之差截面排名。正倾斜=收益主要来自放量日。
- [volume_u_shape_score](#volume_u_shape_score)  — U型分布评分因子，成交量日内分布与U型模板的相关性截面排名（取负向=极端U型=操纵风险排后）。
- [volume_weighted_ret](#volume_weighted_ret)  — 成交量加权收益因子，Σ(ret_i×vol_i)/Σvol_i截面排名（量价配合=真实涨跌排前）。
- [vp_consistency_20](#vp_consistency_20)  — 量价一致性20日均值因子：四象限一致性得分的20日均值截面排名（持续量价健康排前）。
- [vp_consistency_score](#vp_consistency_score)  — 量价一致性得分因子：四象限一致性(放量涨+缩量跌−缩量涨−放量跌)/全天量截面排名（量价健康排前）。
- [vp_expand_down_am_share](#vp_expand_down_am_share)  — 早盘放量下跌占比因子：早盘(09:30-11:30)放量下跌量占全天放量下跌量比例截面排名（早盘恐慌释放排前）。
- [vp_expand_down_share](#vp_expand_down_share)  — 放量下跌量占比因子：日内放量下跌分钟量占全天量比例截面排名（负向，恐慌抛售排后）。
- [vp_expand_price_pos](#vp_expand_price_pos)  — 放量价格位置因子：放量分钟的量加权日内位置均值截面排名（负向，低位放量=吸筹排前）。
- [vp_expand_ret_gap](#vp_expand_ret_gap)  — 放缩量收益差因子：放量分钟均收益−缩量分钟均收益截面排名（放量推动价格排前）。
- [vp_expand_up_share](#vp_expand_up_share)  — 放量上涨量占比因子：日内放量(超20日同时段均量)上涨分钟量占全天量比例截面排名（涨有量排前）。
- [vp_shrink_down_share](#vp_shrink_down_share)  — 缩量下跌量占比因子：日内缩量下跌分钟量占全天量比例截面排名（跌无量排前）。
- [vp_shrink_up_share](#vp_shrink_up_share)  — 缩量上涨量占比因子：日内缩量上涨分钟量占全天量比例截面排名（负向，无量反弹排后）。
- [vwap_am_pm_gap_factor](#vwap_am_pm_gap_factor)  — 上午/下午VWAP差因子：上午VWAP相对下午VWAP偏离截面排名（上午价格水平高排前）。
- [vwap_close_ratio](#vwap_close_ratio)  — 收盘价/分钟VWAP偏离因子：close/(amount/vol)-1 截面排名（负向）。
- [vwap_daily_deviation](#vwap_daily_deviation)  — 日频VWAP偏离因子：close/(amount/vol)−1截面排名（收盘高于日均价=尾盘强势排前）。
- [vwap_dev_1d](#vwap_dev_1d)  — 日频VWAP偏离因子：close/(amount/vol)-1（收盘价高于日均价排前）。
- [vwap_deviation](#vwap_deviation)  — VWAP偏离因子，(收盘-VWAP)/VWAP截面排名。收盘价相对日均价的位置。
- [vwap_momentum_5d](#vwap_momentum_5d)  — VWAP动量因子，(5日VWAP均值/20日VWAP均值-1)截面排名。
- [williams_r_14](#williams_r_14)  — 14日威廉%R因子：(HH14−C)/(HH14−LL14)×(−100)截面排名（超卖排前）。
- [winner_momentum_combo_20](#winner_momentum_combo_20)  — 获利盘×动量：winner_rate×momentum_20。获利盘多+上涨=浮盈筹码形成支撑。
- [winner_rate](#winner_rate)  — 获利盘比例因子，winner_rate截面排名（取负向=高获利盘为反转信号）。
- [winner_rate_acceleration](#winner_rate_acceleration)  — 获利盘比例加速度因子，winner_rate的5日变化截面排名。
- [winner_rate_change_20d](#winner_rate_change_20d)  — 获利盘比例20日变化。获利盘增加=上涨趋势中筹码逐步盈利，排名高。
- [winner_rate_change_5d](#winner_rate_change_5d)  — 获利盘5日变化因子，winner_rate - winner_rate.shift(5)截面排名（取负向）。
- [winner_rate_factor_momentum_20](#winner_rate_factor_momentum_20)  — 获利盘因子20日动量 (筹码结构变化方向)。
- [winner_rate_momentum_5d](#winner_rate_momentum_5d)  — 获利盘变化率因子，winner_rate的5日变化截面排名（获利盘快速增加=短期过热排后）。
- [winner_rate_reversal_signal](#winner_rate_reversal_signal)  — 获利盘极端反转因子，-|winner_rate-0.5|截面排名（50%附近=方向不确定=不确定溢价排前）。
- [winner_rs_combo_60](#winner_rs_combo_60)  — 获利盘×相对强度复合因子：winner_rate×rs_60截面排名（筹码压力小且强势排前）。
- [zero_return_fraction_20](#zero_return_fraction_20)  — 零收益占比因子：20日|pct_chg|<0.1%的天数占比截面排名（负向，交投冷淡排后）。
- [zscore_amihud_intraday](#zscore_amihud_intraday)  — 时序Z-score变换日内Amihud非流动性因子，(原始值-252日均值)/252日标准差截面排名。
- [zscore_realized_spread_5min](#zscore_realized_spread_5min)  — 时序Z-score变换5分钟已实现价差因子，(原始值-252日均值)/252日标准差截面排名。
- [zscore_turnover_20](#zscore_turnover_20)  — 时序Z-score变换20日换手率因子，(原始值-252日均值)/252日标准差截面排名。
- [zscore_turnover_f_20](#zscore_turnover_f_20)  — 时序Z-score变换20日自由流通换手率因子，(原始值-252日均值)/252日标准差截面排名。
- [zscore_volume_rv_ratio](#zscore_volume_rv_ratio)  — 时序Z-score变换成交量-RV比率因子，(原始值-252日均值)/252日标准差截面排名。
