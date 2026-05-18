# 因子开发指南

> 本文档汇总了上游数据资产和当前因子体系的全貌，供因子研究员分析现有覆盖度、发现空白领域，规划下一步因子开发方向。

---

## 目录

1. [上游数据资产](#1-上游数据资产)
2. [数据加载机制](#2-数据加载机制)
3. [当前因子体系](#3-当前因子体系)
4. [因子开发基础设施](#4-因子开发基础设施)
5. [可开发方向分析](#5-可开发方向分析)

---

## 1. 上游数据资产

所有上游数据在 `/root/shared-nvme/lingqiData/data/`，为 parquet 格式。数据覆盖约 **2542 只 A 股**，日频数据最早到 **2019-01-02**。

### 1.1 日频行情数据

#### `daily_adj.parquet` (190M, 11 列) — 复权日线

| 字段 | 类型 | 说明 |
|---|---|---|
| stock_code | string | 股票代码，如 `000001` |
| trade_date | string | 交易日期，YYYYMMDD |
| open | double | 复权开盘价 |
| high | double | 复权最高价 |
| low | double | 复权最低价 |
| close | double | 复权收盘价 |
| change | double | 价格变动 |
| pct_chg | double | 涨跌幅(%) |
| vol | double | 成交量 |
| amount | double | 成交额 |
| stock_name | string | 股票名称 |

**已用于**: 所有 price/enhanced/neutral 类因子、target 标签的底层数据。

#### `daily.parquet` (206M, 12 列) — 不复权日线

| 字段 | 类型 | 说明 |
|---|---|---|
| stock_code | string | 股票代码 |
| trade_date | string | 交易日期 |
| open / high / low / close | double | 不复权 OHLC |
| pre_close | double | 前收盘价 |
| change / pct_chg | double | 涨跌额 / 涨跌幅 |
| vol / amount | double | 成交量 / 成交额 |
| stock_name | string | 股票名称 |

**已用于**: shadow_upper_20, shadow_lower_20, gap_ratio_20, atr_20（需 pre_close 计算 TR）。

**与 daily_adj 的差异**: daily_adj 已复权，适合计算收益率；daily 含 pre_close，适合计算跳空、影线、真实波幅等需要真实价格几何结构的因子。

---

### 1.2 日频估值与交易指标

#### `finance.parquet` (211M, 20 列)

| 字段 | 类型 | 说明 |
|---|---|---|
| stock_code | string | 股票代码 |
| trade_date | string | 交易日期 |
| close | double | 收盘价 |
| turnover_rate | double | 总股本换手率 |
| turnover_rate_f | double | 自由流通换手率 |
| volume_ratio | double | 量比（当日量/5日均量） |
| pe / pe_ttm | double | 市盈率 / 滚动市盈率 |
| pe_ttm_percentile | double | PE_TTM 历史分位 |
| pb | double | 市净率 |
| ps / ps_ttm | double | 市销率 / 滚动市销率 |
| dv_ratio / dv_ttm | double | 报告期股息率 / 滚动股息率 |
| total_share / float_share / free_share | double | 总股本 / 流通股本 / 自由流通股本 |
| total_mv / circ_mv | double | 总市值 / 流通市值 |
| stock_name | string | 股票名称 |

**已用字段**: pb, ps_ttm, dv_ttm, dv_ratio, total_mv, circ_mv, turnover_rate, turnover_rate_f, volume_ratio, pe_ttm, pe_ttm_percentile, free_share, total_share

**未用字段**: `pe`, `float_share`, `close`

---

### 1.3 财务数据（报告期 → 日频前向填充）

#### `financial_indicator.parquet` (159M, 167 列)

核心字段（完整 167 列见附录）：

| 类别 | 已用字段 | 未用字段（举例） |
|---|---|---|
| 盈利能力 | roe, roa, roic, roe_waa, roe_dt, gross_margin, netprofit_margin | roe_yearly, roa_yearly, roic_yearly, npta, ebit, ebitda, fcff, fcfe, eps, bps, ocfps |
| 偿债能力 | debt_to_assets, current_ratio, quick_ratio, cash_ratio, ebit_to_interest | assets_to_eqt, debt_to_eqt, ocf_to_debt, longdeb_to_debt |
| 运营效率 | assets_turn, inv_turn, ar_turn | ca_turn, fa_turn, turn_days |
| 成长性 | or_yoy, netprofit_yoy, equity_yoy, assets_yoy | op_yoy, tr_yoy, ebt_yoy, q_gr_yoy, q_sales_yoy, q_profit_yoy, q_netprofit_yoy |
| 现金流质量 | ocf_to_profit, salescash_to_or | ocf_to_or, ocf_to_opincome, ocfps, cfps |
| 季度单季 | q_roe, q_gsprofit_margin, q_netprofit_margin, q_sales_yoy, q_netprofit_yoy, q_profit_yoy, q_ocf_to_sales, q_eps | q_opincome, q_dt_roe 等 |

#### `balancesheet.parquet` (61M, 157 列)

**已用字段**: goodwill, total_hldr_eqy_exc_min_int, inventories, total_assets, accounts_receiv, fix_assets, acct_payable

**未用字段举例**:
- **资产结构**: money_cap(货币资金), trad_asset(交易性金融资产), notes_receiv(应收票据), intan_assets(无形资产), r_and_d(开发支出), cip(在建工程), right_of_use_assets(使用权资产), lease_liab(租赁负债), contract_assets(合同资产), contract_liab(合同负债)
- **负债结构**: st_borr(短期借款), lt_borr(长期借款), bond_payable(应付债券), total_cur_liab(流动负债合计), total_ncl(非流动负债合计)
- **权益**: treasury_share(库存股), minority_int(少数股东权益), oth_eqt_tools(其他权益工具)

#### `income.parquet` (36M, 96 列)

**已用字段**: operate_profit, total_profit, revenue, rd_exp, sell_exp, admin_exp, fin_exp, invest_income, oper_cost

**未用字段举例**:
- **收入拆分**: total_revenue, int_income, comm_income, n_oth_income
- **利润质量**: ebit, ebitda, non_oper_income, non_oper_exp, nca_disploss(非流动资产处置损失), credit_impair_loss(信用减值损失), assets_impair_loss(资产减值损失)
- **EPS 系列**: basic_eps, diluted_eps

#### `cashflow.parquet` (48M, 97 列)

**已用字段**: free_cashflow, n_cashflow_act, n_cash_flows_fnc_act, c_inf_fr_operate_a

**未用字段举例**:
- n_cashflow_inv_act(投资活动现金流净额)
- c_recp_borrow(取得借款收到的现金)
- c_pay_dist_dpcp_int_exp(分配股利/偿付利息)
- c_paid_for_taxes(支付的各项税费)
- depr_fa_coga_dpba(固定资产折旧)
- amort_intang_assets(无形资产摊销)
- 间接法现金流量调整项约 30 个字段

---

### 1.4 资金流数据

#### `main_fund_flow.parquet` (596M, 20 列)

按大/中/小/特大单拆分的买卖量和金额：

| 字段 | 说明 |
|---|---|
| buy_sm_vol / buy_sm_amount | 小单买入量/额 |
| sell_sm_vol / sell_sm_amount | 小单卖出量/额 |
| buy_md_vol / buy_md_amount | 中单买入量/额 |
| sell_md_vol / sell_md_amount | 中单卖出量/额 |
| buy_lg_vol / buy_lg_amount | 大单买入量/额 |
| sell_lg_vol / sell_lg_amount | 大单卖出量/额 |
| buy_elg_vol / buy_elg_amount | 特大单买入量/额 |
| sell_elg_vol / sell_elg_amount | 特大单卖出量/额 |
| net_mf_vol / net_mf_amount | 主力净流入量/额 |

**已用字段**: 全部买卖金额字段, net_mf_amount。**未用字段**: 所有 vol 字段。

#### `margin_detail.parquet` (161M, 11 列)

| 字段 | 说明 |
|---|---|
| stock_code / trade_date | 股票代码 / 日期 |
| exchange_id | 交易所 |
| rzye | 融资余额 |
| rzmre | 融资买入额 |
| rzche | 融资偿还额 |
| rqye | 融券余量 |
| rqmcl | 融券卖出量 |
| rzrqye | 两融总余额 |
| rqyl | 融券余量(另一种口径) |
| is_backfill | 是否回补 |

**已用字段**: rzye, rzmre, rzche, rqye, rzrqye

---

### 1.5 事件数据

#### `top_list.parquet` (5.9M, 15 列) — 龙虎榜（上榜日才有值）

| 字段 | 说明 |
|---|---|
| trade_date / stock_code | 日期 / 代码 |
| name | 股票名称 |
| close / pct_change | 收盘价 / 涨跌幅 |
| turnover_rate / amount | 换手率 / 成交额 |
| l_sell / l_buy / l_amount | 龙虎榜卖出/买入/总成交额 |
| net_amount / net_rate | 净买额 / 净买率 |
| amount_rate | 龙虎榜成交占比 |
| float_values | 流通市值 |
| reason | 上榜原因 |

#### `dragon_tiger.parquet` (2.6M, 10 列) — 龙虎榜席位明细

| 字段 | 说明 |
|---|---|
| trade_date / stock_code | 日期 / 代码 |
| org_name | 席位名称 |
| buy_amount / buy_ratio | 买入金额 / 占比 |
| sell_amount / sell_ratio | 卖出金额 / 占比 |
| net_buy_amount | 净买额 |
| direction | 方向 |
| reason | 原因 |

#### `limit_up.parquet` (4.1M, 17 列) — 涨停板

| 字段 | 说明 |
|---|---|
| trade_date / stock_code | 日期 / 代码 |
| price / change_percent | 涨停价 / 涨跌幅 |
| first_limit_time / final_limit_time | 首次/最终封板时间 |
| consecutive_days | 连板天数 |
| sealed_volume / sealed_amount | 封单量 / 封单额 |
| sealed_turnover_ratio / sealed_flow_ratio | 封单换手率 / 封单流比 |
| open_count | 开板次数 |
| boards / limit_type | 板类型 / 涨跌停类型 |
| is_limit_up | 是否涨停 |
| reason_text | 原因 |

#### `limit_list.parquet` (7.1M, 18 列) — 涨跌停列表

| 字段 | 说明 |
|---|---|
| trade_date / stock_code | 日期 / 代码 |
| close / pct_chg / amount | 收盘价 / 涨跌幅 / 成交额 |
| limit_amount | 封单额 |
| float_mv / total_mv | 流通市值 / 总市值 |
| turnover_ratio | 换手率 |
| fd_amount | 封单额(另一种统计) |
| first_time / last_time | 首次/最后时间 |
| open_times | 开板次数 |
| limit | L=涨停, Z=跌停 |
| limit_times | 连板/跌停次数 |

---

### 1.6 其他频率行情

| 文件 | 大小 | 列数 | 字段 | 已用 |
|---|---|---|---|---|
| `kline_adj_weekly.parquet` | 45M | 11 | OHLC+vol+amount(trade_date) | mom_4w, mom_12w |
| `kline_adj_monthly.parquet` | 11M | 11 | 同上 | mom_3m, mom_6m |
| `kline_weekly.parquet` | 49M | 12 | OHLC+pre_close+vol+amount | **未用** |
| `kline_monthly.parquet` | 12M | 12 | 同上 | **未用** |

---

### 1.7 股东与质押

| 文件 | 大小 | 列数 | 字段 | 已用 |
|---|---|---|---|---|
| `holder_number.parquet` | 1.9M | 4 | stock_code, ann_date, end_date, holder_num | holder_num_change |
| `pledge_stat.parquet` | 739K | 7 | stock_code, end_date, pledge_count, unrest_pledge, rest_pledge, total_share, pledge_ratio | pledge_ratio |

---

### 1.8 辅助数据

| 文件 | 大小 | 列数 | 字段 | 说明 |
|---|---|---|---|---|
| `calendar.parquet` | 15K | 2 | date, is_open | 交易日历 |
| `stock_list.parquet` | 222K | 11 | stock_code, name, area, industry, list_date, list_status, delist_date, is_hs 等 | 股票列表（含行业、地区） |
| `list.parquet` | 222K | 11 | 同 stock_list | 另一个股票列表副本 |

---

### 1.9 尚未使用的上游数据（高价值）

#### THS 行业/板块体系

| 文件 | 大小 | 列数 | 字段 |
|---|---|---|---|
| `ths_daily.parquet` | 53M | 12 | ths_code, trade_date, OHLC, pre_close, avg_price, vol, turnover_rate |
| `ths_constituent_stocks.parquet` | 1.4M | 5 | index_code, code, stock_code, stock_name, plate_name |
| `ths_sector_categories.parquet` | 39K | 7 | index_code, name, count, exchange, list_date, type, code |

**价值**: THS 行业分类比申万更细（约 300+ 个板块），ths_daily 有板块指数日线，可构建**行业轮动因子**、**板块内相对强度**、**板块趋势因子**。

#### 指数权重

| 文件 | 大小 | 列数 | 字段 |
|---|---|---|---|
| `index_weight.parquet` | 286K | 4 | index_code, stock_code, trade_date, weight |

**价值**: 可构建**指数成分股效应因子**（沪深300/中证500成分股的身份溢价）、**权重变化因子**（调入调出效应）。

---

### 1.10 按股票分文件存储的数据（大量未使用）

#### 分钟级行情 (per-stock parquet, ~3310 只/目录)

| 目录 | 大小 | 字段 |
|---|---|---|
| `min_adj_15min/` | 2.0G | stock_code, trade_time, OHLC, vol, amount **(复权)** |
| `min_adj_30min/` | 1.1G | 同上 |
| `min_adj_60min/` | 583M | 同上 |
| `history_15min/` | 2.0G | 同上 **(不复权)** |
| `history_30min/` | 1.1G | 同上 |
| `history_60min/` | 588M | 同上 |

**价值**: 可构建**日内因子**——开盘强度、尾盘拉升、日内波动率、日内反转、盘中最大回撤、VWAP 偏离等。这是当前因子体系最明显的空白。

#### CYQ 筹码分布 (per-stock parquet, 3194 只)

| 目录 | 大小 | 字段 |
|---|---|---|
| `cyq_chips/` | 152M | trade_date, stock_code, price, percent |

**价值**: 可构建**筹码结构因子**——获利盘比例、筹码集中度、支撑/压力位距离、筹码峰位置等。

---

## 2. 数据加载机制

### 2.1 两种加载方式

```python
# 日频面板：直接加载，自动规范化为 (Date, Code) MultiIndex
context.load("daily_adj.parquet")

# 报告期财务数据：自动 forward-fill 为日频面板
context.load_financial(
    "financial_indicator.parquet",
    value_cols=["roe", "roa"],
    date_col="ann_date",           # 默认，也可用 "end_date"
)
```

### 2.2 关键约定

- **内部计算格式**: (Date, Code) MultiIndex 的 Series，因子函数内做运算，保持此格式即可
- **最终输出格式**: Date × Code 宽表（`ensure_single_factor_frame` 自动 pivot）
- **股票池过滤**: 自动通过 `stock_list.parquet` 中 `list_status == "L"` 和 `Code_num.txt` 白名单过滤
- **日期截断**: 当日 18:00 前以 `daily_adj.parquet` 最新日期为准（当日 ETL 尚未完成）
- **增量构建**: base(.fea) + incr(_incr.fea) 双文件，日常只重建增量

### 2.3 因子注册模板

```python
@register_factor(
    name="factor_name",
    description="因子描述（一句话）",
    category="category",        # price/timeseries/valuation/quality/event/fund_flow/financial/sector/neutral/target
    thesis="因子背后的逻辑/依据",
    dependencies=("上游文件1.parquet", "上游文件2.parquet"),
)
def factor_func(context: FactorContext):
    data = context.load("上游文件1.parquet")
    # ... 计算逻辑 ...
    return cross_sectional_rank(result)  # 统一截面排名输出
```

---

## 3. 当前因子体系

**共计 154 个因子**: 150 个特征因子 + 4 个目标标签。

### 3.1 各类别概览

| 类别 | 数量 | 上游数据 | 核心方向 |
|---|---|---|---|
| **price** | 31 | daily_adj, daily, kline_adj_weekly, kline_adj_monthly | 动量(8)、反转(3)、波动率(3)、振幅(1)、乖离(1)、RSI(1)、最大回撤(1)、偏度/峰度(2)、成交量/额比(2)、影线(2)、跳空(1)、价格位置(1)、均线收敛(1)、突破(1)、ATR(1)、涨跌幅波动(1) |
| **timeseries** | 4 | daily_adj | 动量加速度(1)、波动率历史分位(1)、价格自身分位(1)、成交量Z-score(1) |
| **valuation** | 13 | finance | 价值(BP/SP/DP/PE)(4)、规模(2)、自由流通占比(1)、换手率(3)、量比(1)、PE分位(1)、股息综合(1) |
| **quality** | 30 | financial_indicator + calendar | 盈利(ROE/ROA/ROIC等)(5)、利息保障(1)、利润率(2)、杠杆/偿债(4)、现金流质量(2)、运营效率(3)、成长(4)、单季度(8)、质量综合(1) |
| **event** | 12 | top_list, limit_up, limit_list, dragon_tiger | 龙虎榜(4)、涨停(4)、跌停(2)、换手强度(1)、席位分析(2)（部分仅上榜日有值） |
| **fund_flow** | 10 | main_fund_flow, margin_detail, daily_adj | 主力净流入(2)、订单规模(3)、融资融券(3)、资金趋势(1)、价量背离(1) |
| **margin** | 3 | margin_detail, finance, daily_adj | 杠杆比率(1)、买入强度(1)、做空压力(1) |
| **financial** | 24 | balancesheet, income, cashflow, holder_number, pledge_stat 等 + calendar | 资产质量(3)、资产结构(4)、利润质量(4)、现金流(4)、创新(1)、费用管控(1)、筹码(1)、质押(1)、运营(2)、稳定性(1)、折旧摊销(1)、投资强度(1) |
| **enhanced** | 5 | daily_adj | 波动率调整动量(2)、波动率调整反转(1)、收益稳定性/类Sharpe(1)、价量相关(1) |
| **sector** | 7 | ths_daily, ths_constituent_stocks, ths_sector_categories + daily_adj/finance | 板块内相对强度(1)、板块动量(1)、板块Beta(1)、板块内排名(2)、板块轮动(1)、板块分散度(1) |
| **neutral** | 11 | daily_adj/finance/financial_indicator/limit_up + stock_list | 行业中性化：BP、动量、价量相关、连板天数、ROE、ROA、毛利率、市销率、FCF收益率、换手率、波动率 |
| **target** | 4 | daily_adj | 未来 1/5/10/20 日开盘买入收益率 |

### 3.2 已有因子完整清单

#### price (31个)

| 因子名 | 类型 | 说明 |
|---|---|---|
| mom_5, mom_10, mom_20, mom_60 | 动量 | 5/10/20/60 日收益率截面排名 |
| mom_120_skip5 | 动量 | 120 日动量（跳过最近 5 日），剔除短期反转 |
| mom_4w, mom_12w | 周频动量 | 4/12 周收益率 |
| mom_3m, mom_6m | 月频动量 | 3/6 月收益率 |
| reversal_1, reversal_5, reversal_10 | 反转 | 负 1/5/10 日收益率 |
| volatility_20, volatility_60 | 波动率 | 20/60 日收益率标准差（低波排前） |
| downside_vol_20 | 波动率 | 20 日下行波动率 |
| pct_chg_vol_20 | 波动率 | 20 日涨跌幅波动率 |
| amplitude_20 | 振幅 | 20 日均振幅（低振幅排前） |
| bias_20 | 乖离 | close/MA20 - 1 |
| rsi_14 | 超买超卖 | 14 日 RSI（高 RSI 排后） |
| max_drawdown_60 | 回撤 | 60 日最大回撤（回撤大排前=反转预期） |
| ret_skew_20 | 分布 | 20 日收益偏度 |
| ret_kurt_20 | 分布 | 20 日收益峰度（高峰度排后） |
| volume_ratio_20 | 成交量 | 20 日相对成交量（放量排后） |
| amount_ratio_20 | 成交额 | 20 日相对成交额（放量排后） |
| shadow_upper_20 | K线形态 | 20 日均上影线比例 |
| shadow_lower_20 | K线形态 | 20 日均下影线比例 |
| gap_ratio_20 | K线形态 | 20 日均跳空比率 |
| price_position_60 | 价格位置 | (close-60d_low)/(60d_high-60d_low) |
| ma_convergence_20_60 | 均线 | MA20/MA60 - 1 |
| breakout_60 | 突破 | close/max(high,60) - 1 |
| atr_20 | 波幅 | 20 日 ATR（低 ATR 排前） |

#### timeseries (4个)

| 因子名 | 类型 | 说明 |
|---|---|---|
| ts_mom_accel_20 | 动量加速度 | 20 日动量与 20 日前动量的差值 |
| ts_vol_regime_60 | 波动率体制 | 当前 20 日波动率在自身 252 日历史中的分位数 |
| ts_price_self_rank_60 | 价格位置 | 收盘价在自身 60 日高低区间的相对位置 |
| ts_volume_zscore_20 | 成交量异常 | 当日成交量偏离 20 日均值的标准差数（高放量排后） |

#### valuation (13个)

| 因子名 | 说明 |
|---|---|
| bp | 1/PB |
| sp_ttm | 1/PS_TTM |
| dp_ttm | 滚动股息率 |
| pe_ttm | 滚动市盈率截面排名（低PE排前） |
| log_total_mv | 对数总市值（负值=小盘） |
| log_circ_mv | 对数流通市值（负值=小盘） |
| float_mv_ratio | 自由流通占比（负值排前） |
| turnover_20 | 20 日均换手率（低换手排前） |
| turnover_f_20 | 20 日均自由流通换手率 |
| turnover_vol_20 | 20 日换手率波动 |
| volume_ratio | 量比（负值排前） |
| pe_ttm_percentile | PE_TTM 历史分位（低分位排前） |
| dv_composite | dv_ratio + dv_ttm 综合 |

#### quality (30个)

| 因子名 | 类型 | 说明 |
|---|---|---|
| roe, roa, roic | 盈利 | ROE/ROA/ROIC |
| roe_waa, roe_dt | 盈利 | 加权平均 ROE / 扣非 ROE |
| gross_margin, netprofit_margin | 利润率 | 毛利率 / 净利率 |
| debt_to_assets | 杠杆 | 资产负债率（低排前） |
| current_ratio, quick_ratio, cash_ratio | 偿债 | 流动/速动/现金比率 |
| ebit_to_interest | 偿债 | 利息保障倍数 |
| ocf_to_profit | 现金流质量 | 经营现金流/净利润 |
| salescash_to_or | 现金流质量 | 销售收现/营收 |
| assets_turn | 运营效率 | 总资产周转率 |
| inv_turn | 运营效率 | 存货周转率 |
| ar_turn | 运营效率 | 应收账款周转率 |
| or_yoy | 成长 | 营收同比增速 |
| netprofit_yoy | 成长 | 净利润同比增速 |
| equity_yoy | 成长 | 净资产同比增速 |
| assets_yoy | 成长 | 总资产同比增速 |
| q_roe | 单季度 | 单季度 ROE |
| q_gsprofit_margin | 单季度 | 单季度毛利率 |
| q_netprofit_margin | 单季度 | 单季度净利率 |
| q_sales_yoy | 单季度 | 单季度营收同比 |
| q_netprofit_yoy | 单季度 | 单季度净利润同比 |
| q_profit_yoy | 单季度 | 单季度利润总额同比 |
| q_ocf_to_sales | 单季度 | 单季度经营现金流/营收 |
| q_eps | 单季度 | 单季度每股收益 |
| quality_composite | 综合 | ROE+ROA+毛利率+OCF 等权 |

#### event (12个)

| 因子名 | 说明 | 覆盖度 |
|---|---|---|
| dragon_tiger_net_rate | 龙虎榜净买率 | 仅上榜日 |
| dragon_tiger_amount_rate | 龙虎榜成交占比 | 仅上榜日 |
| dragon_tiger_buy_sell_ratio | 龙虎榜买卖比 | 仅上榜日 |
| top_list_turnover_intensity | 龙虎榜换手强度 | 仅上榜日 |
| dt_org_count | 龙虎榜机构席位数量 | 仅上榜日 |
| dt_top_net_rate | 龙虎榜前五席位净买集中度 | 仅上榜日 |
| limit_up_consecutive | 连板天数 | 仅涨停日 |
| limit_up_sealed_flow_ratio | 封单流比 | 仅涨停日 |
| limit_up_open_count_neg | 开板次数（负向） | 仅涨停日 |
| limit_up_sealed_amount | 封单金额 | 仅涨停日 |
| limit_down_open_times | 跌停开板次数 | 仅跌停日 |
| limit_down_pct_chg | 跌停跌幅 | 仅跌停日 |

#### fund_flow (10个)

| 因子名 | 说明 |
|---|---|
| mf_net_inflow_ratio | 主力净流入/成交额 |
| mf_net_inflow_5d | 5 日累计主力净流入率 |
| mf_big_order_ratio | (特大+大)单净买入/成交额 |
| mf_small_order_ratio | 小单净买入率（负向） |
| mf_big_small_divergence | (大单净买-小单净买)/成交额 |
| mf_net_inflow_trend_5d | 5 日主力净流入趋势斜率 |
| big_order_divergence | rank(大单净买)-rank(pct_chg) |
| margin_buy_strength | 融资净买/融资余额 |
| margin_balance_change | 融资余额变化率 |
| margin_short_pressure | 融券余额/两融余额（负向） |

#### margin (3个)

| 因子名 | 说明 |
|---|---|
| margin_leverage_ratio | 融资余额/流通市值（负向） |
| margin_buy_intensity_5d | 5 日累计融资买入/5 日总成交额 |
| short_sale_pressure_5d | 5 日融券余量变化率（负向） |

#### financial (24个)

| 因子名 | 类型 | 说明 |
|---|---|---|
| goodwill_risk | 资产质量 | 商誉/净资产（负向） |
| inventory_pressure | 资产质量 | 存货/总资产（负向） |
| receivable_pressure | 资产质量 | 应收账款/总资产（负向） |
| fix_asset_ratio | 资产结构 | 固定资产/总资产（轻资产排前） |
| contract_liab_ratio | 资产结构 | 合同负债/营收（预收款质量） |
| cash_to_assets | 资产结构 | 货币资金/总资产（现金充裕度） |
| short_term_debt_ratio | 资产结构 | 短期借款/总资产（负向） |
| intan_assets_ratio | 资产结构 | 无形资产/总资产 |
| operating_profit_purity | 利润质量 | 营业利润/利润总额 |
| non_oper_profit_ratio | 利润质量 | 非经常性损益占比（负向） |
| credit_impair_risk | 利润质量 | 信用减值损失/营业利润（负向） |
| assets_impair_risk | 利润质量 | 资产减值损失/营业利润（负向） |
| rd_intensity | 创新 | 研发费用/营收 |
| expense_control | 费用管控 | 三费/营收（低费用排前） |
| invest_income_reliance | 利润质量 | 投资收益/营业利润（负向） |
| fcf_yield | 现金流 | 自由现金流/总市值 |
| ncf_act_to_revenue | 现金流 | 经营现金流/营收 |
| financing_dependency | 现金流 | 筹资现金流/经营现金流（负向） |
| investment_intensity | 现金流 | 投资活动现金流/总资产 |
| depr_amort_to_revenue | 现金流 | 折旧摊销/营收（负向） |
| holder_num_change | 筹码 | 股东户数环比变化（减少排前） |
| pledge_ratio | 风险 | 股权质押比例（负向） |
| cash_conversion_cycle | 运营 | 存货+应收-应付周转天数（短排前） |
| gross_margin_stability_8q | 稳定性 | 8 季度毛利率标准差（低波动排前） |

#### enhanced (5个)

| 因子名 | 说明 |
|---|---|
| mom_20_vol_adj | mom_20 / volatility_60 |
| mom_60_vol_adj | mom_60 / volatility_60 |
| reversal_5_vol_adj | -ret_5 / vol_20 |
| return_stability_60 | 60 日收益率/60 日波动率（类 Sharpe） |
| price_volume_corr_20 | 20 日量价相关系数 |

#### neutral (11个)

| 因子名 | 说明 |
|---|---|
| bp_neutral | 行业内 BP 排名 |
| mom_20_neutral | 行业内 mom_20 排名 |
| price_volume_corr_20_neutral | 行业内价量相关排名 |
| limit_up_consecutive_neutral | 行业内连板天数排名 |
| roe_neutral | 行业内 ROE 排名 |
| roa_neutral | 行业内 ROA 排名 |
| gross_margin_neutral | 行业内毛利率排名 |
| sp_ttm_neutral | 行业内市销率倒数排名 |
| fcf_yield_neutral | 行业内 FCF 收益率排名 |
| turnover_20_neutral | 行业内换手率排名 |
| volatility_20_neutral | 行业内波动率排名 |

#### sector (7个)

| 因子名 | 说明 |
|---|---|
| sector_rel_strength_20 | 个股相对所属THS板块的20日超额收益 |
| sector_momentum_20 | 所属THS板块的20日均收益率 |
| sector_beta_60 | 个股60日收益率对板块收益率的滚动Beta |
| sector_mv_rank | 个股在所属THS板块内的市值排名 |
| sector_amount_rank | 个股在所属THS板块内的成交额排名 |
| sector_rotation_20 | 板块截面排名20日变化的绝对值（高轮动排后） |
| sector_diversification | 个股所属THS板块数量（高分散排后） |

---

## 4. 因子开发基础设施

### 4.1 目录结构

```
featureengineering/
├── build_factors.py          # 入口：构建因子
├── src/featureengineering/
│   ├── cli.py                # CLI 参数解析
│   ├── builder.py            # 构建调度（增量判断、并行）
│   ├── storage.py            # 读写 .fea 文件
│   ├── dataset.py            # 上游数据加载 + 缓存
│   ├── registry.py           # 因子注册表 + FactorSpec/FactorContext
│   ├── settings.py           # 路径配置
│   ├── utils.py              # 截面排名、滚动统计等工具函数
│   ├── ic_analysis.py        # IC/ICIR 分析
│   └── factors/              # 因子定义
│       ├── price.py          # 31 个量价 + 4 个时间序列因子
│       ├── valuation.py      # 13 个估值因子
│       ├── quality.py        # 30 个质量因子
│       ├── event.py          # 12 个事件因子
│       ├── fund_flow.py      # 10 个资金流因子
│       ├── margin.py         # 3 个融资融券因子
│       ├── deep_finance.py   # 24 个深度财务因子
│       ├── enhanced.py       # 5 个增强因子
│       ├── sector.py         # 7 个板块因子
│       ├── neutral.py        # 11 个行业中性化因子
│       └── target/ret.py     # 4 个目标标签
├── data/
│   ├── factors/              # 因子输出 (.fea + .json manifest)
│   └── targets/              # target 输出
└── FACTOR_DEV_GUIDE.md       # 本文档
```

### 4.2 输出格式

- `.fea` 文件：Date × Code 宽表（1784 个交易日 × 2542 只股票）
- `.json` manifest：元数据（行数、覆盖率、最后日期等）
- 所有因子经 `cross_sectional_rank` 输出截面百分位排名 (0~1)

### 4.3 运行方式

```bash
python build_factors.py --all              # 构建全部因子
python build_factors.py --factor mom_20    # 构建单个因子
python build_factors.py --list            # 列出所有因子
python build_factors.py --check-dates     # 检查因子数据时效
python build_factors.py --targets-only    # 只构建 target
```

---

## 5. 可开发方向分析

### 5.1 尚未使用的上游数据 → 全新因子类别

#### A. 分钟级数据 (min_adj + history) — **日内因子**

6 个目录共 3310×6 个文件，总数据量 ~7.4G。这是最大的未开发领域。

| 方向 | 具体思路 | 所需数据 |
|---|---|---|
| **开盘强度** | 开盘价相对前收盘的跳空幅度在日内被回补的比例（开盘方向可靠性） | min_adj_15min |
| **尾盘效应** | 最后 30min 收益率 vs 全天收益率、尾盘拉升/砸盘频率 | min_adj_30min/60min |
| **日内波动率** | 日内高低点振幅、(High-Low)/Open，日内已实现波动率 | min_adj |
| **日内反转** | 上午跌下午涨的频率/幅度——反映抄底力量 | min_adj_60min |
| **VWAP 偏离** | 收盘价相对 VWAP 的偏离度（机构行为信号） | min_adj_15min |
| **开盘半小时成交占比** | 开盘前 30 分钟的成交额/全天成交额——机构参与度 | min_adj_30min |
| **盘中最大回撤** | 日内从最高点到最低点的最大回撤幅度 | min_adj |
| **区间成交量分布** | 各时段成交量占比的稳定性（信息均匀释放 vs 集中释放） | min_adj_60min |

#### B. CYQ 筹码分布 — **筹码结构因子**

| 方向 | 具体思路 |
|---|---|
| **获利盘比例** | 当前价以下筹码占比——浮盈压力 |
| **筹码集中度** | 筹码分布的离散度（标准差/峰度）——主力控盘度 |
| **支撑/压力距离** | 筹码密集峰距当前价的距离——技术支撑/压力位 |
| **筹码峰变化** | 筹码峰的迁移速度与方向——主力吸筹/出货痕迹 |
| **套牢盘比例** | 当前价以上的筹码占比——上方抛压 |

#### C. THS 行业/板块体系 — **板块因子**（已实现，见 sector.py）

| 方向 | 具体思路 | 状态 |
|---|---|---|
| **板块内相对强度** | 个股涨跌幅 - 所属 THS 板块指数涨跌幅 | 已实现 |
| **板块趋势强度** | 板块指数的动量/均线状态 | 已实现 |
| **板块轮动速度** | 板块排名的日度变化率 | 已实现 |
| **板块关联度** | 个股与板块指数的 beta/相关系数 | 已实现 |
| **龙头识别** | 个股在板块内的市值/成交额占比 | 已实现 |

#### D. 指数权重 — **指数效应因子**

| 方向 | 具体思路 |
|---|---|
| **成分股身份** | 是否属于沪深300/中证500/中证1000 |
| **权重变化** | 指数调仓前后权重变化 |
| **调入调出预期** | 基于市值排名的调入概率 |

---

### 5.2 已用数据中尚未利用的字段 → 现有类别扩展

#### finance.parquet
| 未用字段 | 可构建因子 | 状态 |
|---|---|---|
| pe | PE 估值因子（绝对值，pe_ttm 已实现） | -- |
| ps | PS 估值因子（目前仅有 PS_TTM） | 待开发 |
| float_share | 流通股本变化因子 | 待开发 |

#### financial_indicator.parquet（167 列中约 130+ 列未用）

| 方向 | 可用字段 | 状态 |
|---|---|---|
| **单季度指标** | q_roe, q_gsprofit_margin, q_netprofit_margin, q_sales_yoy 等 20+ 个 | 已实现 8 个 |
| **盈利质量** | ebit, ebitda, npta, fcff, fcfe, ocfps, cfps | 待开发 |
| **偿债深度** | ebit_to_interest, ocf_to_debt, ocf_to_shortdebt | 已实现 ebit_to_interest |
| **营运效率** | inv_turn, ar_turn, ca_turn, fa_turn, turn_days | 已实现 inv_turn, ar_turn |
| **YoY 系列** | roe_yoy, bps_yoy, cfps_yoy, ocf_yoy, basic_eps_yoy, dt_eps_yoy | 待开发 |

#### balancesheet.parquet（157 列中约 135+ 列未用）

| 方向 | 可用字段 | 状态 |
|---|---|---|
| **货币资金** | money_cap / total_assets——现金充裕度 | 已实现 |
| **无形与研发资产** | intan_assets + r_and_d——知识资产密集度 | 已实现 intan_assets |
| **商誉细分** | goodwill / (goodwill + total_hldr_eqy)——更细的商誉风险 | 待开发 |
| **借款到期压力** | st_borr / lt_borr——短期 vs 长期债务结构 | 已实现 st_borr |
| **合同负债** | contract_liab / revenue——预收款质量（茅台指标） | 已实现 |
| **使用权资产** | right_of_use_assets / total_assets——经营租赁依赖（新准则） | 待开发 |

#### income.parquet（96 列中约 75+ 列未用）

| 方向 | 可用字段 | 状态 |
|---|---|---|
| **利润稳定性** | non_oper_income, non_oper_exp——非经常性损益占比 | 已实现 |
| **减值风险** | credit_impair_loss, assets_impair_loss——减值损失 | 已实现 |
| **EPS 增速** | basic_eps, diluted_eps, dt_eps_yoy | 待开发 |
| **利息覆盖** | ebit / fin_exp——利息保障倍数 | 待开发 |

#### cashflow.parquet（97 列中约 85+ 列未用）

| 方向 | 可用字段 | 状态 |
|---|---|---|
| **投资强度** | n_cashflow_inv_act / total_assets——资本开支力度 | 已实现 |
| **折旧摊销** | depr_fa_coga_dpba + amort_intang_assets——维持性资本支出 | 已实现 |
| **分红能力** | c_pay_dist_dpcp_int_exp / net_profit——分红率 | 待开发 |
| **间接法细节** | 约 30 个间接法调整项——应计项目质量 | 待开发 |

---

### 5.3 因子方法论空白 → 跨类别或方法论创新

#### A. 时间序列因子（已实现 4 个基础版）

已实现 ts_mom_accel_20、ts_vol_regime_60、ts_price_self_rank_60、ts_volume_zscore_20（见 price.py 末尾）。可进一步扩展的方向：

| 方向 | 说明 | 状态 |
|---|---|---|
| **自身历史分位** | 扩展到更多因子（ROE历史分位、换手率历史分位等） | 待扩展 |
| **加速/减速因子** | 扩展到更多周期（5日加速、60日加速） | 待扩展 |
| **波动率聚集** | GARCH 类条件波动率建模 | 待开发 |
| **异常检测** | 扩展到多维 Z-score 异常检测 | 待开发 |

#### B. 非线性因子

| 方向 | 说明 |
|---|---|
| **极值因子** | 过去 N 日是否出现涨跌停、是否出现天量——事件型二值因子 |
| **阈值交互** | 动量 × 波动率低于某阈值时的特殊表现 |
| **顶部/底部识别** | 价格在 N 日最高/最低的附近天数 |

#### C. 行业中性化扩展（已实现 11 个）

已实现 bp、mom_20、price_volume_corr_20、limit_up_consecutive、roe、roa、gross_margin、sp_ttm、fcf_yield、turnover_20、volatility_20 的行业中性化版本（见 neutral.py）。可进一步扩展的方向：

- valuation 类：dp_ttm, pe_ttm_percentile 等
- quality 类：netprofit_margin, q_roe 等都受行业影响显著
- financial 类：goodwill_risk, cash_conversion_cycle 等

#### D. 因子交互/合成

| 方向 | 说明 |
|---|---|
| **动量+低波** | mom_20 × (-volatility_20)——动量与低波的交集 |
| **价值+质量** | bp × roe——好公司+好价格 |
| **资金流确认** | mom_20 × mf_net_inflow_ratio——量价配合 |
| **多因子等权组合** | 按类别做等权 composite（如 quality_composite 已有，可扩展到 value/momentum composite） |

#### E. 周频/月频直接因子

当前动量有周频/月频版本（mom_4w/mom_12w/mom_3m/mom_6m），但以下因子也可以做周频/月频版本：

| 方向 | 说明 |
|---|---|
| **周频波动率** | kline_adj_weekly 可直接算，避免日频叠加噪声 |
| **周频反转** | 周线级别的反转信号 |
| **周频换手率** | 周频换手率比日频更稳定 |

---

### 5.4 未使用的现成文件

| 文件 | 潜在用途 |
|---|---|
| `kline_weekly.parquet` | 周频影线、跳空、ATR（不复权周线） |
| `kline_monthly.parquet` | 月频形态因子（不复权月线） |
| `list.parquet` | 作为 stock_list 的替代/验证 |

---

---

## 附录：开发新因子的检查清单

- [ ] 确定因子类别（category），确认不在已有因子清单中
- [ ] 确定上游数据依赖（dependencies），确认文件存在且字段可用
- [ ] 财务数据优先考虑 `load_financial` 的 `value_cols` 参数做列级加载
- [ ] 因子逻辑写清 thesis（一句话，解释为什么这个因子应该有效）
- [ ] 输出经 `cross_sectional_rank` 处理为截面排名
- [ ] 仅在上榜日/事件日有值的因子，需考虑非事件日的 NaN 填充策略
- [ ] 行业中性化因子复用 `_industry_neutral_rank` 辅助函数
- [ ] 运行 `python build_factors.py --factor <name> --force` 测试
- [ ] 运行 IC 分析验证因子对 target 的预测力
