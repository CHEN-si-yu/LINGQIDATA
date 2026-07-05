# 上游未使用字段分析与因子开发报告

> 生成日期: 2026-07-02
> 股票池: `/autodl-fs/data/lingqiData/Code_num.txt` (1782只)
> 时间范围: 2020-01-01 至 2026-07-01
> 新因子文件: `featureengineering/src/featureengineering/factors/unused_fields_factors.py`

---

## 一、数据全景

### 1.1 上游数据源总览

| 数据源 | 列数 | 频率 | 日期范围 | 股票覆盖率 | NaN 平均率 |
|--------|------|------|----------|-----------|-----------|
| `daily.parquet` | 9 | 日频 | 2020-01-02 ~ 2026-07-01 | 1782/1782 (100%) | 0.00% |
| `daily_adj.parquet` | 8 | 日频 | 2020-01-02 ~ 2026-07-01 | 1782/1782 (100%) | 0.00% |
| `finance.parquet` | 17 | 日频 | 2020-01-02 ~ 2026-07-01 | 1782/1782 (100%) | 0.09% |
| `cyq_perf.parquet` | 9 | 日频 | 2020-01-02 ~ 2026-07-01 | 1782/1782 (100%) | 0.20% |
| `main_fund_flow.parquet` | 18 | 日频 | 2020-01-02 ~ 2026-07-01 | 1782/1782 (100%) | 0.00% |
| `margin_detail.parquet` | 7 | 日频 | 2020-01-02 ~ 2026-06-30 | 1782/1782 (100%) | 0.08% |
| `dragon_tiger.parquet` | 5 | 事件 | 2020-01-02 ~ 2026-07-01 | 1701/1782 (95.5%) | 0.00% |
| `top_list.parquet` | 11 | 事件 | 2020-01-02 ~ 2026-07-01 | 1689/1782 (94.8%) | 0.00% |
| `limit_list.parquet` | 13 | 事件 | 2020-01-02 ~ 2026-07-01 | 1772/1782 (99.4%) | 7.52% |
| `limit_up.parquet` | 11 | 事件 | 2020-01-02 ~ 2026-07-01 | 1767/1782 (99.2%) | 0.88% |
| `index_weight.parquet` | 1 | 月频 | 2020-01-01 ~ 2026-04-01 | 1636/1782 (91.8%) | 0.00% |
| `holder_number.parquet` | 1 | 报告期 | 2020-01-02 ~ 2026-06-30 | 1782/1782 (100%) | 0.00% |
| `pledge_stat.parquet` | 5 | 月频 | 2020-04-30 ~ 2026-04-30 | 1776/1782 (99.7%) | 0.00% |
| `balancesheet.parquet` | 149 | 季频 | 2020-03-31 ~ 2026-03-31 | 1782/1782 (100%) | 0.28% |
| `income.parquet` | 88 | 季频 | 2020-03-31 ~ 2026-03-31 | 1782/1782 (100%) | 0.93% |
| `cashflow.parquet` | 89 | 季频 | 2020-03-31 ~ 2026-03-31 | 1782/1782 (100%) | 0.08% |
| `financial_indicator.parquet` | 163 | 季频 | 2020-03-31 ~ 2026-03-31 | 1782/1782 (100%) | 0.02% |

### 1.2 已加载的数据源（现有因子已使用）

通过 `context.load()` 直接加载：`daily.parquet`, `daily_adj.parquet`, `finance.parquet`, `cyq_perf.parquet`, `main_fund_flow.parquet`, `margin_detail.parquet`, `limit_list.parquet`, `limit_up.parquet`

通过 `context.load_financial()` 加载：`balancesheet.parquet`, `income.parquet`, `cashflow.parquet`, `financial_indicator.parquet`, `holder_number.parquet`, `pledge_stat.parquet`, `index_weight.parquet`

---

## 二、未使用字段分析

### 2.1 各数据源未使用字段统计

| 数据源 | 总值列 | 已使用 | 未使用 | 利用率 |
|--------|--------|--------|--------|--------|
| `daily.parquet` | 9 | 9 | 0 | 100% |
| `daily_adj.parquet` | 8 | 8 | 0 | 100% |
| `cyq_perf.parquet` | 9 | 9 | 0 | 100% |
| `margin_detail.parquet` | 7 | 7 | 0 | 100% |
| `holder_number.parquet` | 1 | 1 | 0 | 100% |
| `pledge_stat.parquet` | 5 | 5 | 0 | 100% |
| `index_weight.parquet` | 1 | 1 | 0 | 100% |
| `finance.parquet` | 17 | 16 | **1** | 94.1% |
| `limit_up.parquet` | 11 | 8 | **3** | 72.7% |
| `main_fund_flow.parquet` | 18 | 9 | **9** | 50.0% |
| `limit_list.parquet` | 13 | 5 | **8** | 38.5% |
| `dragon_tiger.parquet` | 5 | 0 | **5** | 0% |
| `top_list.parquet` | 11 | 4 | **7** | 36.4% |
| `financial_indicator.parquet` | 163 | 78 | **85** | 47.9% |
| `income.parquet` | 88 | 23 | **65** | 26.1% |
| `balancesheet.parquet` | 149 | 35 | **114** | 23.5% |
| `cashflow.parquet` | 89 | 15 | **74** | 16.9% |

### 2.2 按优先级分类的未使用字段

#### 🟢 高优先级（低NaN、高覆盖、清晰因子逻辑）

| 字段 | 来源 | NaN | Zero% | 可开发因子 |
|------|------|-----|-------|-----------|
| `fcff` | financial_indicator | 0% | 4.3% | FCF Yield, FCF Conversion |
| `fcfe` | financial_indicator | 0% | 4.6% | FCFE Yield |
| `working_capital` | financial_indicator | 0% | 4.6% | WC/Assets 效率因子 |
| `invest_capital` | financial_indicator | 0% | 4.6% | IC Turnover |
| `retained_earnings` | financial_indicator | 0% | 0% | 留存收益/总资产 (F-Score组件) |
| `interestdebt` | financial_indicator | 0% | 6.4% | 有息负债率 |
| `netdebt` | financial_indicator | 0% | 4.6% | 净负债/EBITDA |
| `q_roe` | financial_indicator | 0% | 0.2% | 单季度ROE |
| `q_netprofit_qoq` | financial_indicator | 0% | 0% | 季度盈利环比增速 |
| `q_sales_qoq` | financial_indicator | 0% | 0.1% | 季度营收环比增速 |
| `q_ocf_to_sales` | financial_indicator | 0% | 0.2% | 季度经营现金流质量 |
| `q_netprofit_margin` | financial_indicator | 0% | 0.2% | 季度净利率 |
| `dt_eps` | financial_indicator | 0% | 2.0% | 稀释EPS Yield |
| `cogs_of_sales` | financial_indicator | 0% | 4.6% | 营业成本率 |
| `expense_of_sales` | financial_indicator | 0% | 4.6% | 期间费用率 |
| `goodwill` | balancesheet | 0% | 38.9% | 商誉/净资产 |
| `lt_eqt_invest` | balancesheet | 0% | 15.0% | 长期股权投资/总资产 |
| `contract_liab` | balancesheet | 0% | 5.6% | 合同负债/营收 |
| `minority_int` | balancesheet | 0% | 9.6% | 少数股东权益/净资产 |
| `defer_tax_assets` | balancesheet | 0% | 1.3% | 递延所得税资产/总资产 |
| `r_and_d` | balancesheet | 0% | 73.1% | R&D/营收（科技股专用） |
| `rd_exp` | income | 0% | 11.7% | 研发费用/营收 |
| `basic_eps` | income | 0% | 0.2% | EPS raw value |
| `fin_exp_int_exp` | income | 0% | 9.0% | 利息费用分解 |
| `fin_exp_int_inc` | income | 0% | 6.1% | 利息收入分解 |
| `ass_invest_income` | income | 0% | 23.4% | 联营投资收益/利润 |
| `biz_tax_surchg` | income | 0% | 0.02% | 营业税金率 |
| `c_fr_sale_sg` | cashflow | 0% | 4.3% | 销售收现比 |
| `c_paid_goods_s` | cashflow | 0% | 4.4% | 采购付现比 |
| `c_recp_borrow` | cashflow | 0% | 16.8% | 借款占比 |
| `n_incr_cash_cash_equ` | cashflow | 0% | 0% | 现金净变化率 |
| `stot_inflows_inv_act` | cashflow | 0% | 3.4% | 投资活动总流入 |
| `stot_cash_in_fnc_act` | cashflow | 0% | 8.7% | 筹资活动总流入 |
| Volume columns (×9) | main_fund_flow | 0% | 0-21% | 量价背离、订单熵 |
| `open_times` | limit_list | 0% | 31.9% | 涨停开板频率 |
| `first_time` | limit_list | 0% | 0% | 首次涨停时间信号 |
| `turnover_ratio` | limit_list | 0% | 0% | 涨停日换手率 |
| `ps` | finance | 0% | — | 未调整市销率 |

#### 🟡 中优先级（有一定NaN或特殊行业适用）

| 字段 | 来源 | NaN | 备注 |
|------|------|-----|------|
| `total_nca` / `total_ncl` | balancesheet | 0% | 非流动资产负债结构 |
| `defer_tax_liab` | balancesheet | 0% | 递延税负债(税务筹划信号) |
| `minority_gain` | income | 0% | 少数股东损益占比 |
| `eff_fx_flu_cash` | cashflow | 0% | 汇率影响(涉外企业) |
| `stot_cashout_fnc_act` | cashflow | 0% | 筹资活动总流出 |

#### 🔴 低优先级（高NaN、高零值、仅特定行业有效）

大部分 `balancesheet.parquet` 中的保险/银行专用科目（如 `premium_receiv`, `reinsur_receiv`, `depos` 等）和已废弃的会计准则科目（如 `fa_avail_for_sale`, `htm_invest` 等），以及 `income.parquet` 中的保险专项科目。

---

## 三、新增因子清单（54个）

### A. 自由现金流质量 (6个)
- `fcf_yield` — FCFF/总市值
- `fcfe_yield` — FCFE/总市值
- `fcf_conversion` — (FCFF-OCF)/|OCF|，资本开支强度
- `invest_capital_turnover` — 营收/投入资本
- `cash_flow_quality_composite` — 收现比+经营CF占比+现金变化率

### B. 盈利结构质量 (8个)
- `non_operating_profit_ratio` — 非经营利润/利润总额 (越低越好)
- `operating_profit_to_ebt` — 经营利润/EBT
- `deducted_profit_ratio` — 扣非净利润/净利润
- `earnings_structure_quality` — 三因子合成
- `cogs_ratio` — 营业成本率 (越低越好)
- `expense_to_sales` — 期间费用率 (越低越好)
- `tax_burden_ratio` — 实际税率
- `net_interest_margin` — 净利息收入/营收

### C. 季度高频率因子 (5个)
- `q_netprofit_qoq` — 单季度归母净利润环比
- `q_sales_qoq` — 单季度营收环比
- `q_roe` — 单季度ROE
- `q_netprofit_margin` — 单季度净利率
- `q_ocf_to_sales` — 单季度经营现金流/营收

### D. 杠杆与资本结构 (5个)
- `net_debt_to_ebitda` — 净负债/EBITDA
- `interest_bearing_debt_ratio` — 有息负债率
- `equity_to_invested_capital` — 权益/投入资本
- `int_bearing_debt_to_capital` — 有息负债/总资本
- `working_capital_to_assets` — 营运资本/总资产

### E. 资产负债表新增 (7个)
- `goodwill_to_equity` — 商誉/净资产 (越低越好)
- `rd_to_revenue` — R&D/营收 (bs口径)
- `lt_equity_invest_to_assets` — 长期股权投资/总资产
- `deferred_tax_asset_ratio` — 递延所得税资产率
- `deferred_tax_liab_ratio` — 递延所得税负债率
- `minority_interest_ratio` — 少数股东权益比
- `contract_liab_to_revenue` — 合同负债/营收
- `non_current_asset_ratio` — 非流动资产比
- `non_current_liab_ratio` — 非流动负债比

### F. 利润表新增 (6个)
- `rd_expense_to_revenue` — 研发费用/营收 (is口径, 更准)
- `financial_expense_ratio` — 财务费用率
- `associate_invest_income_ratio` — 联营投资收益/利润
- `minority_gain_ratio` — 少数股东损益/净利润
- `diluted_eps_yield` — 稀释EPS/股价
- `eps_growth_4q_qoq` — 稀释EPS 4Q增速

### G. 现金流量表新增 (7个)
- `operating_cf_to_total_inflow` — 经营活动CF/总现金流入
- `cash_from_sales_to_revenue` — 销售收现比
- `cash_paid_goods_to_cogs` — 采购付现比
- `borrowing_to_cash_inflow` — 借款/筹资流入
- `financing_cf_gap` — 筹资活动CF/总资产
- `cash_change_ratio` — 现金净变化/期初现金
- `fx_impact_on_cash` — 汇率影响

### H. 资金流量价分析 (4个)
- `mf_big_order_vol_ratio` — 大单成交量占比
- `mf_amount_vol_divergence` — 成交额vs成交量背离
- `mf_net_vol_intensity` — 净流入量/总成交量
- `mf_order_size_entropy` — 订单规模分布熵

### I. 涨停事件增强 (3个)
- `limit_open_frequency_20d` — 20日涨停开板频率
- `limit_first_time_signal` — 平均首次涨停时间
- `limit_turnover_intensity` — 涨停日换手率

### J. 估值 (1个)
- `sp_raw` — 未调整市销率 (ps)

---

## 四、集成指南

### 4.1 构建新因子

```bash
# 进入项目目录
cd /autodl-fs/data/lingqiData/featureengineering

# 构建所有新因子 (Class 1, 因为依赖不含 cyq_chips/history_1min/__factors__)
python build_factors.py --only-class 1 --all

# 或只构建指定因子
python build_factors.py --factor fcf_yield fcfe_yield q_roe goodwill_to_equity

# 带质量检查
python build_factors.py --only-class 1 --all --quality-check-days 5 --dashboard
```

### 4.2 新因子注册机制

新因子通过 `factor_loader.py` 的 `pkgutil.walk_packages` 自动发现，无需修改任何配置文件。添加文件后直接构建即可。

### 4.3 Class 分类

新因子都属于 **Class 1 (Panel)**，因为依赖只包含 `.parquet` 文件：
- 不含 `cyq_chips` → 不是 Class 2
- 不含 `history_1min` → 不是 Class 3
- 不含 `__factors__` → 不是 Class 4

### 4.4 如需添加更多因子

编辑 `/autodl-fs/data/lingqiData/featureengineering/src/featureengineering/factors/unused_fields_factors.py`，按以下模式添加：

```python
@register_factor(
    name="your_factor_name",
    description="因子中文描述",
    category="quality",  # quality/valuation/growth/fund_flow/event
    thesis="投资逻辑/学术理论基础",
    dependencies=("source.parquet", "calendar.parquet"),
)
def factor_your_factor_name(context: FactorContext):
    # 日频数据：context.load("source.parquet")
    # 财务数据：context.load_financial("source.parquet", value_cols=["col1", "col2"])
    # 计算逻辑...
    return cross_sectional_rank(result)
```

---

## 五、关键注意事项

### 5.1 数据覆盖结论
所有推荐开发的字段在目标股票池(1782只)和时间范围(2020-2026)内覆盖率均 >95%，NaN 率 <5%。数据质量足以支撑因子开发。

### 5.2 废弃/不推荐开发的字段
以下字段虽然未使用但**不建议开发**：
- **旧准则科目**: `fa_avail_for_sale`, `htm_invest` (新准则已弃用，99%+为零)
- **保险/银行专有科目**: `premium_receiv`, `reinsur_*`, `depos_*` 等 (仅金融企业有效，对全市场因子无意义)
- **高零值科目**: `forex_differ` (100%零), `amor_exp` (100%零), `undist_profit`/`distable_profit` (100%零)
- **高NaN科目**: `limit_amount` (85.6% NaN), `continued_net_profit`/`net_profit_before_tax` (>7% NaN且>92%零)

### 5.3 文件路径
- 代码实际路径: `/autodl-fs/data/lingqiData/` (不是 `/root/autodl-fs/lingqiData/`)
- 数据路径: `/autodl-fs/data/lingqiData/data/`
- 因子路径: `/autodl-fs/data/lingqiData/featureengineering/`
