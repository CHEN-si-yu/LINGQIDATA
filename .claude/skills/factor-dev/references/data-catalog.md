# Upstream Data Catalog

All data files live in the source root (default: `../data` relative to project root).

## Data Source Inventory

### Daily Market Data

| File | Rows | Time Range | Key Columns |
|------|------|-----------|-------------|
| `daily_adj.parquet` | 8.4M | 2019-now | stock_code, trade_date, open, high, low, close, change, pct_chg, vol, amount |
| `daily.parquet` | 8.4M | 2019-now | **Same + pre_close** (unadjusted prices, has pre_close for gap calculations) |

**Difference**: `daily.parquet` has `pre_close` and `change` (price change). `daily_adj.parquet` has adjusted prices (for factor calculation). Neither is currently used — all factors use `daily_adj.parquet`.

### Daily Financial/Valuation

| File | Rows | Time Range | Key Columns |
|------|------|-----------|-------------|
| `finance.parquet` | 8.4M | 2019-now | pe, pe_ttm, pe_ttm_percentile, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv, turnover_rate, turnover_rate_f, volume_ratio |

### Financial Statements (report-frequency → daily ffill)

| File | Rows | Time Range | Frequency | Key Columns |
|------|------|-----------|-----------|-------------|
| `financial_indicator.parquet` | 245K | 2019-now | 季/年 | 130+ cols: roe, roa, roic, gross_margin, netprofit_margin, eps, bps, ocfps, cfps, ebit, ebitda, fcff, fcfe, current_ratio, quick_ratio, debt_to_assets, equity_yoy, or_yoy, q_* (quarterly) ...  ⚠️ **DATA QUALITY**: All `q_*` columns are zero-filled (~95% of records) for 2019-2022. Do NOT use q_* fields for lookback-window factors before 2023. Non-q fields (eps, roe, etc.) are unaffected. |
| `balancesheet.parquet` | 151K | 2019-now | 季/年 | 150+ cols: total_assets, total_liab, total_hldr_eqy_exc_min_int, goodwill, inventories, accounts_receiv, fix_assets, intan_assets, contract_liab, st_borr, lt_borr, money_cap, rd_exp ... |
| `income.parquet` | 153K | 2019-now | 季/年 | 100+ cols: total_revenue, revenue, operate_profit, total_profit, n_income, ebit, ebitda, fin_exp, sell_exp, admin_exp, invest_income, non_oper_income ... |
| `cashflow.parquet` | 140K | 2019-now | 季/年 | 100+ cols: n_cashflow_act, n_cashflow_inv_act, n_cash_flows_fnc_act, free_cashflow, c_inf_fr_operate_a, depr_fa_coga_dpba ... |

### Fund Flow & Margin

| File | Rows | Time Range | Key Columns |
|------|------|-----------|-------------|
| `main_fund_flow.parquet` | 8.0M | 2019-now | buy_sm_amount, sell_sm_amount, buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount, buy_elg_amount, sell_elg_amount, net_mf_amount, net_mf_vol |
| `margin_detail.parquet` | 5.1M | 2019-now | rzye (融资余额), rzmre (融资买入额), rzche (融资偿还额), rqye (融券余额), rqmcl (融券卖出量), rzrqye (融资融券余额) |

### Event Data

| File | Rows | Time Range | Key Columns |
|------|------|-----------|-------------|
| `top_list.parquet` | 100K | 2019-now | l_sell, l_buy, l_amount, net_amount, net_rate, amount_rate, float_values, reason |
| `dragon_tiger.parquet` | 104K | 2019-now | org_name, buy_amount, buy_ratio, sell_amount, sell_ratio, net_buy_amount, direction, reason |
| `limit_up.parquet` | 123K | 2019-now | consecutive_days, sealed_volume, sealed_amount, sealed_turnover_ratio, sealed_flow_ratio, open_count, boards, is_limit_up |
| `limit_list.parquet` | 154K | 2019-now | limit_amount, float_mv, total_mv, turnover_ratio, fd_amount, first_time, last_time, open_times, up_stat, limit_times |

### Sector/Industry

| File | Rows | Time Range | Key Columns |
|------|------|-----------|-------------|
| `stock_list.parquet` | 5.8K | — | stock_code, name, industry, area, list_date, list_status |
| `list.parquet` | 5.8K | — | Same + delisted stocks (D status) |
| `ths_daily.parquet` | 1.4M | 2019-now | ths_code, trade_date, open, high, low, close, pct_change, vol, turnover_rate |
| `ths_constituent_stocks.parquet` | 233K | 2019-now | index_code, stock_code, plate_name |
| `ths_sector_categories.parquet` | 1.2K | — | index_code, name, type (I=industry), exchange |

### 🔴 Completely Unused Data

| File | Rows | Size | Time Range | Key Columns |
|------|------|------|-----------|-------------|
| `cyq_chips.parquet` | 461M | 811MB | 2019-now | trade_date, stock_code, **price, percent** (chip distribution histogram) |
| `cyq_perf.parquet` | 8.3M | 64MB | 2019-now | his_low, his_high, **cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct, weight_avg, winner_rate**, is_backfill |
| `holder_number.parquet` | 321K | 1.9MB | 2019-now | stock_code, ann_date, end_date, **holder_num** |
| `index_weight.parquet` | 368K | 1MB | 2019-now (restored) | index_code, stock_code, trade_date, **weight** (monthly, 8 major indices). ⚠️ Must pass `full_history=True` to fetch pre-2024 data |
| `pledge_stat.parquet` | 86K | 739KB | 2019-now | stock_code, end_date, pledge_count, unrest_pledge, rest_pledge, total_share, **pledge_ratio** |
| `history_1min/` | 2,542 files | 16GB | 2019-now | stock_code, trade_time, open, high, low, close, vol, amount (per-stock files) |
| `daily_dump_1min/` | 5 files | ~35MB | 5 days | trade_date, trade_time, stock_code, open, high, low, close, vol, amount (daily dump files) |

### Reference Tables

| File | Rows | Key Columns |
|------|------|-------------|
| `calendar.parquet` | 2.7K | date, is_open |

## Current Factor Data Usage Map

```
daily_adj.parquet     →  price (31), enhanced (5), target (4), neutral (3), sector (0-indirect)
finance.parquet       →  valuation (10), margin (2), neutral (6), deep_finance (3)
financial_indicator   →  quality (30), deep_finance (5), neutral (4)
balancesheet.parquet  →  deep_finance (15)
income.parquet        →  deep_finance (8)
cashflow.parquet      →  deep_finance (5)
main_fund_flow        →  fund_flow (10)
margin_detail         →  margin (2)
top_list              →  event (6)
limit_up              →  event (5)
limit_list            →  event (1)
dragon_tiger          →  event (3)
ths_daily             →  sector (3)
ths_constituent       →  sector (indirect)
stock_list            →  dataset.py (industry map)

cyq_chips             →  UNUSED
cyq_perf              →  UNUSED
holder_number         →  UNUSED (holder_num_change exists but uses different source)
index_weight          →  index_weight (5 factors: inclusion_recency, membership_count, weight_hs300, weight_change, diversification)
pledge_stat           →  UNUSED
history_1min          →  UNUSED
daily_dump_1min       →  UNUSED
daily.parquet         →  UNUSED
list.parquet          →  UNUSED
```

## DataRepository API

```python
class DataRepository:
    # Daily panel: (Date, Code) MultiIndex
    def load_panel(self, relative_path, min_date=None, max_date=None, lookback_days=0) -> pd.DataFrame

    # Financial panel: forward-filled from report frequency to daily
    def load_financial_panel(self, relative_path, value_cols=None, date_col="ann_date",
                              min_date=None, max_date=None, lookback_days=0) -> pd.DataFrame

    # Stock pool
    def load_stock_pool(self) -> pd.DataFrame           # columns: Code, industry, area
    def load_industry_map(self) -> pd.Series            # Code → industry
    def available_codes(self) -> pd.Index               # All valid stock codes
```

### FactorContext (what factor functions receive)

```python
class FactorContext:
    repo: DataRepository
    start_date: str | None   # YYYYMMDD — for incremental builds
    end_date: str | None     # YYYYMMDD — effective end date

    # Load daily panel (shortcut to repo.load_panel)
    def load(self, relative_path: str) -> pd.DataFrame

    # Load financial panel with auto daily-ffill (shortcut to repo.load_financial_panel)
    def load_financial(self, relative_path, value_cols=None, date_col="ann_date") -> pd.DataFrame
```
