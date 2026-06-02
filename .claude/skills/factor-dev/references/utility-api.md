# Utility API Reference

## Core Utilities (`utils.py`)

```python
from ..utils import (
    cross_sectional_rank,
    safe_divide,
    rolling_group_mean,
    rolling_group_std,
    rolling_group_sum,
    rolling_group_max,
    rolling_group_min,
)
```

### `cross_sectional_rank(series, winsorize=True, lower=0.01, upper=0.99)`

Cross-sectional percentile rank [0,1] within each Date. Handles inf/NaN internally.
When `winsorize=True`, clips to `[lower, upper]` quantiles per date before ranking.
This is the **standard final step** for almost all factors (except targets).

### `safe_divide(left, right)`

Element-wise division with safety: replaces zero denominators with NaN and drops inf results.

### `rolling_group_*` functions

Per-Code rolling statistics (internally uses `groupby(level="Code").transform`):

```python
rolling_group_mean(series, window, min_periods=None)   # default min_periods = window//2
rolling_group_std(series, window, min_periods=None)
rolling_group_sum(series, window, min_periods=None)
rolling_group_max(series, window)                       # min_periods=1
rolling_group_min(series, window)                       # min_periods=1
```

## FactorContext API

The `context` object passed to factor functions:

```python
class FactorContext:
    repo: DataRepository
    start_date: str | None    # YYYYMMDD
    end_date: str | None      # YYYYMMDD

    def load(self, relative_path: str) -> pd.DataFrame: ...
    def load_financial(self, relative_path: str, value_cols: list[str] | None = None,
                       date_col: str = "ann_date") -> pd.DataFrame: ...
```

### `context.load(path)`
Loads a daily-frequency parquet file. Returns `pd.DataFrame` with `(Date, Code)` MultiIndex.
The returned frame is already date-filtered (with the 252-day lookback buffer for rolling windows)
and stock-pool-filtered.

### `context.load_financial(path, value_cols, date_col)`
Loads a report-frequency parquet file and forward-fills to daily frequency using trading calendar.
Returns `pd.DataFrame` with `(Date, Code)` MultiIndex.

- `value_cols=None` → auto-detects all value columns (slower; prefer explicit)
- `date_col="ann_date"` → anchor date column. Use `"end_date"` for data like `pledge_stat.parquet`.

## DataRepository Additional Methods

```python
# Stock pool
context.repo.load_stock_pool()       # pd.DataFrame: Code, industry, area
context.repo.load_industry_map()     # pd.Series: Code → industry
context.repo.available_codes()       # pd.Index of all valid stock codes
```

## Industry Neutralization (`neutral.py`)

```python
from .neutral import _industry_neutral_rank

neutral_signal = _industry_neutral_rank(raw_signal, context)
```

Computes within-industry percentile rank. The raw_signal must be a `pd.Series` with `(Date, Code)` MultiIndex.
Adds `"stock_list.parquet"` to dependencies when using this.

## Build & Registration

```python
from ..registry import FactorContext, register_factor

@register_factor(
    name="factor_name",              # snake_case, unique
    description="中文描述。",          # One-liner
    category="category_name",        # Must match existing or be added to CATEGORY_ORDER
    thesis="投资逻辑简述。",           # Investment thesis (why this should work)
    dependencies=("source.parquet",), # Tuple of required upstream files
)
def factor_xxx(context: FactorContext):
    ...
    return cross_sectional_rank(signal)
```

## Storage Helpers (`storage.py`)

Rarely needed directly in factor code (builder calls these automatically):

```python
from .storage import load_factor      # Merge base + incr .fea files
from .storage import factor_base_path  # {name}.fea path
from .storage import factor_incr_path  # {name}_incr.fea path
```

## IC Analysis (`ic_analysis.py`)

```python
from .ic_analysis import compute_ic, ICResult

result = compute_ic("factor_name", "label_ret_5d", method="rank")
# result.ic_mean, result.ic_std, result.icir, result.ic_tstat, result.ic_positive_ratio
```

## Finance Factor Column Quick Reference

Common columns from `finance.parquet`:
```
pe, pe_ttm, pe_ttm_percentile, pb, ps, ps_ttm, dv_ratio, dv_ttm,
total_share, float_share, free_share, total_mv, circ_mv,
turnover_rate, turnover_rate_f, volume_ratio
```

Common columns from `financial_indicator.parquet`:
```
roe, roe_waa, roe_dt, roa, roic, npta,
gross_margin, netprofit_margin, grossprofit_margin,
eps, dt_eps, bps, ocfps, cfps, fcff, fcfe,
ebit, ebitda, ebit_ps, fcff_ps, fcfe_ps,
current_ratio, quick_ratio, cash_ratio,
debt_to_assets, assets_to_eqt, ebt_to_interest,
inv_turn, ar_turn, assets_turn,
or_yoy, op_yoy, tr_yoy, equity_yoy, netprofit_yoy,
salescash_to_or, ocf_to_profit, ocf_to_or, ocf_to_shortdebt, ocf_to_debt,
q_roe, q_eps, q_netprofit_margin, q_gsprofit_margin,
q_sales_yoy, q_profit_yoy, q_netprofit_yoy, q_ocf_to_sales,
q_gr_yoy, q_gr_qoq, q_sales_qoq, q_op_qoq, q_profit_qoq, q_netprofit_qoq,
```

Common columns from `balancesheet.parquet`:
```
total_assets, total_liab, total_hldr_eqy_exc_min_int, total_hldr_eqy_inc_min_int,
goodwill, inventories, accounts_receiv, fix_assets, intan_assets,
money_cap, contract_liab, st_borr, lt_borr,
total_cur_assets, total_nca, total_cur_liab, total_ncl,
right_of_use_assets, lease_liab
```

Common columns from `income.parquet`:
```
total_revenue, revenue, operate_profit, total_profit, n_income,
ebit, ebitda, fin_exp, sell_exp, admin_exp, rd_exp,
invest_income, non_oper_income, non_oper_exp,
income_tax, n_income_attr_p
```

Common columns from `cashflow.parquet`:
```
n_cashflow_act, n_cashflow_inv_act, n_cash_flows_fnc_act,
free_cashflow, c_inf_fr_operate_a,
depr_fa_coga_dpba, amort_intang_assets
```
