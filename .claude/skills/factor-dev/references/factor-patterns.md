# Factor Development Patterns

## Factor Registration

Every factor is decorated with `@register_factor`. The decorator takes:

```python
from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank

@register_factor(
    name="factor_name",              # snake_case, unique across all factors
    description="中文描述。",          # One-line description
    category="category_name",        # Must match one of CATEGORY_ORDER or be new
    thesis="投资逻辑简述。",           # Why this factor should work (investment thesis)
    dependencies=("source.parquet",), # Tuple of required parquet files
)
def factor_xxx(context: FactorContext):
    ...
```

### Existing Categories (from CATEGORY_ORDER in builder.py)

```
price, timeseries, valuation, quality, event, fund_flow,
financial, sector, neutral, index, target, other
```

Currently used: `price`, `valuation`, `quality`, `event`, `fund_flow`, `financial`, `sector`, `neutral`, `target`, `enhanced`

## Factor Compute Patterns

### Pattern 1: Simple daily data → cross-sectional rank

Most common pattern. Load daily panel, compute signal, rank within date.

```python
@register_factor(
    name="mom_20",
    category="price",
    dependencies=("daily_adj.parquet",),
    ...
)
def factor_mom_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    mom = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(20)
    )
    return cross_sectional_rank(mom)
```

### Pattern 2: Financial statement data → daily ffill

Use `context.load_financial()` for report-frequency data. Only specify needed columns for efficiency.

```python
@register_factor(
    name="roe",
    category="quality",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
    ...
)
def factor_roe(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe"]
    )
    return cross_sectional_rank(fin["roe"])
```

Note: `calendar.parquet` must be in dependencies when using `load_financial()`.

### Pattern 3: Multiple data sources joined

```python
@register_factor(
    name="margin_leverage_ratio",
    category="margin",
    dependencies=("margin_detail.parquet", "finance.parquet"),
    ...
)
def factor_margin_leverage_ratio(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    finance = context.load("finance.parquet")

    rzye = margin["rzye"]
    circ_mv = finance["circ_mv"]

    common = rzye.index.intersection(circ_mv.index)
    ratio = rzye.loc[common] / circ_mv.loc[common].replace(0, np.nan)
    return cross_sectional_rank(-ratio)  # negative: high leverage = bad
```

### Pattern 4: Industry-neutral version

```python
from .neutral import _industry_neutral_rank

@register_factor(
    name="mom_20_neutral",
    category="neutral",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
    ...
)
def factor_mom_20_neutral(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    mom = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(20)
    )
    neutral = _industry_neutral_rank(mom, context)  # within-industry percentile
    return cross_sectional_rank(neutral)
```

### Pattern 5: Target label

Target labels live in `factors/target/`. They do NOT return cross-sectional ranks — they return raw forward returns.

```python
@register_factor(
    name="label_ret_5d",
    category="target",
    ...
)
def factor_label_ret_5d(context: FactorContext):
    ...
    return raw_forward_returns  # NOT cross_sectional_rank
```

### Pattern 6: Rolling window per-stock (using groupby+transform)

```python
# Via utils
from ..utils import rolling_group_mean, rolling_group_std, rolling_group_sum, rolling_group_max, rolling_group_min

# Or directly
series.groupby(level="Code").transform(
    lambda s: s.rolling(window, min_periods=n).mean()
)
```

## Important Conventions

1. **Direction**: High rank = good (positive alpha expected). Use `-signal` when the raw signal direction is reversed (e.g., high leverage → negative signal).

2. **Dependencies list**: Include ALL parquet files that the factor reads. This is how the build system decides whether to rebuild.

3. **Return type**: The function must return `pd.Series` or `pd.DataFrame` with `(Date, Code)` MultiIndex. The storage layer converts it to wide format.

4. **Single column**: Output must have exactly one column (the factor name). For targets, the column name should match the factor name.

5. **NaN handling**: `cross_sectional_rank()` handles inf/nan internally. For custom calculations, use `.replace(0, np.nan)` before division and `.replace([np.inf, -np.inf], np.nan)` after.

6. **New files**: If creating a new category file under `factors/`, just put `.py` files there — `factor_loader.py` auto-discovers them via `pkgutil.walk_packages`.

## Build Phase Classification

### Phase 1 — Regular Factors (parallel)
All factors using daily-level data sources. These run in parallel via `ProcessPoolExecutor`.
- Sources: daily_adj, daily, finance, financial_indicator, balancesheet, income, cashflow, main_fund_flow, margin_detail, top_list, limit_up, limit_list, dragon_tiger, ths_daily, cyq_perf, holder_number, index_weight, pledge_stat

### Phase 2 — Heavy Data Factors (sequential)
Factors using **cyq_chips** or **history_1min**. These data sources are too large for parallel builds.
Process them sequentially with a single worker, separately from Phase 1.

## Checklist: Creating a New Factor

1. [ ] **Stock pool**: Only the 2542 stocks in `Code_num.txt` (DataRepository handles this automatically)
2. [ ] **Frequency**: Output must be daily. If using lower-freq data, forward-fill via `load_financial()` or manual ffill per stock
3. [ ] **No forward-peeking**: All information on date T must be known at T's close
4. [ ] **Build phase**: Determine Phase 1 (parallel) or Phase 2 (heavy: cyq_chips / history_1min)
5. [ ] Choose the right category file (or create a new one if needed)
6. [ ] Verify the data source exists and has the needed columns (consult `data-catalog.md`)
7. [ ] Decide: `context.load()` for daily data, `context.load_financial()` for report-frequency data
8. [ ] Write the `@register_factor` decorator with complete metadata
9. [ ] Write the compute function — **vectorized operations only**, no Python loops over stocks/dates
10. [ ] **Always specify `value_cols`** in `load_financial()` for columnar read efficiency
11. [ ] Ensure the function returns a `(Date, Code)` MultiIndex Series
12. [ ] Apply `cross_sectional_rank()` (unless it's a target label)
13. [ ] **Direction**: High rank = positive alpha. Negate signal if needed before ranking
14. [ ] **Output alignment**: Wide format, Date rows × Code columns, YYYYMMDD strings, 6-digit codes
15. [ ] After build: run `python scripts/validate_factors.py` to check last-date and NaN rate
16. [ ] **NaN rate** (non-event factors): Must be < 15% from 2020-01-01. Investigate if higher.
17. [ ] New file is auto-discovered by `factor_loader.py` — no import changes needed
