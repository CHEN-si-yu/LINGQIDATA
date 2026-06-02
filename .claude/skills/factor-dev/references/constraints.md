# Hard Constraints

Violating any of these rules means the factor implementation is invalid.

---

## 1. Stock Pool

**Only these 2542 stocks.** Stock pool is defined in `/root/shared-nvme/lingqiData/Code_num.txt` (6-digit codes, one per line).

The `DataRepository` already filters to this pool automatically via `_filter_by_allowed()`. No extra work needed in factor code — but every factor's output must only contain these stocks.

## 2. Daily Frequency — Mandatory

**Every factor must be daily frequency.** Output index is `Date` (YYYYMMDD string), one row per trading day.

When using lower-frequency source data:
- **Quarterly/Annual financial reports**: Use `context.load_financial()` which forward-fills to daily via trading calendar. Forward-fill means: each day carries the **latest known** reported value (NOT future-peeking).
- **Monthly data** (e.g., index_weight, pledge_stat): Must also be forward-filled to daily. Use `context.load_financial()` with the appropriate `date_col`.
- **Weekly aggregation**: If computing a weekly signal, it must be forward-filled to daily — each day uses the latest completed week's value.

**Never** produce a factor that has gaps on non-report days. Every trading day must have a value for every stock (subject to data availability — stocks not yet listed will have NaN, which is acceptable).

## 3. No Forward-Peeking (Look-Ahead Bias)

This is the cardinal sin of quant factor development. Specific rules:

- **Financial data**: Use `load_financial()` which forward-fills from `ann_date`. The date on which a report is announced is the first day its data becomes visible. Before that day, use the previous report's data.
- **Target labels**: Already computed as `open[t+1+N]/open[t+1] - 1` in `target/ret.py`. New target labels must follow the same convention.
- **Rolling windows**: Always use `.shift(1)` before computing rolling statistics if the current value would not be known at the decision point.
- **Signal date = decision date**: The factor value on date T must be computable using only information available at or before T's market close (or T's open for overnight signals).

## 4. Build Phases

The build system has two conceptual phases:

### Phase 1 — Regular Factors (parallel)
All factors using daily-level and above data sources run in parallel via `ProcessPoolExecutor`.
Data sources in this phase: daily_adj, daily, finance, financial_indicator, balancesheet, income, cashflow, main_fund_flow, margin_detail, top_list, limit_up, limit_list, dragon_tiger, ths_daily, cyq_perf, holder_number, index_weight, pledge_stat, calendar, stock_list.

### Phase 2 — Heavy Data Factors (sequential/single)
Factors using **cyq_chips** or **history_1min** data must be processed separately. These data sources are too large (811MB for cyq_chips, 16GB for history_1min) for parallel processing without memory blowup. Phase 2 factors should be built sequentially with a single worker.

When implementing a Phase 2 factor:
- Mark it with `_PHASE = 2` or place in a separate build step
- These factors should NOT be mixed with Phase 1 parallel builds

## 5. Output Format Alignment

Output must match the existing convention exactly:

```
Format: Feather (.fea) file
Layout: Wide — Date index (rows) × Code columns
Date format: YYYYMMDD string
Code format: 6-digit zero-padded string (e.g., "000001")
Values: Cross-sectional percentile rank [0, 1] (except target labels)
Column name: {factor_name}
Index name: "Date"
Columns name: "Code"
```

## 6. NaN Rate Requirement

**For non-event-type factors**, the NaN rate from 2020-01-01 to the latest date must be measured and reported.

Event-type factors (dragon_tiger, limit_up, etc.) are exempt because they are sparse by nature — they only have values on event days.

NaN rate = (total cells where factor is NaN) / (total cells = trading_days × active_stocks) from 2020-01-01 onward.

- NaN rate < 5%: Excellent
- NaN rate 5-15%: Acceptable (e.g., newer stocks not yet listed)
- NaN rate > 15%: Needs investigation — the factor may have a computation bug or the data source has poor coverage

## 7. Algorithm Efficiency

Factor computation must be efficient at scale (2,542 stocks × ~1,800 trading days ≈ 4.6M cell dataset, larger for chip/min data):

- **Use vectorized pandas operations** — never iterate over stocks or dates in Python loops
- **Use `groupby(level="Code").transform()`** for per-stock rolling calculations (already the standard pattern)
- **Specify `value_cols` explicitly** in `load_financial()` — columnar parquet reads skip unnecessary columns
- **Leverage the LRU cache** — DataRepository caches 16 panels, so repeated loads of the same file are free
- **Avoid `pd.DataFrame.apply()`** with axis=1; prefer column-wise vectorized operations
- **For cyq_chips**: Pre-aggregate per stock per date before the factor function (consider pre-computing summary stats)
- **For history_1min**: Only load the date range needed; use the `daily_dump_1min/` format when possible (single file per day)

## 8. Factor Direction Convention

- **High rank = High expected return** (positive alpha)
- If the raw signal has the opposite direction (e.g., high leverage → low return), negate it before ranking: `cross_sectional_rank(-signal)`
- Target labels: high value = high forward return (no rank applied)

## 9. Dependencies Declaration

The `dependencies` tuple in `@register_factor` must list every upstream parquet file the factor reads. This is used by `decide_build_action()` to determine whether the factor is stale.

For `load_financial()`, always include `"calendar.parquet"` in addition to the data file.
For industry-neutral factors, always include `"stock_list.parquet"`.

## 10. Known Data Quality Issues

### q_* Fields (Single-Quarter Financial Indicators)

All `q_*` columns in `financial_indicator.parquet` (q_eps, q_roe, q_gsprofit_margin, q_netprofit_margin, q_sales_yoy, etc.) are **zero-filled for ~95% of records from 2019-2022**. Real values only appear from mid-2023.

**Consequence**: Factors using rolling statistics on q_* fields (e.g., CV/std of q_eps) produce NaN for virtually all stocks before 2023. The `earnings_yield_stability` factor was removed for this reason.

**Guideline**: Do NOT develop new factors that depend on q_* columns unless you can accept NaN for 2019-2023. Prefer the non-q equivalents (eps, roe, gross_margin, etc.) which have complete history.

### index_weight.parquet Historical Data

The `fetch_index_weight.py` script defaults `full_history=False` which caps downloads to 24 months. Always use `--full-history` for initial data loads. The restored data now spans 2019-2026 (88 months, 368K rows).

---

## 11. Factor Naming Convention

- snake_case, lowercase
- Use descriptive names: `momentum_20d`, NOT `m20`
- Suffix with window period: `mom_20`, `volatility_60`
- Neutralized versions: `{name}_neutral`
- Target labels: `label_ret_{horizon}d`
