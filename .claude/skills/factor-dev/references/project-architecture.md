# Project Architecture

## Directory Layout

```
featureengineering/
├── build_factors.py                     # Main entry point (direct run or CLI)
├── src/featureengineering/              # Core package
│   ├── builder.py                       # Factor build engine (sequential/parallel/incremental)
│   ├── registry.py                      # Factor registry (@register_factor decorator)
│   ├── dataset.py                       # DataRepository — LRU cache, daily panel, financial ffill
│   ├── storage.py                       # Output: write .fea + JSON manifest
│   ├── settings.py                      # Path config (env vars / defaults)
│   ├── state.py                         # BuildState — resume-after-interrupt tracking
│   ├── cli.py                           # CLI argument parser
│   ├── report.py                        # Terminal color-coded status reports
│   ├── ic_analysis.py                   # IC / ICIR evaluation (rank & pearson)
│   ├── factor_loader.py                 # Auto-import all factor modules via pkgutil
│   ├── utils.py                         # cross_sectional_rank, rolling_group_*, safe_divide
│   ├── progress.py                      # Stage constants for progress callbacks
│   └── factors/                         # Factor implementations by category
│       ├── __init__.py                  # (empty)
│       ├── price.py                     # 31 factors — momentum, reversal, volatility, technical
│       ├── quality.py                   # 30 factors — ROE/ROA, margins, profitability quality
│       ├── deep_finance.py              # 23 factors — balance sheet, income, cashflow deep dive
│       ├── valuation.py                 # 13 factors — BP, EP, SP, DP, size, FCF yield
│       ├── event.py                     # 12 factors — dragon tiger, limit up/down, block trades
│       ├── neutral.py                   # 11 factors — industry-neutralized versions
│       ├── fund_flow.py                 # 10 factors — main fund net inflow, order-size analysis
│       ├── enhanced.py                  #  5 factors — volatility-adjusted, stability
│       ├── sector.py                    #  3 factors — sector diversification, within-sector rank
│       ├── margin.py                    #  2 factors — margin leverage, margin buy intensity
│       └── target/
│           ├── __init__.py
│           └── ret.py                   #  4 factors — label_ret_{1d,5d,10d,20d}
└── data/
    ├── factors/                         # 140 output .fea files (Date x Code wide format)
    ├── targets/                         # 4 target label .fea files
    ├── manifests/                       # JSON metadata per factor
    ├── build_state.json                 # Build progress state for resume
    └── logs/
```

## Key Data Flow

```
Upstream parquet data  →  DataRepository.load / load_financial
                                  │
                    ┌─────────────┴─────────────┐
                    │   FactorContext.load()     │
                    │   FactorContext.load_financial() │
                    └─────────────┬─────────────┘
                                  │
                    Factor compute function
                    (pd.Series with Date/Code MultiIndex)
                                  │
                    ensure_single_factor_frame()
                    (→ wide format: Date rows × Code cols)
                                  │
                    write_factor() / write_factor_incremental()
                    (→ .fea feather file + JSON manifest)
```

## Three Build Actions

`decide_build_action()` in builder.py compares factor last date vs source data last date:

| Action | Trigger | Behavior |
|--------|---------|----------|
| **skip** | factor date >= source date | Nothing to do |
| **incremental** | factor date < source date | Compute only new dates, merge to `_incr.fea` |
| **rebuild** | no factor file / force=True / date anomaly | Full historical rebuild to `{name}.fea` |

## EOD Cutoff Rule

`_EOD_CUTOFF_HOUR = 18` — before 18:00, today's trading data is considered unavailable.
The effective end date is capped at `daily_adj.parquet` max date, with the 18:00 rule as a belt-and-suspenders safeguard.

## Factor Registry

- `FACTOR_REGISTRY` is a global `dict[str, FactorSpec]`
- Factors are auto-discovered by `ensure_builtin_factors_loaded()` which walks `factors/` package
- Each `FactorSpec` contains: name, description, category, thesis, dependencies, compute function

## Output Format

- **Format**: Feather (`.fea`) files
- **Layout**: Wide — `Date` index (rows) × `Code` columns
- **Values**: Cross-sectional percentile ranks [0, 1]
- **Storage**: Base file (`{name}.fea`) + incremental file (`{name}_incr.fea`), merged on read

## Key Configuration

Set via environment variables or `configure_paths()`:
- `FEATURE_ENGINEERING_PROJECT_ROOT` — project root (default: package parent)
- `FEATURE_ENGINEERING_SOURCE_ROOT` — upstream data root (default: `../data`)
- `FEATURE_ENGINEERING_FACTOR_OUTPUT_DIR` — factor output dir
- `FEATURE_ENGINEERING_MANIFEST_OUTPUT_DIR` — manifest output dir
