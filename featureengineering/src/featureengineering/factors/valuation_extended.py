"""
Valuation extended factors -- Class 1 panel factors.

These use untapped columns from finance.parquet and financial_indicator.parquet:
  - dv_ratio / dv_ttm: dividend yield (completely new signal)
  - ps_ttm: price-to-sales TTM (more responsive than annual ps)
  - q_eps: quarterly EPS yield (faster than annual eps)
  - turnover_rate: raw daily turnover rate from finance.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Dividend Yield Factors (from finance.parquet -- completely untapped)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="dv_yield_rank",
    description="股息率因子，dv_ratio截面排名（高股息排前）。",
    category="valuation",
    thesis=(
        "Dividend yield is a classic value and quality factor. "
        "High-dividend stocks in A-shares have become increasingly important "
        "under regulatory guidance encouraging dividend payouts. "
        "Unlike BP (book-to-price), dividend yield provides a direct cash "
        "return to shareholders and is harder to manipulate than earnings. "
        "The dv_ratio field in finance.parquet represents trailing 12-month "
        "dividend yield and has 0% NaN coverage."
    ),
    dependencies=("finance.parquet",),
)
def factor_dv_yield_rank(context: FactorContext):
    fin = context.load("finance.parquet")
    dv = fin["dv_ratio"].clip(0, 20)  # winsorize extreme yields
    return cross_sectional_rank(dv)


@register_factor(
    name="dv_ttm_rank",
    description="股息率TTM因子，dv_ttm截面排名（高TTM股息排前）。",
    category="valuation",
    thesis=(
        "TTM dividend yield is slightly more current than dv_ratio. "
        "Both measures are highly correlated but dv_ttm updates faster "
        "when companies announce new dividend policies. "
        "Using both provides a more robust dividend signal."
    ),
    dependencies=("finance.parquet",),
)
def factor_dv_ttm_rank(context: FactorContext):
    fin = context.load("finance.parquet")
    dv = fin["dv_ttm"].clip(0, 20)
    return cross_sectional_rank(dv)


@register_factor(
    name="dv_stability_4q",
    description="股息稳定性因子，过去4季dv_ratio变异系数截面排名（股息稳定排前）。",
    category="valuation",
    thesis=(
        "Dividend stability is as important as dividend level. "
        "Companies that maintain stable dividends signal confidence "
        "in future cash flows. Erratic dividends signal uncertainty. "
        "CV of dividend yield over 4 quarters captures consistency."
    ),
    dependencies=("finance.parquet",),
)
def factor_dv_stability_4q(context: FactorContext):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Price-to-Sales Factors (from finance.parquet -- untapped)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ps_ttm_rank",
    description="市销率TTM因子，ps_ttm截面排名（低市销率排前）。",
    category="valuation",
    thesis=(
        "Price-to-Sales TTM is a fundamental valuation metric that works "
        "for companies with negative earnings (where PE fails). "
        "Low P/S stocks tend to be overlooked by earnings-focused investors "
        "and can offer significant upside when profitability improves. "
        "TTM variant is more responsive than annual P/S. "
        "0% NaN -- excellent coverage."
    ),
    dependencies=("finance.parquet",),
)
def factor_ps_ttm_rank(context: FactorContext):
    fin = context.load("finance.parquet")
    ps = fin["ps_ttm"].clip(0, 500)
    return cross_sectional_rank(-ps)  # low P/S ranks high


@register_factor(
    name="ps_ttm_sector_neutral",
    description="行业中性市销率因子，(ps_ttm排名-行业均值排名)截面排名。",
    category="valuation",
    thesis=(
        "P/S ratios vary dramatically by industry (tech vs utilities). "
        "Industry-neutral P/S captures within-industry relative cheapness, "
        "which is more predictive than absolute P/S level. "
        "Uses THS sector classification."
    ),
    dependencies=("finance.parquet", "ths_constituent_stocks.parquet"),
)
def factor_ps_ttm_sector_neutral(context: FactorContext):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Quarterly Earnings Factors (from financial_indicator.parquet)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="q_eps_yield_rank",
    description="季度EPS收益率因子，(q_eps*4/close)截面排名（高盈利收益率排前）。",
    category="valuation",
    thesis=(
        "Quarterly EPS annualized and divided by price gives a fast-updating "
        "earnings yield. Unlike annual/ttm EPS which lags by quarters, "
        "quarterly EPS yield captures the most recent quarter earnings power. "
        "Early detection of earnings acceleration before annual figures update."
    ),
    dependencies=("financial_indicator.parquet", "calendar.parquet", "daily_adj.parquet"),
)
def factor_q_eps_yield_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_eps"],
    )
    daily = context.load("daily_adj.parquet")
    close = daily["close"]

    # Annualize quarterly EPS
    eps_annual = fin["q_eps"] * 4

    # Align to daily dates
    common = eps_annual.index.intersection(close.index)
    eps_a = eps_annual.loc[common]
    price = close.loc[common]
    eps_yield = safe_divide(eps_a, price + 1e-10)
    eps_yield = eps_yield.clip(-1, 1)
    return cross_sectional_rank(eps_yield)


@register_factor(
    name="q_gr_qoq_rank",
    description="季度营收环比增长因子，q_gr_qoq截面排名（高环比增速排前）。",
    category="financial",
    thesis=(
        "Quarter-over-quarter revenue growth captures sequential business "
        "momentum missed by YoY comparisons. A company can have weak YoY "
        "growth (tough comparison) but strong QoQ (improving trajectory). "
        "QoQ growth is a leading indicator for future YoY growth acceleration."
    ),
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_gr_qoq_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_gr_qoq"],
    )
    growth = fin["q_gr_qoq"].clip(-200, 500)
    return cross_sectional_rank(growth)


@register_factor(
    name="q_gsprofit_margin_rank",
    description="季度毛利率因子，q_gsprofit_margin截面排名（高毛利率排前）。",
    category="financial",
    thesis=(
        "Quarterly gross margin is the fastest-updating profitability metric. "
        "Changes in gross margin precede changes in operating/net margins "
        "because COGS is the first line item affected by input costs and "
        "pricing power. Quarterly margin compression is an early warning."
    ),
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_gsprofit_margin_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_gsprofit_margin"],
    )
    margin = fin["q_gsprofit_margin"].clip(-100, 100)
    return cross_sectional_rank(margin)


@register_factor(
    name="q_netprofit_yoy_rank",
    description="季度利润同比增长因子，q_netprofit_yoy截面排名（高利润增速排前）。",
    category="financial",
    thesis=(
        "Quarterly net profit YoY growth is the most direct measure of "
        "earnings momentum. Unlike annual growth which averages 4 quarters, "
        "quarterly growth captures the exact inflection point. "
        "Faster signal for earnings surprises than annual figures."
    ),
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_netprofit_yoy_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_netprofit_yoy"],
    )
    growth = fin["q_netprofit_yoy"].clip(-500, 500)
    return cross_sectional_rank(growth)


