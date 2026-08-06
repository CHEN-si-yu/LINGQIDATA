"""
Direct field utilization factors — Class 1 panel factors.

These factors exploit raw data fields that are completely unused or severely
underused in the existing 560+ factor library:

  Completely UNUSED fields:
    - daily.parquet: change (raw price difference)
    - finance.parquet:   pe, pe_ttm (raw PE values)
    - stock_list.parquet: is_hs, act_ent_type, area, list_date

  Severely UNDERUSED fields:
    - finance.parquet:   turnover_rate_f (only 1 existing factor: turnover_f_20)

All factors are vectorized Class 1 (panel parquet) operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    stack_date_code,
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    safe_divide,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: pad stock code to 6-digit format
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_code(code: str) -> str:
    """Strip exchange suffix and zero-pad to 6 digits."""
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: load full stock_list with identity fields
# ═══════════════════════════════════════════════════════════════════════════════

def _load_stock_identity(context: FactorContext) -> pd.DataFrame:
    """Load stock_list.parquet with identity fields, cached at module level.

    Returns a DataFrame with columns: Code, industry, area, act_ent_type, is_hs, list_date.
    Only includes currently listed stocks (list_status == 'L') within the allowed pool.
    """
    cache = getattr(_load_stock_identity, "_cache", None)
    if cache is not None:
        return cache

    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "stock_list.parquet")
    pool = raw[raw["list_status"] == "L"].copy()
    pool["Code"] = pool["stock_code"].apply(_pad_code)

    # Keep identity columns
    cols = ["Code"]
    for c in ["industry", "area", "act_ent_type", "is_hs", "list_date"]:
        if c in pool.columns:
            cols.append(c)
    pool = pool[cols]

    # Filter to allowed stock pool
    allowed = context.repo.allowed_codes
    if allowed:
        pool = pool[pool["Code"].isin(allowed)]

    pool = pool.drop_duplicates(subset=["Code"]).set_index("Code").sort_index()
    _load_stock_identity._cache = pool
    return pool

def _broadcast_identity_to_daily(
    identity_series: pd.Series,
    context: FactorContext,
) -> pd.Series:
    """Broadcast a per-stock identity Series to the daily panel index.

    The identity values are static per stock and are replicated across all
    trading dates in the daily panel.
    """
    daily_adj = context.load("daily.parquet")
    all_dates = daily_adj.index.get_level_values("Date").unique()
    all_codes = daily_adj.index.get_level_values("Code").unique()

    # Align identity to the daily panel's stock universe
    aligned = identity_series.reindex(all_codes)

    # Broadcast to all dates
    full_idx = pd.MultiIndex.from_product(
        [all_dates, all_codes], names=["Date", "Code"]
    )
    result = pd.Series(index=full_idx, dtype=float)
    for date in all_dates:
        result.loc[date] = aligned.values
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Raw PE / PE_TTM Valuation Factors (finance.parquet: pe, pe_ttm)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="pe_ttm_absolute",
    description="PE_TTM原始值截面排名取负，低PE=价值信号（当期所有股票的截面比较）。",
    category="valuation",
    thesis=(
        "PE_TTM原始值截面排名取负。低PE是经典的价值信号——低PE股票在A股中长期有超额收益。"
        "本因子做当期截面比较(当期所有股票PE排序)。注:finance.parquet 的"
        "pe_ttm_percentile 字段(历史分位)属禁用数据源,不用于任何因子。"
    ),
    dependencies=("finance.parquet",),
)
def factor_pe_ttm_absolute(context: FactorContext):
    f = context.load("finance.parquet")
    pe = f["pe_ttm"].clip(0, 500)
    return cross_sectional_rank(-pe)


@register_factor(
    name="pe_ttm_change_20d",
    description="PE_TTM 20日变化率取反，估值收缩=价值改善，估值扩张=均值回归风险。",
    category="valuation",
    thesis=(
        "PE_TTM的20日变化率取反排名。PE下降(盈利增长或价格下跌使估值趋于合理)意味着估值回归，"
        "是价值改善的信号；PE上升(估值扩张)意味着股价上涨快于盈利增长，可能面临均值回归压力。"
    ),
    dependencies=("finance.parquet",),
)
def factor_pe_ttm_change_20d(context: FactorContext):
    f = context.load("finance.parquet")
    pe = f["pe_ttm"]
    chg = pe.groupby(level="Code").transform(lambda s: s.pct_change(20))
    chg = chg.clip(-0.5, 1.0)
    return cross_sectional_rank(-chg)


@register_factor(
    name="pe_pb_divergence",
    description="PE与PB百分位排名差。正偏离=轻资产高盈利(优质)，负偏离=周期/重资产。",
    category="valuation",
    thesis=(
        "PE与PB截面百分位排名的差值。PE排名高但PB排名低的股票(正偏离)往往是轻资产高盈利公司"
        "(如消费、医药)——市场对盈利给予溢价但资产价值未被重估，是优质信号。"
        "PE排名低但PB排名高的股票(负偏离)可能是周期股或重资产低盈利公司。"
    ),
    dependencies=("finance.parquet",),
)
def factor_pe_pb_divergence(context: FactorContext):
    f = context.load("finance.parquet")
    pe_rank = f["pe_ttm"].groupby(level="Date").rank(pct=True)
    pb_rank = f["pb"].groupby(level="Date").rank(pct=True)
    divergence = pe_rank - pb_rank
    return cross_sectional_rank(divergence)


@register_factor(
    name="float_share_ratio",
    description="自由流通股占比取反排名。流通盘小=筹码稀缺+弹性大，排名高。",
    category="valuation",
    thesis=(
        "流通股/总股本比率取反排名。流通股占比低意味着大量股份锁定期未满或大股东高比例控股，"
        "实际可交易盘小——筹码稀缺效应在上涨时放大弹性。该字段(float_share)在现有因子中零引用。"
    ),
    dependencies=("finance.parquet",),
)
def factor_float_share_ratio(context: FactorContext):
    f = context.load("finance.parquet")
    ratio = safe_divide(f["float_share"], f["total_share"])
    ratio = ratio.clip(0, 1)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ps_ttm_momentum_20d",
    description="PS_TTM 20日变化率取反，PS下降=变便宜，是价值改善信号。",
    category="valuation",
    thesis=(
        "PS_TTM的20日变化率取反排名。PS_TTM下降意味着股价下跌快于收入增长——公司变得更便宜，"
        "是价值回归的潜在信号。PS_TTM上升意味着估值扩张。PS不受利润波动影响，比PE更稳定。"
    ),
    dependencies=("finance.parquet",),
)
def factor_ps_ttm_momentum_20d(context: FactorContext):
    f = context.load("finance.parquet")
    ps = f["ps_ttm"]
    chg = ps.groupby(level="Code").transform(lambda s: s.pct_change(20))
    chg = chg.clip(-0.5, 1.0)
    return cross_sectional_rank(-chg)

# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Free-Float Turnover Factor (finance.parquet: turnover_rate_f)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="turnover_f_raw",
    description="自由流通换手率原始值因子（低换手排前）。",
    category="valuation",
    thesis=(
        "turnover_rate_f剔除大股东锁定股份，更精确反映真实可交易盘的换手活跃度。"
        "现有因子中仅turnover_f_20使用了20日均值，该因子使用每日原始值"
        "以更及时地捕捉自由流通盘的交易强度变化。"
        "低换手代表筹码稳定、投机度低，在A股截面中具有正向预测力。"
    ),
    dependencies=("finance.parquet",),
)
def factor_turnover_f_raw(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate_f"]
    turnover = turnover.clip(0, 50)
    return cross_sectional_rank(-turnover)

@register_factor(
    name="turnover_f_delta_5",
    description="自由流通换手率5日变化因子（换手率下降=浮筹减少排前）。",
    category="valuation",
    thesis=(
        "自由流通换手率的短期下降意味着浮动筹码被逐步吸收，"
        "筹码趋于集中。换手率持续下降往往伴随价格筑底，"
        "是筹码锁定度改善的领先信号。"
    ),
    dependencies=("finance.parquet",),
)
def factor_turnover_f_delta_5(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate_f"]
    delta = turnover.groupby(level="Code").transform(
        lambda s: s.pct_change(5)
    )
    delta = delta.clip(-1, 3)
    return cross_sectional_rank(-delta)

