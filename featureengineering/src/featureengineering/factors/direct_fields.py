"""
Direct field utilization factors — Class 1 panel factors.

These factors exploit raw data fields that are completely unused or severely
underused in the existing 560+ factor library:

  Completely UNUSED fields:
    - daily_adj.parquet: change (raw price difference)
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
    daily_adj = context.load("daily_adj.parquet")
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
# Section A: Raw Price Change Factor (daily_adj.parquet: change field)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="change_raw",
    description="价格变动绝对值因子，change截面排名（大变动=高波动弹性排前）。",
    category="price",
    thesis=(
        "change字段是原始价格变动（close - pre_close的绝对值方向），"
        "不同于pct_chg的百分比度量。change捕捉的是绝对资本变动幅度，"
        "高价股有更大的change区间。change的截面差异同时反映了价格水平和波动性，"
        "高change股票往往伴随更强的日内资金博弈。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_change_raw(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    change = daily_adj["change"].replace([np.inf, -np.inf], np.nan)
    return cross_sectional_rank(change)

@register_factor(
    name="change_amplitude_ratio",
    description="价格变动振幅比因子，|change|/(high-low)截面排名（高占比=单边行情排前）。",
    category="price",
    thesis=(
        "change绝对值占日内振幅(high-low)的比例反映了价格运动的单边程度。"
        "高占比意味着日内价格以趋势性运动为主、往返较少，"
        "是动量持续性的微观结构信号。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_change_amplitude_ratio(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    amplitude = daily_adj["high"] - daily_adj["low"]
    change_abs = daily_adj["change"].abs()
    ratio = safe_divide(change_abs, amplitude)
    ratio = ratio.clip(0, 1.5)
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Raw PE / PE_TTM Valuation Factors (finance.parquet: pe, pe_ttm)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="pe_ttm_absolute",
    description="PE_TTM原始值截面排名取负，低PE=价值信号。与pe_ttm_percentile互补——该因子做截面比较而非历史比较。",
    category="valuation",
    thesis=(
        "PE_TTM原始值截面排名取负。低PE是经典的价值信号——低PE股票在A股中长期有超额收益。"
        "现有因子pe_ttm_percentile做的是历史分位比较(当前PE在过去5年的位置)，"
        "该因子做的是截面比较(当期所有股票PE排序)，提供独立于历史分位的估值维度。"
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

# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Stock Identity Factors (stock_list.parquet)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="area_return_effect",
    description="地区动量效应因子（同area股票20日动量均值+个股偏离截面排名）。",
    category="sector",
    thesis=(
        "同一地区的上市公司共享本地经济环境、政策影响和投资者关注度。"
        "地区内股票存在协同效应：同地区股票平均动量代表了该地区的"
        "系统性信息冲击，个股动量偏离地区均值反映了公司特异性信息。"
        "两者结合捕捉了区域层面的alpha分层。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_area_return_effect(context: FactorContext):
    identity = _load_stock_identity(context)
    daily_adj = context.load("daily_adj.parquet")

    # Calculate 20-day momentum per stock
    close = daily_adj["close"]
    mom_20 = close.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    mom_20 = mom_20.clip(-0.5, 1.0)

    # Build area→codes mapping
    area_map: dict[str, list[str]] = {}
    for code, row in identity.iterrows():
        area = str(row.get("area", "")).strip()
        if area and area != "nan":
            area_map.setdefault(area, []).append(code)

    # Unstack for cross-sectional operations
    mom_frame = mom_20.unstack("Code")  # Date × Code

    # For each area, compute area-mean momentum and individual deviation
    result_parts = []
    for area, codes in area_map.items():
        available = [c for c in codes if c in mom_frame.columns]
        if len(available) < 2:
            continue
        area_mom = mom_frame[available]
        area_mean = area_mom.mean(axis=1)
        # Individual deviation from area mean
        for c in available:
            dev = area_mom[c] - area_mean
            dev.name = c
            result_parts.append(dev.to_frame())

    if not result_parts:
        return cross_sectional_rank(mom_20)

    combined = pd.concat(result_parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T  # average across areas if multi-mapped
    result = stack_date_code(combined)
    result.name = "area_return_effect"
    return cross_sectional_rank(result)

@register_factor(
    name="owner_type_momentum_div",
    description="实控人类型动量偏离因子（个股动量偏离同实控人类型均值排前=加速）。",
    category="sector",
    thesis=(
        "不同实控人类型（国企/民企/外资等）具有不同的典型动量水平。"
        "国企由于市值大、机构持仓多，动量通常低于民企。"
        "在同类型内比较动量偏离可以更公平地识别加速/减速股票，"
        "避免将民企系统性高动量误判为alpha信号。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_owner_type_momentum_div(context: FactorContext):
    identity = _load_stock_identity(context)
    daily_adj = context.load("daily_adj.parquet")

    # 20-day momentum
    close = daily_adj["close"]
    mom_20 = close.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    mom_20 = mom_20.clip(-0.5, 1.0)

    # Build owner type → codes mapping
    owner_map: dict[str, list[str]] = {}
    for code, row in identity.iterrows():
        owner = str(row.get("act_ent_type", "")).strip()
        if owner and owner != "nan":
            owner_map.setdefault(owner, []).append(code)

    mom_frame = mom_20.unstack("Code")

    result_parts = []
    for owner, codes in owner_map.items():
        available = [c for c in codes if c in mom_frame.columns]
        if len(available) < 3:
            continue
        owner_mom = mom_frame[available]
        owner_mean = owner_mom.mean(axis=1)
        for c in available:
            dev = owner_mom[c] - owner_mean
            dev.name = c
            result_parts.append(dev.to_frame())

    if not result_parts:
        return cross_sectional_rank(mom_20)

    combined = pd.concat(result_parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    result = stack_date_code(combined)
    result.name = "owner_type_momentum_div"
    return cross_sectional_rank(result)

