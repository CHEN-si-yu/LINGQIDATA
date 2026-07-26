"""
Stock identity characteristic factors — Class 1 panel factors.

These factors leverage stock_list.parquet identity fields (area, act_ent_type, 
is_hs, list_date) combined with market data to create characteristic-based
cross-sectional signals.

Identity fields are static per stock; the factors broadcast them to daily
frequency and compute market-data interactions at each cross-section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    stack_date_code,
    cross_sectional_rank,
    rolling_group_mean,
    safe_divide,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_code(code: str) -> str:
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)

def _load_stock_identity(context: FactorContext) -> pd.DataFrame:
    """Load stock_list with identity fields, cached at module level."""
    cache = getattr(_load_stock_identity, "_cache", None)
    if cache is not None:
        return cache

    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "stock_list.parquet")
    pool = raw[raw["list_status"] == "L"].copy()
    pool["Code"] = pool["stock_code"].apply(_pad_code)

    cols = ["Code"]
    for c in ["industry", "area", "act_ent_type", "is_hs", "list_date"]:
        if c in pool.columns:
            cols.append(c)
    pool = pool[cols]

    allowed = context.repo.allowed_codes
    if allowed:
        pool = pool[pool["Code"].isin(allowed)]

    pool = pool.drop_duplicates(subset=["Code"]).set_index("Code").sort_index()
    _load_stock_identity._cache = pool
    return pool

# ═══════════════════════════════════════════════════════════════════════════════
# Factor 1: Ownership dispersion (BP/momentum dispersion within owner type)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Factor 2: HS Connect Flow Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="hs_connect_flow_sensitivity",
    description="港股通资金敏感度因子（HS标的且成交量异常的截面排名）。",
    category="sector",
    thesis=(
        "港股通标的对境外资金流动和全球风险偏好更为敏感。"
        "当HS标的出现异常成交量（高于自身20日均量2倍）时，"
        "往往反映了外资的大额进出。该因子结合HS资格和成交量异常，"
        "捕捉了'外资异动'信号。HS标的+高量比=外资可能正在交易。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_hs_connect_flow_sensitivity(context: FactorContext):
    identity = _load_stock_identity(context)
    daily_adj = context.load("daily_adj.parquet")

    # is_hs flag: S/H = HS connect eligible
    hs_flag = identity["is_hs"].map(
        lambda x: 1.0 if str(x).strip() in ("S", "H") else 0.0
    )

    # Volume ratio: current vol / 20-day average vol
    vol = daily_adj["vol"]
    vol_ma20 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    vol_ratio = safe_divide(vol, vol_ma20)
    vol_ratio = vol_ratio.clip(0, 10)

    # Get vol_ratio values for all codes
    vol_frame = vol_ratio.unstack("Code")
    all_codes = vol_frame.columns
    all_dates = vol_frame.index

    # Align HS flag to vol_frame
    hs_aligned = hs_flag.reindex(all_codes).fillna(0)

    # Score: HS eligible * volume ratio anomaly
    score = vol_frame.multiply(hs_aligned.values, axis=1)

    result = stack_date_code(score)
    result.name = "hs_connect_flow_sensitivity"
    return cross_sectional_rank(result)

# ═══════════════════════════════════════════════════════════════════════════════
# Factor 3: Area Dividend Preference
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="area_dividend_preference",
    description="区域股息偏好因子（个股dv_ratio偏离同区域均值的幅度截面排名）。",
    category="sector",
    thesis=(
        "不同地区的上市公司在分红文化上存在系统性差异："
        "沿海发达地区公司更倾向于高分红（成熟的投资者关系文化），"
        "内陆地区公司更倾向于保留利润再投资（成长导向）。"
        "在同区域内比较股息偏离可以识别公司治理的异常信号："
        "区域内股息显著偏高的公司可能面临成长机会不足，"
        "区域内股息显著偏低的公司可能有治理问题（不愿回报股东）。"
    ),
    dependencies=("finance.parquet",),
)
def factor_area_dividend_preference(context: FactorContext):
    identity = _load_stock_identity(context)
    finance = context.load("finance.parquet")

    dv = finance["dv_ratio"].clip(0, 20)

    # Build area → codes mapping
    area_map: dict[str, list[str]] = {}
    for code, row in identity.iterrows():
        area = str(row.get("area", "")).strip()
        if area and area != "nan":
            area_map.setdefault(area, []).append(code)

    dv_frame = dv.unstack("Code")

    result_parts = []
    for area, codes in area_map.items():
        available = [c for c in codes if c in dv_frame.columns]
        if len(available) < 3:
            continue
        area_dv = dv_frame[available]
        area_mean = area_dv.mean(axis=1)
        for c in available:
            dev = area_dv[c] - area_mean
            dev.name = c
            result_parts.append(dev.to_frame())

    if not result_parts:
        return cross_sectional_rank(dv)

    combined = pd.concat(result_parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    result = stack_date_code(combined)
    result.name = "area_dividend_preference"
    return cross_sectional_rank(result)  # higher dividend vs area peers = ranks high

# ═══════════════════════════════════════════════════════════════════════════════
# Factor 4: New Listing Momentum Effect
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="owner_type_turnover_div",
    description="实控人类型换手偏离因子（个股换手偏离同实控人类型均值截面排名）。",
    category="sector",
    thesis=(
        "不同实控人类型的股票具有不同的典型换手率水平。"
        "民企由于流通盘小、机构持仓少，换手率通常高于国企。"
        "在同类型内比较换手偏离可以识别异常交易活动："
        "国企异常高换手=可能有重大事项或资金异动；"
        "民企异常低换手=可能被边缘化或筹码高度锁定。"
    ),
    dependencies=("finance.parquet",),
)
def factor_owner_type_turnover_div(context: FactorContext):
    identity = _load_stock_identity(context)
    finance = context.load("finance.parquet")

    turnover = finance["turnover_rate"].clip(0, 50)

    owner_map: dict[str, list[str]] = {}
    for code, row in identity.iterrows():
        owner = str(row.get("act_ent_type", "")).strip()
        if owner and owner != "nan":
            owner_map.setdefault(owner, []).append(code)

    to_frame = turnover.unstack("Code")

    result_parts = []
    for owner, codes in owner_map.items():
        available = [c for c in codes if c in to_frame.columns]
        if len(available) < 3:
            continue
        owner_to = to_frame[available]
        owner_mean = owner_to.mean(axis=1)
        for c in available:
            dev = owner_to[c] - owner_mean
            dev.name = c
            result_parts.append(dev.to_frame())

    if not result_parts:
        return cross_sectional_rank(turnover)

    combined = pd.concat(result_parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    result = stack_date_code(combined)
    result.name = "owner_type_turnover_div"
    return cross_sectional_rank(result)
