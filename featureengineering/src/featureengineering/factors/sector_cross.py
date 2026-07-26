"""
Cross-sector interactive factors (跨板块交互因子) — Class 1.

These factors compute sector-level aggregates from constituent stocks using
THS industry sector membership, then derive sector-relative and cross-sector
signals for individual stocks.

⚠️  We do NOT use ``ths_daily.parquet`` (data starts 2023-01-03, violating the
2020-01-01 constraint).  Instead, sector aggregates are built from individual
stock data (daily_adj, finance, etc.) grouped by THS sector membership.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    safe_divide,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_code(code: str) -> str:
    """Strip exchange suffix and zero-pad to 6 digits."""
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)

def _load_stock_sector_map(context: FactorContext) -> dict[str, list[str]]:
    """Build stock_code (6-digit) → list of THS industry sector codes.

    Only sectors of type 'I' (industry classification).  Cached at module level.
    .. warning::
       **数据泄露风险**: ``ths_constituent_stocks.parquet`` 不含日期字段，
       为静态快照。板块/概念成分股会随时间变化（新增、剔除），
       但此映射将所有历史日期统一应用当前快照，可能引入前瞻偏差：
       - 对于 type='I'（行业板块），成分股变化较慢，影响有限；
       - 对于 type='N'（概念板块），主题板块更动态，泄漏风险更高。
       若数据源提供历史成分股快照，应改为按日期动态加载。
       当前实现假设成分股变动对因子影响在可接受范围内。
    """
    cache = getattr(_load_stock_sector_map, "_cache", None)
    if cache is not None:
        return cache

    src = context.repo.paths.source_root
    cs = context.repo._read_parquet(src / "ths_constituent_stocks.parquet")
    sc = context.repo._read_parquet(src / "ths_sector_categories.parquet")

    import logging
    _logger = logging.getLogger(__name__)

    industry_codes = set(sc[sc["type"] == "I"]["index_code"])
    cs_industry = cs[cs["index_code"].isin(industry_codes)]
    if cs_industry.empty:
        _logger.debug(
            "sector_cross: type=I industry sectors not found in constituent data, "
            "trying stock_list.parquet industry field (申万行业) as fallback."
        )
        # ── Fallback 1: use stock_list.parquet industry field ──
        try:
            stock_list = context.repo._read_parquet(src / "stock_list.parquet")
            if "industry" in stock_list.columns:
                stock_map_fb: dict[str, list[str]] = {}
                for _, row in stock_list.iterrows():
                    code = _pad_code(row["stock_code"])
                    ind = (row.get("industry") or "").strip()
                    if ind:
                        stock_map_fb.setdefault(code, []).append(ind)
                if stock_map_fb:
                    n_sec = len({i for v in stock_map_fb.values() for i in v})
                    _logger.debug(
                        "sector_cross: using stock_list industry (%d sectors, %d stocks).",
                        n_sec, len(stock_map_fb),
                    )
                    _load_stock_sector_map._cache = stock_map_fb
                    return stock_map_fb
        except Exception:
            _logger.warning(
                "sector_cross: stock_list fallback failed, trying type=BB/N."
            )

        # ── Fallback 2: BB/N market-relative ──
        _logger.warning(
            "sector_cross: falling back to type=BB/N — factors become market-relative."
        )
        industry_codes = set(sc[sc["type"].isin(["BB", "N"])]["index_code"])
        cs_industry = cs[cs["index_code"].isin(industry_codes)]

    stock_map: dict[str, list[str]] = {}
    for _, row in cs_industry.iterrows():
        code = _pad_code(row["stock_code"])
        ths = row["index_code"]
        stock_map.setdefault(code, []).append(ths)

    _load_stock_sector_map._cache = stock_map
    return stock_map

def _build_primary_sector_map(context: FactorContext) -> pd.Series:
    """Return a Series mapping stock_code → primary THS industry sector code.

    The primary sector is the first listed industry sector for each stock.
    """
    stock_map = _load_stock_sector_map(context)
    mapping = {code: sectors[0] if sectors else None for code, sectors in stock_map.items()}
    return pd.Series(mapping, name="sector")

def _build_sector_return_panel(context: FactorContext) -> pd.DataFrame:
    """Build a (Date × sector_code) DataFrame of equal-weighted sector daily returns.

    Returns are computed from daily_adj.close, then equal-weighted within each sector.
    """
    cache = getattr(_build_sector_return_panel, "_cache", None)
    if cache is not None:
        return cache

    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    primary = _build_primary_sector_map(context)
    codes = ret.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"ret": ret.values, "sector": sectors.values}, index=ret.index)
    df = df.dropna(subset=["sector"])

    # Equal-weighted sector return
    sector_ret = df.groupby(["Date", "sector"])["ret"].mean()
    panel = sector_ret.unstack("sector")  # Date × sector
    panel = panel.sort_index()

    _build_sector_return_panel._cache = panel
    return panel

def _build_sector_mv_panel(context: FactorContext) -> pd.DataFrame:
    """Build a (Date × sector_code) DataFrame of total sector market cap."""
    cache = getattr(_build_sector_mv_panel, "_cache", None)
    if cache is not None:
        return cache

    finance = context.load("finance.parquet")
    mv = finance["total_mv"].where(finance["total_mv"] > 0, np.nan)

    primary = _build_primary_sector_map(context)
    codes = mv.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"mv": mv.values, "sector": sectors.values}, index=mv.index)
    df = df.dropna(subset=["sector"])

    sector_mv = df.groupby(["Date", "sector"])["mv"].sum()
    panel = sector_mv.unstack("sector")
    panel = panel.sort_index()

    _build_sector_mv_panel._cache = panel
    return panel

def _build_sector_turnover_panel(context: FactorContext) -> pd.DataFrame:
    """Build a (Date × sector_code) DataFrame of sector average turnover rate."""
    cache = getattr(_build_sector_turnover_panel, "_cache", None)
    if cache is not None:
        return cache

    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]

    primary = _build_primary_sector_map(context)
    codes = turnover.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"to": turnover.values, "sector": sectors.values}, index=turnover.index)
    df = df.dropna(subset=["sector"])

    sector_to = df.groupby(["Date", "sector"])["to"].mean()
    panel = sector_to.unstack("sector")
    panel = panel.sort_index()

    _build_sector_turnover_panel._cache = panel
    return panel

def _build_all_a_return(context: FactorContext) -> pd.Series:
    """Build the equal-weighted all-A-share daily return series (Date-indexed)."""
    cache = getattr(_build_all_a_return, "_cache", None)
    if cache is not None:
        return cache

    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    all_a = ret.groupby(level="Date").mean()

    _build_all_a_return._cache = all_a
    return all_a

def _sector_panel_to_stock(panel: pd.DataFrame, context: FactorContext,
                           date_index: pd.Index | None = None) -> pd.Series:
    """Map a (Date × sector) panel back to (Date, Code) Series for individual stocks.

    Uses the primary sector of each stock to look up sector-level values.
    """
    primary = _build_primary_sector_map(context)
    if date_index is None:
        date_index = panel.index

    # For each date, get the sector value for each stock's primary sector
    result_parts = []
    for date in date_index:
        if date not in panel.index:
            continue
        row = panel.loc[date]
        stock_vals = primary.map(row)
        stock_vals = stock_vals.dropna()
        idx = pd.MultiIndex.from_arrays(
            [[date] * len(stock_vals), stock_vals.index],
            names=["Date", "Code"],
        )
        result_parts.append(pd.Series(stock_vals.values, index=idx))

    if not result_parts:
        idx = pd.MultiIndex.from_tuples([], names=['Date', 'Code'])
        return pd.Series(dtype=float, index=idx, name='value')
    result = pd.concat(result_parts)
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# A — Sector-Market Relationship Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="stock_sector_corr_60",
    description="个股-板块相关性因子，个股60日收益与所属板块等权收益的相关系数截面排名（高相关=跟随板块排前）。",
    category="sector",
    thesis="个股与其板块的相关性反映该股票是'板块代表股'还是'独立走势股'——高相关意味着板块涨它就涨，是投资该板块的纯beta工具。低相关则可能有独立的alpha驱动。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_stock_sector_corr_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    sector_ret_panel = _build_sector_return_panel(context)
    primary = _build_primary_sector_map(context)

    # For each stock, get its sector's return series and compute rolling correlation
    ret_wide = ret.unstack("Code")  # Date × Code
    common_dates = ret_wide.index.intersection(sector_ret_panel.index)

    corr_series = pd.Series(np.nan, index=ret.index)

    for code in ret_wide.columns:
        if code not in primary.index:
            continue
        sec = primary.loc[code]
        if pd.isna(sec) or sec not in sector_ret_panel.columns:
            continue

        stock_r = ret_wide.loc[common_dates, code]
        sector_r = sector_ret_panel.loc[common_dates, sec]
        corr = stock_r.rolling(60, min_periods=30).corr(sector_r)

        idx = pd.MultiIndex.from_arrays(
            [corr.index, [code] * len(corr)], names=["Date", "Code"]
        )
        tmp = pd.Series(corr.values, index=idx)
        corr_series.update(tmp)

    corr_series = corr_series.dropna()
    corr_series.index = corr_series.index.set_names(["Date", "Code"])
    corr_series = corr_series.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(corr_series)

@register_factor(
    name="stock_sector_beta_60",
    description="个股-板块Beta因子，个股60日相对所属板块的Beta系数截面排名（高Beta=弹性大排前）。",
    category="sector",
    thesis="个股相对板块的Beta反映其在板块内的进攻性——高Beta个股在板块上涨时弹性更大，是板块内的高杠杆标的。在确认板块方向后，高Beta个股是放大收益的工具。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_stock_sector_beta_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    sector_ret_panel = _build_sector_return_panel(context)
    primary = _build_primary_sector_map(context)

    ret_wide = ret.unstack("Code")
    common_dates = ret_wide.index.intersection(sector_ret_panel.index)

    beta_series = pd.Series(np.nan, index=ret.index)

    for code in ret_wide.columns:
        if code not in primary.index:
            continue
        sec = primary.loc[code]
        if pd.isna(sec) or sec not in sector_ret_panel.columns:
            continue

        stock_r = ret_wide.loc[common_dates, code]
        sector_r = sector_ret_panel.loc[common_dates, sec]

        cov = stock_r.rolling(60, min_periods=30).cov(sector_r)
        var = sector_r.rolling(60, min_periods=30).var()
        beta = cov / var.replace(0, np.nan)

        idx = pd.MultiIndex.from_arrays(
            [beta.index, [code] * len(beta)], names=["Date", "Code"]
        )
        tmp = pd.Series(beta.values, index=idx)
        beta_series.update(tmp)

    beta_series = beta_series.dropna()
    beta_series.index = beta_series.index.set_names(["Date", "Code"])
    beta_series = beta_series.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(beta_series)

@register_factor(
    name="stock_sector_vol_ratio_20",
    description="个股-板块波动率比因子，个股20日波动率/所属板块20日波动率截面排名（低比值=比板块更稳定排前）。",
    category="sector",
    thesis="个股相对板块的波动率比值反映其风险特征——波动率低于板块均值的个股是行业内的'防御型'标的，在板块下跌时往往跌幅更小。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_stock_sector_vol_ratio_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_20 = ret.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())

    primary = _build_primary_sector_map(context)
    codes = vol_20.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"vol": vol_20.values, "sector": sectors.values}, index=vol_20.index)
    df = df.dropna(subset=["sector"])
    df["sector_vol"] = df.groupby(["Date", "sector"])["vol"].transform("mean")
    df["vol_ratio"] = df["vol"] / df["sector_vol"].replace(0, np.nan)

    return cross_sectional_rank(-df["vol_ratio"])

@register_factor(
    name="stock_sector_timing",
    description="个股-板块领先滞后因子，个股5日收益与板块1-5日前收益的最大相关系数截面排名（领先板块=排前）。",
    category="sector",
    thesis="个股相对板块的领先/滞后关系可以识别行业龙头——领先板块上涨的个股往往是行业的风向标，信息优势和定价权使其最先启动。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_stock_sector_timing(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    sector_ret_panel = _build_sector_return_panel(context)
    primary = _build_primary_sector_map(context)

    ret_wide = ret.unstack("Code")
    common_dates = ret_wide.index.intersection(sector_ret_panel.index)

    timing_series = pd.Series(np.nan, index=ret.index)

    for code in ret_wide.columns:
        if code not in primary.index:
            continue
        sec = primary.loc[code]
        if pd.isna(sec) or sec not in sector_ret_panel.columns:
            continue

        stock_r = ret_wide.loc[common_dates, code]
        sector_r = sector_ret_panel.loc[common_dates, sec]

        # Lead-lag: corr(stock_ret_t, sector_ret_t-lag) for lag=0..5
        max_corr = pd.Series(0.0, index=stock_r.index)
        for lag in range(6):
            sector_lagged = sector_r.shift(lag)
            corr = stock_r.rolling(20, min_periods=10).corr(sector_lagged)
            max_corr = pd.concat([max_corr, corr.abs()], axis=1).max(axis=1)

        # Timing score: high correlation at any lag = stock is connected to sector
        idx = pd.MultiIndex.from_arrays(
            [max_corr.index, [code] * len(max_corr)], names=["Date", "Code"]
        )
        tmp = pd.Series(max_corr.values, index=idx)
        timing_series.update(tmp)

    timing_series = timing_series.dropna()
    timing_series.index = timing_series.index.set_names(["Date", "Code"])
    timing_series = timing_series.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(timing_series)

# ═══════════════════════════════════════════════════════════════════════════════
# H — Sector Spillover & PE Factors
# ═══════════════════════════════════════════════════════════════════════════════

def _load_concept_stock_map(context: FactorContext) -> dict[str, list[str]]:
    """Build stock → list of THS concept (类型='N') sector codes."""
    cache = getattr(_load_concept_stock_map, "_cache", None)
    if cache is not None:
        return cache

    src = context.repo.paths.source_root
    cs = context.repo._read_parquet(src / "ths_constituent_stocks.parquet")
    sc = context.repo._read_parquet(src / "ths_sector_categories.parquet")

    concept_codes = set(sc[sc["type"] == "N"]["index_code"])
    cs_concept = cs[cs["index_code"].isin(concept_codes)]

    stock_map: dict[str, list[str]] = {}
    for _, row in cs_concept.iterrows():
        code = _pad_code(row["stock_code"])
        ths = row["index_code"]
        stock_map.setdefault(code, []).append(ths)

    _load_concept_stock_map._cache = stock_map
    return stock_map

def _build_concept_primary_map(context: FactorContext) -> pd.Series:
    """Stock → primary concept sector (first one)."""
    cmap = _load_concept_stock_map(context)
    mapping = {code: sectors[0] if sectors else None for code, sectors in cmap.items()}
    return pd.Series(mapping, name="concept")

# ═══════════════════════════════════════════════════════════════════════════════
