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

    industry_codes = set(sc[sc["type"] == "I"]["index_code"])
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
        return pd.Series(dtype=float)
    result = pd.concat(result_parts)
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# A — Sector-Market Relationship Factors
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="sector_corr_to_market_20",
    description="板块-市场相关性因子，个股所属THS行业板块20日收益与全A等权收益的相关系数截面排名（高相关排前=与市场同步）。",
    category="sector",
    thesis="板块与市场的相关性反映该板块是顺周期还是逆周期——高相关板块在大盘涨时确定性更高，低相关板块提供分散化价值。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_corr_to_market_20(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)
    all_a = _build_all_a_return(context)

    # Rolling 20-day correlation of each sector with all-A return
    common_dates = sector_ret.index.intersection(all_a.index)
    sector_ret_aligned = sector_ret.loc[common_dates]
    all_a_aligned = all_a.loc[common_dates]

    corr_panel = sector_ret_aligned.rolling(20, min_periods=10).corr(all_a_aligned)
    # After rolling corr, we get a DataFrame with same shape as sector_ret_aligned
    # Actually rolling().corr() with another Series gives a Series

    # Compute rolling correlation for each sector column
    corr_df = pd.DataFrame(index=sector_ret_aligned.index, columns=sector_ret_aligned.columns)
    for col in sector_ret_aligned.columns:
        s = sector_ret_aligned[col]
        corr_df[col] = s.rolling(20, min_periods=10).corr(all_a_aligned)

    result = _sector_panel_to_stock(corr_df, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_corr_to_leader_20",
    description="板块-龙头相关性因子，个股所属板块与20日最强板块收益的相关系数截面排名。",
    category="sector",
    thesis="板块与市场龙头的相关性捕捉资金轮动效应——高相关意味着板块跟随龙头趋势，低相关则独立于市场热点。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_corr_to_leader_20(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)

    # The "leader" sector on each day = highest 20-day cumulative return
    cum_20 = sector_ret.rolling(20, min_periods=10).sum()
    # Safe idxmax — all-NaN rows raise ValueError in pandas
    valid_mask = cum_20.notna().any(axis=1)
    leader_col = pd.Series(np.nan, index=cum_20.index, dtype=object)
    if valid_mask.any():
        leader_col[valid_mask] = cum_20.loc[valid_mask].idxmax(axis=1)

    # Build leader return series: for each date, get the return of the leading sector
    leader_ret = pd.Series(np.nan, index=sector_ret.index)
    for date in sector_ret.index:
        if date in leader_col.index:
            lc = leader_col.loc[date]
            if pd.notna(lc) and lc in sector_ret.columns:
                leader_ret.loc[date] = sector_ret.loc[date, lc]

    corr_df = pd.DataFrame(index=sector_ret.index, columns=sector_ret.columns)
    for col in sector_ret.columns:
        s = sector_ret[col]
        corr_df[col] = s.rolling(20, min_periods=10).corr(leader_ret)

    result = _sector_panel_to_stock(corr_df, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_corr_stability_60",
    description="板块相关性稳定性因子，板块-市场20日滚动相关系数在60日内的标准差截面排名（取负向=不稳定排后）。",
    category="sector",
    thesis="板块-市场相关性的稳定性反映该板块的系统性风险是否可预期——相关性剧烈波动意味着板块驱动因素在变化，预测难度大。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_corr_stability_60(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)
    all_a = _build_all_a_return(context)

    common_dates = sector_ret.index.intersection(all_a.index)
    sector_ret_aligned = sector_ret.loc[common_dates]
    all_a_aligned = all_a.loc[common_dates]

    # Rolling 20-day corr, then std of that corr over 60 days
    corr_df = pd.DataFrame(index=sector_ret_aligned.index, columns=sector_ret_aligned.columns)
    for col in sector_ret_aligned.columns:
        s = sector_ret_aligned[col]
        corr_df[col] = s.rolling(20, min_periods=10).corr(all_a_aligned)

    stability = corr_df.rolling(60, min_periods=30).std()

    result = _sector_panel_to_stock(stability, context)
    return cross_sectional_rank(-result)


# ═══════════════════════════════════════════════════════════════════════════════
# B — Sector Beta & Alpha Factors
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="sector_beta_60",
    description="板块Beta因子，所属板块60日相对全A等权收益的Beta系数截面排名（高Beta排前=进攻型板块）。",
    category="sector",
    thesis="板块Beta量化了板块对市场波动的敏感度——高Beta板块在牛市中弹性更大，但下行风险也更高。是行业配置的核心参数。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_beta_60(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)
    all_a = _build_all_a_return(context)

    common_dates = sector_ret.index.intersection(all_a.index)
    sector_ret_aligned = sector_ret.loc[common_dates]
    all_a_aligned = all_a.loc[common_dates]

    # Rolling Beta = Cov(sector, market) / Var(market)
    market_var = all_a_aligned.rolling(60, min_periods=30).var()

    beta_df = pd.DataFrame(index=sector_ret_aligned.index, columns=sector_ret_aligned.columns)
    for col in sector_ret_aligned.columns:
        s = sector_ret_aligned[col]
        cov = s.rolling(60, min_periods=30).cov(all_a_aligned)
        beta_df[col] = cov / market_var.replace(0, np.nan)

    result = _sector_panel_to_stock(beta_df, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_beta_change_20",
    description="板块Beta变化因子，板块60日Beta的20日变化截面排名（Beta上升排前=资金加速流入）。",
    category="sector",
    thesis="Beta的变化方向反映资金对板块态度的边际转变——Beta上升意味着板块正从防御转向进攻，资金在加配该板块。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_beta_change_20(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)
    all_a = _build_all_a_return(context)

    common_dates = sector_ret.index.intersection(all_a.index)
    sector_ret_aligned = sector_ret.loc[common_dates]
    all_a_aligned = all_a.loc[common_dates]

    market_var = all_a_aligned.rolling(60, min_periods=30).var()

    beta_df = pd.DataFrame(index=sector_ret_aligned.index, columns=sector_ret_aligned.columns)
    for col in sector_ret_aligned.columns:
        s = sector_ret_aligned[col]
        cov = s.rolling(60, min_periods=30).cov(all_a_aligned)
        beta_df[col] = cov / market_var.replace(0, np.nan)

    beta_change = beta_df.diff(20)

    result = _sector_panel_to_stock(beta_change, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_alpha_20",
    description="板块Alpha因子，板块20日超额收益（实际收益-Beta×市场收益）截面排名。",
    category="sector",
    thesis="板块Alpha是剔除市场Beta影响后的纯粹超额收益——高Alpha板块有独立于大盘的上涨驱动力，Alpha持续性比Beta驱动的收益更强。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_alpha_20(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)
    all_a = _build_all_a_return(context)

    common_dates = sector_ret.index.intersection(all_a.index)
    sector_ret_aligned = sector_ret.loc[common_dates]
    all_a_aligned = all_a.loc[common_dates]

    market_var = all_a_aligned.rolling(60, min_periods=30).var()

    cum_sector_20 = sector_ret_aligned.rolling(20, min_periods=10).sum()

    alpha_df = pd.DataFrame(index=sector_ret_aligned.index, columns=sector_ret_aligned.columns)
    for col in sector_ret_aligned.columns:
        s = sector_ret_aligned[col]
        cov = s.rolling(60, min_periods=30).cov(all_a_aligned)
        beta = cov / market_var.replace(0, np.nan)
        market_cum = all_a_aligned.rolling(20, min_periods=10).sum()
        expected = beta * market_cum
        alpha_df[col] = cum_sector_20[col] - expected

    result = _sector_panel_to_stock(alpha_df, context)
    return cross_sectional_rank(result)


# ═══════════════════════════════════════════════════════════════════════════════
# C — Sector Rotation & Momentum Factors
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="sector_rotation_accel_5d",
    description="板块轮动加速度因子，板块20日动量排名的5日变化（二阶导）截面排名（加速上升的板块排前）。",
    category="sector",
    thesis="动量排名的变化速度（加速度）比排名本身更早捕捉轮动方向——排名加速上升的板块正在获得资金关注。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_rotation_accel_5d(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)

    # 20-day cumulative return for each sector
    cum_20 = sector_ret.rolling(20, min_periods=10).sum()

    # Cross-sectional rank of sectors by 20-day return (within each date)
    sector_rank = cum_20.rank(axis=1, pct=True)

    # Acceleration = 5-day change in sector rank
    accel = sector_rank.diff(5)

    result = _sector_panel_to_stock(accel, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_relative_strength_60d",
    description="板块相对强度因子，板块60日收益率-全A等权60日收益率截面排名。",
    category="sector",
    thesis="板块60日相对强度是中期行业轮动的核心指标——持续跑赢市场的板块往往处于产业上升周期，强者恒强的行业趋势在A股中可持续2-3个季度。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_relative_strength_60d(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)
    all_a = _build_all_a_return(context)

    common_dates = sector_ret.index.intersection(all_a.index)
    sector_ret_aligned = sector_ret.loc[common_dates]
    all_a_aligned = all_a.loc[common_dates]

    cum_sector_60 = sector_ret_aligned.rolling(60, min_periods=30).sum()
    cum_market_60 = all_a_aligned.rolling(60, min_periods=30).sum()

    rs = cum_sector_60.sub(cum_market_60, axis=0)

    result = _sector_panel_to_stock(rs, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_momentum_reversal_5d",
    description="板块动量反转因子，-(板块20日收益排名最高板块的5日反转)截面排名（超买板块=反转风险高排后）。",
    category="sector",
    thesis="涨得最猛的板块短期面临获利回吐压力——20日动量排名极高的板块在接下来5日往往跑输。均值回复在板块层面尤为显著。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_momentum_reversal_5d(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)

    cum_20 = sector_ret.rolling(20, min_periods=10).sum()
    sector_rank = cum_20.rank(axis=1, pct=True)

    # Reversal = -5d return (high past rank + negative recent return = reversal)
    ret_5d = sector_ret.rolling(5, min_periods=3).sum()
    # Higher reversal risk = high rank but negative 5d return
    reversal_risk = sector_rank * (-ret_5d)

    result = _sector_panel_to_stock(reversal_risk, context)
    return cross_sectional_rank(-result)


# ═══════════════════════════════════════════════════════════════════════════════
# D — Sector Turnover & Volume Factors
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="sector_turnover_divergence",
    description="板块换手率背离因子，(板块20日收益排名-板块换手率排名)截面排名（收益强+换手低=质量好排前）。",
    category="sector",
    thesis="板块收益排名高于换手率排名意味着用较少的换手实现了较大的涨幅——这是板块上涨质量的体现，类似个股的'低换手高收益'质量特征。",
    dependencies=("daily_adj.parquet", "finance.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_turnover_divergence(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)
    sector_to = _build_sector_turnover_panel(context)

    common_dates = sector_ret.index.intersection(sector_to.index)
    common_cols = sector_ret.columns.intersection(sector_to.columns)

    cum_20 = sector_ret.loc[common_dates, common_cols].rolling(20, min_periods=10).sum()
    to_avg = sector_to.loc[common_dates, common_cols].rolling(20, min_periods=10).mean()

    ret_rank = cum_20.rank(axis=1, pct=True)
    to_rank = to_avg.rank(axis=1, pct=True)

    divergence = ret_rank - to_rank  # positive = good return with low turnover

    result = _sector_panel_to_stock(divergence, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_up_volume_ratio_20",
    description="板块上涨量比因子，板块内20日上涨日成交量之和/总成交量截面排名（上涨放量=健康排前）。",
    category="sector",
    thesis="板块上涨日的成交量占比反映买盘的真实力度——上涨放量+下跌缩量是健康的量价关系，说明资金在积极买入而非被动持有。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_up_volume_ratio_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    vol = daily_adj["vol"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    primary = _build_primary_sector_map(context)
    codes = ret.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"ret": ret.values, "vol": vol.values, "sector": sectors.values}, index=ret.index)
    df = df.dropna(subset=["sector"])
    df["is_up"] = (df["ret"] > 0).astype(float)
    df["up_vol"] = df["is_up"] * df["vol"]

    sector_up_vol = df.groupby(["Date", "sector"])["up_vol"].sum()
    sector_total_vol = df.groupby(["Date", "sector"])["vol"].sum()

    up_ratio = (sector_up_vol / sector_total_vol.replace(0, np.nan))
    panel = up_ratio.unstack("sector")
    panel = panel.rolling(20, min_periods=10).mean()

    result = _sector_panel_to_stock(panel, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_amount_flow_5d",
    description="板块资金流向因子，板块内个股5日成交额变化率均值截面排名（成交放量=资金关注排前）。",
    category="sector",
    thesis="板块层面的成交额变化反映资金是否在涌入该板块——成交额是资金关注度的直接体现，板块成交额持续放大意味着机构在布局该赛道。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_amount_flow_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    amount = daily_adj["amount"]

    # 5-day amount change per stock
    amt_chg = amount.groupby(level="Code").transform(lambda s: s.pct_change(5))

    primary = _build_primary_sector_map(context)
    codes = amt_chg.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"amt_chg": amt_chg.values, "sector": sectors.values}, index=amt_chg.index)
    df = df.dropna(subset=["sector"])

    sector_flow = df.groupby(["Date", "sector"])["amt_chg"].mean()
    panel = sector_flow.unstack("sector")

    result = _sector_panel_to_stock(panel, context)
    return cross_sectional_rank(result)


# ═══════════════════════════════════════════════════════════════════════════════
# E — Sector Breadth & Extreme Indicators
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="sector_new_high_ratio_20",
    description="板块新高比例因子，板块内20日创60日新高股票占比截面排名（新高扩散=强势排前）。",
    category="sector",
    thesis="板块内创新高的股票比例是行业趋势强度的重要确认——高比例意味着行业上涨不是个别龙头拉动而是全面扩散，趋势持续性更强。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_new_high_ratio_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    # 60-day high per stock
    high_60 = close.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).max())
    is_new_high = (close >= high_60.shift(1)).astype(float)  # new high today vs previous 60d high

    primary = _build_primary_sector_map(context)
    codes = is_new_high.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"new_high": is_new_high.values, "sector": sectors.values}, index=is_new_high.index)
    df = df.dropna(subset=["sector"])

    sector_ratio = df.groupby(["Date", "sector"])["new_high"].mean()
    panel = sector_ratio.unstack("sector")
    panel = panel.rolling(20, min_periods=10).mean()

    result = _sector_panel_to_stock(panel, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_new_low_ratio_20",
    description="板块新低比例因子，-(板块内20日创60日新低股票占比)截面排名（新低扩散=弱势排后）。",
    category="sector",
    thesis="板块内创新低股票占比反映行业下跌的广度——新低比例持续高企意味着行业基本面或预期在恶化，应回避。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_new_low_ratio_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    low_60 = close.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).min())
    is_new_low = (close <= low_60.shift(1)).astype(float)

    primary = _build_primary_sector_map(context)
    codes = is_new_low.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"new_low": is_new_low.values, "sector": sectors.values}, index=is_new_low.index)
    df = df.dropna(subset=["sector"])

    sector_ratio = df.groupby(["Date", "sector"])["new_low"].mean()
    panel = sector_ratio.unstack("sector")
    panel = panel.rolling(20, min_periods=10).mean()

    result = _sector_panel_to_stock(panel, context)
    return cross_sectional_rank(-result)


@register_factor(
    name="sector_advance_decline_20",
    description="板块涨跌比因子，板块内20日上涨股票数/下跌股票数截面排名。",
    category="sector",
    thesis="涨跌比(AD ratio)是技术分析中衡量市场广度的经典指标——高AD比意味着板块内多数股票在上涨，行业处于普涨格局，趋势可靠。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_advance_decline_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    primary = _build_primary_sector_map(context)
    codes = ret.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"ret": ret.values, "sector": sectors.values}, index=ret.index)
    df = df.dropna(subset=["sector"])
    df["is_up"] = (df["ret"] > 0).astype(float)
    df["is_down"] = (df["ret"] < 0).astype(float)

    up_count = df.groupby(["Date", "sector"])["is_up"].sum()
    down_count = df.groupby(["Date", "sector"])["is_down"].sum()

    ad_ratio = up_count / down_count.replace(0, np.nan)
    panel = ad_ratio.unstack("sector")
    panel = panel.rolling(20, min_periods=10).mean()

    result = _sector_panel_to_stock(panel, context)
    return cross_sectional_rank(result)


# ═══════════════════════════════════════════════════════════════════════════════
# F — Sector Market Cap Share
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="sector_mc_share_change_20",
    description="板块市值占比变化因子，板块总市值/全市场总市值的20日变化截面排名。",
    category="sector",
    thesis="板块市值占比的变化反映产业结构的长期变迁——市值占比持续提升的板块是经济结构转型的受益者，具有长期配置价值。",
    dependencies=("finance.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_mc_share_change_20(context: FactorContext):
    sector_mv = _build_sector_mv_panel(context)

    total_mv = sector_mv.sum(axis=1)
    share = sector_mv.div(total_mv, axis=0)
    share_change = share.diff(20)

    result = _sector_panel_to_stock(share_change, context)
    return cross_sectional_rank(result)


# ═══════════════════════════════════════════════════════════════════════════════
# G — Stock-Sector Relationship Factors
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


@register_factor(
    name="sector_spillover_impact_20",
    description="板块溢出效应因子，关联板块（同属更大类）20日平均超额收益截面排名。",
    category="sector",
    thesis="同一大类行业下的细分板块之间存在溢出效应——关联板块的上涨会通过产业链、资金关注和情绪传导至相关板块。捕捉行业内轮动的前兆信号。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_spillover_impact_20(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)
    all_a = _build_all_a_return(context)

    common_dates = sector_ret.index.intersection(all_a.index)
    sector_ret_aligned = sector_ret.loc[common_dates]
    all_a_aligned = all_a.loc[common_dates]

    # Sector excess return over market
    cum_sec_20 = sector_ret_aligned.rolling(20, min_periods=10).sum()
    cum_mkt_20 = all_a_aligned.rolling(20, min_periods=10).sum()
    excess = cum_sec_20.sub(cum_mkt_20, axis=0)

    # For each sector, average excess return of all OTHER sectors (spillover)
    n_sectors = len(excess.columns)
    total_excess = excess.sum(axis=1)
    spillover = pd.DataFrame(index=excess.index, columns=excess.columns)
    for col in excess.columns:
        spillover[col] = (total_excess - excess[col]) / max(n_sectors - 1, 1)

    result = _sector_panel_to_stock(spillover, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_relative_pe_20",
    description="板块相对估值因子，板块综合PE在20日内的分位数截面排名（低估值分位=安全边际排前）。",
    category="sector",
    thesis="板块PE的历史分位数是行业择时中常用的估值指标——PE处于自身历史低位时板块有估值修复动力，处于高位则有回调风险。",
    dependencies=("finance.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_relative_pe_20(context: FactorContext):
    finance = context.load("finance.parquet")
    pe = finance["pe"].where(finance["pe"] > 0, np.nan)

    primary = _build_primary_sector_map(context)
    codes = pe.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"pe": pe.values, "sector": sectors.values}, index=pe.index)
    df = df.dropna(subset=["sector", "pe"])

    # Sector median PE
    sector_pe = df.groupby(["Date", "sector"])["pe"].median()
    panel = sector_pe.unstack("sector")

    # 252-day rolling percentile rank within each sector
    pe_pct = pd.DataFrame(index=panel.index, columns=panel.columns)
    for col in panel.columns:
        pe_pct[col] = panel[col].rolling(252, min_periods=60).rank(pct=True)

    result = _sector_panel_to_stock(pe_pct, context)
    return cross_sectional_rank(-result)  # low percentile = cheap = good


@register_factor(
    name="sector_pe_expansion_20",
    description="板块PE扩张因子，板块中位数PE 20日变化率截面排名（PE扩张=估值修复排前）。",
    category="sector",
    thesis="板块估值扩张（PE上升）反映市场对该行业未来盈利预期的上调——估值扩张期的板块处于戴维斯双击的'估值提升'阶段，短期动量强。",
    dependencies=("finance.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_pe_expansion_20(context: FactorContext):
    finance = context.load("finance.parquet")
    pe = finance["pe"].where(finance["pe"] > 0, np.nan)

    primary = _build_primary_sector_map(context)
    codes = pe.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"pe": pe.values, "sector": sectors.values}, index=pe.index)
    df = df.dropna(subset=["sector", "pe"])

    sector_pe = df.groupby(["Date", "sector"])["pe"].median()
    panel = sector_pe.unstack("sector")

    expansion = panel.pct_change(20)

    result = _sector_panel_to_stock(expansion, context)
    return cross_sectional_rank(result)


# ═══════════════════════════════════════════════════════════════════════════════
# I — Concept Board Factors (概念板块)
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


@register_factor(
    name="concept_momentum_5d",
    description="概念动量因子，个股所属概念板块5日等权平均收益截面排名。",
    category="sector",
    thesis="A股概念板块轮动频繁——概念板块的短期动量是最直接的交易信号。个股所属概念近期表现好意味着它处于当前市场热点中，短期关注度和资金流入概率更高。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_concept_momentum_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    cum_5 = ret.groupby(level="Code").transform(lambda s: s.rolling(5, min_periods=3).sum())

    concept_map = _load_concept_stock_map(context)

    # Compute concept-level average 5d return
    codes_all = cum_5.index.get_level_values("Code").unique()
    concept_ret: dict[str, pd.Series] = {}

    for code in codes_all:
        if code not in concept_map:
            continue
        concepts = concept_map[code]
        for cpt in concepts:
            if cpt not in concept_ret:
                concept_ret[cpt] = pd.Series(np.nan, index=cum_5.index.get_level_values("Date").unique())

    # For each concept, compute average return of member stocks
    ret_wide = cum_5.unstack("Code")
    for cpt in list(concept_ret.keys()):
        member_codes = [c for c in concept_map if cpt in concept_map[c] and c in ret_wide.columns]
        if len(member_codes) < 3:
            del concept_ret[cpt]
            continue
        concept_ret[cpt] = ret_wide[member_codes].mean(axis=1)

    # Map concept return to each stock (average across all concepts the stock belongs to)
    result_parts = []
    for code in codes_all:
        if code not in concept_map:
            continue
        concepts = [c for c in concept_map[code] if c in concept_ret]
        if not concepts:
            continue
        stock_concept_ret = pd.DataFrame({c: concept_ret[c] for c in concepts}).mean(axis=1)
        stock_concept_ret = stock_concept_ret.dropna()
        if stock_concept_ret.empty:
            continue
        idx = pd.MultiIndex.from_arrays(
            [stock_concept_ret.index, [code] * len(stock_concept_ret)],
            names=["Date", "Code"],
        )
        result_parts.append(pd.Series(stock_concept_ret.values, index=idx))

    if not result_parts:
        return cross_sectional_rank(pd.Series(dtype=float))

    result = pd.concat(result_parts)
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(result)


@register_factor(
    name="concept_turnover_heat",
    description="概念热度因子，个股所属概念板块5日换手率变化截面排名（热度上升排前）。",
    category="sector",
    thesis="概念板块换手率的变化反映市场关注度的边际变化——换手率快速上升的概念正处于舆论和资金的风口，但也需警惕过度拥挤。",
    dependencies=("finance.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_concept_turnover_heat(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    to_chg = turnover.groupby(level="Code").transform(lambda s: s.diff(5))

    concept_map = _load_concept_stock_map(context)

    to_chg_wide = to_chg.unstack("Code")
    codes_all = to_chg_wide.columns

    concept_chg: dict[str, pd.Series] = {}
    for code in codes_all:
        if code not in concept_map:
            continue
        for cpt in concept_map[code]:
            if cpt not in concept_chg:
                concept_chg[cpt] = pd.Series(np.nan, index=to_chg_wide.index)

    for cpt in list(concept_chg.keys()):
        member_codes = [c for c in concept_map if cpt in concept_map[c] and c in to_chg_wide.columns]
        if len(member_codes) < 3:
            del concept_chg[cpt]
            continue
        concept_chg[cpt] = to_chg_wide[member_codes].mean(axis=1)

    result_parts = []
    for code in codes_all:
        if code not in concept_map:
            continue
        concepts = [c for c in concept_map[code] if c in concept_chg]
        if not concepts:
            continue
        stock_heat = pd.DataFrame({c: concept_chg[c] for c in concepts}).mean(axis=1)
        stock_heat = stock_heat.dropna()
        if stock_heat.empty:
            continue
        idx = pd.MultiIndex.from_arrays(
            [stock_heat.index, [code] * len(stock_heat)], names=["Date", "Code"]
        )
        result_parts.append(pd.Series(stock_heat.values, index=idx))

    if not result_parts:
        return cross_sectional_rank(pd.Series(dtype=float))

    result = pd.concat(result_parts)
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(result)


@register_factor(
    name="concept_concentration",
    description="概念集中度因子，-(个股所属概念板块数量/总概念数)HHI截面排名（多概念=分散风险排前）。",
    category="sector",
    thesis="所属概念数量反映公司的'题材多样性'——概念覆盖度高的公司有更多催化剂的可能，但也可能主业不聚焦。适度的概念多样性（3-8个）在A股历史上表现最佳。",
    dependencies=("ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_concept_concentration(context: FactorContext):
    concept_map = _load_concept_stock_map(context)
    total_concepts = len(set(c for codes in concept_map.values() for c in codes))

    daily_adj = context.load("daily_adj.parquet")
    all_dates = daily_adj.index.get_level_values("Date").unique()
    all_codes = daily_adj.index.get_level_values("Code").unique()

    concept_count = pd.Series({code: len(sectors) for code, sectors in concept_map.items()})
    concept_count = concept_count.reindex(all_codes).fillna(0)

    # HHI of concept concentration: higher = more concentrated (fewer concepts)
    hhi = (concept_count / max(total_concepts, 1)) ** 2

    result = hhi.to_frame("hhi")
    result["Date"] = all_dates[0]
    result = result.set_index("Date", append=True).reorder_levels(["Date", "Code"])
    result = result["hhi"]
    full_idx = pd.MultiIndex.from_product([all_dates, all_codes], names=["Date", "Code"])
    result = result.reindex(full_idx).ffill()

    return cross_sectional_rank(-result)  # less concentrated = better diversified


@register_factor(
    name="concept_rotation_signal",
    description="概念轮动信号因子，概念板块5日收益排名vs 20日收益排名的差值截面排名（短期领先于中期=加速排前）。",
    category="sector",
    thesis="概念板块短期排名超过中期排名意味着轮动资金正在涌入——是概念启动的早期信号。此因子捕捉概念轮动的加速度，比单纯的动量排名更早发现热点切换。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_concept_rotation_signal(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    concept_map = _load_concept_stock_map(context)
    ret_wide = ret.unstack("Code")
    codes_all = ret_wide.columns

    cum_5 = ret_wide.rolling(5, min_periods=3).sum()
    cum_20 = ret_wide.rolling(20, min_periods=10).sum()

    # Concept-level returns
    concept_ret_5: dict[str, pd.Series] = {}
    concept_ret_20: dict[str, pd.Series] = {}

    for code in codes_all:
        if code not in concept_map:
            continue
        for cpt in concept_map[code]:
            if cpt not in concept_ret_5:
                concept_ret_5[cpt] = pd.Series(np.nan, index=cum_5.index)
                concept_ret_20[cpt] = pd.Series(np.nan, index=cum_20.index)

    for cpt in list(concept_ret_5.keys()):
        member_codes = [c for c in concept_map if cpt in concept_map[c] and c in cum_5.columns]
        if len(member_codes) < 3:
            del concept_ret_5[cpt]
            del concept_ret_20[cpt]
            continue
        concept_ret_5[cpt] = cum_5[member_codes].mean(axis=1)
        concept_ret_20[cpt] = cum_20[member_codes].mean(axis=1)

    # Rank concepts within each date
    concept_5_df = pd.DataFrame(concept_ret_5)
    concept_20_df = pd.DataFrame(concept_ret_20)
    rank_5 = concept_5_df.rank(axis=1, pct=True)
    rank_20 = concept_20_df.rank(axis=1, pct=True)

    rotation = rank_5 - rank_20  # positive = short-term leading long-term

    result_parts = []
    for code in codes_all:
        if code not in concept_map:
            continue
        concepts = [c for c in concept_map[code] if c in rotation.columns]
        if not concepts:
            continue
        stock_rot = rotation[concepts].mean(axis=1).dropna()
        if stock_rot.empty:
            continue
        idx = pd.MultiIndex.from_arrays(
            [stock_rot.index, [code] * len(stock_rot)], names=["Date", "Code"]
        )
        result_parts.append(pd.Series(stock_rot.values, index=idx))

    if not result_parts:
        return cross_sectional_rank(pd.Series(dtype=float))

    result = pd.concat(result_parts)
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(result)


@register_factor(
    name="concept_overlap_premium",
    description="多热点概念溢价因子，个股同时属于多个高收益概念的溢价截面排名。",
    category="sector",
    thesis="同时属于多个近期强势概念的股票享有'多热点叠加'效应——每个热点概念都能为该股带来增量关注和资金。多概念叠加的股票在轮动行情中持续受益。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_concept_overlap_premium(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    concept_map = _load_concept_stock_map(context)
    ret_wide = ret.unstack("Code")

    cum_5 = ret_wide.rolling(5, min_periods=3).sum()

    concept_ret: dict[str, pd.Series] = {}
    for code in ret_wide.columns:
        if code not in concept_map:
            continue
        for cpt in concept_map[code]:
            if cpt not in concept_ret:
                concept_ret[cpt] = pd.Series(np.nan, index=cum_5.index)

    for cpt in list(concept_ret.keys()):
        member_codes = [c for c in concept_map if cpt in concept_map[c] and c in cum_5.columns]
        if len(member_codes) < 3:
            del concept_ret[cpt]
            continue
        concept_ret[cpt] = cum_5[member_codes].mean(axis=1)

    concept_ret_df = pd.DataFrame(concept_ret)
    concept_rank = concept_ret_df.rank(axis=1, pct=True)  # high rank = hot concept

    # Premium = average rank of concepts a stock belongs to, weighted by how many hot concepts
    result_parts = []
    for code in ret_wide.columns:
        if code not in concept_map:
            continue
        concepts = [c for c in concept_map[code] if c in concept_rank.columns]
        if len(concepts) < 2:
            continue

        stock_ranks = concept_rank[concepts]
        # Weight: number of hot concepts (top 20%) × average rank
        is_hot = (stock_ranks > 0.8).astype(float)
        hot_count = is_hot.sum(axis=1)
        avg_rank = stock_ranks.mean(axis=1)
        premium = avg_rank * (1 + hot_count / len(concepts))

        premium = premium.dropna()
        if premium.empty:
            continue
        idx = pd.MultiIndex.from_arrays(
            [premium.index, [code] * len(premium)], names=["Date", "Code"]
        )
        result_parts.append(pd.Series(premium.values, index=idx))

    if not result_parts:
        return cross_sectional_rank(pd.Series(dtype=float))

    result = pd.concat(result_parts)
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(result)


# ═══════════════════════════════════════════════════════════════════════════════
# J — Extended Sector Factors
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="sector_fund_flow_heat_5d",
    description="板块资金流热度因子，板块内主力资金净流入5日均值截面排名（资金流入=热点排前）。",
    category="sector",
    thesis="板块层面的主力资金净流入反映机构在板块间的配置行为——资金持续流入的板块是机构当前青睐的方向，具有趋势跟随价值。",
    dependencies=("main_fund_flow.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_fund_flow_heat_5d(context: FactorContext):
    mff = context.load("main_fund_flow.parquet")
    net_mf = mff["net_mf_amount"]

    primary = _build_primary_sector_map(context)
    codes = net_mf.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"net_mf": net_mf.values, "sector": sectors.values}, index=net_mf.index)
    df = df.dropna(subset=["sector"])

    sector_net = df.groupby(["Date", "sector"])["net_mf"].sum()
    panel = sector_net.unstack("sector")
    panel = panel.rolling(5, min_periods=3).mean()

    result = _sector_panel_to_stock(panel, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_margin_trend_5d",
    description="板块融资趋势因子，板块内融资余额5日变化率均值截面排名（融资增加=看多排前）。",
    category="sector",
    thesis="融资资金是A股中最敏感的杠杆资金——板块融资余额的变化方向反映了高风险偏好资金对板块的态度。融资持续增加意味着杠杆资金在积极做多该板块。",
    dependencies=("margin_detail.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_margin_trend_5d(context: FactorContext):
    margin = context.load("margin_detail.parquet")
    rzye = margin["rzye"]  # 融资余额

    rzye_chg = rzye.groupby(level="Code").transform(lambda s: s.pct_change(5))

    primary = _build_primary_sector_map(context)
    codes = rzye_chg.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"chg": rzye_chg.values, "sector": sectors.values}, index=rzye_chg.index)
    df = df.dropna(subset=["sector"])

    sector_chg = df.groupby(["Date", "sector"])["chg"].mean()
    panel = sector_chg.unstack("sector")

    result = _sector_panel_to_stock(panel, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_cipin_concentration",
    description="板块筹码集中度因子，板块内winner_rate 20日变化均值截面排名（获利盘增加=筹码集中排前）。",
    category="sector",
    thesis="板块内筹码集中度的提升（获利盘比例上升）意味着板块正在从分散走向集中——这是主力资金在板块层面系统性收集筹码的标志。",
    dependencies=("cyq_perf.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_cipin_concentration(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    winner = cyq["winner_rate"]

    winner_chg = winner.groupby(level="Code").transform(lambda s: s.diff(20))

    primary = _build_primary_sector_map(context)
    codes = winner_chg.index.get_level_values("Code")
    sectors = codes.map(primary)

    df = pd.DataFrame({"chg": winner_chg.values, "sector": sectors.values}, index=winner_chg.index)
    df = df.dropna(subset=["sector"])

    sector_chg = df.groupby(["Date", "sector"])["chg"].mean()
    panel = sector_chg.unstack("sector")

    result = _sector_panel_to_stock(panel, context)
    return cross_sectional_rank(result)


@register_factor(
    name="sector_multi_dimension_score",
    description="板块多维度综合评分因子，(动量排名+资金流排名+估值排名+新高比排名)/4截面排名。",
    category="sector",
    thesis="综合动量、资金、估值和质量四个维度对板块进行评分——多维度共振的板块具有最强的投资确定性。单一维度可能产生假信号，多维度确认则显著提高胜率。",
    dependencies=("daily_adj.parquet", "finance.parquet", "main_fund_flow.parquet",
                  "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_multi_dimension_score(context: FactorContext):
    sector_ret = _build_sector_return_panel(context)

    cum_20 = sector_ret.rolling(20, min_periods=10).sum()
    mom_rank = cum_20.rank(axis=1, pct=True)

    # Fund flow dimension
    mff = context.load("main_fund_flow.parquet")
    net_mf = mff["net_mf_amount"]
    primary = _build_primary_sector_map(context)
    codes = net_mf.index.get_level_values("Code")
    sectors = codes.map(primary)
    df_flow = pd.DataFrame({"net_mf": net_mf.values, "sector": sectors.values}, index=net_mf.index).dropna(subset=["sector"])
    sector_flow = df_flow.groupby(["Date", "sector"])["net_mf"].sum().unstack("sector")
    sector_flow = sector_flow.rolling(5, min_periods=3).mean()
    common_dates = mom_rank.index.intersection(sector_flow.index)
    common_cols = mom_rank.columns.intersection(sector_flow.columns)
    flow_rank = sector_flow.loc[common_dates, common_cols].rank(axis=1, pct=True)
    mom_rank_aligned = mom_rank.loc[common_dates, common_cols]

    # Valuation dimension (inverse PE = cheap is good)
    finance = context.load("finance.parquet")
    pe = finance["pe"].where(finance["pe"] > 0, np.nan)
    codes_pe = pe.index.get_level_values("Code")
    sectors_pe = codes_pe.map(primary)
    df_pe = pd.DataFrame({"pe": pe.values, "sector": sectors_pe.values}, index=pe.index).dropna(subset=["sector", "pe"])
    sector_pe_med = df_pe.groupby(["Date", "sector"])["pe"].median().unstack("sector")
    common_dates2 = common_dates.intersection(sector_pe_med.index)
    common_cols2 = common_cols.intersection(sector_pe_med.columns)
    pe_rank = (-sector_pe_med.loc[common_dates2, common_cols2]).rank(axis=1, pct=True)  # low PE = high rank

    # New high ratio dimension
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    high_60 = close.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).max())
    is_new_high = (close >= high_60.shift(1)).astype(float)
    codes_nh = is_new_high.index.get_level_values("Code")
    sectors_nh = codes_nh.map(primary)
    df_nh = pd.DataFrame({"nh": is_new_high.values, "sector": sectors_nh.values}, index=is_new_high.index).dropna(subset=["sector"])
    sector_nh = df_nh.groupby(["Date", "sector"])["nh"].mean().unstack("sector")
    sector_nh = sector_nh.rolling(20, min_periods=10).mean()
    common_dates3 = common_dates2.intersection(sector_nh.index)
    common_cols3 = common_cols2.intersection(sector_nh.columns)
    nh_rank = sector_nh.loc[common_dates3, common_cols3].rank(axis=1, pct=True)

    # Align all
    fd = common_dates3
    fc = common_cols3
    composite = (
        mom_rank_aligned.loc[fd, fc]
        + flow_rank.loc[fd, fc]
        + pe_rank.loc[fd, fc]
        + nh_rank.loc[fd, fc]
    ) / 4.0

    result = _sector_panel_to_stock(composite, context)
    return cross_sectional_rank(result)
