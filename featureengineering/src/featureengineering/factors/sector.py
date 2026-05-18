from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── THS 板块数据加载辅助 ───────────────────────────────────────────────────────

def _pad_code(code: str) -> str:
    """Strip exchange suffix and zero-pad to 6 digits, e.g. '000001.SZ' → '000001'."""
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _load_ths_sector_panel(context: FactorContext) -> pd.DataFrame:
    """Load THS sector daily close prices as a Date x ths_code wide DataFrame."""
    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "ths_daily.parquet")
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    panel = raw.pivot(index="trade_date", columns="ths_code", values="close")
    panel.index.name = "Date"
    panel.columns.name = "ths_code"
    return panel.sort_index()


def _load_stock_sector_map(context: FactorContext) -> dict[str, list[str]]:
    """Build a mapping: stock_code (6-digit) -> list of THS industry sector codes.

    Only includes sectors of type 'I' (industry classification).
    Cached at module level to avoid redundant I/O.
    """
    cache = getattr(_load_stock_sector_map, "_cache", None)
    if cache is not None:
        return cache

    src = context.repo.paths.source_root
    cs = context.repo._read_parquet(src / "ths_constituent_stocks.parquet")
    sc = context.repo._read_parquet(src / "ths_sector_categories.parquet")

    # Only use industry sectors
    industry_codes = set(sc[sc["type"] == "I"]["index_code"])
    cs_industry = cs[cs["index_code"].isin(industry_codes)]

    stock_map: dict[str, list[str]] = {}
    for _, row in cs_industry.iterrows():
        code = _pad_code(row["stock_code"])
        ths = row["index_code"]
        stock_map.setdefault(code, []).append(ths)

    _load_stock_sector_map._cache = stock_map
    return stock_map


def _stock_sector_returns(
    context: FactorContext, window: int
) -> pd.Series:
    """Compute average sector return for each (date, stock) over *window* days.

    Returns a Series with (Date, Code) MultiIndex.
    """
    sector_panel = _load_ths_sector_panel(context)
    sector_ret = sector_panel.pct_change(window, fill_method=None)

    stock_map = _load_stock_sector_map(context)

    # Align dates: keep only dates present in both sector data and allowed pool
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    stock_dates = close.index.get_level_values("Date").unique()

    # For each stock, get its sector returns and average
    records: list[dict] = []
    all_codes = sorted(stock_map.keys())
    for code in all_codes:
        sectors = stock_map[code]
        available = [s for s in sectors if s in sector_ret.columns]
        if not available:
            continue
        avg_ret = sector_ret[available].mean(axis=1).dropna()
        for date_val in avg_ret.index:
            if date_val in stock_dates:
                records.append({"Date": date_val, "Code": code, "sector_ret": avg_ret[date_val]})

    if not records:
        result = pd.Series(dtype=float, name="sector_ret")
        result.index = pd.MultiIndex.from_tuples([], names=["Date", "Code"])
        return result

    result = pd.DataFrame(records).set_index(["Date", "Code"])["sector_ret"]
    result.index = result.index.set_names(["Date", "Code"])
    return result.sort_index()


# ── THS 板块因子 ───────────────────────────────────────────────────────────────

@register_factor(
    name="sector_rel_strength_20",
    description="板块内相对强度因子，个股20日收益率减去所属THS行业板块20日均收益率的截面排名。",
    category="sector",
    thesis="剥离板块贝塔后的个股alpha是纯净的选股信号。强于板块的个股表明公司层面的积极因素正在被定价，弱于板块的个股则有公司层面风险。板块内相对强度比绝对动量更具可比性。",
    dependencies=("daily_adj.parquet", "ths_daily.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_rel_strength_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    stock_ret = close.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    sector_ret = _stock_sector_returns(context, 20)

    aligned = pd.concat([stock_ret.rename("stock"), sector_ret.rename("sector")], axis=1)
    rel_strength = aligned["stock"] - aligned["sector"]
    return cross_sectional_rank(rel_strength)


@register_factor(
    name="sector_momentum_20",
    description="板块动量因子，个股所属THS行业板块的20日均收益率截面排名。",
    category="sector",
    thesis="行业轮动是A股重要的收益来源，强势板块中的个股享有贝塔红利。板块动量因子将行业趋势信号映射到个股层面，捕捉行业层面的动量效应。",
    dependencies=("ths_daily.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_momentum_20(context: FactorContext):
    sector_ret = _stock_sector_returns(context, 20)
    return cross_sectional_rank(sector_ret)


@register_factor(
    name="sector_beta_60",
    description="板块Beta因子，个股60日收益率对所属THS行业板块收益率的滚动Beta截面排名。",
    category="sector",
    thesis="高板块Beta的个股在板块上涨时弹性更大但下跌时跌幅也更大。在趋势明确的行情中高Beta占优，在震荡市中低Beta更稳健。板块Beta反映了个股对行业系统性风险的暴露程度。",
    dependencies=("daily_adj.parquet", "ths_daily.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_beta_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    stock_ret_1d = close.groupby(level="Code").transform(
        lambda s: s.pct_change(1)
    )

    sector_ret_1d = _stock_sector_returns(context, 1)

    aligned = pd.concat(
        [stock_ret_1d.rename("stock"), sector_ret_1d.rename("sector")], axis=1
    ).dropna()

    def _rolling_beta(grp: pd.DataFrame) -> pd.Series:
        stock = grp["stock"]
        sector = grp["sector"]
        cov = stock.rolling(60, min_periods=30).cov(sector)
        var = sector.rolling(60, min_periods=30).var()
        return cov / var.replace(0, np.nan)

    beta = aligned.groupby(level="Code").apply(_rolling_beta)
    if isinstance(beta.index, pd.MultiIndex) and beta.index.nlevels > 2:
        beta = beta.droplevel(0)
    beta = beta.clip(-5, 5)
    return cross_sectional_rank(beta)


@register_factor(
    name="sector_mv_rank",
    description="板块内市值占比因子，个股总市值在所属THS行业板块内的截面排名。",
    category="sector",
    thesis="板块内市值最大的公司通常是行业龙头，享有流动性溢价、机构关注度和定价权优势。板块内市值排名比绝对市值排名更能反映公司在细分赛道中的竞争地位。",
    dependencies=("finance.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_mv_rank(context: FactorContext):
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"].where(finance["total_mv"] > 0, np.nan)

    stock_map = _load_stock_sector_map(context)
    sector_panel = _load_ths_sector_panel(context)

    # Build sector→stocks mapping
    sector_stocks: dict[str, list[str]] = {}
    for code, sectors in stock_map.items():
        for ths in sectors:
            if ths in sector_panel.columns:
                sector_stocks.setdefault(ths, []).append(code)

    # For each sector on each date, rank stocks by market value within the sector
    mv_frame = total_mv.unstack("Code")  # Date × Code
    rank_parts: list[pd.Series] = []

    for ths, codes in sector_stocks.items():
        available = [c for c in codes if c in mv_frame.columns]
        if len(available) < 3:
            continue
        sector_mv = mv_frame[available]
        sector_rank = sector_mv.rank(axis=1, pct=True)  # within-sector rank
        rank_parts.append(sector_rank)

    if not rank_parts:
        return cross_sectional_rank(total_mv)

    # Average within-sector rank across all sectors each stock belongs to
    combined = pd.concat(rank_parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    combined = combined.stack().reorder_levels(["Date", "Code"]).sort_index()
    combined.name = "sector_mv_rank"
    return cross_sectional_rank(combined)


@register_factor(
    name="sector_amount_rank",
    description="板块内成交额占比因子，个股成交额在所属THS行业板块内的截面排名。",
    category="sector",
    thesis="板块内成交额占比高的个股是资金关注的焦点，具有更好的流动性和价格发现效率。成交额占比持续领先的个股往往是板块的情绪龙头或机构重仓标的。",
    dependencies=("daily_adj.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_amount_rank(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    amount = daily_adj["amount"].where(daily_adj["amount"] > 0, np.nan)

    stock_map = _load_stock_sector_map(context)
    sector_panel = _load_ths_sector_panel(context)

    sector_stocks: dict[str, list[str]] = {}
    for code, sectors in stock_map.items():
        for ths in sectors:
            if ths in sector_panel.columns:
                sector_stocks.setdefault(ths, []).append(code)

    amt_frame = amount.unstack("Code")
    rank_parts: list[pd.DataFrame] = []

    for ths, codes in sector_stocks.items():
        available = [c for c in codes if c in amt_frame.columns]
        if len(available) < 3:
            continue
        sector_amt = amt_frame[available]
        sector_rank = sector_amt.rank(axis=1, pct=True)
        rank_parts.append(sector_rank)

    if not rank_parts:
        return cross_sectional_rank(amount)

    combined = pd.concat(rank_parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    combined = combined.stack().reorder_levels(["Date", "Code"]).sort_index()
    combined.name = "sector_amount_rank"
    return cross_sectional_rank(combined)


@register_factor(
    name="sector_rotation_20",
    description="板块轮动速度因子，板块截面排名20日变化的绝对值截面排名（高轮动排后）。",
    category="sector",
    thesis="板块轮动速度过快意味着市场缺乏主线、资金频繁切换，系统性风险较高；轮动速度低意味着市场风格稳定、趋势可持续，选股Alpha更容易实现。",
    dependencies=("ths_daily.parquet", "ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_rotation_20(context: FactorContext):
    sector_panel = _load_ths_sector_panel(context)
    sector_ret = sector_panel.pct_change(20, fill_method=None)

    # Cross-sectional rank of sector returns each day
    sector_rank_today = sector_ret.rank(axis=1, pct=True)
    sector_rank_20d_ago = sector_ret.shift(20).rank(axis=1, pct=True)

    # Rank change magnitude (high change = fast rotation)
    rank_change = (sector_rank_today - sector_rank_20d_ago).abs()

    # Map sector rotation to stocks
    stock_map = _load_stock_sector_map(context)
    stock_codes = sorted(stock_map.keys())

    records: list[dict] = []
    for code in stock_codes:
        sectors = stock_map[code]
        available = [s for s in sectors if s in rank_change.columns]
        if available:
            avg_change = rank_change[available].mean(axis=1).dropna()
            for date_val, val in avg_change.items():
                records.append({"Date": date_val, "Code": code, "rotation": val})

    result = pd.DataFrame(records).set_index(["Date", "Code"])["rotation"]
    result.index = result.index.set_names(["Date", "Code"])
    result = result.sort_index()
    return cross_sectional_rank(-result)


@register_factor(
    name="sector_diversification",
    description="板块分散度因子，个股所属THS行业板块数量的截面排名（高分散排后=主业聚焦偏好）。",
    category="sector",
    thesis="所属板块数量越多意味着业务多元化程度越高，但也可能主业不突出。A股历史上主业聚焦的公司长期表现优于过度多元化的公司，本质是对'专注溢价'的量化表达。",
    dependencies=("ths_constituent_stocks.parquet", "ths_sector_categories.parquet"),
)
def factor_sector_diversification(context: FactorContext):
    stock_map = _load_stock_sector_map(context)

    daily_adj = context.load("daily_adj.parquet")
    all_dates = daily_adj.index.get_level_values("Date").unique()
    all_codes = daily_adj.index.get_level_values("Code").unique()

    sector_count = pd.Series(
        {code: len(sectors) for code, sectors in stock_map.items()}
    )
    sector_count = sector_count.reindex(all_codes).fillna(0)

    result = sector_count.to_frame("count")
    result["Date"] = all_dates[0]
    result = result.set_index("Date", append=True).reorder_levels(["Date", "Code"])
    result = result["count"]

    # Broadcast to all dates
    full_idx = pd.MultiIndex.from_product([all_dates, all_codes], names=["Date", "Code"])
    result = result.reindex(full_idx).ffill()
    return cross_sectional_rank(-result)
