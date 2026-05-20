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



# ── THS 板块因子 ───────────────────────────────────────────────────────────────



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
