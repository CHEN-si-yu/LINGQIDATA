from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, stack_date_code

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

    .. warning::
       **数据泄露风险**: ``ths_constituent_stocks.parquet`` 不含日期字段，
       为静态快照。板块/概念成分股会随时间变化（新增、剔除），
       但此映射将所有历史日期统一应用当前快照，可能引入前瞻偏差。
       对于 type='I'（行业板块）成分股变化较慢、影响有限；
       对于 type='N'（概念板块）主题板块更动态、泄漏风险更高。
       若数据源提供历史成分股快照，应改为按日期动态加载。
       当前实现仅使用 type='I' 行业板块以最小化泄露风险。
    """
    cache = getattr(_load_stock_sector_map, "_cache", None)
    if cache is not None:
        return cache

    src = context.repo.paths.source_root
    cs = context.repo._read_parquet(src / "ths_constituent_stocks.parquet")
    sc = context.repo._read_parquet(src / "ths_sector_categories.parquet")

    import logging
    _logger = logging.getLogger(__name__)

    # Only use industry sectors
    industry_codes = set(sc[sc["type"] == "I"]["index_code"])
    cs_industry = cs[cs["index_code"].isin(industry_codes)]
    if cs_industry.empty:
        _logger.debug(
            "sector: type=I industry sectors not found in constituent data, "
            "trying stock_list.parquet industry field (申万行业) as fallback."
        )
        # ── Fallback 1: stock_list.parquet industry field ──
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
                        "sector: using stock_list industry (%d sectors, %d stocks).",
                        n_sec, len(stock_map_fb),
                    )
                    _load_stock_sector_map._cache = stock_map_fb
                    return stock_map_fb
        except Exception:
            _logger.warning("sector: stock_list fallback failed, trying type=BB/N.")

        # ── Fallback 2: BB/N market-relative ──
        _logger.warning(
            "sector: falling back to type=BB/N — factors become market-relative."
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
    # When using non-THS sector codes (e.g. stock_list industry names),
    # skip the sector_panel filter since codes won't match ths_daily columns.
    _any_ths_code = any(str(s).endswith(".TI") for sectors in stock_map.values() for s in sectors)
    sector_stocks: dict[str, list[str]] = {}
    for code, sectors in stock_map.items():
        for ths in sectors:
            if (not _any_ths_code) or (ths in sector_panel.columns):
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
    combined = stack_date_code(combined)
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

    # When using non-THS sector codes (e.g. stock_list industry names),
    # skip the sector_panel filter since codes won't match ths_daily columns.
    _any_ths_code = any(str(s).endswith(".TI") for sectors in stock_map.values() for s in sectors)
    sector_stocks: dict[str, list[str]] = {}
    for code, sectors in stock_map.items():
        for ths in sectors:
            if (not _any_ths_code) or (ths in sector_panel.columns):
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
    combined = stack_date_code(combined)
    combined.name = "sector_amount_rank"
    return cross_sectional_rank(combined)

# ═══════════════════════════════════════════════════════════════════════════════
