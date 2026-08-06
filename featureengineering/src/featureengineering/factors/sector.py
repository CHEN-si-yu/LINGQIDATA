"""
行业板块因子模块（基于 stock_list.parquet 的静态行业映射）。

⚠️ 时点风险：stock_list.parquet 为当前快照，行业归属按当前值回填历史日期。
行业板块成分变动缓慢（区别于概念板块），影响有限；此警示保留。

历史说明：本模块原依赖 ths_constituent_stocks / ths_sector_categories /
ths_daily，但 type='I' 行业板块在现有数据中成分数为 0（5000 行成分全部为
type='BB'），THS 查询路径必然回退到 stock_list.industry，并白读 57.5MB 的
ths_daily.parquet。2026-07-31 重构：回退路径提升为唯一实现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, stack_date_code


def _pad_code(code: str) -> str:
    """Strip exchange suffix and zero-pad to 6 digits, e.g. '000001.SZ' → '000001'."""
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _load_industry_map(context: FactorContext) -> dict[str, list[str]]:
    """Build a mapping: stock_code (6-digit) -> list of industry names.

    Uses ``stock_list.parquet``'s static ``industry`` field (申万/东财行业).
    Cached at module level to avoid redundant I/O.
    """
    cache = getattr(_load_industry_map, "_cache", None)
    if cache is not None:
        return cache

    src = context.repo.paths.source_root
    stock_list = context.repo._read_parquet(src / "stock_list.parquet")
    allowed = context.repo.allowed_codes

    stock_map: dict[str, list[str]] = {}
    for _, row in stock_list.iterrows():
        code = _pad_code(row["stock_code"])
        if allowed and code not in allowed:
            continue
        ind = (row.get("industry") or "").strip()
        if ind:
            stock_map.setdefault(code, []).append(ind)

    _load_industry_map._cache = stock_map
    return stock_map


def _build_sector_stocks(
    stock_map: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Invert the stock→industry map into industry→[stocks]."""
    sector_stocks: dict[str, list[str]] = {}
    for code, industries in stock_map.items():
        for ind in industries:
            sector_stocks.setdefault(ind, []).append(code)
    return sector_stocks


def _map_sector_metric_to_stocks(
    sector_metric: pd.DataFrame,
    sector_stocks: dict[str, list[str]],
) -> pd.Series:
    """Broadcast a Date × industry metric back to (Date, Code), averaging
    across a stock's industries when it belongs to several."""
    parts: list[pd.DataFrame] = []
    for ind, codes in sector_stocks.items():
        if ind not in sector_metric.columns:
            continue
        ind_values = sector_metric[ind]
        available_codes = [c for c in codes]
        if not available_codes:
            continue
        df = pd.DataFrame(
            {code: ind_values for code in available_codes},
            index=ind_values.index,
        )
        parts.append(df)

    if not parts:
        idx = pd.MultiIndex.from_tuples([], names=["Date", "Code"])
        return pd.Series(dtype=float, index=idx, name="value")

    combined = pd.concat(parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    combined.columns.name = "Code"
    combined = stack_date_code(combined)
    combined.name = "value"
    return combined


# ── 行业内排名因子 ─────────────────────────────────────────────────────────

@register_factor(
    name="sector_mv_rank",
    description="行业内市值占比因子，个股总市值在所属行业内的截面排名。",
    category="sector",
    thesis="行业内市值最大的公司通常是行业龙头，享有流动性溢价、机构关注度和定价权优势。行业内排名比绝对市值排名更能反映公司在细分赛道中的竞争地位。",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_sector_mv_rank(context: FactorContext):
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"].where(finance["total_mv"] > 0, np.nan)

    stock_map = _load_industry_map(context)
    sector_stocks = _build_sector_stocks(stock_map)

    # For each industry on each date, rank stocks by market value within it
    mv_frame = total_mv.unstack("Code")  # Date × Code
    rank_parts: list[pd.Series] = []

    for ind, codes in sector_stocks.items():
        available = [c for c in codes if c in mv_frame.columns]
        if len(available) < 3:
            continue
        sector_mv = mv_frame[available]
        sector_rank = sector_mv.rank(axis=1, pct=True)  # within-industry rank
        rank_parts.append(sector_rank)

    if not rank_parts:
        return cross_sectional_rank(total_mv)

    # Average within-industry rank across all industries each stock belongs to
    combined = pd.concat(rank_parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    combined = stack_date_code(combined)
    combined.name = "sector_mv_rank"
    return cross_sectional_rank(combined)


@register_factor(
    name="sector_amount_rank",
    description="行业内成交额占比因子，个股成交额在所属行业内的截面排名。",
    category="sector",
    thesis="行业内成交额占比高的个股是资金关注的焦点，具有更好的流动性和价格发现效率。成交额占比持续领先的个股往往是行业的情绪龙头或机构重仓标的。",
    dependencies=("daily.parquet", "stock_list.parquet"),
)
def factor_sector_amount_rank(context: FactorContext):
    daily = context.load("daily.parquet")
    amount = daily["amount"].where(daily["amount"] > 0, np.nan)

    stock_map = _load_industry_map(context)
    sector_stocks = _build_sector_stocks(stock_map)

    amt_frame = amount.unstack("Code")
    rank_parts: list[pd.DataFrame] = []

    for ind, codes in sector_stocks.items():
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
