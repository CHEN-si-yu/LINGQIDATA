"""
THS 板块日频因子模块 — 基于 ths_daily.parquet 未充分利用字段。

thds_daily.parquet 包含 2,570 个 THS 板块指数（行业+概念+风格）的日频 OHLCV 数据，
当前因子库仅使用了 close 和 turnover_rate 两个字段。本模块系统性利用以下 8 个
此前未使用的字段，构建板块层面的日频 alpha 因子：

  pct_change  — 板块涨跌幅（%）
  vol         — 板块成交量
  open        — 板块开盘价
  high        — 板块最高价
  low         — 板块最低价
  pre_close   — 板块前收盘价
  change      — 板块价格变动（绝对值）
  avg_price   — 板块日内均价（VWAP 代理）

架构：遵循 sector_momentum.py 的 helper 模式：
  1. 加载 ths_daily.parquet → pivot 为 Date × ths_code 宽表
  2. 加载 stock→sector 映射（从 ths_constituent_stocks + ths_sector_categories）
  3. 在板块层面计算指标 → _map_sector_metric_to_stocks() 映射到个股
  4. cross_sectional_rank() 截面排名

分类：
  A. 板块收益因子（pct_change）         12 factors
  B. 板块相对强度因子（个股 vs 板块）    8 factors
  C. 板块成交量因子（vol）              10 factors
  D. 板块日内结构因子（open/high/low）   12 factors
  E. 板块均价因子（avg_price）           6 factors
  F. 板块资金流向代理因子（vol×price）    6 factors
  G. 板块交互/复合因子                  10 factors
  H. 多板块整合因子                      6 factors
  ─────────────────────────────────────────────
  合计                                  70 factors
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, safe_rank, stack_date_code

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: code padding
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_code(code: str) -> str:
    """Strip exchange suffix and zero-pad to 6 digits, e.g. '000001.SZ' -> '000001'."""
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: stock ↔ sector mapping (reused from sector_momentum.py pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_stock_sector_map(
    context: FactorContext,
    sector_types: tuple[str, ...] = ("I",),
) -> dict[str, list[str]]:
    """Build stock_code (6-digit) → list of THS sector codes mapping.

    Cached at module level. Falls back to stock_list.parquet industry field
    or BB/N type sectors when type='I' yields empty results.
    """
    cache = getattr(_load_stock_sector_map, "_cache", None)
    if cache is None:
        cache = {}
        _load_stock_sector_map._cache = cache
    if sector_types in cache:
        return cache[sector_types]

    src = context.repo.paths.source_root
    cs = context.repo._read_parquet(src / "ths_constituent_stocks.parquet")
    sc = context.repo._read_parquet(src / "ths_sector_categories.parquet")

    allowed_codes = set(sc[sc["type"].isin(sector_types)]["index_code"])
    cs_filtered = cs[cs["index_code"].isin(allowed_codes)]

    if cs_filtered.empty and sector_types == ("I",):
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
                    cache[sector_types] = stock_map_fb
                    return stock_map_fb
        except Exception:
            pass
        allowed_codes = set(sc[sc["type"].isin(["BB", "N"])]["index_code"])
        cs_filtered = cs[cs["index_code"].isin(allowed_codes)]

    stock_map: dict[str, list[str]] = {}
    for _, row in cs_filtered.iterrows():
        code = _pad_code(row["stock_code"])
        ths = row["index_code"]
        stock_map.setdefault(code, []).append(ths)

    cache[sector_types] = stock_map
    return stock_map

def _build_sector_stocks(
    stock_map: dict[str, list[str]],
    valid_sectors: set[str],
) -> dict[str, list[str]]:
    """Reverse stock_map into sector → list of stock codes, filtered to valid_sectors."""
    sector_stocks: dict[str, list[str]] = {}
    for code, sectors in stock_map.items():
        for ths in sectors:
            if ths in valid_sectors:
                sector_stocks.setdefault(ths, []).append(code)
    return sector_stocks

def _map_sector_metric_to_stocks(
    sector_metric: pd.DataFrame,
    sector_stocks: dict[str, list[str]],
) -> pd.Series:
    """Map sector-level (Date × ths_code) metric to stock-level (Date × Code) Series.

    Multi-sector stocks get the unweighted average across their sectors.
    """
    parts: list[pd.DataFrame] = []
    for ths, codes in sector_stocks.items():
        if ths not in sector_metric.columns:
            continue
        ths_values = sector_metric[ths]
        available_codes = [c for c in codes]
        if not available_codes:
            continue
        df = pd.DataFrame(
            {code: ths_values for code in available_codes},
            index=ths_values.index,
        )
        parts.append(df)

    if not parts:
        idx = pd.MultiIndex.from_tuples([], names=['Date', 'Code'])
        return pd.Series(dtype=float, index=idx, name='value')

    combined = pd.concat(parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    combined.columns.name = "Code"
    combined = stack_date_code(combined)
    combined.name = "value"
    return combined

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: multi-field THS panel loaders (cached)
# ═══════════════════════════════════════════════════════════════════════════════

_ALL_A_CODE = "700001.TI"

def _load_ths_raw(context: FactorContext) -> pd.DataFrame:
    """Load ths_daily.parquet raw data with normalized dates. Cached."""
    cache = getattr(_load_ths_raw, "_cache", None)
    if cache is not None:
        return cache.copy()
    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "ths_daily.parquet")
    raw = raw.copy()
    raw["trade_date"] = (
        raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    )
    _load_ths_raw._cache = raw.copy()
    return raw

def _load_ths_panel(context: FactorContext, field: str) -> pd.DataFrame:
    """Load a single field from ths_daily as Date × ths_code wide panel. Cached per field."""
    cache = getattr(_load_ths_panel, "_cache", None)
    if cache is None:
        cache = {}
        _load_ths_panel._cache = cache
    if field in cache:
        return cache[field].copy()

    raw = _load_ths_raw(context)
    panel = raw.pivot(index="trade_date", columns="ths_code", values=field)
    panel.index.name = "Date"
    panel.columns.name = "ths_code"
    panel = panel.sort_index()
    cache[field] = panel.copy()
    return panel

def _load_ths_multi_panel(context: FactorContext, fields: list[str]) -> dict[str, pd.DataFrame]:
    """Load multiple fields from ths_daily as Date × ths_code panels. Cached per field."""
    result = {}
    for f in fields:
        result[f] = _load_ths_panel(context, f)
    return result

# Convenience loaders for commonly-used fields
def _sector_close(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "close")

def _sector_pct_chg(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "pct_change")

def _sector_vol(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "vol")

def _sector_open(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "open")

def _sector_high(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "high")

def _sector_low(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "low")

def _sector_pre_close(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "pre_close")

def _sector_change(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "change")

def _sector_avg_price(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "avg_price")

def _sector_turnover(context: FactorContext) -> pd.DataFrame:
    return _load_ths_panel(context, "turnover_rate")

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build sector_stocks once for a factor group
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sector_context(context: FactorContext):
    """Return (stock_map, sector_stocks) for THS industry sectors. Cached."""
    cache = getattr(_get_sector_context, "_cache", None)
    if cache is not None:
        return cache
    stock_map = _load_stock_sector_map(context)
    close_panel = _sector_close(context)
    sector_stocks = _build_sector_stocks(stock_map, set(close_panel.columns))
    result = (stock_map, sector_stocks)
    _get_sector_context._cache = result
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# Section A: 板块收益因子 (pct_change) — 12 factors
# ═══════════════════════════════════════════════════════════════════════════════

THS_DEPS = (
    "ths_daily.parquet",
    "ths_constituent_stocks.parquet",
    "ths_sector_categories.parquet",
)
THS_DAILY_DEPS = (
    "daily_adj.parquet",
    "ths_daily.parquet",
    "ths_constituent_stocks.parquet",
    "ths_sector_categories.parquet",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section B: 板块相对强度因子（个股 vs 板块）— 8 factors
# ═══════════════════════════════════════════════════════════════════════════════

def _stock_returns(context: FactorContext, window: int) -> pd.Series:
    """Compute per-stock return over window days from daily_adj.close."""
    daily = context.load("daily_adj.parquet")
    close = daily["close"]
    return close.groupby(level="Code").transform(lambda s: s.pct_change(window))

# ═══════════════════════════════════════════════════════════════════════════════
# Section C: 板块成交量因子 (vol) — 10 factors
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section D: 板块日内结构因子 (open/high/low/pre_close/change) — 12 factors
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section E: 板块均价因子 (avg_price) — 6 factors
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section F: 板块资金流向代理因子 (vol × price 近似成交额) — 6 factors
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section G: 板块交互/复合因子 — 10 factors
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section H: 多板块整合因子 — 6 factors
# ═══════════════════════════════════════════════════════════════════════════════

