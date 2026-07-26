"""
股票身份/特征扩展因子 — 基于 stock_list.parquet 未充分利用字段。

stock_list.parquet 包含 5,867 只股票的基本身份信息（11 列），
当前因子库仅使用了 7 个字段。本模块系统利用以下此前未使用/未充分利用的字段：

  symbol         — 交易代码（前缀可识别板块归属：60/00/30/68）
  act_name       — 实际控制人名称（可用于细化国企/民企分类）
  delist_date    — 退市日期（退市风险预警）

  交叉使用 ths_sector_categories.name — 板块名称关键词分类

架构：
  1. 通过 context.repo._read_parquet 读取 stock_list.parquet（原始数据）
  2. 通过 _broadcast_identity_to_daily 广播到日频面板
  3. cross_sectional_rank 截面排名

分类：
  A. 板块归属因子（symbol 前缀）      6 factors
  B. 实际控制人因子（act_name）       6 factors
  C. 退市风险因子（delist_date）      3 factors
  D. 板块名称因子（ths_sector_categories.name） 5 factors
  ─────────────────────────────────────────────
  合计                              20 factors
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank

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

def _load_stock_identity(context: FactorContext) -> pd.DataFrame:
    """Load stock_list.parquet with full identity fields, cached at module level."""
    cache = getattr(_load_stock_identity, "_cache", None)
    if cache is not None:
        return cache

    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "stock_list.parquet")
    pool = raw[raw["list_status"] == "L"].copy()
    pool["Code"] = pool["stock_code"].apply(_pad_code)

    cols = ["Code"]
    for c in ["industry", "area", "act_ent_type", "is_hs", "list_date",
              "symbol", "act_name", "delist_date", "name"]:
        if c in pool.columns:
            cols.append(c)
    pool = pool[cols]

    allowed = context.repo.allowed_codes
    if allowed:
        pool = pool[pool["Code"].isin(allowed)]

    pool = pool.drop_duplicates(subset=["Code"]).set_index("Code").sort_index()
    _load_stock_identity._cache = pool
    return pool

def _broadcast_to_daily(
    identity_series: pd.Series,
    context: FactorContext,
) -> pd.Series:
    """Broadcast a per-stock static Series to the full (Date, Code) daily panel."""
    daily_adj = context.load("daily_adj.parquet")
    all_dates = daily_adj.index.get_level_values("Date").unique()
    all_codes = daily_adj.index.get_level_values("Code").unique()

    aligned = identity_series.reindex(all_codes)
    full_idx = pd.MultiIndex.from_product(
        [all_dates, all_codes], names=["Date", "Code"]
    )
    result = pd.Series(index=full_idx, dtype=float)
    for date in all_dates:
        result.loc[date] = aligned.values
    return result

def _load_sector_categories(context: FactorContext) -> pd.DataFrame:
    """Load ths_sector_categories.parquet with name field, cached."""
    cache = getattr(_load_sector_categories, "_cache", None)
    if cache is not None:
        return cache
    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "ths_sector_categories.parquet")
    _load_sector_categories._cache = raw
    return raw

def _load_constituent_stocks(context: FactorContext) -> pd.DataFrame:
    """Load ths_constituent_stocks.parquet, cached."""
    cache = getattr(_load_constituent_stocks, "_cache", None)
    if cache is not None:
        return cache
    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "ths_constituent_stocks.parquet")
    _load_constituent_stocks._cache = raw
    return raw

def _build_sector_name_features(context: FactorContext) -> pd.Series:
    """Build per-stock sector name keyword features. Cached.

    For each stock, checks the names of all its THS industry sectors (type='I')
    for keyword matches and returns a binary flag Series.
    """
    cache = getattr(_build_sector_name_features, "_cache", None)
    if cache is not None:
        return cache

    identity = _load_stock_identity(context)
    cs = _load_constituent_stocks(context)
    sc = _load_sector_categories(context)

    # Filter to industry sectors
    industry_codes = set(sc[sc["type"] == "I"]["index_code"])
    cs_industry = cs[cs["index_code"].isin(industry_codes)]

    # Build stock → sector_names mapping
    stock_sector_names: dict[str, list[str]] = {}
    for _, row in cs_industry.iterrows():
        code = _pad_code(row["stock_code"])
        idx_code = row["index_code"]
        # Get sector name
        name_row = sc[sc["index_code"] == idx_code]
        if name_row.empty:
            continue
        sector_name = str(name_row.iloc[0]["name"])
        stock_sector_names.setdefault(code, []).append(sector_name)

    allowed_codes = identity.index.tolist()

    # Keyword categories
    tech_kw = ["科技", "信息", "电子", "计算机", "软件", "通信", "互联网", "芯片", "半导体", "人工智能", "机器人"]
    finance_kw = ["金融", "银行", "保险", "证券", "信托", "房地产"]
    medical_kw = ["医药", "医疗", "生物", "制药", "中药", "化学药", "器械", "疫苗"]
    manufacturing_kw = ["制造", "工业", "机械", "汽车", "设备", "化工", "材料", "钢铁", "有色"]
    new_energy_kw = ["新能源", "光伏", "风电", "锂电", "电池", "储能", "核电", "氢能", "充电桩", "碳中和"]

    def _has_keyword(names: list[str], keywords: list[str]) -> float:
        for name in names:
            for kw in keywords:
                if kw in name:
                    return 1.0
        return 0.0

    result = {}
    for code in allowed_codes:
        names = stock_sector_names.get(code, [])
        result.setdefault("sector_has_tech", {})[code] = _has_keyword(names, tech_kw)
        result.setdefault("sector_has_finance", {})[code] = _has_keyword(names, finance_kw)
        result.setdefault("sector_has_medical", {})[code] = _has_keyword(names, medical_kw)
        result.setdefault("sector_has_manufacturing", {})[code] = _has_keyword(names, manufacturing_kw)
        result.setdefault("sector_has_new_energy", {})[code] = _has_keyword(names, new_energy_kw)

    output = {k: pd.Series(v, name=k) for k, v in result.items()}
    _build_sector_name_features._cache = output
    return output

# ═══════════════════════════════════════════════════════════════════════════════
# Section A: 板块归属因子 (symbol 前缀) — 6 factors
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section B: 实际控制人因子 (act_name) — 6 factors
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section C: 退市风险因子 (delist_date) — 3 factors
# ═══════════════════════════════════════════════════════════════════════════════

