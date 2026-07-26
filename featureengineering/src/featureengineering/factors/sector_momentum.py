"""
THS 行业 / 概念动量因子模块。

基于 ths_daily.parquet（THS 板块指数日行情）、ths_constituent_stocks.parquet（成分股）
和 ths_sector_categories.parquet（板块分类）数据，构建行业板块层面的动量类因子
以及概念热度因子，填补板块指数日行情数据在因子库中的空白。

依赖的数据集：
- ths_daily.parquet         约 1.4M 行，1 575 个 THS 板块代码，日频 OHLCV + 换手率
- ths_constituent_stocks.parquet  板块一成分股对应关系
- ths_sector_categories.parquet   板块元信息（类型／名称／交易所等）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, stack_date_code

# ── THS 板块数据加载辅助 ───────────────────────────────────────────────────────

def _pad_code(code: str) -> str:
    """Strip exchange suffix and zero-pad to 6 digits, e.g. '000001.SZ' -> '000001'."""
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
    raw["trade_date"] = (
        raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    )
    panel = raw.pivot(index="trade_date", columns="ths_code", values="close")
    panel.index.name = "Date"
    panel.columns.name = "ths_code"
    return panel.sort_index()

def _load_stock_sector_map(context: FactorContext, sector_types: tuple[str, ...] = ("I",)) -> dict[str, list[str]]:
    """Build a mapping: stock_code (6-digit) -> list of THS sector codes.

    Only includes sectors whose type is in *sector_types*.
    Cached at module level to avoid redundant I/O.

    .. warning::
       **数据泄露风险**: ``ths_constituent_stocks.parquet`` 不含日期字段，
       为静态快照。板块/概念成分股会随时间变化（新增、剔除），
       但此映射将所有历史日期统一应用当前快照，可能引入前瞻偏差。
       对于 type='I'（行业板块）成分股变化较慢、影响有限；
       对于 type='N'（概念板块）主题板块更动态、泄漏风险更高。
       若数据源提供历史成分股快照，应改为按日期动态加载。
       当前实现假设成分股变动对因子影响在可接受范围内。
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
        import logging
        _logger = logging.getLogger(__name__)
        _logger.debug(
            "sector_momentum: type=I industry sectors not found in constituent data, "
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
                        "sector_momentum: using stock_list industry "
                        "(%d sectors, %d stocks). "
                        "Note: factors depending on ths_daily.parquet "
                        "will have reduced effectiveness.",
                        n_sec, len(stock_map_fb),
                    )
                    cache[sector_types] = stock_map_fb
                    return stock_map_fb
        except Exception:
            _logger.warning(
                "sector_momentum: stock_list fallback failed, trying type=BB/N."
            )

        # ── Fallback 2: BB/N market-relative ──
        _logger.warning(
            "sector_momentum: falling back to type=BB/N "
            "— factors become market-relative."
        )
        allowed_codes = set(sc[sc["type"].isin(["BB", "N"])]["index_code"])
        cs_filtered = cs[cs["index_code"].isin(allowed_codes)]

    stock_map: dict[str, list[str]] = {}
    for _, row in cs_filtered.iterrows():
        code = _pad_code(row["stock_code"])
        ths = row["index_code"]
        stock_map.setdefault(code, []).append(ths)

    cache[sector_types] = stock_map
    return stock_map

def _load_sector_turnover_panel(context: FactorContext) -> pd.DataFrame:
    """Load THS sector daily turnover_rate as a Date x ths_code wide DataFrame."""
    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "ths_daily.parquet")
    raw = raw.copy()
    raw["trade_date"] = (
        raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    )
    panel = raw.pivot(index="trade_date", columns="ths_code", values="turnover_rate")
    panel.index.name = "Date"
    panel.columns.name = "ths_code"
    return panel.sort_index()

# ── 通用构造／映射辅助 ─────────────────────────────────────────────────────────

def _build_sector_stocks(
    stock_map: dict[str, list[str]],
    valid_sectors: set[str],
) -> dict[str, list[str]]:
    """Reverse *stock_map* into sector -> list of stock codes, filtered to *valid_sectors*."""
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
    """Map a sector-level (Date x ths_code) metric to stock-level (Date x Code) Series.

    For each sector, broadcasts the sector value to all its constituent stocks.
    Stocks belonging to multiple sectors get the (unweighted) average across them.
    """
    parts: list[pd.DataFrame] = []
    for ths, codes in sector_stocks.items():
        if ths not in sector_metric.columns:
            continue
        ths_values = sector_metric[ths]
        available_codes = [c for c in codes]
        if not available_codes:
            continue
        # Every stock in this sector gets the same sector-level value
        df = pd.DataFrame(
            {code: ths_values for code in available_codes},
            index=ths_values.index,
        )
        parts.append(df)

    if not parts:
        idx = pd.MultiIndex.from_tuples([], names=['Date', 'Code'])
        return pd.Series(dtype=float, index=idx, name='value')

    combined = pd.concat(parts, axis=1)
    # Average across sectors for stocks that belong to multiple sectors
    combined = combined.T.groupby(level=0).mean().T
    combined.columns.name = "Code"
    combined = stack_date_code(combined)
    combined.name = "value"
    return combined

# ── THS 全 A 指数代码 ─────────────────────────────────────────────────────────

_ALL_A_CODE = "700001.TI"

# ═══════════════════════════════════════════════════════════════════════════════
# 因子 1: sector_rel_strength_5d
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="sector_rel_strength_5d",
    description=(
        "行业相对强度因子(5日)：个股所属THS行业板块的5日超额收益"
        "（减去全A指数），板块内取均值后截面排名。"
    ),
    category="sector",
    thesis=(
        "个股的行业板块短期相对强度反映资金在行业层面的轮动方向。"
        "相比个股自身动量，行业层面的动量更稳定且具有更低的换手率。"
        "叠加全A基准后更加纯净地捕捉行业alpha。"
    ),
    dependencies=(
        "daily_adj.parquet",
        "ths_daily.parquet",
        "ths_constituent_stocks.parquet",
        "ths_sector_categories.parquet",
    ),
)
def factor_sector_rel_strength_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    stock_map = _load_stock_sector_map(context)
    sector_panel = _load_ths_sector_panel(context)
    sector_stocks = _build_sector_stocks(stock_map, set(sector_panel.columns))

    # Sector 5d returns  (Date x ths_code)
    sector_ret_5 = sector_panel.pct_change(5)

    # Subtract all-A index 5d return for excess return
    if _ALL_A_CODE in sector_ret_5.columns:
        all_a_ret_5 = sector_ret_5[_ALL_A_CODE]
        sector_ret_5 = sector_ret_5.sub(all_a_ret_5.fillna(0), axis=0)

    stock_metric = _map_sector_metric_to_stocks(sector_ret_5, sector_stocks)

    if stock_metric.empty:
        # Fallback: stock-level 5d return (no sector decomposition available)
        fallback = daily_adj["close"].groupby(level="Code").transform(
            lambda s: s.pct_change(5)
        )
        return cross_sectional_rank(fallback)

    return cross_sectional_rank(stock_metric)

# ═══════════════════════════════════════════════════════════════════════════════
# 因子 2: sector_rel_strength_20d
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="sector_rel_strength_20d",
    description=(
        "行业相对强度因子(20日)：个股所属THS行业板块的20日超额收益"
        "（减去全A指数），板块内取均值后截面排名。"
    ),
    category="sector",
    thesis=(
        "中期(20日)行业相对强度捕捉的是机构资金在行业层面调仓的趋势，"
        "比5日信号更持久、噪音更低。与sector_rel_strength_5d形成短中期互补。"
    ),
    dependencies=(
        "daily_adj.parquet",
        "ths_daily.parquet",
        "ths_constituent_stocks.parquet",
        "ths_sector_categories.parquet",
    ),
)
def factor_sector_rel_strength_20d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    stock_map = _load_stock_sector_map(context)
    sector_panel = _load_ths_sector_panel(context)
    sector_stocks = _build_sector_stocks(stock_map, set(sector_panel.columns))

    # Sector 20d returns
    sector_ret_20 = sector_panel.pct_change(20)

    if _ALL_A_CODE in sector_ret_20.columns:
        all_a_ret_20 = sector_ret_20[_ALL_A_CODE]
        sector_ret_20 = sector_ret_20.sub(all_a_ret_20.fillna(0), axis=0)

    stock_metric = _map_sector_metric_to_stocks(sector_ret_20, sector_stocks)

    if stock_metric.empty:
        fallback = daily_adj["close"].groupby(level="Code").transform(
            lambda s: s.pct_change(20)
        )
        return cross_sectional_rank(fallback)

    return cross_sectional_rank(stock_metric)

# ═══════════════════════════════════════════════════════════════════════════════
# 因子 3: sector_turnover_breakout
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 因子 4: sector_vol_ratio
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 因子 5: sector_momentum_rotation
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 因子 6: sector_leader_laggard_spread
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="sector_leader_laggard_spread",
    description=(
        "板块内部分化因子：个股所属行业板块内5日收益率最高与最低之差，"
        "映射到个股后截面排名。差值越大说明板块内部分化越严重。"
    ),
    category="sector",
    thesis=(
        "板块内收益率分化程度反映板块共识度：分化越大，说明板块内资金存在"
        "严重分歧或龙头独立行情；分化小的板块齐涨共跌，趋势延续性更强。"
        "该因子对板块配置和风格判断有参考价值。"
    ),
    dependencies=(
        "daily_adj.parquet",
        "ths_constituent_stocks.parquet",
        "ths_sector_categories.parquet",
    ),
)
def factor_sector_leader_laggard_spread(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    stock_map = _load_stock_sector_map(context)

    # Build sector -> stocks mapping (no sector-panel filter needed;
    # this factor works with stock-level returns)
    sector_stocks: dict[str, list[str]] = {}
    for code, sectors in stock_map.items():
        for ths in sectors:
            sector_stocks.setdefault(ths, []).append(code)

    # Stock-level 5d returns
    close = daily_adj["close"]
    ret_5 = close.groupby(level="Code").transform(lambda s: s.pct_change(5))
    ret_5_frame = ret_5.unstack("Code")  # Date x Code

    spread_parts: list[pd.DataFrame] = []
    for ths, codes in sector_stocks.items():
        available = [c for c in codes if c in ret_5_frame.columns]
        if len(available) < 3:
            continue
        sector_ret = ret_5_frame[available]
        # Only compute spread on dates where >= 3 stocks have valid data
        valid_counts = sector_ret.notna().sum(axis=1)
        spread = sector_ret.max(axis=1) - sector_ret.min(axis=1)
        spread = spread.where(valid_counts >= 3, np.nan)
        # Broadcast spread to all stocks in this sector
        df = pd.DataFrame(
            {code: spread for code in available},
            index=spread.index,
        )
        spread_parts.append(df)

    if not spread_parts:
        return pd.Series(
            index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Code"]),
            dtype=float,
        )

    combined = pd.concat(spread_parts, axis=1)
    combined = combined.T.groupby(level=0).mean().T
    combined.columns.name = "Code"
    combined = stack_date_code(combined)
    combined.name = "sector_leader_laggard_spread"
    return cross_sectional_rank(combined)

# ═══════════════════════════════════════════════════════════════════════════════
# 因子 7: sector_amount_momentum_5d
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="sector_amount_momentum_5d",
    description=(
        "板块成交额动量因子：个股所属行业板块平均成交额5日变化率，"
        "映射到个股后截面排名。反映资金在板块层面的流入/流出动能。"
    ),
    category="sector",
    thesis=(
        "行业板块成交额的变化是机构资金调仓的代理变量。"
        "板块成交额持续放大意味着资金正在系统性流入该板块，"
        "是板块级别行情启动的重要先行指标。"
        "比个股成交额信号更稳定，不易被个别大单扰动。"
    ),
    dependencies=(
        "daily_adj.parquet",
        "ths_constituent_stocks.parquet",
        "ths_sector_categories.parquet",
    ),
)
def factor_sector_amount_momentum_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    stock_map = _load_stock_sector_map(context)

    # Build sector -> stocks mapping
    sector_stocks: dict[str, list[str]] = {}
    for code, sectors in stock_map.items():
        for ths in sectors:
            sector_stocks.setdefault(ths, []).append(code)

    amount = daily_adj["amount"].where(daily_adj["amount"] > 0, np.nan)
    amount_frame = amount.unstack("Code")  # Date x Code

    # Compute sector-average amount for each sector
    sector_avg_amount: dict[str, pd.Series] = {}
    for ths, codes in sector_stocks.items():
        available = [c for c in codes if c in amount_frame.columns]
        if not available:
            continue
        sector_avg_amount[ths] = amount_frame[available].mean(axis=1)

    if not sector_avg_amount:
        # Fallback: stock-level amount 5d change rate
        return cross_sectional_rank(
            amount.groupby(level="Code").transform(lambda s: s.pct_change(5))
        )

    sector_avg_df = pd.DataFrame(sector_avg_amount)  # Date x ths_code

    # 5d change rate of sector average amount
    sector_amount_mom_5 = sector_avg_df.pct_change(5)
    sector_amount_mom_5 = sector_amount_mom_5.replace([np.inf, -np.inf], np.nan)

    stock_metric = _map_sector_metric_to_stocks(sector_amount_mom_5, sector_stocks)

    if stock_metric.empty:
        return pd.Series(
            index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Code"]),
            dtype=float,
        )

    return cross_sectional_rank(stock_metric)

# ── Supplementary sector momentum factors ──────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
