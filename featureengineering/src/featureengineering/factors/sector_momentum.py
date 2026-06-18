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
from ..utils import cross_sectional_rank


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
        return pd.Series(dtype=float)

    combined = pd.concat(parts, axis=1)
    # Average across sectors for stocks that belong to multiple sectors
    combined = combined.T.groupby(level=0).mean().T
    combined.columns.name = "Code"
    combined = combined.stack().reorder_levels(["Date", "Code"]).sort_index()
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

@register_factor(
    name="sector_turnover_breakout",
    description=(
        "板块换手率突破因子：个股所属行业板块当日换手率/20日均换手率-1，"
        "板块内均值后截面排名。"
    ),
    category="sector",
    thesis=(
        "行业板块换手率的骤然放大通常意味着有新增资金入场或市场关注度急剧提升，"
        "是板块行情启动/加速的前兆。突破幅度越大，短期演绎概率越高。"
    ),
    dependencies=(
        "ths_daily.parquet",
        "ths_constituent_stocks.parquet",
        "ths_sector_categories.parquet",
    ),
)
def factor_sector_turnover_breakout(context: FactorContext):
    stock_map = _load_stock_sector_map(context)
    turnover_panel = _load_sector_turnover_panel(context)
    sector_stocks = _build_sector_stocks(stock_map, set(turnover_panel.columns))

    # Sector 20d average turnover
    turnover_20d_avg = turnover_panel.rolling(20, min_periods=5).mean()

    # Breakout ratio: today / trailing average - 1
    breakout = turnover_panel / turnover_20d_avg.replace(0, np.nan) - 1.0
    breakout = breakout.replace([np.inf, -np.inf], np.nan)

    stock_metric = _map_sector_metric_to_stocks(breakout, sector_stocks)

    if stock_metric.empty:
        return pd.Series(
            index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Code"]),
            dtype=float,
        )

    return cross_sectional_rank(stock_metric)


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 4: sector_vol_ratio
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="sector_vol_ratio",
    description=(
        "板块波动率比率因子：个股所属行业板块20日波动率/全A指数20日波动率，"
        "映射到个股后截面排名。"
    ),
    category="sector",
    thesis=(
        "高波板块相对于全A的波动率比率反映了板块的beta属性。"
        "当比率放大时，板块的系统性风险上升；"
        "持续高比的板块具有更高弹性和交易型机会。"
    ),
    dependencies=(
        "ths_daily.parquet",
        "ths_constituent_stocks.parquet",
        "ths_sector_categories.parquet",
    ),
)
def factor_sector_vol_ratio(context: FactorContext):
    stock_map = _load_stock_sector_map(context)
    sector_panel = _load_ths_sector_panel(context)
    sector_stocks = _build_sector_stocks(stock_map, set(sector_panel.columns))

    # Sector daily returns and 20d rolling volatility
    sector_ret = sector_panel.pct_change(1)
    sector_vol_20 = sector_ret.rolling(20, min_periods=10).std()

    # Ratio against all-A index vol
    if _ALL_A_CODE in sector_vol_20.columns:
        all_a_vol_20 = sector_vol_20[_ALL_A_CODE]
        vol_ratio = sector_vol_20.div(all_a_vol_20.replace(0, np.nan).fillna(1), axis=0)
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)
    else:
        # All-A index not available — use absolute sector vol as fallback
        vol_ratio = sector_vol_20

    stock_metric = _map_sector_metric_to_stocks(vol_ratio, sector_stocks)

    if stock_metric.empty:
        return pd.Series(
            index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Code"]),
            dtype=float,
        )

    return cross_sectional_rank(stock_metric)


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 5: sector_momentum_rotation
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="sector_momentum_rotation",
    description=(
        "板块动量轮动因子：个股所属行业板块5日收益率在所有板块中的百分位排名，"
        "映射到个股后截面排名。衡量个股所在板块是否为当前热点。"
    ),
    category="sector",
    thesis=(
        "当某个行业板块的收益率排名处于所有板块前列时，该板块处于资金追逐的"
        "热点状态。将板块热度映射到个股，可以捕捉到板块轮动中个股的跟涨机会。"
        "动量轮动信号比单纯收益率信号更稳定，因为它相对于全市场板块排序。"
    ),
    dependencies=(
        "ths_daily.parquet",
        "ths_constituent_stocks.parquet",
        "ths_sector_categories.parquet",
    ),
)
def factor_sector_momentum_rotation(context: FactorContext):
    stock_map = _load_stock_sector_map(context)
    sector_panel = _load_ths_sector_panel(context)
    sector_stocks = _build_sector_stocks(stock_map, set(sector_panel.columns))

    # Sector 5d returns  (Date x ths_code)
    sector_ret_5 = sector_panel.pct_change(5)

    # Cross-sectional percentile rank of sectors (within each date)
    sector_rank = sector_ret_5.rank(axis=1, pct=True)

    stock_metric = _map_sector_metric_to_stocks(sector_rank, sector_stocks)

    if stock_metric.empty:
        return pd.Series(
            index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Code"]),
            dtype=float,
        )

    return cross_sectional_rank(stock_metric)


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
    combined = combined.stack().reorder_levels(["Date", "Code"]).sort_index()
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


@register_factor(
    name="sector_size_factor",
    description="板块内小市值效应因子 (小市值相对大市值的超额)。",
    category="sector",
    thesis="在板块内部，小市值股票相对大市值股票存在系统性超额收益",
    dependencies=("finance.parquet", "stock_list.parquet", "ths_constituent_stocks.parquet"),
)
def factor_sector_size_factor(context: FactorContext):
    finance = context.load("finance.parquet")
    mv = finance["total_mv"]
    industry_map = context.repo.load_industry_map()
    codes = mv.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"mv": mv.values, "industry": industries.values}, index=mv.index)
    df = df.dropna(subset=["industry"])
    df["mv_rank"] = df.groupby(["Date", "industry"])["mv"].transform(
        lambda x: x.rank(pct=True)
    )
    # Small cap (low rank) is good
    return cross_sectional_rank(-df["mv_rank"])


@register_factor(
    name="sector_earnings_consistency",
    description="板块盈利一致性因子 (行业内盈利正增长占比)。",
    category="sector",
    thesis="板块内多数公司盈利正增长意味着行业景气上行，个股受益于行业Beta",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_sector_earnings_consistency(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["netprofit_yoy"])
    industry_map = context.repo.load_industry_map()
    np_growth = fin["netprofit_yoy"]
    codes = np_growth.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"np_g": np_growth.values, "industry": industries.values}, index=np_growth.index)
    df = df.dropna(subset=["industry"])
    df["positive"] = (df["np_g"] > 0).astype(float)
    df["consistency"] = df.groupby(["Date", "industry"])["positive"].transform("mean")
    return cross_sectional_rank(df["consistency"])

    return cross_sectional_rank(stock_metric)
