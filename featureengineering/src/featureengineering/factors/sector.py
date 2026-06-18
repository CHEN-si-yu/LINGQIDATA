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


# ── Sector-relative volatility ─────────────────────────────────────────────

@register_factor(
    name="sector_relative_volatility_20",
    description="行业内相对波动率因子，个股20日波动率/行业内中位数波动率截面排名（低比排前）。",
    category="sector",
    thesis="行业内低波动股票在风险调整后表现更优，剔除行业波动特征后的相对低波才是真正的低风险alpha。与volatility_20互补：一个看绝对波动的截面排名，一个看行业内相对波动的排名。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_sector_relative_volatility_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    ret = daily_adj["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    codes = vol_20.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"vol_20": vol_20.values, "industry": industries.values}, index=vol_20.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["vol_20"].transform("median")
    df["relative"] = df["vol_20"] / df["sector_median"].replace(0, np.nan)
    return cross_sectional_rank(-df["relative"])


# ── Sector-relative short-term return ──────────────────────────────────────

@register_factor(
    name="sector_relative_ret_5",
    description="行业内相对短期收益因子，个股5日收益率/行业内中位数收益率截面排名。",
    category="sector",
    thesis="行业内短期相对强度捕捉的是个股在行业轮动中的alpha——剔除了行业整体走势后的个股短期动量信号更纯净，与sector_relative_momentum_20互补覆盖短中期。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_sector_relative_ret_5(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    close = daily_adj["close"]
    ret_5 = close.groupby(level="Code").transform(lambda s: s.pct_change(5))
    codes = ret_5.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"ret_5": ret_5.values, "industry": industries.values}, index=ret_5.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["ret_5"].transform("median")
    df["relative"] = df["ret_5"] - df["sector_median"]
    return cross_sectional_rank(df["relative"])


# ── Supplementary sector factors ───────────────────────────────────────────


@register_factor(
    name="sector_breadth_20",
    description="行业内20日趋势广度因子 (行业内上涨股票占比)。",
    category="sector",
    thesis="行业广度衡量行业内多少股票在上涨——高广度意味着行业全面上行，非仅龙头拉动",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_sector_breadth_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    close = daily_adj["close"]
    ret_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    codes = ret_20.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"ret_20": ret_20.values, "industry": industries.values}, index=ret_20.index)
    df = df.dropna(subset=["industry"])
    df["is_up"] = (df["ret_20"] > 0).astype(float)
    df["breadth"] = df.groupby(["Date", "industry"])["is_up"].transform("mean")
    return cross_sectional_rank(df["breadth"])


@register_factor(
    name="sector_concentration_risk",
    description="行业集中度风险因子 (行业内市值HHI, 高集中排后, 负向)。",
    category="sector",
    thesis="行业内市值过于集中在少数龙头意味着尾部风险——龙头一旦调整板块无支撑",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_sector_concentration_risk(context: FactorContext):
    finance = context.load("finance.parquet")
    industry_map = context.repo.load_industry_map()
    mv = finance["total_mv"]
    codes = mv.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"mv": mv.values, "industry": industries.values}, index=mv.index)
    df = df.dropna(subset=["industry"])
    df["mv_pct"] = df.groupby(["Date", "industry"])["mv"].transform(
        lambda x: x / x.sum()
    )
    df["hhi"] = df.groupby(["Date", "industry"])["mv_pct"].transform(
        lambda x: (x ** 2).sum()
    )
    return cross_sectional_rank(-df["hhi"])


@register_factor(
    name="sector_turnover_ratio_5d",
    description="行业内相对换手率5日变化因子 (换手率提升排前)。",
    category="sector",
    thesis="行业内换手率的边际提升反映关注度上升，换手率提升意味着流动性改善",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_sector_turnover_ratio_5d(context: FactorContext):
    finance = context.load("finance.parquet")
    industry_map = context.repo.load_industry_map()
    turnover = finance["turnover_rate"]
    codes = turnover.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"to": turnover.values, "industry": industries.values}, index=turnover.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["to"].transform("median")
    df["relative"] = df["to"] / df["sector_median"].replace(0, np.nan)
    df["change"] = df.groupby("Code")["relative"].transform(lambda s: s.diff(5))
    return cross_sectional_rank(df["change"])


# ── Sector momentum dispersion ──────────────────────────────────────────

@register_factor(
    name="sector_momentum_dispersion",
    description="行业内动量离散度因子，-(行业内个股收益率20日标准差)截面排名（离散度低=行业协同上涨排前）。",
    category="sector",
    thesis="行业内收益率离散度反映上涨的结构质量——离散度低意味着行业全面上涨(龙头+跟风一起涨)、趋势性强；离散度高意味着仅少数个股上涨、行业动能弱。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_sector_momentum_dispersion(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    close = daily_adj["close"]
    ret_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    codes = ret_20.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"ret_20": ret_20.values, "industry": industries.values}, index=ret_20.index)
    df = df.dropna(subset=["industry"])
    df["dispersion"] = df.groupby(["Date", "industry"])["ret_20"].transform("std")
    return cross_sectional_rank(-df["dispersion"])


# ── Sector leader gap ───────────────────────────────────────────────────

@register_factor(
    name="sector_leader_gap",
    description="行业龙头差距因子，(个股20日收益-行业内最大20日收益)截面排名。",
    category="sector",
    thesis="个股相对行业内最强者的差距——差距小意味着接近龙头表现、可能成为新龙头；差距大意味着落后板块较多，需要更强催化剂才能修复。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_sector_leader_gap(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    close = daily_adj["close"]
    ret_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    codes = ret_20.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"ret_20": ret_20.values, "industry": industries.values}, index=ret_20.index)
    df = df.dropna(subset=["industry"])
    df["sector_max"] = df.groupby(["Date", "industry"])["ret_20"].transform("max")
    df["gap"] = df["ret_20"] - df["sector_max"]
    return cross_sectional_rank(df["gap"])


# ── Sector relative volume ratio ────────────────────────────────────────

@register_factor(
    name="sector_relative_volume_ratio",
    description="行业内相对量比因子，个股成交量/行业内中位数成交量截面排名。",
    category="sector",
    thesis="行业内成交量的相对位置反映资金关注度——行业内成交量占比提升意味着资金从同行转向该股、关注度的边际变化。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_sector_relative_volume_ratio(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    vol = daily_adj["vol"]
    codes = vol.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"vol": vol.values, "industry": industries.values}, index=vol.index)
    df = df.dropna(subset=["industry"])
    df["sector_median_vol"] = df.groupby(["Date", "industry"])["vol"].transform("median")
    df["relative_vol"] = df["vol"] / df["sector_median_vol"].replace(0, np.nan)
    return cross_sectional_rank(df["relative_vol"])


# ── Sector size leadership ──────────────────────────────────────────────

@register_factor(
    name="sector_size_leadership",
    description="行业规模领先因子，行业内总市值排名(百分位)截面排名。",
    category="sector",
    thesis="行业内市值排名反映公司的行业地位——大市值龙头享有定价权和规模效应，在行业景气上行时弹性最大。",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_sector_size_leadership(context: FactorContext):
    finance = context.load("finance.parquet")
    industry_map = context.repo.load_industry_map()
    mv = finance["total_mv"]
    codes = mv.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"mv": mv.values, "industry": industries.values}, index=mv.index)
    df = df.dropna(subset=["industry"])
    df["sector_mv_rank"] = df.groupby(["Date", "industry"])["mv"].rank(pct=True)
    return cross_sectional_rank(df["sector_mv_rank"])


# ── Sector EPS leadership ───────────────────────────────────────────────

@register_factor(
    name="sector_eps_leadership",
    description="行业盈利领先因子，行业内ROE排名(百分位)截面排名。",
    category="sector",
    thesis="行业内ROE排名反映公司的相对盈利能力——ROE在行业内领先意味着公司在产业链中占据最有利的位置。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_sector_eps_leadership(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["roe"])
    industry_map = context.repo.load_industry_map()
    roe = fin["roe"]
    codes = roe.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"roe": roe.values, "industry": industries.values}, index=roe.index)
    df = df.dropna(subset=["industry"])
    df["sector_roe_rank"] = df.groupby(["Date", "industry"])["roe"].rank(pct=True)
    return cross_sectional_rank(df["sector_roe_rank"])


# ── Sector composite rank ───────────────────────────────────────────────

@register_factor(
    name="sector_composite_rank",
    description="行业综合排名因子，(市值排名+ROE排名+成交额排名)/3截面排名（行业龙头综合排前）。",
    category="sector",
    thesis="行业内市占率+盈利能力+成交活跃度的综合排名——三维度综合评定的行业龙头地位，比单维度更全面。",
    dependencies=("finance.parquet", "daily_adj.parquet", "financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_sector_composite_rank(context: FactorContext):
    finance = context.load("finance.parquet")
    daily_adj = context.load("daily_adj.parquet")
    fin = context.load_financial("financial_indicator.parquet", value_cols=["roe"])
    industry_map = context.repo.load_industry_map()

    mv = finance["total_mv"]
    amount = daily_adj["amount"]
    roe = fin["roe"]

    codes_mv = mv.index.get_level_values("Code")
    codes_amt = amount.index.get_level_values("Code")
    codes_roe = roe.index.get_level_values("Code")

    industries_mv = codes_mv.map(industry_map)
    industries_amt = codes_amt.map(industry_map)
    industries_roe = codes_roe.map(industry_map)

    df_mv = pd.DataFrame({"mv": mv.values, "industry": industries_mv.values}, index=mv.index).dropna(subset=["industry"])
    df_amt = pd.DataFrame({"amount": amount.values, "industry": industries_amt.values}, index=amount.index).dropna(subset=["industry"])
    df_roe = pd.DataFrame({"roe": roe.values, "industry": industries_roe.values}, index=roe.index).dropna(subset=["industry"])

    df_mv["mv_rank"] = df_mv.groupby(["Date", "industry"])["mv"].rank(pct=True)
    df_amt["amt_rank"] = df_amt.groupby(["Date", "industry"])["amount"].rank(pct=True)
    df_roe["roe_rank"] = df_roe.groupby(["Date", "industry"])["roe"].rank(pct=True)

    common = df_mv.index.intersection(df_amt.index).intersection(df_roe.index)
    composite = (df_mv.loc[common, "mv_rank"] + df_amt.loc[common, "amt_rank"] + df_roe.loc[common, "roe_rank"]) / 3.0
    return cross_sectional_rank(composite)
