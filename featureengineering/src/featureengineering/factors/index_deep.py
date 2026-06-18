"""
Extended index factors (扩展指数因子) — Class 1.

These factors extend the 8 basic index factors in ``index_weight.py`` with
style exposures, rebalance anticipation, passive flow estimation, benchmark-
relative metrics, market regime sensitivity, and index overlap analysis.

Data source: ``index_weight.parquet`` (monthly, 2019-01 onward, 8 major indices).
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


def _monthly_to_daily(series: pd.Series, context: FactorContext,
                      cap_date: str | None = None) -> pd.Series:
    """Forward-fill a monthly (Date, Code) series to daily using trading calendar."""
    calendar = context.repo._read_parquet(
        context.repo.paths.source_root / "calendar.parquet"
    )
    dates = series.index.get_level_values("Date")
    codes = series.index.get_level_values("Code")
    wide = pd.DataFrame({"Date": dates, "Code": codes, "val": series.values})
    all_codes = sorted(wide["Code"].unique())
    wide = wide.pivot(index="Date", columns="Code", values="val")
    wide = wide.reindex(columns=all_codes)

    if "is_open" in calendar.columns:
        trading = calendar.loc[calendar["is_open"].astype(bool)]
    else:
        trading = calendar
    calendar_dates = trading["date"].astype(str).str.replace("-", "").str.slice(0, 8)
    wide.index = wide.index.astype(str)
    wide = wide.reindex(calendar_dates).ffill()

    if cap_date is not None:
        wide = wide[wide.index <= cap_date]

    stacked = wide.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()

    allowed = context.repo.allowed_codes
    if allowed:
        codes_mask = stacked.index.get_level_values("Code").isin(allowed)
        stacked = stacked.loc[codes_mask]
    return stacked


def _load_index_weight_raw(context: FactorContext) -> pd.DataFrame:
    """Load index_weight.parquet with normalized codes and dates."""
    cache = getattr(_load_index_weight_raw, "_cache", None)
    if cache is not None:
        return cache

    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "index_weight.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)

    _load_index_weight_raw._cache = raw
    return raw


def _build_index_weight_panel(context: FactorContext,
                               index_code: str | None = None) -> pd.DataFrame:
    """Build a (Date × stock_code) panel of index weights.

    If index_code is None, returns total weight across all indices.
    """
    raw = _load_index_weight_raw(context)

    if index_code is not None:
        raw = raw[raw["index_code"] == index_code]

    panel = raw.pivot_table(
        index="trade_date", columns="stock_code", values="weight", aggfunc="sum"
    )
    panel.index.name = "Date"
    panel.columns.name = "Code"
    return panel.sort_index()


def _get_index_weight_series(context: FactorContext,
                              index_code: str | None = None) -> pd.Series:
    """Get a (Date, Code) MultiIndex Series of index weights, daily forward-filled."""
    panel = _build_index_weight_panel(context, index_code)
    stacked = panel.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()

    daily = _monthly_to_daily(stacked, context, cap_date=context.end_date)
    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# A — Composite & Risk Factors
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="index_weight_rank_composite",
    description="多指数综合权重排名因子，个股在沪深300+中证500+上证50+创业板指中的综合权重截面排名。",
    category="index",
    thesis="多指数综合权重反映个股在被动投资体系中的总重要性——被多个核心指数纳入且权重高的股票享受最大的被动资金配置规模，是'被动投资红利'的最直接受益者。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_rank_composite(context: FactorContext):
    core_indices = ["000300.SH", "000905.SH", "000016.SH", "399006.SZ"]
    raw = _load_index_weight_raw(context)
    core_data = raw[raw["index_code"].isin(core_indices)]

    composite = core_data.groupby(["trade_date", "stock_code"])["weight"].sum()
    composite.index = composite.index.set_names(["Date", "Code"])
    composite = composite.reorder_levels(["Date", "Code"]).sort_index()

    daily = _monthly_to_daily(composite, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


@register_factor(
    name="index_weight_volatility_12m",
    description="指数权重波动率因子，个股沪深300权重12个月标准差截面排名（取负向=权重不稳排后）。",
    category="index",
    thesis="指数权重的稳定性反映个股在指数中的'粘性'——权重波动大的股票可能处于调入/调出的边缘，被动资金配置的不确定性高，应给予风险折扣。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_volatility_12m(context: FactorContext):
    panel = _build_index_weight_panel(context, "000300.SH")
    vol = panel.rolling(12, min_periods=6).std()

    stacked = vol.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()

    daily = _monthly_to_daily(stacked, context, cap_date=context.end_date)
    return cross_sectional_rank(-daily)


@register_factor(
    name="index_weight_seasonality",
    description="指数调仓季节性因子，3月/9月调仓月份权重变化的绝对值截面排名（调仓敏感度）。",
    category="index",
    thesis="A股主要指数在每年6月和12月进行定期调仓（提前2周公布）。权重在调仓月份变化大的股票面临更大的被动交易冲击——提前布局调仓方向可以获取'指数效应'超额收益。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_seasonality(context: FactorContext):
    raw = _load_index_weight_raw(context)
    hs300 = raw[raw["index_code"] == "000300.SH"].copy()
    hs300 = hs300.set_index(["trade_date", "stock_code"])["weight"]
    hs300.index = hs300.index.set_names(["Date", "Code"])
    hs300 = hs300.reorder_levels(["Date", "Code"]).sort_index()

    # Monthly weight change
    chg = hs300.groupby(level="Code").diff(1).abs()

    # Flag rebalance months (June=06, December=12) and nearby months
    date_strs = chg.index.get_level_values("Date")
    month = date_strs.str[4:6]
    is_rebalance = month.isin(["05", "06", "11", "12"])  # announcement + effective

    seasonality = chg * is_rebalance.astype(float)
    # Rolling 12-month average of rebalance-month changes
    seasonality_avg = seasonality.groupby(level="Code").transform(
        lambda s: s.rolling(12, min_periods=1).mean()
    )

    daily = _monthly_to_daily(seasonality_avg, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


# ═══════════════════════════════════════════════════════════════════════════════
# B — Inclusion/Exclusion Probability
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="index_inclusion_probability",
    description="纳入概率因子，基于市值排名+流动性排名的沪深300纳入概率截面排名。",
    category="index",
    thesis="处于沪深300边缘（中证500头部）的股票有较高的纳入预期——'纳入预期'本身就能带来增量买盘，因为主动基金会在正式纳入前提前布局。基于市值300-400名+日均成交额前500名的股票纳入概率最高。",
    dependencies=("finance.parquet", "daily_adj.parquet"),
)
def factor_index_inclusion_probability(context: FactorContext):
    finance = context.load("finance.parquet")
    daily_adj = context.load("daily_adj.parquet")

    mv = finance["total_mv"]
    amount = daily_adj["amount"]

    # Market cap rank (cross-sectional)
    mv_rank = mv.groupby(level="Date").rank(pct=True)
    # Liquidity rank
    amt_rank = amount.groupby(level="Date").rank(pct=True)

    # Inclusion probability: high for stocks ranked 200-400 by market cap
    # with good liquidity (highest around pct=0.92 = rank ~400 in 5000 stocks)
    mv_proximity = 1.0 - np.abs(mv_rank - 0.92) * 5  # peak at 92nd percentile
    mv_proximity = mv_proximity.clip(lower=0)

    inclusion_prob = mv_proximity * amt_rank  # liquidity bonus
    return cross_sectional_rank(inclusion_prob)


@register_factor(
    name="index_exclusion_risk",
    description="剔除风险因子，-(沪深300中权重最低的20%股票的权重排名)截面排名（边缘股=剔除风险高排后）。",
    category="index",
    thesis="指数中权重最低的股票面临被剔除的风险——剔除意味着被动基金将在调仓日卖出，带来负面价格冲击。持有剔除风险高的股票需要额外的风险补偿。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_exclusion_risk(context: FactorContext):
    panel = _build_index_weight_panel(context, "000300.SH")

    # Weight percentile within HS300 constituents (lower = closer to exclusion)
    in_index = panel > 0
    weight_rank = panel.where(in_index).rank(axis=1, pct=True)

    # Exclusion risk = 1 - weight_rank (high for bottom weights)
    risk = 1.0 - weight_rank
    risk = risk.where(in_index, 0.5)  # not in index = medium risk

    stacked = risk.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()

    daily = _monthly_to_daily(stacked, context, cap_date=context.end_date)
    return cross_sectional_rank(-daily)


# ═══════════════════════════════════════════════════════════════════════════════
# C — Style Exposure Factors
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="index_float_adj_weight",
    description="自由流通权重因子，个股自由流通市值排名/总市值排名截面排名（高自由流通=指数权重上限高排前）。",
    category="index",
    thesis="指数的权重计算基于自由流通市值——自由流通占比高的股票在同等总市值下能获得更高的指数权重。在指数化投资时代，自由流通占比是影响被动资金配置的关键变量。",
    dependencies=("finance.parquet",),
)
def factor_index_float_adj_weight(context: FactorContext):
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"]
    circ_mv = finance["circ_mv"]

    float_ratio = circ_mv / total_mv.replace(0, np.nan)
    float_ratio = float_ratio.clip(0, 1)

    return cross_sectional_rank(float_ratio)


@register_factor(
    name="index_style_exposure_value",
    description="价值风格暴露因子，基于BP排名估算价值指数（如中证价值）的隐含权重截面排名。",
    category="index",
    thesis="股票在价值/成长风格维度上的暴露决定了其在风格指数中的权重——高BP股票在价值指数中权重更高，当价值风格占优时受益更多。风格暴露是理解因子周期性的关键。",
    dependencies=("finance.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_index_style_exposure_value(context: FactorContext):
    finance = context.load("finance.parquet")
    pb = finance["pb"].where(finance["pb"] > 0, np.nan)
    bp = 1.0 / pb  # Book-to-price

    return cross_sectional_rank(bp)


@register_factor(
    name="index_style_exposure_growth",
    description="成长风格暴露因子，基于营收增速+ROE排名的成长指数隐含权重截面排名。",
    category="index",
    thesis="营收增速和ROE是成长指数选股的核心指标——高增速+高ROE的股票在成长指数中权重更高。成长风格暴露高的股票在成长风格占优周期中弹性最大。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_index_style_exposure_growth(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
                                  value_cols=["or_yoy", "roe"])
    rev_growth = fin["or_yoy"]
    roe = fin["roe"]

    rev_rank = rev_growth.groupby(level="Date").rank(pct=True)
    roe_rank = roe.groupby(level="Date").rank(pct=True)

    growth_exposure = (rev_rank + roe_rank) / 2.0
    return cross_sectional_rank(growth_exposure)


@register_factor(
    name="index_style_exposure_size",
    description="规模风格暴露因子，基于总市值排名的规模指数隐含权重截面排名（大市值=大盘风格排前）。",
    category="index",
    thesis="市值是风格分类中最稳定的维度——大市值股票在大盘指数中权重更高。市值风格暴露决定了股票在不同市场环境下的Beta特征。",
    dependencies=("finance.parquet",),
)
def factor_index_style_exposure_size(context: FactorContext):
    finance = context.load("finance.parquet")
    mv = finance["total_mv"]
    return cross_sectional_rank(mv)


# ═══════════════════════════════════════════════════════════════════════════════
# D — Weight Distribution & Gini
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="index_weight_gini",
    description="指数权重Gini系数因子，-(个股跨8个指数的权重Gini不均度)截面排名（权重均分=分散排前,高Gini=依赖单一指数排后）。",
    category="index",
    thesis="跨指数权重的Gini系数衡量个股被动资金来源的集中度——高Gini意味着严重依赖单一指数，该指数调仓时冲击大。低Gini（权重分散在多个指数中）则被动资金更稳定。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_gini(context: FactorContext):
    raw = _load_index_weight_raw(context)

    # For each (date, stock), compute Gini of weights across indices
    def _gini(x):
        x = x[x > 0]
        if len(x) < 2:
            return 0.0
        x_sorted = np.sort(x.values)
        n = len(x_sorted)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x_sorted) - (n + 1) * np.sum(x_sorted)) / (n * np.sum(x_sorted))

    gini = raw.groupby(["trade_date", "stock_code"])["weight"].apply(_gini)
    gini.index = gini.index.set_names(["Date", "Code"])
    gini = gini.reorder_levels(["Date", "Code"]).sort_index()

    daily = _monthly_to_daily(gini, context, cap_date=context.end_date)
    return cross_sectional_rank(-daily)


# ═══════════════════════════════════════════════════════════════════════════════
# E — Tracking & Rebalance
# ═══════════════════════════════════════════════════════════════════════════════





@register_factor(
    name="index_rebalance_anticipation",
    description="调仓预期因子，(距下次调仓月数×权重边际变化方向)截面排名（买入预期=正权重变化×临近调仓排前）。",
    category="index",
    thesis="指数调仓日附近的交易机会是A股中少数具有确定性的alpha来源——提前2-4周布局预计被调入的股票（权重从0变正）或增持权重提升的股票，可以在被动基金调仓日获得流动性溢价。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_rebalance_anticipation(context: FactorContext):
    raw = _load_index_weight_raw(context)
    hs300 = raw[raw["index_code"] == "000300.SH"].copy()

    hs300 = hs300.set_index(["trade_date", "stock_code"])["weight"]
    hs300.index = hs300.index.set_names(["Date", "Code"])
    hs300 = hs300.reorder_levels(["Date", "Code"]).sort_index()

    # Weight change direction
    chg = hs300.groupby(level="Code").diff(1)

    # Distance to next rebalance (June=06, December=12)
    date_strs = pd.Index(chg.index.get_level_values("Date"))
    months = pd.to_numeric(date_strs.str[4:6], errors="coerce")
    # Months until next rebalance: rebalance happens in month 6 and 12
    dist_to_jun = (6 - months) % 12
    dist_to_dec = (12 - months) % 12
    dist = pd.Series(np.minimum(dist_to_jun, dist_to_dec), index=chg.index)
    # Closer to rebalance = higher score (inverse distance)
    proximity = 1.0 / (dist.clip(lower=1).astype(float))

    anticipation = chg * proximity

    daily = _monthly_to_daily(anticipation, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


@register_factor(
    name="index_passive_flow_estimate",
    description="被动资金流估算因子，沪深300权重×估算AUM(约2000亿)截面排名（高权重=高被动流入排前）。",
    category="index",
    thesis="跟踪沪深300的指数基金AUM合计约2000-3000亿元——个股的指数权重直接乘以AUM就是被动基金需要配置的金额。这是最直接的'被动资金流'量化指标。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_passive_flow_estimate(context: FactorContext):
    weight = _get_index_weight_series(context, "000300.SH")

    # Estimated total AUM tracking HS300: ~2000亿 (rough order of magnitude)
    # The cross-sectional rank is invariant to the exact AUM number
    estimated_flow = weight * 2000  # in 亿 RMB

    return cross_sectional_rank(estimated_flow)


@register_factor(
    name="index_weight_drift",
    description="权重漂移因子，个股累计收益-沪深300累计收益截面排名（价格跑赢=实际权重>目标权重排前）。",
    category="index",
    thesis="在两次调仓之间，股票价格的变化导致实际权重偏离目标权重——涨幅超过指数的股票实际权重被动上升（正漂移），在下次调仓时面临卖出压力。负漂移（涨少了）的股票则有被动增持预期。",
    dependencies=("daily_adj.parquet",),
)
def factor_index_weight_drift(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    cum_ret = ret.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).sum())

    market_ret = ret.groupby(level="Date").mean()
    market_cum = market_ret.rolling(60, min_periods=30).sum()

    codes = cum_ret.index.get_level_values("Code")
    dates = cum_ret.index.get_level_values("Date")
    market_mapped = pd.Series(market_cum.loc[dates].values, index=cum_ret.index)

    drift = cum_ret - market_mapped
    # Negative drift = stock lagged market = more likely to be bought at rebalance
    return cross_sectional_rank(-drift)


# ═══════════════════════════════════════════════════════════════════════════════
# F — Market Participation & Benchmark Relative
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="market_participation_20",
    description="市场参与率因子，过去20日个股与全A等权同涨同跌的比例截面排名（高参与=跟随市场排前）。",
    category="index",
    thesis="市场上涨日参与率衡量个股的'跟涨'能力——市场涨时它也涨的比例。高参与率意味着股票的系统性Beta有效，不会出现'大盘涨它不涨'的尴尬。低参与率的股票可能有独立的风险因素。",
    dependencies=("daily_adj.parquet",),
)
def factor_market_participation_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    market_ret = ret.groupby(level="Date").mean()
    market_up = (market_ret > 0).astype(float)

    ret_wide = ret.unstack("Code")

    participation = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns)
    for code in ret_wide.columns:
        stock_up = (ret_wide[code] > 0).astype(float)
        same_dir = (stock_up == market_up).astype(float)
        participation[code] = same_dir.rolling(20, min_periods=10).mean()

    stacked = participation.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(stacked)


@register_factor(
    name="market_cap_tier_rank",
    description="市值分层排名因子，个股在市值分层（大盘/中盘/小盘）内的截面排名。",
    category="index",
    thesis="同一市值层级内的排名比全市场排名更能反映个股在其'竞争组'中的相对地位——大盘股内部排前10%和中盘股内部排前10%的含义完全不同。分层排名消除了市值因子的干扰。",
    dependencies=("finance.parquet",),
)
def factor_market_cap_tier_rank(context: FactorContext):
    finance = context.load("finance.parquet")
    mv = finance["total_mv"]

    # Split into terciles by market cap within each date
    def _tier_rank(group):
        if len(group) < 10:
            return pd.Series(0.5, index=group.index)
        tercile = pd.qcut(group, 3, labels=False, duplicates="drop")
        result = pd.Series(np.nan, index=group.index)
        for t in range(3):
            mask = tercile == t
            if mask.sum() > 1:
                result[mask] = group[mask].rank(pct=True)
        return result

    tier_rank = mv.groupby(level="Date").transform(_tier_rank)
    return cross_sectional_rank(tier_rank)


@register_factor(
    name="benchmark_relative_return_20",
    description="基准相对收益因子，个股20日收益-沪深300 20日收益截面排名（跑赢基准排前）。",
    category="index",
    thesis="相对沪深300的超额收益是机构考核的核心指标——持续跑赢基准的股票是主动基金超额收益的来源，也是量化多头的核心alpha标的。",
    dependencies=("daily_adj.parquet",),
)
def factor_benchmark_relative_return_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    cum_20 = ret.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).sum())

    # Use equal-weighted market as broad benchmark (HS300 proxy not available daily)
    market_cum_20 = ret.groupby(level="Date").mean().rolling(20, min_periods=10).sum()

    codes = cum_20.index.get_level_values("Code")
    dates = cum_20.index.get_level_values("Date")
    market_mapped = pd.Series(market_cum_20.loc[dates].values, index=cum_20.index)

    relative = cum_20 - market_mapped
    return cross_sectional_rank(relative)


@register_factor(
    name="all_a_relative_strength_20",
    description="全A相对强度因子，个股20日收益-全A等权20日收益截面排名。",
    category="index",
    thesis="相对全A等权指数的超额收益是'选股Alpha'的最纯粹度量——剔除了市场Beta后，剩下的就是选股能力。全A等权指数比市值加权指数更能代表'平均股票'的表现。",
    dependencies=("daily_adj.parquet",),
)
def factor_all_a_relative_strength_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    cum_20 = ret.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).sum())

    all_a_cum = ret.groupby(level="Date").mean().rolling(20, min_periods=10).sum()

    codes = cum_20.index.get_level_values("Code")
    dates = cum_20.index.get_level_values("Date")
    all_a_mapped = pd.Series(all_a_cum.loc[dates].values, index=cum_20.index)

    rs = cum_20 - all_a_mapped
    return cross_sectional_rank(rs)


@register_factor(
    name="board_relative_strength_20",
    description="板块相对强度因子，个股20日收益-所属板块（主板/创业板/科创板）等权收益截面排名。",
    category="index",
    thesis="不同板块（主板、创业板、科创板）有不同的投资者结构和流动性特征——相对所属板块的超额收益剔除了板块效应，更纯粹地反映个股在同类标的中的相对优势。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_board_relative_strength_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    cum_20 = ret.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).sum())

    # Determine board from stock code prefix
    codes = cum_20.index.get_level_values("Code")
    board = pd.Series("other", index=codes)
    board[codes.str.startswith("60")] = "main_sh"
    board[codes.str.startswith("00")] = "main_sz"
    board[codes.str.startswith("30")] = "chinext"
    board[codes.str.startswith("68")] = "star"

    df = pd.DataFrame({"cum_20": cum_20.values, "board": board.values}, index=cum_20.index)
    df["board_avg"] = df.groupby(["Date", "board"])["cum_20"].transform("mean")
    df["relative"] = df["cum_20"] - df["board_avg"]

    return cross_sectional_rank(df["relative"])


# ═══════════════════════════════════════════════════════════════════════════════
# G — Style Tilt & Market Regime
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="style_tilt_momentum",
    description="风格动量因子，(价值暴露排名变化+成长暴露排名变化)/2截面排名（风格强化排前）。",
    category="index",
    thesis="风格暴露的变化方向反映市场对股票的风格重新定价——价值暴露在上升的股票正在被市场重估为价值股，应跟随其风格变化方向做多。风格动量是因子动量在风格维度的体现。",
    dependencies=("finance.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_style_tilt_momentum(context: FactorContext):
    finance = context.load("finance.parquet")
    pb = finance["pb"].where(finance["pb"] > 0, np.nan)
    bp = 1.0 / pb
    bp_rank = bp.groupby(level="Date").rank(pct=True)
    bp_rank_chg = bp_rank.groupby(level="Code").diff(20)

    fin = context.load_financial("financial_indicator.parquet", value_cols=["or_yoy", "roe"])
    growth = fin["or_yoy"].groupby(level="Date").rank(pct=True) + fin["roe"].groupby(level="Date").rank(pct=True)
    growth_chg = growth.groupby(level="Code").diff(20)

    style_mom = (bp_rank_chg.fillna(0) + growth_chg.fillna(0)) / 2.0
    return cross_sectional_rank(style_mom)


@register_factor(
    name="market_regime_sensitivity",
    description="市场状态敏感度因子，个股在上涨市Beta/下跌市Beta的比值截面排名（上涨弹性>下跌弹性排前）。",
    category="index",
    thesis="非对称Beta（上涨Beta vs 下跌Beta的差异）反映了股票的风险回报非对称性——上涨Beta高+下跌Beta低的股票是理想的'涨多跌少'标的。A股中此类股票往往具有品牌壁垒或定价权优势。",
    dependencies=("daily_adj.parquet",),
)
def factor_market_regime_sensitivity(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    market_ret = ret.groupby(level="Date").mean()
    is_up = market_ret > 0
    is_down = market_ret < 0

    ret_wide = ret.unstack("Code")

    up_beta = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns)
    down_beta = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns)

    for code in ret_wide.columns:
        stock_r = ret_wide[code]
        # Up market beta: only use up-market days
        up_mask = is_up & stock_r.notna()
        if up_mask.sum() > 30:
            market_up = market_ret[up_mask]
            stock_up = stock_r[up_mask]
            up_cov = stock_up.rolling(60, min_periods=30).cov(market_up)
            up_var = market_up.rolling(60, min_periods=30).var()
            up_beta[code] = up_cov / up_var.replace(0, np.nan)

        # Down market beta
        down_mask = is_down & stock_r.notna()
        if down_mask.sum() > 30:
            market_down = market_ret[down_mask]
            stock_down = stock_r[down_mask]
            down_cov = stock_down.rolling(60, min_periods=30).cov(market_down)
            down_var = market_down.rolling(60, min_periods=30).var()
            down_beta[code] = down_cov / down_var.replace(0, np.nan)

    # Sensitivity = up_beta / down_beta (higher = better asymmetry)
    sensitivity = up_beta / down_beta.replace(0, np.nan)

    stacked = sensitivity.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(stacked)


# ═══════════════════════════════════════════════════════════════════════════════
# H — Index Arbitrage & Overlap
# ═══════════════════════════════════════════════════════════════════════════════


@register_factor(
    name="index_overlap_premium",
    description="多指数覆盖溢价因子，个股被几个核心指数同时纳入的截面排名（多数纳入=稳定被动资金排前）。",
    category="index",
    thesis="被多个核心指数同时纳入的股票享受'指数重叠效应'——不同的指数基金都需要配置该股，形成多层次的被动买盘。多指数覆盖的股票流动性溢价更高，估值也系统性偏高。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_overlap_premium(context: FactorContext):
    raw = _load_index_weight_raw(context)
    core_indices = ["000001.SH", "000016.SH", "000300.SH", "000852.SH",
                    "000905.SH", "399001.SZ", "399006.SZ"]

    core_data = raw[raw["index_code"].isin(core_indices)]
    has_weight = (core_data["weight"] > 0).astype(float)
    # Group by the original DataFrame columns (has_weight is a Series)
    count = has_weight.groupby([core_data["trade_date"], core_data["stock_code"]]).sum()
    count.index = count.index.set_names(["Date", "Code"])
    count = count.reorder_levels(["Date", "Code"]).sort_index()

    daily = _monthly_to_daily(count, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


@register_factor(
    name="index_weight_momentum_3m",
    description="指数权重动量因子，沪深300权重3个月变化截面排名（权重提升=被动增配排前）。",
    category="index",
    thesis="权重在3个月窗口内的边际变化是最稳定的被动资金流指标——3个月足够平滑单月噪音，同时捕捉调仓趋势。权重持续提升的股票是'被动投资牛市'中最确定的受益者。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_momentum_3m(context: FactorContext):
    panel = _build_index_weight_panel(context, "000300.SH")

    # 3-month weight change
    chg = panel.diff(3)

    stacked = chg.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()

    daily = _monthly_to_daily(stacked, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


@register_factor(
    name="index_weight_acceleration",
    description="指数权重加速度因子，沪深300权重3月变化-权重6月变化/2截面排名（二阶导=加速纳入排前）。",
    category="index",
    thesis="权重变化的加速度（二阶导）比一阶导更早发现纳入趋势——权重从'缓慢增加'变为'快速增加'意味着指数基金正在加速配置该股，可能是被纳入新指数的前兆。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_acceleration(context: FactorContext):
    panel = _build_index_weight_panel(context, "000300.SH")

    chg_3m = panel.diff(3)
    chg_6m = panel.diff(6)

    # Acceleration = recent change - longer-term average change
    accel = chg_3m - chg_6m / 2.0

    stacked = accel.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()

    daily = _monthly_to_daily(stacked, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


@register_factor(
    name="index_sector_neutral_weight",
    description="行业中性权重因子，沪深300权重/行业平均权重截面排名（行业内高权重=相对超配排前）。",
    category="index",
    thesis="行业中性化后的权重反映了股票在行业内的'指数代表性'——同一行业内，指数权重更高的股票是被动基金在该行业中配置最重的标的。行业中性权重剔除了行业间权重差异。",
    dependencies=("index_weight.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_index_sector_neutral_weight(context: FactorContext):
    weight = _get_index_weight_series(context, "000300.SH")
    industry_map = context.repo.load_industry_map()

    codes = weight.index.get_level_values("Code")
    industries = codes.map(industry_map)

    df = pd.DataFrame({"weight": weight.values, "industry": industries.values}, index=weight.index)
    df = df.dropna(subset=["industry"])
    df["industry_avg"] = df.groupby(["Date", "industry"])["weight"].transform("mean")
    df["neutral_weight"] = df["weight"] / df["industry_avg"].replace(0, np.nan)

    return cross_sectional_rank(df["neutral_weight"])


@register_factor(
    name="index_passive_demand_pressure",
    description="被动需求压力因子，权重×调仓临近度截面排名（高权重+临近调仓=买入压力大排前）。",
    category="index",
    thesis="被动需求压力综合了权重大小和时间紧迫度两个维度——权重高且离调仓日近的股票，被动基金必须在短期内完成配置，对价格的推升作用最显著。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_passive_demand_pressure(context: FactorContext):
    weight = _get_index_weight_series(context, "000300.SH")

    # Proximity to rebalance
    date_strs = weight.index.get_level_values("Date")
    months = pd.to_numeric(date_strs.str[4:6], errors="coerce")
    dist_to_jun = (6 - months) % 12
    dist_to_dec = (12 - months) % 12
    dist = pd.Series(np.minimum(dist_to_jun, dist_to_dec), index=weight.index)
    proximity = 1.0 / dist.clip(lower=1).astype(float)

    pressure = weight * proximity
    return cross_sectional_rank(pressure)


@register_factor(
    name="index_style_rotation_20",
    description="风格轮动信号因子，(价值暴露排名20日变化-成长暴露排名20日变化)截面排名（正=转向价值排前）。",
    category="index",
    thesis="风格暴露排名的相对变化捕捉风格轮动——价值排名上升+成长排名下降意味着市场正在重新定价该股为价值股。风格轮动信号可以帮助识别因子风格的切换时点。",
    dependencies=("finance.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_index_style_rotation_20(context: FactorContext):
    finance = context.load("finance.parquet")
    pb = finance["pb"].where(finance["pb"] > 0, np.nan)
    bp = 1.0 / pb
    value_rank = bp.groupby(level="Date").rank(pct=True)
    value_chg = value_rank.groupby(level="Code").diff(20)

    fin = context.load_financial("financial_indicator.parquet", value_cols=["or_yoy", "roe"])
    growth_rank = fin["or_yoy"].groupby(level="Date").rank(pct=True)
    growth_chg = growth_rank.groupby(level="Code").diff(20)

    rotation = value_chg.fillna(0) - growth_chg.fillna(0)
    return cross_sectional_rank(rotation)


@register_factor(
    name="index_marginal_contribution",
    description="指数边际贡献因子，(个股总市值×1%权重)/指数总市值截面排名（每1%指数权重的资金增量）。",
    category="index",
    thesis="每1%的指数权重对应多少增量资金取决于指数总AUM和个股自身的市值——小市值股票获得1%指数权重时，增量资金占其市值的比例更大，价格冲击更显著。",
    dependencies=("index_weight.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_index_marginal_contribution(context: FactorContext):
    weight = _get_index_weight_series(context, "000300.SH")

    finance = context.load("finance.parquet")
    mv = finance["total_mv"]

    # Align
    common_idx = weight.index.intersection(mv.index)
    w_aligned = weight.loc[common_idx]
    mv_aligned = mv.loc[common_idx]

    # Marginal contribution: if this stock gets +1% weight in HS300,
    # passive inflow / its market cap = proportional impact
    marginal = (w_aligned * 2000) / mv_aligned.replace(0, np.nan)  # 2000亿 AUM estimate

    return cross_sectional_rank(marginal)
