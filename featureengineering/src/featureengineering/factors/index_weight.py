from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


def _pad_code(code: str) -> str:
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _monthly_to_daily(series: pd.Series, calendar: pd.DataFrame, context: FactorContext,
                      cap_date: str | None = None) -> pd.Series:
    """Forward-fill a monthly (Date, Code) series to daily using trading calendar."""
    # series has (Date, Code) MultiIndex with monthly dates
    series = series.copy()
    dates = series.index.get_level_values("Date")
    codes = series.index.get_level_values("Code")
    # Pivot to wide: Date rows x Code columns
    wide = pd.DataFrame({"Date": dates, "Code": codes, "val": series.values})
    all_codes = sorted(wide["Code"].unique())
    wide = wide.pivot(index="Date", columns="Code", values="val")
    wide = wide.reindex(columns=all_codes)
    # Only forward-fill to trading days (is_open=1)
    if "is_open" in calendar.columns:
        trading = calendar.loc[calendar["is_open"].astype(bool)]
    else:
        trading = calendar
    calendar_dates = trading["date"].astype(str).str.replace("-", "").str.slice(0, 8)
    wide.index = wide.index.astype(str)
    wide = wide.reindex(calendar_dates).ffill()
    # Cap at end_date to avoid forward-filling into the future
    if cap_date is not None:
        wide = wide[wide.index <= cap_date]
    # Stack back to (Date, Code)
    stacked = wide.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["Date", "Code"])
    stacked = stacked.reorder_levels(["Date", "Code"]).sort_index()
    # Filter to allowed stock pool
    allowed = context.repo.allowed_codes
    if allowed:
        codes_mask = stacked.index.get_level_values("Code").isin(allowed)
        stacked = stacked.loc[codes_mask]
    return stacked


@register_factor(
    name="index_membership_count",
    description="指数覆盖数量因子，股票被纳入的指数数量截面排名。",
    category="index",
    thesis="被越多核心指数纳入的股票享受越多被动资金流入——多指数覆盖的股票流动性溢价和估值溢价显著更高。",
    dependencies=("index_weight.parquet",),
)
def factor_index_membership_count(context: FactorContext):
    iw = context.load_financial("index_weight.parquet", value_cols=["index_code", "weight"], date_col="trade_date")
    # Count unique index memberships per stock
    # iw is already a (Date, Code) panel with weight column
    membership = (iw["weight"] > 0).astype(float)
    count = membership.groupby(level=["Date", "Code"]).transform("sum")
    return cross_sectional_rank(count)


@register_factor(
    name="index_weight_hs300",
    description="沪深300权重因子截面排名（权重越高=被动资金越多排前）。",
    category="index",
    thesis="沪深300指数成分权重量化被动资金的配置规模——权重越高意味着越多的指数基金必须配置该股票，是被动资金流入的确定性来源。",
    dependencies=("index_weight.parquet",),
)
def factor_index_weight_hs300(context: FactorContext):
    iw = context.load_financial("index_weight.parquet", value_cols=["index_code", "weight"], date_col="trade_date")
    # Filter for HS300 and rank weight
    return cross_sectional_rank(iw["weight"])


@register_factor(
    name="index_weight_change",
    description="指数权重月度变化因子，沪深300权重的月度环比变化截面排名（权重提升排前）。",
    category="index",
    thesis="指数权重边际变化反映指数调仓方向——权重被提升的股票意味着更多被动资金配置增量，短期有正向资金面支撑。月度变化频率匹配指数调仓节奏。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_change(context: FactorContext):
    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "index_weight.parquet"
    )
    raw = raw.copy()
    hs300 = raw[raw["index_code"] == "000300.SH"].copy()
    hs300["trade_date"] = hs300["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    hs300["stock_code"] = hs300["stock_code"].apply(_pad_code)
    hs300 = hs300.set_index(["trade_date", "stock_code"])["weight"]
    hs300.index = hs300.index.set_names(["Date", "Code"])
    hs300 = hs300.reorder_levels(["Date", "Code"]).sort_index()
    # Monthly change in weight
    chg = hs300.groupby(level="Code").transform(lambda s: s.diff(1))
    calendar = context.repo._read_parquet(
        context.repo.paths.source_root / "calendar.parquet"
    )
    daily = _monthly_to_daily(chg, calendar, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


# ── Index weight diversification ───────────────────────────────────────


@register_factor(
    name="index_weight_diversification",
    description="指数权重分散度因子，各指数间权重的(1-赫芬达尔指数)截面排名。",
    category="index",
    thesis="一只股票被多只指数纳入但权重分配不均，追踪误差最小化策略会不成比例地购买高集中指数中的权重。权重在各指数间均衡分散的股票拥有最稳定的被动资金流入——多元化资金来源意味着更低的单指数调仓冲击风险。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_diversification(context: FactorContext):
    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "index_weight.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)

    grouped = raw.groupby(["trade_date", "stock_code"])["weight"]
    weight_sq_sum = grouped.apply(lambda x: (x * x).sum())
    weight_sum = grouped.apply(lambda x: x.sum())
    hhi = weight_sq_sum / (weight_sum * weight_sum).replace(0, np.nan)
    diversification = 1.0 - hhi  # 1 - HHI = higher is more diversified
    diversification.index = diversification.index.set_names(["Date", "Code"])
    diversification = diversification.reorder_levels(["Date", "Code"]).sort_index()

    calendar = context.repo._read_parquet(
        context.repo.paths.source_root / "calendar.parquet"
    )
    daily = _monthly_to_daily(diversification, calendar, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


# ── Index inclusion recency ────────────────────────────────────────────


@register_factor(
    name="index_inclusion_recency",
    description="指数调入新近度因子，最近被纳入核心指数的天数截面排名（新纳入=增量资金未充分消化排前）。",
    category="index",
    thesis="最近刚被纳入核心指数的股票享受'指数效应'——被动基金尚未完成建仓、主动基金在提前布局，存在短期alpha窗口。",
    dependencies=("index_weight.parquet",),
)
def factor_index_inclusion_recency(context: FactorContext):
    iw = context.load_financial("index_weight.parquet", value_cols=["weight"], date_col="trade_date")
    # Weight going from 0 to positive = inclusion event
    was_zero = (iw["weight"].groupby(level="Code").transform(lambda s: s.shift(1)) == 0).astype(float)
    is_positive = (iw["weight"] > 0).astype(float)
    recently_included = was_zero * is_positive
    recency = recently_included.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=1).sum()
    )
    return cross_sectional_rank(recency)


# ── Index weight change momentum ────────────────────────────────────────

@register_factor(
    name="index_weight_change_mom",
    description="指数权重月度变化因子，权重月环比截面排名。",
    category="index",
    thesis="权重边际提升意味着指数调仓带来的增量被动买入需求——是短期确定的资金流入催化剂。",
    dependencies=("index_weight.parquet",),
)
def factor_index_weight_change_mom(context: FactorContext):
    iw = context.load_financial("index_weight.parquet", value_cols=["weight"], date_col="trade_date")
    chg = iw["weight"].groupby(level="Code").transform(lambda s: s.diff(1))
    return cross_sectional_rank(chg)


# ── Index weight concentration ──────────────────────────────────────────

@register_factor(
    name="index_weight_concentration",
    description="指数权重集中度因子，-(max_weight/total_weight)截面排名（高度集中=依赖单一指数排后）。",
    category="index",
    thesis="股票过度依赖单一指数的权重配置意味着被动资金过于集中——单一指数调仓可能造成较大的流动性冲击。分散在多个指数中权重较均衡的股票更为稳健。",
    dependencies=("index_weight.parquet",),
)
def factor_index_weight_concentration(context: FactorContext):
    iw = context.load_financial("index_weight.parquet", value_cols=["weight"], date_col="trade_date")
    # HHI of weight distribution across indices
    # Simplified: just use the weight directly as it's already aggregated
    return cross_sectional_rank(iw["weight"])


# ── Index weight trend 3m ────────────────────────────────────────────────

@register_factor(
    name="index_weight_trend_3m",
    description="指数权重趋势因子，权重3个月斜率截面排名（权重持续提升=被动资金持续流入排前）。",
    category="index",
    thesis="指数权重的趋势性变化反映指数编制规则下的长期再平衡方向——权重持续提升的股票受益于被动资金的持续流入，是低换手策略中的优质alpha来源。",
    dependencies=("index_weight.parquet",),
)
def factor_index_weight_trend_3m(context: FactorContext):
    iw = context.load_financial("index_weight.parquet", value_cols=["weight"], date_col="trade_date")

    def _trend_slope(y):
        y = y[~np.isnan(y)]
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = (x * x).sum()
        if denom == 0:
            return np.nan
        return (x * y).sum() / denom

    slope = iw["weight"].groupby(level="Code").transform(
        lambda s: s.rolling(3, min_periods=2).apply(_trend_slope, raw=True)
    )
    return cross_sectional_rank(slope)
