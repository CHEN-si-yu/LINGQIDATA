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

@register_factor(
    name="index_weight_trend_3m",
    description="指数权重趋势因子（沪深300），权重3个月斜率截面排名（权重持续提升=被动资金持续流入排前）。",
    category="index",
    thesis="沪深300指数权重的趋势性变化反映指数编制规则下的长期再平衡方向——权重持续提升的股票受益于被动资金的持续流入，是低换手策略中的优质alpha来源。仅使用沪深300避免多指数混合干扰趋势信号。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_trend_3m(context: FactorContext):
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

    def _trend_slope(y):
        y_arr = np.asarray(y[~np.isnan(y)])
        if len(y_arr) < 2:
            return np.nan
        x = np.arange(len(y_arr), dtype=float)
        x = x - x.mean()
        y_arr = y_arr - y_arr.mean()
        denom = (x * x).sum()
        if denom == 0:
            return np.nan
        return (x * y_arr).sum() / denom

    slope = hs300.groupby(level="Code").transform(
        lambda s: s.rolling(3, min_periods=2).apply(_trend_slope, raw=True)
    )
    calendar = context.repo._read_parquet(
        context.repo.paths.source_root / "calendar.parquet"
    )
    daily = _monthly_to_daily(slope, calendar, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)
