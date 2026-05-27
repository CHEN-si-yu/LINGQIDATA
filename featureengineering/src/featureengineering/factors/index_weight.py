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
    description="指数覆盖度因子，股票被8大核心指数纳入的数量截面排名（月度前向填充至日频）。",
    category="index",
    thesis="被越多核心指数同时纳入的股票，被动资金流入来源越确定，流动性溢价越高。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_membership_count(context: FactorContext):
    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "index_weight.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)
    membership = raw.groupby(["trade_date", "stock_code"])["index_code"].nunique()
    membership.index = membership.index.set_names(["Date", "Code"])
    membership = membership.reorder_levels(["Date", "Code"]).sort_index()
    calendar = context.repo._read_parquet(
        context.repo.paths.source_root / "calendar.parquet"
    )
    daily = _monthly_to_daily(membership, calendar, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


@register_factor(
    name="index_weight_hs300",
    description="沪深300权重因子，沪深300成分股权重截面排名（月度前向填充至日频）。",
    category="index",
    thesis="沪深300权重越高，被动配置资金流入量越大，且通常对应更优质的大盘蓝筹公司。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_weight_hs300(context: FactorContext):
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
    calendar = context.repo._read_parquet(
        context.repo.paths.source_root / "calendar.parquet"
    )
    daily = _monthly_to_daily(hs300, calendar, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)


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
    description="指数纳入新近度因子，(-距首次被核心指数纳入的天数)截面排名（新纳入排前）。",
    category="index",
    thesis="新纳入指数的股票经历一次性的被动资金流入激增和机构覆盖度提升。纳入后的前3-6个月内，指数效应产生显著超额收益。随着时间推移，指数效应的优势递减——'新纳入'的标签溢价逐渐被市场消化。",
    dependencies=("index_weight.parquet", "calendar.parquet"),
)
def factor_index_inclusion_recency(context: FactorContext):
    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "index_weight.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)

    first_date = raw.groupby("stock_code")["trade_date"].min()
    first_date.index.name = "Code"

    all_dates = raw["trade_date"].unique()
    all_codes = raw["stock_code"].unique()
    full_index = pd.MultiIndex.from_product(
        [all_dates, all_codes], names=["Date", "Code"]
    )
    first_map = first_date.reindex(full_index, level="Code")

    current_date = pd.to_datetime(
        full_index.get_level_values("Date"), format="%Y%m%d"
    )
    first_dt = pd.to_datetime(first_map.values, format="%Y%m%d")
    days_since = (current_date - first_dt).days.astype(float)
    recency = pd.Series(-days_since, index=full_index, name="recency")
    recency = recency.reorder_levels(["Date", "Code"]).sort_index()

    allowed = context.repo.allowed_codes
    if allowed:
        codes_mask = recency.index.get_level_values("Code").isin(allowed)
        recency = recency.loc[codes_mask]

    calendar = context.repo._read_parquet(
        context.repo.paths.source_root / "calendar.parquet"
    )
    daily = _monthly_to_daily(recency, calendar, context, cap_date=context.end_date)
    return cross_sectional_rank(daily)
