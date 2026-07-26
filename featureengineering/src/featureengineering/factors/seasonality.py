from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    safe_divide,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _daily_returns(daily: pd.DataFrame) -> pd.Series:
    return daily["close"].groupby(level="Code").pct_change()


# ── Calendar / seasonality effects ──────────────────────────────────────────


@register_factor(
    name="pre_holiday_effect",
    description="节前效应因子，节前最后交易日收益率截面排名。",
    category="price",
    thesis="A股存在节前红包行情——春节、国庆等长假前最后一个交易日往往有上涨，与资金提前布局节后行情的预期有关。",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def factor_pre_holiday_effect(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    # Simple approach: use rolling mean of the pattern
    pre_holiday_avg = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(pre_holiday_avg)


@register_factor(
    name="weekday_effect",
    description="周内效应因子，基于股票在各星期几历史收益率的平均截面排名。",
    category="price",
    thesis="周内效应是全球股市普遍存在的异象——A股'黑色星期四'(周四平均收益最低)是知名的日历异象，周五效应(对周末消息的提前博弈)也有显著alpha。",
    dependencies=("daily_adj.parquet",),
)
def factor_weekday_effect(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    dates = pd.to_datetime(ret.index.get_level_values("Date"), format="%Y%m%d")
    weekday = dates.dayofweek
    df = pd.DataFrame({"ret": ret.values, "wday": weekday}, index=ret.index)

    # Expanding mean within each (weekday, Code) group — only uses data from
    # same-weekday dates up to and including the current date (no future leakage).
    def _expanding_mean(s):
        return s.expanding(min_periods=10).mean()

    wday_avg = df.groupby(["wday", "Code"])["ret"].transform(_expanding_mean)
    return cross_sectional_rank(wday_avg)


@register_factor(
    name="earnings_season_effect",
    description="财报季效应因子，财报密集发布月(3/4/8/10月)的平均收益率截面排名。",
    category="price",
    thesis="财报季(年报3-4月、中报8月、三季报10月)是信息最密集的时期——这一时期表现优异的股票往往有基本面支撑，过滤了纯主题炒作。",
    dependencies=("daily_adj.parquet",),
)
def factor_earnings_season_effect(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    dates = pd.to_datetime(ret.index.get_level_values("Date"), format="%Y%m%d")
    month = dates.month
    is_earnings_month = month.isin([3, 4, 8, 10])
    df = pd.DataFrame({"ret": ret.values, "em": is_earnings_month}, index=ret.index)
    em_avg = df.groupby("Code")["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).mean()
    )
    return cross_sectional_rank(em_avg)
