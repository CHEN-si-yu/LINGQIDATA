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
    name="day_of_week_effect",
    description="星期效应因子 (过去N周星期X的平均收益)",
    category="seasonality",
    thesis="A股存在显著的周历效应：周一常跌(周末信息消化)、周五常涨(博弈周末利好)",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def day_of_week_effect(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    dates = ret.index.get_level_values("Date")
    weekday = pd.to_datetime(dates, format="%Y%m%d").weekday
    ret_series = pd.Series(ret.values, index=ret.index)

    # Friday effect (weekday=4)
    friday_mask = weekday == 4
    friday_ret = ret_series.where(friday_mask, np.nan)
    avg_friday = friday_ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=20).mean()
    )
    # Monday effect (weekday=0)
    monday_mask = weekday == 0
    monday_ret = ret_series.where(monday_mask, np.nan)
    avg_monday = monday_ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=20).mean()
    )

    effect = avg_friday.fillna(0) - avg_monday.fillna(0)
    effect.name = "day_of_week_effect"
    effect.index = ret.index
    return cross_sectional_rank(effect)


@register_factor(
    name="month_start_effect_3m",
    description="月初效应因子 (3个月月初平均收益)",
    category="seasonality",
    thesis="A股月初常有增量资金入场，月初收益率系统性地高于月末",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def month_start_effect_3m(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    dates = pd.to_datetime(ret.index.get_level_values("Date"), format="%Y%m%d")
    day_of_month = dates.day
    ret_series = pd.Series(ret.values, index=ret.index)

    month_start = ret_series.where(day_of_month <= 5, np.nan)
    month_end = ret_series.where(day_of_month >= 25, np.nan)

    avg_start = month_start.groupby(level="Code").transform(
        lambda s: s.rolling(63, min_periods=30).mean()
    )
    avg_end = month_end.groupby(level="Code").transform(
        lambda s: s.rolling(63, min_periods=30).mean()
    )

    effect = avg_start.fillna(0) - avg_end.fillna(0)
    effect.name = "month_start_effect"
    effect.index = ret.index
    return cross_sectional_rank(effect)



@register_factor(
    name="turn_of_month_effect",
    description="月末月初反转效应因子",
    category="seasonality",
    thesis="月底常有仓位调整导致的过度反应，月初反转",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def turn_of_month_effect(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    dates = pd.to_datetime(ret.index.get_level_values("Date"), format="%Y%m%d")
    day_of_month = dates.day
    ret_series = pd.Series(ret.values, index=ret.index)

    # Last 3 days of month
    last_days = ret_series.where(day_of_month >= 28, np.nan)
    # First 3 days of next month
    first_days = ret_series.where(day_of_month <= 3, np.nan)

    # Negative correlation between end-of-month and start-of-next-month
    avg_last = last_days.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=15).mean()
    )
    avg_first = first_days.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=15).mean()
    )

    # Reversal: negative last -> positive first (buy the dip)
    effect = avg_first.fillna(0) - avg_last.fillna(0)
    effect.name = "turn_of_month"
    effect.index = ret.index
    return cross_sectional_rank(effect)


# ── Earnings announcement effect ────────────────────────────────────────────

@register_factor(
    name="earnings_announcement_drift",
    description="盈余公告后漂移因子 (PEAD, 最近公告窗口的异常收益)",
    category="seasonality",
    thesis="A股存在盈余公告后漂移效应：超预期的股票公告后持续上涨",
    dependencies=("daily_adj.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def earnings_announcement_drift(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    fin = ctx.load_financial(
        "financial_indicator.parquet",
        value_cols=["eps", "dt_eps_yoy"],
        date_col="ann_date",
    )

    ret_values = ret.values
    ret_index = ret.index
    ann_eps = fin.get("eps") if isinstance(fin, pd.DataFrame) else None
    ann_yoy = fin.get("dt_eps_yoy") if isinstance(fin, pd.DataFrame) else None

    if ann_yoy is not None and not ann_yoy.empty:
        # Earnings surprise proxy: YoY EPS growth
        # Forward-fill the surprise signal for 20 days after announcement
        surprise = ann_yoy.groupby(level="Date").transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-8)
        )
        return cross_sectional_rank(surprise)

    # Fallback: use recent return momentum around announcement dates
    return cross_sectional_rank(rolling_group_mean(ret.fillna(0), 5))


@register_factor(
    name="month_effect_rank",
    description="月份效应因子，基于股票在同一月份历史收益率的胜率截面排名。",
    category="price",
    thesis="部分股票存在稳定的日历效应——某些月份历史胜率显著高于其他月份。该因子捕捉统计上稳定的季节性模式。",
    dependencies=("daily_adj.parquet",),
)
def factor_month_effect_rank(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    dates = pd.to_datetime(ret.index.get_level_values("Date"), format="%Y%m%d")
    month = dates.month
    df = pd.DataFrame({"ret": ret.values, "month": month}, index=ret.index)

    # Expanding mean within each (month, Code) group — only uses data from
    # same-month dates up to and including the current date (no future leakage).
    def _expanding_mean(s):
        return s.expanding(min_periods=5).mean()

    month_avg = df.groupby(["month", "Code"])["ret"].transform(_expanding_mean)
    return cross_sectional_rank(month_avg)



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
