"""
Extended momentum and reversal factors (动量延伸因子) — Class 1.

将动量分解为隔夜/日内成分，构建残差动量（剔除市场+行业后的纯alpha动量），
以及特质波动率调整的动量纯度指标。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide


# ── 动量分解 ──────────────────────────────────────────────────────────────

@register_factor(
    name="overnight_momentum_20",
    description="隔夜动量因子，20日隔夜收益累计截面排名（隔夜涨=机构布局排前）。",
    category="price",
    thesis="隔夜收益（收盘到次日开盘）和日内收益（开盘到收盘）反映不同类型投资者的行为——隔夜收益主要由机构在盘后和集合竞价的大单推动，日内收益受散户和短线交易主导。隔夜动量比日内动量更能代表'聪明钱'方向。",
    dependencies=("daily_adj.parquet",),
)
def factor_overnight_momentum_20(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    close = daily["close"]
    open_ = daily["open"]

    # Overnight return: (open / prev_close - 1)
    prev_close = close.groupby(level="Code").shift(1)
    overnight_ret = (open_ - prev_close) / prev_close.replace(0, np.nan)

    # 20-day cumulative overnight return
    mom = overnight_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    return cross_sectional_rank(mom)


@register_factor(
    name="intraday_momentum_20",
    description="日内动量因子，20日日间收益（开盘到收盘）累计截面排名。",
    category="price",
    thesis="日内收益反映的是交易时段的市场博弈——日内收益累计为正说明买方在交易时段持续主导，是短线强势的信号。与隔夜动量结合使用：隔夜+日内双强=最佳动量，仅日内强=游资短炒。",
    dependencies=("daily_adj.parquet",),
)
def factor_intraday_momentum_20(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    close = daily["close"]
    open_ = daily["open"]

    # Intraday return: (close / open - 1)
    intraday_ret = (close - open_) / open_.replace(0, np.nan)

    mom = intraday_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    return cross_sectional_rank(mom)


@register_factor(
    name="overnight_intraday_divergence_20",
    description="隔夜日内背离因子，(隔夜20日动量-日内20日动量)截面排名（隔夜强+日内弱=机构主导排前）。",
    category="price",
    thesis="隔夜动量与日内动量的差值反映定价权的归属——隔夜强+日内弱说明盘后/集合竞价期间机构在向上定价、但日间散户在卖出，是机构主导型牛股的信号。相反（日内强+隔夜弱）则可能是散户追涨型。",
    dependencies=("daily_adj.parquet",),
)
def factor_overnight_intraday_divergence_20(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    close = daily["close"]
    open_ = daily["open"]

    prev_close = close.groupby(level="Code").shift(1)
    overnight_ret = (open_ - prev_close) / prev_close.replace(0, np.nan)
    intraday_ret = (close - open_) / open_.replace(0, np.nan)

    overnight_mom = overnight_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    intraday_mom = intraday_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )

    divergence = overnight_mom - intraday_mom
    return cross_sectional_rank(divergence)


# ── 残差动量 ──────────────────────────────────────────────────────────────

@register_factor(
    name="residual_momentum_60",
    description="残差动量因子，60日收益剔除市场+行业收益后的残差截面排名（纯alpha动量排前）。",
    category="price",
    thesis="总动量=市场beta+行业动量+特质alpha。残差动量剥离了市场和行业的影响，反映了纯粹的选股alpha——一只股票跑赢其行业和市场的部分才是管理能力的体现。学术文献(Gutierrez & Prinsky, 2007)证实残差动量比总动量更稳定。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_residual_momentum_60(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()

    close = daily["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    mom_60 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).sum()
    )

    # Market return: equal-weighted average
    mkt_ret = ret.groupby(level="Date").mean()

    # Industry return
    codes = ret.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"ret": ret.values, "industry": industries.values}, index=ret.index)
    df = df.dropna(subset=["industry"])
    ind_ret = df.groupby(["Date", "industry"])["ret"].mean()

    # Align market and industry returns
    aligned_mkt = pd.Series(mkt_ret.loc[df.index.get_level_values("Date")].values, index=df.index)
    aligned_ind = pd.Series(
        ind_ret.loc[[(d, i) for d, i in zip(df.index.get_level_values("Date"), df["industry"])]].values,
        index=df.index,
    )

    # Residual = total - market - (industry - market) = total - industry
    residual = df["ret"] - aligned_ind.values

    # 60-day residual momentum
    res_mom = residual.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).sum()
    )

    return cross_sectional_rank(res_mom)


# ── 52周高点 ──────────────────────────────────────────────────────────────

@register_factor(
    name="high_52w_proximity_acceleration",
    description="52周高点接近加速度因子，(当前距离-1月前距离)/1月前距离截面排名（加速接近=突破在即排前）。",
    category="price",
    thesis="接近52周高点的速度变化比距离本身更重要——距离快速缩窄（加速度大）意味着股价正在加速突破，是突破前的'蓄力'阶段。匀速接近可能只是随波逐流，加速接近才是主动上攻的信号。",
    dependencies=("daily_adj.parquet",),
)
def factor_high_52w_proximity_acceleration(context: FactorContext):
    daily = context.load("daily_adj.parquet")
    close = daily["close"]

    high_52w = close.groupby(level="Code").transform(
        lambda s: s.rolling(252, min_periods=126).max()
    )
    proximity = close / high_52w.replace(0, np.nan)  # 0~1, 1=at high

    # Acceleration: change in proximity over 20 days
    accel = proximity.groupby(level="Code").transform(
        lambda s: s.diff(20)
    )

    return cross_sectional_rank(accel)
