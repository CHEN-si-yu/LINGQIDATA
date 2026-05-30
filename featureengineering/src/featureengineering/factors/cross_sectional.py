"""Cross-sectional and interaction factors.

Market beta, idiosyncratic volatility, sector-relative strength,
time-series momentum, and lead-lag effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


# ── Market Beta ─────────────────────────────────────────────────────────

@register_factor(
    name="sector_relative_momentum_20",
    description="行业相对动量因子，20日个股收益-行业内中位数收益截面排名。",
    category="price",
    thesis="剔除行业轮动效应后的个股动量才是真正的alpha。行业内相对动量捕捉的是个股层面的趋势强度，比绝对动量信号更纯净。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_sector_relative_momentum_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    close = daily_adj["close"]
    ret_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))

    codes = ret_20.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"ret_20": ret_20.values, "industry": industries.values}, index=ret_20.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["ret_20"].transform("median")
    df["relative"] = df["ret_20"] - df["sector_median"]
    return cross_sectional_rank(df["relative"])


@register_factor(
    name="sector_relative_turnover",
    description="行业相对换手率因子，个股20日均换手率/行业中位数换手率截面排名（低比排前）。",
    category="price",
    thesis="行业内异常高换手往往意味着投机性交易或信息不对称，低换手在行业内的相对稀缺性代表筹码稳定性。",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_sector_relative_turnover(context: FactorContext):
    fin = context.load("finance.parquet")
    industry_map = context.repo.load_industry_map()
    turnover = fin["turnover_rate"]
    to_20 = rolling_group_mean(turnover, 20)
    codes = to_20.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"to_20": to_20.values, "industry": industries.values}, index=to_20.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["to_20"].transform("median")
    df["relative"] = df["to_20"] / df["sector_median"].replace(0, np.nan)
    return cross_sectional_rank(-df["relative"])


# ── Time-series momentum (Moskowitz et al. 2012) ────────────────────────

@register_factor(
    name="ts_mom_sign_60",
    description="时间序列动量符号因子，60日收益>0为1否则为0截面排名。Moskowitz et al.(2012)框架。",
    category="price",
    thesis="时间序列动量（看自身过去收益的符号）不依赖横截面对比，捕捉的是绝对趋势持续性。与横截面动量互补：ts_mom在趋势市场中更优，cs_mom在震荡市中更优。",
    dependencies=("daily_adj.parquet",),
)
def factor_ts_mom_sign_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_60 = close.groupby(level="Code").transform(lambda s: s.pct_change(60))
    sign = (ret_60 > 0).astype(float)
    return cross_sectional_rank(sign)


@register_factor(
    name="ts_mom_vol_scaled_60",
    description="波动率缩放时序动量因子，60日收益/60日波动率截面排名。",
    category="price",
    thesis="波动率缩放后的时序动量对不同波动水平的股票具有可比性，解决了传统时序动量偏向高波动股票的问题。",
    dependencies=("daily_adj.parquet",),
)
def factor_ts_mom_vol_scaled_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_60 = close.groupby(level="Code").transform(lambda s: s.pct_change(60))
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_60 = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    scaled = safe_divide(ret_60, vol_60)
    return cross_sectional_rank(scaled)


# ── Co-momentum / lead-lag ──────────────────────────────────────────────

@register_factor(
    name="cs_dispersion_20",
    description="横截面离散度因子，行业内20日收益标准差截面排名。行业内收益分歧度。",
    category="price",
    thesis="行业内收益离散度高意味着行业处于分歧/转折期，离散度低代表行业趋势一致性强。在一致性强行业中做多动量效应更强。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_cs_dispersion_20(context: FactorContext):
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


# ── Downside beta ───────────────────────────────────────────────────────

