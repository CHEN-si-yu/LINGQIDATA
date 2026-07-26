"""
Higher-moment and volatility surface factors — Class 1 panel factors.

These factors extend beyond first and second moments (mean, variance) to
capture:
  - Return distribution asymmetry (skewness)
  - Tail risk (kurtosis)
  - Downside risk separation (downside vol ratio)
  - Volatility stability (vol-of-vol)

All factors use daily_adj.parquet with rolling windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    safe_divide,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Higher Moments
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="skewness_60",
    description="60日收益偏度因子（负偏=左尾风险预警排后，正偏=彩票特征排后，零偏=正态分布排前）。",
    category="risk",
    thesis=(
        "收益分布的偏度（skewness）捕捉了非对称风险。"
        "负偏度股票：频繁小涨+偶尔大跌（做空波动率策略的标的），隐含尾部崩盘风险；"
        "正偏度股票：频繁小跌+偶尔大涨（彩票型股票），投资者可能过度支付。"
        "在A股中，极端偏度（无论正负）往往预示着未来收益反转："
        "负偏度过度悲观后反弹，正偏度过度投机后回落。"
        "该因子取偏度绝对值的负向排名，偏爱接近正态分布的收益结构。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_skewness_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    # Daily returns
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    ret = ret.clip(-0.2, 0.2)

    # 60-day rolling skewness
    skew = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).skew()
    )
    skew = skew.clip(-5, 5)

    # Penalize extreme skewness in either direction
    return cross_sectional_rank(-skew.abs())


@register_factor(
    name="kurtosis_60",
    description="60日收益峰度因子（高峰度=尾部风险大排后）。",
    category="risk",
    thesis=(
        "峰度（kurtosis）度量了收益分布的'肥尾'程度。"
        "高峰度意味着极端收益（暴涨暴跌）出现的频率高于正态分布预期，"
        "是纯粹的尾部风险度量。与波动率不同，峰度捕捉的是'意外'的集中程度"
        "而非'意外'的平均大小。高峰度股票需要更高的风险溢价。"
        "在A股中，高峰度往往与游资炒作、概念轮动相关——高风险信号。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_kurtosis_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    ret = ret.clip(-0.2, 0.2)

    # 60-day rolling kurtosis (excess kurtosis)
    kurt = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).kurt()
    )
    kurt = kurt.clip(-5, 20)

    return cross_sectional_rank(-kurt)  # low kurtosis = normal tails = good


@register_factor(
    name="downside_vol_ratio",
    description="下行波动率比率因子（60日下行波动率/总波动率截面排名，低下行比=收益质量高排前）。",
    category="risk",
    thesis=(
        "将波动率分解为上行和下行两部分：下行波动率只计入负收益日的波动贡献。"
        "低下行/总波动率比率意味着股票的波动主要来自上涨日——"
        "这是优质股票的特征（compounders）。"
        "高比率意味着波动主要由下跌驱动——风险积聚的信号。"
        "该比率是Sortino比率思想在横截面上的应用。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_downside_vol_ratio(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # Total volatility
    total_vol = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )

    # Downside volatility: only negative returns contribute
    downside = ret.clip(upper=0)  # keep only negative returns, zero otherwise
    downside_vol = downside.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )

    ratio = safe_divide(downside_vol, total_vol)
    ratio = ratio.clip(0, 2)

    return cross_sectional_rank(-ratio)  # low downside/total = quality


@register_factor(
    name="vol_of_vol_20",
    description="波动率之波动因子（20日波动率的20日标准差截面排名，波动率不稳定=风险排后）。",
    category="risk",
    thesis=(
        "波动率的波动率（vol-of-vol）是二阶风险度量。"
        "波动率本身不稳定的股票更难进行风险管理——"
        "你不知道明天的风险是今天的1倍还是3倍。"
        "稳定的波动率允许更精确的头寸调整和风险预算，"
        "因此vol-of-vol低的股票享有'可预测性溢价'。"
        "vol-of-vol突然上升往往是市场错位的前兆。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_vol_of_vol_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # 20-day rolling volatility
    vol_20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )

    # Standard deviation of vol_20 over 20 days
    vol_of_vol = vol_20.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )

    # Normalize by mean vol level for cross-sectional comparability
    mean_vol = vol_20.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    normalized = safe_divide(vol_of_vol, mean_vol + 1e-10)
    normalized = normalized.clip(0, 5)

    return cross_sectional_rank(-normalized)


@register_factor(
    name="tail_risk_60",
    description="尾部风险因子（60日1%VaR截面排名，VaR越大=尾部风险越高排后）。",
    category="risk",
    thesis=(
        "Value-at-Risk (VaR) 是经典的尾部风险度量。"
        "60日滚动窗口的1%分位数VaR捕捉了'每100个交易日会出现一次'的极端损失。"
        "与波动率不同，VaR只关注左尾（损失端），不受右尾（暴涨端）的干扰。"
        "高VaR股票是天然的'黑天鹅'候选者，需要风险回避。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_tail_risk_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # 60-day rolling 1% VaR (negative of 1st percentile)
    var_99 = ret.groupby(level="Code").transform(
        lambda s: -s.rolling(60, min_periods=30).quantile(0.01)
    )
    var_99 = var_99.clip(0, 0.3)

    return cross_sectional_rank(-var_99)  # low VaR = safer


@register_factor(
    name="up_down_capture_60",
    description="涨跌捕获比率因子（60日正收益均值/|负收益均值|截面排名，高比率=收益不对称偏多排前）。",
    category="risk",
    thesis=(
        "涨跌捕获比率（Up/Down Capture）比较了上涨日和下跌日的平均幅度。"
        "比率>1意味着上涨日的平均涨幅大于下跌日的平均跌幅——"
        "这是理想的非对称收益结构。比率<1意味着下跌日伤害更大。"
        "与偏度不同，该度量更直观——直接比较了'好日子'和'坏日子'的典型大小。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_up_down_capture_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # Rolling mean of positive returns
    up = ret.clip(lower=0).replace(0, np.nan)
    up_mean = up.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=20).mean()
    )

    # Rolling mean of negative returns (absolute value)
    down = ret.clip(upper=0).replace(0, np.nan)
    down_mean = down.groupby(level="Code").transform(
        lambda s: s.abs().rolling(60, min_periods=20).mean()
    )

    ratio = safe_divide(up_mean, down_mean)
    ratio = ratio.clip(0, 5)

    return cross_sectional_rank(ratio)  # high up/down = favorable asymmetry
