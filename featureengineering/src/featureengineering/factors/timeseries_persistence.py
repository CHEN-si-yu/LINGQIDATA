"""
Time-series persistence and predictability factors — Class 1 panel factors.

These factors capture the time-series properties of individual stocks:
  - Hurst exponent (long-memory / trend persistence)
  - Variance ratio (random walk vs trending vs mean-reverting)
  - Serial correlation (AR(1) structure)
  - Price efficiency

All factors use daily_adj.parquet.
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
# Section A: Hurst Exponent
# ═══════════════════════════════════════════════════════════════════════════════

def _hurst_rs(series: np.ndarray) -> float:
    """Compute Hurst exponent via Rescaled Range (R/S) analysis.

    H > 0.5: trending (persistent)
    H < 0.5: mean-reverting (anti-persistent)
    H = 0.5: random walk
    """
    n = len(series)
    if n < 20:
        return np.nan

    # Use multiple lag windows
    lags = [max(5, n // 8), max(5, n // 4), max(5, n // 2)]
    rs_values = []

    for lag in lags:
        if lag < 5:
            continue
        # Divide series into chunks of size lag
        chunks = n // lag
        if chunks < 2:
            continue
        rs_chunk = []
        for i in range(chunks):
            chunk = series[i * lag : (i + 1) * lag]
            if len(chunk) < 5:
                continue
            mean = chunk.mean()
            if np.isnan(mean):
                continue
            deviate = chunk - mean
            cumsum_dev = np.cumsum(deviate)
            r = cumsum_dev.max() - cumsum_dev.min()
            s = chunk.std()
            if s is None or np.isnan(s) or s == 0:
                continue
            rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append((np.log(lag), np.log(np.mean(rs_chunk))))

    if len(rs_values) < 2:
        return np.nan

    # Linear regression: log(R/S) = H * log(lag) + c
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    try:
        H = np.polyfit(x, y, 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
    return H


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Variance Ratio
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="variance_ratio_20",
    description="方差比率因子（Var(20日收益)/(20*Var(1日收益))截面排名，VR>1=有趋势成分排前）。",
    category="timeseries",
    thesis=(
        "方差比率(Variance Ratio)检验是Lo-MacKinlay(1988)提出的随机游走检验统计量。"
        "VR=Var(r_20d)/(20×Var(r_1d))，在随机游走下VR=1。"
        "VR>1=存在正自相关（趋势/动量），VR<1=存在负自相关（均值回归/反转）。"
        "VR偏离1的程度反映了价格发现效率：偏离越大，定价效率越低（更多可预测性）。"
        "在A股中，小盘股VR通常偏离1更多（定价效率低=更多alpha机会）。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_variance_ratio_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    # 1-day returns
    ret_1d = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # 20-day variance
    var_20d = ret_1d.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=15).var()
    )

    # Variance of 20-day returns
    ret_20d = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    var_ret_20d = ret_20d.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=40).var()
    )

    # Longer window for the 20d return variance to get stable estimates
    # VR = Var(20d_ret) / (20 * mean_1d_var)
    mean_var_1d_60 = ret_1d.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=40).var()
    )

    vr = safe_divide(var_ret_20d, 20.0 * mean_var_1d_60 + 1e-10)
    vr = vr.clip(0, 5)

    # VR > 1 = trending (positive autocorrelation)
    return cross_sectional_rank(vr)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Serial Correlation
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="serial_correlation_5",
    description="自相关系数因子（5日日收益一阶自相关系数截面排名，负自相关=反转特性排前）。",
    category="timeseries",
    thesis=(
        "日收益的一阶自相关系数AR(1)度量了收益的可预测性。"
        "负自相关=昨日涨→今日倾向于跌（日内反转），适合反转策略；"
        "正自相关=昨日涨→今日倾向于涨（短期动量），适合趋势策略。"
        "AR(1)系数的方向告知了最优的交易方向。"
        "5日滚动窗口足够短以适应当前市场状态，又足够长以获得可靠估计。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_serial_correlation_5(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # 1-day lagged return
    ret_lag = ret.groupby(level="Code").shift(1)

    # 5-day rolling correlation between ret and ret_lag
    def _rolling_corr(x, y, window=5):
        x_roll = x.rolling(window, min_periods=5)
        y_roll = y.rolling(window, min_periods=5)
        cov = (x * y).rolling(window, min_periods=5).mean() - x_roll.mean() * y_roll.mean()
        denom = x_roll.std() * y_roll.std()
        return safe_divide(cov, denom)

    ar1 = ret.groupby(level="Code").transform(
        lambda s: _rolling_corr(s, ret_lag.loc[s.index].fillna(0), 5)
    )
    ar1 = ar1.clip(-1, 1)

    # Negative AR(1) = reversal tendency = can profit from fading
    return cross_sectional_rank(-ar1)


@register_factor(
    name="price_efficiency_ratio_20",
    description="价格效率比率因子（|20日净变动|/20日路径长度截面排名，高效率=趋势明确排前）。",
    category="timeseries",
    thesis=(
        "Kaufman的效率比率(Efficiency Ratio, ER)度量了价格运动的'直线性'。"
        "ER = |Close_t - Close_{t-20}| / Σ|Close_i - Close_{i-1}|。"
        "ER→1=价格以直线方式移动（高效趋势），适合趋势跟踪；"
        "ER→0=价格在区间内来回震荡（低效/噪音），适合反转交易。"
        "与波动率不同，ER区分了'有方向的波动'和'无方向的噪音'。"
        "在A股中，低ER+高波动=典型的游资博弈股票。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_price_efficiency_ratio_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    # Absolute net change over 20 days
    net_chg = close.groupby(level="Code").transform(
        lambda s: s.diff(20).abs()
    )

    # Total path length (sum of absolute daily changes)
    daily_chg = close.groupby(level="Code").transform(
        lambda s: s.diff(1).abs()
    )
    path_length = daily_chg.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )

    er = safe_divide(net_chg, path_length + 1e-10)
    er = er.clip(0, 1)

    return cross_sectional_rank(er)  # high ER = efficient trending = good
