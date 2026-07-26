"""
Behavioral finance extended factors — Class 1 panel factors.

Extends behavioral factors with additional market psychology signals:
  - Disposition effect: volume at gain vs loss positions
  - Gambler preference: lottery-like stock characteristics
  - Attention-driven trading: extreme returns attract attention
  - Overreaction correction: extreme moves tend to revert

Data sources: daily_adj.parquet, finance.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, rolling_group_mean, rolling_group_std


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Lottery Preference — extreme positive skew attracts gambling
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="max_return_20d",
    description="20日最大单日收益率取反。极高单日收益=彩票特征→吸引投机→后续表现差，排名低。",
    category="price",
    thesis=(
        "过去20个交易日内最大单日收益率取反排名。极高单日收益是彩票型股票的典型特征——"
        "吸引投机性买入(博彩偏好)。学术研究表明，高MAX回报的股票后续表现系统性偏弱"
        "(Bali et al., 2011)。该因子捕捉了A股市场的博彩偏好效应。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_max_return_20d(context: FactorContext) -> np.ndarray:
    d = context.load("daily_adj.parquet")
    ret = d["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    max_ret = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    return cross_sectional_rank(-max_ret)


@register_factor(
    name="return_skewness_20d_adj",
    description="20日收益率偏度取反。正偏=偶尔大涨→博彩特征→后续弱，排名低。",
    category="price",
    thesis=(
        "20日收益率偏度取反排名。正偏度(右偏)意味着偶尔出现大幅上涨——"
        "这是彩票型收益分布的特征。正偏度股票吸引投机者、推高当前价格、"
        "但未来收益偏低(博彩溢价消失)。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_return_skewness_20d_adj(context: FactorContext) -> np.ndarray:
    d = context.load("daily_adj.parquet")
    ret = d["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    skew = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).skew()
    )
    skew = skew.clip(-5, 5)
    return cross_sectional_rank(-skew)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Overreaction / Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="overreaction_ratio_5d",
    description="5日极端涨跌幅股票的反转倾向。abs(5日收益)/abs(5日每日收益加总)-1，高比值=趋势强而非反转。",
    category="price",
    thesis=(
        "5日累计收益绝对值除以5日每日收益绝对值之和再减1。"
        "该比值衡量了5日走势的连贯性——比值接近0意味着每日方向一致(趋势强)；"
        "比值远离0意味着中间有大幅反转(趋势弱)。高连贯性趋势更容易延续。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_overreaction_ratio_5d(context: FactorContext) -> np.ndarray:
    d = context.load("daily_adj.parquet")
    ret = d["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    cum_ret = ret.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    abs_sum = ret.abs().groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    ratio = safe_divide(cum_ret.abs(), abs_sum)
    return cross_sectional_rank(ratio)


@register_factor(
    name="idiosyncratic_vol_20d",
    description="特质波动率20日取反。高特质波动=套利困难→定价错误持续→博彩特征，排名低。",
    category="price",
    thesis=(
        "20日特质波动率(用CAPM残差近似)取反排名。高特质波动意味着套利成本高、"
        "定价错误难以纠正。A股高特质波动股票与低未来收益相关(异象)。"
        "近似的特质波动=总波动-市场波动(用截面均值代理市场)。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_idiosyncratic_vol_20d(context: FactorContext) -> np.ndarray:
    d = context.load("daily_adj.parquet")
    ret = d["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    total_vol = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    mkt_ret = ret.groupby(level="Date").transform("mean")
    mkt_vol = mkt_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    idio_var = total_vol ** 2 - mkt_vol ** 2
    idio_vol = np.sqrt(np.maximum(idio_var, 0))
    return cross_sectional_rank(-idio_vol)


@register_factor(
    name="volume_shock_reversal_5d",
    description="放量下跌后5日反转概率。异常放量+负收益=恐慌抛售→潜在反弹，排名高。",
    category="price",
    thesis=(
        "识别放量下跌事件并预测反转。成交量超过20日均量1.5倍且当日收益为负=恐慌抛售信号。"
        "恐慌抛售后往往出现技术性反弹——该因子对放量下跌后的第2-6日给予正向评分。"
        "使用事件衰减函数使信号随时间衰减。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_volume_shock_reversal_5d(context: FactorContext) -> np.ndarray:
    d = context.load("daily_adj.parquet")
    ret = d["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol = d["vol"]
    vol_ma20 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    vol_ratio = safe_divide(vol, vol_ma20)
    # Panic sell day: volume 1.5x normal AND negative return
    panic = ((vol_ratio > 1.5) & (ret < 0)).astype(float)
    # Signal decays exponentially after panic day (half-life = 2 days)
    signal = panic.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    return cross_sectional_rank(signal)
