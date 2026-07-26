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

