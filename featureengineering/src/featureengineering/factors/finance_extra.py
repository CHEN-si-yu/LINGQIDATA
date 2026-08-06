"""
财务日频字段增量因子 — Class 1 (finance.parquet only)。

对 finance.parquet 中尚未被充分利用的字段做增量变换:
- pb 的 20 日变化(估值动量,区别于 pe_ttm_change_20d 的水平变化);
- volume_ratio(量比,当日量/5日均量的比率)自身的 5 日动量;
- free_share/total_share 的 20 日变化(自由流通股扩张≈限售解禁事件的日频代理)。

不依赖 pe_ttm_percentile、不依赖财务报告表(非日频),NaN 覆盖率与
finance 面板一致(< 20%)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


@register_factor(
    name="pb_change_20d",
    description="市净率变化因子：pb的20日变化率截面排名（负向，估值抬升排后）。",
    category="valuation",
    thesis="pb 的 20 日变化率衡量估值扩张/收缩的速度——pb 快速抬升=股价涨幅超过"
           "净资产增长(估值透支,未来回报被摊薄);pb 收缩=股价相对净资产走低"
           "(估值回归价值区)。与 pe_ttm_change_20d(盈利口径)互补,构成估值动量维度。",
    dependencies=("finance.parquet",),
)
def factor_pb_change_20d(context: FactorContext):
    fin = context.load("finance.parquet")
    pb_lag = fin["pb"].groupby(level="Code").shift(20)
    change = safe_divide(fin["pb"] - pb_lag, pb_lag)
    return cross_sectional_rank(-change)


@register_factor(
    name="volume_ratio_momentum_5d",
    description="量比动量因子：量比(volume_ratio)的5日变化截面排名（量能扩张排前）。",
    category="fund_flow",
    thesis="volume_ratio(当日成交量/过去5日均量)是量能热度的即时度量——其5日变化"
           "捕捉量能扩张/收缩的速度:量比持续抬升=关注度快速升温(资金进场确认);"
           "量比持续走低=热度退潮。与成交量水平类因子正交,是量能的一阶导。",
    dependencies=("finance.parquet",),
)
def factor_volume_ratio_momentum_5d(context: FactorContext):
    fin = context.load("finance.parquet")
    vr = fin["volume_ratio"]
    change = vr - vr.groupby(level="Code").shift(5)
    return cross_sectional_rank(change)


@register_factor(
    name="free_float_expansion_20d",
    description="自由流通股扩张因子：自由流通股/总股本占比的20日变化截面排名（负向，解禁抛压排后）。",
    category="valuation",
    thesis="自由流通股占比的抬升主要来自限售股解禁/转流通(事件驱动研报:解禁后的"
           "抛售压力与筹码扩容)——占比上升=潜在供给增加(承压);占比稳定=筹码结构"
           "不变。finance.parquet 的 free_share/total_share 为日频字段,20日变化"
           "即解禁事件的近似代理,与事件驱动因子族互补。",
    dependencies=("finance.parquet",),
)
def factor_free_float_expansion_20d(context: FactorContext):
    fin = context.load("finance.parquet")
    ffr = safe_divide(fin["free_share"], fin["total_share"])
    change = ffr - ffr.groupby(level="Code").shift(20)
    return cross_sectional_rank(-change)
