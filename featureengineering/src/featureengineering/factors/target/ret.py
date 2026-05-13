from __future__ import annotations

import numpy as np

from ...registry import FactorContext, register_factor

"""
Target / 目标标签因子

统一交易逻辑：T 日收盘后计算因子信号 → T+1 日开盘买入 → 持有 N 个交易日 →
T+1+N 日开盘卖出。因此 label = open[t+1+N] / open[t+1] - 1。

严禁将 target 类因子作为选股特征使用。
"""


def _forward_open_ret(series, horizon: int):
    """open[t+1+N] / open[t+1] - 1，按 Code 分组计算。"""
    buy_price = series.shift(-1)
    sell_price = series.shift(-(1 + horizon))
    return sell_price / buy_price - 1.0


@register_factor(
    name="label_ret_1d",
    description="T+1开盘买入、T+2开盘卖出，1日目标收益。",
    category="target",
    thesis="隔日开盘买入次日开盘卖出，最短持股周期，适合超短线截面策略。",
    dependencies=("daily_adj.parquet",),
)
def factor_label_ret_1d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    open_price = daily_adj["open"].replace(0, float("nan"))
    raw = open_price.groupby(level="Code").transform(
        lambda s: _forward_open_ret(s, 1)
    )
    raw.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    return raw.rename("label_ret_1d")


@register_factor(
    name="label_ret_5d",
    description="T+1开盘买入、T+6开盘卖出，5日目标收益。",
    category="target",
    thesis="一周持股周期，是A股中频截面策略最常用的目标标签。",
    dependencies=("daily_adj.parquet",),
)
def factor_label_ret_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    open_price = daily_adj["open"].replace(0, float("nan"))
    raw = open_price.groupby(level="Code").transform(
        lambda s: _forward_open_ret(s, 5)
    )
    raw.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    return raw.rename("label_ret_5d")


@register_factor(
    name="label_ret_10d",
    description="T+1开盘买入、T+11开盘卖出，10日目标收益。",
    category="target",
    thesis="两周持股周期，平衡信号衰减与换手成本。",
    dependencies=("daily_adj.parquet",),
)
def factor_label_ret_10d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    open_price = daily_adj["open"].replace(0, float("nan"))
    raw = open_price.groupby(level="Code").transform(
        lambda s: _forward_open_ret(s, 10)
    )
    raw.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    return raw.rename("label_ret_10d")


@register_factor(
    name="label_ret_20d",
    description="T+1开盘买入、T+21开盘卖出，20日目标收益。",
    category="target",
    thesis="月度持股周期，IC衰减但换手成本大幅降低，适合低频截面策略。",
    dependencies=("daily_adj.parquet",),
)
def factor_label_ret_20d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    open_price = daily_adj["open"].replace(0, float("nan"))
    raw = open_price.groupby(level="Code").transform(
        lambda s: _forward_open_ret(s, 20)
    )
    raw.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    return raw.rename("label_ret_20d")
