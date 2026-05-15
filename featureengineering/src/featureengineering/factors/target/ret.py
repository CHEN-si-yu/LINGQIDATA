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


def _trim_tail(series, horizon: int):
    """去掉每只股票尾部因前视数据不足产生的 NaN。

    label_ret 需要未来 open 数据，最近 (1+horizon) 个日期无法计算有效值。
    这里直接截断尾部，而非用 NaN 填充。
    """
    drop = 1 + horizon
    if len(series) <= drop:
        return series.iloc[:0]
    return series.iloc[:-drop]


def _compute_label_ret(daily_adj, horizon: int, name: str):
    open_price = daily_adj["open"].replace(0, float("nan"))
    raw = open_price.groupby(level="Code").transform(
        lambda s: _forward_open_ret(s, horizon)
    )
    raw.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    raw = raw.groupby(level="Code", group_keys=False).apply(
        _trim_tail, horizon=horizon,
    )
    return raw.rename(name)


@register_factor(
    name="label_ret_1d",
    description="T+1开盘买入、T+2开盘卖出，1日目标收益。",
    category="target",
    thesis="隔日开盘买入次日开盘卖出，最短持股周期，适合超短线截面策略。",
    dependencies=("daily_adj.parquet",),
)
def factor_label_ret_1d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    return _compute_label_ret(daily_adj, 1, "label_ret_1d")


@register_factor(
    name="label_ret_5d",
    description="T+1开盘买入、T+6开盘卖出，5日目标收益。",
    category="target",
    thesis="一周持股周期，是A股中频截面策略最常用的目标标签。",
    dependencies=("daily_adj.parquet",),
)
def factor_label_ret_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    return _compute_label_ret(daily_adj, 5, "label_ret_5d")


@register_factor(
    name="label_ret_10d",
    description="T+1开盘买入、T+11开盘卖出，10日目标收益。",
    category="target",
    thesis="两周持股周期，平衡信号衰减与换手成本。",
    dependencies=("daily_adj.parquet",),
)
def factor_label_ret_10d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    return _compute_label_ret(daily_adj, 10, "label_ret_10d")


@register_factor(
    name="label_ret_20d",
    description="T+1开盘买入、T+21开盘卖出，20日目标收益。",
    category="target",
    thesis="月度持股周期，IC衰减但换手成本大幅降低，适合低频截面策略。",
    dependencies=("daily_adj.parquet",),
)
def factor_label_ret_20d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    return _compute_label_ret(daily_adj, 20, "label_ret_20d")
