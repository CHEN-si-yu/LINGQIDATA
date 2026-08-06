"""
Momentum factors rebuilt on a point-in-time adjusted price base.

The base series ``adj = cumprod(1 + pct_chg/100)`` is built from
``daily.parquet``'s vendor-computed ``pct_chg`` (dividend-adjusted daily
return, derived from adjusted ``pre_close``).  Unlike ``daily_adj.parquet``
(forward-adjusted: future dividends rewrite the whole history), this base is
stable point-in-time — a new ex-dividend event only affects the new day and
never rewrites past values, and ex-dividend dates have no price jump.

Precision note: ``pct_chg`` is rounded to 2 decimals; after ~1500 trading
days the accumulated rounding error is ~1e-3 relative — negligible compared
to the 4.67% mean distortion of close.pct_change() on ex-dividend days.

All factors are Class 1 (daily.parquet only), vectorized, no future data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


def _adjusted_close(daily: pd.DataFrame) -> pd.Series:
    """Point-in-time adjusted close: cumprod(1 + pct_chg/100) per Code.

    pct_chg is the vendor's dividend-adjusted daily return in percent.
    Returns a Series aligned to daily's (Date, Code) index.
    """
    ret = daily["pct_chg"] / 100.0
    # 修正: 必须对 (1+r) 累乘 —— 原实现对 r 本身累乘,下溢后基座塌缩成常量 1.0
    return (1.0 + ret).groupby(level="Code").cumprod()


@register_factor(
    name="momentum_5",
    description="5日后复权动量因子（自建后复权基座，无除权失真），截面排名。",
    category="timeseries",
    thesis="基于 pct_chg 累乘的自建后复权基座的短周期动量。消除除权日跳变失真，"
    "point-in-time 稳定（新分红不回溯改写历史）。短期动量捕捉 1 周趋势动能。",
    dependencies=("daily.parquet",),
)
def factor_momentum_5(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    mom = adj.groupby(level="Code").transform(
        lambda s: s.pct_change(5, fill_method=None)
    )
    return cross_sectional_rank(mom)


@register_factor(
    name="momentum_10",
    description="10日后复权动量因子（自建后复权基座，无除权失真），截面排名。",
    category="timeseries",
    thesis="基于 pct_chg 累乘的自建后复权基座的短中期动量，捕捉 2 周趋势动能。",
    dependencies=("daily.parquet",),
)
def factor_momentum_10(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    mom = adj.groupby(level="Code").transform(
        lambda s: s.pct_change(10, fill_method=None)
    )
    return cross_sectional_rank(mom)


@register_factor(
    name="momentum_20",
    description="20日后复权动量因子（自建后复权基座，无除权失真），截面排名。",
    category="timeseries",
    thesis="经典 Jegadeesh-Titman 一个月动量区间。基于 pct_chg 累乘的自建后复权基座，"
    "A 股月度动量效应显著，且无未复权数据在除权日的跳空污染。",
    dependencies=("daily.parquet",),
)
def factor_momentum_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    mom = adj.groupby(level="Code").transform(
        lambda s: s.pct_change(20, fill_method=None)
    )
    return cross_sectional_rank(mom)


@register_factor(
    name="momentum_60",
    description="60日后复权动量因子（自建后复权基座，无除权失真），截面排名。",
    category="timeseries",
    thesis="中周期（季度）动量，捕捉中期趋势延续。基于 pct_chg 累乘的自建后复权基座，"
    "与短周期动量互补，衡量趋势的持续性而非短期动能。",
    dependencies=("daily.parquet",),
)
def factor_momentum_60(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    mom = adj.groupby(level="Code").transform(
        lambda s: s.pct_change(60, fill_method=None)
    )
    return cross_sectional_rank(mom)


@register_factor(
    name="short_term_reversal_5",
    description="5日短周期反转因子（负向5日动量），截面排名。",
    category="timeseries",
    thesis="A 股短周期（1-2 周）存在显著反转效应：近 5 日涨幅过大的股票随后回调概率高。"
    "取自建后复权基座 5 日动量的负向，排名靠前=短期超买待回落。",
    dependencies=("daily.parquet",),
)
def factor_short_term_reversal_5(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    mom5 = adj.groupby(level="Code").transform(
        lambda s: s.pct_change(5, fill_method=None)
    )
    return cross_sectional_rank(-mom5)


@register_factor(
    name="momentum_stability_20_60",
    description="20/60日动量趋势差因子（短期动量-中期动量=趋势加速度），截面排名。",
    category="timeseries",
    thesis="短周期动量与中周期动量之差视为趋势加速度：差值为正=短线强于中线（加速上行），"
    "差值转负=短线动能衰竭（减速）。区分动量水平与动量变化率，"
    "在趋势延续与反转之间提供先行信号。",
    dependencies=("daily.parquet",),
)
def factor_momentum_stability_20_60(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    g = adj.groupby(level="Code")
    mom20 = g.transform(lambda s: s.pct_change(20, fill_method=None))
    mom60 = g.transform(lambda s: s.pct_change(60, fill_method=None))
    accel = mom20 - mom60
    return cross_sectional_rank(accel)


# ── 52周高/回撤族（8.10 因未复权口径删除，自建基座上合规重建）────────────

@register_factor(
    name="price_to_52w_high",
    description="52周高点接近度因子（close/252日最高收盘价-1），截面排名。",
    category="timeseries",
    thesis="George-Hwang 52周高点效应：接近一年高点的股票在A股同样呈现动量延续，"
    "突破/接近高点时套牢盘释放完毕、上方阻力最小。基于 pct_chg 累乘的自建后复权基座，"
    "不受除权污染（8.10 曾因未复权口径删除 price_to_52w_high，此为合规重建）。",
    dependencies=("daily.parquet",),
)
def factor_price_to_52w_high(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    high_252 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(252, min_periods=120).max()
    )
    proximity = adj / high_252.replace(0, np.nan) - 1.0
    return cross_sectional_rank(proximity)


@register_factor(
    name="time_since_52w_high",
    description="距252日新高天数因子（最近一次创年内新高距今的天数，反向排名）。",
    category="timeseries",
    thesis="距创新高的时间越短，动量状态越新鲜；长期未能创新高说明趋势持续走弱。"
    "与 price_to_52w_high（接近度）互补：接近度衡量空间、时间衡量趋势新鲜度。"
    "自建后复权基座上判断新高，无除权假突破。",
    dependencies=("daily.parquet",),
)
def factor_time_since_52w_high(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    high_252 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(252, min_periods=120).max()
    )
    at_high = adj >= high_252
    dates_dt = pd.to_datetime(
        daily.index.get_level_values("Date"), format="%Y%m%d"
    )
    # Keep only True rows, forward-fill the last new-high timestamp per stock.
    hit_ts = at_high.where(at_high).mul(dates_dt.astype("int64"))
    last_hit_ts = hit_ts.groupby(level="Code").ffill()
    days_since = (dates_dt.astype("int64") - last_hit_ts) / 86_400_000_000_000
    return cross_sectional_rank(-days_since)


@register_factor(
    name="drawdown_60",
    description="60日最大回撤因子（当前价格相对60日窗口峰值回撤，反向排名）。",
    category="timeseries",
    thesis="回撤深度反映近期下行风险与筹码套牢程度：回撤浅的股票趋势完整、"
    "上行斜率健康，回撤深的股票面临解套抛压。基于自建后复权基座，"
    "无未复权数据除权日的假回撤（8.10 删除的 max_drawdown_120 的合规重建）。",
    dependencies=("daily.parquet",),
)
def factor_drawdown_60(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    peak = adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    )
    drawdown = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0
    # drawdown is non-positive, so ranking it directly puts values closest to
    # zero (the shallowest drawdowns) first, as required by the factor thesis.
    return cross_sectional_rank(drawdown)


@register_factor(
    name="drawdown_120",
    description="120日最大回撤因子（当前价格相对120日窗口峰值回撤，反向排名）。",
    category="timeseries",
    thesis="中期（半年）回撤深度衡量趋势的完整性与牛熊状态：回撤浅=仍处上行趋势，"
    "回撤深=趋势已破坏。与 drawdown_60 互补短中期风险维度。"
    "基于自建后复权基座，无除权假回撤。",
    dependencies=("daily.parquet",),
)
def factor_drawdown_120(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    peak = adj.groupby(level="Code").transform(
        lambda s: s.rolling(120, min_periods=60).max()
    )
    drawdown = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0
    return cross_sectional_rank(drawdown)


@register_factor(
    name="ulcer_index_20",
    description="溃疡指数因子（20日窗口内回撤平方均值的平方根，反向排名）。",
    category="timeseries",
    thesis="溃疡指数度量回撤的深度与持续性的综合风险（回撤面积）："
    "它同时惩罚大回撤和长时间不回本，比单一最大回撤更平滑稳健。"
    "低溃疡指数=持有体验好、趋势平稳，A股稳健溢价支持其正alpha。"
    "基于自建后复权基座，无除权失真（8.10 删除的 ulcer_index_20 的合规重建）。",
    dependencies=("daily.parquet",),
)
def factor_ulcer_index_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    peak = adj.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    dd = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0
    dd_sq = (dd ** 2).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    ulcer = dd_sq.pow(0.5)
    return cross_sectional_rank(-ulcer)
