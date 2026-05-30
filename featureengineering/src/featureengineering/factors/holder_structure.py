from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


@register_factor(
    name="holder_num_acceleration",
    description="股东户数变化加速度因子，-holder_num二阶段变化率截面排名（户数减少加速=吸筹加快排前）。",
    category="financial",
    thesis="股东户数变化率本身是筹码集中度的流量指标，而户数变化的加速度是筹码集中度变化的方向性信号——下降加速意味着主力吸筹速度在提升，后续股价上涨动力更强。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_acceleration(context: FactorContext):
    """Compute second-order QoQ change of holder_num. Negative accel = speeding up concentration."""
    holder = context.load_financial(
        "holder_number.parquet", value_cols=["holder_num"], date_col="end_date"
    )
    change = holder["holder_num"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    accel = change.groupby(level="Code").transform(lambda s: s.diff(1))
    return cross_sectional_rank(-accel)


@register_factor(
    name="holder_num_trend_4q",
    description="股东户数趋势因子，-holder_num近4季度线性回归斜率截面排名（户数持续下降=趋势性集中排前）。",
    category="financial",
    thesis="4季度滚动回归斜率比单季度变化率更可靠——过滤了季报时点的噪音，捕捉股东户数的趋势性变化方向。下降斜率为负=筹码持续集中=中长期看好。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_trend_4q(context: FactorContext):
    """Compute linear regression slope of holder_num over last 4 quarters. Rank negative slope."""
    holder = context.load_financial(
        "holder_number.parquet", value_cols=["holder_num"], date_col="end_date"
    )

    def _trend_slope(y):
        y = y[~np.isnan(y)]
        if len(y) < 3:
            return np.nan
        x = np.arange(len(y), dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = (x * x).sum()
        if denom == 0:
            return np.nan
        return (x * y).sum() / denom

    slope = holder["holder_num"].groupby(level="Code").transform(
        lambda s: s.rolling(4, min_periods=3).apply(_trend_slope, raw=True)
    )
    return cross_sectional_rank(-slope)


@register_factor(
    name="holder_avg_mv",
    description="户均市值因子，总市值/股东户数截面排名（户均市值高=机构化程度高排前）。",
    category="financial",
    thesis="户均市值是衡量股东结构机构化程度的有效指标——户均市值越高代表大户/机构持股比例越高，机构重仓股往往有更好的基本面支撑和更低的波动性。",
    dependencies=("holder_number.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_holder_avg_mv(context: FactorContext):
    """Compute total_mv / holder_num = average market value per holder. Higher = more institutional."""
    holder = context.load_financial(
        "holder_number.parquet", value_cols=["holder_num"], date_col="end_date"
    )
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"]
    holder_num = holder["holder_num"]
    common = total_mv.index.intersection(holder_num.index)
    avg_mv = total_mv.loc[common] / holder_num.loc[common].replace(0, np.nan)
    return cross_sectional_rank(avg_mv)


@register_factor(
    name="holder_price_divergence",
    description="量价背离因子，sign(price_change)*(-sign(holder_change))截面排名（价涨量缩=主力吸筹排前）。",
    category="financial",
    thesis="价格上涨同时股东户数下降（量价背离）是典型的主力吸筹特征——聪明钱在上涨中收集筹码，后续趋势延续性强。价涨量增则可能意味着散户追涨，跟风盘过多。",
    dependencies=("holder_number.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_holder_price_divergence(context: FactorContext):
    """Compute sign(price_change) * (-sign(holder_change)). Positive = whale accumulation signal."""
    holder = context.load_financial(
        "holder_number.parquet", value_cols=["holder_num"], date_col="end_date"
    )
    daily_adj = context.load("daily_adj.parquet")

    holder_change = holder["holder_num"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    price_change = daily_adj["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))

    common = holder_change.index.intersection(price_change.index)
    divergence = np.sign(price_change.loc[common]) * (-np.sign(holder_change.loc[common]))
    return cross_sectional_rank(divergence)


@register_factor(
    name="holder_num_stability",
    description="股东户数稳定性因子，-holder_num环比变化率4季度滚动标准差截面排名（户数稳定=筹码锁定好排前）。",
    category="financial",
    thesis="股东户数频繁大幅波动代表筹码不稳定、短线资金进出频繁，而户数稳定说明持仓者结构清晰、中长期资金占主导，是质量信号。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_stability(context: FactorContext):
    """Compute rolling std of holder_num QoQ change over 4 quarters. Rank negative (unstable = bad)."""
    holder = context.load_financial(
        "holder_number.parquet", value_cols=["holder_num"], date_col="end_date"
    )
    change = holder["holder_num"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    stability = change.groupby(level="Code").transform(
        lambda s: s.rolling(4, min_periods=3).std()
    )
    return cross_sectional_rank(-stability)
