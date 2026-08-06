"""
日频流动性与换手异常因子。

覆盖 skill.md §3.4 建议的宽表滚动相关方案（量价相关），以及 Amihud
非流动性、换手率/量比 z-score 异常。全部为日频数据源、向量化实现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, stack_date_code


def _zscore_group(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Per-stock rolling z-score of a (Date, Code) Series."""
    mean = series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods).mean()
    )
    std = series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )
    return safe_divide(series - mean, std)


@register_factor(
    name="amihud_daily_20",
    description="Amihud日频非流动性因子（20日平均|收益|/成交额，正向排名）。",
    category="price",
    thesis="Amihud 非流动性=单位成交额对应的价格冲击，是流动性的经典度量。"
    "高非流动性股票存在流动性溢价补偿，且往往被市场忽视。"
    "与日内 amihud_intraday（分钟级口径）互补——此为日频标准口径。",
    dependencies=("daily.parquet",),
)
def factor_amihud_daily_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret_abs = daily["pct_chg"].abs() / 100.0
    amihud = safe_divide(ret_abs, daily["amount"])
    amihud20 = amihud.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(amihud20)


@register_factor(
    name="turnover_zscore_20",
    description="换手率20日Z-score因子（当日换手偏离自身20日均值的标准差数，反向排名）。",
    category="price",
    thesis="换手率异象：个股换手相对自身历史水平异常放大=交易过热、筹码换手频繁，"
    "未来收益倾向偏低（A股高换手负溢价）。z-score 消除了换手率的水平差异"
    "（低换手大盘股与高换手小盘股不可直接比较），只保留异常度。",
    dependencies=("finance.parquet",),
)
def factor_turnover_zscore_20(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"].clip(0, 100)
    z = _zscore_group(turnover, 20, 10)
    return cross_sectional_rank(-z)


@register_factor(
    name="volume_price_corr_20",
    description="量价相关因子（20日滚动收益与对数成交量的相关性，正向排名）。",
    category="price",
    thesis="量价齐升（正相关）表示放量上涨的健康趋势、资金持续参与；"
    "量价背离（负相关）=缩量上涨（动能不足）或放量下跌（派发）。"
    "采用 skill.md §3.4 的宽表 rolling corr 方案。",
    dependencies=("daily.parquet",),
)
def factor_volume_price_corr_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = daily["pct_chg"] / 100.0
    log_vol = np.log(daily["vol"].where(daily["vol"] > 0, np.nan))
    wide_r = ret.unstack("Code")
    wide_v = log_vol.unstack("Code")
    corr = wide_r.rolling(20, min_periods=10).corr(wide_v)
    return cross_sectional_rank(stack_date_code(corr))


@register_factor(
    name="volume_ratio_zscore_20",
    description="量比20日Z-score因子（当日量比偏离自身20日均值的标准差数，反向排名）。",
    category="price",
    thesis="finance.volume_ratio 为当日成交量/近5日均量；其自身 z-score 度量"
    "量比异常的持续性放大。量比连续异常高企=放量滞涨/派发风险，"
    "与 turnover_zscore_20 互为补充（一个是成交量口径、一个是换手率口径）。",
    dependencies=("finance.parquet",),
)
def factor_volume_ratio_zscore_20(context: FactorContext):
    finance = context.load("finance.parquet")
    volume_ratio = finance["volume_ratio"].clip(0, 20)
    z = _zscore_group(volume_ratio, 20, 10)
    return cross_sectional_rank(-z)
