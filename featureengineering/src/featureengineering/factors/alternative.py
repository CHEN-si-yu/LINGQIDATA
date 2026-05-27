from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


@register_factor(
    name="pledge_risk",
    description="股权质押风险因子，-pledge_ratio截面排名（高质押比例=风险信号排后）。",
    category="quality",
    thesis="高股权质押比例=大股东资金链紧张/爆仓风险/潜在控制权转移，是尾部风险预警信号。",
    dependencies=("pledge_stat.parquet", "calendar.parquet"),
)
def factor_pledge_risk(context: FactorContext):
    pledge = context.load_financial(
        "pledge_stat.parquet", value_cols=["pledge_ratio"], date_col="end_date"
    )
    return cross_sectional_rank(-pledge["pledge_ratio"])


@register_factor(
    name="overnight_gap",
    description="隔夜跳空因子，-(open-pre_close)/pre_close截面排名（跳空高开=反转信号排后）。",
    category="price",
    thesis="A股隔夜跳空存在均值回归特征，大幅高开/低开后倾向于日内反转。",
    dependencies=("daily.parquet",),
)
def factor_overnight_gap(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
    return cross_sectional_rank(-gap)


@register_factor(
    name="intraday_ret",
    description="日内收益率因子，(close-open)/open截面排名。",
    category="price",
    thesis="日内收益与隔夜收益相关性低，提供独立于传统动量的alpha维度。强势股日内持续走高。",
    dependencies=("daily.parquet",),
)
def factor_intraday_ret(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = (daily["close"] - daily["open"]) / daily["open"].replace(0, np.nan)
    return cross_sectional_rank(ret)
