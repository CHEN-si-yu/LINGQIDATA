from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank

# pledge_stat.parquet lacks ann_date (only has end_date).  Quarterly reports
# are not publicly available until weeks after period end.  We apply a 45
# trading-day shift (~2 calendar months) to approximate the reporting lag and
# prevent future data leakage.  Q4 annual reports can take up to 120 calendar
# days, so a uniform 45-day lag is conservative for Q1/Q3, correct for Q2,
# and partially mitigates Q4 leakage.
_PLEDGE_LAG = 45  # trading days

def _load_pledge(context, value_cols):
    """Load pledge_stat.parquet with reporting lag applied."""
    df = context.load_financial("pledge_stat.parquet", value_cols=value_cols, date_col="end_date")
    # Shift by reporting lag so new quarterly data only becomes available after disclosure delay.
    # After shift, ffill restores continuity using the previous quarter's data during the lag window.
    return df.groupby(level="Code").shift(_PLEDGE_LAG).groupby(level="Code").ffill()


@register_factor(
    name="pledge_ratio_momentum",
    description="质押比例动量因子，pledge_ratio季度环比变化截面排名（质押比例上升排后）。",
    category="quality",
    thesis="质押比例的边际变化比静态质押比例更具预警意义——质押比例快速上升意味着大股东资金链趋紧，后续爆仓风险增加。与pledge_ratio_change（年度同比维度）互补：一个看短期边际变化（流量），一个看年度趋势（存量变化速率）。",
    dependencies=("pledge_stat.parquet", "calendar.parquet"),
)
def factor_pledge_ratio_momentum(context: FactorContext):
    """Compute pledge_ratio QoQ change. Use diff since pledge_ratio is already a ratio."""
    pledge = _load_pledge(context, ["pledge_ratio"])
    change = pledge["pledge_ratio"].groupby(level="Code").transform(lambda s: s.diff(1))
    return cross_sectional_rank(-change)


@register_factor(
    name="pledge_concentration",
    description="质押集中度因子，pledge_count/total_share截面排名（高=质押笔数分散排前）。",
    category="quality",
    thesis="质押笔数相对总股本的比率反映质押的分散程度——笔数越多说明质押越分散（如多个小股东分别质押），单笔违约风险较低；笔数少但比率高则高度集中，单一大股东风险集中。",
    dependencies=("pledge_stat.parquet", "calendar.parquet"),
)
def factor_pledge_concentration(context: FactorContext):
    """Compute pledge_count / total_share. More fragmented pledging = less concentrated risk."""
    pledge = _load_pledge(context, ["pledge_count", "total_share"])
    concentration = pledge["pledge_count"] / pledge["total_share"].replace(0, np.nan)
    return cross_sectional_rank(concentration)


@register_factor(
    name="pledge_intensity",
    description="质押覆盖强度因子，(unrest_pledge+rest_pledge)/total_share截面排名（高=质押覆盖广排前）。",
    category="quality",
    thesis="已质押股份（含解禁+未解禁）占总股本的比例衡量股权质押的总体覆盖深度——质押覆盖越广，股价下跌触发补充质押或平仓线的风险面越大。",
    dependencies=("pledge_stat.parquet", "calendar.parquet"),
)
def factor_pledge_intensity(context: FactorContext):
    """Compute (unrest_pledge + rest_pledge) / total_share = alternative pledge coverage."""
    pledge = _load_pledge(context, ["unrest_pledge", "rest_pledge", "total_share"])
    total_pledged = pledge["unrest_pledge"] + pledge["rest_pledge"]
    intensity = total_pledged / pledge["total_share"].replace(0, np.nan)
    return cross_sectional_rank(intensity)


@register_factor(
    name="pledge_ratio_acceleration",
    description="质押比例加速度因子，-pledge_ratio变动的4季度变化（二阶导数）截面排名（质押加速上升=恶化排后）。",
    category="quality",
    thesis="质押比例的二阶导数（加速度）比一阶变化更具前瞻性——质押比例上升速度本身在加快意味着大股东压力在加速恶化，是最需要警惕的信号。负向排名。",
    dependencies=("pledge_stat.parquet", "calendar.parquet"),
)
def factor_pledge_ratio_acceleration(context: FactorContext):
    """Compute 4-quarter change of pledge_ratio QoQ change (second derivative).
    Accelerating pledge ratio = worsening stress. Rank negative.
    """
    pledge = _load_pledge(context, ["pledge_ratio"])
    change = pledge["pledge_ratio"].groupby(level="Code").transform(lambda s: s.diff(1))
    accel = change.groupby(level="Code").transform(lambda s: s.diff(4))
    return cross_sectional_rank(-accel)
