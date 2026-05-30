from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


@register_factor(
    name="margin_turnover_ratio",
    description="融资周转率因子，融资买入额/融资余额截面排名（高周转=投机性强排前）。",
    category="fund_flow",
    thesis="融资买入额相对融资余额的比率反映融资资金的活跃度和周转速度——高周转意味着融资资金快进快出、投机属性强，低周转则意味着持仓稳定。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_turnover_ratio(context: FactorContext):
    """Compute rzmre / rzye = margin turnover rate. High turnover = speculative."""
    margin = context.load("margin_detail.parquet")
    ratio = margin["rzmre"] / margin["rzye"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


@register_factor(
    name="margin_short_ratio_change_5d",
    description="融券占比5日变化因子，-(rqye/rzrqye)的5日变化截面排名（融券占比上升=看空情绪增强排后）。",
    category="fund_flow",
    thesis="融券余额占两融总余额的比例上升意味着空头力量在持续增强——越来越多的杠杆资金选择做空而非做多，是市场情绪转向的领先指标。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_short_ratio_change_5d(context: FactorContext):
    """Compute 5d change in rqye/rzrqye ratio. Rank negative (rising short ratio = bearish)."""
    margin = context.load("margin_detail.parquet")
    short_ratio = margin["rqye"] / margin["rzrqye"].replace(0, np.nan)
    change_5d = short_ratio.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(-change_5d)


@register_factor(
    name="margin_net_flow_acceleration",
    description="融资净流入加速度因子，(rzmre-rzche)/rzye的5日变化截面排名（正加速度=融资买入加速排前）。",
    category="fund_flow",
    thesis="融资净开仓率(rzmre-rzche)/rzye的边际变化比单日净流量更具信息量——净买入持续加速意味着杠杆资金在加大多头押注，是趋势加强信号。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_net_flow_acceleration(context: FactorContext):
    """Compute 5d change of (rzmre - rzche) / rzye = acceleration of margin net buying."""
    margin = context.load("margin_detail.parquet")
    net_open = (margin["rzmre"] - margin["rzche"]) / margin["rzye"].replace(0, np.nan)
    accel = net_open.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(accel)


@register_factor(
    name="margin_short_crowding",
    description="融券拥挤度因子，-rqye/流通市值截面排名（融券余额高=做空拥挤风险大排后）。",
    category="fund_flow",
    thesis="融券余额相对流通市值的比例衡量做空拥挤度——高比例意味着大量股票被借券做空，一旦出现利好消息空头被迫回补可能引发轧空行情，但短期仍以风险信号为主。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_margin_short_crowding(context: FactorContext):
    """Compute rqye / float_mv (circ_mv). High short balance relative to float cap = short squeeze candidate.
    Rank negative.
    """
    margin = context.load("margin_detail.parquet")
    finance = context.load("finance.parquet")

    rqye = margin["rqye"]
    circ_mv = finance["circ_mv"]

    common = rqye.index.intersection(circ_mv.index)
    crowding = rqye.loc[common] / circ_mv.loc[common].replace(0, np.nan)
    return cross_sectional_rank(-crowding)


@register_factor(
    name="margin_sentiment_composite",
    description="融资情绪综合因子，rzmre/(rzmre+rzche)-0.5截面排名（正=净买入情绪排前）。",
    category="fund_flow",
    thesis="融资买入额占融资买入+偿还总额的比例反映杠杆资金的多空倾向——比例>0.5意味着融资者以买入为主、情绪偏乐观，<0.5则偿还主导、情绪偏谨慎。与margin_net_open互补：一个看构成比例，一个看差额。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_sentiment_composite(context: FactorContext):
    """Compute rzmre / (rzmre + rzche) - 0.5. >0 = net buying sentiment."""
    margin = context.load("margin_detail.parquet")
    denom = (margin["rzmre"] + margin["rzche"]).replace(0, np.nan)
    sentiment = margin["rzmre"] / denom - 0.5
    return cross_sectional_rank(sentiment)
