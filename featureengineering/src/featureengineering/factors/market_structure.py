from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


@register_factor(
    name="circ_mv_to_total_mv",
    description="流通市值/总市值截面排名（高流通比排前）。",
    category="valuation",
    thesis="流通市值占总市值的比例——高流通比意味着限售股压力小、流通性好，低流通比意味着未来解禁后有大量潜在抛压。",
    dependencies=("finance.parquet",),
)
def factor_circ_mv_to_total_mv(context: FactorContext):
    finance = context.load("finance.parquet")
    ratio = finance["circ_mv"] / finance["total_mv"].replace(0, np.nan)
    return cross_sectional_rank(ratio)

@register_factor(
    name="market_cap_concentration_20d",
    description="市值集中度因子，log_total_mv的20日波动率截面排名。",
    category="valuation",
    thesis="市值在短期内的剧烈波动反映公司基本面的不确定性——市值频繁大幅波动意味着市场对公司价值的共识度低、信息不对称高。",
    dependencies=("finance.parquet",),
)
def factor_market_cap_concentration_20d(context: FactorContext):
    finance = context.load("finance.parquet")
    mv = np.log(finance["total_mv"].replace(0, np.nan))
    mv_vol_20 = mv.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-mv_vol_20)
