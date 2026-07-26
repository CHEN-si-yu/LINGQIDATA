"""
Short selling factors — Class 1 panel factors.

Uses margin_detail.parquet short-selling columns:
  - rqye: short-selling balance (融券余额)
  - rqmcl: short-selling volume (融券卖出量)
  - rqyl: short-selling remaining shares (融券余量)

Also uses rzye for margin-short divergence factors.
Coverage: ~85% of stocks for margin; short-side fields have similar coverage.
All factors use context.load() for daily-panel access to margin_detail.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Short Interest Level Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="short_balance_5d",
    description="融券余额5日变化率。余额快速增加=看空情绪升温，排名高(取正)。",
    category="fund_flow",
    thesis="融券余额(rqye)5日变化率反映做空力量的短期变化。融券余额快速增加意味着"
           "看空情绪升温、做空力量增强，是短期负向信号；余额下降意味着空头回补、看空缓解。",
    dependencies=("margin_detail.parquet",),
)
def factor_short_balance_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    chg = m["rqye"].groupby(level="Code").transform(lambda s: s.pct_change(5))
    chg = chg.clip(-0.5, 1.0)
    return cross_sectional_rank(chg)


@register_factor(
    name="short_balance_20d",
    description="融券余额20日变化率排名取反。持续增长=空头压力累积，排名低。",
    category="fund_flow",
    thesis="融券余额20日变化率反映中期做空趋势。融券余额持续增长意味着累积的空头压力增大、"
           "后市承压概率高；余额持续下降意味着空头逐步离场。排名取反使得空头压力越大排名越低。",
    dependencies=("margin_detail.parquet",),
)
def factor_short_balance_20d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    chg = m["rqye"].groupby(level="Code").transform(lambda s: s.pct_change(20))
    chg = chg.clip(-0.5, 1.0)
    return cross_sectional_rank(-chg)


@register_factor(
    name="short_interest_change_5d",
    description="融券余量5日变化率取反。余量增加=新增卖空力量，排名低。",
    category="fund_flow",
    thesis="融券余量(rqyl,尚未偿还的融券股数)5日变化率取反排名。"
           "融券余量增加意味着新增卖空力量(新开仓)超出平仓力量，是看空信号；"
           "余量减少意味着空头集中平仓(空头回补)，可能推动股价上涨。",
    dependencies=("margin_detail.parquet",),
)
def factor_short_interest_change_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    chg = m["rqyl"].groupby(level="Code").transform(lambda s: s.pct_change(5))
    chg = chg.clip(-0.5, 1.0)
    return cross_sectional_rank(-chg)


# ═══════════════════════════════════════════════════════════════════════════════
# Short Flow Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="short_sell_intensity_5d",
    description="融券卖出5日均量相对融券余额的强度取反。高强度=空头攻击力度大，排名低。",
    category="fund_flow",
    thesis="融券卖出5日均量相对融券余额的强度。高强度意味着近期做空交易极其活跃、"
           "空头攻击力度大——可能出现持续的下跌压力。排名取反使得空头压力越大排名越低。",
    dependencies=("margin_detail.parquet",),
)
def factor_short_sell_intensity_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    sell_ma5 = m["rqmcl"].groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    intensity = safe_divide(sell_ma5, m["rqye"])
    intensity = intensity.clip(0, 10)
    return cross_sectional_rank(-intensity)


# ═══════════════════════════════════════════════════════════════════════════════
# Total Leverage Indicator / Margin-Short Divergence
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="margin_short_divergence_20d",
    description="融资余额与融券余额20日变化率差值。正值=多头占优，排名高。",
    category="fund_flow",
    thesis="融资余额20日变化率减去融券余额20日变化率。正值=融资增长快于融券"
           "(看多力量占优)，负值=融券增长快于融资(看空力量占优)。"
           "多空两股力量的相对变化是市场情绪的先行指标——融资加速+融券减速=强烈看多。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_short_divergence_20d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    rzye_chg = m["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(20))
    rqye_chg = m["rqye"].groupby(level="Code").transform(lambda s: s.pct_change(20))
    divergence = rzye_chg - rqye_chg
    divergence = divergence.clip(-1.0, 1.0)
    return cross_sectional_rank(divergence)
