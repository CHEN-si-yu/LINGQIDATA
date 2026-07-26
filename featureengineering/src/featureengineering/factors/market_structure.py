from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_max,
    rolling_group_mean,
    rolling_group_min,
    rolling_group_std,
    rolling_group_sum,
    safe_divide,
)

# ── Spread proxies ──────────────────────────────────────────────────────────

@register_factor(
    name="roll_spread_20",
    description="20日Roll价差估计因子 (高spread排后, 负向)",
    category="market_structure",
    thesis="Roll(1984)模型通过价格反转的协方差估计有效价差，高spread意味着高交易成本",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def roll_spread_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    close = daily["close"]
    ret = close.groupby(level="Code").pct_change()

    def _roll_spread(s, window):
        s = s.dropna()
        if len(s) < window:
            return pd.Series(np.nan, index=s.index)
        cov = s.rolling(window, min_periods=max(1, window // 2)).cov(s.shift(1))
        cov_neg = cov.clip(upper=0)
        spread = 2 * np.sqrt(-cov_neg)
        return spread

    spread = ret.groupby(level="Code").transform(lambda s: _roll_spread(s, 20))
    return cross_sectional_rank(-spread)

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

# ── Market impact / liquidity depth ──────────────────────────────────────

@register_factor(
    name="amihud_illiq_5",
    description="Amihud非流动性5日版因子截面排名。",
    category="price",
    thesis="5日窗口的Amihud非流动性指标比20日版对短期流动性冲击更敏感——捕捉近期的价格冲击成本和流动性恶化。",
    dependencies=("daily_adj.parquet",),
)
def factor_amihud_illiq_5(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    amount = daily_adj["amount"]
    illiq = (ret.abs() / amount.replace(0, np.nan))
    illiq_5 = illiq.groupby(level="Code").transform(lambda s: s.rolling(5, min_periods=3).mean())
    return cross_sectional_rank(illiq_5)

@register_factor(
    name="price_impact_proxy",
    description="价格冲击代理因子，-(|ret|/turnover)截面排名（高冲击=流动性差排后）。",
    category="price",
    thesis="单位换手率引起的价格变动幅度——价格冲击越大意味着流动性越差、交易成本越高，投资者会要求更高的流动性补偿(高收益期望)。",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def factor_price_impact_proxy(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    finance = context.load("finance.parquet")
    ret = daily_adj["close"].groupby(level="Code").transform(lambda s: s.pct_change(1)).abs()
    turnover = finance["turnover_rate"]
    common = ret.index.intersection(turnover.index)
    impact = ret.loc[common] / turnover.loc[common].replace(0, np.nan)
    return cross_sectional_rank(impact)

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
