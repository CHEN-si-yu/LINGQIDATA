"""Behavioral finance and sentiment factors.

Retail attention proxies, turnover anomaly, and dispersion.
(52-week anchoring factors removed 2026-07-31: unadjusted close is
distorted by ex-dividend jumps — see skill.md §8.)
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean


# ── Retail attention proxy ──────────────────────────────────────────────

@register_factor(
    name="retail_attention",
    description="散户关注度代理因子，异常高换手率(当日换手/20日均换手-1)与大单净流出交乘截面排名（取负向）。",
    category="price",
    thesis="高换手+大单流出=散户接盘信号。机构通过大单出货、散户通过中小单接盘，这种成交量结构预示后续下跌。",
    dependencies=("finance.parquet", "main_fund_flow.parquet"),
)
def factor_retail_attention(context: FactorContext):
    fin = context.load("finance.parquet")
    ff = context.load("main_fund_flow.parquet")
    turnover = fin["turnover_rate"]
    to_20_mean = rolling_group_mean(turnover, 20)
    abnormal_to = turnover / to_20_mean.replace(0, np.nan) - 1.0

    # Total amount and big order net
    total_amount = (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)
    big_net = (ff["buy_lg_amount"] + ff["buy_elg_amount"]
               - ff["sell_lg_amount"] - ff["sell_elg_amount"])
    big_net_ratio = big_net / total_amount

    common = abnormal_to.index.intersection(big_net_ratio.index)
    # Retail attention = high abnormal turnover + big order selling
    signal = abnormal_to.loc[common] * (-big_net_ratio.loc[common])
    return cross_sectional_rank(-signal)


# ── Turnover anomaly ────────────────────────────────────────────────────

@register_factor(
    name="turnover_anomaly_20",
    description="换手率异常因子，20日均换手/60日均换手-1截面排名（取负向=异常高换手排后）。",
    category="price",
    thesis="换手率短期飙升往往伴随投机性交易或信息事件，高异常换手率预示短期反转。中长期低换手率则与低波动溢价相关。",
    dependencies=("finance.parquet",),
)
def factor_turnover_anomaly_20(context: FactorContext):
    fin = context.load("finance.parquet")
    turnover = fin["turnover_rate"]
    to_20 = rolling_group_mean(turnover, 20)
    to_60 = rolling_group_mean(turnover, 60)
    anomaly = to_20 / to_60.replace(0, np.nan) - 1.0
    return cross_sectional_rank(-anomaly)



@register_factor(
    name="turnover_anomaly_mean_20d",
    description="换手率异象因子，-(近20日平均换手率)截面排名（高换手=投机性强排后）。",
    category="price",
    thesis="A股换手率异象是全球最显著的——高换手率股票后续收益显著偏低，原因在于散户过度交易和投机炒作后的均值回复。",
    dependencies=("finance.parquet",),
)
def factor_turnover_anomaly_mean_20d(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    to_20 = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-to_20)
