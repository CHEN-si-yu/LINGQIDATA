"""
杠杆资金结构因子 — Class 1 (margin_detail.parquet + finance.parquet)。

⚠️ margin 跨源混算对齐(2026-08-05 审计):margin 面板在数据层已统一
groupby(level="Code").shift(1)(Date=T 上是原始 T-1 值),与 finance 的
total_mv 混算时,分母必须同步 shift(1),否则 1 日错配。

覆盖率:margin_detail 覆盖约 85% 股票 → 因子 NaN ≈ 15%,符合 skill.md §2.1。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


@register_factor(
    name="margin_leverage_change_20d",
    description="融资杠杆变化因子：融资余额/流通市值占比的20日变化截面排名（杠杆资金加仓排前）。",
    category="fund_flow",
    thesis="融资余额相对流通市值的占比是杠杆资金参与度的标准度量——占比20日上升="
           "杠杆资金持续加仓(增量资金确认,常伴随行情启动);下降=杠杆资金撤退。"
           "与总杠杆水平(total_leverage_ratio)互补:本因子捕捉方向而非水平。"
           "分母 total_mv 同步 shift(1) 与 margin 面板对齐,无1日错配。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_margin_leverage_change_20d(context: FactorContext):
    m = context.load("margin_detail.parquet")
    fin = context.load("finance.parquet")
    total_mv_lag = fin["total_mv"].groupby(level="Code").shift(1)
    # ⚠️ 必须 reindex 到 margin 面板索引:直接相除会做索引并集,
    # 把无融资数据股票的 25% 缺行以 NaN 拉进结果(与既有 margin 因子同口径,
    # .fea 仅覆盖融资标的,NaN 由覆盖范围决定而非对齐方式)。
    total_mv_m = total_mv_lag.reindex(m.index)
    leverage = safe_divide(m["rzye"], total_mv_m)
    change = leverage.groupby(level="Code").diff(20)
    return cross_sectional_rank(change)


@register_factor(
    name="short_balance_ratio_change_20d",
    description="融券余额占比变化因子：融券余额/流通市值占比的20日变化截面排名（负向，空头加仓排后）。",
    category="fund_flow",
    thesis="融券余额相对流通市值的占比上升=看空力量持续加码(空头筹码累积),"
           "其后若轧空则涨幅剧烈;占比下降=空头回补。与融券成交类因子(short_sell_volume_ratio)"
           "互补:本因子刻画存量而非流量。占比为0的股票变化也为0(有效信息),"
           "仅覆盖约85%有融资融券资格的股票。",
    dependencies=("margin_detail.parquet", "finance.parquet"),
)
def factor_short_balance_ratio_change_20d(context: FactorContext):
    m = context.load("margin_detail.parquet")
    fin = context.load("finance.parquet")
    total_mv_lag = fin["total_mv"].groupby(level="Code").shift(1)
    # 与 margin_leverage_change_20d 同口径:reindex 到 margin 面板索引,
    # 避免索引并集引入 25% 缺行 NaN。
    total_mv_m = total_mv_lag.reindex(m.index)
    short_ratio = safe_divide(m["rqye"], total_mv_m)
    change = short_ratio.groupby(level="Code").diff(20)
    return cross_sectional_rank(-change)
