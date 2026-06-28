from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide


# ── Limit Down (跌停) ──────────────────────────────────────────────────

@register_factor(
    name="limit_down_open_times",
    description="跌停开板次数因子截面排名（仅跌停日有值，开板多=抄底资金活跃）。",
    category="event",
    thesis="跌停被撬开代表有资金抄底，多次开板的跌停后续修复概率更高。",
    dependencies=("limit_list.parquet",),
)
def factor_limit_down_open_times(context: FactorContext):
    limit_list = context.load("limit_list.parquet")
    is_down = limit_list["limit"] == "Z"
    open_times = limit_list.loc[is_down, "open_times"].astype(float)
    return cross_sectional_rank(open_times)


@register_factor(
    name="limit_down_pct_chg",
    description="跌停跌幅因子，跌停日跌幅绝对值截面排名（跌幅越大排越前=反转预期）。",
    category="event",
    thesis="跌停幅度反映了市场恐慌程度，极端跌停后存在修复性反弹机会。",
    dependencies=("limit_list.parquet",),
)
def factor_limit_down_pct_chg(context: FactorContext):
    limit_list = context.load("limit_list.parquet")
    is_down = limit_list["limit"] == "Z"
    pct = limit_list.loc[is_down, "pct_chg"]
    return cross_sectional_rank(pct)
