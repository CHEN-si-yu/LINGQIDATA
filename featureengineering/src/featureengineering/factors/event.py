from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── Dragon Tiger (龙虎榜) ───────────────────────────────────────────────

@register_factor(
    name="dragon_tiger_net_rate",
    description="龙虎榜净买率因子截面排名（仅龙虎榜上榜日有值）。",
    category="event",
    thesis="龙虎榜净买率是上榜股票当日机构/游资博弈的综合结果，高净买率代表多方占优。",
    dependencies=("top_list.parquet",),
)
def factor_dragon_tiger_net_rate(context: FactorContext):
    top_list = context.load("top_list.parquet")
    return cross_sectional_rank(top_list["net_rate"])


@register_factor(
    name="dragon_tiger_amount_rate",
    description="龙虎榜成交占比因子截面排名。",
    category="event",
    thesis="龙虎榜成交额占总成交比例越高，说明上榜席位的定价权越大。",
    dependencies=("top_list.parquet",),
)
def factor_dragon_tiger_amount_rate(context: FactorContext):
    top_list = context.load("top_list.parquet")
    return cross_sectional_rank(top_list["amount_rate"])


@register_factor(
    name="dragon_tiger_buy_sell_ratio",
    description="龙虎榜买卖比因子，l_buy/l_sell截面排名。",
    category="event",
    thesis="龙虎榜买入金额相对卖出的比例反映多空力量对比。",
    dependencies=("top_list.parquet",),
)
def factor_dragon_tiger_buy_sell_ratio(context: FactorContext):
    top_list = context.load("top_list.parquet")
    ratio = top_list["l_buy"] / top_list["l_sell"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Limit Up (涨停) ─────────────────────────────────────────────────────

@register_factor(
    name="limit_up_consecutive",
    description="连板天数因子，当日涨停股票的连板天数截面排名。",
    category="event",
    thesis="连板高度是涨停强度的核心指标，但极端连板后存在显著的反转压力。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_consecutive(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    return cross_sectional_rank(limit_up["consecutive_days"].astype(float))


@register_factor(
    name="limit_up_sealed_flow_ratio",
    description="涨停封单流比因子，封单流比截面排名（仅涨停日有值）。",
    category="event",
    thesis="封单流比（封单额/流通市值）是封板质量的核心指标，高封流比预示次日溢价。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_sealed_flow_ratio(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    ratio = limit_up["sealed_flow_ratio"]
    return cross_sectional_rank(ratio)


@register_factor(
    name="limit_up_open_count_neg",
    description="涨停开板次数负向因子，开板次数越多质量越差。",
    category="event",
    thesis="开板次数多代表封板不牢、抛压大，次日溢价空间有限。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_open_count_neg(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    return cross_sectional_rank(-limit_up["open_count"])


@register_factor(
    name="limit_up_sealed_amount",
    description="涨停封单金额因子，封单额截面排名。",
    category="event",
    thesis="封单金额反映封板资金的绝对力度，大封单是强势涨停的标志。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_sealed_amount(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    return cross_sectional_rank(limit_up["sealed_amount"])


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


# ── Event recency ───────────────────────────────────────────────────────

@register_factor(
    name="top_list_turnover_intensity",
    description="龙虎榜换手强度因子，上榜日换手率截面排名。",
    category="event",
    thesis="龙虎榜上榜时的高换手率通常意味着多空激烈博弈，博弈后的方向选择有预测价值。",
    dependencies=("top_list.parquet",),
)
def factor_top_list_turnover_intensity(context: FactorContext):
    top_list = context.load("top_list.parquet")
    return cross_sectional_rank(top_list["turnover_rate"])


# ── Dragon Tiger seat-level aggregation ──────────────────────────────────

@register_factor(
    name="dt_org_count",
    description="龙虎榜机构席位数量因子，当日上榜的机构专用席位数量截面排名。",
    category="event",
    thesis="机构席位的参与数量反映机构投资者对异动股票的关注度和参与深度，机构参与的股票后市表现更稳健。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_org_count(context: FactorContext):
    dt = context.load("dragon_tiger.parquet")
    is_inst = dt["org_name"].str.contains("机构专用", na=False)
    # Count institutional seats per stock-date, convert to float
    inst_count = (
        is_inst.groupby(level=["Date", "Code"]).sum().astype(float)
    )
    return cross_sectional_rank(inst_count)


@register_factor(
    name="dt_top_net_rate",
    description="龙虎榜前五大席位净买入占比因子，按净买入额排序取top5的净买入合计/总净买入合计截面排名。",
    category="event",
    thesis="前五大席位的净买入集中度反映核心参与者的方向一致性和力度，高集中度净买入是强信号。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_top_net_rate(context: FactorContext):
    dt = context.load("dragon_tiger.parquet")
    net = dt["net_buy_amount"]

    def _top5_share(grp):
        top5_net = grp.nlargest(5).sum()
        total_abs = grp.abs().sum()
        if total_abs == 0:
            return np.nan
        return top5_net / total_abs

    top5 = net.groupby(level=["Date", "Code"]).apply(_top5_share)
    # groupby.apply on MultiIndex may produce an extra level; ensure 1-D Series
    if isinstance(top5, pd.DataFrame):
        top5 = top5.iloc[:, 0]
    if isinstance(top5.index, pd.MultiIndex) and top5.index.nlevels > 2:
        top5 = top5.droplevel([0])
    top5.name = "dt_top_net_rate"
    return cross_sectional_rank(top5)
