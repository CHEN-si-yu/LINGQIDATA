from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean


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


# ── Limit-up turnover tightness ────────────────────────────────────────


@register_factor(
    name="limit_up_turnover_tightness",
    description="封板换手紧密度因子，涨停封板金额/流通市值的截面排名。",
    category="event",
    thesis="封板换手率衡量涨停板上的实际筹码交换深度——低封板换手率意味着卖方惜售、封板坚决（正向，强势持续），高封板换手率意味着大量筹码在涨停价获利了结（负向，封板松动预警）。该指标与consecutive_days互补：一个看封板质量，一个看封板数量。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_turnover_tightness(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    return cross_sectional_rank(-limit_up["sealed_turnover_ratio"])


# ── Top list concentration ─────────────────────────────────────────────


@register_factor(
    name="top_list_concentration",
    description="龙虎榜成交集中度因子，(龙虎榜成交额/流通市值)截面排名。",
    category="event",
    thesis="龙虎榜成交额占流通市值的比例衡量上榜期间的筹码集中转移深度——高占比意味着定价权从连续竞价转移到龙虎榜大资金手中。与top_list_turnover_intensity互补：一个看龙虎榜成交占总成交比（换手率维度），一个看龙虎榜成交占流通市值比（筹码转移维度）。",
    dependencies=("top_list.parquet", "finance.parquet"),
)
def factor_top_list_concentration(context: FactorContext):
    top_list = context.load("top_list.parquet")
    finance = context.load("finance.parquet")
    l_amount = top_list["l_amount"]
    circ_mv = finance["circ_mv"]
    common = l_amount.index.intersection(circ_mv.index)
    concentration = l_amount.loc[common] / circ_mv.loc[common].replace(0, np.nan)
    return cross_sectional_rank(concentration)


# ── Dragon Tiger Deep ────────────────────────────────────────────────────


@register_factor(
    name="dragon_tiger_buyer_concentration",
    description="龙虎榜买方集中度因子，l_buy/abs(l_amount)截面排名（买方集中=定价权集中排前）。",
    category="event",
    thesis="龙虎榜买方集中度反映单一买方主导程度——高买方集中度意味着有强势资金在定向收集筹码，后续推动股价的意愿和能力更强。分散买方则缺乏统一方向性。",
    dependencies=("top_list.parquet",),
)
def factor_dragon_tiger_buyer_concentration(context: FactorContext):
    top_list = context.load("top_list.parquet")
    conc = top_list["l_buy"] / top_list["l_amount"].abs().replace(0, np.nan)
    return cross_sectional_rank(conc)


@register_factor(
    name="dragon_tiger_retail_inst_divergence",
    description="龙虎榜散户机构背离因子，(l_buy-l_sell)/abs(l_amount)截面排名。",
    category="event",
    thesis="龙虎榜买卖净额方向与规模综合反映机构vs散户博弈结果——净买入占比高且规模大表明机构资金主导，反之则散户抛压占优。",
    dependencies=("top_list.parquet",),
)
def factor_dragon_tiger_retail_inst_divergence(context: FactorContext):
    top_list = context.load("top_list.parquet")
    net_ratio = (top_list["l_buy"] - top_list["l_sell"]) / top_list["l_amount"].abs().replace(0, np.nan)
    return cross_sectional_rank(net_ratio)


# ── Limit Up Deep ────────────────────────────────────────────────────────


@register_factor(
    name="limit_seal_speed",
    description="涨停封板速度因子，封板距离开盘分钟数截面排名（封板越早=质量越高排前）。",
    category="event",
    thesis="封板时间越早（如9:35封板远优于14:50封板），多方抢筹意愿越强、空方抛压越轻。早盘秒板的股票通常具有最强的次日溢价和最高的连板概率，反映资金的抢筹紧迫度。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_seal_speed(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    first_time = limit_up["first_limit_time"].astype(str)
    # Parse HH:MM:SS to minutes from 9:30 (market open)
    hour = pd.to_numeric(first_time.str[:2], errors="coerce")
    minute = pd.to_numeric(first_time.str[3:5], errors="coerce")
    mins_from_open = (hour - 9) * 60 + minute - 30
    mins_from_open = mins_from_open.clip(lower=0)
    return cross_sectional_rank(-mins_from_open)


@register_factor(
    name="limit_seal_stability",
    description="涨停封板稳定性因子，-open_count/consecutive_days截面排名（开板次数越少=封板越稳排前）。",
    category="event",
    thesis="连板过程中开板次数是封板质量的核心维度——开板次数多说明封板过程反复、抛压大，即使最终封板，次日溢价空间也有限。与limit_seal_speed互补：一个看封板快慢，一个看封板过程中的稳定性。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_seal_stability(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    cd = limit_up["consecutive_days"].replace(0, np.nan)
    oc = limit_up["open_count"]
    stability = oc / cd
    return cross_sectional_rank(-stability)


@register_factor(
    name="limit_consecutive_deceleration",
    description="连板衰减因子，-连板天数变化率截面排名（连板加速=强势持续排前）。",
    category="event",
    thesis="连续涨停中的加速/减速反映资金的追涨热情变化——如果今天涨停但连板天数的加速度在下降（从连板3天缩到2天），说明追涨资金在退潮，短期顶部概率上升。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_consecutive_deceleration(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    cd = limit_up["consecutive_days"].astype(float)
    chg = cd.groupby(level="Code").transform(lambda s: s.diff(1))
    return cross_sectional_rank(-chg)


# ── Limit List ───────────────────────────────────────────────────────────


@register_factor(
    name="limit_down_intensity",
    description="跌停强度因子，-跌停次数截面排名（跌停频繁=风险信号排后）。",
    category="event",
    thesis="跌停频率是尾部风险的最直接度量——频繁触及跌停的股票在基本面和流动性上均存在结构性问题，后续收益显著为负。",
    dependencies=("limit_list.parquet",),
)
def factor_limit_down_intensity(context: FactorContext):
    limit_list = context.load("limit_list.parquet")
    is_down = (limit_list["limit"] == "D").astype(float)
    # Count down-limit occurrences per stock over last 60 trading days
    down_count = is_down.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=1).sum()
    )
    return cross_sectional_rank(-down_count)


@register_factor(
    name="limit_float_mv_intensity",
    description="涨停市值效应因子，-float_mv截面排名（小市值涨停延续性更强排前）。",
    category="event",
    thesis="小市值股票涨停后延续性系统性强于大市值——大市值涨停需要消耗巨额资金，次日获利盘抛压大；小市值涨停封板资金需求小，涨停溢价空间更足。负向排名=小市值排前。",
    dependencies=("limit_list.parquet",),
)
def factor_limit_float_mv_intensity(context: FactorContext):
    limit_list = context.load("limit_list.parquet")
    float_mv = limit_list["float_mv"].replace(0, np.nan)
    return cross_sectional_rank(-float_mv)


@register_factor(
    name="limit_turnover_intensity",
    description="涨停换手强度因子，-涨停日换手率截面排名（高换手=抛压大排后）。",
    category="event",
    thesis="涨停日换手率反映获利盘了结意愿——低换手涨停代表持仓者惜售、后续抛压轻；高换手涨停（尤其是连续涨停后的巨量换手）往往是主力出货的信号。",
    dependencies=("limit_list.parquet",),
)
def factor_limit_turnover_intensity(context: FactorContext):
    limit_list = context.load("limit_list.parquet")
    turnover = limit_list["turnover_ratio"].replace(0, np.nan)
    return cross_sectional_rank(-turnover)


# ── Supplementary event factors ────────────────────────────────────────────


@register_factor(
    name="limit_up_count_20d",
    description="20日涨停次数因子 (涨停次数多排前)。",
    category="event",
    thesis="多次涨停是强势股的标志，涨停频率高的个股存在持续性强势溢价",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_count_20d(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    is_limit = limit_up.get("is_limit_up", limit_up.get("limit_type", None))
    if is_limit is not None:
        count = is_limit.astype(bool).astype(float)
        cum_count = count.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).sum()
        )
        return cross_sectional_rank(cum_count)
    # Fallback: use consecutive_days
    consec = limit_up.get("consecutive_days", None)
    if consec is not None:
        return cross_sectional_rank(consec)
    return cross_sectional_rank(pd.Series(0, index=limit_up.index))


@register_factor(
    name="limit_down_count_20d",
    description="20日跌停次数因子 (跌停多排后, 负向)。",
    category="event",
    thesis="频繁跌停是质的恶化的极端信号，跌停次数多的股票风险极高",
    dependencies=("limit_list.parquet",),
)
def factor_limit_down_count_20d(context: FactorContext):
    limit_list = context.load("limit_list.parquet")
    up_stat = limit_list.get("up_stat", limit_list.get("limit", None))
    if up_stat is not None:
        is_down = (up_stat == 0) | (up_stat == "跌停") | (up_stat.astype(str).str.contains("跌"))
        count = is_down.astype(float)
        cum_count = count.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).sum()
        )
        return cross_sectional_rank(-cum_count)
    return cross_sectional_rank(-pd.Series(1, index=limit_list.index))


@register_factor(
    name="dt_net_amount_persistent_5d",
    description="5日龙虎榜净买入持续性因子 (连续净买入天数)。",
    category="event",
    thesis="龙虎榜连续净买入意味着机构持续建仓，比单日净买入更有说服力",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dt_net_amount_persistent_5d(context: FactorContext):
    dt = context.load("dragon_tiger.parquet")
    net = dt["net_buy_amount"]
    is_buy = (net > 0).astype(float)
    persist = is_buy.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    return cross_sectional_rank(persist)


@register_factor(
    name="top_list_net_rate_ma5",
    description="龙虎榜净买率5日均值因子。",
    category="event",
    thesis="龙虎榜净买率平滑后过滤单日噪音，均值趋势反映机构中期态度",
    dependencies=("top_list.parquet",),
)
def factor_top_list_net_rate_ma5(context: FactorContext):
    top_list = context.load("top_list.parquet")
    net_rate = top_list.get("net_rate", None)
    if net_rate is not None:
        ma5 = rolling_group_mean(net_rate, 5)
        return cross_sectional_rank(ma5)
    return cross_sectional_rank(pd.Series(0, index=top_list.index))


@register_factor(
    name="limit_seal_speed_reversal",
    description="涨停封板速度变化因子 (封板速度的5日变化)。",
    category="event",
    thesis="封板速度在变快=资金抢筹意愿增强，是强势股的加速信号",
    dependencies=("limit_up.parquet",),
)
def factor_limit_seal_speed_reversal(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    first_time = limit_up.get("first_limit_time", None)
    if first_time is not None:
        # Convert time string to minutes from open (e.g. "09:35" -> 5)
        if first_time.dtype == object:
            def _to_minutes(t):
                try:
                    parts = str(t).split(":")
                    return int(parts[0]) * 60 + int(parts[1]) - 570
                except Exception:
                    return 240
            minutes = first_time.apply(_to_minutes)
            # Faster seal = fewer minutes, better
            speed = -minutes
            speed_chg = speed.groupby(level="Code").diff(5)
            return cross_sectional_rank(speed_chg)
    return cross_sectional_rank(pd.Series(0, index=limit_up.index))


# ── Limit down (跌停) factors ─────────────────────────────────────────────

@register_factor(
    name="limit_down_consecutive",
    description="连续跌停天数因子（连续跌停天数截面排名）。",
    category="event",
    thesis="连续跌停代表极端恐慌和流动性危机——极度超卖后存在显著的短期反弹机会，是逆向买入的信号。",
    dependencies=("limit_list.parquet",),
)
def factor_limit_down_consecutive(context: FactorContext):
    limit_list = context.load("limit_list.parquet")
    # limit_list has 'limit' column where D=跌停, U=涨停
    is_down = (limit_list["limit"] == "D").astype(float) if "limit" in limit_list.columns else pd.Series(0, index=limit_list.index)
    return cross_sectional_rank(is_down)


@register_factor(
    name="limit_up_durability",
    description="涨停封板耐久度因子，(final_limit_time-first_limit_time)以分钟计截面排名（耐久=封板时间长排前）。",
    category="event",
    thesis="涨停后封板时间的长度反映封板的坚定程度——全天封板到尾盘的涨停次日溢价最高，而尾盘偷袭涨停的次日表现较差。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_durability(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    first = pd.to_datetime(limit_up["first_limit_time"], format="%H%M%S", errors="coerce")
    last = pd.to_datetime(limit_up["final_limit_time"], format="%H%M%S", errors="coerce")
    duration_minutes = (last - first).dt.total_seconds() / 60
    return cross_sectional_rank(duration_minutes)


@register_factor(
    name="limit_up_sealed_volume_ratio",
    description="涨停封单金额/成交额截面排名（高封单比=强封板排前）。",
    category="event",
    thesis="封单金额相对实际成交额的比例——高比例意味着买盘挂单远超实际成交，封板强度极高、次日连板概率大。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_sealed_volume_ratio(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    ratio = limit_up["sealed_turnover_ratio"]
    return cross_sectional_rank(ratio)


# ── Dragon tiger depth ───────────────────────────────────────────────────

@register_factor(
    name="dragon_tiger_institution_dominance",
    description="龙虎榜机构主导度因子，买方机构数/总席位数截面排名。",
    category="event",
    thesis="龙虎榜中买方席位的机构属性占比反映上涨的'质量'——机构主导的上涨比游资主导的上涨更具持续性和基本面支撑。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dragon_tiger_institution_dominance(context: FactorContext):
    dt = context.load("dragon_tiger.parquet")
    # Group by trade_date + stock_code, compute buy_ratio / (buy_ratio + sell_ratio)
    buy_ratio = dt["buy_ratio"]
    sell_ratio = dt["sell_ratio"]
    total = buy_ratio + sell_ratio
    buy_dominance = buy_ratio / total.replace(0, np.nan)
    return cross_sectional_rank(buy_dominance)


@register_factor(
    name="dragon_tiger_net_amount_rate",
    description="龙虎榜净额/成交额截面排名。",
    category="event",
    thesis="龙虎榜席位净买入金额占总成交额的比例——高净买率意味着上榜席位整体看多且行动一致。",
    dependencies=("dragon_tiger.parquet",),
)
def factor_dragon_tiger_net_amount_rate(context: FactorContext):
    dt = context.load("dragon_tiger.parquet")
    net = dt["buy_amount"] - dt["sell_amount"]
    total = dt["buy_amount"] + dt["sell_amount"]
    net_rate = net / total.replace(0, np.nan)
    return cross_sectional_rank(net_rate)


# ── Limit event composite ─────────────────────────────────────────────────

@register_factor(
    name="limit_event_intensity",
    description="极端行情强度因子，涨停/跌停事件强度(频次×幅度)5日滚动截面排名。",
    category="event",
    thesis="频繁触及涨跌停的股票处于极端行情状态——虽然短期波动大，但也意味着强烈的信息冲击和方向信号。区分涨跌停方向后，该因子能捕捉极端事件驱动的alpha。",
    dependencies=("limit_list.parquet", "daily_adj.parquet"),
)
def factor_limit_event_intensity(context: FactorContext):
    limit_list = context.load("limit_list.parquet")
    daily_adj = context.load("daily_adj.parquet")
    pct_chg = limit_list["pct_chg"]
    # Count events per stock per day
    event_intensity = pct_chg.abs().groupby(level=["Date", "Code"]).transform("sum")
    # Rolling 5-day sum
    intensity_5d = event_intensity.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    return cross_sectional_rank(intensity_5d)


# ── Top list depth factors ────────────────────────────────────────────────

@register_factor(
    name="top_list_net_flow_persistence",
    description="龙虎榜净流入持续性因子，(近3次上榜的net_rate平均值)截面排名。",
    category="event",
    thesis="持续净流入的上榜股票代表机构/游资的持续关注和建仓行为——非单次脉冲，后续仍有空间。",
    dependencies=("top_list.parquet",),
)
def factor_top_list_net_flow_persistence(context: FactorContext):
    top_list = context.load("top_list.parquet")
    net_rate = top_list["net_rate"]
    avg_net_3 = net_rate.groupby(level="Code").transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    return cross_sectional_rank(avg_net_3)


@register_factor(
    name="dragon_tiger_turnover",
    description="龙虎榜换手率因子，上榜日换手率截面排名。",
    category="event",
    thesis="上榜股票的换手率反映筹码交换的充分程度——高换手上榜意味着多空双方充分博弈、方向信号更可靠；低换手则可能是少量资金操作、信号质量差。",
    dependencies=("top_list.parquet",),
)
def factor_dragon_tiger_turnover(context: FactorContext):
    top_list = context.load("top_list.parquet")
    return cross_sectional_rank(top_list["turnover_rate"])


# ── Limit event recurrence ────────────────────────────────────────────────

@register_factor(
    name="limit_up_recurrence_20d",
    description="20日涨停复现率因子，近20日涨停天数截面排名。",
    category="event",
    thesis="涨停频率反映股票的'妖股'属性——频繁涨停的股票是游资最活跃的标的，具有强者恒强的特征但也伴随极高波动。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_recurrence_20d(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    is_up = (limit_up["is_limit_up"] == 1).astype(float) if "is_limit_up" in limit_up.columns else pd.Series(1.0, index=limit_up.index)
    count_20 = is_up.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    return cross_sectional_rank(count_20)


@register_factor(
    name="consecutive_limit_up_seal",
    description="连板封板强度因子，consecutive_days×sealed_flow_ratio截面排名。",
    category="event",
    thesis="连板天数与封单流比的交互——连板数多且封单流比大=最强连板龙头，兼具趋势强度和封板质量。",
    dependencies=("limit_up.parquet",),
)
def factor_consecutive_limit_up_seal(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    strength = limit_up["consecutive_days"].astype(float) * limit_up["sealed_flow_ratio"]
    return cross_sectional_rank(strength)


@register_factor(
    name="dragon_tiger_sell_power",
    description="龙虎榜卖出席位强度因子，-(l_sell/l_amount)截面排名（卖出集中=负面排后）。",
    category="event",
    thesis="龙虎榜卖出席位占总成交的比例——高卖出占比意味着上榜席位整体在撤离、卖方力量集中。",
    dependencies=("top_list.parquet",),
)
def factor_dragon_tiger_sell_power(context: FactorContext):
    top_list = context.load("top_list.parquet")
    sell_power = top_list["l_sell"] / top_list["l_amount"].replace(0, np.nan)
    return cross_sectional_rank(-sell_power)


@register_factor(
    name="limit_up_first_time_rank",
    description="涨停首次封板时间因子截面排名（越早封板=越强势排前）。",
    category="event",
    thesis="首次封板时间越早代表多方力量越强、态度越坚决——早盘涨停(10:30前)的质量和次日溢价远高于尾盘涨停。",
    dependencies=("limit_up.parquet",),
)
def factor_limit_up_first_time_rank(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    first = pd.to_datetime(limit_up["first_limit_time"], format="%H%M%S", errors="coerce")
    minutes_from_open = (first - first.dt.normalize() - pd.Timedelta(hours=9, minutes=30)).dt.total_seconds() / 60
    # Earlier = better => rank negative of minutes (small minutes = early)
    return cross_sectional_rank(-minutes_from_open)
