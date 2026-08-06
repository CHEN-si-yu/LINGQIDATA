"""
Event-driven factors — Class 1 (daily.parquet only).

事件驱动族:把稀疏事件(涨停/跌停/跳空/放量/新高/极端波动)转为连续衰减信号。
事件判定全部基于 daily.parquet 的 pct_chg / pre_close / vol / 复权基座,
无未来函数:所有"事件后表现"类因子均用滞后事件(shift(5) 等)对齐过去事件。

- 涨跌停近似:pct_chg >= 9.8% / <= -9.8%(主板 10% 口径;ST 5%、创业板/科创板 20%
  边界不同,9.8% 只是主板的近似,创业板/科创板涨停不会被判定,属覆盖范围问题,
  不做板块区分)。
- 稀疏事件统一经 utils.event_decay 转 ffill + 指数衰减信号再截面排名。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, event_decay, safe_divide
from .momentum_rebuilt import _adjusted_close


# ═══════════════════════════════════════════════════════════════════════════════
# 涨跌停事件
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="limit_up_event_5",
    description="涨停事件衰减因子：pct_chg≥9.8%近似涨停事件后指数衰减（半衰期3日，强势延续排前）。",
    category="event",
    thesis="涨停是A股最强的情绪事件——涨停后通常有2-3日的惯性溢价(策略10追三板逻辑)。"
           "用event_decay(半衰期3日)把稀疏涨停事件转为连续衰减信号：刚涨停的股票信号最强，"
           "随后衰减。涨停频发+近期=连板强势，排前。",
    dependencies=("daily.parquet",),
)
def factor_limit_up_event_5(context: FactorContext):
    daily = context.load("daily.parquet")
    event = daily["pct_chg"].ge(9.8).astype(float).where(
        daily["pct_chg"].ge(9.8), np.nan
    )
    decayed = event_decay(event, half_life=3)
    return cross_sectional_rank(decayed)


@register_factor(
    name="limit_down_event_5",
    description="跌停事件衰减因子：pct_chg≤−9.8%近似跌停事件后指数衰减（半衰期3日，恐慌延续排前）。",
    category="event",
    thesis="跌停是极端恐慌事件——跌停后惯性下杀与超跌反弹并存。衰减信号捕捉恐慌的"
           "时间结构：刚跌停的股票承压最强。作为风险警示因子，排名高=近期恐慌。",
    dependencies=("daily.parquet",),
)
def factor_limit_down_event_5(context: FactorContext):
    daily = context.load("daily.parquet")
    down = daily["pct_chg"].le(-9.8)
    event = down.astype(float).where(down, np.nan)
    decayed = event_decay(event, half_life=3)
    return cross_sectional_rank(decayed)


@register_factor(
    name="consecutive_limit_up",
    description="连续涨停天数因子：连续涨停(近似)的连板计数截面排名（连板高度排前）。",
    category="event",
    thesis="策略10追三板的核心变量是连板高度——连板天数越长，市场关注度与情绪溢价越高，"
           "但连板越高断板风险也越大。计数采用连续段逻辑：断板即归零。",
    dependencies=("daily.parquet",),
)
def factor_consecutive_limit_up(context: FactorContext):
    daily = context.load("daily.parquet")
    is_lu = daily["pct_chg"].ge(9.8).astype(int)
    code = is_lu.index.get_level_values("Code")
    seg = (~is_lu.astype(bool)).groupby(level="Code").cumsum()
    count = is_lu.groupby([code, seg]).cumsum()
    return cross_sectional_rank(count)


@register_factor(
    name="limit_up_fade_10",
    description="涨停后表现因子：10日内有涨停事件的股票其10日累计收益（涨停后延续排前）。",
    category="event",
    thesis="涨停后的中期表现是情绪溢价的检验——过去10日内涨停过的股票，其10日累计收益"
           "衡量「追涨停资金」的盈亏状态：收益为正=涨停后行情延续(强者恒强)，"
           "收益为负=涨停后回落(情绪退潮)。滞后视角无未来函数。",
    dependencies=("daily.parquet",),
)
def factor_limit_up_fade_10(context: FactorContext):
    daily = context.load("daily.parquet")
    is_lu = daily["pct_chg"].ge(9.8)
    has_lu_10 = is_lu.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=1).max()
    )
    ret = daily["pct_chg"] / 100.0
    ret_10 = ret.groupby(level="Code").transform(
        lambda s: (1.0 + s).cumprod().pct_change(10, fill_method=None)
    )
    fade = has_lu_10.astype(float) * ret_10
    return cross_sectional_rank(fade)


# ═══════════════════════════════════════════════════════════════════════════════
# 跳空 / 放量 / 极端波动事件
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="gap_event_decay_5",
    description="跳空事件衰减因子：|open/pre_close−1|>5%跳空事件后指数衰减（半衰期5日）。",
    category="event",
    thesis="大幅跳空(>5%)是隔夜信息冲击的直接体现——跳空方向与幅度携带信息。"
           "衰减信号区分近期跳空(信息新鲜)与远期跳空(已被消化)。"
           "open/pre_close 为复权口径，除权日不产生假跳空。",
    dependencies=("daily.parquet",),
)
def factor_gap_event_decay_5(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
    is_gap = gap.abs().gt(0.05)
    event = is_gap.astype(float).where(is_gap, np.nan)
    decayed = event_decay(event, half_life=5)
    return cross_sectional_rank(decayed)


@register_factor(
    name="extreme_move_event",
    description="极端波动事件衰减因子：|pct_chg|>7%事件后指数衰减（半衰期10日，异动后惯性排前）。",
    category="event",
    thesis="单日|涨跌|>7%是极端波动事件——大涨(利好兑现)或大跌(利空冲击)后通常有"
           "数日的方向惯性或反转(研报《如何捕捉短线反弹机会》)。半衰期10日捕捉中期效应。",
    dependencies=("daily.parquet",),
)
def factor_extreme_move_event(context: FactorContext):
    daily = context.load("daily.parquet")
    is_ext = daily["pct_chg"].abs().gt(7.0)
    event = is_ext.astype(float).where(is_ext, np.nan)
    decayed = event_decay(event, half_life=10)
    return cross_sectional_rank(decayed)


# ═══════════════════════════════════════════════════════════════════════════════
# 新高/新低事件(复权基座)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="new_high_60_event",
    description="60日新高事件衰减因子：复权价创60日新高事件后指数衰减（半衰期5日，突破强势排前）。",
    category="event",
    thesis="创60日新高是趋势突破的事件化表达——突破后惯性延续(海龟/Donchian逻辑)。"
           "基于复权基座判定新高，除权日不产生假突破(与7-31 52周修复同口径)。"
           "衰减信号衡量突破的新鲜度。",
    dependencies=("daily.parquet",),
)
def factor_new_high_60_event(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    is_high = adj.eq(adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    ))
    event = is_high.astype(float).where(is_high, np.nan)
    decayed = event_decay(event, half_life=5)
    return cross_sectional_rank(decayed)


@register_factor(
    name="new_low_60_event",
    description="60日新低事件衰减因子：复权价创60日新低事件后指数衰减（半衰期5日，破位风险排前）。",
    category="event",
    thesis="创60日新低是趋势破位的事件化表达——新低后惯性下杀与超跌反弹并存。"
           "基于复权基座，除权日不产生假新低。作为风险因子与 new_high_60_event 镜像。",
    dependencies=("daily.parquet",),
)
def factor_new_low_60_event(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    is_low = adj.eq(adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).min()
    ))
    event = is_low.astype(float).where(is_low, np.nan)
    decayed = event_decay(event, half_life=5)
    return cross_sectional_rank(decayed)


@register_factor(
    name="new_high_frequency_60",
    description="60日新高频率因子：60日内创新高天数占比截面排名（趋势强势频率排前）。",
    category="event",
    thesis="创新高的频率比单次新高更稳健——60日内多次创新高=持续趋势(强势股票)；"
           "仅一次新高=脉冲行情。基于复权基座，除权日不产生假新高。",
    dependencies=("daily.parquet",),
)
def factor_new_high_frequency_60(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    is_high = adj.eq(adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    ))
    freq = is_high.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    return cross_sectional_rank(freq)


# ═══════════════════════════════════════════════════════════════════════════════
# 形态 / 交替
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="three_black_crows",
    description="三只黑鸦形态因子：连续阴线(收盘<开盘且低于前收)天数计数截面排名（连续阴跌排前）。",
    category="event",
    thesis="三只黑鸦是经典顶部形态(策略51)：连续三根阴线收盘逐步走低=空头主导。"
           "实现为连续阴跌天数计数(不限于3天)，连续阴跌越长空头动能越强。",
    dependencies=("daily.parquet",),
)
def factor_three_black_crows(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]
    # pre_close is the ex-date reference close supplied by daily.parquet;
    # raw close.shift(1) fabricates a bearish step on corporate-action days.
    cond = (close < open_) & (close < daily["pre_close"])
    cond_i = cond.astype(int)
    code = cond_i.index.get_level_values("Code")
    seg = (~cond_i.astype(bool)).groupby(level="Code").cumsum()
    count = cond_i.groupby([code, seg]).cumsum()
    return cross_sectional_rank(count)


@register_factor(
    name="limit_alternation_20",
    description="涨跌停交替因子：20日内涨停与跌停事件数量的乘积截面排名（情绪极端反转排前）。",
    category="event",
    thesis="同一股票20日内既出现涨停又出现跌停=情绪剧烈反转(多空激烈博弈)——"
           "研报《基于板块效应动量反转特征》指出情绪反转点常伴随大级别变盘。"
           "涨停数与跌停数的乘积大=两种极端同时频繁出现。",
    dependencies=("daily.parquet",),
)
def factor_limit_alternation_20(context: FactorContext):
    daily = context.load("daily.parquet")
    is_lu = daily["pct_chg"].ge(9.8).astype(float)
    is_ld = daily["pct_chg"].le(-9.8).astype(float)
    lu_20 = is_lu.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    ld_20 = is_ld.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    alternation = lu_20 * ld_20
    return cross_sectional_rank(alternation)


@register_factor(
    name="big_gap_reversal_5",
    description="高开回补因子：5日前高开(>3%)事件的5日累计收益截面排名（高开后走强排前）。",
    category="event",
    thesis="大幅高开(>3%)后是否回补是判断跳空性质的试金石——高开后5日继续上涨="
           "真实突破(缺口为支撑)，高开后回落=假突破(缺口被回补)。"
           "用5日前的高开事件与5日累计收益对齐，滞后视角无未来函数。",
    dependencies=("daily.parquet",),
)
def factor_big_gap_reversal_5(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
    gap_event = gap.gt(0.03).astype(float)
    # 5日前的跳空事件: shift 需按 Code 分组
    event_lag5 = gap_event.groupby(level="Code").shift(5)
    ret = daily["pct_chg"] / 100.0
    ret_5 = ret.groupby(level="Code").transform(
        lambda s: (1.0 + s).cumprod().pct_change(5, fill_method=None)
    )
    reversal = event_lag5 * ret_5
    return cross_sectional_rank(reversal)
