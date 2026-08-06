"""
K线形态与事件结构因子 — Class 1 (daily.parquet only)。

形态族:一字板/炸板/孕线/吞没/十字星/连续计数/跳空结构,全部向量化实现。

关键原则(与 skill.md 一致):
- 涨跌停近似:pct_chg >= 9.8% / <= -9.8%(主板口径,不区分板块);
- 跨日比较(昨日高低/前收)一律用复权口径:pre_close 为数据商复权昨收,
  跨日极值用 scale=adj/close 折算 high/low(与 cci_20/eom_14 同口径);
- 当日内部比较(open/close/high/low 关系)无需复权;
- 稀疏事件不引入逐日循环,全部走 rolling 窗口计数/占比;
- 无未来函数:事件类因子仅用当日及历史信息。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide
from .momentum_rebuilt import _adjusted_close


def _scale_hl(daily: pd.DataFrame, adj: pd.Series) -> tuple[pd.Series, pd.Series]:
    """按 scale=adj/close 折算 high/low 到复权口径(跨日比较用)。"""
    scale = adj / daily["close"].replace(0, np.nan)
    return daily["high"] * scale, daily["low"] * scale


def _roll_mean(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.groupby(level="Code").transform(
        lambda x: x.rolling(window, min_periods=min_periods).mean()
    )


def _roll_sum(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.groupby(level="Code").transform(
        lambda x: x.rolling(window, min_periods=min_periods).sum()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 涨跌停事件结构
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="consecutive_limit_down",
    description="连续跌停天数因子：连续跌停(近似)的连跌计数截面排名（连跌高度排前）。",
    category="event",
    thesis="连续跌停是流动性枯竭与恐慌加速的标志(与连续涨停镜像)——连跌越深,"
           "恐慌盘释放越充分,超跌反弹的概率与幅度也越大(研报《如何捕捉短线反弹机会》)。"
           "计数采用连续段逻辑:断板即归零。",
    dependencies=("daily.parquet",),
)
def factor_consecutive_limit_down(context: FactorContext):
    daily = context.load("daily.parquet")
    is_ld = daily["pct_chg"].le(-9.8).astype(int)
    code = is_ld.index.get_level_values("Code")
    seg = (~is_ld.astype(bool)).groupby(level="Code").cumsum()
    count = is_ld.groupby([code, seg]).cumsum()
    return cross_sectional_rank(count)


@register_factor(
    name="one_word_limit_up_freq_20",
    description="一字涨停频率因子：20日一字板(全天无波动且涨停)天数占比截面排名（一字连板强势排前）。",
    category="event",
    thesis="一字涨停(open=high=low=close)意味着买盘封死、筹码完全锁仓——"
           "是A股最强情绪形态(策略10追三板的高阶形态)。一字板频繁出现=强庄控盘,"
           "但也伴随流动性风险。20日频率平滑稀疏事件。",
    dependencies=("daily.parquet",),
)
def factor_one_word_limit_up_freq_20(context: FactorContext):
    daily = context.load("daily.parquet")
    no_range = (daily["high"] - daily["low"]).abs().lt(1e-6)
    one_word = (no_range & daily["pct_chg"].ge(9.8)).astype(float)
    freq = _roll_mean(one_word, 20, 5)
    return cross_sectional_rank(freq)


@register_factor(
    name="one_word_limit_down_freq_20",
    description="一字跌停频率因子：20日一字跌停天数占比截面排名（恐慌锁死排前，负向排名）。",
    category="event",
    thesis="一字跌停=卖盘封死、无法出逃的流动性冻结——是极端恐慌的信号,"
           "一字跌停频发股票的流动性风险与后续踩踏风险高。与一字涨停镜像,负向排名。",
    dependencies=("daily.parquet",),
)
def factor_one_word_limit_down_freq_20(context: FactorContext):
    daily = context.load("daily.parquet")
    no_range = (daily["high"] - daily["low"]).abs().lt(1e-6)
    one_word = (no_range & daily["pct_chg"].le(-9.8)).astype(float)
    freq = _roll_mean(one_word, 20, 5)
    return cross_sectional_rank(-freq)


@register_factor(
    name="limit_up_open_fail_freq_20",
    description="炸板频率因子：20日盘中触板(高点≥9.8%)但收盘未封板的天数占比截面排名（炸板频发排前）。",
    category="event",
    thesis="炸板(触板未封)是情绪由强转弱的直接证据——涨停封单被抛压击穿,"
           "说明该价位承接不足。炸板频繁=情绪票、主力出货特征,是追高策略的反向指标。"
           "high 与当日 pre_close 同尺度比较,除权日不产生假触板。",
    dependencies=("daily.parquet",),
)
def factor_limit_up_open_fail_freq_20(context: FactorContext):
    daily = context.load("daily.parquet")
    # 同日比较:high 与当日 pre_close 天然同尺度(pre_close 在除权日已复权到
    # 当日价格尺度),无需 scale 折算——折算反而会把相对量级(adj≈1.0)混入真实价格。
    high_pct = safe_divide(daily["high"] - daily["pre_close"], daily["pre_close"]) * 100.0
    touched = high_pct.ge(9.8)
    sealed = daily["pct_chg"].ge(9.8)
    fail = (touched & ~sealed).astype(float)
    freq = _roll_mean(fail, 20, 5)
    return cross_sectional_rank(freq)


@register_factor(
    name="limit_down_rebound_10",
    description="跌停后表现因子：10日内有跌停事件的股票其10日累计收益（跌停后反弹排前）。",
    category="event",
    thesis="跌停后的中期表现检验恐慌是否过度——过去10日跌停过的股票,其10日累计收益"
           "衡量超跌反弹的兑现程度(研报《如何捕捉短线反弹机会》:跌停次日反弹胜率高)。"
           "滞后视角无未来函数,与 limit_up_fade_10 镜像。",
    dependencies=("daily.parquet",),
)
def factor_limit_down_rebound_10(context: FactorContext):
    daily = context.load("daily.parquet")
    is_ld = daily["pct_chg"].le(-9.8)
    has_ld_10 = is_ld.astype(float).groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=1).max()
    )
    ret = daily["pct_chg"] / 100.0
    ret_10 = ret.groupby(level="Code").transform(
        lambda s: (1.0 + s).cumprod().pct_change(10, fill_method=None)
    )
    rebound = has_ld_10 * ret_10
    return cross_sectional_rank(rebound)


# ═══════════════════════════════════════════════════════════════════════════════
# K线形态结构(孕线/吞没/十字星/大振幅日)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="inside_bar_count_20",
    description="孕线频率因子：20日孕线(当日高低点完全落入昨日区间)天数截面排名（收敛蓄势排前）。",
    category="price",
    thesis="孕线=当日波动完全被昨日区间包裹,是多空分歧收敛、变盘前的压缩形态"
           "(突破前的能量积蓄)。孕线密集=波动压缩到位,后续突破方向一旦确立"
           "往往伴随主升/主跌段。跨日比较用复权口径高低点。",
    dependencies=("daily.parquet",),
)
def factor_inside_bar_count_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    high_adj, low_adj = _scale_hl(daily, adj)
    prev_high = high_adj.groupby(level="Code").shift(1)
    prev_low = low_adj.groupby(level="Code").shift(1)
    inside = (high_adj.le(prev_high) & low_adj.ge(prev_low)).astype(float)
    count = _roll_sum(inside, 20, 5)
    return cross_sectional_rank(count)


@register_factor(
    name="outside_bar_count_20",
    description="吞没/突破频率因子：20日收盘突破昨日全天区间(吞没)天数截面排名（强吞没走势排前）。",
    category="price",
    thesis="吞没形态(收盘突破昨日全区间)是多空力量的单边碾压——向上吞没=多方完全"
           "收复昨日空方阵地(强势确认),向下吞没=空方碾压。频率高=趋势性股票"
           "持续单边运行,与震荡股形成截面区分。用复权 close 与复权昨日高低比较。",
    dependencies=("daily.parquet",),
)
def factor_outside_bar_count_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    high_adj, low_adj = _scale_hl(daily, adj)
    prev_high = high_adj.groupby(level="Code").shift(1)
    prev_low = low_adj.groupby(level="Code").shift(1)
    outside = (adj.gt(prev_high) | adj.lt(prev_low)).astype(float)
    count = _roll_sum(outside, 20, 5)
    return cross_sectional_rank(count)


@register_factor(
    name="doji_frequency_20",
    description="十字星频率因子：20日十字星(|实体|≤10%振幅)天数占比截面排名（分歧整理排前）。",
    category="price",
    thesis="十字星=开盘价与收盘价几乎重合,多空当日势均力敌——连续十字星=高位滞涨"
           "或底部吸筹的分歧期。十字星密集股短线方向不明,作为低趋势确认度的反指标"
           "与趋势效率因子互补。单日内比较,无需复权。",
    dependencies=("daily.parquet",),
)
def factor_doji_frequency_20(context: FactorContext):
    daily = context.load("daily.parquet")
    rng = daily["high"] - daily["low"]
    body = (daily["close"] - daily["open"]).abs()
    doji = (body.le(0.1 * rng) & rng.gt(0)).astype(float)
    freq = _roll_mean(doji, 20, 5)
    return cross_sectional_rank(freq)


@register_factor(
    name="big_range_day_freq_20",
    description="大振幅日频率因子：20日振幅(复权)≥5%的天数占比截面排名（剧烈波动排前）。",
    category="price",
    thesis="单日振幅≥5%是多空激烈博弈的异动日——振幅日的频率衡量股票的波动性格"
           "(题材股高频、银行股低频)。高振幅频率=情绪驱动与信息冲击频繁,"
           "与波动率水平正交但捕捉其分布形态。高低点折算复权口径,除权日不失真。",
    dependencies=("daily.parquet",),
)
def factor_big_range_day_freq_20(context: FactorContext):
    daily = context.load("daily.parquet")
    # 同日比较:(high−low) 与当日 pre_close 同尺度,无需复权折算;
    # pre_close 在除权日已调整到当日价格尺度,振幅口径与涨跌停判定一致。
    amp = safe_divide(daily["high"] - daily["low"], daily["pre_close"]) * 100.0
    is_big = amp.ge(5.0).astype(float)
    freq = _roll_mean(is_big, 20, 5)
    return cross_sectional_rank(freq)


# ═══════════════════════════════════════════════════════════════════════════════
# 连续状态计数(连涨/连跌/缩量)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="current_up_streak",
    description="当前连涨天数因子：截至今日连续收涨天数截面排名（连涨动能排前）。",
    category="price",
    thesis="当前连续上涨天数衡量趋势的即时动能状态——连涨天数长=买盘连续性强,"
           "但过长也积累短期超买风险。与滚动窗口动量不同,本因子是状态变量:"
           "反映「今天正在发生什么」而非「过去发生了什么」。close 与数据商复权"
           "昨收 pre_close 比较,除权日判定正确。",
    dependencies=("daily.parquet",),
)
def factor_current_up_streak(context: FactorContext):
    daily = context.load("daily.parquet")
    is_up = (daily["close"] > daily["pre_close"]).astype(int)
    code = is_up.index.get_level_values("Code")
    seg = (~is_up.astype(bool)).groupby(level="Code").cumsum()
    streak = is_up.groupby([code, seg]).cumsum()
    return cross_sectional_rank(streak)


@register_factor(
    name="current_down_streak",
    description="当前连跌天数因子：截至今日连续收跌天数截面排名（连跌超卖排前，反向排名）。",
    category="price",
    thesis="连续下跌天数反映下跌的持续性——连跌越深,恐慌释放越充分、超卖越极端"
           "(反转候选),但连跌本身也说明空头完全掌控(趋势未止)。"
           "与 current_up_streak 镜像,作为左侧反转的时间维度信号。",
    dependencies=("daily.parquet",),
)
def factor_current_down_streak(context: FactorContext):
    daily = context.load("daily.parquet")
    is_down = (daily["close"] < daily["pre_close"]).astype(int)
    code = is_down.index.get_level_values("Code")
    seg = (~is_down.astype(bool)).groupby(level="Code").cumsum()
    streak = is_down.groupby([code, seg]).cumsum()
    return cross_sectional_rank(-streak)


@register_factor(
    name="current_vol_shrink_streak",
    description="当前缩量连天数因子：截至今日成交量连续递减天数截面排名（持续缩量排前）。",
    category="price",
    thesis="成交量连续递减=交投热度持续退潮(无人问津)或惜售锁仓(缩量洗盘)——"
           "缩量连天在底部区域是筑底特征,在顶部是上涨动力衰竭的先行信号。"
           "与价格连涨/连跌正交,补充量能的节奏状态维度。",
    dependencies=("daily.parquet",),
)
def factor_current_vol_shrink_streak(context: FactorContext):
    daily = context.load("daily.parquet")
    prev_vol = daily["vol"].groupby(level="Code").shift(1)
    is_shrink = (daily["vol"] < prev_vol).astype(int)
    code = is_shrink.index.get_level_values("Code")
    seg = (~is_shrink.astype(bool)).groupby(level="Code").cumsum()
    streak = is_shrink.groupby([code, seg]).cumsum()
    return cross_sectional_rank(streak)


# ═══════════════════════════════════════════════════════════════════════════════
# 跳空结构
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="gap_open_follow_ratio_20",
    description="跳空方向延续率因子：20日跳空(>0.5%)日中跳空方向与日内方向一致占比截面排名（顺延结构排前）。",
    category="price",
    thesis="跳空后日内继续同向=隔夜信息被市场认可(趋势延续);跳空后反向回补=缺口"
           "是情绪陷阱(反转结构)。方向延续率高=跳空质量高、缺口具支撑/压力意义,"
           "是海龟/缺口交易逻辑的统计刻画。open/pre_close 均为复权口径。",
    dependencies=("daily.parquet",),
)
def factor_gap_open_follow_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
    intra = safe_divide(daily["close"], daily["open"]) - 1.0
    has_gap = gap.abs().gt(0.005)
    align = ((np.sign(gap) == np.sign(intra)) & has_gap).astype(float)
    n_align = _roll_sum(align, 20, 3)
    n_gap = _roll_sum(has_gap.astype(float), 20, 3)
    ratio = safe_divide(n_align, n_gap)
    return cross_sectional_rank(ratio)
