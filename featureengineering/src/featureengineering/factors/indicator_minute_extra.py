"""
Class 4 (indicator_1min) 补充因子 — 回退注册函数。

主构建路径为 cli → build_indicator_1min_new(INDICATOR_FACTOR_SPEC 扇出,
见 indicator_minute.py),本文件的注册函数仅作为单因子直连构建(build_many)
时的回退实现,方向与 INDICATOR_FACTOR_SPEC 中同名条目一一对应:
  pos → rank(+metric), neg → rank(−metric), neg_abs → rank(−|metric|)。

主题:未映射指标的补位 + 日内区间位置/斜率结构 + 跨日指标结构。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank
from .indicator_minute import _compute_indicator_factor


def _metric(context: FactorContext, name: str) -> pd.Series:
    return _compute_indicator_factor(
        context.repo.paths.source_root, context.repo.allowed_codes, name,
        on_progress=context.repo.on_progress,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 未映射指标补位
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="price_vs_ma10_deviation",
    description="价格对MA10偏离因子：close相对分钟MA10的偏离截面排名（负向绝对值，大幅偏离排后）。",
    category="intraday",
    thesis="分钟级 MA10 是短周期均价中枢——价格大幅偏离 MA10=短期过热/超卖"
           "(均值回归风险),贴均线运行=健康趋势。与 price_vs_ma20/ma60 构成"
           "不同周期的偏离谱系,MA10 对短线择时更敏感。",
    dependencies=("indicator_1min",),
)
def factor_price_vs_ma10_deviation(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "price_vs_ma10").abs())


@register_factor(
    name="price_vs_ma30_deviation",
    description="价格对MA30偏离因子：close相对分钟MA30的偏离截面排名（负向绝对值，大幅偏离排后）。",
    category="intraday",
    thesis="MA30(6个交易日约)介于短中周期之间——价格显著高于 MA30=短期涨幅透支,"
           "显著低于=超跌待修复。负向绝对值排名:偏离越大越靠后(极端偏离回归"
           "概率高)。与 MA20/MA60 版本互补,细化偏离周期。",
    dependencies=("indicator_1min",),
)
def factor_price_vs_ma30_deviation(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "price_vs_ma30").abs())


@register_factor(
    name="kdj_dead_cross_count",
    description="KDJ死叉次数因子：日内K下穿D的次数截面排名（负向，死叉频繁排后）。",
    category="intraday",
    thesis="日内 K 下穿 D 的次数衡量空头信号的反复强度——死叉频繁=多方反击乏力、"
           "空头持续压制;死叉稀少=趋势结构稳定。与 kdj_cross_signal(净金叉)互补:"
           "本因子单看空头侧。",
    dependencies=("indicator_1min",),
)
def factor_kdj_dead_cross_count(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "kdj_dead_cross_count"))


# ═══════════════════════════════════════════════════════════════════════════════
# 日内区间位置与斜率结构
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="rsi_day_position",
    description="RSI日内区间位置因子：(RSI收盘−RSI最低)/(RSI最高−RSI最低)截面排名（收盘靠上沿排前）。",
    category="intraday",
    thesis="RSI 收盘在当日区间中的位置衡量日内动能的方向终态——收盘靠上沿=日内"
           "多方逐步占据优势(收盘确认强势);靠下沿=空方压制。与 rsi_range(区间"
           "宽度)正交:宽度管振幅、位置管方向。",
    dependencies=("indicator_1min",),
)
def factor_rsi_day_position(context: FactorContext):
    return cross_sectional_rank(_metric(context, "rsi_day_position"))


@register_factor(
    name="j_day_position",
    description="KDJ J值日内区间位置因子：(J收盘−J最低)/(J最高−J最低)截面排名（J收盘靠上沿排前）。",
    category="intraday",
    thesis="J 值对价格变动最敏感,其收盘在当日区间的位置是日内短线动能方向的"
           "终态判断——靠上沿=尾盘动能向上(次日惯性概率大);靠下沿=尾盘走弱。"
           "与 j_range(波动宽度)、kdj_j_value_close(绝对水平)互补。",
    dependencies=("indicator_1min",),
)
def factor_j_day_position(context: FactorContext):
    return cross_sectional_rank(_metric(context, "j_day_position"))


@register_factor(
    name="macd_bar_energy",
    description="MACD柱能量因子：日内|macd|均值截面排名（负向，柱体活跃排后）。",
    category="intraday",
    thesis="MACD 柱的绝对均值衡量多空动能释放的强度——柱体能量大=趋势推动力"
           "强但波动剧烈(方向切换频繁);能量小=动能枯竭(变盘前兆)。与"
           "macd_daily_range(柱极差)互补:能量看均值、极差看极端。",
    dependencies=("indicator_1min",),
)
def factor_macd_bar_energy(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "macd_bar_energy"))


@register_factor(
    name="ma10_slope",
    description="MA10斜率因子：分钟MA10收盘相对开盘的斜率截面排名（MA10上倾排前）。",
    category="intraday",
    thesis="MA10 斜率是短周期趋势的方向计——上倾=短线均线系统多头结构;与已有"
           "ma5_slope/ma20_slope 构成完整斜率谱系,MA10 的斜率差异可用于判断"
           "短线趋势的传导层级(MA5 上穿 MA10 的加速阶段)。",
    dependencies=("indicator_1min",),
)
def factor_ma10_slope(context: FactorContext):
    return cross_sectional_rank(_metric(context, "ma10_slope"))


@register_factor(
    name="ma30_slope",
    description="MA30斜率因子：分钟MA30收盘相对开盘的斜率截面排名（MA30上倾排前）。",
    category="intraday",
    thesis="MA30 斜率捕捉6日尺度的均线方向——MA30 上倾=中期均线结构转多"
           "(趋势级别提升);下倾=中期结构走弱。与 ma20_slope 的背离可识别"
           "均线系统的内部换档(20日线斜率与30日线斜率的交叉区)。",
    dependencies=("indicator_1min",),
)
def factor_ma30_slope(context: FactorContext):
    return cross_sectional_rank(_metric(context, "ma30_slope"))


@register_factor(
    name="ma5_ma10_gap",
    description="MA5/MA10乖离因子：分钟MA5相对MA10的偏离截面排名（短均线在上排前）。",
    category="intraday",
    thesis="MA5 在 MA10 上方=短周期动能强于短中期(短线攻击结构);MA5 跌破 MA10="
           "短线动能衰减(均线死叉前兆)。是分钟级均线系统的内部结构度量,"
           "与日频 ma_distance_5_10 区分(分钟口径)。",
    dependencies=("indicator_1min",),
)
def factor_ma5_ma10_gap(context: FactorContext):
    return cross_sectional_rank(_metric(context, "ma5_ma10_gap"))


# ═══════════════════════════════════════════════════════════════════════════════
# 跨日结构
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="boll_width_5d_change",
    description="布林带宽5日变化因子：分钟布林带宽的5日变化截面排名（带宽扩张排前）。",
    category="intraday",
    thesis="布林带宽5日扩张=波动率周期启动(趋势行情开启的前兆);持续收窄=横盘"
           "蓄势。与 boll_width_change(日内变化)、boll_squeeze(绝对宽度)互补,"
           "捕捉带宽的中期方向——A股波动率具有明显状态切换特征。",
    dependencies=("indicator_1min",),
)
def factor_boll_width_5d_change(context: FactorContext):
    return cross_sectional_rank(_metric(context, "boll_width_5d_chg"))


@register_factor(
    name="rsi_trend_ma5",
    description="RSI超额5日均值因子：(RSI−50)的5日移动平均截面排名（中期动能排前）。",
    category="intraday",
    thesis="RSI 相对50中线的超额部分5日均值=中期动能水平(过滤单日噪音)——持续"
           "正值=中期多方占优;持续负值=中期空头主导。与 rsi_14_excess(当日)"
           "互补:本因子为5日平滑版本,更贴近趋势状态。",
    dependencies=("indicator_1min",),
)
def factor_rsi_trend_ma5(context: FactorContext):
    return cross_sectional_rank(_metric(context, "rsi_excess_ma5"))


@register_factor(
    name="kdj_bull_frac_5d_change",
    description="KDJ多头占比5日变化因子：日内K>D占比的5日变化截面排名（多头增强排前）。",
    category="intraday",
    thesis="K>D 的日内占比衡量全天金叉状态的时间份额——其5日变化捕捉多头结构"
           "的增强/衰减:占比持续上升=KDJ 系统的多头状态正在建立;下降=金叉"
           "质量恶化。与 kdj_bull_frac(水平)互补,捕捉方向。",
    dependencies=("indicator_1min",),
)
def factor_kdj_bull_frac_5d_change(context: FactorContext):
    return cross_sectional_rank(_metric(context, "kdj_bull_frac_5d_chg"))


@register_factor(
    name="kdj_j_5d_acceleration",
    description="KDJ J值5日加速度因子：J收盘的5日变化截面排名（J值加速排前）。",
    category="intraday",
    thesis="J 值5日变化=KDJ 动能的加速度——J 持续抬升=短线动能加速(主升段特征);"
           "J 回落=动能衰竭。与 j_day_position(日内位置)互补:本因子为跨日"
           "动量维度,捕捉KDJ 系统的趋势性变化。",
    dependencies=("indicator_1min",),
)
def factor_kdj_j_5d_acceleration(context: FactorContext):
    return cross_sectional_rank(_metric(context, "j_close_5d_gap"))
