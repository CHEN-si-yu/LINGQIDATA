"""
Class 3 (history_1min) 补充因子 — 回退注册函数。

主构建路径为 cli → build_intraday_new(INTRADAY_FACTOR_SPEC 扇出,见 intraday.py),
本文件的注册函数仅作为单因子直连构建(build_many)时的回退实现,
方向与 INTRADAY_FACTOR_SPEC 中同名条目一一对应:
  pos → cross_sectional_rank(+metric),  neg → cross_sectional_rank(−metric)。

主题:日内时段结构——上午/下午、开盘/尾盘30分钟的量价分布,
补充已有指标(全日口径)无法刻画的日内形态差异。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank
from .intraday import _compute_intraday_factor


def _metric(context: FactorContext, name: str) -> pd.Series:
    return _compute_intraday_factor(
        context.repo.paths.source_root, context.repo.allowed_codes, name,
        on_progress=context.repo.on_progress,
    )


@register_factor(
    name="am_momentum_intraday",
    description="上午动量因子：上午收盘相对上午开盘的涨跌幅截面排名（上午走强排前）。",
    category="intraday",
    thesis="上午动量(am_close/am_open−1)刻画上午半场的多空胜负——上午强=早盘信息"
           "被消化后多方持续占优;与已有 pm_momentum_intraday 对称,两者之差"
           "即日内方向的时段归属。分钟级口径,与日频 am_pm_return_ratio 互补。",
    dependencies=("history_1min",),
)
def factor_am_momentum_intraday(context: FactorContext):
    return cross_sectional_rank(_metric(context, "am_momentum"))


@register_factor(
    name="open_30_momentum",
    description="开盘30分钟动量因子：开盘半小时(09:30-10:00)涨跌幅截面排名（开盘强势排前）。",
    category="intraday",
    thesis="开盘半小时是隔夜信息定价最集中的时段——开盘30分钟动量强=资金抢筹"
           "坚决(集合竞价+早盘首波买盘的真实意愿);弱=高开低走或低开无承接。"
           "与 open_auction_ret(开盘跳空)区分:本因子度量跳空之后的半小时走势。",
    dependencies=("history_1min",),
)
def factor_open_30_momentum(context: FactorContext):
    return cross_sectional_rank(_metric(context, "open_30_mom"))


@register_factor(
    name="close_30_momentum",
    description="尾盘30分钟动量因子：收盘半小时(14:30-15:00)涨跌幅截面排名（尾盘拉升排前）。",
    category="intraday",
    thesis="尾盘半小时动量反映收盘定调——A股尾盘拉升常是主力做收盘价/次日溢价"
           "的行为(次日惯性),尾盘砸盘=次日低开风险。尾盘方向对次日开盘有"
           "预测力,与 close_auction_impact(最后3分钟)区分:本因子为30分钟窗口。",
    dependencies=("history_1min",),
)
def factor_close_30_momentum(context: FactorContext):
    return cross_sectional_rank(_metric(context, "close_30_mom"))


@register_factor(
    name="open_close_momentum_gap",
    description="首尾动量差因子：开盘30分钟动量−尾盘30分钟动量截面排名（负向，冲高回落排后）。",
    category="intraday",
    thesis="开盘强而尾盘弱=早盘透支、全天动能衰竭(冲高回落风险,次日低开概率大);"
           "开盘弱而尾盘强=尾盘修复(资金尾盘进场,次日惯性延续)。首尾强弱差"
           "捕捉日内动能的迁移方向,是单时段动量不具备的结构信息。",
    dependencies=("history_1min",),
)
def factor_open_close_momentum_gap(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "open_close_30_mom_ratio"))


@register_factor(
    name="am_pm_range_ratio",
    description="上午/下午振幅比因子：上午振幅/下午振幅截面排名（负向，早盘波动主导排后）。",
    category="intraday",
    thesis="上午振幅占优=价格波动集中在早盘(隔夜信息冲击大、方向快速博弈);"
           "下午振幅占优=午后多空拉锯(方向不明)。早盘主导波动=日内方向早定,"
           "交易效率高;下午主导=全天反复,趋势质量差。",
    dependencies=("history_1min",),
)
def factor_am_pm_range_ratio(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "am_pm_hl_range_ratio"))


@register_factor(
    name="tail_volume_share",
    description="尾盘量能占比因子：尾盘30分钟成交量占全天比例截面排名（尾盘放量排前）。",
    category="intraday",
    thesis="尾盘量能占比高=资金在收盘前集中进场/对倒(尾盘定调行为活跃)——"
           "A股尾盘放量常与次日高开正相关;尾盘占比低=量能集中在早盘"
           "(消息驱动型)。与 rel_vol_last_hour(最后1小时)区分:30分钟窗口更聚焦。",
    dependencies=("history_1min",),
)
def factor_tail_volume_share(context: FactorContext):
    return cross_sectional_rank(_metric(context, "close_30_vol_share"))


@register_factor(
    name="open_volume_share",
    description="开盘量能占比因子：开盘30分钟成交量占全天比例截面排名（负向，早盘情绪交易排后）。",
    category="intraday",
    thesis="开盘30分钟量能占比高=交易集中于早盘(散户情绪化参与高峰、信息冲击日);"
           "占比低=全天交易节奏均匀(机构化、策略化交易)。早盘集中放量常伴随"
           "日内冲高回落,负向排名。",
    dependencies=("history_1min",),
)
def factor_open_volume_share(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "open_30_vol_share"))


@register_factor(
    name="vwap_am_pm_gap_factor",
    description="上午/下午VWAP差因子：上午VWAP相对下午VWAP偏离截面排名（上午价格水平高排前）。",
    category="intraday",
    thesis="上午VWAP高于下午=价格重心逐段下移(早盘高买盘午后撤退);上午低于下午="
           "全天重心抬升(早盘低吸午后拉升)。VWAP是时段内真实成交均价,"
           "对时段强弱比收盘价比较更抗尾盘扰动。",
    dependencies=("history_1min",),
)
def factor_vwap_am_pm_gap_factor(context: FactorContext):
    return cross_sectional_rank(_metric(context, "vwap_am_pm_gap"))


@register_factor(
    name="open_30_range_share",
    description="开盘振幅占比因子：开盘30分钟振幅占全天振幅比例截面排名（负向，早盘大幅博弈排后）。",
    category="intraday",
    thesis="开盘30分钟振幅占全天比例高=全天的价格区间在早盘就已确定(方向快速)"
           "或早盘多空剧烈拉锯(方向未定);占比低=全天逐步展开。早盘定区间=日内"
           "趋势延续性强;早盘反复=方向博弈激烈,负向排名。",
    dependencies=("history_1min",),
)
def factor_open_30_range_share(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "open_30_range_pct"))


@register_factor(
    name="am_close_position",
    description="上午收盘位置因子：上午收盘在上午振幅区间中的相对位置截面排名（上午收高排前）。",
    category="intraday",
    thesis="上午收盘位于上午区间上沿=上午多方完全控制(下午大概率延续);位于下沿="
           "上午空方主导(下午承压)。与全日 close_position 区分:本因子以午间"
           "为分界,捕捉日内方向的时段归属与下午的延续概率。",
    dependencies=("history_1min",),
)
def factor_am_close_position(context: FactorContext):
    return cross_sectional_rank(_metric(context, "am_hl_position"))
