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


@register_factor(
    name="intraday_high_time",
    description="日内最高价时点因子：日内最高价出现的归一化时段截面排名（尾盘创新高排前）。",
    category="intraday",
    thesis="日内最高价出现时点是多空掌控力的指纹:强势股尾盘创新高(时点→1),"
           "弱势股早盘冲高回落(时点→0.1)。与 volume_peak_time(量能时点)正交:"
           "早盘放量+尾盘新高=有效放量,早盘放量+早盘见顶=出货特征。",
    dependencies=("history_1min",),
)
def factor_intraday_high_time(context: FactorContext):
    return cross_sectional_rank(_metric(context, "intraday_high_time"))


@register_factor(
    name="min_limit_touch_frac_20",
    description="分钟触板密度因子：20日(分钟涨幅≥9.8%占比)均值截面排名（封板维持时间长排前）。",
    category="intraday",
    thesis="日频炸板率(limit_up_open_fail_freq_20)只有「封没封上」两个状态,"
           "分钟线给出「在板上的时间」——封板时间长=封单牢固(秒板/回封与"
           "临收盘偷袭板在此分离)。用 pre_close×1.098 近似涨停价,"
           "与全库 9.8% 口径一致,除权日无假触板。",
    dependencies=("history_1min", "daily.parquet"),
)
def factor_min_limit_touch_frac_20(context: FactorContext):
    return cross_sectional_rank(_metric(context, "min_limit_touch_frac_20"))


@register_factor(
    name="min_bar_gap_freq_20",
    description="分钟跳空频率因子：20日(|分钟开盘/前分钟收盘−1|>0.2%)占比均值截面排名（负向，盘口断档排后）。",
    category="intraday",
    thesis="1 分钟级频繁跳空=连续竞价断档=流动性薄/大单砸单;低频率=盘口连续="
           "流动性好(机构单边托单)。与 rjump_5min(跳变幅度)区分:本因子是"
           "跳空发生的频率(结构维),流动性差的题材小票频率数倍于蓝筹。",
    dependencies=("history_1min",),
)
def factor_min_bar_gap_freq_20(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "min_bar_gap_freq_20"))


@register_factor(
    name="min_vwap_dev_std",
    description="VWAP贴合度因子：20日(分钟价对当日VWAP偏离的标准差)均值截面排名（负向，价格围绕VWAP剧烈摆动排后）。",
    category="intraday",
    thesis="价格围绕当日 VWAP 的日内摆动幅度:稳定贴合=机构按 VWAP 单边建仓/控盘,"
           "剧烈摆动=多空拉锯(趋势质量差)。与 vwap_daily_deviation(仅收盘一个点)"
           "区分:本因子用全日内分布,趋势票与对倒票分离。"
           "口径修正:1min vol=手(实证),×100 后 amount/vol=真 VWAP。",
    dependencies=("history_1min",),
)
def factor_min_vwap_dev_std(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "min_vwap_dev_std"))


# ═══════════════════════════════════════════════════════════════════════════════
# 2026-08-11 分钟级量价四象限 (放量/缩量 × 上涨/下跌)
# ═══════════════════════════════════════════════════════════════════════════════
# 主构建路径为 INTRADAY_FACTOR_SPEC 扇出(见 intraday.py),本段注册函数为
# 单因子直连构建的回退实现,方向与 spec 同名条目一一对应:
#   pos → cross_sectional_rank(+metric), neg → cross_sectional_rank(−metric)。
# 放量基准 = 分钟量 vs 过去20日同时段均量(时段基准校正 U 型曲线)。


@register_factor(
    name="vp_expand_up_share",
    description="放量上涨量占比因子：日内放量(超20日同时段均量)上涨分钟量占全天量比例截面排名（涨有量排前）。",
    category="intraday",
    thesis="上涨分钟中放量的量占比衡量「上涨的量能质量」——放量上涨=资金真实做多"
           "(增量承接,趋势可持续);缩量上涨=无量反弹(诱多嫌疑,见 vp_shrink_up_share)。"
           "与日频 up_day_volume_ratio_20 区分:本因子为分钟粒度,同一天内即可区分"
           "「涨时放量/跌时缩量」与「跌时放量/涨时缩量」的微观差异。",
    dependencies=("history_1min",),
)
def factor_vp_expand_up_share(context: FactorContext):
    return cross_sectional_rank(_metric(context, "vp_expand_up_share"))


@register_factor(
    name="vp_expand_down_share",
    description="放量下跌量占比因子：日内放量下跌分钟量占全天量比例截面排名（负向，恐慌抛售排后）。",
    category="intraday",
    thesis="放量下跌=恐慌抛售/主力出货(跌有量,承接不足);占比高=当日筹码在下跌中"
           "被大量换手,抛压沉重。与 panic_selling_ratio_60(日频放量×下跌联合)区分:"
           "本因子为分钟粒度,放量基准为同时段均量,能捕捉日内局部的放量砸盘。"
           "方向 neg:放量下跌占比低(下跌无量)排前。",
    dependencies=("history_1min",),
)
def factor_vp_expand_down_share(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "vp_expand_down_share"))


@register_factor(
    name="vp_shrink_up_share",
    description="缩量上涨量占比因子：日内缩量上涨分钟量占全天量比例截面排名（负向，无量反弹排后）。",
    category="intraday",
    thesis="上涨但缩量(量低于同时段均量)=无量反弹/对倒拉升——价格上涨缺乏成交确认,"
           "典型诱多形态(拉高无人跟风,后续易回落)。占比高=当日上涨质量差,"
           "方向 neg。与 vp_expand_up_share 互补,二者合计=上涨分钟的量占比。",
    dependencies=("history_1min",),
)
def factor_vp_shrink_up_share(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "vp_shrink_up_share"))


@register_factor(
    name="vp_shrink_down_share",
    description="缩量下跌量占比因子：日内缩量下跌分钟量占全天量比例截面排名（跌无量排前）。",
    category="intraday",
    thesis="下跌但缩量=无量阴跌/洗盘回调——卖盘枯竭、抛压轻(浮筹锁定),常是主力"
           "洗盘而非出逃(出逃必然放量)。占比高=当日下跌质量好,方向 pos。"
           "与 vp_expand_down_share 互补:跌时量能结构刻画恐慌 vs 洗盘。",
    dependencies=("history_1min",),
)
def factor_vp_shrink_down_share(context: FactorContext):
    return cross_sectional_rank(_metric(context, "vp_shrink_down_share"))


@register_factor(
    name="vp_consistency_score",
    description="量价一致性得分因子：四象限一致性(放量涨+缩量跌−缩量涨−放量跌)/全天量截面排名（量价健康排前）。",
    category="intraday",
    thesis="四象限联合得分:涨有量+跌无量(量价健康)为正,涨无量+跌有量(背离/出货)"
           "为负。该得分把「诱多(缩量涨)」「诱空(放量跌)」两类陷阱统一到一条"
           "量纲一致的轴上,是量价关系的综合度量,与各象限占比互补。",
    dependencies=("history_1min",),
)
def factor_vp_consistency_score(context: FactorContext):
    return cross_sectional_rank(_metric(context, "vp_consistency_score"))


@register_factor(
    name="vp_consistency_20",
    description="量价一致性20日均值因子：四象限一致性得分的20日均值截面排名（持续量价健康排前）。",
    category="intraday",
    thesis="单日一致性噪声大,20日均值度量「量价配合的持续性」——持续涨有量跌无量"
           "=资金长期驻留(吸筹/控盘特征);持续背离=出货周期。与 vp_consistency_score"
           "区分:本因子过滤日内噪声,信号更稳,适合与动量/筹码因子做 Class 5 耦合。",
    dependencies=("history_1min",),
)
def factor_vp_consistency_20(context: FactorContext):
    return cross_sectional_rank(_metric(context, "vp_consistency_20"))


@register_factor(
    name="up_minute_vol_share",
    description="上涨分钟量占比因子：当日上涨分钟成交量占全天量比例截面排名（涨时整体放量排前）。",
    category="intraday",
    thesis="当日上涨分钟的量占比——分钟级版 up_day_volume_ratio_20(日频按天判涨跌)。"
           "与 up_minutes_ratio(上涨分钟数占比)区分:本因子是量维度,资金重仓参与的"
           "上涨与零星反弹在量占比上分离。",
    dependencies=("history_1min",),
)
def factor_up_minute_vol_share(context: FactorContext):
    return cross_sectional_rank(_metric(context, "up_minute_vol_share"))


@register_factor(
    name="minute_ret_vol_corr",
    description="分钟量价相关因子：日内分钟收益与分钟量的皮尔逊相关截面排名（量价同步排前）。",
    category="intraday",
    thesis="分钟级收益-量相关是量价同步的微观度量:正相关=涨放量跌缩量(健康);"
           "负相关=涨缩量跌放量(背离)。与日频 turnover_ret_corr_20/volume_price_corr_20"
           "区分:本因子用日内分钟样本,不受跨日窗口长度影响,捕捉盘中即时响应。",
    dependencies=("history_1min",),
)
def factor_minute_ret_vol_corr(context: FactorContext):
    return cross_sectional_rank(_metric(context, "minute_ret_vol_corr"))


@register_factor(
    name="vp_expand_ret_gap",
    description="放缩量收益差因子：放量分钟均收益−缩量分钟均收益截面排名（放量推动价格排前）。",
    category="intraday",
    thesis="放量分钟的涨幅相对缩量分钟的涨幅之差——度量「放量是否推得动价格」。"
           "放量上涨而价格不动(差值≈0/负)=对倒/出货(大单对敲吸引跟风);"
           "放量即涨=增量资金真实进场。是识别诱多(放量不涨)的关键信号,"
           "与 volume_price_divergence_score(日频价量动量差)区分:本因子为日内分钟口径。",
    dependencies=("history_1min",),
)
def factor_vp_expand_ret_gap(context: FactorContext):
    return cross_sectional_rank(_metric(context, "vp_expand_ret_gap"))


@register_factor(
    name="vp_expand_price_pos",
    description="放量价格位置因子：放量分钟的量加权日内位置均值截面排名（负向，低位放量=吸筹排前）。",
    category="intraday",
    thesis="放量发生在日内什么价位:低位(接近当日低点)放量=承接吸筹(主力低位吃货);"
           "高位放量=追高/出货(拉高出货特征)。方向 neg:高位放量排后。"
           "与 close_position(收盘单点位)区分:本因子用全部放量分钟的分布,"
           "是「吸筹 vs 出货」的日内位置刻画。",
    dependencies=("history_1min",),
)
def factor_vp_expand_price_pos(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "vp_expand_price_pos"))


@register_factor(
    name="vp_expand_down_am_share",
    description="早盘放量下跌占比因子：早盘(09:30-11:30)放量下跌量占全天放量下跌量比例截面排名（早盘恐慌释放排前）。",
    category="intraday",
    thesis="放量下跌发生在早盘=恐慌在开盘集中释放(洗筹,午后修复概率大,A股常见"
           "「早盘杀跌午后V」);发生在尾盘=出货延续(次日低开风险)。方向 pos:"
           "早盘集中排前。与 vp_expand_down_share(全天放量下跌总量)区分:"
           "本因子度量恐慌的时段归属,是诱空识别的重要维度。",
    dependencies=("history_1min",),
)
def factor_vp_expand_down_am_share(context: FactorContext):
    return cross_sectional_rank(_metric(context, "vp_expand_down_am_share"))
