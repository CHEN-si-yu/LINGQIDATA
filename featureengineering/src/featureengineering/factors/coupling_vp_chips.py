"""
Class 5 耦合因子(2026-08-11 批次) — 分钟级量价 × 筹码 × 动量 交叉耦合。

方向契约:输入 .fea 均为「高=好方向」编码(2026-08-11 复核):
  正向直接用:vp_consistency_20(量价一致)/ vp_expand_up_share(放量上涨)/
    vp_shrink_down_share(缩量下跌)/ vp_expand_price_pos(低位放量,spec 已取负)/
    minute_ret_vol_corr(量价同步)/ vp_expand_down_am_share(早盘恐慌释放)/
    chip_win_peak_frac(获利筹码集中)/ momentum_20 / short_term_reversal_5(超跌)/
    mf_net_inflow_ratio(主力净流入)。
  全部无需 (1.0 − X) 翻回,相乘即「高=多信号共振」。

构建顺序:本文件 6 个因子依赖 2026-08-11 批次的 Class 3 新因子
  (vp_* / minute_ret_vol_corr)与 Class 2 新因子(chip_win_peak_frac)的 .fea,
  必须先 `--only-class 3,2`(以及既有 momentum_20 / short_term_reversal_5 /
  mf_net_inflow_ratio 的 .fea)构建后再构建本模块。

与既有耦合区分:volume_price_liftoff_20 族用的是日频放量突破+动量+回撤;
  本模块全部基于分钟级量价新基因(四象限/量价相关/放量位置),描述
  「放量上涨是否有效」「缩量下跌是洗盘还是出货」「低位放量是否吸筹」等
  微观量价行为与筹码结构的联合确认。
"""

from __future__ import annotations

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


@register_factor(
    name="coupling_vp_chip_consistency_20",
    description="量价筹码共振因子：vp_consistency_20×chip_win_peak_frac截面排名（量价健康且筹码锁筹排前）。",
    category="coupling",
    thesis="分钟级量价一致性(涨有量跌无量)与获利筹码集中度(主力成本密集)的共振:"
           "两者同高=主力控盘+筹码锁定的健康趋势股(拉抬无抛压);量价一致但筹码"
           "分散=浮筹多,涨时兑现压力大。筹码维度把「量价健康」从市场行为升级为"
           "筹码结构确认,是吸筹完成后的典型状态。",
    dependencies=("__factors__", "vp_consistency_20", "chip_win_peak_frac"),
)
def factor_coupling_vp_chip_consistency_20(ctx: FactorContext):
    vp = ctx.load_factor("vp_consistency_20")
    chip = ctx.load_factor("chip_win_peak_frac").reindex(vp.index)
    return cross_sectional_rank(vp * chip)


@register_factor(
    name="coupling_vp_expand_up_mom_20",
    description="放量上涨动量确认因子：vp_expand_up_share×momentum_20截面排名（放量且动量确认排前）。",
    category="coupling",
    thesis="日内放量上涨的量占比与20日动量的共振:放量上涨+动量向上=上涨有量能"
           "与趋势双重确认(有效上涨,可持续);放量上涨但动量停滞=放量滞涨"
           "(对倒/出货嫌疑)。与 momentum_volume_resonance_20(日频量比×动量)"
           "区分:本因子用分钟级四象限的放量上涨占比,日内结构更细。",
    dependencies=("__factors__", "vp_expand_up_share", "momentum_20"),
)
def factor_coupling_vp_expand_up_mom_20(ctx: FactorContext):
    vp = ctx.load_factor("vp_expand_up_share")
    mom = ctx.load_factor("momentum_20").reindex(vp.index)
    return cross_sectional_rank(vp * mom)


@register_factor(
    name="coupling_vp_shrink_down_rev_5",
    description="缩量下跌反转确认因子：vp_shrink_down_share×short_term_reversal_5截面排名（洗盘缩量回调的超跌排前）。",
    category="coupling",
    thesis="缩量下跌(抛压轻/洗盘特征)与5日超跌的共振:两者同高=洗盘式回调后的"
           "超跌(浮筹清洗完毕、卖盘枯竭,反转概率大);缩量下跌但不超跌=正常回调"
           "未到买点。与 panic_selling_ratio_60(放量恐慌,负向逻辑)互为镜像——"
           "本因子捕捉「跌无量」的洗盘侧信号,是诱空识别的买侧视角。",
    dependencies=("__factors__", "vp_shrink_down_share", "short_term_reversal_5"),
)
def factor_coupling_vp_shrink_down_rev_5(ctx: FactorContext):
    vp = ctx.load_factor("vp_shrink_down_share")
    rev = ctx.load_factor("short_term_reversal_5").reindex(vp.index)
    return cross_sectional_rank(vp * rev)


@register_factor(
    name="coupling_vp_lowpos_accumulate_20",
    description="低位放量吸筹确认因子：vp_expand_price_pos×chip_win_peak_frac×mf_net_inflow_ratio截面排名（三因子吸筹确认排前）。",
    category="coupling",
    thesis="放量发生在日内低位(吸筹承接)×获利筹码集中(成本峰形成)×主力资金净流入"
           "的三重吸筹确认:放量位置、筹码结构、资金方向三个独立信号同时指向"
           "「主力在低位吸筹」,任一单信号都易被洗盘/对倒混淆,三者共振才是"
           "吸筹完成的高置信信号。与 smart_capital_liftoff_20(资金×动量×突破)"
           "区分:本因子锚定日内低位,是左侧吸筹视角。",
    dependencies=(
        "__factors__", "vp_expand_price_pos", "chip_win_peak_frac",
        "mf_net_inflow_ratio",
    ),
)
def factor_coupling_vp_lowpos_accumulate_20(ctx: FactorContext):
    pos = ctx.load_factor("vp_expand_price_pos")
    chip = ctx.load_factor("chip_win_peak_frac").reindex(pos.index)
    mf = ctx.load_factor("mf_net_inflow_ratio").reindex(pos.index)
    return cross_sectional_rank(pos * chip * mf)


@register_factor(
    name="coupling_vp_retvol_mom_20",
    description="量价同步动量因子：minute_ret_vol_corr×momentum_20截面排名（微观量价同步且趋势向上排前）。",
    category="coupling",
    thesis="日内分钟收益-量相关(微观量价同步度)与20日动量的共振:分钟级涨放量"
           "跌缩量+动量向上=趋势的微观基础健康(资金在每个价位都真实承接);"
           "动量向上但分钟量价脱钩(相关≈0/负)=对倒拉升,趋势脆弱。"
           "与 turnover_ret_corr_20(日频换手×收益)区分:本因子用日内分钟样本,"
           "捕捉盘中即时响应而非跨日窗口。",
    dependencies=("__factors__", "minute_ret_vol_corr", "momentum_20"),
)
def factor_coupling_vp_retvol_mom_20(ctx: FactorContext):
    corr = ctx.load_factor("minute_ret_vol_corr")
    mom = ctx.load_factor("momentum_20").reindex(corr.index)
    return cross_sectional_rank(corr * mom)


@register_factor(
    name="coupling_vp_amfade_rev_20",
    description="早盘恐慌反转因子：vp_expand_down_am_share×short_term_reversal_5截面排名（早盘恐慌释放的超跌排前）。",
    category="coupling",
    thesis="放量下跌集中在早盘(恐慌开盘释放)+5日超跌的共振:早盘恐慌杀跌是A股"
           "经典「诱空」形态(洗盘式砸盘),叠加超跌=恐慌抛压接近枯竭,午后/次日"
           "修复概率大;尾盘放量下跌(份额低)则相反=出货延续。与 vp_expand_down_share"
           "(恐慌总量)区分:本因子锚定恐慌的时段归属与超跌状态,是诱空识别的"
           "完整买侧信号。",
    dependencies=("__factors__", "vp_expand_down_am_share", "short_term_reversal_5"),
)
def factor_coupling_vp_amfade_rev_20(ctx: FactorContext):
    am = ctx.load_factor("vp_expand_down_am_share")
    rev = ctx.load_factor("short_term_reversal_5").reindex(am.index)
    return cross_sectional_rank(am * rev)
