"""
Class 5 耦合因子(第五批) — 时间维度耦合 + 前四类新基因多因子耦合。

时间维度耦合(本批核心新增,全部滞后 k ∈ {5, 10, 20, 60} ≤ 60 天):
  1. 跨期自共振  X_t × X_{t-k} —— 同基因跨时点持续性(融资净流入 20 日持续)。
  2. 领先-滞后耦合  X_{t-k} × Y_t —— 前期信号对当前状态的确认(资金领先价格、
     量领先价、聪明钱领先动量)。
  3. 因子时间加速度  X_t − X_{t-k} —— rank 漂移 = 基因自身在时间轴上的改善
     (动量加速、资金流加速、筹码成本加速)。
  shift(k) 产生的样本期外(每股前 k 个交易日)NaN 按规范保留,不填充。

多因子耦合(新基因):Class 2 筹码(chip_support_strength / chip_peak_shift)、
  Class 3 盘中(tail_volume_share / smart_money_share)、Class 4 分钟指标
  (min_ma_alignment_frac_20)首次进入耦合层;risk / fund_flow / margin / event
  基因与动量/反转做共振。

方向契约:输入 .fea 均为「高=好方向」rank 编码,直接相乘/相减;
  负向因子在注册时已取反(rv_term_structure_slope 高=期限结构平缓、short_term_
  reversal_5 高=超跌、downside_frequency_60 高=下跌日少、margin_buy_pressure
  高=买入意愿强)。

⚠️ margin 系 .fea 覆盖仅约 85% 股票:含 margin 输入的耦合一律以 margin 因子
  索引为基准 reindex 其余因子,防索引并集把缺行以 NaN 拉入(8.13 先例)。

构建顺序:全部输入 .fea 已存在(截至 2026-08-08 全量构建),无跨批次依赖。
"""

from __future__ import annotations

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


def _lag(series, k: int):
    """Shift a (Date, Code) factor series back *k* trading days per stock.

    The first *k* trading days of each stock become NaN (out-of-range of the
    available history) — kept as NaN per the time-window contract (k ≤ 60).
    """
    return series.groupby(level="Code").shift(k)


# ══ 一、时间维度耦合:跨期自共振 ─────────────────────────────────────────────

@register_factor(
    name="coupling_flow_persistence_20",
    description="融资净流入跨期自共振：margin_net_flow_ratio_t × margin_net_flow_ratio_t-20 截面排名（20日持续净流入排前）。",
    category="coupling",
    thesis="融资净流入的跨时点持续性:今天与 20 天前都在净流入=杠杆资金的中期建仓行为,"
           "区别于单日脉冲式流入(游资一日游)。持续性资金比脉冲资金对趋势的支撑更可靠。"
           "纯单基因时间耦合,无并集风险。",
    dependencies=("__factors__", "margin_net_flow_ratio"),
)
def factor_coupling_flow_persistence_20(ctx: FactorContext):
    mg = ctx.load_factor("margin_net_flow_ratio")
    return cross_sectional_rank(mg * _lag(mg, 20))


@register_factor(
    name="coupling_margin_buy_persist_10",
    description="融资买入意愿跨期自共振：margin_buy_pressure_t × margin_buy_pressure_t-10 截面排名（买入意愿持续强于偿还排前）。",
    category="coupling",
    thesis="融资买入意愿(买入/偿还比)跨 10 日持续 > 0.5 = 多头杠杆资金在稳定加仓,"
           "而非一日冲高后的回落。持续性买入意愿的确认度高于单日读数。",
    dependencies=("__factors__", "margin_buy_pressure"),
)
def factor_coupling_margin_buy_persist_10(ctx: FactorContext):
    mbp = ctx.load_factor("margin_buy_pressure")
    return cross_sectional_rank(mbp * _lag(mbp, 10))


# ══ 二、时间维度耦合:领先-滞后交叉 ──────────────────────────────────────────

@register_factor(
    name="coupling_moneyflow_lead_momentum_10",
    description="资金领先动量：mf_net_inflow_ratio_t-10 × momentum_20_t 截面排名（10日前主力净流入+当前动量排前）。",
    category="coupling",
    thesis="领先-滞后耦合:10 天前的主力净流入是机构提前布局的证据,当前动量是布局后的"
           "价格兑现。资金先于价格(吸筹→拉升),故用 t-10 的资金确认 t 的价格趋势,"
           "比同日耦合(资金与价格同向共振)多一层因果时序,过滤'拉升中才追入'的跟风盘。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "momentum_20"),
)
def factor_coupling_moneyflow_lead_momentum_10(ctx: FactorContext):
    mf = ctx.load_factor("mf_net_inflow_ratio")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(_lag(mf, 10) * mom)


@register_factor(
    name="coupling_volume_lead_momentum_5",
    description="量能领先动量：volume_momentum_5_t-5 × momentum_20_t 截面排名（5日前量能扩张+当前动量排前）。",
    category="coupling",
    thesis="量在价先:5 天前的量能扩张先行,当前 20 日动量是量能推动的价格表现。"
           "与 momentum_volume_resonance_20(同日量价共振)区分:本因子要求量能"
           "提前确认,排除'当日才放量'的脉冲行情,捕捉量能持续推动的趋势。",
    dependencies=("__factors__", "volume_momentum_5", "momentum_20"),
)
def factor_coupling_volume_lead_momentum_5(ctx: FactorContext):
    vm = ctx.load_factor("volume_momentum_5")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(_lag(vm, 5) * mom)


@register_factor(
    name="coupling_smartmoney_lead_momentum_10",
    description="聪明钱领先动量：smart_money_share_t-10 × momentum_20_t 截面排名（10日前信息型交易活跃+当前动量排前）。",
    category="coupling",
    thesis="信息型交易(聪明钱)的活跃度领先于价格趋势:10 天前异动分钟成交占比高的"
           "股票,当前动量更可能由知情资金驱动而非散户跟风。Class 3 盘中基因"
           "首次以滞后形式进入耦合层。",
    dependencies=("__factors__", "smart_money_share", "momentum_20"),
)
def factor_coupling_smartmoney_lead_momentum_10(ctx: FactorContext):
    sms = ctx.load_factor("smart_money_share")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(_lag(sms, 10) * mom)


@register_factor(
    name="coupling_margin_lead_trend_60",
    description="杠杆资金领先趋势：margin_net_flow_ratio_t-60 × momentum_60_t 截面排名（60日前融资净流入+中期动量排前）。",
    category="coupling",
    thesis="最大滞后(60 天)的领先-滞后耦合:融资资金在 60 天前即开始净流入的中期趋势"
           "=杠杆资金的长线建仓行为,趋势级别高于短期资金推动。滞后 60 天恰好落在"
           "时间窗口平移上限。⚠️ 以 margin 因子索引 reindex 防并集。",
    dependencies=("__factors__", "margin_net_flow_ratio", "momentum_60"),
)
def factor_coupling_margin_lead_trend_60(ctx: FactorContext):
    mg = ctx.load_factor("margin_net_flow_ratio")
    mom = ctx.load_factor("momentum_60").reindex(mg.index)
    return cross_sectional_rank(_lag(mg, 60) * mom)


# ══ 三、时间维度耦合:因子时间加速度 ─────────────────────────────────────────

@register_factor(
    name="coupling_momentum_drift_20",
    description="动量时间加速度：momentum_20_t − momentum_20_t-20 截面排名（动量排名20日抬升=趋势加速排前）。",
    category="coupling",
    thesis="动量的动量:20 日动量截面排名相对 20 天前的漂移量。排名抬升=趋势在加速"
           "(新资金推动短周期走强),排名回落=动能衰竭的前兆。水平动量(动量_20 本身)"
           "与加速度解耦——加速度能更早捕捉拐点,避免在动量顶点追高。",
    dependencies=("__factors__", "momentum_20"),
)
def factor_coupling_momentum_drift_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(mom - _lag(mom, 20))


@register_factor(
    name="coupling_fundflow_accel_10",
    description="主力资金时间加速度：mf_net_inflow_ratio_t − mf_net_inflow_ratio_t-10 截面排名（净流入排名10日抬升排前）。",
    category="coupling",
    thesis="主力净流入截面排名的 10 日漂移:排名抬升=资金在转强(流出一致性减弱→流入"
           "一致性增强),排名回落=资金在转弱。与基于原始字段的 mf_flow_acceleration_5d"
           "区分:本因子作用于 rank 序列,天然截面可比、剔除个股量纲差异。",
    dependencies=("__factors__", "mf_net_inflow_ratio"),
)
def factor_coupling_fundflow_accel_10(ctx: FactorContext):
    mf = ctx.load_factor("mf_net_inflow_ratio")
    return cross_sectional_rank(mf - _lag(mf, 10))


@register_factor(
    name="coupling_chip_cost_accel_20",
    description="筹码成本上移加速度：chip_median_momentum_t − chip_median_momentum_t-20 截面排名（成本上移20日加速排前）。",
    category="coupling",
    thesis="筹码中位成本动量排名的 20 日漂移:排名持续抬升=吸筹成本在加速上移,"
           "主力在更高价位继续收集筹码,成本重心跟随价格上行=健康拉升;"
           "排名回落=成本重心滞涨,价格与筹码成本背离,警惕派发。",
    dependencies=("__factors__", "chip_median_momentum"),
)
def factor_coupling_chip_cost_accel_20(ctx: FactorContext):
    cm = ctx.load_factor("chip_median_momentum")
    return cross_sectional_rank(cm - _lag(cm, 20))


# ══ 四、多因子耦合:Class 2/3/4 新基因 ──────────────────────────────────────

@register_factor(
    name="coupling_chip_support_reversal_5",
    description="筹码支撑超跌因子：short_term_reversal_5 × chip_support_strength 截面排名（超跌且有筹码支撑排前）。",
    category="coupling",
    thesis="超跌反弹的质量过滤:单纯超跌(short_term_reversal_5 高)可能继续阴跌,"
           "但在筹码成本 15pct 支撑位上的超跌=下方有真实承接盘,反弹安全边际高。"
           "Class 2 筹码基因首次与反转因子耦合。",
    dependencies=("__factors__", "short_term_reversal_5", "chip_support_strength"),
)
def factor_coupling_chip_support_reversal_5(ctx: FactorContext):
    rev = ctx.load_factor("short_term_reversal_5")
    sup = ctx.load_factor("chip_support_strength").reindex(rev.index)
    return cross_sectional_rank(rev * sup)


@register_factor(
    name="coupling_chip_trend_confirm_20",
    description="筹码成本确认趋势：momentum_20 × chip_peak_shift 截面排名（趋势+成本峰20日上移排前）。",
    category="coupling",
    thesis="价格趋势与筹码成本重心同向:20 日动量上涨的同时,筹码峰(成本中位数)也在"
           "20 日上移=上涨由真实换手成本抬升支撑,而非缩量虚涨。与 chip_momentum_resonance_20"
           "(cr3 集中度)区分:本因子用成本重心位移,捕捉'换手推升'。",
    dependencies=("__factors__", "momentum_20", "chip_peak_shift"),
)
def factor_coupling_chip_trend_confirm_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    cps = ctx.load_factor("chip_peak_shift").reindex(mom.index)
    return cross_sectional_rank(mom * cps)


@register_factor(
    name="coupling_intraday_tail_momentum_20",
    description="尾盘资金确认趋势：momentum_20 × tail_volume_share 截面排名（趋势+尾盘放量排前）。",
    category="coupling",
    thesis="尾盘 30 分钟量能是当日资金态度的浓缩:上涨趋势中尾盘放量=资金当日尾段"
           "继续买入(次日延续性强),尾盘缩量=拉高无力承接。Class 3 盘中基因"
           "(tail_volume_share)首次与日线动量耦合。",
    dependencies=("__factors__", "momentum_20", "tail_volume_share"),
)
def factor_coupling_intraday_tail_momentum_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    tvs = ctx.load_factor("tail_volume_share").reindex(mom.index)
    return cross_sectional_rank(mom * tvs)


@register_factor(
    name="coupling_min_align_momentum_20",
    description="分钟均线确认趋势：momentum_20 × min_ma_alignment_frac_20 截面排名（趋势+分钟均线多头排列占比高排前）。",
    category="coupling",
    thesis="日内趋势结构稳固的日线趋势:20 日分钟均线多头排列占比高(ma5>ma10>ma20>"
           "ma30 的分钟占比均值高)=日内买盘持续占优,日线动量由日内结构支撑;"
           "日线动量高但分钟排列差=尾盘偷袭/脉冲行情。Class 4 分钟指标基因"
           "首次与日线动量耦合。",
    dependencies=("__factors__", "momentum_20", "min_ma_alignment_frac_20"),
)
def factor_coupling_min_align_momentum_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    align = ctx.load_factor("min_ma_alignment_frac_20").reindex(mom.index)
    return cross_sectional_rank(mom * align)


# ══ 五、多因子耦合:risk / fund_flow / margin / event / vol ───────────────

@register_factor(
    name="coupling_lowrisk_momentum_60",
    description="低风险趋势因子：momentum_60 × downside_frequency_60 截面排名（中期动量+下行频率低排前）。",
    category="coupling",
    thesis="趋势质量的另一个维度:60 日动量相同的股票,负收益日占比低者=上涨由连续的"
           "正收益构成(台阶式上行),而非大涨大跌的脉冲。与 defensive_momentum_combo_60"
           "(低beta×动量×低溃疡)区分:本因子直接用下行频率度量趋势的'流畅度'。",
    dependencies=("__factors__", "momentum_60", "downside_frequency_60"),
)
def factor_coupling_lowrisk_momentum_60(ctx: FactorContext):
    mom = ctx.load_factor("momentum_60")
    ds = ctx.load_factor("downside_frequency_60").reindex(mom.index)
    return cross_sectional_rank(mom * ds)


@register_factor(
    name="coupling_quality_trend_60",
    description="高质量趋势因子：momentum_60 × sortino_ratio_60 截面排名（中期动量+Sortino高排前）。",
    category="coupling",
    thesis="风险调整后的趋势:Sortino 比率(均收益/下行标准差)高的 60 日动量=上涨"
           "质量高(下行风险小)。与 coupling_lowrisk_momentum_60(下行频率)互补:"
           "本因子加权下行幅度,对'少而大的下跌'更敏感。",
    dependencies=("__factors__", "momentum_60", "sortino_ratio_60"),
)
def factor_coupling_quality_trend_60(ctx: FactorContext):
    mom = ctx.load_factor("momentum_60")
    so = ctx.load_factor("sortino_ratio_60").reindex(mom.index)
    return cross_sectional_rank(mom * so)


@register_factor(
    name="coupling_stableflow_momentum_20",
    description="稳定资金流趋势：momentum_20 × mf_flow_stability_20d 截面排名（趋势+主力资金连续同向排前）。",
    category="coupling",
    thesis="资金流方向一致性的确认:主力资金 20 日连续同向占比高=资金态度稳定(持续"
           "吸筹或持续撤退),叠加动量=稳定吸筹中的趋势;资金流频繁转向(不稳定)下的"
           "动量=博弈盘推动,持续性存疑。与 moneyflow_momentum_resonance_20(净流入"
           "水平)区分:本因子用方向稳定性而非水平。",
    dependencies=("__factors__", "momentum_20", "mf_flow_stability_20d"),
)
def factor_coupling_stableflow_momentum_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    st = ctx.load_factor("mf_flow_stability_20d").reindex(mom.index)
    return cross_sectional_rank(mom * st)


@register_factor(
    name="coupling_flowaccel_breakout_60",
    description="资金加速突破因子：big_order_net_accel_10 × new_high_60_event 截面排名（大单加速+60日新高排前）。",
    category="coupling",
    thesis="有效突破的资金验证:大单净额 5 日相对前 5 日加速流入(资金在突破前吸筹)"
           "+60 日新高事件=资金驱动的真突破;新高但大单净流出=拉高出货的假突破。"
           "与 liftoff 族(动量×放量×回撤)区分:本因子用事件衰减形态的新高+大单加速度。",
    dependencies=("__factors__", "big_order_net_accel_10", "new_high_60_event"),
)
def factor_coupling_flowaccel_breakout_60(ctx: FactorContext):
    acc = ctx.load_factor("big_order_net_accel_10")
    nh = ctx.load_factor("new_high_60_event").reindex(acc.index)
    return cross_sectional_rank(acc * nh)


@register_factor(
    name="coupling_margin_buy_trend_20",
    description="融资买入确认趋势：margin_buy_pressure × momentum_20 截面排名（买入意愿强+动量排前）。",
    category="coupling",
    thesis="杠杆资金参与度确认的趋势:融资买入/偿还比高=多头杠杆资金在主动加仓,"
           "动量叠加杠杆买入=两类资金合力;动量高但杠杆买入意愿弱=散户行情。"
           "⚠️ 以 margin 因子索引 reindex 防并集。",
    dependencies=("__factors__", "margin_buy_pressure", "momentum_20"),
)
def factor_coupling_margin_buy_trend_20(ctx: FactorContext):
    mbp = ctx.load_factor("margin_buy_pressure")
    mom = ctx.load_factor("momentum_20").reindex(mbp.index)
    return cross_sectional_rank(mbp * mom)


@register_factor(
    name="coupling_limitup_momentum_20",
    description="涨停延续动量：limit_up_fade_10 × momentum_20 截面排名（涨停后延续+动量排前）。",
    category="coupling",
    thesis="事件后的趋势延续:10 日内有涨停事件且其后累计收益为正(涨停后延续强)"
           "叠加 20 日动量=强势事件驱动的趋势;涨停后即回落(limit_up_fade_10 低)"
           "的动量=情绪顶点出货。用事件条件过滤动量中的'最后一段'。",
    dependencies=("__factors__", "limit_up_fade_10", "momentum_20"),
)
def factor_coupling_limitup_momentum_20(ctx: FactorContext):
    fade = ctx.load_factor("limit_up_fade_10")
    mom = ctx.load_factor("momentum_20").reindex(fade.index)
    return cross_sectional_rank(fade * mom)


@register_factor(
    name="coupling_volterm_momentum_60",
    description="平缓期限结构趋势：momentum_60 × rv_term_structure_slope 截面排名（中期动量+短期波动不陡峭排前）。",
    category="coupling",
    thesis="波动率期限结构(5min RV/60min RV)平缓=短期波动未放大=趋势未被噪声扰动,"
           "中期动量更'干净';期限结构陡峭(短期波动高)=筹码高度分歧,趋势随时"
           "被日内噪声打断。高 rv_term_structure_slope 已是低陡峭编码,直接相乘。",
    dependencies=("__factors__", "momentum_60", "rv_term_structure_slope"),
)
def factor_coupling_volterm_momentum_60(ctx: FactorContext):
    mom = ctx.load_factor("momentum_60")
    slope = ctx.load_factor("rv_term_structure_slope").reindex(mom.index)
    return cross_sectional_rank(mom * slope)
