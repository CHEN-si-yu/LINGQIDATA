"""
Class 5 耦合因子(第三批) — 资金流×杠杆×筹码×动量 交叉验证。

方向契约(2026-08-08 复核,输入 .fea 均为「高=好方向」编码):
  正向直接用:mf_net_inflow_ratio(主力净流入)/ margin_net_flow_ratio(融资净流入)/
    net_turnover_rate_20(净换手)/ lhb_proxy_score_60(博弈活跃度)/
    margin_chip_cost_gap(融资成本不高于市场成本,注册时已取负)/
    momentum_20 / short_term_reversal_5(超跌) / drawdown_60(浅回撤)/
    winner_rate 需要翻回:(1.0 − winner_rate) = 高获利盘。

构建顺序:本文件 3 个因子依赖 2026-08-08 批次的 Class 1 新因子
  (net_turnover_rate_20 / lhb_proxy_score_60 / margin_chip_cost_gap)的 .fea,
  必须先 `--only-class 1` 构建后再构建本模块。

⚠️ margin 系 .fea 覆盖仅约 85% 股票:含 margin 输入的耦合一律以 margin
  因子索引为基准 reindex 其余因子,防索引并集把缺行以 NaN 拉入(8.13 先例)。
"""

from __future__ import annotations

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


@register_factor(
    name="coupling_bigflow_margin_buy_20",
    description="大单融资双确认因子：mf_net_inflow_ratio×margin_net_flow_ratio截面排名（两类聪明钱同向流入排前）。",
    category="coupling",
    thesis="场内大单(主力资金)与场外杠杆资金(融资净流入)同时净流入=两类聪明钱"
           "互相印证,信号质量最高;单边流入但另一边撤退(分歧)则被乘法稀释。"
           "与 margin_price_resonance_20(融资×价格)区分:本因子是资金×资金"
           "的双确认,不依赖价格方向。⚠️ 以 margin 因子索引 reindex 防并集。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "margin_net_flow_ratio"),
)
def factor_coupling_bigflow_margin_buy_20(ctx: FactorContext):
    mg = ctx.load_factor("margin_net_flow_ratio")
    mf = ctx.load_factor("mf_net_inflow_ratio").reindex(mg.index)
    return cross_sectional_rank(mg * mf)


@register_factor(
    name="coupling_winner_bigflow_20",
    description="获利盘大单共振因子：(1−winner_rate)×mf_net_inflow_ratio截面排名（获利盘多且主力净买入排前）。",
    category="coupling",
    thesis="获利盘占比抬升(筹码在涨,散户跟风)+大单持续净买(主力拉升)=健康上行"
           "的正反馈;大单流出但获利盘上升(诱多)与两者同步(健康上行)在此分离。"
           "winner_rate .fea 高=低获利盘,须 (1.0−X) 翻回(8.11 二次取反先例)。",
    dependencies=("__factors__", "winner_rate", "mf_net_inflow_ratio"),
)
def factor_coupling_winner_bigflow_20(ctx: FactorContext):
    win = ctx.load_factor("winner_rate")
    mf = ctx.load_factor("mf_net_inflow_ratio").reindex(win.index)
    return cross_sectional_rank((1.0 - win) * mf)


@register_factor(
    name="coupling_margin_chip_cost_20",
    description="融资筹码成本复合因子：margin_chip_cost_gap×momentum_20×drawdown_60截面排名（杠杆结构健康的中期趋势排前）。",
    category="coupling",
    thesis="融资盘建仓成本不高于市场筹码平均成本(未高位接盘,止损踩踏风险低)×"
           "中期动量×趋势完整(回撤浅)=杠杆结构健康的中期上行趋势。三因子共振"
           "过滤「杠杆资金高位接盘后的动量陷阱」。⚠️ 以 margin 因子索引"
           "reindex 其余因子防索引并集。",
    dependencies=("__factors__", "margin_chip_cost_gap", "momentum_20", "drawdown_60"),
)
def factor_coupling_margin_chip_cost_20(ctx: FactorContext):
    gap = ctx.load_factor("margin_chip_cost_gap")
    mom = ctx.load_factor("momentum_20").reindex(gap.index)
    dd = ctx.load_factor("drawdown_60").reindex(gap.index)
    return cross_sectional_rank(gap * mom * dd)


@register_factor(
    name="coupling_net_turnover_momentum_20",
    description="净换手动量共振因子：net_turnover_rate_20×momentum_20截面排名（净买入驱动的动量排前）。",
    category="coupling",
    thesis="净换手率(主动净买量/自由流通股本)与动量共振:净买入驱动的上涨=资金"
           "真实承接(区别于对倒/缩量假涨);净卖出中的上涨=出货嫌疑。"
           "与 momentum_volume_resonance_20(总量能)区分:本因子用主动买卖"
           "的净量口径,剔除被动成交噪声。",
    dependencies=("__factors__", "net_turnover_rate_20", "momentum_20"),
)
def factor_coupling_net_turnover_momentum_20(ctx: FactorContext):
    ntr = ctx.load_factor("net_turnover_rate_20")
    mom = ctx.load_factor("momentum_20").reindex(ntr.index)
    return cross_sectional_rank(ntr * mom)


@register_factor(
    name="coupling_lhb_reversal_20",
    description="博弈票超跌反弹因子：lhb_proxy_score_60×short_term_reversal_5截面排名（游资惯犯且超跌排前）。",
    category="coupling",
    thesis="龙虎榜替代活跃度(游资博弈票)×5日超跌=游资关注的超跌反弹标的:博弈票"
           "弹性大、超跌后反弹兑现快;单纯超跌但无人问津的票(低活跃度)被过滤。"
           "与 reversal_oversold_combo_5(通用超跌)区分:本因子限定游资博弈"
           "人群,聚焦题材票的反弹窗口。",
    dependencies=("__factors__", "lhb_proxy_score_60", "short_term_reversal_5"),
)
def factor_coupling_lhb_reversal_20(ctx: FactorContext):
    lhb = ctx.load_factor("lhb_proxy_score_60")
    rev = ctx.load_factor("short_term_reversal_5").reindex(lhb.index)
    return cross_sectional_rank(lhb * rev)
