"""
Daily-factor coupling module — Class 5 (factor-coupling, __factors__).

把新开发的日频因子族(RS相对强度、事件驱动、波动/流动性)与既有基因子
(momentum/bp/turnover/idio_vol/parkinson_vol/amihud)做组合。

方向契约(2026-08-05 复核):
- momentum_20/60 高=动量强;bp 高=低估;idio_vol_60 高=低特质波;
  parkinson_vol 高=低波;amihud_intraday 高=高流动性;
- turnover_20 高=低换手(.fea 已取反),需要"高换手"时用 (1.0 - to);
- rs_60 / limit_up_event_5 为新因子,高=信号强。

构建顺序依赖:依赖新 .fea(rs_60/limit_up_event_5)的耦合因子须等
Class 1 因子构建完成后构建。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


@register_factor(
    name="momentum_rs_resonance_20",
    description="动量×RS共振因子：momentum_20×rs_60截面排名（绝对与相对动量双强排前）。",
    category="coupling",
    thesis="绝对动量(自身涨幅)与相对动量(跑赢市场)双确认=趋势既有内生动力又有相对优势——"
           "RS强但动量弱=刚启动(左侧)；动量强但RS弱=市场beta贡献(虚胖)。双强共振最可靠。",
    dependencies=("__factors__", "momentum_20", "rs_60"),
)
def factor_momentum_rs_resonance_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    rs = ctx.load_factor("rs_60")
    return cross_sectional_rank(mom * rs)


@register_factor(
    name="lowvol_momentum_rs_20",
    description="低波×动量×RS复合因子：idio_vol_60×momentum_20×rs_60截面排名。",
    category="coupling",
    thesis="低波动异象+动量+相对强度的三重叠加——低特质波动的股票动量信号更干净"
           "(噪音少)，叠加RS确认相对优势，是风险调整后最强的趋势暴露。",
    dependencies=("__factors__", "idio_vol_60", "momentum_20", "rs_60"),
)
def factor_lowvol_momentum_rs_20(ctx: FactorContext):
    ivol = ctx.load_factor("idio_vol_60")
    mom = ctx.load_factor("momentum_20")
    rs = ctx.load_factor("rs_60")
    return cross_sectional_rank(ivol * mom * rs)


@register_factor(
    name="value_event_combo_20",
    description="价值×事件复合因子：bp×limit_up_event_5截面排名（低估且刚涨停排前）。",
    category="coupling",
    thesis="低估股票(bp高)出现涨停事件=价值发现启动(事件驱动研报的逻辑)——"
           "涨停带来关注度重估，低估提供安全边际，事件确认了催化剂。组合捕捉"
           "价值股的「戴维斯双击」启动点。",
    dependencies=("__factors__", "bp", "limit_up_event_5"),
)
def factor_value_event_combo_20(ctx: FactorContext):
    bp = ctx.load_factor("bp")
    ev = ctx.load_factor("limit_up_event_5")
    return cross_sectional_rank(bp * ev)


@register_factor(
    name="rs_value_divergence_20",
    description="RS-估值背离因子：rs_60−bp截面排名（强RS但高估值排前，谨慎信号）。",
    category="coupling",
    thesis="相对强度与估值的背离——RS强但bp低(估值贵)=趋势透支基本面的风险暴露；"
           "RS弱但bp高(便宜)=超跌价值股的潜在修复。排名高=趋势与估值脱节(谨慎)。",
    dependencies=("__factors__", "rs_60", "bp"),
)
def factor_rs_value_divergence_20(ctx: FactorContext):
    rs = ctx.load_factor("rs_60")
    bp = ctx.load_factor("bp")
    return cross_sectional_rank(rs - bp)


@register_factor(
    name="event_momentum_divergence_20",
    description="事件-动量背离因子：limit_up_event_5−momentum_20截面排名（事件强动量弱排前）。",
    category="coupling",
    thesis="涨停事件刚发生但20日动量尚未跟上=行情刚启动(事件领先于趋势)；"
           "动量已高但事件衰减=趋势中后段。背离项捕捉事件驱动的早期阶段。",
    dependencies=("__factors__", "limit_up_event_5", "momentum_20"),
)
def factor_event_momentum_divergence_20(ctx: FactorContext):
    ev = ctx.load_factor("limit_up_event_5")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(ev - mom)


@register_factor(
    name="turnover_event_confirmation_20",
    description="高换手×事件确认因子：(1−turnover_20)×limit_up_event_5截面排名。",
    category="coupling",
    thesis="涨停事件配合高换手=筹码充分交换、参与者进场充分(事件有效性高)；"
           "涨停但低换手=惜售一字板(事件可持续性存疑)。turnover_20的.fea高=低换手，"
           "故用(1−to)翻回高换手。",
    dependencies=("__factors__", "turnover_20", "limit_up_event_5"),
)
def factor_turnover_event_confirmation_20(ctx: FactorContext):
    to = ctx.load_factor("turnover_20")
    ev = ctx.load_factor("limit_up_event_5")
    return cross_sectional_rank((1.0 - to) * ev)


@register_factor(
    name="vol_liquidity_resonance_20",
    description="低波×高流动性共振因子：parkinson_vol×amihud_intraday截面排名。",
    category="coupling",
    thesis="低波动(可预测)与高流动性(可交易)共振=最干净的可交易标的——低波异象的收益"
           "在高流动性股票上可实际捕获(低流动性的低波股交易成本侵蚀收益)。",
    dependencies=("__factors__", "parkinson_vol", "amihud_intraday"),
)
def factor_vol_liquidity_resonance_20(ctx: FactorContext):
    vol = ctx.load_factor("parkinson_vol")
    amihud = ctx.load_factor("amihud_intraday")
    return cross_sectional_rank(vol * amihud)


@register_factor(
    name="momentum_stability_combo_60",
    description="动量×稳定性复合因子：momentum_60×momentum_stability_20_60截面排名。",
    category="coupling",
    thesis="60日动量叠加趋势加速度(20-60动量差)——动量强且正在加速=趋势中段最强形态；"
           "动量强但减速=趋势衰竭前兆。加速度确认后的动量暴露更安全。",
    dependencies=("__factors__", "momentum_60", "momentum_stability_20_60"),
)
def factor_momentum_stability_combo_60(ctx: FactorContext):
    mom60 = ctx.load_factor("momentum_60")
    accel = ctx.load_factor("momentum_stability_20_60")
    return cross_sectional_rank(mom60 * accel)
