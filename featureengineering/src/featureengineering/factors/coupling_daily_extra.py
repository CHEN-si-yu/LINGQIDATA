"""
日频耦合因子补充 — Class 5 (factor-coupling, __factors__)。

在 coupling_daily.py 基础上新增 8 个组合,全部使用既有 .fea 因子(均已构建)。

方向契约(与 skill.md §8.11 一致,耦合前已核对源因子注册方向):
- momentum_60/20 高=动量强;bp 高=低估;price_to_52w_high 高=接近新高;
- parkinson_vol 高=低波;kama_efficiency_20 高=趋势效率高;
- short_term_reversal_5 高=近5日超跌;williams_r_14 高=超卖;
- winner_rate 高=低获利盘(上方套牢少);rs_60 高=相对强势;
- volume_spike_event 高=放量事件新鲜;margin_net_flow_ratio 高=融资净流入强;
- smart_money_net_bias 高=聪明钱净买入;drawdown_60 高=回撤浅(深回撤用 1.0-drawdown_60);
- 需要"反向"一律用 (1.0 - X) 翻回,不二次取反。

构建顺序:全部依赖既有 .fea,无新依赖,可与 Class 1 新因子并行构建。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


@register_factor(
    name="value_reversal_combo_60",
    description="价值×深回撤复合因子：bp×(1−drawdown_60)截面排名（低估且深度回撤排前）。",
    category="coupling",
    thesis="低估(bp高)叠加深度回撤(回撤深=左侧)是价值投资的经典买点组合——"
           "回撤压制了短期情绪,低估提供了安全边际,两者共振=超跌价值股的修复空间"
           "最大(极值视角选股研报:估值与回撤双极端)。",
    dependencies=("__factors__", "bp", "drawdown_60"),
)
def factor_value_reversal_combo_60(ctx: FactorContext):
    bp = ctx.load_factor("bp")
    dd = ctx.load_factor("drawdown_60")
    return cross_sectional_rank(bp * (1.0 - dd))


@register_factor(
    name="momentum_high_proximity_combo_20",
    description="动量×52周高接近度复合因子：momentum_60×price_to_52w_high截面排名。",
    category="coupling",
    thesis="60日动量与52周高点接近度双确认=趋势既有动能(涨幅)又有空间状态"
           "(接近新高、套牢盘出清)——动量强但离高点远=反弹未到压力位;"
           "动量强且接近新高=突破在即的最强形态(George-Hwang 52周效应)。",
    dependencies=("__factors__", "momentum_60", "price_to_52w_high"),
)
def factor_momentum_high_proximity_combo_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_60")
    prox = ctx.load_factor("price_to_52w_high")
    return cross_sectional_rank(mom * prox)


@register_factor(
    name="lowvol_trend_efficiency_combo_20",
    description="低波×趋势效率复合因子：parkinson_vol×kama_efficiency_20截面排名（干净趋势排前）。",
    category="coupling",
    thesis="低波动(可预测、回撤浅)与高趋势效率(单边运行)共振=最干净的趋势行情——"
           "KAMA效率高但波动大=波动剧烈方向不稳定;低波但效率低=横盘整理。"
           "双条件共振过滤噪音,是低波异象与趋势跟踪的交集。",
    dependencies=("__factors__", "parkinson_vol", "kama_efficiency_20"),
)
def factor_lowvol_trend_efficiency_combo_20(ctx: FactorContext):
    vol = ctx.load_factor("parkinson_vol")
    eff = ctx.load_factor("kama_efficiency_20")
    return cross_sectional_rank(vol * eff)


@register_factor(
    name="reversal_oversold_combo_5",
    description="超跌×超卖复合因子：short_term_reversal_5×williams_r_14截面排名（超跌且超卖排前）。",
    category="coupling",
    thesis="短期超跌(近5日回落)与威廉%R超卖(价格贴近14日低区)双条件=时间维度与"
           "空间维度的超卖共振——单一条件可能只是普通回调,双条件同时满足时"
           "反弹的概率与弹性最大(短线反弹捕捉研报逻辑)。",
    dependencies=("__factors__", "short_term_reversal_5", "williams_r_14"),
)
def factor_reversal_oversold_combo_5(ctx: FactorContext):
    rev = ctx.load_factor("short_term_reversal_5")
    wr = ctx.load_factor("williams_r_14")
    return cross_sectional_rank(rev * wr)


@register_factor(
    name="winner_rs_combo_60",
    description="获利盘×相对强度复合因子：winner_rate×rs_60截面排名（筹码压力小且强势排前）。",
    category="coupling",
    thesis="获利盘占比低(上方套牢筹码出清、抛压小)且相对市场强势(资金持续流入)="
           "反弹阻力最小与上涨动力最强的组合——套牢盘少让涨势无解套抛压,"
           "RS强确认资金认可。筹码结构与相对强弱两个正交维度共振。",
    dependencies=("__factors__", "winner_rate", "rs_60"),
)
def factor_winner_rs_combo_60(ctx: FactorContext):
    wr = ctx.load_factor("winner_rate")
    rs = ctx.load_factor("rs_60")
    return cross_sectional_rank(wr * rs)


@register_factor(
    name="margin_price_resonance_20",
    description="融资流入×动量复合因子：margin_net_flow_ratio×momentum_20截面排名（杠杆加仓且上涨排前）。",
    category="coupling",
    thesis="杠杆资金净流入(融资买入>偿还)与价格动量共振=增量资金推动的趋势,"
           "真实性高于单纯动量(杠杆资金成本敏感、行为更谨慎)。融资加仓+上涨="
           "确认行情;融资流出+上涨=存量博弈。NaN≈15%(融资融券覆盖范围)。",
    dependencies=("__factors__", "margin_net_flow_ratio", "momentum_20"),
)
def factor_margin_price_resonance_20(ctx: FactorContext):
    mf = ctx.load_factor("margin_net_flow_ratio")
    # ⚠️ margin .fea 仅覆盖融资标的(约1.64M行),直接相乘会做索引并集,
    # 把25%缺行以NaN拉入结果 → 以 margin 索引为基准 reindex(2026-08-05修复)。
    mom = ctx.load_factor("momentum_20").reindex(mf.index)
    return cross_sectional_rank(mf * mom)


@register_factor(
    name="smart_money_momentum_combo_20",
    description="聪明钱×动量复合因子：smart_money_net_bias×momentum_20截面排名（聪明钱净买且上涨排前）。",
    category="coupling",
    thesis="聪明钱净方向与价格动量的共振=知情资金与市场趋势的方向一致——"
           "聪明钱净买入且动量向上=信息型资金推动的上涨(最可信);聪明钱净卖出"
           "但动量向上=散户接盘推动(危险)。用机构行为确认价格信号。",
    dependencies=("__factors__", "smart_money_net_bias", "momentum_20"),
)
def factor_smart_money_momentum_combo_20(ctx: FactorContext):
    bias = ctx.load_factor("smart_money_net_bias")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(bias * mom)
