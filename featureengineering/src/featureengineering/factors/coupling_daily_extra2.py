"""
多因子耦合补充 2 — Class 5 (factor-coupling, __factors__)。

升级方向(2026-08-05 第二轮):
1. 从双因子耦合升级为 **3 因子耦合**(量价起飞、三重左侧、质量三合等);
2. A股特色组合:量价起飞(放量突破+动量+趋势完整)、小盘起飞、主力资金起飞;
3. 全部依赖**已构建**的 .fea(8.11 批次未构建的 rs_60/kdj_daily_j/obv_slope_20
   等不在依赖内,避免构建死锁)。

方向契约(与 skill.md §8.11 一致,均先核对源因子注册方向):
- momentum_* 高=动量强;bp 高=低估;log_circ_mv 高=小盘;
- turnover_20 高=低换手(高换手用 1.0−X);amihud_intraday 高=高流动性;
- parkinson_vol 高=低波;idio_vol_60 高=低特质波;winner_rate 高=低获利盘;
- beta_60 高=低beta;corr_market_60 高=低市场相关;ulcer_index_20 高=低溃疡;
- drawdown_60/120 高=浅回撤(深回撤用 1.0−X);short_term_reversal_5 高=超跌;
- price_to_52w_high 高=接近新高;momentum_stability_20_60 高=趋势加速;
- volume_breakout_confirm_20 高=放量突破确认;volume_momentum_5 高=量增;
- amount_surge_count_20 高=资金脉冲频繁;mf_net_inflow_5d 高=主力净流入;
- margin_net_flow_ratio 高=融资净流入(⚠️ margin .fea 仅覆盖融资标的,
  参与耦合时以 margin 索引为基准 reindex,避免索引并集引入 25% 缺行 NaN)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ═══════════════════════════════════════════════════════════════════════════════
# A股特色:量价起飞族(放量突破 × 动量 × 趋势状态)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="volume_price_liftoff_20",
    description="量价起飞因子：momentum_20×volume_breakout_confirm_20×drawdown_60截面排名。",
    category="coupling",
    thesis="A股「量价起飞」的量化刻画:价格突破(动量+接近突破位)必须有量能确认"
           "(放量突破)且趋势完整(回撤浅)——三者共振=资金推动的实质行情启动,"
           "区别于缩量假突破与深回撤后的弱反弹。三重条件过滤最严格的起飞信号。",
    dependencies=("__factors__", "momentum_20", "volume_breakout_confirm_20", "drawdown_60"),
)
def factor_volume_price_liftoff_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    brk = ctx.load_factor("volume_breakout_confirm_20")
    dd = ctx.load_factor("drawdown_60")
    return cross_sectional_rank(mom * brk * dd)


@register_factor(
    name="smallcap_liftoff_combo_60",
    description="小盘量价起飞因子：log_circ_mv×momentum_60×volume_momentum_5截面排名。",
    category="coupling",
    thesis="A股小市值+量增+中期动量的经典起飞组合——小盘股弹性大,量能放大是"
           "资金进场的先行信号,60日动量确认趋势级别。小盘×量增×动量三重共振"
           "捕捉「小盘股放量启动」行情(小市值策略的动量增强版)。",
    dependencies=("__factors__", "log_circ_mv", "momentum_60", "volume_momentum_5"),
)
def factor_smallcap_liftoff_combo_60(ctx: FactorContext):
    size = ctx.load_factor("log_circ_mv")
    mom = ctx.load_factor("momentum_60")
    vm = ctx.load_factor("volume_momentum_5")
    return cross_sectional_rank(size * mom * vm)


@register_factor(
    name="value_liftoff_combo_20",
    description="价值量价起飞因子：bp×volume_breakout_confirm_20×momentum_20截面排名。",
    category="coupling",
    thesis="低估价值股出现放量突破且动量转正=价值发现的启动点(戴维斯双击的"
           "量价版本)——bp 提供安全边际,放量突破确认资金进场,动量确认趋势"
           "方向。比双因子 value_event_combo_20(仅bp×事件)多一重动量确认。",
    dependencies=("__factors__", "bp", "volume_breakout_confirm_20", "momentum_20"),
)
def factor_value_liftoff_combo_20(ctx: FactorContext):
    bp = ctx.load_factor("bp")
    brk = ctx.load_factor("volume_breakout_confirm_20")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(bp * brk * mom)


@register_factor(
    name="lowvol_liftoff_combo_20",
    description="低波量价起飞因子：parkinson_vol×volume_breakout_confirm_20×momentum_20截面排名。",
    category="coupling",
    thesis="低波动股票的放量突破=低波异象与突破信号的叠加——低波股突破的成功率"
           "高于高波股(噪音少、突破真实),放量确认+动量支持下的低波突破是最"
           "干净的趋势启动形态,与 high_quality_liquidity_combo 互补。",
    dependencies=("__factors__", "parkinson_vol", "volume_breakout_confirm_20", "momentum_20"),
)
def factor_lowvol_liftoff_combo_20(ctx: FactorContext):
    vol = ctx.load_factor("parkinson_vol")
    brk = ctx.load_factor("volume_breakout_confirm_20")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(vol * brk * mom)


@register_factor(
    name="liftoff_pulse_combo_20",
    description="资金脉冲起飞因子：amount_surge_count_20×momentum_10×drawdown_60截面排名。",
    category="coupling",
    thesis="资金脉冲频繁(成交额异常放大反复出现)+短期动量+趋势完整=大资金反复"
           "进出的活跃票正处于启动段——脉冲是资金行为痕迹,动量确认方向,"
           "浅回撤排除高位派发。与 volume_price_liftoff_20 的突破口径互补"
           "(脉冲口径不要求突破事件)。",
    dependencies=("__factors__", "amount_surge_count_20", "momentum_10", "drawdown_60"),
)
def factor_liftoff_pulse_combo_20(ctx: FactorContext):
    pulse = ctx.load_factor("amount_surge_count_20")
    mom = ctx.load_factor("momentum_10")
    dd = ctx.load_factor("drawdown_60")
    return cross_sectional_rank(pulse * mom * dd)


@register_factor(
    name="smart_capital_liftoff_20",
    description="主力资金起飞因子：mf_net_inflow_5d×momentum_20×volume_momentum_5截面排名。",
    category="coupling",
    thesis="主力资金持续净流入(5日)+动量向上+成交量放大=「聪明钱推动的起飞」——"
           "主力净流入是机构行为证据,量增提供流动性配合,动量确认方向。与"
           "smart_money_momentum_combo_20(分钟级聪明钱)区分:本因子为日频"
           "主力资金口径。",
    dependencies=("__factors__", "mf_net_inflow_5d", "momentum_20", "volume_momentum_5"),
)
def factor_smart_capital_liftoff_20(ctx: FactorContext):
    mf = ctx.load_factor("mf_net_inflow_5d")
    mom = ctx.load_factor("momentum_20")
    vm = ctx.load_factor("volume_momentum_5")
    return cross_sectional_rank(mf * mom * vm)


# ═══════════════════════════════════════════════════════════════════════════════
# A股特色:左侧价值 / 质量 / 防御
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="deep_value_reversal_combo_60",
    description="三重左侧复合因子：bp×(1−drawdown_120)×short_term_reversal_5截面排名。",
    category="coupling",
    thesis="低估(bp高)+深度回撤(半年尺度)+短期超跌=价值/回撤/反转三个维度的"
           "左侧共振——深度回撤压制情绪、低估提供安全边际、短期超跌提供弹性,"
           "三条件同时满足的股票是超跌价值修复的最佳候选。",
    dependencies=("__factors__", "bp", "drawdown_120", "short_term_reversal_5"),
)
def factor_deep_value_reversal_combo_60(ctx: FactorContext):
    bp = ctx.load_factor("bp")
    dd = ctx.load_factor("drawdown_120")
    rev = ctx.load_factor("short_term_reversal_5")
    return cross_sectional_rank(bp * (1.0 - dd) * rev)


@register_factor(
    name="lowvol_quality_momentum_60",
    description="低波质量动量因子：parkinson_vol×momentum_60×momentum_stability_20_60截面排名。",
    category="coupling",
    thesis="低波动+中期动量+趋势加速的三重确认——低波过滤噪音、60日动量确认"
           "趋势级别、20-60日动量差确认加速方向。三重共振=「低波+加速趋势」"
           "的最强形态,是低波异象与动量异象的乘积结构。",
    dependencies=("__factors__", "parkinson_vol", "momentum_60", "momentum_stability_20_60"),
)
def factor_lowvol_quality_momentum_60(ctx: FactorContext):
    vol = ctx.load_factor("parkinson_vol")
    mom = ctx.load_factor("momentum_60")
    acc = ctx.load_factor("momentum_stability_20_60")
    return cross_sectional_rank(vol * mom * acc)


@register_factor(
    name="chip_price_resonance_20",
    description="筹码×价格共振因子：winner_rate×momentum_20×price_to_52w_high截面排名。",
    category="coupling",
    thesis="筹码压力小(获利盘占比低、上方套牢出清)+动量向上+接近52周新高="
           "趋势的筹码结构与价格结构双重确认——筹码干净让上涨无解套抛压,"
           "接近新高确认空间打开。筹码(winner_rate)、动量(momentum)、"
           "位置(52w高)三个正交维度共振。",
    dependencies=("__factors__", "winner_rate", "momentum_20", "price_to_52w_high"),
)
def factor_chip_price_resonance_20(ctx: FactorContext):
    wr = ctx.load_factor("winner_rate")
    mom = ctx.load_factor("momentum_20")
    prox = ctx.load_factor("price_to_52w_high")
    return cross_sectional_rank(wr * mom * prox)


@register_factor(
    name="defensive_momentum_combo_60",
    description="防御动量复合因子：beta_60×momentum_60×ulcer_index_20截面排名。",
    category="coupling",
    thesis="低beta(市场敏感度低)+中期动量+低溃疡(回撤浅而短)=「不靠市场也能涨」"
           "的防御型趋势——低beta过滤系统性行情依赖,溃疡指数约束回撤体验,"
           "动量确认趋势。是低beta异象与动量异象的稳健交集。",
    dependencies=("__factors__", "beta_60", "momentum_60", "ulcer_index_20"),
)
def factor_defensive_momentum_combo_60(ctx: FactorContext):
    beta = ctx.load_factor("beta_60")
    mom = ctx.load_factor("momentum_60")
    ulcer = ctx.load_factor("ulcer_index_20")
    return cross_sectional_rank(beta * mom * ulcer)


@register_factor(
    name="multi_horizon_momentum_combo_20",
    description="多周期动量共振因子：momentum_5×momentum_10×momentum_20截面排名。",
    category="coupling",
    thesis="5/10/20日三个周期的动量同时为正且都强=短中周期趋势方向一致"
           "(多周期共振,信号最可靠)——单一周期动量可能只是噪音,三周期"
           "同向=趋势的内外结构同步,过滤了周期错位的伪信号。",
    dependencies=("__factors__", "momentum_5", "momentum_10", "momentum_20"),
)
def factor_multi_horizon_momentum_combo_20(ctx: FactorContext):
    m5 = ctx.load_factor("momentum_5")
    m10 = ctx.load_factor("momentum_10")
    m20 = ctx.load_factor("momentum_20")
    return cross_sectional_rank(m5 * m10 * m20)


@register_factor(
    name="fund_flow_alpha_combo_60",
    description="主力资金独立alpha因子：mf_net_inflow_5d×momentum_60×corr_market_60截面排名。",
    category="coupling",
    thesis="主力净流入+中期动量+低市场相关=「资金推动的独立行情」——低相关"
           "排除市场beta贡献(独立alpha),主力流入提供机构证据,动量确认趋势。"
           "与 beta_60(无条件暴露)互补:本因子显式要求独立性。",
    dependencies=("__factors__", "mf_net_inflow_5d", "momentum_60", "corr_market_60"),
)
def factor_fund_flow_alpha_combo_60(ctx: FactorContext):
    mf = ctx.load_factor("mf_net_inflow_5d")
    mom = ctx.load_factor("momentum_60")
    corr = ctx.load_factor("corr_market_60")
    return cross_sectional_rank(mf * mom * corr)


@register_factor(
    name="quality_liquidity_combo_20",
    description="质量流动性三合因子：amihud_intraday×parkinson_vol×idio_vol_60截面排名。",
    category="coupling",
    thesis="高流动性×低波动×低特质波动的三重质量筛选——高流动性保证策略可交易"
           "(冲击成本低),低波动保证可预测,低特质波动排除噪音型股票。三合="
           "最干净的可交易标的池,是 vol_liquidity_resonance_20 的特质波升级版。",
    dependencies=("__factors__", "amihud_intraday", "parkinson_vol", "idio_vol_60"),
)
def factor_quality_liquidity_combo_20(ctx: FactorContext):
    liq = ctx.load_factor("amihud_intraday")
    vol = ctx.load_factor("parkinson_vol")
    ivol = ctx.load_factor("idio_vol_60")
    return cross_sectional_rank(liq * vol * ivol)


@register_factor(
    name="reversal_liquidity_combo_5",
    description="超跌流动性复合因子：short_term_reversal_5×amihud_intraday×turnover_20截面排名。",
    category="coupling",
    thesis="短期超跌+高流动性+低换手=「可交易的缩量超跌」——超跌提供反弹弹性,"
           "高流动性保证实际可买入(超跌但流动性枯竭的股票无法交易),低换手"
           "说明抛压衰竭(缩量超跌=跌无可跌)。三个条件过滤出最具操作性的"
           "反弹候选。",
    dependencies=("__factors__", "short_term_reversal_5", "amihud_intraday", "turnover_20"),
)
def factor_reversal_liquidity_combo_5(ctx: FactorContext):
    rev = ctx.load_factor("short_term_reversal_5")
    liq = ctx.load_factor("amihud_intraday")
    to = ctx.load_factor("turnover_20")
    return cross_sectional_rank(rev * liq * to)


@register_factor(
    name="margin_trend_combo_20",
    description="杠杆趋势复合因子：margin_net_flow_ratio×momentum_20×drawdown_60截面排名。",
    category="coupling",
    thesis="融资净流入+价格动量+趋势完整的杠杆资金确认——融资加仓是杠杆资金"
           "的真金白银表态,动量确认方向,浅回撤确认趋势健康。三因子共振="
           "杠杆推动的趋势中段最强形态。⚠️ 以 margin 因子索引为基准 reindex,"
           "避免索引并集把 25% 缺行以 NaN 拉入(.fea 仅覆盖融资标的,"
           "与既有 margin 因子同口径)。",
    dependencies=("__factors__", "margin_net_flow_ratio", "momentum_20", "drawdown_60"),
)
def factor_margin_trend_combo_20(ctx: FactorContext):
    mf = ctx.load_factor("margin_net_flow_ratio")
    mom = ctx.load_factor("momentum_20").reindex(mf.index)
    dd = ctx.load_factor("drawdown_60").reindex(mf.index)
    return cross_sectional_rank(mf * mom * dd)


@register_factor(
    name="margin_value_combo_20",
    description="杠杆价值复合因子：margin_net_flow_ratio×bp×log_circ_mv截面排名。",
    category="coupling",
    thesis="融资净流入+低估+小盘=「杠杆资金抄底小盘价值」——融资盘在小盘价值股"
           "上的净流入往往是短线资金博弈与价值修复的混合信号,小盘放大弹性,"
           "低估提供安全边际。以 margin 索引为基准,与 margin_trend_combo_20 "
           "同口径。",
    dependencies=("__factors__", "margin_net_flow_ratio", "bp", "log_circ_mv"),
)
def factor_margin_value_combo_20(ctx: FactorContext):
    mf = ctx.load_factor("margin_net_flow_ratio")
    bp = ctx.load_factor("bp").reindex(mf.index)
    size = ctx.load_factor("log_circ_mv").reindex(mf.index)
    return cross_sectional_rank(mf * bp * size)
