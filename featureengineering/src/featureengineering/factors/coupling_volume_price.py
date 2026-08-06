"""
Volume-price coupling factors — Class 5 (factor-coupling, __factors__).

量价耦合因子:把从未被耦合的动量/反转基因子引入耦合层,与量能(换手/资金流/流动性)、
筹码(集中度/获利盘)、估值(bp/dv)、规模、波动率基因子做交互。

核心缺口(2026-08-01 审计):momentum_* 家族、short_term_reversal_5 此前从未被任何
Class5 因子引用;本模块以 momentum_20 为主输入构建共振/背离/复合三类模式。

注意:基因子 momentum_5/10/20/60 等的 .fea 需先按修复后的 _adjusted_close 重建
(2026-08-01 修复),否则构建本模块因子会读到旧 bug 值(每日截面唯一值=1)。
"""

from __future__ import annotations

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── 动量 × 量能 ───────────────────────────────────────────────────────────

@register_factor(
    name="momentum_volume_resonance_20",
    description="动量×换手共振：momentum_20与turnover_20双高排名。放量上涨=资金确认的趋势。",
    category="coupling",
    thesis="价格动量与换手率同时高=上涨由持续的交易参与推动(量价共振)；"
           "动量高但换手低=缩量上涨,趋势未获资金确认。共振项捕捉'有人气的趋势'。",
    dependencies=("__factors__", "momentum_20", "turnover_20"),
)
def factor_momentum_volume_resonance_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    # turnover_20 的 .fea 为 rank(-换手),高=低换手,故 (1.0 - to) 翻回"高换手"
    to = ctx.load_factor("turnover_20")
    return cross_sectional_rank(mom * (1.0 - to))


@register_factor(
    name="momentum_volume_divergence_20",
    description="量价背离：momentum_20−volume_momentum_5。价升量缩(背离)排名高=缺乏资金确认。",
    category="coupling",
    thesis="价格趋势与成交量趋势背离(价升量缩/价跌量增)意味着走势缺乏真实资金支持，"
           "持续性弱、反转风险大。该因子排名高=背离显著,作为趋势的谨慎信号。",
    dependencies=("__factors__", "momentum_20", "volume_momentum_5"),
)
def factor_momentum_volume_divergence_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    vol_mom = ctx.load_factor("volume_momentum_5")
    return cross_sectional_rank(mom - vol_mom)


@register_factor(
    name="momentum_liquidity_resonance_20",
    description="动量×高流动性：momentum_20×amihud_intraday。流动性好的股票动量更可靠。",
    category="coupling",
    thesis="高流动性股票的动量来自真实成交、可交易性强；低流动性股票的动量易被少量资金扭曲"
           "(虚假拉升),且交易成本侵蚀收益。amihud_intraday 的 .fea 高=高流动性,直接作权重。",
    dependencies=("__factors__", "momentum_20", "amihud_intraday"),
)
def factor_momentum_liquidity_resonance_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    amihud = ctx.load_factor("amihud_intraday")
    return cross_sectional_rank(mom * amihud)


@register_factor(
    name="momentum_lowvol_combo_20",
    description="动量×低波动：momentum_20×parkinson_vol。低波动趋势股的风险调整后动量。",
    category="coupling",
    thesis="低波动异象(低波股经风险调整后收益更高)与动量结合:低波动+强趋势="
           "风险调整后动量最强,回撤可控。parkinson_vol 的 .fea 高=低波,直接作权重。",
    dependencies=("__factors__", "momentum_20", "parkinson_vol"),
)
def factor_momentum_lowvol_combo_20(ctx: FactorContext):
    mom = ctx.load_factor("momentum_20")
    vol = ctx.load_factor("parkinson_vol")
    return cross_sectional_rank(mom * vol)


# ── 动量 × 估值 ───────────────────────────────────────────────────────────

@register_factor(
    name="bp_momentum_combo_20",
    description="价值+动量复合：(bp+momentum_20)/2。低估且走强的股票双因子确认。",
    category="coupling",
    thesis="经典 HML×MOM 的 A股实现:既便宜(bp高)又在上涨(momentum_20高)的股票"
           "同时获得价值与趋势资金的支撑,是价值-动量复合中最常见的alpha组合。",
    dependencies=("__factors__", "bp", "momentum_20"),
)
def factor_bp_momentum_combo_20(ctx: FactorContext):
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank((bp + mom) / 2.0)


@register_factor(
    name="bp_momentum_divergence_20",
    description="估值-趋势背离：momentum_20−bp。趋势强但估值贵的股票(估值透支)排名高。",
    category="coupling",
    thesis="趋势与估值背离的两种情形:①趋势强+估值贵=涨幅透支基本面,回调风险大(排名高=谨慎);"
           "②趋势弱+估值便宜=超跌价值股,可能被错杀。该因子作为估值约束的动量信号。",
    dependencies=("__factors__", "bp", "momentum_20"),
)
def factor_bp_momentum_divergence_20(ctx: FactorContext):
    bp = ctx.load_factor("bp")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(mom - bp)


# ── 反转 × 量能 ───────────────────────────────────────────────────────────

@register_factor(
    name="reversal_turnover_resonance_5",
    description="高换手反转：short_term_reversal_5×turnover_20。高换手的超跌股反转概率更高。",
    category="coupling",
    thesis="短期反转效应在高换手股票上显著更强(高换手=投资者情绪驱动、过度反应更极端)。"
           "超跌(反转因子高)×高换手=博弈资金开始回补,反转行情启动概率高。",
    dependencies=("__factors__", "short_term_reversal_5", "turnover_20"),
)
def factor_reversal_turnover_resonance_5(ctx: FactorContext):
    rev = ctx.load_factor("short_term_reversal_5")
    # turnover_20 的 .fea 高=低换手, (1.0 - to) 翻回"高换手"
    to = ctx.load_factor("turnover_20")
    return cross_sectional_rank(rev * (1.0 - to))


@register_factor(
    name="moneyflow_reversal_divergence_5",
    description="资金×反转背离：mf_net_inflow_ratio−short_term_reversal_5。主力流入但股价超跌=潜在反转。",
    category="coupling",
    thesis="主力资金持续流入(净流入排名高)但股价仍在超跌区(反转因子低)——"
           "机构在低位吸筹而散户情绪仍悲观,是经典的左侧反转买点。背离越大信号越强。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "short_term_reversal_5"),
)
def factor_moneyflow_reversal_divergence_5(ctx: FactorContext):
    mf = ctx.load_factor("mf_net_inflow_ratio")
    rev = ctx.load_factor("short_term_reversal_5")
    return cross_sectional_rank(mf - rev)


# ── 动量 × 资金流 ─────────────────────────────────────────────────────────

@register_factor(
    name="moneyflow_momentum_resonance_20",
    description="资金确认动量：mf_net_inflow_ratio×momentum_20。主力净流入+上涨=趋势有资金背书。",
    category="coupling",
    thesis="上涨趋势若同时有主力资金净流入背书,则趋势由机构资金驱动而非散户跟风,"
           "延续性更强。资金与价格双确认是趋势交易的核心过滤条件。",
    dependencies=("__factors__", "mf_net_inflow_ratio", "momentum_20"),
)
def factor_moneyflow_momentum_resonance_20(ctx: FactorContext):
    mf = ctx.load_factor("mf_net_inflow_ratio")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(mf * mom)


@register_factor(
    name="bigorder_momentum_resonance_20",
    description="大单×动量：mf_big_order_ratio×momentum_20。大单占比高+上涨=机构主导的行情。",
    category="coupling",
    thesis="大单占比反映机构/大户参与度——上涨行情中若大单占比持续高,说明是机构主导的拉升"
           "(散户小单无法推动大行情),行情的级别和持续性更高。",
    dependencies=("__factors__", "mf_big_order_ratio", "momentum_20"),
)
def factor_bigorder_momentum_resonance_20(ctx: FactorContext):
    big = ctx.load_factor("mf_big_order_ratio")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(big * mom)


# ── 动量 × 筹码 ───────────────────────────────────────────────────────────

@register_factor(
    name="chip_momentum_resonance_20",
    description="筹码集中×动量：chip_cr3_factor×momentum_20。筹码集中且走强=主力控盘的趋势。",
    category="coupling",
    thesis="筹码集中度(前3大成本区占比)高+价格走强=主力吸筹完成后的拉升阶段;"
           "筹码分散+走强=跟风盘推动、随时可能抛压出逃。筹码结构确认的趋势更可靠。",
    dependencies=("__factors__", "chip_cr3_factor", "momentum_20"),
)
def factor_chip_momentum_resonance_20(ctx: FactorContext):
    chip = ctx.load_factor("chip_cr3_factor")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(chip * mom)


@register_factor(
    name="winner_momentum_combo_20",
    description="获利盘×动量：winner_rate×momentum_20。获利盘多+上涨=浮盈筹码形成支撑。",
    category="coupling",
    thesis="获利盘占比高说明多数持仓者浮盈——上涨趋势中浮盈筹码锁仓意愿强、抛压小,"
           "形成'惜售-上涨'的正反馈;叠加动量确认后趋势的自我强化特征更明确。",
    dependencies=("__factors__", "winner_rate", "momentum_20"),
)
def factor_winner_momentum_combo_20(ctx: FactorContext):
    # winner_rate 的 .fea 为 rank(-获利盘),高=低获利盘;描述要求"获利盘多+上涨排前",
    # 故翻回 (1.0 - win)。修复前 win*mom = 低获利盘×动量,与描述相反(2026-08-05)。
    win = ctx.load_factor("winner_rate")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank((1.0 - win) * mom)


# ── 动量结构 ──────────────────────────────────────────────────────────────

@register_factor(
    name="momentum_accel_20_60",
    description="动量加速度：momentum_20−momentum_60。短中期动量差=趋势加速/减速。",
    category="coupling",
    thesis="20日动量强于60日动量=趋势在加速(新资金入场推动短周期走强);"
           "20日动量弱于60日=趋势在衰减(动能枯竭的前兆)。加速度比水平更能捕捉拐点。",
    dependencies=("__factors__", "momentum_20", "momentum_60"),
)
def factor_momentum_accel_20_60(ctx: FactorContext):
    mom20 = ctx.load_factor("momentum_20")
    mom60 = ctx.load_factor("momentum_60")
    return cross_sectional_rank(mom20 - mom60)


@register_factor(
    name="size_momentum_combo_20",
    description="小盘×动量：log_circ_mv×momentum_20。小盘股的动量效应更强。",
    category="coupling",
    thesis="小盘股动量效应显著强于大盘股(交易者异质性更高、价格调整更慢、羊群效应更强)。"
           "log_circ_mv 的 .fea 高=小盘(已取反),直接作权重,优先在小盘上暴露动量。",
    dependencies=("__factors__", "log_circ_mv", "momentum_20"),
)
def factor_size_momentum_combo_20(ctx: FactorContext):
    mv = ctx.load_factor("log_circ_mv")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(mv * mom)


@register_factor(
    name="idio_vol_momentum_combo_20",
    description="低特质波动×动量：momentum_20×idio_vol_60。低特质波动的动量更稳。",
    category="coupling",
    thesis="特质波动率高的股票噪音大、动量信号被干扰,且高特质波动与低收益相关(低波动异象);"
           "在低特质波动股上保留动量暴露,风险调整后收益更高。idio_vol_60 的 .fea 高=低特质波,直接作权重。",
    dependencies=("__factors__", "idio_vol_60", "momentum_20"),
)
def factor_idio_vol_momentum_combo_20(ctx: FactorContext):
    ivol = ctx.load_factor("idio_vol_60")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(mom * ivol)


@register_factor(
    name="dv_momentum_combo_20",
    description="高股息×动量：dv_ttm_rank×momentum_20。高股息+走强=股息资金与趋势资金共振。",
    category="coupling",
    thesis="高股息股票提供下行保护(股息现金流+估值锚),叠加动量确认后攻守兼备——"
           "股息资金(险资/固收替代)与趋势资金形成双买盘,回撤更小、持有体验更好。",
    dependencies=("__factors__", "dv_ttm_rank", "momentum_20"),
)
def factor_dv_momentum_combo_20(ctx: FactorContext):
    dv = ctx.load_factor("dv_ttm_rank")
    mom = ctx.load_factor("momentum_20")
    return cross_sectional_rank(dv * mom)
