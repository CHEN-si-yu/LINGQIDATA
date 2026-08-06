"""
Risk metrics factors — Class 1 (daily.parquet only).

风险度量族:VaR / Sortino / 回撤结构 / 波动状态等风险管理指标。
- 收益一律用 pct_chg(复权口径),回撤/新高用 _adjusted_close 后复权基座;
- 风险类因子(除 Sortino/修复率外)统一负向排名:风险越大排名越低;
- 连续段计数用组内分组技巧,rolling corr 用宽表。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, stack_date_code
from .momentum_rebuilt import _adjusted_close


def _ret(daily: pd.DataFrame) -> pd.Series:
    """复权日收益序列 (Date, Code)。"""
    return daily["pct_chg"] / 100.0


def _consecutive_count(cond: pd.Series) -> pd.Series:
    """每股连续满足条件的天数(不满足则归零)。"""
    cond_i = cond.astype(int)
    code = cond_i.index.get_level_values("Code")
    seg = (~cond_i.astype(bool)).groupby(level="Code").cumsum()
    return cond_i.groupby([code, seg]).cumsum()


# ═══════════════════════════════════════════════════════════════════════════════
# 下行风险
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="downside_frequency_60",
    description="下行频率因子：60日负收益天数占比截面排名（负向，频繁下跌排后）。",
    category="risk",
    thesis="60日负收益天数占比衡量「钝刀子割肉」式的持续阴跌风险——单日大跌可由事件解释，"
           "但高频负收益=股票缺乏上行弹性。与波动率正交：低波动+高频下行=阴跌股，风险高。",
    dependencies=("daily.parquet",),
)
def factor_downside_frequency_60(context: FactorContext):
    daily = context.load("daily.parquet")
    neg = _ret(daily).lt(0)
    freq = neg.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    return cross_sectional_rank(-freq)


@register_factor(
    name="var_95_20",
    description="20日VaR(95%)因子：收益5%分位数截面排名（负向，尾部损失大排后）。",
    category="risk",
    thesis="VaR(95%, 20日)衡量20日窗口内的最坏单日损失——5%分位数越负=尾部风险越大。"
           "是比标准差更贴合损失视角的风险度量(研报《A股市场特征研究》尾部相关性语境)。",
    dependencies=("daily.parquet",),
)
def factor_var_95_20(context: FactorContext):
    daily = context.load("daily.parquet")
    var = _ret(daily).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).quantile(0.05)
    )
    return cross_sectional_rank(var)


@register_factor(
    name="loss_probability_20",
    description="损失概率因子：20日负收益频率截面排名（负向，高损失概率排后）。",
    category="risk",
    thesis="20日损失概率衡量短期亏钱体验——概率高=高频小幅下跌(负期望暴露)。"
           "与 downside_frequency_60 的区别在于窗口更短、更贴近近期状态。",
    dependencies=("daily.parquet",),
)
def factor_loss_probability_20(context: FactorContext):
    daily = context.load("daily.parquet")
    neg = _ret(daily).lt(0)
    freq = neg.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-freq)


@register_factor(
    name="sortino_ratio_60",
    description="60日Sortino比率因子：均收益/下行标准差截面排名（正向，风险调整收益高排前）。",
    category="risk",
    thesis="Sortino比率只惩罚下行波动(优于Sharpe)——同收益下下行波动小的股票"
           "持有体验更好、回撤更浅。60日均收益/60日下行标准差，正向排名。",
    dependencies=("daily.parquet",),
)
def factor_sortino_ratio_60(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = _ret(daily)
    mean60 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    down = ret.where(ret < 0)
    # 下行样本稀疏:60日窗口内 min_periods=5 个负收益日即可估计
    down_std = down.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=5).std()
    )
    sortino = safe_divide(mean60, down_std + 1e-10)
    return cross_sectional_rank(sortino)


# ═══════════════════════════════════════════════════════════════════════════════
# 波动率结构
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="vol_regime_switch_20",
    description="波动状态切换因子：20日波动/60日波动截面排名（负向，波动骤升排后）。",
    category="risk",
    thesis="20日波动相对60日波动的比值>1=近期波动放大(状态切换至高风险区)，"
           "<1=波动收敛(蓄势)。波动放大常伴随方向不明的剧烈博弈，负向排名。",
    dependencies=("daily.parquet",),
)
def factor_vol_regime_switch_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = _ret(daily)
    vol20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    vol60 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    ratio = safe_divide(vol20, vol60 + 1e-10)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="vol_decay_ratio_20",
    description="波动衰减因子：20日波动/10日波动截面排名（负向，波动持续放大排后）。",
    category="risk",
    thesis="20日波动相对10日波动的比值衡量波动的时间结构——>1=近期波动仍处高位(波动未衰减)，"
           "<1=波动正在收敛(风险释放)。与 vol_regime_switch_20 互补(更长基期)。",
    dependencies=("daily.parquet",),
)
def factor_vol_decay_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = _ret(daily)
    vol20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    vol10 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).std()
    )
    ratio = safe_divide(vol20, vol10 + 1e-10)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="vol_clustering_20",
    description="波动聚集因子：|收益|的20日自相关截面排名（负向，波动持续聚集排后）。",
    category="risk",
    thesis="波动聚集(GARCH特征)：|收益|自相关高=大波动日扎堆出现(风险持续时间长)，"
           "自相关低/负=波动独立分散(风险快速释放)。用宽表 rolling corr 向量化。",
    dependencies=("daily.parquet",),
)
def factor_vol_clustering_20(context: FactorContext):
    daily = context.load("daily.parquet")
    abs_ret = _ret(daily).abs().unstack("Code")
    lag = abs_ret.shift(1)
    ac = abs_ret.rolling(20, min_periods=10).corr(lag)
    return cross_sectional_rank(-stack_date_code(ac))


@register_factor(
    name="tail_risk_pct_60",
    description="尾风险频率因子：60日内|z|>2极端收益占比截面排名（负向，极端波动频发排后）。",
    category="risk",
    thesis="60日窗口内超过2倍标准差的极端收益频率=厚尾风险(研报《A股市场特征研究》尾部相关性)。"
           "极端波动频发=信息冲击不稳定、定价效率低，负向排名。",
    dependencies=("daily.parquet",),
)
def factor_tail_risk_pct_60(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = _ret(daily)
    mean60 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    std60 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    z = safe_divide(ret - mean60, std60 + 1e-10)
    freq = z.abs().gt(2).groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    return cross_sectional_rank(-freq)


@register_factor(
    name="downside_upside_vol_60",
    description="下行/上行波动比因子：60日负收益波动/正收益波动截面排名（负向，下行风险主导排后）。",
    category="risk",
    thesis="下行波动大于上行波动=下跌比上涨更剧烈(空头主导、接盘意愿弱)；"
           "比值<1=上涨比下跌更有力度。衡量收益分布的非对称风险，与偏度互补。"
           "(命名加60与已删除的 downside_vol_ratio_20 区分)",
    dependencies=("daily.parquet",),
)
def factor_downside_upside_vol_60(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = _ret(daily)
    down = ret.where(ret < 0)
    up = ret.where(ret > 0)
    # 正/负样本各自稀疏:60日窗口内 min_periods=5 即可估计
    down_std = down.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=5).std()
    )
    up_std = up.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=5).std()
    )
    ratio = safe_divide(down_std, up_std + 1e-10)
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# 回撤结构(复权基座)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="drawdown_duration_120",
    description="回撤持续期因子：当前回撤(复权价低于前高)持续天数截面排名（负向，深陷回撤排后）。",
    category="risk",
    thesis="回撤持续期=从前期高点回落至今的天数——持续期长=趋势破坏久、套牢盘积压重。"
           "基于复权基座的 cummax 判定前高，除权日无假回撤。负向排名。",
    dependencies=("daily.parquet",),
)
def factor_drawdown_duration_120(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    cummax = adj.groupby(level="Code").cummax()
    in_dd = adj.lt(cummax * 0.999)
    # 回撤段:从回到前高起计数,回撤中每天+1
    duration = _consecutive_count(in_dd)
    return cross_sectional_rank(-duration)


@register_factor(
    name="drawdown_recovery_60",
    description="回撤修复率因子：复权价/60日新高截面排名（正向，接近新高排前）。",
    category="risk",
    thesis="当前价相对60日新高的距离=回撤修复程度——接近新高=趋势修复完成(强势)，"
           "远离新高=回撤未修复(弱势)。与 drawdown_duration 互补(深度 vs 时间)。"
           "基于复权基座，除权日无假修复。",
    dependencies=("daily.parquet",),
)
def factor_drawdown_recovery_60(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    high60 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    )
    recovery = safe_divide(adj, high60 + 1e-10)
    return cross_sectional_rank(recovery)


@register_factor(
    name="max_consecutive_loss_20",
    description="最长连亏因子：20日内最长连续亏损天数截面排名（负向，长连亏排后）。",
    category="risk",
    thesis="20日内最长连续亏损天数衡量下跌的「韧性」——连亏7天比断续亏7天伤害更大"
           "(持仓者更容易在底部割肉)。连续段计数后取20日窗口最大值。",
    dependencies=("daily.parquet",),
)
def factor_max_consecutive_loss_20(context: FactorContext):
    daily = context.load("daily.parquet")
    neg = _ret(daily).lt(0)
    consec = _consecutive_count(neg)
    max_loss = consec.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).max()
    )
    return cross_sectional_rank(-max_loss)


@register_factor(
    name="range_vol_ratio_20",
    description="区间波动比因子：20日高低价区间/20日收益波动截面排名（负向，极端区间排后）。",
    category="risk",
    thesis="20日(高-低)区间相对收益波动的比值——比值高=价格在区间内大幅摆动"
           "(游资博弈、方向不明)，比值低=单边趋势(方向确定)。"
           "区间用复权折算后的high/low(除权日不失真)。",
    dependencies=("daily.parquet",),
)
def factor_range_vol_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    scale = _adjusted_close(daily) / close.replace(0, np.nan)
    adj = _adjusted_close(daily)
    adj_high = daily["high"] * scale
    adj_low = daily["low"] * scale
    hi20 = adj_high.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    lo20 = adj_low.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).min()
    )
    vol20 = _ret(daily).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    # 分母用复权基座 adj 与分子(复权空间区间)同口径——未复权 close 在除权日
    # 跳变会瞬时改变分母标尺(2026-08-05 修复)
    ratio = safe_divide(hi20 - lo20, vol20 * adj + 1e-10)
    return cross_sectional_rank(-ratio)
