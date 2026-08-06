"""
Class 2 (cyq_chips) 补充因子 — 回退注册函数。

主构建路径为 cli → build_cyq_chips_new(CHIP_FACTOR_SPEC 扇出,见 chip_deep.py),
本文件的注册函数仅作为单因子直连构建(build_many)时的回退实现,
方向与 CHIP_FACTOR_SPEC 中同名条目一一对应:
  pos           → rank(+metric)
  neg           → rank(−metric)
  momentum      → rank(5日变化)
  momentum_rev  → rank(−5日变化)
  momentum_20d  → rank(20日变化)
依赖 close 的指标(chip_below_90 / chip_upper_110)回退时加载 close 折算表,
与 chip_peak_ratio 的回退同口径(cyq_chips 成本价为复权口径)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank
from .chip_deep import _compute_chip_factor, _load_close_maps


def _chip_series(
    context: FactorContext, metric: str, need_close: bool = False
) -> pd.Series:
    """Compute one raw chip metric via the shared per-stock fallback path."""
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    if need_close:
        close_map, close_adj_map = _load_close_maps(source_root, allowed)
        return _compute_chip_factor(
            source_root, allowed, metric,
            close_map=close_map, close_adj_map=close_adj_map,
            on_progress=on_progress,
        )
    return _compute_chip_factor(source_root, allowed, metric, on_progress=on_progress)


def _momentum(s: pd.Series, window: int = 5) -> pd.Series:
    return s.groupby(level="Code").transform(lambda x: x.diff(window))


# ═══════════════════════════════════════════════════════════════════════════════
# Close-relative trap ratios
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="chip_deep_trap_ratio",
    description="深套筹码占比因子：成本价>1.1×close的筹码比例截面排名（负向，深套盘多排后）。",
    category="price",
    thesis="深套盘(成本在现价10%以上)是反弹的抛压来源——深套盘越重,上方解套卖出"
           "的意愿越强,反弹持续性越差;深套盘轻=筹码干净、拉升阻力小。"
           "与 chip_above_ratio(全部上方筹码)区分:本因子聚焦深度套牢区。",
    dependencies=("cyq_chips", "daily.parquet"),
)
def factor_chip_deep_trap_ratio(context: FactorContext):
    s = _chip_series(context, "chip_upper_110", need_close=True)
    return cross_sectional_rank(-s)


@register_factor(
    name="chip_high_float_ratio",
    description="高浮盈筹码占比因子：成本价<0.9×close的筹码比例截面排名（负向，深度获利盘多排后）。",
    category="price",
    thesis="高浮盈筹码(成本在现价10%以下)是获利兑现的来源——浮筹占比高=大量持仓者"
           "盈利丰厚、随时可能卖出锁定利润(上涨持续性存疑);浮筹少=上方无阻力"
           "或获利盘已充分换手。是获利盘因子的分布细粒度版本。",
    dependencies=("cyq_chips", "daily.parquet"),
)
def factor_chip_high_float_ratio(context: FactorContext):
    s = _chip_series(context, "chip_below_90", need_close=True)
    return cross_sectional_rank(-s)


# ═══════════════════════════════════════════════════════════════════════════════
# Distribution shape extras
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="chip_bimodality",
    description="筹码双峰因子：|主峰价−中位价|/标准差截面排名（负向，双峰分歧排后）。",
    category="price",
    thesis="双峰分布=筹码聚集在两个分离的价格带(多空两大阵营成本分离),是分歧"
           "加剧、变盘前的典型结构;单峰分布=筹码共识度高。以模式价与中位价的"
           "分离度衡量双峰性,对分布形态比对偏度(三阶矩)更直接。",
    dependencies=("cyq_chips",),
)
def factor_chip_bimodality(context: FactorContext):
    s = _chip_series(context, "chip_mode_median_gap")
    return cross_sectional_rank(-s)


@register_factor(
    name="chip_range_skew_factor",
    description="筹码分位数偏斜因子：(p90−p50)/(p50−p10)截面排名（负向，右偏上方筹码厚排后）。",
    category="price",
    thesis="分位数偏斜衡量筹码分布的不对称性:>1=上方(高价侧)筹码带更宽(套牢/浮筹"
           "堆积,上方压力大);<1=下方筹码带更宽(支撑厚)。比三阶矩偏度对离群"
           "价格更稳健,是 chip_skew_factor 的互补口径。",
    dependencies=("cyq_chips",),
)
def factor_chip_range_skew_factor(context: FactorContext):
    s = _chip_series(context, "chip_range_skew")
    return cross_sectional_rank(-s)


# ═══════════════════════════════════════════════════════════════════════════════
# Cost-centre momentum
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="chip_median_momentum",
    description="筹码中位成本动量因子：中位成本价5日变化截面排名（成本上移排前）。",
    category="price",
    thesis="中位成本价上移=新进筹码成本抬升(资金在更高位接筹,趋势获确认);"
           "下移=成本重心下降(接盘乏力)。与价格动量正交:价格可因缩量上涨而"
           "动量高,但筹码成本不动=虚涨。成本动量为动量的筹码结构确认。",
    dependencies=("cyq_chips",),
)
def factor_chip_median_momentum(context: FactorContext):
    s = _chip_series(context, "chip_median_price")
    return cross_sectional_rank(_momentum(s))


@register_factor(
    name="chip_weighted_mean_momentum",
    description="筹码平均成本动量因子：加权平均成本价5日变化截面排名（成本上移排前）。",
    category="price",
    thesis="平均成本比中位成本对高价筹码更敏感(右偏分布下更高)——平均成本上移="
           "大量资金在高位换手、整体持仓成本抬升,是趋势推进的筹码证据;"
           "与 chip_median_momentum 互补(均值 vs 中位数)。",
    dependencies=("cyq_chips",),
)
def factor_chip_weighted_mean_momentum(context: FactorContext):
    s = _chip_series(context, "chip_weighted_mean")
    return cross_sectional_rank(_momentum(s))


# ═══════════════════════════════════════════════════════════════════════════════
# Dispersion / risk momentum
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="chip_iqr_momentum_20d",
    description="筹码四分位距20日变化因子：IQR的20日变化截面排名（收窄排前）。",
    category="price",
    thesis="四分位距的20日变化捕捉中期筹码收敛/发散——IQR 持续收窄=筹码向中枢"
           "凝聚(主力吸筹特征),发散=筹码松动(派发特征)。比5日变化更平滑,"
           "适合中期筹码结构判断。",
    dependencies=("cyq_chips",),
)
def factor_chip_iqr_momentum_20d(context: FactorContext):
    s = _chip_series(context, "chip_iqr")
    return cross_sectional_rank(_momentum(s, window=20))


@register_factor(
    name="chip_semi_std_momentum",
    description="筹码下行半方差动量因子：下行半标准差的5日变化截面排名（负向，下行离散扩大排后）。",
    category="price",
    thesis="下行半方差(均值下方筹码的离散度)扩大=低位筹码带正在拉宽,承接力量"
           "分散、下方支撑弱化(风险信号);收窄=下方筹码集中、支撑增强。"
           "与 chip_downside_risk(水平)互补,捕捉其方向。",
    dependencies=("cyq_chips",),
)
def factor_chip_semi_std_momentum(context: FactorContext):
    s = _chip_series(context, "chip_semi_std")
    return cross_sectional_rank(-_momentum(s))


@register_factor(
    name="chip_width_ratio_momentum",
    description="筹码相对宽度动量因子：宽度比(std/close)的5日变化截面排名（负向，变宽排后）。",
    category="price",
    thesis="筹码相对宽度(std/close)收窄=分布向现价收敛(价格与筹码结构同步凝聚,"
           "突破能量积蓄);变宽=分布发散(换手分散)。宽度比已对价格水平归一,"
           "其变化衡量筹码结构相对价格的动态松紧。",
    dependencies=("cyq_chips",),
)
def factor_chip_width_ratio_momentum(context: FactorContext):
    s = _chip_series(context, "chip_width_ratio")
    return cross_sectional_rank(-_momentum(s))


@register_factor(
    name="chip_percentile_20d",
    description="筹码价格分位20日变化因子：当前价在筹码分布中的分位20日变化截面排名（分位抬升排前）。",
    category="price",
    thesis="当前价格在筹码分布中的分位=「筹码被套比例」的连续版本——分位20日"
           "抬升=价格正穿越筹码密集区上行(套牢盘被解放,趋势获得筹码确认);"
           "分位下降=价格跌破成本中枢(抛压涌现)。比分位水平更具动态信息。",
    dependencies=("cyq_chips", "daily.parquet"),
)
def factor_chip_percentile_20d(context: FactorContext):
    s = _chip_series(context, "chip_percentile", need_close=True)
    return cross_sectional_rank(_momentum(s, window=20))
