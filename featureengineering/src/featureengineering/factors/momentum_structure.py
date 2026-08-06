"""
动量结构与量价序列结构因子 — Class 1 (daily.parquet only)。

二阶结构族:把动量/流动性/收益序列的一阶量(水平)升级为二阶量(加速度、
自相关、不对称、跨期结构),捕捉一阶因子不敏感的状态切换与序列依赖。

关键原则:
- 连续价格一律走 _adjusted_close 复权基座,除权日无跳变;
- 滚动相关一律 unstack 宽表向量化(rolling corr 在 groupby transform 中极慢);
- 收益率一律 pct_chg/100(数据商复权口径);
- Amihud 类因子:illiq = |ret|/amount,与 amihud_intraday 同族但为日频口径。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, stack_date_code
from .momentum_rebuilt import _adjusted_close


def _ret(daily: pd.DataFrame) -> pd.Series:
    return daily["pct_chg"] / 100.0


def _ret_wide(daily: pd.DataFrame) -> pd.DataFrame:
    return _ret(daily).unstack("Code")


def _amihud(daily: pd.DataFrame) -> pd.Series:
    """日频 Amihud 非流动性: |ret| / 成交额(元)。"""
    return safe_divide(_ret(daily).abs(), daily["amount"])


def _roll_sum(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.groupby(level="Code").transform(
        lambda x: x.rolling(window, min_periods=min_periods).sum()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 动量结构
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="price_distance_from_52w_low",
    description="距52周低点距离因子：(adj−252日最低)/252日最低截面排名（远离年内低点排前）。",
    category="timeseries",
    thesis="George-Hwang 52周效应在低点侧:远离一年低点=趋势处于健康区间,"
           "贴着52周低点运行=持续弱势阴跌(研报《上市公司动量反转》:极值附近的反转"
           "与动量并存)。基于复权基座,除权日不产生假新低(与 price_to_52w_high 同族)。",
    dependencies=("daily.parquet",),
)
def factor_price_distance_from_52w_low(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    low_252 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(252, min_periods=120).min()
    )
    dist = safe_divide(adj - low_252, low_252)
    return cross_sectional_rank(dist)


@register_factor(
    name="momentum_accel_60_120",
    description="60/120日动量加速度因子：(60日动量−120日动量)截面排名（中期趋势加速排前）。",
    category="timeseries",
    thesis="60日动量相对120日动量的差衡量季度尺度的趋势加速度——60日动量强于120日="
           "近三个月正在加速上行,弱于120日=中期动能衰竭。与 momentum_stability_20_60"
           "(月尺度)互补,构成动量的二阶结构:水平(60)+加速度(60-120)。",
    dependencies=("daily.parquet",),
)
def factor_momentum_accel_60_120(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    g = adj.groupby(level="Code")
    mom60 = g.transform(lambda s: s.pct_change(60, fill_method=None))
    mom120 = g.transform(lambda s: s.pct_change(120, fill_method=None))
    accel = mom60 - mom120
    return cross_sectional_rank(accel)


@register_factor(
    name="rebound_from_low_20",
    description="20日低点反弹幅度因子：当前价相对20日低点的涨幅截面排名（强劲反弹排前）。",
    category="timeseries",
    thesis="当前价相对20日低点的反弹幅度=下跌后的修复弹性——反弹幅度大=买盘承接有力,"
           "趋势由弱转强;反弹幅度小=趴在低点(阴跌未止)。与 drawdown_60(相对峰值回撤)"
           "从两端刻画同一趋势:回撤管顶部、反弹管底部。复权基座,无除权假反弹。",
    dependencies=("daily.parquet",),
)
def factor_rebound_from_low_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    low_20 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).min()
    )
    rebound = safe_divide(adj - low_20, low_20)
    return cross_sectional_rank(rebound)


# ═══════════════════════════════════════════════════════════════════════════════
# 量能/收益序列结构(自相关与领先滞后)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="volume_autocorr_5",
    description="成交量5日自相关因子：vol与昨日vol的5日滚动相关截面排名（量能节奏规律排前）。",
    category="price",
    thesis="成交量的一阶自相关衡量放缩量的节奏规律性——高自相关=量能变化有持续"
           "惯性(持续放量或持续缩量,资金行为一致);低/负自相关=量能忽大忽小"
           "(消息驱动的离散博弈)。5日窗口捕捉短节奏。",
    dependencies=("daily.parquet",),
)
def factor_volume_autocorr_5(context: FactorContext):
    daily = context.load("daily.parquet")
    vol_w = daily["vol"].unstack("Code")
    ac = vol_w.rolling(5, min_periods=3).corr(vol_w.shift(1))
    return cross_sectional_rank(stack_date_code(ac))


@register_factor(
    name="volume_autocorr_20",
    description="成交量20日自相关因子：vol与昨日vol的20日滚动相关截面排名（量能惯性排前）。",
    category="price",
    thesis="20日窗口的量能自相关衡量中期资金行为一致性——高=机构建仓/出货的持续"
           "性放量结构;低=散户化的随机博弈。与5日版本互补:5日看短节奏、20日看"
           "中期行为模式。",
    dependencies=("daily.parquet",),
)
def factor_volume_autocorr_20(context: FactorContext):
    daily = context.load("daily.parquet")
    vol_w = daily["vol"].unstack("Code")
    ac = vol_w.rolling(20, min_periods=10).corr(vol_w.shift(1))
    return cross_sectional_rank(stack_date_code(ac))


@register_factor(
    name="ret_vol_lead_corr_20",
    description="量领先价相关性因子：昨日vol与今日收益的20日滚动相关截面排名（量能前瞻有效排前）。",
    category="price",
    thesis="量领先价相关(corr(vol_t−1, ret_t))衡量成交量的前瞻信息含量——高=昨日放量"
           "的股票次日倾向上涨(量是价格的先行指标,资金提前布局);低/负=放量次日"
           "回落(放量出货)。是「量在价先」逻辑的统计检验,与同期相关(volume_price_corr_20)"
           "互为先后手。",
    dependencies=("daily.parquet",),
)
def factor_ret_vol_lead_corr_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret_w = _ret_wide(daily)
    lag_vol_w = daily["vol"].unstack("Code").shift(1)
    corr = ret_w.rolling(20, min_periods=10).corr(lag_vol_w)
    return cross_sectional_rank(stack_date_code(corr))


@register_factor(
    name="ret_autocorr_1d_20",
    description="日收益一阶自相关因子：ret与昨日ret的20日滚动相关截面排名（趋势性排前）。",
    category="price",
    thesis="日收益的一阶自相关直接度量「涨后跟涨/跌后跟跌」的延续性——正自相关="
           "趋势性股票(动量内部结构稳固);负自相关=均值回归股票(高低切频繁)。"
           "与成交量自相关正交(收益维度),与分钟级 ret_autocorr_5min 区分(日频口径)。",
    dependencies=("daily.parquet",),
)
def factor_ret_autocorr_1d_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret_w = _ret_wide(daily)
    ac = ret_w.rolling(20, min_periods=10).corr(ret_w.shift(1))
    return cross_sectional_rank(stack_date_code(ac))


# ═══════════════════════════════════════════════════════════════════════════════
# 流动性结构(日频 Amihud 二阶)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="amihud_trend_20_60",
    description="Amihud流动性趋势因子：20日Amihud/60日Amihud截面排名（负向，流动性恶化排后）。",
    category="risk",
    thesis="近期非流动性相对中期水平抬升=流动性正在恶化(接盘变少、冲击成本上升)——"
           "流动性恶化的股票在下跌市中更易踩踏,是隐性的下行风险放大因子。"
           "与 amihud_daily_20(水平)互补:本因子捕捉趋势方向而非绝对水平。",
    dependencies=("daily.parquet",),
)
def factor_amihud_trend_20_60(context: FactorContext):
    daily = context.load("daily.parquet")
    illiq_w = _amihud(daily).unstack("Code")
    a20 = illiq_w.rolling(20, min_periods=10).mean()
    a60 = illiq_w.rolling(60, min_periods=30).mean()
    ratio = safe_divide(a20, a60 + 1e-12)
    return cross_sectional_rank(-stack_date_code(ratio))


@register_factor(
    name="amihud_asymmetry_20",
    description="涨跌日流动性不对称因子：下跌日Amihud/上涨日Amihud截面排名（负向，恐慌性难出货排后）。",
    category="risk",
    thesis="下跌日与上涨日冲击成本的差异揭示抛售的性质——下跌日流动性显著更差="
           "恐慌性单边出货(无人接盘),上涨日更差=缩量阴跌(观望情绪)。"
           "不对称度高=流动性风险在坏消息时集中爆发,负向排名。",
    dependencies=("daily.parquet",),
)
def factor_amihud_asymmetry_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret_w = _ret_wide(daily)
    illiq_w = _amihud(daily).unstack("Code")
    up_illiq = illiq_w.where(ret_w > 0).rolling(20, min_periods=5).mean()
    down_illiq = illiq_w.where(ret_w < 0).rolling(20, min_periods=5).mean()
    asym = safe_divide(down_illiq, up_illiq + 1e-12)
    return cross_sectional_rank(-stack_date_code(asym))


# ═══════════════════════════════════════════════════════════════════════════════
# 隔夜/日内收益结构
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="overnight_return_share_20",
    description="隔夜收益占比因子：20日|隔夜跳空|占(跳空+日内)总波动的比例截面排名（隔夜驱动排前）。",
    category="price",
    thesis="每日收益可分解为隔夜(open/pre_close−1)与日内(close/open−1)两部分——"
           "隔夜占比高=股票由隔夜信息驱动(公告/海外/政策敏感,跳空定价);"
           "日内占比高=盘中博弈驱动(题材/情绪)。隔夜驱动型股票的信息传导更快,"
           "与 am_pm_return_ratio(分钟级)区分:本因子为日频分解口径。",
    dependencies=("daily.parquet",),
)
def factor_overnight_return_share_20(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
    intra = safe_divide(daily["close"], daily["open"]) - 1.0
    abs_gap = gap.abs()
    abs_intra = intra.abs()
    num = _roll_sum(abs_gap, 20, 10)
    den = _roll_sum(abs_gap + abs_intra, 20, 10)
    share = safe_divide(num, den)
    return cross_sectional_rank(share)
