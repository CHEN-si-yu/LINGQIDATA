"""
风险结构补充因子 — Class 1 (daily.parquet only)。

在 risk_metrics.py / higher_moments.py 基础上补充尾部条件风险、量价条件风险、
波动周期位置、涨跌不对称与市场状态敏感度。全部为日频口径、宽表向量化,
与已有因子构成二阶/条件化区分:
- var_95_20(分位数) → cvar_95_120(条件均值,120日估计窗,尾部更深);
- tail_risk_pct_60(|z|>2 双向) → extreme_gain_freq_20(上行单侧 1.5σ);
- fear_index_20(下行极端频率) → panic_selling_ratio_60(放量×下跌联合条件);
- vol_regime_switch_20(20/60 均值比) → vol_cycle_position_120(相对自身 120 日最低);
- downside_upside_vol_60(二阶矩比) → gain_loss_asymmetry_60(一阶矩比);
- beta_60(无条件) → market_regime_sensitivity_60(市场涨/跌状态条件化)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, stack_date_code
from .market_relative import _market_proxy, _ret_wide


@register_factor(
    name="cvar_95_120",
    description="120日CVaR因子：低于5%分位收益日的均值（条件尾部损失），排名高=尾部风险小。",
    category="risk",
    thesis="CVaR(条件VaR)取最坏5%日收益的均值,比 var_95_20 的单点分位数更稳健地"
           "度量尾部损失的期望深度——两只股票 VaR 相同但 CVaR 更负者尾部更肥"
           "(厚尾风险,研报《A股市场特征研究》尾部相关性语境)。120日窗口保证"
           "尾部样本充足(期望6个极端日),估计稳定、覆盖率高;排名方向与 var_95_20 一致。",
    dependencies=("daily.parquet",),
)
def factor_cvar_95_120(context: FactorContext):
    daily = context.load("daily.parquet")
    ret_w = _ret_wide(daily)
    q05 = ret_w.rolling(120, min_periods=60).quantile(0.05)
    tail = ret_w.where(ret_w <= q05)
    cvar_w = tail.rolling(120, min_periods=3).mean()
    return cross_sectional_rank(stack_date_code(cvar_w))


@register_factor(
    name="panic_selling_ratio_60",
    description="放量下跌占比因子：60日放量(vol>1.5×20日均量)且下跌天数/放量天数（恐慌抛售排前）。",
    category="risk",
    thesis="放量下跌是恐慌性抛售的直接证据(量大且方向向下)——放量日下跌占比高="
           "筹码在下跌中充分换手、情绪集中释放,超跌反转候选(与 fear_index_20 同逻辑);"
           "放量日多为上涨=资金进场推动。60日估计窗保证低放量频率股票也有足够"
           "样本;联合量价条件,区分于纯量(volume_spike_event)与纯价(downside_frequency_60)因子。",
    dependencies=("daily.parquet",),
)
def factor_panic_selling_ratio_60(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = daily["pct_chg"] / 100.0
    vol = daily["vol"]
    ma20 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    big = vol.gt(1.5 * ma20)
    down = ret.lt(0)
    panic = (big & down).astype(float)
    # min_periods=1:只要有 ≥1 个放量日即可估计占比
    n_big = big.astype(float).groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=1).sum()
    )
    n_panic = panic.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=1).sum()
    )
    ratio = safe_divide(n_panic, n_big)
    return cross_sectional_rank(ratio)


@register_factor(
    name="vol_cycle_position_120",
    description="波动周期位置因子：20日波动/120日内最低20日波动截面排名（负向，波动扩张排后）。",
    category="risk",
    thesis="当前20日波动相对自身120日滚动最低水平的比值=波动周期中的绝对位置——"
           "接近1=波动压缩到极致(变盘前兆,方向未知但爆发在即);远高于1=波动扩张"
           "进行中(风险释放未完成)。与 vol_regime_switch_20(相对均值)互补,"
           "本因子以自身极值为基准,对波动率的「地量地价」更敏感。",
    dependencies=("daily.parquet",),
)
def factor_vol_cycle_position_120(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = daily["pct_chg"] / 100.0
    vol20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    vol20_min120 = vol20.groupby(level="Code").transform(
        lambda s: s.rolling(120, min_periods=30).min()
    )
    pos = safe_divide(vol20, vol20_min120 + 1e-10)
    return cross_sectional_rank(-pos)


@register_factor(
    name="gain_loss_asymmetry_60",
    description="涨跌幅度不对称因子：60日平均涨幅/|平均跌幅|截面排名（涨多跌少排前）。",
    category="risk",
    thesis="平均涨幅与平均跌幅的比值捕捉收益分布的第一阶不对称——涨多跌少=多头"
           "占据主动(买盘强于卖盘,趋势质量好);涨少跌多=空头主导。与"
           "downside_upside_vol_60(二阶矩比)互补:一只股票可以波动对称但幅度不对称。",
    dependencies=("daily.parquet",),
)
def factor_gain_loss_asymmetry_60(context: FactorContext):
    daily = context.load("daily.parquet")
    ret_w = _ret_wide(daily)
    mean_up = ret_w.where(ret_w > 0).rolling(60, min_periods=10).mean()
    mean_down = ret_w.where(ret_w < 0).rolling(60, min_periods=10).mean()
    asym = safe_divide(mean_up, mean_down.abs() + 1e-10)
    return cross_sectional_rank(stack_date_code(asym))


@register_factor(
    name="market_regime_sensitivity_60",
    description="市场状态敏感度因子：60日下跌市均收益/|上涨市均收益|（下跌市抗跌排前）。",
    category="risk",
    thesis="把收益按市场涨/跌状态条件化:下跌市中的均收益相对上涨市中的均收益之比,"
           "衡量股票对市场状态的对称性——下跌市跌得比上涨市涨得多=高beta且不对称"
           "(危机敏感股);下跌市抗跌=防御属性(低系统性下行暴露)。是无条件 beta_60 "
           "的状态条件化版本,对尾部市况的刻画更直接。",
    dependencies=("daily.parquet",),
)
def factor_market_regime_sensitivity_60(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    mkt = _market_proxy(wide)
    stock_on_down = wide.where(mkt < 0).rolling(60, min_periods=10).mean()
    stock_on_up = wide.where(mkt > 0).rolling(60, min_periods=10).mean()
    sens = safe_divide(stock_on_down, stock_on_up.abs() + 1e-10)
    return cross_sectional_rank(stack_date_code(sens))


@register_factor(
    name="extreme_gain_freq_20",
    description="上行极端收益频率因子：20日收益高于均值+1.5σ的天数占比截面排名（负向，彩票偏好排后）。",
    category="risk",
    thesis="彩票偏好异象的上行侧刻画:上行极端收益频发=收益分布呈彩票形态,"
           "散户高估其概率而推高价格,未来收益系统性偏低。与 fear_index_20(下行极端,"
           "正向反转逻辑)镜像,本因子取负向排名——上行极端频繁排后。",
    dependencies=("daily.parquet",),
)
def factor_extreme_gain_freq_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = daily["pct_chg"] / 100.0
    mean20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    std20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    extreme_up = ret.gt(mean20 + 1.5 * std20)
    freq = extreme_up.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-freq)
