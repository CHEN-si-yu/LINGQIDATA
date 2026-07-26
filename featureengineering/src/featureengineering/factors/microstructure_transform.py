"""
Microstructure factor distribution transformations — Class 5 coupling factors.

These factors load the highest-|IC| volatility/intraday/microstructure factors
and apply distribution-normalizing transformations (log, sqrt, z-score) so the
MLP model can learn from them more effectively.  The original factors have
skewed, fat-tailed distributions that the model systematically ignores
(importance rank ~950-1200 out of 1223).

Additionally, cross-factor engineered signals (vol-of-vol, term structure,
volume-vol coupling, illiquidity premium) provide second-order microstructure
insights.

All factors declare ``"__factors__"`` in their dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    safe_divide,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (0-1)."""
    return s.groupby(level="Date").rank(pct=True)


def _zscore_cross(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score per date."""
    mu = s.groupby(level="Date").transform("mean")
    sg = s.groupby(level="Date").transform("std")
    return safe_divide(s - mu, sg + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# A — Log-Transformed Volatility / RV Factors  (right-tail compression)
# ═══════════════════════════════════════════════════════════════════════════════

_LOG_VOL_FACTORS = [
    ("rv_5min",      "5分钟已实现波动率"),
    ("rv_10min",     "10分钟已实现波动率"),
    ("rv_15min",     "15分钟已实现波动率"),
    ("rv_30min",     "30分钟已实现波动率"),
    ("rv_60min",     "60分钟已实现波动率"),
    ("gk_vol",       "Garman-Klass波动率"),
    ("parkinson_vol","Parkinson波动率"),
    ("hl_range_intraday", "日内高低价差"),
    ("ret_std_intraday",  "日内收益标准差"),
    ("amplitude_20", "20日振幅"),
    ("high_low_volatility_20", "20日高低波动率"),
    ("idiosyncratic_vol_60",   "60日特质波动率"),
    ("volatility_20",  "20日波动率"),
    ("volatility_5",   "5日波动率"),
    ("intraday_high_low_volatility", "日内高低波动率"),
    ("rv_rolling_5d_std", "5日RV波动率"),
]


def _make_log_vol_factor(base_name: str, desc_cn: str):
    """Factory: create a log-transformed volatility factor."""

    @register_factor(
        name=f"log_{base_name}",
        description=f"对数变换{desc_cn}因子，log({base_name}+ε)截面排名（低波动排前）。",
        category="risk",
        thesis=f"原始{desc_cn}因子分布极度右偏（少数高波动日极端值主导），"
               f"对数变换后消除异方差性，使MLP能稳定学习波动率信号与未来收益的非线性关系。"
               f"低波动股票在A股市场长期具有alpha溢价。",
        dependencies=("__factors__", base_name),
    )
    def _compute(ctx: FactorContext) -> pd.Series:
        raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
        log_val = np.log(raw.clip(lower=1e-10))
        return cross_sectional_rank(-log_val)  # low vol ranks higher

    _compute.__name__ = f"factor_log_{base_name}"
    _compute.__qualname__ = f"factor_log_{base_name}"
    return _compute


# Generate all log-vol factors
for _bn, _desc in _LOG_VOL_FACTORS:
    _make_log_vol_factor(_bn, _desc)


# ═══════════════════════════════════════════════════════════════════════════════
# B — Sqrt-Transformed Signed Factors  (preserve sign, compress magnitude)
# ═══════════════════════════════════════════════════════════════════════════════

_SQRT_FACTORS = [
    ("max_ret_20",             "20日最大收益"),
    ("max_ret_intraday",       "日内最大收益"),
    ("herding_intensity",      "羊群效应强度"),
    ("lottery_stock_indicator","彩票股指标"),
    ("relative_spread",        "相对价差"),
    ("am_hl_range_intraday",   "上午高低价差"),
    ("turnover_std_20",        "20日换手标准差"),
    ("turnover_vol_20",        "20日换手波动率"),
    ("ret_range_20",           "20日收益范围"),
]


def _make_sqrt_factor(base_name: str, desc_cn: str):
    """Factory: create a sqrt-transformed factor (preserves sign)."""

    @register_factor(
        name=f"sqrt_{base_name}",
        description=f"平方根变换{desc_cn}因子，sign×√(|{base_name}|)截面排名。",
        category="risk",
        thesis=f"原始{desc_cn}因子极端值被少数异常交易日主导。"
               f"平方根变换压缩极端值的影响同时保留符号方向，"
               f"使截面排名更稳定、更少受离群值扰动。",
        dependencies=("__factors__", base_name),
    )
    def _compute(ctx: FactorContext) -> pd.Series:
        raw = pd.to_numeric(ctx.load_factor(base_name), errors="coerce")
        sqrt_val = np.sign(raw) * np.sqrt(np.abs(raw))
        return cross_sectional_rank(sqrt_val)

    _compute.__name__ = f"factor_sqrt_{base_name}"
    _compute.__qualname__ = f"factor_sqrt_{base_name}"
    return _compute


for _bn, _desc in _SQRT_FACTORS:
    _make_sqrt_factor(_bn, _desc)


# ═══════════════════════════════════════════════════════════════════════════════
# C — Z-Score Transformed Factors  (time-series normalization)
# ═══════════════════════════════════════════════════════════════════════════════

_ZSCORE_FACTORS = [
    ("turnover_f_20",   "20日自由流通换手率"),
    ("turnover_20",     "20日换手率"),
    ("amihud_intraday", "日内Amihud非流动性"),
    ("realized_spread_5min", "5分钟已实现价差"),
    ("volume_rv_ratio", "成交量-RV比率"),
    ("up_volatility_20",  "20日上行波动率"),
    ("down_volatility_20","20日下行波动率"),
]


def _make_zscore_factor(base_name: str, desc_cn: str):
    """Factory: create a rolling-zscore factor per stock."""

    @register_factor(
        name=f"zscore_{base_name}",
        description=f"时序Z-score变换{desc_cn}因子，(原始值-252日均值)/252日标准差截面排名。",
        category="risk",
        thesis=f"原始{desc_cn}因子的绝对水平受股票自身特征(市值、行业、流动性)严重影响。"
               f"时序Z-score标准化到股票自身历史分布后，"
               f"捕捉的是该因子在当前时点相对于其自身历史的异常程度——"
               f"这比绝对水平更具预测价值。",
        dependencies=("__factors__", base_name),
    )
    def _compute(ctx: FactorContext) -> pd.Series:
        raw = ctx.load_factor(base_name)
        roll_mean = rolling_group_mean(raw, 252, min_periods=60)
        roll_std = rolling_group_std(raw, 252, min_periods=60)
        zscore = safe_divide(raw - roll_mean, roll_std + 1e-8)
        return cross_sectional_rank(zscore)

    _compute.__name__ = f"factor_zscore_{base_name}"
    _compute.__qualname__ = f"factor_zscore_{base_name}"
    return _compute


for _bn, _desc in _ZSCORE_FACTORS:
    _make_zscore_factor(_bn, _desc)


# ═══════════════════════════════════════════════════════════════════════════════
# D — Cross-Factor Engineered Signals  (second-order microstructure)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="vol_rv_divergence",
    description="波动率-RV背离因子，(volatility_20排名/rv_5min排名)截面排名（日间波动相对日内波动异常高排前）。",
    category="risk",
    thesis="日间波动率远高于日内RV意味着隔夜风险(跳空)大——"
           "这种背离通常发生在重大消息公布前后或流动性恶化时。"
           "背离程度大的股票隔夜跳跃风险高，排后；两者一致的股票信息消化充分，排前。",
    dependencies=("__factors__", "volatility_20", "rv_5min"),
)
def factor_vol_rv_divergence(ctx: FactorContext) -> pd.Series:
    df = ctx.load_factors(["volatility_20", "rv_5min"])
    vol_r = _rank(df["volatility_20"])
    rv_r = _rank(df["rv_5min"])
    # High vol relative to RV = overnight risk (rank low)
    divergence = safe_divide(vol_r, rv_r + 1e-8)
    return cross_sectional_rank(-divergence)


@register_factor(
    name="rv_term_structure",
    description="RV期限结构因子，rv_5min排名/rv_60min排名截面排名（短期波动相对长期波动高=波动正在集聚排后）。",
    category="risk",
    thesis="已实现波动率的期限结构反映波动率的短期vs长期动态——"
           "短期RV远高于长期RV=波动率正在急剧上升(通常是负面事件驱动)；"
           "短期RV远低于长期RV=波动率正在消退(不确定性解除)。"
           "波动率消退期的股票回报率更高。",
    dependencies=("__factors__", "rv_5min", "rv_60min"),
)
def factor_rv_term_structure(ctx: FactorContext) -> pd.Series:
    df = ctx.load_factors(["rv_5min", "rv_60min"])
    term = safe_divide(df["rv_5min"], df["rv_60min"] + 1e-10)
    return cross_sectional_rank(-term)  # low short-term RV relative to long-term = good


@register_factor(
    name="vol_of_rv",
    description="波动率的波动率因子，rv_5min的20日滚动标准差截面排名（波动率不稳定排后）。",
    category="risk",
    thesis="已实现波动率本身也有波动率——波动率不稳定=市场对股票的定价不确定性高。"
           "波动率稳定的股票信息环境清晰、定价效率高，未来收益的可预测性更强。"
           "vol-of-vol是波动率维度的二阶风险度量。",
    dependencies=("__factors__", "rv_5min"),
)
def factor_vol_of_rv(ctx: FactorContext) -> pd.Series:
    rv = ctx.load_factor("rv_5min")
    rv_std = rolling_group_std(rv, 20, min_periods=10)
    return cross_sectional_rank(-rv_std)  # stable vol ranks higher


@register_factor(
    name="turnover_rv_interaction",
    description="换手-波动耦合因子，turnover_20排名×rv_5min排名截面排名（量价共振强度排前）。",
    category="risk",
    thesis="换手率和已实现波动率的乘积捕获量价共振——"
           "高换手×高波动=市场分歧大、交易活跃但方向不明(排后)；"
           "低换手×低波动=市场共识强、价格稳定(排前)；"
           "高换手×低波动=资金在稳定吸筹(积极信号)；"
           "低换手×高波动=流动性枯竭+价格剧烈波动(危险信号)。",
    dependencies=("__factors__", "turnover_20", "rv_5min"),
)
def factor_turnover_rv_interaction(ctx: FactorContext) -> pd.Series:
    df = ctx.load_factors(["turnover_20", "rv_5min"])
    to_r = _rank(df["turnover_20"])
    rv_r = _rank(df["rv_5min"])
    # High turnover + low RV = accumulation signal (rank high)
    interaction = to_r * (1 - rv_r)
    return cross_sectional_rank(interaction)


@register_factor(
    name="amihud_parkinson_ratio",
    description="非流动性-波动率比因子，amihud_intraday排名/parkinson_vol排名截面排名（高冲击成本相对低波动的异常排后）。",
    category="risk",
    thesis="Amihud非流动性与Parkinson波动率的比率度量单位波动率的流动性成本——"
           "同样的价格波动下，流动性成本越高的股票交易执行质量越差。"
           "低比率=高效率交易、低比率=低隐性成本，alpha来自于交易效率的差异。",
    dependencies=("__factors__", "amihud_intraday", "parkinson_vol"),
)
def factor_amihud_parkinson_ratio(ctx: FactorContext) -> pd.Series:
    df = ctx.load_factors(["amihud_intraday", "parkinson_vol"])
    amihud_r = _rank(df["amihud_intraday"])
    park_r = _rank(df["parkinson_vol"])
    ratio = safe_divide(amihud_r, park_r + 1e-8)
    return cross_sectional_rank(-ratio)  # low illiquidity per unit vol = good


@register_factor(
    name="vol_asymmetry_ratio",
    description="波动非对称比因子，down_volatility_20排名/up_volatility_20排名截面排名（下跌波动大于上涨波动=风险不对称排后）。",
    category="risk",
    thesis="下行波动率与上行波动率的比率反映波动率的非对称性——"
           "下行波动>上行波动=负面消息冲击大于正面(风险溢价要求更高、排后)；"
           "上行波动>下行波动=正面消息推动力强(上涨有量、下跌有支撑、排前)。"
           "波动不对称是市场情绪在波动率维度的体现。",
    dependencies=("__factors__", "down_volatility_20", "up_volatility_20"),
)
def factor_vol_asymmetry_ratio(ctx: FactorContext) -> pd.Series:
    df = ctx.load_factors(["down_volatility_20", "up_volatility_20"])
    asymmetry = safe_divide(df["down_volatility_20"], df["up_volatility_20"] + 1e-10)
    return cross_sectional_rank(-asymmetry)  # less downside vol asymmetry = better


@register_factor(
    name="microstructure_efficiency",
    description="微观结构效率因子，(rv_5min/parkinson_vol排名)截面排名（日内效率高=信息消化快排前）。",
    category="risk",
    thesis="RV(5min)与Parkinson波动率(日高低价)的比率反映市场微观结构效率——"
           "比值接近1=日内价格发现效率高、信息被均匀消化；"
           "比值远大于1=日内波动远大于日间波动=价格发现效率低、"
           "信息在日内被过度反应后修正。高效率股票的未来收益可预测性更强。",
    dependencies=("__factors__", "rv_5min", "parkinson_vol"),
)
def factor_microstructure_efficiency(ctx: FactorContext) -> pd.Series:
    df = ctx.load_factors(["rv_5min", "parkinson_vol"])
    efficiency = safe_divide(df["rv_5min"], df["parkinson_vol"] + 1e-10)
    return cross_sectional_rank(-efficiency)  # closer to 1 = more efficient


@register_factor(
    name="intraday_overreaction_pressure",
    description="日内过度反应压力因子，(max_ret_intraday排名×herding_intensity排名)截面排名（日内过度反应+羊群=反转压力排后）。",
    category="risk",
    thesis="日内最大收益与羊群效应的共振——"
           "日内出现极端正收益且羊群效应强=追涨行为集中、短期回调压力大(排后)；"
           "日内温和收益且羊群效应弱=理性定价、趋势可持续(排前)。"
           "日内反转压力是短线交易的重要风险预警。",
    dependencies=("__factors__", "max_ret_intraday", "herding_intensity"),
)
def factor_intraday_overreaction_pressure(ctx: FactorContext) -> pd.Series:
    df = ctx.load_factors(["max_ret_intraday", "herding_intensity"])
    reversal = _rank(df["max_ret_intraday"]) * _rank(df["herding_intensity"])
    return cross_sectional_rank(-reversal)  # high reversal pressure = rank low


