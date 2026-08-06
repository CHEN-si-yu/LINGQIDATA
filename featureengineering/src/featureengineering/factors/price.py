from __future__ import annotations

import numpy as np

import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide
from ..utils import rolling_group_max, rolling_group_min, rolling_group_mean, rolling_group_std
from .momentum_rebuilt import _adjusted_close


# ── Amplitude / Range ───────────────────────────────────────────────────

@register_factor(
    name="amplitude_20",
    description="20日均振幅因子，(high-low)/close 的20日均值截面排名（低振幅排前）。",
    category="price",
    thesis="振幅是流动性与不确定性的综合指标，低振幅反映筹码稳定性。",
    dependencies=("daily.parquet",),
)
def factor_amplitude_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    amp = (daily_panel["high"] - daily_panel["low"]) / daily_panel["close"].replace(0, np.nan)
    avg_amp = amp.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_amp)


# ── MA Bias ─────────────────────────────────────────────────────────────

@register_factor(
    name="bias_20",
    description="20日均线乖离率，close/ma_20 - 1 的截面排名。",
    category="price",
    thesis="均线乖离反映价格对中期成本的偏离程度，极端乖离预示均值回归。",
    dependencies=("daily.parquet",),
)
def factor_bias_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    # 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
    adj = _adjusted_close(daily_panel)
    ma_20 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    bias = adj / ma_20.replace(0, np.nan) - 1.0
    return cross_sectional_rank(bias)


# ── Volume ──────────────────────────────────────────────────────────────

@register_factor(
    name="volume_ratio_20",
    description="20日相对成交量因子，vol/avg_vol_20 - 1 截面排名。",
    category="price",
    thesis="放量上涨和缩量下跌都是技术面确认信号，成交量异常值得关注。",
    dependencies=("daily.parquet",),
)
def factor_volume_ratio_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    vol = daily_panel["vol"]
    avg_vol = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    v_ratio = vol / avg_vol.replace(0, np.nan) - 1.0
    return cross_sectional_rank(-v_ratio)


@register_factor(
    name="amount_ratio_20",
    description="20日相对成交额因子，amount/avg_amount_20 - 1 截面排名。",
    category="price",
    thesis="成交额比成交量更能反映资金参与度，异常放量常伴随趋势转折。",
    dependencies=("daily.parquet",),
)
def factor_amount_ratio_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    amount = daily_panel["amount"]
    avg_amount = amount.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    a_ratio = amount / avg_amount.replace(0, np.nan) - 1.0
    return cross_sectional_rank(-a_ratio)


# ── Shadow / Gap (daily.parquet — non-adjusted for real price geometry) ─

@register_factor(
    name="shadow_upper_20",
    description="20日均上影线比例，上影线/(high-low) 截面排名。",
    category="price",
    thesis="上影线反映高位抛压，长期高上影线比例是上涨阻力信号。",
    dependencies=("daily.parquet",),
)
def factor_shadow_upper_20(context: FactorContext):
    daily = context.load("daily.parquet")
    upper_shadow = daily["high"] - daily[["open", "close"]].max(axis=1)
    body_range = daily["high"] - daily["low"]
    ratio = upper_shadow / body_range.replace(0, np.nan)
    avg_ratio = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_ratio)


@register_factor(
    name="shadow_lower_20",
    description="20日均下影线比例，下影线/(high-low) 截面排名（高值=强支撑）。",
    category="price",
    thesis="下影线反映低位承接力，高下影线比例是底部支撑信号。",
    dependencies=("daily.parquet",),
)
def factor_shadow_lower_20(context: FactorContext):
    daily = context.load("daily.parquet")
    lower_shadow = daily[["open", "close"]].min(axis=1) - daily["low"]
    body_range = daily["high"] - daily["low"]
    ratio = lower_shadow / body_range.replace(0, np.nan)
    avg_ratio = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(avg_ratio)


@register_factor(
    name="gap_ratio_20",
    description="20日均跳空比率因子，open/pre_close - 1 截面排名。",
    category="price",
    thesis="向上跳空缺口反映隔夜利好信息，跳空后短期存在反转压力。",
    dependencies=("daily.parquet",),
)
def factor_gap_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = daily["open"] / daily["pre_close"].replace(0, np.nan) - 1.0
    avg_gap = gap.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_gap)


@register_factor(
    name="gap_fill_5d_reversal",
    description="缺口回补反转因子，5日内出现向下跳空后的回补倾向截面排名。",
    category="price",
    thesis="A股'缺口必补'的民间规律有一定统计基础——向下跳空缺口在短期内面临均值回归压力。因子计算：(open-pre_close)/pre_close在5日内的min，取负向（即缺口越深=回补概率越大=正向预期）。",
    dependencies=("daily.parquet",),
)
def factor_gap_fill_5d_reversal(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = daily["open"] / daily["pre_close"].replace(0, np.nan) - 1.0
    # 5-day rolling minimum gap (most negative gap in recent 5 days)
    min_gap_5d = gap.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).min()
    )
    return cross_sectional_rank(-min_gap_5d)


# ── Price position ──────────────────────────────────────────────────────

@register_factor(
    name="price_position_60",
    description="60日价格位置，(close-60d_low)/(60d_high-60d_low) 截面排名。",
    category="price",
    thesis="价格在近60日区间内的相对位置反映短期趋势强度。",
    dependencies=("daily.parquet",),
)
def factor_price_position_60(context: FactorContext):
    daily = context.load("daily.parquet")
    # Use the point-in-time adjusted base (cumprod(1+pct_chg/100)): raw close
    # rolling extrema stay polluted for up to 60 days after an ex-dividend
    # event (same class of bug the 7-31 audit fixed for the 252-day factors).
    adj = _adjusted_close(daily)
    high_60 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    )
    low_60 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).min()
    )
    position = (adj - low_60) / (high_60 - low_60).replace(0, np.nan)
    return cross_sectional_rank(position)


@register_factor(
    name="ma_convergence_20_60",
    description="均线收敛因子，20日均线与60日均线的距离比率截面排名。",
    category="price",
    thesis="短均线相对长均线的偏离程度反映趋势加速/减速，极端收敛后常伴随趋势突破。",
    dependencies=("daily.parquet",),
)
def factor_ma_convergence_20_60(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    # 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会污染均线距离
    adj = _adjusted_close(daily_panel)
    ma_20 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    ma_60 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    convergence = ma_20 / ma_60.replace(0, np.nan) - 1.0
    return cross_sectional_rank(convergence)


# ── Breakout ──────────────────────────────────────────────────────────────

@register_factor(
    name="breakout_60",
    description="60日价格突破强度因子，close/max(high,60)-1截面排名。",
    category="price",
    thesis="价格突破近期高点反映上涨动能强劲，突破强度越高趋势延续性越强。",
    dependencies=("daily.parquet",),
)
def factor_breakout_60(context: FactorContext):
    daily = context.load("daily.parquet")
    # Raw high has no point-in-time adjusted counterpart; use the adjusted
    # close base's rolling maximum (George-Hwang style approximation), which
    # is immune to ex-dividend jumps in the 60-day window.
    adj = _adjusted_close(daily)
    previous = adj.groupby(level="Code").shift(1)
    adj_max = previous.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    )
    breakout = adj / adj_max.replace(0, np.nan) - 1.0
    return cross_sectional_rank(breakout)


# ── ATR ───────────────────────────────────────────────────────────────────

@register_factor(
    name="atr_20",
    description="20日平均真实波幅(ATR)因子，低ATR排前。",
    category="price",
    thesis="低ATR股票波动平稳、筹码稳定，高ATR意味着剧烈波动风险，低波异象支持低ATR溢价。",
    dependencies=("daily.parquet",),
)
def factor_atr_20(context: FactorContext):
    daily = context.load("daily.parquet")
    high = daily["high"]
    low = daily["low"]
    pre_close = daily["pre_close"]

    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    # Row-wise operation — no per-stock groupby needed
    tr = np.maximum(
        high - low,
        np.maximum((high - pre_close).abs(), (low - pre_close).abs()),
    )

    atr = tr.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-atr)


# ── Timeseries factors ───────────────────────────────────────────────────

@register_factor(
    name="ts_price_self_rank_60",
    description="价格自身60日位置因子，收盘价在自身60日高低区间的相对位置截面排名。",
    category="timeseries",
    thesis="现有price_position_60是截面对比，ts_price_self_rank_60是个股价格在自身60日范围内的位置，捕捉个股自身的超买超卖状态，与截面因子互补。",
    dependencies=("daily.parquet",),
)
def factor_ts_price_self_rank_60(context: FactorContext):
    daily = context.load("daily.parquet")
    # Adjusted base (cumprod(1+pct_chg/100)) — raw close extrema are
    # ex-dividend polluted for up to 60 days (see price_position_60).
    adj = _adjusted_close(daily)

    high_60 = rolling_group_max(adj, 60)
    low_60 = rolling_group_min(adj, 60)

    position = (adj - low_60) / (high_60 - low_60).replace(0, np.nan)
    return cross_sectional_rank(position)


@register_factor(
    name="ts_volume_zscore_20",
    description="成交量20日Z-score因子，当日成交量偏离20日均值的标准差数截面排名（高放量排后）。",
    category="timeseries",
    thesis="成交量异常放大（高Z-score）往往伴随信息冲击、主力进出或市场过度关注，后续可能面临反转压力。低Z-score（缩量）则可能处于蓄势阶段。Z-score标准化使不同股票的成交量更具可比性。",
    dependencies=("daily.parquet",),
)
def factor_ts_volume_zscore_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    vol = daily_panel["vol"]

    mean_20 = rolling_group_mean(vol, 20)
    std_20 = rolling_group_std(vol, 20)

    zscore = (vol - mean_20) / std_20.replace(0, np.nan)
    return cross_sectional_rank(-zscore)


# ── Market microstructure / liquidity ───────────────────────────────────

@register_factor(
    name="close_position_intraday_20",
    description="收盘价日内位置因子，20日均(close-low)/(high-low)截面排名。",
    category="price",
    thesis="收盘价在日内区间的相对位置揭示买卖压力——连续在区间高位收盘代表买盘主导（正向），在低位收盘代表卖盘主导。均线平滑后过滤单日噪音。",
    dependencies=("daily.parquet",),
)
def factor_close_position_intraday_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    denom = (daily_panel["high"] - daily_panel["low"]).replace(0, np.nan)
    position = (daily_panel["close"] - daily_panel["low"]) / denom
    avg_pos = position.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(avg_pos)


@register_factor(
    name="volume_surge_3d",
    description="成交量脉冲因子，3日最大(vol/60日中位数vol)截面排名（负向：脉冲后反转）。",
    category="price",
    thesis="成交量短期急剧放大往往是信息冲击或情绪顶点的标志。极端放量后A股存在显著的反转效应——放量脉冲越大，后续回调压力越强。使用中位数而非均值避免极端日污染基线。",
    dependencies=("daily.parquet",),
)
def factor_volume_surge_3d(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    vol = daily_panel["vol"]
    vol_median_60 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).median()
    )
    vol_ratio = vol / vol_median_60.replace(0, np.nan)
    surge = vol_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    return cross_sectional_rank(-surge)


# ── Parkinson (High-Low) volatility ────────────────────────────────────────

@register_factor(
    name="high_low_volatility_20",
    description="Parkinson波动率因子，20日基于最高最低价的波动率估计截面排名（高波排后）。",
    category="price",
    thesis="Parkinson(1980)波动率使用日内高低价范围，比收盘价波动率效率高5.2倍——在同窗口下能更精确地捕捉真实波动。高HL波动率=价格振幅大=不确定性高，预期收益为负。与volatility_20互补：一个用极差估计波动，一个用收盘收益率估计。",
    dependencies=("daily.parquet",),
)
def factor_high_low_volatility_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    high = daily_panel["high"]
    low = daily_panel["low"]

    # Parkinson estimator: sqrt(1/(4*ln(2)*n) * sum(ln(H/L)^2))
    hl_ratio = high / low.replace(0, np.nan)
    hl_ratio = hl_ratio.where(hl_ratio > 0, np.nan)  # guard against log(<=0)
    hl_ratio_log = np.log(hl_ratio)
    hl_sq = hl_ratio_log ** 2
    parkinson_raw = np.sqrt(hl_sq.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    ) / (4.0 * np.log(2)))
    return cross_sectional_rank(-parkinson_raw)


# ── Turnover volatility (liquidity risk) ───────────────────────────────────

@register_factor(
    name="turnover_std_20",
    description="换手率波动率因子，20日换手率标准差截面排名（高换手波动排后=流动性风险）。",
    category="price",
    thesis="换手率剧烈波动意味着流动性不稳定——要么是资金突击进出、要么是筹码松动。与turnover_20互补：一个看换手水平，一个看换手稳定性。高换手波动代表流动性风险溢价。",
    dependencies=("finance.parquet",),
)
def factor_turnover_std_20(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    to_vol = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-to_vol)


# ── Volume indicators ─────────────────────────────────────────────────────

@register_factor(
    name="volume_momentum_5",
    description="5日成交量动量因子 (量增排前)。",
    category="price",
    thesis="成交量短期增长意味着关注度提升，量先于价是A股常见规律",
    dependencies=("daily.parquet",),
)
def factor_volume_momentum_5(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    vol = daily_panel["vol"]
    ma_5 = rolling_group_mean(vol, 5)
    ma_20 = rolling_group_mean(vol, 20)
    ratio = safe_divide(ma_5, ma_20 + 1e-8)
    return cross_sectional_rank(ratio)


# ── MA distance ───────────────────────────────────────────────────────────

@register_factor(
    name="distance_from_ma_5",
    description="收盘价/5日均线-1因子截面排名。",
    category="price",
    thesis="短期偏离均线过大存在回归压力，但强势股可维持正偏离",
    dependencies=("daily.parquet",),
)
def factor_distance_from_ma_5(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    # 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
    adj = _adjusted_close(daily_panel)
    ma_5 = rolling_group_mean(adj, 5)
    return cross_sectional_rank(safe_divide(adj - ma_5, ma_5 + 1e-8))


@register_factor(
    name="distance_from_ma_120",
    description="收盘价/120日均线-1因子截面排名 (负向：远离均线=回归压力)。",
    category="price",
    thesis="价格大幅偏离半年线后均值回复力量增强，低偏离股更安全",
    dependencies=("daily.parquet",),
)
def factor_distance_from_ma_120(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    # 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
    adj = _adjusted_close(daily_panel)
    ma_120 = rolling_group_mean(adj, 120)
    return cross_sectional_rank(-safe_divide(adj - ma_120, ma_120 + 1e-8).abs())


# ── Overnight / gap ───────────────────────────────────────────────────────

@register_factor(
    name="overnight_gap_vol_20",
    description="20日隔夜跳空波动率因子 (高波动排后, 负向)。",
    category="price",
    thesis="隔夜跳空波动大意味着信息不确定性高，可能存在信息不对称风险",
    dependencies=("daily.parquet",),
)
def factor_overnight_gap_vol_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    close = daily_panel["close"]
    open_p = daily_panel["open"]
    # pre_close is dividend-adjusted at ex-dividend dates, unlike close.shift(1)
    prev_close = daily_panel["pre_close"]
    overnight_ret = safe_divide(open_p - prev_close, prev_close + 1e-8)
    gap_vol = overnight_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-gap_vol)
