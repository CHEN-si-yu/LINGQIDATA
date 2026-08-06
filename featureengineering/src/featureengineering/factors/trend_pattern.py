from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_max,
    rolling_group_mean,
    rolling_group_min,
    rolling_group_std,
    safe_divide,
)
from .momentum_rebuilt import _adjusted_close


# ── ADX / trend strength ────────────────────────────────────────────────────

def _true_range(high, low, pre_close) -> pd.Series:
    # pre_close is dividend-adjusted at ex-dividend dates, unlike close.shift(1)
    tr1 = high - low
    tr2 = (high - pre_close).abs()
    tr3 = (low - pre_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _directional_movement(high, low, pre_close, scale, window=14):
    # scale = 后复权基座/未复权close,把未复权 high/low 折算到复权空间后再 diff:
    # 未复权 high/low 在除权日同幅跳变,直接 diff 会制造巨额虚假方向移动
    # (TR 部分已用 pre_close,此处为 DM 部分同口径修复,2026-08-05)。
    adj_high = high * scale
    adj_low = low * scale
    up_move = adj_high.groupby(level="Code").diff()
    down_move = adj_low.groupby(level="Code").diff()
    dm_plus = up_move.where((up_move > 0) & (up_move > down_move.abs()), 0.0)
    dm_minus = (-down_move).where((down_move < 0) & (down_move.abs() > up_move), 0.0)
    atr = _true_range(high, low, pre_close).groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=window // 2).mean()
    )
    di_plus = 100 * dm_plus.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=window // 2).mean()
    ) / atr
    di_minus = 100 * dm_minus.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=window // 2).mean()
    ) / atr
    return di_plus, di_minus


@register_factor(
    name="adx_14",
    description="14日趋势强度因子 (ADX)",
    category="price",
    thesis="ADX衡量趋势强度(非方向)，趋势明确的股票动量策略更有效",
    dependencies=("daily.parquet",),
)
def adx_14(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    di_plus, di_minus = _directional_movement(daily["high"], daily["low"], daily["pre_close"], scale, 14)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-8)
    adx = dx.groupby(level="Code").transform(
        lambda s: s.rolling(14, min_periods=7).mean()
    )
    return cross_sectional_rank(adx)


@register_factor(
    name="di_plus_minus_ratio_14",
    description="14日DI+/DI-比率因子 (多头趋势强度)",
    category="price",
    thesis="DI+>DI-意味着上升趋势，DI+/DI-比率衡量多头相对空头的优势",
    dependencies=("daily.parquet",),
)
def di_plus_minus_ratio_14(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    di_plus, di_minus = _directional_movement(daily["high"], daily["low"], daily["pre_close"], scale, 14)
    ratio = safe_divide(di_plus, di_minus + 1e-8)
    return cross_sectional_rank(ratio)

# ── Bollinger Bands ─────────────────────────────────────────────────────────

@register_factor(
    name="bollinger_position_20",
    description="20日布林带位置因子 (%B, 高位排后, 负向)",
    category="price",
    thesis="%B>1意味着突破上轨(短期超买)，<0意味着突破下轨(超卖)，均值回归视角",
    dependencies=("daily.parquet",),
)
def bollinger_position_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    # 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染均线与带宽
    adj = _adjusted_close(daily)
    ma = rolling_group_mean(adj, 20)
    std = rolling_group_std(adj, 20)
    b_upper = ma + 2 * std
    b_lower = ma - 2 * std
    pct_b = (adj - b_lower) / (b_upper - b_lower + 1e-8)
    return cross_sectional_rank(-pct_b)  # lower position = more mean-reversion upside


@register_factor(
    name="bollinger_width_20",
    description="20日布林带宽度因子 (窄幅排前, 负向)",
    category="price",
    thesis="布林带宽缩减(bandwidth squeeze)预示突破即将到来，窄幅是低波动蓄力",
    dependencies=("daily.parquet",),
)
def bollinger_width_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    # 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染带宽
    adj = _adjusted_close(daily)
    ma = rolling_group_mean(adj, 20)
    std = rolling_group_std(adj, 20)
    width = safe_divide(4 * std, ma)
    return cross_sectional_rank(-width)  # narrow bandwidth = potential breakout


# ── Donchian channels ───────────────────────────────────────────────────────

@register_factor(
    name="donchian_position_60",
    description="60日Donchian通道位置因子 (低位置排前, 负向)",
    category="price",
    thesis="价格在60日高低区间内的位置，低位意味着潜在反转上行",
    dependencies=("daily.parquet",),
)
def donchian_position_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    # Adjusted base (cumprod(1+pct_chg/100)) — raw close 60d extrema are
    # ex-dividend polluted for up to 60 days (see price.py price_position_60).
    adj = _adjusted_close(daily)
    hh = rolling_group_max(adj, 60)
    ll = rolling_group_min(adj, 60)
    position = (adj - ll) / (hh - ll + 1e-8)
    return cross_sectional_rank(-position)


@register_factor(
    name="donchian_breakout_20",
    description="20日Donchian突破强度因子",
    category="price",
    thesis="突破20日最高价是趋势启动的信号，突破幅度越大趋势越强",
    dependencies=("daily.parquet",),
)
def donchian_breakout_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    # 与 donchian_position_60 同口径:用后复权基座,除权日不产生假突破;
    # shift(1) 须按 Code 分组,排除当日(T-20..T-1 窗口),避免全局 shift 跨股错位。
    adj = _adjusted_close(daily)
    hh = rolling_group_max(adj.groupby(level="Code").shift(1), 20)
    breakout = safe_divide(adj - hh, hh)
    return cross_sectional_rank(breakout)


# ── Keltner channels ────────────────────────────────────────────────────────

@register_factor(
    name="keltner_position_20",
    description="20日Keltner通道位置因子 (高位排后, 负向)",
    category="price",
    thesis="Keltner通道使用ATR估计带宽，对波动率变化更敏感，位置过高意味着短期过度上涨",
    dependencies=("daily.parquet",),
)
def keltner_position_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    high, low = daily["high"], daily["low"]
    # 主轨 MA 走复权基座,与已复权的 ATR 保持同口径(未复权 close 跨日窗口受除权污染)
    adj = _adjusted_close(daily)
    tr = _true_range(high, low, daily["pre_close"])
    atr = tr.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    ma = rolling_group_mean(adj, 20)
    k_upper = ma + 1.5 * atr
    k_lower = ma - 1.5 * atr
    position = (adj - k_lower) / (k_upper - k_lower + 1e-8)
    return cross_sectional_rank(-position)


# ── MA crossover / distance ─────────────────────────────────────────────────

@register_factor(
    name="ma_distance_5_20",
    description="5日-20日均线距离因子 (乖离)",
    category="price",
    thesis="短期均线高于长期均线=上升趋势，乖离大小反映趋势强度",
    dependencies=("daily.parquet",),
)
def ma_distance_5_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    # 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
    adj = _adjusted_close(daily)
    ma_5 = rolling_group_mean(adj, 5)
    ma_20 = rolling_group_mean(adj, 20)
    distance = safe_divide(ma_5 - ma_20, ma_20)
    return cross_sectional_rank(distance)


@register_factor(
    name="ma_distance_5_60",
    description="5日-60日均线距离因子 (中期乖离)",
    category="price",
    thesis="短期均线与中期均线的距离衡量中期趋势强度",
    dependencies=("daily.parquet",),
)
def ma_distance_5_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    # 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
    adj = _adjusted_close(daily)
    ma_5 = rolling_group_mean(adj, 5)
    ma_60 = rolling_group_mean(adj, 60)
    distance = safe_divide(ma_5 - ma_60, ma_60)
    return cross_sectional_rank(distance)


@register_factor(
    name="ma_distance_20_60",
    description="20日-60日均线距离因子 (趋势一致性)",
    category="price",
    thesis="20日均线>60日均线=上升趋势，距离扩大=趋势加速",
    dependencies=("daily.parquet",),
)
def ma_distance_20_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    # 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
    adj = _adjusted_close(daily)
    ma_20 = rolling_group_mean(adj, 20)
    ma_60 = rolling_group_mean(adj, 60)
    distance = safe_divide(ma_20 - ma_60, ma_60)
    return cross_sectional_rank(distance)


# ── Price patterns ──────────────────────────────────────────────────────────

@register_factor(
    name="higher_highs_20",
    description="20日不断抬高的高点因子 (趋势延续)",
    category="price",
    thesis="不断创出更高高点是强势上升趋势的特征",
    dependencies=("daily.parquet",),
)
def higher_highs_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    # 跨20日分段比较 high 极值,需折算到复权空间,未复权 high 在除权日阶跃会伪造假高点结构
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    high = daily["high"] * scale

    def _hh_count(s, window):
        half = max(1, window // 2)
        s1 = s.rolling(half, min_periods=half // 2).max()
        s2 = s.shift(half).rolling(half, min_periods=half // 2).max()
        return (s1 > s2).astype(float)

    hh = high.groupby(level="Code").transform(lambda s: _hh_count(s, 20))
    return cross_sectional_rank(hh)


@register_factor(
    name="close_to_high_ratio_20",
    description="20日均收盘/最高价比率因子 (收盘强势)",
    category="price",
    thesis="收盘价接近最高价意味着持续的日内外买盘力量，收盘位置高预示次日强势",
    dependencies=("daily.parquet",),
)
def close_to_high_ratio_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily.parquet")
    ratio = (daily["close"] - daily["low"]) / (daily["high"] - daily["low"] + 1e-8)
    avg_ratio = rolling_group_mean(ratio, 20)
    return cross_sectional_rank(avg_ratio)


# ── Price channel patterns ────────────────────────────────────────────────

@register_factor(
    name="bollinger_position",
    description="布林带位置因子，(close-下轨)/(上轨-下轨)截面排名。",
    category="price",
    thesis="布林带位置衡量价格在波动区间中的相对位置——接近上轨=强势/超买，接近下轨=弱势/超卖。与传统布林带%B等价，是均值回复策略的基础信号。",
    dependencies=("daily.parquet",),
)
def factor_bollinger_position(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    # 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染均线与带宽
    adj = _adjusted_close(daily_panel)
    ma_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    std_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
    upper = ma_20 + 2 * std_20
    lower = ma_20 - 2 * std_20
    position = (adj - lower) / (upper - lower).replace(0, np.nan)
    return cross_sectional_rank(position)


@register_factor(
    name="bollinger_squeeze",
    description="布林带收缩因子，-(上轨-下轨)/均价截面排名（带宽窄=挤压突破前兆排前）。",
    category="price",
    thesis="布林带收缩(带宽变小)是波动率压缩的信号——低波动后往往伴随着剧烈的方向性突破。带宽处于历史低位时预示着即将出现趋势性行情。",
    dependencies=("daily.parquet",),
)
def factor_bollinger_squeeze(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    # 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染带宽
    adj = _adjusted_close(daily_panel)
    ma_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    std_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
    bandwidth = 4 * std_20 / ma_20.replace(0, np.nan)
    # Rank negative: narrow band = squeeze = ranked high
    return cross_sectional_rank(-bandwidth)


# ── Candlestick patterns ─────────────────────────────────────────────────

@register_factor(
    name="marubozu_ratio_10d",
    description="光头光脚阳线频率因子，近10日实体阳线(上下影极短)天数截面排名。",
    category="price",
    thesis="光头光脚阳线(没有上下影或极短的实体大阳线)代表日内空方被完全压制——出现频率高意味着买方压倒性强势。",
    dependencies=("daily.parquet",),
)
def factor_marubozu_ratio_10d(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    body = (daily_panel["close"] - daily_panel["open"]).abs()
    upper_shadow = daily_panel["high"] - daily_panel[["open", "close"]].max(axis=1)
    lower_shadow = daily_panel[["open", "close"]].min(axis=1) - daily_panel["low"]
    total_range = daily_panel["high"] - daily_panel["low"]
    is_marubozu = ((upper_shadow + lower_shadow) / total_range.replace(0, np.nan) < 0.1).astype(float)
    is_green = (daily_panel["close"] > daily_panel["open"]).astype(float)
    signal = is_marubozu * is_green
    ratio = signal.groupby(level="Code").transform(lambda s: s.rolling(10, min_periods=5).mean())
    return cross_sectional_rank(ratio)


@register_factor(
    name="hammer_ratio_20d",
    description="锤子线频率因子，近20日下影线>实体2倍且实体小的天数截面排名（反转信号排前）。",
    category="price",
    thesis="锤子线(长下影+小实体)是经典的技术反转形态——出现在下跌趋势中往往预示底部反转，长下影代表空方在盘中打压失败后被多方强势收复。",
    dependencies=("daily.parquet",),
)
def factor_hammer_ratio_20d(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    body = (daily_panel["close"] - daily_panel["open"]).abs()
    lower_shadow = daily_panel[["open", "close"]].min(axis=1) - daily_panel["low"]
    upper_shadow = daily_panel["high"] - daily_panel[["open", "close"]].max(axis=1)
    total_range = daily_panel["high"] - daily_panel["low"]
    is_hammer = ((lower_shadow > 2 * body) & (upper_shadow < 0.3 * total_range) & (body > 0)).astype(float)
    ratio = is_hammer.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    return cross_sectional_rank(ratio)


# ── Volume pattern ───────────────────────────────────────────────────────

@register_factor(
    name="volume_climax",
    description="放量异动因子，今日成交量/20日均量截面排名。",
    category="price",
    thesis="成交量突然放大数倍于平均水平是异动信号——或为机构建仓/出货、或为消息驱动的大规模换手。高量比往往意味着趋势变盘的前兆。",
    dependencies=("daily.parquet",),
)
def factor_volume_climax(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    vol = daily_panel["vol"]
    vol_ma_20 = vol.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    vol_ratio = vol / vol_ma_20.replace(0, np.nan)
    return cross_sectional_rank(vol_ratio)


@register_factor(
    name="volume_dry_up",
    description="缩量因子，-(20日最低成交量/20日均量)截面排名（极度缩量=变盘前兆排前）。",
    category="price",
    thesis="成交量极度萎缩(地量)代表市场交投意愿降至冰点——往往是趋势即将反转的前兆。在下跌趋势中地量=抛压枯竭=可能见底，上涨趋势中地量=追涨意愿不足=可能见顶。",
    dependencies=("daily.parquet",),
)
def factor_volume_dry_up(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    vol = daily_panel["vol"]
    vol_min_20 = vol.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).min())
    vol_ma_20 = vol.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    dryness = vol_min_20 / vol_ma_20.replace(0, np.nan)
    return cross_sectional_rank(-dryness)
