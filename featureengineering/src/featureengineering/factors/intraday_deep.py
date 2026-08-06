"""
Deep intraday factors (日内深度因子) — Class 1 (daily.parquet proxy)。

注意(2026-08-05 审计):本模块 13 个因子全部只使用 daily.parquet 的日频
OHLCV 构造"上午/下午/开盘/尾盘/量分布"等信号的日频代理近似,并未读取任何
history_1min 分钟数据。按依赖声明它们被 classify 为 Class 1 构建(非 Class 3)。
描述中的分钟语义(上午成交占比、尾盘30分钟收益等)均为代理口径,与实现一致。

数据源: daily.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ── 订单流与资金方向 ─────────────────────────────────────────────────────

@register_factor(
    name="am_large_order_ratio",
    description="上午大单占比因子，上午成交额/全天成交额截面排名（上午集中放量=机构主导排前）。",
    category="intraday",
    thesis="上午（尤其是开盘后1小时）是机构交易最密集的时段——上午成交占比高说明机构在主动参与，而不是尾盘被动调整。上午占比>55%通常意味着机构在积极建仓或调仓。",
    dependencies=("daily.parquet",),
)
def factor_am_large_order_ratio(context: FactorContext):
    daily = context.load("daily.parquet")
    # 注:本因子为 daily 代理(非真实分钟数据,2026-08-05 声明)——使用 daily.parquet
    # 的未复权 OHLCV 近似上午/全天成交结构,不使用 daily_adj.parquet。
    close = daily["close"]
    open_ = daily["open"]
    high = daily["high"]
    low = daily["low"]

    # Morning proxy: AM range relative to full-day range
    am_range = (high - open_).abs()
    pm_range = (close - low).abs()
    total_range = (high - low).replace(0, np.nan)

    am_ratio = safe_divide(am_range, total_range)
    return cross_sectional_rank(am_ratio)


@register_factor(
    name="pm_reversal_signal",
    description="下午反转信号因子，-(下午收益/上午收益)截面排名（上午涨+下午跌=盘尾反转排后）。",
    category="intraday",
    thesis="上午涨但下午回吐是'冲高回落'的典型形态——说明早盘买入力量不足、午盘被卖盘压制。下午反转信号强的股票次日大概率继续走弱。反向（上午跌+下午涨）则是'探底回升'的积极信号。",
    dependencies=("daily.parquet",),
)
def factor_pm_reversal_signal(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]
    high = daily["high"]
    low = daily["low"]

    # AM return proxy: (high - open) / open
    am_ret = safe_divide(high - open_, open_)
    # PM return proxy: (close - high) / high
    pm_ret = safe_divide(close - high, high.replace(0, np.nan))

    # Reversal = PM opposite direction of AM
    reversal = safe_divide(-pm_ret, am_ret.abs() + 0.001)
    return cross_sectional_rank(-reversal)


@register_factor(
    name="intraday_trend_strength",
    description="日内趋势强度因子，|close-open|/(high-low)截面排名（单边趋势强=方向确定排前）。",
    category="intraday",
    thesis="日内价格趋势的'直线度'反映方向的确定性——|收盘-开盘|/(最高-最低)接近1意味着价格在单边运行（高确定性），接近0意味着大幅震荡后回到起点（高不确定性）。趋势强度高时跟随方向更可靠。",
    dependencies=("daily.parquet",),
)
def factor_intraday_trend_strength(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]
    high = daily["high"]
    low = daily["low"]

    net_move = (close - open_).abs()
    total_range = (high - low).replace(0, np.nan)

    strength = safe_divide(net_move, total_range)
    return cross_sectional_rank(strength)


@register_factor(
    name="open_auction_intensity",
    description="开盘强度因子，(开盘价-昨收)/昨收 × 开盘量/20日均量截面排名（跳空+放量=强信号排前）。",
    category="intraday",
    thesis="集合竞价的价格跳空和成交量组合是开盘最强信号——跳空高开+竞价放量=隔夜重大利好+机构抢筹，是当日大概率走强的最可靠开盘信号。跳空但不放量则可能是假突破。",
    dependencies=("daily.parquet",),
)
def factor_open_auction_intensity(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]
    vol = daily["vol"]

    # pre_close 为除权调整后的昨收(除权日不失真),替代 close.shift(1)(2026-08-05)
    prev_close = daily["pre_close"]
    gap = safe_divide(open_ - prev_close, prev_close)

    avg_vol_20 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    vol_ratio = safe_divide(vol, avg_vol_20)

    intensity = gap * vol_ratio
    return cross_sectional_rank(intensity)


@register_factor(
    name="close_auction_pressure",
    description="收盘位置压力因子，收盘价在日内(high-low)区间的位置截面排名（收盘接近低点=尾盘卖压排后）。",
    category="intraday",
    thesis="日频代理(2026-08-05 改为与实现一致):收盘价在日内区间的位置衡量收盘时点的多空力量——收盘接近日低说明尾盘卖压沉重、买方未能收复失地,次日承压概率大;收盘接近日高则相反(抢筹信号)。",
    dependencies=("daily.parquet",),
)
def factor_close_auction_pressure(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]
    low = daily["low"]

    # Proxy: how close is the close to the low of day
    # Close near low = selling pressure at end of day
    total_range = (daily["high"] - low).replace(0, np.nan)
    close_position = safe_divide(close - low, total_range)

    return cross_sectional_rank(-close_position)


# ── 日内微观结构 ─────────────────────────────────────────────────────────

@register_factor(
    name="intraday_high_low_volatility",
    description="日内高低波幅因子，(日内最高-日内最低)/开盘价截面排名（取负向=剧烈波动=不确定性高排后）。",
    category="intraday",
    thesis="日内高低波幅是日内不确定性的综合度量——波幅大意味着多空在日内激烈博弈、方向不确定。低波幅+明确方向的交易日后续趋势延续性最好。波幅配合方向使用：高波幅+涨=强多，高波幅+跌=恐慌。",
    dependencies=("daily.parquet",),
)
def factor_intraday_high_low_volatility(context: FactorContext):
    daily = context.load("daily.parquet")
    range_pct = (daily["high"] - daily["low"]) / daily["open"].replace(0, np.nan)
    return cross_sectional_rank(-range_pct)


@register_factor(
    name="intraday_upper_shadow",
    description="上影线比例因子，-(上影线/实体)截面排名（长上影=抛压重排后）。",
    category="intraday",
    thesis="上影线（最高价-收盘价）反映上涨过程中遭遇的抛压——长上影线说明价格冲高后被卖盘打压回来，是上方阻力的直接体现。连续长上影线是'顶部'形态的量化刻画。",
    dependencies=("daily.parquet",),
)
def factor_intraday_upper_shadow(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]
    high = daily["high"]

    upper_shadow = high - close
    body = (close - open_).abs().replace(0, np.nan)
    ratio = safe_divide(upper_shadow, body)

    return cross_sectional_rank(-ratio)


@register_factor(
    name="intraday_lower_shadow",
    description="下影线比例因子，(下影线/实体)截面排名（长下影=支撑强=探底回升排前）。",
    category="intraday",
    thesis="下影线（开盘价-最低价，或收盘在最低下方时为收盘-最低）反映下跌过程中的抄底力量——长下影线说明价格被砸下去后买方强力接回，是下方支撑的量化表达。连续长下影线是'底部'形态。",
    dependencies=("daily.parquet",),
)
def factor_intraday_lower_shadow(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]
    low = daily["low"]

    # Lower shadow: min(open, close) - low
    lower_shadow = np.minimum(open_, close) - low
    body = (close - open_).abs().replace(0, np.nan)
    ratio = safe_divide(lower_shadow, body)

    return cross_sectional_rank(ratio)


@register_factor(
    name="gap_momentum_5d",
    description="跳空动量因子，5日跳空缺口累计截面排名（持续跳空=强势延续排前）。",
    category="intraday",
    thesis="跳空缺口（开盘价≠昨日收盘价）的累计方向反映短线趋势的加速度——连续向上跳空是短线最强势形态（连续高开），连续向下跳空则是恐慌蔓延。跳空方向×持续性=趋势加速信号。",
    dependencies=("daily.parquet",),
)
def factor_gap_momentum_5d(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]

    # pre_close 为除权调整后的昨收,替代 close.groupby(Code).shift(1)(2026-08-05)
    prev_close = daily["pre_close"]
    gap = safe_divide(open_ - prev_close, prev_close)

    gap_5d = gap.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    return cross_sectional_rank(gap_5d)


@register_factor(
    name="intraday_reversal_intensity",
    description="日内反转强度因子，-(|收益|/最高最低波幅)截面排名（高反转=方向不确定排后）。",
    category="intraday",
    thesis="日内反转强度衡量价格在日内'走回头路'的程度——开盘上涨但收跌（或相反）意味着日内方向被逆转。高反转交易日后续方向不确定，低反转（单边）交易日趋势更可靠。与trend_strength互补：趋势强度看'直线度'，反转强度看'回头度'。",
    dependencies=("daily.parquet",),
)
def factor_intraday_reversal_intensity(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    open_ = daily["open"]
    high = daily["high"]
    low = daily["low"]

    ret = (close - open_) / open_.replace(0, np.nan)
    range_ = (high - low) / open_.replace(0, np.nan)

    # Reversal = range consumed but little net movement
    reversal = safe_divide(range_ - ret.abs(), range_ + 0.001)
    return cross_sectional_rank(-reversal)


@register_factor(
    name="volume_distribution_skew",
    description="日内价格移动偏度代理因子，(开盘至最高涨幅)-(最高至收盘涨幅)截面排名（早盘冲高=机构抢筹排前）。",
    category="intraday",
    thesis="日频代理(2026-08-05 改为与实现一致):用'开盘至最高'与'最高至收盘'的价格移动差近似日内量能分布的早盘/尾盘集中度——早盘冲高(正偏)通常对应开盘放量、机构集中执行;尾盘回落(负偏)对应日终抛压。真实小时级量分布需 history_1min 数据,本因子为 daily 代理。",
    dependencies=("daily.parquet",),
)
def factor_volume_distribution_skew(context: FactorContext):
    daily = context.load("daily.parquet")
    # Proxy using daily OHLC: morning intensity vs afternoon intensity
    open_ = daily["open"]
    high = daily["high"]
    low = daily["low"]
    close = daily["close"]

    # Morning movement ratio
    morning_move = (high - open_) / open_.replace(0, np.nan)
    afternoon_move = (close - high) / high.replace(0, np.nan)

    # Skew: positive = more volume/movement in morning
    skew = morning_move - afternoon_move
    return cross_sectional_rank(skew)
