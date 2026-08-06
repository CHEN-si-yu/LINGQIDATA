"""
Volume-price coupling factors — Class 1 (daily.parquet only).

量价耦合因子:成交量/成交额与价格走势的交互结构。全部只依赖 daily.parquet
的未复权 OHLCV + 复权口径收益率,point-in-time 安全(无 future data):
- 收益一律走 pct_chg(pre_close 已复权,除权日无跳变)
- 需要连续价格序列处使用 momentum_rebuilt._adjusted_close 自建后复权基座
  (cumprod(1+pct_chg/100)),禁止 close.pct_change()/close.shift(1)
- 全部向量化(groupby.transform 每股 rolling),无 rolling().apply
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide
from .momentum_rebuilt import _adjusted_close


@register_factor(
    name="volume_tilt_20",
    description="量能倾斜：20日量加权收益与等权收益之差截面排名。正倾斜=收益主要来自放量日。",
    category="price",
    thesis="若20日收益主要来自放量日(量加权收益>等权收益)，说明上涨由真实资金推动而非小量偷袭，"
           "趋势可信度高；若上涨全靠缩量日(倾斜为负)，参与者稀少、反转风险大。"
           "该因子量化了'收益的资金确认度'。",
    dependencies=("daily.parquet",),
)
def factor_volume_tilt_20(context: FactorContext):
    daily = context.load("daily.parquet")
    pct = daily["pct_chg"]
    vol = daily["vol"]
    wsum = (pct * vol).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    vsum = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    vw = safe_divide(wsum, vsum)
    ew = pct.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    tilt = vw - ew
    return cross_sectional_rank(tilt)


@register_factor(
    name="ret_efficiency_20",
    description="价格效率：20日累计收益/20日累计成交额截面排名。单位成交额推动的价格变动效率。",
    category="price",
    thesis="同样的成交额推动更大的价格变动=筹码锁定好、抛压小(amihud 的收益侧有符号版本)；"
           "价格效率高的股票上涨更'省力'，趋势阻力小。该因子与 amihud 互补：amihud 看成本的绝对水平，"
           "本因子看收益的相对效率。",
    dependencies=("daily.parquet",),
)
def factor_ret_efficiency_20(context: FactorContext):
    daily = context.load("daily.parquet")
    pct_sum = daily["pct_chg"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    amt_sum = daily["amount"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    eff = safe_divide(pct_sum, amt_sum)
    return cross_sectional_rank(eff)


# ── 突破/跳空 × 量能确认 ─────────────────────────────────────────────────

@register_factor(
    name="volume_breakout_confirm_20",
    description="突破量能确认：20日内突破前20日高点的量比累计截面排名。突破+放量=强确认。",
    category="price",
    thesis="价格突破关键高点后，量能是否配合决定突破的有效性：放量突破=新资金入场接力，"
           "缩量突破=大概率假突破。以自建后复权基座判定突破(避免除权日伪突破)，"
           "对突破日的 vol/vol_ma20 量比做20日累计，无突破则记0。",
    dependencies=("daily.parquet",),
)
def factor_volume_breakout_confirm_20(context: FactorContext):
    daily = context.load("daily.parquet")
    vol = daily["vol"]
    adj = _adjusted_close(daily)
    prior_high = adj.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max().shift(1)
    )
    breakout = (adj > prior_high) & prior_high.notna()
    vol_ma20 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    vol_ratio = safe_divide(vol, vol_ma20)
    confirm = (breakout.astype(float) * vol_ratio).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    return cross_sectional_rank(confirm)


@register_factor(
    name="gap_volume_interaction_20",
    description="跳空×量能交互：20日均值(跳空幅度×量比)截面排名。高开且放量=资金抢筹。",
    category="price",
    thesis="跳空高开(open/pre_close)反映隔夜信息冲击，量比反映当日参与度——高开+放量=资金抢筹、"
           "方向确认；高开+缩量=高开低走风险；低开+放量=恐慌抛售。交互项同时捕捉方向与确认度。"
           "pre_close 已复权，跳空口径无除权失真。",
    dependencies=("daily.parquet",),
)
def factor_gap_volume_interaction_20(context: FactorContext):
    daily = context.load("daily.parquet")
    vol = daily["vol"]
    gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
    vol_ma20 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    vol_ratio = safe_divide(vol, vol_ma20)
    gi = (gap * vol_ratio).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(gi)


# ── 量能结构 ──────────────────────────────────────────────────────────────

@register_factor(
    name="max_vol_day_contribution_20",
    description="单日脉冲主导度：20日内最大量日的|收益|占20日|收益|总和比例截面排名。",
    category="price",
    thesis="若一段行情的收益集中在单日脉冲(占比高)，说明行情靠事件驱动、不可持续，"
           "后续波动加大；收益均匀分布=筹码换手充分、趋势扎实。该因子度量行情的时间结构。",
    dependencies=("daily.parquet",),
)
def factor_max_vol_day_contribution_20(context: FactorContext):
    daily = context.load("daily.parquet")
    vol = daily["vol"]
    abs_pct = daily["pct_chg"].abs()
    # 近似取 rolling 窗口内 vol 最大的 1-2 个交易日(同值并列均计入)
    max_vol = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    is_max_day = (vol == max_vol) & (vol > 0)
    max_day_abs = (abs_pct * is_max_day.astype(float)).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    total_abs = abs_pct.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    contrib = safe_divide(max_day_abs, total_abs)
    return cross_sectional_rank(-contrib)


@register_factor(
    name="close_position_vol_weighted_20",
    description="量能加权收盘位置：20日均值(量比×日内收盘位置)截面排名。量能集中于高位收盘=强承接。",
    category="price",
    thesis="日内收盘位置((close-low)/(high-low))反映当日多空结果，乘上相对量比后，"
           "量能大的日子权重更高——若放量日都收在日内高位，说明大资金在承接，后续看涨；"
           "放量日收在低位则是出货特征。",
    dependencies=("daily.parquet",),
)
def factor_close_position_vol_weighted_20(context: FactorContext):
    daily = context.load("daily.parquet")
    vol = daily["vol"]
    pos = safe_divide(daily["close"] - daily["low"], daily["high"] - daily["low"])
    vol_ma20 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    vol_norm = safe_divide(vol, vol_ma20)
    vp = (vol_norm * pos).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(vp)


@register_factor(
    name="amount_surge_count_20",
    description="资金脉冲频率：20日内成交额>1.5倍20日均额的交易日占比截面排名。",
    category="price",
    thesis="成交额异常放大(>1.5×均额)的频率衡量资金进出的脉冲性：频繁脉冲=有大资金在反复进出，"
           "关注度与博弈程度高(情绪票特征)；无脉冲=交投冷清、被市场遗忘。频率而非幅度，"
           "避免被单日巨量主导。",
    dependencies=("daily.parquet",),
)
def factor_amount_surge_count_20(context: FactorContext):
    daily = context.load("daily.parquet")
    amount = daily["amount"]
    amt_ma20 = amount.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    surge = (amount > 1.5 * amt_ma20).astype(float)
    frac = surge.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(frac)
