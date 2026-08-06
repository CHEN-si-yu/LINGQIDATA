"""
Volume-price dynamics factors — Class 1 (daily.parquet / finance.parquet).

量价动力学族:VWAP 偏离、量积累、流动性冲击、Amivest、A-H 交叉等。

- 当日比值类指标(量占比、VWAP偏离)除权日局部失真可接受;
- 需要跨日极值/方向判定的(AD线方向)用 pct_chg 符号或复权基座;
- A-H 交叉因子(ah_vol/ah_amount)仅约百余只 A+H 股有值,其余为 NaN——
  覆盖范围天然受限,开发保留(质量评估阶段按 skill.md NaN 标准裁决);
- rolling corr 一律宽表。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, stack_date_code
from .momentum_rebuilt import _adjusted_close


# ═══════════════════════════════════════════════════════════════════════════════
# VWAP / 均价偏离
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="vwap_daily_deviation",
    description="日频VWAP偏离因子：close/(amount/vol)−1截面排名（收盘高于日均价=尾盘强势排前）。",
    category="price",
    thesis="收盘价相对当日VWAP(成交量加权均价)的偏离——收盘高于VWAP=尾盘买方主导"
           "(策略38聪明钱语境中的价格位置)，低于VWAP=尾盘抛压。"
           "日频口径的 VWAP 偏离，与分钟级 vwap_deviation 互补。",
    dependencies=("daily.parquet",),
)
def factor_vwap_daily_deviation(context: FactorContext):
    daily = context.load("daily.parquet")
    vwap = safe_divide(daily["amount"], daily["vol"])
    deviation = safe_divide(daily["close"] - vwap, vwap)
    return cross_sectional_rank(deviation)


@register_factor(
    name="avg_price_trend_20",
    description="均价趋势因子：当日VWAP的20日变化率截面排名（成交均价抬升排前）。",
    category="price",
    thesis="成交均价(amount/vol)的20日变化率衡量持仓成本的迁移方向——均价持续抬升="
           "增量资金以更高价位进场(换手充分、筹码上移)，均价下移=筹码下移(套牢加深)。",
    dependencies=("daily.parquet",),
)
def factor_avg_price_trend_20(context: FactorContext):
    daily = context.load("daily.parquet")
    vwap = safe_divide(daily["amount"], daily["vol"])
    # VWAP 为当日实际成交均价(未复权),跨日 pct_change 在除权日会产生伪位移;
    # 乘 scale=adj/close 折算到复权空间后再比较(2026-08-05 修复)
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    trend = (vwap * scale).groupby(level="Code").transform(
        lambda s: s.pct_change(20, fill_method=None)
    )
    return cross_sectional_rank(trend)


# ═══════════════════════════════════════════════════════════════════════════════
# 量能结构
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="up_day_volume_ratio_20",
    description="上涨日量占比因子：20日上涨日成交量占总量的比例截面排名（上涨放量排前）。",
    category="price",
    thesis="上涨日的成交量占比衡量「量能的方向质量」——涨时放量(占比>50%)=资金主动做多；"
           "涨时缩量=无量反弹(不可持续)。是量价共振的日频简化版。",
    dependencies=("daily.parquet",),
)
def factor_up_day_volume_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    vol = daily["vol"]
    up = (daily["pct_chg"] > 0).astype(float)
    up_vol = (up * vol).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    tot_vol = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    ratio = safe_divide(up_vol, tot_vol)
    return cross_sectional_rank(ratio)


@register_factor(
    name="volume_skew_5d",
    description="5日量能偏度因子：5日成交量的偏度截面排名（放量脉冲排前，负向排名）。",
    category="price",
    thesis="5日成交量偏度正=量能脉冲式放大(单日巨量主导)，负=均匀放量。"
           "正偏度常对应事件驱动的一次性放量，脉冲放量后量能难以持续，负向排名。",
    dependencies=("daily.parquet",),
)
def factor_volume_skew_5d(context: FactorContext):
    daily = context.load("daily.parquet")
    skew = daily["vol"].groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).skew()
    )
    return cross_sectional_rank(-skew)


@register_factor(
    name="zero_return_fraction_20",
    description="零收益占比因子：20日|pct_chg|<0.1%的天数占比截面排名（负向，交投冷淡排后）。",
    category="price",
    thesis="Amivest流动性视角：零收益(涨跌幅<0.1%)占比高=成交稀疏、价格惰性"
           "(流动性差，交易成本高)；占比低=价格活跃、定价充分。"
           "流动性差的股票难以交易，负向排名。",
    dependencies=("daily.parquet",),
)
def factor_zero_return_fraction_20(context: FactorContext):
    daily = context.load("daily.parquet")
    zero = daily["pct_chg"].abs().lt(0.1).astype(float)
    freq = zero.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-freq)


@register_factor(
    name="liquidity_shock_20",
    description="流动性冲击因子：Amihud(20日均值)相对前20日的变化截面排名（负向，冲击放大排后）。",
    category="price",
    thesis="Amihud=|收益|/成交额衡量单位成交额的价格冲击——其20日均值相对更早20日的"
           "放大=流动性恶化(冲击成本上升，常伴随恐慌抛售)；收敛=流动性修复。"
           "先log再求比，消除量纲差异。",
    dependencies=("daily.parquet",),
)
def factor_liquidity_shock_20(context: FactorContext):
    daily = context.load("daily.parquet")
    amihud = safe_divide((daily["pct_chg"] / 100.0).abs(), daily["amount"])
    log_amihud = np.log(amihud + 1e-12)
    ma_now = log_amihud.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    ma_prev = ma_now.groupby(level="Code").shift(20)
    shock = ma_now - ma_prev
    return cross_sectional_rank(-shock)


@register_factor(
    name="turnover_ret_corr_20",
    description="换手率-收益相关因子：20日换手率与收益的相关性截面排名（量价同步排前）。",
    category="price",
    thesis="换手率与收益的正相关=上涨伴随放量(量价同步、趋势健康)，负相关=上涨缩量/"
           "下跌放量(背离)。用 finance.turnover_rate 与 pct_chg 的20日滚动相关"
           "(宽表向量化)，捕捉资金与价格的同步性。",
    dependencies=("daily.parquet", "finance.parquet"),
)
def factor_turnover_ret_corr_20(context: FactorContext):
    daily = context.load("daily.parquet")
    fin = context.load("finance.parquet")
    ret = (daily["pct_chg"] / 100.0).unstack("Code")
    to = fin["turnover_rate"].unstack("Code")
    common_cols = ret.columns.intersection(to.columns)
    ret = ret[common_cols]
    to = to[common_cols]
    corr = ret.rolling(20, min_periods=10).corr(to)
    return cross_sectional_rank(stack_date_code(corr))


# ═══════════════════════════════════════════════════════════════════════════════
# 量积累 / 背离
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="accumulation_distribution_20",
    description="量积累线斜率因子：AD线20日变化截面排名（资金净积累排前）。",
    category="price",
    thesis="Accumulation/Distribution线=Σ CLV×vol，CLV=(收盘位置)衡量每根K线的资金"
           "进出——AD线20日变化=近一月资金净积累方向。CLV用当日高低区间比值"
           "(除权日局部失真可接受)。",
    dependencies=("daily.parquet",),
)
def factor_accumulation_distribution_20(context: FactorContext):
    daily = context.load("daily.parquet")
    high = daily["high"]
    low = daily["low"]
    close = daily["close"]
    hl = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl
    ad = (clv * daily["vol"]).groupby(level="Code").cumsum()
    slope = ad.groupby(level="Code").diff(20)
    avg_vol = daily["vol"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    norm = safe_divide(slope, avg_vol + 1e-10)
    return cross_sectional_rank(norm)


@register_factor(
    name="volume_price_divergence_score",
    description="量价背离得分因子：价动量排名−量动量排名的背离截面排名（价强量弱背离排前）。",
    category="price",
    thesis="价格动量与成交量动量的方向背离——价升量缩=上涨缺乏确认(顶部背离预警)；"
           "价跌量增=恐慌放量(底部临近)。排名高=价强量弱的可疑上涨，作谨慎信号。",
    dependencies=("daily.parquet",),
)
def factor_volume_price_divergence_score(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    price_mom = adj.groupby(level="Code").pct_change(20, fill_method=None)
    vol_mom = daily["vol"].groupby(level="Code").transform(
        lambda s: s.pct_change(20, fill_method=None)
    )
    pr = price_mom.groupby(level="Date").rank(pct=True)
    vr = vol_mom.groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(pr - vr)
