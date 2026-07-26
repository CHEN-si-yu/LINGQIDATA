"""
Volume-price elasticity and dynamics factors — Class 1 panel factors.

These factors capture the interaction between volume and price:
  - Volume-price elasticity (Kyle's lambda style)
  - Volume-turnover divergence
  - Price reversal volume confirmation
  - Volume acceleration

All factors use daily_adj.parquet and finance.parquet.
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
# Section A: Volume-Price Elasticity
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="volume_price_elasticity_20",
    description="量价弹性因子（20日|return|/|volume_change|配对均值截面排名，高弹性=薄盘排前）。",
    category="market_structure",
    thesis=(
        "量价弹性（Volume-Price Elasticity）度量了单位成交量变化引起的价格变化幅度。"
        "高弹性股票：少量资金即可推动大涨幅（流动性薄、筹码集中）——"
        "适合小资金但大资金进出困难。低弹性股票：大量资金才能推动小涨幅（流动性深）——"
        "更适合大资金配置。弹性是Kyle's lambda在日频的近似："
        "|ΔP| / |ΔV| 的平均值捕捉了'每一块钱成交额的价格冲击'。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_volume_price_elasticity_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")

    close = daily_adj["close"]
    amount = daily_adj["amount"]

    # Daily return absolute
    ret_abs = close.groupby(level="Code").transform(
        lambda s: s.pct_change(1).abs()
    )
    # Amount change absolute
    amt_chg_abs = amount.groupby(level="Code").transform(
        lambda s: s.pct_change(1).abs()
    )

    # Elasticity per day: |return| / |amount_change|
    elasticity_daily = safe_divide(ret_abs, amt_chg_abs + 1e-10)
    elasticity_daily = elasticity_daily.clip(0, 10)

    # 20-day average elasticity
    elasticity_20 = elasticity_daily.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    return cross_sectional_rank(elasticity_20)


@register_factor(
    name="volume_return_corr_20",
    description="量价相关性因子（20日vol变化率与return的Spearman截面排名，高相关=量价配合排前）。",
    category="market_structure",
    thesis=(
        "成交量变化与价格变化的20日相关性反映了量价配合的程度。"
        "高正相关（量增价涨+量缩价跌）=量价健康配合，趋势可靠性高；"
        "负相关（量增价跌=恐慌抛售，量缩价涨=资金撤出）=量价背离，趋势不可持续。"
        "该因子是技术分析中'量价配合原则'的量化表达。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_volume_return_corr_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")

    close = daily_adj["close"]
    vol = daily_adj["vol"]

    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_chg = vol.groupby(level="Code").transform(lambda s: s.pct_change(1))

    ret = ret.clip(-0.15, 0.15)
    vol_chg = vol_chg.clip(-5, 10)

    # 20-day rolling rank correlation (Spearman-style via rolling rank)
    def _rolling_rank_corr(x, y, window=20):
        xr = x.rolling(window, min_periods=10).apply(
            lambda s: s.rank().corr(pd.Series(range(len(s))).rank()), raw=False
        )
        return xr

    # Simplified: rolling Pearson on ranked values (approximates Spearman)
    ret_rank = ret.groupby(level="Date").rank(pct=True)
    vol_rank = vol_chg.groupby(level="Date").rank(pct=True)

    # Rolling correlation
    def _roll_corr(s, other, window=20):
        s_roll = s.rolling(window, min_periods=10)
        o_roll = other.rolling(window, min_periods=10)
        cov = (s * other).rolling(window, min_periods=10).mean() - s_roll.mean() * o_roll.mean()
        denom = s_roll.std() * o_roll.std()
        return safe_divide(cov, denom)

    corr = ret_rank.groupby(level="Code").transform(
        lambda s: _roll_corr(s, vol_rank.loc[s.index], 20)
    )
    corr = corr.clip(-1, 1)

    return cross_sectional_rank(corr)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Volume-Turnover Divergence
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="volume_turnover_divergence",
    description="量换手偏离因子（成交量20日变化率 - 换手率20日变化率截面排名）。",
    category="market_structure",
    thesis=(
        "成交量(vol)和换手率(turnover rate)应当同步变化——"
        "换手率 = vol / total_shares，分母为总股本（相对稳定）。"
        "当成交量变化显著大于换手率变化时，意味着公司的总股本发生了变化"
        "（增发、回购、解禁等），成交量的增长更多来自结构因素而非交易活跃度。"
        "当换手率变化大于成交量变化时，可能反映了限售股解禁等股本扩张。"
        "两者的偏离捕捉了成交量变化的'质量'。"
    ),
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def factor_volume_turnover_divergence(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    finance = context.load("finance.parquet")

    vol = daily_adj["vol"]
    turnover = finance["turnover_rate"]

    # 20-day change rate
    vol_chg = vol.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    to_chg = turnover.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )

    common = vol_chg.index.intersection(to_chg.index)
    vol_a = vol_chg.loc[common].clip(-1, 5)
    to_a = to_chg.loc[common].clip(-1, 5)

    divergence = vol_a - to_a

    return cross_sectional_rank(divergence)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Price Reversal Volume Confirmation
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="price_reversal_volume_confirm",
    description="价格反转成交量确认因子（反转日缩量程度截面排名，缩量反转=可信排前）。",
    category="market_structure",
    thesis=(
        "价格反转日的成交量大小决定了反转的可靠性。"
        "缩量反转（价格反转但成交量低）：原趋势动能衰竭，反转可信；"
        "放量反转（价格反转但成交量高）：可能是新的反向力量入场，"
        "也可能是散户恐慌/追涨，反转信号更不确定。"
        "该因子将反转幅度与成交量变化结合："
        "反转幅度大+缩量=高可信度反转（排前）；反转幅度大+放量=可信度存疑。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_price_reversal_volume_confirm(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")

    close = daily_adj["close"]
    vol = daily_adj["vol"]

    # 5-day return (for reversal detection)
    ret_5d = close.groupby(level="Code").transform(
        lambda s: s.pct_change(5)
    )
    # 1-day return (for daily direction)
    ret_1d = close.groupby(level="Code").transform(
        lambda s: s.pct_change(1)
    )

    # Volume change vs 5-day average
    vol_ma5 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    vol_ratio = safe_divide(vol, vol_ma5)

    # Reversal signal: 5d return and 1d return have opposite signs
    reversal = -(ret_5d * ret_1d)  # positive when signs differ
    reversal = reversal.clip(lower=0)  # only consider reversal days

    # Confidence: reversal strength * (1/vol_ratio) = reversal × shrinking volume
    confidence = reversal * safe_divide(1.0, vol_ratio)
    confidence = confidence.clip(0, 5)

    # 20-day average confidence
    avg_confidence = confidence.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    return cross_sectional_rank(avg_confidence)


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Volume Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="volume_acceleration_20",
    description="成交量加速度因子（(MA5_vol - MA20_vol)/MA20_vol截面排名，放量加速排前）。",
    category="market_structure",
    thesis=(
        "成交量加速度（volume的二阶导数）捕捉了交易活跃度的变化速度。"
        "短均线(5日)与长均线(20日)的偏离度量了成交量的趋势强度："
        "正偏离（短期放量）=交易活跃度加速上升，往往伴随价格突破；"
        "负偏离（短期缩量）=交易活跃度下降，可能预示变盘。"
        "该因子在量比因子(volume_ratio)的基础上增加了趋势维度。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_volume_acceleration_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    vol = daily_adj["vol"]

    vol_ma5 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    vol_ma20 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    acceleration = safe_divide(vol_ma5 - vol_ma20, vol_ma20 + 1e-10)
    acceleration = acceleration.clip(-1, 5)

    return cross_sectional_rank(acceleration)


@register_factor(
    name="amount_concentration_20",
    description="成交额集中度因子（20日top5%成交额日占比截面排名，高集中度=资金异动排前）。",
    category="market_structure",
    thesis=(
        "成交额在时间维度上的集中度反映了资金的'脉冲性'。"
        "成交额高度集中在少数几天意味着资金在特定时间点大进大出——"
        "可能是事件驱动的机构交易或游资进出。"
        "成交额均匀分布意味着稳定的交易节奏，更适合中长期持有。"
        "集中度异常上升是'大事发生'的量化信号。"
    ),
    dependencies=("daily_adj.parquet",),
)
def factor_amount_concentration_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    amount = daily_adj["amount"]

    # Ratio of max daily amount to mean amount over 20 days
    amt_max = amount.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    amt_mean = amount.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    concentration = safe_divide(amt_max, amt_mean + 1e-10)
    concentration = concentration.clip(0, 20)

    return cross_sectional_rank(concentration)
