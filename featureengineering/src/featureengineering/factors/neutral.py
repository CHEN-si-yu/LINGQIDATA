from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std
from .enhanced import _rolling_corr_series


def _industry_neutral_rank(signal: pd.Series, context: FactorContext) -> pd.Series:
    """Within-industry cross-sectional percentile rank.

    signal: Series with (Date, Code) MultiIndex, raw factor values
    Returns a Series with the same index, values are within-industry percentile ranks.
    """
    industry_map = context.repo.load_industry_map()
    codes = signal.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame(
        {"signal": signal.values, "industry": industries.values},
        index=signal.index,
    )
    df = df.dropna(subset=["industry"])
    with np.errstate(invalid="ignore"):
        df["rank"] = df.groupby(["Date", "industry"])["signal"].rank(pct=True)
    return df["rank"]


# ── BP neutral ────────────────────────────────────────────────────────────

def _size_neutral_rank(signal: pd.Series, context: FactorContext, n_buckets: int = 10) -> pd.Series:
    """Within-size-bucket cross-sectional percentile rank.

    signal: Series with (Date, Code) MultiIndex, raw factor values.
    Stocks are bucketed by market cap within each date, then ranked within each bucket.
    """
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"]
    common = signal.index.intersection(total_mv.index)
    signal = signal.loc[common]
    total_mv = total_mv.loc[common]

    df = pd.DataFrame({
        "signal": signal.values,
        "total_mv": total_mv.values,
    }, index=signal.index)

    def _bucket_rank(grp):
        if len(grp) < n_buckets * 2:
            grp["bucket"] = 0
        else:
            grp["bucket"] = pd.qcut(grp["total_mv"], n_buckets, labels=False, duplicates="drop")
        grp["rank"] = grp.groupby("bucket")["signal"].rank(pct=True)
        return grp["rank"]

    df = df.groupby(level="Date", group_keys=False).apply(_bucket_rank)
    return df


@register_factor(
    name="bp_size_neutral",
    description="规模中性化BP因子，市值分桶内截面排名。消除市值与估值相关性。",
    category="neutral",
    thesis="BP与市值存在系统性相关性（大盘股普遍PB较低），规模中性化后提取更纯粹的估值信号，避免选到全是银行股的'价值陷阱'。",
    dependencies=("finance.parquet",),
)
def factor_bp_size_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    bp = 1.0 / finance["pb"].replace(0, np.nan)
    neutral = _size_neutral_rank(bp, context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="mom_20_size_neutral",
    description="规模中性化20日动量因子，市值分桶内截面排名。",
    category="neutral",
    thesis="小盘股动量往往强于大盘股，规模中性化后可提取同等市值级别中的相对动量强度。",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def factor_mom_20_size_neutral(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    mom = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    neutral = _size_neutral_rank(mom, context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="volatility_20_size_neutral",
    description="规模中性化波动率因子，市值分桶内低波排名。",
    category="neutral",
    thesis="小盘股波动率天然高于大盘股，规模中性化后可对比同市值级别内的低波溢价。",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def factor_volatility_20_size_neutral(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_1d = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_20 = ret_1d.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    neutral = _size_neutral_rank(-vol_20, context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="turnover_20_size_neutral",
    description="规模中性化换手率因子，市值分桶内低换手排名。",
    category="neutral",
    thesis="小盘股换手率天然高，规模中性化后找到同市值级别中换手率偏低（筹码稳定）的股票。",
    dependencies=("finance.parquet",),
)
def factor_turnover_20_size_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    to_20 = rolling_group_mean(turnover, 20)
    neutral = _size_neutral_rank(-to_20, context)
    return cross_sectional_rank(neutral)


