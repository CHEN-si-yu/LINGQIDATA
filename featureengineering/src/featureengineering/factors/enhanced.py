from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── Risk-adjusted momentum ──────────────────────────────────────────────

@register_factor(
    name="mom_20_vol_adj",
    description="波动率调整动量因子，mom_20/volatility_60截面排名。",
    category="enhanced",
    thesis="将动量用波动率标准化后可比较不同波动水平股票的动量质量，高IR动量优于裸动量。",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_20_vol_adj(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_60 = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    ratio = ret_20 / vol_60.replace(0, np.nan)
    return cross_sectional_rank(ratio)


@register_factor(
    name="mom_60_vol_adj",
    description="波动率调整60日动量因子截面排名。",
    category="enhanced",
    thesis="中长期动量经波动率调整后对趋势质量的刻画更精细。",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_60_vol_adj(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_60 = close.groupby(level="Code").transform(lambda s: s.pct_change(60))
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_60 = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    ratio = ret_60 / vol_60.replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Risk-adjusted reversal ──────────────────────────────────────────────

@register_factor(
    name="reversal_5_vol_adj",
    description="波动率调整反转因子，-ret_5/vol_20截面排名。",
    category="enhanced",
    thesis="高波动环境下的反转比低波动环境下的反转更可靠，经波动率调整可提纯反转信号。",
    dependencies=("daily_adj.parquet",),
)
def factor_reversal_5_vol_adj(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_5 = close.groupby(level="Code").transform(lambda s: -s.pct_change(5))
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_20 = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    ratio = ret_5 / vol_20.replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Stability (Sharpe-like) ─────────────────────────────────────────────

@register_factor(
    name="return_stability_60",
    description="收益稳定性因子，60日收益率/60日波动率截面排名（类Sharpe）。",
    category="enhanced",
    thesis="高Sharpe比率股票在风险调整后表现更优，是动量与低波的融合因子。",
    dependencies=("daily_adj.parquet",),
)
def factor_return_stability_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_60 = close.groupby(level="Code").transform(lambda s: s.pct_change(60))
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_60 = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    sharpe_60 = ret_60 / vol_60.replace(0, np.nan)
    return cross_sectional_rank(sharpe_60)


# ── Price-volume coordination ───────────────────────────────────────────

def _rolling_corr_series(daily_ret: pd.Series, vol_chg: pd.Series) -> pd.Series:
    """Compute per-code rolling 20d correlation, returning a Series with the original index.

    Uses covariance decomposition:
        corr(a,b) = (E[ab] - E[a]E[b]) / (std(a) * std(b))
    All components are C-level vectorized rolling ops — no per-window Python loop.
    """
    WINDOW = 20
    MIN_PERIODS = 10

    product = daily_ret * vol_chg
    gp = daily_ret.groupby(level="Code")
    gv = vol_chg.groupby(level="Code")

    mean_ret = gp.transform(lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).mean())
    mean_vol = gv.transform(lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).mean())
    mean_prod = product.groupby(level="Code").transform(
        lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    )
    std_ret = gp.transform(lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).std())
    std_vol = gv.transform(lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).std())

    cov = mean_prod - mean_ret * mean_vol
    denom = std_ret * std_vol
    corr = cov / denom.replace(0, np.nan)
    return corr.clip(-1, 1)


@register_factor(
    name="price_volume_corr_20",
    description="价量相关系数因子，20日收益率与成交量变化率的相关性截面排名。",
    category="enhanced",
    thesis="价量正相关=趋势配合，价量背离=潜在反转，价量一致性是技术面重要的确认指标。",
    dependencies=("daily_adj.parquet",),
)
def factor_price_volume_corr_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    vol = daily_adj["vol"]
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_chg = vol.groupby(level="Code").transform(lambda s: s.pct_change(1))
    corr = _rolling_corr_series(daily_ret, vol_chg)
    return cross_sectional_rank(corr)
