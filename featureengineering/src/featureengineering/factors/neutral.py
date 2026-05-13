from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank
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
    df["rank"] = df.groupby(["Date", "industry"])["signal"].rank(pct=True)
    return df["rank"]


# ── BP neutral ────────────────────────────────────────────────────────────

@register_factor(
    name="bp_neutral",
    description="行业中性化账面市值比因子，行业内截面排名后的BP。",
    category="neutral",
    thesis="行业中性化可剔除BP因子中行业间估值差异的干扰，提取行业内相对价值信号。",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_bp_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    bp = 1.0 / finance["pb"].replace(0, np.nan)
    neutral = _industry_neutral_rank(bp, context)
    return cross_sectional_rank(neutral)


# ── EP neutral ────────────────────────────────────────────────────────────

@register_factor(
    name="ep_ttm_neutral",
    description="行业中性化盈利市值比因子，行业内截面排名后的EP_TTM。",
    category="neutral",
    thesis="EP因子在不同行业间天然水平差异大，行业中性化后可提取行业内相对盈利能力信号。",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_ep_ttm_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    ep = 1.0 / finance["pe_ttm"].replace(0, np.nan)
    neutral = _industry_neutral_rank(ep, context)
    return cross_sectional_rank(neutral)


# ── Momentum neutral ──────────────────────────────────────────────────────

@register_factor(
    name="mom_20_neutral",
    description="行业中性化20日动量因子，行业内截面排名后的mom_20。",
    category="neutral",
    thesis="动量在不同行业间差异明显（如行业轮动），行业中性化后可提取个股层面的相对动量。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_mom_20_neutral(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    mom = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(20)
    )
    neutral = _industry_neutral_rank(mom, context)
    return cross_sectional_rank(neutral)


# ── Price-volume correlation neutral ──────────────────────────────────────

@register_factor(
    name="price_volume_corr_20_neutral",
    description="行业中性化价量相关系数因子。",
    category="neutral",
    thesis="价量相关性在不同行业间存在结构性差异，行业中性化后信号更纯净。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_price_volume_corr_20_neutral(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    vol = daily_adj["vol"]
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_chg = vol.groupby(level="Code").transform(lambda s: s.pct_change(1))
    corr = _rolling_corr_series(daily_ret, vol_chg)
    neutral = _industry_neutral_rank(corr, context)
    return cross_sectional_rank(neutral)


# ── Limit-up consecutive neutral ──────────────────────────────────────────

@register_factor(
    name="limit_up_consecutive_neutral",
    description="行业中性化连板天数因子，行业内截面排名后的连板天数。",
    category="neutral",
    thesis="不同行业的连板概率和高度差异巨大（如科技vs银行），行业中性化后连板信号更可比。",
    dependencies=("limit_up.parquet", "stock_list.parquet"),
)
def factor_limit_up_consecutive_neutral(context: FactorContext):
    limit_up = context.load("limit_up.parquet")
    raw = limit_up["consecutive_days"].astype(float)
    neutral = _industry_neutral_rank(raw, context)
    return cross_sectional_rank(neutral)
