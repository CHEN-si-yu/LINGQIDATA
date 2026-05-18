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


# ── 行业中性化扩展：quality 类 ─────────────────────────────────────────────

@register_factor(
    name="roe_neutral",
    description="行业中性化ROE因子，行业内ROE截面排名后再截面排名。",
    category="neutral",
    thesis="ROE受行业杠杆水平和盈利模式影响显著（金融vs科技），行业中性化后可以提取行业内相对盈利质量信号，更纯粹地反映公司层面的经营差异。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_roe_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe"]
    )
    neutral = _industry_neutral_rank(fin["roe"], context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="roa_neutral",
    description="行业中性化ROA因子，行业内ROA截面排名后再截面排名。",
    category="neutral",
    thesis="ROA消除了杠杆的影响，但在不同资产密集度的行业间仍有系统性差异，行业中性化后信号更可比。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_roa_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roa"]
    )
    neutral = _industry_neutral_rank(fin["roa"], context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="gross_margin_neutral",
    description="行业中性化毛利率因子，行业内毛利率截面排名后再截面排名。",
    category="neutral",
    thesis="毛利率在不同行业间天然差异巨大（软件vs零售），行业中性化后才能识别出行业内定价权更强的公司。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_gross_margin_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["gross_margin"]
    )
    neutral = _industry_neutral_rank(fin["gross_margin"], context)
    return cross_sectional_rank(neutral)


# ── 行业中性化扩展：valuation 类 ────────────────────────────────────────────

@register_factor(
    name="sp_ttm_neutral",
    description="行业中性化市销率倒数因子，行业内1/PS_TTM截面排名后再截面排名。",
    category="neutral",
    thesis="市销率在不同行业（高毛利vs低毛利）差异明显，行业中性化消除行业估值中枢差异后，提取行业内相对便宜/昂贵的信号。",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_sp_ttm_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    sp_ttm = 1.0 / finance["ps_ttm"].replace(0, np.nan)
    neutral = _industry_neutral_rank(sp_ttm, context)
    return cross_sectional_rank(neutral)


# ── 行业中性化扩展：financial 类 ───────────────────────────────────────────

@register_factor(
    name="fcf_yield_neutral",
    description="行业中性化自由现金流收益率因子，行业内FCF/总市值截面排名后再截面排名。",
    category="neutral",
    thesis="FCF收益率受行业资本开支周期影响（重资产vs轻资产），行业中性化后可对比同行业内现金回报能力。",
    dependencies=("cashflow.parquet", "finance.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_fcf_yield_neutral(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["free_cashflow"]
    )
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"].where(finance["total_mv"] > 0, np.nan)
    merged = pd.concat([cf["free_cashflow"], total_mv], axis=1)
    fcf_yield = merged["free_cashflow"] / merged["total_mv"].replace(0, np.nan)
    neutral = _industry_neutral_rank(fcf_yield, context)
    return cross_sectional_rank(neutral)


# ── 行业中性化扩展：valuation/price 类 ─────────────────────────────────────

@register_factor(
    name="turnover_20_neutral",
    description="行业中性化换手率因子，行业内20日均换手率截面排名后再截面排名。",
    category="neutral",
    thesis="换手率在不同行业和市值规模间差异巨大（小盘科技vs大盘银行），行业中性化后识别行业内换手率异常的股票，低换手往往对应筹码稳定。",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_turnover_20_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    turnover_20_mean = rolling_group_mean(turnover, 20)
    neutral = _industry_neutral_rank(turnover_20_mean, context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="volatility_20_neutral",
    description="行业中性化波动率因子，行业内20日波动率截面排名后再截面排名。",
    category="neutral",
    thesis="波动率在不同行业间有结构性差异（周期股vs公用事业），行业中性化后提取行业内相对低波的信号，低波动在行业内也具有超额收益潜力。",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_volatility_20_neutral(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_1d = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_20 = rolling_group_std(ret_1d, 20)
    neutral = _industry_neutral_rank(vol_20, context)
    return cross_sectional_rank(-neutral)
