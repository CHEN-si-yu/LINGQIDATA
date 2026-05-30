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


# ── 行业中性化扩展：event/fund_flow 类 ─────────────────────────────────────

@register_factor(
    name="winner_rate_neutral",
    description="行业中性化获利盘比例因子，行业内winner_rate截面排名后再截面排名（取负向）。",
    category="neutral",
    thesis="获利盘比例在不同行业间有系统性差异（周期股vs成长股），行业中性化后更可比。",
    dependencies=("cyq_perf.parquet", "stock_list.parquet"),
)
def factor_winner_rate_neutral(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    neutral = _industry_neutral_rank(-perf["winner_rate"], context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="mf_net_inflow_ratio_neutral",
    description="行业中性化主力净流入率因子，行业内mf_net_inflow_ratio截面排名后再截面排名。",
    category="neutral",
    thesis="主力资金行为在不同行业间结构差异明显（大盘vs小盘行业），行业中性化后提取行业内相对资金流向信号。",
    dependencies=("main_fund_flow.parquet", "stock_list.parquet"),
)
def factor_mf_net_inflow_ratio_neutral(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_amount = (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)
    ratio = ff["net_mf_amount"] / total_amount
    neutral = _industry_neutral_rank(ratio, context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="chip_concentration_neutral",
    description="行业中性化筹码集中度因子，行业内chip_concentration截面排名后再截面排名（取负向=窄区间排前）。",
    category="neutral",
    thesis="筹码集中度在不同市值的行业间差异显著，行业中性化后识别行业内筹码结构更优的股票。",
    dependencies=("cyq_perf.parquet", "stock_list.parquet"),
)
def factor_chip_concentration_neutral(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    spread = (perf["cost_95pct"] - perf["cost_5pct"]) / perf["cost_50pct"].replace(0, np.nan)
    neutral = _industry_neutral_rank(-spread, context)
    return cross_sectional_rank(neutral)


# ── 规模中性化因子 ───────────────────────────────────────────────────────

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
    name="roe_size_neutral",
    description="规模中性化ROE因子，市值分桶内ROE截面排名。",
    category="neutral",
    thesis="大市值公司ROE通常更高更稳定，规模中性化后可以找到小市值中ROE突出的'隐形冠军'。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_roe_size_neutral(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["roe"])
    neutral = _size_neutral_rank(fin["roe"], context)
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


@register_factor(
    name="fcf_yield_size_neutral",
    description="规模中性化自由现金流收益率因子，市值分桶内截面排名。",
    category="neutral",
    thesis="大市值公司FCF通常更高，规模中性化后找到同市值级别中现金回报更优的公司。",
    dependencies=("cashflow.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_fcf_yield_size_neutral(context: FactorContext):
    cf = context.load_financial("cashflow.parquet", value_cols=["free_cashflow"])
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"].where(finance["total_mv"] > 0, np.nan)
    merged = pd.concat([cf["free_cashflow"], total_mv], axis=1)
    fcf_yield = merged["free_cashflow"] / merged["total_mv"].replace(0, np.nan)
    neutral = _size_neutral_rank(fcf_yield, context)
    return cross_sectional_rank(neutral)


# ── Supplementary neutral factors ──────────────────────────────────────────


@register_factor(
    name="mom_60_neutral",
    description="行业中性化60日动量因子。",
    category="neutral",
    thesis="去除行业beta后的动量更纯净，跨行业比较时不受行业轮动干扰",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_mom_60_neutral(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    close = daily_adj["close"]
    ret_60 = close.groupby(level="Code").transform(lambda s: s.pct_change(60))
    codes = ret_60.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"ret_60": ret_60.values, "industry": industries.values}, index=ret_60.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["ret_60"].transform("median")
    neutral = df["ret_60"] - df["sector_median"]
    return cross_sectional_rank(neutral)


@register_factor(
    name="volatility_20_industry_neutral",
    description="行业中性化20日波动率因子 (低波排前, 负向)。",
    category="neutral",
    thesis="行业中性化后的波动率衡量个股的异质波动，更适合跨行业优选",
    dependencies=("daily_adj.parquet", "stock_list.parquet"),
)
def factor_volatility_20_neutral(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    industry_map = context.repo.load_industry_map()
    ret = daily_adj["close"].groupby(level="Code").pct_change()
    vol = ret.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
    codes = vol.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"vol": vol.values, "industry": industries.values}, index=vol.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["vol"].transform("median")
    neutral = df["vol"] / df["sector_median"].replace(0, np.nan) - 1.0
    return cross_sectional_rank(-neutral)


@register_factor(
    name="turnover_20_industry_neutral",
    description="行业中性化20日换手率因子 (低换手排前, 负向)。",
    category="neutral",
    thesis="行业内相对换手率过滤行业轮动效应，纯粹的流动性偏好信号",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_turnover_20_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    industry_map = context.repo.load_industry_map()
    turnover = finance["turnover_rate"]
    to_ma20 = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    codes = to_ma20.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"to": to_ma20.values, "industry": industries.values}, index=to_ma20.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["to"].transform("median")
    neutral = df["to"] / df["sector_median"].replace(0, np.nan) - 1.0
    return cross_sectional_rank(-neutral)


@register_factor(
    name="bp_industry_neutral",
    description="行业中性化BP因子。",
    category="neutral",
    thesis="行业中性BP消除行业间估值差异，适用于跨行业价值选股",
    dependencies=("finance.parquet", "stock_list.parquet"),
)
def factor_bp_industry_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    industry_map = context.repo.load_industry_map()
    bp = 1.0 / finance["pb"].replace(0, np.nan)
    codes = bp.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"bp": bp.values, "industry": industries.values}, index=bp.index)
    df = df.dropna(subset=["industry"])
    df["sector_median"] = df.groupby(["Date", "industry"])["bp"].transform("median")
    neutral = df["bp"] / df["sector_median"].replace(0, np.nan) - 1.0
    return cross_sectional_rank(neutral)
