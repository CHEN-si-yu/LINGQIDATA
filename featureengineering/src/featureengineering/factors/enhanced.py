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


# ── Cross-factor composites (value × quality, momentum × chip) ─────────

@register_factor(
    name="bp_x_roe",
    description="价值×质量交互因子，BP排名×ROE排名的等权复合截面排名。",
    category="enhanced",
    thesis="高BP+高ROE是Greenblatt神奇公式的精髓——又好又便宜。单独的高BP可能陷入价值陷阱（坏公司理应便宜），单独的高ROE可能过于昂贵，两者交集才是真正的投资机会。",
    dependencies=("finance.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_bp_x_roe(context: FactorContext):
    finance = context.load("finance.parquet")
    bp = 1.0 / finance["pb"].replace(0, np.nan)
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe"]
    )
    with np.errstate(invalid="ignore"):
        rank_bp = bp.groupby(level="Date").rank(pct=True)
        rank_roe = fin["roe"].groupby(level="Date").rank(pct=True)
    composite = (rank_bp + rank_roe) / 2.0
    return composite.rename("bp_x_roe")


@register_factor(
    name="mom_x_winner",
    description="动量×筹码交互因子，20日动量排名×获利盘比例排名的复合截面排名。",
    category="enhanced",
    thesis="动量策略最大的风险是追高被套——高动量但获利盘比例也高的股票面临获利了结压力。将动量与筹码验证结合：高动量+低获利盘（=筹码仍未充分获利、上涨空间大）才是更可靠的动量信号。",
    dependencies=("daily_adj.parquet", "cyq_perf.parquet"),
)
def factor_mom_x_winner(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    perf = context.load("cyq_perf.parquet")
    mom = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(20)
    )
    winner_rate = -perf["winner_rate"]
    common = mom.index.intersection(winner_rate.index)
    with np.errstate(invalid="ignore"):
        rank_mom = mom.loc[common].groupby(level="Date").rank(pct=True)
        rank_winner = winner_rate.loc[common].groupby(level="Date").rank(pct=True)
    composite = (rank_mom + rank_winner) / 2.0
    return composite.rename("mom_x_winner")


@register_factor(
    name="margin_of_safety",
    description="安全边际因子，BP排名×经营现金流质量排名的等权复合截面排名。",
    category="enhanced",
    thesis="安全边际不仅仅是便宜（低PB），还要有现金利润支撑。高BP+高OCF质量=企业的账面净资产能产生真实现金回报，此时低估值才是真正安全的机会，而非会计意义上的价值陷阱。",
    dependencies=("finance.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_margin_of_safety(context: FactorContext):
    finance = context.load("finance.parquet")
    bp = 1.0 / finance["pb"].replace(0, np.nan)
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_profit"]
    )
    common = bp.index.intersection(fin["ocf_to_profit"].index)
    with np.errstate(invalid="ignore"):
        rank_bp = bp.loc[common].groupby(level="Date").rank(pct=True)
        rank_ocf = fin["ocf_to_profit"].loc[common].groupby(level="Date").rank(pct=True)
    composite = (rank_bp + rank_ocf) / 2.0
    return composite.rename("margin_of_safety")


# ── Overnight × intraday separation ─────────────────────────────────────

@register_factor(
    name="overnight_intraday_divergence",
    description="隔夜-日内背离因子，20日隔夜累计收益-20日日内累计收益截面排名。",
    category="enhanced",
    thesis="隔夜收益反映机构主导的隔夜信息定价，日内收益反映散户主导的盘中交易。隔夜-日内背离度大（隔夜强/日内弱）代表机构吸筹而散户抛售，是聪明钱vs散户的分化信号。",
    dependencies=("daily.parquet",),
)
def factor_overnight_intraday_divergence(context: FactorContext):
    daily = context.load("daily.parquet")
    overnight_ret = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
    intraday_ret = (daily["close"] - daily["open"]) / daily["open"].replace(0, np.nan)
    cum_overnight = overnight_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    cum_intraday = intraday_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    divergence = cum_overnight - cum_intraday
    return cross_sectional_rank(divergence)


# ── GARP (Growth at Reasonable Price) ──────────────────────────────────


@register_factor(
    name="garp_composite",
    description="合理价格增长因子，(rank(净利润增速)+rank(1/PE))/2截面排名。",
    category="enhanced",
    thesis="GARP策略寻找以合理价格增长的公司——避免纯粹高增长的高估值陷阱，也避免纯粹低估值无增长的价值陷阱。净利润增速与低PE的交集是机构选股的核心框架，在A股中该交集因子比单独的增长或估值因子更稳健。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_garp_composite(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["netprofit_yoy"],
    )
    fin_panel = context.load("finance.parquet")
    pe_recip = 1.0 / fin_panel["pe_ttm"].replace(0, np.nan)
    common = fin["netprofit_yoy"].index.intersection(pe_recip.index)
    with np.errstate(invalid="ignore"):
        rank_growth = fin["netprofit_yoy"].loc[common].groupby(level="Date").rank(pct=True)
        rank_pe_inv = pe_recip.loc[common].groupby(level="Date").rank(pct=True)
    composite = (rank_growth + rank_pe_inv) / 2.0
    return composite.rename("garp_composite")


# ── Quality × Low volatility ───────────────────────────────────────────


@register_factor(
    name="quality_volatility_composite",
    description="质量×低波复合因子，(rank(ROE)+rank(-vol_60))/2截面排名。",
    category="enhanced",
    thesis="高质量+低波动是'防御性增长'主题的核心——高ROE企业的盈利稳定性天然伴随更低的股价波动。两者交集捕捉的是经营质量高且不被市场过度交易、定价有效的好公司。在A股中质量和低波的联合信号比单独信号更不易被套利。",
    dependencies=("financial_indicator.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_quality_volatility_composite(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["roe"],
    )
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    low_vol = -ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    common = fin["roe"].index.intersection(low_vol.index)
    with np.errstate(invalid="ignore"):
        rank_q = fin["roe"].loc[common].groupby(level="Date").rank(pct=True)
        rank_v = low_vol.loc[common].groupby(level="Date").rank(pct=True)
    composite = (rank_q + rank_v) / 2.0
    return composite.rename("quality_volatility_composite")


# ── Acceleration × Value ───────────────────────────────────────────────


@register_factor(
    name="acceleration_value_composite",
    description="盈利加速度×价值复合因子，(rank(ROE动量)+rank(BP))/2截面排名。",
    category="enhanced",
    thesis="寻找'正在改善的便宜公司'——ROE在加速提升且估值仍低（高BP）。这是GARP的动态版本：不仅看增长水平，更关注增长方向。盈利改善中的低估值公司享受基本面拐点和估值修复的双重驱动力。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_acceleration_value_composite(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["roe"],
    )
    roe_mom = fin["roe"].groupby(level="Code").transform(lambda s: s.diff(4))
    fin_panel = context.load("finance.parquet")
    bp = 1.0 / fin_panel["pb"].replace(0, np.nan)
    common = roe_mom.index.intersection(bp.index)
    with np.errstate(invalid="ignore"):
        rank_roe_mom = roe_mom.loc[common].groupby(level="Date").rank(pct=True)
        rank_bp = bp.loc[common].groupby(level="Date").rank(pct=True)
    composite = (rank_roe_mom + rank_bp) / 2.0
    return composite.rename("acceleration_value_composite")


# ── Momentum × Earnings stability ──────────────────────────────────────


@register_factor(
    name="momentum_quality_trend",
    description="动量质量趋势因子，(rank(mom_60_vol_adj)+rank(eps_stability_8q))/2截面排名。使用eps（非q_eps）的8期滚动变异系数度量盈利稳定性。",
    category="enhanced",
    thesis="由高盈利稳定性支撑的上涨趋势比单纯的动量更可靠——稳定盈利公司享受更低的投资者分歧和更集中的买方共识。动量质量趋势捕捉的是'好公司正在上涨'而非'任何东西在上涨'，对动量崩盘风险有更好的抵御力。",
    dependencies=("daily_adj.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_momentum_quality_trend(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_60 = close.groupby(level="Code").transform(lambda s: s.pct_change(60))
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_60 = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    mom_vol_adj = ret_60 / vol_60.replace(0, np.nan)

    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["eps"],
    )
    eps = fin["eps"]
    eps_std_8 = eps.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    eps_mean_8 = eps.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    stability = -eps_std_8 / eps_mean_8.replace(0, np.nan)

    common = mom_vol_adj.index.intersection(stability.index)
    with np.errstate(invalid="ignore"):
        rank_mom = mom_vol_adj.loc[common].groupby(level="Date").rank(pct=True)
        rank_stab = stability.loc[common].groupby(level="Date").rank(pct=True)
    composite = (rank_mom + rank_stab) / 2.0
    return composite.rename("momentum_quality_trend")


# ── Winner rate × reversal divergence ───────────────────────────────────


@register_factor(
    name="winner_rate_reversal_divergence",
    description="获利盘-反转背离因子，(rank(-winner_rate)+rank(reversal_5))/2截面排名。",
    category="enhanced",
    thesis="当获利盘比例高但近期价格下跌时（高reversal信号+高winner_rate=筹码获利+价格下行），往往是'恐慌性获利了结'而非趋势反转——有盈利的投资者在下跌中加速抛出获利头寸。这种背离往往构成短期底部信号，与纯技术反转互补。",
    dependencies=("cyq_perf.parquet", "daily_adj.parquet"),
)
def factor_winner_rate_reversal_divergence(context: FactorContext):
    perf = context.load("cyq_perf.parquet")
    daily_adj = context.load("daily_adj.parquet")
    winner = -perf["winner_rate"]  # low winner rate = good (opposite of normal)
    reversal = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: -s.pct_change(5)  # -ret_5 = reversal (high = big drop)
    )
    common = winner.index.intersection(reversal.index)
    with np.errstate(invalid="ignore"):
        rank_w = winner.loc[common].groupby(level="Date").rank(pct=True)
        rank_r = reversal.loc[common].groupby(level="Date").rank(pct=True)
    composite = (rank_w + rank_r) / 2.0
    return composite.rename("winner_rate_reversal_divergence")
