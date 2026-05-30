from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


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
# ── Residual momentum ──────────────────────────────────────────────────────

@register_factor(
    name="residual_momentum_20",
    description="残差动量因子，20日收益率对市场等权收益回归取残差截面排名。",
    category="enhanced",
    thesis="剔除市场Beta后的残差动量是更纯粹的个股alpha信号——在牛市中不被市场上涨掩盖，在熊市中不被系统性下跌拖累。与mom_20互补：残差动量过滤了系统性风险暴露。",
    dependencies=("daily_adj.parquet",),
)
def factor_residual_momentum_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    ret_1 = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    ret_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))

    ret_1_panel = ret_1.unstack("Code")
    ret_20_panel = ret_20.unstack("Code")
    mkt_ret = ret_1_panel.mean(axis=1)

    cov = ret_1_panel.rolling(60, min_periods=30).cov(mkt_ret, pairwise=False)
    var = mkt_ret.rolling(60, min_periods=30).var()
    beta_panel = cov.div(var, axis=0)

    mkt_panel = pd.DataFrame(
        {c: mkt_ret for c in ret_20_panel.columns}, index=ret_20_panel.index
    )
    residual_panel = ret_20_panel - beta_panel * mkt_panel
    residual = residual_panel.stack().reorder_levels(["Date", "Code"]).sort_index()
    return cross_sectional_rank(residual)


# ── Intraday × overnight combo ─────────────────────────────────────────────

@register_factor(
    name="intraday_overnight_combo",
    description="日内隔夜综合因子，intraday_reversal的截面排名 + overnight_gap的截面排名的等权均值。",
    category="enhanced",
    thesis="日内反转和隔夜缺口是A股两个独立的短期alpha来源——日内反转捕捉高频交易者行为偏差，隔夜缺口反映机构盘后信息优势。两者等权结合比单独使用稳定性更强。",
    dependencies=("daily_adj.parquet",),
)
def factor_intraday_overnight_combo(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    open_price = daily_adj["open"]

    # Intraday return = close/open - 1 (approximate intraday reversal signal)
    denom = open_price.replace(0, np.nan)
    intraday_ret = close / denom - 1.0
    intraday_rank = intraday_ret.groupby(level="Date").rank(pct=True)

    # Overnight gap = open/prev_close - 1
    prev_close = close.groupby(level="Code").shift(1)
    overnight_gap = open_price / prev_close.replace(0, np.nan) - 1.0
    overnight_rank = overnight_gap.groupby(level="Date").rank(pct=True)

    combo = (intraday_rank + overnight_rank) / 2.0
    return cross_sectional_rank(combo)


# ── Supplementary enhanced / composite factors ────────────────────────────


@register_factor(
    name="mom_20_sharpe",
    description="20日动量夏普比因子 (动量/波动率) 截面排名。",
    category="enhanced",
    thesis="风险调整后的动量比原始动量更干净，去除高波动噪音，关注真正稳定的趋势",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_20_sharpe(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    mom = rolling_group_mean(ret, 20) * 252
    vol = rolling_group_std(ret, 20) * np.sqrt(252)
    sharpe = safe_divide(mom, vol + 1e-8)
    return cross_sectional_rank(sharpe)


@register_factor(
    name="mom_60_sharpe",
    description="60日动量夏普比因子 (动量/波动率) 截面排名。",
    category="enhanced",
    thesis="中期动量的风险调整版本，60日窗口的夏普比更稳定",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_60_sharpe(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    mom = rolling_group_mean(ret, 60) * 252
    vol = rolling_group_std(ret, 60) * np.sqrt(252)
    sharpe = safe_divide(mom, vol + 1e-8)
    return cross_sectional_rank(sharpe)


@register_factor(
    name="bp_x_mom_20",
    description="BP x 20日动量交互因子 (价值+趋势双排名均值)。",
    category="enhanced",
    thesis="低估值+近期上涨=基本面反转确认，价值与趋势的互补性很强",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def factor_bp_x_mom_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    finance = context.load("finance.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    pb = finance["pb"]
    bp = safe_divide(1.0, pb)
    common = ret.index.intersection(bp.index)
    bp_rank = bp.loc[common].groupby(level="Date").rank(pct=True)
    mom_rank = ret.loc[common].groupby(level="Date").rank(pct=True)
    combo = (bp_rank + mom_rank) / 2.0
    return cross_sectional_rank(combo)


@register_factor(
    name="quality_x_value",
    description="质量x价值综合因子 (ROE排名 x BP排名) 截面排名。",
    category="enhanced",
    thesis="高ROE+低PB是最经典的价值投资框架，GARP的极端形式",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_quality_x_value(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["roe"])
    finance = context.load("finance.parquet")
    roe = fin["roe"]
    pb = finance["pb"]
    bp = safe_divide(1.0, pb)
    common = roe.index.intersection(bp.index)
    roe_rank = roe.loc[common].groupby(level="Date").rank(pct=True)
    bp_rank = bp.loc[common].groupby(level="Date").rank(pct=True)
    combo = (roe_rank + bp_rank) / 2.0
    return cross_sectional_rank(combo)


@register_factor(
    name="trend_quality_composite",
    description="趋势质量复合因子 (动量+低波+量一致性三排名均值)。",
    category="enhanced",
    thesis="动量、低波、量价一致性三个维度结合，识别健康趋势股",
    dependencies=("daily_adj.parquet",),
)
def factor_trend_quality_composite(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    vol = daily_adj["vol"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    mom = rolling_group_mean(ret, 20)
    mom_rank = mom.groupby(level="Date").rank(pct=True)

    vol_20 = rolling_group_std(ret, 20)
    vol_rank = (-vol_20).groupby(level="Date").rank(pct=True)

    vol_chg = vol.groupby(level="Code").pct_change()
    consistent = ((ret > 0) & (vol_chg > 0)) | ((ret < 0) & (vol_chg < 0))
    consist_ratio = consistent.astype(float).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    consist_rank = consist_ratio.groupby(level="Date").rank(pct=True)

    combo = (mom_rank + vol_rank + consist_rank) / 3.0
    return cross_sectional_rank(combo)


@register_factor(
    name="low_risk_composite",
    description="低风险综合因子 (低波+低回撤双排名均值)。",
    category="enhanced",
    thesis="结合低波动和低回撤两个风险管理维度，构建稳健的低风险组合",
    dependencies=("daily_adj.parquet",),
)
def factor_low_risk_composite(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))

    vol_20 = rolling_group_std(ret, 20)
    vol_rank = (-vol_20).groupby(level="Date").rank(pct=True)

    cummax = close.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    )
    dd = (close - cummax) / cummax
    max_dd = dd.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).min()
    )
    dd_rank = max_dd.groupby(level="Date").rank(pct=True)

    combo = (vol_rank + dd_rank) / 2.0
    return cross_sectional_rank(combo)


@register_factor(
    name="sentiment_momentum_composite",
    description="情绪动量复合因子 (动量+量趋势+换手趋势三排名均值)。",
    category="enhanced",
    thesis="结合价格动量、成交量趋势和换手率变化的市场情绪综合度量",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def factor_sentiment_momentum_composite(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    finance = context.load("finance.parquet")
    close = daily_adj["close"]
    vol = daily_adj["vol"]
    turnover = finance["turnover_rate"]

    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    mom = rolling_group_mean(ret, 20)
    mom_rank = mom.groupby(level="Date").rank(pct=True)

    vol_ma_5 = rolling_group_mean(vol, 5)
    vol_ma_20 = rolling_group_mean(vol, 20)
    vol_trend = safe_divide(vol_ma_5 - vol_ma_20, vol_ma_20 + 1e-8)
    vol_rank = vol_trend.groupby(level="Date").rank(pct=True)

    to_ma_5 = rolling_group_mean(turnover, 5)
    to_ma_20 = rolling_group_mean(turnover, 20)
    to_trend = safe_divide(to_ma_5 - to_ma_20, to_ma_20 + 1e-8)
    to_rank = to_trend.groupby(level="Date").rank(pct=True)

    combo = (mom_rank + vol_rank + to_rank) / 3.0
    return cross_sectional_rank(combo)


# ── Value-Momentum composites ─────────────────────────────────────────────

@register_factor(
    name="bp_x_roe",
    description="价值×质量交互因子，bp截面排名×roe截面排名（高=低估+高质量排前）。",
    category="enhanced",
    thesis="BP×ROE是经典的价值+质量交叉——避免'价值陷阱'（低BP但ROE差）和'质量溢价过高'（高ROE但估值太贵），同时满足便宜+优质的条件。",
    dependencies=("__factors__", "bp", "roe"),
)
def factor_bp_x_roe(context: FactorContext):
    bp = context.load_factor("bp")
    roe = context.load_factor("roe")
    common = bp.index.intersection(roe.index)
    bp_r = bp.loc[common].groupby(level="Date").rank(pct=True)
    roe_r = roe.loc[common].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(bp_r * roe_r)


@register_factor(
    name="mom_x_winner",
    description="动量×筹码验证因子，mom_20截面排名×winner_rate截面排名（有筹码支撑的动量更可靠排前）。",
    category="enhanced",
    thesis="动量效应与筹码结构交叉验证——有获利盘支撑的动量趋势比纯价格动量更可靠，减少了追高被套的风险。",
    dependencies=("__factors__", "mom_20", "winner_rate"),
)
def factor_mom_x_winner(context: FactorContext):
    mom = context.load_factor("mom_20")
    wr = context.load_factor("winner_rate")
    common = mom.index.intersection(wr.index)
    mom_r = mom.loc[common].groupby(level="Date").rank(pct=True)
    wr_r = wr.loc[common].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(mom_r * wr_r)


@register_factor(
    name="margin_of_safety",
    description="安全边际因子，bp排名×(1-资产负债率排名)截面排名。",
    category="enhanced",
    thesis="BP(低估值)×低杠杆=安全边际——既便宜又财务稳健的公司在市场下行时有双重保护：估值支撑+财务不爆雷。",
    dependencies=("__factors__", "bp", "debt_to_assets"),
)
def factor_margin_of_safety(context: FactorContext):
    bp = context.load_factor("bp")
    debt = context.load_factor("debt_to_assets")
    common = bp.index.intersection(debt.index)
    bp_r = bp.loc[common].groupby(level="Date").rank(pct=True)
    debt_r = debt.loc[common].groupby(level="Date").rank(pct=True)
    safety = bp_r * (1 - debt_r)
    return cross_sectional_rank(safety)


@register_factor(
    name="quality_momentum_composite",
    description="质量动量综合因子，roe_momentum_4q排名×mom_20排名截面排名。",
    category="enhanced",
    thesis="盈利改善+价格动量同时发生=基本面和技术面共振——是最可靠的趋势信号，盈利驱动+价格验证。",
    dependencies=("__factors__", "roe_momentum_4q", "mom_20"),
)
def factor_quality_momentum_composite(context: FactorContext):
    roe_mom = context.load_factor("roe_momentum_4q")
    mom = context.load_factor("mom_20")
    common = roe_mom.index.intersection(mom.index)
    roe_r = roe_mom.loc[common].groupby(level="Date").rank(pct=True)
    mom_r = mom.loc[common].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(roe_r * mom_r)


@register_factor(
    name="value_quality_volatility",
    description="价值质量低波综合因子，(bp+roe-振幅)三维度截面排名。",
    category="enhanced",
    thesis="低估值+高质量+低波动的三维交叉——低估值提供安全边际、高质量提供盈利保障、低波动提供持有体验，是长线资金最偏好的组合特征。",
    dependencies=("__factors__", "bp", "roe", "volatility_20"),
)
def factor_value_quality_volatility(context: FactorContext):
    bp = context.load_factor("bp")
    roe = context.load_factor("roe")
    vol = context.load_factor("volatility_20")
    common = bp.index.intersection(roe.index).intersection(vol.index)
    bp_r = bp.loc[common].groupby(level="Date").rank(pct=True)
    roe_r = roe.loc[common].groupby(level="Date").rank(pct=True)
    vol_r = vol.loc[common].groupby(level="Date").rank(pct=True)
    composite = bp_r + roe_r + (1 - vol_r)
    return cross_sectional_rank(composite)


@register_factor(
    name="growth_at_reasonable_price",
    description="GARP因子，or_yoy排名/(1+bp排名)截面排名（成长性/估值=性价比排前）。",
    category="enhanced",
    thesis="Growth At Reasonable Price(GARP)策略——寻找成长性好但估值合理(不过贵)的股票，是成长投资和价值投资的交汇点。",
    dependencies=("__factors__", "bp", "or_yoy"),
)
def factor_growth_at_reasonable_price(context: FactorContext):
    bp = context.load_factor("bp")
    growth = context.load_factor("or_yoy")
    common = bp.index.intersection(growth.index)
    bp_r = bp.loc[common].groupby(level="Date").rank(pct=True)
    growth_r = growth.loc[common].groupby(level="Date").rank(pct=True)
    garp = growth_r / (1 + bp_r)
    return cross_sectional_rank(garp)


@register_factor(
    name="earnings_momentum_quality",
    description="盈利动量质量因子，ROE动量×OCF/利润排名截面排名（盈利改善+现金流确认排前）。",
    category="enhanced",
    thesis="盈利改善但需要现金流的确认——ROE动量提升+高现金流质量的组合过滤了'纸面利润改善'，确保盈利改善是真正有现金支撑的。",
    dependencies=("__factors__", "roe_momentum_4q", "ocf_to_profit"),
)
def factor_earnings_momentum_quality(context: FactorContext):
    roe_mom = context.load_factor("roe_momentum_4q")
    ocf_q = context.load_factor("ocf_to_profit")
    common = roe_mom.index.intersection(ocf_q.index)
    roe_r = roe_mom.loc[common].groupby(level="Date").rank(pct=True)
    ocf_r = ocf_q.loc[common].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(roe_r * ocf_r)


@register_factor(
    name="reversal_with_volume_confirmation",
    description="放量反转确认因子，-mom_5排名×volume_ratio排名截面排名（放量反转=信号确认排前）。",
    category="enhanced",
    thesis="反转信号的可靠性需要成交量验证——缩量反转可能是噪音，放量反转(尤其是底部放量反弹)是趋势转变的强力确认。",
    dependencies=("__factors__", "mom_5", "volume_ratio"),
)
def factor_reversal_with_volume_confirmation(context: FactorContext):
    mom5 = context.load_factor("mom_5")
    vol_ratio = context.load_factor("volume_ratio")
    common = mom5.index.intersection(vol_ratio.index)
    reversal = (1 - mom5.loc[common].groupby(level="Date").rank(pct=True))
    vol_r = vol_ratio.loc[common].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(reversal * vol_r)


@register_factor(
    name="quality_turnover_divergence",
    description="质量换手背离因子，roe排名×(1-turnover_20排名)截面排名（高质量+低换手=筹码锁定排前）。",
    category="enhanced",
    thesis="高质量股票+低换手率=机构长线持有不交易——是'被遗忘的优质股'信号，低换手意味着市场关注度低但基本面好，有重估潜力。",
    dependencies=("__factors__", "roe", "turnover_20"),
)
def factor_quality_turnover_divergence(context: FactorContext):
    roe = context.load_factor("roe")
    turnover = context.load_factor("turnover_20")
    common = roe.index.intersection(turnover.index)
    roe_r = roe.loc[common].groupby(level="Date").rank(pct=True)
    to_r = (1 - turnover.loc[common].groupby(level="Date").rank(pct=True))
    return cross_sectional_rank(roe_r * to_r)


@register_factor(
    name="breakout_with_fund_flow",
    description="突破+资金确认因子，mom_20排名×net_mf_amount_intensity排名截面排名。",
    category="enhanced",
    thesis="价格突破趋势+主力资金净流入=资金驱动型突破——价格突破得到主力资金的确认，趋势持续性更强。",
    dependencies=("__factors__", "mom_20", "net_mf_amount_intensity"),
)
def factor_breakout_with_fund_flow(context: FactorContext):
    mom = context.load_factor("mom_20")
    mf = context.load_factor("net_mf_amount_intensity")
    common = mom.index.intersection(mf.index)
    mom_r = mom.loc[common].groupby(level="Date").rank(pct=True)
    mf_r = mf.loc[common].groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(mom_r * mf_r)


@register_factor(
    name="liquidity_discount_factor",
    description="流动性折价因子，(1-bp排名)×(1-turnover_20排名)截面排名（低估值+低流动性=流动性折价排前）。",
    category="enhanced",
    thesis="低估值的低流动性股票有双重折价——估值折价+流动性折价。当流动性改善时(如被纳入指数)，流动性折价修复会带来显著的alpha。",
    dependencies=("__factors__", "bp", "turnover_20"),
)
def factor_liquidity_discount_factor(context: FactorContext):
    bp = context.load_factor("bp")
    turnover = context.load_factor("turnover_20")
    common = bp.index.intersection(turnover.index)
    bp_r = bp.loc[common].groupby(level="Date").rank(pct=True)
    low_liq = (1 - turnover.loc[common].groupby(level="Date").rank(pct=True))
    return cross_sectional_rank(bp_r * low_liq)


@register_factor(
    name="sentiment_divergence_factor",
    description="情绪背离因子，(资金流入排名-动量排名)截面排名（资金流入>动量=资金提前布局排前）。",
    category="enhanced",
    thesis="资金流向与价格动量的背离蕴含信息——资金流入领先于价格上涨(聪明钱提前布局)，资金流出但价格不跌(筹码吸收)。正向背离是alpha来源。",
    dependencies=("__factors__", "mom_20", "net_mf_amount_intensity"),
)
def factor_sentiment_divergence_factor(context: FactorContext):
    mom = context.load_factor("mom_20")
    mf = context.load_factor("net_mf_amount_intensity")
    common = mom.index.intersection(mf.index)
    mom_r = mom.loc[common].groupby(level="Date").rank(pct=True)
    mf_r = mf.loc[common].groupby(level="Date").rank(pct=True)
    divergence = mf_r - mom_r
    return cross_sectional_rank(divergence)
