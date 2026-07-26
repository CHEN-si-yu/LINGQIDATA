from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_max,
    rolling_group_mean,
    rolling_group_min,
    rolling_group_std,
    safe_divide,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _daily_returns(daily: pd.DataFrame) -> pd.Series:
    """Daily percentage returns from close price."""
    close = daily["close"]
    return close.groupby(level="Code").pct_change()


def _mkt_return(ret: pd.Series) -> pd.Series:
    """Equal-weighted cross-sectional mean return per date."""
    return ret.groupby(level="Date").mean()


def _unstacked_beta(ret_panel: pd.DataFrame, mkt_ret: pd.Series, window: int) -> pd.Series:
    """Vectorised rolling beta for every stock vs equal-weighted market."""
    min_p = max(1, window // 2)
    roll_cov = ret_panel.rolling(window, min_periods=min_p).cov(mkt_ret)
    roll_var = mkt_ret.rolling(window, min_periods=min_p).var()
    beta_panel = roll_cov.div(roll_var, axis=0)
    beta = beta_panel.stack()
    beta.index = beta.index.set_names(["Date", "Code"])
    beta.replace([np.inf, -np.inf], np.nan, inplace=True)
    return beta


# ── Beta family ─────────────────────────────────────────────────────────────

@register_factor(
    name="beta_60",
    description="60日市场贝塔因子 (低贝塔排前)",
    category="risk",
    thesis="低beta股票风险调整后收益更高 (Frazzini & Pedersen 2014)",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def beta_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")
    beta = _unstacked_beta(ret_panel, mkt, 60)
    return cross_sectional_rank(-beta)  # low beta ranks higher


@register_factor(
    name="beta_120",
    description="120日市场贝塔因子 (低贝塔排前)",
    category="risk",
    thesis="长期低beta提供更稳定的风险溢价",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def beta_120(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")
    beta = _unstacked_beta(ret_panel, mkt, 120)
    return cross_sectional_rank(-beta)


@register_factor(
    name="downside_beta_60",
    description="60日下行贝塔因子 (市场下跌时的贝塔)",
    category="risk",
    thesis="下行贝塔捕捉尾部风险暴露，高下行贝塔的股票在市场下跌时亏损更大",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def downside_beta_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")
    # Keep only market-down days
    mask = mkt < 0
    ret_down = ret_panel.where(mask, 0.0)
    mkt_down = mkt.where(mask, 0.0)
    beta = _unstacked_beta(ret_down, mkt_down, 60)
    return cross_sectional_rank(-beta)


@register_factor(
    name="upside_beta_60",
    description="60日上行贝塔因子 (市场上涨时的贝塔)",
    category="risk",
    thesis="上行贝塔捕捉市场上涨时的参与度",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def upside_beta_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")
    mask = mkt > 0
    ret_up = ret_panel.where(mask, 0.0)
    mkt_up = mkt.where(mask, 0.0)
    beta = _unstacked_beta(ret_up, mkt_up, 60)
    return cross_sectional_rank(beta)


@register_factor(
    name="beta_asymmetry_60",
    description="60日贝塔不对称因子 (上行beta/下行beta)",
    category="risk",
    thesis="贝塔不对称性的溢价：上行beta远高于下行beta是正面信号",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def beta_asymmetry_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")

    mask_up = mkt > 0
    mask_dn = mkt < 0

    beta_up = _unstacked_beta(ret_panel.where(mask_up, 0.0), mkt.where(mask_up, 0.0), 60)
    beta_dn = _unstacked_beta(ret_panel.where(mask_dn, 0.0), mkt.where(mask_dn, 0.0), 60)

    ratio = safe_divide(beta_up, beta_dn.abs() + 1e-8)
    return cross_sectional_rank(ratio)


@register_factor(
    name="beta_stability_60",
    description="60日贝塔稳定性 (贝塔的波动率, 负向)",
    category="risk",
    thesis="贝塔稳定的股票风险特征明确，便于定价；贝塔波动大意味着风险特征漂移",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def beta_stability_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")

    betas = []
    for w in [20, 40, 60]:
        b = _unstacked_beta(ret_panel, mkt, w)
        b.name = f"beta_{w}"
        betas.append(b)
    beta_df = pd.concat(betas, axis=1)
    beta_vol = beta_df.std(axis=1, skipna=True)
    beta_vol.name = "beta_stability"

    return cross_sectional_rank(-beta_vol)


# ── Drawdown / pain index family ────────────────────────────────────────────

@register_factor(
    name="max_drawdown_120",
    description="120日最大回撤因子 (低回撤排前)",
    category="risk",
    thesis="回撤小的股票下行保护更好，低回撤溢价比低波动更直接",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def max_drawdown_120(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    close = daily["close"]
    cummax = close.groupby(level="Code").transform(
        lambda s: s.rolling(120, min_periods=60).max()
    )
    dd = (close - cummax) / cummax
    max_dd = dd.groupby(level="Code").transform(
        lambda s: s.rolling(120, min_periods=60).min()
    )
    return cross_sectional_rank(max_dd)  # less negative = higher rank


@register_factor(
    name="ulcer_index_20",
    description="20日溃疡指数因子 (低溃疡排前, 负向)",
    category="risk",
    thesis="溃疡指数综合衡量回撤深度和持续时间，比最大回撤更全面",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def ulcer_index_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    close = daily["close"]

    def _ulcer(s, window):
        cummax = s.rolling(window, min_periods=max(1, window // 2)).max()
        pct_dd = (s - cummax) / cummax
        return np.sqrt((pct_dd ** 2).rolling(window, min_periods=max(1, window // 2)).mean())

    ulcer = close.groupby(level="Code").transform(lambda s: _ulcer(s, 20))
    return cross_sectional_rank(-ulcer)


@register_factor(
    name="calmar_ratio_60",
    description="60日Calmar比率因子 (年化收益/最大回撤)",
    category="risk",
    thesis="Calmar比率同时考虑收益和回撤，高Calmar意味着更好的风险调整收益",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def calmar_ratio_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    close = daily["close"]

    ann_ret = rolling_group_mean(ret, 60) * 252
    cummax = close.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    )
    dd = (close - cummax) / cummax
    max_dd = dd.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).min()
    )

    calmar = safe_divide(ann_ret, -max_dd)
    return cross_sectional_rank(calmar)


@register_factor(
    name="sortino_ratio_60",
    description="60日Sortino比率因子 (年化收益/下行标准差)",
    category="risk",
    thesis="Sortino只惩罚下行波动，比Sharpe更合理",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def sortino_ratio_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)

    ann_ret = rolling_group_mean(ret, 60) * 252

    def _downside_std(s, window):
        neg = s.where(s < 0, 0.0)
        return neg.rolling(window, min_periods=max(1, window // 2)).std()

    down_std = ret.groupby(level="Code").transform(lambda s: _downside_std(s, 60))
    down_std_ann = down_std * np.sqrt(252)

    sortino = safe_divide(ann_ret, down_std_ann)
    return cross_sectional_rank(sortino)


# ── Tail risk ───────────────────────────────────────────────────────────────

@register_factor(
    name="var_95_20",
    description="20日95%在险价值因子 (高VaR排后, 负向)",
    category="risk",
    thesis="高VaR意味着更厚的左尾风险，低VaR提供更好的下行保护",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def var_95_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)

    def _rolling_var(s, window):
        return s.rolling(window, min_periods=max(1, window // 2)).quantile(0.05)

    var = ret.groupby(level="Code").transform(lambda s: _rolling_var(s, 20))
    return cross_sectional_rank(var)  # higher VaR (less negative) = better

@register_factor(
    name="tail_ratio_60",
    description="60日尾部比率因子 (95%分位收益/|5%分位收益|)",
    category="risk",
    thesis="尾部比率衡量收益分布的对称性，高比率意味着右尾厚左尾薄",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def tail_ratio_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)

    def _rolling_q95(s, window):
        return s.rolling(window, min_periods=max(1, window // 2)).quantile(0.95)

    def _rolling_q05(s, window):
        return s.rolling(window, min_periods=max(1, window // 2)).quantile(0.05)

    q95 = ret.groupby(level="Code").transform(lambda s: _rolling_q95(s, 60))
    q05 = ret.groupby(level="Code").transform(lambda s: _rolling_q05(s, 60))

    ratio = safe_divide(q95, q05.abs())
    return cross_sectional_rank(ratio)


@register_factor(
    name="tail_risk_dd_20",
    description="20日尾部回撤因子 (5日最大回撤, 负向)",
    category="risk",
    thesis="短期极端回撤信号预示流动性风险",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def tail_risk_dd_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    close = daily["close"]

    def _rolling_worst_5d(s, window):
        ret_5d = s.pct_change(5)
        return ret_5d.rolling(window, min_periods=max(1, window // 2)).min()

    worst5 = close.groupby(level="Code").transform(lambda s: _rolling_worst_5d(s, 20))
    return cross_sectional_rank(worst5)


# ── Correlation / co-moment ─────────────────────────────────────────────────

@register_factor(
    name="market_corr_60",
    description="60日市场相关性因子 (低相关排前)",
    category="risk",
    thesis="与市场相关性低的股票提供分散化收益，有独特alpha",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def market_corr_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)

    ret_panel = ret.unstack("Code")
    min_p = max(1, 30)
    roll_cov = ret_panel.rolling(60, min_periods=min_p).cov(mkt)
    roll_std_stock = ret_panel.rolling(60, min_periods=min_p).std()
    roll_std_mkt = mkt.rolling(60, min_periods=min_p).std()

    corr_panel = roll_cov.div(roll_std_stock.mul(roll_std_mkt, axis=0), axis=0)
    corr = corr_panel.stack()
    corr.index = corr.index.set_names(["Date", "Code"])
    corr.replace([np.inf, -np.inf], np.nan, inplace=True)

    return cross_sectional_rank(-corr)


@register_factor(
    name="correlation_change_20",
    description="市场相关性20日变化因子 (相关下降排前)",
    category="risk",
    thesis="市场相关性快速下降的股票可能在积累独立行情",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def correlation_change_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")

    min_p = max(1, 30)
    roll_cov = ret_panel.rolling(60, min_periods=min_p).cov(mkt)
    roll_std_stock = ret_panel.rolling(60, min_periods=min_p).std()
    roll_std_mkt = mkt.rolling(60, min_periods=min_p).std()
    corr_panel = roll_cov.div(roll_std_stock.mul(roll_std_mkt, axis=0), axis=0)

    corr_change = corr_panel.diff(20)
    corr_s = corr_change.stack()
    corr_s.index = corr_s.index.set_names(["Date", "Code"])
    corr_s.replace([np.inf, -np.inf], np.nan, inplace=True)

    return cross_sectional_rank(-corr_s)


@register_factor(
    name="idiosyncratic_vol_60",
    description="60日特质波动率因子 (低特质波动排前, 负向)",
    category="risk",
    thesis="特质波动率谜题：低特质波动率的股票未来收益更高 (Ang et al. 2006)",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def idiosyncratic_vol_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")

    beta = _unstacked_beta(ret_panel, mkt, 60)
    beta.name = "beta"

    # Align beta and compute residual
    beta_unstack = beta.unstack("Code") if isinstance(beta, pd.Series) else beta
    # Recompute: residual return = stock_ret - beta * mkt_ret
    aligned_mkt = mkt_ret_aligned = mkt.reindex(ret_panel.index)
    systematic = ret_panel.mul(beta_unstack.reindex(ret_panel.index), axis=0) * 0
    # Simpler approach: compute residual vol from total vol and systematic vol
    total_vol = ret_panel.rolling(60, min_periods=30).std() * np.sqrt(252)

    # Systematic vol = |beta| * market_vol
    mkt_vol = mkt.rolling(60, min_periods=30).std() * np.sqrt(252)
    systematic_vol = beta_unstack.abs().mul(mkt_vol, axis=0)

    # Idio vol = sqrt(total_vol^2 - systematic_vol^2)
    idio_var = total_vol ** 2 - systematic_vol ** 2
    idio_var = idio_var.clip(lower=0)
    idio_vol = np.sqrt(idio_var)

    idio_s = idio_vol.stack()
    idio_s.index = idio_s.index.set_names(["Date", "Code"])
    idio_s.replace([np.inf, -np.inf], np.nan, inplace=True)

    return cross_sectional_rank(-idio_s)


@register_factor(
    name="coskewness_60",
    description="60日协偏度因子 (高协偏度排前)",
    category="risk",
    thesis="协偏度高的股票在市场下跌时跌幅较小，提供下行保护",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def coskewness_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    mkt = _mkt_return(ret)
    ret_panel = ret.unstack("Code")

    # Standardize
    mkt_demean = mkt - mkt.rolling(60, min_periods=30).mean()
    mkt_std = mkt.rolling(60, min_periods=30).std()
    mkt_z = mkt_demean / mkt_std

    ret_demean = ret_panel.sub(ret_panel.rolling(60, min_periods=30).mean())
    ret_std = ret_panel.rolling(60, min_periods=30).std()
    ret_z = ret_demean.div(ret_std, axis=0)

    # Co-skewness: E[r_i * mkt^2]
    coskew_num = ret_z.mul(mkt_z ** 2, axis=0).rolling(60, min_periods=30).mean()
    coskew_denom = mkt_z.rolling(60, min_periods=30).skew()

    result = coskew_num.stack()
    result.index = result.index.set_names(["Date", "Code"])
    result.replace([np.inf, -np.inf], np.nan, inplace=True)

    return cross_sectional_rank(result)


# ── Volatility of volatility ────────────────────────────────────────────────

@register_factor(
    name="vol_stability_20",
    description="20日波动率稳定性因子 (低波动稳定排前, 负向)",
    category="risk",
    thesis="波动率波动大的股票面临不确定性风险，市场偏好波动率稳定的股票",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def vol_stability_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    ret = _daily_returns(daily)
    vol_5 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).std()
    )
    vol_of_vol = vol_5.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-vol_of_vol)
