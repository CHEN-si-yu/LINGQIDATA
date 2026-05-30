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
    rolling_group_sum,
    safe_divide,
)


# ── Spread proxies ──────────────────────────────────────────────────────────

@register_factor(
    name="roll_spread_20",
    description="20日Roll价差估计因子 (高spread排后, 负向)",
    category="market_structure",
    thesis="Roll(1984)模型通过价格反转的协方差估计有效价差，高spread意味着高交易成本",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def roll_spread_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    close = daily["close"]
    ret = close.groupby(level="Code").pct_change()

    def _roll_spread(s, window):
        s = s.dropna()
        if len(s) < window:
            return pd.Series(np.nan, index=s.index)
        cov = s.rolling(window, min_periods=max(1, window // 2)).cov(s.shift(1))
        cov_neg = cov.clip(upper=0)
        spread = 2 * np.sqrt(-cov_neg)
        return spread

    spread = ret.groupby(level="Code").transform(lambda s: _roll_spread(s, 20))
    return cross_sectional_rank(-spread)


@register_factor(
    name="cs_spread_20",
    description="20日Corwin-Schultz价差估计因子 (高spread排后, 负向)",
    category="market_structure",
    thesis="Corwin-Schultz(2012)利用日内最高最低价估计买卖价差，适合没有逐笔数据的场景",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def cs_spread_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    high = daily["high"]
    low = daily["low"]

    ratio_beta = (high / low.replace(0, np.nan))
    beta = np.log(ratio_beta.where(ratio_beta > 0, np.nan)) ** 2
    ratio_gamma = (high.rolling(2).max() / low.rolling(2).min().replace(0, np.nan))
    gamma = np.log(ratio_gamma.where(ratio_gamma > 0, np.nan)) ** 2

    def _cs_est(s, window):
        beta_s = beta.reindex(s.index)
        gamma_s = gamma.reindex(s.index)
        alpha = (np.sqrt(2) - 1) / 3 - np.sqrt(2) / (3 - 2 * np.sqrt(2))
        spread = (2 * (np.exp(beta_s.rolling(window, min_periods=10).mean() * alpha) - 1)
                  / (1 + np.exp(beta_s.rolling(window, min_periods=10).mean() * alpha)))
        # Clamp to reasonable range
        spread = spread.clip(0, 0.1)
        return spread

    spread = high.groupby(level="Code").transform(lambda s: _cs_est(s, 20))
    return cross_sectional_rank(-spread)


@register_factor(
    name="high_low_spread_20",
    description="20日高低价差比率因子 (高spread排后, 负向)",
    category="market_structure",
    thesis="日内高低价差/均价反映流动性，spread大意味着做市成本高",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def high_low_spread_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    spread = (daily["high"] - daily["low"]) / ((daily["high"] + daily["low"]) / 2)
    avg_spread = rolling_group_mean(spread, 20)
    return cross_sectional_rank(-avg_spread)


# ── Kyle's lambda & price impact ────────────────────────────────────────────

@register_factor(
    name="amihud_illiq_60",
    description="60日Amihud非流动性因子 (高非流动排后, 负向)",
    category="market_structure",
    thesis="Amihud(2002)非流动性衡量单笔交易的价格冲击，高ILLIQ意味着流动性差",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def amihud_illiq_60(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    abs_ret = daily["close"].groupby(level="Code").pct_change().abs()
    # Use 'amount' as trading volume proxy (yuan)
    amount = daily["amount"]
    daily_illiq = safe_divide(abs_ret, amount / 1e6)  # scale for numerical stability

    illiq_60 = daily_illiq.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    return cross_sectional_rank(-illiq_60)


@register_factor(
    name="kyle_lambda_20",
    description="20日Kyle's lambda代理因子 (高冲击排后, 负向)",
    category="market_structure",
    thesis="Kyle's lambda衡量成交量对价格的影响，高lambda意味着大额交易成本高",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def kyle_lambda_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    abs_ret = daily["close"].groupby(level="Code").pct_change().abs()
    amount_change = daily["amount"].groupby(level="Code").pct_change()
    lambda_daily = safe_divide(abs_ret, amount_change.abs() + 1e-8)

    lambda_20 = lambda_daily.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-lambda_20)


@register_factor(
    name="price_impact_ratio_20",
    description="20日价格冲击比率 (涨跌幅/换手率, 负向)",
    category="market_structure",
    thesis="每单位换手率推动的价格变化越大，流动性越差",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def price_impact_ratio_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    finance = ctx.load("finance.parquet")

    abs_ret = daily["close"].groupby(level="Code").pct_change().abs()
    turnover = finance["turnover_rate"]

    impact = safe_divide(abs_ret, turnover + 1e-8)
    impact_20 = impact.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-impact_20)


# ── Market depth & liquidity ────────────────────────────────────────────────

@register_factor(
    name="dollar_volume_20",
    description="20日均成交额因子",
    category="market_structure",
    thesis="高成交额意味着高流动性和低交易成本，同时反映市场关注度",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def dollar_volume_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    avg_amount = rolling_group_mean(daily["amount"], 20)
    return cross_sectional_rank(avg_amount)


@register_factor(
    name="dollar_volume_growth_20",
    description="20日成交额增长因子",
    category="market_structure",
    thesis="成交额趋势性增长预示关注度上升和流动性改善",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def dollar_volume_growth_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    amount = daily["amount"]
    ma_5 = rolling_group_mean(amount, 5)
    ma_20 = rolling_group_mean(amount, 20)
    growth = safe_divide(ma_5 - ma_20, ma_20)
    return cross_sectional_rank(growth)


@register_factor(
    name="volume_stability_20",
    description="20日成交量稳定性因子 (高稳定排前, 负向)",
    category="market_structure",
    thesis="成交量稳定的股票流动性可预测，交易成本更低",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def volume_stability_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    vol = daily["vol"]
    vol_cv = safe_divide(rolling_group_std(vol, 20), rolling_group_mean(vol, 20))
    return cross_sectional_rank(-vol_cv)


@register_factor(
    name="liquidity_depth_20",
    description="20日流动性深度因子 (换手率稳定×高成交额)",
    category="market_structure",
    thesis="流动性深度综合反映市场承接能力，深度好的股票大资金进出成本低",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def liquidity_depth_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    finance = ctx.load("finance.parquet")
    amount = daily["amount"]
    turnover = finance["turnover_rate"]
    # Combine: stable turnover + high dollar volume
    t_stability = -safe_divide(rolling_group_std(turnover, 20), rolling_group_mean(turnover, 20) + 1e-8)
    mean_amount = rolling_group_mean(amount, 20)
    log_amount = np.log(mean_amount.where(mean_amount > 0, np.nan))
    depth = log_amount.groupby(level="Date").transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-8)
    ) + t_stability.groupby(level="Date").transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-8)
    )
    return cross_sectional_rank(depth)


# ── Price efficiency ────────────────────────────────────────────────────────

@register_factor(
    name="price_efficiency_ratio_20",
    description="20日价格效率比率 (|总收益|/路径长度, 高效排前)",
    category="market_structure",
    thesis="Kaufman效率比率衡量价格运动的趋势性，高效率意味着更确定的定价",
    dependencies=("daily_adj.parquet", "calendar.parquet"),
)
def price_efficiency_ratio_20(ctx: FactorContext) -> pd.Series:
    daily = ctx.load("daily_adj.parquet")
    close = daily["close"]

    def _eff_ratio(s, window):
        net = (s - s.shift(window)).abs()
        path = s.diff().abs().rolling(window, min_periods=max(1, window // 2)).sum()
        return safe_divide(net, path)

    eff = close.groupby(level="Code").transform(lambda s: _eff_ratio(s, 20))
    return cross_sectional_rank(eff)


# ── Market cap distribution ───────────────────────────────────────────────

@register_factor(
    name="size_percentile_252d",
    description="市值分位因子，当前总市值在252日滚动窗口中的百分位截面排名。",
    category="valuation",
    thesis="当前市值在历史市值分布中的位置——市值处于历史高位时(大市值)享有流动性溢价但弹性下降，处于历史低位时(缩水严重)可能面临退市风险但也可能是反弹机会。",
    dependencies=("finance.parquet",),
)
def factor_size_percentile_252d(context: FactorContext):
    finance = context.load("finance.parquet")
    mv = finance["total_mv"]
    pct_252 = mv.groupby(level="Code").transform(
        lambda s: s.rolling(252, min_periods=126).apply(
            lambda x: (x < x.iloc[-1]).mean(), raw=False
        )
    )
    return cross_sectional_rank(pct_252)


@register_factor(
    name="free_float_ratio",
    description="自由流通比例因子，free_share/total_share截面排名。",
    category="valuation",
    thesis="自由流通股占比影响股票的流动性、波动性和定价效率——高自由流通比意味着股票被广泛持有、流动性好，低自由流通比可能存在大股东控制权集中的折价或溢价。",
    dependencies=("finance.parquet",),
)
def factor_free_float_ratio(context: FactorContext):
    finance = context.load("finance.parquet")
    ff_ratio = finance["free_share"] / finance["total_share"].replace(0, np.nan)
    return cross_sectional_rank(ff_ratio)


@register_factor(
    name="circ_mv_to_total_mv",
    description="流通市值/总市值截面排名（高流通比排前）。",
    category="valuation",
    thesis="流通市值占总市值的比例——高流通比意味着限售股压力小、流通性好，低流通比意味着未来解禁后有大量潜在抛压。",
    dependencies=("finance.parquet",),
)
def factor_circ_mv_to_total_mv(context: FactorContext):
    finance = context.load("finance.parquet")
    ratio = finance["circ_mv"] / finance["total_mv"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Market impact / liquidity depth ──────────────────────────────────────

@register_factor(
    name="amihud_illiq_5",
    description="Amihud非流动性5日版因子截面排名。",
    category="price",
    thesis="5日窗口的Amihud非流动性指标比20日版对短期流动性冲击更敏感——捕捉近期的价格冲击成本和流动性恶化。",
    dependencies=("daily_adj.parquet",),
)
def factor_amihud_illiq_5(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj["close"].groupby(level="Code").transform(lambda s: s.pct_change(1))
    amount = daily_adj["amount"]
    illiq = (ret.abs() / amount.replace(0, np.nan))
    illiq_5 = illiq.groupby(level="Code").transform(lambda s: s.rolling(5, min_periods=3).mean())
    return cross_sectional_rank(illiq_5)


@register_factor(
    name="price_impact_proxy",
    description="价格冲击代理因子，-(|ret|/turnover)截面排名（高冲击=流动性差排后）。",
    category="price",
    thesis="单位换手率引起的价格变动幅度——价格冲击越大意味着流动性越差、交易成本越高，投资者会要求更高的流动性补偿(高收益期望)。",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def factor_price_impact_proxy(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    finance = context.load("finance.parquet")
    ret = daily_adj["close"].groupby(level="Code").transform(lambda s: s.pct_change(1)).abs()
    turnover = finance["turnover_rate"]
    common = ret.index.intersection(turnover.index)
    impact = ret.loc[common] / turnover.loc[common].replace(0, np.nan)
    return cross_sectional_rank(impact)


@register_factor(
    name="market_cap_concentration_20d",
    description="市值集中度因子，log_total_mv的20日波动率截面排名。",
    category="valuation",
    thesis="市值在短期内的剧烈波动反映公司基本面的不确定性——市值频繁大幅波动意味着市场对公司价值的共识度低、信息不对称高。",
    dependencies=("finance.parquet",),
)
def factor_market_cap_concentration_20d(context: FactorContext):
    finance = context.load("finance.parquet")
    mv = np.log(finance["total_mv"].replace(0, np.nan))
    mv_vol_20 = mv.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-mv_vol_20)
