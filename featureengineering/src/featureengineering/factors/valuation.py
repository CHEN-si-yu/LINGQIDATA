from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std


# ── Value ───────────────────────────────────────────────────────────────

@register_factor(
    name="bp",
    description="账面市值比(BP)因子，1/PB截面排名，高值代表价值股。",
    category="valuation",
    thesis="价值因子是Fama-French三因子之一，高BP股票长期有超额收益。",
    dependencies=("finance.parquet",),
)
def factor_bp(context: FactorContext):
    finance = context.load("finance.parquet")
    bp = 1.0 / finance["pb"].replace(0, np.nan)
    return cross_sectional_rank(bp)


@register_factor(
    name="sp_ttm",
    description="市销率倒数(SP_TTM)因子，1/PS_TTM截面排名。",
    category="valuation",
    thesis="PS估值对净利润为负的公司仍有定义域，在A股覆盖面优于PE类因子。",
    dependencies=("finance.parquet",),
)
def factor_sp_ttm(context: FactorContext):
    finance = context.load("finance.parquet")
    sp = 1.0 / finance["ps_ttm"].replace(0, np.nan)
    return cross_sectional_rank(sp)


@register_factor(
    name="dp_ttm",
    description="滚动股息率因子，截面排名。",
    category="valuation",
    thesis="高股息率股票在低利率环境中具备配置价值，且在下跌市中具有防御属性。",
    dependencies=("finance.parquet",),
)
def factor_dp_ttm(context: FactorContext):
    finance = context.load("finance.parquet")
    dp = finance["dv_ttm"]
    return cross_sectional_rank(dp)


# ── Size ────────────────────────────────────────────────────────────────

@register_factor(
    name="log_total_mv",
    description="对数总市值因子，负对数总市值（小市值排前）。",
    category="valuation",
    thesis="规模因子是A股最显著的单因子之一，小市值效应长期存在。",
    dependencies=("finance.parquet",),
)
def factor_log_total_mv(context: FactorContext):
    finance = context.load("finance.parquet")
    log_mv = np.log(finance["total_mv"].replace(0, np.nan))
    return cross_sectional_rank(-log_mv)


@register_factor(
    name="log_circ_mv",
    description="对数流通市值因子，负对数流通市值截面排名。",
    category="valuation",
    thesis="流通市值比总市值更精确反映可交易盘规模，对小盘效应捕捉更纯。",
    dependencies=("finance.parquet",),
)
def factor_log_circ_mv(context: FactorContext):
    finance = context.load("finance.parquet")
    log_cmv = np.log(finance["circ_mv"].replace(0, np.nan))
    return cross_sectional_rank(-log_cmv)


@register_factor(
    name="float_mv_ratio",
    description="自由流通市值占比因子，free_share/total_share截面排名。",
    category="valuation",
    thesis="自由流通盘占比低代表筹码锁定度高、实际流通盘小，可能伴随更高的波动弹性。",
    dependencies=("finance.parquet",),
)
def factor_float_mv_ratio(context: FactorContext):
    finance = context.load("finance.parquet")
    ratio = finance["free_share"] / finance["total_share"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Liquidity ───────────────────────────────────────────────────────────

@register_factor(
    name="turnover_20",
    description="20日平均换手率因子（总股本换手率），低换手排前。",
    category="valuation",
    thesis="低换手率反映筹码稳定、投机度低，在A股中具有正向截面预测力。",
    dependencies=("finance.parquet",),
)
def factor_turnover_20(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    avg_turnover = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_turnover)


@register_factor(
    name="turnover_f_20",
    description="20日平均自由流通换手率因子，低换手排前。",
    category="valuation",
    thesis="自由流通换手率剔除大股东锁定股份，更精确反映真实交易活跃度。",
    dependencies=("finance.parquet",),
)
def factor_turnover_f_20(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate_f"]
    avg_turnover = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_turnover)


@register_factor(
    name="turnover_vol_20",
    description="20日换手率波动因子，换手率标准差截面排名（低波动排前）。",
    category="valuation",
    thesis="换手率剧烈波动常反映资金博弈激烈，是风险信号。",
    dependencies=("finance.parquet",),
)
def factor_turnover_vol_20(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    vol = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-vol)


@register_factor(
    name="volume_ratio",
    description="量比因子（当日成交量相对5日均量），截面排名。",
    category="valuation",
    thesis="量比是盘中常用指标，极端量比常伴随短期反转。",
    dependencies=("finance.parquet",),
)
def factor_volume_ratio(context: FactorContext):
    finance = context.load("finance.parquet")
    vol_ratio = finance["volume_ratio"]
    return cross_sectional_rank(-vol_ratio)


# ── PE percentile ───────────────────────────────────────────────────────

@register_factor(
    name="pe_ttm_percentile",
    description="PE_TTM历史分位因子（低分位=估值处于历史低位，排前）。",
    category="valuation",
    thesis="估值相对于自身历史的低位是价值回归的潜在信号。",
    dependencies=("finance.parquet",),
)
def factor_pe_ttm_percentile(context: FactorContext):
    finance = context.load("finance.parquet")
    percentile = finance["pe_ttm_percentile"]
    return cross_sectional_rank(-percentile)


# ── Dividend composite ────────────────────────────────────────────────────

@register_factor(
    name="dv_composite",
    description="股息率综合因子，dv_ratio与dv_ttm的等权平均截面排名。",
    category="valuation",
    thesis="dv_ratio（报告期股息率）与dv_ttm（滚动股息率）从不同维度度量股息回报，综合后信号更稳定。",
    dependencies=("finance.parquet",),
)
def factor_dv_composite(context: FactorContext):
    finance = context.load("finance.parquet")
    rank_ratio = finance["dv_ratio"].groupby(level="Date").rank(pct=True)
    rank_ttm = finance["dv_ttm"].groupby(level="Date").rank(pct=True)
    composite = (rank_ratio + rank_ttm) / 2.0
    return composite.rename("dv_composite")
