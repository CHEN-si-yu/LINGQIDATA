from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    safe_divide,
)


# ── Stability / persistence factors ─────────────────────────────────────────

@register_factor(
    name="roe_stability_8q",
    description="ROE 8季度稳定性因子 (变异系数取负, 稳定排前)",
    category="quality",
    thesis="ROE持续稳定意味着竞争优势持久，波动大的ROE往往包含一次性项目",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def roe_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["roe"])
    roe = fin["roe"]
    roe_std = roe.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    roe_mean = roe.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    cv = safe_divide(roe_std, roe_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="roa_stability_8q",
    description="ROA 8季度稳定性因子 (稳定排前)",
    category="quality",
    thesis="ROA稳定反映资产盈利能力的持续性和管理层的审慎经营",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def roa_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["roa"])
    roa = fin["roa"]
    roa_std = roa.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    roa_mean = roa.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    cv = safe_divide(roa_std, roa_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="gross_margin_stability_8q",
    description="毛利率8季度稳定性因子 (稳定排前)",
    category="quality",
    thesis="毛利率稳定反映定价权和成本控制能力，毛利率大幅波动是竞争恶化的信号",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def gross_margin_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["gross_margin"])
    gm = fin["gross_margin"]
    gm_std = gm.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    gm_mean = gm.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    cv = safe_divide(gm_std, gm_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="net_margin_stability_8q",
    description="净利率8季度稳定性因子 (稳定排前)",
    category="quality",
    thesis="净利率稳定意味着盈利模式成熟，不受非经常损益大幅扰动",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def net_margin_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["netprofit_margin"])
    npm = fin["netprofit_margin"]
    npm_std = npm.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    npm_mean = npm.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    cv = safe_divide(npm_std, npm_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="revenue_stability_8q",
    description="营业收入8季度稳定性因子 (稳定排前)",
    category="quality",
    thesis="营收稳定表明需求端稳固，大客户和市场份额稳定",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def revenue_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["or_yoy"])
    rev_growth = fin["or_yoy"]
    growth_std = rev_growth.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    return cross_sectional_rank(-growth_std)


@register_factor(
    name="profit_volatility_8q",
    description="8季度利润波动率因子 (低波动排前, 负向)",
    category="quality",
    thesis="利润大幅波动是基本面不稳定的信号，低波动意味着可预测性强",
    dependencies=("income.parquet", "calendar.parquet"),
)
def profit_volatility_8q(ctx: FactorContext) -> pd.Series:
    inc = ctx.load_financial("income.parquet", value_cols=["n_income"])
    ni = inc["n_income"]
    ni_std = ni.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    ni_mean = ni.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    cv = safe_divide(ni_std, ni_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


# ── Earnings quality ────────────────────────────────────────────────────────

@register_factor(
    name="earnings_smoothness_8q",
    description="8季度盈利平滑度因子 (平滑排前)",
    category="quality",
    thesis="盈利平滑反映应计与现金流的协方差，高平滑度意味着更高的盈余质量",
    dependencies=("income.parquet", "cashflow.parquet", "calendar.parquet"),
)
def earnings_smoothness_8q(ctx: FactorContext) -> pd.Series:
    inc = ctx.load_financial("income.parquet", value_cols=["n_income"])
    cf = ctx.load_financial("cashflow.parquet", value_cols=["n_cashflow_act"])
    ni = inc["n_income"]
    ocf = cf["n_cashflow_act"]
    # Smoothness = -|std(ni)/std(ocf) - 1|  (closer to 1 = more natural relationship)
    ni_std = ni.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    ocf_std = ocf.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    ratio = safe_divide(ni_std, ocf_std + 1e-8)
    smoothness = -(ratio - 1).abs()
    return cross_sectional_rank(smoothness)


@register_factor(
    name="accruals_quality_8q",
    description="8季度应计质量因子 (低应计排前, 负向)",
    category="quality",
    thesis="净应计=NI-OCF，高应计意味着盈利质量差，低应计意味着现金支持好",
    dependencies=("income.parquet", "cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def accruals_quality_8q(ctx: FactorContext) -> pd.Series:
    inc = ctx.load_financial("income.parquet", value_cols=["n_income"])
    cf = ctx.load_financial("cashflow.parquet", value_cols=["n_cashflow_act"])
    bs = ctx.load_financial("balancesheet.parquet", value_cols=["total_assets"])
    ni = inc["n_income"]
    ocf = cf["n_cashflow_act"]
    ta = bs["total_assets"]
    accruals = (ni - ocf) / (ta.abs() + 1e-8)
    avg_accruals = accruals.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    return cross_sectional_rank(-avg_accruals)


# ── Growth quality ──────────────────────────────────────────────────────────

@register_factor(
    name="revenue_growth_acceleration",
    description="营收增速加速度因子 (营收YoY增速的变化)",
    category="quality",
    thesis="营收增速加快意味着需求加速渗透，Growth on Growth是成长股的核心驱动力",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def revenue_growth_acceleration(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["or_yoy"])
    rev_growth = fin["or_yoy"]
    accel = rev_growth.groupby(level="Code").diff(4)  # YoY growth change
    return cross_sectional_rank(accel)


@register_factor(
    name="eps_growth_4q",
    description="4季度EPS同比增速因子",
    category="quality",
    thesis="EPS增长是最基础的盈利质量信号，持续正增长是价值创造的前提",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def eps_growth_4q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["eps", "dt_eps_yoy"])
    if "dt_eps_yoy" in (fin.columns if isinstance(fin, pd.DataFrame) else []):
        return cross_sectional_rank(fin["dt_eps_yoy"])
    eps = fin["eps"]
    eps_growth = eps.groupby(level="Code").pct_change(4)
    return cross_sectional_rank(eps_growth)


@register_factor(
    name="netprofit_growth_acceleration",
    description="净利润增速加速度因子 (YoY增速的环比变化)",
    category="quality",
    thesis="利润增速的边际变化比增速水平更有预测力 (earning momentum的延伸)",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def netprofit_growth_acceleration(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["netprofit_yoy"])
    np_growth = fin["netprofit_yoy"]
    accel = np_growth.groupby(level="Code").diff(4)
    return cross_sectional_rank(accel)


@register_factor(
    name="sustainable_growth_rate",
    description="可持续增长率因子 (ROE x 留存比率)",
    category="quality",
    thesis="SGR=ROE×(1-分红率)衡量内生增长上限，高SGR意味着不需要融资即可成长",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def sustainable_growth_rate(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["roe", "profit_dedt"])
    roe = fin["roe"]
    # profit_dedt / n_income ≈ payout ratio proxy
    payout = fin.get("profit_dedt")
    if payout is not None:
        # Approximate retention ratio
        retention = 1 - payout.abs() / (payout.abs() + 100)
        sgr = roe * retention
    else:
        sgr = roe * 0.7  # assume 30% payout as fallback
    return cross_sectional_rank(sgr)


# ── Cash flow quality ──────────────────────────────────────────────────────

@register_factor(
    name="ocf_growth_4q",
    description="4季度经营现金流同比增速因子",
    category="quality",
    thesis="经营现金流增长是盈利质量改善的最直接体现，比利润增长更可靠",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def ocf_growth_4q(ctx: FactorContext) -> pd.Series:
    cf = ctx.load_financial("cashflow.parquet", value_cols=["n_cashflow_act"])
    ocf = cf["n_cashflow_act"]
    ocf_growth = ocf.groupby(level="Code").pct_change(4)
    ocf_growth.replace([np.inf, -np.inf], np.nan, inplace=True)
    return cross_sectional_rank(ocf_growth)


@register_factor(
    name="fcf_to_equity",
    description="股权自由现金流因子 (FCFE/总市值)",
    category="quality",
    thesis="FCFE衡量股东可支配的现金流，FCFE/市值高意味着现金回报潜力大",
    dependencies=("cashflow.parquet", "finance.parquet", "calendar.parquet"),
)
def fcf_to_equity(ctx: FactorContext) -> pd.Series:
    cf = ctx.load_financial("cashflow.parquet", value_cols=["free_cashflow"])
    fin = ctx.load("finance.parquet")
    fcf = cf["free_cashflow"]
    total_mv = fin["total_mv"]
    fcfe_yield = safe_divide(fcf, total_mv)
    return cross_sectional_rank(fcfe_yield)


@register_factor(
    name="cf_volatility_8q",
    description="8季度经营现金流波动率因子 (低波动排前, 负向)",
    category="quality",
    thesis="OCF波动大意味着经营不确定性高，低波动的OCF意味着稳定的商业模式",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def cf_volatility_8q(ctx: FactorContext) -> pd.Series:
    cf = ctx.load_financial("cashflow.parquet", value_cols=["n_cashflow_act"])
    ocf = cf["n_cashflow_act"]
    ocf_std = ocf.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    ocf_mean = ocf.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    cv = safe_divide(ocf_std, ocf_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="capex_to_assets",
    description="资本支出/总资产因子 (低capex排前, 负向)",
    category="quality",
    thesis="高capex意味着高资本消耗和维护负担，低capex的企业轻资产运营更灵活",
    dependencies=("cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def capex_to_assets(ctx: FactorContext) -> pd.Series:
    cf = ctx.load_financial("cashflow.parquet", value_cols=["c_pay_acq_const_fiolta"])
    bs = ctx.load_financial("balancesheet.parquet", value_cols=["total_assets"])
    capex = cf["c_pay_acq_const_fiolta"].abs()
    ta = bs["total_assets"]
    ratio = safe_divide(capex, ta)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="asset_turn_stability_8q",
    description="总资产周转率8季度稳定性因子 (变异系数取负, 稳定排前)",
    category="quality",
    thesis="资产周转率稳定反映收入与资产规模的均衡增长——周转率大幅波动可能意味着资产扩张节奏失控或收入确认异常",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def asset_turn_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["assets_turn"])
    s = fin["assets_turn"]
    s_std = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).std())
    s_mean = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).mean())
    cv = safe_divide(s_std, s_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="ocfps_stability_8q",
    description="每股经营现金流8季度稳定性因子 (稳定排前)",
    category="quality",
    thesis="每股现金流的稳定性是商业模式可预测性的终极验证——现金流稳定意味着客户粘性强、成本可控",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def ocfps_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["ocfps"])
    s = fin["ocfps"]
    s_std = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).std())
    s_mean = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).mean())
    cv = safe_divide(s_std, s_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="eps_growth_stability_8q",
    description="EPS同比增速8季度稳定性因子 (稳定排前)",
    category="quality",
    thesis="EPS增速的稳定性是高质量成长股的核心特征——增速稳定意味着增长动能持续而非一次性",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def eps_growth_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["dt_eps_yoy"])
    s = fin["dt_eps_yoy"]
    s_std = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).std())
    s_mean = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).mean())
    cv = safe_divide(s_std, s_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="op_growth_stability_8q",
    description="营业利润增速8季度稳定性因子 (稳定排前)",
    category="quality",
    thesis="营业利润增速稳定代表主营业务增长具有可持续性",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def op_growth_stability_8q(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["op_yoy"])
    s = fin["op_yoy"]
    s_std = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).std())
    s_mean = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).mean())
    cv = safe_divide(s_std, s_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)


@register_factor(
    name="ebitda_margin_rank",
    description="EBITDA利润率截面排名 (ebitda / total_revenue_ps 近似)",
    category="quality",
    thesis="EBITDA利润率是经营效率的综合指标——高EBITDA利润率代表强定价权、低成本结构和高效运营",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def ebitda_margin_rank(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["ebitda", "total_revenue_ps"])
    margin = safe_divide(fin["ebitda"], fin["total_revenue_ps"].abs() + 1e-8)
    return cross_sectional_rank(margin)


@register_factor(
    name="capital_reserve_ps_rank",
    description="每股资本公积截面排名",
    category="quality",
    thesis="每股资本公积高意味着历史上高溢价融资——是市场对公司的历史定价认可的度量",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def capital_reserve_ps_rank(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["capital_rese_ps"])
    return cross_sectional_rank(fin["capital_rese_ps"])


@register_factor(
    name="op_to_profit_rank",
    description="营业利润/利润总额截面排名 (高占比=利润来自主营、可持续性强)",
    category="quality",
    thesis="营业利润占利润总额的比重反映利润来源的持续性——高占比=利润来自主营、可持续性强",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def op_to_profit_rank(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["op_to_ebt"])
    return cross_sectional_rank(fin["op_to_ebt"])


@register_factor(
    name="invest_income_to_ebt",
    description="投资收益/利润总额截面排名 (高占比=利润依赖投资，排后, 负向)",
    category="quality",
    thesis="投资收益占利润比重过高意味着公司依赖非主营的投资收益维持利润——可持续性差、波动性大",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def invest_income_to_ebt(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["investincome_of_ebt"])
    return cross_sectional_rank(-fin["investincome_of_ebt"])


@register_factor(
    name="noncurrent_exint_rank",
    description="长期债务/总债务截面排名",
    category="quality",
    thesis="长期债务占比高意味着债务结构更稳定——但也意味着更高的最低利息负担",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def noncurrent_exint_rank(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["longdeb_to_debt"])
    return cross_sectional_rank(fin["longdeb_to_debt"])


@register_factor(
    name="current_debt_ratio",
    description="流动负债/总负债截面排名 (高流动负债占比=再融资风险高，排后, 负向)",
    category="quality",
    thesis="流动负债占比较高意味着企业频繁面临再融资压力——在经济下行或信贷紧缩时风险加剧",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def current_debt_ratio(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["currentdebt_to_debt"])
    return cross_sectional_rank(-fin["currentdebt_to_debt"])


@register_factor(
    name="ocf_to_sales_rank",
    description="经营现金流/营业收入截面排名",
    category="quality",
    thesis="经营现金流占营收比重是现金流质量的核心——高占比意味着每元收入有更多现金流入",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def ocf_to_sales_rank(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["ocf_to_or"])
    return cross_sectional_rank(fin["ocf_to_or"])


@register_factor(
    name="op_to_debt_rank",
    description="营业利润/总负债截面排名",
    category="quality",
    thesis="营业利润对总负债的覆盖反映通过主营业务盈利偿还全部债务的能力",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def op_to_debt_rank(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["op_to_debt"])
    return cross_sectional_rank(fin["op_to_debt"])


@register_factor(
    name="eqt_to_debt_rank",
    description="净资产/总负债截面排名 (高=偿债安全垫厚)",
    category="quality",
    thesis="净资产相对负债的比例是资产负债表偿债能力的核心——净资产是债务的最后安全垫",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def eqt_to_debt_rank(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["eqt_to_debt"])
    return cross_sectional_rank(fin["eqt_to_debt"])


@register_factor(
    name="long_debt_to_working_capital",
    description="长期负债/营运资本截面排名 (高=长期负债挤压运营资金，排后, 负向)",
    category="quality",
    thesis="长期负债相对营运资本的比例衡量债务期限结构对运营资金的压力——高于1意味着长期债务吞噬了运营资金",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def long_debt_to_working_capital(ctx: FactorContext) -> pd.Series:
    fin = ctx.load_financial("financial_indicator.parquet", value_cols=["longdebt_to_workingcapital"])
    return cross_sectional_rank(-fin["longdebt_to_workingcapital"])


@register_factor(
    name="n_cashflow_stability_8q",
    description="经营现金流净额8季度稳定性因子 (变异系数取负, 稳定排前)",
    category="quality",
    thesis="经营现金流净额的稳定性是商业模式可预测性的关键度量——稳定意味着内生造血能力持续",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def n_cashflow_stability_8q(ctx: FactorContext) -> pd.Series:
    cf = ctx.load_financial("cashflow.parquet", value_cols=["n_cashflow_act"])
    s = cf["n_cashflow_act"]
    s_std = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).std())
    s_mean = s.groupby(level="Code").transform(lambda x: x.rolling(8, min_periods=4).mean())
    cv = safe_divide(s_std, s_mean.abs() + 1e-8)
    return cross_sectional_rank(-cv)
