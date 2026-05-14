from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── Balance sheet factors ───────────────────────────────────────────────

@register_factor(
    name="goodwill_risk",
    description="商誉风险因子，商誉/净资产截面排名（高商誉占比排后=风险信号）。",
    category="financial",
    thesis="高商誉占比在减值测试中面临巨大风险，是财务暴雷的重要预警信号。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_goodwill_risk(context: FactorContext):
    bs = context.repo.load_financial_panel(
        "balancesheet.parquet",
        value_cols=["goodwill", "total_hldr_eqy_exc_min_int"],
    )
    ratio = bs["goodwill"] / bs["total_hldr_eqy_exc_min_int"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="inventory_pressure",
    description="存货压力因子，存货/总资产截面排名（高占比排后）。",
    category="financial",
    thesis="存货占比过高意味着资金被库存占用，在需求下行时面临减值风险。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_inventory_pressure(context: FactorContext):
    bs = context.repo.load_financial_panel(
        "balancesheet.parquet",
        value_cols=["inventories", "total_assets"],
    )
    ratio = bs["inventories"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="receivable_pressure",
    description="应收账款压力因子，应收账款/总资产截面排名（高占比排后）。",
    category="financial",
    thesis="高应收账款占比意味着回款能力弱、坏账风险大，是利润含金量低的信号。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_receivable_pressure(context: FactorContext):
    bs = context.repo.load_financial_panel(
        "balancesheet.parquet",
        value_cols=["accounts_receiv", "total_assets"],
    )
    ratio = bs["accounts_receiv"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="fix_asset_ratio",
    description="固定资产占比因子，固定资产/总资产截面排名（轻资产排前）。",
    category="financial",
    thesis="轻资产模式通常意味着更高的运营灵活性和更低的固定成本，长期表现更优。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_fix_asset_ratio(context: FactorContext):
    bs = context.repo.load_financial_panel(
        "balancesheet.parquet",
        value_cols=["fix_assets", "total_assets"],
    )
    ratio = bs["fix_assets"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Income statement factors ────────────────────────────────────────────

@register_factor(
    name="operating_profit_purity",
    description="经营利润纯度因子，营业利润/利润总额截面排名。",
    category="financial",
    thesis="营业利润占比越高说明利润来自主营业务而非一次性收益，盈利质量更好。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_operating_profit_purity(context: FactorContext):
    inc = context.repo.load_financial_panel(
        "income.parquet",
        value_cols=["operate_profit", "total_profit"],
    )
    ratio = inc["operate_profit"] / inc["total_profit"].replace(0, np.nan)
    # Winsorize extreme values
    ratio = ratio.clip(-1, 2)
    return cross_sectional_rank(ratio)


@register_factor(
    name="rd_intensity",
    description="研发强度因子，研发费用/营业收入截面排名。",
    category="financial",
    thesis="适度的高研发投入代表创新驱动的成长潜力。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_rd_intensity(context: FactorContext):
    inc = context.repo.load_financial_panel(
        "income.parquet",
        value_cols=["rd_exp", "revenue"],
    )
    ratio = inc["rd_exp"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


@register_factor(
    name="expense_control",
    description="费用控制因子，(销售+管理+财务费用)/营收截面排名（低费用排前）。",
    category="financial",
    thesis="三费占比低代表轻运营模式或高效管理，费用控制是盈利能力的重要支撑。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_expense_control(context: FactorContext):
    inc = context.repo.load_financial_panel(
        "income.parquet",
        value_cols=["sell_exp", "admin_exp", "fin_exp", "revenue"],
    )
    total_exp = inc["sell_exp"].fillna(0) + inc["admin_exp"].fillna(0) + inc["fin_exp"].fillna(0)
    ratio = total_exp / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="invest_income_reliance",
    description="投资收益依赖度因子，投资收益/营业利润截面排名（高依赖度排后）。",
    category="financial",
    thesis="高投资收益依赖意味着主业盈利能力弱、业绩波动大，可持续性差。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_invest_income_reliance(context: FactorContext):
    inc = context.repo.load_financial_panel(
        "income.parquet",
        value_cols=["invest_income", "operate_profit"],
    )
    ratio = inc["invest_income"] / inc["operate_profit"].replace(0, np.nan)
    ratio = ratio.clip(-5, 5)
    return cross_sectional_rank(-ratio)


# ── Cashflow factors ────────────────────────────────────────────────────

@register_factor(
    name="fcf_yield",
    description="自由现金流收益率因子，FCF/总市值（日频前向填充FCF）截面排名。",
    category="financial",
    thesis="自由现金流是股东可支配的现金回报，高FCF收益率兼具价值与质量属性。",
    dependencies=("cashflow.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_fcf_yield(context: FactorContext):
    cf = context.repo.load_financial_panel(
        "cashflow.parquet",
        value_cols=["free_cashflow"],
    )
    finance = context.load("finance.parquet")
    # Align on index
    fcf = cf["free_cashflow"]
    mkt_cap = finance["total_mv"]
    common = fcf.index.intersection(mkt_cap.index)
    fcf_y = fcf.loc[common] / mkt_cap.loc[common].replace(0, np.nan)
    return cross_sectional_rank(fcf_y)


@register_factor(
    name="ncf_act_to_revenue",
    description="经营现金流/营收因子截面排名。",
    category="financial",
    thesis="经营现金流相对营收的比例是现金流质量的核心指标，高比例代表健康的现金循环。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ncf_act_to_revenue(context: FactorContext):
    cf = context.repo.load_financial_panel(
        "cashflow.parquet",
        value_cols=["n_cashflow_act", "c_inf_fr_operate_a"],
    )
    ratio = cf["n_cashflow_act"] / cf["c_inf_fr_operate_a"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


@register_factor(
    name="financing_dependency",
    description="融资依赖度因子，筹资现金流/经营现金流（负值=依赖外部融资，排后）。",
    category="financial",
    thesis="持续依赖外部融资的企业面临再融资风险，内生现金流充足的企业更稳健。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_financing_dependency(context: FactorContext):
    cf = context.repo.load_financial_panel(
        "cashflow.parquet",
        value_cols=["n_cash_flows_fnc_act", "n_cashflow_act"],
    )
    ratio = cf["n_cash_flows_fnc_act"] / cf["n_cashflow_act"].replace(0, np.nan)
    ratio = ratio.clip(-3, 3)
    return cross_sectional_rank(-ratio)


# ── Shareholder concentration ───────────────────────────────────────────

@register_factor(
    name="holder_num_change",
    description="股东户数变化率因子，股东户数季度环比变化截面排名（减少=筹码集中，排前）。",
    category="financial",
    thesis="股东户数减少代表筹码从散户向机构集中，是经典的筹码集中度信号。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_change(context: FactorContext):
    hn = context.repo.load_financial_panel(
        "holder_number.parquet",
        value_cols=["holder_num"],
    )
    holder = hn["holder_num"]
    chg = holder.groupby(level="Code").transform(lambda s: s.pct_change(63))
    return cross_sectional_rank(-chg)


# ── Pledge risk ─────────────────────────────────────────────────────────

@register_factor(
    name="pledge_ratio",
    description="股权质押比例因子截面排名（高质押比例排后=风险信号）。",
    category="financial",
    thesis="高质押比例在大幅下跌时面临强制平仓风险，是潜在的黑天鹅信号。",
    dependencies=("pledge_stat.parquet", "calendar.parquet"),
)
def factor_pledge_ratio(context: FactorContext):
    ps = context.repo.load_financial_panel(
        "pledge_stat.parquet",
        value_cols=["pledge_ratio"],
        date_col="end_date",
    )
    return cross_sectional_rank(-ps["pledge_ratio"])


# ── Cash conversion cycle ─────────────────────────────────────────────────

@register_factor(
    name="cash_conversion_cycle",
    description="现金转换周期因子，存货周转天数+应收周转天数-应付周转天数截面排名（短周期排前）。",
    category="financial",
    thesis="CCC越短说明企业从投入资金到收回现金的周期越短、运营效率越高，短周期企业现金流更健康。",
    dependencies=("financial_indicator.parquet", "balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_cash_conversion_cycle(context: FactorContext):
    fin = context.repo.load_financial_panel(
        "financial_indicator.parquet",
        value_cols=["invturn_days", "arturn_days"],
    )
    bs = context.repo.load_financial_panel(
        "balancesheet.parquet",
        value_cols=["acct_payable"],
    )
    inc = context.repo.load_financial_panel(
        "income.parquet",
        value_cols=["oper_cost"],
    )

    inv_days = fin["invturn_days"]
    ar_days = fin["arturn_days"]
    ap = bs["acct_payable"]
    oper_cost = inc["oper_cost"]

    # AP turnover days = AP / (oper_cost / 365)
    apturn_days = ap / (oper_cost / 365).replace(0, np.nan)

    # Align on common index
    common = inv_days.index.intersection(ar_days.index).intersection(apturn_days.index)
    ccc = inv_days.loc[common] + ar_days.loc[common] - apturn_days.loc[common]
    ccc = ccc.clip(-1000, 1000)
    return cross_sectional_rank(-ccc)


# ── Gross margin stability ────────────────────────────────────────────────

@register_factor(
    name="gross_margin_stability_8q",
    description="过去8个季度毛利率标准差因子（负值：低波动排前）。",
    category="financial",
    thesis="毛利率稳定性是公司盈利模式稳定、竞争壁垒牢靠的信号，波动剧烈的毛利率意味着经营风险高。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_gross_margin_stability_8q(context: FactorContext):
    fin = context.repo.load_financial_panel(
        "financial_indicator.parquet",
        value_cols=["q_gsprofit_margin"],
    )
    gm = fin["q_gsprofit_margin"]
    # 8 quarters ~ 504 trading days
    gm_std = gm.groupby(level="Code").transform(
        lambda s: s.rolling(504, min_periods=126).std()
    )
    return cross_sectional_rank(-gm_std)
