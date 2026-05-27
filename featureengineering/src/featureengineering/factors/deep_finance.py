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
    bs = context.load_financial(
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
    bs = context.load_financial(
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
    bs = context.load_financial(
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
    bs = context.load_financial(
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
    inc = context.load_financial(
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
    inc = context.load_financial(
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
    inc = context.load_financial(
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
    inc = context.load_financial(
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
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["free_cashflow"],
    )
    finance = context.load("finance.parquet")
    fcf = cf["free_cashflow"]
    mkt_cap = finance["total_mv"]
    fcf_y = fcf / mkt_cap.replace(0, np.nan)
    return cross_sectional_rank(fcf_y)


@register_factor(
    name="ncf_act_to_revenue",
    description="经营现金流/营收因子截面排名。",
    category="financial",
    thesis="经营现金流相对营收的比例是现金流质量的核心指标，高比例代表健康的现金循环。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ncf_act_to_revenue(context: FactorContext):
    cf = context.load_financial(
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
    cf = context.load_financial(
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
    hn = context.load_financial(
        "holder_number.parquet",
        value_cols=["holder_num"],
    )
    holder = hn["holder_num"]
    chg = holder.groupby(level="Code").transform(lambda s: s.pct_change(63, fill_method=None))
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
    ps = context.load_financial(
        "pledge_stat.parquet",
        value_cols=["pledge_ratio"],
        date_col="end_date",
    )
    return cross_sectional_rank(-ps["pledge_ratio"])





# ── REMOVED: gross_margin_stability_8q — uses q_gsprofit_margin which is
# zero-filled 2019-2022. Do not re-add without vendor data backfill.
# See: memory/vendor-data-quality.md


# ── Asset structure: balancesheet 扩展 ───────────────────────────────────

@register_factor(
    name="contract_liab_ratio",
    description="合同负债/营收因子（预收款质量）截面排名。",
    category="financial",
    thesis="合同负债代表客户预付款，高占比意味着对下游议价能力强、收入可预见性高，是A股'茅台指标'——预收款充沛的公司盈利质量和确定性更优。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_contract_liab_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["contract_liab"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = bs["contract_liab"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


@register_factor(
    name="cash_to_assets",
    description="货币资金/总资产因子（现金充裕度）截面排名。",
    category="financial",
    thesis="货币资金占比高的公司财务弹性强，在经济下行期具有更强的抗风险能力和逆势扩张能力，同时现金充沛也是分红回购的潜在信号。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_cash_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["money_cap", "total_assets"]
    )
    ratio = bs["money_cap"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


@register_factor(
    name="short_term_debt_ratio",
    description="短期借款/总资产因子截面排名（高短期负债占比排后）。",
    category="financial",
    thesis="短期借款占比过高意味着企业依赖短期融资，面临再融资滚动压力和利率波动风险，债务期限结构越短、财务脆弱性越高。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_short_term_debt_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["st_borr", "total_assets"]
    )
    ratio = bs["st_borr"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="intan_assets_ratio",
    description="无形资产/总资产因子截面排名。",
    category="financial",
    thesis="无形资产（含专利权、商标权、软件著作权等）占比高代表知识资产密集，在科技和消费品牌领域是护城河的重要组成部分，但也需关注无形资产的质量和减值风险。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_intan_assets_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["intan_assets", "total_assets"]
    )
    ratio = bs["intan_assets"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Profit quality: income 扩展 ──────────────────────────────────────────

@register_factor(
    name="non_oper_profit_ratio",
    description="非经常性损益占比因子，非经常性损益/利润总额截面排名（高占比排后）。",
    category="financial",
    thesis="非经常性损益占比高说明利润主要来自一次性收益而非主营业务，盈利的可持续性和质量较差，是识别'注水利润'的重要指标。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_non_oper_profit_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["non_oper_income", "non_oper_exp", "total_profit"],
    )
    non_oper = inc["non_oper_income"].fillna(0) + inc["non_oper_exp"].fillna(0)
    ratio = non_oper.abs() / inc["total_profit"].abs().replace(0, np.nan)
    ratio = ratio.clip(0, 2)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="credit_impair_risk",
    description="信用减值损失/营业利润因子截面排名（高减值占比排后）。",
    category="financial",
    thesis="信用减值损失高意味着应收账款或贷款面临较大坏账风险，是资产质量的负面信号，尤其在经济下行期需要重点关注。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_credit_impair_risk(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["credit_impair_loss", "operate_profit"],
    )
    ratio = inc["credit_impair_loss"].abs() / inc["operate_profit"].abs().replace(0, np.nan)
    ratio = ratio.clip(0, 1)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="assets_impair_risk",
    description="资产减值损失/营业利润因子截面排名（高减值占比排后）。",
    category="financial",
    thesis="资产减值损失高意味着固定资产、存货、商誉等面临减值压力，是资产质量的负面信号，尤其商誉减值集中在年报期需要警惕。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_assets_impair_risk(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["assets_impair_loss", "operate_profit"],
    )
    ratio = inc["assets_impair_loss"].abs() / inc["operate_profit"].abs().replace(0, np.nan)
    ratio = ratio.clip(0, 1)
    return cross_sectional_rank(-ratio)


# ── Cashflow depth: cashflow 扩展 ────────────────────────────────────────

@register_factor(
    name="investment_intensity",
    description="投资活动现金流净额/总资产因子截面排名。",
    category="financial",
    thesis="投资活动现金流净额反映企业资本开支和对外投资力度，适度投资是成长的基础，但过度投资（大额净流出）可能意味着盲目扩张和未来减值风险。",
    dependencies=("cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_investment_intensity(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["n_cashflow_inv_act"]
    )
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["total_assets"]
    )
    ratio = cf["n_cashflow_inv_act"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


@register_factor(
    name="depr_amort_to_revenue",
    description="折旧摊销/营收因子截面排名（高占比排后）。",
    category="financial",
    thesis="折旧摊销占营收比重反映企业的资本密集度，高占比意味着维持现有业务需要大量资本支出，是'重资产'的量化指标，轻资产模式长期表现更优。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_depr_amort_to_revenue(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["depr_fa_coga_dpba", "amort_intang_assets"],
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    depr_amort = cf["depr_fa_coga_dpba"].fillna(0) + cf["amort_intang_assets"].fillna(0)
    ratio = depr_amort / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Holder number extensions ──────────────────────────────────────────────

@register_factor(
    name="holder_num_change_yoy",
    description="股东户数年度变化率因子，-holder_num年度环比截面排名。",
    category="financial",
    thesis="年度股东户数变化过滤季报波动，捕捉中长期筹码集中趋势。持续减少=机构持续吸筹。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_change_yoy(context: FactorContext):
    hn = context.load_financial(
        "holder_number.parquet", value_cols=["holder_num"], date_col="ann_date"
    )
    change = -hn["holder_num"].groupby(level="Code").transform(lambda s: s.pct_change(4))
    return cross_sectional_rank(change)


@register_factor(
    name="avg_holding_mv",
    description="户均持股市值因子，circ_mv/holder_num截面排名。",
    category="financial",
    thesis="户均持股市值高=机构化程度高/大户主导，信息效率更高，波动率更低。",
    dependencies=("holder_number.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_avg_holding_mv(context: FactorContext):
    hn = context.load_financial(
        "holder_number.parquet", value_cols=["holder_num"], date_col="ann_date"
    )
    finance = context.load("finance.parquet")
    circ_mv = finance["circ_mv"]
    hn_s = hn["holder_num"]
    common = hn_s.index.intersection(circ_mv.index)
    ratio = circ_mv.loc[common] / hn_s.loc[common].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Per-share metrics ─────────────────────────────────────────────────────

@register_factor(
    name="eps_rank",
    description="每股收益(EPS)因子截面排名。",
    category="quality",
    thesis="EPS是最基础的盈利能力指标，高EPS公司通常具备持续竞争优势。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_eps_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["eps"]
    )
    return cross_sectional_rank(fin["eps"])


@register_factor(
    name="ocfps_rank",
    description="每股经营现金流(OCFPS)因子截面排名。",
    category="quality",
    thesis="每股经营现金流反映真实的现金创造能力，比EPS更难被会计操纵，含金量更高。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ocfps_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocfps"]
    )
    return cross_sectional_rank(fin["ocfps"])


# ── Holder concentration depth ───────────────────────────────────────────

@register_factor(
    name="holder_num_percentile",
    description="股东户数历史分位因子，当前股东户数在过去2年滚动窗口中的百分位截面排名（低分位=户数处于历史低位=筹码高度集中，排前）。",
    category="financial",
    thesis="股东户数在历史上的分位位置比绝对变化量更能反映筹码集中度的极端程度。建仓过程中股东户数持续下降，当达到2年最低分位时意味着筹码已极度集中到机构手中，后续拉升阻力最小。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_percentile(context: FactorContext):
    hn = context.load_financial(
        "holder_number.parquet", value_cols=["holder_num"]
    )
    holder = hn["holder_num"]

    def _rolling_pct(s):
        return s.rolling(504, min_periods=126).apply(
            lambda x: (x < x.iloc[-1]).mean(), raw=False
        )

    pct = holder.groupby(level="Code").transform(_rolling_pct)
    return cross_sectional_rank(-pct)


# ── Pledge risk change ───────────────────────────────────────────────────

@register_factor(
    name="pledge_ratio_change",
    description="股权质押比例半年变化因子，质押比例的6个月环比变化截面排名（增加=质押风险上升，排后）。",
    category="financial",
    thesis="质押比例的边际变化比绝对水平更具预警意义——快速增加的质押意味着大股东资金链可能正在紧张、急需补充质押物或借新还旧，是风险加速暴露的信号。",
    dependencies=("pledge_stat.parquet", "calendar.parquet"),
)
def factor_pledge_ratio_change(context: FactorContext):
    ps = context.load_financial(
        "pledge_stat.parquet",
        value_cols=["pledge_ratio"],
        date_col="end_date",
    )
    pledge = ps["pledge_ratio"]
    chg = pledge.groupby(level="Code").transform(lambda s: s.diff(2))
    return cross_sectional_rank(-chg)


# ── Lease leverage (off-balance-sheet) ───────────────────────────────────

@register_factor(
    name="lease_leverage",
    description="租赁杠杆因子，(使用权资产+租赁负债)/总资产截面排名（高租赁杠杆排后）。",
    category="financial",
    thesis="新租赁准则(IFRS 16/CAS 21)将经营租赁表内化后，使用权资产和租赁负债成为衡量企业表外杠杆的关键指标。高租赁杠杆公司在航空、零售、酒店等租赁密集行业真实负债被严重低估，是财务风险的新维度。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_lease_leverage(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["right_of_use_assets", "lease_liab", "total_assets"],
    )
    lease_assets = bs["right_of_use_assets"].fillna(0) + bs["lease_liab"].fillna(0)
    ratio = lease_assets / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Contract assets (potential revenue) ──────────────────────────────────

@register_factor(
    name="contract_assets_ratio",
    description="合同资产/总资产因子截面排名。",
    category="financial",
    thesis="合同资产代表企业已履约但尚未取得无条件收款权的资产，高合同资产占比意味着未来1-2期有确定性较高的收入转为应收账款，是收入前瞻指标。与合同负债互补——前者是'我已履约等收款'，后者是'我预收了款等履约'。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_contract_assets_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["contract_assets", "total_assets"],
    )
    ratio = bs["contract_assets"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Deferred tax × earnings quality ──────────────────────────────────────

@register_factor(
    name="deferred_tax_liab_ratio",
    description="递延所得税负债/总资产因子截面排名。",
    category="financial",
    thesis="递延所得税负债主要来自固定资产加速折旧与税法差异，高递延税负债占比通常意味着企业在积极进行资本投资（加速折旧抵税），是投资力度和税务筹划能力的间接信号。但需关注递延税来源的质量——来自亏损弥补的递延税资产可能意味着盈利可持续性问题。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_deferred_tax_liab_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["defer_tax_liab", "total_liab"],
    )
    ratio = bs["defer_tax_liab"] / bs["total_liab"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Accruals (Sloan 1996) ──────────────────────────────────────────────


@register_factor(
    name="accruals_ratio",
    description="应计利润因子，(经营现金流-净利润)/总资产截面排名（负向：高应计=盈利质量差）。",
    category="financial",
    thesis="Sloan(1996)应计利润异象——高应计利润企业（会计利润远超现金利润）未来收益显著走低。应计代表管理层会计选择和盈利操纵空间，高应计利润通常伴随后续反转。这是学术上验证最充分的盈利质量因子之一。",
    dependencies=("cashflow.parquet", "income.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_accruals_ratio(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["n_cashflow_act"],
    )
    inc = context.load_financial(
        "income.parquet",
        value_cols=["n_income"],
    )
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["total_assets"],
    )
    common = cf["n_cashflow_act"].index.intersection(inc["n_income"].index).intersection(
        bs["total_assets"].index
    )
    accruals = (
        cf["n_cashflow_act"].loc[common] - inc["n_income"].loc[common]
    ) / bs["total_assets"].loc[common].replace(0, np.nan)
    return cross_sectional_rank(-accruals)


# ── Operating leverage ─────────────────────────────────────────────────


@register_factor(
    name="operating_leverage",
    description="经营杠杆因子，(营业利润年增速)/(营业收入年增速)条件截面排名。",
    category="financial",
    thesis="经营杠杆衡量企业固定成本结构——高固定成本企业在收入增长时利润增速更快（正向放大），但在收入下滑时亏损加速恶化（负向放大）。条件方向：当收入同比增长时高杠杆排前（放大利润），收入下滑时高杠杆排后（放大亏损）。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_operating_leverage(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["operate_profit", "revenue"],
    )
    op = inc["operate_profit"]
    rev = inc["revenue"]
    op_yoy = op.groupby(level="Code").transform(lambda s: s.pct_change(252))
    rev_yoy = rev.groupby(level="Code").transform(lambda s: s.pct_change(252))
    with np.errstate(invalid="ignore"):
        dol = op_yoy / rev_yoy
        dol_signed = np.where(rev_yoy > 0, dol, -dol)
    dol_signed = pd.Series(dol_signed, index=op.index, name="operating_leverage")
    return cross_sectional_rank(dol_signed)


# ── Interest coverage trend ────────────────────────────────────────────


@register_factor(
    name="interest_coverage_change_4q",
    description="利息覆盖变化因子，(ebit/利息)的年度同比变化截面排名。",
    category="quality",
    thesis="利息覆盖率的同比变化比绝对值对信用风险恶化更敏感。恶化中的利息覆盖是财务困境的早期预警——许多违约案例中，利息覆盖恶化领先评级下调2-4个季度。与debt_to_assets互补：一个看债务存量的可持续性，一个看偿债能力的边际变化。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_interest_coverage_change_4q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["ebit_to_interest"],
    )
    coverage = fin["ebit_to_interest"]
    chg = coverage.groupby(level="Code").transform(lambda s: s.diff(4))
    return cross_sectional_rank(chg)


# ── Effective tax rate trend ───────────────────────────────────────────


@register_factor(
    name="tax_rate_change_4q",
    description="有效税率变化因子，(所得税/利润总额)的年度同比变化截面排名（负向：税率上升=利润压力）。",
    category="quality",
    thesis="有效税率的突然上升意味着税收优惠或亏损结转过期，是利润质量下降或监管压力上升的前瞻信号。税率持续下降的公司通常具有更积极的税务筹划能力，盈利可持续性更强。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_tax_rate_change_4q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["tax_to_ebt"],
    )
    tax = fin["tax_to_ebt"]
    chg = tax.groupby(level="Code").transform(lambda s: s.diff(4))
    return cross_sectional_rank(-chg)


# ── Debt maturity structure ────────────────────────────────────────────


@register_factor(
    name="long_term_debt_ratio",
    description="长期债务占比因子，长期借款/总负债截面排名。",
    category="financial",
    thesis="债务期限结构反映再融资风险——短期债务占比高=高频展期风险（已有short_term_debt_ratio覆盖），长期债务占比高=债务结构更稳定但利息成本可能更高。与short_term_debt_ratio互补，共同描绘完整的债务期限画像。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_long_term_debt_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["lt_borr", "total_liab"],
    )
    ratio = bs["lt_borr"] / bs["total_liab"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Minority interest ──────────────────────────────────────────────────


@register_factor(
    name="minority_interest_ratio",
    description="少数股东权益占比因子，(含少数股东权益-不含)/含少数股东权益截面排名（负向：高占比=利润归属稀释）。",
    category="financial",
    thesis="少数股东权益占比高意味着合并利润中有较大比例归属外部股东，归母净利润的可预测性和质量被稀释。高少数股东权益占比在并购频繁的企业中尤为常见——子公司利润被少数股东分享，母公司股东实际可得利润低于报表归母数。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_minority_interest_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["total_hldr_eqy_inc_min_int", "total_hldr_eqy_exc_min_int"],
    )
    minority = (
        bs["total_hldr_eqy_inc_min_int"] - bs["total_hldr_eqy_exc_min_int"]
    )
    ratio = minority / bs["total_hldr_eqy_inc_min_int"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Income statement effective tax rate ────────────────────────────────


@register_factor(
    name="effective_tax_by_ebt",
    description="利润表有效税率因子，(所得税费用)/(利润总额)截面排名（负向：高税率=盈利留存低）。",
    category="financial",
    thesis="利润表直接计算的所得税/利润总额比率，不同于预计算的tax_to_ebt指标，能捕捉非经常性税务调整（递延税转回、税务争议拨备等）。异常高的有效税率意味着利润被税务侵蚀严重，异常的税率意味着税务优惠依赖度高。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_effective_tax_by_ebt(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["income_tax", "total_profit"],
    )
    ratio = inc["income_tax"] / inc["total_profit"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)
