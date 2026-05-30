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

# ── Credit impairment risk ──────────────────────────────────────────────

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


# ── Investment intensity ────────────────────────────────────────────────

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


# ── EPS rank ────────────────────────────────────────────────────────────

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


# ── Lease leverage ──────────────────────────────────────────────────────

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


# ── Contract assets ratio ───────────────────────────────────────────────

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


# ── Deferred tax liability ratio ────────────────────────────────────────

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


# ── Operating leverage ──────────────────────────────────────────────────

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


# ── Intangible amortization burden ────────────────────────────────────────

@register_factor(
    name="amortization_burden",
    description="摊销负担因子，(无形资产摊销+长期待摊费用)/营收截面排名（高摊销占比=资产沉没成本高，排后）。",
    category="financial",
    thesis="无形资产摊销和长期待摊费用摊销代表过去资本化的沉没成本正在被消耗，高的摊销/营收比意味着过去的大额资本化正在侵蚀当期利润，也限制未来盈利弹性。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_amortization_burden(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["amort_intang_assets", "lt_amort_deferred_exp"],
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    amort = cf["amort_intang_assets"].fillna(0) + cf["lt_amort_deferred_exp"].fillna(0)
    ratio = amort / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Advance receipts quality ─────────────────────────────────────────────

@register_factor(
    name="advance_receipts_ratio",
    description="预收款项/营收因子截面排名（高预收款=客户预付款意愿强=需求确定性高，排前）。",
    category="financial",
    thesis="预收款项代表客户提前支付但尚未确认为收入的款项——高预收款意味着产品/服务供不应求，客户愿意提前付款锁定产能，是极强的需求验证信号。与contract_liab互补：一个是传统的预收款项科目，一个是新准则下的合同负债。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_advance_receipts_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["adv_receipts"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = bs["adv_receipts"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Payables leverage ────────────────────────────────────────────────────

@register_factor(
    name="payables_power",
    description="应付账款/营业成本因子截面排名（高应付款占比=对上游议价能力强，排前）。",
    category="financial",
    thesis="应付账款占营业成本的比例反映企业对供应商的议价能力——高占比意味着企业占用供应商资金周期长，无息负债带来的经营杠杆提升资本效率。这是OPM(Other People's Money)策略的量化表达。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_payables_power(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["acct_payable", "notes_payable"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["oper_cost"]
    )
    payables = bs["acct_payable"].fillna(0) + bs["notes_payable"].fillna(0)
    ratio = payables / inc["oper_cost"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Interest burden ──────────────────────────────────────────────────────

@register_factor(
    name="interest_burden",
    description="利息负担因子，财务费用/营业利润截面排名（高财务费用侵蚀利润=排后）。",
    category="financial",
    thesis="财务费用（主要是利息支出）与营业利润的比例反映债务的利润侵蚀程度——高利息负担意味着大量利润用于支付债权人而非归属股东，在经济下行期利率上升+利润下降可能形成双重挤压。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_interest_burden(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["fin_exp", "operate_profit"],
    )
    ratio = inc["fin_exp"] / inc["operate_profit"].replace(0, np.nan)
    ratio = ratio.clip(-1, 3)
    return cross_sectional_rank(-ratio)


# ── Admin expense ratio ──────────────────────────────────────────────────

@register_factor(
    name="admin_expense_ratio",
    description="管理费用率因子，管理费用/营收截面排名（高管理费用率=管理效率低，排后）。",
    category="financial",
    thesis="管理费用率反映企业管理效率——高管理费用率意味着臃肿的管理层和过多行政支出，低费用率代表精简高效的管理结构。与expense_control互补：expense_control看三费总和，admin_expense_ratio专注于管理效率维度。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_admin_expense_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["admin_exp", "revenue"],
    )
    ratio = inc["admin_exp"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Sell expense ratio ───────────────────────────────────────────────────

@register_factor(
    name="sell_expense_ratio",
    description="销售费用率因子，销售费用/营收截面排名（高销售费用率=获客成本高=竞争激烈，排后）。",
    category="financial",
    thesis="销售费用率反映市场竞争激烈程度和客户获取成本——高销售费用率意味着公司需要大量投入营销/渠道/广告维持收入，品牌力或产品力可能不足。但在消费品行业，适度的销售费用投资可能带来未来的品牌资产。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_sell_expense_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["sell_exp", "revenue"],
    )
    ratio = inc["sell_exp"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Fair value exposure ──────────────────────────────────────────────────

@register_factor(
    name="fair_value_dependency",
    description="公允价值变动依赖因子，公允价值变动损益/营业利润的绝对值截面排名（高依赖=利润波动风险大，排后）。",
    category="financial",
    thesis="公允价值变动损益是非现金的未实现损益，高度依赖市场行情和估值模型假设——公允价值变动占利润比重过高意味着利润质量差、波动大、可预测性低，尤其在金融市场剧烈波动时容易出现大幅回撤。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_fair_value_dependency(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["fv_value_chg_gain", "operate_profit"],
    )
    ratio = inc["fv_value_chg_gain"].abs() / inc["operate_profit"].abs().replace(0, np.nan)
    ratio = ratio.clip(0, 2)
    return cross_sectional_rank(-ratio)


# ── Forex exposure ───────────────────────────────────────────────────────

@register_factor(
    name="forex_exposure",
    description="汇兑风险暴露因子，汇兑收益/营业利润的绝对值截面排名（高风险暴露排后）。",
    category="financial",
    thesis="汇兑损益敞口大的公司面临汇率波动风险——人民币升值周期中出口企业利润受压，贬值周期中进口成本上升。高汇兑敞口意味着盈利可预测性被宏观因素稀释。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_forex_exposure(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["forex_gain", "operate_profit"],
    )
    ratio = inc["forex_gain"].abs() / inc["operate_profit"].abs().replace(0, np.nan)
    ratio = ratio.clip(0, 2)
    return cross_sectional_rank(-ratio)


# ── Notes receivable ratio ───────────────────────────────────────────────

@register_factor(
    name="notes_receivable_ratio",
    description="应收票据/总资产因子截面排名（高应收票据占比排前=票据信用优于应收账款）。",
    category="financial",
    thesis="应收票据（尤其是银行承兑汇票）的信用风险远低于应收账款——高应收票据占比意味着回款更有保障、坏账风险更低。与receivable_pressure互补：一个看应收总量的风险，一个看应收结构中的优质比例。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_notes_receivable_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["notes_receiv", "total_assets"],
    )
    ratio = bs["notes_receiv"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Other receivables risk ────────────────────────────────────────────────

@register_factor(
    name="other_receivables_risk",
    description="其他应收款/总资产因子截面排名（高其他应收款=关联方占款风险，排后）。",
    category="financial",
    thesis="其他应收款往往是关联方资金占用、非经营性往来款的通道——高其他应收款占比是大股东资金占用的经典预警信号，是财务暴雷的常见前兆（如康得新、康美药业案例）。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_other_receivables_risk(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["oth_receiv", "total_assets"],
    )
    ratio = bs["oth_receiv"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Payroll burden ───────────────────────────────────────────────────────

@register_factor(
    name="payroll_burden",
    description="人工成本负担因子，应付职工薪酬/营收截面排名。",
    category="financial",
    thesis="应付职工薪酬占营收比例反映企业的人力成本刚性——高人工占比意味成本结构偏固定、经营杠杆高，在收入下滑时利润恶化更快。但科技行业中高人工占比也可能意味着人才密集型的核心竞争壁垒。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_payroll_burden(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["payroll_payable"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = bs["payroll_payable"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Taxes payable ratio ──────────────────────────────────────────────────

@register_factor(
    name="taxes_payable_ratio",
    description="应交税费/营收因子截面排名（高税费占比=税务合规性强=盈利真实性高，排前）。",
    category="financial",
    thesis="应交税费相对营收的比例是盈利真实性的旁证——持续有大量应交税费意味着公司有实际的应税收入，造假公司通常难以同时伪造税务缴纳记录。这是通过税务维度识别财务造假的简单有效指标。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_taxes_payable_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["taxes_payable"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = bs["taxes_payable"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Dividend payable ratio ───────────────────────────────────────────────

@register_factor(
    name="dividend_payable_ratio",
    description="应付股利/净资产因子截面排名（高应付股利=积极回报股东，排前）。",
    category="financial",
    thesis="应付股利（已宣告未发放的股利）代表公司对股东的现金回报承诺——高应付股利占比意味着公司有分红意愿和能力，是对股东友好的信号，也是盈利真实性的间接验证。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_dividend_payable_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["div_payable", "total_hldr_eqy_exc_min_int"],
    )
    ratio = bs["div_payable"] / bs["total_hldr_eqy_exc_min_int"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Cash paid to employees ───────────────────────────────────────────────

@register_factor(
    name="labor_intensity",
    description="劳动力密集度因子，(支付给职工现金)/营收截面排名（高劳动力密集度排后）。",
    category="financial",
    thesis="支付给职工的现金占营收比重反映劳动力密集度——高占比意味着人工成本是主要成本驱动因素，在劳动力成本上升或社保政策收紧时面临更大的利润压缩风险。轻资产+低劳动力密集是最优的商业模式特征。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_labor_intensity(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_paid_to_for_empl"],
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = cf["c_paid_to_for_empl"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Tax paid ratio ──────────────────────────────────────────────────────

@register_factor(
    name="tax_burden",
    description="税负因子，支付的各项税费/营收截面排名（高税负排后）。",
    category="financial",
    thesis="实际支付的税费占营收比重反映企业综合税负——高税负意味着政府占有更多企业创造的价值，留给股东的更少。但税负过低可能来自亏损或税收优惠，需结合盈利能力综合判断。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_tax_burden(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_paid_for_taxes"],
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = cf["c_paid_for_taxes"] / inc["revenue"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Retained earnings quality ────────────────────────────────────────────

@register_factor(
    name="retained_earnings_ratio",
    description="留存收益/净资产因子截面排名（高留存=持续盈利积累=质量信号，排前）。",
    category="financial",
    thesis="留存收益（未分配利润+盈余公积）是企业在扣除分红后累计留存的利润——高留存收益/净资产意味着企业长期盈利能力强、持续为股东创造价值，且留存资金可用于未来再投资或分红。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_retained_earnings_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["undistr_porfit", "surplus_rese", "total_hldr_eqy_exc_min_int"],
    )
    retained = bs["undistr_porfit"].fillna(0) + bs["surplus_rese"].fillna(0)
    ratio = retained / bs["total_hldr_eqy_exc_min_int"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Capital reserve ratio ────────────────────────────────────────────────

@register_factor(
    name="capital_reserve_ratio",
    description="资本公积/净资产因子截面排名（高资本公积=历史溢价融资=市场认可，排前）。",
    category="financial",
    thesis="资本公积主要来自IPO和增发中的股本溢价——高资本公积/净资产意味着公司历史上以高溢价成功融资，反映资本市场对公司价值的认可，也为未来股本转增提供了空间。但需注意高资本公积+低留存收益可能意味着融资后盈利能力未跟上。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_capital_reserve_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["cap_rese", "total_hldr_eqy_exc_min_int"],
    )
    ratio = bs["cap_rese"] / bs["total_hldr_eqy_exc_min_int"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Total borrowings structure ───────────────────────────────────────────

@register_factor(
    name="total_borrowings_ratio",
    description="总借款/总资产因子截面排名（高借款占比排后）。",
    category="financial",
    thesis="短期借款+长期借款+应付债券合计占总资产的比例是更全面的债务负担衡量——覆盖了银行借款和债券融资两类主要付息债务，比单独看短期借款或资产负债率更精确地衡量企业的主动融资杠杆。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_total_borrowings_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["st_borr", "lt_borr", "bond_payable", "total_assets"],
    )
    total_borrow = bs["st_borr"].fillna(0) + bs["lt_borr"].fillna(0) + bs["bond_payable"].fillna(0)
    ratio = total_borrow / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Net debt ratio ───────────────────────────────────────────────────────

@register_factor(
    name="net_debt_ratio",
    description="净负债/净资产因子截面排名（高净负债排后）。",
    category="financial",
    thesis="（有息负债-货币资金）/净资产是剔除现金后的真实杠杆水平——手头现金充裕但借了大量债的公司可能是在做套息交易或储备流动性，净负债能更准确衡量真实的财务杠杆。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_net_debt_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["st_borr", "lt_borr", "bond_payable", "money_cap", "total_hldr_eqy_exc_min_int"],
    )
    total_debt = bs["st_borr"].fillna(0) + bs["lt_borr"].fillna(0) + bs["bond_payable"].fillna(0)
    net_debt = total_debt - bs["money_cap"].fillna(0)
    ratio = net_debt / bs["total_hldr_eqy_exc_min_int"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Inventory turnover days change ───────────────────────────────────────

@register_factor(
    name="inventory_efficiency_change",
    description="存货效率变化因子，营业收入/(存货+1)的年度同比变化截面排名（存货效率改善排前）。",
    category="financial",
    thesis="存货效率（营收/存货）的同比变化反映库存管理改善或恶化趋势——效率提升意味着去库存顺利、销售加速；效率下降可能意味着滞销、库存积压，是盈利下行的前瞻信号。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_inventory_efficiency_change(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["inventories"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    inv_eff = inc["revenue"] / bs["inventories"].replace(0, np.nan)
    chg = inv_eff.groupby(level="Code").transform(lambda s: s.diff(4))
    return cross_sectional_rank(chg)


# ── Other payables ratio ─────────────────────────────────────────────────

@register_factor(
    name="other_payables_risk",
    description="其他应付款/总资产因子截面排名（高其他应付款=潜在表外负债，排后）。",
    category="financial",
    thesis="其他应付款科目常常是大股东或关联方的资金往来通道——高占比可能意味着大股东垫付但尚未结算的资金或潜在的表外承诺，也可能隐藏未披露的对外担保和关联交易。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_other_payables_risk(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["oth_payable", "total_assets"],
    )
    ratio = bs["oth_payable"] / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Interest income ratio ─────────────────────────────────────────────────

@register_factor(
    name="interest_income_ratio",
    description="利息收入/营业利润因子截面排名（高利息收入=资金充裕=财务弹性高，排前）。",
    category="financial",
    thesis="利息收入占营业利润比重高意味着公司持有大量可产生利息的金融资产——这类公司通常现金充裕、财务弹性强，在经济下行或信贷紧缩时有更强的抗风险能力。但也需关注是否主营不振转而依赖理财收益。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_interest_income_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["int_income", "operate_profit"],
    )
    ratio = inc["int_income"] / inc["operate_profit"].replace(0, np.nan)
    ratio = ratio.clip(-0.5, 2)
    return cross_sectional_rank(ratio)


# ── Financing cashflow sustainability ────────────────────────────────────

@register_factor(
    name="equity_vs_debt_financing",
    description="股权vs债权融资偏好因子，(吸收投资收到现金-偿还债务支付现金)/总资产截面排名（正=偏股权融资=财务稳健排前）。",
    category="financial",
    thesis="企业的外部融资结构反映融资策略的审慎性——偏好股权融资（增发、配股）意味着降低杠杆风险但稀释现有股东权益；偏好债务融资意味着保持股权集中但也增加财务风险。该比率衡量净融资中股权与债权的相对力度。",
    dependencies=("cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_equity_vs_debt_financing(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_recp_cap_contrib", "c_prepay_amt_borr"],
    )
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["total_assets"]
    )
    net_financing = cf["c_recp_cap_contrib"].fillna(0) - cf["c_prepay_amt_borr"].fillna(0)
    ratio = net_financing / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(ratio)
