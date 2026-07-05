"""
Deep extended cash flow statement factors — Class 1 (Wave 2).

Second-wave factorisation of remaining unused cashflow.parquet fields covering
operating detail, investing/financing sub-items, supplementary indirect-method
adjustments, debt/equity conversions, and cash balance items.

Sections
--------
A. Operating Cash Detail       — other operating inflows/outflows, tax refunds, financing expense
B. Investing Cash Detail       — investment withdrawal, returns, asset disposal, M&A outflows
C. Financing Cash Detail       — bond issuance, other financing flows, dividend/interest paid
D. Supplementary (Indirect)    — depreciation, provisions, working capital changes, deferred tax
E. Conversions & Capital       — debt-to-equity swaps, bonds due within year, lease financing
F. Cash Balance Items          — cash equivalents, capital contributions, securities

Data source: ``cashflow.parquet`` (97 columns, quarterly via forward-fill)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    safe_divide,
)


# ═══════════════════════════════════════════════════════════════════════════
# A.  Operating Cash Detail
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_other_operating_cf_in_to_revenue",
    description="其他经营活动现金流入/营收比率因子，c_fr_oth_operate_a/revenue。",
    category="quality",
    thesis="其他经营活动现金流入占比高意味着经营现金流中非主营收入比例大——"
    "可能是押金、保证金、政府补助等非持续性来源。"
    "持续高比率需关注经营现金流的'含金量'。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_other_operating_cf_in_to_revenue(context: FactorContext):
    cf = context.load_financial("cashflow.parquet", value_cols=["c_fr_oth_operate_a"])
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    ratio = safe_divide(cf["c_fr_oth_operate_a"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_other_operating_cf_out_to_revenue",
    description="其他经营活动现金流出/营收比率因子，oth_cash_pay_oper_act/revenue。",
    category="quality",
    thesis="其他经营活动现金流出占比高可能意味着大量非主营支出——"
    "如保证金支出、往来款、罚款等。这是经营现金流中的'黑箱'，"
    "高比率需要结合附注信息判断合理性。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_other_operating_cf_out_to_revenue(context: FactorContext):
    cf = context.load_financial("cashflow.parquet", value_cols=["oth_cash_pay_oper_act"])
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    ratio = safe_divide(cf["oth_cash_pay_oper_act"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_total_operating_outflow_to_revenue",
    description="经营活动现金总流出/营收比率因子，st_cash_out_act/revenue。",
    category="quality",
    thesis="经营现金流出总额占营收比衡量企业的现金经营成本密度。"
    "比率>1意味着经营现金支出大于收入（烧钱经营），比率<0.7意味着"
    "较强的现金生成能力。该比率的趋势比绝对水平更有意义。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_total_operating_outflow_to_revenue(context: FactorContext):
    cf = context.load_financial("cashflow.parquet", value_cols=["st_cash_out_act"])
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    ratio = safe_divide(cf["st_cash_out_act"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_tax_refund_to_revenue",
    description="税费返还/营收比率因子，recp_tax_rends/revenue截面排名。",
    category="quality",
    thesis="税费返还占营收比重高意味着企业享受大量税收优惠或出口退税——"
    "可能是政策扶持信号，但也可能是利润对税收优惠的过度依赖（脆弱性）。"
    "税收优惠到期可能导致盈利断崖式下跌。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_tax_refund_to_revenue(context: FactorContext):
    cf = context.load_financial("cashflow.parquet", value_cols=["recp_tax_rends"])
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    ratio = safe_divide(cf["recp_tax_rends"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_financing_expense_cf_to_revenue",
    description="财务费用(CF口径)/营收比率因子，finan_exp/revenue截面排名。",
    category="quality",
    thesis="现金流量表补充资料中的财务费用（finan_exp）反映了实际支付的"
    "利息净额。与利润表财务费用（fin_exp）的差异来自利息资本化和应付利息变动。"
    "finan_exp>fin_exp意味着利息支付>计提——现金流出压力大于账面费用。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_financing_expense_cf_to_revenue(context: FactorContext):
    cf = context.load_financial("cashflow.parquet", value_cols=["finan_exp"])
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    ratio = safe_divide(cf["finan_exp"].abs(), inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Investing Cash Detail
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_investment_withdrawal_ratio",
    description="投资收回/投资流入比率因子，c_disp_withdrwl_invest/stot_inflows_inv_act。",
    category="quality",
    thesis="投资收回（撤回投资）占投资活动流入的比率反映了企业的投资组合管理策略。"
    "高比率意味着企业正在收缩投资（可能是回收资金或止损），"
    "低比率意味着投资组合相对稳定。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_investment_withdrawal_ratio(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_disp_withdrwl_invest", "stot_inflows_inv_act"],
    )
    ratio = safe_divide(cf["c_disp_withdrwl_invest"], cf["stot_inflows_inv_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_investment_return_ratio",
    description="投资收益现金回流/投资流入比率因子，c_recp_return_invest/stot_inflows_inv_act。",
    category="quality",
    thesis="投资收益收到的现金占投资活动流入比重反映投资的实际现金回报率。"
    "高比率意味着投资产生了实际现金回报（而非仅仅是账面收益）。"
    "低比率可能意味着投资未产生预期的现金回流。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_investment_return_ratio(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_recp_return_invest", "stot_inflows_inv_act"],
    )
    ratio = safe_divide(cf["c_recp_return_invest"], cf["stot_inflows_inv_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_fixed_asset_disposal_inflow",
    description="固定资产处置现金流入/投资流入比率因子。",
    category="quality",
    thesis="处置固定资产收到的现金占投资流入比重高可能意味着："
    "企业正在优化资产结构（正面）、或变卖资产度日（负面）。"
    "结合营收和利润趋势判断——如果营收也在增长则可能是设备更新换代。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_fixed_asset_disposal_inflow(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["n_recp_disp_fiolta", "stot_inflows_inv_act"],
    )
    ratio = safe_divide(cf["n_recp_disp_fiolta"], cf["stot_inflows_inv_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_subsidiary_disposal_inflow",
    description="子公司处置现金流入/投资流入比率因子。",
    category="quality",
    thesis="处置子公司收到的现金占投资流入比重大意味着企业正在进行业务重组——"
    "可能是'瘦身'聚焦主业（正面）、或出售优质资产自救（负面）。"
    "需要结合被处置子公司的业务性质判断。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_subsidiary_disposal_inflow(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["n_recp_disp_sobu", "stot_inflows_inv_act"],
    )
    ratio = safe_divide(cf["n_recp_disp_sobu"], cf["stot_inflows_inv_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_subsidiary_acquisition_outflow",
    description="并购子公司现金流出/投资流出比率因子，n_disp_subs_oth_biz/stot_out_inv_act。",
    category="quality",
    thesis="取得子公司支付的现金占投资流出比重高意味着企业正在进行"
    "大规模并购扩张。并购是双刃剑——成功的并购创造价值，"
    "失败的并购毁灭价值（商誉减值风险）。需关注并购后的整合效果。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_subsidiary_acquisition_outflow(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["n_disp_subs_oth_biz", "stot_out_inv_act"],
    )
    ratio = safe_divide(cf["n_disp_subs_oth_biz"], cf["stot_out_inv_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# C.  Financing Cash Detail
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_bond_issuance_ratio",
    description="债券融资依赖度因子，proc_issue_bonds/stot_cash_in_fnc_act截面排名。",
    category="quality",
    thesis="发行债券收到的现金占筹资流入比重反映企业对债券市场的依赖。"
    "债券融资比例高意味着企业能够进入债券市场（信用认可），"
    "但同时也面临利率风险和到期再融资风险。银行借款+债券的合理搭配是最优结构。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_bond_issuance_ratio(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["proc_issue_bonds", "stot_cash_in_fnc_act"],
    )
    ratio = safe_divide(cf["proc_issue_bonds"], cf["stot_cash_in_fnc_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_dividend_interest_paid_ratio",
    description="分红付息/筹资流出比率因子，incl_dvd_profit_paid_sc_ms/stot_cashout_fnc_act。",
    category="quality",
    thesis="分配股利和偿付利息支付的现金占筹资活动流出比重反映"
    "回报股东和债权人的力度。高比率是股东友好型企业的特征。"
    "但需确保分红有充足的自由现金流支撑（而非靠借新债分红）。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_dividend_interest_paid_ratio(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["incl_dvd_profit_paid_sc_ms", "stot_cashout_fnc_act"],
    )
    ratio = safe_divide(cf["incl_dvd_profit_paid_sc_ms"], cf["stot_cashout_fnc_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# D.  Supplementary — Indirect Method
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_depreciation_provision_ratio_cf",
    description="折旧与准备/净利润比率因子(CF口径)，prov_depr_assets/abs(net_profit)。",
    category="quality",
    thesis="资产减值准备和折旧是净利润→经营现金流的最大调节项。"
    "prov_depr_assets占比高意味着大量非现金费用——经营现金流远超净利润。"
    "重资产行业的经营现金流质量天然更高（折旧是非现金支出）。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_depreciation_provision_ratio_cf(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["prov_depr_assets", "net_profit"],
    )
    ratio = safe_divide(cf["prov_depr_assets"].abs(), cf["net_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_working_capital_change_cf",
    description="营运资本变动/净利润比率因子(CF口径)，(存货+应收+应付变动)/净利润。",
    category="quality",
    thesis="营运资本变动(间接法)=存货减少+经营性应收减少+经营性应付增加。"
    "正值意味着营运资本释放了现金（正面），负值意味着占用了更多现金。"
    "持续负值（现金被营运资本吞噬）是增长质量差的信号。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_working_capital_change_cf(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["decr_inventories", "decr_oper_payable", "incr_oper_payable", "net_profit"],
    )
    wc_change = (
        cf["decr_inventories"].fillna(0)
        - cf["decr_oper_payable"].fillna(0)
        + cf["incr_oper_payable"].fillna(0)
    )
    ratio = safe_divide(wc_change, cf["net_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_deferred_tax_change_cf",
    description="递延所得税变动/净利润比率因子(CF口径)。",
    category="quality",
    thesis="递延所得税资产减少+递延所得税负债增加反映应纳税暂时性差异——"
    "意味着当期实际缴税少于账面税费（现金节省）。持续大额递延税变动"
    "需要关注是否来自激进的税务筹划。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_deferred_tax_change_cf(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["decr_def_inc_tax_assets", "incr_def_inc_tax_liab", "net_profit"],
    )
    dt_change = cf["decr_def_inc_tax_assets"].fillna(0) + cf["incr_def_inc_tax_liab"].fillna(0)
    ratio = safe_divide(dt_change, cf["net_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_fair_value_loss_cf",
    description="公允价值变动损失/净利润比率因子(CF口径)（低值排前）。",
    category="quality",
    thesis="loss_fv_chg是间接法中从净利润调整到经营现金流的项目——"
    "公允价值变动损失是未实现损失，不产生现金流出。该调整项大"
    "意味着利润受到了公允价值波动的显著冲击（非现金），盈利波动性大。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_fair_value_loss_cf(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["loss_fv_chg", "net_profit"],
    )
    ratio = safe_divide(cf["loss_fv_chg"].abs(), cf["net_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_imputed_ocf_adjustment",
    description="间接法其他调整/净利润比率因子（低值排前），im_net_cashflow_oper_act/abs(net_profit)。",
    category="quality",
    thesis="间接法的'其他'调整项(im_net_cashflow_oper_act)是净利润→经营现金流的"
    "剩余调节项——包含所有未单独列示的项目。该项绝对值大意味着"
    "经营现金流中包含大量'说不清楚'的调节——透明度差的信号。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_imputed_ocf_adjustment(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["im_net_cashflow_oper_act", "net_profit"],
    )
    ratio = safe_divide(cf["im_net_cashflow_oper_act"].abs(), cf["net_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_credit_impairment_cf_supp",
    description="信用减值损失(CF补充)/净利润比率因子（低值排前）。",
    category="quality",
    thesis="现金流量表补充资料中的信用减值损失(credit_impa_loss)反映了"
    "预期信用损失模型下的减值计提。大额信用减值意味着"
    "应收账款或债权投资的信用质量恶化——领先于实际坏账的信号。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_credit_impairment_cf_supp(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["credit_impa_loss", "net_profit"],
    )
    ratio = safe_divide(cf["credit_impa_loss"].abs(), cf["net_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_right_of_use_dep_cf_supp",
    description="使用权资产折旧(CF补充)/净利润比率因子。",
    category="quality",
    thesis="use_right_asset_dep（使用权资产折旧）是新租赁准则下的新增调节项。"
    "该折旧是非现金支出，调增经营现金流。高使用权资产折旧占比"
    "意味着企业大量使用租赁资产——轻资产运营模式的现金流量表特征。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_right_of_use_dep_cf_supp(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["use_right_asset_dep", "net_profit"],
    )
    ratio = safe_divide(cf["use_right_asset_dep"].abs(), cf["net_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# E.  Conversions & Capital
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_debt_to_equity_conversion",
    description="债转股/总资产比率因子，conv_debt_into_cap/total_assets截面排名。",
    category="quality",
    thesis="债转股(conv_debt_into_cap)将债务转换为资本——降低杠杆但稀释股权。"
    "正金额意味着发生了债转股（债务重组信号），"
    "通常发生在财务困境企业。对正常经营企业，债转股金额应为零。",
    dependencies=("cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_debt_to_equity_conversion(context: FactorContext):
    cf = context.load_financial("cashflow.parquet", value_cols=["conv_debt_into_cap"])
    bs = context.load_financial("balancesheet.parquet", value_cols=["total_assets"])
    ratio = safe_divide(cf["conv_debt_into_cap"], bs["total_assets"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_bonds_due_within_year",
    description="一年内到期应付债券/总负债比率因子（低值排前）。",
    category="quality",
    thesis="一年内到期的应付债券(conv_copbonds_due_within_1y)占总负债比重"
    "是短期再融资压力的直接度量。高比率意味着大量债券即将到期需要"
    "续发或偿还——在信用紧缩期面临再融资风险。",
    dependencies=("cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_bonds_due_within_year(context: FactorContext):
    cf = context.load_financial("cashflow.parquet", value_cols=["conv_copbonds_due_within_1y"])
    bs = context.load_financial("balancesheet.parquet", value_cols=["total_liab"])
    ratio = safe_divide(cf["conv_copbonds_due_within_1y"], bs["total_liab"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# F.  Cash Balance Items
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_cash_equivalents_change",
    description="现金等价物变动率因子，c_cash_equ_end_period/c_cash_equ_beg_period截面排名。",
    category="quality",
    thesis="现金及现金等价物期末/期初比率直接反映企业的现金池变动。"
    "比率>1=现金积累（正面），<1=现金消耗。该比率比n_incr_cash_cash_equ"
    "更直观——直接给出变动倍率而非绝对额。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_cash_equivalents_change(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_cash_equ_end_period", "c_cash_equ_beg_period"],
    )
    ratio = safe_divide(cf["c_cash_equ_end_period"], cf["c_cash_equ_beg_period"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_capital_contribution_cf",
    description="吸收投资收到的现金/筹资流入比率因子，c_recp_cap_contrib/stot_cash_in_fnc_act。",
    category="quality",
    thesis="吸收投资（股权融资）占筹资流入比重高意味着企业依赖增发融资——"
    "可能是高成长企业的正常行为（融资扩张），也可能是盈利能力不足、"
    "依赖外部输血。需结合ROE和营收增速判断。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_capital_contribution_cf(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_recp_cap_contrib", "stot_cash_in_fnc_act"],
    )
    ratio = safe_divide(cf["c_recp_cap_contrib"], cf["stot_cash_in_fnc_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_net_cash_from_securities",
    description="证券投资净现金/净利润比率因子，net_cash_rece_sec/abs(net_profit)。",
    category="quality",
    thesis="net_cash_rece_sec（证券投资收到的现金净额）占净利润比重大"
    "意味着企业有大量金融资产交易——盈利受到金融市场波动的显著影响。"
    "对于非金融企业，持续大额的证券投资净现金流入/流出需警惕。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_net_cash_from_securities(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["net_cash_rece_sec", "net_profit"],
    )
    ratio = safe_divide(cf["net_cash_rece_sec"].abs(), cf["net_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)
