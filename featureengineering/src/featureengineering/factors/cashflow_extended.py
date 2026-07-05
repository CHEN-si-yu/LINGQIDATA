"""
Extended cash flow statement factors — Class 1.

Systematic factorisation of remaining unused cashflow.parquet fields.
After existing factors and unused_fields_factors.py (~7 factors from CF),
many detailed cash flow sub-items remain untapped — investing/financing
efficiency, capex, depreciation, free cash flow, tax/employee burden,
and supplementary information.

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
# A.  Investing & Financing Cash Flow
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_investing_cf_ratio",
    description="投资活动现金流效率因子，投资活动现金流出/流入截面排名。",
    category="quality",
    thesis="投资活动流出/流入比率反映企业的投资节奏。比率>1意味着"
    "企业处于扩张期（资本开支>投资回收），比率<1意味着收缩期。"
    "适度的扩张（略>1）结合营收增长是健康的，过高的比率可能过度投资。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_investing_cf_ratio(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["stot_out_inv_act", "stot_inflows_inv_act"],
    )
    ratio = safe_divide(cf["stot_out_inv_act"], cf["stot_inflows_inv_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_financing_cf_ratio",
    description="筹资活动现金流结构因子，筹资流出/筹资流入截面排名。",
    category="quality",
    thesis="筹资流出/流入比率反映企业的融资-偿债/分红平衡。"
    "高比率（偿还>融资）意味着去杠杆或大额分红——财务趋于保守。"
    "低比率（融资>偿还）意味着加杠杆或增发——需关注资金用途。"
    "可持续的状态是比率围绕1波动。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_financing_cf_ratio(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["stot_cashout_fnc_act", "stot_cash_in_fnc_act"],
    )
    ratio = safe_divide(cf["stot_cashout_fnc_act"], cf["stot_cash_in_fnc_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Key Expenditure Ratios
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_capex_to_revenue",
    description="资本开支/营收比率因子，c_pay_acq_const_fiolta/revenue截面排名。",
    category="quality",
    thesis="资本开支（购建固定资产/无形资产支付的现金）占营收比重"
    "反映企业的再投资强度。高capex/营收意味着企业处于扩张期，"
    "但也意味着大量现金流出——自由现金流（FCF = OCF - capex）减少。"
    "维持在合理水平的capex是长期增长的基础。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_capex_to_revenue(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["c_pay_acq_const_fiolta"]
    )
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    ratio = safe_divide(cf["c_pay_acq_const_fiolta"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_depreciation_to_revenue",
    description="折旧摊销/营收比率因子，depr_fa_coga_dpba/revenue截面排名。",
    category="quality",
    thesis="固定资产折旧占营收比重反映现有资产的折旧负担。"
    "高折旧率常见于重资产行业（制造、能源、交通），低折旧率常见于轻资产行业。"
    "折旧/营收比率突然上升（营收未同步增长）意味着产能利用率下降。"
    "该比率与capex/revenue结合可判断企业的投资周期阶段。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_depreciation_to_revenue(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["depr_fa_coga_dpba"]
    )
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    ratio = safe_divide(cf["depr_fa_coga_dpba"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_free_cashflow_raw",
    description="自由现金流原始值因子，free_cashflow/总市值截面排名（高值排前）。",
    category="quality",
    thesis="free_cashflow（cashflow.parquet中预计算的企业自由现金流）"
    "是企业可自由支配的现金。正值FCF意味着企业有现金可用于分红、回购或再投资。"
    "负值FCF需要外部融资来弥补缺口。配合市值标准化后是优秀估值因子。",
    dependencies=("cashflow.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_ext_free_cashflow_raw(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["free_cashflow"]
    )
    finance = context.load("finance.parquet")
    fcf_yield = safe_divide(cf["free_cashflow"], finance["total_mv"])
    return cross_sectional_rank(fcf_yield)


# ═══════════════════════════════════════════════════════════════════════════
# C.  Obligation Ratios
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_dividend_payout_cf",
    description="分红付现比因子，分配股利利润支付的现金/经营现金流截面排名。",
    category="quality",
    thesis="分红现金/经营CF比率反映了分红的可持续性——"
    "比率<0.5意味着分红有充足的经营现金流支撑，可持续性强。"
    "比率>1意味着企业借钱分红（动用储备或新增融资）——不可持续。"
    "持续合理的分红付现比是红利因子的质量过滤器。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_dividend_payout_cf(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_pay_dist_dpcp_int_exp", "n_cashflow_act"],
    )
    ratio = safe_divide(
        cf["c_pay_dist_dpcp_int_exp"],
        cf["n_cashflow_act"].abs().replace(0, np.nan),
    )
    return cross_sectional_rank(-ratio)  # lower payout ratio = more sustainable


@register_factor(
    name="ext_employee_pay_to_revenue",
    description="员工薪酬/营收比率因子，c_paid_to_for_empl/revenue截面排名。",
    category="quality",
    thesis="支付给职工以及为职工支付的现金/营收比率反映了人力成本密度。"
    "高比率常见于服务业、科技业（人才密集型），低比率常见于制造业（资本密集型）。"
    "该比率结合人均创收可以判断企业的人力资源效率。"
    "人均薪酬上升+人均创收上升=良性循环；人均薪酬上升+人均创收下降=效率恶化。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_employee_pay_to_revenue(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["c_paid_to_for_empl"]
    )
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    ratio = safe_divide(cf["c_paid_to_for_empl"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_tax_paid_to_pretax",
    description="税款付现比因子，支付的各项税费/(收入+利润)截面排名。",
    category="quality",
    thesis="支付的税费/营业总收入(total_revenue)比率反映了实际税负水平。"
    "实际税负显著低于法定税率(25%)可能意味着：税收优惠（正面）、"
    "利润操纵（负面）、或海外业务占比大（结构性）。"
    "使用现金流量表的实缴税费比利润表的所得税费用更真实（不可操纵）。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_ext_tax_paid_to_pretax(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["c_paid_for_taxes"]
    )
    inc = context.load_financial("income.parquet", value_cols=["total_revenue"])
    ratio = safe_divide(cf["c_paid_for_taxes"], inc["total_revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# D.  Cash Balance & Supplementary
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_cash_balance_change",
    description="现金余额变动率因子，(期末-期初)/期初现金截面排名。",
    category="quality",
    thesis="现金及等价物余额的期间变动率直接反映企业的现金消耗/积累速度。"
    "期末余额/期初余额比率>1意味着现金积累（正面），<1意味着现金消耗（需关注）。"
    "使用end_bal_cash和beg_bal_cash（与n_incr_cash_cash_equ互补）。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_ext_cash_balance_change(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["end_bal_cash", "beg_bal_cash"],
    )
    ratio = safe_divide(
        cf["end_bal_cash"] - cf["beg_bal_cash"],
        cf["beg_bal_cash"].abs().replace(0, np.nan),
    )
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_ocf_to_total_debt",
    description="经营现金流/总债务比率因子，n_cashflow_act/(st_borr+lt_borr)截面排名（高值排前）。",
    category="quality",
    thesis="经营现金流净额/有息负债总额是衡量真实偿债能力的核心指标。"
    "比率>0.5意味着经营现金流2年可覆盖全部有息负债——非常安全。"
    "比率<0.1意味着需要10年以上——偿债压力大。"
    "该指标比利润指标更能预警债务危机（现金流先于利润恶化）。",
    dependencies=("cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_ocf_to_total_debt(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["n_cashflow_act"]
    )
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["st_borr", "lt_borr"]
    )
    total_debt = bs["st_borr"].fillna(0) + bs["lt_borr"].fillna(0)
    ratio = safe_divide(cf["n_cashflow_act"], total_debt.abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)
