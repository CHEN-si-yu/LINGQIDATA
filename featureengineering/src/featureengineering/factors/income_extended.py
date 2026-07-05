"""
Extended income statement factors — Class 1.

Systematic factorisation of remaining unused income.parquet fields.
After existing factors and unused_fields_factors.py (~5 factors from IS),
many P&L detail columns remain untapped — margins, profit decomposition,
and expense structure.

Skipped:
- Insurance-specific columns (prem_earned, compens_payout, reins_*, div_payt, etc.)
- High-NaN (continued_net_profit >7% NaN)
- Profit distribution detail (transfer_*, withdra_*, workers_welfare — not investor-relevant)

Data source: ``income.parquet`` (96 columns, quarterly via forward-fill)
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
# A.  Margin Analysis
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_ebitda_margin_is",
    description="EBITDA利润率因子(IS口径)，ebitda/revenue截面排名（高值排前）。",
    category="quality",
    thesis="使用income.parquet中的ebitda和revenue字段计算EBITDA利润率。"
    "与financial_indicator口径（ebitda/total_revenue_ps）形成互补和交叉验证。"
    "IS口径使用原始财务报表数据，不受标准化调整影响。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_ebitda_margin_is(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["ebitda", "revenue"]
    )
    margin = safe_divide(inc["ebitda"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(margin)


@register_factor(
    name="ext_ebit_margin",
    description="EBIT利润率因子，ebit/revenue截面排名（高值排前）。",
    category="quality",
    thesis="息税前利润率（EBIT Margin）剔除了财务费用和所得税的影响，"
    "是衡量企业经营效率的最佳单一指标之一。EBIT Margin稳定或上升"
    "意味着企业拥有定价权或成本控制能力——竞争壁垒的体现。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_ebit_margin(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["ebit", "revenue"]
    )
    margin = safe_divide(inc["ebit"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(margin)


@register_factor(
    name="ext_sg_and_a_to_revenue",
    description="销售管理费用率因子(IS口径)（低值排前），(sell_exp+admin_exp)/revenue。",
    category="quality",
    thesis="SG&A比率是评价管理层费用管控能力的重要指标。"
    "使用income.parquet的sell_exp和admin_exp原始值。"
    "费用率持续下降+营收增长=经营杠杆正效应（最理想的增长模式）。"
    "费用率上升+营收增长放缓=费用失控风险。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_sg_and_a_to_revenue(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["sell_exp", "admin_exp", "revenue"]
    )
    sga = inc["sell_exp"].fillna(0) + inc["admin_exp"].fillna(0)
    ratio = safe_divide(sga, inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Profit Decomposition
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_asset_impairment_ratio",
    description="资产减值/营收比率因子（低值排前），assets_impair_loss/revenue。",
    category="quality",
    thesis="资产减值损失占营收比重大幅上升是利润质量的重大红旗——"
    "意味着企业承认之前的资产估值存在问题（商誉、存货、应收等）。"
    "减值集中的季度往往伴随股价大幅调整。持续低减值率是会计稳健的信号。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_asset_impairment_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["assets_impair_loss", "revenue"]
    )
    ratio = safe_divide(inc["assets_impair_loss"].abs(), inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_investment_income_ratio",
    description="投资收益/利润总额比率因子（低值排前），invest_income/total_profit。",
    category="quality",
    thesis="投资收益占利润总额比重大意味着企业盈利依赖非主营业务——"
    "可能是持有大量金融资产（如保险公司）、子公司分红（如控股集团）、"
    "或炒股收益（非金融企业）。对于非金融企业，高投资收益占比是盈利质量差的信号。"
    "持续性经营利润才是企业价值的真正来源。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_investment_income_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["invest_income", "total_profit"]
    )
    ratio = safe_divide(inc["invest_income"].abs(), inc["total_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_fair_value_change_ratio",
    description="公允价值变动/利润总额比率因子（低值排前），fv_value_chg_gain/total_profit。",
    category="quality",
    thesis="公允价值变动收益是未实现损益，不产生现金流。高FV收益占比意味着"
    "利润中有大量'纸面富贵'——市场反转时这些收益可能迅速变为亏损。"
    "持有大量交易性金融资产的企业（如部分'炒股'上市公司）该比率波动大。"
    "持续高FV收益占比是盈利质量差的信号。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_fair_value_change_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["fv_value_chg_gain", "total_profit"]
    )
    ratio = safe_divide(inc["fv_value_chg_gain"].abs(), inc["total_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_non_operating_net_ratio",
    description="营业外净收支/利润总额比率因子（低值排前），(non_oper_income-non_oper_exp)/total_profit。",
    category="quality",
    thesis="营业外净收支占比反映了非经常性项目对利润的贡献。"
    "正值且占比高=利润依赖一次性收益（如资产处置、政府补贴）。"
    "负值且占比大=营业外支出拖累（如罚款、捐赠）。"
    "持续接近零是企业经营专注度高的表现。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_non_operating_net_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["non_oper_income", "non_oper_exp", "total_profit"],
    )
    net_non_op = inc["non_oper_income"].fillna(0) - inc["non_oper_exp"].fillna(0)
    ratio = safe_divide(net_non_op.abs(), inc["total_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_credit_impairment_ratio",
    description="信用减值损失/营收比率因子（低值排前），credit_impair_loss/revenue。",
    category="quality",
    thesis="信用减值损失（新金融工具准则）是应收款项和债权投资预期损失的提前确认。"
    "大额信用减值意味着企业的客户/债务人出现信用问题——这是领先于实际违约的信号。"
    "银行、地产、供应链金融企业的信用减值率是核心风险指标。"
    "对于一般企业，突然出现大额信用减值需高度警惕。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_credit_impairment_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["credit_impair_loss", "revenue"]
    )
    ratio = safe_divide(inc["credit_impair_loss"].abs(), inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)
