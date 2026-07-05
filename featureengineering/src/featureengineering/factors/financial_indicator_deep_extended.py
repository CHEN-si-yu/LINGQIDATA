"""
Deep extended financial indicator factors — Class 1 (Wave 2).

Second-wave factorisation of remaining unused financial_indicator.parquet fields
discovered through a comprehensive column-level audit.  These are high-quality
pre-computed ratios that were missed in earlier passes.

Sections
--------
A. Earnings Quality Depth      — extra_item, op_income, valuechange_income, daa, impai_ttm, etc.
B. Margin Decomposition        — gc_of_gr, op_of_gr, ebit_of_gr, profit_prefin_exp
C. ROE / ROA Variants          — roe_yearly, roe_avg, roa_yearly, roa_dp, roic_yearly
D. Asset Structure Ratios      — dp_assets_to_eqt, tbassets_to_totalassets, etc.
E. Quarterly Level Signals     — q_opincome, q_investincome, q_dtprofit, q_exp_to_sales, etc.
F. Quarterly Growth Composites — earnings breadth, financial health, quality momentum
G. Cash Flow Solvency          — ocf_to_opincome, op_to_liqdebt, cash_to_liqdebt_withinterest

Data source: ``financial_indicator.parquet`` (167 columns, daily-panel via forward-fill)
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
# A.  Earnings Quality Depth
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_extra_item_to_profit",
    description="非经常性损益/利润总额比率因子（低值排前），extra_item/abs(profit_to_gr)。",
    category="quality",
    thesis="extra_item（非经常性损益）是利润中的一次性项目。"
    "占比高意味着当期利润不可持续——扣非后利润可能大幅缩水。"
    "持续低extra_item占比的企业盈利质量更高、预测性更强。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_extra_item_to_profit(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["extra_item", "profit_to_gr"],
    )
    ratio = safe_divide(fin["extra_item"].abs(), fin["profit_to_gr"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_op_income_ratio",
    description="营业利润/利润总额比率因子(fi口径)（高值排前），op_income/profit_to_gr。",
    category="quality",
    thesis="op_income占利润总额比例反映经营性收益的主导地位。"
    "高比率意味着利润主要来自主营业务（可持续），低比率意味着依赖营业外收入。"
    "使用financial_indicator口径与income口径交叉验证。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_op_income_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["op_income", "profit_to_gr"],
    )
    ratio = safe_divide(fin["op_income"], fin["profit_to_gr"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_valuechange_income_ratio",
    description="公允价值变动收益/利润总额比率因子(fi口径)（低值排前）。",
    category="quality",
    thesis="valuechange_income是未实现损益。高占比意味着利润中有大量'纸面富贵'——"
    "市场反转时可能迅速变为亏损。使用financial_indicator预计算口径。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_valuechange_income_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["valuechange_income", "profit_to_gr"],
    )
    ratio = safe_divide(fin["valuechange_income"].abs(), fin["profit_to_gr"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_interest_income_ratio",
    description="利息收入/利润总额比率因子（低值排前），interst_income/profit_to_gr。",
    category="quality",
    thesis="利息收入占利润比重高可能意味着：大量闲置资金存银行（效率低）、"
    "或金融业务占比大（非实体经济）。对于实体企业，高利息收入占比"
    "暗示资金未能有效用于主业扩张。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_interest_income_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["interst_income", "profit_to_gr"],
    )
    ratio = safe_divide(fin["interst_income"].abs(), fin["profit_to_gr"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_daa_to_revenue",
    description="折旧摊销/营收比率因子，daa/total_revenue_ps截面排名。",
    category="quality",
    thesis="折旧与摊销（D&A）占营收比重反映资本密集度和资产老化程度。"
    "高D&A/营收常见于重资产行业（制造、能源、交通）。D&A/营收突然上升"
    "（营收未同步增长）意味着产能利用率下降。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_daa_to_revenue(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["daa", "total_revenue_ps"],
    )
    ratio = safe_divide(fin["daa"], fin["total_revenue_ps"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_impairment_ttm_ratio",
    description="TTM减值/营收比率因子（低值排前），impai_ttm/total_revenue_ps。",
    category="quality",
    thesis="滚动十二个月（TTM）减值损失占营收比重是盈利质量的持续性指标。"
    "TTM减值率持续下降意味着资产质量改善，持续上升则需警惕。"
    "TTM口径平滑了单季波动，更适合中长期判断。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_impairment_ttm_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["impai_ttm", "total_revenue_ps"],
    )
    ratio = safe_divide(fin["impai_ttm"].abs(), fin["total_revenue_ps"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Margin Decomposition
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_gross_cost_ratio",
    description="营业成本率因子（低值排前），gc_of_gr=营业成本/营业总收入。",
    category="quality",
    thesis="gc_of_gr是毛利率的镜像（=1-毛利率），直接反映产品或服务的成本结构。"
    "高成本率（低毛利率）意味着微利经营，对原材料价格和人工成本高度敏感。"
    "成本率的趋势性下降（毛利率提升）是竞争力改善的最佳证据。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_gross_cost_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["gc_of_gr"]
    )
    return cross_sectional_rank(-fin["gc_of_gr"])


@register_factor(
    name="ext_operating_profit_to_revenue_fi",
    description="营业利润率因子(fi口径)，op_of_gr=营业利润/营业总收入截面排名（高值排前）。",
    category="quality",
    thesis="营业利润率(op_of_gr)衡量扣除三项费用后的经营效率。"
    "营业利润率>毛利率的一半说明费用控制良好。"
    "使用financial_indicator口径与income口径(operate_profit/revenue)互补。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_operating_profit_to_revenue_fi(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["op_of_gr"]
    )
    return cross_sectional_rank(fin["op_of_gr"])


@register_factor(
    name="ext_ebit_to_revenue_fi",
    description="EBIT利润率因子(fi口径)，ebit_of_gr截面排名（高值排前）。",
    category="quality",
    thesis="ebit_of_gr（EBIT/营业总收入）剔除了财务费用和所得税，"
    "适合跨资本结构比较。使用financial_indicator预计算口径。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_ebit_to_revenue_fi(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebit_of_gr"]
    )
    return cross_sectional_rank(fin["ebit_of_gr"])


@register_factor(
    name="ext_profit_prefin_exp_ratio",
    description="息税前利润/利润总额比率因子（高值排前），profit_prefin_exp/profit_to_gr。",
    category="quality",
    thesis="profit_prefin_exp是扣除财务费用前的利润。与利润总额的比率"
    "反映了财务费用对利润的侵蚀程度。比率接近1说明财务费用低（财务健康），"
    "比率>>1说明大量利润用于支付利息（高杠杆风险）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_profit_prefin_exp_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["profit_prefin_exp", "profit_to_gr"],
    )
    ratio = safe_divide(fin["profit_prefin_exp"], fin["profit_to_gr"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_networking_capital_to_assets_fi",
    description="净营运资本/总资产比率因子，networking_capital/tangible_asset截面排名。",
    category="quality",
    thesis="networking_capital（净营运资本）与working_capital口径不同——"
    "networking_capital通常=流动资产-流动负债（不含现金及短期债务），"
    "更精准地衡量经营性营运资本。低/负净营运资本=强势商业模式（占用他人资金）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_networking_capital_to_assets_fi(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["networking_capital", "tangible_asset"],
    )
    ratio = safe_divide(fin["networking_capital"], fin["tangible_asset"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# C.  ROE / ROA Variants
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_roe_yearly_signal",
    description="年化ROE因子，roe_yearly=ROE年化值截面排名（高值排前）。",
    category="quality",
    thesis="roe_yearly将最近季度的ROE年化，提供了全年的盈利预测。"
    "比单季度ROE更能反映完整年度的盈利能力。结合roe_avg使用可判断"
    "盈利趋势（年化ROE>平均ROE=盈利加速）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_roe_yearly_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe_yearly"]
    )
    return cross_sectional_rank(fin["roe_yearly"])


@register_factor(
    name="ext_roe_avg_signal",
    description="平均ROE因子，roe_avg=多期平均ROE截面排名（高值排前）。",
    category="quality",
    thesis="roe_avg平滑了单期波动，反映企业长期稳定的盈利能力。"
    "平均ROE高且稳定是优质企业的核心特征。与单期ROE的差异可判断"
    "盈利质量——单期ROE远高于平均可能包含一次性收益。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_roe_avg_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe_avg"]
    )
    return cross_sectional_rank(fin["roe_avg"])


@register_factor(
    name="ext_roe_yoy_signal",
    description="ROE同比增长率因子，roe_yoy截面排名（高值排前）。",
    category="growth",
    thesis="ROE的同比增长率是盈利改善的领先信号。ROE增长来自"
    "利润率提升、周转加快或杠杆增加——前两者是质量信号，后者需警惕。"
    "ROE增速>净利润增速意味着企业可能在进行有效的资本配置。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_roe_yoy_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe_yoy"]
    )
    return cross_sectional_rank(fin["roe_yoy"])


@register_factor(
    name="ext_roa_yearly_signal",
    description="年化ROA因子，roa_yearly截面排名（高值排前）。",
    category="quality",
    thesis="roa_yearly将ROA年化，剔除了杠杆影响（与ROE互补）。"
    "年化ROA>行业均值意味着企业拥有竞争优势。ROA的稳定性比绝对值更重要。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_roa_yearly_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roa_yearly"]
    )
    return cross_sectional_rank(fin["roa_yearly"])


@register_factor(
    name="ext_roa_dp_signal",
    description="杜邦分析ROA因子，roa_dp截面排名（高值排前）。",
    category="quality",
    thesis="roa_dp（杜邦口径ROA = 净利润率×总资产周转率）分解了ROA的驱动力。"
    "高roa_dp来自高利润率或高周转率的组合——两者都是竞争优势的标志。"
    "roa_dp的变化可以追溯至利润率变化或效率变化。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_roa_dp_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roa_dp"]
    )
    return cross_sectional_rank(fin["roa_dp"])


@register_factor(
    name="ext_roic_yearly_signal",
    description="年化ROIC因子，roic_yearly截面排名（高值排前）。",
    category="quality",
    thesis="roic_yearly（年化投入资本回报率）是价值投资的核心指标。"
    "ROIC>WACC=创造价值，ROIC<WACC=毁灭价值。年化口径平滑了季节性。"
    "持续高ROIC是经济护城河的最佳定量证据。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_roic_yearly_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roic_yearly"]
    )
    return cross_sectional_rank(fin["roic_yearly"])


# ═══════════════════════════════════════════════════════════════════════════
# D.  Asset Structure Ratios
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_dp_assets_to_equity",
    description="杜邦权益乘数因子，dp_assets_to_eqt截面排名。",
    category="quality",
    thesis="dp_assets_to_eqt（资产/权益，即杜邦公式中的权益乘数）衡量财务杠杆。"
    "高权益乘数放大ROE但也放大风险。权益乘数的变化方向比绝对水平更重要——"
    "下降中的权益乘数=去杠杆（正面），上升中的=加杠杆（需关注）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_dp_assets_to_equity(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["dp_assets_to_eqt"]
    )
    return cross_sectional_rank(-fin["dp_assets_to_eqt"])


@register_factor(
    name="ext_tb_assets_ratio",
    description="有形资产/总资产比率因子，tbassets_to_totalassets截面排名（高值排前）。",
    category="quality",
    thesis="有形资产占比（tbassets_to_totalassets）反映资产的'硬度'。"
    "高有形资产占比意味着清算价值有保障（银行关注的抵押价值），"
    "低有形资产占比常见于科技/服务公司（轻资产模式）。"
    "该比率在信用风险评估中尤为重要。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_tb_assets_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["tbassets_to_totalassets"]
    )
    return cross_sectional_rank(fin["tbassets_to_totalassets"])


@register_factor(
    name="ext_equity_to_interest_debt",
    description="权益/有息负债比率因子，eqt_to_interestdebt截面排名（高值排前）。",
    category="quality",
    thesis="权益覆盖有息负债的倍数衡量偿债安全垫。"
    "高倍数（>5）意味着即使利润大幅下降也能偿还利息。"
    "低倍数（<1）意味着权益不足以覆盖有息负债——资不抵债边缘。"
    "该指标比D/E更聚焦（仅关注有息负债而非全部负债）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_equity_to_interest_debt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["eqt_to_interestdebt"]
    )
    return cross_sectional_rank(fin["eqt_to_interestdebt"])


@register_factor(
    name="ext_total_fa_turnover",
    description="总固定资产周转率因子，total_fa_trun截面排名（高值排前）。",
    category="quality",
    thesis="total_fa_trun（总固定资产周转率）与fa_turn互补——"
    "total_fa_trun包含了在建工程、使用权资产等更广泛的长期资产口径。"
    "两者的差异反映了在建工程等尚未产生收入的资产占比。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_total_fa_turnover(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["total_fa_trun"]
    )
    return cross_sectional_rank(fin["total_fa_trun"])


# ═══════════════════════════════════════════════════════════════════════════
# E.  Quarterly Level Signals
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_q_operating_income_level",
    description="单季度营业利润规模因子，q_opincome/total_revenue_ps截面排名（高值排前）。",
    category="quality",
    thesis="单季度营业利润（q_opincome）的绝对值水平与年度化营收的比率"
    "反映单季度盈利能力相对于企业规模的水平。该比率高意味着最近一个季度"
    "的盈利动能强劲（可能是拐点信号）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_operating_income_level(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_opincome", "total_revenue_ps"],
    )
    ratio = safe_divide(fin["q_opincome"], fin["total_revenue_ps"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_q_investment_income_level",
    description="单季度投资收益依赖度因子（低值排前），q_investincome/q_opincome。",
    category="quality",
    thesis="单季度投资收益占营业利润的比例反映了短期盈利对非主营业务的依赖。"
    "q_investincome/q_opincome突然升高可能意味着主业经营恶化、靠投资收益粉饰。"
    "持续低比率意味着盈利来源纯粹。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_investment_income_level(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_investincome", "q_opincome"],
    )
    ratio = safe_divide(fin["q_investincome"].abs(), fin["q_opincome"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_q_deducted_profit_level",
    description="单季度扣非净利润率因子，q_dtprofit/q_opincome截面排名（高值排前）。",
    category="quality",
    thesis="扣非净利润/营业利润比率越高说明非经常性项目占比越小。"
    "q_dtprofit/q_opincome接近1是最理想的状态。"
    "该比率的变化方向对判断盈利质量趋势很有价值。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_deducted_profit_level(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_dtprofit", "q_opincome"],
    )
    ratio = safe_divide(fin["q_dtprofit"], fin["q_opincome"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_q_expense_ratio_fi",
    description="单季度费用率因子（低值排前），q_exp_to_sales截面排名。",
    category="quality",
    thesis="单季度费用率（q_exp_to_sales）提供比年化数据更及时的"
    "费用控制信号。费用率突然跳升可能预示渠道扩张、研发投入或管理失控。"
    "q_exp_to_sales的边际变化比绝对水平更重要。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_expense_ratio_fi(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_exp_to_sales"]
    )
    return cross_sectional_rank(-fin["q_exp_to_sales"])


@register_factor(
    name="ext_q_profit_margin_fi",
    description="单季度利润率因子，q_profit_to_gr截面排名（高值排前）。",
    category="quality",
    thesis="q_profit_to_gr（单季度利润总额/营业总收入）是单季度盈利能力"
    "的综合度量。与年度利润率对比可判断利润率的边际变化趋势。"
    "利润率连续两个季度改善是基本面拐点的强烈信号。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_profit_margin_fi(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_profit_to_gr"]
    )
    return cross_sectional_rank(fin["q_profit_to_gr"])


@register_factor(
    name="ext_q_sga_ratio_fi",
    description="单季度SG&A费用率因子（低值排前），q_saleexp_to_gr+q_adminexp_to_gr。",
    category="quality",
    thesis="单季度销售+管理费用率是管理效率的高频监测指标。"
    "费用率趋势性下降=管理改善+经营杠杆正效应。"
    "费用率跳升需结合营收增速判断是扩张性投入还是效率恶化。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_sga_ratio_fi(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_saleexp_to_gr", "q_adminexp_to_gr"],
    )
    sga = fin["q_saleexp_to_gr"].fillna(0) + fin["q_adminexp_to_gr"].fillna(0)
    return cross_sectional_rank(-sga)


@register_factor(
    name="ext_q_impairment_burden",
    description="单季度减值TTM负担因子（低值排前），q_impair_to_gr_ttm截面排名。",
    category="quality",
    thesis="TTM口径的单季度减值率（q_impair_to_gr_ttm）平滑了季节性。"
    "减值负担持续上升是资产质量恶化的滞后确认——通常是负面信号。"
    "对于银行、地产等行业，该指标是资产质量的核心先行指标。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_impairment_burden(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_impair_to_gr_ttm"]
    )
    return cross_sectional_rank(-fin["q_impair_to_gr_ttm"].abs())


@register_factor(
    name="ext_q_dt_roe_signal",
    description="单季度摊薄ROE因子，q_dt_roe截面排名（高值排前）。",
    category="quality",
    thesis="q_dt_roe（单季度摊薄ROE）使用最新股本计算，比基本ROE更保守——"
    "考虑了潜在稀释（期权、可转债等）。当q_dt_roe显著低于roe时，"
    "意味着存在大量潜在稀释工具，未来EPS可能被摊薄。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_dt_roe_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_dt_roe"]
    )
    return cross_sectional_rank(fin["q_dt_roe"])


@register_factor(
    name="ext_q_npta_signal",
    description="单季度总资产净利润率因子，q_npta截面排名（高值排前）。",
    category="quality",
    thesis="q_npta（单季度净利润/总资产）是ROA的单季度版本，"
    "提供了比年化ROA更及时的总资产盈利能力信号。"
    "连续两个季度npta改善通常领先于股价上升。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_npta_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_npta"]
    )
    return cross_sectional_rank(fin["q_npta"])


@register_factor(
    name="ext_q_ocf_to_or_signal",
    description="单季度经营现金流/营收比率因子，q_ocf_to_or截面排名（高值排前）。",
    category="quality",
    thesis="q_ocf_to_or是单季度收入现金转化率——比年度数据更敏感。"
    "单季度OCF/OR突然恶化可能预示应收账款问题或渠道压货。"
    "连续两个季度的OCF/OR<0.5需要高度警惕收入质量。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_q_ocf_to_or_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_ocf_to_or"]
    )
    return cross_sectional_rank(fin["q_ocf_to_or"])


# ═══════════════════════════════════════════════════════════════════════════
# F.  Cash Flow Solvency
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_ocf_to_opincome_signal",
    description="经营现金流/营业利润比率因子，ocf_to_opincome截面排名（高值排前）。",
    category="quality",
    thesis="经营现金流与营业利润的比率（ocf_to_opincome）度量盈利的现金含量。"
    "比率>1说明营业利润有充足的现金支撑（健康），比率<0.5则利润可能"
    "有'水分'（应收账款膨胀或收入确认激进）。是最重要的盈利质量指标之一。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_ocf_to_opincome_signal(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_opincome"]
    )
    return cross_sectional_rank(fin["ocf_to_opincome"])


@register_factor(
    name="ext_op_to_liquid_debt",
    description="营业利润/流动负债比率因子，op_to_liqdebt截面排名（高值排前）。",
    category="quality",
    thesis="营业利润对流动负债的覆盖率（op_to_liqdebt）衡量短期债务偿还能力。"
    "比率>1意味着一年营业利润可覆盖全部流动负债——非常安全。"
    "比率<0.2意味着流动性高度依赖再融资或资产变现。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_op_to_liquid_debt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["op_to_liqdebt"]
    )
    return cross_sectional_rank(fin["op_to_liqdebt"])


@register_factor(
    name="ext_cash_to_liquid_debt_with_interest",
    description="现金/（流动负债+利息）比率因子，cash_to_liqdebt_withinterest截面排名（高值排前）。",
    category="quality",
    thesis="cash_to_liqdebt_withinterest比普通现金比率更严格——"
    "分母包含了流动负债和应付利息。这是最保守的短期偿付能力测试。"
    "比率>1意味着现金足以覆盖所有短期债务+利息（极端安全）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_cash_to_liquid_debt_with_interest(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["cash_to_liqdebt_withinterest"]
    )
    return cross_sectional_rank(fin["cash_to_liqdebt_withinterest"])
