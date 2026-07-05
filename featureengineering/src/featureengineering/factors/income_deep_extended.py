"""
Deep extended income statement factors — Class 1 (Wave 2).

Second-wave factorisation of remaining unused income.parquet fields:
operating detail, net income attribution, comprehensive income,
profit distribution, and dividend signals.

Sections
--------
A. Operating Detail            — biz_tax_surchg, other_bus_cost, nca_disploss, pn_op_profit, oth_income
B. Net Income Attribution      — n_income_attr_p, non_fin_year_net_inc, continu_oper_np, termin_oper_np
C. Comprehensive Income        — OCI, TCI, financial asset remeasurement
D. Profit Distribution         — legal surplus, discretionary reserves, shareholder distribution, adjustments
E. Dividend Signals            — pref/ common dividend coverage, stock dividend ratio

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
# A.  Operating Detail
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_biz_tax_surchg_to_revenue",
    description="营业税金及附加/营收比率因子，biz_tax_surchg/revenue截面排名。",
    category="quality",
    thesis="营业税金及附加占营收比重反映企业的流转税负担（增值税附加、消费税等）。"
    "高比率常见于烟草、白酒、成品油等高消费税行业——"
    "这也意味着这些行业具有较高的政策壁垒（高税收=政府'入股'）。"
    "比率的行业间差异远大于行业内差异。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_biz_tax_surchg_to_revenue(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["biz_tax_surchg", "revenue"]
    )
    ratio = safe_divide(inc["biz_tax_surchg"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_other_bus_cost_to_revenue",
    description="其他业务成本/营收比率因子，other_bus_cost/revenue截面排名。",
    category="quality",
    thesis="其他业务成本占比高意味着非主营业务占用了大量成本资源——"
    "可能是'不务正业'的信号。持续高于行业均值需要关注其他业务的"
    "性质和盈利能力。高比率+低毛利=其他业务可能在亏损。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_other_bus_cost_to_revenue(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["other_bus_cost", "revenue"]
    )
    ratio = safe_divide(inc["other_bus_cost"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_nca_disploss_to_revenue",
    description="非流动资产处置损失/营收比率因子（低值排前）。",
    category="quality",
    thesis="非流动资产处置损失(nca_disploss)占比高意味着企业"
    "频繁处置固定资产/无形资产且以亏损价格出售——可能是"
    "设备陈旧或被迫出售的信号。该科目应接近于零，持续大额损失是红旗。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_nca_disploss_to_revenue(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["nca_disploss", "revenue"]
    )
    ratio = safe_divide(inc["nca_disploss"].abs(), inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_pn_op_profit_margin",
    description="营业利润率因子(IS alternate口径)，pn_op_profit/revenue截面排名（高值排前）。",
    category="quality",
    thesis="pn_op_profit（营业利润，扣除了财务费用前的利润口径）与operate_profit"
    "可能使用不同的计算口径。两个口径的差异反映了对财务费用和"
    "投资收益等项目的不同处理方式。使用pn_op_profit作为稳健性检验。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_pn_op_profit_margin(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["pn_op_profit", "revenue"]
    )
    margin = safe_divide(inc["pn_op_profit"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(margin)


@register_factor(
    name="ext_oth_income_to_revenue",
    description="其他收益/营收比率因子，oth_income/revenue截面排名。",
    category="quality",
    thesis="其他收益（oth_income）主要是与企业日常活动相关的政府补助——"
    "如增值税即征即退、研发补贴等。高占比意味着利润对政府补贴的依赖度高。"
    "补贴政策的变化可能导致盈利断崖式下跌。"
    "持续高额补贴的企业需要评估补贴的可持续性。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_oth_income_to_revenue(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["oth_income", "revenue"]
    )
    ratio = safe_divide(inc["oth_income"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Net Income Attribution
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_parent_share_of_ni",
    description="归母净利润/总净利润比率因子，n_income_attr_p/n_income截面排名（高值排前）。",
    category="quality",
    thesis="归母净利润占比反映归属于母公司股东的利润份额。"
    "占比高说明少数股东权益（非全资子公司）分走的利润少——"
    "控制力强。占比下降可能意味着核心子公司引入了外部投资者"
    "（稀释控制权）或非全资子公司的利润占比上升。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_parent_share_of_ni(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["n_income_attr_p", "n_income"]
    )
    ratio = safe_divide(inc["n_income_attr_p"], inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_continu_oper_np_ratio",
    description="持续经营净利润/总净利润比率因子（高值排前）。",
    category="quality",
    thesis="持续经营净利润占比(continu_oper_np/n_income)反映利润"
    "来自企业正常持续经营的程度。比率接近1说明利润来源稳定，"
    "比率<1说明终止经营业务贡献了部分利润（不可持续）。"
    "终止经营利润是一次性的——估值时应剔除。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_continu_oper_np_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["continu_oper_np", "n_income"]
    )
    ratio = safe_divide(inc["continu_oper_np"], inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_termin_oper_np_ratio",
    description="终止经营净利润/总净利润比率因子（低值排前）。",
    category="quality",
    thesis="终止经营净利润占比>0意味着企业正在退出某些业务——"
    "这可能是战略聚焦（剥离非核心业务，正面）或被迫止损（负面）。"
    "终止经营业务不再贡献未来利润——当前的利润水平不可外推。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_termin_oper_np_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["termin_oper_np", "n_income"]
    )
    ratio = safe_divide(inc["termin_oper_np"].abs(), inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# C.  Comprehensive Income
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_oci_to_net_income",
    description="其他综合收益/净利润比率因子，oth_compr_income/n_income截面排名。",
    category="quality",
    thesis="其他综合收益(OCI)与净利润的比率反映了'不计入利润但影响净资产'"
    "的项目的重要性。OCI绝对值接近或超过净利润意味着大量未实现损益"
    "（如可供出售金融资产、外币折算差额）——企业的真实经济表现与账面利润"
    "存在重大差异。负OCI（损失）可能预示着未来的利润减值。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_oci_to_net_income(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["oth_compr_income", "n_income"]
    )
    ratio = safe_divide(inc["oth_compr_income"].abs(), inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_tci_to_net_income",
    description="综合收益总额/净利润比率因子，t_compr_income/n_income截面排名。",
    category="quality",
    thesis="综合收益总额(t_compr_income = 净利润 + OCI)与净利润的比率"
    "揭示了OCI对整体业绩的影响方向。比率>1=OCI正贡献（净资产增加>利润），"
    "比率<1=OCI负贡献（净资产增加<利润——部分利润被OCI损失抵消）。"
    "持续比率<1意味着账面利润夸大了真实的价值创造。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_tci_to_net_income(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["t_compr_income", "n_income"]
    )
    ratio = safe_divide(inc["t_compr_income"], inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_net_fin_assets_remeasurement",
    description="金融资产重新计量净损益/净利润比率因子（低值排前）。",
    category="quality",
    thesis="net_exp_meas_fin_assets（金融资产重新计量产生的净损益）"
    "反映了金融资产分类或计量属性变更产生的一次性损益。"
    "该科目不应经常出现——频繁出现说明企业可能在通过金融资产"
    "重分类操纵利润。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_net_fin_assets_remeasurement(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["net_exp_meas_fin_assets", "n_income"]
    )
    ratio = safe_divide(inc["net_exp_meas_fin_assets"].abs(), inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# D.  Profit Distribution
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_legal_surplus_withdraw_ratio",
    description="法定盈余公积提取率因子，withdra_legal_surplus/n_income截面排名。",
    category="quality",
    thesis="法定盈余公积的提取比例（法定10%，达到注册资本50%后可不提）。"
    "提取率低于10%意味着企业已积累足够盈余公积——长期盈利的标志。"
    "持续按10%提取意味着盈余公积尚未达到上限——可能是新公司或盈利不多。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_legal_surplus_withdraw_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["withdra_legal_surplus", "n_income"]
    )
    ratio = safe_divide(inc["withdra_legal_surplus"], inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_profit_distribution_rate",
    description="利润分配率因子，distr_profit_shrhder/n_income截面排名（高值排前）。",
    category="quality",
    thesis="distr_profit_shrhder（应付股东利润）占净利润的比重"
    "反映了公司对股东的利润分配意愿。高分配率是股东友好型公司，"
    "但需确保分配后有足够资金用于再投资和发展。"
    "分配率的稳定性（而非高低）是优质红利股的特征。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_profit_distribution_rate(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["distr_profit_shrhder", "n_income"]
    )
    ratio = safe_divide(inc["distr_profit_shrhder"], inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_prf_reserve_to_profit",
    description="利润储备/利润总额比率因子，prf_reserve/total_profit截面排名（高值排前）。",
    category="quality",
    thesis="prf_reserve（利润储备）反映企业从利润中提取的各类准备金——"
    "体现了管理层对风险的审慎态度（平滑未来利润波动）。"
    "高利润储备率意味着利润确认偏保守（正面），但也可能是"
    "'cookie jar'储备（为未来释放利润做准备）。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_prf_reserve_to_profit(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["prf_reserve", "total_profit"]
    )
    ratio = safe_divide(inc["prf_reserve"], inc["total_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_adj_lossgain_ratio",
    description="以前年度损益调整/净利润比率因子（低值排前），adj_lossgain/n_income。",
    category="quality",
    thesis="以前年度损益调整(adj_lossgain)是对前期会计差错的更正——"
    "正金额=前期利润低估（本期调增），负金额=前期利润高估（本期调减）。"
    "大额以前年度调整意味着之前的财务报表存在重大差错——"
    "是财务报告质量差的信号。多次出现可能面临监管关注。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_adj_lossgain_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["adj_lossgain", "n_income"]
    )
    ratio = safe_divide(inc["adj_lossgain"].abs(), inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# E.  Dividend Signals
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_pref_dividend_commitment",
    description="优先股应付股息/净利润比率因子（低值排前）。",
    category="quality",
    thesis="prfshare_payable_dvd（应付优先股股利）是优先于普通股的分红义务。"
    "对于发行了优先股的企业，优先股股息是固定财务负担——"
    "必须在普通股分红前支付。高比率意味着优先股侵蚀了大量"
    "可用于普通股股东的利润。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_pref_dividend_commitment(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["prfshare_payable_dvd", "n_income"]
    )
    ratio = safe_divide(inc["prfshare_payable_dvd"].abs(), inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_common_dividend_coverage",
    description="普通股应付股利/净利润比率因子，comshare_payable_dvd/n_income截面排名。",
    category="quality",
    thesis="comshare_payable_dvd（应付普通股股利）占净利润比重"
    "是分红率的应付口径——反映了已宣告但可能尚未支付的分红。"
    "与现金流量表中的实际支付口径互补：应付>实付=分红递延（现金流压力），"
    "应付<实付=前期分红在本期支付。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_common_dividend_coverage(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["comshare_payable_dvd", "n_income"]
    )
    ratio = safe_divide(inc["comshare_payable_dvd"], inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_capitalized_dividend_ratio",
    description="股票股利/净利润比率因子，capit_comstock_div/n_income截面排名。",
    category="quality",
    thesis="capit_comstock_div（资本化普通股股利，即股票股利/送股）"
    "占净利润比重反映了公司以股票代替现金分红的倾向。"
    "送股不产生现金流出——在现金流紧张时可能是替代现金分红的策略。"
    "频繁送股而少现金分红=利润的'纸面'分配。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_ext_capitalized_dividend_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["capit_comstock_div", "n_income"]
    )
    ratio = safe_divide(inc["capit_comstock_div"], inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)
