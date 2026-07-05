"""
Deep extended balance sheet factors — Class 1 (Wave 2).

Second-wave factorisation of remaining unused balancesheet.parquet fields:
asset detail, liability opacity, equity quality, new standard items,
and composite opacity scores.

Sections
--------
A. Asset Detail          — prepayment, receivables sub-items, investment assets, bio/oil assets
B. Liability Detail      — bond payables, accrued items, deferred items, specific obligations
C. Equity & Other        — treasury shares, OCI, equity instruments, held-for-sale, FV assets
D. Opacity Composites    — balance sheet opacity score, receivables opacity, payables opacity

Data source: ``balancesheet.parquet`` (157 columns, quarterly via forward-fill)
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
# A.  Asset Detail
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_prepayment_to_assets",
    description="预付款项/总资产比率因子，prepayment/total_assets截面排名。",
    category="quality",
    thesis="预付款项占比反映企业对上游供应商的议价能力。"
    "高预付款占比意味着企业需要先付钱才能拿货——议价能力弱。"
    "低预付款占比（或大量预收）意味着企业可以占用他人资金——议价能力强。"
    "持续上升的预付款占比是供应链地位恶化的信号。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_prepayment_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["prepayment", "total_assets"]
    )
    ratio = safe_divide(bs["prepayment"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_div_receiv_to_assets",
    description="应收股利/总资产比率因子，div_receiv/total_assets截面排名。",
    category="quality",
    thesis="应收股利占比高意味着企业持有大量股权投资且已宣告分红——"
    "但现金尚未到账。这是未来现金流入的领先指标。"
    "但异常高的应收股利需关注子公司的分红能力和资金汇回限制。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_div_receiv_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["div_receiv", "total_assets"]
    )
    ratio = safe_divide(bs["div_receiv"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_nca_within_1y_to_assets",
    description="一年内到期非流动资产/总资产比率因子。",
    category="quality",
    thesis="一年内到期的非流动资产（nca_within_1y）代表了即将变现的长期资产。"
    "该比率高意味着未来12个月将有大量长期资产转化为现金（或应收）——"
    "是资产负债表流动性的前瞻指标。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_nca_within_1y_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["nca_within_1y", "total_assets"]
    )
    ratio = safe_divide(bs["nca_within_1y"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_oth_cur_assets_opacity",
    description="其他流动资产/总资产比率因子（低值排前），透明度信号。",
    category="quality",
    thesis="其他流动资产(oth_cur_assets)是资产负债表中的'黑箱'——"
    "包含了无法归入标准科目的各种流动资产。高占比意味着"
    "资产结构不透明，可能存在资金占用或隐藏问题。越低越好。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_oth_cur_assets_opacity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["oth_cur_assets", "total_assets"]
    )
    ratio = safe_divide(bs["oth_cur_assets"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_invest_real_estate_to_assets",
    description="投资性房地产/总资产比率因子，invest_real_estate/total_assets截面排名。",
    category="quality",
    thesis="投资性房地产占比高意味着企业持有大量非自用房产——"
    "可能是'隐形地产股'（如部分制造业企业持有大量升值房产），"
    "也可能是主业不振、转向地产投资。需关注公允价值计量方式。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_invest_real_estate_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["invest_real_estate", "total_assets"]
    )
    ratio = safe_divide(bs["invest_real_estate"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_const_materials_to_assets",
    description="工程物资/总资产比率因子，const_materials/total_assets截面排名。",
    category="quality",
    thesis="工程物资占比反映在建工程的备料规模——是未来capex的领先指标。"
    "工程物资增加（结合在建工程增加）意味着产能扩张正在推进，"
    "未来1-2年可能转固并开始产生收入。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_const_materials_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["const_materials", "total_assets"]
    )
    ratio = safe_divide(bs["const_materials"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_biological_assets_to_assets",
    description="生产性生物资产/总资产比率因子，produc_bio_assets/total_assets截面排名。",
    category="quality",
    thesis="生产性生物资产（produc_bio_assets）是农林牧渔企业的核心资产"
    "（如种畜、果树、林木）。该科目的估值高度依赖管理层判断，"
    "历史上多次出现生物资产造假案例（如扇贝逃跑）。"
    "高生物资产占比的企业需要额外审慎。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_biological_assets_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["produc_bio_assets", "total_assets"]
    )
    ratio = safe_divide(bs["produc_bio_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_oil_gas_assets_to_assets",
    description="油气资产/总资产比率因子，oil_and_gas_assets/total_assets截面排名。",
    category="quality",
    thesis="油气资产占比是能源企业的核心特征。油气资产的价值与国际油价"
    "高度相关——油价下跌时面临减值风险。该比率结合油价趋势可判断"
    "能源企业的资产质量周期。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_oil_gas_assets_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["oil_and_gas_assets", "total_assets"]
    )
    ratio = safe_divide(bs["oil_and_gas_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_lt_amor_exp_to_assets",
    description="长期待摊费用/总资产比率因子（低值排前）。",
    category="quality",
    thesis="长期待摊费用（lt_amor_exp）是已经支付但需分期摊销的支出——"
    "如装修费、大修理费等。高占比意味着大量支出被资本化而非费用化"
    "（当期费用被低估、利润被高估）。持续上升的长期待摊费用是盈余管理的常见信号。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_lt_amor_exp_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["lt_amor_exp", "total_assets"]
    )
    ratio = safe_divide(bs["lt_amor_exp"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_oth_nca_opacity",
    description="其他非流动资产/总资产比率因子（低值排前），透明度信号。",
    category="quality",
    thesis="其他非流动资产(oth_nca)是长期资产端的'黑箱'科目。"
    "高占比意味着大量资产无法归入标准分类——不透明。"
    "财务造假企业往往在'其他'科目中隐藏问题。越低越透明。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_oth_nca_opacity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["oth_nca", "total_assets"]
    )
    ratio = safe_divide(bs["oth_nca"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_debt_invest_to_assets",
    description="债权投资/总资产比率因子(新准则)，debt_invest/total_assets截面排名。",
    category="quality",
    thesis="debt_invest（新金融工具准则下的债权投资，以摊余成本计量）"
    "反映企业持有的债券等固定收益资产。高占比常见于保险公司和财务公司。"
    "对一般企业，高债权投资占比意味着资金未用于主业。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_debt_invest_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["debt_invest", "total_assets"]
    )
    ratio = safe_divide(bs["debt_invest"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_oth_debt_invest_to_assets",
    description="其他债权投资/总资产比率因子(新准则)，oth_debt_invest/total_assets。",
    category="quality",
    thesis="oth_debt_invest（以公允价值计量且变动计入OCI的债权投资）"
    "的市值波动直接影响其他综合收益(OCI)。高占比意味着"
    "企业的净资产受到利率和信用利差波动的显著影响。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_oth_debt_invest_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["oth_debt_invest", "total_assets"]
    )
    ratio = safe_divide(bs["oth_debt_invest"], bs["total_assets"])
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Liability Detail
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_st_bonds_payable_to_assets",
    description="应付短期债券/总资产比率因子，st_bonds_payable/total_assets截面排名。",
    category="quality",
    thesis="应付短期债券占比反映了企业对短期债券融资（如超短融SCP）的依赖。"
    "短期债券需要频繁滚动发行——在信用事件中面临'展期失败'风险。"
    "短期债券+短期借款双高是流动性危机的经典前兆。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_st_bonds_payable_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["st_bonds_payable", "total_assets"]
    )
    ratio = safe_divide(bs["st_bonds_payable"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_accrued_expenses_to_assets",
    description="预提费用/总资产比率因子，acc_exp/total_assets截面排名。",
    category="quality",
    thesis="预提费用（acc_exp）是企业已发生但尚未支付的费用——"
    "代表了即将产生的现金流出。占比突然上升可能意味着费用确认激进"
    "（当期费用高估→利润低估→未来利润释放）或费用支付延迟（现金流问题）。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_accrued_expenses_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["acc_exp", "total_assets"]
    )
    ratio = safe_divide(bs["acc_exp"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_deferred_inc_to_assets",
    description="递延收益/总资产比率因子，deferred_inc/total_assets截面排名（高值排前）。",
    category="quality",
    thesis="递延收益（deferred_inc）主要是与资产相关的政府补助——"
    "政府给钱让企业买设备/建产线，补助随折旧分期确认收益。"
    "高递延收益占比意味着企业获得了大量政府支持（政策优势），"
    "未来折旧期内将持续确认非经常性收益。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_deferred_inc_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["deferred_inc", "total_assets"]
    )
    ratio = safe_divide(bs["deferred_inc"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_oth_cur_liab_opacity",
    description="其他流动负债/总资产比率因子（低值排前），透明度信号。",
    category="quality",
    thesis="其他流动负债(oth_cur_liab)是短期负债端的'黑箱'——"
    "包含了无法归入标准科目的各种短期债务。高占比是财务不透明的信号。"
    "投资者无法判断其中是否隐藏了未披露的担保或借款。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_oth_cur_liab_opacity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["oth_cur_liab", "total_assets"]
    )
    ratio = safe_divide(bs["oth_cur_liab"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_estimated_liab_to_assets",
    description="预计负债/总资产比率因子，estimated_liab/total_assets截面排名。",
    category="quality",
    thesis="预计负债（estimated_liab）是企业对很可能发生的或有事项的"
    "预先计提——如未决诉讼、产品质量保证、资产 restoration义务。"
    "占比突然上升意味着企业面临重大不确定性事件（如环境污染诉讼）。"
    "该科目的大小直接反映了表外风险的表内化程度。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_estimated_liab_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["estimated_liab", "total_assets"]
    )
    ratio = safe_divide(bs["estimated_liab"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_oth_ncl_opacity",
    description="其他非流动负债/总资产比率因子（低值排前），透明度信号。",
    category="quality",
    thesis="其他非流动负债(oth_ncl)是长期负债端的'黑箱'。"
    "高占比意味着有大量无法归类的长期债务——不透明。"
    "投资者应查看附注中oth_ncl的具体构成以判断风险。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_oth_ncl_opacity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["oth_ncl", "total_assets"]
    )
    ratio = safe_divide(bs["oth_ncl"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# C.  Equity & Other Items
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_treasury_share_to_equity",
    description="库存股/归母权益比率因子（高值排前），treasury_share/total_hldr_eqy_exc_min_int（取正）。",
    category="quality",
    thesis="库存股（treasury_share）是公司回购并持有的自身股票——"
    "通常记录为权益的减项（负值）。库存股绝对值大意味着公司"
    "进行了大规模的股票回购——这是管理层认为股价低估的强烈信号。"
    "回购注销可以提高EPS和ROE。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_treasury_share_to_equity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["treasury_share", "total_hldr_eqy_exc_min_int"],
    )
    # treasury_share is usually negative (equity reduction), take abs value
    ratio = safe_divide(bs["treasury_share"].abs(), bs["total_hldr_eqy_exc_min_int"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_invest_loss_unconf_to_equity",
    description="未确认投资损失/归母权益比率因子（低值排前）。",
    category="quality",
    thesis="invest_loss_unconf（未确认的投资损失）是权益的减项——"
    "代表了被投资单位发生的、超过投资账面价值的超额亏损。"
    "该科目为正（减项大）意味着被投资企业严重亏损，"
    "可能面临进一步的投资减值或追加投资（'输血'）压力。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_invest_loss_unconf_to_equity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["invest_loss_unconf", "total_hldr_eqy_exc_min_int"],
    )
    ratio = safe_divide(bs["invest_loss_unconf"].abs(), bs["total_hldr_eqy_exc_min_int"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_oth_comp_income_to_equity",
    description="其他综合收益/归母权益比率因子，oth_comp_income/total_hldr_eqy_exc_min_int。",
    category="quality",
    thesis="其他综合收益（OCI）占归母权益的比重反映净资产受"
    "非利润表项目（如外币报表折算、AFS公允价值变动、养老金精算）"
    "的影响程度。OCI波动大意味着净资产不稳定——虽然不影响当期利润，"
    "但影响企业真实的价值创造。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_oth_comp_income_to_equity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["oth_comp_income", "total_hldr_eqy_exc_min_int"],
    )
    ratio = safe_divide(bs["oth_comp_income"].abs(), bs["total_hldr_eqy_exc_min_int"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_hfs_assets_to_assets",
    description="持有待售资产/总资产比率因子，hfs_assets/total_assets截面排名。",
    category="quality",
    thesis="持有待售资产（hfs_assets，held-for-sale）是企业已决定出售"
    "并很可能在一年内完成的资产组——不再计提折旧。高持有待售占比"
    "意味着企业正在进行重大资产重组或业务剥离。需要关注出售目的"
    "（聚焦主业vs自救变现）。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_hfs_assets_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["hfs_assets", "total_assets"]
    )
    ratio = safe_divide(bs["hfs_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_fair_value_fin_assets_ratio",
    description="公允价值金融资产/总资产比率因子，fair_value_fin_assets/total_assets。",
    category="quality",
    thesis="以公允价值计量的金融资产（fair_value_fin_assets）"
    "市值波动直接影响当期损益（FVTPL）或OCI（FVOCI）。"
    "高占比意味着企业利润或净资产受到金融市场波动的显著影响。"
    "在熊市中面临公允价值损失风险。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_fair_value_fin_assets_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["fair_value_fin_assets", "total_assets"]
    )
    ratio = safe_divide(bs["fair_value_fin_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_cost_fin_assets_to_assets",
    description="摊余成本金融资产/总资产比率因子，cost_fin_assets/total_assets。",
    category="quality",
    thesis="以摊余成本计量的金融资产（cost_fin_assets）不受市价波动影响——"
    "收益来自利息收入而非公允价值变动。高占比意味着利润稳定性好，"
    "但也意味着资产收益率受限于合同利率。主要是债券投资和贷款。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_cost_fin_assets_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["cost_fin_assets", "total_assets"]
    )
    ratio = safe_divide(bs["cost_fin_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_cip_total_to_assets",
    description="在建工程(合计)/总资产比率因子，cip_total/total_assets截面排名。",
    category="quality",
    thesis="在建工程合计(cip_total,新准则)占总资产比重反映未来的产能扩张。"
    "在建工程占比高+营收增长=健康的扩张周期。"
    "在建工程占比高+营收停滞=过度投资或工程延误风险。"
    "需关注在建工程转固的节奏——延迟转固可能是避免折旧的盈余管理。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_cip_total_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["cip_total", "total_assets"]
    )
    ratio = safe_divide(bs["cip_total"], bs["total_assets"])
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# D.  Opacity Composites
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_bs_opacity_score",
    description="资产负债表不透明度综合因子（低值排前），四个'其他'科目合计/总资产。",
    category="quality",
    thesis="资产负债表的不透明度综合评分 = (oth_cur_assets + oth_nca + "
    "oth_cur_liab + oth_ncl) / total_assets。'其他'类科目是财务报表"
    "中最不透明的部分——高分意味着大量资产负债无法归入标准分类。"
    "学术研究表明不透明度与未来负收益、财务重述概率正相关。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_bs_opacity_score(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["oth_cur_assets", "oth_nca", "oth_cur_liab", "oth_ncl", "total_assets"],
    )
    opacity = (
        bs["oth_cur_assets"].fillna(0)
        + bs["oth_nca"].fillna(0)
        + bs["oth_cur_liab"].fillna(0)
        + bs["oth_ncl"].fillna(0)
    )
    ratio = safe_divide(opacity, bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_receivables_opacity",
    description="应收款项不透明度因子（低值排前），(oth_receiv+oth_rcv_total+lt_rec)/总资产。",
    category="quality",
    thesis="非标准应收款项占比=(其他应收款+其他应收款合计+长期应收款)/总资产。"
    "其他应收款是'垃圾桶'科目——经常被用于隐藏关联方资金占用、"
    "大股东占款等。大额其他应收款是财务造假的经典红旗信号。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_receivables_opacity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["oth_receiv", "oth_rcv_total", "lt_rec", "total_assets"],
    )
    opacity = (
        bs["oth_receiv"].fillna(0)
        + bs["oth_rcv_total"].fillna(0)
        + bs["lt_rec"].fillna(0)
    )
    ratio = safe_divide(opacity, bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_payables_opacity",
    description="应付款项不透明度因子（低值排前），(oth_payable+oth_pay_total+oth_cur_liab)/总资产。",
    category="quality",
    thesis="非标准应付款项占比=(其他应付款+其他应付款合计+其他流动负债)/总资产。"
    "与应收款的不透明度对称——其他应付款可能隐藏了未披露的借款、"
    "关联方资金拆借或表外负债。高度不透明的负债结构是风险评估的重点。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_payables_opacity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["oth_payable", "oth_pay_total", "oth_cur_liab", "total_assets"],
    )
    opacity = (
        bs["oth_payable"].fillna(0)
        + bs["oth_pay_total"].fillna(0)
        + bs["oth_cur_liab"].fillna(0)
    )
    ratio = safe_divide(opacity, bs["total_assets"])
    return cross_sectional_rank(-ratio)
