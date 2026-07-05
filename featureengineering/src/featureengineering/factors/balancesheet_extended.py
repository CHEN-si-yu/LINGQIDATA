"""
Extended balance sheet factors — Class 1.

Systematic factorisation of remaining unused balancesheet.parquet fields.
After existing factors and unused_fields_factors.py (~10 factors from BS),
many asset-structure, liability-structure, and equity-quality columns remain
untapped.

Skipped:
- Insurance/banking columns (premium_receiv, reinsur_*, depos_*, reser_*, etc.)
- Old accounting standards (fa_avail_for_sale, htm_invest — 99%+ zero)
- 100%-zero columns (forex_differ, amor_exp)
- Identifier/metadata columns

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
# A.  Asset Structure
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_cash_to_assets",
    description="现金/总资产比率因子，money_cap/total_assets截面排名（高值排前）。",
    category="quality",
    thesis="现金占总资产比率反映企业的流动性储备和财务安全垫。"
    "高现金比意味着企业在经济下行期有更强的抗风险能力，"
    "同时具备逆势扩张（收购/回购）的弹药。现金为王。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_cash_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["money_cap", "total_assets"]
    )
    ratio = safe_divide(bs["money_cap"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_inventory_to_assets",
    description="存货/总资产比率因子（低值排前），inventories/total_assets截面排名。",
    category="quality",
    thesis="存货占比高意味着大量资金沉淀在库存中，面临减值风险和资金占用成本。"
    "存货占比上升而营收没有同步增长是需求疲软的预警信号。"
    "轻库存模式（低存货占比）是现代优秀企业的共同特征。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_inventory_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["inventories", "total_assets"]
    )
    ratio = safe_divide(bs["inventories"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_receivables_to_assets",
    description="应收款项/总资产比率因子（低值排前），(notes_receiv+accounts_receiv)/total_assets。",
    category="quality",
    thesis="应收账款和应收票据占总资产比例高意味着企业的收入质量差——"
    "产品卖出去了但钱没收到。应收占比持续上升是典型的'纸面利润'信号，"
    "后续可能面临大额坏账计提风险。该因子对识别财务造假特别有效。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_receivables_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["notes_receiv", "accounts_receiv", "total_assets"],
    )
    receivables = bs["notes_receiv"].fillna(0) + bs["accounts_receiv"].fillna(0)
    ratio = safe_divide(receivables, bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_fixed_assets_ratio",
    description="固定资产/总资产比率因子，fix_assets/total_assets截面排名。",
    category="quality",
    thesis="固定资产占比反映了企业的资产结构和经营模式。"
    "重资产模式（高固资占比）经营杠杆高——景气期利润弹性大，衰退期折旧负担重。"
    "轻资产模式（低固资占比）更加灵活，但可能缺乏护城河。"
    "该因子对判断企业周期性特征有参考价值。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_fixed_assets_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["fix_assets", "total_assets"]
    )
    ratio = safe_divide(bs["fix_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_intangible_assets_ratio",
    description="无形资产/总资产比率因子，intan_assets/total_assets截面排名（高值排前）。",
    category="quality",
    thesis="无形资产占比高通常意味着企业拥有专利、商标、技术授权等知识产权——"
    "这些是经济护城河的重要组成部分。但需注意：商誉不计入此指标（使用goodwill单独评估）。"
    "高无形资产占比在科技和医药行业中尤为正面。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_intangible_assets_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["intan_assets", "total_assets"]
    )
    ratio = safe_divide(bs["intan_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_current_assets_ratio",
    description="流动资产/总资产比率因子，total_cur_assets/total_assets截面排名。",
    category="quality",
    thesis="流动资产占比反映资产结构的流动性。高流动资产占比意味着"
    "企业资产变现能力强，但收益率可能较低。低流动资产占比意味着"
    "长期投资多、固定资产多——可能处于扩张期或属于重资产行业。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_current_assets_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["total_cur_assets", "total_assets"]
    )
    ratio = safe_divide(bs["total_cur_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Liability Structure
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_short_term_debt_ratio",
    description="短期借款/总资产比率因子（低值排前），st_borr/total_assets。",
    category="quality",
    thesis="短期借款占比高意味着企业对短期融资的依赖度大。短期借款需要频繁续贷，"
    "在信用紧缩期面临再融资风险。长短债结构合理的企业（长期债务为主）财务更稳健。"
    "高短期借款占比常见于资金链紧张的企业。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_short_term_debt_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["st_borr", "total_assets"]
    )
    ratio = safe_divide(bs["st_borr"], bs["total_assets"])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="ext_long_term_debt_ratio",
    description="长期借款/总资产比率因子，lt_borr/total_assets截面排名。",
    category="quality",
    thesis="长期借款占比反映企业获取长期资金的能力。能够获得长期借款"
    "（特别是银行贷款）意味着银行对企业的信用认可。但过高可能意味着"
    "过度依赖债务扩张，财务费用侵蚀利润。需要结合行业特征判断。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_long_term_debt_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["lt_borr", "total_assets"]
    )
    ratio = safe_divide(bs["lt_borr"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_trade_payables_ratio",
    description="应付账款/总资产比率因子（高值排前），(notes_payable+acct_payable)/total_assets。",
    category="quality",
    thesis="应付账款（含应付票据）占比高意味着企业占用上游供应商资金——"
    "这是'OPM战略'（Other People's Money）的体现。能够大量占用供应商资金"
    "说明企业在产业链中具有强势地位（如沃尔玛、茅台）。"
    "这是比有息负债'更好'的负债——无息且不增加财务费用。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_trade_payables_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["notes_payable", "acct_payable", "total_assets"],
    )
    payables = bs["notes_payable"].fillna(0) + bs["acct_payable"].fillna(0)
    ratio = safe_divide(payables, bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_advance_receipts_ratio",
    description="预收款项/总资产比率因子（高值排前），adv_receipts/total_assets。",
    category="quality",
    thesis="预收款项占比高是最强的商业模式信号之一——客户先付钱后拿货"
    "（如高端白酒、软件订阅、教育预付费）。预收款既是无息负债，"
    "又锁定了未来收入。高预收款占比的企业拥有卓越的定价权和客户粘性。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_advance_receipts_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["adv_receipts", "total_assets"]
    )
    ratio = safe_divide(bs["adv_receipts"], bs["total_assets"])
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# C.  Equity Quality
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_capital_reserve_ratio",
    description="资本公积/归母权益比率因子，cap_rese/total_hldr_eqy_exc_min_int截面排名。",
    category="quality",
    thesis="资本公积占归母权益的比例反映了股东权益的'质量'结构。"
    "高资本公积占比意味着权益主要来自股本溢价（IPO/增发），而非经营积累——"
    "企业尚未通过经营证明其价值创造能力。低资本公积占比+高留存收益占比"
    "意味着权益主要靠盈利积累，是内生增长的标志。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_capital_reserve_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["cap_rese", "total_hldr_eqy_exc_min_int"],
    )
    ratio = safe_divide(bs["cap_rese"], bs["total_hldr_eqy_exc_min_int"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)  # lower capital reserve ratio = more earned equity


@register_factor(
    name="ext_surplus_reserve_ratio",
    description="盈余公积/归母权益比率因子，surplus_rese/total_hldr_eqy_exc_min_int截面排名。",
    category="quality",
    thesis="盈余公积是从净利润中提取的法定积累，反映了企业历史上的盈利留存。"
    "高盈余公积占比意味着企业有长期的盈利记录和稳健的利润分配政策。"
    "盈余公积是利润分配的安全垫——可用于补亏或转增股本。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_surplus_reserve_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["surplus_rese", "total_hldr_eqy_exc_min_int"],
    )
    ratio = safe_divide(bs["surplus_rese"], bs["total_hldr_eqy_exc_min_int"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_undistributed_profit_ratio",
    description="未分配利润/归母权益比率因子，undistr_porfit/total_hldr_eqy_exc_min_int截面排名。",
    category="quality",
    thesis="未分配利润占比代表了企业历史上累积的、可供未来分配的利润池。"
    "高未分配利润占比意味着企业盈利能力强且尚未过度分红——"
    "这些留存利润可用于再投资（内生增长）或未来高分红。"
    "是'蓄水池'型企业的核心标志。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_undistributed_profit_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["undistr_porfit", "total_hldr_eqy_exc_min_int"],
    )
    ratio = safe_divide(bs["undistr_porfit"], bs["total_hldr_eqy_exc_min_int"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# D.  New Accounting Standard Items
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_right_of_use_assets_ratio",
    description="使用权资产/总资产比率因子，right_of_use_assets/total_assets截面排名。",
    category="quality",
    thesis="使用权资产（新租赁准则）反映了企业通过经营租赁使用的资产规模。"
    "高使用权资产占比常见于零售（门店租赁）、航空（飞机租赁）、"
    "物流（仓库租赁）等行业。新准则下租赁负债也同步确认，"
    "不影响资产负债表平衡但增加了表内杠杆。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_right_of_use_assets_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["right_of_use_assets", "total_assets"],
    )
    ratio = safe_divide(bs["right_of_use_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_lease_liability_ratio",
    description="租赁负债/总负债比率因子，lease_liab/total_liab截面排名。",
    category="quality",
    thesis="租赁负债占比（新准则）反映了企业的表外负债'回表'程度。"
    "高租赁负债占比意味着企业大量依赖租赁而非自持资产——"
    "在利率上升期租赁成本会增加，但同时也保持了资产的灵活性。"
    "波动大的行业中，租赁优于购买是理性的经营选择。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_lease_liability_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["lease_liab", "total_liab"],
    )
    ratio = safe_divide(bs["lease_liab"], bs["total_liab"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="ext_contract_assets_ratio",
    description="合同资产/总资产比率因子，contract_assets/total_assets截面排名。",
    category="quality",
    thesis="合同资产（新收入准则）是企业已履约但尚未取得无条件收款权的对价。"
    "与应收账款不同——合同资产还需满足其他条件才能收款（如里程碑验收）。"
    "合同资产占比高可能意味着收入确认进度快于实际收款进度，需关注。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_ext_contract_assets_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["contract_assets", "total_assets"],
    )
    ratio = safe_divide(bs["contract_assets"], bs["total_assets"])
    return cross_sectional_rank(ratio)
