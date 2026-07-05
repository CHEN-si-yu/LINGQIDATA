"""
Factor implementations based on previously unused upstream data fields.

This module fills gaps identified in the comprehensive field audit (2026-07-02).
All factors use fields that were NOT referenced in any existing factor code,
yet have excellent data quality (>95% coverage, <5% NaN for target stock pool).

Data coverage baseline: 1,782 stocks in Code_num.txt, 2020-01-01 to 2026-07-01.

Sections
--------
A.  Financial Indicator (financial_indicator.parquet) — unused fields
B.  Balance Sheet (balancesheet.parquet) — unused fields
C.  Income Statement (income.parquet) — unused fields
D.  Cash Flow (cashflow.parquet) — unused fields
E.  Main Fund Flow Volume (main_fund_flow.parquet) — volume-based ratios
F.  Limit List (limit_list.parquet) — unused event fields
G.  Finance Panel (finance.parquet) — unused field
H.  Comprehensive / Composite factors from unused fields
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    rolling_group_sum,
    safe_divide,
)

# ═══════════════════════════════════════════════════════════════════════════
# A.  Financial Indicator — previously unused fields
#    Source: financial_indicator.parquet (163 columns, 85 unused)
#    Coverage: 89,090 rows, 1782/1782 stocks, 2020-03-31 to 2026-03-31
# ═══════════════════════════════════════════════════════════════════════════

# ── A.1  Free Cash Flow Quality ────────────────────────────────────────────

@register_factor(
    name="uf_fcf_yield",
    description="企业自由现金流收益率因子，FCFF/总市值截面排名。",
    category="quality",
    thesis="自由现金流是股东可支配的真实现金回报，FCFF Yield高的公司被低估，具有显著的价值溢价。"
    "FCFF相比净利润更难被会计操纵。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_uf_fcf_yield(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["fcff"]
    )
    finance = context.load("finance.parquet")
    fcff_per_share = fin["fcff"]
    total_mv = finance["total_mv"]
    # FCFF / Total Market Value
    fcf_yield = safe_divide(fcff_per_share, total_mv)
    return cross_sectional_rank(fcf_yield)


@register_factor(
    name="fcfe_yield",
    description="股权自由现金流收益率因子，FCFE/总市值截面排名。",
    category="quality",
    thesis="FCFE(Yield)是股东角度的可分配现金流，扣除债务本息后更准确地衡量股东回报能力。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_fcfe_yield(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["fcfe"]
    )
    finance = context.load("finance.parquet")
    fcfe_val = fin["fcfe"]
    total_mv = finance["total_mv"]
    fcfe_yield = safe_divide(fcfe_val, total_mv)
    return cross_sectional_rank(fcfe_yield)


@register_factor(
    name="uf_fcf_conversion",
    description="现金流转化率因子，(FCFF-OCF)/OCF绝对值，度量自由现金流与经营现金流的偏离。",
    category="quality",
    thesis="自由现金流与经营现金流的差异反映资本开支强度，高转化率代表轻资产模式。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_uf_fcf_conversion(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["fcff", "ocfps"]
    )
    ocf = fin["ocfps"].abs().replace(0, np.nan)
    conversion = (fin["fcff"] - fin["ocfps"]) / ocf  # negative = heavy capex
    return cross_sectional_rank(-conversion)  # higher rank = lighter capex


# ── A.2  Capital Structure & Efficiency ────────────────────────────────────

@register_factor(
    name="invest_capital_turnover",
    description="投入资本周转率因子，营收/投入资本截面排名。",
    category="quality",
    thesis="投入资本（Invested Capital）衡量企业经营实际占用的资本，高周转率意味着更高效的资本利用。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_invest_capital_turnover(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["invest_capital", "total_revenue_ps"],
    )
    ic_per_share = fin["invest_capital"]
    rev_per_share = fin["total_revenue_ps"]
    turnover = safe_divide(rev_per_share, ic_per_share)
    return cross_sectional_rank(turnover)


@register_factor(
    name="uf_net_debt_to_ebitda",
    description="净负债/EBITDA因子（低值排前），衡量真实偿债压力。",
    category="quality",
    thesis="净负债（有息负债-现金）/EBITDA是更准确的杠杆指标，扣除了可立即偿债的现金。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_uf_net_debt_to_ebitda(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["netdebt", "ebitda"]
    )
    ratio = safe_divide(fin["netdebt"], fin["ebitda"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)  # lower leverage = higher rank


@register_factor(
    name="uf_working_capital_to_assets",
    description="营运资本/总资产因子，度量短期经营效率。",
    category="quality",
    thesis="营运资本占比反映企业对上下游的资金占用能力，负营运资本代表占用他人资金经营（好商业模式）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_uf_working_capital_to_assets(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["working_capital", "tangible_asset"],
    )
    # Using tangible_asset as proxy for total assets scaling
    ratio = safe_divide(fin["working_capital"], fin["tangible_asset"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)  # negative WC = good


@register_factor(
    name="interest_bearing_debt_ratio",
    description="有息负债率因子，(current_exint+noncurrent_exint)/总资产，低值排前。",
    category="quality",
    thesis="有息负债才是真正的财务负担，区分于应付账款等无息负债，更准确衡量财务风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_interest_bearing_debt_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["current_exint", "noncurrent_exint", "tangible_asset"],
    )
    total_int_debt = fin["current_exint"].fillna(0) + fin["noncurrent_exint"].fillna(0)
    ratio = safe_divide(total_int_debt, fin["tangible_asset"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ── A.3  Earnings Quality & Composition ────────────────────────────────────

@register_factor(
    name="non_operating_profit_ratio",
    description="非经营利润占比因子（低值排前），度量盈利的可持续性。",
    category="quality",
    thesis="非经营利润（non_op_profit/利润总额）占比高意味着盈利来自一次性项目，质量和可持续性差。"
    "持续经营利润占比高的公司享有质量溢价。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_non_operating_profit_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["non_op_profit", "profit_to_gr"],
    )
    ratio = safe_divide(fin["non_op_profit"].abs(), fin["profit_to_gr"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="operating_profit_to_ebt",
    description="经营利润/EBT比率因子，衡量盈利来源的纯度。",
    category="quality",
    thesis="经营利润/利润总额比率高说明盈利主要来自主营业务，可持续性强。低比率暗示依赖投资收益或非经常性项目。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_operating_profit_to_ebt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["opincome_of_ebt"]
    )
    return cross_sectional_rank(fin["opincome_of_ebt"])


@register_factor(
    name="deducted_profit_ratio",
    description="扣非净利润/净利润比率因子，度量盈余质量。",
    category="quality",
    thesis="扣非净利润占比越接近100%，盈利越真实可靠。低于1说明存在大量非经常性损益。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_deducted_profit_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["dtprofit_to_profit"]
    )
    return cross_sectional_rank(fin["dtprofit_to_profit"])


# ── A.4  Cost Structure ────────────────────────────────────────────────────

@register_factor(
    name="cogs_ratio",
    description="营业成本率因子（低值排前），度量主营业务成本控制能力。",
    category="quality",
    thesis="营业成本率低的公司具有供应链优势或产品定价权。低营业成本率通常对应高毛利率。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_cogs_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["cogs_of_sales"]
    )
    return cross_sectional_rank(-fin["cogs_of_sales"])


@register_factor(
    name="expense_to_sales",
    description="期间费用率因子（低值排前），(销售+管理+财务费用)/营收。",
    category="quality",
    thesis="费用率低的公司运营效率高，管理能力强。费用率的下降趋势是盈利改善的先行指标。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_expense_to_sales(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["expense_of_sales"]
    )
    return cross_sectional_rank(-fin["expense_of_sales"])


@register_factor(
    name="tax_burden_ratio",
    description="实际税率因子（低值排前），所得税/利润总额。",
    category="quality",
    thesis="实际税率异常偏高可能暗示利润质量差（不可抵扣费用多），而高新技术企业享受低税率。"
    "合理范围内的低税率是企业竞争优势的体现。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_tax_burden_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["tax_to_ebt"]
    )
    return cross_sectional_rank(-fin["tax_to_ebt"])


# ── A.5  Quarterly Growth (Higher Frequency) ───────────────────────────────

@register_factor(
    name="q_netprofit_qoq",
    description="单季度归母净利润环比增速因子截面排名。",
    category="quality",
    thesis="季度环比增速捕捉最新的盈利动量变化，相比年度同比更及时地反映经营趋势转折。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_netprofit_qoq(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_netprofit_qoq"]
    )
    return cross_sectional_rank(fin["q_netprofit_qoq"])


@register_factor(
    name="q_sales_qoq",
    description="单季度营收环比增速因子截面排名。",
    category="quality",
    thesis="营收环比增速是盈利增长的先行指标，收入先于利润拐点。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_sales_qoq(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_sales_qoq"]
    )
    return cross_sectional_rank(fin["q_sales_qoq"])


@register_factor(
    name="q_roe",
    description="单季度ROE因子截面排名，更高频率的盈利能力度量。",
    category="quality",
    thesis="单季度ROE比TTM ROE对盈利拐点更敏感，适合捕捉业绩反转。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_roe(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_roe"]
    )
    return cross_sectional_rank(fin["q_roe"])


@register_factor(
    name="q_netprofit_margin",
    description="单季度净利率因子截面排名。",
    category="quality",
    thesis="单季度净利率剔除了历史季度的平滑效应，能更纯粹地反映当前经营质量。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_netprofit_margin(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_netprofit_margin"]
    )
    return cross_sectional_rank(fin["q_netprofit_margin"])


@register_factor(
    name="q_ocf_to_sales",
    description="单季度经营现金流/营收因子，度量盈利的现金含量。",
    category="quality",
    thesis="高OCF/营收意味着利润伴随真实现金流入，低比率可能暗示应收账款堆积或盈余管理。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_ocf_to_sales(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_ocf_to_sales"]
    )
    return cross_sectional_rank(fin["q_ocf_to_sales"])


# ── A.6  EPS Variants ──────────────────────────────────────────────────────

@register_factor(
    name="diluted_eps_yield",
    description="稀释EPS/股价因子截面排名（使用dt_eps），考虑潜在摊薄后的真实每股收益。",
    category="valuation",
    thesis="稀释EPS考虑了可转债、期权等潜在摊薄因素，比基本EPS更保守、更真实。"
    "dt_eps_yield = dt_eps / price",
    dependencies=("financial_indicator.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_diluted_eps_yield(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["dt_eps"]
    )
    daily = context.load("daily_adj.parquet")
    eps_val = fin["dt_eps"]
    price = daily["close"]
    eps_yield = safe_divide(eps_val, price)
    return cross_sectional_rank(eps_yield)


@register_factor(
    name="eps_growth_4q_qoq",
    description="单季度EPS的4个季度累计同比增速因子。",
    category="growth",
    thesis="基于dt_eps计算的4Q累计增速比basic_eps_yoy更真实地反映盈利增长（考虑了潜在摊薄）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_eps_growth_4q_qoq(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["dt_eps_yoy", "dt_netprofit_yoy"]
    )
    # Average of diluted EPS YoY and diluted net profit YoY
    growth = fin["dt_eps_yoy"].fillna(0) * 0.5 + fin["dt_netprofit_yoy"].fillna(0) * 0.5
    return cross_sectional_rank(growth)


# ── A.7  Capital Structure (Advanced) ──────────────────────────────────────

@register_factor(
    name="equity_to_invested_capital",
    description="权益/投入资本比率因子，度量资本结构中的股权依赖度。",
    category="quality",
    thesis="权益占比高意味着低杠杆、财务稳健。债务占比高的公司享受税盾但也承担更大的破产风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_equity_to_invested_capital(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["eqt_to_talcapital"]
    )
    return cross_sectional_rank(fin["eqt_to_talcapital"])


@register_factor(
    name="int_bearing_debt_to_capital",
    description="有息负债/总资本因子（低值排前），度量真实杠杆水平。",
    category="quality",
    thesis="仅有息负债才构成财务费用负担，该比率相比传统的debt_to_assets更能区分真实财务风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_int_bearing_debt_to_capital(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["int_to_talcap"]
    )
    return cross_sectional_rank(-fin["int_to_talcap"])


@register_factor(
    name="retained_earnings_to_assets",
    description="留存收益/总资产因子，度量企业自我积累能力。",
    category="quality",
    thesis="留存收益占比高的公司有长期稳定的盈利积累，是历史上盈利能力的综合证明。"
    "高留存收益资产比是Piotroski F-Score的重要组成部分。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_retained_earnings_to_assets(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["retained_earnings", "tangible_asset"],
    )
    ratio = safe_divide(fin["retained_earnings"], fin["tangible_asset"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Balance Sheet — previously unused fields
#    Source: balancesheet.parquet (149 value columns, 114 unused)
#    Coverage: 48,075 rows, 1782/1782 stocks, 2020-03-31 to 2026-03-31
# ═══════════════════════════════════════════════════════════════════════════

# ── B.1  Intangible & Goodwill Quality ─────────────────────────────────────

@register_factor(
    name="goodwill_to_equity",
    description="商誉/净资产因子（低值排前），度量并购风险的累积。",
    category="quality",
    thesis="商誉占比高意味着大量溢价并购，减值风险大。A股历史上商誉减值潮导致大量暴雷，"
    "低商誉占比是财务稳健的信号。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_goodwill_to_equity(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["goodwill", "total_hldr_eqy_exc_min_int"],
    )
    ratio = safe_divide(bs["goodwill"], bs["total_hldr_eqy_exc_min_int"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="rd_to_revenue",
    description="研发费用/营收因子（高值排前），度量创新投入强度。",
    category="quality",
    thesis="研发投入占比高的公司具有长期增长潜力，特别是在科技和医药行业。"
    "但需要结合资本化政策判断研发质量。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_rd_to_revenue(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["r_and_d"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = safe_divide(bs["r_and_d"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ── B.2  Strategic Investment ──────────────────────────────────────────────

@register_factor(
    name="lt_equity_invest_to_assets",
    description="长期股权投资/总资产因子，度量战略投资布局程度。",
    category="quality",
    thesis="长期股权投资反映了企业的产业链整合与战略联盟布局。"
    "适度的长期股权投资对协同发展有利，过高则可能隐藏表外风险。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_lt_equity_invest_to_assets(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["lt_eqt_invest", "total_assets"],
    )
    ratio = safe_divide(bs["lt_eqt_invest"], bs["total_assets"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ── B.3  Deferred Tax Analysis ─────────────────────────────────────────────

@register_factor(
    name="deferred_tax_asset_ratio",
    description="递延所得税资产/总资产因子（低值排前），度量税务质量。",
    category="quality",
    thesis="递延所得税资产过高可能意味着大量可抵扣亏损（盈利差），或激进的收入确认政策。"
    "低递延税资产的利润更真实可靠。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_deferred_tax_asset_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["defer_tax_assets", "total_assets"],
    )
    ratio = safe_divide(bs["defer_tax_assets"], bs["total_assets"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="uf_deferred_tax_liab_ratio",
    description="递延所得税负债/总资产因子，度量加速折旧等税务筹划效果。",
    category="quality",
    thesis="递延所得税负债通常由加速折旧产生，代表企业充分利用了税务优惠。"
    "适度递延税负债是税务管理的积极信号（无息递延税款）。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_uf_deferred_tax_liab_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["defer_tax_liab", "total_assets"],
    )
    ratio = safe_divide(bs["defer_tax_liab"], bs["total_assets"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ── B.4  Minority Interest ─────────────────────────────────────────────────

@register_factor(
    name="uf_minority_interest_ratio",
    description="少数股东权益/净资产因子（低值排前），度量归母净利润的纯度。",
    category="quality",
    thesis="少数股东权益占比高意味着大量利润归属于外部股东而非上市公司股东。"
    "归母净利润与合并净利润的差异越大，归母利润的含金量越低。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_uf_minority_interest_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["minority_int", "total_hldr_eqy_inc_min_int"],
    )
    ratio = safe_divide(bs["minority_int"], bs["total_hldr_eqy_inc_min_int"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ── B.5  Contract Liability (Revenue Quality) ──────────────────────────────

@register_factor(
    name="contract_liab_to_revenue",
    description="合同负债/营收因子（高值排前），度量预收款质量和未来收入保障。",
    category="quality",
    thesis="合同负债（预收账款）代表已收款但尚未确认的收入，高合同负债意味着"
    "未来收入的确定性高，是收入质量的先行指标（白酒行业的典型特征）。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_contract_liab_to_revenue(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["contract_liab"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = safe_divide(bs["contract_liab"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ── B.6  Asset Structure ───────────────────────────────────────────────────

@register_factor(
    name="non_current_asset_ratio",
    description="非流动资产/总资产因子，度量资产结构轻重。",
    category="quality",
    thesis="非流动资产占比反映企业的资产结构（轻资产vs重资产）。"
    "轻资产模式通常具有更高的经营灵活性和ROIC。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_non_current_asset_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["total_nca", "total_assets"],
    )
    ratio = safe_divide(bs["total_nca"], bs["total_assets"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)  # lighter asset = higher rank


@register_factor(
    name="non_current_liab_ratio",
    description="非流动负债/总负债因子（低值排前），度量负债期限结构。",
    category="quality",
    thesis="非流动负债占比高意味着短期偿债压力小，债务结构稳健。"
    "短债长投是经典的财务风险信号。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_non_current_liab_ratio(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["total_ncl", "total_liab"],
    )
    ratio = safe_divide(bs["total_ncl"], bs["total_liab"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# C.  Income Statement — previously unused fields
#    Source: income.parquet (88 value columns, 65 unused)
#    Coverage: 48,059 rows, 1782/1782 stocks, 2020-03-31 to 2026-03-31
# ═══════════════════════════════════════════════════════════════════════════

# ── C.1  R&D Intensity (from income, more accurate) ────────────────────────

@register_factor(
    name="rd_expense_to_revenue",
    description="研发费用/营收因子（来自利润表rd_exp字段），度量真实研发投入。",
    category="quality",
    thesis="利润表中的研发费用是当期实际费用化的研发支出，相比资产负债表的r_and_d更能反映"
    "当期研发投入强度。该因子在科技和医药行业截面有显著的alpha。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_rd_expense_to_revenue(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["rd_exp", "revenue"]
    )
    ratio = safe_divide(inc["rd_exp"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ── C.2  Financial Income/Expense Detail ───────────────────────────────────

@register_factor(
    name="net_interest_margin",
    description="净利息收入/营收因子（金融企业为正），度量利息收支结构。",
    category="quality",
    thesis="利息收入与利息费用的差额反映了企业的净现金管理能力。"
    "利息收入大于费用（正值）意味着企业是净债权人，财务稳健。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_net_interest_margin(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["fin_exp_int_inc", "fin_exp_int_exp", "revenue"]
    )
    net_int = inc["fin_exp_int_inc"].fillna(0) - inc["fin_exp_int_exp"].fillna(0)
    ratio = safe_divide(net_int, inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="financial_expense_ratio",
    description="财务费用率因子（低值排前），度量融资成本负担。",
    category="quality",
    thesis="财务费用率高意味着依赖债务融资，财务负担重。在利率上升周期中，"
    "高财务费用率的公司面临更大的盈利压力。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_financial_expense_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["fin_exp", "revenue"]
    )
    ratio = safe_divide(inc["fin_exp"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ── C.3  Associate Investment Income ───────────────────────────────────────

@register_factor(
    name="associate_invest_income_ratio",
    description="联营/合营企业投资收益/利润总额因子，度量对外投资依赖度。",
    category="quality",
    thesis="联营企业投资收益占比过高意味着主业盈利能力不足，利润来自非控制的被投资企业。"
    "该收益对上市公司而言缺乏现金支配力（通常不并表分红）。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_associate_invest_income_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["ass_invest_income", "total_profit"]
    )
    ratio = safe_divide(inc["ass_invest_income"].abs(), inc["total_profit"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ── C.4  Minor Gain / EPS Quality ──────────────────────────────────────────

@register_factor(
    name="minority_gain_ratio",
    description="少数股东损益/净利润因子（低值排前），度量归母利润纯度。",
    category="quality",
    thesis="少数股东损益占比高意味着上市公司虽然是合并报表主体，但利润大量流向了子公司少数股东。"
    "该因子与minority_interest_ratio配合使用效果更佳。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_minority_gain_ratio(context: FactorContext):
    inc = context.load_financial(
        "income.parquet", value_cols=["minority_gain", "n_income"]
    )
    ratio = safe_divide(inc["minority_gain"].abs(), inc["n_income"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ═══════════════════════════════════════════════════════════════════════════
# D.  Cash Flow — previously unused fields
#    Source: cashflow.parquet (89 value columns, 74 unused)
#    Coverage: 44,539 rows, 1782/1782 stocks, 2020-03-31 to 2026-03-31
# ═══════════════════════════════════════════════════════════════════════════

# ── D.1  Cash Flow Structure ───────────────────────────────────────────────

@register_factor(
    name="operating_cf_to_total_inflow",
    description="经营活动现金流入/总现金流入因子，度量现金流来源的持续性。",
    category="quality",
    thesis="经营活动现金流占比高意味着企业现金流来自主营业务，可持续性强。"
    "投资和筹资现金流占比过高暗示主业造血能力不足。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_operating_cf_to_total_inflow(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_inf_fr_operate_a", "stot_inflows_inv_act", "stot_cash_in_fnc_act"],
    )
    total_inflow = (
        cf["c_inf_fr_operate_a"].fillna(0)
        + cf["stot_inflows_inv_act"].fillna(0)
        + cf["stot_cash_in_fnc_act"].fillna(0)
    )
    ratio = safe_divide(cf["c_inf_fr_operate_a"], total_inflow.abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="cash_from_sales_to_revenue",
    description="销售商品收到的现金/营收因子，度量收入现金含量。",
    category="quality",
    thesis="销售收现比高意味着营收伴随真实现金流入，低比率暗示应收账款膨胀或收入确认激进。"
    "该因子是识别收入造假的核心指标之一。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_cash_from_sales_to_revenue(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["c_fr_sale_sg"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )
    ratio = safe_divide(cf["c_fr_sale_sg"], inc["revenue"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="cash_paid_goods_to_cogs",
    description="购买商品支付的现金/营业成本因子（低值排前），度量采购现金效率。",
    category="quality",
    thesis="购买商品现金支出与营业成本的比率反映了存货和应付账款管理效果。"
    "低比率意味着公司有效利用供应商信用（占用上游资金）。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_cash_paid_goods_to_cogs(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["c_paid_goods_s"]
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["total_cogs"]
    )
    ratio = safe_divide(cf["c_paid_goods_s"], inc["total_cogs"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ── D.2  Financing Cash Flow Analysis ──────────────────────────────────────

@register_factor(
    name="borrowing_to_cash_inflow",
    description="借款收到的现金/筹资总流入因子（低值排前），度量融资结构。",
    category="quality",
    thesis="筹资活动主要依赖借款而非权益融资，表明公司偏好债务融资。"
    "高借款融资占比在紧缩周期中面临再融资风险。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_borrowing_to_cash_inflow(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_recp_borrow", "stot_cash_in_fnc_act"],
    )
    ratio = safe_divide(cf["c_recp_borrow"], cf["stot_cash_in_fnc_act"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


@register_factor(
    name="financing_cf_gap",
    description="筹资活动现金净流量/总资产因子，度量对外部融资的依赖度。",
    category="quality",
    thesis="持续大额筹资活动现金流入意味着企业造血能力不足以支撑经营和投资，"
    "需要外部输血维持。这在长期是不可持续的。",
    dependencies=("cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_financing_cf_gap(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet", value_cols=["n_cash_flows_fnc_act"]
    )
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["total_assets"]
    )
    ratio = safe_divide(cf["n_cash_flows_fnc_act"], bs["total_assets"].abs().replace(0, np.nan))
    return cross_sectional_rank(-ratio)


# ── D.3  Cash Position ─────────────────────────────────────────────────────

@register_factor(
    name="cash_change_ratio",
    description="现金净增加额/期初现金因子，度量现金流动态。",
    category="quality",
    thesis="现金及等价物的净变化率反映了企业的整体现金流健康状况。"
    "持续现金净流出的企业将面临流动性危机。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_cash_change_ratio(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["n_incr_cash_cash_equ", "c_cash_equ_beg_period"],
    )
    ratio = safe_divide(cf["n_incr_cash_cash_equ"], cf["c_cash_equ_beg_period"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


@register_factor(
    name="fx_impact_on_cash",
    description="汇率变动对现金的影响/期初现金因子，度量汇率风险敞口。",
    category="quality",
    thesis="汇率变动对现金有显著影响的企业具有较大的外汇风险敞口。"
    "在人民币波动加大的环境下，该因子可识别受益/受损企业。",
    dependencies=("cashflow.parquet", "calendar.parquet"),
)
def factor_fx_impact_on_cash(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["eff_fx_flu_cash", "c_cash_equ_beg_period"],
    )
    ratio = safe_divide(cf["eff_fx_flu_cash"], cf["c_cash_equ_beg_period"].abs().replace(0, np.nan))
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════
# E.  Main Fund Flow — Volume-based Ratios (previously unused volume columns)
#    Source: main_fund_flow.parquet
#    All 9 volume columns were unused: buy/sell_sm/md/lg/elg_vol + net_mf_vol
#    Coverage: 2,791,194 rows, 1782/1782 stocks, 2020-01-02 to 2026-07-01
# ═══════════════════════════════════════════════════════════════════════════

def _total_vol(ff):
    """Return total volume from main fund flow, zero replaced with NaN."""
    return (
        ff["buy_sm_vol"] + ff["sell_sm_vol"]
        + ff["buy_md_vol"] + ff["sell_md_vol"]
        + ff["buy_lg_vol"] + ff["sell_lg_vol"]
        + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    ).replace(0, np.nan)


@register_factor(
    name="mf_big_order_vol_ratio",
    description="大单+特大单成交量占比因子，基于成交量口径的机构行为度量。",
    category="fund_flow",
    thesis="成交量口径的大单占比与成交额口径互补：成交量剔除了高价股偏差，"
    "更能反映真实的交易笔数集中度。两个口径的信号差异本身也是信息。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_big_order_vol_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    big_vol = (
        ff["buy_lg_vol"] - ff["sell_lg_vol"]
        + ff["buy_elg_vol"] - ff["sell_elg_vol"]
    )
    ratio = big_vol / _total_vol(ff)
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_amount_vol_divergence",
    description="资金流量价背离因子，（大单净买入额占比 - 大单净买入量占比）截面排名。",
    category="fund_flow",
    thesis="成交额占比与成交量占比的差异反映了不同价位上的交易分布。"
    "大单净买入额占比 > 大单净买入量占比意味着大资金偏好高价买入（更强的买入意愿）。"
    "额量背离是聪明钱强度的重要信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_amount_vol_divergence(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    # Big order net amount ratio
    big_net_amt = (
        ff["buy_lg_amount"] - ff["sell_lg_amount"]
        + ff["buy_elg_amount"] - ff["sell_elg_amount"]
    )
    total_amt = (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)
    amt_ratio = big_net_amt / total_amt

    # Big order net volume ratio
    big_net_vol = (
        ff["buy_lg_vol"] - ff["sell_lg_vol"]
        + ff["buy_elg_vol"] - ff["sell_elg_vol"]
    )
    vol_ratio = big_net_vol / _total_vol(ff)

    divergence = amt_ratio - vol_ratio
    return cross_sectional_rank(divergence)


@register_factor(
    name="mf_net_vol_intensity",
    description="主力净流入量/总成交量因子，成交量口径的资金净流向强度。",
    category="fund_flow",
    thesis="净流入量占比为正意味着买方在数量上占优，配合净流入额占比使用"
    "可以识别量价配合vs量价背离的信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_vol_intensity(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    ratio = ff["net_mf_vol"] / _total_vol(ff)
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_order_size_entropy",
    description="订单规模分布的熵因子（高值排前），度量资金参与结构的多样性。",
    category="fund_flow",
    thesis="当各规模订单（小/中/大/特大）的参与比例较为均匀时，市场参与结构健康。"
    "当某一类订单占比畸高时（例如全靠大单拉动），行情的持续性存疑。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_order_size_entropy(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_vol = _total_vol(ff)

    # Volume share per order size category
    sm_share = (ff["buy_sm_vol"] + ff["sell_sm_vol"]) / total_vol
    md_share = (ff["buy_md_vol"] + ff["sell_md_vol"]) / total_vol
    lg_share = (ff["buy_lg_vol"] + ff["sell_lg_vol"]) / total_vol
    elg_share = (ff["buy_elg_vol"] + ff["sell_elg_vol"]) / total_vol

    # Entropy: -sum(p * ln(p)), higher = more diverse participation
    eps = 1e-10
    entropy = -(
        sm_share * np.log(sm_share + eps)
        + md_share * np.log(md_share + eps)
        + lg_share * np.log(lg_share + eps)
        + elg_share * np.log(elg_share + eps)
    )
    return cross_sectional_rank(entropy)


# ═══════════════════════════════════════════════════════════════════════════
# F.  Limit List — previously unused event fields
#    Source: limit_list.parquet
#    Unused: limit_amount, float_mv, turnover_ratio, fd_amount,
#            first_time, last_time, open_times, limit_times
#    Coverage: 67,066 rows (event data, sparse per stock), 1772/1782 stocks
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="limit_open_frequency_20d",
    description="20日涨停开板频率因子（低值排前），盘中开板次数越多说明封板越弱。",
    category="event",
    thesis="涨停被打开的次数是封板强度的直接度量。频发开板意味着多空分歧大、"
    "封板资金信心不足。开板次数少的涨停股后续溢价更高。",
    dependencies=("limit_list.parquet", "calendar.parquet"),
)
def factor_limit_open_frequency_20d(context: FactorContext):
    ll = context.load("limit_list.parquet")
    # Filter to limit-up (u) stocks only
    lu = ll[ll["up_stat"].notna()] if "up_stat" in ll.columns else ll
    # Sum open_times per stock over last 20 days
    open_count = lu.groupby(level="Code")["open_times"].transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    return cross_sectional_rank(-open_count)


@register_factor(
    name="limit_first_time_signal",
    description="平均首次涨停时间因子（早封板排前），越早封板越强。",
    category="event",
    thesis="首次涨停时间反映了封板资金的坚决程度。早盘秒板（如09:30-10:00封板）"
    "通常意味着主力资金做多意愿极强，而尾盘拉板则可能是弱势封板。",
    dependencies=("limit_list.parquet",),
)
def factor_limit_first_time_signal(context: FactorContext):
    ll = context.load("limit_list.parquet")
    # Parse first_time as HHMMSS integer -> minutes from midnight
    ft = pd.to_numeric(ll["first_time"], errors="coerce")
    minutes = (ft // 10000) * 60 + ((ft % 10000) // 100)  # 143627 -> 14*60+36 = 876
    # Rank: earlier = higher (negative)
    return cross_sectional_rank(-minutes)


@register_factor(
    name="limit_turnover_intensity",
    description="涨停日换手率因子（低值排前），封板期间低换手说明筹码锁定好。",
    category="event",
    thesis="涨停日换手率低意味着持筹者惜售，封板后的抛压小。"
    "高换手涨停（如>20%）可能是出货板，后续回调风险大。",
    dependencies=("limit_list.parquet",),
)
def factor_limit_turnover_intensity(context: FactorContext):
    ll = context.load("limit_list.parquet")
    to = ll["turnover_ratio"]
    return cross_sectional_rank(-to)


# ═══════════════════════════════════════════════════════════════════════════
# G.  Finance Panel — ps (Price-to-Sales unadjusted)
#     Source: finance.parquet
#     Only 1 unused field: ps (vs ps_ttm which is already used)
#     ps is the unadjusted trailing P/S using reported revenue
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="sp_raw",
    description="未调整市销率因子（低值排前），ps = 总市值/最近报告期营收。",
    category="valuation",
    thesis="ps字段提供了一种不同于ps_ttm的市销率计算口径（使用最近一期报告而非TTM），"
    "两者之间的差异本身包含了信息（TTM调整的方向和幅度）。",
    dependencies=("finance.parquet",),
)
def factor_sp_raw(context: FactorContext):
    finance = context.load("finance.parquet")
    return cross_sectional_rank(-finance["ps"])


# ═══════════════════════════════════════════════════════════════════════════
# H.  Composite Factors from Unused Fields
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="cash_flow_quality_composite",
    description="现金流质量综合因子（等权合成）：cash_from_sales_to_revenue + ocf_to_total_inflow + cash_change_ratio。",
    category="quality",
    thesis="将收现比、经营现金流占比、现金变化率三个维度合成综合现金流质量评分，"
    "比单一指标更稳健地度量和预测企业的财务健康度。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_cash_flow_quality_composite(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["c_fr_sale_sg", "c_inf_fr_operate_a",
                     "stot_inflows_inv_act", "stot_cash_in_fnc_act",
                     "n_incr_cash_cash_equ", "c_cash_equ_beg_period"],
    )
    inc = context.load_financial(
        "income.parquet", value_cols=["revenue"]
    )

    # Component 1: Cash from sales / Revenue
    c1 = safe_divide(cf["c_fr_sale_sg"], inc["revenue"].abs().replace(0, np.nan))

    # Component 2: Operating CF / Total Inflow
    total_inflow = (
        cf["c_inf_fr_operate_a"].fillna(0)
        + cf["stot_inflows_inv_act"].fillna(0)
        + cf["stot_cash_in_fnc_act"].fillna(0)
    )
    c2 = safe_divide(cf["c_inf_fr_operate_a"], total_inflow.abs().replace(0, np.nan))

    # Component 3: Cash change rate
    c3 = safe_divide(cf["n_incr_cash_cash_equ"], cf["c_cash_equ_beg_period"].abs().replace(0, np.nan))

    # Equal-weight composite
    composite = (
        cross_sectional_rank(c1)
        + cross_sectional_rank(c2)
        + cross_sectional_rank(c3)
    ) / 3.0
    return composite


@register_factor(
    name="earnings_structure_quality",
    description="盈利结构质量综合因子：operating_profit_to_ebt - non_operating_profit_ratio + deducted_profit_ratio。",
    category="quality",
    thesis="综合经营利润占比、非经营利润占比、扣非利润比率三个维度，"
    "全面度量盈利来源的可持续性。高得分意味着利润主要来自可重复的主营业务。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_earnings_structure_quality(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["opincome_of_ebt", "non_op_profit", "profit_to_gr", "dtprofit_to_profit"],
    )
    # C1: operating profit / EBT
    c1 = fin["opincome_of_ebt"]
    # C2: 1 - |non_op_profit|/profit (lower non-op = better)
    non_op_ratio = safe_divide(fin["non_op_profit"].abs(), fin["profit_to_gr"].abs().replace(0, np.nan))
    c2 = 1.0 - non_op_ratio.fillna(0)
    # C3: deducted profit ratio
    c3 = fin["dtprofit_to_profit"]

    composite = (
        cross_sectional_rank(c1)
        + cross_sectional_rank(c2)
        + cross_sectional_rank(c3)
    ) / 3.0
    return composite
