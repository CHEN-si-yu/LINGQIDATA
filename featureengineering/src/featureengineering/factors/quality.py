from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── Profitability ───────────────────────────────────────────────────────

@register_factor(
    name="roe",
    description="ROE因子，净资产收益率（日频前向填充版）截面排名。",
    category="quality",
    thesis="高ROE是巴菲特式质量投资的核心指标，长期稳定高ROE的公司享有估值溢价。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roe(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe"]
    )
    return cross_sectional_rank(fin["roe"])


@register_factor(
    name="roa",
    description="ROA因子，总资产收益率截面排名。",
    category="quality",
    thesis="ROA衡量资产使用效率，不受资本结构影响，比ROE更适合跨行业比较。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roa(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roa"]
    )
    return cross_sectional_rank(fin["roa"])


@register_factor(
    name="roic",
    description="ROIC因子，投入资本回报率截面排名。",
    category="quality",
    thesis="ROIC衡量企业经营资本回报，高ROIC代表护城河与竞争优势。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roic(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roic"]
    )
    return cross_sectional_rank(fin["roic"])


@register_factor(
    name="roe_waa",
    description="加权平均ROE因子截面排名。",
    category="quality",
    thesis="加权平均ROE考虑了权益变动时间加权，比简单ROE更精确。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roe_waa(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe_waa"]
    )
    return cross_sectional_rank(fin["roe_waa"])


@register_factor(
    name="roe_dt",
    description="扣非ROE因子截面排名。",
    category="quality",
    thesis="扣非ROE剔除非经常性损益，更真实反映主营业务的盈利能力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roe_dt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe_dt"]
    )
    return cross_sectional_rank(fin["roe_dt"])


# ── Margins ─────────────────────────────────────────────────────────────

@register_factor(
    name="gross_margin",
    description="毛利率因子截面排名。",
    category="quality",
    thesis="高毛利率代表定价权与竞争壁垒，是护城河的核心量化指标。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_gross_margin(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["gross_margin"]
    )
    return cross_sectional_rank(fin["gross_margin"])


@register_factor(
    name="netprofit_margin",
    description="净利率因子截面排名。",
    category="quality",
    thesis="净利率综合考虑毛利、费用与税收，反映企业全链条盈利能力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_netprofit_margin(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["netprofit_margin"]
    )
    return cross_sectional_rank(fin["netprofit_margin"])


# ── Leverage / Solvency ─────────────────────────────────────────────────

@register_factor(
    name="debt_to_assets",
    description="资产负债率因子截面排名（低负债排前）。",
    category="quality",
    thesis="低杠杆企业在经济下行周期中具有更强的抗风险能力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_debt_to_assets(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["debt_to_assets"]
    )
    return cross_sectional_rank(-fin["debt_to_assets"])


@register_factor(
    name="current_ratio",
    description="流动比率因子截面排名。",
    category="quality",
    thesis="高流动比率代表短期偿债能力强，财务安全性高。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_current_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["current_ratio"]
    )
    return cross_sectional_rank(fin["current_ratio"])


@register_factor(
    name="quick_ratio",
    description="速动比率因子截面排名。",
    category="quality",
    thesis="速动比率剔除存货，比流动比率更严苛地衡量短期偿债能力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_quick_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["quick_ratio"]
    )
    return cross_sectional_rank(fin["quick_ratio"])


@register_factor(
    name="cash_ratio",
    description="现金比率因子截面排名。",
    category="quality",
    thesis="现金比率是流动性最严格的定义，反映极端情况下的即时偿债能力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_cash_ratio(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["cash_ratio"]
    )
    return cross_sectional_rank(fin["cash_ratio"])


# ── Cash flow quality ──────────────────────────────────────────────────

@register_factor(
    name="ocf_to_profit",
    description="经营现金流/净利润比率因子截面排名。",
    category="quality",
    thesis="现金流利润匹配度是盈利质量的核心指标，高比率代表利润含金量高。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ocf_to_profit(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_profit"]
    )
    return cross_sectional_rank(fin["ocf_to_profit"])


@register_factor(
    name="salescash_to_or",
    description="销售收现/营业收入因子截面排名。",
    category="quality",
    thesis="销售收现比直接衡量营收的现金质量，高比率代表真金白银的确认收入。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_salescash_to_or(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["salescash_to_or"]
    )
    return cross_sectional_rank(fin["salescash_to_or"])


# ── Efficiency ──────────────────────────────────────────────────────────

@register_factor(
    name="assets_turn",
    description="总资产周转率因子截面排名。",
    category="quality",
    thesis="高周转率代表运营效率高、资产利用充分，是轻资产商业模式的核心特征。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_assets_turn(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["assets_turn"]
    )
    return cross_sectional_rank(fin["assets_turn"])


# ── Growth ──────────────────────────────────────────────────────────────

@register_factor(
    name="or_yoy",
    description="营业收入同比增速因子截面排名。",
    category="quality",
    thesis="营收增长是成长性的基础维度，持续高增长的股票享有成长溢价。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_or_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["or_yoy"]
    )
    return cross_sectional_rank(fin["or_yoy"])


@register_factor(
    name="netprofit_yoy",
    description="净利润同比增速因子截面排名。",
    category="quality",
    thesis="利润增速比营收增速更直接反映股东回报的成长性。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_netprofit_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["netprofit_yoy"]
    )
    return cross_sectional_rank(fin["netprofit_yoy"])


@register_factor(
    name="equity_yoy",
    description="净资产（权益）同比增速因子截面排名。",
    category="quality",
    thesis="净资产增长反映企业内生积累或融资能力，是长期成长的基础。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_equity_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["equity_yoy"]
    )
    return cross_sectional_rank(fin["equity_yoy"])


@register_factor(
    name="assets_yoy",
    description="总资产同比增速因子截面排名。",
    category="quality",
    thesis="总资产增速反映企业扩张节奏，过快或过慢都可能包含信息。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_assets_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["assets_yoy"]
    )
    return cross_sectional_rank(fin["assets_yoy"])


# ── Quality composite ───────────────────────────────────────────────────

@register_factor(
    name="quality_composite",
    description="质量综合因子，ROE+ROA+毛利率+现金流质量四个维度的等权平均截面排名。",
    category="quality",
    thesis="多维度质量因子综合可提升对企业真实质量的识别能力，降低单一指标的误判风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_quality_composite(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["roe", "roa", "gross_margin", "ocf_to_profit"],
    )
    with np.errstate(invalid="ignore"):
        rank_roe = fin["roe"].groupby(level="Date").rank(pct=True)
        rank_roa = fin["roa"].groupby(level="Date").rank(pct=True)
        rank_gm = fin["gross_margin"].groupby(level="Date").rank(pct=True)
        rank_ocf = fin["ocf_to_profit"].groupby(level="Date").rank(pct=True)
    composite = (rank_roe + rank_roa + rank_gm + rank_ocf) / 4.0
    return composite.rename("quality_composite")


# ── Solvency depth: financial_indicator 扩展 ──────────────────────────────

@register_factor(
    name="ebit_to_interest",
    description="利息保障倍数因子，EBIT/利息支出截面排名。",
    category="quality",
    thesis="利息保障倍数衡量企业利润覆盖利息支出的能力，高倍数代表低财务风险，是学术验证最强的信用质量指标之一，与现有资产负债率互补（前者看利润表覆盖，后者看资产负债表杠杆）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ebit_to_interest(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebit_to_interest"]
    )
    return cross_sectional_rank(fin["ebit_to_interest"])


# ── Operational efficiency: financial_indicator 扩展 ─────────────────────

@register_factor(
    name="inv_turn",
    description="存货周转率因子截面排名。",
    category="quality",
    thesis="存货周转率反映企业销售效率和库存管理能力，高周转代表产品畅销、资金占用少，低周转可能意味着产品滞销或存货积压风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_inv_turn(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["inv_turn"]
    )
    return cross_sectional_rank(fin["inv_turn"])


@register_factor(
    name="ar_turn",
    description="应收账款周转率因子截面排名。",
    category="quality",
    thesis="应收账款周转率反映企业回款效率和对下游的议价能力，高周转代表回款快、坏账风险低、利润含金量高。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ar_turn(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ar_turn"]
    )
    return cross_sectional_rank(fin["ar_turn"])


# ── Single-quarter financial indicator 扩展 ──────────────────────────────

@register_factor(
    name="q_roe",
    description="单季度ROE因子，季度净资产收益率截面排名。",
    category="quality",
    thesis="单季度ROE比TTM ROE更敏感，能更早捕捉企业盈利能力的边际变化。季度数据避免了TTM的平滑效应，对盈利拐点的识别更及时。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_roe(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_roe"]
    )
    return cross_sectional_rank(fin["q_roe"])


@register_factor(
    name="q_gsprofit_margin",
    description="单季度毛利率因子，季度毛利率截面排名。",
    category="quality",
    thesis="单季度毛利率变化是定价权和成本控制能力的及时信号，毛利率的季度波动对竞争格局变化和原材料价格冲击的反映比TTM版本更灵敏。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_gsprofit_margin(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_gsprofit_margin"]
    )
    return cross_sectional_rank(fin["q_gsprofit_margin"])


@register_factor(
    name="q_netprofit_margin",
    description="单季度净利率因子，季度净利率截面排名。",
    category="quality",
    thesis="净利率的季度变化反映费用控制和经营效率的短期波动，单季度数据能更早暴露利润率拐点。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_netprofit_margin(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_netprofit_margin"]
    )
    return cross_sectional_rank(fin["q_netprofit_margin"])


@register_factor(
    name="q_sales_yoy",
    description="单季度营收同比增速因子截面排名。",
    category="quality",
    thesis="单季度营收同比增速消除了季节性因素，同时比TTM同比更及时反映增长趋势的变化。高增速意味着产品需求旺盛、市场份额提升。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_sales_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_sales_yoy"]
    )
    return cross_sectional_rank(fin["q_sales_yoy"])


@register_factor(
    name="q_netprofit_yoy",
    description="单季度净利润同比增速因子截面排名。",
    category="quality",
    thesis="净利润单季度同比增速是盈利增长最直接的度量，剔除了季节性但保留了季度敏感度，对盈利拐点的信号比TTM增速领先1-2个季度。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_netprofit_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_netprofit_yoy"]
    )
    return cross_sectional_rank(fin["q_netprofit_yoy"])


@register_factor(
    name="q_profit_yoy",
    description="单季度利润总额同比增速因子截面排名。",
    category="quality",
    thesis="利润总额同比增速比净利润更少受非经常性损益干扰，反映主营业务的真实增长动能。单季度版本对经营拐点的敏感度优于TTM版本。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_profit_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_profit_yoy"]
    )
    return cross_sectional_rank(fin["q_profit_yoy"])


@register_factor(
    name="q_ocf_to_sales",
    description="单季度经营现金流/营收因子截面排名。",
    category="quality",
    thesis="经营现金流/营收比反映了单季度收入转化为现金的能力，高比率意味着收入含金量高、应收账款可控。季度版本能更及时暴露现金流质量问题。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_ocf_to_sales(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_ocf_to_sales"]
    )
    return cross_sectional_rank(fin["q_ocf_to_sales"])


@register_factor(
    name="q_eps",
    description="单季度每股收益因子截面排名。",
    category="quality",
    thesis="单季度EPS是每股层面盈利能力的最基本度量，剔除股本变动影响后的季度EPS反映了每股价值创造的速度。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_q_eps(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["q_eps"]
    )
    return cross_sectional_rank(fin["q_eps"])
