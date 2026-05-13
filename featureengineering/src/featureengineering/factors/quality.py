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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
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
    fin = context.repo.load_financial_panel(
        "financial_indicator.parquet",
        value_cols=["roe", "roa", "gross_margin", "ocf_to_profit"],
    )
    rank_roe = fin["roe"].groupby(level="Date").rank(pct=True)
    rank_roa = fin["roa"].groupby(level="Date").rank(pct=True)
    rank_gm = fin["gross_margin"].groupby(level="Date").rank(pct=True)
    rank_ocf = fin["ocf_to_profit"].groupby(level="Date").rank(pct=True)
    composite = (rank_roe + rank_roa + rank_gm + rank_ocf) / 4.0
    return composite.rename("quality_composite")
