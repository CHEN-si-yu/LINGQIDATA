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


# ── REMOVED: All q_* single-quarter factors (q_roe, q_gsprofit_margin,
# q_netprofit_margin, q_sales_yoy, q_netprofit_yoy, q_profit_yoy,
# q_ocf_to_sales, q_eps) — upstream financial_indicator.parquet q_* columns
# are zero-filled for ~95% of records 2019-2022. Ranking zero values produces
# noise. Do not re-add unless vendor backfills historical q_* data.
# See: memory/vendor-data-quality.md


# ── Per-share metrics ────────────────────────────────────────────────────

@register_factor(
    name="bps_rank",
    description="每股净资产(BPS)因子截面排名。",
    category="quality",
    thesis="每股净资产是股票内在价值的账面锚定，高BPS代表更强的资产安全垫，在价值投资中与BP互补——一个看每股资产，一个看市价相对资产。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_bps_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["bps"]
    )
    return cross_sectional_rank(fin["bps"])


# ── Cash flow solvency ───────────────────────────────────────────────────

@register_factor(
    name="ocf_coverage",
    description="经营现金流短期债务覆盖因子，OCF/短期债务截面排名。",
    category="quality",
    thesis="经营现金流覆盖短期债务的能力是企业短期财务安全的核心指标，高覆盖率意味企业可以依靠内生现金流偿还到期债务，无需再融资，是信用质量的及时度量。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ocf_coverage(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_shortdebt"]
    )
    return cross_sectional_rank(fin["ocf_to_shortdebt"])


# ── Earnings stability ───────────────────────────────────────────────────

# REMOVED: earnings_yield_stability — upstream financial_indicator.parquet q_eps
# is zero-filled for ~95% of records before 2023 (vendor data quality issue).
# CV = std/mean = 0/0 → NaN for virtually all stocks 2019-2023.
# Do not re-add unless vendor backfills historical q_eps data.


# ── REMOVED: q_sales_qoq, q_profit_qoq, q_gr_yoy — q_* fields are
# zero-filled 2019-2022. Do not re-add without vendor data backfill.


# ── ROE momentum ─────────────────────────────────────────────────────────

@register_factor(
    name="roe_momentum_4q",
    description="ROE季度动量因子，roe - roe.shift(4)截面排名，即ROE同比变化量。",
    category="quality",
    thesis="ROE的边际变化比静态ROE水平更具预测力——盈利加速改善的公司往往处于成长加速期，而盈利恶化的公司即使静态ROE不低也可能面临基本面下行。与roe因子互补：一个看水平一个看变化。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roe_momentum_4q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe"]
    )
    roe = fin["roe"]
    delta = roe.groupby(level="Code").transform(lambda s: s.diff(4))
    return cross_sectional_rank(delta)


# ── DuPont component ─────────────────────────────────────────────────────

@register_factor(
    name="assets_to_eqt",
    description="权益乘数（总资产/净资产）截面排名（低杠杆=低乘数排前）。",
    category="quality",
    thesis="权益乘数是杜邦分析中的杠杆维度，低权益乘数意味着企业经营更多依赖自有资金而非债务，财务风险更低。在信用收缩或利率上行期，低杠杆企业的抗风险能力显著更强。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_assets_to_eqt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["assets_to_eqt"]
    )
    return cross_sectional_rank(-fin["assets_to_eqt"])


# ── Pre-tax earnings quality ─────────────────────────────────────────────

@register_factor(
    name="tax_to_ebt",
    description="实际税率因子，所得税/利润总额截面排名（高税率排后）。",
    category="quality",
    thesis="实际税率高意味着企业享受的税收优惠少，在同等税前利润下留给股东的净利润更少。但需注意过低的实际税率可能来自非经常性损益或会计处理，需结合盈利质量综合判断。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_tax_to_ebt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["tax_to_ebt"]
    )
    return cross_sectional_rank(-fin["tax_to_ebt"])


# ── Asset quality ────────────────────────────────────────────────────────

@register_factor(
    name="npta",
    description="非不良资产/总资产因子，资产质量截面排名。",
    category="quality",
    thesis="NPTA衡量经不良调整后的资产质量，高值代表资产'含金量'高、不良风险低。与不良贷款率不同，NPTA覆盖了更广泛的风险资产类别。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_npta(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["npta"]
    )
    return cross_sectional_rank(fin["npta"])


# ── REMOVED: q_gr_qoq — q_* fields are zero-filled 2019-2022.

@register_factor(
    name="op_yoy",
    description="营业利润同比增速因子截面排名。",
    category="quality",
    thesis="营业利润（operating profit）剔除了投资收益和营业外收支的扰动，其同比增速比净利润增速更纯，反映主营业务盈利的真实增长趋势。高营业利润增速=主业强劲、可持续性强。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_op_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["op_yoy"]
    )
    return cross_sectional_rank(fin["op_yoy"])


@register_factor(
    name="ebit_ps_rank",
    description="每股EBIT因子，息税前利润/总股本截面排名。",
    category="quality",
    thesis="每股EBIT剔除了利息和所得税的结构性差异，比EPS更适合跨资本结构比较。高EBITPS代表企业核心经营业务具有更强的每股盈利能力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ebit_ps_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebit_ps"]
    )
    return cross_sectional_rank(fin["ebit_ps"])


# ── Long-term growth trend ───────────────────────────────────────────────

@register_factor(
    name="tr_yoy",
    description="营业总收入同比增速因子截面排名。",
    category="quality",
    thesis="营业总收入是公司最上线的收入口径（含主营业务+其他业务），其同比增速反映公司全业务线的综合增长能力。高总营收增速意味着公司在多条战线上持续扩张。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_tr_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["tr_yoy"]
    )
    return cross_sectional_rank(fin["tr_yoy"])


@register_factor(
    name="ebt_yoy",
    description="利润总额同比增速因子截面排名。",
    category="quality",
    thesis="利润总额同比增速是税前盈利的综合增长度量，包含了营业利润和非经常性损益的完整效应，是盈利增长最全面的指标。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ebt_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebt_yoy"]
    )
    return cross_sectional_rank(fin["ebt_yoy"])
