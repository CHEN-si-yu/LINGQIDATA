from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


# ── Profitability ───────────────────────────────────────────────────────

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
# q_ocf_to_or, q_eps) — upstream financial_indicator.parquet q_* columns
# are zero-filled for ~95% of records 2019-2022. Ranking zero values produces
# noise. Do not re-add unless vendor backfills historical q_* data.
# See: memory/vendor-data-quality.md


# ── Per-share metrics ────────────────────────────────────────────────────

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

# ── Gross profitability (Novy-Marx 2013) ───────────────────────────────────

@register_factor(
    name="gross_profitability",
    description="毛利率资产比因子，毛利/总资产截面排名（Novy-Marx质量因子）。",
    category="quality",
    thesis="Novy-Marx(2013)发现毛利率/总资产(gross profitability)的预测力与BP相当且与BP正交——高毛利资产比的企业将更多收入转化为利润。这是Fama-French五因子模型之外最重要的异象之一，与roe/roa互补：GP看收入转化效率，ROE看股东回报。",
    dependencies=("income.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_gross_profitability(context: FactorContext):
    inc = context.load_financial(
        "income.parquet",
        value_cols=["revenue", "total_cogs"],
    )
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["total_assets"],
    )
    gp = inc["revenue"] - inc["total_cogs"]
    gp_ratio = gp / bs["total_assets"].replace(0, np.nan)
    return cross_sectional_rank(gp_ratio)


# ── Asset growth (Cooper et al. 2008) ──────────────────────────────────────

@register_factor(
    name="asset_growth",
    description="总资产增长率因子，总资产同比增长率截面排名（负向：高增长排后=资产扩张异象）。",
    category="quality",
    thesis="Cooper et al.(2008)资产增长异象——资产快速扩张的企业未来收益显著低于保守扩张的企业。高资产增长通常伴随过度投资、管理层empire-building和随后的均值回归。这是学术界验证最充分的负向alpha因子之一。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_asset_growth(context: FactorContext):
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["total_assets"],
    )
    assets = bs["total_assets"]
    growth = assets.groupby(level="Code").transform(lambda s: s.pct_change(4))
    return cross_sectional_rank(-growth)

# ── Core profitability ──────────────────────────────────────────────────

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


# ── Sales cash ratio ────────────────────────────────────────────────────

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


# ── BPS rank ────────────────────────────────────────────────────────────

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


# ── NPTA ────────────────────────────────────────────────────────────────

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


# ── EBIT per share ──────────────────────────────────────────────────────

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


# ── Supplementary quality factors ──────────────────────────────────────────


@register_factor(
    name="gross_profit_to_assets",
    description="毛利/总资产因子 (Novy-Marx GP/A, 高毛利润排前)。",
    category="quality",
    thesis="毛利/总资产(GP/A)是Novy-Marx(2013)提出的最干净的质量因子，不受会计操纵影响",
    dependencies=("financial_indicator.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_gross_profit_to_assets(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["grossprofit_margin"])
    bs = context.load_financial("balancesheet.parquet", value_cols=["total_assets"])
    gp_margin = fin["grossprofit_margin"]
    ta = bs["total_assets"]
    common = gp_margin.index.intersection(ta.index)
    gpa = safe_divide(gp_margin.loc[common] * 100, ta.loc[common])
    return cross_sectional_rank(gpa)


@register_factor(
    name="debt_to_equity",
    description="负债权益比因子 (低负债排前, 负向)。",
    category="quality",
    thesis="D/E比是国际通用的杠杆指标，低D/E意味着财务稳健、抗风险能力强",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_debt_to_equity(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_liab", "total_hldr_eqy_exc_min_int"])
    de = safe_divide(bs["total_liab"], bs["total_hldr_eqy_exc_min_int"] + 1e-8)
    return cross_sectional_rank(-de)


@register_factor(
    name="working_capital_to_assets",
    description="营运资本/总资产因子 (高运营效率排前)。",
    category="quality",
    thesis="营运资本/总资产比率低意味着轻资产运营、资金效率高",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_working_capital_to_assets(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_cur_assets", "total_cur_liab", "total_assets"])
    wc = bs["total_cur_assets"] - bs["total_cur_liab"]
    ratio = safe_divide(wc, bs["total_assets"] + 1e-8)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="earnings_quality_composite",
    description="盈余质量综合因子 (OCF/利润+低应计+毛利率稳定三排名均值)。",
    category="quality",
    thesis="OCF覆盖利润、低应计、毛利率稳定三维度综合判断盈余质量",
    dependencies=("financial_indicator.parquet", "cashflow.parquet", "calendar.parquet"),
)
def factor_earnings_quality_composite(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["ocf_to_profit", "gross_margin", "netprofit_margin"])
    ocf_to_ni = fin["ocf_to_profit"]
    gm = fin["gross_margin"]

    ocf_rank = ocf_to_ni.groupby(level="Date").rank(pct=True)
    gm_std = gm.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    gm_mean = gm.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    gm_cv = safe_divide(gm_std, gm_mean.abs() + 1e-8)
    gm_stable_rank = (-gm_cv).groupby(level="Date").rank(pct=True)

    common = ocf_rank.index.intersection(gm_stable_rank.index)
    combo = (ocf_rank.loc[common] + gm_stable_rank.loc[common]) / 2.0
    return cross_sectional_rank(combo)


@register_factor(
    name="ebitda_rank",
    description="EBITDA截面排名。",
    category="quality",
    thesis="EBITDA是未经过折旧摊销和资本结构扭曲的经营利润，比净利润更适合跨行业比较企业的经营现金流产生能力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ebitda_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebitda"]
    )
    return cross_sectional_rank(fin["ebitda"])


@register_factor(
    name="ebit_rank",
    description="EBIT截面排名。",
    category="quality",
    thesis="EBIT剔除了资本结构和税率差异，是可比性最强的盈利指标。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ebit_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebit"]
    )
    return cross_sectional_rank(fin["ebit"])


@register_factor(
    name="bps_yoy",
    description="每股净资产同比增速截面排名。",
    category="quality",
    thesis="BPS增长代表每股内含价值的持续积累，高BPS增速意味着公司持续为股东创造账面价值。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_bps_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["bps_yoy"]
    )
    return cross_sectional_rank(fin["bps_yoy"])


@register_factor(
    name="cfps_rank",
    description="每股经营现金流截面排名。",
    category="quality",
    thesis="每股经营现金流反映真实的每股现金创造能力，与EPS互补验证盈利质量。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_cfps_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["cfps"]
    )
    return cross_sectional_rank(fin["cfps"])


@register_factor(
    name="cfps_yoy",
    description="每股经营现金流同比增速截面排名。",
    category="quality",
    thesis="每股现金流的增长趋势是盈利质量改善的最直接信号。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_cfps_yoy(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["cfps_yoy"]
    )
    return cross_sectional_rank(fin["cfps_yoy"])


@register_factor(
    name="fcfe_ps_rank",
    description="每股股权自由现金流截面排名。",
    category="quality",
    thesis="FCFE per share衡量股东可支配的每股现金——扣除资本开支和债务偿付后剩余的自由现金归属股东部分。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_fcfe_ps_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["fcfe_ps"]
    )
    return cross_sectional_rank(fin["fcfe_ps"])


@register_factor(
    name="fcff_ps_rank",
    description="每股公司自由现金流截面排名。",
    category="quality",
    thesis="FCFF per share衡量公司整体（含债权人和股东）可支配的每股现金，不受资本结构影响。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_fcff_ps_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["fcff_ps"]
    )
    return cross_sectional_rank(fin["fcff_ps"])


@register_factor(
    name="debt_to_equity_rank",
    description="负债权益比（财务杠杆截面排名，高杠杆排后）。",
    category="quality",
    thesis="D/E比是国际通用的杠杆指标，与debt_to_assets互补——一个看负债相对权益的比例，一个看负债相对总资产的比例。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_debt_to_equity_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["debt_to_eqt"]
    )
    return cross_sectional_rank(-fin["debt_to_eqt"])


@register_factor(
    name="ca_to_assets",
    description="流动资产/总资产（资产流动性截面排名）。",
    category="quality",
    thesis="流动资产占比高意味着资产变现能力强、流动性风险低，但过高可能意味着非流动资产投资不足。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ca_to_assets(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ca_to_assets"]
    )
    return cross_sectional_rank(fin["ca_to_assets"])


@register_factor(
    name="nca_to_assets",
    description="非流动资产/总资产（反向：重资产排后）。",
    category="quality",
    thesis="非流动资产占比高意味着重资产模式——在产能过剩或技术迭代时重资产面临更大减值风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_nca_to_assets(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["nca_to_assets"]
    )
    return cross_sectional_rank(-fin["nca_to_assets"])


@register_factor(
    name="fa_turn",
    description="固定资产周转率截面排名。",
    category="quality",
    thesis="固定资产周转率反映企业单位固定资产创造收入的能力，是资本密集型企业最关键的效率指标。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_fa_turn(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["fa_turn"]
    )
    return cross_sectional_rank(fin["fa_turn"])


@register_factor(
    name="ca_turn",
    description="流动资产周转率截面排名。",
    category="quality",
    thesis="流动资产周转率衡量短期资产的运用效率，高周转=资金被有效利用而非闲置。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ca_turn(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ca_turn"]
    )
    return cross_sectional_rank(fin["ca_turn"])


@register_factor(
    name="cf_short_debt_cover",
    description="经营现金流/流动负债（短期偿债现金流覆盖）截面排名。",
    category="quality",
    thesis="经营性现金流对短期债务的覆盖能力——高覆盖意味着企业可以用内生现金流偿还到期债务，无需借新还旧。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_cf_short_debt_cover(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["cash_to_liqdebt"]
    )
    return cross_sectional_rank(fin["cash_to_liqdebt"])


@register_factor(
    name="ocf_to_debt_rank",
    description="经营现金流/总负债截面排名。",
    category="quality",
    thesis="经营现金流覆盖总负债的能力是长期信用质量的核心——高覆盖表明企业即使停止融资也能依靠经营现金持续偿还债务。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ocf_to_debt_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_debt"]
    )
    return cross_sectional_rank(fin["ocf_to_debt"])


@register_factor(
    name="ocf_to_interest_debt",
    description="经营现金流/有息负债截面排名。",
    category="quality",
    thesis="经营现金流对有息负债（银行贷款+应付债券）的覆盖——更精确衡量企业对刚性债务的偿还能力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ocf_to_interest_debt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_interestdebt"]
    )
    return cross_sectional_rank(fin["ocf_to_interestdebt"])


@register_factor(
    name="ebitda_to_debt_rank",
    description="EBITDA/总负债截面排名。",
    category="quality",
    thesis="EBITDA相对总负债的比例综合反映了税前经营利润对债务的覆盖——债权人视角的偿债安全边际。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ebitda_to_debt_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebitda_to_debt"]
    )
    return cross_sectional_rank(fin["ebitda_to_debt"])


@register_factor(
    name="ocf_to_net_debt",
    description="经营现金流/净负债截面排名。",
    category="quality",
    thesis="经营现金流覆盖净负债（总负债-现金）的能力——剔除可直接清偿的现金后，衡量真正的偿债缺口。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ocf_to_net_debt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_netdebt"]
    )
    return cross_sectional_rank(fin["ocf_to_netdebt"])


@register_factor(
    name="working_capital_rank",
    description="营运资本截面排名。",
    category="quality",
    thesis="充足的营运资本是日常经营的润滑剂——正的营运资本意味着流动资产覆盖流动负债后的盈余，具备短期财务缓冲。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_working_capital_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["working_capital"]
    )
    return cross_sectional_rank(fin["working_capital"])


@register_factor(
    name="ocf_yoy_rank",
    description="经营现金流同比增速截面排名。",
    category="quality",
    thesis="经营现金流增长是盈利质量改善的核心信号——现金增长比利润增长更难操纵、更可持续。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ocf_yoy_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_yoy"]
    )
    return cross_sectional_rank(fin["ocf_yoy"])


@register_factor(
    name="roa2_yearly",
    description="年化ROA（含非经常损益版本）截面排名。",
    category="quality",
    thesis="年化ROA比单季度ROA更稳定，过滤季节性波动。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roa2_yearly(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roa2_yearly"]
    )
    return cross_sectional_rank(fin["roa2_yearly"])


@register_factor(
    name="retained_earnings_ps",
    description="每股留存收益截面排名。",
    category="quality",
    thesis="每股留存收益是企业在扣除分红后累计留存的每股金额——高留存意味着企业有大量内部积累可用于再投资或未来分红。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_retained_earnings_ps(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["retainedps"]
    )
    return cross_sectional_rank(fin["retainedps"])


@register_factor(
    name="surplus_reserve_ps",
    description="每股盈余公积截面排名。",
    category="quality",
    thesis="盈余公积是强制或自愿从净利润中提取的留存——高盈余公积代表企业财务政策审慎、合规性强。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_surplus_reserve_ps(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["surplus_rese_ps"]
    )
    return cross_sectional_rank(fin["surplus_rese_ps"])


@register_factor(
    name="undistributed_profit_ps",
    description="每股未分配利润截面排名。",
    category="quality",
    thesis="每股未分配利润是未来分红和股本转增的弹药库——高未分配利润意味着强大的分红潜力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_undistributed_profit_ps(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["undist_profit_ps"]
    )
    return cross_sectional_rank(fin["undist_profit_ps"])


@register_factor(
    name="total_revenue_ps",
    description="每股营业总收入截面排名。",
    category="quality",
    thesis="每股营收是公司业务规模的标准化度量——高每股营收代表公司具有规模效应。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_total_revenue_ps(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["total_revenue_ps"]
    )
    return cross_sectional_rank(fin["total_revenue_ps"])


@register_factor(
    name="revenue_ps_rank",
    description="每股营业收入截面排名。",
    category="quality",
    thesis="每股营业收入剔除了非主营收入后的核心业务规模——比total_revenue_ps更纯地反映主营业务规模。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_revenue_ps_rank(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["revenue_ps"]
    )
    return cross_sectional_rank(fin["revenue_ps"])


@register_factor(
    name="tang_asset_to_debt",
    description="有形资产/总负债（资产担保能力）截面排名。",
    category="quality",
    thesis="有形资产相对负债的比例是债权人最看重的担保能力指标——高比例意味着即使清算也有足够的资产覆盖债务。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_tang_asset_to_debt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["tangibleasset_to_debt"]
    )
    return cross_sectional_rank(fin["tangibleasset_to_debt"])


@register_factor(
    name="tang_asset_to_net_debt",
    description="有形资产/净负债截面排名。",
    category="quality",
    thesis="有形净资产覆盖净负债的能力——更严格的偿付能力测试，是信用质量的核心指标。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_tang_asset_to_net_debt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["tangibleasset_to_netdebt"]
    )
    return cross_sectional_rank(fin["tangibleasset_to_netdebt"])


@register_factor(
    name="tang_asset_to_int_debt",
    description="有形资产/有息负债截面排名。",
    category="quality",
    thesis="有形资产对有息负债的覆盖——银行信贷审批中最关注的偿债能力指标，高覆盖=低信用风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_tang_asset_to_int_debt(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["tangasset_to_intdebt"]
    )
    return cross_sectional_rank(fin["tangasset_to_intdebt"])


@register_factor(
    name="ebt_growth_momentum",
    description="利润总额增速动量（diff(4) of ebt_yoy）截面排名。",
    category="quality",
    thesis="利润增速的二阶变化（加速度）——增速本身在加快意味着基本面的改善趋势在增强，是成长股的alpha来源。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ebt_growth_momentum(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebt_yoy"]
    )
    growth = fin["ebt_yoy"]
    accel = growth.groupby(level="Code").transform(lambda s: s.diff(4))
    return cross_sectional_rank(accel)


@register_factor(
    name="or_growth_acceleration",
    description="营收增速加速度（diff(4) of or_yoy）截面排名。",
    category="quality",
    thesis="营收增速的边际变化比增速水平更具前瞻性——营收增速在加快意味着需求端的加速渗透，是营收增长的质量维度。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_or_growth_acceleration(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["or_yoy"]
    )
    growth = fin["or_yoy"]
    accel = growth.groupby(level="Code").transform(lambda s: s.diff(4))
    return cross_sectional_rank(accel)


# ════════════════════════════════════════════════════════════════════════════
# Phase 2a: Industry-Neutralized Quality Factors
# 行业中性化版本 — 剔除行业间差异，提取行业内相对质量信号
# ════════════════════════════════════════════════════════════════════════════

from .neutral import _industry_neutral_rank


@register_factor(
    name="roe_dt_neutral",
    description="行业中性化扣非ROE因子，行业内截面排名后统一排名。",
    category="quality",
    thesis="ROE在不同行业间的合理水平差异极大（金融杠杆vs科技轻资产），行业中性化后提取行业内相对盈利质量信号，更纯粹地反映公司层面的经营差异。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_roe_dt_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe_dt"]
    )
    neutral = _industry_neutral_rank(fin["roe_dt"], context)
    return cross_sectional_rank(neutral)



@register_factor(
    name="netprofit_margin_neutral",
    description="行业中性化净利率因子。",
    category="quality",
    thesis="净利率同样受行业模式影响——费用结构（销售费用vs研发费用占比）行业差异显著，中性化后反映公司层面费用管控效率。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_netprofit_margin_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["netprofit_margin"]
    )
    neutral = _industry_neutral_rank(fin["netprofit_margin"], context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="debt_to_assets_neutral",
    description="行业中性化资产负债率因子（取负，低杠杆排前）。",
    category="quality",
    thesis="杠杆率天然因行业而异——金融/地产/公用事业高杠杆是常态，行业中性化后才能识别同行业内真正过度负债的公司。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_debt_to_assets_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["debt_to_assets"]
    )
    neutral = _industry_neutral_rank(-fin["debt_to_assets"], context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="asset_turn_neutral",
    description="行业中性化资产周转率因子。",
    category="quality",
    thesis="资产周转效率在不同行业间不可比（零售高周转vs基建低周转），中性化后提取行业内相对运营效率信号。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_asset_turn_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["assets_turn"]
    )
    neutral = _industry_neutral_rank(fin["assets_turn"], context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="fcff_ps_neutral",
    description="行业中性化自由现金流/总资产因子。",
    category="quality",
    thesis="自由现金流生成能力因行业资本密集度差异显著，中性化后提取行业内相对现金流效率——高FCF每股意味着更强的内生增长与分红能力。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_fcff_ps_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["fcff_ps"]
    )
    neutral = _industry_neutral_rank(fin["fcff_ps"], context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="ocf_to_or_neutral",
    description="行业中性化经营现金流/营收因子。",
    category="quality",
    thesis="现金回收率(OCF/营收)反映盈利的真实性——在行业中性化后能识别出同行业中'纸面利润'vs'真金白银'的差异。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_ocf_to_or_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_or"]
    )
    neutral = _industry_neutral_rank(fin["ocf_to_or"], context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="roe_stability_8q_neutral",
    description="行业中性化ROE稳定性因子（取负CV，稳定排前）。",
    category="quality",
    thesis="ROE稳定性(8Q CV)在行业中性化后剔除行业周期性的干扰——识别同行业内真正盈利稳健的公司。",
    dependencies=("financial_indicator.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_roe_stability_8q_neutral(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["roe"])
    roe = fin["roe"]
    roe_std = roe.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    roe_mean = roe.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    cv = safe_divide(roe_std, roe_mean.abs() + 1e-8)
    neutral = _industry_neutral_rank(-cv, context)
    return cross_sectional_rank(neutral)


# ════════════════════════════════════════════════════════════════════════════
# Phase 2b: Time-Series Acceleration Quality Factors
# 质量指标的边际变化（加速度）——捕捉质量改善趋势而非静态水平
# ════════════════════════════════════════════════════════════════════════════

def _quarterly_slope(series: pd.Series, n_quarters: int = 4) -> pd.Series:
    """Compute linear trend slope over the last N quarterly observations."""
    def _slope(y):
        y_arr = np.asarray(y[~np.isnan(y)])
        if len(y_arr) < n_quarters:
            return np.nan
        y_arr = y_arr[-n_quarters:]
        x = np.arange(n_quarters, dtype=float) - (n_quarters - 1) / 2.0
        y_dm = y_arr - y_arr.mean()
        denom = (x * x).sum()
        if denom == 0:
            return np.nan
        return (x * y_dm).sum() / denom

    return series.groupby(level="Code").transform(
        lambda s: s.rolling(n_quarters, min_periods=n_quarters).apply(_slope, raw=True)
    )


@register_factor(
    name="roe_acceleration_4q",
    description="ROE四季度线性斜率因子——ROE改善趋势截面排名。",
    category="quality",
    thesis="ROE的趋势性改善(加速)比ROE的水平值更具前瞻性——改善中的公司通常处于竞争优势强化阶段，未来超额收益更显著。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roe_acceleration_4q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roe_dt"]
    )
    slope = _quarterly_slope(fin["roe_dt"], n_quarters=4)
    return cross_sectional_rank(slope)


@register_factor(
    name="gross_margin_acceleration_4q",
    description="毛利率四季度斜率因子——毛利率改善趋势截面排名。",
    category="quality",
    thesis="毛利率的持续改善意味着定价权提升或成本结构优化——毛利率趋势比毛利率水平更及时地反映竞争格局变化。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_gross_margin_acceleration_4q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["gross_margin"]
    )
    slope = _quarterly_slope(fin["gross_margin"], n_quarters=4)
    return cross_sectional_rank(slope)


@register_factor(
    name="asset_turn_acceleration_4q",
    description="资产周转率四季度斜率因子。",
    category="quality",
    thesis="资产周转效率的趋势性提升意味着公司正在更有效地利用资产——需求改善或产能优化正在释放运营杠杆。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_asset_turn_acceleration_4q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["assets_turn"]
    )
    slope = _quarterly_slope(fin["assets_turn"], n_quarters=4)
    return cross_sectional_rank(slope)


@register_factor(
    name="fcf_improvement_4q",
    description="自由现金流/总资产四季度斜率因子。",
    category="quality",
    thesis="自由现金流的趋势性改善意味着公司内生增长能力的增强——从'烧钱'到'造血'的转变是价值重估的核心催化剂。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_fcf_improvement_4q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["fcff_ps"]
    )
    slope = _quarterly_slope(fin["fcff_ps"], n_quarters=4)
    return cross_sectional_rank(slope)


@register_factor(
    name="debt_reduction_4q",
    description="资产负债率四季度下降斜率因子（斜率取负=去杠杆排前）。",
    category="quality",
    thesis="主动去杠杆（负债率趋势性下降）意味着公司财务风险边际改善——偿债压力减轻释放自由现金流，同时降低尾部风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_debt_reduction_4q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["debt_to_assets"]
    )
    slope = _quarterly_slope(fin["debt_to_assets"], n_quarters=4)
    return cross_sectional_rank(-slope)


@register_factor(
    name="eps_growth_acceleration_8q",
    description="EPS增速二阶加速度因子（diff(8) of eps_yoy），8季度diff截面排名。",
    category="quality",
    thesis="EPS增长率本身的加速（加速度）识别盈利拐点——增速由负转正加速或正增速进一步提速都是基本面的最强信号。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_eps_growth_acceleration_8q(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["basic_eps_yoy"]
    )
    growth = fin["basic_eps_yoy"]
    # diff(8) over quarterly data = 2-year change in YoY growth rate
    accel = growth.groupby(level="Code").transform(lambda s: s.diff(8))
    return cross_sectional_rank(accel)


# ════════════════════════════════════════════════════════════════════════════
# Phase 2c: Composite Quality Factors
# 多维度质量综合 — 单一指标噪音大，综合分数更稳定
# ════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="quality_composite_golden",
    description="质量综合因子，ROE+毛利率+资产周转率+FCF每股的等权综合排名。",
    category="quality",
    thesis="多维度质量指标的等权综合较单一维度更稳定——ROE(盈利水平)+毛利率(护城河)+周转率(运营效率)+FCF每股(现金真实性)四维度捕捉质量的不同侧面，彼此互补降低噪音。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_quality_composite_golden(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["roe_dt", "gross_margin", "assets_turn", "fcff_ps"],
    )
    rank_roe = cross_sectional_rank(fin["roe_dt"])
    rank_margin = cross_sectional_rank(fin["gross_margin"])
    rank_turn = cross_sectional_rank(fin["assets_turn"])
    rank_fcf = cross_sectional_rank(fin["fcff_ps"])
    composite = (rank_roe + rank_margin + rank_turn + rank_fcf) / 4.0
    return cross_sectional_rank(composite)


@register_factor(
    name="quality_earnings_composite",
    description="盈利质量综合因子，(应计质量+盈利平滑度+ROE稳定性)等权综合。",
    category="quality",
    thesis="盈利质量的三个核心维度：应计低(现金真实性)、平滑度低(无恶意平滑)、ROE稳定(竞争优势持续)——三者综合比单一维度更准确识别真实的盈利质量。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_quality_earnings_composite(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["roe"],
    )
    roe = fin["roe"]
    roe_std = roe.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    roe_mean = roe.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    roe_cv = safe_divide(roe_std, roe_mean.abs() + 1e-8)
    # Lower CV = more stable, higher rank
    rank_stability = cross_sectional_rank(-roe_cv)
    # For earnings smoothness proxy: use 8Q autocorrelation of quarterly changes
    roe_chg = roe.groupby(level="Code").diff(1)
    autocorr = roe_chg.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).apply(
            lambda x: np.asarray(x).var() / (np.asarray(x[1:]).var() + 1e-8) if len(x) >= 4 else np.nan, raw=True
        )
    )
    # High autocorr of changes = possible smoothing, lower rank
    rank_smooth = cross_sectional_rank(-autocorr)
    composite = (rank_stability + rank_smooth) / 2.0
    return cross_sectional_rank(composite)


@register_factor(
    name="quality_growth_composite",
    description="质量成长综合因子，(ROE加速度+毛利率加速度+FCF改善+营收增速)等权综合。",
    category="quality",
    thesis="质量+成长的综合捕捉：纯质量因子忽视成长性，纯成长因子忽视质量——四维度加速度综合识别'高质量成长'公司（盈利改善+护城河增强+现金流优化+规模扩张同时发生）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_quality_growth_composite(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["roe_dt", "gross_margin", "fcff_ps", "or_yoy"],
    )
    roe_slope = _quarterly_slope(fin["roe_dt"], n_quarters=4)
    margin_slope = _quarterly_slope(fin["gross_margin"], n_quarters=4)
    fcf_slope = _quarterly_slope(fin["fcff_ps"], n_quarters=4)
    rank_roe_acc = cross_sectional_rank(roe_slope)
    rank_margin_acc = cross_sectional_rank(margin_slope)
    rank_fcf_acc = cross_sectional_rank(fcf_slope)
    rank_rev_growth = cross_sectional_rank(fin["or_yoy"])
    composite = (rank_roe_acc + rank_margin_acc + rank_fcf_acc + rank_rev_growth) / 4.0
    return cross_sectional_rank(composite)
