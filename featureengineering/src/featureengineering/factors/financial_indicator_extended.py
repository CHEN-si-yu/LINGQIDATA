"""
Extended financial indicator factors — Class 1.

Systematic factorisation of remaining unused financial_indicator.parquet fields.
The financial_indicator table has 167 columns of pre-computed financial ratios
and per-share metrics.  After unused_fields_factors.py (~29 factors) and other
existing modules, many high-quality fields remain untapped.

Sections
--------
A. Per-Share Metrics          — bps, ocfps, cfps, revenue_ps, ebit_ps, retainedps
B. Turnover / Efficiency      — assets_turn, ca_turn, fa_turn, inv_turn, ar_turn
C. Profitability Quality      — grossprofit_margin, ebitda margin, SGA ratio, etc.
D. Solvency / Leverage        — current_ratio, quick_ratio, cash_ratio, debt_to_eqt, etc.
E. Growth / YoY               — or_yoy, netprofit_yoy, op_yoy, assets_yoy, etc.

Data source: ``financial_indicator.parquet`` (167 columns, daily-panel via forward-fill)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    safe_divide,
)


# ═══════════════════════════════════════════════════════════════════════════
# A.  Per-Share Metrics — direct value anchors
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_bps_factor",
    description="每股净资产因子（低估值排前），bps=归属母公司股东权益/总股本。",
    category="valuation",
    thesis="每股净资产（Book Value Per Share）是最基础的价值锚点。高BPS意味着"
    "每股含有的净资产多，提供了更厚的安全边际。与股价结合即为P/B。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_ext_bps_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["bps"])
    finance = context.load("finance.parquet")
    # BPS / Price = Book-to-Price (higher = cheaper)
    close = finance["close"]
    bp_ratio = safe_divide(fin["bps"], close)
    return cross_sectional_rank(bp_ratio)


@register_factor(
    name="ext_ocfps_factor",
    description="每股经营现金流因子，ocfps/收盘价截面排名（高值排前）。",
    category="quality",
    thesis="每股经营现金流（Operating CF Per Share）衡量每股实际产生的现金。"
    "OCFPS高的公司有更强的内生增长能力和分红潜力，且不易受会计操纵。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_ext_ocfps_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["ocfps"])
    finance = context.load("finance.parquet")
    ocf_yield = safe_divide(fin["ocfps"], finance["close"])
    return cross_sectional_rank(ocf_yield)


@register_factor(
    name="ext_cfps_factor",
    description="每股现金流因子，cfps/收盘价截面排名（高值排前）。",
    category="quality",
    thesis="cfps（Cash Flow Per Share）包含经营+投资+筹资的净现金流。"
    "与ocfps互补，cfps反映了企业整体的现金生成能力，包括投融资活动。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_ext_cfps_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["cfps"])
    finance = context.load("finance.parquet")
    cf_yield = safe_divide(fin["cfps"], finance["close"])
    return cross_sectional_rank(cf_yield)


@register_factor(
    name="ext_revenue_ps_factor",
    description="每股营收因子，revenue_ps/收盘价截面排名（高值排前）。",
    category="valuation",
    thesis="每股营收（Revenue Per Share）衡量每股对应的业务规模。"
    "高营收PS意味着每元投资获得更多的业务量，适合评估尚未盈利的成长型公司。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_ext_revenue_ps_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["revenue_ps"])
    finance = context.load("finance.parquet")
    rev_yield = safe_divide(fin["revenue_ps"], finance["close"])
    return cross_sectional_rank(rev_yield)


@register_factor(
    name="ext_ebit_ps_factor",
    description="每股EBIT因子，ebit_ps/收盘价截面排名（高值排前）。",
    category="valuation",
    thesis="每股息税前利润（EBIT Per Share）剔除了资本结构和税率差异，"
    "更适合跨行业比较企业的经营盈利能力。EBIT/Price是EV/EBIT的简化版。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_ext_ebit_ps_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["ebit_ps"])
    finance = context.load("finance.parquet")
    ebit_yield = safe_divide(fin["ebit_ps"], finance["close"])
    return cross_sectional_rank(ebit_yield)


@register_factor(
    name="ext_retained_eps_factor",
    description="每股留存收益因子，retainedps/收盘价截面排名（高值排前）。",
    category="quality",
    thesis="每股留存收益反映了企业历史上累计的未分配利润。"
    "高留存收益意味着企业长期盈利且再投资能力强，是F-Score的核心组件之一。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_ext_retained_eps_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["retainedps"])
    finance = context.load("finance.parquet")
    re_yield = safe_divide(fin["retainedps"], finance["close"])
    return cross_sectional_rank(re_yield)


# ═══════════════════════════════════════════════════════════════════════════
# B.  Turnover / Efficiency Ratios
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_assets_turnover_factor",
    description="总资产周转率因子，assets_turn=营收/总资产截面排名（高值排前）。",
    category="quality",
    thesis="总资产周转率衡量企业利用全部资产产生收入的效率。"
    "高周转率意味着轻资产、高效率的商业模式——每元资产产生更多收入。"
    "该因子在制造业和零售业中区分度最高。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_assets_turnover_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["assets_turn"])
    return cross_sectional_rank(fin["assets_turn"])


@register_factor(
    name="ext_ca_turnover_factor",
    description="流动资产周转率因子，ca_turn截面排名（高值排前）。",
    category="quality",
    thesis="流动资产周转率聚焦于短期资产的运营效率。高CA周转率说明企业"
    "用较少的流动资产（现金+存货+应收）支撑了较大的营收规模，资金使用效率高。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_ca_turnover_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["ca_turn"])
    return cross_sectional_rank(fin["ca_turn"])


@register_factor(
    name="ext_fa_turnover_factor",
    description="固定资产周转率因子，fa_turn截面排名（高值排前）。",
    category="quality",
    thesis="固定资产周转率衡量长期资产的生产效率。高FA周转率代表"
    "每元固定资产产生更多收入，是资本密集型行业（制造业、能源）的核心效率指标。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_fa_turnover_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["fa_turn"])
    return cross_sectional_rank(fin["fa_turn"])


@register_factor(
    name="ext_inventory_turnover_factor",
    description="存货周转率因子，inv_turn截面排名（高值排前）。",
    category="quality",
    thesis="存货周转率反映存货管理效率。高周转率意味着存货变现快、"
    "资金占用少、存货减值风险低。异常低的存货周转率可能预示存货积压或滞销。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_inventory_turnover_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["inv_turn"])
    return cross_sectional_rank(fin["inv_turn"])


@register_factor(
    name="ext_ar_turnover_factor",
    description="应收账款周转率因子，ar_turn截面排名（高值排前）。",
    category="quality",
    thesis="应收账款周转率高意味着回款速度快、坏账风险低。"
    "AR周转率异常低可能是虚增收入（应收账款挂账）的预警信号。"
    "该因子对识别财务造假有特殊价值。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_ar_turnover_factor(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["ar_turn"])
    return cross_sectional_rank(fin["ar_turn"])


# ═══════════════════════════════════════════════════════════════════════════
# C.  Profitability Quality
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_gross_margin_factor",
    description="毛利率因子，grossprofit_margin截面排名（高值排前）。",
    category="quality",
    thesis="毛利率（grossprofit_margin）反映产品或服务的基本盈利能力，"
    "是商业模式竞争力的最直接体现。高毛利率意味着强定价权、品牌溢价或技术壁垒。"
    "使用财务指标口径的grossprofit_margin作为现有毛利率因子的补充。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_gross_margin_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["grossprofit_margin"]
    )
    return cross_sectional_rank(fin["grossprofit_margin"])


@register_factor(
    name="ext_ebitda_margin_factor",
    description="EBITDA利润率因子，ebitda/total_revenue_ps截面排名（高值排前）。",
    category="quality",
    thesis="EBITDA Margin剔除了折旧摊销和资本结构的影响，最适合跨行业比较。"
    "高EBITDA Margin代表强运营效率和现金生成能力。常用于并购估值和企业价值比较。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_ebitda_margin_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebitda", "total_revenue_ps"]
    )
    margin = safe_divide(fin["ebitda"], fin["total_revenue_ps"].abs().replace(0, np.nan))
    return cross_sectional_rank(margin)


@register_factor(
    name="ext_sga_to_revenue",
    description="销售管理费用率因子（低值排前），(saleexp+adminexp)/营收截面排名。",
    category="quality",
    thesis="销售费用+管理费用占营收比（SG&A Ratio）衡量期间费用的管控效率。"
    "费用率持续下降是管理改善的信号，费用率突然跳升可能预示渠道扩张或管理失控。"
    "相比单独看销售费用率或管理费用率，SG&A合计更加综合。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_sga_to_revenue(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["saleexp_to_gr", "adminexp_of_gr"],
    )
    sga = fin["saleexp_to_gr"].fillna(0) + fin["adminexp_of_gr"].fillna(0)
    return cross_sectional_rank(-sga)


@register_factor(
    name="ext_financial_expense_to_revenue_fi",
    description="财务费用率因子(fin_indicator口径)（低值排前），finaexp_of_gr截面排名。",
    category="quality",
    thesis="财务费用率反映企业的融资成本和债务负担。高财务费用率意味着"
    "利息支出侵蚀了大量利润，盈利质量打折扣。该指标在加息周期中尤为重要。"
    "使用financial_indicator口径与income口径互补。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_financial_expense_to_revenue_fi(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["finaexp_of_gr"]
    )
    return cross_sectional_rank(-fin["finaexp_of_gr"])


@register_factor(
    name="ext_profit_to_op_factor",
    description="利润总额/营业利润比率因子（高值排前），度量非经营项目的贡献。",
    category="quality",
    thesis="profit_to_op = 利润总额/营业利润，当比率显著>1时说明非经营收入贡献大"
    "（可能是投资收益、政府补贴或资产处置），当比率<1时说明营业外支出拖累利润。"
    "接近1的值意味着利润来源纯粹。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_profit_to_op_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["profit_to_op"]
    )
    # Values near 1 = purer earnings; high absolute deviation is the signal
    deviation = (fin["profit_to_op"] - 1.0).abs()
    return cross_sectional_rank(-deviation)


# ═══════════════════════════════════════════════════════════════════════════
# D.  Solvency / Leverage
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_current_ratio_factor",
    description="流动比率因子（高值排前），current_ratio=流动资产/流动负债。",
    category="quality",
    thesis="流动比率是短期偿债能力的经典指标。高流动比率意味着企业有足够"
    "的流动性应对短期到期债务。但过高的流动比率也可能意味着资产利用效率不足。"
    "通常>2被认为是健康的，<1则存在流动性风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_current_ratio_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["current_ratio"]
    )
    return cross_sectional_rank(fin["current_ratio"])


@register_factor(
    name="ext_quick_ratio_factor",
    description="速动比率因子（高值排前），quick_ratio=(流动资产-存货)/流动负债。",
    category="quality",
    thesis="速动比率比流动比率更保守——扣除了不易变现的存货。"
    "高速动比率代表即使存货全部无法变现，企业仍有能力偿还短期债务。"
    "这是更严苛的流动性测试。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_quick_ratio_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["quick_ratio"]
    )
    return cross_sectional_rank(fin["quick_ratio"])


@register_factor(
    name="ext_cash_ratio_factor",
    description="现金比率因子（高值排前），cash_ratio=现金及等价物/流动负债。",
    category="quality",
    thesis="现金比率是最保守的流动性指标——仅考虑可以立即动用的现金。"
    "高现金比率代表极端情况下无需变现任何资产即可偿债。"
    "该指标在信用紧缩期对识别财务危机有特殊价值。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_cash_ratio_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["cash_ratio"]
    )
    return cross_sectional_rank(fin["cash_ratio"])


@register_factor(
    name="ext_debt_to_equity_factor",
    description="产权比率因子（低值排前），debt_to_eqt=总负债/股东权益。",
    category="quality",
    thesis="产权比率（Debt-to-Equity）是财务杠杆的核心度量。高D/E意味着"
    "企业依赖债务融资的程度高，财务风险大。同时高杠杆在景气上行期放大收益，"
    "下行期放大亏损。D/E对预测财务困境有显著效力。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_debt_to_equity_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["debt_to_eqt"]
    )
    return cross_sectional_rank(-fin["debt_to_eqt"])


@register_factor(
    name="ext_equity_to_asset_factor",
    description="权益乘数因子（高值排前），assets_to_eqt的倒数=股东权益/总资产。",
    category="quality",
    thesis="股东权益占比（Equity/Assets）反映资本结构的稳健性。"
    "高权益占比意味着企业主要依靠自有资金经营，财务风险低。"
    "使用assets_to_eqt的倒数将其转化为正向指标（高值=稳健）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_equity_to_asset_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["assets_to_eqt"]
    )
    equity_ratio = safe_divide(1.0, fin["assets_to_eqt"])
    return cross_sectional_rank(equity_ratio)


@register_factor(
    name="ext_interest_coverage_factor",
    description="利息保障倍数因子，ebit_to_interest=EBIT/利息费用截面排名（高值排前）。",
    category="quality",
    thesis="利息保障倍数衡量企业用经营利润覆盖利息支出的能力。"
    "倍数<1意味着经营利润不足以支付利息（技术性破产风险）。"
    "倍数>5意味着较强的偿息能力。是信用分析中最核心的指标之一。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_interest_coverage_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebit_to_interest"]
    )
    return cross_sectional_rank(fin["ebit_to_interest"])


# ═══════════════════════════════════════════════════════════════════════════
# E.  Growth / YoY
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_revenue_yoy_factor",
    description="营业收入同比增长率因子，or_yoy截面排名（高值排前）。",
    category="growth",
    thesis="营收同比增长率（or_yoy）是成长性的第一度量。持续高营收增长意味着"
    "市场份额扩大或行业景气上行。营收增长的稳定性（低波动）也是质量信号。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_revenue_yoy_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["or_yoy"]
    )
    return cross_sectional_rank(fin["or_yoy"])


@register_factor(
    name="ext_netprofit_yoy_factor",
    description="归母净利润同比增长率因子，netprofit_yoy截面排名（高值排前）。",
    category="growth",
    thesis="净利润增速是成长投资最关注的指标。净利润增长>营收增长意味着"
    "利润率在改善（经营杠杆效应）。净利润增速放缓往往先于股价见顶。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_netprofit_yoy_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["netprofit_yoy"]
    )
    return cross_sectional_rank(fin["netprofit_yoy"])


@register_factor(
    name="ext_op_yoy_factor",
    description="营业利润同比增长率因子，op_yoy截面排名（高值排前）。",
    category="growth",
    thesis="营业利润增速剔除了非经常性损益的干扰，比净利润增速更纯粹地反映"
    "主营业务的增长趋势。营业利润增速与净利润增速的差异可识别'粉饰'行为。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_op_yoy_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["op_yoy"]
    )
    return cross_sectional_rank(fin["op_yoy"])


@register_factor(
    name="ext_ebt_yoy_factor",
    description="利润总额同比增长率因子，ebt_yoy截面排名（高值排前）。",
    category="growth",
    thesis="利润总额（EBT）增速包含了营业利润和非经营项目，但未扣除所得税。"
    "EBT增速>净利润增速可能意味着税负上升。作为成长因子与净利润增速互补。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_ebt_yoy_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebt_yoy"]
    )
    return cross_sectional_rank(fin["ebt_yoy"])


@register_factor(
    name="ext_assets_yoy_factor",
    description="总资产同比增长率因子，assets_yoy截面排名（高值排前）。",
    category="growth",
    thesis="资产增速反映企业规模的扩张速度。但需区分有机增长（内生）和外延增长"
    "（并购）。过快的资产增长可能伴随商誉堆积和整合风险。"
    "配合营收增速使用——资产增速>营收增速可能效率在下降。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_assets_yoy_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["assets_yoy"]
    )
    return cross_sectional_rank(fin["assets_yoy"])


@register_factor(
    name="ext_equity_yoy_factor",
    description="股东权益同比增长率因子，eqt_yoy截面排名（高值排前）。",
    category="growth",
    thesis="权益增速是内生增长（留存收益积累）+外延增长（增发融资）的综合结果。"
    "权益增长>资产增长意味着杠杆率在下降（财务结构改善）。"
    "稳定的权益增长是可持续分红的保障。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_equity_yoy_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["eqt_yoy"]
    )
    return cross_sectional_rank(fin["eqt_yoy"])


@register_factor(
    name="ext_ocf_yoy_factor",
    description="经营现金流同比增长率因子，ocf_yoy截面排名（高值排前）。",
    category="growth",
    thesis="经营现金流增速是成长质量的关键校验。OCF增速>净利润增速代表"
    "增长的'含金量'高（现金正在跟随利润增长）。OCF增速<净利润增速则需警惕"
    "应收账款膨胀或利润质量恶化。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_ocf_yoy_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_yoy"]
    )
    return cross_sectional_rank(fin["ocf_yoy"])


@register_factor(
    name="ext_bps_yoy_factor",
    description="每股净资产同比增长率因子，bps_yoy截面排名（高值排前）。",
    category="growth",
    thesis="BPS增速是价值投资者关注的成长指标。BPS的增长来自留存收益的积累"
    "（而非增发摊薄），代表股东价值的真实增长。BPS增速>EPS增速时"
    "可能存在增发摊薄，BPS增速<EPS增速是健康的增长模式。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_bps_yoy_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["bps_yoy"]
    )
    return cross_sectional_rank(fin["bps_yoy"])


# ═══════════════════════════════════════════════════════════════════════════
# F.  Additional Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_roic_factor",
    description="投入资本回报率因子，roic截面排名（高值排前）。",
    category="quality",
    thesis="ROIC（Return on Invested Capital）衡量企业使用全部投入资本"
    "（股东权益+有息负债）产生的回报。ROIC>WACC意味着企业创造价值，"
    "ROIC<WACC意味着企业毁灭价值。是巴菲特最看重的指标之一。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_roic_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roic"]
    )
    return cross_sectional_rank(fin["roic"])


@register_factor(
    name="ext_roa_factor",
    description="总资产收益率因子，roa截面排名（高值排前）。",
    category="quality",
    thesis="ROA（Return on Assets）衡量每元资产产生的净利润。ROA综合了"
    "利润率（盈利性）和资产周转率（效率性）两个维度，是杜邦分析的核心。"
    "相比ROE，ROA不受杠杆率影响，更适合跨行业比较。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_roa_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["roa"]
    )
    return cross_sectional_rank(fin["roa"])


@register_factor(
    name="ext_capitalized_to_da_factor",
    description="资本化率因子（低值排前），capitalized_to_da=资本化/折旧摊销。",
    category="quality",
    thesis="资本化率（资本化支出/折旧摊销）反映企业将支出资本化而非费用化的程度。"
    "高资本化率意味着当期费用被低估、利润被高估。异常高的资本化率是盈余管理的"
    "常见手段（特别是软件和研发密集型企业）。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_capitalized_to_da_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["capitalized_to_da"]
    )
    return cross_sectional_rank(-fin["capitalized_to_da"])


@register_factor(
    name="ext_ocf_to_or_factor",
    description="经营现金流/营业收入比率因子，ocf_to_or截面排名（高值排前）。",
    category="quality",
    thesis="经营现金流与营业收入的比率（ocf_to_or）度量收入转化为现金的效率。"
    "比率>1意味着收到了比当期收入更多的现金（预收款模式/应收款回收），"
    "比率持续<0.5则需要关注收入质量和应收账款风险。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_ocf_to_or_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ocf_to_or"]
    )
    return cross_sectional_rank(fin["ocf_to_or"])


@register_factor(
    name="ext_salescash_to_or_factor",
    description="销售收现比因子，salescash_to_or截面排名（高值排前）。",
    category="quality",
    thesis="销售商品提供劳务收到的现金/营业收入（salescash_to_or）是最重要的"
    "收入质量指标。比率>1意味着收入有真金白银支撑（甚至预收），"
    "比率<0.8则需警惕虚增收入。该因子对识别财务造假有实战价值。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ext_salescash_to_or_factor(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["salescash_to_or"]
    )
    return cross_sectional_rank(fin["salescash_to_or"])
