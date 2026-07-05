"""
Financial deep extended factors — Class 1 panel factors.

These factors tap underutilised columns in financial_indicator.parquet (167 cols)
that have 0% NaN and strong theoretical basis.  The 301 existing financial factors
average |IC| = 0.005 — these are designed to improve that by targeting:
  - Cash-flow quality (sales cash, OCF)
  - Earnings quality (tax, margin stability)
  - Growth surprise (sector-relative profit growth)
  - Capital intensity (asset-heavy vs asset-light)

All factors use ``context.load_financial()`` for report-frequency data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# A — Cash-Flow Quality Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="salescash_quality_ext",
    description="销售收现质量因子，salescash_to_or截面排名（高收现比排前）。",
    category="financial",
    thesis="销售商品收到的现金/营业收入比率反映收入含金量——"
           "比率>1意味着不仅当期销售全收回现金、还可能回收了往期应收；"
           "比率<1意味着收入增长快但现金回收慢、可能有大量应收账款累积。"
           "收现质量是识别财务造假的核心指标之一。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_salescash_quality_ext(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["salescash_to_or"],
    )
    return cross_sectional_rank(fin["salescash_to_or"])


@register_factor(
    name="ocf_quality_gap_ext",
    description="经营现金流质量背离因子，(ocf_to_profit排名-ocf_to_or排名)截面排名（利润现金含量>收入现金含量=优质排前）。",
    category="financial",
    thesis="OCF/营业利润与OCF/营业收入的背离反映盈利与收现的质量差异——"
           "利润的现金转化率高于收入的现金转化率=利润质量特别高(排前)；"
           "反之利润增长但现金跟不上=可能存在盈余管理。"
           "两个维度的差距捕捉单一指标无法发现的盈利质量信号。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ocf_quality_gap_ext(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["ocf_to_profit", "ocf_to_or"],
    )
    ocf_profit = fin["ocf_to_profit"].clip(-5, 5)
    ocf_rev = fin["ocf_to_or"].clip(-5, 5)
    # Higher ocf_to_profit relative to ocf_to_or = profit has more cash backing
    gap = ocf_profit - ocf_rev
    return cross_sectional_rank(gap)


# ═══════════════════════════════════════════════════════════════════════════════
# B — Margin & Earnings Quality Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ebit_margin_stability_ext",
    description="EBIT利润率稳定性因子，ebit_of_gr的8Q滚动标准差截面排名（低波动=稳定盈利排前）。",
    category="financial",
    thesis="EBIT利润率的时序稳定性是盈利可预测性的核心度量——"
           "利润率波动大的公司业绩难以预测、估值折价高；"
           "利润率稳定的公司市场对其盈利有高度共识、估值溢价。"
           "8个季度覆盖一个完整业绩周期(含淡旺季)。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_ebit_margin_stability_ext(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["ebit_of_gr"],
    )
    margin = fin["ebit_of_gr"].clip(-1, 1)
    # Rolling std over 8 quarters per stock
    roll_std = margin.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    return cross_sectional_rank(-roll_std)  # stable margin ranks high


@register_factor(
    name="gross_margin_stability_ext",
    description="毛利率稳定性因子，grossprofit_margin的8Q滚动标准差截面排名（低波动=定价权强排前）。",
    category="financial",
    thesis="毛利率的稳定性反映公司的定价权和竞争壁垒——"
           "毛利率波幅大=受原材料/竞争/需求波动影响大、缺乏护城河；"
           "毛利率稳定=产品或服务有差异化优势、客户粘性强。"
           "毛利率稳定性是护城河在财务数据中的量化表达。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_gross_margin_stability_ext(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["grossprofit_margin"],
    )
    margin = fin["grossprofit_margin"].clip(-1, 1)
    roll_std = margin.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    return cross_sectional_rank(-roll_std)


@register_factor(
    name="tax_quality_ext",
    description="税率质量因子，|tax_to_ebt-0.25|截面排名（税率接近法定25%=利润真实排后；偏离大=利润操纵风险排前）。",
    category="financial",
    thesis="实际税率大幅偏离25%法定税率是盈余管理的信号——"
           "税率异常低可能通过税收优惠、递延税项调高利润(不可持续)；"
           "税率异常高可能对历史亏损进行补提(一次性)。"
           "实际税率接近法定税率意味着利润真实、会计保守。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_tax_quality_ext(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["tax_to_ebt"],
    )
    rate = fin["tax_to_ebt"].clip(-0.5, 1.0)
    # Distance from statutory 25% rate
    deviation = np.abs(rate - 0.25)
    return cross_sectional_rank(-deviation)  # close to 25% = good


# ═══════════════════════════════════════════════════════════════════════════════
# C — Growth & Capital Intensity Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="earnings_surprise_ext",
    description="盈利惊喜因子，q_profit_yoy的行业内Z-score截面排名（相对同业的超预期增长排前）。",
    category="financial",
    thesis="单季利润增速在同行业内的相对排名是盈利惊喜的量化度量——"
           "行业内排名靠前=业绩超预期；行业内排名靠后=业绩不及预期。"
           "行业中性化后剥离了行业beta，剩余的alpha更纯粹。"
           "A股对盈利超预期的反应有显著的漂移效应(post-earnings drift)。",
    dependencies=("financial_indicator.parquet", "calendar.parquet",
                  "ths_constituent_stocks.parquet"),
)
def factor_earnings_surprise_ext(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["q_profit_yoy"],
    )
    growth = fin["q_profit_yoy"].clip(-10, 10)

    # Load sector mapping
    industry_map = context.repo.load_industry_map()

    # Build DataFrame with industry info
    codes = growth.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"growth": growth.values, "industry": industries.values},
                      index=growth.index)
    df = df.dropna(subset=["industry"])

    # Sector-relative z-score
    ind_mean = df.groupby(["Date", "industry"])["growth"].transform("mean")
    ind_std = df.groupby(["Date", "industry"])["growth"].transform("std")
    zscore = safe_divide(df["growth"] - ind_mean, ind_std + 1e-8)

    return cross_sectional_rank(zscore)


@register_factor(
    name="capital_intensity_ext",
    description="资本密集度因子，(daa/total_revenue_ps)截面排名（高折旧摊销/营收=重资产排后）。",
    category="financial",
    thesis="折旧摊销与营业收入的比率度量资本密集度——"
           "高比率=重资产模式(制造业、航空、钢铁)、固定成本高、经营杠杆大；"
           "低比率=轻资产模式(软件、服务、消费品牌)、边际成本低、扩张灵活。"
           "在A股，轻资产模式长期获得估值溢价，尤其在利率上行周期。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_capital_intensity_ext(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["daa", "total_revenue_ps"],
    )
    ratio = safe_divide(fin["daa"], fin["total_revenue_ps"] + 1e-10)
    ratio = ratio.clip(0, 2)
    return cross_sectional_rank(-ratio)  # light-asset ranks high


