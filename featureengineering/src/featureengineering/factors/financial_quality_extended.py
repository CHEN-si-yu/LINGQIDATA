"""
Extended financial quality factors (财务深度质量因子) — Class 1.

基于三张报表(income/balancesheet/cashflow)和financial_indicator的深度财务质量因子。
关注现金流质量、盈利稳定性、研发投入和应计异象。
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide

# Reporting lag: quarterly reports disclosed ~30-45 days after quarter end
_FIN_LAG = 45  # trading days


# ── 现金流质量 ────────────────────────────────────────────────────────────

@register_factor(
    name="ocf_profit_ratio",
    description="经营现金流/净利润因子截面排名（OCF/利润>1=利润质量高，<0.5=纸面富贵排后）。",
    category="quality",
    thesis="经营现金流与净利润的比值是利润质量的核心标尺——>1意味着每1元利润都有真金白银支撑（高质量），<0.5意味着利润主要来自应收账款和存货增值（纸面富贵）。A股中OCF/Profit>1的公司长期超额收益显著。",
    dependencies=("income.parquet", "cashflow.parquet", "calendar.parquet"),
)
def factor_ocf_profit_ratio(context: FactorContext):
    income = context.load_financial(
        "income.parquet",
        value_cols=["continued_net_profit"],
        date_col="ann_date",
    )
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["n_cashflow_act"],
        date_col="ann_date",
    )

    # Apply reporting lag
    income = income.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()
    cf = cf.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()

    common = income.index.intersection(cf.index)
    profit = income.loc[common, "continued_net_profit"]
    ocf = cf.loc[common, "n_cashflow_act"]

    ratio = safe_divide(ocf, profit.abs())
    return cross_sectional_rank(ratio)


@register_factor(
    name="free_cf_yield",
    description="自由现金流收益率因子，(OCF-购建固定资产)/总市值截面排名（高FCF收益率=价值低估排前）。",
    category="quality",
    thesis="自由现金流收益率=FCF/总市值，是巴菲特最看重的价值指标——FCF是股东真正可以支配的现金，高FCF收益率意味着公司以低成本产生高现金回报。在A股，FCF Yield top20%的股票年化超额收益约8-12%。",
    dependencies=("cashflow.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_free_cf_yield(context: FactorContext):
    cf = context.load_financial(
        "cashflow.parquet",
        value_cols=["n_cashflow_act", "c_paid_invest"],
        date_col="ann_date",
    )
    finance = context.load("finance.parquet")

    cf = cf.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()

    fcf = cf["n_cashflow_act"] - cf["c_paid_invest"]  # Free Cash Flow
    mv = finance["total_mv"]

    common = fcf.index.intersection(mv.index)
    fcf_y = safe_divide(fcf.loc[common], mv.loc[common])

    return cross_sectional_rank(fcf_y)


# ── 盈利稳定性 ────────────────────────────────────────────────────────────

@register_factor(
    name="gross_margin_stability_8q_ext",
    description="毛利率稳定性因子，-(毛利率8季度标准差)截面排名（毛利率稳定=护城河深排前）。",
    category="quality",
    thesis="毛利率的稳定性是经济护城河的量化指标——毛利率越稳定，公司定价权越强（不受竞争威胁）。高且稳定的毛利率=宽护城河，高但波动大=可能在衰退。Morningstar的护城河评级中毛利率稳定性是核心指标。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_gross_margin_stability_8q_ext(context: FactorContext):
    income = context.load_financial(
        "income.parquet",
        value_cols=["revenue", "oper_cost"],
        date_col="ann_date",
    )
    income = income.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()

    gross_margin = (income["revenue"] - income["oper_cost"]) / income["revenue"].replace(0, np.nan)

    stability = gross_margin.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    return cross_sectional_rank(-stability)


@register_factor(
    name="roa_stability_8q_ext",
    description="ROA稳定性因子，-(ROA 8季度标准差/|ROA均值|)截面排名（盈利稳定=质量高排前）。",
    category="quality",
    thesis="盈利的稳定性比盈利的绝对水平更能预测未来收益——盈利波动大的公司有更高的不确定性折价。ROA变异系数(CV)低=盈利可预测性高=高质量公司。这个因子在A股中IC稳定性极好。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_roa_stability_8q_ext(context: FactorContext):
    fi = context.load_financial(
        "financial_indicator.parquet",
        value_cols=["roa"],
        date_col="ann_date",
    )
    fi = fi.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()

    roa = fi["roa"]
    std_8q = roa.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    mean_8q = roa.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    cv = safe_divide(std_8q, mean_8q.abs())

    return cross_sectional_rank(-cv)


# ── 成长质量 ──────────────────────────────────────────────────────────────

@register_factor(
    name="rd_to_revenue_growth",
    description="研发投入强度变化因子，(研发费用/营收)的同比变化截面排名（研发加码=未来增长排前）。",
    category="quality",
    thesis="研发投入强度的边际变化是企业创新意愿的前瞻指标——研发占比提升意味着公司在投资未来增长（即使短期拉低利润）。A股科技/医药行业中研发强度提升>2%的公司次年收入增速显著高于同行。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_rd_to_revenue_growth(context: FactorContext):
    income = context.load_financial(
        "income.parquet",
        value_cols=["revenue", "rd_exp"],
        date_col="ann_date",
    )
    income = income.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()

    rd_ratio = safe_divide(income["rd_exp"].fillna(0), income["revenue"].replace(0, np.nan))
    chg = rd_ratio.groupby(level="Code").transform(lambda s: s.diff(4))  # YoY

    return cross_sectional_rank(chg)


@register_factor(
    name="inventory_revenue_divergence",
    description="存货-营收背离因子，-(存货增速-营收增速)截面排名（存货积压>营收增长=需求恶化排后）。",
    category="quality",
    thesis="存货增速显著快于营收增速是需求疲弱的预警信号——产品卖不出去在仓库堆积，未来可能需要降价清库存（压缩毛利率）。存货营收背离是领先基本面恶化2-3个季度的先行指标。",
    dependencies=("income.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_inventory_revenue_divergence(context: FactorContext):
    income = context.load_financial(
        "income.parquet",
        value_cols=["revenue"],
        date_col="ann_date",
    )
    bs = context.load_financial(
        "balancesheet.parquet",
        value_cols=["inventories"],
        date_col="ann_date",
    )
    income = income.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()
    bs = bs.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()

    rev_growth = income["revenue"].groupby(level="Code").transform(lambda s: s.pct_change(4))
    inv_growth = bs["inventories"].groupby(level="Code").transform(lambda s: s.pct_change(4))

    common = rev_growth.index.intersection(inv_growth.index)
    divergence = inv_growth.loc[common] - rev_growth.loc[common]

    return cross_sectional_rank(-divergence)


@register_factor(
    name="accruals_to_assets",
    description="应计异象因子，-(净利润-OCF)/总资产截面排名（高应计=低质量=未来收益低排后）。",
    category="quality",
    thesis="应计异象(Sloan, 1996)是会计学最著名的异象：应计利润（=净利润-经营现金流）高的公司未来收益系统性偏低——因为应计来自会计估计（如应收账款计提），存在主观操纵空间。低应计=利润更真实=未来收益更高。",
    dependencies=("income.parquet", "cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_accruals_to_assets(context: FactorContext):
    income = context.load_financial("income.parquet", value_cols=["continued_net_profit"], date_col="ann_date")
    cf = context.load_financial("cashflow.parquet", value_cols=["n_cashflow_act"], date_col="ann_date")
    bs = context.load_financial("balancesheet.parquet", value_cols=["total_assets"], date_col="ann_date")

    for df in [income, cf, bs]:
        df = df.groupby(level="Code").shift(_FIN_LAG).groupby(level="Code").ffill()

    common = income.index.intersection(cf.index).intersection(bs.index)
    profit = income.loc[common, "continued_net_profit"]
    ocf = cf.loc[common, "n_cashflow_act"]
    assets = bs.loc[common, "total_assets"]

    accruals = (profit - ocf) / assets.replace(0, np.nan)
    return cross_sectional_rank(-accruals)
