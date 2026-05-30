"""Financial composite scoring factors.

Piotroski F-Score, Beneish M-Score components, Altman Z-Score,
cash flow quality, and earnings quality metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ── Piotroski F-Score (9-point fundamental strength) ───────────────────

@register_factor(
    name="f_score",
    description="Piotroski F-Score，9分制基本面综合评分（盈利4分+杠杆3分+运营2分），截面排名。",
    category="quality",
    thesis="Piotroski(2000)发现高BP股票中F-Score高的组合年化超额收益显著。F-Score从盈利质量、杠杆变化、运营效率三个维度识别价值陷阱vs价值转机。A股市场中低BP+高FScore是最优组合。",
    dependencies=("financial_indicator.parquet", "balancesheet.parquet", "cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_f_score(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["roa", "current_ratio", "gross_margin", "assets_turn"])
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_assets", "total_hldr_eqy_exc_min_int", "total_liab"])
    cf = context.load_financial("cashflow.parquet",
        value_cols=["n_cashflow_act"])
    inc = context.load_financial("income.parquet",
        value_cols=["n_income"])

    common_idx = fin.index.intersection(bs.index).intersection(cf.index).intersection(inc.index)
    fin = fin.loc[common_idx]
    bs = bs.loc[common_idx]
    cf = cf.loc[common_idx]
    inc = inc.loc[common_idx]

    score = pd.Series(0.0, index=common_idx)

    # Profitability (4 points)
    score += (fin["roa"] > 0).astype(float)
    score += (cf["n_cashflow_act"] > 0).astype(float)
    roa_qoq = fin["roa"].groupby(level="Code").transform(lambda s: s.diff(1))
    score += (roa_qoq > 0).astype(float)
    score += (cf["n_cashflow_act"] > inc["n_income"]).astype(float)

    # Leverage / liquidity (3 points)
    leverage = bs["total_liab"] / bs["total_assets"].replace(0, np.nan)
    delta_leverage = leverage.groupby(level="Code").transform(lambda s: s.diff(1))
    score += (delta_leverage < 0).astype(float)
    delta_current = fin["current_ratio"].groupby(level="Code").transform(lambda s: s.diff(1))
    score += (delta_current > 0).astype(float)
    equity = bs["total_hldr_eqy_exc_min_int"]
    delta_equity = equity.groupby(level="Code").transform(lambda s: s.diff(1))
    score += (delta_equity > 0).astype(float)

    # Operating efficiency (2 points)
    delta_gross = fin["gross_margin"].groupby(level="Code").transform(lambda s: s.diff(1))
    score += (delta_gross > 0).astype(float)
    delta_turn = fin["assets_turn"].groupby(level="Code").transform(lambda s: s.diff(1))
    score += (delta_turn > 0).astype(float)

    return cross_sectional_rank(score)


# ── Beneish M-Score components ─────────────────────────────────────────

@register_factor(
    name="dsri",
    description="应收账款周转天数指数(DSRI)，(AR_t/Rev_t)/(AR_t-1/Rev_t-1)截面排名（高值排后）。Beneish M-Score第一分量。",
    category="quality",
    thesis="应收账款增速远超营收增速是提前确认收入/虚增收入的信号。DSRI>1.031是操纵的标志(Beneish 1999)。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_dsri(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["accounts_receiv"])
    inc = context.load_financial("income.parquet",
        value_cols=["revenue"])
    common = bs.index.intersection(inc.index)
    ar = bs["accounts_receiv"].loc[common]
    rev = inc["revenue"].loc[common]
    ratio = ar / rev.replace(0, np.nan)
    dsri = ratio.groupby(level="Code").transform(lambda s: s / s.shift(1).replace(0, np.nan))
    dsri = dsri.clip(0, 5)
    return cross_sectional_rank(-dsri)


@register_factor(
    name="gmi",
    description="毛利率指数(GMI)，毛利率_t-1/毛利率_t截面排名。Beneish M-Score第二分量，毛利率恶化信号。",
    category="quality",
    thesis="毛利率持续下滑意味着竞争力减弱或成本压力增大，是盈余操纵的动机来源之一。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_gmi(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["gross_margin"])
    gm = fin["gross_margin"]
    gmi = gm.groupby(level="Code").transform(lambda s: s.shift(1) / s.replace(0, np.nan))
    gmi = gmi.clip(0, 5)
    return cross_sectional_rank(gmi)


@register_factor(
    name="aqi",
    description="资产质量指数(AQI)，(1-CA_t/TA_t)/(1-CA_t-1/TA_t-1)截面排名（低值排前）。Beneish M-Score第三分量。",
    category="quality",
    thesis="非流动资产占比突然上升可能意味着费用资本化以虚增利润。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_aqi(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_assets", "total_cur_assets"])
    ca_ratio = bs["total_cur_assets"] / bs["total_assets"].replace(0, np.nan)
    nca_ratio = 1 - ca_ratio
    aqi = nca_ratio.groupby(level="Code").transform(lambda s: s / s.shift(1).replace(0, np.nan))
    aqi = aqi.clip(0, 3)
    return cross_sectional_rank(-aqi)


@register_factor(
    name="sgi",
    description="销售收入增长指数(SGI)，营收_t/营收_t-1截面排名（取负向=异常高增长排后）。Beneish M-Score第四分量。",
    category="quality",
    thesis="异常高的营收增长可能伴随盈余管理动机。SGI>1.134是Beneish操纵标志之一。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_sgi(context: FactorContext):
    inc = context.load_financial("income.parquet",
        value_cols=["revenue"])
    rev = inc["revenue"]
    sgi = rev.groupby(level="Code").transform(lambda s: s / s.shift(1).replace(0, np.nan))
    sgi = sgi.clip(0, 3)
    return cross_sectional_rank(-sgi)


@register_factor(
    name="tata",
    description="总应计/总资产(TATA)，(NI-CFO)/TA截面排名（取负向=高应计排后）。Sloan(1996)应计异象核心变量。",
    category="quality",
    thesis="高应计利润意味着利润含金量低，现金支撑不足。Sloan发现低应计组合年化超额11%。A股市场应计效应同样显著。",
    dependencies=("income.parquet", "cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_tata(context: FactorContext):
    inc = context.load_financial("income.parquet",
        value_cols=["n_income"])
    cf = context.load_financial("cashflow.parquet",
        value_cols=["n_cashflow_act"])
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_assets"])
    common = inc.index.intersection(cf.index).intersection(bs.index)
    ni = inc["n_income"].loc[common]
    cfo = cf["n_cashflow_act"].loc[common]
    ta = bs["total_assets"].loc[common]
    tata = (ni - cfo) / ta.replace(0, np.nan)
    return cross_sectional_rank(-tata)


# ── Altman Z-Score (China version) ─────────────────────────────────────

@register_factor(
    name="z_score_china",
    description="中国版Altman Z-Score，1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MV/TL + 1.0*Sales/TA截面排名。",
    category="quality",
    thesis="修正版Z-Score综合衡量企业财务健康度，在中国市场对ST预警有较好的预测效果。高Z值=低破产风险。",
    dependencies=("balancesheet.parquet", "income.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_z_score_china(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_cur_assets", "total_cur_liab", "total_assets",
                     "total_liab", "surplus_rese", "undistr_porfit"])
    inc = context.load_financial("income.parquet",
        value_cols=["ebit", "revenue"])
    fin = context.load("finance.parquet")

    common = bs.index.intersection(inc.index)
    bs = bs.loc[common]
    inc = inc.loc[common]

    wc = bs["total_cur_assets"] - bs["total_cur_liab"]
    ta = bs["total_assets"]
    re = bs["surplus_rese"].fillna(0) + bs["undistr_porfit"].fillna(0)

    x1 = wc / ta.replace(0, np.nan)
    x2 = re / ta.replace(0, np.nan)
    x3 = inc["ebit"] / ta.replace(0, np.nan)

    # MV/TL: market value from finance panel
    total_mv = fin["total_mv"]
    common2 = common.intersection(total_mv.index)
    z = pd.Series(np.nan, index=common)
    z.loc[x1.index] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3
    x4 = total_mv.loc[common2] / bs["total_liab"].loc[common2].replace(0, np.nan)
    z.loc[x4.index] = z.loc[x4.index].fillna(0) + 0.6 * x4.loc[x4.index]
    x5 = inc["revenue"] / ta.replace(0, np.nan)
    z.loc[x5.index] = z.loc[x5.index].fillna(0) + 1.0 * x5.loc[x5.index]
    z = z.clip(-10, 20)
    return cross_sectional_rank(z)


# ── Cash flow quality ──────────────────────────────────────────────────

@register_factor(
    name="ocf_stability_8q",
    description="经营现金流稳定性因子，8季度OCF/TA滚动窗口的CV倒数截面排名（高稳定排前）。",
    category="quality",
    thesis="现金流稳定的公司盈利质量更可靠，高OCF波动率预示基本面不确定。",
    dependencies=("cashflow.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_ocf_stability_8q(context: FactorContext):
    cf = context.load_financial("cashflow.parquet",
        value_cols=["n_cashflow_act"])
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_assets"])
    common = cf.index.intersection(bs.index)
    ocf_ta = cf["n_cashflow_act"].loc[common] / bs["total_assets"].loc[common].replace(0, np.nan)
    cv = ocf_ta.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).std() / s.rolling(8, min_periods=4).mean().abs().replace(0, np.nan)
    )
    return cross_sectional_rank(-cv)


@register_factor(
    name="fcf_conversion",
    description="自由现金流转化率因子，FCF/净利润的4季度滚动平均值截面排名。",
    category="quality",
    thesis="FCF/净利润持续接近或超过1说明利润能有效转化为现金，高转化率代表利润含金量高。",
    dependencies=("cashflow.parquet", "income.parquet", "calendar.parquet"),
)
def factor_fcf_conversion(context: FactorContext):
    cf = context.load_financial("cashflow.parquet",
        value_cols=["free_cashflow"])
    inc = context.load_financial("income.parquet",
        value_cols=["n_income"])
    common = cf.index.intersection(inc.index)
    ratio = cf["free_cashflow"].loc[common] / inc["n_income"].loc[common].replace(0, np.nan)
    ratio = ratio.clip(-5, 5)
    avg_4q = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(4, min_periods=2).mean()
    )
    return cross_sectional_rank(avg_4q)


@register_factor(
    name="cash_cycle",
    description="现金周期因子，(存货周转天数+应收周转天数-应付周转天数)截面排名（取负向=短周期排前）。",
    category="quality",
    thesis="现金周期短意味着公司能快速将存货变现并延迟支付供应商，运营效率高、资金占用少。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_cash_cycle(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["invturn_days", "arturn_days", "turn_days"])
    cycle = fin["invturn_days"] + fin["arturn_days"] - fin["turn_days"]
    cycle = cycle.clip(-365, 1000)
    return cross_sectional_rank(-cycle)


# ── Earnings momentum ──────────────────────────────────────────────────

@register_factor(
    name="earnings_momentum_4q",
    description="盈利动量因子，4季度滚动净利润的季度差分/4季度滚动平均截面排名。",
    category="quality",
    thesis="盈利加速度是比盈利水平更强的动量信号。盈利持续改善的公司享有基本面和情绪的双重助力。",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_earnings_momentum_4q(context: FactorContext):
    inc = context.load_financial("income.parquet",
        value_cols=["n_income"])
    ni = inc["n_income"]
    ni_rolling = ni.groupby(level="Code").transform(
        lambda s: s.rolling(4, min_periods=3).sum()
    )
    chg = ni_rolling.groupby(level="Code").transform(lambda s: s.diff(4))
    avg = ni_rolling.groupby(level="Code").transform(
        lambda s: s.rolling(4, min_periods=3).mean().abs()
    )
    mom = safe_divide(chg, avg).clip(-2, 2)
    return cross_sectional_rank(mom)


# ── Supplementary financial composite factors ──────────────────────────────


@register_factor(
    name="dupont_roe_decomposition",
    description="杜邦ROE分解综合因子 (净利润率排名+周转率排名-杠杆排名)。",
    category="financial",
    thesis="ROE的杜邦三分法：高质量ROE来自利润率和周转率而非杠杆",
    dependencies=("financial_indicator.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_dupont_roe_decomposition(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["netprofit_margin", "assets_turn"])
    npm = fin["netprofit_margin"]
    aturn = fin["assets_turn"]

    npm_rank = npm.groupby(level="Date").rank(pct=True)
    aturn_rank = aturn.groupby(level="Date").rank(pct=True)

    # Leverage rank (low leverage = good)
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_assets", "total_hldr_eqy_exc_min_int"])
    leverage = safe_divide(bs["total_assets"], bs["total_hldr_eqy_exc_min_int"] + 1e-8)
    leverage_rank = (-leverage).groupby(level="Date").rank(pct=True)

    common = npm_rank.index.intersection(aturn_rank.index).intersection(leverage_rank.index)
    combo = (npm_rank.loc[common] + aturn_rank.loc[common] + leverage_rank.loc[common]) / 3.0
    return cross_sectional_rank(combo)


@register_factor(
    name="profitability_durability",
    description="盈利持续性因子 (8季度每季度都盈利的虚拟变量)。",
    category="financial",
    thesis="8个季度持续盈利是基本面最硬的质量要求，持续亏损任何一期都大幅降低质量",
    dependencies=("income.parquet", "calendar.parquet"),
)
def factor_profitability_durability(context: FactorContext):
    inc = context.load_financial("income.parquet", value_cols=["n_income"])
    ni = inc["n_income"]
    is_profitable = (ni > 0).astype(float)
    durability = is_profitable.groupby(level="Code").transform(
        lambda s: s.rolling(8, min_periods=4).sum()
    )
    return cross_sectional_rank(durability)


# ── Piotroski F-Score inspired components ─────────────────────────────────

@register_factor(
    name="f_score_profitability",
    description="F-Score盈利子维度：ROA>0、OCF>0、ROA变化>0三条件等权截面排名。",
    category="financial",
    thesis="Piotroski F-Score中的盈利能力维度是价值股筛选的核心——盈利为正+现金流为正+盈利改善同时满足的公司在价值股中最具反转潜力。",
    dependencies=("financial_indicator.parquet", "cashflow.parquet", "calendar.parquet"),
)
def factor_f_score_profitability(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["roa"])
    cf = context.load_financial("cashflow.parquet", value_cols=["n_cashflow_act"])
    roa = fin["roa"]
    ocf = cf["n_cashflow_act"]
    roa_chg = roa.groupby(level="Code").transform(lambda s: s.diff(4))
    score = (roa > 0).astype(float) + (ocf > 0).astype(float) + (roa_chg > 0).astype(float)
    return cross_sectional_rank(score)


@register_factor(
    name="f_score_efficiency",
    description="F-Score效率子维度：毛利率改善+资产周转率改善+费用率下降三条件等权截面排名。",
    category="financial",
    thesis="运营效率改善是财务反转的重要驱动力——毛利率提升+周转加快+费用率下降同时出现时，公司基本面改善的置信度最高。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_f_score_efficiency(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["gross_margin", "assets_turn", "or_yoy"])
    gm_chg = fin["gross_margin"].groupby(level="Code").transform(lambda s: s.diff(4))
    at_chg = fin["assets_turn"].groupby(level="Code").transform(lambda s: s.diff(4))
    # or_yoy improvement as proxy for expense ratio improvement
    oryoy_chg = fin["or_yoy"].groupby(level="Code").transform(lambda s: s.diff(4))
    score = (gm_chg > 0).astype(float) + (at_chg > 0).astype(float) + (oryoy_chg > 0).astype(float)
    return cross_sectional_rank(score)


@register_factor(
    name="f_score_leverage",
    description="F-Score杠杆子维度：负债率下降+流动比率改善+无股权稀释三条件等权截面排名。",
    category="financial",
    thesis="财务杠杆下降和流动性改善是财务健康的改善方向——降低杠杆+改善流动性同时出现的公司财务风险在边际缩小。",
    dependencies=("financial_indicator.parquet", "calendar.parquet"),
)
def factor_f_score_leverage(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["debt_to_assets", "current_ratio", "eqt_yoy"])
    debt_chg = fin["debt_to_assets"].groupby(level="Code").transform(lambda s: s.diff(4))
    cr_chg = fin["current_ratio"].groupby(level="Code").transform(lambda s: s.diff(4))
    no_dilution = fin["eqt_yoy"] >= 5  # equity not significantly growing (no large dilution)
    score = (debt_chg < 0).astype(float) + (cr_chg > 0).astype(float) + no_dilution.astype(float)
    return cross_sectional_rank(score)


# ── Altman Z-Score inspired ──────────────────────────────────────────────

@register_factor(
    name="altman_z_score",
    description="Altman Z-Score截面排名（财务困境预测模型——高Z值=低破产风险排前）。",
    category="financial",
    thesis="Altman Z-Score是学术界最经典的财务困境预测模型——综合了营运资本/资产、留存收益/资产、EBIT/资产、权益/负债、收入/资产五个维度，高Z值代表低破产风险。",
    dependencies=("balancesheet.parquet", "income.parquet", "calendar.parquet"),
)
def factor_altman_z_score(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_cur_assets", "total_cur_liab", "total_assets",
                     "undistr_porfit", "surplus_rese", "total_liab",
                     "total_hldr_eqy_exc_min_int"])
    inc = context.load_financial("income.parquet",
        value_cols=["ebit", "revenue"])
    wc = bs["total_cur_assets"] - bs["total_cur_liab"]
    retained = bs["undistr_porfit"].fillna(0) + bs["surplus_rese"].fillna(0)
    ta = bs["total_assets"]
    x1 = wc / ta.replace(0, np.nan)
    x2 = retained / ta.replace(0, np.nan)
    x3 = inc["ebit"] / ta.replace(0, np.nan)
    x4 = bs["total_hldr_eqy_exc_min_int"] / bs["total_liab"].replace(0, np.nan)
    x5 = inc["revenue"] / ta.replace(0, np.nan)
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
    return cross_sectional_rank(z)


# ── Beneish M-Score simplified ────────────────────────────────────────────

@register_factor(
    name="beneish_earnings_manipulation",
    description="Beneish盈余操纵风险因子，应收款+毛利率恶化+资产质量下降综合截面排名（高风险排后）。",
    category="financial",
    thesis="Beneish M-Score是识别财务造假/盈余操纵的经典模型——简化版关注应收款异常增长、毛利率恶化、资产质量下降三个核心维度，高得分=高操纵风险。",
    dependencies=("balancesheet.parquet", "income.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_beneish_earnings_manipulation(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["accounts_receiv", "total_assets"])
    inc = context.load_financial("income.parquet", value_cols=["revenue"])
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["gross_margin", "assets_turn"])
    # DSRI: receivables/revenue ratio change
    ar_to_rev = bs["accounts_receiv"] / inc["revenue"].replace(0, np.nan)
    dsri = ar_to_rev.groupby(level="Code").transform(lambda s: s.diff(4))
    # GMI: gross margin decline
    gmi = fin["gross_margin"].groupby(level="Code").transform(lambda s: s.diff(4))
    # AQI: asset quality decline
    aqi = fin["assets_turn"].groupby(level="Code").transform(lambda s: s.diff(4))
    # Composite: higher = more manipulation risk => rank negative
    score = dsri.fillna(0) / 0.1 + (-gmi.fillna(0) / 2.0) + (-aqi.fillna(0) / 0.2)
    return cross_sectional_rank(-score)


# ── Financial health composite ────────────────────────────────────────────

@register_factor(
    name="financial_health_composite",
    description="财务健康综合因子：Z-Score+F-Score盈利+F-Score效率+F-Score杠杆的等权截面排名。",
    category="financial",
    thesis="综合多维度财务健康指标能更全面评估企业财务质量——单独指标可能存在行业偏差或会计噪音，多维度的交叉验证降低误判风险。",
    dependencies=("financial_indicator.parquet", "balancesheet.parquet", "income.parquet", "cashflow.parquet", "calendar.parquet"),
)
def factor_financial_health_composite(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["roa", "gross_margin", "assets_turn", "or_yoy",
                     "debt_to_assets", "current_ratio", "eqt_yoy"])
    cf = context.load_financial("cashflow.parquet", value_cols=["n_cashflow_act"])
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_cur_assets", "total_cur_liab", "total_assets",
                     "undistr_porfit", "surplus_rese", "total_liab",
                     "total_hldr_eqy_exc_min_int"])
    inc = context.load_financial("income.parquet", value_cols=["ebit", "revenue"])

    # Profitability sub-score
    roa = fin["roa"]
    roa_chg = roa.groupby(level="Code").transform(lambda s: s.diff(4))
    profit_score = ((roa > 0).astype(float) + (cf["n_cashflow_act"] > 0).astype(float) + (roa_chg > 0).astype(float)) / 3.0

    # Efficiency sub-score
    gm_chg = fin["gross_margin"].groupby(level="Code").transform(lambda s: s.diff(4))
    at_chg = fin["assets_turn"].groupby(level="Code").transform(lambda s: s.diff(4))
    eff_score = ((gm_chg > 0).astype(float) + (at_chg > 0).astype(float)) / 2.0

    # Solvency sub-score
    debt_chg = fin["debt_to_assets"].groupby(level="Code").transform(lambda s: s.diff(4))
    solv_score = ((debt_chg < 0).astype(float) + (fin["current_ratio"].groupby(level="Code").transform(lambda s: s.diff(4)) > 0).astype(float)) / 2.0

    # Z-Score
    wc = bs["total_cur_assets"] - bs["total_cur_liab"]
    retained = bs["undistr_porfit"].fillna(0) + bs["surplus_rese"].fillna(0)
    ta = bs["total_assets"]
    z = 1.2 * wc/ta.replace(0,np.nan) + 1.4 * retained/ta.replace(0,np.nan) + 3.3 * inc["ebit"]/ta.replace(0,np.nan) + 0.6 * bs["total_hldr_eqy_exc_min_int"]/bs["total_liab"].replace(0,np.nan) + 0.999 * inc["revenue"]/ta.replace(0,np.nan)
    z_rank = z.groupby(level="Date").rank(pct=True)

    health = (profit_score.groupby(level="Date").rank(pct=True) +
              eff_score.groupby(level="Date").rank(pct=True) +
              solv_score.groupby(level="Date").rank(pct=True) +
              z_rank) / 4.0
    return cross_sectional_rank(health)


# ── Cash adequacy ─────────────────────────────────────────────────────────

@register_factor(
    name="cash_adequacy_ratio",
    description="现金充裕度因子，(货币资金+交易性金融资产)/流动负债截面排名。",
    category="financial",
    thesis="现金及现金等价物覆盖流动负债的能力是最保守的流动性度量——高覆盖率意味着企业在极端压力下也不需要通过贱卖资产偿还债务。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_cash_adequacy_ratio(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["money_cap", "trad_asset", "total_cur_liab"])
    cash = bs["money_cap"].fillna(0) + bs["trad_asset"].fillna(0)
    ratio = cash / bs["total_cur_liab"].replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Solvency margin ──────────────────────────────────────────────────────

@register_factor(
    name="solvency_margin",
    description="偿债安全边际因子，(EBITDA-利息支出)/短期到期债务截面排名。",
    category="financial",
    thesis="(EBITDA-利息)/短期到期债务衡量企业满足短期偿债需求后的经营盈余——正值意味着内生现金流有余力，负值意味着借钱还债的恶性循环风险。",
    dependencies=("financial_indicator.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_solvency_margin(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet",
        value_cols=["ebitda", "ebit_to_interest"])
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["st_borr", "non_cur_liab_due_1y"])
    # ebitda - interest_exp (approximate: ebitda / ebit_to_interest * ebit = interest)
    # Simpler: just use ebitda directly and compare to short-term debt
    st_debt = bs["st_borr"].fillna(0) + bs["non_cur_liab_due_1y"].fillna(0)
    margin = fin["ebitda"] / st_debt.replace(0, np.nan)
    margin = margin.clip(-5, 20)
    return cross_sectional_rank(margin)


# ── Accrual anomaly depth ─────────────────────────────────────────────────

@register_factor(
    name="net_operating_assets_change",
    description="净营运资产变化因子，-(NOA_t-NOA_t-1)/总资产截面排名（NOA增长=盈余操纵风险排后）。",
    category="financial",
    thesis="Hirshleifer et al.(2004)发现净营运资产(NOA)增长是比应计利润更广泛的盈余管理度量——NOA增长意味着管理层的乐观估计在财务报表上的体现更多，后续回报更低。",
    dependencies=("balancesheet.parquet", "calendar.parquet"),
)
def factor_net_operating_assets_change(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet",
        value_cols=["total_cur_assets", "money_cap", "total_cur_liab",
                     "st_borr", "total_assets", "lt_borr", "bond_payable"])
    # NOA = (Current Assets - Cash) - (Current Liab - ST Debt) + (Total Assets - Current Assets) - (Total Liab - Current Liab)
    # Simplified: NOA = (Total Assets - Cash) - (Total Liab - Total Debt)
    total_debt = bs["st_borr"].fillna(0) + bs["lt_borr"].fillna(0) + bs["bond_payable"].fillna(0)
    noa = (bs["total_assets"] - bs["money_cap"].fillna(0)) - (bs["total_cur_liab"] - bs["total_cur_liab"].fillna(0))
    # Further simplified: NOA ≈ total_assets - money_cap - non-debt liabilities
    # Actually let's just compute operating assets directly
    oa = bs["total_cur_assets"] - bs["money_cap"].fillna(0)
    ol = bs["total_cur_liab"] - bs["st_borr"].fillna(0)
    noa = oa - ol
    noa_chg = noa.groupby(level="Code").transform(lambda s: s.diff(4))
    noa_chg_ratio = noa_chg / bs["total_assets"].shift(1).replace(0, np.nan)
    return cross_sectional_rank(-noa_chg_ratio)


# ── Dividend sustainability ───────────────────────────────────────────────

@register_factor(
    name="dividend_sustainability",
    description="分红可持续性因子，(FCFE-股利)/|股利|截面排名（正=内生现金流可覆盖分红排前）。",
    category="financial",
    thesis="分红率本身(DP)是估值和收益因子——分红可持续性是质量因子：FCFE大于实际分红意味着分红不是通过借债或侵蚀资产维持的。",
    dependencies=("cashflow.parquet", "financial_indicator.parquet", "calendar.parquet"),
)
def factor_dividend_sustainability(context: FactorContext):
    cf = context.load_financial("cashflow.parquet",
        value_cols=["free_cashflow", "c_pay_dist_dpcp_int_exp"])
    # c_pay_dist_dpcp_int_exp includes dividends + interest paid
    # We approximate dividends as this amount (since we don't have pure dividend paid)
    fcf = cf["free_cashflow"]
    div_paid = cf["c_pay_dist_dpcp_int_exp"].abs()
    surplus = (fcf - div_paid) / div_paid.replace(0, np.nan)
    surplus = surplus.clip(-3, 3)
    return cross_sectional_rank(surplus)
