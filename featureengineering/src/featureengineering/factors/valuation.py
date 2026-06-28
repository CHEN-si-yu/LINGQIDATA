from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ── Value ───────────────────────────────────────────────────────────────

@register_factor(
    name="bp",
    description="账面市值比(BP)因子，1/PB截面排名，高值代表价值股。",
    category="valuation",
    thesis="价值因子是Fama-French三因子之一，高BP股票长期有超额收益。",
    dependencies=("finance.parquet",),
)
def factor_bp(context: FactorContext):
    finance = context.load("finance.parquet")
    bp = 1.0 / finance["pb"].replace(0, np.nan)
    return cross_sectional_rank(bp)


@register_factor(
    name="sp_ttm",
    description="市销率倒数(SP_TTM)因子，1/PS_TTM截面排名。",
    category="valuation",
    thesis="PS估值对净利润为负的公司仍有定义域，在A股覆盖面优于PE类因子。",
    dependencies=("finance.parquet",),
)
def factor_sp_ttm(context: FactorContext):
    finance = context.load("finance.parquet")
    sp = 1.0 / finance["ps_ttm"].replace(0, np.nan)
    return cross_sectional_rank(sp)


@register_factor(
    name="dp_ttm",
    description="滚动股息率因子，截面排名。",
    category="valuation",
    thesis="高股息率股票在低利率环境中具备配置价值，且在下跌市中具有防御属性。",
    dependencies=("finance.parquet",),
)
def factor_dp_ttm(context: FactorContext):
    finance = context.load("finance.parquet")
    dp = finance["dv_ttm"]
    return cross_sectional_rank(dp)


# ── Size ────────────────────────────────────────────────────────────────

@register_factor(
    name="log_total_mv",
    description="对数总市值因子，负对数总市值（小市值排前）。",
    category="valuation",
    thesis="规模因子是A股最显著的单因子之一，小市值效应长期存在。",
    dependencies=("finance.parquet",),
)
def factor_log_total_mv(context: FactorContext):
    finance = context.load("finance.parquet")
    log_mv = np.log(finance["total_mv"].replace(0, np.nan))
    return cross_sectional_rank(-log_mv)


@register_factor(
    name="log_circ_mv",
    description="对数流通市值因子，负对数流通市值截面排名。",
    category="valuation",
    thesis="流通市值比总市值更精确反映可交易盘规模，对小盘效应捕捉更纯。",
    dependencies=("finance.parquet",),
)
def factor_log_circ_mv(context: FactorContext):
    finance = context.load("finance.parquet")
    log_cmv = np.log(finance["circ_mv"].replace(0, np.nan))
    return cross_sectional_rank(-log_cmv)


@register_factor(
    name="float_mv_ratio",
    description="自由流通市值占比因子，free_share/total_share截面排名。",
    category="valuation",
    thesis="自由流通盘占比低代表筹码锁定度高、实际流通盘小，可能伴随更高的波动弹性。",
    dependencies=("finance.parquet",),
)
def factor_float_mv_ratio(context: FactorContext):
    finance = context.load("finance.parquet")
    ratio = finance["free_share"] / finance["total_share"].replace(0, np.nan)
    return cross_sectional_rank(-ratio)


# ── Liquidity ───────────────────────────────────────────────────────────

@register_factor(
    name="turnover_20",
    description="20日平均换手率因子（总股本换手率），低换手排前。",
    category="valuation",
    thesis="低换手率反映筹码稳定、投机度低，在A股中具有正向截面预测力。",
    dependencies=("finance.parquet",),
)
def factor_turnover_20(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    avg_turnover = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_turnover)


@register_factor(
    name="turnover_f_20",
    description="20日平均自由流通换手率因子，低换手排前。",
    category="valuation",
    thesis="自由流通换手率剔除大股东锁定股份，更精确反映真实交易活跃度。",
    dependencies=("finance.parquet",),
)
def factor_turnover_f_20(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate_f"]
    avg_turnover = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_turnover)


@register_factor(
    name="turnover_vol_20",
    description="20日换手率波动因子，换手率标准差截面排名（低波动排前）。",
    category="valuation",
    thesis="换手率剧烈波动常反映资金博弈激烈，是风险信号。",
    dependencies=("finance.parquet",),
)
def factor_turnover_vol_20(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    vol = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-vol)


@register_factor(
    name="volume_ratio",
    description="量比因子（当日成交量相对5日均量），截面排名。",
    category="valuation",
    thesis="量比是盘中常用指标，极端量比常伴随短期反转。",
    dependencies=("finance.parquet",),
)
def factor_volume_ratio(context: FactorContext):
    finance = context.load("finance.parquet")
    vol_ratio = finance["volume_ratio"]
    return cross_sectional_rank(-vol_ratio)


# ── PE percentile ───────────────────────────────────────────────────────

@register_factor(
    name="pe_ttm_percentile",
    description="PE_TTM历史分位因子（低分位=估值处于历史低位，排前）。",
    category="valuation",
    thesis="估值相对于自身历史的低位是价值回归的潜在信号。",
    dependencies=("finance.parquet",),
)
def factor_pe_ttm_percentile(context: FactorContext):
    finance = context.load("finance.parquet")
    percentile = finance["pe_ttm_percentile"]
    return cross_sectional_rank(-percentile)


# ── Dividend composite ────────────────────────────────────────────────────

@register_factor(
    name="dv_composite",
    description="股息率综合因子，dv_ratio与dv_ttm的等权平均截面排名。",
    category="valuation",
    thesis="dv_ratio（报告期股息率）与dv_ttm（滚动股息率）从不同维度度量股息回报，综合后信号更稳定。",
    dependencies=("finance.parquet",),
)
def factor_dv_composite(context: FactorContext):
    finance = context.load("finance.parquet")
    with np.errstate(invalid="ignore"):
        rank_ratio = finance["dv_ratio"].groupby(level="Date").rank(pct=True)
        rank_ttm = finance["dv_ttm"].groupby(level="Date").rank(pct=True)
    composite = (rank_ratio + rank_ttm) / 2.0
    return composite.rename("dv_composite")


# ── Enterprise value based ───────────────────────────────────────────────

@register_factor(
    name="ebitda_to_ev",
    description="企业价值倍数因子，EBITDA/(总市值+总负债-现金)截面排名。",
    category="valuation",
    thesis="EBITDA/EV剔除了资本结构和折旧政策的影响，比PE更适合跨行业比较。高EBITDA/EV意味着企业相对其经营盈利能力被低估，是Greenblatt神奇公式的核心维度之一。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_ebitda_to_ev(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebitda"]
    )
    bs = context.load_financial(
        "balancesheet.parquet", value_cols=["total_liab", "money_cap"]
    )
    finance = context.load("finance.parquet")
    ebitda = fin["ebitda"]
    total_mv = finance["total_mv"]
    total_liab = bs["total_liab"]
    cash = bs["money_cap"]
    common = ebitda.index.intersection(total_mv.index).intersection(
        total_liab.index).intersection(cash.index)
    ev = total_mv.loc[common] + total_liab.loc[common] - cash.loc[common]
    ratio = ebitda.loc[common] / ev.replace(0, np.nan)
    return cross_sectional_rank(ratio)


# ── Supplementary valuation factors ───────────────────────────────────────


@register_factor(
    name="pb_percentile_5y",
    description="PB 5年历史分位因子 (低位排前, 负向)。",
    category="valuation",
    thesis="PB历史分位低意味着估值处于历史低位区间，均值回复力量强于绝对估值水平",
    dependencies=("finance.parquet",),
)
def factor_pb_percentile_5y(context: FactorContext):
    finance = context.load("finance.parquet")
    pb = finance["pb"]

    def _percentile(s, window):
        return s.rolling(window, min_periods=window // 2).apply(
            lambda x: (x.iloc[-1] > x).mean(), raw=False
        )

    pct = pb.groupby(level="Code").transform(lambda s: _percentile(s, 1260))
    return cross_sectional_rank(-pct)


@register_factor(
    name="ev_to_total_assets",
    description="企业价值/总资产因子截面排名。",
    category="valuation",
    thesis="EV/Total Assets衡量整个企业相对于其资产基础的价值，越低越被低估",
    dependencies=("finance.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_ev_to_total_assets(context: FactorContext):
    bs = context.load_financial("balancesheet.parquet", value_cols=["total_liab", "money_cap", "total_assets"])
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"]
    total_liab = bs["total_liab"]
    cash = bs["money_cap"]
    ta = bs["total_assets"]
    common = total_mv.index.intersection(total_liab.index).intersection(cash.index).intersection(ta.index)
    ev = total_mv.loc[common] + total_liab.loc[common] - cash.loc[common]
    ratio = safe_divide(ev, ta.loc[common])
    return cross_sectional_rank(-ratio)


@register_factor(
    name="mv_to_ebitda",
    description="市值/EBITDA比率因子截面排名 (低排前, 负向)。",
    category="valuation",
    thesis="MV/EBITDA是EV/EBITDA的简化版，适合快速筛选低估值股票",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_mv_to_ebitda(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["ebitda"])
    finance = context.load("finance.parquet")
    ebitda = fin["ebitda"]
    total_mv = finance["total_mv"]
    common = ebitda.index.intersection(total_mv.index)
    ratio = safe_divide(total_mv.loc[common], ebitda.loc[common].abs() + 1e-8)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="pe_to_eps_growth",
    description="PE/ESP增速因子 (PEG代理, 低排前, 负向)。",
    category="valuation",
    thesis="PEG=PE/增长率，低PEG意味着成长被低估。使用dt_eps_yoy作为增长率代理",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_pe_to_eps_growth(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["dt_eps_yoy"])
    finance = context.load("finance.parquet")
    eps_growth = fin["dt_eps_yoy"]
    pe = finance["pe_ttm"]
    common = eps_growth.index.intersection(pe.index)
    growth_pos = eps_growth.loc[common].clip(lower=1.0)
    peg = safe_divide(pe.loc[common], growth_pos)
    return cross_sectional_rank(-peg)


@register_factor(
    name="revenue_to_ev",
    description="营业收入/企业价值因子截面排名。",
    category="valuation",
    thesis="Revenue/EV是一个不受会计政策影响的估值指标，高比率意味着营收能力相对企业价值强",
    dependencies=("income.parquet", "finance.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_revenue_to_ev(context: FactorContext):
    inc = context.load_financial("income.parquet", value_cols=["total_revenue"])
    bs = context.load_financial("balancesheet.parquet", value_cols=["total_liab", "money_cap"])
    finance = context.load("finance.parquet")
    revenue = inc["total_revenue"]
    total_mv = finance["total_mv"]
    total_liab = bs["total_liab"]
    cash = bs["money_cap"]
    common = revenue.index.intersection(total_mv.index).intersection(total_liab.index).intersection(cash.index)
    ev = total_mv.loc[common] + total_liab.loc[common] - cash.loc[common]
    ratio = safe_divide(revenue.loc[common], ev)
    return cross_sectional_rank(ratio)


@register_factor(
    name="net_debt_to_ebitda",
    description="净负债/EBITDA因子 (低排前, 负向)。",
    category="valuation",
    thesis="净负债/EBITDA是杠杆与盈利能力的综合度量，低比率意味着偿债能力强、财务健康",
    dependencies=("financial_indicator.parquet", "balancesheet.parquet", "calendar.parquet"),
)
def factor_net_debt_to_ebitda(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["ebitda"])
    bs = context.load_financial("balancesheet.parquet", value_cols=["total_liab", "money_cap"])
    ebitda = fin["ebitda"]
    total_liab = bs["total_liab"]
    cash = bs["money_cap"]
    common = ebitda.index.intersection(total_liab.index).intersection(cash.index)
    net_debt = total_liab.loc[common] - cash.loc[common]
    ratio = safe_divide(net_debt, ebitda.loc[common].abs() + 1e-8)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="pb_to_roe",
    description="PB/ROE因子截面排名 (低排前, 负向)。",
    category="valuation",
    thesis="PB-ROE框架：高ROE理应高PB，低PB+高ROE=低估。PB/ROE越低越有投资价值",
    dependencies=("financial_indicator.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_pb_to_roe(context: FactorContext):
    fin = context.load_financial("financial_indicator.parquet", value_cols=["roe"])
    finance = context.load("finance.parquet")
    roe = fin["roe"]
    pb = finance["pb"]
    common = roe.index.intersection(pb.index)
    pb_roe = safe_divide(pb.loc[common], roe.loc[common].abs() + 1e-8)
    return cross_sectional_rank(-pb_roe)


# ════════════════════════════════════════════════════════════════════════════
# Phase 2d: Industry-Neutralized Valuation Factors
# 行业中性化估值因子 — 剔除行业估值中枢差异
# ════════════════════════════════════════════════════════════════════════════

from .neutral import _industry_neutral_rank




@register_factor(
    name="ebitda_to_ev_neutral",
    description="行业中性化EBITDA/EV因子。",
    category="valuation",
    thesis="EBITDA/EV剔除资本结构和折旧政策干扰，行业中性化后是更纯粹的行业内相对估值指标。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_ebitda_to_ev_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebitda"]
    )
    finance = context.load("finance.parquet")
    ebitda = fin["ebitda"]
    mv = finance["total_mv"]
    total_liab = context.load_financial(
        "balancesheet.parquet", value_cols=["total_liab"]
    )["total_liab"]
    cash = context.load_financial(
        "balancesheet.parquet", value_cols=["money_cap"]
    )["money_cap"]
    common = ebitda.index.intersection(mv.index).intersection(
        total_liab.index
    ).intersection(cash.index)
    ev = mv.loc[common] + total_liab.loc[common] - cash.loc[common]
    ratio = safe_divide(ebitda.loc[common], ev.abs() + 1e-8)
    neutral = _industry_neutral_rank(ratio, context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="mv_to_ebitda_neutral",
    description="行业中性化市值/EBITDA因子（取负=低估值排前）。",
    category="valuation",
    thesis="EV/EBITDA的简化版(市值/EBITDA)在行业中性化后可作为行业内相对估值指标——低倍数意味着估值修复空间更大。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_mv_to_ebitda_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["ebitda"]
    )
    finance = context.load("finance.parquet")
    ebitda = fin["ebitda"]
    mv = finance["total_mv"]
    common = ebitda.index.intersection(mv.index)
    ratio = safe_divide(mv.loc[common], ebitda.loc[common].abs() + 1e-8)
    neutral = _industry_neutral_rank(-ratio, context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="pe_to_eps_growth_neutral",
    description="行业中性化PEG因子（PE/eps_yoy，取负=低PEG排前）。",
    category="valuation",
    thesis="PEG比率因行业增长中枢差异而不可比，行业中性化后识别同行业内低估值-高增长的个股。",
    dependencies=("financial_indicator.parquet", "finance.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_pe_to_eps_growth_neutral(context: FactorContext):
    fin = context.load_financial(
        "financial_indicator.parquet", value_cols=["basic_eps_yoy"]
    )
    finance = context.load("finance.parquet")
    eps_g = fin["basic_eps_yoy"]
    pe = finance["pe_ttm"]
    common = eps_g.index.intersection(pe.index)
    growth = eps_g.loc[common].clip(lower=1.0)
    peg = safe_divide(pe.loc[common], growth.abs() + 1e-8)
    neutral = _industry_neutral_rank(-peg, context)
    return cross_sectional_rank(neutral)
