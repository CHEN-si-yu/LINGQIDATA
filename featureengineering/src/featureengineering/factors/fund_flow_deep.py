"""
Deep fund flow factors (主力资金流深度因子) — Class 1.

基于 main_fund_flow.parquet 的四档订单规模数据（小单/中单/大单/特大单），
构建机构行为识别和资金流质量评估因子。

数据源: main_fund_flow.parquet
字段: buy_sm/md/lg/elg_vol/amount, sell_sm/md/lg/elg_vol/amount, net_mf_vol/amount
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, safe_divide


def _total_amount(ff):
    """Return total turnover amount from main fund flow."""
    return (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)


def _big_net(ff):
    """大单+特大单净买入额。"""
    return (
        ff["buy_lg_amount"] - ff["sell_lg_amount"]
        + ff["buy_elg_amount"] - ff["sell_elg_amount"]
    )


def _small_net(ff):
    """小单净买入额（散户方向）。"""
    return ff["buy_sm_amount"] - ff["sell_sm_amount"]


# ── 订单规模分析 ─────────────────────────────────────────────────────────

@register_factor(
    name="lg_order_imbalance",
    description="超大单不平衡因子，(特大买-特大卖)/(特大买+特大卖)截面排名（特大单净买=机构抢筹排前）。",
    category="fund_flow",
    thesis="特大单（>100万元/笔）几乎完全代表机构行为——特大单净买入占比高说明机构在主动收集筹码，方向信号极其可靠。特大单的方向性比大单更纯粹（排除了游资干扰）。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_lg_order_imbalance(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    elg_buy = ff["buy_elg_amount"]
    elg_sell = ff["sell_elg_amount"]
    imbalance = (elg_buy - elg_sell) / (elg_buy + elg_sell).replace(0, np.nan)
    return cross_sectional_rank(imbalance)


@register_factor(
    name="lg_sm_divergence",
    description="大小单背离因子，(大单净买-小单净卖)/总成交额截面排名（机构买+散户卖=最佳组合排前）。",
    category="fund_flow",
    thesis="大单净买入同时小单净卖出是'聪明钱在吸、散户在抛'的背离信号——这种组合表明筹码正在从弱手（散户）转移到强手（机构），是经典的底部吸筹特征。反向（大单卖+小单买）则是顶部出货信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_lg_sm_divergence(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    big = _big_net(ff)
    small = _small_net(ff)
    total = _total_amount(ff)
    divergence = (big - small) / total
    return cross_sectional_rank(divergence)


@register_factor(
    name="mf_net_persistent_5d",
    description="主力净流入持续性因子，5日主力净流入为正的天数截面排名（持续净流入=坚定看多排前）。",
    category="fund_flow",
    thesis="主力资金的持续性比单日力度更重要——连续5天净流入说明机构在系统性建仓而非短线博弈。持续净流入的股票中期趋势延续概率显著高于脉冲式流入的股票。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_persistent_5d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    net = ff["net_mf_amount"]
    is_positive = (net > 0).astype(float)
    persist = is_positive.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    return cross_sectional_rank(persist)


@register_factor(
    name="mf_flow_volatility_20d",
    description="资金流波动率因子，-(主力净流入20日标准差)截面排名（资金流稳定=有序建仓排前）。",
    category="fund_flow",
    thesis="主力资金流的波动率区分'有序建仓'和'游资短炒'——机构建仓通常表现为持续、稳定的净流入（低波动），而游资操作则呈现大进大出（高波动）。低波动+正流入是最优信号。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_flow_volatility_20d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    net = ff["net_mf_amount"]
    total = _total_amount(ff)
    net_ratio = net / total
    vol = net_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-vol)


@register_factor(
    name="order_size_ratio_change",
    description="订单规模比变化因子，(大单+特大)/总成交的5日变化截面排名（大单占比提升=机构参与加深排前）。",
    category="fund_flow",
    thesis="大单占比的边际变化比绝对水平更有信息量——大单占比从10%升到20%意味着机构刚开始介入（最佳买点），而占比已经在40%高位意味着机构可能已经开始出货。二阶变化捕捉拐点。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_order_size_ratio_change(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    big_total = (
        ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    )
    total = _total_amount(ff)
    big_ratio = big_total / total
    chg = big_ratio.groupby(level="Code").transform(
        lambda s: s.diff(5)
    )
    return cross_sectional_rank(chg)


@register_factor(
    name="mf_price_divergence",
    description="资金流-价格背离因子，(主力净流入排名-涨跌幅排名)截面排名（流入但不涨=压盘吸筹排前）。",
    category="fund_flow",
    thesis="主力资金持续流入但股价不涨甚至下跌是'压盘吸筹'的典型特征——主力通过大单拆小、限价挂单等方式隐藏买入意图，在低位默默收集筹码。这种背离积累到一定程度后通常伴随爆发性上涨。",
    dependencies=("main_fund_flow.parquet", "daily_adj.parquet"),
)
def factor_mf_price_divergence(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    daily = context.load("daily_adj.parquet")

    net = ff["net_mf_amount"]
    total = _total_amount(ff)
    net_ratio = net / total

    close = daily["close"]
    ret_5d = close.groupby(level="Code").transform(
        lambda s: s.pct_change(5)
    )

    common = net_ratio.index.intersection(ret_5d.index)
    net_rank = net_ratio.loc[common].groupby(level="Date").rank(pct=True)
    ret_rank = ret_5d.loc[common].groupby(level="Date").rank(pct=True)

    divergence = net_rank - ret_rank
    return cross_sectional_rank(divergence)


@register_factor(
    name="mf_sector_relative",
    description="行业相对资金流因子，个股主力净流入率-行业均值截面排名（行业内资金吸引力排前）。",
    category="fund_flow",
    thesis="行业内相对资金流捕捉的是'板块内轮动选股'——即使整个板块资金在流出，相对流出更少（或流入更多）的个股说明其在板块内获得资金偏好，是板块轮动策略的核心指标。",
    dependencies=("main_fund_flow.parquet", "stock_list.parquet"),
)
def factor_mf_sector_relative(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    industry_map = context.repo.load_industry_map()

    net = ff["net_mf_amount"]
    total = _total_amount(ff)
    net_ratio = net / total

    codes = net_ratio.index.get_level_values("Code")
    industries = codes.map(industry_map)

    df = net_ratio.to_frame("net_ratio")
    df["industry"] = industries.values
    df = df.dropna(subset=["industry"])

    # Subtract industry mean
    industry_mean = df.groupby(["Date", "industry"])["net_ratio"].transform("mean")
    relative = df["net_ratio"] - industry_mean

    return cross_sectional_rank(relative)
