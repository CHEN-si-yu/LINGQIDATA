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
    name="smart_money_concentration",
    description="聪明钱集中度因子，(特大单净买-小单净卖)/abs(all net)截面排名（聪明钱相对噪音交易者越集中排前）。",
    category="fund_flow",
    thesis="将订单按规模分为聪明钱(特大+大单)和噪音(小+中单)——"
           "聪明钱净买入远大于噪音交易者净买入时=机构主导定价权、方向可靠；"
           "噪音交易者主导时=散户情绪驱动、方向不确定。"
           "该比率度量的是谁的边际定价权更强。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_smart_money_concentration(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    smart = (
        ff["buy_elg_amount"] - ff["sell_elg_amount"]
        + ff["buy_lg_amount"] - ff["sell_lg_amount"]
    )
    noise = (
        ff["buy_sm_amount"] - ff["sell_sm_amount"]
        + ff["buy_md_amount"] - ff["sell_md_amount"]
    )
    total = _total_amount(ff)
    # Smart money net minus noise net, scaled by total turnover
    score = (smart - noise) / total
    return cross_sectional_rank(score)

@register_factor(
    name="order_size_concentration",
    description="订单规模集中度因子，四个规模档的成交额HHI截面排名（集中度高=机构交易主导排前）。",
    category="fund_flow",
    thesis="四个订单规模档位(sm/md/lg/elg)的成交额赫芬达尔指数(HHI)——"
           "HHI高=交易集中在某个规模档(通常是特大单或小单)=交易者类型单一、方向明确；"
           "HHI低=交易分散在四个档=多空分歧大、方向不明。"
           "特大单主导的高HHI=机构行动一致(排前)；小单主导的高HHI=散户情绪集中(排后)。"
           "结合方向判断后区分两种高HHI情形。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_order_size_concentration(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    sm_total = ff["buy_sm_amount"] + ff["sell_sm_amount"]
    md_total = ff["buy_md_amount"] + ff["sell_md_amount"]
    lg_total = ff["buy_lg_amount"] + ff["sell_lg_amount"]
    elg_total = ff["buy_elg_amount"] + ff["sell_elg_amount"]
    total_amount = sm_total + md_total + lg_total + elg_total

    # HHI = sum of squared market shares
    hhi = (
        (sm_total / total_amount) ** 2
        + (md_total / total_amount) ** 2
        + (lg_total / total_amount) ** 2
        + (elg_total / total_amount) ** 2
    )
    # Direction sign: large+elg net direction
    big_net = (lg_total - 2 * (ff["sell_lg_amount"] - ff["buy_lg_amount"]).abs() / total_amount)  # proxy
    # Simplified: if ELG+LG net is positive, HHI is positive; if negative, HHI is negative
    smart_direction = (
        ff["buy_elg_amount"] - ff["sell_elg_amount"]
        + ff["buy_lg_amount"] - ff["sell_lg_amount"]
    )
    signed_hhi = hhi * np.sign(smart_direction)
    return cross_sectional_rank(signed_hhi)

@register_factor(
    name="mf_flow_acceleration_ext",
    description="资金流加速度因子，主力净流入率的5日变化截面排名（流入在加速=趋势加强排前）。",
    category="fund_flow",
    thesis="资金流的二阶导(加速度)比一阶导(方向)更具前瞻性——"
           "净流入从正到更正向=买盘在加速(最强信号)；"
           "净流入从负到正(转正)=趋势可能反转(注意跟进)；"
           "净流入从正到负(减弱)=主力在撤退(预警信号)。"
           "加速度为正意味着资金的边际态度在改善。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_flow_acceleration_ext(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    net = ff["net_mf_amount"]
    total = _total_amount(ff)
    net_ratio = net / total
    # 5-day change in net_ratio (acceleration)
    accel = net_ratio.groupby(level="Code").transform(
        lambda s: s.diff(5)
    )
    return cross_sectional_rank(accel)

