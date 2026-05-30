"""Behavioral finance and sentiment factors.

52-week high/low anchoring, MAX effect (lottery preference),
retail attention proxies, turnover anomaly, and dispersion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, rolling_group_max, rolling_group_min


# ── 52-week high / low (anchoring) ──────────────────────────────────────

@register_factor(
    name="price_to_52w_high",
    description="52周高点距离因子，(close-52w_low)/(52w_high-52w_low)截面排名。锚定效应——接近高点预示继续走强。",
    category="price",
    thesis="George&Hwang(2004)发现52周高点距离比传统动量更能预测未来收益。投资者以52周高点为锚：接近高点时视为强势，远离高点时视为弱势。A股中该效应叠加散户追涨行为更为显著。",
    dependencies=("daily_adj.parquet",),
)
def factor_price_to_52w_high(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    high_52w = rolling_group_max(close, 252)
    low_52w = rolling_group_min(close, 252)
    position = (close - low_52w) / (high_52w - low_52w).replace(0, np.nan)
    return cross_sectional_rank(position)


@register_factor(
    name="price_to_52w_high_dist",
    description="52周高点相对距离因子，close/52w_high截面排名。直接度量现价在52周高点的百分比。",
    category="price",
    thesis="close/52w_high<0.7是深度回调信号（可能反弹），>0.95是突破前夜（可能继续突破）。非线性效应——极端值更有预测力。",
    dependencies=("daily_adj.parquet",),
)
def factor_price_to_52w_high_dist(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    high_52w = rolling_group_max(close, 252)
    ratio = close / high_52w.replace(0, np.nan)
    return cross_sectional_rank(ratio)


@register_factor(
    name="price_to_52w_low_dist",
    description="52周低点相对距离因子，close/52w_low - 1截面排名。反弹强度度量。",
    category="price",
    thesis="距52周低点越远说明反弹越强劲。接近52周低点的股票存在'死猫反弹'风险（诱多后继续下跌）。",
    dependencies=("daily_adj.parquet",),
)
def factor_price_to_52w_low_dist(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    low_52w = rolling_group_min(close, 252)
    dist = close / low_52w.replace(0, np.nan) - 1.0
    return cross_sectional_rank(dist)


# ── MAX effect (lottery-type stock preference) ──────────────────────────

@register_factor(
    name="max_ret_20",
    description="MAX效应因子，20日内最大单日收益截面排名（取负向=高MAX排后）。Bali et al.(2011)彩票偏好。",
    category="price",
    thesis="散户投资者偏好彩票型股票（高MAX、高偏度），需求推动价格高估后均值回归。高MAX股票未来收益显著低于低MAX股票。A股散户占比高，MAX效应比美股更强。",
    dependencies=("daily_adj.parquet",),
)
def factor_max_ret_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    max_ret = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    return cross_sectional_rank(-max_ret)


@register_factor(
    name="max_ret_vol_ratio",
    description="极端收益/波动率比因子，max_ret_20/volatility_20截面排名（取负向）。彩票收益的风险调整度量。",
    category="price",
    thesis="高MAX/低波动率组合是最典型的彩票型股票特征——偶尔大幅上涨但整体波动低，吸引散户追入。风险调整后的MAX效应更纯粹。",
    dependencies=("daily_adj.parquet",),
)
def factor_max_ret_vol_ratio(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    max_ret = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    vol_20 = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    ratio = max_ret / vol_20.replace(0, np.nan)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="min_ret_20",
    description="极端负收益因子，20日内最小单日收益截面排名。尾部风险度量，大幅下跌日的预测能力。",
    category="price",
    thesis="大幅下跌日后存在短期反弹（恐慌性超卖）和长期弱势（基本面恶化信号）两种相反力量。A股中'大跌次日反弹'效应在特定条件下有效。",
    dependencies=("daily_adj.parquet",),
)
def factor_min_ret_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    daily_ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    min_ret = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).min()
    )
    return cross_sectional_rank(min_ret)


# ── Retail attention proxy ──────────────────────────────────────────────

@register_factor(
    name="retail_attention",
    description="散户关注度代理因子，异常高换手率(当日换手/20日均换手-1)与大单净流出交乘截面排名（取负向）。",
    category="price",
    thesis="高换手+大单流出=散户接盘信号。机构通过大单出货、散户通过中小单接盘，这种成交量结构预示后续下跌。",
    dependencies=("finance.parquet", "main_fund_flow.parquet"),
)
def factor_retail_attention(context: FactorContext):
    fin = context.load("finance.parquet")
    ff = context.load("main_fund_flow.parquet")
    turnover = fin["turnover_rate"]
    to_20_mean = rolling_group_mean(turnover, 20)
    abnormal_to = turnover / to_20_mean.replace(0, np.nan) - 1.0

    # Total amount and big order net
    total_amount = (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)
    big_net = (ff["buy_lg_amount"] + ff["buy_elg_amount"]
               - ff["sell_lg_amount"] - ff["sell_elg_amount"])
    big_net_ratio = big_net / total_amount

    common = abnormal_to.index.intersection(big_net_ratio.index)
    # Retail attention = high abnormal turnover + big order selling
    signal = abnormal_to.loc[common] * (-big_net_ratio.loc[common])
    return cross_sectional_rank(-signal)


# ── Turnover anomaly ────────────────────────────────────────────────────

@register_factor(
    name="turnover_anomaly_20",
    description="换手率异常因子，20日均换手/60日均换手-1截面排名（取负向=异常高换手排后）。",
    category="price",
    thesis="换手率短期飙升往往伴随投机性交易或信息事件，高异常换手率预示短期反转。中长期低换手率则与低波动溢价相关。",
    dependencies=("finance.parquet",),
)
def factor_turnover_anomaly_20(context: FactorContext):
    fin = context.load("finance.parquet")
    turnover = fin["turnover_rate"]
    to_20 = rolling_group_mean(turnover, 20)
    to_60 = rolling_group_mean(turnover, 60)
    anomaly = to_20 / to_60.replace(0, np.nan) - 1.0
    return cross_sectional_rank(-anomaly)


@register_factor(
    name="turnover_divergence",
    description="量价背离因子，20日换手率变化与20日收益的符号一致性截面排名（背离排后=量价不符）。",
    category="price",
    thesis="放量上涨和缩量下跌是健康的量价关系。放量下跌或缩量上涨（量价背离）预示趋势不可持续。此因子捕捉量价之间的信息一致性。",
    dependencies=("daily_adj.parquet", "finance.parquet"),
)
def factor_turnover_divergence(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    fin = context.load("finance.parquet")
    close = daily_adj["close"]
    ret_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    turnover = fin["turnover_rate"]
    to_chg = turnover.groupby(level="Code").transform(lambda s: s / s.shift(20).replace(0, np.nan) - 1.0)

    common = ret_20.index.intersection(to_chg.index)
    # Divergence = sign difference between return and turnover change
    # Positive: return and turnover agree (healthy). Negative: they diverge.
    agreement = np.sign(ret_20.loc[common]) * np.sign(to_chg.loc[common])
    return cross_sectional_rank(agreement)


# ── Disposition effect proxy ────────────────────────────────────────────

@register_factor(
    name="disposition_effect",
    description="处置效应代理因子，winner_rate(90%+)且价格接近52周高点时排名靠后（获利了结压力）。",
    category="price",
    thesis="A股散户普遍存在'出盈保亏'的处置效应。当获利盘比例极高(>90%)且价格在52周高点附近时，获利了结压力最大，短期反转概率高。",
    dependencies=("daily_adj.parquet", "cyq_perf.parquet"),
)
def factor_disposition_effect(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    perf = context.load("cyq_perf.parquet")
    close = daily_adj["close"]
    high_52w = rolling_group_max(close, 252)
    near_high = close / high_52w.replace(0, np.nan)

    winner_rate = perf["winner_rate"]
    common = near_high.index.intersection(winner_rate.index)
    near_high = near_high.loc[common]
    winner_rate = winner_rate.loc[common]

    # Disposition pressure = high winner rate + near 52-week high
    pressure = winner_rate * near_high
    return cross_sectional_rank(-pressure)


# ── Anchoring bias ─────────────────────────────────────────────────────────

@register_factor(
    name="anchoring_to_52w_high",
    description="52周高点锚定因子，(close-52周最高)/52周最高截面排名。",
    category="price",
    thesis="投资者锚定52周高点是行为金融学最稳定的效应之一——接近52周高点具有动量效应(突破锚定)，远离52周高点具有反转效应(均值回归锚定)。该因子量化距离52周高点的程度。",
    dependencies=("daily_adj.parquet",),
)
def factor_anchoring_to_52w_high(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    high_52w = close.groupby(level="Code").transform(lambda s: s.rolling(252, min_periods=126).max())
    distance = (close - high_52w) / high_52w.replace(0, np.nan)
    return cross_sectional_rank(distance)


@register_factor(
    name="anchoring_to_52w_low",
    description="52周低点锚定因子，(52周最低-close)/close截面排名（接近低点=恐慌超卖排前）。",
    category="price",
    thesis="接近52周低点引发投资者的损失厌恶和恐慌——极端接近历史低点时往往出现超卖，随后有均值回复。",
    dependencies=("daily_adj.parquet",),
)
def factor_anchoring_to_52w_low(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    low_52w = close.groupby(level="Code").transform(lambda s: s.rolling(252, min_periods=126).min())
    distance = (close - low_52w) / low_52w.replace(0, np.nan)
    return cross_sectional_rank(distance)


@register_factor(
    name="disposition_effect_proxy",
    description="处置效应代理因子，-(winner_rate×avg_return_20)截面排名（获利+近期上涨=卖出倾向排后）。",
    category="price",
    thesis="处置效应——投资者倾向于过早卖出盈利股票、过久持有亏损股票。高获利+近期上涨的股票面临更多获利了结压力(反转信号)，该因子捕捉这种卖出倾向的强度。",
    dependencies=("daily_adj.parquet", "cyq_perf.parquet"),
)
def factor_disposition_effect_proxy(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    cyq = context.load("cyq_perf.parquet")
    close = daily_adj["close"]
    ret_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    wr = cyq["winner_rate"]
    common = ret_20.index.intersection(wr.index)
    disposition = wr.loc[common] * ret_20.loc[common].clip(lower=0)
    return cross_sectional_rank(-disposition)


@register_factor(
    name="lottery_stock_indicator",
    description="彩票股指标因子，-(max_return_20×idiosyncratic_vol)截面排名（彩票特征越强排后）。",
    category="price",
    thesis="Bali et al.(2011)发现具有彩票特征(高最大日收益+高异质波动)的股票未来收益显著偏低——投资者偏好彩票型股票支付了过高的价格。",
    dependencies=("daily_adj.parquet",),
)
def factor_lottery_stock_indicator(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    max_ret_20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    # idiosyncratic vol approximation = daily return std
    idio_vol = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    lottery = max_ret_20 * idio_vol
    return cross_sectional_rank(-lottery)


@register_factor(
    name="overreaction_indicator_5d",
    description="过度反应指标因子，-(5日收益率的绝对值)截面排名（过度反应=反转可能大排后）[负向:过度上涨=抛压]。",
    category="price",
    thesis="过度反应假说(De Bondt & Thaler, 1985)——5日内大幅涨跌的股票在后续时期存在显著反转。该因子捕捉短期过度反应的程度，为逆向投资提供信号。",
    dependencies=("daily_adj.parquet",),
)
def factor_overreaction_indicator_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret_5 = close.groupby(level="Code").transform(lambda s: s.pct_change(5))
    # Overreaction = extreme absolute return, expect reversal
    return cross_sectional_rank(ret_5)


@register_factor(
    name="attention_proxy_20d",
    description="投资者关注度代理因子，20日极端日(涨跌幅>5%)天数截面排名。",
    category="price",
    thesis="极端涨跌停事件吸引投资者的有限注意力——关注度上升导致短期买入压力和随后的反转。该因子是Barber & Odean(2008)注意力驱动交易假设的量化。",
    dependencies=("daily_adj.parquet",),
)
def factor_attention_proxy_20d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    extreme = (ret.abs() > 0.05).astype(float)
    attention = extreme.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    return cross_sectional_rank(attention)


@register_factor(
    name="herding_intensity",
    description="羊群效应强度因子，-(|ret_i-ret_market|的20日平均)截面排名（偏离市场越小=跟风越强排后）。",
    category="price",
    thesis="个股与市场平均收益的偏离度反映个股独立行情的程度——偏离度极低意味着股票随大流（羊群行为）、缺乏独立alpha；适度偏离意味着有特异性信息驱动。",
    dependencies=("daily_adj.parquet",),
)
def factor_herding_intensity(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    market_ret = ret.groupby(level="Date").transform("mean")
    deviation = (ret - market_ret).abs()
    avg_dev_20 = deviation.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    # Moderate deviation is positive (independent alpha); extreme herding (low deviation) is negative
    return cross_sectional_rank(avg_dev_20)


@register_factor(
    name="turnover_anomaly_mean_20d",
    description="换手率异象因子，-(近20日平均换手率)截面排名（高换手=投机性强排后）。",
    category="price",
    thesis="A股换手率异象是全球最显著的——高换手率股票后续收益显著偏低，原因在于散户过度交易和投机炒作后的均值回复。",
    dependencies=("finance.parquet",),
)
def factor_turnover_anomaly_mean_20d(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    to_20 = turnover.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-to_20)
