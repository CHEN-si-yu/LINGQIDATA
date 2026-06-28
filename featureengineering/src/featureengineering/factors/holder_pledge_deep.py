"""
Extended holder & pledge factors (扩展股东/质押因子) — Class 1.

Deep analysis of shareholder structure changes and pledge risks: concentration
dynamics, retail/institutional ratio proxies, pledge coverage and liquidation
risk, governance scoring, insider confidence signals.

Data sources (periodic, forward-filled to daily):
- ``holder_number.parquet`` (quarterly-ish, 2019-01-03 onward)
- ``pledge_stat.parquet`` (quarterly-ish, 2019-04-30 onward)
- ``daily_adj.parquet``, ``finance.parquet``, ``financial_indicator.parquet``
- ``calendar.parquet``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    safe_divide,
)

_PLEDGE_LAG = 45  # trading days (~2 calendar months) to approximate quarterly reporting lag

def _load_pledge(context, value_cols):
    """Load pledge_stat.parquet with reporting lag applied.

    pledge_stat.parquet only has end_date (period end), not ann_date (announcement).
    Quarterly reports are not publicly available until weeks after period end.
    We shift by _PLEDGE_LAG trading days and re-forward-fill to prevent future data leakage.
    """
    df = context.load_financial("pledge_stat.parquet", value_cols=value_cols, date_col="end_date")
    return df.groupby(level="Code").shift(_PLEDGE_LAG).groupby(level="Code").ffill()

# ═══════════════════════════════════════════════════════════════════════════════
# A — Holder Concentration Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="holder_concentration_speed",
    description="股东集中速度因子，-(股东数季度环比变化率)截面排名（股东数下降=筹码集中排前）。",
    category="financial",
    thesis="股东人数下降是筹码集中的最经典指标——股东数减少意味着股票正在从散户手中转移到少数人手中，是主力吸筹的直接证据。下降速度越快，吸筹力度越大。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_concentration_speed(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    holder_num = holder["holder_num"]

    # Quarterly change rate
    chg = holder_num.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # Negate: decreasing holder count = good concentration
    return cross_sectional_rank(-chg)

@register_factor(
    name="holder_dispersion_index",
    description="股东分散度因子，1/股东人数截面排名（取负向=高分散排后=低集中度）。",
    category="financial",
    thesis="股东数的倒数是最简单的持股集中度代理——股东数越少，平均每人持股越多，筹码越集中。A股历史上股东集中度与未来收益呈正向关系，'股东数减少+股价上涨'是最强的中线信号。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_dispersion_index(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    holder_num = holder["holder_num"]

    dispersion = 1.0 / holder_num.replace(0, np.nan)

    return cross_sectional_rank(-dispersion)

@register_factor(
    name="holder_change_volatility",
    description="股东变化波动率因子，股东数季度环比变化的4季度标准差截面排名（取负向=剧烈变动排后）。",
    category="financial",
    thesis="股东数变化的稳定性本身包含信息——剧烈波动（忽增忽减）意味着筹码在快速换手，可能是主力在对倒或派发。平稳下降是最健康的吸筹模式。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_change_volatility(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    holder_num = holder["holder_num"]
    chg = holder_num.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol = rolling_group_std(chg, 4)

    return cross_sectional_rank(-vol)

@register_factor(
    name="holder_seasonal_pattern",
    description="股东季节性因子，Q1股东变化-Q3股东变化（年报集中披露效应）截面排名。",
    category="financial",
    thesis="股东数的变化具有明显的季节性——年报/一季报披露期（Q1）股东数通常大幅变动，因为机构调仓集中在这个窗口。Q1股东集中（减少）的信号含金量高于其他季度。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_seasonal_pattern(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    holder_num = holder["holder_num"]
    chg = holder_num.groupby(level="Code").transform(lambda s: s.pct_change(1))

    # Flag Q1 months (end_date March = Q1)
    dates = chg.index.get_level_values("Date")
    months = dates.str[4:6]
    is_q1 = months.isin(["03", "04"]).astype(float)

    seasonal = chg * is_q1

    return cross_sectional_rank(-seasonal)

# ═══════════════════════════════════════════════════════════════════════════════
# B — Retail/Institutional Ratio Proxies
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="holder_retail_inst_ratio",
    description="散户机构比代理因子，-(股东数×人均持股市值)截面排名（取负向=散户多排后）。",
    category="financial",
    thesis="在无法直接获得机构持股数据时，股东数×人均持股市值是散户化程度的有效代理——股东多+人均持股少=散户为主。A股历史上散户占比高的股票长期收益系统性偏低，因为散户主导的定价噪音大。",
    dependencies=("holder_number.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_holder_retail_inst_ratio(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    finance = context.load("finance.parquet")

    holder_num = holder["holder_num"]
    circ_mv = finance["circ_mv"]

    common = holder_num.index.intersection(circ_mv.index)
    holder_aligned = holder_num.loc[common]
    mv_aligned = circ_mv.loc[common]

    # avg holding value per holder
    avg_holding = mv_aligned / holder_aligned.replace(0, np.nan)

    # Retail proxy: low avg holding = many retail investors
    return cross_sectional_rank(avg_holding)

@register_factor(
    name="holder_smart_money_proxy",
    description="股东吸筹信号因子，股东数下降+股价上涨截面排名（筹码集中+价格确认=吸筹排前）。",
    category="financial",
    thesis="股东减少+股价上涨是经典的主力吸筹模式——散户在卖出（股东减少）而价格在上涨（主力在买）。这是'股东-价格'双确认信号，比单独股东减少的信号更可靠。",
    dependencies=("holder_number.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_holder_smart_money_proxy(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    daily_adj = context.load("daily_adj.parquet")

    holder_num = holder["holder_num"]
    holder_chg = -holder_num.groupby(level="Code").transform(lambda s: s.pct_change(1))

    close = daily_adj["close"]
    ret_60d = close.groupby(level="Code").transform(lambda s: s.pct_change(60))

    common = holder_chg.index.intersection(ret_60d.index)
    chg_aligned = holder_chg.loc[common]
    ret_aligned = ret_60d.loc[common]

    chg_rank = chg_aligned.groupby(level="Date").rank(pct=True)
    ret_rank = ret_aligned.groupby(level="Date").rank(pct=True)

    smart = chg_rank * ret_rank
    return cross_sectional_rank(smart)

@register_factor(
    name="holder_distribution_signal",
    description="股东派发信号因子，-(股东数增加+股价横盘或下跌)截面排名（派发=主力出货排后）。",
    category="financial",
    thesis="股东增加+股价横盘是'派发'的典型特征——主力在高位把筹码分散卖给散户，股价维持不跌以吸引更多跟风。股东持续增加而股价不动是危险信号，说明买方力量在消耗。",
    dependencies=("holder_number.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_holder_distribution_signal(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    daily_adj = context.load("daily_adj.parquet")

    holder_num = holder["holder_num"]
    holder_chg = holder_num.groupby(level="Code").transform(lambda s: s.pct_change(1))

    close = daily_adj["close"]
    ret_60d = close.groupby(level="Code").transform(lambda s: s.pct_change(60))

    common = holder_chg.index.intersection(ret_60d.index)
    chg_aligned = holder_chg.loc[common]
    ret_aligned = ret_60d.loc[common]

    # Distribution: increasing holders + flat/down price
    holder_inc_rank = chg_aligned.groupby(level="Date").rank(pct=True)
    price_weak_rank = (-ret_aligned).groupby(level="Date").rank(pct=True)

    distribution = holder_inc_rank * price_weak_rank
    return cross_sectional_rank(-distribution)

# ═══════════════════════════════════════════════════════════════════════════════
# C — Holder-Price Divergence & Confirmation
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="holder_price_divergence_4q",
    description="股东-价格背离因子，4季度累计股东变化方向-4季度累计价格变化方向截面排名（背离=价格未反映筹码变化排前）。",
    category="financial",
    thesis="股东和价格的4季度背离包含了中期趋势反转的信号——股东在减少(筹码集中)但价格还没涨，是'潜伏期'的典型特征。4季度窗口覆盖了完整的年报周期，过滤了季度性噪音。",
    dependencies=("holder_number.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_holder_price_divergence_4q(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    daily_adj = context.load("daily_adj.parquet")

    holder_num = holder["holder_num"]
    # 4-quarter cumulative change
    holder_cum = holder_num.groupby(level="Code").transform(lambda s: s.pct_change(4))

    close = daily_adj["close"]
    # Approx 4-quarter cumulative return (~240 trading days)
    price_cum = close.groupby(level="Code").transform(lambda s: s.pct_change(240))

    common = holder_cum.index.intersection(price_cum.index)
    h_aligned = holder_cum.loc[common]
    p_aligned = price_cum.loc[common]

    # Divergence: holder concentration happening (-h_chg) but price not yet up
    h_rank = (-h_aligned).groupby(level="Date").rank(pct=True)
    p_rank = (1 - p_aligned.groupby(level="Date").rank(pct=True))

    divergence = h_rank * p_rank
    return cross_sectional_rank(divergence)

@register_factor(
    name="holder_momentum_confirmation",
    description="股东动量确认因子，价格动量+股东集中信号截面排名（趋势获筹码确认排前）。",
    category="financial",
    thesis="价格动量被股东集中所确认是'趋势+筹码'双共振——动量股如果同时有股东集中的支撑，说明上涨不是泡沫而是有资金在持续收集筹码。这种双确认动量比单纯动量更可靠。",
    dependencies=("holder_number.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_holder_momentum_confirmation(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    daily_adj = context.load("daily_adj.parquet")

    holder_num = holder["holder_num"]
    holder_chg = -holder_num.groupby(level="Code").transform(lambda s: s.pct_change(1))

    close = daily_adj["close"]
    ret_60d = close.groupby(level="Code").transform(lambda s: s.pct_change(60))

    common = holder_chg.index.intersection(ret_60d.index)
    h_aligned = holder_chg.loc[common]
    p_aligned = ret_60d.loc[common]

    h_rank = h_aligned.groupby(level="Date").rank(pct=True)
    p_rank = p_aligned.groupby(level="Date").rank(pct=True)

    confirmation = h_rank * p_rank
    return cross_sectional_rank(confirmation)

# ═══════════════════════════════════════════════════════════════════════════════
# D — Pledge Risk Factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="pledge_liquidation_risk",
    description="质押平仓风险因子，-(估算平仓线/现价-1)截面排名（高比值=距平仓线近=高风险排后）。",
    category="financial",
    thesis="股价距估算平仓线的距离是质押风险最直接的度量——距离越近，继续下跌触发平仓的概率越高，连锁抛售的尾部风险越大。通常估算平仓线在质押时股价的130%-150%。",
    dependencies=("pledge_stat.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_pledge_liquidation_risk(context: FactorContext):
    pledge = _load_pledge(context, ["pledge_ratio"])
    daily_adj = context.load("daily_adj.parquet")

    pledge_ratio = pledge["pledge_ratio"]
    close = daily_adj["close"]

    common = pledge_ratio.index.intersection(close.index)
    pledge_aligned = pledge_ratio.loc[common]
    close_aligned = close.loc[common]

    # Simplified liquidation risk: higher pledge + lower price (relative to recent high)
    high_60d = close.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).max())
    drawdown = close / high_60d.replace(0, np.nan) - 1.0

    common2 = pledge_aligned.index.intersection(drawdown.index)
    p = pledge_aligned.loc[common2]
    d = drawdown.loc[common2]

    # Risk = pledge ratio × drawdown (more pledged + more drawn down = higher risk)
    risk = p * (-d).clip(lower=0)

    return cross_sectional_rank(-risk)

@register_factor(
    name="pledge_market_impact",
    description="质押市场冲击因子，-(质押率×60日波动率)截面排名（高质押+高波动=尾部风险排后）。",
    category="financial",
    thesis="质押率与波动率的乘积是质押尾部风险的量化——高质押+高波动的组合是最危险的：股价剧烈波动时容易触发质押平仓线，而平仓又会进一步加剧波动，形成恶性循环。",
    dependencies=("pledge_stat.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_pledge_market_impact(context: FactorContext):
    pledge = _load_pledge(context, ["pledge_ratio"])
    daily_adj = context.load("daily_adj.parquet")

    pledge_ratio = pledge["pledge_ratio"]
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_60 = rolling_group_std(ret, 60)

    common = pledge_ratio.index.intersection(vol_60.index)
    p_aligned = pledge_ratio.loc[common]
    v_aligned = vol_60.loc[common]

    impact = p_aligned * v_aligned
    return cross_sectional_rank(-impact)

@register_factor(
    name="pledge_change_signal",
    description="质押率变动信号因子，-(质押率季度同比变化>20%)截面排名（大幅变动=不确定性排后）。",
    category="financial",
    thesis="质押率的大幅变动（无论方向）都是治理风险的信号——大幅增加可能意味着大股东急需资金（流动性危机），大幅减少可能是质权人减持（被迫或主动）。质押率的稳定才是健康状态。",
    dependencies=("pledge_stat.parquet", "calendar.parquet"),
)
def factor_pledge_change_signal(context: FactorContext):
    pledge = _load_pledge(context, ["pledge_ratio"])
    pledge_ratio = pledge["pledge_ratio"]

    # Year-over-year change (4 quarters)
    chg = pledge_ratio.groupby(level="Code").transform(lambda s: s.diff(4))

    # Large change (>20%) is bad regardless of direction
    large_change = (chg.abs() > 20).astype(float)
    signal = large_change.groupby(level="Code").transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )

    return cross_sectional_rank(-signal)

@register_factor(
    name="pledge_industry_comparison",
    description="质押率行业比较因子，-(个股质押率/行业平均质押率)截面排名（取负向=高相对质押排后）。",
    category="financial",
    thesis="质押率在行业内的相对位置比绝对质押率更有信息量——同一行业内，质押率显著高于同行的公司通常面临更大的财务压力或治理问题。行业中性化后的质押率是更纯粹的治理风险指标。",
    dependencies=("pledge_stat.parquet", "stock_list.parquet", "calendar.parquet"),
)
def factor_pledge_industry_comparison(context: FactorContext):
    pledge = _load_pledge(context, ["pledge_ratio"])
    industry_map = context.repo.load_industry_map()

    pledge_ratio = pledge["pledge_ratio"]
    codes = pledge_ratio.index.get_level_values("Code")
    industries = codes.map(industry_map)

    df = pd.DataFrame({"pledge": pledge_ratio.values, "industry": industries.values},
                      index=pledge_ratio.index)
    df = df.dropna(subset=["industry"])
    df["industry_avg"] = df.groupby(["Date", "industry"])["pledge"].transform("mean")
    df["relative_pledge"] = df["pledge"] / df["industry_avg"].replace(0, np.nan)

    return cross_sectional_rank(-df["relative_pledge"])

# ═══════════════════════════════════════════════════════════════════════════════
# E — Governance & Composite Risk
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="governance_risk_composite",
    description="治理风险综合因子，-(质押率排名+股东分散度排名+质押变动排名)/3截面排名（治理差=风险排后）。",
    category="financial",
    thesis="公司治理风险需要多维度综合评估——高质押+高分散+质押大幅变动的公司存在多维治理缺陷。治理风险综合评分是从'避险'角度筛选标的的有力工具，治理差的公司在黑天鹅事件中跌幅最大。",
    dependencies=("holder_number.parquet", "pledge_stat.parquet", "calendar.parquet"),
)
def factor_governance_risk_composite(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    pledge = _load_pledge(context, ["pledge_ratio"])

    holder_num = holder["holder_num"]
    pledge_ratio = pledge["pledge_ratio"]

    # Holder dispersion (higher = worse)
    dispersion = 1.0 / holder_num.replace(0, np.nan)

    # Pledge level (higher = worse)
    pledge_chg = pledge_ratio.groupby(level="Code").transform(lambda s: s.diff(4))

    # Combine: higher = worse governance
    common = dispersion.index.intersection(pledge_ratio.index).intersection(pledge_chg.index)
    d_rank = dispersion.loc[common].groupby(level="Date").rank(pct=True)
    p_rank = pledge_ratio.loc[common].groupby(level="Date").rank(pct=True)
    pc_rank = pledge_chg.loc[common].abs().groupby(level="Date").rank(pct=True)

    risk = (d_rank + p_rank + pc_rank) / 3.0
    return cross_sectional_rank(-risk)

@register_factor(
    name="insider_confidence_proxy",
    description="内部人信心代理因子，-(低质押率排名+高股东集中度排名)/2截面排名（低质押+集中=内部信心足排前）。",
    category="financial",
    thesis="低质押率+高股东集中度是内部人对公司前景有信心的间接证据——大股东不缺钱(不需要质押)+筹码集中(不急于减持)。这种组合在A股中往往对应着优质公司。",
    dependencies=("holder_number.parquet", "pledge_stat.parquet", "calendar.parquet"),
)
def factor_insider_confidence_proxy(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    pledge = _load_pledge(context, ["pledge_ratio"])

    holder_num = holder["holder_num"]
    pledge_ratio = pledge["pledge_ratio"]

    concentration = 1.0 / holder_num.replace(0, np.nan)

    common = concentration.index.intersection(pledge_ratio.index)
    conc_aligned = concentration.loc[common]
    pledge_aligned = pledge_ratio.loc[common]

    conc_rank = conc_aligned.groupby(level="Date").rank(pct=True)
    low_pledge_rank = (-pledge_aligned).groupby(level="Date").rank(pct=True)

    confidence = (conc_rank + low_pledge_rank) / 2.0
    return cross_sectional_rank(confidence)

@register_factor(
    name="corporate_governance_quality",
    description="公司治理质量因子，-(质押率+股东分散度+质押变动)截面排名（取负=低质量排后）。",
    category="financial",
    thesis="公司治理质量是一个多维度的概念——需要从质押风险、股东结构稳定性、内部人行为等多个角度综合衡量。治理质量因子是对冲'治理风险'的alpha来源，治理好的公司长期表现更优。",
    dependencies=("holder_number.parquet", "pledge_stat.parquet", "calendar.parquet"),
)
def factor_corporate_governance_quality(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    pledge = _load_pledge(context, ["pledge_ratio"])

    holder_num = holder["holder_num"]
    pledge_ratio = pledge["pledge_ratio"]

    holder_chg = holder_num.groupby(level="Code").transform(lambda s: s.pct_change(1))
    pledge_chg = pledge_ratio.groupby(level="Code").transform(lambda s: s.diff(4))

    dispersion = 1.0 / holder_num.replace(0, np.nan)

    common = dispersion.index.intersection(pledge_ratio.index).intersection(holder_chg.index).intersection(pledge_chg.index)

    # Components (lower = better governance)
    d_rank = dispersion.loc[common].groupby(level="Date").rank(pct=True)  # high dispersion = bad
    p_rank = pledge_ratio.loc[common].groupby(level="Date").rank(pct=True)  # high pledge = bad
    hc_rank = holder_chg.loc[common].groupby(level="Date").rank(pct=True)  # increasing = bad
    pc_rank = pledge_chg.loc[common].abs().groupby(level="Date").rank(pct=True)  # large change = bad

    quality = -(d_rank + p_rank + hc_rank + pc_rank) / 4.0
    return cross_sectional_rank(quality)

@register_factor(
    name="pledge_pressure_index",
    description="质押压力指数因子，-(质押率×max(0, 距高点跌幅-20%)×波动率)截面排名（三高=极端压力排后）。",
    category="financial",
    thesis="质押压力指数综合了质押水平、价格跌幅和波动率三个维度——同时处于高质押+大幅下跌+高波动状态的股票面临最严峻的质押平仓危机。这是A股中预测'闪崩'的重要先行指标。",
    dependencies=("pledge_stat.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_pledge_pressure_index(context: FactorContext):
    pledge = _load_pledge(context, ["pledge_ratio"])
    daily_adj = context.load("daily_adj.parquet")

    pledge_ratio = pledge["pledge_ratio"]
    close = daily_adj["close"]

    # Distance from 60-day high
    high_60 = close.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).max())
    drawdown = (1.0 - close / high_60.replace(0, np.nan)).clip(lower=0)

    # Volatility
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol = rolling_group_std(ret, 20)

    common = pledge_ratio.index.intersection(drawdown.index).intersection(vol.index)
    p = pledge_ratio.loc[common]
    d = drawdown.loc[common]
    v = vol.loc[common]

    # Pressure = pledge × max(0, drawdown - 20%) × volatility
    pressure = p * (d - 0.2).clip(lower=0) * v

    return cross_sectional_rank(-pressure)

@register_factor(
    name="holder_concentration_momentum",
    description="股东集中度动量因子，股东集中度（1/股东数）的4季度变化截面排名（集中加速排前）。",
    category="financial",
    thesis="股东集中度的变化速度（加速度）比集中度水平更具预测能力——集中度在加速上升意味着主力正在加大吸筹力度，股价启动在即。集中度减速则意味着吸筹趋于完成或主力开始派发。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_concentration_momentum(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    holder_num = holder["holder_num"]
    concentration = 1.0 / holder_num.replace(0, np.nan)

    # 4-quarter momentum of concentration
    momentum = concentration.groupby(level="Code").transform(lambda s: s.pct_change(4))

    return cross_sectional_rank(momentum)

@register_factor(
    name="pledge_tail_risk",
    description="质押尾部风险因子，-(质押率>50% + 距60日高点跌幅>30%)截面排名（高质押+深跌=极端风险排后）。",
    category="financial",
    thesis="质押率超过50%且股价距高点跌幅超过30%是'质押爆仓高危区'——此时大量质押仓位已触及或接近平仓线，继续下跌将引发大规模强制平仓。A股历史上多次'闪崩'行情均由质押集中爆仓引发。",
    dependencies=("pledge_stat.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_pledge_tail_risk(context: FactorContext):
    pledge = _load_pledge(context, ["pledge_ratio"])
    daily_adj = context.load("daily_adj.parquet")

    pledge_ratio = pledge["pledge_ratio"]
    close = daily_adj["close"]
    high_60 = close.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).max())
    drawdown = 1.0 - close / high_60.replace(0, np.nan)

    common = pledge_ratio.index.intersection(drawdown.index)
    p = pledge_ratio.loc[common]
    d = drawdown.loc[common]

    high_pledge = (p > 50).astype(float)
    deep_dd = (d > 0.3).astype(float)

    tail_risk = high_pledge * deep_dd * p * d
    return cross_sectional_rank(-tail_risk)

@register_factor(
    name="holder_structure_stability",
    description="股东结构稳定性因子，连续4季度股东数变化率绝对值<10%的天数占比截面排名（稳定=可预期排前）。",
    category="financial",
    thesis="股东结构的稳定性本身是公司质量的一个维度——股东数长期稳定的公司通常具有稳定的投资者结构，定价效率高、波动率低。股东结构剧烈波动（无论方向）都是不确定性的来源。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_structure_stability(context: FactorContext):
    holder = context.load_financial("holder_number.parquet",
                                     value_cols=["holder_num"],
                                     date_col="ann_date")
    holder_num = holder["holder_num"]
    chg = holder_num.groupby(level="Code").transform(lambda s: s.pct_change(1))

    is_stable = (chg.abs() < 0.1).astype(float)
    stability = rolling_group_mean(is_stable, 4)

    return cross_sectional_rank(stability)
