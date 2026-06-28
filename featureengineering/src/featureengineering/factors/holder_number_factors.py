"""
Holder number based factors (股东人数因子) — Class 1.

A股经典筹码集中度信号：股东户数的变化反映筹码在散户和主力之间的转移。
户数减少=筹码集中=主力吸筹，户数增加=筹码分散=主力出货。

数据源: holder_number.parquet (季频, holder_num 字段)
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide

# 季报披露滞后：季报在季度结束后约30-45天披露。
# 我们使用 45 个交易日作为保守的滞后偏移。
_HOLDER_LAG = 45  # trading days


def _load_holder(context: FactorContext):
    """Load holder_number.parquet with reporting lag applied."""
    df = context.load_financial(
        "holder_number.parquet",
        value_cols=["holder_num"],
        date_col="end_date",
    )
    # Shift by reporting lag, then ffill to carry forward until next disclosure
    return df.groupby(level="Code").shift(_HOLDER_LAG).groupby(level="Code").ffill()


# ── 股东人数变化 ────────────────────────────────────────────────────────

@register_factor(
    name="holder_num_change_qoq",
    description="股东户数环比变化率因子，holder_num季度环比截面排名（取负向=户数减少=筹码集中排前）。",
    category="quality",
    thesis="股东户数环比减少反映筹码从散户向主力集中——户数下降越快，主力吸筹力度越大，后续股价上涨动力越强。这是A股最经典的反转信号之一。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_change_qoq(context: FactorContext):
    holder = _load_holder(context)
    hn = holder["holder_num"]
    chg = hn.groupby(level="Code").transform(lambda s: s.pct_change(1))
    return cross_sectional_rank(-chg)


@register_factor(
    name="holder_num_change_4q",
    description="股东户数年度变化率因子，holder_num四季度同比截面排名（取负向=年度集中趋势排前）。",
    category="quality",
    thesis="四个季度的户数变化反映年度级别的筹码转移趋势——持续一年户数减少是主力长线建仓的标志，比单季变化更可靠。年度户数降幅>20%通常是牛股启动的前兆。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_change_4q(context: FactorContext):
    holder = _load_holder(context)
    hn = holder["holder_num"]
    chg = hn.groupby(level="Code").transform(lambda s: s.pct_change(4))
    return cross_sectional_rank(-chg)


@register_factor(
    name="holder_num_change_acceleration",
    description="股东户数加速度因子，环比变化-前季环比变化截面排名（取负向=集中加速排前）。",
    category="quality",
    thesis="股东户数减少的加速度（二阶导）比一阶变化更具前瞻性——户数下降速度本身在加快意味着主力在建仓后期的加速吸筹阶段，是拉升前最后的信号窗口。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_num_change_acceleration(context: FactorContext):
    holder = _load_holder(context)
    hn = holder["holder_num"]
    chg = hn.groupby(level="Code").transform(lambda s: s.pct_change(1))
    accel = chg.groupby(level="Code").transform(lambda s: s.diff(1))
    return cross_sectional_rank(-accel)


@register_factor(
    name="holder_num_price_divergence",
    description="股东户数-股价背离因子，(股价排名-户数变化排名)截面排名（股价跌+户数降=洗盘背离排前）。",
    category="quality",
    thesis="股价下跌但股东户数也在减少是最经典的'洗盘吸筹'模式——主力利用股价下跌吓退散户同时暗中吸筹，筹码在悄无声息中集中。这种背离信号比同向信号（价涨户降）更具反向投资价值。",
    dependencies=("holder_number.parquet", "daily_adj.parquet", "calendar.parquet"),
)
def factor_holder_num_price_divergence(context: FactorContext):
    holder = _load_holder(context)
    daily = context.load("daily_adj.parquet")

    hn = holder["holder_num"]
    hn_chg = hn.groupby(level="Code").transform(lambda s: s.pct_change(1))

    close = daily["close"]
    price_chg = close.groupby(level="Code").transform(
        lambda s: s.pct_change(63)  # ~1 quarter
    )

    common = hn_chg.index.intersection(price_chg.index)
    hn_rank = hn_chg.loc[common].groupby(level="Date").rank(pct=True)
    price_rank = price_chg.loc[common].groupby(level="Date").rank(pct=True)

    divergence = (1 - price_rank) * hn_rank
    return cross_sectional_rank(divergence)


@register_factor(
    name="avg_holding_mv_change",
    description="户均持股市值变化率因子，(总市值/户数)季度环比截面排名（户均市值升=大户主导排前）。",
    category="quality",
    thesis="户均持股市值=总市值/股东户数，剔除了股票数量变化的影响——户均市值上升意味着市场中的大户/机构在增持，散户在退出。户均市值持续上升是牛股的共同特征。",
    dependencies=("holder_number.parquet", "finance.parquet", "calendar.parquet"),
)
def factor_avg_holding_mv_change(context: FactorContext):
    holder = _load_holder(context)
    finance = context.load("finance.parquet")

    hn = holder["holder_num"]
    mv = finance["total_mv"]

    common = hn.index.intersection(mv.index)
    avg_mv = mv.loc[common] / hn.loc[common].replace(0, np.nan)
    chg = avg_mv.groupby(level="Code").transform(lambda s: s.pct_change(1))

    return cross_sectional_rank(chg)


@register_factor(
    name="holder_concentration_volatility",
    description="股东集中度波动因子，-(户数4季标准差/户数均值)截面排名（低波动=稳定集中排前）。",
    category="quality",
    thesis="股东户数的波动率反映筹码转移的稳定性——户数波动大意味着筹码在反复换手（可能是游资短炒），户数稳定下降才是真正的主力长线建仓。低波动+下降是最优组合。",
    dependencies=("holder_number.parquet", "calendar.parquet"),
)
def factor_holder_concentration_volatility(context: FactorContext):
    holder = _load_holder(context)
    hn = holder["holder_num"]

    std_4q = hn.groupby(level="Code").transform(
        lambda s: s.rolling(4, min_periods=3).std()
    )
    mean_4q = hn.groupby(level="Code").transform(
        lambda s: s.rolling(4, min_periods=3).mean()
    )
    cv = safe_divide(std_4q, mean_4q)
    return cross_sectional_rank(-cv)
