"""
Volume-based fund flow factors — Class 1 panel factors.

The existing fund_flow.py (37 factors) and fund_flow_deep.py (9 factors)
almost exclusively use the **_amount (成交额)** variants of main_fund_flow.parquet.
The **_vol (成交量)** variants are severely underused despite containing
complementary information:

Key distinction:
  - Amount-based: weighted by trade price, reflects capital commitment
  - Volume-based: weighted by share count, reflects trading activity breadth

This module fills the gap with volume-based fund flow factors that provide
an orthogonal perspective to the amount-based signals.

Data source: main_fund_flow.parquet (both _vol and _amount variants)
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

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _total_vol(ff: pd.DataFrame) -> pd.Series:
    """Total trading volume across all order sizes, zero → NaN."""
    return (
        ff["buy_sm_vol"] + ff["sell_sm_vol"]
        + ff["buy_md_vol"] + ff["sell_md_vol"]
        + ff["buy_lg_vol"] + ff["sell_lg_vol"]
        + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    ).replace(0, np.nan)


def _total_amount(ff: pd.DataFrame) -> pd.Series:
    """Total trading amount across all order sizes, zero → NaN."""
    return (
        ff["buy_sm_amount"] + ff["sell_sm_amount"]
        + ff["buy_md_amount"] + ff["sell_md_amount"]
        + ff["buy_lg_amount"] + ff["sell_lg_amount"]
        + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    ).replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Medium-Order Volume Factors ("Smart Retail")
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_md_order_vol_ratio",
    description="中单成交量占比因子（(buy_md_vol+sell_md_vol)/总成交量截面排名）。",
    category="fund_flow",
    thesis=(
        "中单（4-20万元）成交量占总成交量的比例反映了'聪明散户'或小型机构的参与度。"
        "中单资金在A股具有独特的信息优势：资金量足够影响盘面但又不是太大以至于暴露意图。"
        "成交额口径已有类似因子（ext_mf_medium_order_amount_ratio），"
        "成交量口径的中单占比提供了无价格偏差的参与度度量。"
        "两者差异反映中单资金的'单价'变化方向。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_md_order_vol_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    md_vol = ff["buy_md_vol"] + ff["sell_md_vol"]
    ratio = safe_divide(md_vol, _total_vol(ff))
    return cross_sectional_rank(ratio)


@register_factor(
    name="mf_sm_order_vol_ratio",
    description="小单成交量占比因子（散户成交量占比截面排名，低占比=机构化排前）。",
    category="fund_flow",
    thesis=(
        "小单（<4万元）成交量占比是散户参与度的纯粹度量。"
        "高散户占比的股票更容易出现行为偏差驱动的错误定价。"
        "低散户占比意味着机构化程度高，定价更有效。"
        "成交量口径避免了散户偏好低价股的偏差（成交额口径会低估散户在低价股中的真实参与度）。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_sm_order_vol_ratio(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    sm_vol = ff["buy_sm_vol"] + ff["sell_sm_vol"]
    ratio = safe_divide(sm_vol, _total_vol(ff))
    return cross_sectional_rank(-ratio)  # low retail = institutionalized = good


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Tier Volume Balance / Concentration
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_vol_tier_balance",
    description="各规模成交量平衡度因子（各档vol占比两两差异绝对值之和截面排名，均衡=健康排前）。",
    category="fund_flow",
    thesis=(
        "当四个订单规模档（小/中/大/特大）的成交量占比趋于均衡时，"
        "市场参与结构健康——各类资金共同参与、互相制衡。"
        "当某一档占比严重偏高时（如全靠大单），行情持续性存疑。"
        "成交量口径的平衡度比成交额口径更强调'参与广度'而非'资金强度'。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_vol_tier_balance(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total = _total_vol(ff)

    sm_share = (ff["buy_sm_vol"] + ff["sell_sm_vol"]) / total
    md_share = (ff["buy_md_vol"] + ff["sell_md_vol"]) / total
    lg_share = (ff["buy_lg_vol"] + ff["sell_lg_vol"]) / total
    elg_share = (ff["buy_elg_vol"] + ff["sell_elg_vol"]) / total

    # Pairwise absolute differences — lower = more balanced
    imbalance = (
        (sm_share - md_share).abs()
        + (sm_share - lg_share).abs()
        + (sm_share - elg_share).abs()
        + (md_share - lg_share).abs()
        + (md_share - elg_share).abs()
        + (lg_share - elg_share).abs()
    )

    return cross_sectional_rank(-imbalance)  # low imbalance = balanced = good


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Net Volume Trend
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_net_vol_trend_3d",
    description="3日净成交量趋势因子（net_mf_vol的3日变化率截面排名，净流入加速排前）。",
    category="fund_flow",
    thesis=(
        "净成交量（net_mf_vol = 总买量 - 总卖量）的短期趋势捕捉了"
        "资金流向的加速度。净流入加速=买方力量在增强；净流出减速=卖方力量在衰竭。"
        "3日窗口足够短以捕捉转折点，又足够长以过滤单日噪音。"
        "成交量口径的净流趋势比成交额口径更能反映交易行为的广泛性变化。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_vol_trend_3d(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    net_vol = ff["net_mf_vol"]

    # 3-day change
    delta = net_vol.groupby(level="Code").transform(
        lambda s: s.diff(3)
    )
    # Normalize by total volume for cross-sectional comparability
    normalized = safe_divide(delta, _total_vol(ff))
    normalized = normalized.clip(-0.5, 0.5)

    return cross_sectional_rank(normalized)


@register_factor(
    name="mf_net_vol_ma_divergence",
    description="净成交量均线偏离因子（净成交量/(净成交量20日均值)-1截面排名，当前>均值=加速排前）。",
    category="fund_flow",
    thesis=(
        "净成交量相对于其20日均值的偏离反映了资金流向的异常强度。"
        "当前净流入远超均值=资金正在'抢筹'；当前净流出远超均值=资金正在'逃离'。"
        "与趋势因子(net_vol_trend_3d)不同，该因子关注的是'当前是否异常'而非'方向是否改变'。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_net_vol_ma_divergence(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    net_vol = ff["net_mf_vol"]

    ma_20 = net_vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    divergence = safe_divide(net_vol, ma_20.abs() + 1e-10) - 1.0
    divergence = divergence.clip(-3, 5)

    return cross_sectional_rank(divergence)


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Volume-Amount Correlation / Divergence
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_vol_amount_corr_20",
    description="量额相关性因子（20日各档vol/amount日变化率相关性截面排名，低相关=价格偏差大排后）。",
    category="fund_flow",
    thesis=(
        "成交量和成交额应当高度正相关。当两者背离时，意味着成交均价发生了变化——"
        "同样的股数在更高的价格成交（均价上升：买方急迫）或在更低的价格成交（均价下降：卖方急迫）。"
        "低vol-amount相关性意味着成交均价不稳定，可能是信息不对称的标志。"
        "该因子捕捉了'成交背后的价格故事'。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_vol_amount_corr_20(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")
    total_vol = _total_vol(ff)
    total_amt = _total_amount(ff)

    # Daily change rates
    vol_chg = total_vol.groupby(level="Code").transform(
        lambda s: s.pct_change(1)
    )
    amt_chg = total_amt.groupby(level="Code").transform(
        lambda s: s.pct_change(1)
    )

    # 20-day rolling correlation of daily changes
    def _rolling_corr(vol_s, amt_s, window=20):
        """Vectorized rolling correlation."""
        vol_roll = vol_s.rolling(window, min_periods=10)
        amt_roll = amt_s.rolling(window, min_periods=10)
        cov = (vol_s * amt_s).rolling(window, min_periods=10).mean() - vol_roll.mean() * amt_roll.mean()
        denom = vol_roll.std() * amt_roll.std()
        return safe_divide(cov, denom)

    corr = vol_chg.groupby(level="Code").transform(
        lambda s: _rolling_corr(s, amt_chg.loc[s.index], 20)
    )
    corr = corr.clip(-1, 1)

    return cross_sectional_rank(corr)  # high correlation = stable pricing = good


@register_factor(
    name="mf_avg_trade_price_momentum",
    description="成交均价动量因子（VWAP(大单)/VWAP(小单)的5日变化率截面排名，比率上升=机构买入紧迫度升排前）。",
    category="fund_flow",
    thesis=(
        "大单VWAP与小单VWAP的比率反映了机构相对于散户的交易价格水平。"
        "比率上升=机构愿意以越来越高的价格买入（买入紧迫度增加）；"
        "比率下降=机构在降价出货。该比率的变化方向比绝对值更有信息量。"
    ),
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_avg_trade_price_momentum(context: FactorContext):
    ff = context.load("main_fund_flow.parquet")

    # Big order (LG + ELG) VWAP
    big_vol = ff["buy_lg_vol"] + ff["sell_lg_vol"] + ff["buy_elg_vol"] + ff["sell_elg_vol"]
    big_amt = ff["buy_lg_amount"] + ff["sell_lg_amount"] + ff["buy_elg_amount"] + ff["sell_elg_amount"]
    big_vwap = safe_divide(big_amt, big_vol)

    # Small order (SM) VWAP
    sm_vol = ff["buy_sm_vol"] + ff["sell_sm_vol"]
    sm_amt = ff["buy_sm_amount"] + ff["sell_sm_amount"]
    sm_vwap = safe_divide(sm_amt, sm_vol)

    # Ratio: big/small VWAP
    ratio = safe_divide(big_vwap, sm_vwap)
    ratio = ratio.clip(0.5, 3.0)

    # 5-day momentum of ratio
    mom = ratio.groupby(level="Code").transform(
        lambda s: s.pct_change(5)
    )
    mom = mom.clip(-0.3, 0.5)

    return cross_sectional_rank(mom)
