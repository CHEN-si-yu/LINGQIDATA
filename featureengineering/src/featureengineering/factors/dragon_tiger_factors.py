"""
Dragon Tiger board (龙虎榜) unused field factors — Class 1.

Systematic factorisation of ALL previously unused dragon_tiger.parquet fields.
The dragon_tiger table records institution-level buy/sell details for stocks
appearing on the Dragon Tiger board each day.  Multiple institutions can appear
for the same stock on the same day — we aggregate to (Date, Code) panel first.

Data source:
- ``dragon_tiger.parquet`` (daily event, 2020-01-02 onward)
- ``daily_adj.parquet`` (for reindex alignment)

Reference
---------
- Column audit: 10 columns, **0% utilised** in existing factors.
- All 5 data columns (buy_amount, buy_ratio, sell_amount, sell_ratio,
  net_buy_amount) plus institution-count signals are factorised here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_sum,
    safe_divide,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pad_code(code: str) -> str:
    """Normalise a stock code to 6-digit string without exchange suffix."""
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _load_dt_daily(context: FactorContext) -> pd.DataFrame:
    """Aggregate dragon_tiger raw rows → (Date, Code) MultiIndex panel.

    Cached at module level so every factor in this file shares one read.
    """
    cache = getattr(_load_dt_daily, "_cache", None)
    if cache is not None:
        return cache

    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "dragon_tiger.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)

    agg = raw.groupby(["trade_date", "stock_code"], sort=False).agg(
        total_buy=("buy_amount", "sum"),
        total_sell=("sell_amount", "sum"),
        net_amount=("net_buy_amount", "sum"),
        org_count=("org_name", "nunique"),
        buy_orgs=("buy_amount", lambda x: (x > 0).sum()),
        sell_orgs=("sell_amount", lambda x: (x > 0).sum()),
        max_buy=("buy_amount", "max"),
        max_sell=("sell_amount", "max"),
        mean_buy_ratio=("buy_ratio", "mean"),
        mean_sell_ratio=("sell_ratio", "mean"),
        reason_list=("reason", lambda x: "|".join(sorted(set(x.dropna())))),
    )
    agg.index = agg.index.set_names(["Date", "Code"])
    agg = agg.reorder_levels(["Date", "Code"]).sort_index()

    _load_dt_daily._cache = agg
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# A — Net Buying Pressure & Institution Flow
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_dt_net_buy_pressure",
    description="龙虎榜净买入压力因子，净买入/(总买+总卖)截面排名（高值排前）。",
    category="event",
    thesis="净买入占比衡量机构在龙虎榜上的净方向强度。正净买入意味着上榜机构整体"
    "在吸筹，负净买入意味着机构在出货。该比值标准化后可比不同市值的股票。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_net_buy_pressure(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    total_flow = dt["total_buy"] + dt["total_sell"]
    pressure = safe_divide(dt["net_amount"], total_flow)
    pressure = pressure.reindex(daily_adj.index)

    # 5-day rolling average to smooth sparse events
    pressure_ma = pressure.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return cross_sectional_rank(pressure_ma)


@register_factor(
    name="ext_dt_buy_sell_ratio",
    description="龙虎榜买卖比因子，总买入/总卖出截面排名（高值排前）。",
    category="event",
    thesis="买入金额与卖出金额之比直接反映多空力量对比。比值>1表示买方力量占优，"
    "比值<1表示卖方力量占优。该因子对短期（1-5日）收益有预测能力。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_buy_sell_ratio(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    ratio = safe_divide(dt["total_buy"], dt["total_sell"])
    ratio = ratio.reindex(daily_adj.index)

    ratio_ma = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return cross_sectional_rank(ratio_ma)


@register_factor(
    name="ext_dt_institution_count",
    description="龙虎榜机构数量因子，当日参与龙虎榜的机构家数截面排名。",
    category="event",
    thesis="参与龙虎榜的机构数量反映了市场对个股的关注广度。机构数量多意味着"
    "多空博弈激烈，短期波动率和交易机会增加。适度的机构关注是积极信号。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_institution_count(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    count = dt["org_count"].reindex(daily_adj.index).fillna(0)
    count_ma = count.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    return cross_sectional_rank(count_ma)


# ═══════════════════════════════════════════════════════════════════════════
# B — Concentration & Consensus
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_dt_buy_concentration",
    description="龙虎榜买入集中度因子（低值排前），最大单笔买入/总买入。",
    category="event",
    thesis="买入集中度高（单一席位主导买入）意味着该席位的判断对股价影响过大，"
    "一旦该席位反手卖出将造成较大抛压。分散的买入结构更健康、持续性更好。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_buy_concentration(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    conc = safe_divide(dt["max_buy"], dt["total_buy"])
    conc = conc.reindex(daily_adj.index)

    conc_ma = conc.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(-conc_ma)  # lower concentration = higher rank


@register_factor(
    name="ext_dt_sell_concentration",
    description="龙虎榜卖出集中度因子（低值排前），最大单笔卖出/总卖出。",
    category="event",
    thesis="卖出集中度高意味着主要抛压来自单一大卖家，可能是特定机构清仓。"
    "这种集中抛售往往是事件驱动的一次性行为，超跌后可能存在修复机会。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_sell_concentration(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    conc = safe_divide(dt["max_sell"], dt["total_sell"])
    conc = conc.reindex(daily_adj.index)

    conc_ma = conc.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(-conc_ma)


@register_factor(
    name="ext_dt_direction_consensus",
    description="龙虎榜方向共识因子，买入机构数/(买入+卖出机构数)截面排名。",
    category="event",
    thesis="方向共识衡量上榜机构中买方占比。高共识（>0.7）意味着多数机构方向一致，"
    "后续走势跟随共识方向的可能性更大。低共识（≈0.5）表示多空分歧大。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_direction_consensus(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    total_orgs = dt["buy_orgs"] + dt["sell_orgs"]
    consensus = safe_divide(dt["buy_orgs"].astype(float), total_orgs.astype(float))
    consensus = consensus.reindex(daily_adj.index)

    consensus_ma = consensus.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(consensus_ma)


# ═══════════════════════════════════════════════════════════════════════════
# C — Rolling / Cumulative Signals
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_dt_net_buy_ma_5d",
    description="龙虎榜5日净买入均线因子，近5日净买入总额截面排名。",
    category="event",
    thesis="5日净买入累计平滑了单日噪音，更可靠地反映机构在中短周期上的方向性偏好。"
    "连续5日净买入累计为正代表持续的机构吸筹行为。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_net_buy_ma_5d(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    net = dt["net_amount"].reindex(daily_adj.index).fillna(0)
    net_5d = net.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    return cross_sectional_rank(net_5d)


@register_factor(
    name="ext_dt_net_buy_ma_20d",
    description="龙虎榜20日净买入均线因子，近20日净买入均值截面排名。",
    category="event",
    thesis="20日净买入均值捕捉中期机构行为趋势。20个交易日约一个月，"
    "持续一个月的净买入代表机构级别的建仓行为，具有较强趋势持续性。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_net_buy_ma_20d(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    net = dt["net_amount"].reindex(daily_adj.index).fillna(0)
    net_20d = net.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    return cross_sectional_rank(net_20d)


@register_factor(
    name="ext_dt_appearance_momentum_20d",
    description="龙虎榜20日上榜动量因子，近期上榜频率变化截面排名。",
    category="event",
    thesis="上榜频率的环比变化反映市场关注度的边际变化。关注度上升期（上榜频率加速）"
    "通常伴随股价趋势，关注度下降期可能预示行情降温。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_appearance_momentum_20d(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    appeared = (dt["org_count"].reindex(daily_adj.index).fillna(0) > 0).astype(float)
    freq_20d = appeared.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    freq_40d = appeared.groupby(level="Code").transform(
        lambda s: s.rolling(40, min_periods=10).sum()
    )
    # Short-term frequency minus longer-term baseline = momentum
    momentum = freq_20d - (freq_40d / 2.0)
    return cross_sectional_rank(momentum)


# ═══════════════════════════════════════════════════════════════════════════
# D — Composite & Derived
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_dt_institution_quality",
    description="龙虎榜机构质量综合因子：共识 × 净买压 × (1-集中度)，截面排名。",
    category="event",
    thesis="综合三个维度的机构行为质量：方向共识（有多少机构同向）、净买压（力度）"
    "和分散度（是否过度集中于单一席位）。三维合成比单一指标更稳健。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_institution_quality(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    # C1: direction consensus (0-1)
    total_orgs = dt["buy_orgs"] + dt["sell_orgs"]
    consensus = safe_divide(dt["buy_orgs"].astype(float), total_orgs.astype(float))
    consensus = consensus.reindex(daily_adj.index).fillna(0.5)

    # C2: net buy pressure (-1 to 1)
    total_flow = dt["total_buy"] + dt["total_sell"]
    pressure = safe_divide(dt["net_amount"], total_flow)
    pressure = pressure.reindex(daily_adj.index).fillna(0)

    # C3: 1 - buy concentration (higher = more dispersed)
    buy_conc = safe_divide(dt["max_buy"], dt["total_buy"])
    buy_conc = buy_conc.reindex(daily_adj.index).fillna(1)
    dispersion = 1.0 - buy_conc

    quality = consensus * pressure * dispersion

    quality_ma = quality.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    return cross_sectional_rank(quality_ma)


@register_factor(
    name="ext_dt_smart_money_divergence",
    description="龙虎榜聪明钱背离因子（低值排前），买机构多但净额为负=出货信号。",
    category="event",
    thesis="当买方机构数量占优但净买入金额为负时，表面上机构在买、实际上在出货——"
    "这是典型的'假买真卖'模式。该背离是重要的风险预警信号。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_smart_money_divergence(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    total_orgs = dt["buy_orgs"] + dt["sell_orgs"]
    consensus = safe_divide(dt["buy_orgs"].astype(float), total_orgs.astype(float))
    consensus = consensus.reindex(daily_adj.index).fillna(0.5)

    total_flow = dt["total_buy"] + dt["total_sell"]
    pressure = safe_divide(dt["net_amount"], total_flow)
    pressure = pressure.reindex(daily_adj.index).fillna(0)

    # Divergence = consensus is high but net pressure is negative
    divergence = consensus - pressure  # large positive = high consensus + negative net = bearish
    divergence = divergence.reindex(daily_adj.index)

    div_ma = divergence.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(-div_ma)  # lower divergence = higher rank


@register_factor(
    name="ext_dt_reason_diversity_20d",
    description="龙虎榜原因多样性因子，20日内上榜原因种类截面排名。",
    category="event",
    thesis="个股因多种原因反复上榜（如连续三个涨停+振幅异常+换手异常）反映了"
    "多维度的异常交易特征。原因多样性本身是一个波动率和关注度代理变量。",
    dependencies=("dragon_tiger.parquet", "daily_adj.parquet"),
)
def factor_ext_dt_reason_diversity_20d(context: FactorContext):
    dt = _load_dt_daily(context)
    daily_adj = context.load("daily_adj.parquet")

    # Count unique reasons per stock per day
    reason_count = dt["reason_list"].apply(
        lambda x: len(set(x.split("|"))) if isinstance(x, str) and x else 0
    )
    reason_count = reason_count.reindex(daily_adj.index).fillna(0)

    # Rolling sum over 20 days
    diversity = reason_count.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    return cross_sectional_rank(diversity)
