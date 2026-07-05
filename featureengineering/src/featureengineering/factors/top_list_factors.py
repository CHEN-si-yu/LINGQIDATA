"""
Top List (龙虎榜每日明细) unused field factors — Class 1.

Systematic factorisation of ALL previously unused top_list.parquet fields.
The top_list table is the daily summary of Dragon Tiger board activity — one row
per stock per day with seat-level aggregates.  Unlike dragon_tiger.parquet (which
has per-institution rows), top_list is already at (Date, Code) granularity.

Data source:
- ``top_list.parquet`` (daily event, 2020-01-02 onward)
- ``daily_adj.parquet`` (for reindex alignment)
- ``finance.parquet`` (for market-cap scaling)

Reference
---------
- Column audit: 15 columns, ~4 used in existing factors.
- Unused fields factorised here: l_sell, l_buy, l_amount, net_amount, net_rate,
  amount_rate, float_values, reason.
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
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


def _load_top_list_panel(context: FactorContext) -> pd.DataFrame:
    """Load top_list.parquet as (Date, Code) MultiIndex panel.

    Cached at module level.
    """
    cache = getattr(_load_top_list_panel, "_cache", None)
    if cache is not None:
        return cache

    raw = context.repo._read_parquet(
        context.repo.paths.source_root / "top_list.parquet"
    )
    raw = raw.copy()
    raw["trade_date"] = raw["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    raw["stock_code"] = raw["stock_code"].apply(_pad_code)
    raw = raw.set_index(["trade_date", "stock_code"])
    raw.index = raw.index.set_names(["Date", "Code"])
    raw = raw.reorder_levels(["Date", "Code"]).sort_index()

    _load_top_list_panel._cache = raw
    return raw


# ═══════════════════════════════════════════════════════════════════════════
# A — Seat Flow & Concentration
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_top_list_seat_concentration",
    description="龙虎榜席位集中度因子，席位交易额/总交易额截面排名。",
    category="event",
    thesis="席位交易额占比反映龙虎榜席位对当日总交易的参与程度。占比高说明"
    "当日交易主要由上榜席位驱动，信息含量高；占比低说明上榜席位只是参与者之一。"
    "高集中度事件对后续走势的信号更明确。",
    dependencies=("top_list.parquet", "daily_adj.parquet"),
)
def factor_ext_top_list_seat_concentration(context: FactorContext):
    tl = _load_top_list_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    conc = safe_divide(tl["l_amount"], tl["amount"])
    conc = conc.reindex(daily_adj.index)

    conc_ma = conc.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(conc_ma)


@register_factor(
    name="ext_top_list_net_seat_flow",
    description="龙虎榜席位净流向因子，净买入额/席位交易额截面排名。",
    category="event",
    thesis="席位净买入额占其自身交易额的比例反映了上榜席位的净方向强度。"
    "净买入比例高说明上榜席位整体在买入而非卖出，方向明确。"
    "这一口径比净额/总交易额更能体现上榜资金本身的态度。",
    dependencies=("top_list.parquet", "daily_adj.parquet"),
)
def factor_ext_top_list_net_seat_flow(context: FactorContext):
    tl = _load_top_list_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    net_ratio = safe_divide(tl["net_amount"], tl["l_amount"].abs().replace(0, np.nan))
    net_ratio = net_ratio.reindex(daily_adj.index)

    net_ratio_ma = net_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return cross_sectional_rank(net_ratio_ma)


@register_factor(
    name="ext_top_list_buy_sell_imbalance",
    description="龙虎榜席位买卖失衡因子，(l_buy-l_sell)/(l_buy+l_sell)截面排名。",
    category="event",
    thesis="席位买卖失衡度是席位层面最直接的买卖力量对比。正值表示席位总体买入占优，"
    "负值表示席位总体卖出占优。绝对值大小反映方向的坚定程度。",
    dependencies=("top_list.parquet", "daily_adj.parquet"),
)
def factor_ext_top_list_buy_sell_imbalance(context: FactorContext):
    tl = _load_top_list_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    total_seat = tl["l_buy"] + tl["l_sell"]
    imbalance = safe_divide(tl["l_buy"] - tl["l_sell"], total_seat)
    imbalance = imbalance.reindex(daily_adj.index)

    imb_ma = imbalance.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return cross_sectional_rank(imb_ma)


@register_factor(
    name="ext_top_list_amount_ratio",
    description="龙虎榜成交占比因子，席位成交额/总成交额(amount_rate)截面排名。",
    category="event",
    thesis="amount_rate直接给出了上榜席位在整个市场交易中的参与占比。"
    "该指标比l_amount/amount更精确（数据源已计算）。"
    "高amount_rate的龙虎榜事件对价格发现的贡献更大。",
    dependencies=("top_list.parquet", "daily_adj.parquet"),
)
def factor_ext_top_list_amount_ratio(context: FactorContext):
    tl = _load_top_list_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    ar = tl["amount_rate"].reindex(daily_adj.index)

    ar_ma = ar.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(ar_ma)


# ═══════════════════════════════════════════════════════════════════════════
# B — Market-Cap Context & Net Flow
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_top_list_float_mv_scale",
    description="龙虎榜流通市值因子，上榜日流通市值(float_values)截面排名（低值排前=小盘）。",
    category="event",
    thesis="float_values记录上榜当日的流通市值，直接反映上榜个股的规模特征。"
    "小盘股上榜通常意味着更高的波动和弹性，大盘股上榜则更值得关注基本面变化。"
    "该因子可用于识别龙虎榜事件中的小盘效应。",
    dependencies=("top_list.parquet", "daily_adj.parquet"),
)
def factor_ext_top_list_float_mv_scale(context: FactorContext):
    tl = _load_top_list_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    fv = tl["float_values"].reindex(daily_adj.index)

    # Log scale for better distribution
    fv_log = np.log(fv.replace(0, np.nan))
    fv_ma = fv_log.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    return cross_sectional_rank(-fv_ma)  # smaller float MV = higher rank


@register_factor(
    name="ext_top_list_net_flow_ma_10d",
    description="龙虎榜10日净流向均线因子，近10日席位净买入累计截面排名。",
    category="event",
    thesis="10日席位净买入累计值跟踪中期席位资金流向。10个交易日约两周，"
    "持续正净买入累计说明席位在该股上的做多意愿不是一日游。",
    dependencies=("top_list.parquet", "daily_adj.parquet"),
)
def factor_ext_top_list_net_flow_ma_10d(context: FactorContext):
    tl = _load_top_list_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    net = tl["net_amount"].reindex(daily_adj.index).fillna(0)
    net_10d = net.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).sum()
    )
    return cross_sectional_rank(net_10d)


@register_factor(
    name="ext_top_list_appearance_20d",
    description="龙虎榜20日上榜次数因子，近20日top_list出现次数截面排名。",
    category="event",
    thesis="top_list上榜次数是股票活跃度的直接度量。20日上榜次数多说明个股处于"
    "高关注期，短期交易机会丰富。但极端高频（>8次/20日）也可能意味过度投机。",
    dependencies=("top_list.parquet", "daily_adj.parquet"),
)
def factor_ext_top_list_appearance_20d(context: FactorContext):
    tl = _load_top_list_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    appeared = tl["amount"].notna().reindex(daily_adj.index).fillna(0).astype(float)
    count_20d = appeared.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    return cross_sectional_rank(count_20d)


# ═══════════════════════════════════════════════════════════════════════════
# C — Divergence & Quality
# ═══════════════════════════════════════════════════════════════════════════

@register_factor(
    name="ext_top_list_seat_flow_divergence",
    description="龙虎榜量价背离因子，net_rate与pct_change的差分截面排名。",
    category="event",
    thesis="席位净买入比例与当日涨跌幅的背离是重要的技术信号："
    "大跌+席位净买入=逆势吸筹（积极信号）；"
    "大涨+席位净卖出=顺势出货（消极信号）。"
    "背离度的方向性包含了聪明钱的择时信息。",
    dependencies=("top_list.parquet", "daily_adj.parquet"),
)
def factor_ext_top_list_seat_flow_divergence(context: FactorContext):
    tl = _load_top_list_panel(context)
    daily_adj = context.load("daily_adj.parquet")

    net_rate = tl["net_rate"].reindex(daily_adj.index).fillna(0)
    pct_chg = tl["pct_change"].reindex(daily_adj.index).fillna(0)

    # Normalise both to ranks within date, then compare
    net_rank = net_rate.groupby(level="Date").rank(pct=True)
    pct_rank = (-pct_chg).groupby(level="Date").rank(pct=True)  # negated: low return = high rank

    # Divergence = net flow rank minus (negative) return rank
    # High positive = strong net buying despite weak price → bullish
    divergence = net_rank - pct_rank

    div_ma = divergence.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    return cross_sectional_rank(div_ma)
