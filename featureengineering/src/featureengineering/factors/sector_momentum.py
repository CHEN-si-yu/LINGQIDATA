"""
行业板块成交额动量与行业中性因子（基于 stock_list.parquet 静态行业映射）。

⚠️ 时点风险：行业映射为当前快照，按当前归属回填历史；行业成分变动缓慢，
影响有限（详见 sector.py 头部说明）。

历史说明：本模块原尝试从 ths_constituent_stocks / ths_sector_categories
加载 type='I' 行业板块，但现有数据中该类板块成分数为 0，必然回退到
stock_list.industry。2026-07-31 重构：回退路径提升为唯一实现，并补充
行业中性动量因子。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank
from .momentum_rebuilt import _adjusted_close
from .sector import _build_sector_stocks, _load_industry_map, _map_sector_metric_to_stocks


# ── 板块成交额动量因子 ──────────────────────────────────────────────────────

@register_factor(
    name="sector_amount_momentum_5d",
    description=(
        "板块成交额动量因子：个股所属行业平均成交额5日变化率，"
        "映射到个股后截面排名。反映资金在板块层面的流入/流出动能。"
    ),
    category="sector",
    thesis=(
        "行业板块成交额的变化是机构资金调仓的代理变量。"
        "板块成交额持续放大意味着资金正在系统性流入该板块，"
        "是板块级别行情启动的重要先行指标。"
    ),
    dependencies=("daily.parquet", "stock_list.parquet"),
)
def factor_sector_amount_momentum_5d(context: FactorContext):
    daily = context.load("daily.parquet")
    stock_map = _load_industry_map(context)
    sector_stocks = _build_sector_stocks(stock_map)

    amount = daily["amount"].where(daily["amount"] > 0, np.nan)
    amount_frame = amount.unstack("Code")

    sector_avg_amount: dict[str, pd.Series] = {}
    for ind, codes in sector_stocks.items():
        available = [c for c in codes if c in amount_frame.columns]
        if not available:
            continue
        sector_avg_amount[ind] = amount_frame[available].mean(axis=1)

    if not sector_avg_amount:
        return cross_sectional_rank(
            amount.groupby(level="Code").transform(
                lambda s: s.pct_change(5, fill_method=None)
            )
        )

    sector_avg_df = pd.DataFrame(sector_avg_amount)
    sector_amount_mom_5 = sector_avg_df.pct_change(5, fill_method=None)
    sector_amount_mom_5 = sector_amount_mom_5.replace([np.inf, -np.inf], np.nan)

    stock_metric = _map_sector_metric_to_stocks(sector_amount_mom_5, sector_stocks)

    if stock_metric.empty:
        return pd.Series(
            index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Code"]),
            dtype=float,
        )

    return cross_sectional_rank(stock_metric)


# ── 行业中性动量因子 ────────────────────────────────────────────────────────

@register_factor(
    name="industry_relative_momentum_20",
    description="行业中性20日动量因子（个股20日动量减所属行业均值动量），截面排名。",
    category="sector",
    thesis=(
        "将个股动量剥离行业系统性成分后，剩余的是行业内相对强弱："
        "同行业中动量显著强于均值的个股，在资金抱团与板块轮动中更具持续性。"
        "行业中性化同时消除了动量因子对行业风格的暴露，"
        "与 momentum_20（绝对动量）形成互补。"
    ),
    dependencies=("daily.parquet", "stock_list.parquet"),
)
def factor_industry_relative_momentum_20(context: FactorContext):
    daily = context.load("daily.parquet")
    stock_map = _load_industry_map(context)
    sector_stocks = _build_sector_stocks(stock_map)

    adj = _adjusted_close(daily)
    mom20 = adj.groupby(level="Code").transform(
        lambda s: s.pct_change(20, fill_method=None)
    )
    mom_frame = mom20.unstack("Code")

    # Industry mean 20d momentum (Date × industry), then broadcast back to
    # stocks and subtract: residual = stock momentum minus industry momentum.
    industry_mean: dict[str, pd.Series] = {}
    for ind, codes in sector_stocks.items():
        available = [c for c in codes if c in mom_frame.columns]
        if len(available) < 3:
            continue
        industry_mean[ind] = mom_frame[available].mean(axis=1)

    if not industry_mean:
        return cross_sectional_rank(mom20)

    ind_mean_df = pd.DataFrame(industry_mean)
    stock_ind_mean = _map_sector_metric_to_stocks(ind_mean_df, sector_stocks)

    if stock_ind_mean.empty:
        return cross_sectional_rank(mom20)

    residual = mom20 - stock_ind_mean.reindex(mom20.index)
    return cross_sectional_rank(residual)
