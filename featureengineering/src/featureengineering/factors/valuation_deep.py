"""
Deep valuation factors — Class 1 panel factors.

Extends the valuation factor family with:
  - PE-turnover regime interaction (non-linear valuation × liquidity)
  - Industry-adjusted PB (within-sector book-to-price)
  - PE/Dividend ratio (growth vs income dimension)
  - Dividend growth proxies
  - Valuation momentum factors

All factors use vectorized operations on finance.parquet.
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
# Section A: PE-Turnover Regime Interaction
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="pb_turnover_regime",
    description="PB-换手率状态因子（低PB+高换手=价值重估排前）。",
    category="valuation",
    thesis=(
        "PB与换手率的交互与PE逻辑类似但侧重资产价值维度。"
        "低PB+高换手=资产价值被重新发现（排前）；"
        "低PB+低换手=资产价值被长期忽视（中性）；"
        "高PB+低换手=高ROE的合理溢价（中性偏正）；"
        "高PB+高换手=高估值下的筹码博弈（排后）。"
    ),
    dependencies=("finance.parquet",),
)
def factor_pb_turnover_regime(context: FactorContext):
    finance = context.load("finance.parquet")

    pb = finance["pb"].replace(0, np.nan).clip(lower=0.1, upper=100)
    turnover = finance["turnover_rate"].clip(0, 50)

    pb_rank = pb.groupby(level="Date").rank(pct=True)  # high = expensive
    to_rank = turnover.groupby(level="Date").rank(pct=True)

    value_score = (1.0 - pb_rank)
    activity_bonus = 0.5 + 0.5 * to_rank
    regime_score = value_score * activity_bonus

    return cross_sectional_rank(regime_score)

# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Industry-Adjusted PB / Deep Value
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="pb_industry_adjusted",
    description="行业调整市净率因子，1/PB在同THS行业内截面排名（低PB行业内排前）。",
    category="valuation",
    thesis=(
        "不同行业的PB水平存在系统性差异：金融行业PB通常<1，"
        "科技行业PB>5。不做行业调整的BP因子会将金融股系统性排前，"
        "科技股系统性排后——这并非alpha信号，而是行业偏差。"
        "行业调整后的PB消除了这种偏差，提取了真正的行业内相对价值信号。"
    ),
    dependencies=("finance.parquet",),
)
def factor_pb_industry_adjusted(context: FactorContext):
    finance = context.load("finance.parquet")
    pb = finance["pb"].replace(0, np.nan).clip(lower=0.1, upper=100)
    bp = 1.0 / pb

    # Load industry mapping
    industry_map = context.repo.load_industry_map()

    codes = bp.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame({"bp": bp.values, "industry": industries.values}, index=bp.index)
    df = df.dropna(subset=["industry"])

    # Rank BP within Date + industry groups
    bp_rank = df.groupby(["Date", "industry"])["bp"].rank(pct=True)
    return cross_sectional_rank(bp_rank)

# ═══════════════════════════════════════════════════════════════════════════════
# Section C: PE/Dividend & Growth-vs-Income Dimension
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Valuation Momentum / Acceleration
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="bp_momentum_20",
    description="BP动量因子（BP的20日变化率截面排名，BP上升=价值增强排前）。",
    category="valuation",
    thesis=(
        "BP的变化来自两方面：价格变化（分母）和净资产变化（分子）。"
        "BP短期上升（价格下跌快于净资产或净资产增长）意味着价值属性增强，"
        "可能预示着价值回归机会。BP动量捕捉了价值因子的动态变化，"
        "比静态BP水平更能反映边际变化。"
    ),
    dependencies=("finance.parquet",),
)
def factor_bp_momentum_20(context: FactorContext):
    finance = context.load("finance.parquet")
    pb = finance["pb"].replace(0, np.nan).clip(lower=0.1, upper=100)
    bp = 1.0 / pb

    delta = bp.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    delta = delta.clip(-0.5, 1.0)
    return cross_sectional_rank(delta)  # BP rising = more value = ranks high

@register_factor(
    name="sp_ttm_momentum_20",
    description="SP_TTM动量因子（1/PS_TTM的20日变化率截面排名，SP上升排前）。",
    category="valuation",
    thesis=(
        "SP_TTM（市销率倒数）的变化反映了销售额相对价格的变化。"
        "SP上升可能源于：收入增长（基本面改善）或价格下跌（变得便宜）。"
        "与BP动量互补，SP动量对轻资产、高PB公司（如科技、消费）更为有效。"
    ),
    dependencies=("finance.parquet",),
)
def factor_sp_ttm_momentum_20(context: FactorContext):
    finance = context.load("finance.parquet")
    ps = finance["ps_ttm"].replace(0, np.nan).clip(lower=0.1, upper=500)
    sp = 1.0 / ps

    delta = sp.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    delta = delta.clip(-0.5, 1.0)
    return cross_sectional_rank(delta)

# ═══════════════════════════════════════════════════════════════════════════════
# Section E: Valuation Dispersion / Stability
# ═══════════════════════════════════════════════════════════════════════════════
