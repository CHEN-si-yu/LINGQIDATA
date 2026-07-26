"""
Valuation deep extended factors — Class 1 panel factors.

Extends valuation factor coverage using underutilized finance.parquet fields:
  - turnover_rate_f / turnover_rate divergence
  - circ_mv share change (equity structure dynamics)
  - volume_ratio extreme detection
  - dividend yield marginal change

All factors use context.load() for daily-panel access to finance.parquet.
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Free-Float vs Total Turnover Divergence
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="turnover_f_divergence",
    description="自由流通换手率/总换手率。>1=交易集中于自由流通盘，存量筹码活跃，排名高。",
    category="valuation",
    thesis=(
        "自由流通换手率/总股本换手率反映交易在自由流通盘中的集中度。"
        "比率>1意味着交易高度集中于自由流通盘、存量筹码充分交换——筹码活跃度高；"
        "比率<1意味着大股东也在参与交易(异常信号)。该比率捕捉了交易结构的质量差异。"
    ),
    dependencies=("finance.parquet",),
)
def factor_turnover_f_divergence(context: FactorContext) -> np.ndarray:
    f = context.load("finance.parquet")
    ratio = safe_divide(f["turnover_rate_f"], f["turnover_rate"])
    ratio = ratio.clip(0, 2)
    return cross_sectional_rank(ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Market Cap Share Change — equity structure dynamics
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="circ_mv_share_change_20d",
    description="流通市值占比20日变化。占比增加=限售解禁/减持压力，排名低；占比减少=回购/增持，排名高。",
    category="valuation",
    thesis=(
        "流通市值占总市值比重的20日变化捕捉了股本结构变化对价格的影响。"
        "流通占比增加=限售股解禁或大股东减持(可能带来卖压)；"
        "流通占比减少=大股东增持或回购注销(利好信号)。"
    ),
    dependencies=("finance.parquet",),
)
def factor_circ_mv_share_change_20d(context: FactorContext) -> np.ndarray:
    f = context.load("finance.parquet")
    circ_share = safe_divide(f["circ_mv"], f["total_mv"])
    chg = circ_share.groupby(level="Code").transform(lambda s: s.diff(20))
    chg = chg.clip(-0.1, 0.1)
    return cross_sectional_rank(-chg)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Volume Ratio Extreme — liquidity anomaly detection
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="volume_ratio_extreme",
    description="量比极端值因子取反。异常放量(vr>3)=量能衰竭→反转，异常缩量(vr<0.3)=无人问津→可能启动。",
    category="valuation",
    thesis=(
        "量比极端偏离1的程度取反排名。量比>3(异常放量)往往伴随短期量能衰竭后的反转——"
        "放量见顶；量比<0.3(异常缩量)意味着无人问津但地量见地价——可能反转启动。"
        "该因子捕捉极端成交量的均值回归特性。"
    ),
    dependencies=("finance.parquet",),
)
def factor_volume_ratio_extreme(context: FactorContext) -> np.ndarray:
    f = context.load("finance.parquet")
    vr = f["volume_ratio"]
    extreme = (vr - 1.0).abs()
    # Only flag when vr is truly extreme (>2.5 or <0.4)
    is_extreme = (vr > 2.5) | (vr < 0.4)
    signal = extreme * is_extreme.astype(float)
    return cross_sectional_rank(-signal)


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Dividend Yield Marginal Change
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Additional finance field factors (added 2026-07-26 batch 2)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="pe_ttm_percentile_momentum",
    description="PE_TTM历史分位20日变化。分位下降=估值在历史中变得更便宜,排名高。",
    category="valuation",
    thesis=(
        "PE_TTM历史分位(pe_ttm_percentile)的20日变化取反排名。"
        "分位下降=当前PE在自身5年历史中变得更便宜(估值回归)——价值改善信号；"
        "分位上升=PE在历史中变得更贵(估值扩张)——可能高估。"
        "pe_ttm_percentile仅在一处被使用(作为静态排名),其变化率维度是全新的。"
    ),
    dependencies=("finance.parquet",),
)
def factor_pe_ttm_percentile_momentum(context: FactorContext) -> np.ndarray:
    f = context.load("finance.parquet")
    pct = f["pe_ttm_percentile"]
    chg = pct.groupby(level="Code").transform(lambda s: s.diff(20))
    chg = chg.clip(-50, 50)
    return cross_sectional_rank(-chg)


@register_factor(
    name="free_share_turnover_ratio",
    description="自由流通换手/总换手比。高比值=实际可交易筹码在充分换手,排名高。",
    category="valuation",
    thesis=(
        "用自由流通股本(free_share)调整后的换手率: volume / free_share。"
        "相比总换手率(turnover_rate=vol/total_share),自由流通换手率更真实反映"
        "实际可交易筹码的周转速度——剔除了大股东锁定股份的干扰。"
        "free_share字段仅1处引用(valuation.py),该因子是其第二个使用者。"
    ),
    dependencies=("finance.parquet", "daily_adj.parquet"),
)
def factor_free_share_turnover_ratio(context: FactorContext) -> np.ndarray:
    f = context.load("finance.parquet")
    d = context.load("daily_adj.parquet")
    common = f.index.intersection(d.index)
    free_to = safe_divide(d.loc[common, "vol"], f.loc[common, "free_share"])
    free_to = free_to.clip(0, 1e10)
    return cross_sectional_rank(free_to)




@register_factor(
    name="pe_ttm_pb_spread",
    description="PE_TTM百分位与PB百分位的差值。PE分位>PB分位=盈利被低估(优质),排名高。",
    category="valuation",
    thesis=(
        "PE_TTM历史分位排名减去PB截面排名的差值。PE分位高(历史看贵)但PB低(截面看便宜)"
        "=公司盈利能力在历史高位但资产未被重估——盈利质量优异的信号；"
        "PE分位低(历史看便宜)但PB高(截面看贵)=当前低盈利但资产溢价——可能是周期底部。"
        "该因子结合了历史维度(pe_ttm_percentile)和截面维度(pb rank)。"
    ),
    dependencies=("finance.parquet",),
)
def factor_pe_ttm_pb_spread(context: FactorContext) -> np.ndarray:
    f = context.load("finance.parquet")
    pe_pct_rank = f["pe_ttm_percentile"].groupby(level="Date").rank(pct=True)
    pb_rank = f["pb"].groupby(level="Date").rank(pct=True)
    spread = pe_pct_rank - pb_rank
    return cross_sectional_rank(spread)
