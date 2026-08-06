from __future__ import annotations

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank


# ── Liquidity discount ─────────────────────────────────────────────────────

@register_factor(
    name="liquidity_discount_factor",
    description="流动性折价因子，(1-bp排名)×(1-turnover_20排名)截面排名（低估值+低流动性=流动性折价排前）。",
    category="enhanced",
    thesis="低估值的低流动性股票有双重折价——估值折价+流动性折价。当流动性改善时(如被纳入指数)，流动性折价修复会带来显著的alpha。",
    dependencies=("__factors__", "bp", "turnover_20"),
)
def factor_liquidity_discount_factor(context: FactorContext):
    bp = context.load_factor("bp")
    turnover = context.load_factor("turnover_20")
    common = bp.index.intersection(turnover.index)
    bp_r = bp.loc[common].groupby(level="Date").rank(pct=True)
    low_liq = (1 - turnover.loc[common].groupby(level="Date").rank(pct=True))
    return cross_sectional_rank(bp_r * low_liq)
