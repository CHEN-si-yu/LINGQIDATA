"""
Extended margin trading factors (扩展融资融券因子) — Class 1.

These factors extend the basic margin factors in ``margin.py`` with flow
dynamics, short-selling signals, squeeze detection, leverage metrics,
composite indicators, and divergence/confirmation signals.

Data source: ``margin_detail.parquet`` (daily, 2019-01-02 onward).
Fields: rzye (融资余额), rzmre (融资买入额), rzche (融资偿还额),
        rqye (融券余额), rqmcl (融券卖出量), rzrqye (融资融券余额),
        rqyl (融券余量), is_backfill
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import (
    cross_sectional_rank,
    rolling_group_mean,
    rolling_group_std,
    rolling_group_sum,
    safe_divide,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _net_margin_flow(margin: pd.DataFrame) -> pd.Series:
    """Daily net margin flow = 融资买入 - 融资偿还."""
    return margin["rzmre"] - margin["rzche"]

# ═══════════════════════════════════════════════════════════════════════════════
# A — Net Flow & Flow Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# B — Short Selling Signals
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# C — Leverage & Cost Metrics
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# D — Composite Indicators
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# E — Divergence & Confirmation Signals
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# F — Extended Margin Factors
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Exchange-Level Margin Factors — utilising the previously unused exchange_id field
# ═══════════════════════════════════════════════════════════════════════════════

def _load_margin_exchange_map(context: FactorContext) -> pd.Series:
    """Load exchange_id for each stock from margin_detail.parquet. Cached.

    Returns a Series mapping stock_code → exchange_id string.
    exchange_id values: 'SH' (上海), 'SZ' (深圳).
    """
    cache = getattr(_load_margin_exchange_map, "_cache", None)
    if cache is not None:
        return cache
    src = context.repo.paths.source_root
    raw = context.repo._read_parquet(src / "margin_detail.parquet")
    # Get unique (stock_code, exchange_id) pairs
    mapping = raw[["stock_code", "exchange_id"]].drop_duplicates(subset=["stock_code"])
    # Normalize codes
    mapping["Code"] = mapping["stock_code"].apply(
        lambda x: str(x).strip().replace(".SZ", "").replace(".SH", "").replace(".BSE", "").zfill(6)
    )
    result = mapping.set_index("Code")["exchange_id"]
    _load_margin_exchange_map._cache = result
    return result

