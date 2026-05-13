from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank

# Module-level cache to avoid re-reading 3368 chip files across 3 factors
_chip_metrics_cache: pd.DataFrame | None = None


def _pad_code(code: str) -> str:
    """Strip exchange suffix and zero-pad to 6 digits."""
    code = str(code).strip()
    if code.upper().endswith((".SZ", ".SH")):
        code = code[:-3]
    elif code.upper().endswith(".BSE"):
        code = code[:-4]
    return code.zfill(6)


def _build_chip_metrics(source_root, daily_adj: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cyq_chips/ per-stock files into a daily panel of chip metrics.

    Returns a DataFrame with (Date, Code) MultiIndex and columns:
      chip_hhi, chip_avg_cost, chip_profit_pct
    """
    global _chip_metrics_cache
    if _chip_metrics_cache is not None:
        return _chip_metrics_cache

    chips_dir = source_root / "cyq_chips"
    close_series = daily_adj["close"]
    allowed_codes = set(daily_adj.index.get_level_values("Code").unique())
    results = []

    for fpath in chips_dir.glob("*.parquet"):
        try:
            raw = pd.read_parquet(fpath)
        except Exception:
            continue

        code = _pad_code(str(raw["stock_code"].iloc[0]))
        if code not in allowed_codes:
            continue

        raw["trade_date"] = (
            raw["trade_date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
        )
        raw = raw.rename(columns={"trade_date": "date"})

        # Get close prices for this stock
        try:
            stock_close = close_series.xs(code, level="Code")
        except KeyError:
            continue

        # Compute metrics per date
        grouped = raw.groupby("date")
        hhi_map = grouped.apply(lambda g: (g["percent"] ** 2).sum())
        avg_cost_map = grouped.apply(
            lambda g: (g["price"] * g["percent"]).sum() / g["percent"].sum()
            if g["percent"].sum() > 0 else np.nan
        )

        for date_val, grp in grouped:
            if date_val not in stock_close.index:
                continue
            close_val = stock_close.loc[date_val]
            if isinstance(close_val, pd.Series):
                close_val = close_val.iloc[0]

            total_pct = grp["percent"].sum()
            if total_pct == 0:
                continue
            below = grp.loc[grp["price"] < close_val, "percent"].sum()
            profit_pct = below / total_pct

            results.append({
                "Date": date_val,
                "Code": code,
                "chip_hhi": hhi_map.loc[date_val] if date_val in hhi_map.index else np.nan,
                "chip_avg_cost": avg_cost_map.loc[date_val] if date_val in avg_cost_map.index else np.nan,
                "chip_profit_pct": profit_pct,
            })

    if not results:
        _chip_metrics_cache = pd.DataFrame(
            columns=["chip_hhi", "chip_avg_cost", "chip_profit_pct"]
        )
        _chip_metrics_cache.index = pd.MultiIndex(
            levels=[[], []], codes=[[], []], names=["Date", "Code"]
        )
        return _chip_metrics_cache

    panel = pd.DataFrame(results)
    panel = panel.set_index(["Date", "Code"]).sort_index()
    _chip_metrics_cache = panel
    return panel


# ── Chip concentration (HHI) ──────────────────────────────────────────────

@register_factor(
    name="chip_concentration",
    description="筹码集中度因子（HHI），赫芬达尔指数截面排名（高集中度排前）。",
    category="chips",
    thesis="筹码集中度反映筹码在少数价格区间的聚集程度，高集中度意味着持仓成本一致性强、突破后趋势性更明确。",
    dependencies=("daily_adj.parquet",),
)
def factor_chip_concentration(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    metrics = _build_chip_metrics(context.repo.paths.source_root, daily_adj)
    return cross_sectional_rank(metrics["chip_hhi"])


# ── Chip profit ratio ─────────────────────────────────────────────────────

@register_factor(
    name="chip_profit_ratio",
    description="获利盘比例因子，现价高于筹码成本区间的筹码占比截面排名。",
    category="chips",
    thesis="获利盘比例反映持仓者的盈亏状态，高获利盘意味着抛压更大，低获利盘（套牢盘多）则有解套卖出压力。",
    dependencies=("daily_adj.parquet",),
)
def factor_chip_profit_ratio(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    metrics = _build_chip_metrics(context.repo.paths.source_root, daily_adj)
    return cross_sectional_rank(metrics["chip_profit_pct"])


# ── Chip cost deviation ───────────────────────────────────────────────────

@register_factor(
    name="chip_cost_deviation",
    description="现价偏离平均筹码成本因子，(close-avg_cost)/avg_cost截面排名。",
    category="chips",
    thesis="现价偏离筹码平均成本的程度反映整体持仓者的盈亏幅度，大幅偏离后存在均值回归动力。",
    dependencies=("daily_adj.parquet",),
)
def factor_chip_cost_deviation(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    metrics = _build_chip_metrics(context.repo.paths.source_root, daily_adj)

    close = daily_adj["close"]
    avg_cost = metrics["chip_avg_cost"]
    common = close.index.intersection(avg_cost.index)
    deviation = close.loc[common] / avg_cost.loc[common].replace(0, np.nan) - 1.0
    return cross_sectional_rank(deviation)
