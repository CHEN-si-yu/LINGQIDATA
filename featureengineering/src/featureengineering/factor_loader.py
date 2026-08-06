from __future__ import annotations

import importlib
import json
import logging
import os
import pkgutil

_LOADED = False

# ── Classification of upstream data sources by frequency ────────────────
# Mirrors the same classification in prepared_data.py's
# _identify_continuous_daily_factors().
_DAILY_SOURCES = {
    "daily.parquet", "finance.parquet",
    "cyq_perf.parquet", "main_fund_flow.parquet", "margin_detail.parquet",
    "limit_up.parquet", "dragon_tiger.parquet", "top_list.parquet",
    "limit_list.parquet", "ths_daily.parquet", "ths_sector_categories.parquet",
    "ths_constituent_stocks.parquet", "pledge_stat.parquet",
    "history_1min", "cyq_chips", "indicator_1min",
}

_NON_DAILY_SOURCES = {
    "financial_indicator.parquet", "balancesheet.parquet",
    "income.parquet", "cashflow.parquet",
    "holder_number.parquet", "index_weight.parquet",
    "stock_list.parquet",
}

_META_SOURCES = {"calendar.parquet"}
_FACTOR_MARKER = "__factors__"

logger = logging.getLogger(__name__)


def _filter_non_continuous_factors() -> int:
    """Remove non-continuous (non-daily-frequency) factors from the registry.

    Reads factor manifests from ``data/manifests/`` and classifies each
    factor as continuous or non-continuous based on whether all of its
    upstream data sources deliver new information every trading day.

    Returns the number of factors removed.
    """
    from .registry import FACTOR_REGISTRY

    # Resolve manifest directory relative to the project root
    # __file__ is at .../src/featureengineering/factor_loader.py
    # parents[2] = .../featureengineering/ (project root)
    from pathlib import Path as _Path
    _package_root = str(_Path(__file__).resolve().parents[2])
    manifest_dir = os.path.join(
        os.environ.get("FEATURE_ENGINEERING_PROJECT_ROOT", _package_root),
        "data", "manifests",
    )

    if not os.path.isdir(manifest_dir):
        logger.debug(
            "Manifest directory not found (%s) — skipping non-continuous filter",
            manifest_dir,
        )
        return 0

    # ── Load manifests ────────────────────────────────────────────────
    factors: dict[str, dict] = {}
    for fname in os.listdir(manifest_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(manifest_dir, fname)) as fh:
            data = json.load(fh)
        name = data["name"]
        deps = data.get("dependencies", [])
        factor_refs: list[str] = []
        data_refs: list[str] = []
        for d in deps:
            if d == _FACTOR_MARKER:
                continue
            if (d in _DAILY_SOURCES or d in _NON_DAILY_SOURCES
                    or d in _META_SOURCES or d.endswith(".parquet")):
                data_refs.append(d)
            else:
                factor_refs.append(d)
        factors[name] = {
            "category": data.get("category", "unknown"),
            "factor_refs": factor_refs,
            "data_refs": data_refs,
        }

    if not factors:
        logger.debug("No manifests found — skipping non-continuous filter")
        return 0

    # ── Classify ───────────────────────────────────────────────────────
    daily: set[str] = set()
    non_daily: set[str] = set()

    for name, info in factors.items():
        real_sources = set(info["data_refs"]) - _META_SOURCES
        if real_sources & _NON_DAILY_SOURCES:
            non_daily.add(name)
        elif real_sources & _DAILY_SOURCES:
            daily.add(name)

    changed = True
    while changed:
        changed = False
        for name, info in factors.items():
            if name in daily or name in non_daily:
                continue
            real_sources = set(info["data_refs"]) - _META_SOURCES
            if real_sources & _NON_DAILY_SOURCES:
                non_daily.add(name)
                changed = True
                continue
            elif real_sources & _DAILY_SOURCES:
                daily.add(name)
                changed = True
                continue
            if info["factor_refs"]:
                if all(r in daily for r in info["factor_refs"]):
                    daily.add(name)
                    changed = True
                elif any(r in non_daily for r in info["factor_refs"]):
                    non_daily.add(name)
                    changed = True

    unresolved = set(factors) - daily - non_daily
    if unresolved:
        logger.warning(
            "%d factors could not be classified: %s",
            len(unresolved), sorted(unresolved)[:10],
        )

    # ── Remove non-continuous factors from the registry ────────────────
    removed = 0
    for name in list(FACTOR_REGISTRY):
        if name in non_daily:
            del FACTOR_REGISTRY[name]
            removed += 1

    if removed:
        logger.info(
            "Filtered %d non-continuous factors from registry (%d remaining)",
            removed, len(FACTOR_REGISTRY),
        )

    return removed


def ensure_builtin_factors_loaded() -> None:
    global _LOADED

    if _LOADED:
        return

    from . import factors as factors_package

    for module_info in pkgutil.walk_packages(
        factors_package.__path__,
        prefix=f"{factors_package.__name__}.",
    ):
        if module_info.name.startswith("_"):
            continue
        # Skip package-level __init__ files
        if module_info.ispkg:
            continue
        try:
            importlib.import_module(module_info.name)
        except ImportError as exc:
            logger.warning(
                "Failed to import factor module %s: %s",
                module_info.name, exc,
            )

    # ── Post-load: remove any remaining non-continuous factors ────────
    _filter_non_continuous_factors()

    _LOADED = True
