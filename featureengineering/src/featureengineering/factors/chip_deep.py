"""Class 2 factors — cyq_chips per-stock directory.

Each stock has its own .parquet file in the cyq_chips/ directory.
The unified builder reads each file once and computes all metrics in a single pass,
fanning out to the registered factors — same pattern as intraday.py for history_1min.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank
from .chip import _close_adj_basis

logger = logging.getLogger(__name__)


def _pad_code(code: str) -> str:
    code = str(code).strip()
    for suffix in (".SZ", ".SH", ".BSE"):
        if code.upper().endswith(suffix):
            code = code[: -len(suffix)]
    return code.zfill(6)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-stock metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_single_date_metrics(sub_df: pd.DataFrame,
                                   close_val: float | None = None) -> pd.Series:
    """Compute all chip distribution metrics for a single (trade_date) group.

    Returns a Series with up to 9 metric columns.
    """
    prices = sub_df["price"].values.astype(np.float64)
    percents = sub_df["percent"].values.astype(np.float64)

    total_pct = percents.sum()
    if total_pct <= 0:
        idx = pd.MultiIndex.from_tuples([], names=['Date', 'Code'])
        return pd.Series(dtype=float, index=idx, name='value')

    pcts = percents / total_pct  # normalised probability mass

    # ── Weighted mean ──
    mean = np.sum(prices * pcts)

    # ── Weighted central moments ──
    centered = prices - mean
    variance = np.sum(pcts * centered ** 2)
    std = np.sqrt(variance) if variance > 1e-12 else 0.0

    if std > 1e-12:
        skewness = np.sum(pcts * centered ** 3) / (std ** 3)
        kurtosis = np.sum(pcts * centered ** 4) / (std ** 4) - 3.0
    else:
        skewness = 0.0
        kurtosis = -3.0

    # ── Entropy: -Σ(p_i * log(p_i)) ──
    pos = pcts > 0
    entropy = -np.sum(pcts[pos] * np.log(pcts[pos]))

    # ── Peak dominance ──
    peak_dominance = percents.max() / total_pct

    # ── Peak price ──
    peak_price = prices[np.argmax(percents)]

    # ── Weighted quantiles ──
    sort_idx = np.argsort(prices)
    psorted = prices[sort_idx]
    wsorted = pcts[sort_idx]
    cumsum = np.cumsum(wsorted)

    def _weighted_quantile(q: float) -> float:
        idx = int(np.searchsorted(cumsum, q))
        return float(psorted[min(idx, len(psorted) - 1)])

    p10 = _weighted_quantile(0.10)
    p25 = _weighted_quantile(0.25)
    p50 = _weighted_quantile(0.50)
    p75 = _weighted_quantile(0.75)
    p90 = _weighted_quantile(0.90)
    iqr = p75 - p25

    # ── Coefficient of variation (normalised dispersion) ──
    cv = std / mean if mean > 1e-12 else 0.0

    # ── CR3: top-3 peak concentration ──
    k = min(3, len(percents))
    topk = np.partition(percents, -k)[-k:]
    cr3 = topk.sum() / total_pct

    # ── Gini concentration of probability mass across price bins ──
    # A Gini/Lorenz curve sorts the measured masses themselves.  Sorting by
    # price would make the result depend on whether the dominant bin happened
    # to be cheap or expensive and could even produce a signed location proxy.
    w_sorted = np.sort(pcts.astype(np.float64))
    cs = np.cumsum(w_sorted)
    # G = 1 - 2 * area_under_Lorenz = 1 - (1/n) * Σ (cs_i + cs_{i-1})
    cs_prev = np.concatenate([[0.0], cs[:-1]])
    n = len(w_sorted)
    gini = 1.0 - np.sum(cs + cs_prev) / n if n > 0 else 0.0

    # ── Mode-mean gap (normalised) ──
    mode_mean_gap = abs(peak_price - mean) / mean if mean > 1e-12 else 0.0

    # ── P90-P10 range ──
    p90_p10 = p90 - p10

    # Additional distribution shape metrics
    # Semi-variance: downside chip dispersion (bearish if chips are spread below)
    below_mean = prices <= mean
    if below_mean.any() and below_mean.sum() > 0:
        semi_var = np.sum(pcts[below_mean] * centered[below_mean] ** 2)
        semi_std = np.sqrt(semi_var) if semi_var > 1e-12 else 0.0
    else:
        semi_std = 0.0
    # Upside semi-std
    above_mean = prices > mean
    if above_mean.any() and above_mean.sum() > 0:
        up_semi_var = np.sum(pcts[above_mean] * centered[above_mean] ** 2)
        up_semi_std = np.sqrt(up_semi_var) if up_semi_var > 1e-12 else 0.0
    else:
        up_semi_std = 0.0

    # Chip asymmetry ratio: downside/upside dispersion
    chip_asymmetry = semi_std / up_semi_std if up_semi_std > 1e-12 else 1.0

    # Price percentile within chip distribution (where is current price?)
    chip_percentile = float(np.interp(close_val, psorted, cumsum)) if (close_val is not None and not np.isnan(close_val)) else 0.5

    # Distribution width relative to price
    if close_val is not None and not np.isnan(close_val) and close_val > 1e-12:
        chip_width_ratio = std / close_val
    else:
        chip_width_ratio = std / mean if mean > 1e-12 else 0.0

    # ── Deep-trap / high-float ratios (close-relative, 2026-08-05) ──
    if close_val is not None and not np.isnan(close_val):
        chip_below_90 = percents[prices < 0.9 * close_val].sum() / total_pct
        chip_upper_110 = percents[prices > 1.1 * close_val].sum() / total_pct
    else:
        chip_below_90 = np.nan
        chip_upper_110 = np.nan

    # ── Mode-median gap: bimodality proxy (双峰=模式价与中位价分离) ──
    chip_mode_median_gap = abs(peak_price - p50) / std if std > 1e-12 else 0.0

    # ── Quantile-based skewness (对离群值比三阶矩更稳健) ──
    chip_range_skew = (p90 - p50) / (p50 - p10) if (p50 - p10) > 1e-12 else 1.0

    result = {
        "chip_peak_price": peak_price,
        "chip_weighted_mean": mean,
        "chip_weighted_std": std,
        "chip_cv": cv,
        "chip_cr3": cr3,
        "chip_gini": gini,
        "chip_mode_mean_gap": mode_mean_gap,
        "chip_skewness": skewness,
        "chip_kurtosis": kurtosis,
        "chip_entropy": entropy,
        "chip_peak_dominance": peak_dominance,
        "chip_iqr": iqr,
        "chip_p90_p10": p90_p10,
        "chip_median_price": p50,
        "chip_semi_std": semi_std,
        "chip_up_semi_std": up_semi_std,
        "chip_asymmetry": chip_asymmetry,
        "chip_percentile": chip_percentile,
        "chip_width_ratio": chip_width_ratio,
        "chip_below_90": chip_below_90,
        "chip_upper_110": chip_upper_110,
        "chip_mode_median_gap": chip_mode_median_gap,
        "chip_range_skew": chip_range_skew,
    }

    if close_val is not None and not np.isnan(close_val):
        below_sum = percents[prices <= close_val].sum()
        result["chip_below_ratio"] = below_sum / total_pct
        # ── 获利/套牢筹码的峰集中度 (2026-08-11) ──
        # 现价下方(获利)筹码的最大单箱占比:高=获利筹码集中成峰(主力成本密集
        # 锁筹,吸筹完成特征);低=获利筹码分散(浮筹)。上方(套牢)同理:
        # 集中=单点压力位,分散=套牢盘零散。
        below_pcts = percents[prices <= close_val]
        if below_sum > 1e-12:
            result["chip_win_peak_frac"] = below_pcts.max() / below_sum
        else:
            result["chip_win_peak_frac"] = np.nan
        above_pcts = percents[prices > close_val]
        above_sum = above_pcts.sum()
        if above_sum > 1e-12:
            result["chip_loss_peak_frac"] = above_pcts.max() / above_sum
        else:
            result["chip_loss_peak_frac"] = np.nan

    return pd.Series(result)


def _chip_daily_metrics(stock_df: pd.DataFrame,
                        close_series: pd.Series | None = None) -> pd.DataFrame:
    """Compute daily chip metrics for one stock in a single pass.

    Returns DataFrame indexed by trade_date (YYYYMMDD) with all metric columns.
    """
    df = stock_df.copy()
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    df = df.sort_values(["trade_date", "price"], kind="stable")

    MAX_PRICE = 10000.0
    df = df[(df["price"] > 0) & (df["price"] < MAX_PRICE) & (df["percent"] > 0)]
    if df.empty:
        return pd.DataFrame()

    close_dict: dict[str, float] = {}
    if close_series is not None:
        close_s = close_series.rename("close")
        merged = df[["trade_date"]].drop_duplicates().merge(
            close_s, left_on="trade_date", right_index=True, how="left",
        )
        close_dict = dict(zip(merged["trade_date"], merged["close"]))

    rows: list[pd.Series] = []
    for name, group in df.groupby("trade_date", sort=False):
        row = _compute_single_date_metrics(group, close_dict.get(name))
        if not row.empty:
            row.name = name
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
# Factor metric mapping
# ═══════════════════════════════════════════════════════════════════════════════

CHIP_FACTOR_SPEC: dict[str, tuple[str, str]] = {
    # ── Existing (3) ──
    "chip_peak_distance":       ("chip_peak_price",     "peak"),
    "chip_peak_ratio":          ("chip_below_ratio",    "pos"),
    "chip_below_momentum":      ("chip_below_ratio",    "momentum"),
    # ── Static distribution rank (6) ──
    "chip_dispersion":          ("chip_weighted_std",   "neg"),
    "chip_skew_factor":         ("chip_skewness",        "pos"),
    "chip_entropy_signal":      ("chip_entropy",         "neg"),
    "chip_peak_purity":         ("chip_peak_dominance",  "pos"),
    "chip_tail_risk":           ("chip_kurtosis",        "neg"),
    "chip_iqr_factor":          ("chip_iqr",             "neg"),
    # ── Distribution momentum — 5-day change (5) ──
    "chip_dispersion_momentum": ("chip_weighted_std",   "momentum_rev"),
    "chip_skew_momentum":       ("chip_skewness",        "momentum"),
    "chip_entropy_convergence": ("chip_entropy",         "momentum_rev"),
    "chip_peak_growing":        ("chip_peak_dominance",  "momentum"),
    "chip_tail_risk_change":    ("chip_kurtosis",        "momentum_rev"),
    # ── Close-distance (1) ──
    "chip_mean_distance":       ("chip_weighted_mean",   "peak"),
    # ── 2nd batch — variant/concentration stats (6) ──
    "chip_cv_factor":           ("chip_cv",              "neg"),
    "chip_cr3_factor":          ("chip_cr3",             "pos"),
    "chip_gini_factor":         ("chip_gini",            "pos"),
    "chip_mode_mean_gap":       ("chip_mode_mean_gap",   "neg"),
    "chip_p90_p10_factor":      ("chip_p90_p10",         "neg"),
    "chip_median_distance":     ("chip_median_price",    "peak"),
    # ── 2nd batch — momentum (5) ──
    "chip_cv_momentum":         ("chip_cv",              "momentum_rev"),
    "chip_cr3_momentum":        ("chip_cr3",             "momentum"),
    "chip_gini_momentum":       ("chip_gini",            "momentum"),
    "chip_mode_mean_convergence": ("chip_mode_mean_gap", "momentum_rev"),
    "chip_p90_p10_momentum":    ("chip_p90_p10",         "momentum_rev"),
    # ── New: downside risk & asymmetry (3) ──
    "chip_downside_risk":       ("chip_semi_std",        "neg"),
    "chip_upside_potential":    ("chip_up_semi_std",     "pos"),
    "chip_asymmetry_factor":    ("chip_asymmetry",       "pos"),
    # ── New: price-position in distribution (3) ──
    "chip_percentile_factor":   ("chip_percentile",      "pos"),
    "chip_width_ratio_factor":  ("chip_width_ratio",     "neg"),
    "chip_percentile_momentum": ("chip_percentile",      "momentum"),
    # ── New: additional transforms of existing metrics (4) ──
    "chip_skew_momentum_rev":   ("chip_skewness",        "momentum_rev"),
    "chip_kurtosis_momentum":   ("chip_kurtosis",        "momentum"),
    "chip_gini_momentum_rev":   ("chip_gini",            "momentum_rev"),
    "chip_cv_pos":              ("chip_cv",              "pos"),
    # ── 2026-08-05: close-relative trap ratios & shape extras (4) ──
    "chip_deep_trap_ratio":     ("chip_upper_110",       "neg"),
    "chip_high_float_ratio":    ("chip_below_90",        "neg"),
    "chip_bimodality":          ("chip_mode_median_gap", "neg"),
    "chip_range_skew_factor":   ("chip_range_skew",      "neg"),
    # ── 2026-08-05: cost-centre momentum (2) ──
    "chip_median_momentum":     ("chip_median_price",    "momentum"),
    "chip_weighted_mean_momentum": ("chip_weighted_mean","momentum"),
    # ── 2026-08-05: dispersion/risk momentum (4) ──
    "chip_iqr_momentum_20d":    ("chip_iqr",             "momentum_20d"),
    "chip_semi_std_momentum":   ("chip_semi_std",        "momentum_rev"),
    "chip_width_ratio_momentum":("chip_width_ratio",     "momentum_rev"),
    "chip_percentile_20d":      ("chip_percentile",      "momentum_20d"),
    # ── 2026-08-11: 获利/套牢筹码峰集中度 (3) ──
    "chip_win_peak_frac":       ("chip_win_peak_frac",   "pos"),
    "chip_loss_peak_frac":      ("chip_loss_peak_frac",  "neg"),
    "chip_win_peak_growth":     ("chip_win_peak_frac",   "momentum"),
}

_CHIP_METRIC_COLS = {
    "chip_peak_price",
    "chip_below_ratio",
    "chip_weighted_mean",
    "chip_weighted_std",
    "chip_cv",
    "chip_cr3",
    "chip_gini",
    "chip_mode_mean_gap",
    "chip_skewness",
    "chip_kurtosis",
    "chip_entropy",
    "chip_peak_dominance",
    "chip_iqr",
    "chip_p90_p10",
    "chip_median_price",
    "chip_semi_std",
    "chip_up_semi_std",
    "chip_asymmetry",
    "chip_percentile",
    "chip_width_ratio",
    "chip_below_90",
    "chip_upper_110",
    "chip_mode_median_gap",
    "chip_range_skew",
    "chip_win_peak_frac",
    "chip_loss_peak_frac",
}


def _make_multiindex_series(values: np.ndarray, dates: pd.Index, code: str,
                            name: str) -> pd.Series:
    """Build a (Date, Code) MultiIndex Series from a date-indexed array."""
    idx = pd.MultiIndex.from_arrays(
        [dates, [code] * len(dates)], names=["Date", "Code"]
    )
    return pd.Series(values, index=idx, name=name)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch worker (ProcessPoolExecutor)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level global set once per worker process by _init_chip_worker.
# Avoids pickling the large close_map dict (~100 MB) for every batch submission —
# for ~5,000 stocks that saves ~200 redundant serializations.
_worker_close_map: dict[str, pd.Series] | None = None
_worker_close_adj_map: dict[str, pd.Series] | None = None


def _init_chip_worker(
    close_map: dict[str, pd.Series] | None,
    close_adj_map: dict[str, pd.Series] | None = None,
) -> None:
    """ProcessPoolExecutor initializer — sets close_map once per worker."""
    global _worker_close_map, _worker_close_adj_map
    _worker_close_map = close_map
    _worker_close_adj_map = close_adj_map


def _process_chip_batch(
    batch: list[tuple[str, str]],
    min_trade_date: str | None = None,
) -> dict[str, list[pd.Series]]:
    """Process a batch of stock files in a worker process.

    Args:
        batch: list of (code, filepath_str) tuples.
        min_trade_date: if set, filter rows to trade_date >= min_trade_date.

    close_map is read from the module-level global ``_worker_close_map``
    set by ``_init_chip_worker`` — it is *not* pickled per submission.

    Returns:
        dict mapping metric_col → list of pd.Series (one per stock).
    """
    accum: dict[str, list[pd.Series]] = defaultdict(list)
    for code, fpath_str in batch:
        try:
            stock_df = pd.read_parquet(fpath_str)
        except Exception:
            continue
        if stock_df.empty:
            continue
        # Normalize trade_date to YYYYMMDD so comparisons with min_trade_date
        # work correctly regardless of the source file format (e.g. "2026-06-01").
        stock_df["trade_date"] = (
            stock_df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
        )
        if min_trade_date is not None:
            stock_df = stock_df[stock_df["trade_date"] >= min_trade_date]
            if stock_df.empty:
                continue
        try:
            close_s = _worker_close_adj_map.get(code) if _worker_close_adj_map else (
                _worker_close_map.get(code) if _worker_close_map else None
            )
            daily = _chip_daily_metrics(stock_df, close_series=close_s)
        except Exception:
            continue
        for col in _CHIP_METRIC_COLS:
            if col not in daily.columns:
                continue
            s = daily[col].dropna()
            if s.empty:
                continue
            accum[col].append(
                _make_multiindex_series(s.values, s.index, code, col)
            )
    return dict(accum)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified builder entry point
# ═══════════════════════════════════════════════════════════════════════════════

def build_cyq_chips_unified(
    paths,                      # ProjectPaths
    factor_names: list[str],
    allowed_codes: set[str],
    force: bool = False,
    on_progress=None,           # (stage: str, current: int, total: int) -> None
    max_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all cyq_chips factors in a **single pass** over per-stock files.

    Uses ProcessPoolExecutor to parallelise metric computation across batches
    of stocks.  Each stock file is read once and ``_chip_daily_metrics`` is
    called once, producing metric columns that are fanned out to the registered
    cyq_chips factors.

    Returns ``{factor_name: wide_DataFrame}`` ready for ``write_factor`` or
    ``write_factor_incremental``.
    """
    from ..builder import _resolve_effective_end_date, _read_factor_max_date
    from ..storage import ensure_single_factor_frame

    chip_dir = paths.source_root / "cyq_chips"
    if not chip_dir.exists():
        raise FileNotFoundError(f"cyq_chips directory not found: {chip_dir}")

    # ── Determine incremental filter date ──────────────────────────────────
    effective_end = _resolve_effective_end_date(paths.source_root)

    min_trade_date: str | None = None
    earliest_existing: str | None = None
    if not force:
        for name in factor_names:
            factor_path = paths.factor_output_dir / f"{name}.fea"
            if factor_path.exists():
                fm = _read_factor_max_date(factor_path)
                if fm:
                    if earliest_existing is None or fm < earliest_existing:
                        earliest_existing = fm
        if earliest_existing and earliest_existing < effective_end:
            from datetime import datetime, timedelta
            _lb_dt = datetime.strptime(earliest_existing, "%Y%m%d") - timedelta(days=45)
            min_trade_date = _lb_dt.strftime("%Y%m%d")

    # ── Build file list ────────────────────────────────────────────────────
    all_files = sorted([f for f in os.listdir(chip_dir) if f.endswith(".parquet")])
    file_map: dict[str, str] = {}
    for fname in all_files:
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if not allowed_codes or code in allowed_codes:
            file_map[code] = str(chip_dir / fname)

    if not file_map:
        return {n: pd.DataFrame() for n in factor_names}

    total_files = len(file_map)
    if on_progress:
        on_progress("chip_scan", 0, total_files)

    # ── Pre-load close prices (needed for chip_below_ratio) ────────────────
    daily_path = paths.source_root / "daily.parquet"
    close_df = pd.read_parquet(daily_path, columns=["trade_date", "stock_code", "close", "pct_chg"])
    close_df["trade_date"] = close_df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    close_df["stock_code"] = close_df["stock_code"].apply(_pad_code)
    if allowed_codes:
        close_df = close_df[close_df["stock_code"].isin(allowed_codes)]
    close_map, close_adj_map = _build_close_maps(close_df)
    del close_df

    # ── Determine parallelism ──────────────────────────────────────────────
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4))

    batch_size = max(1, total_files // max_workers)
    file_items = list(file_map.items())
    batches = [file_items[i:i + batch_size] for i in range(0, len(file_items), batch_size)]

    # ── Process batches in parallel ─────────────────────────────────────────
    accumulators: dict[str, list[pd.Series]] = defaultdict(list)
    completed = 0
    t_start = time.perf_counter()
    batch_file_counts = [len(b) for b in batches]

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_chip_worker,
        initargs=(close_map, close_adj_map),
    ) as executor:
        futures = {
            executor.submit(
                _process_chip_batch, batch, min_trade_date,
            ): idx
            for idx, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            try:
                batch_result = fut.result()
            except Exception:
                completed += batch_file_counts[batch_idx]
                if on_progress:
                    on_progress("chip_scan", completed, total_files)
                continue
            for col, series_list in batch_result.items():
                accumulators[col].extend(series_list)
            completed += batch_file_counts[batch_idx]
            if on_progress:
                on_progress("chip_scan", completed, total_files)

    scan_elapsed = time.perf_counter() - t_start

    # ── Concatenate per-stock results ───────────────────────────────────────
    if on_progress:
        on_progress("chip_concat", 0, len(factor_names))

    raw_metrics: dict[str, pd.Series] = {}
    for col in _CHIP_METRIC_COLS:
        parts = accumulators.get(col, [])
        if parts:
            s = pd.concat(parts)
            if s.index.has_duplicates:
                s = s[~s.index.duplicated(keep="last")]
            if isinstance(s.index, pd.MultiIndex):
                s.index = s.index.set_names(["Date", "Code"])
            raw_metrics[col] = s.sort_index()
        else:
            raw_metrics[col] = pd.Series(dtype=float, name=col)

    # ── Build final factor frames ───────────────────────────────────────────
    output: dict[str, pd.DataFrame] = {}
    for idx, name in enumerate(factor_names):
        if on_progress:
            on_progress("chip_build", idx + 1, len(factor_names))

        if name not in CHIP_FACTOR_SPEC:
            continue
        metric_col, direction = CHIP_FACTOR_SPEC[name]

        raw = raw_metrics.get(metric_col)
        if raw is None or raw.empty:
            output[name] = pd.DataFrame()
            continue

        if direction == "momentum":
            chg = raw.groupby(level="Code").transform(lambda s: s.diff(5))
            ranked = cross_sectional_rank(chg)
        elif direction == "momentum_rev":
            chg = raw.groupby(level="Code").transform(lambda s: s.diff(5))
            ranked = cross_sectional_rank(-chg)
        elif direction == "momentum_20d":
            chg = raw.groupby(level="Code").transform(lambda s: s.diff(20))
            ranked = cross_sectional_rank(chg)
        elif direction == "peak":
            # cyq_chips 的 chip_peak_price/weighted_mean/median_price 为复权口径
            # 成本价,close 需折算到同一空间(见 _build_close_maps 注释)
            close_adj_all = close_map_to_series(close_adj_map)
            common = raw.index.intersection(close_adj_all.index)
            distance = (
                close_adj_all.loc[common] - raw.loc[common]
            ) / close_adj_all.loc[common].replace(0, np.nan)
            ranked = cross_sectional_rank(distance)
        elif direction == "pos":
            ranked = cross_sectional_rank(raw)
        elif direction == "neg":
            ranked = cross_sectional_rank(-raw)
        else:
            ranked = cross_sectional_rank(raw)

        frame = ensure_single_factor_frame(ranked, name, paths=paths)
        output[name] = frame

    # ── Incremental: keep only new dates in output frames ──────────────────
    if not force and earliest_existing and min_trade_date is not None:
        for name in list(output.keys()):
            frame = output.get(name)
            if frame is None or frame.empty:
                continue
            frame = frame[frame.index > earliest_existing]
            if frame.empty:
                output[name] = pd.DataFrame()
            else:
                output[name] = frame

    return output


def close_map_to_series(close_map: dict[str, pd.Series]) -> pd.Series:
    """Convert a code→Series[trade_date→close] dict to a (Date, Code) MultiIndex Series."""
    parts = []
    for code, s in close_map.items():
        if s.empty:
            continue
        idx = pd.MultiIndex.from_arrays(
            [s.index, [code] * len(s)], names=["Date", "Code"]
        )
        parts.append(pd.Series(s.values, index=idx, name="close"))
    if not parts:
        return pd.Series(dtype=float, name="close")
    result = pd.concat(parts)
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    return result


def _build_close_maps(close_df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """从 daily 面板构建 (close_map, close_adj_map)，两者均为未复权 close。

    close_adj_map 不再做 cumprod(1+pct_chg/100) × K(末值锚定) 折算:除权日
    pct_chg 按"除权参考价"口径计算使 cumprod 基座与 close 失去一致性,锚点跨
    过除权日会把全历史 close_adj 重标定,同一日期因子值随构建日变化(未来函数,
    见 chip._close_adj_basis 的说明)。直接使用未复权 close:除权日与 cyq 复权
    成本价短暂失真,但历史值稳定、只由 ≤当日 数据决定。
    """
    close_map: dict[str, pd.Series] = {}
    close_adj_map: dict[str, pd.Series] = {}
    for code, grp in close_df.groupby("stock_code"):
        grp = grp.sort_values("trade_date")
        close_map[code] = grp.set_index("trade_date")["close"]
        close_adj_map[code] = close_map[code]
    return close_map, close_adj_map


# ═══════════════════════════════════════════════════════════════════════════════
# New unified builder entry point (--new flag)
# ═══════════════════════════════════════════════════════════════════════════════

def build_cyq_chips_new(
    factor_names: list[str],
    paths,                      # ProjectPaths
    force: bool = False,
    max_workers: int | None = None,
    quality_check_days: int = 0,
) -> list:
    """Build all Class 2 cyq_chips factors in a single pass with enhanced progress.

    This is the --new entry point.  Each per-stock parquet file is opened **once**,
    all 15 raw metrics are computed together, then fanned out to every requested
    factor.  Multi-stock batching runs in parallel via ProcessPoolExecutor.

    The progress display uses a redesigned multi-phase layout:
      Phase 1 — File discovery & validation
      Phase 2 — Parallel metric computation (stocks batched across workers)
      Phase 3 — Factor frame assembly
      Phase 4 — Writing .fea output files

    Parameters
    ----------
    quality_check_days:
        If > 0, inspect the last N trading days of existing .fea files.
        Factors whose recent rows are all-NaN will be force-rebuilt.

    Returns a list of ``BuildResult`` objects compatible with CLI expectations.
    """
    from ..builder import (
        BuildResult,
        _resolve_effective_end_date,
        _read_factor_max_date,
        _check_factor_recent_quality,
        get_factor,
    )
    from ..dataset import _load_allowed_codes
    from ..storage import (
        ensure_single_factor_frame,
        record_build_elapsed,
        write_factor,
        write_factor_incremental,
    )
    allowed = _load_allowed_codes(paths.stock_pool_file)
    chip_dir = paths.source_root / "cyq_chips"

    if not chip_dir.exists():
        raise FileNotFoundError(f"cyq_chips directory not found: {chip_dir}")

    # ── Phase 1 header ───────────────────────────────────────────────────
    total_factors = len(factor_names)
    n_workers = min(max_workers or 32, (os.cpu_count() or 4))

    print(f"\n{'='*64}")
    print(f"  CLASS 2 — UNIFIED SINGLE-PASS BUILDER")
    print(f"  Factors: {total_factors}   Workers: {n_workers}")
    print(f"  Strategy: open each stock file once, compute all metrics together")
    print(f"{'='*64}")

    # ── Phase 1: File discovery ──────────────────────────────────────────
    t_phase1 = time.perf_counter()
    print(f"\n  Phase 1/4 — File discovery")

    all_files = sorted([f for f in os.listdir(chip_dir) if f.endswith(".parquet")])
    file_map: dict[str, str] = {}
    for fname in all_files:
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if not allowed or code in allowed:
            file_map[code] = str(chip_dir / fname)

    total_stocks = len(file_map)
    print(f"  Stocks found: {total_stocks}  (filtered from {len(all_files)} files)")

    if not file_map:
        return [BuildResult(
            factor_name=n, action="skip", elapsed=0.0, rows=0,
            factor_path=paths.factor_output_dir / f"{n}.fea",
            manifest_path=paths.manifest_output_dir / f"{n}.json",
        ) for n in factor_names]

    # ── Incremental path permanently disabled — always full rebuild ──
    effective_end = _resolve_effective_end_date(paths.source_root)

    # All factors are always active — skip logic removed
    active_names = list(factor_names)
    total_factors = len(factor_names)

    # Always full rebuild — no incremental date filtering
    min_trade_date: str | None = None
    earliest_existing: str | None = None  # kept for dead-code compatibility
    mode_str = "full rebuild (incremental disabled)"
    print(f"  Mode: {mode_str}")
    # ── Pre-load close prices ────────────────────────────────────────────
    t_close = time.perf_counter()
    daily_path = paths.source_root / "daily.parquet"
    close_df = pd.read_parquet(daily_path, columns=["trade_date", "stock_code", "close", "pct_chg"])
    close_df["trade_date"] = close_df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    close_df["stock_code"] = close_df["stock_code"].apply(_pad_code)
    if allowed:
        close_df = close_df[close_df["stock_code"].isin(allowed)]
    close_map, close_adj_map = _build_close_maps(close_df)
    del close_df
    print(f"  Close prices loaded: {len(close_map)} stocks  "
          f"({time.perf_counter() - t_close:.1f}s)")

    t_phase1_elapsed = time.perf_counter() - t_phase1
    print(f"  Phase 1 done  ({t_phase1_elapsed:.1f}s)")

    # ── Phase 2: Parallel metric computation ─────────────────────────────
    t_phase2 = time.perf_counter()
    print(f"\n  Phase 2/4 — Computing chip metrics  "
          f"[ProcessPoolExecutor x{n_workers}]")

    # Use small batches (~40 stocks each) so the progress bar updates
    # frequently — the first batch won't block display for minutes.
    BATCH_SIZE = 40
    file_items = list(file_map.items())
    batches = [file_items[i:i + BATCH_SIZE] for i in range(0, len(file_items), BATCH_SIZE)]
    total_batches = len(batches)
    batch_file_counts = [len(b) for b in batches]

    from tqdm import tqdm

    accumulators: dict[str, list[pd.Series]] = defaultdict(list)
    completed_stocks = 0

    pbar = tqdm(
        total=total_stocks, desc="  Stocks", unit="stk",
        bar_format="{desc:>12}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {postfix}]",
    )

    t_batch_start = time.perf_counter()
    batch_errors = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_chip_worker,
        initargs=(close_map, close_adj_map),
    ) as executor:
        futures = {
            executor.submit(
                _process_chip_batch, batch, min_trade_date,
            ): idx
            for idx, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            n_in_batch = batch_file_counts[batch_idx]
            try:
                batch_result = fut.result()
            except Exception as e:
                batch_errors += 1
                completed_stocks += n_in_batch
                pbar.update(n_in_batch)
                if batch_errors <= 3:
                    pbar.write(f"  [!] Batch {batch_idx} failed: {type(e).__name__}: {e}")
                continue
            for col, series_list in batch_result.items():
                accumulators[col].extend(series_list)
            completed_stocks += n_in_batch
            elapsed_batch = time.perf_counter() - t_batch_start
            rate = completed_stocks / elapsed_batch if elapsed_batch > 0 else 0
            pbar.set_postfix_str(f"{rate:.0f} stk/s")
            pbar.update(n_in_batch)

    pbar.close()

    if batch_errors:
        print(f"  [!] {batch_errors}/{total_batches} batches failed")
    total_accumulated = sum(len(v) for v in accumulators.values())
    if not accumulators:
        print(f"  [!] FATAL: all batches failed — no metrics accumulated. "
              f"Check that cyq_chips/*.parquet files are readable.")
    else:
        print(f"  Accumulated {total_accumulated} metric series across "
              f"{len(accumulators)} columns")

    t_phase2_elapsed = time.perf_counter() - t_phase2
    rate_p2 = total_stocks / t_phase2_elapsed if t_phase2_elapsed > 0 else 0
    print(f"  Phase 2 done  ({t_phase2_elapsed:.1f}s, {rate_p2:.0f} stocks/s)")

    # ── Concatenate raw metrics ──────────────────────────────────────────
    t_concat = time.perf_counter()
    raw_metrics: dict[str, pd.Series] = {}
    for col in _CHIP_METRIC_COLS:
        parts = accumulators.get(col, [])
        if parts:
            s = pd.concat(parts)
            if s.index.has_duplicates:
                s = s[~s.index.duplicated(keep="last")]
            if isinstance(s.index, pd.MultiIndex):
                s.index = s.index.set_names(["Date", "Code"])
            raw_metrics[col] = s.sort_index()
        else:
            raw_metrics[col] = pd.Series(dtype=float, name=col)

    # Build close_map_to_series for peak/distance factors
    close_mi_series = close_map_to_series(close_map) if close_map else pd.Series(dtype=float)
    # cyq_chips 成本价为复权口径,close 折算到同一空间后供 peak 类因子使用(见 _build_close_maps)
    close_adj_mi_series = close_map_to_series(close_adj_map) if close_adj_map else pd.Series(dtype=float)

    print(f"  Metrics concatenated  ({time.perf_counter() - t_concat:.1f}s)")

    # ── Phase 3: Factor frame assembly ───────────────────────────────────
    t_phase3 = time.perf_counter()
    print(f"\n  Phase 3/4 — Building factor frames")

    output: dict[str, pd.DataFrame] = {}
    errors_build: list[str] = []
    fac_elapsed: dict[str, float] = {}  # per-factor assembly time (seconds)

    pbar_factor = tqdm(
        total=total_factors, desc="  Factors", unit="fac",
        position=0, leave=True,
        bar_format="{desc:>12}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}] {postfix}",
    )

    for idx, name in enumerate(factor_names):
        pbar_factor.set_postfix_str(f"{name}")
        pbar_factor.update(0)  # refresh
        t_fac = time.perf_counter()

        if name not in CHIP_FACTOR_SPEC:
            output[name] = pd.DataFrame()
            errors_build.append(name)
            pbar_factor.update(1)
            continue

        metric_col, direction = CHIP_FACTOR_SPEC[name]
        raw = raw_metrics.get(metric_col)
        if raw is None or raw.empty:
            output[name] = pd.DataFrame()
            pbar_factor.update(1)
            continue

        try:
            if direction == "momentum":
                chg = raw.groupby(level="Code").transform(lambda s: s.diff(5))
                ranked = cross_sectional_rank(chg)
            elif direction == "momentum_rev":
                chg = raw.groupby(level="Code").transform(lambda s: s.diff(5))
                ranked = cross_sectional_rank(-chg)
            elif direction == "momentum_20d":
                chg = raw.groupby(level="Code").transform(lambda s: s.diff(20))
                ranked = cross_sectional_rank(chg)
            elif direction == "peak":
                common = raw.index.intersection(close_adj_mi_series.index)
                distance = (
                    close_adj_mi_series.loc[common] - raw.loc[common]
                ) / close_adj_mi_series.loc[common].replace(0, np.nan)
                ranked = cross_sectional_rank(distance)
            elif direction == "pos":
                ranked = cross_sectional_rank(raw)
            elif direction == "neg":
                ranked = cross_sectional_rank(-raw)
            else:
                ranked = cross_sectional_rank(raw)

            frame = ensure_single_factor_frame(ranked, name, paths=paths)
            output[name] = frame
            fac_elapsed[name] = time.perf_counter() - t_fac
        except Exception:
            logger.exception("%s: Phase 3 assembly failed", name)
            output[name] = pd.DataFrame()
            errors_build.append(name)

        pbar_factor.update(1)

    pbar_factor.close()

    # ── Incremental: keep only new dates in output frames ──────────────────
    # Factor computation needs full lookback (momentum diff(5)/diff(20)), but
    # we only write dates that don't already exist in the stored .fea files.
    if not force and earliest_existing and min_trade_date is not None:
        for name in list(output.keys()):
            frame = output.get(name)
            if frame is None or frame.empty:
                continue
            frame = frame[frame.index > earliest_existing]
            if frame.empty:
                output[name] = pd.DataFrame()
            else:
                output[name] = frame

    t_phase3_elapsed = time.perf_counter() - t_phase3
    rate_p3 = total_factors / t_phase3_elapsed if t_phase3_elapsed > 0 else 0
    print(f"  Phase 3 done  ({t_phase3_elapsed:.1f}s, {rate_p3:.1f} factors/s)")

    # ── Phase 4: Writing output files ────────────────────────────────────
    t_phase4 = time.perf_counter()
    print(f"\n  Phase 4/4 — Writing .fea output files")

    results: list = []
    write_errors = 0

    pbar_write = tqdm(
        total=total_factors, desc="  Write", unit="file",
        position=0, leave=True,
        bar_format="{desc:>12}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}] {postfix}",
    )

    for name in factor_names:
        frame = output.get(name)
        factor_path = paths.factor_output_dir / f"{name}.fea"

        if frame is None or frame.empty:
            results.append(BuildResult(
                factor_name=name, action="error", elapsed=0.0, rows=0,
                factor_path=factor_path,
                manifest_path=paths.manifest_output_dir / f"{name}.json",
            ))
            write_errors += 1
            pbar_write.set_postfix_str(f"ERR {name}")
            pbar_write.update(1)
            continue
        spec = get_factor(name)
        try:
            # Decide full-rebuild vs incremental write.
            # Quality check issues (all_nan, missing_dates, unreadable)
            # are handled by write_factor_incremental which merges new
            # data over the problematic old rows (keep="last"); a full
            # overwrite would discard all historical data.
            t_write = time.perf_counter()
            if not factor_path.exists() or min_trade_date is None:
                write_factor(spec, frame, paths=paths)
                action = "rebuild"
            else:
                write_factor_incremental(spec, frame, paths=paths)
                action = "incremental"
            # Record the factor's generation time (seconds) in its manifest
            record_build_elapsed(
                paths.manifest_output_dir / f"{name}.json",
                fac_elapsed.get(name, 0.0) + (time.perf_counter() - t_write),
            )
            results.append(BuildResult(
                factor_name=name, action=action, elapsed=0.0,
                rows=len(frame), factor_path=factor_path,
                manifest_path=paths.manifest_output_dir / f"{name}.json",
            ))
            pbar_write.set_postfix_str(f"OK {name}")
        except Exception:
            logger.exception("%s: Phase 4 write failed", name)
            results.append(BuildResult(
                factor_name=name, action="error", elapsed=0.0, rows=0,
                factor_path=factor_path,
                manifest_path=paths.manifest_output_dir / f"{name}.json",
            ))
            write_errors += 1
            pbar_write.set_postfix_str(f"ERR {name}")

        pbar_write.update(1)

    pbar_write.close()

    t_phase4_elapsed = time.perf_counter() - t_phase4
    rate_p4 = total_factors / t_phase4_elapsed if t_phase4_elapsed > 0 else 0
    print(f"  Phase 4 done  ({t_phase4_elapsed:.1f}s, {rate_p4:.1f} files/s)")

    # ── Final summary ────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_phase1
    built = sum(1 for r in results if r.action == "rebuild")
    skipped = sum(1 for r in results if r.action == "skip")
    errors_total = len(errors_build) + write_errors

    print(f"\n{'─'*64}")
    print(f"  BUILD COMPLETE")
    print(f"  Total time:        {total_elapsed:.1f}s")
    print(f"  Stocks processed:  {total_stocks}")
    print(f"  Factors built:     {built}  |  Skipped: {skipped}  |  Errors: {errors_total}")
    print(f"  Phase breakdown:")
    print(f"    Phase 1 (discovery):  {t_phase1_elapsed:>6.1f}s")
    print(f"    Phase 2 (metrics):    {t_phase2_elapsed:>6.1f}s  ({rate_p2:.0f} stocks/s)")
    print(f"    Phase 3 (assembly):   {t_phase3_elapsed:>6.1f}s  ({rate_p3:.1f} factors/s)")
    print(f"    Phase 4 (write):      {t_phase4_elapsed:>6.1f}s  ({rate_p4:.1f} files/s)")
    if errors_total:
        print(f"  Error list: {', '.join(errors_build)}")
    print(f"{'─'*64}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: single-factor directory reader (for individual factor calls)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_chip_factor(
    source_root: Path,
    allowed_codes: set[str],
    metric_col: str,
    close_map: dict[str, pd.Series] | None = None,
    close_adj_map: dict[str, pd.Series] | None = None,
    on_progress=None,
) -> pd.Series:
    """Compute one chip metric for all stocks from per-stock cyq_chips/ files.

    close_adj_map 提供折算到 cyq 成本价口径的 close(见 _build_close_maps),
    供 chip_below_ratio 等需要与复权价格网格比较的指标使用。
    """
    chip_dir = source_root / "cyq_chips"
    if not chip_dir.exists():
        raise FileNotFoundError(f"cyq_chips directory not found: {chip_dir}")

    files = sorted([f for f in os.listdir(chip_dir) if f.endswith(".parquet")])
    parts = []
    total = len(files)

    for i, fname in enumerate(files):
        code = fname.replace(".parquet", "").split(".")[0].zfill(6)
        if allowed_codes and code not in allowed_codes:
            if on_progress:
                on_progress("chip", i + 1, total)
            continue

        filepath = chip_dir / fname
        try:
            stock_df = pd.read_parquet(filepath)
        except Exception:
            if on_progress:
                on_progress("chip", i + 1, total)
            continue

        if stock_df.empty:
            if on_progress:
                on_progress("chip", i + 1, total)
            continue

        try:
            close_adj_s = close_adj_map.get(code) if close_adj_map else (
                close_map.get(code) if close_map else None
            )
            daily = _chip_daily_metrics(stock_df, close_series=close_adj_s)
        except Exception:
            if on_progress:
                on_progress("chip", i + 1, total)
            continue

        if metric_col not in daily.columns:
            if on_progress:
                on_progress("chip", i + 1, total)
            continue

        s = daily[metric_col].dropna()
        if s.empty:
            if on_progress:
                on_progress("chip", i + 1, total)
            continue

        s = pd.DataFrame(
            {metric_col: s.values},
            index=pd.MultiIndex.from_arrays(
                [s.index, [code] * len(s)], names=["Date", "Code"]
            ),
        )[metric_col]
        parts.append(s)

        if on_progress:
            on_progress("chip", i + 1, total)

    if not parts:
        return pd.Series(dtype=float, name=metric_col)

    result = pd.concat(parts)
    result.index = result.index.set_names(["Date", "Code"])
    result = result.reorder_levels(["Date", "Code"]).sort_index()
    # Deduplicate: keep last value for each (Date, Code) pair.
    # groupby().last() can coerce a Series -> DataFrame in edge cases
    # (e.g. single-date data), so use boolean indexing instead.
    dup = result.index.duplicated(keep="last")
    if dup.any():
        result = result[~dup]
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.set_names(["Date", "Code"])
    # Guarantee Series return — groupby().last() may return DataFrame
    if isinstance(result, pd.DataFrame):
        result = result.squeeze(axis=1)
    return result


def _load_close_maps(source_root: Path, allowed_codes: set[str]) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Load (close_map, close_adj_map) from daily.parquet, keyed by 6-digit code.

    两者均为未复权 close(不再做复权折算,见 _build_close_maps / chip._close_adj_basis
    的说明——K 末值锚定跨除权日会使历史值随构建日改写,已移除)。
    """
    daily_path = source_root / "daily.parquet"
    close_df = pd.read_parquet(
        daily_path, columns=["trade_date", "stock_code", "close", "pct_chg"]
    )
    close_df["trade_date"] = close_df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    close_df["stock_code"] = close_df["stock_code"].apply(_pad_code)
    if allowed_codes:
        close_df = close_df[close_df["stock_code"].isin(allowed_codes)]
    return _build_close_maps(close_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Registered factors
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="chip_peak_distance",
    description="筹码峰距离因子，(close-chip_peak_price)/close截面排名。价格在最大筹码峰上方=支撑排前。",
    category="price",
    thesis="最大筹码峰是最密集的持仓成本区，价格在峰上方时有强支撑，在峰下方时变为强阻力。",
    dependencies=("cyq_chips", "daily.parquet"),
)
def factor_chip_peak_distance(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress

    peak_series = _compute_chip_factor(
        source_root, allowed, "chip_peak_price",
        on_progress=on_progress,
    )

    daily_panel = context.load("daily.parquet")
    # cyq_chips 成本价为复权口径,把 close 折算到同一空间后再比较(见 chip._close_adj_basis)
    close_adj = _close_adj_basis(daily_panel)
    common = close_adj.index.intersection(peak_series.index)
    distance = (close_adj.loc[common] - peak_series.loc[common]) / close_adj.loc[common].replace(0, np.nan)
    return cross_sectional_rank(distance)


@register_factor(
    name="chip_peak_ratio",
    description="下方筹码占比因子，当前价格以下的筹码面积占总筹码面积比例截面排名。",
    category="price",
    thesis="价格下方的筹码面积占比越高，意味着越多的持仓者处于盈利状态、且下方支撑越密集。",
    dependencies=("cyq_chips", "daily.parquet"),
)
def factor_chip_peak_ratio(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress

    close_map, close_adj_map = _load_close_maps(source_root, allowed)
    ratio_series = _compute_chip_factor(
        source_root, allowed, "chip_below_ratio",
        close_map=close_map, close_adj_map=close_adj_map, on_progress=on_progress,
    )
    return cross_sectional_rank(ratio_series)


@register_factor(
    name="chip_below_momentum",
    description="下方筹码动量因子，chip_peak_ratio的5日变化截面排名。",
    category="price",
    thesis="下方筹码占比的快速增加意味着价格在向上突破——越来越多的筹码从上方套牢转为下方盈利。",
    dependencies=("cyq_chips", "daily.parquet"),
)
def factor_chip_below_momentum(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress

    close_map, close_adj_map = _load_close_maps(source_root, allowed)
    ratio_series = _compute_chip_factor(
        source_root, allowed, "chip_below_ratio",
        close_map=close_map, close_adj_map=close_adj_map, on_progress=on_progress,
    )
    if isinstance(ratio_series, pd.DataFrame):
        ratio_series = ratio_series.squeeze(axis=1)
    chg = ratio_series.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(chg)


# ── Distribution shape: static rank ─────────────────────────────────────────

@register_factor(
    name="chip_dispersion",
    description="筹码分布宽度因子，chip_weighted_std截面排名（取负=窄分布排前）。窄分布=筹码集中=一致预期强。",
    category="price",
    thesis="分布宽度（加权标准差）衡量芯片在价格轴上的分散程度。标准差小=所有持仓成本集中在窄区间=筹码结构紧凑、主力控盘能力强。与(cost_95pct-cost_5pct)/cost_50pct不同，标准差使用全部分布计算，对离群价格敏感。",
    dependencies=("cyq_chips",),
)
def factor_chip_dispersion(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_weighted_std", on_progress=on_progress)
    return cross_sectional_rank(-s)


@register_factor(
    name="chip_skew_factor",
    description="筹码偏度因子，分布偏度截面排名。右偏（正偏）=上方筹码多=牛市中换手充分、趋势惯性。",
    category="price",
    thesis="偏度衡量筹码分布的对称性。右偏=均价上方有更多筹码分布（上升趋势中换手活跃），通常出现在上升趋势中。左偏=均价下方筹码堆积=套牢盘重。与cost_distribution_skew（仅用3个分位点估计）互补——此因子使用全部分布精确计算。",
    dependencies=("cyq_chips",),
)
def factor_chip_skew_factor(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_skewness", on_progress=on_progress)
    return cross_sectional_rank(s)


@register_factor(
    name="chip_entropy_signal",
    description="筹码熵信号因子，分布信息熵截面排名（取负=低熵排前）。低熵=结构有序=主力控盘明显。",
    category="price",
    thesis="信息熵衡量筹码分布的无序程度。低熵=筹码集中在少数价格水平（有清晰主峰）=市场参与者对估值区间形成共识。高熵=筹码均匀分散在各价位=多空意见分歧大、缺乏明确方向。",
    dependencies=("cyq_chips",),
)
def factor_chip_entropy_signal(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_entropy", on_progress=on_progress)
    return cross_sectional_rank(-s)


@register_factor(
    name="chip_peak_purity",
    description="筹码主峰纯度因子，最大峰占总筹码比例截面排名。主峰突出=清晰的价格锚点=有效的支撑/阻力位。",
    category="price",
    thesis="最大筹码峰占比越高，意味着最多的投资者在同一个价格附近持有筹码。这个价格因此具备最强的支撑/阻力含义。高峰占比=明确的价格锚定=突破该价位后的趋势确定性更强。",
    dependencies=("cyq_chips",),
)
def factor_chip_peak_purity(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_peak_dominance", on_progress=on_progress)
    return cross_sectional_rank(s)


@register_factor(
    name="chip_tail_risk",
    description="筹码尾部风险因子，分布超额峰度截面排名（取负=低峰度排前）。高峰度=肥尾=远离均价的极端筹码堆积、异动风险高。",
    category="price",
    thesis="超额峰度为正（尖峰+肥尾）时，筹码在远离均价的位置有额外堆积。正峰度=大量套牢盘或获利盘堆积在极端价位，当价格靠近时会有剧烈涌出。负峰度（低峰）=筹码分布较为均匀。",
    dependencies=("cyq_chips",),
)
def factor_chip_tail_risk(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_kurtosis", on_progress=on_progress)
    return cross_sectional_rank(-s)


@register_factor(
    name="chip_iqr_factor",
    description="筹码四分位距因子，分布IQR（P75-P25）截面排名（取负=窄IQR排前）。窄IQR=核心50%筹码高度集中。",
    category="price",
    thesis="四分位距仅衡量中间50%筹码的分布宽度，剔除上下25%极端值的影响。IQR窄=主力持仓成本高度集中在一个小区间=突破时的合力和方向确定性更强。与chip_dispersion互补：IQR对尾部异常值不敏感。",
    dependencies=("cyq_chips",),
)
def factor_chip_iqr_factor(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_iqr", on_progress=on_progress)
    return cross_sectional_rank(-s)


# ── Distribution shape: momentum (5-day change) ─────────────────────────────

@register_factor(
    name="chip_dispersion_momentum",
    description="筹码宽度动量因子，chip_weighted_std的5日变化截面排名（取负=收窄排前）。宽度收窄=筹码凝聚=突破前兆。",
    category="price",
    thesis="分布宽度的变化方向比绝对宽度更有前瞻性。宽度快速收窄=资金集中+筹码被持续吸收，是典型的主力收集特征。宽度扩张=筹码分散+出货信号。变化的边际意义大于静态状态。",
    dependencies=("cyq_chips",),
)
def factor_chip_dispersion_momentum(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_weighted_std", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(-chg)


@register_factor(
    name="chip_skew_momentum",
    description="筹码偏度动量因子，分布偏度的5日变化截面排名。偏度右移=筹码重心上移+趋势延续。",
    category="price",
    thesis="偏度的变化比静态偏度更有信号价值。偏度从负转正（或正向加大）=筹码重心向高价区移动=上升趋势有内在惯性。偏度从正转负=趋势可能转向——分布正在向低价区倾斜。",
    dependencies=("cyq_chips",),
)
def factor_chip_skew_momentum(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_skewness", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(chg)


@register_factor(
    name="chip_entropy_convergence",
    description="筹码熵收敛因子，分布熵的5日变化截面排名（取负=熵降排前）。熵下降=信息收敛=预期正在趋于一致。",
    category="price",
    thesis="熵值下降=筹码分布从无序到有序——市场对合理价格区间正在形成共识，这是价格突破前的一个重要信号。连续多日熵降往往预示趋势性行情即将启动。",
    dependencies=("cyq_chips",),
)
def factor_chip_entropy_convergence(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_entropy", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(-chg)


@register_factor(
    name="chip_peak_growing",
    description="筹码主峰增强因子，主峰占比的5日变化截面排名。主峰增强=资金持续在核心价位收集筹码。",
    category="price",
    thesis="主峰占比上升=越来越多的筹码集中到同一价格区间——主力资金在固定的核心价位持续吸筹。主峰强度连续上升往往是拉升前的重要信号。",
    dependencies=("cyq_chips",),
)
def factor_chip_peak_growing(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_peak_dominance", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(chg)


@register_factor(
    name="chip_tail_risk_change",
    description="筹码尾部风险变化因子，分布峰度的5日变化截面排名（取负=尾部收窄排前）。尾部收窄=极端筹码被消化=结构改善。",
    category="price",
    thesis="峰度变化反映尾部风险的边际变化。峰度下降（超额峰度减小）=极端价位的筹码正在被消化=尾部风险降低+筹码结构改善。峰度上升=需警惕极端价位附近的筹码压力积累。",
    dependencies=("cyq_chips",),
)
def factor_chip_tail_risk_change(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_kurtosis", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(-chg)


# ── Distribution shape: close-distance ──────────────────────────────────────

@register_factor(
    name="chip_mean_distance",
    description="筹码均价距离因子，(close-chip_weighted_mean)/close截面排名。价格在加权平均成本上方=多数盈利+强支撑。",
    category="price",
    thesis="与chip_peak_distance（模态成本）类似，但使用全部分布的加权平均成本——这是比模态峰价更稳健的平均持仓成本估计。价格高于均价=多数持仓者盈利=下方支撑可靠。",
    dependencies=("cyq_chips", "daily.parquet"),
)
def factor_chip_mean_distance(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    mean_series = _compute_chip_factor(
        source_root, allowed, "chip_weighted_mean", on_progress=on_progress,
    )
    daily_panel = context.load("daily.parquet")
    close = daily_panel["close"]
    common = close.index.intersection(mean_series.index)
    distance = (close.loc[common] - mean_series.loc[common]) / close.loc[common].replace(0, np.nan)
    return cross_sectional_rank(distance)


# ── 2nd batch: additional distribution statistics — static rank ─────────────

@register_factor(
    name="chip_cv_factor",
    description="筹码变异系数因子，chip_cv=std/mean截面排名（取负=低CV排前）。低CV=分布相对均值集中=风险可控。",
    category="price",
    thesis="变异系数将标准差按均价归一化，消除股价水平的影响。低CV=筹码在均价周围的相对离散度小、持仓成本一致性强。与chip_dispersion互补：CV跨股票可比、chip_dispersion保留价格量纲。",
    dependencies=("cyq_chips",),
)
def factor_chip_cv_factor(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_cv", on_progress=on_progress)
    return cross_sectional_rank(-s)


@register_factor(
    name="chip_cr3_factor",
    description="筹码CR3集中度因子，前三峰筹码占比截面排名。CR3高=多个价格区间筹码集中=多级支撑结构。",
    category="price",
    thesis="CR3衡量前三高筹码峰的总占比。与单峰纯度(chip_peak_purity)不同，CR3能识别多峰分布中的集中度——即使分布是多峰的，只要前三个峰合计占比高，说明筹码仍集中。CR3>0.6通常意味着三峰已覆盖大部分持仓成本。",
    dependencies=("cyq_chips",),
)
def factor_chip_cr3_factor(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_cr3", on_progress=on_progress)
    return cross_sectional_rank(s)


@register_factor(
    name="chip_gini_factor",
    description="筹码基尼系数因子，分布Gini系数截面排名。高Gini=筹码集中在少数价位=价格锚定清晰。",
    category="price",
    thesis="基尼系数是经典的不平等度量。高Gini=少数价位占有大部分筹码=市场对核心成本区形成强共识。低Gini=筹码均匀分布在各价位=多空分歧大。与熵信息互补：Gini对均匀分布更敏感。",
    dependencies=("cyq_chips",),
)
def factor_chip_gini_factor(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_gini", on_progress=on_progress)
    return cross_sectional_rank(s)


@register_factor(
    name="chip_mode_mean_gap",
    description="筹码众数均值偏离因子，|peak_mean|/mean截面排名（取负=缺口小排前）。缺口小=分布对称+单峰结构健康。",
    category="price",
    thesis="众数（筹码峰价格）与加权均值的差距衡量分布的对称性。缺口大=分布非对称/多峰——筹码结构不统一，可能有两组投资者在博弈。缺口小=单峰对称的分布、价格锚定清晰。与众数→右偏不同，此指标捕捉的是局部的结构不一致。",
    dependencies=("cyq_chips",),
)
def factor_chip_mode_mean_gap(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_mode_mean_gap", on_progress=on_progress)
    return cross_sectional_rank(-s)


@register_factor(
    name="chip_p90_p10_factor",
    description="筹码P90-P10范围因子，分布90%筹码价格区间宽度截面排名（取负=窄区间排前）。窄区间=核心筹码高度重叠。",
    category="price",
    thesis="P90-P10衡量90%筹码的价格宽度（剔除上下5%极端值）。比IQR更宽地覆盖分布、比全距更稳健。窄P90-P10=绝大多数持仓成本集中于小区间=共识基础扎实。",
    dependencies=("cyq_chips",),
)
def factor_chip_p90_p10_factor(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_p90_p10", on_progress=on_progress)
    return cross_sectional_rank(-s)


@register_factor(
    name="chip_median_distance",
    description="筹码中位数距离因子，(close-chip_median_price)/close截面排名。价格在持仓成本中位数上方=多数盈利+支撑。",
    category="price",
    thesis="使用中位数成本（而非众数峰价或加权均值）来度量价格偏离。中位数对极端值不敏感，是更稳健的'典型成本'估计。价格高于中位数成本意味着超过一半的持仓者盈利。与chip_peak_distance和chip_mean_distance形成估计方法上的三角互补。",
    dependencies=("cyq_chips", "daily.parquet"),
)
def factor_chip_median_distance(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    median_series = _compute_chip_factor(
        source_root, allowed, "chip_median_price", on_progress=on_progress,
    )
    daily_panel = context.load("daily.parquet")
    close = daily_panel["close"]
    common = close.index.intersection(median_series.index)
    distance = (close.loc[common] - median_series.loc[common]) / close.loc[common].replace(0, np.nan)
    return cross_sectional_rank(distance)


# ── 2nd batch: additional distribution statistics — momentum ────────────────

@register_factor(
    name="chip_cv_momentum",
    description="筹码CV动量因子，变异系数的5日变化截面排名（取负=CV降排前）。CV收窄=相对离散度降低=筹码趋于集中。",
    category="price",
    thesis="变异系数的变化消除了股价波动的影响。CV快速收窄=筹码在均价周围的相对分散度缩小=持仓成本趋同=主力吸筹信号。比绝对宽度的变化更干净，特别是对于股价波动较大的标的。",
    dependencies=("cyq_chips",),
)
def factor_chip_cv_momentum(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_cv", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(-chg)


@register_factor(
    name="chip_cr3_momentum",
    description="筹码CR3动量因子，前三峰占比的5日变化截面排名。CR3上升=筹码向多个核心锚点集中=结构性收集。",
    category="price",
    thesis="CR3上升意味着分布在多个价位形成支撑点——不是单一主力在吸筹，而是多路资金在多个价格区间同时收集。CR3快速上升往往预示更为稳健的多级支撑结构。",
    dependencies=("cyq_chips",),
)
def factor_chip_cr3_momentum(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_cr3", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(chg)


@register_factor(
    name="chip_gini_momentum",
    description="筹码基尼动量因子，Gini系数的5日变化截面排名。Gini上升=筹码向少数价位集中=均衡被打破、新共识形成中。",
    category="price",
    thesis="Gini系数上升（不平等度增加）=筹码从均匀分布转向集中分布=市场在淘汰'非共识'价位上的筹码。Gini的边际变化通常比绝对水平更有信号意义——快速的Gini上升是个强烈的趋势信号。",
    dependencies=("cyq_chips",),
)
def factor_chip_gini_momentum(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_gini", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(chg)


@register_factor(
    name="chip_mode_mean_convergence",
    description="筹码众均收敛因子，|peak-mean|/mean的5日变化截面排名（取负=缺口收窄排前）。众数均值趋于一致=分布从多峰向单峰收敛。",
    category="price",
    thesis="众数-均值缺口收窄=分布结构从复杂（多峰、非对称）走向简单（单峰、对称）——这是筹码结构从混乱走向有序的过程。缺口的快速收窄意味着市场分歧正在消除，主力筹码正在形成统一的价格锚。",
    dependencies=("cyq_chips",),
)
def factor_chip_mode_mean_convergence(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_mode_mean_gap", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(-chg)


@register_factor(
    name="chip_p90_p10_momentum",
    description="筹码P90-P10动量因子，90%筹码范围的5日变化截面排名（取负=收窄排前）。核心区间收窄=筹码进一步浓缩。",
    category="price",
    thesis="P90-P10的收窄是筹码浓缩最直观的体现——90%的持仓成本区间缩小，说明高成本的套牢盘和低成本的获利盘正在被消化，筹码向中心价位收敛。范围快速缩小往往是强烈趋势的前奏。",
    dependencies=("cyq_chips",),
)
def factor_chip_p90_p10_momentum(context: FactorContext):
    source_root = context.repo.paths.source_root
    allowed = context.repo.allowed_codes
    on_progress = context.repo.on_progress
    s = _compute_chip_factor(source_root, allowed, "chip_p90_p10", on_progress=on_progress)
    chg = s.groupby(level="Code").transform(lambda x: x.diff(5))
    return cross_sectional_rank(-chg)


# ── Chip concentration dynamics ────────────────────────────────────────────

@register_factor(
    name="chip_concentration_change_20d",
    description="筹码集中度20日变化因子，-(cost_95pct-cost_5pct)/cost_50pct的20日变化截面排名（凝聚=正向排前）。",
    category="price",
    thesis="筹码集中度的边际变化比绝对集中度更重要——筹码正在凝聚(区间收窄)意味着主力正在控制筹码，后续突破概率增大。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_concentration_change_20d(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    concentration = -(cyq["cost_95pct"] - cyq["cost_5pct"]) / cyq["cost_50pct"].replace(0, np.nan)
    chg = concentration.groupby(level="Code").transform(lambda s: s.diff(20))
    return cross_sectional_rank(chg)


@register_factor(
    name="winner_rate_acceleration",
    description="获利盘比例加速度因子，winner_rate的5日变化截面排名。",
    category="price",
    thesis="获利盘比例的快速变化反映筹码结构的急剧变化——winner_rate快速上升=大量持仓者从亏损转为盈利，可能触发获利了结；快速下降=恐慌抛售导致深套，可能形成支撑位。",
    dependencies=("cyq_perf.parquet",),
)
def factor_winner_rate_acceleration(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    wr = cyq["winner_rate"]
    chg_5 = wr.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(chg_5)


@register_factor(
    name="cost_displacement_extreme",
    description="成本偏离极端度因子，(close-weight_avg)/weight_avg的绝对值截面排名（极度偏离=均值回归压力大排前）。",
    category="price",
    thesis="价格大幅偏离加权平均成本后存在均值回归倾向——无论是大幅盈利(上方偏离)还是大幅亏损(下方偏离)，都有筹码驱动的回归压力。逆向排名：偏离越大越倾向于反转。",
    dependencies=("cyq_perf.parquet", "daily.parquet"),
)
def factor_cost_displacement_extreme(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    daily_panel = context.load("daily.parquet")
    # cyq 成本价为复权口径,把 close 折算到同一空间后再比较(见 chip._close_adj_basis)
    close_adj = _close_adj_basis(daily_panel)
    cost_avg = cyq["weight_avg"]
    common = close_adj.index.intersection(cost_avg.index)
    displacement = (close_adj.loc[common] - cost_avg.loc[common]).abs() / cost_avg.loc[common].replace(0, np.nan)
    # Positive rank for high displacement = high reversion probability (reversal signal)
    return cross_sectional_rank(displacement)


@register_factor(
    name="chip_profit_loss_ratio",
    description="盈亏筹码比因子，(close-cost_95pct)/(cost_5pct-close)截面排名。",
    category="price",
    thesis="上方筹码(套牢盘)与下方筹码(获利盘)的相对比例——比率高=上方套牢盘重(负向)、上升阻力大；比率低=下方获利盘多(正向)、有支撑。",
    dependencies=("cyq_perf.parquet", "daily.parquet"),
)
def factor_chip_profit_loss_ratio(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    daily_panel = context.load("daily.parquet")
    # cyq 成本价为复权口径,把 close 折算到同一空间后再比较(见 chip._close_adj_basis)
    close_adj = _close_adj_basis(daily_panel)
    common = close_adj.index.intersection(cyq["cost_95pct"].index)
    upper = close_adj.loc[common] - cyq["cost_95pct"].loc[common]
    lower = cyq["cost_5pct"].loc[common] - close_adj.loc[common]
    ratio = lower / (upper.abs() + 0.01)
    return cross_sectional_rank(ratio)


@register_factor(
    name="chip_concentration_streak",
    description="筹码凝聚持续性因子，筹码集中度连续改善天数截面排名。",
    category="price",
    thesis="筹码连续凝聚是主力收集筹码过程的表现——凝聚趋势的持续性比单日凝聚程度更重要，连凝天数多=主力持续在收集。",
    dependencies=("cyq_perf.parquet",),
)
def factor_chip_concentration_streak(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    concentration = -(cyq["cost_95pct"] - cyq["cost_5pct"]) / cyq["cost_50pct"].replace(0, np.nan)
    improving = (concentration.groupby(level="Code").transform(lambda s: s.diff(1)) > 0).astype(int)

    def _count_streak(x):
        x = x.values
        streak = np.zeros_like(x, dtype=float)
        cnt = 0
        for i in range(len(x)):
            if x[i] == 1:
                cnt += 1
            else:
                cnt = 0
            streak[i] = cnt
        return streak

    streak = improving.groupby(level="Code").transform(_count_streak)
    return cross_sectional_rank(streak)


@register_factor(
    name="cost_distribution_skew",
    description="成本分布偏度因子，(cost_50pct-cost_15pct)/(cost_85pct-cost_50pct)截面排名。",
    category="price",
    thesis="成本分布左右偏度反映筹码的'重心'偏向——右偏(上方筹码多)=套牢盘重、上升压力大；左偏(下方筹码多)=获利盘多、上升有支撑。",
    dependencies=("cyq_perf.parquet",),
)
def factor_cost_distribution_skew(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    left_tail = cyq["cost_50pct"] - cyq["cost_15pct"]
    right_tail = cyq["cost_85pct"] - cyq["cost_50pct"]
    skew = left_tail / right_tail.replace(0, np.nan)
    return cross_sectional_rank(skew)


@register_factor(
    name="winner_rate_reversal_signal",
    description="获利盘极端反转因子，-|winner_rate-0.5|截面排名（50%附近=方向不确定=不确定溢价排前）。",
    category="price",
    thesis="获利盘比例在50%附近时筹码结构最为平衡——多空双方势均力敌，任何方向突破都可能是大行情的起点。极端获利(>90%)或极端亏损(<10%)则方向确定但反转压力也最大。",
    dependencies=("cyq_perf.parquet",),
)
def factor_winner_rate_reversal_signal(context: FactorContext):
    cyq = context.load("cyq_perf.parquet")
    distance = -(cyq["winner_rate"] - 0.5).abs()
    return cross_sectional_rank(distance)
