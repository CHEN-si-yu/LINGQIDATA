"""
因子评估与筛选模块

提供多维度的因子质量评估与回测框架:

1. 截面 IC 评估
   - Rank IC (Spearman)
   - Pearson IC
   - IC 均值 / 标准差 / ICIR / IC>0 占比 / t-stat

2. 分组收益评估
   - Top/Bottom N% 多空收益
   - 流动性加权多空收益 (参考 fac_filt.py 的 1.5e9 逻辑)

3. 因子稳定性评估
   - IC 自相关 (IC decay)
   - IC 滚动窗口稳定性
   - 换手率 / 覆盖度

4. 因子相关性分析
   - 因子间截面相关性矩阵
   - 因子聚类 / 冗余检测

5. 因子筛选
   - 多条件综合排名
   - 按季度/时间段筛选
   - 去冗余筛选

用法::

    from featureengineering.factor_evaluation import (
        evaluate_factors,
        evaluate_single_factor,
        FactorEvalResult,
        FactorScreener,
    )

参考: fac_filt.py 的筛选逻辑 (IC + 多空收益方向调整 + 分季筛选)
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .settings import ProjectPaths, configure_paths

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_EVAL_METRICS = [
    "rank_ic_mean", "rank_ic_std", "rank_icir", "rank_ic_pos_ratio",
    "pearson_ic_mean", "pearson_ic_std", "pearson_icir", "pearson_ic_pos_ratio",
    "long_ret_top10pct", "short_ret_bot10pct", "long_short_spread_10pct",
    "long_ret_head_amt", "short_ret_tail_amt", "long_short_spread_amt",
    "coverage_ratio", "n_dates", "ic_stability", "ic_decay_5d",
]

# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class FactorEvalResult:
    """Single factor × single target evaluation result."""

    factor_name: str
    target_name: str
    method: str = "rank"

    # IC metrics
    n_dates: int = 0
    rank_ic_mean: float = float("nan")
    rank_ic_std: float = float("nan")
    rank_icir: float = float("nan")
    rank_ic_pos_ratio: float = float("nan")
    pearson_ic_mean: float = float("nan")
    pearson_ic_std: float = float("nan")
    pearson_icir: float = float("nan")
    pearson_ic_pos_ratio: float = float("nan")

    # Return metrics (top/bottom 10%)
    long_ret_top10pct: float = float("nan")
    short_ret_bot10pct: float = float("nan")
    long_short_spread_10pct: float = float("nan")

    # Liquidity-weighted return metrics
    long_ret_head_amt: float = float("nan")
    short_ret_tail_amt: float = float("nan")
    long_short_spread_amt: float = float("nan")

    # Stability
    ic_stability: float = float("nan")   # IC std of rolling 20d mean IC
    ic_decay_5d: float = float("nan")    # autocorrelation of IC at lag 5
    coverage_ratio: float = float("nan")

    # Raw IC series for downstream use
    ic_series: pd.Series | None = None

    @property
    def abs_rank_icir(self) -> float:
        return abs(self.rank_icir) if not np.isnan(self.rank_icir) else float("nan")

    @property
    def abs_long_short_amt(self) -> float:
        return abs(self.long_short_spread_amt) if not np.isnan(self.long_short_spread_amt) else float("nan")

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "ic_series"}
        return d

    def summary(self) -> str:
        lines = [
            f"Factor: {self.factor_name}  ×  Target: {self.target_name}",
            f"  N dates:        {self.n_dates}",
            f"  Rank  IC mean:  {self.rank_ic_mean:+.6f}   std: {self.rank_ic_std:.6f}   ICIR: {self.rank_icir:+.4f}   >0: {self.rank_ic_pos_ratio:.2%}",
            f"  Pear  IC mean:  {self.pearson_ic_mean:+.6f}   std: {self.pearson_ic_std:.6f}   ICIR: {self.pearson_icir:+.4f}   >0: {self.pearson_ic_pos_ratio:.2%}",
            f"  Top10% ret:     {self.long_ret_top10pct:+.6f}   Bot10% ret: {self.short_ret_bot10pct:+.6f}   Spread: {self.long_short_spread_10pct:+.6f}",
            f"  Amt-wt ret:     {self.long_ret_head_amt:+.6f}   Tail Amt: {self.short_ret_tail_amt:+.6f}   Spread: {self.long_short_spread_amt:+.6f}",
            f"  IC stability:   {self.ic_stability:.4f}   IC decay(5d): {self.ic_decay_5d:.4f}   Coverage: {self.coverage_ratio:.2%}",
        ]
        return "\n".join(lines)


@dataclass
class FactorScreener:
    """因子筛选器 —— 复刻 fac_filt.py 的筛选逻辑并增强。

    基础筛选流程:
    1. 按 Rank IC 正负分组，各取前 50%
    2. 若超出目标数量，按 abs(多空收益) 排序截断
    3. 支持额外过滤条件 (覆盖率、IC 稳定性等)
    """

    fac_num_target: int = 1400
    min_coverage: float = 0.5
    min_n_dates: int = 60

    def screen(
        self,
        results: list[FactorEvalResult],
        sort_metric: str = "abs_long_short_amt",
        remove_redundant: bool = False,
        corr_threshold: float = 0.85,
        paths: "ProjectPaths | None" = None,
        keep_metric: str = "abs_rank_icir",
    ) -> list[str]:
        """从评估结果中筛选因子，返回因子名称列表。

        Parameters
        ----------
        results : list[FactorEvalResult]
        sort_metric : str
            第二层排序指标: "abs_rank_icir" | "abs_long_short_amt"
        remove_redundant : bool
            是否进行相关性去冗余 (保留 |ICIR| 更高的因子)。
        corr_threshold : float
            相关性阈值，高于此值的因子对被认为是冗余的。
        paths : ProjectPaths or None
            用于加载因子数据计算相关性。
        keep_metric : str
            决定保留哪个因子的指标。
        """
        df = pd.DataFrame([r.to_dict() for r in results])

        # 基础过滤
        mask = (
            df["rank_icir"].notna()
            & (df["coverage_ratio"] >= self.min_coverage)
            & (df["n_dates"] >= self.min_n_dates)
        )
        df = df.loc[mask].copy()

        if df.empty:
            return []

        # 第一步：分正负 Rank IC，各取前 50%
        pos = df[df["rank_ic_mean"] > 0].sort_values("rank_icir", ascending=False)
        neg = df[df["rank_ic_mean"] < 0].sort_values("rank_icir", ascending=True)
        pos_top = pos.iloc[: max(1, len(pos) // 2)]
        neg_top = neg.iloc[: max(1, len(neg) // 2)]
        half = pd.concat([pos_top, neg_top])

        # 第二步：若超出目标数量，按收益强度排序截断
        if len(half) <= self.fac_num_target:
            selected = half["factor_name"].tolist()
        else:
            if sort_metric == "abs_long_short_amt":
                half = half.copy()
                half["_sort"] = half["long_short_spread_amt"].abs()
            else:
                half = half.copy()
                half["_sort"] = half["rank_icir"].abs()

            final = half.sort_values("_sort", ascending=False).iloc[: self.fac_num_target]
            selected = final["factor_name"].tolist()

        # Optional: redundancy removal (correlation-based deduplication)
        if remove_redundant:
            selected, removed = find_redundant_factors(
                selected,
                corr_threshold=corr_threshold,
                paths=paths,
                keep_metric=keep_metric,
                eval_results=df,
            )
            if removed:
                logger.info(
                    "Redundancy removal: dropped %d factors (corr >= %.2f)",
                    len(removed), corr_threshold,
                )
                for rm_fac, keep_fac, corr_val in removed[:5]:
                    logger.debug("  %s removed (kept %s, corr=%.3f)", rm_fac, keep_fac, corr_val)
                if len(removed) > 5:
                    logger.debug("  ... and %d more pairs", len(removed) - 5)

        return selected

    def rank(
        self,
        results: list[FactorEvalResult],
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """对因子进行加权综合排名。

        默认权重 (经验值，基于 fac_filt.py 的逻辑):
        - rank_icir: 0.30 (IC 信息比率最重要)
        - abs_long_short_spread_amt: 0.25 (多空收益)
        - ic_stability: 0.15 (IC 稳定性)
        - rank_ic_pos_ratio: 0.10 (IC 胜率)
        - ic_decay_5d: 0.10 (IC 衰减 → 负向，衰减越小越好)
        - coverage_ratio: 0.10 (覆盖度)
        """
        if weights is None:
            weights = {
                "rank_icir": 0.30,
                "abs_long_short_amt": 0.25,
                "ic_stability": 0.15,
                "rank_ic_pos_ratio": 0.10,
                "neg_ic_decay": 0.10,
                "coverage_ratio": 0.10,
            }

        df = pd.DataFrame([r.to_dict() for r in results])
        df = df[df["rank_icir"].notna()].copy()

        # 计算各项得分 (等权排名归一化到 [0, 1])
        score_cols = {}
        for metric, w in weights.items():
            if metric == "abs_long_short_amt":
                col = df["long_short_spread_amt"].abs().rank(pct=True)
            elif metric == "neg_ic_decay":
                # IC decay 越小越好 → 取负后越大越好
                col = (-df["ic_decay_5d"]).rank(pct=True)
            elif metric in df.columns:
                col = df[metric].rank(pct=True)
            else:
                continue
            score_cols[metric] = col * w

        df["composite_score"] = pd.DataFrame(score_cols).sum(axis=1)
        df = df.sort_values("composite_score", ascending=False)
        return df[
            ["factor_name", "target_name", "composite_score",
             "rank_icir", "long_short_spread_amt", "ic_stability",
             "coverage_ratio", "n_dates"]
        ]


# ── Core evaluation functions ────────────────────────────────────────────────


def _compute_single_date_metrics(
    fac_row: pd.Series,
    ret_row: pd.Series,
    amt_row: pd.Series | None = None,
    money: float = 1.5e9,
    min_stocks: int = 10,
) -> dict[str, float]:
    """Compute cross-sectional metrics for a single date.

    Parameters
    ----------
    fac_row : pd.Series
        Factor values indexed by stock code.
    ret_row : pd.Series
        Forward returns indexed by stock code.
    amt_row : pd.Series or None
        Daily trade amount (liquidity proxy) indexed by stock code.
    money : float
        Capital allocation budget for liquidity-weighted returns.
    min_stocks : int
        Minimum number of stocks required.

    Returns
    -------
    dict with metric_name → value
    """
    # Align
    common = fac_row.dropna().index.intersection(ret_row.dropna().index)
    if amt_row is not None:
        common = common.intersection(amt_row.dropna().index)

    if len(common) < min_stocks:
        return {}

    f = fac_row.reindex(common)
    r = ret_row.reindex(common)

    metrics = {}

    # Rank IC
    f_rank = f.rank(pct=True, method="average")
    r_rank = r.rank(pct=True, method="dense")
    metrics["rank_ic"] = f_rank.corr(r_rank, method="spearman")
    metrics["pearson_ic"] = f.corr(r, method="pearson")

    # Top/Bottom 10% returns
    head10 = f_rank > 0.9
    tail10 = f_rank < 0.1
    metrics["head10p_ret"] = (head10 * r).sum() / max(head10.sum(), 1)
    metrics["tail10p_ret"] = (tail10 * r).sum() / max(tail10.sum(), 1)

    # Liquidity-weighted returns (head=long tail=short, sign-adjusted later)
    if amt_row is not None:
        a = amt_row.reindex(common).fillna(0)
        ar = (a * r).to_frame("amt_ret")
        ar["amt"] = a
        ar["ret"] = r

        def _htamt_ret(code_list):
            sub = ar.reindex(code_list).fillna(0)
            sub["cum_amt"] = sub["amt"].cumsum()
            sub_ht = sub.loc[sub["cum_amt"] <= money]
            if sub_ht["amt"].sum() == 0:
                return 0.0
            return sub_ht["amt_ret"].sum() / sub_ht["amt"].sum()

        f_sorted = f.sort_values(ascending=False).dropna()
        metrics["head_amt_ret"] = _htamt_ret(f_sorted.index.tolist())

        f_sorted_asc = f.sort_values(ascending=True).dropna()
        metrics["tail_amt_ret"] = _htamt_ret(f_sorted_asc.index.tolist())

    return metrics


def evaluate_single_factor(
    factor_name: str,
    target_name: str,
    paths: ProjectPaths | None = None,
    amt_col: str | None = None,
    min_stocks: int = 10,
) -> FactorEvalResult:
    """评估单个因子对单个 target 的质量。

    Parameters
    ----------
    factor_name : str
        因子文件名 (不含 .fea)。
    target_name : str
        Target 文件名 (不含 .fea)。
    paths : ProjectPaths or None
    amt_col : str or None
        用于流动性加权的金额列名 (若为 None 则跳过流动性加权收益)。
    min_stocks : int
        每截面最低股票数。

    Returns
    -------
    FactorEvalResult
    """
    paths = paths or configure_paths()

    factor_path = paths.factor_output_dir / f"{factor_name}.fea"
    target_path = paths.target_output_dir / f"{target_name}.fea"

    if not factor_path.exists():
        logger.warning("Factor file not found: %s", factor_path)
        return FactorEvalResult(factor_name=factor_name, target_name=target_name)

    if not target_path.exists():
        logger.warning("Target file not found: %s", target_path)
        return FactorEvalResult(factor_name=factor_name, target_name=target_name)

    try:
        factor = pd.read_feather(factor_path)
        target = pd.read_feather(target_path)
    except Exception as e:
        logger.error("Failed to read %s: %s", factor_name, e)
        return FactorEvalResult(factor_name=factor_name, target_name=target_name)

    # Detect index column
    for idx_col in ["Date", "date", "index"]:
        if idx_col in factor.columns:
            factor = factor.set_index(idx_col)
            break
    for idx_col in ["Date", "date", "index"]:
        if idx_col in target.columns:
            target = target.set_index(idx_col)
            break

    # Optionally load liquidity data (same format as factors)
    amt = None
    if amt_col is not None:
        amt_path = paths.factor_output_dir / f"{amt_col}.fea"
        if amt_path.exists():
            amt = pd.read_feather(amt_path)
            for idx_col in ["Date", "date", "index"]:
                if idx_col in amt.columns:
                    amt = amt.set_index(idx_col)
                    break

    # Find common dates
    common_dates = factor.index.intersection(target.index).sort_values()
    if len(common_dates) == 0:
        return FactorEvalResult(factor_name=factor_name, target_name=target_name)

    # Compute per-date metrics
    date_metrics: list[dict] = []
    for date in common_dates:
        try:
            fac_row = factor.loc[date]
            ret_row = target.loc[date]
            amt_row = None
            if amt is not None and date in amt.index:
                amt_row = amt.loc[date]

            m = _compute_single_date_metrics(
                fac_row, ret_row, amt_row, min_stocks=min_stocks,
            )
            if m:
                m["date"] = date
                date_metrics.append(m)
        except Exception:
            continue

    if not date_metrics:
        return FactorEvalResult(factor_name=factor_name, target_name=target_name)

    dm = pd.DataFrame(date_metrics).set_index("date")

    # ── IC metrics ──
    rank_ic = dm.get("rank_ic", pd.Series(dtype=float)).dropna()
    pearson_ic = dm.get("pearson_ic", pd.Series(dtype=float)).dropna()

    def _ic_metrics(ic_series: pd.Series) -> dict:
        if len(ic_series) < 5:
            return {"mean": float("nan"), "std": float("nan"),
                    "icir": float("nan"), "pos_ratio": float("nan"), "n": 0}
        n = len(ic_series)
        mean = float(ic_series.mean())
        std = float(ic_series.std(ddof=1))
        icir = mean / std if std > 0 else float("nan")
        pos_ratio = float((ic_series > 0).mean())
        return {"mean": mean, "std": std, "icir": icir,
                "pos_ratio": pos_ratio, "n": n}

    rank_m = _ic_metrics(rank_ic)
    pear_m = _ic_metrics(pearson_ic)

    # ── Return metrics (sign-adjusted) ──
    # 参考 fac_filt.py: 若 IC > 0, head=long tail=short; 若 IC < 0, 互换
    sign = 1 if rank_m["mean"] >= 0 else -1
    head10p = dm.get("head10p_ret", pd.Series(dtype=float)).dropna()
    tail10p = dm.get("tail10p_ret", pd.Series(dtype=float)).dropna()
    long_10 = head10p.mean() if sign >= 0 else tail10p.mean()
    short_10 = tail10p.mean() if sign >= 0 else head10p.mean()

    head_amt = dm.get("head_amt_ret", pd.Series(dtype=float))
    tail_amt = dm.get("tail_amt_ret", pd.Series(dtype=float))
    if not head_amt.empty and not tail_amt.empty:
        head_amt = head_amt.dropna()
        tail_amt = tail_amt.dropna()
        long_amt = head_amt.mean() if sign >= 0 else tail_amt.mean()
        short_amt = tail_amt.mean() if sign >= 0 else head_amt.mean()
    else:
        long_amt = float("nan")
        short_amt = float("nan")

    # ── Stability metrics ──
    ic_stab = float("nan")
    ic_decay = float("nan")
    if len(rank_ic) >= 40:
        # Rolling 20-day IC std → lower is more stable
        roll_std = rank_ic.rolling(20).std().dropna()
        ic_stab = 1.0 / (1.0 + float(roll_std.mean()))  # [0, 1], higher = more stable
        # IC autocorrelation at lag 5
        if len(rank_ic) >= 25:
            ic_decay = rank_ic.autocorr(lag=5)
            ic_decay = float(ic_decay) if not np.isnan(ic_decay) else float("nan")

    # ── Coverage ──
    coverage = float(len(dm) / len(common_dates)) if len(common_dates) > 0 else 0.0

    return FactorEvalResult(
        factor_name=factor_name,
        target_name=target_name,
        n_dates=rank_m["n"],
        rank_ic_mean=rank_m["mean"],
        rank_ic_std=rank_m["std"],
        rank_icir=rank_m["icir"],
        rank_ic_pos_ratio=rank_m["pos_ratio"],
        pearson_ic_mean=pear_m["mean"],
        pearson_ic_std=pear_m["std"],
        pearson_icir=pear_m["icir"],
        pearson_ic_pos_ratio=pear_m["pos_ratio"],
        long_ret_top10pct=float(long_10) if not isinstance(long_10, pd.Series) else float(long_10.iloc[0]) if len(long_10) > 0 else float("nan"),
        short_ret_bot10pct=float(short_10) if not isinstance(short_10, pd.Series) else float(short_10.iloc[0]) if len(short_10) > 0 else float("nan"),
        long_short_spread_10pct=float(long_10 - short_10) if not isinstance(long_10, pd.Series) else float("nan"),
        long_ret_head_amt=float(long_amt),
        short_ret_tail_amt=float(short_amt),
        long_short_spread_amt=float(long_amt - short_amt) if not np.isnan(long_amt) and not np.isnan(short_amt) else float("nan"),
        ic_stability=ic_stab,
        ic_decay_5d=ic_decay,
        coverage_ratio=coverage,
        ic_series=rank_ic,
    )


def evaluate_factors(
    factor_names: list[str],
    target_names: list[str] | None = None,
    paths: ProjectPaths | None = None,
    max_workers: int | None = None,
    amt_col: str | None = None,
    min_stocks: int = 10,
    progress: bool = True,
) -> pd.DataFrame:
    """批量评估因子 × target 组合。

    Parameters
    ----------
    factor_names : list[str]
    target_names : list[str] or None
        Default: ["label_ret_1d", "label_ret_5d", "label_ret_10d", "label_ret_20d"]
    paths : ProjectPaths or None
    max_workers : int or None
        并行进程数。默认取 CPU 核数的一半。
    amt_col : str or None
        流动性代理列名。
    min_stocks : int
    progress : bool

    Returns
    -------
    pd.DataFrame
        每行一个 factor × target 组合。
    """
    paths = paths or configure_paths()

    if target_names is None:
        target_names = [
            "label_ret_1d", "label_ret_3d", "label_ret_5d",
            "label_ret_10d", "label_ret_20d",
        ]
        # Only use targets that actually exist
        target_names = [t for t in target_names
                        if (paths.target_output_dir / f"{t}.fea").exists()]

    if not target_names:
        logger.error("No target files found in %s", paths.target_output_dir)
        return pd.DataFrame()

    # Deduplicate
    factor_names = sorted(set(factor_names))
    target_names = sorted(set(target_names))

    # Filter to existing factor files
    existing = [f for f in factor_names
                if (paths.factor_output_dir / f"{f}.fea").exists()]
    missing = len(factor_names) - len(existing)
    if missing:
        logger.warning("Skipping %d missing factor files", missing)

    tasks = [(f, t) for f in existing for t in target_names]
    logger.info(
        "Evaluating %d factors × %d targets = %d combinations",
        len(existing), len(target_names), len(tasks),
    )

    results: list[FactorEvalResult] = []

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) // 2)

    if max_workers > 1 and len(tasks) > 5:
        # Parallel evaluation
        chunk_size = max(1, len(tasks) // (max_workers * 4))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_single_factor, f, t, paths, amt_col, min_stocks,
                ): (f, t)
                for f, t in tasks
            }
            it = as_completed(futures)
            if progress:
                it = tqdm(it, total=len(tasks), desc="Evaluating factors")
            for future in it:
                try:
                    results.append(future.result())
                except Exception as e:
                    f, t = futures[future]
                    logger.warning("Eval failed %s × %s: %s", f, t, e)
    else:
        # Sequential
        it = tasks
        if progress:
            it = tqdm(tasks, desc="Evaluating factors")
        for f, t in it:
            try:
                results.append(evaluate_single_factor(
                    f, t, paths, amt_col, min_stocks,
                ))
            except Exception as e:
                logger.warning("Eval failed %s × %s: %s", f, t, e)

    # Build DataFrame
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)

    # Compute derived metrics
    if not df.empty:
        if "long_short_spread_amt" in df.columns:
            df["abs_long_short_amt"] = df["long_short_spread_amt"].abs()
        if "rank_icir" in df.columns:
            df["abs_rank_icir"] = df["rank_icir"].abs()

    return df


# ── Factor correlation / redundancy ──────────────────────────────────────────


def compute_factor_correlation(
    factor_names: list[str],
    date: str | None = None,
    paths: ProjectPaths | None = None,
    min_stocks: int = 30,
    method: str = "spearman",
) -> pd.DataFrame:
    """计算因子间的截面相关性矩阵。

    Parameters
    ----------
    factor_names : list[str]
    date : str or None
        指定截面日期, 若为 None 则用最新公共日期。
    paths : ProjectPaths or None
    min_stocks : int
        因子间至少要有 min_stocks 只公共股票。
    method : str
        "spearman" 或 "pearson"。

    Returns
    -------
    pd.DataFrame
        因子 × 因子的相关性矩阵。
    """
    paths = paths or configure_paths()

    # Load all factors
    factor_series: dict[str, pd.Series] = {}
    common_dates: set | None = None

    for name in factor_names:
        fp = paths.factor_output_dir / f"{name}.fea"
        if not fp.exists():
            continue
        try:
            f = pd.read_feather(fp)
            for idx_col in ["Date", "date", "index"]:
                if idx_col in f.columns:
                    f = f.set_index(idx_col)
                    break
            if f.empty:
                continue
            if common_dates is None:
                common_dates = set(f.index)
            else:
                common_dates &= set(f.index)
            factor_series[name] = f
        except Exception:
            continue

    if not factor_series or not common_dates:
        return pd.DataFrame()

    # Pick date
    if date is None or date not in common_dates:
        date = str(sorted(common_dates)[-1])

    # Extract cross-section
    xs_data: dict[str, pd.Series] = {}
    for name, f in factor_series.items():
        if date in f.index:
            row = f.loc[date].dropna()
            if len(row) >= min_stocks:
                xs_data[name] = row

    if len(xs_data) < 2:
        return pd.DataFrame()

    # Build wide DataFrame
    xs_df = pd.DataFrame(xs_data)
    common_codes = xs_df.dropna().index
    if len(common_codes) < min_stocks:
        return pd.DataFrame()

    if method == "spearman":
        corr = xs_df.loc[common_codes].corr(method="spearman")
    else:
        corr = xs_df.loc[common_codes].corr(method="pearson")

    return corr



def quick_check_factors(
    factor_names: list[str],
    target_name: str | None = None,
    paths: "ProjectPaths | None" = None,
    sample_dates: int = 10,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """Fast pre-check: detect factors that are constant, all-NaN, or too sparse.

    Scans the first and last *sample_dates* trading days of each factor's .fea
    file to detect edge cases before running the full evaluation.

    Parameters
    ----------
    factor_names : list[str]
    target_name : str or None
        If provided, also checks alignment with the target.
    paths : ProjectPaths or None
    sample_dates : int
        Number of dates to sample from start and end of each factor.
    min_stocks : int
        Minimum number of valid stocks per date for a factor to be usable.

    Returns
    -------
    pd.DataFrame
        Columns: factor_name, status, n_unique_start, n_unique_end,
        n_valid_start, n_valid_end, coverage_ratio, recommendation
    """
    import pandas as pd
    import numpy as np

    if paths is None:
        from .settings import configure_paths
        paths = configure_paths()

    factor_dir = paths.factor_output_dir
    results = []

    for fn in factor_names:
        fp = factor_dir / f"{fn}.fea"
        row = {"factor_name": fn, "status": "ok", "n_unique_start": -1,
               "n_unique_end": -1, "n_valid_start": -1, "n_valid_end": -1,
               "coverage_ratio": -1.0, "recommendation": ""}

        if not fp.exists():
            row["status"] = "missing"
            row["recommendation"] = "remove — .fea file missing"
            results.append(row)
            continue

        try:
            f = pd.read_feather(fp)
        except Exception:
            row["status"] = "unreadable"
            row["recommendation"] = "rebuild — .fea file corrupted"
            results.append(row)
            continue

        # Find the value column (last column)
        val_cols = [c for c in f.columns if c not in ("Date", "date", "index", "Code")]
        if not val_cols:
            row["status"] = "no_value_column"
            row["recommendation"] = "rebuild — no factor value column found"
            results.append(row)
            continue

        val_col = val_cols[-1]

        # Check first and last sample_dates
        date_col = next((c for c in ["Date", "date", "index"] if c in f.columns), None)
        if date_col:
            dates = sorted(f[date_col].unique())
            start_dates = dates[:sample_dates]
            end_dates = dates[-sample_dates:]
        else:
            # If no date column, check the whole dataset
            start_dates = [None]
            end_dates = [None]

        def _check_dates(date_list):
            n_unique_vals = []
            n_valid_stocks = []
            for d in date_list:
                if date_col and d is not None:
                    subset = f[f[date_col] == d][val_col]
                else:
                    subset = f[val_col]
                valid = subset.dropna()
                n_valid_stocks.append(len(valid))
                n_unique_vals.append(valid.nunique() if len(valid) > 0 else 0)
            return n_unique_vals, n_valid_stocks

        start_uniq, start_valid = _check_dates(start_dates)
        end_uniq, end_valid = _check_dates(end_dates)

        row["n_unique_start"] = int(np.mean(start_uniq)) if start_uniq else 0
        row["n_unique_end"] = int(np.mean(end_uniq)) if end_uniq else 0
        row["n_valid_start"] = int(np.mean(start_valid)) if start_valid else 0
        row["n_valid_end"] = int(np.mean(end_valid)) if end_valid else 0

        # Determine status
        total_nunique = row["n_unique_end"]
        total_valid = row["n_valid_end"]

        if total_valid == 0:
            row["status"] = "all_nan"
            row["recommendation"] = "fix or remove — all values NaN"
        elif total_nunique <= 2:
            row["status"] = "constant"
            row["recommendation"] = f"fix — only {total_nunique} unique values (needs per-stock variation)"
        elif total_valid < min_stocks:
            row["status"] = "too_sparse"
            row["recommendation"] = f"fix — only {total_valid} valid stocks per date (min={min_stocks})"
        elif total_nunique < 10:
            row["status"] = "low_variance"
            row["recommendation"] = f"review — only {total_nunique} unique values, may have weak signal"

        results.append(row)

    return pd.DataFrame(results)

def find_redundant_factors(
    factor_names: list[str],
    corr_threshold: float = 0.85,
    date: str | None = None,
    paths: ProjectPaths | None = None,
    keep_metric: str = "abs_rank_icir",
    eval_results: pd.DataFrame | None = None,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """检测并移除冗余因子 (高相关性对中保留质量更高的)。

    Parameters
    ----------
    factor_names : list[str]
    corr_threshold : float
        相关性阈值，高于此值的因子对被认为是冗余的。
    date : str or None
    paths : ProjectPaths or None
    keep_metric : str
        用于决定保留哪个因子的指标 (必须存在于 eval_results 中)。
    eval_results : pd.DataFrame or None
        因子评估结果，用于选择保留哪个因子。

    Returns
    -------
    retained : list[str]
        去冗余后保留的因子。
    removed_pairs : list[tuple[str, str, float]]
        被移除的 (removed_factor, retained_factor, correlation)。
    """
    corr = compute_factor_correlation(factor_names, date=date, paths=paths)
    if corr.empty:
        return list(factor_names), []

    # Build quality scores
    quality: dict[str, float] = {}
    if eval_results is not None and not eval_results.empty:
        for _, row in eval_results.iterrows():
            fn = row.get("factor_name")
            if fn in factor_names:
                quality[fn] = abs(float(row.get(keep_metric, 0) or 0))

    # Get pairs above threshold
    pairs: list[tuple[str, str, float]] = []
    names = list(corr.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            val = corr.iloc[i, j]
            if abs(val) >= corr_threshold:
                pairs.append((names[i], names[j], val))

    # Decide which to keep
    removed: set[str] = set()
    removed_pairs: list[tuple[str, str, float]] = []

    for a, b, val in sorted(pairs, key=lambda x: -abs(x[2])):
        if a in removed or b in removed:
            continue
        qa = quality.get(a, 0)
        qb = quality.get(b, 0)
        if qa >= qb:
            removed.add(b)
            removed_pairs.append((b, a, val))
        else:
            removed.add(a)
            removed_pairs.append((a, b, val))

    retained = [n for n in factor_names if n not in removed]
    return retained, removed_pairs


# ── Per-season evaluation (matching fac_filt.py logic) ────────────────────────


def get_eval_date_range(
    all_dates: list[str],
    season: str,
    lookback_days: int = 720,
    gap_days: int = 10,
    exclude_ranges: list[tuple[str, str]] | None = None,
) -> tuple[list[str], str, str]:
    """获取某季度的回测日期范围 (复刻 fac_filt.py 的逻辑)。

    Parameters
    ----------
    all_dates : list[str]
        全部可选日期 (YYYYMMDD)。
    season : str
        季度标签，如 "2023q1"。
    lookback_days : int
        回溯天数。
    gap_days : int
        测试集开始前排除的天数 (防止信息泄露)。
    exclude_ranges : list[tuple[str, str]] or None
        额外排除的日期区间。

    Returns
    -------
    eval_dates : list[str]
    start_date : str
    end_date : str
    """
    year = int(season[:4])
    q = int(season.split("q")[1])
    month = q * 3 - 2
    test_start = f"{year}{month:02d}01"

    from datetime import datetime, timedelta
    start_dt = datetime.strptime(test_start, "%Y%m%d") - timedelta(days=lookback_days)
    train_start = start_dt.strftime("%Y%m%d")

    # Dates between train_start and test_start, excluding gap
    train_dates = [d for d in all_dates if train_start <= d < test_start]
    if len(train_dates) > gap_days:
        train_dates = train_dates[:-gap_days]

    # Exclude anomalous ranges
    if exclude_ranges is None:
        exclude_ranges = [("20240201", "20240223")]
    for lo, hi in exclude_ranges:
        train_dates = [d for d in train_dates if not (lo <= d <= hi)]

    train_dates.sort()
    if not train_dates:
        return [], "", ""

    return train_dates, train_dates[0], train_dates[-1]


def evaluate_by_season(
    factor_names: list[str],
    target_name: str,
    season_list: list[str],
    all_dates: list[str],
    paths: ProjectPaths | None = None,
    lookback_days: int = 720,
) -> dict[str, pd.DataFrame]:
    """按季度评估因子表现 (匹配 fac_filt.py)。

    Parameters
    ----------
    factor_names : list[str]
    target_name : str
    season_list : list[str]
        如 ["2023q1", "2023q2", ...]
    all_dates : list[str]
    paths : ProjectPaths or None
    lookback_days : int

    Returns
    -------
    dict[str, pd.DataFrame]
        season → evaluation DataFrame.
    """
    paths = paths or configure_paths()

    results_by_season: dict[str, pd.DataFrame] = {}

    for season in tqdm(season_list, desc="Evaluating by season"):
        eval_dates, eval_start, eval_end = get_eval_date_range(
            all_dates, season, lookback_days=lookback_days,
        )
        if not eval_dates:
            logger.warning("No eval dates for season %s", season)
            continue

        season_results = []
        for fn in tqdm(factor_names, desc=f"  {season}", leave=False):
            r = evaluate_single_factor(fn, target_name, paths=paths)
            if r.n_dates > 0:
                # Compute mean on eval period
                ic = r.ic_series
                if ic is not None:
                    ic_eval = ic[ic.index.isin(eval_dates)]
                    if len(ic_eval) >= 5:
                        r.rank_ic_mean = float(ic_eval.mean())
                        r.n_dates = len(ic_eval)
                season_results.append(r)

        df = pd.DataFrame([r.to_dict() for r in season_results])
        results_by_season[season] = df

    return results_by_season


# ── Report generation ────────────────────────────────────────────────────────


def generate_eval_report(
    eval_df: pd.DataFrame,
    top_n: int = 50,
) -> str:
    """生成 markdown 格式的评估报告。

    Parameters
    ----------
    eval_df : pd.DataFrame
        evaluate_factors() 的输出。
    top_n : int

    Returns
    -------
    str
        Markdown 报告。
    """
    if eval_df.empty:
        return "# Factor Evaluation Report\n\n**No data available.**\n"

    lines = [
        "# 因子评估报告 (Factor Evaluation Report)",
        "",
        f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**因子总数**: {eval_df['factor_name'].nunique()}",
        f"**Target 数**: {eval_df['target_name'].nunique()}",
        f"**总评估组合**: {len(eval_df)}",
        "",
        "---",
        "",
    ]

    # ── Global statistics ──
    lines.append("## 1. 整体统计 (Global Statistics)")
    lines.append("")
    stats = eval_df.describe().round(6).to_markdown()
    lines.append(stats)
    lines.append("")

    # ── Top factors by Rank ICIR ──
    lines.append("---")
    lines.append("## 2. Top 因子 — Rank ICIR")
    lines.append("")
    for target in sorted(eval_df["target_name"].unique()):
        sub = eval_df[eval_df["target_name"] == target].copy()
        sub = sub.sort_values("abs_rank_icir", ascending=False).head(top_n)
        lines.append(f"### {target}")
        lines.append("")
        cols = ["factor_name", "rank_icir", "rank_ic_mean", "rank_ic_pos_ratio",
                "long_short_spread_10pct", "coverage_ratio", "n_dates"]
        available = [c for c in cols if c in sub.columns]
        lines.append(sub[available].round(6).to_markdown(index=False))
        lines.append("")

    # ── Top factors by Long-Short Spread (Amt-weighted) ──
    if "abs_long_short_amt" in eval_df.columns:
        lines.append("---")
        lines.append("## 3. Top 因子 — 多空收益 (流动性加权)")
        lines.append("")
        for target in sorted(eval_df["target_name"].unique()):
            sub = eval_df[eval_df["target_name"] == target].copy()
            sub = sub.sort_values("abs_long_short_amt", ascending=False).head(top_n)
            lines.append(f"### {target}")
            lines.append("")
            cols = ["factor_name", "long_short_spread_amt", "long_ret_head_amt",
                    "short_ret_tail_amt", "rank_icir", "coverage_ratio"]
            available = [c for c in cols if c in sub.columns]
            lines.append(sub[available].round(6).to_markdown(index=False))
            lines.append("")

    # ── Factor quality distribution ──
    lines.append("---")
    lines.append("## 4. 因子质量分布")
    lines.append("")

    bins_icir = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float("inf")]
    labels_icir = ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)",
                    "[0.75,1.0)", "[1.0,1.5)", "[1.5,2.0)", ">=2.0"]
    if "abs_rank_icir" in eval_df.columns:
        eval_df["_icir_bin"] = pd.cut(
            eval_df["abs_rank_icir"], bins=bins_icir, labels=labels_icir, right=False,
        )
        dist = eval_df.groupby("_icir_bin", observed=False).size().reset_index(name="count")
        lines.append("### |Rank ICIR| 分布")
        lines.append("")
        lines.append(dist.to_markdown(index=False))
        lines.append("")

    # ── Factor count by coverage ──
    lines.append("### 覆盖率分布")
    lines.append("")
    bins_cov = [0, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    labels_cov = ["[0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,0.95)", "[0.95,1.0]"]
    if "coverage_ratio" in eval_df.columns:
        eval_df["_cov_bin"] = pd.cut(
            eval_df["coverage_ratio"], bins=bins_cov, labels=labels_cov, right=False,
        )
        dist2 = eval_df.groupby("_cov_bin", observed=False).size().reset_index(name="count")
        lines.append(dist2.to_markdown(index=False))
        lines.append("")

    # ── Worst factors (negative ICIR) ──
    lines.append("---")
    lines.append("## 5. 无效/负向因子 (|Rank ICIR| < 0.1)")
    lines.append("")
    bad = eval_df[eval_df["abs_rank_icir"] < 0.1].sort_values("abs_rank_icir")
    if len(bad) > 0:
        cols = ["factor_name", "target_name", "rank_icir", "coverage_ratio"]
        available = [c for c in cols if c in bad.columns]
        lines.append(f"共 {len(bad)} 个组合")
        lines.append("")
        lines.append(bad[available].head(100).round(6).to_markdown(index=False))
    else:
        lines.append("无")
    lines.append("")

    return "\n".join(lines)


def save_eval_report(eval_df: pd.DataFrame, output_path: str | Path, top_n: int = 50) -> None:
    """保存评估报告到 markdown 文件。"""
    report = generate_eval_report(eval_df, top_n=top_n)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)


def save_eval_csv(eval_df: pd.DataFrame, output_path: str | Path) -> None:
    """保存评估结果为 CSV。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("CSV saved to %s", output_path)


# ── Optimised: evaluate one factor against all targets ────────────────────────


def evaluate_factor_across_targets(
    factor_name: str,
    target_names: list[str],
    paths: ProjectPaths | None = None,
    amt_col: str | None = None,
    min_stocks: int = 10,
) -> list[FactorEvalResult]:
    """Load a factor once, evaluate against all targets (I/O efficient).

    This is the recommended entry point for batch evaluation when you have
    many targets — it avoids re-reading the same factor file N times.
    """
    paths = paths or configure_paths()

    factor_path = paths.factor_output_dir / f"{factor_name}.fea"
    if not factor_path.exists():
        logger.warning("Factor file not found: %s", factor_path)
        return []

    try:
        factor = pd.read_feather(factor_path)
    except Exception as e:
        logger.error("Failed to read factor %s: %s", factor_name, e)
        return []

    # Ensure Date index
    for idx_col in ["Date", "date", "index"]:
        if idx_col in factor.columns:
            factor = factor.set_index(idx_col)
            break

    # Optionally load liquidity
    amt = None
    if amt_col is not None:
        amt_path = paths.factor_output_dir / f"{amt_col}.fea"
        if amt_path.exists():
            amt = pd.read_feather(amt_path)
            for idx_col in ["Date", "date", "index"]:
                if idx_col in amt.columns:
                    amt = amt.set_index(idx_col)
                    break

    results: list[FactorEvalResult] = []

    for target_name in target_names:
        target_path = paths.target_output_dir / f"{target_name}.fea"
        if not target_path.exists():
            logger.warning("Target file not found: %s", target_path)
            continue

        try:
            target = pd.read_feather(target_path)
        except Exception as e:
            logger.error("Failed to read target %s: %s", target_name, e)
            continue

        for idx_col in ["Date", "date", "index"]:
            if idx_col in target.columns:
                target = target.set_index(idx_col)
                break

        # Common dates
        common_dates = factor.index.intersection(target.index).sort_values()
        if len(common_dates) == 0:
            results.append(FactorEvalResult(
                factor_name=factor_name, target_name=target_name,
            ))
            continue

        # Per-date metrics
        date_metrics: list[dict] = []
        for date in common_dates:
            try:
                fac_row = factor.loc[date]
                ret_row = target.loc[date]
                amt_row = None
                if amt is not None and date in amt.index:
                    amt_row = amt.loc[date]
                m = _compute_single_date_metrics(
                    fac_row, ret_row, amt_row, min_stocks=min_stocks,
                )
                if m:
                    m["date"] = date
                    date_metrics.append(m)
            except Exception:
                continue

        if not date_metrics:
            results.append(FactorEvalResult(
                factor_name=factor_name, target_name=target_name,
            ))
            continue

        dm = pd.DataFrame(date_metrics).set_index("date")

        # ── IC metrics ──
        rank_ic = dm.get("rank_ic", pd.Series(dtype=float)).dropna()
        pearson_ic = dm.get("pearson_ic", pd.Series(dtype=float)).dropna()

        def _ic_metrics(ic_series: pd.Series) -> dict:
            if len(ic_series) < 5:
                return {"mean": float("nan"), "std": float("nan"),
                        "icir": float("nan"), "pos_ratio": float("nan"), "n": 0}
            n = len(ic_series)
            mean = float(ic_series.mean())
            std = float(ic_series.std(ddof=1))
            icir = mean / std if std > 0 else float("nan")
            pos_ratio = float((ic_series > 0).mean())
            return {"mean": mean, "std": std, "icir": icir,
                    "pos_ratio": pos_ratio, "n": n}

        rank_m = _ic_metrics(rank_ic)
        pear_m = _ic_metrics(pearson_ic)

        # ── Return metrics (sign-adjusted) ──
        sign = 1 if rank_m["mean"] >= 0 else -1
        head10p = dm.get("head10p_ret", pd.Series(dtype=float)).dropna()
        tail10p = dm.get("tail10p_ret", pd.Series(dtype=float)).dropna()

        if sign >= 0:
            long_10 = float(head10p.mean()) if len(head10p) > 0 else float("nan")
            short_10 = float(tail10p.mean()) if len(tail10p) > 0 else float("nan")
        else:
            long_10 = float(tail10p.mean()) if len(tail10p) > 0 else float("nan")
            short_10 = float(head10p.mean()) if len(head10p) > 0 else float("nan")

        head_amt = dm.get("head_amt_ret", pd.Series(dtype=float))
        tail_amt = dm.get("tail_amt_ret", pd.Series(dtype=float))
        long_amt = float("nan")
        short_amt = float("nan")
        if not head_amt.empty and not tail_amt.empty:
            head_amt = head_amt.dropna()
            tail_amt = tail_amt.dropna()
            if sign >= 0:
                long_amt = float(head_amt.mean()) if len(head_amt) > 0 else float("nan")
                short_amt = float(tail_amt.mean()) if len(tail_amt) > 0 else float("nan")
            else:
                long_amt = float(tail_amt.mean()) if len(tail_amt) > 0 else float("nan")
                short_amt = float(head_amt.mean()) if len(head_amt) > 0 else float("nan")

        # ── Stability ──
        ic_stab = float("nan")
        ic_decay = float("nan")
        if len(rank_ic) >= 40:
            roll_std = rank_ic.rolling(20).std().dropna()
            ic_stab = 1.0 / (1.0 + float(roll_std.mean()))
            if len(rank_ic) >= 25:
                ic_decay = float(rank_ic.autocorr(lag=5)) if not np.isnan(rank_ic.autocorr(lag=5)) else float("nan")

        coverage = float(len(dm) / len(common_dates)) if len(common_dates) > 0 else 0.0

        results.append(FactorEvalResult(
            factor_name=factor_name,
            target_name=target_name,
            n_dates=rank_m["n"],
            rank_ic_mean=rank_m["mean"],
            rank_ic_std=rank_m["std"],
            rank_icir=rank_m["icir"],
            rank_ic_pos_ratio=rank_m["pos_ratio"],
            pearson_ic_mean=pear_m["mean"],
            pearson_ic_std=pear_m["std"],
            pearson_icir=pear_m["icir"],
            pearson_ic_pos_ratio=pear_m["pos_ratio"],
            long_ret_top10pct=long_10,
            short_ret_bot10pct=short_10,
            long_short_spread_10pct=long_10 - short_10 if not np.isnan(long_10) and not np.isnan(short_10) else float("nan"),
            long_ret_head_amt=long_amt if not isinstance(long_amt, pd.Series) else float(long_amt.iloc[0]) if hasattr(long_amt, 'iloc') and len(long_amt) > 0 else float("nan"),
            short_ret_tail_amt=short_amt if not isinstance(short_amt, pd.Series) else float(short_amt.iloc[0]) if hasattr(short_amt, 'iloc') and len(short_amt) > 0 else float("nan"),
            long_short_spread_amt=float(long_amt - short_amt) if not np.isnan(long_amt) and not np.isnan(short_amt) and not isinstance(long_amt, pd.Series) and not isinstance(short_amt, pd.Series) else float("nan"),
            ic_stability=ic_stab,
            ic_decay_5d=ic_decay,
            coverage_ratio=coverage,
            ic_series=rank_ic,
        ))

    return results


def evaluate_factors_optimized(
    factor_names: list[str],
    target_names: list[str] | None = None,
    paths: ProjectPaths | None = None,
    max_workers: int | None = None,
    amt_col: str | None = None,
    min_stocks: int = 10,
    progress: bool = True,
    dashboard: Any | None = None,
) -> pd.DataFrame:
    """Optimised batch evaluation — loads each factor once for all targets.

    Uses ProcessPoolExecutor at factor granularity: each worker evaluates
    one factor against ALL targets, avoiding redundant file I/O.

    Parameters
    ----------
    factor_names, target_names, paths, max_workers, amt_col, min_stocks, progress
        Same as :func:`evaluate_factors`.
    dashboard : EvalDashboard or None
        Optional live Rich dashboard for progress visualisation.

    Returns
    -------
    pd.DataFrame
    """
    paths = paths or configure_paths()

    if target_names is None:
        target_names = [
            "label_ret_1d", "label_ret_3d", "label_ret_5d",
            "label_ret_10d", "label_ret_20d",
        ]
        target_names = [t for t in target_names
                        if (paths.target_output_dir / f"{t}.fea").exists()]

    if not target_names:
        logger.error("No target files found")
        return pd.DataFrame()

    factor_names = sorted(set(factor_names))
    existing = [f for f in factor_names
                if (paths.factor_output_dir / f"{f}.fea").exists()]
    missing = len(factor_names) - len(existing)
    if missing:
        logger.warning("Skipping %d missing factor files", missing)

    total = len(existing)
    logger.info("Evaluating %d factors × %d targets = %d combinations (optimised)",
                 total, len(target_names), total * len(target_names))

    if max_workers is None:
        max_workers = max(1, min((os.cpu_count() or 4) - 2, 32))

    all_rows: list[dict] = []

    if max_workers > 1 and total > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_factor_across_targets,
                    fn, target_names, paths, amt_col, min_stocks,
                ): fn
                for fn in existing
            }
            it = as_completed(futures)
            if progress:
                it = tqdm(it, total=total, desc="Evaluating factors (opt)")

            for i, future in enumerate(it):
                fn = futures[future]
                try:
                    results = future.result()
                    for r in results:
                        all_rows.append(r.to_dict())
                except Exception as e:
                    logger.warning("Eval failed for %s: %s", fn, e)

                if dashboard is not None:
                    for r in results:
                        icir = r.rank_icir if hasattr(r, 'rank_icir') else None
                        dashboard.mark_done(
                            f"{r.factor_name}×{r.target_name}",
                            icir=icir, ok=not np.isnan(r.rank_icir) if hasattr(r, 'rank_icir') else True,
                        )
    else:
        it = existing
        if progress:
            it = tqdm(it, desc="Evaluating factors (opt)")
        for fn in it:
            try:
                results = evaluate_factor_across_targets(
                    fn, target_names, paths, amt_col, min_stocks,
                )
                for r in results:
                    all_rows.append(r.to_dict())
            except Exception as e:
                logger.warning("Eval failed for %s: %s", fn, e)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        if "long_short_spread_amt" in df.columns:
            df["abs_long_short_amt"] = df["long_short_spread_amt"].abs()
        if "rank_icir" in df.columns:
            df["abs_rank_icir"] = df["rank_icir"].abs()

    return df
