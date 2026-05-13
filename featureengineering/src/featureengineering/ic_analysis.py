"""因子 IC / ICIR 分析模块。

支持任意因子与任意 target 标签之间的截面 IC 评估。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .settings import ProjectPaths, configure_paths


@dataclass(frozen=True)
class ICResult:
    factor_name: str
    target_name: str
    method: str                      # "rank" | "pearson"
    n_dates: int                     # 有效截面数
    ic_mean: float
    ic_std: float
    icir: float                      # IC mean / IC std
    ic_positive_ratio: float         # IC > 0 的日期占比
    ic_tstat: float                  # IC mean / (IC std / sqrt(n))

    def summary(self) -> str:
        return (
            f"{self.factor_name} × {self.target_name}  [{self.method}]\n"
            f"  N dates:    {self.n_dates}\n"
            f"  IC mean:    {self.ic_mean:+.6f}\n"
            f"  IC std:     {self.ic_std:.6f}\n"
            f"  ICIR:       {self.icir:+.4f}\n"
            f"  IC > 0:     {self.ic_positive_ratio:.2%}\n"
            f"  t-stat:     {self.ic_tstat:+.4f}"
        )


def compute_ic(
    factor_name: str,
    target_name: str,
    method: str = "rank",
    paths: ProjectPaths | None = None,
) -> ICResult:
    """计算单个因子对单个 target 的截面 IC。

    Parameters
    ----------
    factor_name : str
        因子名，对应 data/factors/<factor_name>.fea
    target_name : str
        target 名，对应 data/factors/<target_name>.fea
    method : str
        "rank" (Spearman) 或 "pearson" (Pearson)
    paths : ProjectPaths | None
    """
    paths = paths or configure_paths()

    factor = pd.read_feather(paths.factor_output_dir / f"{factor_name}.fea")
    target = pd.read_feather(paths.target_output_dir / f"{target_name}.fea")

    # Align on (Date, Code)
    factor_col = factor_name
    target_col = target_name

    merged = factor[[factor_col]].join(target[[target_col]], how="inner")
    merged = merged.dropna()

    if merged.empty:
        return ICResult(
            factor_name=factor_name, target_name=target_name, method=method,
            n_dates=0, ic_mean=float("nan"), ic_std=float("nan"),
            icir=float("nan"), ic_positive_ratio=float("nan"),
            ic_tstat=float("nan"),
        )

    def _ic_for_date(group: pd.DataFrame) -> float:
        if len(group) < 10:
            return float("nan")
        if method == "rank":
            return group[factor_col].corr(group[target_col], method="spearman")
        else:
            return group[factor_col].corr(group[target_col], method="pearson")

    ic_series = merged.groupby(level="Date").apply(_ic_for_date).dropna()

    if len(ic_series) == 0:
        return ICResult(
            factor_name=factor_name, target_name=target_name, method=method,
            n_dates=0, ic_mean=float("nan"), ic_std=float("nan"),
            icir=float("nan"), ic_positive_ratio=float("nan"),
            ic_tstat=float("nan"),
        )

    n = len(ic_series)
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else float("nan")
    ic_positive_ratio = float((ic_series > 0).mean())
    ic_tstat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else float("nan")

    return ICResult(
        factor_name=factor_name, target_name=target_name, method=method,
        n_dates=n, ic_mean=ic_mean, ic_std=ic_std,
        icir=icir, ic_positive_ratio=ic_positive_ratio, ic_tstat=ic_tstat,
    )


def compute_ic_matrix(
    factor_names: list[str],
    target_names: list[str],
    method: str = "rank",
    paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """计算多个因子 × 多个 target 的 IC 矩阵。"""
    rows = []
    for fn in factor_names:
        for tn in target_names:
            r = compute_ic(fn, tn, method=method, paths=paths)
            rows.append({
                "factor": fn, "target": tn,
                "ic_mean": r.ic_mean, "ic_std": r.ic_std,
                "icir": r.icir, "ic_positive_ratio": r.ic_positive_ratio,
                "n_dates": r.n_dates,
            })
    return pd.DataFrame(rows)


def get_ic_series(
    factor_name: str,
    target_name: str,
    method: str = "rank",
    paths: ProjectPaths | None = None,
) -> pd.Series:
    """返回按日期的 IC 时间序列，用于画图或进一步分析。"""
    paths = paths or configure_paths()

    factor = pd.read_feather(paths.factor_output_dir / f"{factor_name}.fea")
    target = pd.read_feather(paths.target_output_dir / f"{target_name}.fea")

    factor_col = factor_name
    target_col = target_name

    merged = factor[[factor_col]].join(target[[target_col]], how="inner").dropna()

    def _ic_for_date(group: pd.DataFrame) -> float:
        if len(group) < 10:
            return float("nan")
        if method == "rank":
            return group[factor_col].corr(group[target_col], method="spearman")
        else:
            return group[factor_col].corr(group[target_col], method="pearson")

    return merged.groupby(level="Date").apply(_ic_for_date).dropna()
