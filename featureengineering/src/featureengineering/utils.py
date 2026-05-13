from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    right = right.replace(0, np.nan)
    result = left / right
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level="Date").rank(pct=True)


def rolling_group_mean(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods or max(1, window // 2)).mean()
    )


def rolling_group_std(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods or max(1, window // 2)).std()
    )


def rolling_group_sum(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods or max(1, window // 2)).sum()
    )


def rolling_group_max(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="Code").transform(lambda s: s.rolling(window, min_periods=1).max())


def rolling_group_min(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="Code").transform(lambda s: s.rolling(window, min_periods=1).min())


def rolling_group_skew(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=max(1, window // 2)).skew()
    )


def rolling_group_kurt(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=max(1, window // 2)).kurt()
    )


def rolling_group_apply(series: pd.Series, window: int, func, min_periods: int | None = None) -> pd.Series:
    return series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods or max(1, window // 2)).apply(func)
    )


def rolling_corr_by_group(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.DataFrame({"a": a.values, "b": b.values}, index=a.index)
    return (
        df.groupby(level="Code")
        .apply(lambda g: g["a"].rolling(window, min_periods=window).corr(g["b"]))
        .droplevel(0)
        .reindex(a.index)
    )
