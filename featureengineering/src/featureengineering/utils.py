from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(left, right):
    """Divide *left* by *right*, mapping division-by-zero to NaN.

    Handles both pandas Series and scalar inputs, in any combination.
    """
    # Replace zeros in the denominator — scalar path avoids .replace() on numbers
    if isinstance(right, (int, float, np.integer, np.floating)):
        if right == 0:
            right = np.nan
    else:
        right = right.replace(0, np.nan)

    result = left / right

    if isinstance(result, (int, float, np.integer, np.floating)):
        return np.nan if np.isinf(result) else result
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def cross_sectional_rank(
    series: pd.Series,
    winsorize: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Cross-sectional percentile rank within each date.

    When *winsorize* is True (the default), raw values are first clipped
    to [*lower*, *upper*] quantiles per date to limit the influence of
    extreme outliers on the rank distribution.
    """
    series = series.replace([np.inf, -np.inf], np.nan)
    series = pd.to_numeric(series, errors="coerce")

    if winsorize:
        # Vectorized winsorization — 7× faster than groupby.transform(_clip)
        lo = series.groupby(level="Date").quantile(lower)
        hi = series.groupby(level="Date").quantile(upper)
        # Reindex quantiles to full series index for clip()
        date_lo = lo.reindex(series.index, level="Date")
        date_hi = hi.reindex(series.index, level="Date")
        series = series.clip(lower=date_lo, upper=date_hi)

    with np.errstate(invalid="ignore"):
        return series.groupby(level="Date").rank(pct=True)




def safe_rank(
    series: pd.Series,
    winsorize: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
    fallback_value: float = 0.0,
) -> pd.Series:
    """Cross-sectional percentile rank with safe fallback for edge cases.

    Unlike ``cross_sectional_rank``, this function handles:
    - All-NaN series → returns *fallback_value* for all entries
    - Constant series → returns *fallback_value* for all entries  
    - Too few valid entries (<5) per date → returns *fallback_value*
    
    This prevents spurious Spearman correlations when factor values are
    degenerate (no cross-sectional variation).
    """
    import numpy as np
    import pandas as pd

    series = series.replace([np.inf, -np.inf], np.nan)
    series = pd.to_numeric(series, errors="coerce")

    if series.isna().all():
        return pd.Series(fallback_value, index=series.index)

    result = pd.Series(index=series.index, dtype=float)

    for date, group in series.groupby(level="Date"):
        valid = group.dropna()
        if len(valid) < 5 or valid.nunique() <= 1:
            result.loc[group.index] = fallback_value
        else:
            if winsorize:
                lo = valid.quantile(lower)
                hi = valid.quantile(upper)
                valid = valid.clip(lower=lo, upper=hi)
            with np.errstate(invalid="ignore"):
                result.loc[group.index] = valid.rank(pct=True)

    return result

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


def event_decay(series: pd.Series, half_life: int = 5) -> pd.Series:
    """Convert sparse event-factor values into a continuous decaying signal.

    For each stock, forward-fill the event-day values and apply exponential
    decay with the given *half_life* (in trading days).  Non-event days are
    filled with the last event value weighted by exp(-λ·t) where λ = ln(2)/half_life.

    Parameters
    ----------
    series : pd.Series (MultiIndex [Date, Code])
        Sparse event factor: valid (rank) values on event days, NaN on non-event days.
    half_life : int
        Number of trading days after which the signal strength is halved.
        Typical values: 3 (封板), 5 (龙虎榜/涨跌停), 10 (跌停恢复).

    Returns
    -------
    pd.Series with the same MultiIndex, no NaN after each stock's first event.
    """
    import numpy as np

    decay_rate = np.log(2) / half_life

    def _decay_one_stock(s: pd.Series) -> pd.Series:
        # Forward-fill event values
        ffill = s.ffill()
        # Days since last event — group_id increments on each new event
        grouper = s.notna().cumsum()
        days_since = s.groupby(grouper, group_keys=False).cumcount()
        return ffill * np.exp(-decay_rate * days_since)

    return series.groupby(level="Code", group_keys=False).apply(_decay_one_stock)
