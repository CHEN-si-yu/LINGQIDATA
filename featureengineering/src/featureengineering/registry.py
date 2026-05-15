from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .dataset import DataRepository


@dataclass(frozen=True)
class FactorSpec:
    name: str
    description: str
    category: str
    thesis: str
    dependencies: tuple[str, ...]
    compute: Callable[["FactorContext"], pd.Series | pd.DataFrame]


@dataclass
class FactorContext:
    repo: DataRepository
    start_date: str | None = None
    end_date: str | None = None
    _lookback_days: int = 252

    def load(self, relative_path: str) -> pd.DataFrame:
        return self.repo.load_panel(
            relative_path,
            min_date=self.start_date,
            max_date=self.end_date,
            lookback_days=self._lookback_days,
        )

    def load_financial(
        self,
        relative_path: str,
        value_cols: list[str] | None = None,
        date_col: str = "ann_date",
    ) -> pd.DataFrame:
        return self.repo.load_financial_panel(
            relative_path,
            value_cols=value_cols,
            date_col=date_col,
            min_date=self.start_date,
            max_date=self.end_date,
            lookback_days=self._lookback_days,
        )


FACTOR_REGISTRY: dict[str, FactorSpec] = {}


def register_factor(
    *,
    name: str,
    description: str,
    category: str,
    thesis: str,
    dependencies: tuple[str, ...],
) -> Callable[[Callable[[FactorContext], pd.Series | pd.DataFrame]], Callable[[FactorContext], pd.Series | pd.DataFrame]]:
    def decorator(func: Callable[[FactorContext], pd.Series | pd.DataFrame]) -> Callable[[FactorContext], pd.Series | pd.DataFrame]:
        if name in FACTOR_REGISTRY:
            raise ValueError(f"Duplicate factor registration: {name}")
        FACTOR_REGISTRY[name] = FactorSpec(
            name=name,
            description=description,
            category=category,
            thesis=thesis,
            dependencies=dependencies,
            compute=func,
        )
        return func

    return decorator


def get_factor(name: str) -> FactorSpec:
    try:
        return FACTOR_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown factor: {name}") from exc
