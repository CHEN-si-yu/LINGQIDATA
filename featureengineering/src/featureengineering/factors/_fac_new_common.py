"""
fac_new 51 因子公共工具（内部模块，无注册）。

行业相对类因子使用 ``context.repo.load_industry_map()`` 获取行业归属
（stock_list 当前快照派生；与 neutral.py / sector.py 同款，行业归属变化
缓慢，属可接受的轻微时点回溯；PIT 过滤器会移除显式声明 stock_list 依赖的
因子，故此处不声明，仅在 docstring 说明）。
"""

from __future__ import annotations

import pandas as pd

from ..registry import FactorContext


def with_industry(frame: pd.DataFrame, context: FactorContext) -> pd.DataFrame:
    """(Date, Code) 面板 → 追加 industry 列（缺失行业 → 'unknown'）。

    frame 可以是 (Date, Code) MultiIndex 面板，也可以是 reset 后的
    (Date, Code 两列) DataFrame。
    """
    ind_map = context.repo.load_industry_map()
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex) and "Code" in out.index.names:
        codes = out.index.get_level_values("Code")
    else:
        codes = out["Code"]
    out["industry"] = codes.map(ind_map).fillna("unknown")
    return out


def roll(df: pd.DataFrame, col: str, n: int, agg: str, min_periods: int = 1) -> pd.Series:
    """按 Code 分组滚动聚合（df 需含 Code 列，返回与 df 对齐的 Series）。"""
    return df.groupby("Code")[col].transform(
        lambda s: s.rolling(n, min_periods=min_periods).agg(agg)
    )


def ind_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """按 (Date, industry) 分组的等权均值（df 需含 industry 列）。"""
    return df.groupby(["Date", "industry"])[col].transform("mean")


def ind_std(df: pd.DataFrame, col: str) -> pd.Series:
    """按 (Date, industry) 分组的横截面标准差。"""
    return df.groupby(["Date", "industry"])[col].transform("std")
