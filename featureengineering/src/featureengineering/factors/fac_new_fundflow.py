"""
fac_new 第二轮主力资金流因子（3 个，Class 1，数据源 main_fund_flow.parquet）。

来源于 trainingdata/build_fac_new_r2.py（2026-08-13 迭代产物），公式语义
与原脚本一致：net_rel_ind = (net_mf_amount − 行业均值)/行业std 的行业内
z-score，再按 Code 滚动求均值；mf_flow_streak_5d 名义为 streak，实为
5 日内净流入为正的天数占比（照抄保留）。
"""

from __future__ import annotations

import pandas as pd

from ..registry import FactorContext, register_factor
from ._fac_new_common import ind_mean, ind_std, roll, with_industry


def _mf(context: FactorContext) -> pd.DataFrame:
    mf = context.load("main_fund_flow.parquet")
    df = with_industry(mf.reset_index(), context)
    df["net_rel_ind"] = (df["net_mf_amount"] - ind_mean(df, "net_mf_amount")) / (
        ind_std(df, "net_mf_amount") + 1e-8
    )
    return df


def _out(df: pd.DataFrame, name: str, values: pd.Series) -> pd.Series:
    out = df[["Date", "Code"]].copy()
    out[name] = values.to_numpy()
    return out.set_index(["Date", "Code"])[name]


@register_factor(
    name="mf_rel_ind_ma5d",
    description="主力净流入行业相对强度（行业内z-score的5日均值）。",
    category="fund_flow",
    thesis="主力资金（大单+超大单净流入）行业内相对强弱，5日均值平滑单日噪音，"
    "反映短期聪明钱动向。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_rel_ind_ma5d(context: FactorContext):
    df = _mf(context)
    return _out(df, "mf_rel_ind_ma5d", roll(df, "net_rel_ind", 5, "mean"))


@register_factor(
    name="mf_rel_ind_ma20d",
    description="主力净流入行业相对强度（行业内z-score的20日均值）。",
    category="fund_flow",
    thesis="月度主力资金相对强度，过滤短期对倒噪音，刻画机构资金的持续布局方向。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_rel_ind_ma20d(context: FactorContext):
    df = _mf(context)
    return _out(df, "mf_rel_ind_ma20d", roll(df, "net_rel_ind", 20, "mean"))


@register_factor(
    name="mf_flow_streak_5d",
    description="主力净流入天数频率（5日内 net_mf_amount>0 的天数占比）。",
    category="fund_flow",
    thesis="净流入天数占比比净额更稳健，高频净流入=主力持续吸筹，"
    "（注：名义 streak，实为频率，照抄原脚本语义）。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_flow_streak_5d(context: FactorContext):
    df = _mf(context)
    df["pos"] = (df["net_mf_amount"] > 0).astype(float)
    return _out(df, "mf_flow_streak_5d", roll(df, "pos", 5, "mean"))
