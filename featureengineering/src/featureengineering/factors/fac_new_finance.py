"""
fac_new 第三轮财务/市场数据因子（11 个，Class 1，数据源 finance.parquet）。

来源于 trainingdata/build_fac_new_r3.py（2026-08-13 迭代产物），公式语义
与原脚本一致：行业相对类用行业内等权均值之差（绝对差，非比值/对数）；
rel_turnover_ind_ma20 沿袭原脚本语义 = 换手率减个股全期均值的 20 日均值
（名义"行业相对"，实为个股自身 demean，照抄保留）；pe_ttm/pb 0 值置 NaN；
log_mv 用 total_mv 自然对数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ._fac_new_common import ind_mean, roll, with_industry


def _fin(context: FactorContext) -> pd.DataFrame:
    fin = context.load("finance.parquet")
    df = with_industry(fin.reset_index(), context)
    df["pe_ttm_n"] = df["pe_ttm"].replace(0, np.nan)
    df["pb_n"] = df["pb"].replace(0, np.nan)
    df["log_mv"] = np.log(df["total_mv"].replace(0, np.nan))
    return df


def _out(df: pd.DataFrame, name: str, values: pd.Series) -> pd.Series:
    out = df[["Date", "Code"]].copy()
    out[name] = values.to_numpy()
    return out.set_index(["Date", "Code"])[name]


@register_factor(
    name="rel_turnover_ind",
    description="行业内相对换手（换手率 − 行业等权均值）。",
    category="sector",
    thesis="行业内换手相对水平：高换手=资金关注度高但筹码交换剧烈，"
    "低换手=惜售/锁筹；剥离板块整体交投活跃度。",
    dependencies=("finance.parquet",),
)
def factor_rel_turnover_ind(context: FactorContext):
    df = _fin(context)
    vals = df["turnover_rate"] - ind_mean(df, "turnover_rate")
    return _out(df, "rel_turnover_ind", vals)


@register_factor(
    name="rel_turnover_ind_ma20",
    description="换手率相对自身均值20日均值（turnover_rate − 个股全期均值）。",
    category="sector",
    thesis="换手率相对自身历史水平的月度趋势，识别个股交投活跃度中枢上移/下移"
    "（注：沿袭原脚本语义，实为个股自身 demean，非行业相对）。",
    dependencies=("finance.parquet",),
)
def factor_rel_turnover_ind_ma20(context: FactorContext):
    df = _fin(context)
    vals = df.groupby("Code")["turnover_rate"].transform(
        lambda s: (s - s.mean()).rolling(20, min_periods=1).mean()
    )
    return _out(df, "rel_turnover_ind_ma20", vals)


@register_factor(
    name="rel_pe_ind",
    description="行业内相对估值（pe_ttm − 行业等权均值，0值置NaN）。",
    category="valuation",
    thesis="同一行业内相对便宜的股票（负偏离大）估值修复空间更大，"
    "绝对 PE 受行业属性干扰大，行业内相对估值更可比。",
    dependencies=("finance.parquet",),
)
def factor_rel_pe_ind(context: FactorContext):
    df = _fin(context)
    vals = df["pe_ttm_n"] - ind_mean(df, "pe_ttm_n")
    return _out(df, "rel_pe_ind", vals)


@register_factor(
    name="rel_pb_ind",
    description="行业内相对市净率（pb − 行业等权均值，0值置NaN）。",
    category="valuation",
    thesis="行业内相对 PB 刻画净资产定价差异，低相对 PB 标的具备价值修复弹性。",
    dependencies=("finance.parquet",),
)
def factor_rel_pb_ind(context: FactorContext):
    df = _fin(context)
    vals = df["pb_n"] - ind_mean(df, "pb_n")
    return _out(df, "rel_pb_ind", vals)


@register_factor(
    name="turnover_chg_5d",
    description="换手率5日变化率（turnover_rate 5日pct_change）。",
    category="timeseries",
    thesis="换手率短期骤升=资金异动/事件催化，骤降=交投萎缩；变化率捕捉边际拐点。",
    dependencies=("finance.parquet",),
)
def factor_turnover_chg_5d(context: FactorContext):
    df = _fin(context)
    vals = df.groupby("Code")["turnover_rate"].pct_change(5)
    return _out(df, "turnover_chg_5d", vals)


@register_factor(
    name="turnover_chg_20d",
    description="换手率20日变化率（turnover_rate 20日pct_change）。",
    category="timeseries",
    thesis="月度换手变化反映交投活跃度的中期趋势切换，过滤短期脉冲。",
    dependencies=("finance.parquet",),
)
def factor_turnover_chg_20d(context: FactorContext):
    df = _fin(context)
    vals = df.groupby("Code")["turnover_rate"].pct_change(20)
    return _out(df, "turnover_chg_20d", vals)


@register_factor(
    name="vol_ratio_ma5",
    description="量比5日均值（volume_ratio 的5日滚动均值）。",
    category="timeseries",
    thesis="量比=当日成交量/过去5日均量，其均值的抬升反映近期持续放量，"
    "温和放量上行比单日爆量更可持续。",
    dependencies=("finance.parquet",),
)
def factor_vol_ratio_ma5(context: FactorContext):
    df = _fin(context)
    return _out(df, "vol_ratio_ma5", roll(df, "volume_ratio", 5, "mean"))


@register_factor(
    name="vol_ratio_ma20",
    description="量比20日均值（volume_ratio 的20日滚动均值）。",
    category="timeseries",
    thesis="月度量能水平，识别持续放量/缩量阶段的切换，作为趋势确认的辅助信号。",
    dependencies=("finance.parquet",),
)
def factor_vol_ratio_ma20(context: FactorContext):
    df = _fin(context)
    return _out(df, "vol_ratio_ma20", roll(df, "volume_ratio", 20, "mean"))


@register_factor(
    name="log_mv",
    description="对数市值（ln(total_mv)，0值置NaN）。",
    category="valuation",
    thesis="市值是 A 股最经典的风险因子：小市值长期有溢价，对数化压缩右尾，"
    "作为规模基准参与截面定价。",
    dependencies=("finance.parquet",),
)
def factor_log_mv(context: FactorContext):
    df = _fin(context)
    return _out(df, "log_mv", df["log_mv"])


@register_factor(
    name="rel_log_mv_ind",
    description="行业内相对市值（log_mv − 行业等权均值）。",
    category="valuation",
    thesis="行业内相对规模：同一行业中偏小的标的弹性更大，剥离板块整体市值水平后"
    "规模效应更纯粹。",
    dependencies=("finance.parquet",),
)
def factor_rel_log_mv_ind(context: FactorContext):
    df = _fin(context)
    vals = df["log_mv"] - ind_mean(df, "log_mv")
    return _out(df, "rel_log_mv_ind", vals)


@register_factor(
    name="log_mv_chg_20d",
    description="市值20日变化（log_mv 的20日diff）。",
    category="valuation",
    thesis="对数市值变化近似 20 日累计收益率（市值维度），与价格动量互补，"
    "捕捉股本变动（增发/解禁）带来的规模跳变。",
    dependencies=("finance.parquet",),
)
def factor_log_mv_chg_20d(context: FactorContext):
    df = _fin(context)
    vals = df.groupby("Code")["log_mv"].diff(20)
    return _out(df, "log_mv_chg_20d", vals)
