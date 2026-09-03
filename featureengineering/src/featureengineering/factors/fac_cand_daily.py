"""
fac_cand 日频候选因子（4 个，Class 1，数据源 daily.parquet）。

来源于 Model/v8work/build_cand_daily.py（2026-08-14 V8.2 迭代产物：27 个
日频候选电池经 240 天池 1d rankIC 筛查后选入模型的 4 个）。公式语义与原脚本
逐项对齐：收益一律 pct_chg/100 口径（skill.md §8.10），隔夜=open/pre_close-1，
日内=close/open-1，日 VWAP=amount/vol（daily 的 vol 单位=股，无需 ×100）。
ret_ind_rel_1d 的行业归属见 _fac_new_common 说明（不声明 stock_list 依赖，
PIT 过滤器会移除显式声明该依赖的因子）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ._fac_new_common import with_industry


def _daily(context: FactorContext) -> pd.DataFrame:
    """加载 daily 面板并补齐基础列（ret/overnight/intraday/vwap/industry）。"""
    daily = context.load("daily.parquet")
    df = with_industry(daily.reset_index(), context)
    df["ret"] = df["pct_chg"] / 100.0
    df["overnight"] = df["open"] / df["pre_close"] - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    df["vwap"] = df["amount"] / df["vol"].replace(0, np.nan)
    return df


def _out(df: pd.DataFrame, name: str, values: pd.Series) -> pd.Series:
    out = df[["Date", "Code"]].copy()
    out[name] = values.to_numpy()
    return out.set_index(["Date", "Code"])[name]


@register_factor(
    name="overnight_minus_intraday",
    description="隔夜−日内收益差因子：open/pre_close-1 与 close/open-1 之差（隔夜强于日内排前）。",
    category="price",
    thesis="隔夜与日内收益的差值衡量消息定价结构——隔夜涨而日内跌=开盘消化利好"
           "后日内反转压力大；隔夜跌而日内涨=利空出尽、资金盘中承接。V8 筛查 "
           "meanIC 0.030/ICIR 0.20，为 27 个日频候选中最强。",
    dependencies=("daily.parquet",),
)
def factor_overnight_minus_intraday(context: FactorContext):
    df = _daily(context)
    return _out(df, "overnight_minus_intraday", df["overnight"] - df["intraday"])


@register_factor(
    name="vwap_dev_1d",
    description="日频VWAP偏离因子：close/(amount/vol)-1（收盘价高于日均价排前）。",
    category="price",
    thesis="收盘价相对全日成交均价的位置度量尾盘定价方向——收盘显著高于 VWAP="
           "尾盘买盘占优、资金愿意以高于均价的成本拿货；低于 VWAP=尾盘承压。"
           "V8 筛查 meanIC -0.034/|IC|0.13（该池内方向为负，截面排名可逆）。",
    dependencies=("daily.parquet",),
)
def factor_vwap_dev_1d(context: FactorContext):
    df = _daily(context)
    return _out(df, "vwap_dev_1d", df["close"] / df["vwap"] - 1.0)


@register_factor(
    name="overnight_ma5",
    description="隔夜收益5日均值因子：open/pre_close-1 的5日滚动均值（隔夜动能排前）。",
    category="price",
    thesis="隔夜收益衡量集合竞价定价的信息冲击，其5日均值过滤单日噪音后捕捉"
           "隔夜动能的持续性——连续隔夜高开=市场对该股的信息定价持续上修。"
           "V8 筛查 meanIC 0.019/ICIR 0.17。",
    dependencies=("daily.parquet",),
)
def factor_overnight_ma5(context: FactorContext):
    df = _daily(context)
    vals = df.groupby("Code")["overnight"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return _out(df, "overnight_ma5", vals)


@register_factor(
    name="ret_ind_rel_1d",
    description="行业相对1日收益因子：个股pct_chg/100 − 行业等权日收益（跑赢行业排前）。",
    category="sector",
    thesis="剥离行业 β 的当日个股超额收益——同日行业普涨普跌时，个股相对行业的"
           "超额部分更能反映个股层面的资金选择。与 _fac_new_common 的行业相对"
           "族同口径（行业映射用当前快照，变化缓慢，属可接受轻微时点回溯）。",
    dependencies=("daily.parquet",),
)
def factor_ret_ind_rel_1d(context: FactorContext):
    df = _daily(context)
    ind_ret = df.groupby(["Date", "industry"])["ret"].transform("mean")
    return _out(df, "ret_ind_rel_1d", df["ret"] - ind_ret)
