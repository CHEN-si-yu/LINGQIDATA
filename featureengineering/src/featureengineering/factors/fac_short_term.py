"""
fac_short_term —— 第 4 轮新因子：超短周期（2~5 日）1d 信号增强因子（16 个，Class 1）。

数据源仅 daily.parquet（白名单）。全部因子 T 日值只依赖 ≤T 数据（pct_change /
rolling 均无未来函数），PIT 干净。

设计动机：V7.x 诊断显示 1d 目标是短板（V7.0 池 IC 0.0621），而 5d/10d/20d 信号
强得多。超短周期因子（2-5 日反转/动量、5 日波动结构、隔夜/日内拆分、流动性冲击）
旨在为 1d 头提供更贴近持有期的增量信号。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ._fac_new_common import ind_mean, roll, with_industry


def _daily(context: FactorContext) -> pd.DataFrame:
    daily = context.load("daily.parquet")
    df = with_industry(daily.reset_index(), context)
    df["ret"] = df.groupby("Code")["close"].pct_change()
    df["overnight"] = df["open"] / df["pre_close"] - 1
    df["intraday"] = df["close"] / df["open"] - 1
    df["absret"] = df["ret"].abs()
    return df


def _out(df: pd.DataFrame, name: str, values: pd.Series) -> pd.Series:
    out = df[["Date", "Code"]].copy()
    out[name] = values.to_numpy()
    return out.set_index(["Date", "Code"])[name]


# ── 超短动量/反转 ─────────────────────────────────────────────────────────

@register_factor(
    name="reversal_2d",
    description="2 日收益（收盘价 2 日变化率）。",
    category="price",
    thesis="A 股超短周期（1-5 日）以反转为主：2 日大涨的股票次日倾向回吐，"
    "2 日大跌倾向反弹，1d 持有期内信号最直接。",
    dependencies=("daily.parquet",),
)
def factor_reversal_2d(context: FactorContext):
    df = _daily(context)
    return _out(df, "reversal_2d", df.groupby("Code")["close"].pct_change(2))


@register_factor(
    name="momentum_3",
    description="3 日收益（收盘价 3 日变化率）。",
    category="price",
    thesis="3 日介于超短反转（2 日）与周度动量（5 日）之间，是 1d 预测的"
    "敏感窗口，补充 momentum_5 未覆盖的粒度。",
    dependencies=("daily.parquet",),
)
def factor_momentum_3(context: FactorContext):
    df = _daily(context)
    return _out(df, "momentum_3", df.groupby("Code")["close"].pct_change(3))


# ── 超短行业相对 ──────────────────────────────────────────────────────────

def _rel_mom(df: pd.DataFrame, n: int) -> pd.Series:
    df["ret_n"] = df.groupby("Code")["close"].pct_change(n)
    ind_ret_n = df.groupby(["Date", "industry"])["ret_n"].transform("mean")
    return df["ret_n"] - ind_ret_n


@register_factor(
    name="rel_mom_ind_3d",
    description="行业相对动量（个股3日收益 − 行业等权3日收益）。",
    category="sector",
    thesis="3 日行业相对动量为 rel_mom_ind_5d 的短周期补充：行业内超短期"
    "领先-滞后关系在 1d 持有期内衰减最快，需要更短的窗口捕捉。",
    dependencies=("daily.parquet",),
)
def factor_rel_mom_ind_3d(context: FactorContext):
    df = _daily(context)
    return _out(df, "rel_mom_ind_3d", _rel_mom(df, 3))


@register_factor(
    name="ind_ret_ma_3d",
    description="行业收益动能（行业等权日收益的3日均值）。",
    category="sector",
    thesis="3 日行业动能捕捉板块超短期情绪脉冲，与 ind_ret_ma_5d/20d/60d "
    "构成完整周期谱系，1d 预测中近期板块共振是重要背景项。",
    dependencies=("daily.parquet",),
)
def factor_ind_ret_ma_3d(context: FactorContext):
    df = _daily(context)
    df["ind_ret"] = ind_mean(df, "ret")
    vals = roll(df, "ind_ret", 3, "mean")
    return _out(df, "ind_ret_ma_3d", vals)


# ── 5 日波动结构 ──────────────────────────────────────────────────────────

@register_factor(
    name="vol_of_vol_5d",
    description="波动率之波动（5 日 |日收益| 的标准差）。",
    category="price",
    thesis="vol_of_vol_20d/60d 证明波动率二阶矩是强信号；5 日窗口刻画"
    "波动率刚启动/收敛的拐点，与 1d 短期反转效应高度相关。",
    dependencies=("daily.parquet",),
)
def factor_vol_of_vol_5d(context: FactorContext):
    df = _daily(context)
    return _out(df, "vol_of_vol_5d", roll(df, "absret", 5, "std"))


@register_factor(
    name="close_location_5d",
    description="5 日收盘位置（(close − 5日最低价)/(5日最高价 − 5日最低价)）。",
    category="price",
    thesis="close_location_20d 的短周期版：5 日区间内的收盘位置刻画超短期"
    "供需平衡点，高位钝化/低位启动对次日收益有区分度。",
    dependencies=("daily.parquet",),
)
def factor_close_location_5d(context: FactorContext):
    df = _daily(context)
    lo = df.groupby("Code")["low"].transform(lambda s: s.rolling(5, min_periods=1).min())
    hi = df.groupby("Code")["high"].transform(lambda s: s.rolling(5, min_periods=1).max())
    vals = (df["close"] - lo) / (hi - lo + 1e-8)
    return _out(df, "close_location_5d", vals)


@register_factor(
    name="close_to_high_5d",
    description="收盘相对 5 日最高价（close / 5日最高价）。",
    category="price",
    thesis="距离 5 日高点的位置区分「创短期新高」与「深度回调」状态；"
    "贴近新高后的次日行为（突破延续 vs 获利回吐）是 1d 有效信号。",
    dependencies=("daily.parquet",),
)
def factor_close_to_high_5d(context: FactorContext):
    df = _daily(context)
    hi = df.groupby("Code")["high"].transform(lambda s: s.rolling(5, min_periods=1).max())
    return _out(df, "close_to_high_5d", df["close"] / (hi + 1e-8))


# ── 隔夜 / 日内结构 ───────────────────────────────────────────────────────

@register_factor(
    name="gap_abs_ma_5d",
    description="5 日平均绝对跳空（|open/pre_close − 1| 的 5 日均值）。",
    category="price",
    thesis="gap_abs_ma_20d 的短周期版：近期跳空频率与幅度反映消息面活跃度，"
    "高跳空股票 1d 波动与可交易机会更大，与 1d 收益的尾部相关。",
    dependencies=("daily.parquet",),
)
def factor_gap_abs_ma_5d(context: FactorContext):
    df = _daily(context)
    return _out(df, "gap_abs_ma_5d", roll(df, "overnight", 5, "mean"))


@register_factor(
    name="overnight_std_5d",
    description="隔夜收益标准差（5 日）。",
    category="price",
    thesis="overnight_std_20d 的短周期版：隔夜波动反映盘后信息流强度，"
    "5 日窗口对信息事件更敏感，是 1d 波动与方向的先行指标。",
    dependencies=("daily.parquet",),
)
def factor_overnight_std_5d(context: FactorContext):
    df = _daily(context)
    return _out(df, "overnight_std_5d", roll(df, "overnight", 5, "std"))


@register_factor(
    name="intraday_vol_ratio_5d",
    description="日内/隔夜波动比（5 日 |日内收益| 均值 ÷ 5 日 |隔夜收益| 均值）。",
    category="price",
    thesis="日内波动占比高 = 盘中博弈主导（换手型），隔夜占比高 = 信息驱动；"
    "两者主导机制不同，比值识别 1d 收益的驱动类型。",
    dependencies=("daily.parquet",),
)
def factor_intraday_vol_ratio_5d(context: FactorContext):
    df = _daily(context)
    df["_ia"] = df["intraday"].abs()
    df["_oa"] = df["overnight"].abs()
    m_ia = roll(df, "_ia", 5, "mean")
    m_oa = roll(df, "_oa", 5, "mean")
    vals = m_ia / (m_oa + 1e-8)
    return _out(df, "intraday_vol_ratio_5d", vals)


# ── 流动性 / 量能冲击 ─────────────────────────────────────────────────────

@register_factor(
    name="amihud_daily_5",
    description="Amihud 非流动性（|日收益|/成交额×1e8 的 5 日均值）。",
    category="price",
    thesis="amihud_daily_20 的短周期版：5 日非流动性对资金流入流出更敏感，"
    "短期流动性冲击与 1d 收益反转相关。",
    dependencies=("daily.parquet",),
)
def factor_amihud_daily_5(context: FactorContext):
    df = _daily(context)
    df["amihud"] = df["absret"] / df["amount"].replace(0, np.nan) * 1e8
    return _out(df, "amihud_daily_5", roll(df, "amihud", 5, "mean", min_periods=2))


@register_factor(
    name="vol_ratio_ma3",
    description="3 日量比（当日成交量 / 3 日均量）。",
    category="volume",
    thesis="vol_ratio_ma5/ma20 的短周期版：3 日量比捕捉当日异动最锐利，"
    "放量+位置组合是 1d 短期信号经典来源。",
    dependencies=("daily.parquet",),
)
def factor_vol_ratio_ma3(context: FactorContext):
    df = _daily(context)
    mv = df.groupby("Code")["vol"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    vals = df["vol"] / (mv + 1e-8)
    return _out(df, "vol_ratio_ma3", vals)


@register_factor(
    name="amt_surge_3d",
    description="3 日成交额冲击（当日成交额 / 3 日均额）。",
    category="volume",
    thesis="成交额冲击比成交量更贴近资金规模：3 日窗口的突然放量常对应"
    "主力进出，次日有方向性延续或反转。",
    dependencies=("daily.parquet",),
)
def factor_amt_surge_3d(context: FactorContext):
    df = _daily(context)
    ma = df.groupby("Code")["amount"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    vals = df["amount"] / (ma + 1e-8)
    return _out(df, "amt_surge_3d", vals)


@register_factor(
    name="amt_mom_accel",
    description="成交额动量加速（3 日变化率 − 10 日变化率）。",
    category="volume",
    thesis="量能一阶动量反映资金流入方向，二阶（加速）反映资金行为切换；"
    "加速放量与 1d 动量/反转切换点相关。",
    dependencies=("daily.parquet",),
)
def factor_amt_mom_accel(context: FactorContext):
    df = _daily(context)
    g = df.groupby("Code")["amount"]
    p3 = g.pct_change(3)
    p10 = g.pct_change(10)
    vals = p3 - p10
    return _out(df, "amt_mom_accel", vals)


# ── 收益分布形态（短窗） ───────────────────────────────────────────────────

@register_factor(
    name="ret_skew_5d",
    description="5 日收益偏度。",
    category="price",
    thesis="ret_skew_20/60 的短周期版：5 日偏度识别脉冲式拉升/跳水形态，"
    "正偏度（脉冲上涨）后 1d 倾向反转，负偏度后倾向修复。",
    dependencies=("daily.parquet",),
)
def factor_ret_skew_5d(context: FactorContext):
    df = _daily(context)
    vals = df.groupby("Code")["ret"].transform(
        lambda s: s.rolling(5, min_periods=3).skew())
    return _out(df, "ret_skew_5d", vals)


@register_factor(
    name="ret_kurt_5d",
    description="5 日收益峰度。",
    category="price",
    thesis="5 日峰度识别极端行情聚集：高尖峰 = 单日极端波动主导，"
    "随后 1d 波动收缩与均值回归概率上升。",
    dependencies=("daily.parquet",),
)
def factor_ret_kurt_5d(context: FactorContext):
    df = _daily(context)
    vals = df.groupby("Code")["ret"].transform(
        lambda s: s.rolling(5, min_periods=3).kurt())
    return _out(df, "ret_kurt_5d", vals)
