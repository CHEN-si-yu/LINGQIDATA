"""
fac_new 第一、二轮日频因子（35 个，Class 1，数据源 daily.parquet）。

来源于 trainingdata/build_fac_new_r1.py + r2.py 的日频部分（2026-08-13 迭代
产物，沙盒重生成验证 51/51 与 fac_new.fea 一致后注册）。公式语义与原脚本
逐项对齐：收益用 close.pct_change()（原始价），隔夜=open/pre_close-1，
日内=close/open-1，行业归属见 _fac_new_common 说明。

注：amt_rel_ind_ma5 沿袭原脚本语义 = amount/个股全期均值 的 5 日均值
（名义"行业相对"，实际个股自身归一；个股常数不改变横截面排序，照抄保留）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ._fac_new_common import ind_mean, ind_std, roll, with_industry


def _daily(context: FactorContext) -> pd.DataFrame:
    """加载 daily 面板并补齐基础列（ret/overnight/intraday/absret/industry）。"""
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


# ── 行业相对动量 / 行业动量和离散度 ─────────────────────────────────────────

def _rel_mom(df: pd.DataFrame, n: int) -> pd.Series:
    df["ret_n"] = df.groupby("Code")["close"].pct_change(n)
    ind_ret_n = df.groupby(["Date", "industry"])["ret_n"].transform("mean")
    return df["ret_n"] - ind_ret_n


@register_factor(
    name="rel_mom_ind_5d",
    description="行业相对动量（个股5日收益 − 行业等权5日收益）。",
    category="sector",
    thesis="行业内相对动量剥离板块 β：同一行业里跑赢同伴的股票延续性更强，"
    "绝对动量受行业轮动干扰大，行业相对更稳定。",
    dependencies=("daily.parquet",),
)
def factor_rel_mom_ind_5d(context: FactorContext):
    df = _daily(context)
    return _out(df, "rel_mom_ind_5d", _rel_mom(df, 5))


@register_factor(
    name="rel_mom_ind_10d",
    description="行业相对动量（个股10日收益 − 行业等权10日收益）。",
    category="sector",
    thesis="10 日窗口介于 5d 与 20d 之间，捕捉中短期行业内相对趋势，"
    "兼顾响应速度与噪音过滤。",
    dependencies=("daily.parquet",),
)
def factor_rel_mom_ind_10d(context: FactorContext):
    df = _daily(context)
    return _out(df, "rel_mom_ind_10d", _rel_mom(df, 10))


@register_factor(
    name="rel_mom_ind_20d",
    description="行业相对动量（个股20日收益 − 行业等权20日收益）。",
    category="sector",
    thesis="月度行业相对动量，趋势更稳、换手更低，是行业内轮动的主信号。",
    dependencies=("daily.parquet",),
)
def factor_rel_mom_ind_20d(context: FactorContext):
    df = _daily(context)
    return _out(df, "rel_mom_ind_20d", _rel_mom(df, 20))


def _ind_ret_ma(context: FactorContext, n: int, name: str, description: str, thesis: str):
    df = _daily(context)
    ind_ret = ind_mean(df, "ret")
    df["ind_ret"] = ind_ret
    vals = roll(df, "ind_ret", n, "mean")
    return _out(df, name, vals)


@register_factor(
    name="ind_ret_ma_5d",
    description="行业收益动能（行业等权日收益的5日均值）。",
    category="sector",
    thesis="行业层面的短周期动能，反映板块整体资金情绪，作为行业内选股的背景项。",
    dependencies=("daily.parquet",),
)
def factor_ind_ret_ma_5d(context: FactorContext):
    return _ind_ret_ma(context, 5, "ind_ret_ma_5d", "行业收益动能（5日）", "行业短周期动能")


@register_factor(
    name="ind_ret_ma_20d",
    description="行业收益动能（行业等权日收益的20日均值）。",
    category="sector",
    thesis="行业层面的中周期动能，识别处于趋势中的板块。",
    dependencies=("daily.parquet",),
)
def factor_ind_ret_ma_20d(context: FactorContext):
    return _ind_ret_ma(context, 20, "ind_ret_ma_20d", "行业收益动能（20日）", "行业中周期动能")


@register_factor(
    name="ind_ret_ma_60d",
    description="行业收益动能（行业等权日收益的60日均值）。",
    category="sector",
    thesis="季度级别的行业趋势，过滤短期噪音，捕捉中期板块轮动方向。",
    dependencies=("daily.parquet",),
)
def factor_ind_ret_ma_60d(context: FactorContext):
    return _ind_ret_ma(context, 60, "ind_ret_ma_60d", "行业收益动能（60日）", "行业季度动能")


def _ind_disp_ma(context: FactorContext, n: int, name: str, description: str, thesis: str):
    df = _daily(context)
    ind_disp = ind_std(df, "ret")
    df["ind_disp"] = ind_disp
    vals = roll(df, "ind_disp", n, "mean")
    return _out(df, name, vals)


@register_factor(
    name="ind_disp_ma_5d",
    description="行业离散度（行业日收益横截面std的5日均值）。",
    category="sector",
    thesis="行业内个股分化程度：离散度高=行业内机会多但选股难度大，"
    "低=板块共振明显，适合行业 β 交易。",
    dependencies=("daily.parquet",),
)
def factor_ind_disp_ma_5d(context: FactorContext):
    return _ind_disp_ma(context, 5, "ind_disp_ma_5d", "行业离散度（5日）", "行业内分化度")


@register_factor(
    name="ind_disp_ma_20d",
    description="行业离散度（行业日收益横截面std的20日均值）。",
    category="sector",
    thesis="中周期行业分化度，反映板块选股空间与风格切换特征。",
    dependencies=("daily.parquet",),
)
def factor_ind_disp_ma_20d(context: FactorContext):
    return _ind_disp_ma(context, 20, "ind_disp_ma_20d", "行业离散度（20日）", "行业中周期分化度")


@register_factor(
    name="ind_disp_ma_60d",
    description="行业离散度（行业日收益横截面std的60日均值）。",
    category="sector",
    thesis="季度级别的行业分化度，标识行业进入分化/共振状态的持续性。",
    dependencies=("daily.parquet",),
)
def factor_ind_disp_ma_60d(context: FactorContext):
    return _ind_disp_ma(context, 60, "ind_disp_ma_60d", "行业离散度（60日）", "行业季度分化度")


# ── 隔夜/日内收益分解 ───────────────────────────────────────────────────────

def _overnight_ma(context: FactorContext, n: int, name: str, description: str, thesis: str):
    df = _daily(context)
    vals = roll(df, "overnight", n, "mean")
    return _out(df, name, vals)


@register_factor(
    name="overnight_ma_5d",
    description="隔夜收益均值（open/pre_close−1 的5日均值）。",
    category="price",
    thesis="隔夜收益承载消息面与情绪溢价，短窗口均值反映近期隔夜动量方向。",
    dependencies=("daily.parquet",),
)
def factor_overnight_ma_5d(context: FactorContext):
    return _overnight_ma(context, 5, "overnight_ma_5d", "隔夜收益均值（5日）", "隔夜动量")


@register_factor(
    name="overnight_ma_20d",
    description="隔夜收益均值（open/pre_close−1 的20日均值）。",
    category="price",
    thesis="月度隔夜收益方向，过滤单日噪音，刻画消息驱动的持续性溢价。",
    dependencies=("daily.parquet",),
)
def factor_overnight_ma_20d(context: FactorContext):
    return _overnight_ma(context, 20, "overnight_ma_20d", "隔夜收益均值（20日）", "隔夜动量月度")


@register_factor(
    name="overnight_ma_60d",
    description="隔夜收益均值（open/pre_close−1 的60日均值）。",
    category="price",
    thesis="季度隔夜收益倾向，识别长期存在稳定隔夜溢价/折价的标的。",
    dependencies=("daily.parquet",),
)
def factor_overnight_ma_60d(context: FactorContext):
    return _overnight_ma(context, 60, "overnight_ma_60d", "隔夜收益均值（60日）", "隔夜动量季度")


def _intraday_ma(context: FactorContext, n: int, name: str, description: str, thesis: str):
    df = _daily(context)
    vals = roll(df, "intraday", n, "mean")
    return _out(df, name, vals)


@register_factor(
    name="intraday_ma_5d",
    description="日内收益均值（close/open−1 的5日均值）。",
    category="price",
    thesis="日内收益承载交易行为溢价（承接/抛压），短窗口均值反映日内动量。",
    dependencies=("daily.parquet",),
)
def factor_intraday_ma_5d(context: FactorContext):
    return _intraday_ma(context, 5, "intraday_ma_5d", "日内收益均值（5日）", "日内动量")


@register_factor(
    name="intraday_ma_20d",
    description="日内收益均值（close/open−1 的20日均值）。",
    category="price",
    thesis="月度日内收益方向，捕捉资金日内承接能力的持续性。",
    dependencies=("daily.parquet",),
)
def factor_intraday_ma_20d(context: FactorContext):
    return _intraday_ma(context, 20, "intraday_ma_20d", "日内收益均值（20日）", "日内动量月度")


@register_factor(
    name="intraday_ma_60d",
    description="日内收益均值（close/open−1 的60日均值）。",
    category="price",
    thesis="季度日内收益倾向，识别交易行为长期稳定的标的。",
    dependencies=("daily.parquet",),
)
def factor_intraday_ma_60d(context: FactorContext):
    return _intraday_ma(context, 60, "intraday_ma_60d", "日内收益均值（60日）", "日内动量季度")


def _cum_prod(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    return df.groupby("Code")[col].transform(
        lambda s: (1 + s).rolling(n, min_periods=1).apply(np.prod, raw=True) - 1
    )


@register_factor(
    name="overnight_cum_20d",
    description="隔夜收益20日复利累计。",
    category="price",
    thesis="累计隔夜收益=消息面溢价的复利视角，比均值更能反映趋势型隔夜强势股。",
    dependencies=("daily.parquet",),
)
def factor_overnight_cum_20d(context: FactorContext):
    df = _daily(context)
    return _out(df, "overnight_cum_20d", _cum_prod(df, "overnight", 20))


@register_factor(
    name="intraday_cum_20d",
    description="日内收益20日复利累计。",
    category="price",
    thesis="累计日内收益=资金行为溢价的复利视角，捕捉持续被资金承接的标的。",
    dependencies=("daily.parquet",),
)
def factor_intraday_cum_20d(context: FactorContext):
    df = _daily(context)
    return _out(df, "intraday_cum_20d", _cum_prod(df, "intraday", 20))


@register_factor(
    name="overnight_intraday_ratio_20d",
    description="隔夜/日内收益强度比（20日均值之比，分母取绝对值）。",
    category="price",
    thesis="比值高=收益主要来自隔夜（消息/情绪驱动），低=日内交易行为主导；"
    "刻画两类收益来源的相对强弱。",
    dependencies=("daily.parquet",),
)
def factor_overnight_intraday_ratio_20d(context: FactorContext):
    df = _daily(context)
    oma = roll(df, "overnight", 20, "mean")
    ima = roll(df, "intraday", 20, "mean")
    return _out(df, "overnight_intraday_ratio_20d", oma / (ima.abs() + 1e-6))


# ── 波动 / 区间 / 位置 ───────────────────────────────────────────────────────

@register_factor(
    name="rel_vol_ind_20d",
    description="行业内相对波动（个股20日收益std − 行业等权均值）。",
    category="sector",
    thesis="同一行业内波动相对更低的股票风险调整后更优，剥离板块整体波动水平。",
    dependencies=("daily.parquet",),
)
def factor_rel_vol_ind_20d(context: FactorContext):
    df = _daily(context)
    df["vol20"] = roll(df, "ret", 20, "std", min_periods=5)
    vals = df["vol20"] - ind_mean(df, "vol20")
    return _out(df, "rel_vol_ind_20d", vals)


@register_factor(
    name="vol_of_vol_20d",
    description="波动之波动（|日收益|的20日std）。",
    category="risk",
    thesis="波动状态不稳定（vol-of-vol 高）的股票交易成本高、regime 切换频繁，"
    "低 vol-of-vol 标的波动环境一致、更可预测。",
    dependencies=("daily.parquet",),
)
def factor_vol_of_vol_20d(context: FactorContext):
    df = _daily(context)
    return _out(df, "vol_of_vol_20d", roll(df, "absret", 20, "std", min_periods=5))


@register_factor(
    name="vol_of_vol_60d",
    description="波动之波动（|日收益|的60日std）。",
    category="risk",
    thesis="季度级别的波动状态稳定性，比 20d 更稳健地刻画波动 regime 切换频率。",
    dependencies=("daily.parquet",),
)
def factor_vol_of_vol_60d(context: FactorContext):
    df = _daily(context)
    return _out(df, "vol_of_vol_60d", roll(df, "absret", 60, "std", min_periods=10))


@register_factor(
    name="overnight_std_20d",
    description="隔夜收益波动（overnight 的20日std）。",
    category="risk",
    thesis="隔夜波动反映消息面不确定性：高隔夜波动=定价信息冲击频繁，"
    "风险溢价要求更高。",
    dependencies=("daily.parquet",),
)
def factor_overnight_std_20d(context: FactorContext):
    df = _daily(context)
    return _out(df, "overnight_std_20d", roll(df, "overnight", 20, "std", min_periods=5))


@register_factor(
    name="ret_autocorr_20d",
    description="收益自相关（日收益20日滚动 lag-1 自相关）。",
    category="risk",
    thesis="正自相关=动量延续，负自相关=日内反转；刻画收益序列的持续性/均值回归特征。",
    dependencies=("daily.parquet",),
)
def factor_ret_autocorr_20d(context: FactorContext):
    df = _daily(context)
    vals = df.groupby("Code")["ret"].transform(
        lambda s: s.rolling(20, min_periods=10).corr(s.shift(1))
    )
    return _out(df, "ret_autocorr_20d", vals)


def _range_position(df: pd.DataFrame, n: int) -> pd.Series:
    h = roll(df, "high", n, "max")
    l = roll(df, "low", n, "min")
    return (df["close"] - l) / (h - l + 1e-8)


@register_factor(
    name="range_position_20d",
    description="20日区间位置（close 在20日 high-low 区间内的位置）。",
    category="price",
    thesis="接近区间上沿=短期强势/突破在即，接近下沿=超跌；区间位置是经典的趋势位置信号。",
    dependencies=("daily.parquet",),
)
def factor_range_position_20d(context: FactorContext):
    df = _daily(context)
    return _out(df, "range_position_20d", _range_position(df, 20))


@register_factor(
    name="range_position_60d",
    description="60日区间位置（close 在60日 high-low 区间内的位置）。",
    category="price",
    thesis="季度区间位置过滤短期噪音，识别中期趋势所处阶段。",
    dependencies=("daily.parquet",),
)
def factor_range_position_60d(context: FactorContext):
    df = _daily(context)
    return _out(df, "range_position_60d", _range_position(df, 60))


@register_factor(
    name="close_location_20d",
    description="收盘位置（20日收益 ÷ 20日振幅）。",
    category="price",
    thesis="同振幅下收益越高=上攻效率越高，刻画 20 日上涨质量而非单纯涨幅。",
    dependencies=("daily.parquet",),
)
def factor_close_location_20d(context: FactorContext):
    df = _daily(context)
    h20 = roll(df, "high", 20, "max")
    l20 = roll(df, "low", 20, "min")
    chg20 = df.groupby("Code")["close"].shift(20)
    vals = (df["close"] - chg20) / (h20 - l20 + 1e-8)
    return _out(df, "close_location_20d", vals)


@register_factor(
    name="close_to_high_20d",
    description="收盘距20日最高价距离（close/max20(high)−1）。",
    category="price",
    thesis="越接近 20 日高点=突破形态越完整，创新高动量在 A 股有正溢价。",
    dependencies=("daily.parquet",),
)
def factor_close_to_high_20d(context: FactorContext):
    df = _daily(context)
    h20 = roll(df, "high", 20, "max")
    return _out(df, "close_to_high_20d", df["close"] / h20 - 1)


@register_factor(
    name="up_day_freq_20d",
    description="上涨天数频率（20日内 close>前收盘 的天数占比）。",
    category="price",
    thesis="上涨频率比累计涨幅更稳健：高频率=胜率高、路径平稳，规避暴涨暴跌型标的。",
    dependencies=("daily.parquet",),
)
def factor_up_day_freq_20d(context: FactorContext):
    df = _daily(context)
    up = (df.groupby("Code")["close"].diff() > 0).astype(float)
    df["up"] = up
    return _out(df, "up_day_freq_20d", roll(df, "up", 20, "mean"))


# ── 量能 / 流动性 ───────────────────────────────────────────────────────────

@register_factor(
    name="amt_rel_ind",
    description="行业内相对量能（amount 行业内 z-score）。",
    category="price",
    thesis="行业内成交额异常放大=资金关注度提升，z-score 剥离板块整体交投水平。",
    dependencies=("daily.parquet",),
)
def factor_amt_rel_ind(context: FactorContext):
    df = _daily(context)
    vals = (df["amount"] - ind_mean(df, "amount")) / (ind_std(df, "amount") + 1e-8)
    return _out(df, "amt_rel_ind", vals)


@register_factor(
    name="amt_rel_ind_ma5",
    description="量能相对自身均值（amount/个股全期均值 的5日均值）。",
    category="price",
    thesis="成交额相对自身历史水平的短期抬升，反映个股层面的量能异动"
    "（注：沿袭原脚本语义，实为个股自身归一，非行业相对）。",
    dependencies=("daily.parquet",),
)
def factor_amt_rel_ind_ma5(context: FactorContext):
    df = _daily(context)
    vals = df.groupby("Code")["amount"].transform(
        lambda s: (s / s.mean()).rolling(5, min_periods=1).mean()
    )
    return _out(df, "amt_rel_ind_ma5", vals)


@register_factor(
    name="amihud_amt_20d",
    description="Amihud 非流动性（|日收益|/成交额×1e8 的20日均值）。",
    category="price",
    thesis="单位成交额推动的价格变动越大=流动性越差，非流动性溢价在 A 股显著，"
    "高 amihud 小盘股未来收益更高。",
    dependencies=("daily.parquet",),
)
def factor_amihud_amt_20d(context: FactorContext):
    df = _daily(context)
    df["amihud"] = df["absret"] / df["amount"].replace(0, np.nan) * 1e8
    return _out(df, "amihud_amt_20d", roll(df, "amihud", 20, "mean", min_periods=5))


# ── 缺口 / 隔夜形态 ─────────────────────────────────────────────────────────

@register_factor(
    name="gap_abs_ma_20d",
    description="缺口幅度均值（|open/pre_close−1| 的20日均值）。",
    category="price",
    thesis="平均跳空幅度反映信息冲击强度与交易拥挤度，高跳空股波动与成本更高。",
    dependencies=("daily.parquet",),
)
def factor_gap_abs_ma_20d(context: FactorContext):
    df = _daily(context)
    df["gap_abs"] = df["overnight"].abs()
    return _out(df, "gap_abs_ma_20d", roll(df, "gap_abs", 20, "mean"))


@register_factor(
    name="overnight_sign_consistency_20d",
    description="隔夜方向一致性（20日内 overnight>0 的天数占比）。",
    category="price",
    thesis="隔夜收益正占比高=消息面持续偏多，方向一致性强于幅度，稳健捕捉情绪溢价。",
    dependencies=("daily.parquet",),
)
def factor_overnight_sign_consistency_20d(context: FactorContext):
    df = _daily(context)
    df["overnight_pos"] = (df["overnight"] > 0).astype(float)
    return _out(df, "overnight_sign_consistency_20d", roll(df, "overnight_pos", 20, "mean"))


@register_factor(
    name="gap_up_fade_freq_20d",
    description="高开低走频率（20日内 overnight>0 且 intraday<0 的天数占比）。",
    category="price",
    thesis="高开低走=冲高抛压、情绪透支，高频出现说明上方套牢盘重，短期承压。",
    dependencies=("daily.parquet",),
)
def factor_gap_up_fade_freq_20d(context: FactorContext):
    df = _daily(context)
    df["gap_up_fade"] = ((df["overnight"] > 0) & (df["intraday"] < 0)).astype(float)
    return _out(df, "gap_up_fade_freq_20d", roll(df, "gap_up_fade", 20, "mean"))


@register_factor(
    name="gap_down_recover_freq_20d",
    description="低开高走频率（20日内 overnight<0 且 intraday>0 的天数占比）。",
    category="price",
    thesis="低开高走=承接有力、恐慌被消化，高频出现说明买盘韧性好，短期偏强。",
    dependencies=("daily.parquet",),
)
def factor_gap_down_recover_freq_20d(context: FactorContext):
    df = _daily(context)
    df["gap_down_rec"] = ((df["overnight"] < 0) & (df["intraday"] > 0)).astype(float)
    return _out(df, "gap_down_recover_freq_20d", roll(df, "gap_down_rec", 20, "mean"))
