from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, rolling_group_std, rolling_group_skew, rolling_group_kurt
from ..utils import rolling_group_max, rolling_group_min


@register_factor(
    name="mom_20",
    description="20日动量因子，基于复权收盘价的20日收益率。",
    category="price",
    thesis="中期动量效应在A股截面中存在显著的正向预测能力，是量价类最核心的基础因子之一。",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    mom = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(20)
    )
    return cross_sectional_rank(mom)


# ── Momentum ────────────────────────────────────────────────────────────

@register_factor(
    name="mom_5",
    description="5日动量因子，基于复权收盘价的5日收益率截面排名。",
    category="price",
    thesis="短期动量捕捉近一周的价格趋势延续性。",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_5(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(5)
    )
    return cross_sectional_rank(ret)


@register_factor(
    name="mom_10",
    description="10日动量因子，基于复权收盘价的10日收益率截面排名。",
    category="price",
    thesis="双周动量在A股中兼具稳定性和灵敏度。",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_10(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(10)
    )
    return cross_sectional_rank(ret)


@register_factor(
    name="mom_60",
    description="60日动量因子，基于复权收盘价的60日收益率截面排名。",
    category="price",
    thesis="中长期动量捕捉季度的趋势性，是传统动量策略的标准窗口。",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(60)
    )
    return cross_sectional_rank(ret)


@register_factor(
    name="mom_120_skip5",
    description="120日动量（跳过最近5日），排除短期反转效应。",
    category="price",
    thesis="中长期动量剔除最近一周可分离中期趋势与短期反转，提升因子稳定性。",
    dependencies=("daily_adj.parquet",),
)
def factor_mom_120_skip5(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(
        lambda s: s.shift(5) / s.shift(120).replace(0, np.nan) - 1.0
    )
    return cross_sectional_rank(ret)


# ── Reversal ────────────────────────────────────────────────────────────

@register_factor(
    name="reversal_1",
    description="1日反转因子，负的1日收益率截面排名（高值=近期跌幅大）。",
    category="price",
    thesis="A股短期反转效应显著，前一日涨幅大的股票次日倾向于回落。",
    dependencies=("daily_adj.parquet",),
)
def factor_reversal_1(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: -s.pct_change(1)
    )
    return cross_sectional_rank(ret)


@register_factor(
    name="reversal_5",
    description="5日反转因子，负的5日收益率截面排名。",
    category="price",
    thesis="周度反转效应反映短期超买超卖后的均值回归。",
    dependencies=("daily_adj.parquet",),
)
def factor_reversal_5(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: -s.pct_change(5)
    )
    return cross_sectional_rank(ret)


@register_factor(
    name="reversal_10",
    description="10日反转因子，负的10日收益率截面排名。",
    category="price",
    thesis="双周反转捕捉短期过度反应后的价格修正。",
    dependencies=("daily_adj.parquet",),
)
def factor_reversal_10(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: -s.pct_change(10)
    )
    return cross_sectional_rank(ret)


# ── Volatility ──────────────────────────────────────────────────────────

@register_factor(
    name="volatility_20",
    description="20日波动率因子，20日收益率标准差截面排名（低波动排前）。",
    category="price",
    thesis="低波动异象在A股中显著存在，低波动股票未来收益更高。",
    dependencies=("daily_adj.parquet",),
)
def factor_volatility_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    daily_ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(1)
    )
    vol = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-vol)


@register_factor(
    name="volatility_60",
    description="60日波动率因子，60日收益率标准差截面排名（低波动排前）。",
    category="price",
    thesis="中长期低波动异象比短期更稳定。",
    dependencies=("daily_adj.parquet",),
)
def factor_volatility_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    daily_ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(1)
    )
    vol = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    return cross_sectional_rank(-vol)


@register_factor(
    name="downside_vol_20",
    description="20日下行波动率因子，仅计入负收益日的标准差截面排名（下行波动低排前）。",
    category="price",
    thesis="下行波动率比总波动率更精确地刻画尾部风险，投资者对下跌波动更敏感。",
    dependencies=("daily_adj.parquet",),
)
def factor_downside_vol_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    daily_ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(1)
    )
    down_ret = daily_ret.clip(upper=0)
    down_vol = down_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-down_vol)


# ── Amplitude / Range ───────────────────────────────────────────────────

@register_factor(
    name="amplitude_20",
    description="20日均振幅因子，(high-low)/close 的20日均值截面排名（低振幅排前）。",
    category="price",
    thesis="振幅是流动性与不确定性的综合指标，低振幅反映筹码稳定性。",
    dependencies=("daily_adj.parquet",),
)
def factor_amplitude_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    amp = (daily_adj["high"] - daily_adj["low"]) / daily_adj["close"].replace(0, np.nan)
    avg_amp = amp.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_amp)


# ── MA Bias ─────────────────────────────────────────────────────────────

@register_factor(
    name="bias_20",
    description="20日均线乖离率，close/ma_20 - 1 的截面排名。",
    category="price",
    thesis="均线乖离反映价格对中期成本的偏离程度，极端乖离预示均值回归。",
    dependencies=("daily_adj.parquet",),
)
def factor_bias_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ma_20 = close.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    bias = close / ma_20.replace(0, np.nan) - 1.0
    return cross_sectional_rank(bias)


# ── RSI ─────────────────────────────────────────────────────────────────

@register_factor(
    name="rsi_14",
    description="14日相对强弱指标(RSI)截面排名。",
    category="price",
    thesis="RSI是经典超买超卖指标，高RSI股票短期有回调压力。",
    dependencies=("daily_adj.parquet",),
)
def factor_rsi_14(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    delta = close.groupby(level="Code").transform(lambda s: s.diff())
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(level="Code").transform(
        lambda s: s.rolling(14, min_periods=7).mean()
    )
    avg_loss = loss.groupby(level="Code").transform(
        lambda s: s.rolling(14, min_periods=7).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return cross_sectional_rank(-rsi)


# ── Max Drawdown ────────────────────────────────────────────────────────

@register_factor(
    name="max_drawdown_60",
    description="60日最大回撤因子，截面排名（回撤越大排越前=反转预期）。",
    category="price",
    thesis="大幅回撤后的反弹效应在A股中具有一定的截面预测能力。",
    dependencies=("daily_adj.parquet",),
)
def factor_max_drawdown_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    def _max_dd(s):
        peak = s.rolling(60, min_periods=30).max()
        dd = s / peak.replace(0, np.nan) - 1.0
        return dd.rolling(60, min_periods=30).min()

    mdd = close.groupby(level="Code").transform(_max_dd)
    return cross_sectional_rank(mdd)


# ── Return distribution ─────────────────────────────────────────────────

@register_factor(
    name="ret_skew_20",
    description="20日收益率偏度因子截面排名（正偏=右偏，高值更优）。",
    category="price",
    thesis="收益率正偏度反映上涨弹性，正偏股票更受趋势交易者青睐。",
    dependencies=("daily_adj.parquet",),
)
def factor_ret_skew_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    daily_ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(1)
    )
    skew = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).skew()
    )
    return cross_sectional_rank(skew)


@register_factor(
    name="ret_kurt_20",
    description="20日收益率峰度因子截面排名（高峰度排后=极端值风险）。",
    category="price",
    thesis="收益率高峰度意味着极端波动概率高，是风险信号。",
    dependencies=("daily_adj.parquet",),
)
def factor_ret_kurt_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    daily_ret = daily_adj.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(1)
    )
    kurt = daily_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).kurt()
    )
    return cross_sectional_rank(-kurt)


# ── Volume ──────────────────────────────────────────────────────────────

@register_factor(
    name="volume_ratio_20",
    description="20日相对成交量因子，vol/avg_vol_20 - 1 截面排名。",
    category="price",
    thesis="放量上涨和缩量下跌都是技术面确认信号，成交量异常值得关注。",
    dependencies=("daily_adj.parquet",),
)
def factor_volume_ratio_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    vol = daily_adj["vol"]
    avg_vol = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    v_ratio = vol / avg_vol.replace(0, np.nan) - 1.0
    return cross_sectional_rank(-v_ratio)


@register_factor(
    name="amount_ratio_20",
    description="20日相对成交额因子，amount/avg_amount_20 - 1 截面排名。",
    category="price",
    thesis="成交额比成交量更能反映资金参与度，异常放量常伴随趋势转折。",
    dependencies=("daily_adj.parquet",),
)
def factor_amount_ratio_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    amount = daily_adj["amount"]
    avg_amount = amount.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    a_ratio = amount / avg_amount.replace(0, np.nan) - 1.0
    return cross_sectional_rank(-a_ratio)


# ── Shadow / Gap (daily.parquet — non-adjusted for real price geometry) ─

@register_factor(
    name="shadow_upper_20",
    description="20日均上影线比例，上影线/(high-low) 截面排名。",
    category="price",
    thesis="上影线反映高位抛压，长期高上影线比例是上涨阻力信号。",
    dependencies=("daily.parquet",),
)
def factor_shadow_upper_20(context: FactorContext):
    daily = context.load("daily.parquet")
    upper_shadow = daily["high"] - daily[["open", "close"]].max(axis=1)
    body_range = daily["high"] - daily["low"]
    ratio = upper_shadow / body_range.replace(0, np.nan)
    avg_ratio = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_ratio)


@register_factor(
    name="shadow_lower_20",
    description="20日均下影线比例，下影线/(high-low) 截面排名（高值=强支撑）。",
    category="price",
    thesis="下影线反映低位承接力，高下影线比例是底部支撑信号。",
    dependencies=("daily.parquet",),
)
def factor_shadow_lower_20(context: FactorContext):
    daily = context.load("daily.parquet")
    lower_shadow = daily[["open", "close"]].min(axis=1) - daily["low"]
    body_range = daily["high"] - daily["low"]
    ratio = lower_shadow / body_range.replace(0, np.nan)
    avg_ratio = ratio.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(avg_ratio)


@register_factor(
    name="gap_ratio_20",
    description="20日均跳空比率因子，open/pre_close - 1 截面排名。",
    category="price",
    thesis="向上跳空缺口反映隔夜利好信息，跳空后短期存在反转压力。",
    dependencies=("daily.parquet",),
)
def factor_gap_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = daily["open"] / daily["pre_close"].replace(0, np.nan) - 1.0
    avg_gap = gap.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-avg_gap)


# ── Price position ──────────────────────────────────────────────────────

@register_factor(
    name="price_position_60",
    description="60日价格位置，(close-60d_low)/(60d_high-60d_low) 截面排名。",
    category="price",
    thesis="价格在近60日区间内的相对位置反映短期趋势强度。",
    dependencies=("daily_adj.parquet",),
)
def factor_price_position_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    high_60 = close.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    )
    low_60 = close.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).min()
    )
    position = (close - low_60) / (high_60 - low_60).replace(0, np.nan)
    return cross_sectional_rank(position)


@register_factor(
    name="ma_convergence_20_60",
    description="均线收敛因子，20日均线与60日均线的距离比率截面排名。",
    category="price",
    thesis="短均线相对长均线的偏离程度反映趋势加速/减速，极端收敛后常伴随趋势突破。",
    dependencies=("daily_adj.parquet",),
)
def factor_ma_convergence_20_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ma_20 = close.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    ma_60 = close.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    convergence = ma_20 / ma_60.replace(0, np.nan) - 1.0
    return cross_sectional_rank(convergence)


# ── Weekly / Monthly momentum ───────────────────────────────────────────

@register_factor(
    name="mom_4w",
    description="4周动量因子，基于周线复权收盘价的4周收益率截面排名。",
    category="price",
    thesis="周频动量比日频动量噪声更低，4周（约20日）动量信号更稳定。",
    dependencies=("kline_adj_weekly.parquet",),
)
def factor_mom_4w(context: FactorContext):
    weekly = context.load("kline_adj_weekly.parquet")
    ret = weekly.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(4)
    )
    return cross_sectional_rank(ret)


@register_factor(
    name="mom_12w",
    description="12周动量因子，基于周线的12周收益率截面排名（约60日）。",
    category="price",
    thesis="12周动量捕捉季度趋势，周频信号日频应用可降低换手率。",
    dependencies=("kline_adj_weekly.parquet",),
)
def factor_mom_12w(context: FactorContext):
    weekly = context.load("kline_adj_weekly.parquet")
    ret = weekly.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(12)
    )
    return cross_sectional_rank(ret)


@register_factor(
    name="mom_3m",
    description="3月动量因子，基于月线复权收盘价的3月收益率截面排名。",
    category="price",
    thesis="季度动量是长期趋势交易的核心，月频信号稳定性最高。",
    dependencies=("kline_adj_monthly.parquet",),
)
def factor_mom_3m(context: FactorContext):
    monthly = context.load("kline_adj_monthly.parquet")
    ret = monthly.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(3)
    )
    return cross_sectional_rank(ret)


@register_factor(
    name="mom_6m",
    description="6月动量因子，基于月线的6月收益率截面排名。",
    category="price",
    thesis="半年度动量是中长期趋势的经典度量。",
    dependencies=("kline_adj_monthly.parquet",),
)
def factor_mom_6m(context: FactorContext):
    monthly = context.load("kline_adj_monthly.parquet")
    ret = monthly.groupby(level="Code")["close"].transform(
        lambda s: s.pct_change(6)
    )
    return cross_sectional_rank(ret)


# ── Breakout ──────────────────────────────────────────────────────────────

@register_factor(
    name="breakout_60",
    description="60日价格突破强度因子，close/max(high,60)-1截面排名。",
    category="price",
    thesis="价格突破近期高点反映上涨动能强劲，突破强度越高趋势延续性越强。",
    dependencies=("daily_adj.parquet",),
)
def factor_breakout_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    high = daily_adj["high"]
    high_max = high.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).max()
    )
    breakout = close / high_max.replace(0, np.nan) - 1.0
    return cross_sectional_rank(breakout)


# ── ATR ───────────────────────────────────────────────────────────────────

@register_factor(
    name="atr_20",
    description="20日平均真实波幅(ATR)因子，低ATR排前。",
    category="price",
    thesis="低ATR股票波动平稳、筹码稳定，高ATR意味着剧烈波动风险，低波异象支持低ATR溢价。",
    dependencies=("daily.parquet",),
)
def factor_atr_20(context: FactorContext):
    daily = context.load("daily.parquet")
    high = daily["high"]
    low = daily["low"]
    pre_close = daily["pre_close"]

    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    def _tr(grp):
        return pd.DataFrame({
            "hl": grp["high"] - grp["low"],
            "hpc": (grp["high"] - grp["pre_close"]).abs(),
            "lpc": (grp["low"] - grp["pre_close"]).abs(),
        }).max(axis=1)

    df = pd.DataFrame({"high": high, "low": low, "pre_close": pre_close})
    tr = df.groupby(level="Code").apply(_tr)
    if isinstance(tr.index, pd.MultiIndex):
        tr = tr.droplevel(0)
    tr = tr.sort_index()

    atr = tr.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(-atr)


# ── Pct_chg volatility ────────────────────────────────────────────────────

@register_factor(
    name="pct_chg_vol_20",
    description="20日涨跌幅波动率因子，基于pct_chg的20日标准差截面排名（低波动排前）。",
    category="price",
    thesis="涨跌幅波动率是对收益波动的另一种度量，低涨跌幅波动反映价格运行平稳。",
    dependencies=("daily_adj.parquet",),
)
def factor_pct_chg_vol_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    pct_chg = daily_adj["pct_chg"]
    vol = pct_chg.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    return cross_sectional_rank(-vol)
