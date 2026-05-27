from __future__ import annotations

import numpy as np

import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank
from ..utils import rolling_group_max, rolling_group_min, rolling_group_mean, rolling_group_std


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


@register_factor(
    name="gap_fill_5d_reversal",
    description="缺口回补反转因子，5日内出现向下跳空后的回补倾向截面排名。",
    category="price",
    thesis="A股'缺口必补'的民间规律有一定统计基础——向下跳空缺口在短期内面临均值回归压力。因子计算：(open-pre_close)/pre_close在5日内的min，取负向（即缺口越深=回补概率越大=正向预期）。",
    dependencies=("daily.parquet",),
)
def factor_gap_fill_5d_reversal(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = daily["open"] / daily["pre_close"].replace(0, np.nan) - 1.0
    # 5-day rolling minimum gap (most negative gap in recent 5 days)
    min_gap_5d = gap.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).min()
    )
    return cross_sectional_rank(-min_gap_5d)


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
    # Row-wise operation — no per-stock groupby needed
    tr = np.maximum(
        high - low,
        np.maximum((high - pre_close).abs(), (low - pre_close).abs()),
    )

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


# ── Timeseries factors ───────────────────────────────────────────────────

@register_factor(
    name="ts_mom_accel_20",
    description="20日动量加速度因子，当前动量与20日前动量的差值截面排名。",
    category="timeseries",
    thesis="动量加速度捕捉趋势的加速/减速信号。动量上升加速中的股票趋势强度增强，动量减速甚至转为下降的股票趋势衰竭，加速度因子可以提前识别趋势拐点。",
    dependencies=("daily_adj.parquet",),
)
def factor_ts_mom_accel_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    mom_20 = close.groupby(level="Code").transform(
        lambda s: s.pct_change(20)
    )
    mom_20_ago = mom_20.groupby(level="Code").transform(lambda s: s.shift(20))

    accel = mom_20 - mom_20_ago
    return cross_sectional_rank(accel)


@register_factor(
    name="ts_vol_regime_60",
    description="波动率历史分位因子，当前20日波动率在自身252日历史中的分位数截面排名。",
    category="timeseries",
    thesis="波动率具有聚集效应，当前波动率相对自身历史的位置（波动率体制）比绝对波动率更具预测力。高体制（波动率处于历史高位）往往对应恐慌/不确定性，低体制对应平稳期，不同体制下因子表现可能截然不同。",
    dependencies=("daily_adj.parquet",),
)
def factor_ts_vol_regime_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    ret_1d = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_20 = rolling_group_std(ret_1d, 20)

    def _expanding_pct_rank(s: pd.Series) -> pd.Series:
        return s.expanding(min_periods=60).rank(pct=True)

    regime = vol_20.groupby(level="Code").transform(_expanding_pct_rank)
    return cross_sectional_rank(regime)


@register_factor(
    name="ts_price_self_rank_60",
    description="价格自身60日位置因子，收盘价在自身60日高低区间的相对位置截面排名。",
    category="timeseries",
    thesis="现有price_position_60是截面对比，ts_price_self_rank_60是个股价格在自身60日范围内的位置，捕捉个股自身的超买超卖状态，与截面因子互补。",
    dependencies=("daily_adj.parquet",),
)
def factor_ts_price_self_rank_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]

    high_60 = rolling_group_max(close, 60)
    low_60 = rolling_group_min(close, 60)

    position = (close - low_60) / (high_60 - low_60).replace(0, np.nan)
    return cross_sectional_rank(position)


@register_factor(
    name="ts_volume_zscore_20",
    description="成交量20日Z-score因子，当日成交量偏离20日均值的标准差数截面排名（高放量排后）。",
    category="timeseries",
    thesis="成交量异常放大（高Z-score）往往伴随信息冲击、主力进出或市场过度关注，后续可能面临反转压力。低Z-score（缩量）则可能处于蓄势阶段。Z-score标准化使不同股票的成交量更具可比性。",
    dependencies=("daily_adj.parquet",),
)
def factor_ts_volume_zscore_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    vol = daily_adj["vol"]

    mean_20 = rolling_group_mean(vol, 20)
    std_20 = rolling_group_std(vol, 20)

    zscore = (vol - mean_20) / std_20.replace(0, np.nan)
    return cross_sectional_rank(-zscore)


# ── Market microstructure / liquidity ───────────────────────────────────


def _rolling_autocorr(series: pd.Series, window: int = 60, min_periods: int = 30) -> pd.Series:
    """Efficient rolling lag-1 autocorrelation via covariance decomposition.

    Returns a Series aligned with the input, clipped to [-1, 1].
    """
    lagged = series.groupby(level="Code").shift(1)
    product = series * lagged

    mean_s = series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods).mean()
    )
    mean_l = lagged.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods).mean()
    )
    mean_prod = product.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods).mean()
    )
    std_s = series.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )
    std_l = lagged.groupby(level="Code").transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )

    cov = mean_prod - mean_s * mean_l
    denom = std_s * std_l
    return (cov / denom.replace(0, np.nan)).clip(-1, 1)


@register_factor(
    name="amihud_illiq_20",
    description="Amihud非流动性因子，20日日均|收益率|/成交额截面排名（正向）。",
    category="price",
    thesis="Amihud非流动性是学术文献中验证最充分的流动性因子——高非流动性股票面临更大的价格冲击成本，投资者要求更高的预期收益作为流动性补偿。与换手率互补：一个衡量成交频率，一个衡量价格冲击。",
    dependencies=("daily_adj.parquet",),
)
def factor_amihud_illiq_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(lambda s: s.pct_change(1))
    illiq = ret.abs() / daily_adj["amount"].replace(0, np.nan)
    amihud = illiq.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(amihud)


@register_factor(
    name="close_position_intraday_20",
    description="收盘价日内位置因子，20日均(close-low)/(high-low)截面排名。",
    category="price",
    thesis="收盘价在日内区间的相对位置揭示买卖压力——连续在区间高位收盘代表买盘主导（正向），在低位收盘代表卖盘主导。均线平滑后过滤单日噪音。",
    dependencies=("daily_adj.parquet",),
)
def factor_close_position_intraday_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    denom = (daily_adj["high"] - daily_adj["low"]).replace(0, np.nan)
    position = (daily_adj["close"] - daily_adj["low"]) / denom
    avg_pos = position.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(avg_pos)


@register_factor(
    name="volume_surge_3d",
    description="成交量脉冲因子，3日最大(vol/60日中位数vol)截面排名（负向：脉冲后反转）。",
    category="price",
    thesis="成交量短期急剧放大往往是信息冲击或情绪顶点的标志。极端放量后A股存在显著的反转效应——放量脉冲越大，后续回调压力越强。使用中位数而非均值避免极端日污染基线。",
    dependencies=("daily_adj.parquet",),
)
def factor_volume_surge_3d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    vol = daily_adj["vol"]
    vol_median_60 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).median()
    )
    vol_ratio = vol / vol_median_60.replace(0, np.nan)
    surge = vol_ratio.groupby(level="Code").transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    return cross_sectional_rank(-surge)


@register_factor(
    name="tail_risk_dd_5d",
    description="尾部风险因子，5日最大回撤/20日波动率截面排名（正向：极端回撤后反转补偿）。",
    category="price",
    thesis="近期经历剧烈回撤的股票存在反转潜力。回撤幅度相对于波动率越大（'尾部风险夏普比'越高），反弹力量越强。该因子捕捉恐慌性抛售后的超跌反弹机会。",
    dependencies=("daily_adj.parquet",),
)
def factor_tail_risk_dd_5d(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    rolling_max_5 = close.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).max()
    )
    dd = close / rolling_max_5.replace(0, np.nan) - 1.0  # negative, 0 to -1
    ret_1d = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_20 = ret_1d.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    tail_ratio = dd.abs() / vol_20.replace(0, np.nan)
    return cross_sectional_rank(tail_ratio)


@register_factor(
    name="serial_corr_60",
    description="序列相关因子，60日收益率一阶自相关系数截面排名（正相关=趋势持续）。",
    category="timeseries",
    thesis="收益序列相关性是趋势质量的核心指标。正序列相关=动量延续性强（正向），负序列相关=均值回复占主导。与动量幅度因子互补：一个衡量趋势有多强，一个衡量趋势稳不稳。",
    dependencies=("daily_adj.parquet",),
)
def factor_serial_corr_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(lambda s: s.pct_change(1))
    autocorr = _rolling_autocorr(ret, window=60, min_periods=30)
    return cross_sectional_rank(autocorr)


@register_factor(
    name="volume_adjusted_momentum_20",
    description="成交量加权动量因子，(20日sum(ret×vol)/sum(vol))×20截面排名。",
    category="enhanced",
    thesis="传统动量等权对待毎一日的收益，但高成交量日的价格变动蕴含更多信息。成交量加权动量赋予放量日更高权重——量价配合的上涨比缩量上涨更可靠。这是学术上的VWAP动量，实证表现优于简单动量。",
    dependencies=("daily_adj.parquet",),
)
def factor_volume_adjusted_momentum_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    vol = daily_adj["vol"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    weighted_ret = ret * vol
    sum_wr = weighted_ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    sum_vol = vol.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    vwap_mom = (sum_wr / sum_vol.replace(0, np.nan)) * 20.0
    return cross_sectional_rank(vwap_mom)


@register_factor(
    name="realized_skewness_20",
    description="已实现偏度因子，20日日收益率的偏度截面排名（负向：负偏=崩盘风险溢价）。",
    category="price",
    thesis="收益负偏度是崩盘风险的代理变量。历史收益呈负偏分布的股票面临更高的左尾风险，投资者要求更高的预期收益作为补偿。该效应在A股小盘中尤为显著。",
    dependencies=("daily_adj.parquet",),
)
def factor_realized_skewness_20(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    ret = daily_adj.groupby(level="Code")["close"].transform(lambda s: s.pct_change(1))
    skew = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).skew()
    )
    return cross_sectional_rank(-skew)


# ── Volatility term structure ──────────────────────────────────────────


@register_factor(
    name="volatility_term_structure",
    description="波动率期限结构因子，(60日波动率/20日波动率-1)截面排名（负向：陡峭曲线=波动率预期上升）。",
    category="timeseries",
    thesis="借鉴期权定价中的波动率期限结构概念——当短期波动率低于长期波动率时（曲线陡峭/Contango），波动率被预期将上升，这对股票预期收益形成压力。曲线平坦或倒挂（短期已在高波动中）反而是正向信号。",
    dependencies=("daily_adj.parquet",),
)
def factor_volatility_term_structure(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    ret = close.groupby(level="Code").transform(lambda s: s.pct_change(1))
    vol_20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    vol_60 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).std()
    )
    term_slope = vol_60 / vol_20.replace(0, np.nan) - 1.0
    return cross_sectional_rank(-term_slope)


# ── Return acceleration (non-overlapping) ──────────────────────────────


@register_factor(
    name="return_accel_nonoverlap_20_60",
    description="收益加速度因子（非重叠窗口），(近20日动量)-(20日前60日动量)截面排名。",
    category="timeseries",
    thesis="比较两个非重叠窗口的动量——近期20天vs此前60天（80天前至20天前）。正加速度=近期趋势强于历史趋势、资金在加速流入；负加速度=趋势在衰竭。与ts_mom_accel_20互补：一个比较相邻等宽窗口（短期加速度），一个比较非重叠异宽窗口（中长期加速度变化）。",
    dependencies=("daily_adj.parquet",),
)
def factor_return_accel_nonoverlap_20_60(context: FactorContext):
    daily_adj = context.load("daily_adj.parquet")
    close = daily_adj["close"]
    mom_20 = close.groupby(level="Code").transform(lambda s: s.pct_change(20))
    old_mom_60 = close.groupby(level="Code").transform(
        lambda s: s.shift(20).pct_change(60)
    )
    accel = mom_20 - old_mom_60
    return cross_sectional_rank(accel)
