"""
策略挖掘族 — Class 1 (daily / finance / cyq_perf / main_fund_flow / margin_detail)。

素材:99 个聚宽策略 + 券商研报(净换手率/龙虎榜替代/RSRS 量能加权)。

关键原则(继承代码库纪律):
- 连续价格一律走 _adjusted_close 复权基座,除权日无跳变;跨日极值(52周高/
  250日高低/吊灯止损)必须折算到复权空间;
- 同日比较(high/low/open/close vs 当日 pre_close)天然同尺度,直接比较;
- 收益率一律 pct_chg/100(数据商复权口径),禁止 close.pct_change()/shift(1);
- 单位(2026-08-08 实证):daily.vol=股、main_fund_flow vol=手(×100转股)、
  main_fund_flow 金额=万元(×1e4 转元)、finance.free_share/circ_mv=股/元;
- margin_detail 已被数据层 shift(1) 对齐(Date=T 上是原始 T-1 值),与 close
  配价必须用 close 的逐股 shift(1)(T-1 收盘价);margin 与其他源混算必须
  reindex(m.index) 防索引并集;
- 无未来函数:事件后表现类因子用滞后事件对齐。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, stack_date_code
from .momentum_rebuilt import _adjusted_close
from .technical_pattern import _compute_rsrs_beta

# 涨跌停近似口径(全库统一):pct_chg >= 9.8% 视为涨停/封板
_LIMIT_UP = 9.8


# ═══════════════════════════════════════════════════════════════════════════════
# 情绪/动量质量 (策略12均值回归 / 策略21 BIAS / 经典指标补位)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="psy_12",
    description="心理线因子：12日上涨天数占比截面排名（多头情绪浓度排前）。",
    category="price",
    thesis="心理线(PSY)是经典情绪指标——12日内上涨天数占比度量散户情绪的持续性:"
           "占比高=多头氛围浓(趋势延续的群众基础);占比低=空头氛围。"
           "与偏动量/价格类因子正交:只计数涨跌方向、不看幅度,"
           "对温和上涨(幅度小但天数多)的股票给予高排名。",
    dependencies=("daily.parquet",),
)
def factor_psy_12(context: FactorContext):
    daily = context.load("daily.parquet")
    up = (daily["pct_chg"] > 0).astype(float)
    psy = up.groupby(level="Code").transform(
        lambda s: s.rolling(12, min_periods=6).mean()
    )
    return cross_sectional_rank(psy)


@register_factor(
    name="up_down_count_ratio_20",
    description="20日涨跌天数比因子：(涨天数+1)/(跌天数+1)截面排名（涨多跌少排前）。",
    category="timeseries",
    thesis="涨跌天数比刻画动量质量:涨多跌少=稳健多头(上涨有持续性),"
           "涨少跌多=阴跌。与 20 日动量(幅度维)互补——本因子只看方向频率,"
           "可识别「小步慢涨」(动量温和但天数比极高)与「大涨大跌」的区别。",
    dependencies=("daily.parquet",),
)
def factor_up_down_count_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    up = (daily["pct_chg"] > 0).astype(float)
    dn = (daily["pct_chg"] < 0).astype(float)
    g = up.groupby(level="Code")
    up20 = g.transform(lambda s: s.rolling(20, min_periods=10).sum())
    dn20 = dn.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    ratio = safe_divide(up20 + 1.0, dn20 + 1.0)
    return cross_sectional_rank(ratio)


@register_factor(
    name="cmo_20",
    description="Chande动量振荡器因子：20日(涨额和−跌额和)/(涨额和+跌额和)截面排名（多空动能净额排前）。",
    category="timeseries",
    thesis="CMO(Chande Momentum Oscillator)度量动量与反向动量的对称净值——"
           "与 RSI 同族但窗口对称、无钝化:涨跌额旗鼓相当=0,单边=±100。"
           "与 gain_loss_asymmetry_60(60日)窗口互补,20日口径更贴近短周期轮动。"
           "用 pct_chg/100 做涨跌额,除权日无失真。",
    dependencies=("daily.parquet",),
)
def factor_cmo_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = daily["pct_chg"] / 100.0
    up = ret.clip(lower=0.0)
    dn = (-ret).clip(lower=0.0)
    g = up.groupby(level="Code")
    up_sum = g.transform(lambda s: s.rolling(20, min_periods=10).sum())
    dn_sum = dn.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    cmo = safe_divide(up_sum - dn_sum, up_sum + dn_sum)
    return cross_sectional_rank(cmo)


@register_factor(
    name="ppo_signal_12_26_9",
    description="PPO信号差因子：百分比价格振荡器(EMA12−EMA26)/EMA26减去其9日EMA截面排名（动能反转确认排前）。",
    category="timeseries",
    thesis="PPO 是 MACD 的价格水平归一化变形——消除高价股/低价股的绝对量纲,"
           "使横盘低价股与高价股可比。PPO 与其 9 日 EMA 之差=柱状图的百分比版,"
           "突破 0 轴=动能反转确认。与 macd_* 绝对口径互补。"
           "基于自建后复权基座,无除权失真。",
    dependencies=("daily.parquet",),
)
def factor_ppo_signal_12_26_9(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    g = adj.groupby(level="Code")
    ema12 = g.transform(
        lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean()
    )
    ema26 = g.transform(
        lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    ppo = safe_divide(ema12 - ema26, ema26) * 100.0
    signal = ppo.groupby(level="Code").transform(
        lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    return cross_sectional_rank(ppo - signal)


@register_factor(
    name="bias_signal_29_19",
    description="BIAS金叉信号因子：29日乖离率减去其19日均线截面排名（乖离加速扩张排前）。",
    category="price",
    thesis="策略21(BIAS_QL乖离率择时)用 N=29 日乖离与其 M=19 日均线的金叉/死叉"
           "判断乖离的趋势方向:乖离减去其均线>0=乖离仍在加速扩张(主升/超跌反弹"
           "进行中),<0=乖离回归(趋势衰竭)。与既有 bias_20/bias_60(乖离绝对值)"
           "互补:本因子是「乖离的加速度」维度。",
    dependencies=("daily.parquet",),
)
def factor_bias_signal_29_19(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    g = adj.groupby(level="Code")
    ma29 = g.transform(lambda s: s.rolling(29, min_periods=15).mean())
    bias = safe_divide(adj - ma29, ma29)
    bias_ma19 = bias.groupby(level="Code").transform(
        lambda s: s.rolling(19, min_periods=10).mean()
    )
    return cross_sectional_rank(bias - bias_ma19)


# ═══════════════════════════════════════════════════════════════════════════════
# KAMA 自适应均线 (策略01/82: 自适应均线过滤趋势/震荡)
# ═══════════════════════════════════════════════════════════════════════════════

def _kama(s: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average — 单遍 O(N) 递推(逐股)。

    KAMA_t = KAMA_{t−1} + SC_t × (close_t − KAMA_{t−1}),SC_t = (ER_t×(2/(fast+1)
    −2/(slow+1)) + 2/(slow+1))²。ER 用前 period 日的 |Δ| 累和(向量化 cumsum),
    无窗口嵌套,复杂度 O(N) 每股。初始化:首日 SC=1 → KAMA=首日价;
    period 之前 SC=0 → KAMA 保持首日价(数据早期,占比小)。
    """
    vals = s.to_numpy(dtype=np.float64)
    n = len(vals)
    out = np.full(n, np.nan)
    if n < period + 1:
        return pd.Series(out, index=s.index)
    diff = np.abs(np.diff(vals))                       # |x_t − x_{t−1}|, len n−1
    cum = np.concatenate([[0.0], np.cumsum(diff)])     # 前缀和,vol_sum_t = cum[t]−cum[t−period]
    chg = np.abs(vals[period:] - vals[:-period])       # |x_t − x_{t−period}|
    fast_c = 2.0 / (fast + 1.0)
    slow_c = 2.0 / (slow + 1.0)
    kama = np.empty(n)
    kama[0] = vals[0]
    sc = 1.0
    for t in range(1, n):
        if t >= period:
            vol_sum = cum[t] - cum[t - period]
            er = chg[t - period] / vol_sum if vol_sum > 1e-12 else 0.0
            sc = (er * (fast_c - slow_c) + slow_c) ** 2
        kama[t] = kama[t - 1] + sc * (vals[t] - kama[t - 1])
    return pd.Series(kama, index=s.index)


@register_factor(
    name="kama_position_20",
    description="KAMA位置因子：后复权收盘价相对20期自适应均线的偏离截面排名（价格站上KAMA排前）。",
    category="timeseries",
    thesis="KAMA(自适应均线)在震荡市收紧、趋势市放宽,能过滤假突破——价格站上"
           "KAMA=自适应确认的趋势方向向上(策略01/82的自适应均线过滤器)。"
           "与 kama_efficiency_20(效率比 ER)互补:本因子是价格相对 KAMA 线的"
           "位置,直接度量趋势的方位。基于后复权基座计算,无除权失真。",
    dependencies=("daily.parquet",),
)
def factor_kama_position_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    kama = adj.groupby(level="Code").transform(_kama)
    pos = safe_divide(adj - kama, kama)
    return cross_sectional_rank(pos)


# ═══════════════════════════════════════════════════════════════════════════════
# RSRS 变体 (策略77量能加权 / 斜率动量)
# ═══════════════════════════════════════════════════════════════════════════════

def _rsrs_beta_r2(daily: pd.DataFrame, window: int = 18) -> tuple[pd.Series, pd.Series]:
    """复权空间 RSRS β 与 R²(折算逻辑与 rsrs_beta_18 一致)。"""
    scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
    high = daily["high"] * scale
    low = daily["low"] * scale
    return _compute_rsrs_beta(high, low, window=window)


@register_factor(
    name="rsrs_volume_right_deviation",
    description="RSRS量能加权右偏离因子：β的400日Z分数×β×R²×近期量能占比截面排名（趋势信号+量能确认排前）。",
    category="price",
    thesis="策略77的独有变形:RSRS 右偏离(β 相对自身 400 日历史的标准化偏离)叠加"
           "近 9 日/18 日量能占比加权——量能集中在近期=趋势信号获得资金确认,"
           "放量突破的右偏离比缩量阴跌的右偏离更可信。与既有 rsrs_right_deviation"
           "的区别仅在量能加权项,捕捉「信号+量」的共振。",
    dependencies=("daily.parquet",),
)
def factor_rsrs_volume_right_deviation(context: FactorContext):
    daily = context.load("daily.parquet")
    beta, r2 = _rsrs_beta_r2(daily, window=18)
    g = beta.groupby(level="Code")
    beta_mean = g.transform(lambda s: s.rolling(400, min_periods=120).mean())
    beta_std = g.transform(lambda s: s.rolling(400, min_periods=120).std())
    z = safe_divide(beta - beta_mean, beta_std)
    vol = daily["vol"]
    vol9 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(9, min_periods=5).sum()
    )
    vol18 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(18, min_periods=10).sum()
    )
    vol_ratio = safe_divide(2.0 * vol9, vol18)
    raw = z * beta * r2 * vol_ratio
    return cross_sectional_rank(raw)


@register_factor(
    name="rsrs_beta_momentum_5",
    description="RSRS斜率动量因子：18日high~low回归斜率5日变化截面排名（斜率转升排前）。",
    category="price",
    thesis="RSRS β 的 5 日变化度量支撑阻力关系的边际方向——斜率刚转升=阻力相对"
           "支撑抬升的初期(阻力转支撑的早期信号),斜率转降=支撑走弱。与 β 绝对值"
           "(水平)互补:在 β 接近的股票中区分斜率加速与减速者。",
    dependencies=("daily.parquet",),
)
def factor_rsrs_beta_momentum_5(context: FactorContext):
    daily = context.load("daily.parquet")
    beta, _ = _rsrs_beta_r2(daily, window=18)
    mom = beta.groupby(level="Code").transform(lambda s: s.diff(5))
    return cross_sectional_rank(mom)


# ═══════════════════════════════════════════════════════════════════════════════
# 波段结构 / 波动历史分位 (黄金分割 / ATR分位 / 吊灯止损)
# ═══════════════════════════════════════════════════════════════════════════════

_FIB_LEVELS = np.array([0.236, 0.382, 0.5, 0.618, 0.786])


@register_factor(
    name="fib_retracement_proximity",
    description="黄金分割位邻近度因子：250日波段的斐波那契回撤位距离截面排名（靠近关键位排前）。",
    category="price",
    thesis="A股技术派的共识支撑/阻力位:回调到 0.382/0.5/0.618 分位时承接与抛压"
           "结构发生切换(策略 12 均值回归同源)。因子取现价在 250 日波段中的"
           "位置距最近斐波那契位的最小距离,越近=越处于关键博弈价位。"
           "波段高低点基于后复权基座,无除权假高低。",
    dependencies=("daily.parquet",),
)
def factor_fib_retracement_proximity(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    g = adj.groupby(level="Code")
    hh = g.transform(lambda s: s.rolling(250, min_periods=120).max())
    ll = g.transform(lambda s: s.rolling(250, min_periods=120).min())
    r = safe_divide(adj - ll, hh - ll)
    # pandas 3.0 + numpy 2:ufunc 作用于 Series 返回 ndarray,须包回 Series
    # (保留 (Date, Code) 索引供 cross_sectional_rank 使用)
    dists = [np.abs(r - level) for level in _FIB_LEVELS]
    prox = pd.Series(np.minimum.reduce(dists), index=r.index)
    return cross_sectional_rank(-prox)


@register_factor(
    name="atr_position_250",
    description="ATR历史位置因子：ATR20相对自身250日分布的标准化偏离截面排名（负向，波动分位高排后）。",
    category="risk",
    thesis="当前波动率在自身一年历史中的相对位置:低分位=波动压缩末端(变盘前夜),"
           "高分位=波动宣泄中(风险释放,追涨胜率低)。与 vol_cycle_position_120"
           "(成交量周期)区分:本因子是 ATR 价格波动维度,用 z-score 度量"
           "(向量化,滚动分位在 5000 股票规模不可行)。",
    dependencies=("daily.parquet",),
)
def factor_atr_position_250(context: FactorContext):
    daily = context.load("daily.parquet")
    # 同日比较:high/low 与当日 pre_close 天然同尺度,除权日无假 TR
    h = daily["high"]
    l = daily["low"]
    pc = daily["pre_close"]
    tr = pd.Series(
        np.maximum.reduce([h - l, (h - pc).abs(), (l - pc).abs()]),
        index=h.index,
    )
    atr20 = tr.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    g = atr20.groupby(level="Code")
    mean250 = g.transform(lambda s: s.rolling(250, min_periods=120).mean())
    std250 = g.transform(lambda s: s.rolling(250, min_periods=120).std())
    z = safe_divide(atr20 - mean250, std250)
    return cross_sectional_rank(-z)


@register_factor(
    name="chandelier_position",
    description="吊灯止损距离因子：现价距(20日高点−3×ATR)止损线的ATR单位数截面排名（趋势健康度排前）。",
    category="price",
    thesis="海龟系变体(策略01用4×ATR吊灯止损):止损线=20日最高价−3×ATR,价格距"
           "止损线的 ATR 单位数度量趋势的「余量」——距离远=趋势完整、回调尚未"
           "触及风控位;距离近=趋势濒临破坏。ATR 归一使高波与低波趋势股可比。"
           "全部在复权空间计算(high 折算、TR 用复权基座差分),无除权污染。",
    dependencies=("daily.parquet",),
)
def factor_chandelier_position(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    scale = adj / daily["close"].replace(0, np.nan)
    h_adj = daily["high"] * scale
    l_adj = daily["low"] * scale
    prev_adj = adj.groupby(level="Code").shift(1)
    tr = pd.Series(
        np.maximum.reduce([h_adj - l_adj, (h_adj - prev_adj).abs(), (l_adj - prev_adj).abs()]),
        index=h_adj.index,
    )
    atr20 = tr.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    high20 = h_adj.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    stop = high20 - 3.0 * atr20
    pos = safe_divide(adj - stop, atr20)
    return cross_sectional_rank(pos)


# ═══════════════════════════════════════════════════════════════════════════════
# 缺口/高开结构 (开盘价差族: 隔夜缺口与日内收益相关 / 高开低走惯犯)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="gap_intraday_corr_20",
    description="缺口-日内收益相关性因子：20日(高开幅度,日内收益)滚动相关截面排名（高开常延续排前）。",
    category="price",
    thesis="每只股票「高开惯性」vs「高开回落」的固有模式:corr 高=高开常在日内延续"
           "(强势股承接强);corr 低/负=高开必被砸(出货股)。滚动 corr 用宽表向量化"
           "(Date×Code 面板 rolling().corr()),C 级实现。缺口用 open/pre_close、"
           "日内用 close/open,均为同日比较,除权日无假缺口。",
    dependencies=("daily.parquet",),
)
def factor_gap_intraday_corr_20(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
    intra = safe_divide(daily["close"], daily["open"]) - 1.0
    wide_gap = gap.unstack("Code")
    wide_intra = intra.unstack("Code")
    corr = wide_gap.rolling(20, min_periods=10).corr(wide_intra)
    return cross_sectional_rank(stack_date_code(corr))


@register_factor(
    name="high_open_low_close_frac_20",
    description="高开低走频率因子：20日(高开≥2%且收阴)天数占比截面排名（负向，高开低走惯犯排后）。",
    category="event",
    thesis="高开 2% 以上却收阴是主力借高开出货的典型形态——「高开低走惯犯」=上方"
           "抛压沉重、承接乏力。与 gap_open_follow_ratio_20(高开次日跟随)不同:"
           "本因子聚焦当日高开→收阴的日内反转频率,是出货特征的直接证据。",
    dependencies=("daily.parquet",),
)
def factor_high_open_low_close_frac_20(context: FactorContext):
    daily = context.load("daily.parquet")
    gapup = safe_divide(daily["open"], daily["pre_close"]) - 1.0
    fade = (gapup >= 0.02) & (daily["close"] < daily["open"])
    freq = fade.astype(float).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    return cross_sectional_rank(-freq)


# ═══════════════════════════════════════════════════════════════════════════════
# 涨停梯队/封板结构 (策略10追三板: 连板高度 / 缩量涨停 / 连板放量结构)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="limit_board_streak_mean_60",
    description="平均连板高度因子：60日封板天数/连板启动次数截面排名（历史连板惯性排前）。",
    category="event",
    thesis="策略10追三板的核心是连板高度——60日内封板天数除以连板启动次数=该股被"
           "资金连续拉板的惯性禀赋:反复 2-3 板被砸的票与单次 5 板以上的历史妖股"
           "在此分离。与 consecutive_limit_up(当前连板数)互补:本因子是历史均值,"
           "刻画「拉板基因」而非当下状态。",
    dependencies=("daily.parquet",),
)
def factor_limit_board_streak_mean_60(context: FactorContext):
    daily = context.load("daily.parquet")
    sealed = (daily["pct_chg"] >= _LIMIT_UP).astype(float)
    # 连板启动日=当日封板且前一日未封板;shift 后首日 NaN 视为未启动(fillna False)。
    # shift 会把 numpy bool 列提升为 object dtype(无法表示 NaN),必须再转回 bool,
    # 否则对 object 列做位取反会触发 numpy 2.x 的 DeprecationWarning。
    prev_sealed = sealed.astype(bool).groupby(level="Code").shift(1).fillna(False).astype(bool)
    start = (sealed.astype(bool) & ~prev_sealed).astype(float)
    # 涨停事件稀疏:两窗口均 min_periods=1(60 日内 ≥1 个封板日/启动日即估,
    # panic_selling_ratio_60 先例);无任何封板日的股票用 0 中性填充
    days60 = sealed.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=1).sum()
    )
    starts60 = start.astype(float).groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=1).sum()
    )
    avg = safe_divide(days60, starts60).fillna(0.0)
    return cross_sectional_rank(avg)


@register_factor(
    name="limit_up_vol_shrink_60",
    description="缩量涨停率因子：60日涨停日均量/非涨停日均量截面排名（负向，涨停放量分歧排后）。",
    category="event",
    thesis="缩量涨停=惜售=筹码锁定良好(一字板特征),放量涨停=分歧大(换手充分、"
           "随时可能被砸)。涨停日量均相对非涨停日量均的比值连续度量封板形态,"
           "覆盖一字板以外的全部封板方式,与 one_word_limit_up_freq_20(仅数一字板)"
           "互补。",
    dependencies=("daily.parquet",),
)
def factor_limit_up_vol_shrink_60(context: FactorContext):
    daily = context.load("daily.parquet")
    sealed = daily["pct_chg"] >= _LIMIT_UP
    vol = daily["vol"]
    # 涨停日稀疏:涨停侧 min_periods=1(60 日内 ≥1 个涨停日即估均值,panic_selling
    # 先例),非涨停侧稠密用常规 min_periods=30;无涨停日的股票比值为 NaN,
    # 用 1.0 中性填充(与一字板频率类 0 填充同思路,保证全市场覆盖)
    sealed_vol = vol.where(sealed).groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=1).mean()
    )
    plain_vol = vol.where(~sealed).groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    ratio = safe_divide(sealed_vol, plain_vol).fillna(1.0)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="limit_streak_volume_ratio",
    description="连板放量结构因子：当前连板数(≤5)×当日量/近3日最大量截面排名（连板且量创新高排前）。",
    category="event",
    thesis="策略10的量能条件:追三板要求当日量是近 3 日最大——连板且当日量创新高"
           "=接力资金进场(换手板);连板但量萎缩=一字板(无法上车,断板风险由"
           "封单决定)。连板数与放量比的乘积在连板梯队内部再分层,与既有"
           "consecutive_limit_up 只度量高度区分。",
    dependencies=("daily.parquet",),
)
def factor_limit_streak_volume_ratio(context: FactorContext):
    daily = context.load("daily.parquet")
    is_lu = daily["pct_chg"].ge(_LIMIT_UP).astype(int)
    code = is_lu.index.get_level_values("Code")
    seg = (~is_lu.astype(bool)).groupby(level="Code").cumsum()
    streak = is_lu.groupby([code, seg]).cumsum().clip(upper=5.0)
    vol = daily["vol"]
    max3 = vol.groupby(level="Code").transform(
        lambda s: s.rolling(3, min_periods=1).max().shift(1)
    )
    ratio = safe_divide(vol, max3)
    raw = streak * ratio
    return cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# 次新热度 / 净换手率 (策略47/82次新小盘 / 研报《净换手率》)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="listing_age_heat",
    description="次新热度衰减因子：换手率×exp(−上市天数/500)截面排名（次新且换手高排前）。",
    category="price",
    thesis="策略47/82(次新小盘)的核心是次新股的高弹性与高热度——热度随上市时间"
           "指数衰减:上市 250 天内换手仍高的次新=资金未撤离。换手率项保证日频"
           "变化与老股间的正常分层,年龄项放大次新效应。上市年龄取该股在"
           "daily 面板的首条记录(数据始于 2019,老股年龄统一截断,截面可比)。",
    dependencies=("daily.parquet", "finance.parquet"),
)
def factor_listing_age_heat(context: FactorContext):
    daily = context.load("daily.parquet")
    fin = context.load("finance.parquet")
    dates_s = pd.Series(
        pd.to_datetime(daily.index.get_level_values("Date"), format="%Y%m%d"),
        index=daily.index,
    )
    first_dt = dates_s.groupby(level="Code").transform("min")
    age_days = (dates_s - first_dt).dt.days
    turnover = fin["turnover_rate"]
    heat = turnover * np.exp(-age_days / 500.0)
    return cross_sectional_rank(heat)


@register_factor(
    name="net_turnover_rate_20",
    description="净换手率因子：20日(主动买量−主动卖量)/自由流通股本截面排名（净买入比例高排前）。",
    category="fund_flow",
    thesis="研报《净换手率》+策略95:净换手率=(主动买量−主动卖量)/流通股本,"
           "是短线动量维度的高效因子。资金流 buy/sell 即主动买卖代理,"
           "自由流通股本归一给出「多少比例的流通盘被净买入」的直观含义。"
           "与 mf_net_vol_ratio_5d(总成交量归一)区分:本因子用股本归一。"
           "单位:main_fund_flow vol=手(×100转股),free_share=股。",
    dependencies=("main_fund_flow.parquet", "finance.parquet"),
)
def factor_net_turnover_rate_20(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    fin = context.load("finance.parquet")
    buy = (
        mf["buy_sm_vol"] + mf["buy_md_vol"] + mf["buy_lg_vol"] + mf["buy_elg_vol"]
    )
    sell = (
        mf["sell_sm_vol"] + mf["sell_md_vol"] + mf["sell_lg_vol"] + mf["sell_elg_vol"]
    )
    net_shares = (buy - sell) * 100.0  # 手 → 股
    net20 = net_shares.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    free_share = fin["free_share"].reindex(net20.index)
    pct = safe_divide(net20, free_share) * 100.0
    return cross_sectional_rank(pct)


# ═══════════════════════════════════════════════════════════════════════════════
# 龙虎榜替代 (大波动+高换手+大单参与 三条件近似上榜事件)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="lhb_proxy_score_60",
    description="龙虎榜替代活跃度因子：60日(大波动×高换手×大单高参与)事件次数截面排名（博弈票活跃度排前）。",
    category="event",
    thesis="无龙虎榜数据白名单下,用「|涨跌|≥7% × 换手率≥5% × 成交额≥1亿 × "
           "大单净参与≥10%」四条件近似上榜事件——上榜惯犯=游资博弈票,其短期"
           "波动与题材弹性显著高于均值。与 extreme_move_count(仅价格幅度)区分:"
           "本因子叠加换手与大单参与度,筛出真正的资金博弈而非单纯暴涨暴跌。"
           "单位:main_fund_flow 金额=万元(×1e4 转元),amount=元。",
    dependencies=("main_fund_flow.parquet", "finance.parquet", "daily.parquet"),
)
def factor_lhb_proxy_score_60(context: FactorContext):
    daily = context.load("daily.parquet")
    mf = context.load("main_fund_flow.parquet")
    fin = context.load("finance.parquet")
    big_buy = mf["buy_lg_amount"] + mf["buy_elg_amount"]
    big_sell = mf["sell_lg_amount"] + mf["sell_elg_amount"]
    big_net_yuan = (big_buy - big_sell) * 1e4  # 万元 → 元
    amount = daily["amount"].reindex(big_net_yuan.index)
    big_share = safe_divide(big_net_yuan.abs(), amount)
    hit = (
        (daily["pct_chg"].abs().reindex(big_net_yuan.index) >= 7.0)
        & (fin["turnover_rate"].reindex(big_net_yuan.index) >= 5.0)
        & (amount >= 1e8)
        & (big_share >= 0.10)
    )
    score = hit.astype(float).groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=10).sum()
    )
    return cross_sectional_rank(score)


# ═══════════════════════════════════════════════════════════════════════════════
# 融资盘成本结构 (margin 已 shift(1): 配价必须用 close 逐股 shift(1))
# ═══════════════════════════════════════════════════════════════════════════════

def _margin_weighted_cost(margin: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """近20日融资买入额按成交价加权的平均成本(逐股)。

    margin 面板 Date=T 存原始 T-1 值 → 配对价格用 close 的逐股 shift(1)
    (T-1 收盘价),否则 1 日错配。返回与 margin 索引对齐的 Series。
    """
    close_lag = daily["close"].groupby(level="Code").shift(1)
    close_m = close_lag.reindex(margin.index)
    num = (margin["rzmre"] * close_m).groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    den = margin["rzmre"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )
    return safe_divide(num, den)


@register_factor(
    name="margin_buyer_avg_cost_premium",
    description="融资盘成本溢价因子：现价相对近20日融资买入加权平均成本截面排名（融资盘浮盈排前）。",
    category="fund_flow",
    thesis="现价低于融资盘平均建仓成本=融资盘被套(强平/止损压力随时释放);高于"
           "=融资盘浮盈(惜售)。融资盘是 A 股最活跃的杠杆资金,其盈亏状态直接"
           "影响后续抛压。margin 数据已 shift(1),配对价格用 close 逐股 shift(1)"
           "(T-1 收盘),严格对齐无未来函数。与既有 margin_* 量额类因子区分:"
           "本因子是价格×金额的加权成本维度。",
    dependencies=("margin_detail.parquet", "daily.parquet"),
)
def factor_margin_buyer_avg_cost_premium(context: FactorContext):
    daily = context.load("daily.parquet")
    margin = context.load("margin_detail.parquet")
    cost = _margin_weighted_cost(margin, daily)
    close = daily["close"].reindex(cost.index)
    raw = safe_divide(close, cost) - 1.0
    return cross_sectional_rank(raw)


@register_factor(
    name="margin_chip_cost_gap",
    description="融资盘-筹码成本差因子：融资盘平均成本/市场筹码平均成本−1截面排名（负向，杠杆盘高位接盘排后）。",
    category="fund_flow",
    thesis="融资盘平均建仓成本相对 cyq_perf 筹码平均成本(weight_avg)的位置:"
           "融资成本显著高于市场平均=杠杆盘在高位接盘(后续易引发止损踩踏);"
           "低于市场平均=杠杆盘低位建仓(安全边际高)。同一只股票筹码结构不变时,"
           "本因子给出融资盘的增量位置信息,与 margin_buyer_avg_cost_premium"
           "(现价视角)互补为成本结构双视角。",
    dependencies=("margin_detail.parquet", "daily.parquet", "cyq_perf.parquet"),
)
def factor_margin_chip_cost_gap(context: FactorContext):
    daily = context.load("daily.parquet")
    margin = context.load("margin_detail.parquet")
    cyq = context.load("cyq_perf.parquet")
    cost = _margin_weighted_cost(margin, daily)
    weight_avg = cyq["weight_avg"].reindex(cost.index)
    gap = safe_divide(cost, weight_avg) - 1.0
    return cross_sectional_rank(-gap)


# ═══════════════════════════════════════════════════════════════════════════════
# 盈利质量代理 (PS/PE = 净利率,市值恒等式,无财报白名单下的构造)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="margin_proxy_ttm",
    description="净利率代理因子：ps_ttm/pe_ttm(市值恒等式=净利/营收)截面排名（高净利率排前）。",
    category="valuation",
    thesis="ps_ttm/pe_ttm = (市值/营收)/(市值/净利) = 净利率——无财报白名单下"
           "用市值恒等式构造盈利质量维度:银行白酒(净利率 30%+)与低毛利制造业"
           "(<5%)在此分离,是 pe_ttm/pb 之外的独立质量维度。负 PE 置 NaN"
           "(与既有 PE 类因子同口径,NaN≈17% 合格)。",
    dependencies=("finance.parquet",),
)
def factor_margin_proxy_ttm(context: FactorContext):
    fin = context.load("finance.parquet")
    pe = fin["pe_ttm"]
    ps = fin["ps_ttm"]
    raw = safe_divide(ps, pe.where(pe > 0))
    return cross_sectional_rank(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# 资金流深加工 (大中小单分歧 / 大单净峰度 / 大单加速度 / 超大单吸筹)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="mf_tier_net_spread_20",
    description="大中小单分歧度因子：20日四档(小/中/大/超大)净占比极差截面排名（负向，多空分歧排后）。",
    category="fund_flow",
    thesis="四个层级 20 日净占比(单档净流入/成交额)的极差度量资金方向的分歧度:"
           "极差大=机构与散户方向严重对立(变盘前兆);极差小=各层级方向一致"
           "(趋势健康)。与 mf_big_small_divergence(仅大小两档)区分:四档极差"
           "覆盖中单(游资/大户)的独立信号。单位:main_fund_flow 金额=万元"
           "(×1e4 转元),daily.amount=元。",
    dependencies=("main_fund_flow.parquet", "daily.parquet"),
)
def factor_mf_tier_net_spread_20(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    daily = context.load("daily.parquet")
    amount20 = daily["amount"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    ).reindex(mf.index)
    shares = []
    for tier in ("sm", "md", "lg", "elg"):
        net = (mf[f"buy_{tier}_amount"] - mf[f"sell_{tier}_amount"]) * 1e4
        net20 = net.groupby(level="Code").transform(
            lambda s: s.rolling(20, min_periods=10).sum()
        )
        shares.append(safe_divide(net20, amount20))
    spread = pd.Series(
        np.maximum.reduce(shares) - np.minimum.reduce(shares),
        index=shares[0].index,
    )
    return cross_sectional_rank(-spread)


@register_factor(
    name="mf_big_order_net_kurt_20",
    description="大单净流入峰度因子：20日(大单+超大单净流入)峰度截面排名（脉冲式建仓排前）。",
    category="fund_flow",
    thesis="大单净流入的分布形态区分资金行为:峰度高=集中脉冲式建仓(1-2 天巨额"
           "大单砸下,事件驱动);峰度低=每天均匀流入(长期吸筹)。与"
           "mf_flow_volatility_20d/mf_flow_stability_20d(二阶矩)互补,"
           "峰度直达三阶形态。单位一致(万元)内部比较,无需换算。",
    dependencies=("main_fund_flow.parquet",),
)
def factor_mf_big_order_net_kurt_20(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    x = (
        mf["buy_lg_amount"] + mf["buy_elg_amount"]
        - mf["sell_lg_amount"] - mf["sell_elg_amount"]
    )
    kurt = x.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).kurt()
    )
    return cross_sectional_rank(kurt)


@register_factor(
    name="big_order_net_accel_10",
    description="大单净额加速度因子：近5日大单净额−前5日大单净额(占成交额比)截面排名（资金加速流入排前）。",
    category="fund_flow",
    thesis="大单净流入的边际变化比水平更有信息:加速流入=机构刚进场(最佳跟随点),"
           "减速=流入尾声(追高风险)。与 mf_flow_acceleration_5d(全层级 net_mf)"
           "区分:本因子专看 lg+elg 档,剔除散户/中单噪声。以 10 日成交额归一,"
           "跨市值可比。单位:万元×1e4 转元后与 amount 同尺度。",
    dependencies=("main_fund_flow.parquet", "daily.parquet"),
)
def factor_big_order_net_accel_10(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    daily = context.load("daily.parquet")
    x = (
        mf["buy_lg_amount"] + mf["buy_elg_amount"]
        - mf["sell_lg_amount"] - mf["sell_elg_amount"]
    )
    net5 = x.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    net10 = x.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).sum()
    )
    prev5 = net10 - net5
    amount10 = daily["amount"].groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).sum()
    ).reindex(x.index)
    accel = safe_divide((net5 - prev5) * 1e4, amount10)
    return cross_sectional_rank(accel)


@register_factor(
    name="elg_net_60d_to_mv",
    description="超大单60日净买入占流通市值比因子：Σ(超大单买−卖,60日)×1e4/流通市值截面排名（长线吸筹强度排前）。",
    category="fund_flow",
    thesis="无北向持股数据白名单下,超大单(elg)是外资/产业资本的最佳代理——60 日"
           "累计净买入占流通市值比衡量长线吸筹强度(策略73跟随北向资金的日频"
           "替代)。与 mf_cumulative_flow_20d(20日全口径)区分:60 日 elg 专属"
           "口径更长更纯,过滤短线噪音。单位:金额万元×1e4 转元,circ_mv=元。",
    dependencies=("main_fund_flow.parquet", "finance.parquet"),
)
def factor_elg_net_60d_to_mv(context: FactorContext):
    mf = context.load("main_fund_flow.parquet")
    fin = context.load("finance.parquet")
    net = (mf["buy_elg_amount"] - mf["sell_elg_amount"]) * 1e4
    net60 = net.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=20).sum()
    )
    circ_mv = fin["circ_mv"].reindex(net60.index)
    raw = safe_divide(net60, circ_mv)
    return cross_sectional_rank(raw)
