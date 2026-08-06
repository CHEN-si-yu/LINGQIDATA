"""
Daily technical momentum factors — Class 1 (daily.parquet only).

日频技术动量族,基于 99 个聚宽策略 + 券商研报中的经典技术信号开发。

关键原则:
- 所有连续价格序列一律使用 momentum_rebuilt._adjusted_close 后复权基座
  (cumprod(1+pct_chg/100)),除权日无跳变、point-in-time 稳定;
- 收益率一律使用 pct_chg(数据商复权口径),禁止 close.pct_change()/close.shift(1);
- 当日比值类指标(CCI/Williams/RSI 等)在除权日当天允许局部失真(用户确认可接受);
- 跨日极值/回撤/新高新低必须走复权基座;
- rolling corr 一律 unstack 宽表向量化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide, stack_date_code
from .momentum_rebuilt import _adjusted_close
from .market_relative import _ret_wide, _market_proxy, _rolling_beta


# ═══════════════════════════════════════════════════════════════════════════════
# KAMA 效率比率
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="kama_efficiency_20",
    description="KAMA效率比率因子：|20日净变动|/20日累计波动（趋势效率，高效率=单边趋势排前）。",
    category="price",
    thesis="KAMA自适应均线的核心是效率比率ER——净变动占累计波动的比例。ER高=价格单边运行"
           "(强趋势，KAMA快速跟随)，ER低=震荡(均值回归)。是向量化的KAMA代理，"
           "既捕捉趋势强度又规避递归实现的性能问题。",
    dependencies=("daily.parquet",),
)
def factor_kama_efficiency_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    net_move = adj.groupby(level="Code").diff(20).abs()
    gross_move = adj.groupby(level="Code").diff().abs()
    gross_20 = gross_move.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    er = safe_divide(net_move, gross_20)
    return cross_sectional_rank(er)


# ═══════════════════════════════════════════════════════════════════════════════
# TRIX(三重指数平滑)
# ═══════════════════════════════════════════════════════════════════════════════

def _trix_wide(adj_wide: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """TRIX: 三重指数平滑后的变化率(宽表向量化)。"""
    ema1 = adj_wide.ewm(span=n, adjust=False).mean()
    ema2 = ema1.ewm(span=n, adjust=False).mean()
    ema3 = ema2.ewm(span=n, adjust=False).mean()
    return ema3.pct_change(20, fill_method=None)


@register_factor(
    name="trix_12_20",
    description="TRIX趋势因子：三重指数平滑(12)的20日变化率截面排名（趋势加速排前）。",
    category="price",
    thesis="TRIX通过三重平滑滤除短期噪音，其20日变化率反映中长期趋势的加速/减速——"
           "策略42验证 TRIX 在低回撤组合中的有效性。正变化率=中长期趋势向上。",
    dependencies=("daily.parquet",),
)
def factor_trix_12_20(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _adjusted_close(daily).unstack("Code")
    trix = _trix_wide(wide)
    return cross_sectional_rank(stack_date_code(trix))


@register_factor(
    name="trix_signal_gap",
    description="TRIX信号乖离因子：TRIX与其20日信号线的乖离截面排名（强于自身趋势线排前）。",
    category="price",
    thesis="TRIX上穿/下穿其信号线(自身20日均值)是经典买卖信号——乖离扩大=趋势加速中，"
           "乖离收敛=趋势衰竭。捕捉策略42中 TRIX×MATRIX 的交叉逻辑的横截面形态。",
    dependencies=("daily.parquet",),
)
def factor_trix_signal_gap(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _adjusted_close(daily).unstack("Code")
    trix = _trix_wide(wide)
    signal = trix.rolling(20, min_periods=10).mean()
    gap = trix.sub(signal).div(signal.abs() + 1e-10)
    return cross_sectional_rank(stack_date_code(gap))


# ═══════════════════════════════════════════════════════════════════════════════
# BIAS 乖离率(60日)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="bias_60",
    description="60日乖离率因子：(adj-MA60)/MA60截面排名（偏离中期均线排前）。",
    category="price",
    thesis="乖离率衡量价格对中期均线的偏离程度——策略21用乖离及其均线金叉做择时。"
           "60日乖离极端正=短期超买(均值回归风险)，但中期强势延续性也强；"
           "与20日乖离(bias_20)互补，捕捉更慢的回归周期。",
    dependencies=("daily.parquet",),
)
def factor_bias_60(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    ma60 = adj.groupby(level="Code").transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    bias = safe_divide(adj - ma60, ma60)
    return cross_sectional_rank(bias)


# ═══════════════════════════════════════════════════════════════════════════════
# TD Sequential(策略78)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="td_setup_count",
    description="TD序列setup计数因子：连续close≤4日前close的天数截面排名（连续下跌setup排前）。",
    category="price",
    thesis="TD Sequential(策略78 GFTD)的买入setup=连续N日收盘价≤4日前收盘价，"
           "计数达到9通常意味着耗尽性下跌。连续setup天数反映超卖衰竭的程度——"
           "setup计数高=持续阴跌后反转概率积累，作为左侧反转信号。",
    dependencies=("daily.parquet",),
)
def factor_td_setup_count(context: FactorContext):
    daily = context.load("daily.parquet")
    # 跨日比较必须走复权基座,未复权 close 在除权日跳变会伪造连跌setup
    adj = _adjusted_close(daily)
    cond = (adj <= adj.groupby(level="Code").shift(4)).astype(int)
    # 每股连续段计数:段号 = (cond==0) 的累计,段内 = 组内 cumsum
    code = cond.index.get_level_values("Code")
    seg = (~cond.astype(bool)).groupby(level="Code").cumsum()
    count = cond.groupby([code, seg]).cumsum()
    return cross_sectional_rank(count)


# ═══════════════════════════════════════════════════════════════════════════════
# RS 相对全市场强度(策略23祖鲁法则)
# ═══════════════════════════════════════════════════════════════════════════════

def _relative_strength(daily: pd.DataFrame, window: int) -> pd.Series:
    """相对全市场强度: RS = (r_s − r_m) / (1 + r_m), N 日累计口径。"""
    ret = daily["pct_chg"] / 100.0
    wide = ret.unstack("Code")
    mkt = _market_proxy(wide)
    # pandas 3.0 无 Rolling.prod,用 cumprod + shift 求 N 日累计收益
    cum_s = (1.0 + wide).cumprod()
    stock_ret = cum_s.div(cum_s.shift(window)) - 1.0
    cum_m = (1.0 + mkt).cumprod()
    mkt_ret = cum_m.div(cum_m.shift(window)) - 1.0
    # 注意:DataFrame/Series 除法必须显式 axis=0(index 对齐),
    # 默认 axis="columns" 会按列名对齐导致全 NaN (2026-08-05 修复)
    rs = stock_ret.sub(mkt_ret, axis=0).div(1.0 + mkt_ret, axis=0)
    return stack_date_code(rs)


@register_factor(
    name="rs_60",
    description="60日相对强度因子：个股60日收益相对全市场等权收益的RS截面排名（跑赢市场排前）。",
    category="price",
    thesis="祖鲁法则(策略23)的核心确认信号：RS=(r_s−r_m)/(1+r_m)。60日RS衡量中期相对强弱——"
           "持续跑赢市场的股票处于资金聚集区，动量延续概率高。与绝对动量正交(剔市场贝塔)。",
    dependencies=("daily.parquet",),
)
def factor_rs_60(context: FactorContext):
    daily = context.load("daily.parquet")
    return cross_sectional_rank(_relative_strength(daily, 60))


@register_factor(
    name="rs_120",
    description="120日相对强度因子：中期相对全市场的RS截面排名。",
    category="price",
    thesis="120日(半年)RS捕捉中期风格——策略23要求 RS_year>RS_month>0 的渐进强势结构，"
           "120日RS介于30日与365日之间，衡量半年趋势质量。半年持续跑赢=机构持仓换手充分。",
    dependencies=("daily.parquet",),
)
def factor_rs_120(context: FactorContext):
    daily = context.load("daily.parquet")
    return cross_sectional_rank(_relative_strength(daily, 120))


@register_factor(
    name="rs_250",
    description="250日相对强度因子：年度相对全市场的RS截面排名（年度强势排前）。",
    category="price",
    thesis="250日(年)RS是祖鲁法则的最终确认——年度持续跑赢的股票基本面与资金面双重强劲，"
           "是机构「抱团」股票的量化刻画。年RS过滤了短期噪音，信号最稳定但响应最慢。",
    dependencies=("daily.parquet",),
)
def factor_rs_250(context: FactorContext):
    daily = context.load("daily.parquet")
    return cross_sectional_rank(_relative_strength(daily, 250))


# ═══════════════════════════════════════════════════════════════════════════════
# 日频 MACD(12,26,9)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="macd_daily_hist_5d",
    description="日频MACD柱5日变化因子：MACD(12,26,9)柱状图的5日变化截面排名（动能增强排前）。",
    category="price",
    thesis="策略58用MACD金叉死叉做选股——柱状图由负转正/持续放大=多头动能增强。"
           "5日柱变化捕捉动能二阶导。注意与分钟级macd_*区分：本因子为日频口径。",
    dependencies=("daily.parquet",),
)
def factor_macd_daily_hist_5d(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _adjusted_close(daily).unstack("Code")
    ema12 = wide.ewm(span=12, adjust=False).mean()
    ema26 = wide.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    chg = hist.diff(5)
    return cross_sectional_rank(stack_date_code(chg))


# ═══════════════════════════════════════════════════════════════════════════════
# 日频 RSI / KDJ / CCI / ROC / Aroon / DPO / Williams / STOCH
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="rsi_spread_6_14",
    description="日频RSI(6)−RSI(14)因子：短中期超买超卖差截面排名（短期强于中期排前）。",
    category="price",
    thesis="策略42用RSI(6)在55-80区间做超买判断——RSI6−RSI14的正差=短期动能强于中期"
           "(新一波拉升启动)，负差=短期动能衰竭。捕捉RSI的动量而非水平，"
           "与分钟级rsi_*区分(日频口径)。",
    dependencies=("daily.parquet",),
)
def factor_rsi_spread_6_14(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = (daily["pct_chg"] / 100.0).unstack("Code")

    def _rsi(w, n):
        delta = w.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(span=n, adjust=False).mean()
        avg_loss = loss.ewm(span=n, adjust=False).mean()
        rs = safe_divide(avg_gain, avg_loss + 1e-10)
        return 100.0 - 100.0 / (1.0 + rs)

    spread = _rsi(wide, 6) - _rsi(wide, 14)
    return cross_sectional_rank(stack_date_code(spread))


@register_factor(
    name="kdj_daily_j",
    description="日频KDJ的J值因子：J=3K−2D截面排名（超买动能排前，反向排名）。",
    category="price",
    thesis="策略29/40用KDJ金叉做信号——J值最灵敏(3K−2D)，>100超买、<0超卖。"
           "J值衡量短期随机动能，极端J=均值回归风险高，故负向排名。"
           "与分钟级kdj_*区分(日频口径)。",
    dependencies=("daily.parquet",),
)
def factor_kdj_daily_j(context: FactorContext):
    daily = context.load("daily.parquet")
    # 9日极值跨日窗口必须走复权基座,未复权 close 在除权日跳变会污染RSV
    wide = _adjusted_close(daily).unstack("Code")
    ll9 = wide.rolling(9, min_periods=5).min()
    hh9 = wide.rolling(9, min_periods=5).max()
    rsv = safe_divide(wide - ll9, hh9 - ll9 + 1e-10) * 100.0
    k = rsv.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    d = k.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return cross_sectional_rank(-stack_date_code(j))


@register_factor(
    name="cci_20",
    description="20日CCI因子：(TP−MA20)/(0.015×MD)截面排名（超买排后，负向排名）。",
    category="price",
    thesis="CCI(顺势指标)衡量典型价格对20日均价的标准化偏离——>100超买、<−100超卖。"
           "策略51的技术指标开关含CCI。极端正CCI=短期涨幅透支，负向排名做均值回归信号。",
    dependencies=("daily.parquet",),
)
def factor_cci_20(context: FactorContext):
    daily = context.load("daily.parquet")
    # 20日均值/平均偏离为跨日窗口,TP 需走复权口径(scale=adj/close 折算 high/low)
    adj = _adjusted_close(daily)
    scale = adj / daily["close"].replace(0, np.nan)
    tp = (daily["high"] * scale + daily["low"] * scale + adj) / 3.0
    tp_w = tp.unstack("Code")
    ma = tp_w.rolling(20, min_periods=10).mean()
    md = tp_w.sub(ma).abs().rolling(20, min_periods=10).mean()
    cci = safe_divide(tp_w - ma, 0.015 * md + 1e-10)
    return cross_sectional_rank(-stack_date_code(cci))


@register_factor(
    name="roc_12",
    description="12日ROC变动率因子：(adj−adj.shift(12))/adj.shift(12)截面排名（中期加速排前）。",
    category="price",
    thesis="ROC(Price Rate of Change)是经典动量指标——12日变化率衡量中期速度，"
           "正ROC=价格加速上行。基于复权基座计算，除权日无跳变失真。",
    dependencies=("daily.parquet",),
)
def factor_roc_12(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    roc = adj.groupby(level="Code").transform(
        lambda s: s.pct_change(12, fill_method=None)
    )
    return cross_sectional_rank(roc)


@register_factor(
    name="aroon_up_25",
    description="Aroon上行因子：25日窗口内距最近新高的天数位置截面排名（强势突破排前）。",
    category="price",
    thesis="Aroon(阿隆)衡量距离上次创新高的时间——越接近新高=趋势越强。"
           "基于复权基座判断新高(除权日不产生假突破)，25日窗口为经典参数。"
           "新高持续出现=多头主导。",
    dependencies=("daily.parquet",),
)
def factor_aroon_up_25(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _adjusted_close(daily).unstack("Code")
    is_high = wide.eq(wide.rolling(25, min_periods=1).max())
    pos = pd.DataFrame(
        np.tile(np.arange(len(wide))[:, None], (1, wide.shape[1])),
        index=wide.index,
        columns=wide.columns,
    )
    last_high = pos.where(is_high).ffill()
    days_since = pos - last_high
    aroon = safe_divide(25.0 - days_since, 25.0) * 100.0
    return cross_sectional_rank(stack_date_code(aroon))


@register_factor(
    name="aroon_down_25",
    description="Aroon下行因子：25日窗口内距最近新低的距离位置截面排名（破位风险排前，负向排名）。",
    category="price",
    thesis="Aroon下行衡量距离上次创新低的时间——越接近新低=趋势越弱。"
           "与aroon_up_25互补，反映空头主导程度。基于复权基座，除权日不产生假新低。",
    dependencies=("daily.parquet",),
)
def factor_aroon_down_25(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _adjusted_close(daily).unstack("Code")
    is_low = wide.eq(wide.rolling(25, min_periods=1).min())
    pos = pd.DataFrame(
        np.tile(np.arange(len(wide))[:, None], (1, wide.shape[1])),
        index=wide.index,
        columns=wide.columns,
    )
    last_low = pos.where(is_low).ffill()
    days_since = pos - last_low
    aroon = safe_divide(25.0 - days_since, 25.0) * 100.0
    return cross_sectional_rank(-stack_date_code(aroon))


@register_factor(
    name="dpo_20",
    description="20日DPO区间震荡因子：(adj−11日前价格)的20日SMA截面排名（中期趋势偏离排前）。",
    category="price",
    thesis="DPO(Detrended Price Oscillator)去趋势化后衡量价格对中期均衡的偏离——"
           "DPO>0=价格高于其中期趋势(强势)，DPO<0=低于趋势。shift(11)为20日周期的一半。",
    dependencies=("daily.parquet",),
)
def factor_dpo_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    detrended = adj - adj.groupby(level="Code").shift(11)
    dpo = detrended.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(dpo)


@register_factor(
    name="williams_r_14",
    description="14日威廉%R因子：(HH14−C)/(HH14−LL14)×(−100)截面排名（超卖排前）。",
    category="price",
    thesis="威廉%R是KDJ前身的反向随机指标——接近−100=超卖(反弹概率高)，接近0=超买。"
           "用复权基座计算14日高低区间，除权日不产生假极值。负向取值=超卖排前。",
    dependencies=("daily.parquet",),
)
def factor_williams_r_14(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    wide = adj.unstack("Code")
    hh = wide.rolling(14, min_periods=7).max()
    ll = wide.rolling(14, min_periods=7).min()
    wr = safe_divide(hh - wide, hh - ll + 1e-10) * (-100.0)
    return cross_sectional_rank(-stack_date_code(wr))


@register_factor(
    name="stoch_slow_k",
    description="慢速随机%K因子：RSV(9)的三日平滑截面排名（随机动能排前）。",
    category="price",
    thesis="慢速随机指标(策略29 KD)通过平滑RSV过滤噪音——%K上穿%D是经典金叉。"
           "慢速%K衡量短期超买超卖动能，取正向排名(动能强排前)。",
    dependencies=("daily.parquet",),
)
def factor_stoch_slow_k(context: FactorContext):
    daily = context.load("daily.parquet")
    # 9日极值跨日窗口必须走复权基座,未复权 close 在除权日跳变会污染RSV
    wide = _adjusted_close(daily).unstack("Code")
    ll9 = wide.rolling(9, min_periods=5).min()
    hh9 = wide.rolling(9, min_periods=5).max()
    rsv = safe_divide(wide - ll9, hh9 - ll9 + 1e-10) * 100.0
    slow_k = rsv.rolling(3, min_periods=2).mean()
    return cross_sectional_rank(stack_date_code(slow_k))


# ═══════════════════════════════════════════════════════════════════════════════
# Force Index / EOM / MFI(需 vol/amount)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="force_index_13",
    description="13日力指数因子：(close−pre_close)×vol的13日EWMA截面排名（量价合力排前）。",
    category="price",
    thesis="Force Index(力指数)=价格变动×成交量，衡量价格变动的资金合力——"
           "大涨配巨量=强多头力，小涨配巨量=分歧。用close−pre_close(复权口径涨跌额)"
           "替代close diff，除权日不失真。13日EWMA平滑捕捉中期合力。",
    dependencies=("daily.parquet",),
)
def factor_force_index_13(context: FactorContext):
    daily = context.load("daily.parquet")
    price_move = daily["close"] - daily["pre_close"]
    fi = price_move * daily["vol"]
    fi_w = fi.unstack("Code")
    fi_ema = fi_w.ewm(span=13, adjust=False).mean()
    return cross_sectional_rank(stack_date_code(fi_ema))


@register_factor(
    name="eom_14",
    description="14日EOM简易波动因子：价格中点位移×量/区间的14日均值截面排名（放量推动排前）。",
    category="price",
    thesis="Ease of Movement(简易波动指标)=价格中点位移×(成交量/价格区间)——"
           "量能推动价格移动的效率。EOM高=单位波动被大资金推动(有效上涨)；"
           "EOM低=无量空跌(虚假波动)。14日均值平滑。",
    dependencies=("daily.parquet",),
)
def factor_eom_14(context: FactorContext):
    daily = context.load("daily.parquet")
    # 跨日中点位移走复权口径,避免除权日伪位移(scale=adj/close 折算 high/low)
    adj = _adjusted_close(daily)
    scale = adj / daily["close"].replace(0, np.nan)
    adj_h = daily["high"] * scale
    adj_l = daily["low"] * scale
    mid = (adj_h + adj_l) / 2.0
    box_ratio = daily["vol"] / (adj_h - adj_l).replace(0, np.nan)
    distance = mid - mid.groupby(level="Code").shift(1)
    eom = distance * box_ratio
    eom_w = eom.unstack("Code")
    eom_avg = eom_w.rolling(14, min_periods=7).mean()
    return cross_sectional_rank(stack_date_code(eom_avg))


@register_factor(
    name="mfi_14",
    description="14日MFI资金流量指标：正负量流比截面排名（资金流入推动排前）。",
    category="price",
    thesis="Money Flow Index(资金流量指标)用典型价格×量区分买卖压力——"
           "MFI>80超买、<20超卖。与RSI互补(量加权)。TP=(H+L+C)/3,"
           "量流方向由TP相对昨日判定——TP 走复权口径,除权日无伪方向(2026-08-05 修复)。",
    dependencies=("daily.parquet",),
)
def factor_mfi_14(context: FactorContext):
    daily = context.load("daily.parquet")
    # TP 方向判定为跨日比较,走复权口径避免除权日误判方向
    # (scale=adj/close 折算 high/low,close 直接用复权基座)
    adj = _adjusted_close(daily)
    scale = adj / daily["close"].replace(0, np.nan)
    tp = (daily["high"] * scale + daily["low"] * scale + adj) / 3.0
    raw_flow = tp * daily["vol"]
    tp_chg = tp.groupby(level="Code").diff()
    pos_flow = raw_flow.where(tp_chg > 0, 0.0)
    neg_flow = raw_flow.where(tp_chg < 0, 0.0)
    pos_w = pos_flow.unstack("Code").rolling(14, min_periods=7).sum()
    neg_w = neg_flow.unstack("Code").rolling(14, min_periods=7).sum()
    ratio = safe_divide(pos_w, neg_w + 1e-10)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    return cross_sectional_rank(stack_date_code(mfi))


# ═══════════════════════════════════════════════════════════════════════════════
# OBV 能量潮
# ═══════════════════════════════════════════════════════════════════════════════

def _obv_series(daily: pd.DataFrame) -> pd.Series:
    """OBV = Σ sign(pct_chg) × vol,方向用复权收益符号(规避未复权 close diff 符号错乱)。"""
    sign_vol = np.sign(daily["pct_chg"]) * daily["vol"]
    return sign_vol.groupby(level="Code").cumsum()


@register_factor(
    name="obv_slope_20",
    description="20日OBV斜率因子：OBV的20日变化截面排名（量能净流入加速排前）。",
    category="price",
    thesis="OBV(能量潮)累计价涨量正、价跌量负的成交——20日变化=近一月量能净方向。"
           "OBV上升=放量上涨主导(资金主动买入)，下降=放量下跌主导。"
           "方向用pct_chg符号判定，除权日不失真。",
    dependencies=("daily.parquet",),
)
def factor_obv_slope_20(context: FactorContext):
    daily = context.load("daily.parquet")
    obv = _obv_series(daily)
    slope = obv.groupby(level="Code").diff(20)
    # 量纲归一:除以20日平均成交量的对数量级,避免大市值股天然占优
    avg_vol = daily["vol"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    norm = safe_divide(slope, avg_vol + 1e-10)
    return cross_sectional_rank(norm)


@register_factor(
    name="obv_divergence_20",
    description="OBV量价背离因子：价格动量排名−OBV动量排名的背离截面排名（价涨量缩背离排前）。",
    category="price",
    thesis="量价背离是经典顶部/底部信号——价格创新高而OBV未能跟进=上涨缺乏量能确认"
           "(顶部背离，风险信号)；价格新低而OBV企稳=抛压衰竭(底部背离)。"
           "本因子排名高=价强量弱背离，作为趋势谨慎信号。",
    dependencies=("daily.parquet",),
)
def factor_obv_divergence_20(context: FactorContext):
    daily = context.load("daily.parquet")
    adj = _adjusted_close(daily)
    price_mom = adj.groupby(level="Code").pct_change(20, fill_method=None)
    obv = _obv_series(daily)
    obv_mom = obv.groupby(level="Code").diff(20)
    avg_vol = daily["vol"].groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    obv_norm = safe_divide(obv_mom, avg_vol + 1e-10)
    # 截面标准化后求背离:价动量强 且 OBV 弱 = 正背离
    pr = price_mom.groupby(level="Date").rank(pct=True)
    ob = obv_norm.groupby(level="Date").rank(pct=True)
    return cross_sectional_rank(pr - ob)


# ═══════════════════════════════════════════════════════════════════════════════
# 残差动量 / 尾部相关 / 恐惧指数(市场相对)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="residual_momentum_20",
    description="残差动量因子：剔除市场暴露后的残差20日累计截面排名（特质动量排前）。",
    category="price",
    thesis="残差动量(研报:基于残差动量的相对收益动量策略)——个股对全市场回归后的残差"
           "剔除了系统性beta暴露，纯特质部分的中期动量更稳定、拥挤度更低。"
           "基于beta_60回归残差的20日累计，与绝对动量正交。",
    dependencies=("daily.parquet",),
)
def factor_residual_momentum_20(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    mkt = _market_proxy(wide)
    beta = _rolling_beta(wide, mkt, 60, 30)
    resid = wide - beta.multiply(mkt, axis=0)
    resid_mom = resid.rolling(20, min_periods=10).sum()
    return cross_sectional_rank(stack_date_code(resid_mom))


@register_factor(
    name="tail_corr_60",
    description="60日尾部相关性因子：与市场同向极端收益(|z|>1.5)的频率截面排名（尾部同步排前）。",
    category="risk",
    thesis="研报《沪深300样本股尾部相关性》：尾部同向波动反映系统性风险的传导——"
           "与市场同时出现极端收益的股票在危机中无分散价值。"
           "60日窗口内个股|z|>1.5与市场同号的频率越高=尾部相关性越强，负向排名。",
    dependencies=("daily.parquet",),
)
def factor_tail_corr_60(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    mkt = _market_proxy(wide)
    z = wide.sub(wide.rolling(60, min_periods=30).mean()).div(
        wide.rolling(60, min_periods=30).std() + 1e-10
    )
    mkt_z = mkt.sub(mkt.rolling(60, min_periods=30).mean()).div(
        mkt.rolling(60, min_periods=30).std() + 1e-10
    )
    same_sign = np.sign(z).eq(np.sign(mkt_z), axis=0)
    extreme = z.abs().gt(1.5)
    tail_freq = (same_sign & extreme).rolling(60, min_periods=30).mean()
    return cross_sectional_rank(-stack_date_code(tail_freq))


@register_factor(
    name="fear_index_20",
    description="恐惧指数因子：20日窗口内下行极端收益(低于均值1.5σ)的频率截面排名（恐慌频发排前）。",
    category="risk",
    thesis="研报《度量市场恐惧与贪婪的量化择时指标》的个股化：下行极端收益频率"
           "衡量个股的恐慌状态——恐慌频发=情绪释放充分(反转做多机会)但也是高风险标签。"
           "取正向排名：极端下行频率高=超卖反转候选。",
    dependencies=("daily.parquet",),
)
def factor_fear_index_20(context: FactorContext):
    daily = context.load("daily.parquet")
    ret = daily["pct_chg"] / 100.0
    mean20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    std20 = ret.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    extreme_down = ret.lt(mean20 - 1.5 * std20)
    freq = extreme_down.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(freq)
