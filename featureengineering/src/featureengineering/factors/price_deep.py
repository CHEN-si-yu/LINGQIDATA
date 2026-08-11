from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, safe_divide
from .momentum_rebuilt import _adjusted_close


@register_factor(
    name="gap_ratio",
    description="跳空比率因子，(open-pre_close)/pre_close截面排名（正=高开幅度大排前）。",
    category="price",
    thesis="跳空幅度反映隔夜信息冲击强度——大幅跳空高开意味着利好集中释放，但A股存在跳空回补效应，极端跳空方向可能面临反转。",
    dependencies=("daily.parquet",),
)
def factor_gap_ratio(context: FactorContext):
    """Compute (open - pre_close) / pre_close = gap up/down magnitude."""
    daily = context.load("daily.parquet")
    gap = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
    return cross_sectional_rank(gap)


@register_factor(
    name="upper_shadow_ratio",
    description="上影线比率因子，-(high-max(open,close))/(high-low+1e-9)截面排名（长上影=抛压重排后）。",
    category="price",
    thesis="长上影线是典型的日内冲高回落形态，反映空头在高位阻击，大量卖盘在上涨过程中涌现，是短期上涨阻力信号。",
    dependencies=("daily.parquet",),
)
def factor_upper_shadow_ratio(context: FactorContext):
    """Compute upper shadow ratio: (high - max(open, close)) / (high - low). Rank negative."""
    daily = context.load("daily.parquet")
    body_range = daily["high"] - daily["low"]
    upper_shadow = daily["high"] - np.maximum(daily["open"], daily["close"])
    ratio = upper_shadow / (body_range + 1e-9)
    return cross_sectional_rank(-ratio)


@register_factor(
    name="lower_shadow_ratio",
    description="下影线比率因子，(min(open,close)-low)/(high-low+1e-9)截面排名（长下影=承接力强排前）。",
    category="price",
    thesis="长下影线反映日内探底回升过程中买盘积极承接，是重要支撑信号，低位下影线往往预示短期底部确立。",
    dependencies=("daily.parquet",),
)
def factor_lower_shadow_ratio(context: FactorContext):
    """Compute lower shadow ratio: (min(open, close) - low) / (high - low)."""
    daily = context.load("daily.parquet")
    body_range = daily["high"] - daily["low"]
    lower_shadow = np.minimum(daily["open"], daily["close"]) - daily["low"]
    ratio = lower_shadow / (body_range + 1e-9)
    return cross_sectional_rank(ratio)


@register_factor(
    name="shadow_asymmetry",
    description="影线不对称性因子，-(upper_shadow_ratio-lower_shadow_ratio)*(high-low)/close截面排名（上影主导=看空排后）。",
    category="price",
    thesis="加权影线不对称性将上下影线的相对长度与蜡烛实体大小结合——大实体+上影主导是最强烈的日内见顶形态。因子值越负（上影明显长于下影且实体大）越看空。",
    dependencies=("daily.parquet",),
)
def factor_shadow_asymmetry(context: FactorContext):
    """Compute weighted shadow asymmetry: (upper_shadow_ratio - lower_shadow_ratio) * (high-low) / close.
    Rank negative (more upper shadow = bearish).
    """
    daily = context.load("daily.parquet")
    body_range = daily["high"] - daily["low"]
    close = daily["close"]

    upper_shadow = daily["high"] - np.maximum(daily["open"], daily["close"])
    lower_shadow = np.minimum(daily["open"], daily["close"]) - daily["low"]

    upper_shadow_ratio = upper_shadow / (body_range + 1e-9)
    lower_shadow_ratio = lower_shadow / (body_range + 1e-9)

    asymmetry = (upper_shadow_ratio - lower_shadow_ratio) * body_range / close.replace(0, np.nan)
    return cross_sectional_rank(-asymmetry)


@register_factor(
    name="oi_divergence_intensity",
    description="隔夜日内背离强度因子，(close-open)-(open-pre_close)截面排名（正=日内强化跳空方向排前）。",
    category="price",
    thesis="隔夜跳空与日内走势的背离度是判断跳空质量的关键——跳空后日内继续同向运行（正背离）说明跳空方向得到市场认可，反向运行（负背离）说明跳空是情绪过度反应。",
    dependencies=("daily.parquet",),
)
def factor_oi_divergence_intensity(context: FactorContext):
    """Signed intraday return in the direction of the opening gap."""
    daily = context.load("daily.parquet")
    overnight_gap = safe_divide(
        daily["open"] - daily["pre_close"], daily["pre_close"]
    )
    intraday_ret = safe_divide(daily["close"] - daily["open"], daily["open"])
    reinforcement = np.sign(overnight_gap) * intraday_ret
    return cross_sectional_rank(reinforcement)


# ── Overnight / Intraday gap factors ─────────────────────────────────────

@register_factor(
    name="overnight_gap_momentum",
    description="隔夜跳空因子，(open-pre_close)/pre_close截面排名（高开排前=利好消化未完）。",
    category="price",
    thesis="A股隔夜跳空后短期存在动量效应——正跳空(高开)反映隔夜利好信息未完全消化、开盘后仍有追涨动力；负跳空(低开)反映利空。与intraday_ret互补：一个捕捉隔夜信息冲击，一个捕捉日内价格发现。",
    dependencies=("daily.parquet",),
)
def factor_overnight_gap_momentum(context: FactorContext):
    daily = context.load("daily.parquet")
    gap = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
    return cross_sectional_rank(gap)


@register_factor(
    name="intraday_ret_momentum",
    description="日内收益因子，(close-open)/open截面排名。",
    category="price",
    thesis="日内收益与隔夜收益相关性低，提供独立的alpha维度。日内强势(收盘远离开盘价)反映日内买方主导和价格发现效率高。",
    dependencies=("daily.parquet",),
)
def factor_intraday_ret_momentum(context: FactorContext):
    daily = context.load("daily.parquet")
    intraday = (daily["close"] - daily["open"]) / daily["open"].replace(0, np.nan)
    return cross_sectional_rank(intraday)


@register_factor(
    name="overnight_intraday_divergence_daily",
    description="隔夜-日内背离因子，overnight_gap - intraday_ret截面排名。",
    category="price",
    thesis="隔夜和日内方向相反时反映信息不对称——高开低走=隔夜乐观情绪被日内交易否定(短期反转信号)，低开高走=隔夜恐慌被纠正(短期反弹信号)。",
    dependencies=("daily.parquet",),
)
def factor_overnight_intraday_divergence_daily(context: FactorContext):
    daily = context.load("daily.parquet")
    overnight = (daily["open"] - daily["pre_close"]) / daily["pre_close"].replace(0, np.nan)
    intraday = (daily["close"] - daily["open"]) / daily["open"].replace(0, np.nan)
    divergence = overnight - intraday
    return cross_sectional_rank(-divergence)


# ── Price range extremes ─────────────────────────────────────────────────

@register_factor(
    name="price_position_20d",
    description="20日价格位置因子，(close-20日最低)/(20日最高-20日最低)截面排名。",
    category="price",
    thesis="价格在近期高低点区间中的位置反映短期趋势强度——接近区间上沿=强势趋势中(正动量)，接近区间下沿=弱势中(负动量)。与单纯看涨跌幅不同，位置因子在震荡市中也能提供有效区分度。",
    dependencies=("daily.parquet",),
)
def factor_price_position_20d(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    # 用自建后复权基座(pct_chg 累乘)替代未复权 close,避免除权日 20 日区间高低点被污染
    adj = _adjusted_close(daily_panel)
    high_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max())
    low_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).min())
    position = (adj - low_20) / (high_20 - low_20).replace(0, np.nan)
    return cross_sectional_rank(position)


@register_factor(
    name="high_low_amplitude_20",
    description="振幅因子，(20日最高-20日最低)/20日均价截面排名。",
    category="price",
    thesis="振幅是波动性的另一个维度——与收益率标准差互补，振幅衡量的是日内极端价格范围而非收益率离散度。高振幅=投机性强、多空分歧大。",
    dependencies=("daily.parquet",),
)
def factor_high_low_amplitude_20(context: FactorContext):
    daily_panel = context.load("daily.parquet")
    close = daily_panel["close"]
    # 用每日复权系数 (adj/close) 把未复权 high/low 折算到后复权空间,
    # 避免除权日跨日 rolling 极值被污染(与 price_position_20d 同口径)
    adj = _adjusted_close(daily_panel)
    scale = adj / close.replace(0, np.nan)
    adj_high = daily_panel["high"] * scale
    adj_low = daily_panel["low"] * scale
    high_20 = adj_high.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max())
    low_20 = adj_low.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).min())
    mean_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    hl_vol = (high_20 - low_20) / mean_20.replace(0, np.nan)
    return cross_sectional_rank(hl_vol)


# ── Gap fill analysis ────────────────────────────────────────────────────

@register_factor(
    name="gap_fill_tendency_10d",
    description="缺口回补倾向因子，近10日缺口天数/(缺口天数+0.01)截面排名。",
    category="price",
    thesis="A股有'缺口必补'的民间说法——统计上，向上跳空缺口在短期内被回补的概率较高。该因子度量跳空后回补的频率：高频回补的股票可能在缺口后趋势反而较弱。",
    dependencies=("daily.parquet",),
)
def factor_gap_fill_tendency_10d(context: FactorContext):
    daily = context.load("daily.parquet")
    close = daily["close"]
    pre_close = daily["pre_close"]
    # A gap exists if open != pre_close significantly
    gap = (daily["open"] - pre_close) / pre_close.replace(0, np.nan)
    gap_abs = gap.abs()
    is_gap = (gap_abs > 0.01).astype(float)  # >1% gap
    # Gap fills if close crosses back toward pre_close
    gap_filled = ((gap > 0.01) & (close < pre_close)) | ((gap < -0.01) & (close > pre_close))
    gap_filled = gap_filled.astype(float)
    gap_count_10 = is_gap.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).sum()
    )
    fill_count_10 = gap_filled.groupby(level="Code").transform(
        lambda s: s.rolling(10, min_periods=5).sum()
    )
    fill_rate = fill_count_10 / (gap_count_10 + 0.01)
    return cross_sectional_rank(fill_rate)


# ── Consecutive direction ────────────────────────────────────────────────

@register_factor(
    name="gap_up_ratio_20d",
    description="20日高开概率因子，高开(open>pre_close)天数/20截面排名。",
    category="price",
    thesis="高开频率反映市场对股票的持续正面预期——频繁高开的股票往往是机构持续买入或利好信息持续释放的标的。",
    dependencies=("daily.parquet",),
)
def factor_gap_up_ratio_20d(context: FactorContext):
    daily = context.load("daily.parquet")
    gap_up = (daily["open"] > daily["pre_close"]).astype(float)
    ratio = gap_up.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    return cross_sectional_rank(ratio)

# ═══════════════════════════════════════════════════════════════════════════════
# New: Price Microstructure & Intraday Patterns (daily.parquet / daily.parquet)
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor(
    name="close_position_ratio",
    description="收盘价在日内高低点区间中的位置：(close-low)/(high-low)，强势收盘=排名高。",
    category="price",
    thesis=(
        "收盘价在日内高低点区间中的相对位置。接近1=收于日内高点附近(强势收盘、多头主导)，"
        "接近0=收于低点附近(弱势收盘、空头主导)。该指标比单看涨跌幅更能反映日内多空博弈结果——"
        "同样的涨幅，收于高点vs收于低点代表完全不同的日内走势质量。"
    ),
    dependencies=("daily.parquet",),
)
def factor_close_position_ratio(context: FactorContext) -> np.ndarray:
    daily = context.load("daily.parquet")
    range_hl = daily["high"] - daily["low"]
    position = safe_divide(daily["close"] - daily["low"], range_hl)
    position = position.clip(0, 1)
    return cross_sectional_rank(position)


@register_factor(
    name="gap_reversal_5d",
    description="跳空反转信号——跳空方向与日内走势方向相反时标记强度取反。高开低走/低开高走=趋势陷阱。",
    category="price",
    thesis=(
        "识别跳空方向与日内走势方向相反的趋势陷阱。高开低走=多头陷阱(开盘诱多后出货)；"
        "低开高走=空头陷阱(开盘诱空后吸筹)。趋势陷阱是强烈的反转信号，"
        "该因子对陷阱日给予高排名(预期发生反转)。"
    ),
    dependencies=("daily.parquet",),
)
def factor_gap_reversal_5d(context: FactorContext) -> np.ndarray:
    daily = context.load("daily.parquet")
    gap = safe_divide(daily["open"] - daily["pre_close"], daily["pre_close"])
    intraday = safe_divide(daily["close"] - daily["open"], daily["open"])
    # Trapped: gap direction != intraday direction
    trapped = ((np.sign(gap) * np.sign(intraday)) < 0).astype(float)
    reversal_5d = trapped.groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return cross_sectional_rank(reversal_5d)


@register_factor(
    name="high_low_expansion",
    description="日内振幅相对20日均值的扩张程度。振幅扩大=分歧加剧，振幅收缩=方向选择在即。",
    category="price",
    thesis=(
        "日内振幅(high-low)相对20日均值的扩张倍数。振幅突然扩大意味着多空分歧加剧、"
        "波动率突变——通常是重大信息冲击(利好或利空)或主力洗盘/出货的信号。"
        "振幅持续收缩意味着市场关注度下降或方向即将选择(暴风雨前的平静)。"
    ),
    dependencies=("daily.parquet",),
)
def factor_high_low_expansion(context: FactorContext) -> np.ndarray:
    d = context.load("daily.parquet")
    hl_range = d["high"] - d["low"]
    mean_range = hl_range.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    expansion = safe_divide(hl_range, mean_range)
    expansion = expansion.clip(0, 5)
    return cross_sectional_rank(expansion)


@register_factor(
    name="open_price_shock",
    description="开盘跳空幅度取正。大幅跳空=隔夜信息冲击强→短期反转概率高，排名高。",
    category="price",
    thesis=(
        "开盘跳空幅度(绝对值)的截面排名。大幅跳空(无论正负)意味着隔夜信息冲击强烈——"
        "集合竞价阶段出现极端不平衡。这种情况往往导致开盘后短期反转(跳空回补效应)。"
        "高幅度排名意味着反转交易机会更大。"
    ),
    dependencies=("daily.parquet",),
)
def factor_open_price_shock(context: FactorContext) -> np.ndarray:
    daily = context.load("daily.parquet")
    gap_abs = safe_divide(
        (daily["open"] - daily["pre_close"]).abs(),
        daily["pre_close"],
    )
    return cross_sectional_rank(gap_abs)


@register_factor(
    name="overnight_skewness_20d",
    description="20日隔夜收益偏度的绝对值取反。极端偏度=信息冲击不稳定，排名低。",
    category="price",
    thesis=(
        "20日隔夜收益(open/pre_close-1)的偏度。正偏度=偶尔大幅高开(利好集中释放)，"
        "负偏度=偶尔大幅低开(利空突袭)。极端偏度(无论正负)反映信息冲击的不稳定性——"
        "公司基本面存在不确定性或信息不对称严重。取绝对值后排名取反。"
    ),
    dependencies=("daily.parquet",),
)
def factor_overnight_skewness_20d(context: FactorContext) -> np.ndarray:
    daily = context.load("daily.parquet")
    overnight = safe_divide(daily["open"] - daily["pre_close"], daily["pre_close"])
    skew = overnight.groupby(level="Code").transform(
        lambda s: s.rolling(20, min_periods=10).skew()
    )
    skew = skew.clip(-5, 5)
    return cross_sectional_rank(-skew.abs())
