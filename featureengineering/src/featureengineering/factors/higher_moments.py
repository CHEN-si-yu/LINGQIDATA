"""
日频收益高阶矩与下行风险因子。

收益序列直接用 daily.parquet 的 ``pct_chg``（数据商复权口径日收益，除权日
无跳变），无需自建基座。所有滚动矩用宽表一次调用（C 级向量化，符合
skill.md §3.4 的宽表 rolling 方案，避免 groupby 内逐股串行）。

原 stub 于 2026-07-31 因旧实现依赖 close.pct_change() 被清空，
本次以合规口径重建。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, stack_date_code


def _ret_wide(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily adjusted returns (pct_chg/100) as a Date × Code wide panel."""
    ret = daily["pct_chg"] / 100.0
    return ret.unstack("Code")


@register_factor(
    name="ret_skew_20",
    description="20日收益偏度因子（日收益20日滚动偏度，反向排名）。",
    category="risk",
    thesis="彩票偏好：高偏度股票被散户高估、未来收益更低，A股负偏度溢价显著。"
    "低（负）偏度股票收益分布稳健，排名靠前。与日内 rv_skew_intraday 互补——"
    "此处为日频收益率的三阶矩，捕捉跨日分布形态。",
    dependencies=("daily.parquet",),
)
def factor_ret_skew_20(context: FactorContext):
    daily = context.load("daily.parquet")
    skew = _ret_wide(daily).rolling(20, min_periods=15).skew()
    return cross_sectional_rank(-stack_date_code(skew))


@register_factor(
    name="ret_skew_60",
    description="60日收益偏度因子（日收益60日滚动偏度，反向排名）。",
    category="risk",
    thesis="中期收益偏度刻画季度级别的收益分布形态：持续正偏（偶发大涨）的股票"
    "常被市场高估，负偏度则隐含风险已被释放。60日窗口更稳健，与20日版本互补。",
    dependencies=("daily.parquet",),
)
def factor_ret_skew_60(context: FactorContext):
    daily = context.load("daily.parquet")
    skew = _ret_wide(daily).rolling(60, min_periods=40).skew()
    return cross_sectional_rank(-stack_date_code(skew))


@register_factor(
    name="ret_kurt_20",
    description="20日收益峰度因子（日收益20日滚动峰度，反向排名）。",
    category="risk",
    thesis="高峰度=收益分布尾部厚重、极端行情频繁，风险溢价要求更高而实际收益"
    "往往更差（彩票型收益特征）。低峰度股票收益路径平稳，排名靠前。",
    dependencies=("daily.parquet",),
)
def factor_ret_kurt_20(context: FactorContext):
    daily = context.load("daily.parquet")
    kurt = _ret_wide(daily).rolling(20, min_periods=15).kurt()
    return cross_sectional_rank(-stack_date_code(kurt))


@register_factor(
    name="downside_vol_ratio_20",
    description="下行波动占比因子（20日半波动/总波动比，反向排名）。",
    category="risk",
    thesis="下行波动占比衡量收益波动的方向不对称性：比值高=下跌贡献了大部分波动"
    "（负向不对称），风险尚未充分定价。比值低=波动主要由上涨驱动，"
    "持有体验好。",
    dependencies=("daily.parquet",),
)
def factor_downside_vol_ratio_20(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    neg = wide.clip(upper=0.0)
    var_all = (wide ** 2).rolling(20, min_periods=15).mean()
    var_neg = (neg ** 2).rolling(20, min_periods=15).mean()
    ratio = var_neg.pow(0.5) / var_all.pow(0.5).replace(0, np.nan)
    return cross_sectional_rank(-stack_date_code(ratio))


@register_factor(
    name="vol_of_vol_60",
    description="波动率的波动因子（60日收益std的20日std，反向排名）。",
    category="risk",
    thesis="波动率自身的不稳定性：vol-of-vol 高的股票波动状态频繁切换、"
    "regime 不稳定，预测难度与交易成本高；vol-of-vol 低=波动环境一致，"
    "低波异象更可靠。波动率二阶矩是常见风险因子缺项。",
    dependencies=("daily.parquet",),
)
def factor_vol_of_vol_60(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    vol60 = wide.rolling(60, min_periods=30).std()
    vov = vol60.rolling(20, min_periods=10).std()
    return cross_sectional_rank(-stack_date_code(vov))
