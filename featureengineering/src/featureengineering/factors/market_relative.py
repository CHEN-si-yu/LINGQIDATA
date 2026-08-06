"""
市场相对因子：个股收益相对全池等权市场收益的暴露（beta、相关性、特质波动）。

市场代理为 daily.parquet 自身构建：pct_chg 的截面等权均值（含 NaN 自动
跳过停牌股，当日上市股票数 < 100 时置 NaN 防止早期稀疏）。收益序列本身为
复权口径（pct_chg），无需基座。全部滚动估计用宽表一次调用（C 级向量化，
符合 skill.md §3.4）。
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


def _market_proxy(wide: pd.DataFrame) -> pd.Series:
    """Equal-weighted market return per date (skipna over listed stocks)."""
    # pandas 3.0 移除了 DataFrame.mean 的 min_count 参数，改为先计数再掩码
    n_listed = wide.notna().sum(axis=1)
    return wide.mean(axis=1).where(n_listed >= 100)


def _rolling_beta(wide: pd.DataFrame, mkt: pd.Series, window: int,
                  min_periods: int) -> pd.DataFrame:
    """Rolling market beta per stock: cov(ret_i, mkt) / var(mkt)."""
    var_mkt = mkt.rolling(window, min_periods=min_periods).var()
    cov = wide.rolling(window, min_periods=min_periods).cov(mkt)
    return cov.div(var_mkt.replace(0, np.nan), axis=0)


@register_factor(
    name="beta_60",
    description="60日市场贝塔因子（对全池等权市场收益的60日滚动beta，反向排名）。",
    category="risk",
    thesis="低贝塔异象（A股同样成立）：高贝塔股票承担更多系统性风险却未被充分补偿。"
    "低贝塔股票排名靠前。与日内微观结构风险因子互补，是日频系统性风险暴露的基础衡量。",
    dependencies=("daily.parquet",),
)
def factor_beta_60(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    mkt = _market_proxy(wide)
    beta = _rolling_beta(wide, mkt, 60, 30)
    return cross_sectional_rank(-stack_date_code(beta))


@register_factor(
    name="corr_market_60",
    description="60日市场相关性因子（个股日收益与全池等权市场收益的60日滚动相关，反向排名）。",
    category="risk",
    thesis="与市场高度同步的股票缺乏独立alpha来源、且在系统性下跌时无分散价值；"
    "低相关股票更可能由自身基本面驱动。低相关性排名靠前，与 beta_60 互补"
    "（相关性与贝塔的差异在于特质波动成分）。",
    dependencies=("daily.parquet",),
)
def factor_corr_market_60(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    mkt = _market_proxy(wide)
    corr = wide.rolling(60, min_periods=30).corr(mkt)
    return cross_sectional_rank(-stack_date_code(corr))


@register_factor(
    name="idio_vol_60",
    description="60日特质波动率因子（剔除市场暴露后的残差波动，反向排名）。",
    category="risk",
    thesis="特质波动率异象（Ang et al.）：特质波动率高的股票未来收益系统性偏低"
    "（套利限制+彩票偏好），A股广泛验证。残差法：ret - beta*mkt 的60日滚动标准差。"
    "8.2 曾删除了 regime 分片的 idiosyncratic_vol_60_*，此为未分片完整版。",
    dependencies=("daily.parquet",),
)
def factor_idio_vol_60(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    mkt = _market_proxy(wide)
    beta = _rolling_beta(wide, mkt, 60, 30)
    resid = wide - beta.multiply(mkt, axis=0)
    idio = resid.rolling(60, min_periods=30).std()
    return cross_sectional_rank(-stack_date_code(idio))


@register_factor(
    name="market_beta_change_20",
    description="贝塔变化因子（20日贝塔-60日贝塔=系统性风险暴露的短期变化）。",
    category="risk",
    thesis="贝塔加速上升=资金正系统性涌入（风险偏好抬升），常伴随行情启动；"
    "贝塔快速下降=防御性调仓。贝塔动量捕捉市场风格切换的先行信号，"
    "与绝对贝塔水平正交。",
    dependencies=("daily.parquet",),
)
def factor_market_beta_change_20(context: FactorContext):
    daily = context.load("daily.parquet")
    wide = _ret_wide(daily)
    mkt = _market_proxy(wide)
    beta20 = _rolling_beta(wide, mkt, 20, 10)
    beta60 = _rolling_beta(wide, mkt, 60, 30)
    change = beta20 - beta60
    return cross_sectional_rank(stack_date_code(change))
