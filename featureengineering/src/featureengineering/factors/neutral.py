from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor

# ⚠️ 时点风险(2026-08-05 审计声明):_industry_neutral_rank 使用
# stock_list.parquet 的当前快照行业归属回填全部历史日期(今天才知道的行业
# 被用于历史截面)。行业归属变化缓慢,属可接受的轻微时点回溯;若上游提供
# 公告时点行业数据应替换。sector.py 已有同款声明。
from ..utils import cross_sectional_rank, rolling_group_mean


def _industry_neutral_rank(signal: pd.Series, context: FactorContext) -> pd.Series:
    """Within-industry cross-sectional percentile rank.

    signal: Series with (Date, Code) MultiIndex, raw factor values
    Returns a Series with the same index, values are within-industry percentile ranks.
    """
    industry_map = context.repo.load_industry_map()
    codes = signal.index.get_level_values("Code")
    industries = codes.map(industry_map)
    df = pd.DataFrame(
        {"signal": signal.values, "industry": industries.values},
        index=signal.index,
    )
    df = df.dropna(subset=["industry"])
    with np.errstate(invalid="ignore"):
        df["rank"] = df.groupby(["Date", "industry"])["signal"].rank(pct=True)
    return df["rank"]


# ── BP neutral ────────────────────────────────────────────────────────────

def _size_neutral_rank(signal: pd.Series, context: FactorContext, n_buckets: int = 10) -> pd.Series:
    """Within-size-bucket cross-sectional percentile rank.

    signal: Series with (Date, Code) MultiIndex, raw factor values.
    Stocks are bucketed by market cap within each date, then ranked within each bucket.
    """
    finance = context.load("finance.parquet")
    total_mv = finance["total_mv"]
    common = signal.index.intersection(total_mv.index)
    signal = signal.loc[common]
    total_mv = total_mv.loc[common]

    df = pd.DataFrame({
        "signal": signal.values,
        "total_mv": total_mv.values,
    }, index=signal.index)

    def _bucket_rank(grp):
        # 显式构造新列,避免在 .apply 回调内对分组子集做链式赋值
        # (pandas 3.0 下 SettingWithCopy/结果粘连风险,2026-08-05 修复)
        if len(grp) < n_buckets * 2:
            bucket = pd.Series(0, index=grp.index)
        else:
            bucket = pd.Series(
                pd.qcut(grp["total_mv"], n_buckets, labels=False, duplicates="drop"),
                index=grp.index,
            )
        return grp["signal"].groupby(bucket).rank(pct=True)

    df = df.groupby(level="Date", group_keys=False).apply(_bucket_rank)
    return df


@register_factor(
    name="bp_size_neutral",
    description="规模中性化BP因子，市值分桶内截面排名。消除市值与估值相关性。",
    category="neutral",
    thesis="BP与市值存在系统性相关性（大盘股普遍PB较低），规模中性化后提取更纯粹的估值信号，避免选到全是银行股的'价值陷阱'。",
    dependencies=("finance.parquet",),
)
def factor_bp_size_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    bp = 1.0 / finance["pb"].replace(0, np.nan)
    neutral = _size_neutral_rank(bp, context)
    return cross_sectional_rank(neutral)


@register_factor(
    name="turnover_20_size_neutral",
    description="规模中性化换手率因子，市值分桶内低换手排名。",
    category="neutral",
    thesis="小盘股换手率天然高，规模中性化后找到同市值级别中换手率偏低（筹码稳定）的股票。",
    dependencies=("finance.parquet",),
)
def factor_turnover_20_size_neutral(context: FactorContext):
    finance = context.load("finance.parquet")
    turnover = finance["turnover_rate"]
    to_20 = rolling_group_mean(turnover, 20)
    neutral = _size_neutral_rank(-to_20, context)
    return cross_sectional_rank(neutral)


