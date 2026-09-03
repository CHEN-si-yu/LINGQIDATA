"""
fac_new 第二轮融资余额因子（2 个，Class 1，数据源 margin_detail.parquet）。

⚠️ 数据泄露修复（重要）：原脚本 build_fac_new_r2.py 直接读 margin_detail
原始文件（余额日口径，T+1 才到达），导致 T 日的因子用了 T 日余额——存在
1 个交易日的信息泄露。本注册版通过 context.load("margin_detail.parquet")
加载，pipeline 的 load_panel 已对 margin 做 1 个交易日平移（余额日 T-1 重标
为可用日 T），因子值 = 原脚本整体后移 1 个交易日标签，PIT 正确。
"""

from __future__ import annotations

import pandas as pd

from ..registry import FactorContext, register_factor
from ._fac_new_common import ind_mean, with_industry


def _margin(context: FactorContext) -> pd.DataFrame:
    m = context.load("margin_detail.parquet")  # 已平移 1 交易日（PIT）
    df = with_industry(m.reset_index(), context)
    df["rz_chg"] = df.groupby("Code")["rzye"].pct_change(5)
    return df


def _out(df: pd.DataFrame, name: str, values: pd.Series) -> pd.Series:
    out = df[["Date", "Code"]].copy()
    out[name] = values.to_numpy()
    return out.set_index(["Date", "Code"])[name]


@register_factor(
    name="margin_chg_abs_5d",
    description="融资余额5日变化率（rzye 5日pct_change，PIT 平移后）。",
    category="fund_flow",
    thesis="融资余额变化捕捉杠杆资金边际动向：余额增长=杠杆做多升温，"
    "下降=去杠杆承压。本因子为个股自身变化率。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_chg_abs_5d(context: FactorContext):
    df = _margin(context)
    return _out(df, "margin_chg_abs_5d", df["rz_chg"])


@register_factor(
    name="margin_chg_rel_ind_5d",
    description="融资余额5日变化率行业相对（减去行业等权均值，PIT 平移后）。",
    category="fund_flow",
    thesis="行业内相对杠杆变化：同一板块中融资余额加速增长的个股=资金更看好，"
    "剥离板块整体杠杆环境。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_chg_rel_ind_5d(context: FactorContext):
    df = _margin(context)
    vals = df["rz_chg"] - ind_mean(df, "rz_chg")
    return _out(df, "margin_chg_rel_ind_5d", vals)
