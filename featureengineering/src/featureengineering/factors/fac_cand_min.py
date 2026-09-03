"""
Class 3 (history_1min) fac_cand 分钟因子 — 回退注册函数。

来源于 Model/v8work/build_cand_min.py（2026-08-14 V8.4 迭代产物：13 个分钟级
候选经 240 天池 1d rankIC 筛查后选入模型的 9 个；pm_open_gap 因 13:00 分钟条
重复 bug 弃用，open30_ret/open30_vol_share/min_vol_gini 筛查未选入）。

主构建路径为 cli → build_intraday_new(INTRADAY_FACTOR_SPEC 扇出,见 intraday.py),
本文件的注册函数仅作为单因子直连构建(build_many)时的回退实现,
方向与 INTRADAY_FACTOR_SPEC 中同名条目一一对应:
  pos → cross_sectional_rank(+metric),  neg → cross_sectional_rank(−metric)。

与既有因子的等价关系（metric 复用，值或截面排名不变）:
  close_auction_ret ≡ close_auction_impact、last30_ret ≡ intra_rev_ret、
  vwap_close_ratio ≡ vwap_deviation（原脚本 vol 单位×100 仅作常数缩放，
  截面排名不变）。保留 V8 因子名以与 V8.x 训练配置（extra_factor_files）对齐。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank
from .intraday import _compute_intraday_factor


def _metric(context: FactorContext, name: str) -> pd.Series:
    return _compute_intraday_factor(
        context.repo.paths.source_root, context.repo.allowed_codes, name,
        on_progress=context.repo.on_progress,
    )


@register_factor(
    name="min_ret_std",
    description="1分钟收益标准差因子：日内分钟收益波动的截面排名（负向，波动大排后）。",
    category="intraday",
    thesis="1 分钟颗粒度的收益波动捕捉 5 分钟口径（ret_5min_std）无法刻画的"
           "高频噪音结构——分钟波动大=日内定价分歧剧烈、短期持有风险高。"
           "V8 筛查 meanIC -0.038/ICIR -0.22，为分钟候选最强档。",
    dependencies=("history_1min",),
)
def factor_min_ret_std(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "ret_1min_std"))


@register_factor(
    name="min_ret_min",
    description="1分钟收益最小值因子：日内最大分钟跌幅的截面排名（负向绝对值，深跌排后）。",
    category="intraday",
    thesis="日内单分钟最大跌幅刻画瞬时恐慌强度——深跌分钟=盘中瞬时抛压极端释放"
           "(闪崩风险),与 min_ret_max 构成分钟收益的尾部谱系。V8 筛查 "
           "meanIC +0.034/ICIR 0.20（原值方向为正）。",
    dependencies=("history_1min",),
)
def factor_min_ret_min(context: FactorContext):
    return cross_sectional_rank(_metric(context, "ret_1min_min"))


@register_factor(
    name="min_ret_max",
    description="1分钟收益最大值因子：日内最大分钟涨幅的截面排名（负向，急拉排后）。",
    category="intraday",
    thesis="单分钟最大涨幅=瞬时脉冲式拉升(游资点火特征)——急拉分钟频繁出现"
           "的股票短期筹码结构不稳、追高风险大。V8 筛查 meanIC -0.038/"
           "ICIR -0.23。",
    dependencies=("history_1min",),
)
def factor_min_ret_max(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "ret_1min_max"))


@register_factor(
    name="vwap_close_ratio",
    description="收盘价/分钟VWAP偏离因子：close/(amount/vol)-1 截面排名（负向）。",
    category="intraday",
    thesis="与既有 vwap_deviation 同秩（V8 原脚本 vol 单位×100 仅为常数缩放，"
           "截面排名不变），V8 池内方向为负：收盘显著高于分钟 VWAP=尾盘透支,"
           "低于 VWAP=尾盘修复空间。V8 筛查 meanIC -0.034/ICIR -0.20。",
    dependencies=("history_1min",),
)
def factor_vwap_close_ratio(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "vwap_dev"))


@register_factor(
    name="close_auction_ret",
    description="收盘集合竞价收益因子：14:57开盘→15:00收盘涨跌幅截面排名（负向）。",
    category="intraday",
    thesis="集合竞价是收盘定价的最终博弈——竞价段急拉=尾盘抢筹但次日兑现压力大,"
           "竞价段平稳/回落=定价充分。与既有 close_auction_impact 逐值等价。"
           "V8 筛查 meanIC -0.023/ICIR -0.50（方向最稳）。",
    dependencies=("history_1min",),
)
def factor_close_auction_ret(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "close_auction_impact"))


@register_factor(
    name="close_auction_vol_share",
    description="收盘集合竞价量占比因子：14:57-14:59成交额占全天比例截面排名（负向）。",
    category="intraday",
    thesis="竞价量占比=大资金借收盘定价窗口调仓的强度——竞价放量=尾盘博弈激烈、"
           "次日兑现压力大；竞价平静=定价充分。V8 筛查 meanIC -0.0143/"
           "ICIR -0.21，方向与既有 open_volume_share(neg) 一致。",
    dependencies=("history_1min",),
)
def factor_close_auction_vol_share(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "close_auction_vol_share"))


@register_factor(
    name="last30_ret",
    description="尾盘30分钟收益因子：14:30开盘→收盘涨跌幅截面排名（负向）。",
    category="intraday",
    thesis="尾盘30分钟是全天定价的收官段——尾盘急拉=短线资金尾盘偷袭(次日惯性"
           "存疑),尾盘平稳=全天定价充分。与既有 intra_rev_ret 逐值等价。",
    dependencies=("history_1min",),
)
def factor_last30_ret(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "intra_rev_ret"))


@register_factor(
    name="last30_vol_share",
    description="尾盘30分钟量占比因子：14:30-14:59成交额占全天比例截面排名（正向）。",
    category="intraday",
    thesis="尾盘量占比=资金在定价收官段的参与深度——尾盘放量(尤其伴随价格平稳)"
           "=机构调仓/建仓痕迹,次日延续性更强。与既有 tail_volume_share 的"
           "差异仅在 15:00 竞价条是否计入。",
    dependencies=("history_1min",),
)
def factor_last30_vol_share(context: FactorContext):
    return cross_sectional_rank(_metric(context, "last30_vol_share"))


@register_factor(
    name="tail_ret_3d",
    description="近3日尾盘收益累计因子：last30_ret 滚动3日和的截面排名（负向）。",
    category="intraday",
    thesis="连续3日尾盘收益的累计值捕捉尾盘资金的持续性——连续尾盘拉升=短线"
           "资金抱团(拥挤度上升),均值回归压力累积。",
    dependencies=("history_1min",),
)
def factor_tail_ret_3d(context: FactorContext):
    return cross_sectional_rank(-_metric(context, "tail_ret_3d"))
