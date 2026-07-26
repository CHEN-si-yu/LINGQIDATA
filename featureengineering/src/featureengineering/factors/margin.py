"""
Margin trading factors — Class 1 panel factors.

Uses margin_detail.parquet daily-panel fields:
  - rzye: margin financing balance (融资余额)
  - rzmre: margin buy amount (融资买入额)
  - rzche: margin repay amount (融资偿还额)
  - rzrqye: total margin + short balance (融资融券总余额)

Coverage: ~85% of stocks have margin trading (NaN ≈ 15%, acceptable < 20%).
All factors use context.load("margin_detail.parquet") for daily-panel access.
"""

from __future__ import annotations

import numpy as np

from ..registry import FactorContext, register_factor
from ..utils import cross_sectional_rank, rolling_group_mean, safe_divide


# ── Margin Balance Change Factors ────────────────────────────────────────────

@register_factor(
    name="margin_balance_5d",
    description="融资余额5日变化率，反映杠杆资金短期流入/流出速度。余额增长=杠杆做多，排名高。",
    category="fund_flow",
    thesis="融资余额5日变化率捕捉杠杆资金的短期边际变化。余额快速增长意味着融资盘积极做多，"
           "是短期看多信号；余额下降意味着融资盘去杠杆，预示短期承压。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_balance_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    chg = m["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(5))
    chg = chg.clip(-0.5, 1.0)
    return cross_sectional_rank(chg)


@register_factor(
    name="margin_balance_20d",
    description="融资余额20日变化率，反映中期杠杆资金趋势。稳步增长=持续看多共识，排名高。",
    category="fund_flow",
    thesis="融资余额20日变化率过滤了短期噪音，反映中期杠杆资金趋势方向。"
           "稳步增长的融资余额代表持续的看多共识，比5日变化更可靠地捕捉中线资金态度。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_balance_20d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    chg = m["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(20))
    chg = chg.clip(-0.5, 1.0)
    return cross_sectional_rank(chg)


# ── Margin Buy/Sell Pressure Factors ─────────────────────────────────────────

@register_factor(
    name="margin_buy_pressure",
    description="融资买入压力比=融资买入/(融资买入+融资偿还)。>0.5=买入意愿强于偿还，排名高。",
    category="fund_flow",
    thesis="融资买入压力比率衡量杠杆资金的日内买卖倾向。比率>0.5意味着买入意愿强于偿还意愿，"
           "杠杆资金在主动加仓；比率<0.5意味着偿还压力大于买入意愿，杠杆资金在减仓或被动偿还。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_buy_pressure(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    total = m["rzmre"] + m["rzche"]
    pressure = safe_divide(m["rzmre"], total)
    return cross_sectional_rank(pressure)


@register_factor(
    name="margin_net_flow_ratio",
    description="融资净流入相对余额比=(融资买入-融资偿还)/融资余额，标准化净流量强度。",
    category="fund_flow",
    thesis="融资净流入相对余额的比率标准化了净流量的强度，使得不同融资余额规模的股票可比。"
           "高比率说明新的杠杆资金在加速涌入，是增量资金信号；负值意味着杠杆资金净流出。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_net_flow_ratio(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    net_flow = m["rzmre"] - m["rzche"]
    ratio = safe_divide(net_flow, m["rzye"])
    ratio = ratio.clip(-0.1, 0.1)
    return cross_sectional_rank(ratio)


@register_factor(
    name="margin_velocity",
    description="融资周转速度=(融资买入+融资偿还)/融资余额。高速度=投机性强，排名高。",
    category="fund_flow",
    thesis="融资周转速度反映融资盘的交易活跃度。高速度意味着融资盘换手频繁——"
           "投机性强、资金快进快出；低速度意味着融资盘锁定不动——筹码稳定性高、持仓信心强。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_velocity(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    total_flow = m["rzmre"] + m["rzche"]
    velocity = safe_divide(total_flow, m["rzye"])
    velocity = velocity.clip(0, 2)
    return cross_sectional_rank(velocity)


# ── Margin Stability / Risk Factors ──────────────────────────────────────────

@register_factor(
    name="margin_balance_volatility_20d",
    description="融资余额20日波动率(变异系数)，反映杠杆资金稳定性。低波动=方向一致，排名高(取反)。",
    category="fund_flow",
    thesis="融资余额20日波动率(变异系数=std/mean)衡量杠杆资金的稳定性。高波动反映融资盘"
           "多空观点频繁变化、持仓不稳定；低波动意味着融资盘方向一致、持仓信心稳定。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_balance_volatility_20d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    rzye = m["rzye"]
    roll_std = rzye.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
    roll_mean = rzye.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    cv = safe_divide(roll_std, roll_mean)
    cv = cv.clip(0, 0.5)
    return cross_sectional_rank(-cv)


@register_factor(
    name="margin_buy_momentum_5d",
    description="融资买入5日均值相对余额动量。持续买入=信号可靠，排名高。",
    category="fund_flow",
    thesis="融资买入5日均值相对余额反映买入行为的持续性。持续的高买入动量意味着"
           "杠杆资金在一段时间内保持积极进场态势，而非单日脉冲，信号更可靠。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_buy_momentum_5d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    buy_ma5 = m["rzmre"].groupby(level="Code").transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    momentum = safe_divide(buy_ma5, m["rzye"])
    momentum = momentum.clip(0, 0.5)
    return cross_sectional_rank(momentum)


@register_factor(
    name="margin_leverage_trend_10d",
    description="融资融券总余额10日变化率。总杠杆增加=风险偏好提升，排名高。",
    category="fund_flow",
    thesis="融资融券总余额(rzrqye=rzye+rqye)10日变化率反映整体杠杆水平变化趋势。"
           "总杠杆增加意味着市场风险偏好提升；总杠杆减少意味着避险情绪升温。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_leverage_trend_10d(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    chg = m["rzrqye"].groupby(level="Code").transform(lambda s: s.pct_change(10))
    chg = chg.clip(-0.3, 0.3)
    return cross_sectional_rank(chg)


@register_factor(
    name="margin_balance_ma_divergence",
    description="融资余额偏离20日均线幅度，极端偏离预示均值回归。正向偏离=可能超买，排名居中。",
    category="fund_flow",
    thesis="融资余额偏离20日均线的幅度反映杠杆资金的极端情绪。正向偏离(余额在均线上方)"
           "意味着融资盘快速涌入、可能超买；负向偏离意味着融资盘撤离、可能超卖。"
           "这是一个均值回归信号——极端偏离后倾向于回归。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_balance_ma_divergence(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    rzye = m["rzye"]
    ma20 = rzye.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
    div = safe_divide(rzye - ma20, ma20)
    div = div.clip(-0.1, 0.1)
    return cross_sectional_rank(div)


@register_factor(
    name="margin_repay_deceleration",
    description="融资偿还额5日变化率取反。偿还减速=空方力量减弱，排名高。",
    category="fund_flow",
    thesis="融资偿还额5日变化率的反向排名。偿还额下降(正排名)意味着杠杆资金不再急于平仓——"
           "看空力量减弱、持筹信心恢复。偿还额加速上升意味着恐慌性平仓、信心崩溃。",
    dependencies=("margin_detail.parquet",),
)
def factor_margin_repay_deceleration(context: FactorContext) -> np.ndarray:
    m = context.load("margin_detail.parquet")
    chg = m["rzche"].groupby(level="Code").transform(lambda s: s.pct_change(5))
    chg = chg.clip(-0.5, 0.5)
    return cross_sectional_rank(-chg)
