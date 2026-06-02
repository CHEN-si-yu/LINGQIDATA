# Pending Factor Ideas

Factors NOT yet implemented, organized by data source and priority.

---

## P0 — 筹码分布 (cyq_perf + cyq_chips) — Data Ready Immediately

### From cyq_perf (aggregated chip performance — easiest to implement)

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 1 | `winner_rate` | `cross_sectional_rank(winner_rate)` | price | 获利盘比例。高值=多数持仓者盈利，短期存在获利了结压力，反转信号。 |
| 2 | `chip_concentration` | `cross_sectional_rank(-(cost_95pct - cost_5pct) / cost_50pct)` | price | 筹码集中度。窄区间=筹码密集，筹码峰突破后趋势性强（正向动量前置指标）。 |
| 3 | `chip_position` | `cross_sectional_rank((close - cost_5pct) / (cost_95pct - cost_5pct))` | price | 当前价在筹码分布中的相对位置。高位=接近套牢区上沿，上升阻力大。 |
| 4 | `cost_displacement` | `cross_sectional_rank(-(close - weight_avg) / weight_avg)` | price | 现价相对加权平均成本的偏离。大幅偏离=大量获利盘/套牢盘，均值回归倾向。 |
| 5 | `winner_rate_change_5d` | `cross_sectional_rank(winner_rate - winner_rate.shift(5))` | price | 获利盘5日变化。快速上升=上涨中筹码快速获利，加速赶顶信号。 |
| 6 | `chip_cost_skew` | `cross_sectional_rank((cost_50pct - cost_15pct) / (cost_85pct - cost_50pct))` | price | 成本分布偏度。右偏=上方套牢盘重（负向），左偏=下方获利盘多（正向）。 |

**Data**: `cyq_perf.parquet` — columns: cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct, weight_avg, winner_rate, his_low, his_high

### From cyq_chips (raw chip distribution — more complex)

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 7 | `chip_peak_ratio` | `cross_sectional_rank(当前价以下筹码面积 / 总筹码面积)` | price | 下方筹码占比。比例越高=下方支撑越强，下跌空间有限。 |
| 8 | `chip_support_distance` | `cross_sectional_rank((close - 最大筹码峰价格) / close)` | price | 现价相对最大筹码峰的距离。在筹码峰上方=有强支撑（正向），下方=筹码峰变压力（负向）。 |

**Data**: `cyq_chips.parquet` — columns: trade_date, stock_code, price, percent
**Engineering cost**: Medium — need per-date per-stock aggregation from histogram data.

---

## P0 — 股东户数 (holder_number) — Needs Financial FFill

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 9 | `holder_num_change` | `cross_sectional_rank(-holder_num / holder_num.shift(1) + 1)` | quality | 股东户数变化率（季频前向填充）。减少=筹码集中/主力吸筹，正向信号。 |
| 10 | `holder_num_change_yoy` | `cross_sectional_rank(-holder_num / holder_num.shift(4) + 1)` | quality | 年度股东户数变化。捕捉中长期集中度趋势，过滤季节性。 |
| 11 | `holder_num_percentile` | `cross_sectional_rank(-(holder_num的2年滚动百分位))` | quality | 当前股东户数在历史上的分位数。历史低位=极度集中。 |
| 12 | `avg_holding_mv` | `cross_sectional_rank(circ_mv / holder_num)` | quality | 户均持股市值。高值=机构化/大户化，正向信号。 |

**Data**: `holder_number.parquet` (+ `finance.parquet` for circ_mv)
**Engineering cost**: Medium — needs `load_financial()` pipeline for daily forward-fill. Date column is `ann_date`.

---

## P0 — 财务指标未用字段 (financial_indicator.parquet)

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 13 | `eps_rank` | `cross_sectional_rank(eps)` | quality | 每股收益截面排名。最基础的盈利能力因子。 |
| 14 | `bps_rank` | `cross_sectional_rank(bps)` | valuation | 每股净资产截面排名。 |
| 15 | `ocfps_rank` | `cross_sectional_rank(ocfps)` | quality | 每股经营现金流截面排名。现金盈利质量高于会计盈利。 |
| 16 | `ebitda_to_ev` | `cross_sectional_rank(ebitda / (total_mv + netdebt))` | valuation | 企业价值倍数。比PE更适合跨行业比较，不受资本结构影响。 |
| 17 | `ocf_coverage` | `cross_sectional_rank(ocf_to_shortdebt)` | quality | 短期偿债现金流覆盖。OCF覆盖短期债务的能力。 |
---

## P1 — 指数权重 (index_weight) — Needs Monthly FFill

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 19 | `index_membership_count` | `cross_sectional_rank(股票被多少指数纳入)` | index | 指数覆盖度。被越多核心指数纳入，被动资金流入越确定。 |
| 20 | `index_weight_hs300` | `cross_sectional_rank(沪深300权重)` | index | 沪深300权重排名。权重越高，被动配置规模越大。 |
| 21 | `index_weight_change` | `cross_sectional_rank(权重月度环比变化)` | index | 权重边际变化。权重提升=指数调仓带来的被动买入需求。 |

**Data**: `index_weight.parquet` — monthly index constituent weights for 8 major indices.

---

## P1 — 股权质押 (pledge_stat)

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 22 | `pledge_risk` | `cross_sectional_rank(-pledge_ratio)` | quality | 股权质押风险。高质押比例=大股东资金紧张/爆仓风险/控制权转移风险。 |
| 23 | `pledge_ratio_change` | `cross_sectional_rank(-(pledge_ratio的半年变化))` | quality | 质押比例边际变化。快速上升更危险，可能反映大股东链式危机。 |

**Data**: `pledge_stat.parquet` — date_col is `end_date`.

---

## P2 — 未复权日线特有字段 (daily.parquet)

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 24 | `overnight_gap` | `cross_sectional_rank(-(open - pre_close) / pre_close)` | price | 隔夜跳空。A股隔夜跳空通常反转（高开回落/低开反弹）。 |
| 25 | `intraday_ret` | `cross_sectional_rank((close - open) / open)` | price | 日内收益。日内与隔夜收益相关性低，提供独立的alpha维度。 |
| 26 | `gap_fill_5d` | 近5日缺口回补频率 | price | 缺口回补倾向。A股有缺口必补的民间说法，量化验证。 |

**Data**: `daily.parquet` — has `pre_close` column.

---

## P2 — 日内分钟线 (history_1min) — High Engineering Cost

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 27 | `intraday_volatility` | `cross_sectional_rank(-日内分钟收益率的std)` | price | 日内波动率。高日内波动=信息不对称大/投机性强，预期收益低。 |
| 28 | `tail_30m_return` | `cross_sectional_rank(最后30分钟收益率)` | price | 尾盘效应。机构交易集中在尾盘，尾盘涨=聪明钱看好。 |
| 29 | `open_30m_return` | `cross_sectional_rank(开盘30分钟收益率)` | price | 开盘动量。隔夜信息在开盘30分钟内快速消化。 |
| 30 | `am_pm_volume_ratio` | `cross_sectional_rank(上午成交量/下午成交量)` | price | 上午放量=强势信号，下午放量=弱势信号。 |
| 31 | `volume_concentration` | `cross_sectional_rank(-max(30min成交量)/全天成交量)` | price | 成交集中度。集中于某一时段=机构行为/知情交易。 |
| 32 | `intraday_reversal` | `cross_sectional_rank(-(日内最高→收盘回撤))` | price | 冲高回落幅度。大幅冲高回落=短期顶部信号（负向）。 |
| 33 | `intraday_momentum` | `cross_sectional_rank((收盘-开盘)/(最高-最低))` | price | 日内方向持续度。阳线实体占比大=日内买盘持续性强。 |

**Data**: `history_1min/` (2,542 per-stock files) or `daily_dump_1min/` (daily dump format).
**Engineering cost**: HIGH — need minutely aggregation pipeline. Start with `daily_dump_1min/` (5-day rolling window).

---

## P3 — Composite / Interaction Factors — No New Data

| # | Name | Formula | Category | Thesis |
|---|------|---------|----------|--------|
| 34 | `bp_x_roe` | `cross_sectional_rank(bp_rank * roe_rank)` | enhanced | BP×ROE交互。价值+质量交叉，避免价值陷阱。 |
| 35 | `mom_x_winner` | `cross_sectional_rank(mom_20_rank * winner_rate_rank)` | enhanced | 动量×筹码验证。有筹码支撑的动量更可靠。 |
| 36 | `roe_momentum_4q` | `cross_sectional_rank(roe - roe.shift(4))` | quality | ROE季度变化率。盈利改善加速度，比静态ROE更敏感。 |
| 37 | `margin_of_safety` | `cross_sectional_rank(bp_rank * ocf_quality_rank)` | enhanced | 估值×质量综合。低估值+强现金流=安全边际。 |

---

## Edit Checklist

When implementing a factor from this list:
1. Update the factor count in the table
2. Mark the factor as implemented (add ✅ or remove row)
3. If a new data source is tapped, update `data-catalog.md`
