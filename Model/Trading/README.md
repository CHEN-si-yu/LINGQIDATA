# Model/Trading — 普适性交易策略研究 (V1~V20)

以"开盘先卖后买换仓"为核心的 5W 资金、1~2 只持仓、隔日开盘执行、带止盈止损的
普适性交易策略回测框架。结论汇总见 `SUMMARY_REPORT.md`。

## 快速使用

```bash
cd /autodl-fs/data/lingqiData/Model/Trading

# 0) 引擎回归校准 (与旧引擎 V11/strat_backtest.py 逐笔对齐, 应全部 OK)
python3 calibrate.py

# 1) 批量回测 (24 个打分集 × 策略电池, 多进程)
python3 run_battery.py --battery B1        # 主电池 70 配置
python3 run_battery.py --battery B2        # 细化网格 146 配置
python3 run_battery.py --battery B3        # 排名退出组合族 44 配置
python3 run_battery.py --battery SELLMODE  # 卖出口径对照 (含净值落盘)
python3 run_battery.py --battery ALL       # 全部

# 2) 聚合评估 (普适性 U 得分 + 图表 fig1/fig2)
python3 aggregate.py

# 3) 稳健性矩阵 (成本/过滤/本金/口径/三等分)
python3 robustness.py

# 4) 图表 (fig3 卖出口径 / fig4 主推净值 / fig5 逐模型 / fig6 V20 聚焦)
python3 plot_figs.py
```

## 主推协议 (S2_re300_sl8_tr15)

- 持仓 2 只等权 (每仓 2.5W), 每晚打分 → 次日开盘先卖后买换仓;
- 卖出触发 (收盘 → 次日开盘): 止损 -8% / 移动止盈 -15%(自峰值) / 排名掉出前300 (持有≥2日);
- 买入: 空位补入排名最高的未持有标的 (跳过 ST/退/一字涨停);
- 成本: 佣金万2.5(最低5)+印花税万5(卖)+过户费万0.1; 整手100股。
- 跨 24 模型: 中位净收益 +79.0%, 正收益 19/24, 22/24 胜过隔日换仓基线,
  H1/H2 双正 16/24, 中位 MaxDD -17.0%。

## 目录结构

```
Trading/
├── engine.py            # 回测引擎 (open/close_overlap/close_seq 三口径)
├── battery.py           # 策略电池定义 (272 配置)
├── run_battery.py       # 多进程批量回测
├── aggregate.py         # 跨模型聚合/普适性评分/图表
├── robustness.py        # 稳健性检验
├── calibrate.py         # 引擎校准回归
├── dump_equity.py       # 指定 (集×策略) 净值/成交落盘
├── plot_figs.py         # 图表产出
├── SUMMARY_REPORT.md    # 总结报告 (权威汇总)
├── results/             # 全部结果 CSV + equity/ + trades/
└── figures/             # fig1~fig6
```

## 打分集 (24 个)

V1~V17 `all_zscore_score` (16) + V11 mixA/mixB/mixC/ens_w2 (4) + V18 f1/f2 (2)
+ V20 `score_ens_w2` (1) + V11 allz。V19 无打分产物未纳入。
窗口: 因子日 20250901~20260901 (官方 Test 集, 243 交易日), 价格至 20260904。

## 关键口径约定

- 决策只用当日收盘可获得的信息; 执行只在次日开盘。
- close_overlap = 旧 V20 冠军口径 (换仓日开盘买入+收盘卖出), 需要 ≈2× 资金
  (实测峰值 99,909 元), 且隐含 T+1 瑕疵 — 仅用于复现/对照, 不推荐执行。
- 引擎与旧引擎校准差异 < 3pp, 全部来自期末强制清仓 (有意设计)。
