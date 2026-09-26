#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性对比: 冻结慢腿协议 (u_h20_re300_tr15) 下 V8_allz vs best(0901 全量重训) 打分源。
口径 = engine.run_backtest + DEFAULT_CFG (open 执行/含成本/5W), 窗口 20250901~20260901。
用法: python3 _slow_cmp_best_v8.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (load_market, load_scores, run_backtest, SCORE_SETS,
                    WINDOW_START, WINDOW_END)

PROBE = "Model/_cmp_best_on_v8win/model_pred/2026q3/all_zscore_score.fea"
SCORE_SETS["_best_probe"] = PROBE

SLOW_CFG = dict(top_n=2, hold=20, exit_rank=300, min_hold=2, trail_pct=0.15,
                name="慢腿(u_h20_re300_tr15)")

market = load_market()
tds = None
for name in ["V8_allz", "_best_probe"]:
    scores = load_scores(name, window=(WINDOW_START, WINDOW_END))
    m, tr, eq = run_backtest(scores, SLOW_CFG, market)
    print(f"\n=== 慢腿回测 [{name}] 窗口 {WINDOW_START}~{WINDOW_END} "
          f"({m['n_days']} 日) ===")
    print(f"  累计净值  cum_net   = {m['cum_net']:+.2%}")
    print(f"  累计(逐笔) cum_trade = {m['cum_trade']:+.2%}  (trade 数 {m['n_trades']})")
    print(f"  年化 Sharpe          = {m['sharpe']:.2f}")
    print(f"  MaxDD                = {m['maxdd']:.2%}")
    print(f"  H1 / H2 (净值分段)   = {m['h1_cum']:+.2%} / {m['h2_cum']:+.2%}")
    print(f"  胜率 {m['win_rate']:.1%} | 空仓日 {m['empty_days']} | "
          f"成本拖累 {m['cost_drag']:.2%} | 换手 {m['turnover']:.2f}x")
