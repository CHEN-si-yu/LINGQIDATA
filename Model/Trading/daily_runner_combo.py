#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_runner_combo.py — 多系统组合每日决策器 (模型×策略组合层)

组合形态 (iter5/iter6 结论, 稳定性优先):
  默认组合 (0.5/0.5): [V20_ensw2 + h20tr20_t2] × [V11_ensw2 + D01]
    → 真实净值 +301%, Sharpe 3.75, MaxDD -16.0%, 滚动60日正率 97.8%
  可选三系统 (1/3 各): + [V22_20d + S2_re300]
    → +225%, Sharpe 3.82, MaxDD -13.9%, 滚动正率 100%
每个子系统独立纸面状态 (holdings_combo.json), 资本 = 总资金 × 权重, 各自
两阶段结算/计划 (复用 daily_runner_v2.settle_and_plan 口径, 与引擎逐笔一致),
输出合并后的次日开盘指令 (先卖后买)。

用法:
  python3 daily_runner_combo.py                    # 默认 0.5/0.5 组合决策
  python3 daily_runner_combo.py --triple           # 三系统 1/3 各
  python3 daily_runner_combo.py --reset            # 重置全部子系统状态
  python3 daily_runner_combo.py --replay           # 回放校验 (组合净值 vs 引擎基线)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import load_calendar, load_market, load_scores, run_backtest, CAPITAL
import daily_runner_v2 as d2

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, 'holdings_combo.json')

# 组合配置: (名称, score 集, 协议 cfg, 资本权重)
CONFIG_PAIR = [
    ("A_V20h20", "V20_ensw2", dict(top_n=2, hold=20, trail_pct=0.20), 0.5),
    ("B_V11D01", "V11_ensw2", dict(top_n=1, hold=5, stop_loss=0.08), 0.5),
]
CONFIG_TRIPLE = CONFIG_PAIR[:2] + [
    ("C_V22S2", "V22_20d", dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
                                stop_loss=0.08, trail_pct=0.15), 1 / 3),
]


def _scores_for(name):
    if name == "V22_20d":
        df = pd.read_feather(_score_path(name)).set_index('date')
        return dict(scores={d: df.loc[d].dropna().to_dict() for d in df.index},
                    ranked={d: df.loc[d].dropna().sort_values(
                        ascending=False).index.tolist() for d in df.index},
                    top1={d: float(df.loc[d].max()) for d in df.index},
                    dates=sorted(df.index))
    return load_scores(name)


def replay(configs):
    """组合回放: 各子系统独立纸面推进 → 合并净值 → 与单系统基线对照。"""
    market = load_market()
    tds, tdi = load_calendar()
    equity_parts = {}
    for name, sname, proto, frac in configs:
        sc = _scores_for(sname)
        m, tr, eq = run_backtest(sc, {**proto, "capital_mult": frac},
                                 market, tds, tdi)
        equity_parts[name] = eq.set_index("date")["equity"]
    j = None
    for e in equity_parts.values():
        j = e.index if j is None else j.intersection(e.index)
    comb = sum(equity_parts[n].loc[j] for n in equity_parts).to_frame()
    comb.columns = ["equity"]
    comb["ret"] = comb["equity"].pct_change()
    r = comb["ret"].dropna()
    cum = float((1 + r).prod() - 1)
    sh = float(r.mean() / r.std() * np.sqrt(252))
    peak = np.maximum.accumulate(comb["equity"].values)
    dd = float((comb["equity"].values / peak - 1).min())
    h1 = float((1 + r.iloc[:len(r) // 2]).prod() - 1)
    h2 = float((1 + r.iloc[len(r) // 2:]).prod() - 1)
    pos = sum(1 for i in range(60, len(r) + 1)
              if (1 + r.iloc[i - 60:i]).prod() - 1 > 0)
    print(f'[replay] 组合净值: cum {cum:+.2%} | Sharpe {sh:.2f} | '
          f'MaxDD {dd:+.2%} | H1/H2 {h1:+.2%}/{h2:+.2%} | '
          f'滚动60日正率 {pos}/{len(r)-59}')
    return cum


def run_daily(configs):
    market = load_market()
    tds, tdi = load_calendar()
    state = {}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
    names = market['name_last']
    all_sells, all_buys, lines = [], [], []
    for name, sname, proto, frac in configs:
        score = pd.read_feather(_score_path(sname)).set_index('date').sort_index()
        last_fd = score.index.max()
        today = tds[tdi.get(last_fd, len(tds) - 1)]
        sub = state.get(name) or {'slots': [], 'cash': CAPITAL * frac,
                                  'prev_day': None, 'pending': None}
        ex_s, ex_b, tmr_s, tmr_b, sub = d2.settle_and_plan(
            score, market, sub, tds, tdi, today)
        state[name] = sub
        for s in tmr_s:
            all_sells.append((name, s))
        for b in tmr_b:
            all_buys.append((name, b))
        tmr = sub['pending']['exec_day']
        lines.append(f'  [{name}] 净值 {sub["cash"] + sum((market["close"].get((today, q["code"])) or q["buy_prc"])*q["lots"]*100 for q in sub["slots"]):.0f} 元'
                     f' | 今日结算 卖{len(ex_s)}/买{len(ex_b)} | 明日({tmr}) 卖{len(tmr_s)} 买{len(tmr_b)}')
    print('== 组合每日决策 (多系统) ==')
    for l in lines:
        print(l)
    print('  明日开盘卖出 (全部子系统): ' + ('; '.join(
        f"[{n}] {s['code']} {names.get(s['code'],'?')} {s['lots']}手 ({s['reason']})"
        for n, s in all_sells) if all_sells else '无'))
    print('  明日开盘买入 (全部子系统): ' + ('; '.join(
        f"[{n}] {b['code']} {names.get(b['code'],'?')}" for n, b in all_buys)
        if all_buys else '无'))
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f'  组合状态 → {STATE_PATH}')


def _score_path(sname):
    from engine import SCORE_SETS, PROJECT_ROOT
    return PROJECT_ROOT + SCORE_SETS[sname]


if __name__ == '__main__':
    cfg = CONFIG_TRIPLE if '--triple' in sys.argv else CONFIG_PAIR
    if '--reset' in sys.argv:
        if os.path.exists(STATE_PATH):
            os.remove(STATE_PATH)
        print('组合纸面状态已重置')
    elif '--replay' in sys.argv:
        replay(cfg)
    else:
        run_daily(cfg)
