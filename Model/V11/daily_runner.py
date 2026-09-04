#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_runner.py — 冠军系统每日决策器 (norep+hold3 @ 打分文件)

用法:
  python3 daily_runner.py <score.fea> [--holding CODE --since YYYYMMDD]

输出: 今日决策 (买入/继续持有/卖出换仓/持有不足3天) 与 Top5 候选。
状态文件 holdings.json 自动维护 (记录当前持仓与买入日)。
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strat_backtest import load_calendar, load_prices

MIN_HOLD_DAYS = 3

def main():
    score_path = sys.argv[1] if len(sys.argv) > 1 else None
    if score_path is None:
        print('usage: daily_runner.py <score.fea>')
        return
    score = pd.read_feather(score_path).set_index('date').sort_index()
    tds = load_calendar()
    open_map, close_map, prev_close_map, amount_map, name_map = load_prices()
    names = name_map

    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holdings.json')
    state = {'code': None, 'buy_dt': None, 'since': None}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)

    last = score.index.max()
    top1 = score.loc[last].sort_values(ascending=False).index[0]
    top5 = score.loc[last].sort_values(ascending=False).head(5)

    def td_diff(a, b):
        ia, ib = tds.index(a) if a in tds else None, tds.index(b) if b in tds else None
        return (ib - ia) if (ia is not None and ib is not None) else None

    holding = state.get('code')
    print(f'== 每日决策 @ 因子日 {last} (次日开盘执行) ==')
    print(f'  打分文件: {score_path}')
    print(f'  Top1: {top1} {names.get(top1, "?")}  '
          f'(打分 {score.loc[last, top1]:+.2f})')
    print(f'  Top5: ' + ', '.join(f"{c}({names.get(c,'?')} {score.loc[last,c]:+.1f})"
                                   for c in top5.index))
    if holding:
        held_days = td_diff(state.get('buy_dt'), last) if state.get('buy_dt') else None
        print(f'  当前持仓: {holding} {names.get(holding, "?")} '
              f'(买入 {state.get("buy_dt")}, 持有 {held_days} 个交易日)')
        if top1 == holding:
            print('  → 决策: 继续持有 (持仓仍是 Top1, 卖出日顺延)')
        elif held_days is not None and held_days < MIN_HOLD_DAYS:
            print(f'  → 决策: 继续持有 (不足 {MIN_HOLD_DAYS} 天, 明日开盘不操作)')
        else:
            print(f'  → 决策: 次日开盘卖出 {holding}, 买入 {top1}')
            state = {'code': top1, 'buy_dt': _next_td(last, 1, tds), 'since': last}
    else:
        print('  → 决策: 空仓 → 次日开盘买入 Top1')
        state = {'code': top1, 'buy_dt': _next_td(last, 1, tds), 'since': last}
    with open(state_path, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f'  (状态已存 {state_path})')

def _next_td(d, n, tds):
    i = tds.index(d) if d in tds else None
    if i is None:
        return None
    j = i + n
    return tds[j] if j < len(tds) else None

if __name__ == '__main__':
    main()
