#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_v17.py — V17 (5d 专属模型族) 评估: 单独 + 作为冠军第三组件

打分 = Σ_f z(w1·zr1 + w5·zr5 + w3·zr3 + wt·ztop)  (V17 顶部头为 5d 主目标)
策略: hold5s8-收盘卖 (冠军协议), 分半
集成: z(mixA) + 2·z(v13) + w·z(v17)  w ∈ {0.25, 0.5, 1, 1.5}
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'V11'))
from strat_backtest import load_prices, load_calendar, eval_ic

HERE = os.path.dirname(os.path.abspath(__file__))
V17_HEADS = os.path.join(HERE, 'model_pred/2026q3/heads')
V11_DIR = os.path.join(HERE, '..', 'V11', 'model_pred', '2026q3')
V13_HEADS = os.path.join(HERE, '..', 'V13', 'model_pred', '2026q3', 'heads')


def zn(df):
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]


def build(heads_dir, w1, w5, w3, wt, nf=8):
    perfold = {h: {} for h in ['r1', 'r5', 'r3', 'top']}
    for h in perfold:
        for f in range(1, nf + 1):
            p = os.path.join(heads_dir, f'{h}_f{f}.fea')
            if os.path.exists(p):
                perfold[h][f] = pd.read_feather(p).set_index('date')
    score = None
    for f in sorted(perfold['r1'].keys()):
        fs = w1 * perfold['r1'][f]
        if w5 and 'r5' in perfold:
            fs = fs + w5 * perfold['r5'][f]
        if w3 and 'r3' in perfold:
            fs = fs + w3 * perfold['r3'][f]
        if 'top' in perfold:
            fs = fs + wt * perfold['top'][f]
        fs = zn(fs)
        score = fs if score is None else score.add(fs, fill_value=0.0)
    return score


def main():
    open_map, close_map, prev_close_map, amount_map, name_map = load_prices()
    tds = load_calendar()
    F = dict(use_cost=True, exclude_st=True, exclude_limit_up=True, buy_gap_limit=0.095,
             hold=5, stop_loss=0.08)
    # 收盘卖引擎
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'V11', 'strat_backtest.py')).read()
    src_cs = src.replace("sp = open_map.get((p['planned_sell'], code))",
                         "sp = close_map.get((p['planned_sell'], code))")
    mod = {}
    exec(src_cs, mod)
    rb = mod['run_backtest']

    mixA = pd.read_feather(os.path.join(V11_DIR, 'score_mixA_top.fea')).set_index('date')
    v13 = build(V13_HEADS, 1.0, 0.0, 0.0, 1.0)
    champ = pd.read_feather(os.path.join(V11_DIR, 'score_ens_w2.fea')).set_index('date')

    print(f'{"打分":<26}{"hold5s8收盘卖%":>16}{"DD%":>8} | {"H1%":>9}{"H2%":>9}')
    # V17 自身最优 mix 网格 (w5, w3, wt 精简)
    best = (None, None, None, -1e9)
    for w5 in [0.0, 0.25, 0.5]:
        for wt in [0.5, 1.0, 1.5]:
            v17 = zn(build(V17_HEADS, 1.0, w5, 0.0, wt))
            m, _, _ = rb(v17, open_map, close_map, prev_close_map, name_map,
                         amount_map=amount_map, tds=tds, **F)
            if m['net']['cum'] > best[3]:
                best = (w5, wt, v17, m['net']['cum'])
    w5b, wtb, v17_best, v = best
    print(f'V17 自身最优 (w5={w5b}, wt={wtb}): {v * 100:.2f}%')

    rows = [('champ ens_w2', zn(champ))]
    for w in [0.25, 0.5, 1.0, 1.5]:
        sc = zn(mixA).add(2.0 * zn(v13), fill_value=0.0).add(w * zn(v17_best), fill_value=0.0)
        rows.append((f'champ + {w}*V17', sc))
    for name, sc in rows:
        m, _, _ = rb(sc, open_map, close_map, prev_close_map, name_map,
                     amount_map=amount_map, tds=tds, **F)
        h1, _, _ = rb(sc, open_map, close_map, prev_close_map, name_map,
                      amount_map=amount_map, tds=tds,
                      window_start='20250901', window_end='20260227', **F)
        h2, _, _ = rb(sc, open_map, close_map, prev_close_map, name_map,
                      amount_map=amount_map, tds=tds,
                      window_start='20260302', window_end='20260901', **F)
        n = m['net']
        print(f'{name:<26}{n["cum"] * 100:>16.2f}{n["maxdd"] * 100:>8.2f} | '
              f'{h1["net"]["cum"] * 100:>9.2f}{h2["net"]["cum"] * 100:>9.2f}')


if __name__ == '__main__':
    main()
