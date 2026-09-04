#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix_grid2.py — 对任意版本落盘的头部逐折矩阵做权重网格 (官方外部-z 定义)

用法: python3 mix_grid2.py <heads_dir>  (heads_dir 内含 {r1,r5,r3,top}_f{1..8}.fea)
打分 = Σ_folds z( w1*zr1_f + w5*zr5_f + w3*zr3_f + wt*ztop_f )
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strat_backtest import (run_backtest, load_prices, load_calendar, eval_ic)


def main():
    heads_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if heads_dir is None or not os.path.isdir(heads_dir):
        print('usage: mix_grid2.py <heads_dir>')
        return
    head_names = ['r1', 'r5', 'r3', 'top']
    perfold = {}
    n_folds = 0
    for h in head_names:
        files = sorted(glob.glob(os.path.join(heads_dir, f'{h}_f*.fea')))
        folds = {}
        for p in files:
            fi = int(os.path.basename(p).split('_f')[1].split('.')[0])
            folds[fi] = pd.read_feather(p).set_index('date')
        if folds:
            n_folds = max(n_folds, max(folds.keys()))
            perfold[h] = folds
    print(f"[mix2] heads_dir={heads_dir} folds={n_folds}")

    open_map, close_map, prev_close_map, amount_map, name_map = load_prices()
    tds = load_calendar()
    F = dict(use_cost=True, exclude_st=True, exclude_limit_up=True, buy_gap_limit=0.095)

    combos = []
    for w5 in [0.0, 0.25, 0.5]:
        for w3 in [0.0, 0.25, 0.5]:
            for wt in [1.0, 1.5, 2.0]:
                combos.append((1.0, w5, w3, wt))

    rows = []
    for w1, w5, w3, wt in combos:
        score = None
        for f in sorted(perfold['r1'].keys()):
            fs = w1 * perfold['r1'][f]
            if w5 and 'r5' in perfold:
                fs = fs + w5 * perfold['r5'][f]
            if w3 and 'r3' in perfold:
                fs = fs + w3 * perfold['r3'][f]
            fs = fs + wt * perfold['top'][f]
            fs = (fs - fs.mean(axis=1).values[:, None]) / \
                 fs.std(axis=1).values[:, None]
            score = fs if score is None else score.add(fs, fill_value=0.0)
        ic = eval_ic(score)
        s1, _, _ = run_backtest(score, open_map, close_map, prev_close_map,
                                name_map, amount_map=amount_map, tds=tds, **F)
        s5, _, _ = run_backtest(score, open_map, close_map, prev_close_map,
                                name_map, amount_map=amount_map, tds=tds,
                                hold=5, stop_loss=0.08, **F)
        s10, _, _ = run_backtest(score, open_map, close_map, prev_close_map,
                                 name_map, amount_map=amount_map, tds=tds,
                                 hold=10, stop_loss=0.08, **F)
        rows.append({
            'w1': w1, 'w5': w5, 'w3': w3, 'wtop': wt,
            'RankIC': round(ic['RankIC'], 4),
            'ICIR': round(ic['RankICIR'], 4),
            'top_ret': round(ic['top_return'], 4),
            'hold1净%': round(s1['net']['cum'] * 100, 2) if s1 else None,
            'hold1DD%': round(s1['net']['maxdd'] * 100, 2) if s1 else None,
            'hold5s8净%': round(s5['net']['cum'] * 100, 2) if s5 else None,
            'hold5s8DD%': round(s5['net']['maxdd'] * 100, 2) if s5 else None,
            'hold10s8净%': round(s10['net']['cum'] * 100, 2) if s10 else None,
            'hold10s8DD%': round(s10['net']['maxdd'] * 100, 2) if s10 else None,
        })

    table = pd.DataFrame(rows)
    print("\n===== 权重网格 (按 hold5s8净% 排序) =====")
    print(table.sort_values('hold5s8净%', ascending=False).head(25).to_string(index=False))
    out = os.path.join(heads_dir, 'mix_grid2.csv')
    table.to_csv(out, index=False)
    print(f"\n[mix2] 已存 {out}")
    print("\n===== Top-10 by RankICIR =====")
    print(table.sort_values('ICIR', ascending=False).head(10).to_string(index=False))
    print("\n===== Top-10 by hold1净% =====")
    print(table.sort_values('hold1净%', ascending=False).head(10).to_string(index=False))


if __name__ == '__main__':
    main()
