#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix_grid.py — 头部混合权重网格 (官方外部-z 定义)

打分 = Σ_folds z( w1*zr1_f + w5*zr5_f + wt*ztop_f )
即 V9 官方集成定义 (每折先加权求和再 z 归一化) 的推广; w5=0,wt=1 时与 V9 完全一致。
评估: RankIC/ICIR/top_return + 三种策略回测 (hold1成本 / hold5+stop8% / hold10+stop8%)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strat_backtest import (run_backtest, load_prices, load_calendar, eval_ic,
                            WINDOW_START, WINDOW_END)

MIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mix_v9')


def zscore_df(df):
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]


def main():
    perfold = {}
    n_folds = 8
    for h in ['r1', 'r5', 'top']:
        folds = {}
        for f in range(1, n_folds + 1):
            p = os.path.join(MIX_DIR, f'{h}_f{f}.fea')
            if not os.path.exists(p):
                continue
            folds[f] = pd.read_feather(p).set_index('date')
        perfold[h] = folds
        print(f"[mix] {h}: {len(folds)} fold matrices, "
              f"{next(iter(folds.values())).shape}")

    open_map, close_map, prev_close_map, amount_map, name_map = load_prices()
    tds = load_calendar()
    F = dict(use_cost=True, exclude_st=True, exclude_limit_up=True, buy_gap_limit=0.095)

    combos = []
    for w5 in [0.0, 0.25, 0.5, 1.0]:
        for wt in [0.5, 1.0, 1.5, 2.0]:
            combos.append((1.0, w5, wt))

    rows = []
    for w1, w5, wt in combos:
        # 官方定义: 每折加权求和 → z → 跨折求和
        score = None
        for f in sorted(perfold['r1'].keys()):
            fs = w1 * perfold['r1'][f] + w5 * perfold['r5'][f] + wt * perfold['top'][f]
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
            'w1': w1, 'w5': w5, 'wtop': wt,
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
    print("\n===== 头部混合网格 [官方外部-z 定义] (按 hold5s8净% 排序) =====")
    print(table.sort_values('hold5s8净%', ascending=False).to_string(index=False))
    table.to_csv(os.path.join(MIX_DIR, 'mix_grid_outer.csv'), index=False)
    print(f"\n[mix] 已存 {MIX_DIR}/mix_grid_outer.csv")


if __name__ == '__main__':
    main()
