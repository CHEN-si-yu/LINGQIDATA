#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_v11e.py — V15 联合双头 (top1d/top3d) 评估 + 与冠军集成

打分 = Σ_f z( w1·zr1 + w5·zr5 + w3·zr3 + w1d·ztop1d + w3d·ztop3d )
策略: hold5s8 / norep3 / norep4, 分半 + 与 ens_w2 集成
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'V11'))
from strat_backtest import run_backtest, load_prices, load_calendar, eval_ic

HERE = os.path.dirname(os.path.abspath(__file__))
V11E_HEADS = os.path.join(HERE, 'model_pred/2026q3/heads')
ENS_W2 = os.path.join(HERE, '..', 'V11/model_pred/2026q3/score_ens_w2.fea')


def load_heads(names=('r1', 'r5', 'r3', 'top1d', 'top3d'), n_folds=8):
    perfold = {h: {} for h in names}
    for h in names:
        for f in range(1, n_folds + 1):
            p = os.path.join(V11E_HEADS, f'{h}_f{f}.fea')
            if os.path.exists(p):
                perfold[h][f] = pd.read_feather(p).set_index('date')
    return perfold


def build_mix(perfold, w1, w5, w3, w1d, w3d):
    score = None
    for f in sorted(perfold['r1'].keys()):
        fs = w1 * perfold['r1'][f]
        if w5 and 'r5' in perfold:
            fs = fs + w5 * perfold['r5'][f]
        if w3 and 'r3' in perfold:
            fs = fs + w3 * perfold['r3'][f]
        if w1d and 'top1d' in perfold:
            fs = fs + w1d * perfold['top1d'][f]
        if w3d and 'top3d' in perfold:
            fs = fs + w3d * perfold['top3d'][f]
        fs = (fs - fs.mean(axis=1).values[:, None]) / fs.std(axis=1).values[:, None]
        score = fs if score is None else score.add(fs, fill_value=0.0)
    return score


def zn(df):
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]


def main():
    heads = load_heads()
    print(f"[eval] V15 heads: { {h: len(v) for h, v in heads.items()} }")
    open_map, close_map, prev_close_map, amount_map, name_map = load_prices()
    tds = load_calendar()
    F = dict(use_cost=True, exclude_st=True, exclude_limit_up=True, buy_gap_limit=0.095)
    strategies = {
        'hold5s8': dict(**F, hold=5, stop_loss=0.08),
        'norep3': dict(**F, no_repeat_skip=True, hold=3),
    }
    rows = []
    for w5 in [0.0, 0.25]:
        for w3 in [0.0, 0.25, 0.5]:
            for w1d in [0.5, 1.0]:
                for w3d in [1.0, 2.0]:
                    score = build_mix(heads, 1.0, w5, w3, w1d, w3d)
                    ic = eval_ic(score)
                    row = dict(w5=w5, w3=w3, w1d=w1d, w3d=w3d,
                               ic=round(ic['RankIC'], 4), ir=round(ic['RankICIR'], 4))
                    for stn, cfg in strategies.items():
                        m, _, _ = run_backtest(score, open_map, close_map,
                                               prev_close_map, name_map,
                                               amount_map=amount_map, tds=tds, **cfg)
                        h1, _, _ = run_backtest(score, open_map, close_map,
                                                prev_close_map, name_map,
                                                amount_map=amount_map, tds=tds,
                                                window_start='20250901',
                                                window_end='20260227', **cfg)
                        h2, _, _ = run_backtest(score, open_map, close_map,
                                                prev_close_map, name_map,
                                                amount_map=amount_map, tds=tds,
                                                window_start='20260302',
                                                window_end='20260901', **cfg)
                        n = m['net']
                        row[f'{stn}净'] = round(n['cum'] * 100, 2)
                        row[f'{stn}DD'] = round(n['maxdd'] * 100, 2)
                        row[f'{stn}H1'] = round(h1['net']['cum'] * 100, 2)
                        row[f'{stn}H2'] = round(h2['net']['cum'] * 100, 2)
                    rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(HERE, 'v11e_grid.csv'), index=False)
    print('\n===== V15 网格 (按 hold5s8净 排序 TOP12) =====')
    print(t.sort_values('hold5s8净', ascending=False).head(12).to_string(index=False))

    best = t.sort_values('hold5s8净', ascending=False).iloc[0]
    v11e_best = build_mix(heads, 1.0, best['w5'], best['w3'], best['w1d'], best['w3d'])
    ens_w2 = pd.read_feather(ENS_W2).set_index('date')
    print(f'\n===== V15 最优 ({best["w5"]},{best["w3"]},{best["w1d"]},{best["w3d"]}) 与冠军集成 =====')
    print(f'{"集成":<28}{"hold5s8全期%":>14}{"DD%":>8} | {"H1%":>8}{"H2%":>8}')
    cfg = strategies['hold5s8']
    for name, sc in [('ens_w2 (冠军)', zn(ens_w2)),
                     ('v11e_best', zn(v11e_best)),
                     ('ens_w2 + 0.5*v11e', zn(ens_w2).add(0.5 * zn(v11e_best), fill_value=0.0)),
                     ('ens_w2 + 1.0*v11e', zn(ens_w2).add(1.0 * zn(v11e_best), fill_value=0.0)),
                     ('ens_w2 + 1.5*v11e', zn(ens_w2).add(1.5 * zn(v11e_best), fill_value=0.0)),
                     ('ens_w2 + 2.0*v11e', zn(ens_w2).add(2.0 * zn(v11e_best), fill_value=0.0))]:
        m, _, _ = run_backtest(sc, open_map, close_map, prev_close_map, name_map,
                               amount_map=amount_map, tds=tds, **cfg)
        h1, _, _ = run_backtest(sc, open_map, close_map, prev_close_map, name_map,
                                amount_map=amount_map, tds=tds,
                                window_start='20250901', window_end='20260227', **cfg)
        h2, _, _ = run_backtest(sc, open_map, close_map, prev_close_map, name_map,
                                amount_map=amount_map, tds=tds,
                                window_start='20260302', window_end='20260901', **cfg)
        n = m['net']
        print(f'{name:<28}{n["cum"] * 100:>14.2f}{n["maxdd"] * 100:>8.2f} | '
              f'{h1["net"]["cum"] * 100:>8.2f}{h2["net"]["cum"] * 100:>8.2f}')


if __name__ == '__main__':
    main()
