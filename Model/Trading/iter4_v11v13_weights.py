#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter4_v11v13_weights.py — V11/V13 原始头部在新协议族下的权重重搜索

目的: iter1_scores 在 V20 自训头部上搜索未超冠军; 但 V11_ensw2 (原始冠军头部)
在 D01 上 +261% vs V20 +197% — 头部存在差异。本脚本在 **V11/V13 原始头矩阵**
上重搜家族权重与跨族权重, 目标协议 h20tr20_t2 / hold20_t2 / hold20_t1 / D01。
输出: results/iter4_v11v13_weights.csv
"""
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import load_calendar, load_market, run_backtest

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ROOT = "/autodl-fs/data/lingqiData/Model/"
PROTOCOLS = {
    "h20tr20_t2": dict(top_n=2, hold=20, trail_pct=0.20),
    "hold20_t2": dict(top_n=2, hold=20),
    "hold20_t1": dict(top_n=1, hold=20),
    "D01_t1h5sl8": dict(top_n=1, hold=5, stop_loss=0.08),
}

_G = {}


def zn(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def load_heads(base_dir, nf=8):
    perfold = {h: {} for h in ('r1', 'r3', 'r5', 'top')}
    for h in perfold:
        for f in range(1, nf + 1):
            p = os.path.join(base_dir, f'{h}_f{f}.fea')
            df = pd.read_feather(p).set_index('date').astype(np.float32)
            perfold[h][f] = df
    return perfold


def family_score(perfold, w1, w3, w5, wt, nf=8):
    score = None
    for f in range(1, nf + 1):
        fs = w1 * perfold['r1'][f]
        if w5:
            fs = fs + w5 * perfold['r5'][f]
        if w3:
            fs = fs + w3 * perfold['r3'][f]
        fs = fs + wt * perfold['top'][f]
        fs = zn(fs)
        score = fs if score is None else score.add(fs, fill_value=0.0)
    return zn(score)


def df_to_scores(df):
    return dict(
        scores={d: df.loc[d].dropna().to_dict() for d in df.index},
        ranked={d: df.loc[d].dropna().sort_values(ascending=False).index.tolist()
                for d in df.index},
        top1={d: float(df.loc[d].max()) for d in df.index},
        dates=sorted(df.index))


def _ws_run(args):
    name, df = args
    sc = df_to_scores(df)
    m, tr, eq = run_backtest(sc, _G['proto'], _G['mkt'], _G['tds'], _G['tdi'])
    if m is None:
        return None
    return dict(name=name, cum_net=m['cum_net'], sharpe=m['sharpe'],
                maxdd=m['maxdd'], h1=m['h1_cum'], h2=m['h2_cum'],
                win=m['win_rate'], n=m['n_trades'])


def main():
    import multiprocessing as mp
    ha = load_heads(ROOT + "V11/model_pred/2026q3/heads")
    hc = load_heads(ROOT + "V13/model_pred/2026q3/heads")
    combos = {}
    # V11 族 (多目标 1d主): w1=1 固定
    for w3, w5, wt in product((0.0, 0.25, 0.5, 1.0), (0.0, 0.25, 0.5),
                              (0.5, 1.0, 2.0, 3.0)):
        combos[f"A_w3{w3}_w5{w5}_wt{wt}"] = family_score(ha, 1.0, w3, w5, wt)
    # V13 族 (顶部3d): 主权重可变
    for w1, w3, w5, wt in product((0.25, 0.5, 1.0), (0.0, 0.25, 0.5),
                                  (0.0, 0.25), (0.5, 1.0, 2.0)):
        combos[f"C_w1{w1}_w3{w3}_w5{w5}_wt{wt}"] = family_score(hc, w1, w3, w5, wt)
    v11ens = pd.read_feather(ROOT + "V11/model_pred/2026q3/score_ens_w2.fea").set_index("date")
    combos["V11_ensw2"] = v11ens
    print(f"[search] {len(combos)} 家族组合", flush=True)

    mkt = load_market()
    tds, tdi = load_calendar()
    _G.update(mkt=mkt, tds=tds, tdi=tdi)
    all_rows = []
    for pn, pc in PROTOCOLS.items():
        _G['proto'] = pc
        tasks = [(n, c) for n, c in combos.items()]
        pool = mp.Pool(20)
        rows = [r for r in pool.imap_unordered(_ws_run, tasks, chunksize=2) if r]
        pool.close(); pool.join()
        fam = pd.DataFrame(rows)
        fam['proto'] = pn
        all_rows.append(fam)
        # 每族 top-5 → 跨族集成
        a_top = fam[fam['name'].str.startswith('A')].sort_values(
            'cum_net', ascending=False).head(5)['name'].tolist()
        c_top = fam[fam['name'].str.startswith('C')].sort_values(
            'cum_net', ascending=False).head(5)['name'].tolist()
        ens_tasks = []
        for na in a_top:
            for nc in c_top:
                da, dc = combos[na], combos[nc]
                for wc in (0.5, 1.0, 1.5, 2.0, 3.0):
                    ens_tasks.append((f"{na}+{wc}x{nc}",
                                      zn(da).add(wc * zn(dc), fill_value=0.0)))
        pool = mp.Pool(20)
        ens_rows = [r for r in pool.imap_unordered(_ws_run, ens_tasks, chunksize=2) if r]
        pool.close(); pool.join()
        ens = pd.DataFrame(ens_rows)
        ens['proto'] = pn
        all_rows.append(ens)
        print(f"\n== {pn} ens Top8 ==")
        print(ens.sort_values('cum_net', ascending=False).head(8)
              [['name', 'cum_net', 'sharpe', 'maxdd', 'h1', 'h2', 'n']]
              .round(3).to_string(index=False))
    df = pd.concat(all_rows, ignore_index=True)
    out = os.path.join(RESULTS, "iter4_v11v13_weights.csv")
    df.to_csv(out, index=False)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
