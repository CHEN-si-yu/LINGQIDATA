#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter1_scores.py — 新迭代第一轮: 分数侧重加权搜索 (V20/V11 头矩阵在内存中组合)

阶段1: 家族内权重组合 (48 a-combos × 12 c-combos) 作为独立打分集,
       在给定协议清单上回测 (真实净值 cum_net, 含再投资);
阶段2: 阶段1 中每族前若干名做跨族集成 (wc ∈ {0.5,1,1.5,2,3}), 同样协议回测。
输出: results/iter1_scores_fam.csv + results/iter1_scores_ens.csv
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

# 协议清单 (真实净值口径下的冠军候选; 来自 iter1_protocol_grid.csv)
PROTOCOLS = {
    "D01_t1h5sl8": dict(top_n=1, hold=5, stop_loss=0.08),
    "hold20_t1": dict(top_n=1, hold=20),
    "hold30sl5_t1": dict(top_n=1, hold=30, stop_loss=0.05),
    "h20tr20_t2": dict(top_n=2, hold=20, trail_pct=0.20),
    "hold20_t2": dict(top_n=2, hold=20),
    "h25sl8_t3": dict(top_n=3, hold=25, stop_loss=0.08),
    "S2_re300": dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
                     stop_loss=0.08, trail_pct=0.15),
    "UU4_re300h20tr12": dict(top_n=2, hold=20, exit_rank=300, min_hold=2,
                             trail_pct=0.12),
    "N2_re300": dict(top_n=2, hold=None, exit_rank=300, min_hold=2),
}

# 家族权重网格 (w1 固定 1, 因 z 缩放不变性)
W3S = (0.0, 0.25, 0.5, 1.0)
W5S = (0.0, 0.25, 0.5, 1.0)
WTS = (0.5, 1.0, 2.0, 3.0)
WC_S = (0.5, 1.0, 1.5, 2.0, 3.0)
N_TOP_FAM = 5   # 阶段2 每族取前几名

_G = {}


def load_heads(base_dir, nf=8):
    """读一个家族的 4×8 头矩阵 → {h: {f: DataFrame}} (float32)."""
    perfold = {h: {} for h in ('r1', 'r3', 'r5', 'top')}
    for h in perfold:
        for f in range(1, nf + 1):
            p = os.path.join(base_dir, f'{h}_f{f}.fea')
            df = pd.read_feather(p).set_index('date').astype(np.float32)
            perfold[h][f] = df
    return perfold


def zn(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


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
    """DataFrame (index=date, cols=code) → engine scores dict."""
    ranked, scores, top1 = {}, {}, {}
    for d in df.index:
        row = df.loc[d].dropna()
        scores[d] = row.to_dict()
        ranked[d] = row.sort_values(ascending=False).index.tolist()
        top1[d] = float(row.max()) if len(row) else np.nan
    return dict(scores=scores, ranked=ranked, top1=top1,
                dates=sorted(scores.keys()))


def build_family_combos(heads_a, heads_c):
    """返回 {名: DataFrame} 的家族打分组合。"""
    out = {}
    for w3, w5, wt in product(W3S, W5S, WTS):
        out[f"A_w3{w3}_w5{w5}_wt{wt}"] = family_score(heads_a, 1.0, w3, w5, wt)
        out[f"C_w3{w3}_w5{w5}_wt{wt}"] = family_score(heads_c, 1.0, w3, w5, wt)
    return out


def run_task(args):
    name, combos, proto_name, proto = args
    sc = df_to_scores(combos[name])
    m, tr, eq = run_backtest(sc, proto, _G["mkt"], _G["tds"], _G["tdi"])
    if m is None:
        return None
    return dict(score=name, proto=proto_name, cum_net=m["cum_net"],
                cum_trade=m["cum_trade"], sharpe=m["sharpe"],
                maxdd=m["maxdd"], h1_cum=m["h1_cum"], h2_cum=m["h2_cum"],
                h1_cum_t=m["h1_cum_t"], h2_cum_t=m["h2_cum_t"],
                win_rate=m["win_rate"], n_trades=m["n_trades"],
                avg_hold=m["avg_hold"], cost_drag=m["cost_drag"])


def main():
    import multiprocessing as mp
    heads_a = load_heads(ROOT + "V20/model_pred/2026q3/heads_a")
    heads_c = load_heads(ROOT + "V20/model_pred/2026q3/heads_c")
    combos = build_family_combos(heads_a, heads_c)
    print(f"[scores] 家族组合 {len(combos)} 个 (A/C 各 {len(combos)//2})")
    # 基线打分 (V11/V20 ens_w2) + V11/V20 跨代集成 一并回测
    v11 = pd.read_feather(ROOT + "V11/model_pred/2026q3/score_ens_w2.fea").set_index("date")
    v20 = pd.read_feather(ROOT + "V20/model_pred/2026q3/score_ens_w2.fea").set_index("date")
    v11z, v20z = zn(v11.astype(np.float64)), zn(v20.astype(np.float64))
    combos["V11_ensw2"] = v11
    combos["V20_ensw2"] = v20
    for w in (0.5, 1.0, 1.5, 2.0):
        combos[f"ENS_v11_{w}v20"] = zn(v11z + w * v20z)

    mkt = load_market()
    tds, tdi = load_calendar()
    _G.update(mkt=mkt, tds=tds, tdi=tdi)
    names = list(combos.keys())
    tasks = [(n, combos, pn, pc) for n in names for pn, pc in PROTOCOLS.items()]
    print(f"[scores] 阶段1: {len(tasks)} 任务", flush=True)
    pool = mp.Pool(20)
    rows = [r for r in pool.imap_unordered(run_task, tasks, chunksize=4) if r]
    pool.close(); pool.join()
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "iter1_scores_fam.csv")
    df.to_csv(out, index=False)
    print(f"[scores] 阶段1 完成 → {out}")
    pd.set_option("display.width", 250)
    for pn in PROTOCOLS:
        sub = df[df["proto"] == pn].sort_values("cum_net", ascending=False).head(8)
        print(f"\n== {pn} Top8 ==")
        print(sub[["score", "cum_net", "cum_trade", "sharpe", "maxdd",
                   "h1_cum", "h2_cum", "n_trades"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
