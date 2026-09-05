#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter2_v18.py — V18 LightGBM (修复后重训) 在新协议族下的评估 + 与冠军打分集成

内容:
  1) v18_f1/f2/f3/f4/v18 在 h20tr20_t2 / hold20_t2 / hold20_t1 / D01 / S2 上的真实净值
  2) ens 组合: z(V20_ensw2) + w·z(v18) / z(V11_ensw2) + w·z(v18)  (w ∈ {0.25,0.5,1,2})
  3) v18 与 ens_w2 的逐日相关性 (正交性检查)
输出: results/iter2_v18.csv
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
    "S2_re300": dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
                     stop_loss=0.08, trail_pct=0.15),
}

_G = {}


def zn(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def df_to_scores(df):
    ranked, scores, top1 = {}, {}, {}
    for d in df.index:
        row = df.loc[d].dropna()
        scores[d] = row.to_dict()
        ranked[d] = row.sort_values(ascending=False).index.tolist()
        top1[d] = float(row.max()) if len(row) else np.nan
    return dict(scores=scores, ranked=ranked, top1=top1, dates=sorted(scores.keys()))


def run_task(args):
    name, scores, pn, proto = args
    m, tr, eq = run_backtest(scores, proto, _G["mkt"], _G["tds"], _G["tdi"])
    if m is None:
        return None
    return dict(score=name, proto=pn, cum_net=m["cum_net"], cum_trade=m["cum_trade"],
                sharpe=m["sharpe"], maxdd=m["maxdd"], h1=m["h1_cum"], h2=m["h2_cum"],
                win=m["win_rate"], n=m["n_trades"], hold=m["avg_hold"],
                cost=m["cost_drag"])


def main():
    import multiprocessing as mp
    sets = {}
    for n in ("v18_f1", "v18_f2", "v18_f3", "v18_f4", "v18"):
        sets[n] = pd.read_feather(ROOT + f"V18/model_pred/{n}.fea").set_index("date")
    v20 = pd.read_feather(ROOT + "V20/model_pred/2026q3/score_ens_w2.fea").set_index("date")
    v11 = pd.read_feather(ROOT + "V11/model_pred/2026q3/score_ens_w2.fea").set_index("date")
    v20z, v11z, v18z = zn(v20.astype(np.float64)), zn(v11.astype(np.float64)), \
        zn(sets["v18"].astype(np.float64))
    sets["V20_ensw2"], sets["V11_ensw2"] = v20, v11
    for w in (0.25, 0.5, 1.0, 2.0):
        sets[f"V20+w{w}v18"] = zn(v20z + w * v18z)
        sets[f"V11+w{w}v18"] = zn(v11z + w * v18z)
    # 正交性检查
    common = v20z.index.intersection(v18z.index)
    corrs = [v20z.loc[d].corr(v18z.loc[d]) for d in common if v18z.loc[d].notna().sum() > 50]
    print(f"[v18] V20_ensw2 × v18 逐日相关: mean {np.mean(corrs):+.4f}  "
          f"min {np.min(corrs):+.4f}  max {np.max(corrs):+.4f}  (n={len(corrs)})")
    scores_dicts = {n: df_to_scores(df) for n, df in sets.items()}
    mkt = load_market()
    tds, tdi = load_calendar()
    _G.update(mkt=mkt, tds=tds, tdi=tdi)
    tasks = [(n, scores_dicts[n], pn, pc) for n in sets for pn, pc in PROTOCOLS.items()]
    print(f"[v18] {len(tasks)} 任务", flush=True)
    pool = mp.Pool(20)
    rows = [r for r in pool.imap_unordered(run_task, tasks, chunksize=2) if r]
    pool.close(); pool.join()
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "iter2_v18.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    for pn in PROTOCOLS:
        sub = df[df["proto"] == pn].sort_values("cum_net", ascending=False).head(10)
        print(f"\n== {pn} Top10 ==")
        print(sub[["score", "cum_net", "cum_trade", "sharpe", "maxdd", "h1", "h2",
                   "n"]].round(3).to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
