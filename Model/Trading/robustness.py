#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""robustness.py — 冠军族稳健性检验

维度: 成本开/关、过滤开/关、本金 1W/3W/5W、卖出口径 open/close_seq、
      窗口三等分 (T1/T2/T3, 按买入日切) 稳定性。
用法: python3 robustness.py
输出: results/robustness.csv
"""
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import SCORE_SETS, load_calendar, load_market, load_scores, run_backtest

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FINALISTS = {
    "A01_daily_top1": dict(top_n=1, hold=1),
    "D01_h5_sl8_t1": dict(top_n=1, hold=5, stop_loss=0.08),
    "N2_re300": dict(top_n=2, hold=None, exit_rank=300, min_hold=2),
    "P2_re300_sl8": dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.08),
    "S2_re300_sl8_tr15": dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
                              stop_loss=0.08, trail_pct=0.15),
    "T2_re300_mh5": dict(top_n=2, hold=None, exit_rank=300, min_hold=5),
    "T2_re300_mh5_sl8_tr15": dict(top_n=2, hold=None, exit_rank=300, min_hold=5,
                                  stop_loss=0.08, trail_pct=0.15),
    "T2_re250_sl8_tr15": dict(top_n=2, hold=None, exit_rank=250, min_hold=5,
                              stop_loss=0.08, trail_pct=0.15),
    "T2_re400_sl8_tr15": dict(top_n=2, hold=None, exit_rank=400, min_hold=5,
                              stop_loss=0.08, trail_pct=0.15),
    "U2_re300_h20": dict(top_n=2, hold=20, exit_rank=300, min_hold=2),
}

VARIANTS = {
    "base": {},
    "no_cost": dict(use_cost=False),
    "no_filters": dict(exclude_st=False, exclude_limit_up=False, sell_limit_filter=False),
    "cap1w": dict(capital_mult=0.2),
    "cap3w": dict(capital_mult=0.6),
    "close_seq": dict(sell_at="close_seq"),
}


def thirds(tr):
    """按买入日把交易三等分, 返回每段复利净收益。"""
    if tr is None or len(tr) == 0:
        return np.nan, np.nan, np.nan
    tr = tr.sort_values("buy_dt")
    k = max(1, len(tr) // 3)
    b1, b2, b3 = tr.iloc[:k], tr.iloc[k:2 * k], tr.iloc[2 * k:]
    out = []
    for part in (b1, b2, b3):
        out.append(float((1 + part["net_pct"] / 100).prod() - 1))
    return tuple(out)


def run(args):
    s, cname, vname = args
    cfg = {**FINALISTS[cname], **VARIANTS[vname], "name": cname}
    sc = load_scores(s)
    m, tr, eq = run_backtest(sc, cfg, _G["mkt"], _G["tds"], _G["tdi"])
    if m is None:
        return None
    t1, t2, t3 = thirds(tr)
    return dict(set=s, cfg=cname, variant=vname, cum=m["cum_trade"], maxdd=m["maxdd"],
                sharpe=m["sharpe"], sharpe_t=m["sharpe_trade"], win=m["win_rate"],
                n=m["n_trades"], avg_hold=m["avg_hold"], cost_drag=m["cost_drag"],
                turnover=m["turnover"], h1=m["h1_cum_t"], h2=m["h2_cum_t"],
                t1=t1, t2=t2, t3=t3, blocked_sell=m["blocked_sell"])


_G = {}


def main():
    import multiprocessing as mp
    mkt = load_market()
    tds, tdi = load_calendar()
    _G["mkt"], _G["tds"], _G["tdi"] = mkt, tds, tdi
    tasks = list(product(SCORE_SETS.keys(), FINALISTS.keys(), VARIANTS.keys()))
    pool = mp.Pool(24)
    rows = [r for r in pool.imap_unordered(run, tasks, chunksize=8) if r]
    pool.close(); pool.join()
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "robustness.csv")
    df.to_csv(out, index=False)
    print(f"完成 {len(df)} 行 → {out}")
    # 摘要: base 口径下各策略跨模型
    g = df[df["variant"] == "base"].groupby("cfg")["cum"].agg(["median", "mean"])
    g["pos"] = df[df["variant"] == "base"].groupby("cfg")["cum"].apply(lambda x: (x > 0).mean())
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
