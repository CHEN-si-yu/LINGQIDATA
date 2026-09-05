#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter6_combos2.py — 组合权重扫描 + 三系统组合 (稳定性优先)

在 iter5 基础上: 对最优组合对做 capital_split 权重网格 (0.3~0.7),
并测试三系统各 1/3 组合。指标同 iter5 (Sharpe/MaxDD/min(H1,H2)/三等分/滚动正率)。
输出: results/iter6_combos2.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import load_calendar, load_market, run_backtest, load_scores

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PROTO = {
    "h20tr20_t2": dict(top_n=2, hold=20, trail_pct=0.20),
    "D01": dict(top_n=1, hold=5, stop_loss=0.08),
    "hold20_t1": dict(top_n=1, hold=20),
    "S2": dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
               stop_loss=0.08, trail_pct=0.15),
}
PAIRS = [
    ("V20_ensw2", "h20tr20_t2", "V11_ensw2", "D01"),
    ("V20_ensw2", "h20tr20_t2", "V11_ensw2", "hold20_t1"),
    ("V20_ensw2", "h20tr20_t2", "V23_ens23", "S2"),
    ("V20_ensw2", "h20tr20_t2", "V22_20d", "S2"),
]
TRIPLES = [
    ("V20_ensw2", "h20tr20_t2", "V11_ensw2", "D01", "V22_20d", "S2"),
    ("V20_ensw2", "h20tr20_t2", "V11_ensw2", "D01", "V23_ens23", "S2"),
]


def thirds(tr, slots):
    tr = tr.sort_values("buy_dt")
    k = max(1, len(tr) // 3)
    return [float((1 + p["net_pct"] / 100 / slots).prod() - 1)
            for p in (tr.iloc[:k], tr.iloc[k:2 * k], tr.iloc[2 * k:])]


def metrics_of(eq, trades, slots):
    r = eq["ret"].dropna()
    n = len(r)
    cum = float((1 + r).prod() - 1)
    sh = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq["equity"].values)
    dd = float((eq["equity"].values / peak - 1).min())
    h1 = float((1 + r.iloc[:n // 2]).prod() - 1)
    h2 = float((1 + r.iloc[n // 2:]).prod() - 1)
    t1, t2, t3 = thirds(trades, slots)
    pos = sum(1 for i in range(60, len(r) + 1)
              if (1 + r.iloc[i - 60:i]).prod() - 1 > 0)
    return dict(cum=cum, sharpe=sh, maxdd=dd, h1=h1, h2=h2, t1=t1, t2=t2, t3=t3,
                min_h=min(h1, h2), min_t=min(t1, t2, t3), win=(trades["net_pct"] > 0).mean()
                if len(trades) else np.nan, n=len(trades), roll_pos=pos,
                roll_n=len(r) - 59)


def main():
    market = load_market()
    tds, tdi = load_calendar()
    scores = {s: load_scores(s) for s in
              ["V20_ensw2", "V11_ensw2", "V22_20d", "V23_ens23"]}
    rows = []

    def run_one(sname, pname, mult):
        m, tr, eq = run_backtest(scores[sname], {**PROTO[pname],
                                                 "capital_mult": mult},
                                 market, tds, tdi)
        return m, tr, eq

    for (sa, pa, sb, pb) in PAIRS:
        for wa in (0.3, 0.4, 0.5, 0.6, 0.7):
            ma, tra, eqa = run_one(sa, pa, wa)
            mb, trb, eqb = run_one(sb, pb, 1 - wa)
            eqa, eqb = eqa.set_index("date"), eqb.set_index("date")
            j = eqa.index.intersection(eqb.index)
            comb = (eqa.loc[j, "equity"] + eqb.loc[j, "equity"]).to_frame()
            comb.columns = ["equity"]
            comb["ret"] = comb["equity"].pct_change()
            comb_tr = pd.concat([tra, trb], ignore_index=True)
            rows.append(dict(system=f"{wa:.1f}×[{sa}+{pa}] + {1-wa:.1f}×[{sb}+{pb}]",
                             **metrics_of(comb, comb_tr, 2)))
    for (sa, pa, sb, pb, sc2, pc2) in TRIPLES:
        ma, tra, eqa = run_one(sa, pa, 1 / 3)
        mb, trb, eqb = run_one(sb, pb, 1 / 3)
        mc, trc, eqc = run_one(sc2, pc2, 1 / 3)
        eqa, eqb, eqc = eqa.set_index("date"), eqb.set_index("date"), \
            eqc.set_index("date")
        j = eqa.index.intersection(eqb.index).intersection(eqc.index)
        comb = (eqa.loc[j, "equity"] + eqb.loc[j, "equity"]
                + eqc.loc[j, "equity"]).to_frame()
        comb.columns = ["equity"]
        comb["ret"] = comb["equity"].pct_change()
        comb_tr = pd.concat([tra, trb, trc], ignore_index=True)
        rows.append(dict(system=f"1/3×[{sa}+{pa}]×[{sb}+{pb}]×[{sc2}+{pc2}]",
                         **metrics_of(comb, comb_tr, 2)))
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "iter6_combos2.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 260)
    print(df[["system", "cum", "sharpe", "maxdd", "h1", "h2", "min_h", "t1", "t2",
              "t3", "min_t", "win", "n", "roll_pos"]].round(3)
          .to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
