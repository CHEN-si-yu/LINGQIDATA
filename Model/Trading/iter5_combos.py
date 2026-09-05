#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter5_combos.py — 双系统资本对半组合 (模型×策略组合层, 稳定性优先)

思路 (DESIGN_V24 §四): 不同协议喜欢不同模型 → 两套 5W 系统各半资本并行,
净值按日相加 = 组合账户。评估稳定性优先指标:
  cum / sharpe / maxdd / min(H1,H2) / 三等分最小值 / 滚动60日正率 / 正窗口数。
输出: results/iter5_combos.csv
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
    "hold20_t2": dict(top_n=2, hold=20),
}
# 单系统基线: (score, proto)
SYSTEMS = [
    ("V20_ensw2", "h20tr20_t2"),
    ("V11_ensw2", "D01"),
    ("V11_ensw2", "hold20_t1"),
    ("V20_ensw2", "D01"),
    ("V22_20d", "S2"),
    ("V23_ens23", "S2"),
    ("V20_ensw2", "S2"),
]
PAIRS = [
    (("V20_ensw2", "h20tr20_t2"), ("V22_20d", "S2")),
    (("V20_ensw2", "h20tr20_t2"), ("V23_ens23", "S2")),
    (("V20_ensw2", "h20tr20_t2"), ("V11_ensw2", "D01")),
    (("V20_ensw2", "h20tr20_t2"), ("V11_ensw2", "hold20_t1")),
    (("V20_ensw2", "h20tr20_t2"), ("V20_ensw2", "D01")),
    (("V20_ensw2", "h20tr20_t2"), ("V20_ensw2", "S2")),
]

_G = {}


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
    # 滚动 60 交易日正率 (基于组合净值日收益)
    pos = 0
    for i in range(60, len(r) + 1):
        if (1 + r.iloc[i - 60:i]).prod() - 1 > 0:
            pos += 1
    return dict(cum=cum, sharpe=sh, maxdd=dd, h1=h1, h2=h2, t1=t1, t2=t2, t3=t3,
                win=(trades["net_pct"] > 0).mean() if len(trades) else np.nan,
                n=len(trades), roll_pos=pos, roll_n=len(r) - 59)


def main():
    market = load_market()
    tds, tdi = load_calendar()
    scores = {}
    for s in ["V20_ensw2", "V11_ensw2", "V22_20d", "V23_ens23"]:
        try:
            scores[s] = load_scores(s)
        except Exception:
            pass
    # V22_20d / V23 需要手工注册
    import importlib
    from engine import SCORE_SETS
    for s in list(SCORE_SETS.keys()):
        if s.startswith("V22") or s.startswith("V23"):
            try:
                scores[s] = load_scores(s)
            except Exception:
                pass
    rows = []

    def run_one(sname, pname):
        m, tr, eq = run_backtest(scores[sname], {**PROTO[pname],
                                                 "capital_mult": 0.5},
                                 market, tds, tdi)
        return m, tr, eq

    # 单系统基线 (全资本)
    for sname, pname in SYSTEMS:
        m, tr, eq = run_backtest(scores[sname], PROTO[pname], market, tds, tdi)
        if m is None:
            continue
        rows.append(dict(system=f"{sname}+{pname}", kind="single",
                         **metrics_of(eq, tr, PROTO[pname]["top_n"])))
    # 组合
    for (sa, pa), (sb, pb) in PAIRS:
        ma, tra, eqa = run_one(sa, pa)
        mb, trb, eqb = run_one(sb, pb)
        if ma is None or mb is None:
            continue
        eqa = eqa.set_index("date")
        eqb = eqb.set_index("date")
        j = eqa.index.intersection(eqb.index)
        combined = (eqa.loc[j, "equity"] + eqb.loc[j, "equity"]).to_frame()
        combined.columns = ["equity"]
        combined["ret"] = combined["equity"].pct_change()
        comb_tr = pd.concat([tra, trb], ignore_index=True)
        rows.append(dict(system=f"[{sa}+{pa}]×[{sb}+{pb}]", kind="combo",
                         **metrics_of(combined, comb_tr, 2)))
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "iter5_combos.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 260)
    print(df[["system", "kind", "cum", "sharpe", "maxdd", "h1", "h2", "t1", "t2",
              "t3", "win", "n", "roll_pos"]].round(3).to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
