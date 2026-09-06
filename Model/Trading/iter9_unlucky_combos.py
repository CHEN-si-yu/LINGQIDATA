#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter9_unlucky_combos.py — 无幸运种子时, 持仓组合策略的稳健性 (任务: 种子彩票免疫性)

背景问题: 单模型 Top 选股几乎完全由训练种子决定 (Top1 Jaccard ≤0.11, 成交近零
重叠) — 在"只能持仓 1~3 只"的约束下, 单一模型源 = 一张种子彩票。能否通过
**持仓构造**把"赢或输"的彩票变成"赢多赢少"的彩票?

方法: 用全部**已检验的非幸运种子单元** (V20b/V20c/V13b/V13c, 均已证 h20 型
协议下 +33.7~+43.7% 收敛) 替换 iter5/6 组合里的幸运腿 V20 (+352%), 重跑
双腿 0.5/0.5 组合 (腿 A = Top2 长持 ≤2 只, 腿 B = Top1 短持 1 只, 合计 ≤3 只),
对照旧冠军对 [V20+h20]×[V11+D01] (+301% / Sharpe 3.75)。

输出: results/iter9_unlucky_combos.csv
用法: python3 iter9_unlucky_combos.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import SCORE_SETS, load_calendar, load_market, run_backtest, \
    load_scores  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SCORE_SETS["V20b_ensw2"] = "Model/V20b/model_pred/2026q3/score_ens_w2.fea"
SCORE_SETS["V20c_ensw2"] = "Model/V20c/model_pred/2026q3/score_ens_w2.fea"
SCORE_SETS["V13b_allz"] = "Model/V13b/model_pred/2026q3/all_zscore_score.fea"
SCORE_SETS["V13c_allz"] = "Model/V13c/model_pred/2026q3/all_zscore_score.fea"

PROTO = {   # 持仓只数: h20/S2 = Top2 (2只/腿), D01 = Top1 (1只/腿)
    "h20tr20_t2": dict(top_n=2, hold=20, trail_pct=0.20),
    "S2":         dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
                       stop_loss=0.08, trail_pct=0.15),
    "D01":        dict(top_n=1, hold=5, stop_loss=0.08),
}
UNLUCKY = ["V20b_ensw2", "V20c_ensw2", "V13b_allz", "V13c_allz"]


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
    return dict(cum=cum, sharpe=sh, maxdd=dd, h1=h1, h2=h2,
                min_h=min(h1, h2), min_t=min(t1, t2, t3),
                n=len(trades), roll_pos=pos, roll_n=len(r) - 59)


def main():
    market = load_market()
    tds, tdi = load_calendar()
    S = {s: load_scores(s) for s in
         ["V20_ensw2", "V11_ensw2"] + UNLUCKY}
    rows = []

    def leg(sname, pname, mult):
        m, tr, eq = run_backtest(S[sname], {**PROTO[pname],
                                            "capital_mult": mult},
                                 market, tds, tdi)
        return tr, eq

    # ---- 单腿参照 ----
    for s in ["V20_ensw2"] + UNLUCKY + ["V11_ensw2"]:
        for p in (["h20tr20_t2"] if s != "V11_ensw2"
                  else ["h20tr20_t2", "S2", "D01"]):
            tr, eq = leg(s, p, 1.0)
            rows.append(dict(system=f"1×[{s}+{p}]",
                             npos="2只" if p != "D01" else "1只",
                             **metrics_of(eq, tr, 2 if p != "D01" else 1)))

    # ---- 双腿组合 0.5/0.5 (≤3 只: 腿A Top2 + 腿B Top1) ----
    PAIRS = [("V20_ensw2", "h20tr20_t2", "V11_ensw2", "D01")]  # 幸运参照
    for s in UNLUCKY:
        PAIRS.append((s, "h20tr20_t2", "V11_ensw2", "D01"))
    PAIRS += [("V20b_ensw2", "S2", "V11_ensw2", "D01"),
              ("V20b_ensw2", "h20tr20_t2", "V13b_allz", "h20tr20_t2")]

    def pair(sa, pa, sb, pb):
        tra, eqa = leg(sa, pa, 0.5)
        trb, eqb = leg(sb, pb, 0.5)
        eqa, eqb = eqa.set_index("date"), eqb.set_index("date")
        j = eqa.index.intersection(eqb.index)
        comb = (eqa.loc[j, "equity"] + eqb.loc[j, "equity"]).to_frame()
        comb.columns = ["equity"]
        comb["ret"] = comb["equity"].pct_change()
        return pd.concat([tra, trb], ignore_index=True), comb

    for (sa, pa, sb, pb) in PAIRS:
        tr, eq = pair(sa, pa, sb, pb)
        rows.append(dict(system=f"0.5×[{sa}+{pa}] + 0.5×[{sb}+{pb}]",
                         npos="3只", **metrics_of(eq, tr, 2)))

    df = pd.DataFrame(rows)
    df["roll_pct"] = df["roll_pos"] / df["roll_n"]
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "iter9_unlucky_combos.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df[["system", "npos", "cum", "sharpe", "maxdd", "min_h", "min_t",
              "roll_pct"]].round(3).to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
