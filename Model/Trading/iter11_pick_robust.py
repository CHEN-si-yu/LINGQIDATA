#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter11_pick_robust.py — 稳健选型收口 (用户需求: 1~3 只 + 低回撤 + 止盈止损 + 好模型)

回合 2 网格 (iter10 为回合 1, 26 配置 × 高IC 12 集):
  混合体 u_h20_re300_tr15 = Top2 + 持有≤20日 + 排名掉出300退出(min_hold 2)
                           + 移动止盈 15% (无固定止损) — 三口径验证:
  [高IC12]  n_pos 11/12, H1H2 8/12, med +38.0%, med dd -22.0%
  [种子6]   6/6 全正 (V20/V20b/V20c/V13/V13b/V13c), med +36.0%
  [全24]    n_pos 22/24, med +37.6%, med dd -21.9%

3 只双结构 (0.5/0.5): [混合体 on 好模型] + [V11_ensw2+D01 短腿]:
  全部组合 (含非幸运种子腿) cum +128~159%, Sharpe 2.46~2.91,
  MaxDD -13.8~-15.9%, 滚动60日正率 90~100%。

输出: results/iter11_pick_robust.csv (混合体 × 6 种子 + 5 组合 + 最差个案)
用法: python3 iter11_pick_robust.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import SCORE_SETS, load_calendar, load_market, run_backtest, \
    load_scores  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for k, p in [("V18_f1", "Model/V18/model_pred/v18_f1.fea"),
             ("V18_f2", "Model/V18/model_pred/v18_f2.fea"),
             ("V19_meta", "Model/V19/model_pred/2026q3/score_meta.fea"),
             ("V20b_ensw2", "Model/V20b/model_pred/2026q3/score_ens_w2.fea"),
             ("V20c_ensw2", "Model/V20c/model_pred/2026q3/score_ens_w2.fea"),
             ("V13b_allz", "Model/V13b/model_pred/2026q3/all_zscore_score.fea"),
             ("V13c_allz", "Model/V13c/model_pred/2026q3/all_zscore_score.fea")]:
    SCORE_SETS[k] = p

SEED = ["V20_ensw2", "V20b_ensw2", "V20c_ensw2",
        "V13_allz", "V13b_allz", "V13c_allz"]
HYB = dict(top_n=2, hold=20, exit_rank=300, min_hold=2, trail_pct=0.15)
D01 = dict(top_n=1, hold=5, stop_loss=0.08)


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
                n=len(trades), roll_pct=pos / max(1, len(r) - 59))


def main():
    market = load_market()
    tds, tdi = load_calendar()
    rows = []
    for u in SEED:
        sc = load_scores(u)
        m, tr, eq = run_backtest(sc, HYB, market, tds, tdi)
        rows.append(dict(kind="hybrid_seed", leg=u, **metrics_of(eq, tr, 2)))
    for x in ["V20_ensw2", "V20b_ensw2", "V20c_ensw2", "V13b_allz",
              "V13c_allz"]:
        ma, tra, eqa = run_backtest(load_scores(x), {**HYB,
                                                     "capital_mult": 0.5},
                                    market, tds, tdi)
        mb, trb, eqb = run_backtest(load_scores("V11_ensw2"),
                                    {**D01, "capital_mult": 0.5},
                                    market, tds, tdi)
        eqa, eqb = eqa.set_index("date"), eqb.set_index("date")
        j = eqa.index.intersection(eqb.index)
        comb = (eqa.loc[j, "equity"] + eqb.loc[j, "equity"]).to_frame()
        comb.columns = ["equity"]
        comb["ret"] = comb["equity"].pct_change()
        tr = pd.concat([tra, trb], ignore_index=True)
        rows.append(dict(kind="pair_3names", leg=f"0.5×[{x}+HYB]+0.5×[V11+D01]",
                         **metrics_of(comb, tr, 2)))
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "iter11_pick_robust.csv")
    df.to_csv(out, index=False)
    print(df[["kind", "leg", "cum", "sharpe", "maxdd", "h1", "h2",
              "roll_pct"]].round(3).to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
