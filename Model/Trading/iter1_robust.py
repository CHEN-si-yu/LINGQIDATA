#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter1_robust.py — 冠军候选的稳健性层评估 (真实净值口径)

对 (score, protocol) 候选做:
  - 基准 (含成本+过滤, 再投资净值)
  - 去最大 1/2 笔 (逐笔按槽权重重建净值近似)
  - 三等分 T1/T2/T3 (按买入日, 槽权重复利)
  - 无成本 / 1W 资金 / 3W 资金 / 无过滤
  - H1/H2 净值分半 + bootstrap Sharpe 95%CI (P(Sharpe<0))
输出: results/iter1_robust.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import load_market, load_scores, run_backtest

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FINALISTS = [
    # (score, name, cfg)
    ("V20_ensw2", "h20tr20_t2", dict(top_n=2, hold=20, trail_pct=0.20)),
    ("V20_ensw2", "hold20_t2", dict(top_n=2, hold=20)),
    ("V20_ensw2", "hold20_t1", dict(top_n=1, hold=20)),
    ("V20_ensw2", "h25sl8_t3", dict(top_n=3, hold=25, stop_loss=0.08)),
    ("V20_ensw2", "D01_h5sl8", dict(top_n=1, hold=5, stop_loss=0.08)),
    ("V11_ensw2", "hold20_t1", dict(top_n=1, hold=20)),
    ("V11_ensw2", "hold30sl5_t1", dict(top_n=1, hold=30, stop_loss=0.05)),
    ("V11_ensw2", "D01_h5sl8", dict(top_n=1, hold=5, stop_loss=0.08)),
]


def rebuild_cum(tr, n_slots, drop=0):
    """按槽权重逐笔复利 (近似重建净值; 引擎真实值略低于此因再平衡摩擦)。"""
    t = tr
    if drop:
        t = t.sort_values("net_pct", ascending=False).iloc[drop:]
    t = t.sort_values("buy_dt")
    return float((1 + t["net_pct"] / 100 / n_slots).prod() - 1)


def thirds(tr, n_slots):
    tr = tr.sort_values("buy_dt")
    k = max(1, len(tr) // 3)
    out = []
    for part in (tr.iloc[:k], tr.iloc[k:2 * k], tr.iloc[2 * k:]):
        out.append(float((1 + part["net_pct"] / 100 / n_slots).prod() - 1))
    return tuple(out)


def bootstrap_sharpe(r, n_boot=2000, seed=3253):
    rng = np.random.default_rng(seed)
    r = r.dropna().values
    sh = []
    for _ in range(n_boot):
        s = rng.choice(r, size=len(r), replace=True)
        sh.append(s.mean() / s.std() * np.sqrt(252) if s.std() > 1e-12 else 0.0)
    sh = np.array(sh)
    return float(np.quantile(sh, 0.025)), float(np.quantile(sh, 0.975)), \
        float((sh < 0).mean())


def main():
    mkt = load_market()
    rows = []
    for s, name, cfg in FINALISTS:
        sc = load_scores(s)
        m, tr, eq = run_backtest(sc, cfg, mkt)
        if m is None:
            continue
        slots = cfg["top_n"]
        lo, hi, pneg = bootstrap_sharpe(eq["ret"])
        t1, t2, t3 = thirds(tr, slots)
        rows.append(dict(score=s, cfg=name, variant="base", cum=m["cum_net"],
                         cum_trade=m["cum_trade"], sharpe=m["sharpe"],
                         sharpe_lo=lo, sharpe_hi=hi, p_sharpe_neg=pneg,
                         maxdd=m["maxdd"], h1=m["h1_cum"], h2=m["h2_cum"],
                         t1=t1, t2=t2, t3=t3, n=m["n_trades"],
                         win=m["win_rate"], cost_drag=m["cost_drag"],
                         avg_cash=m["avg_cash_ratio"],
                         drop1=rebuild_cum(tr, slots, 1),
                         drop2=rebuild_cum(tr, slots, 2)))
        for vname, vcfg in (("no_cost", dict(use_cost=False)),
                            ("cap1w", dict(capital_mult=0.2)),
                            ("cap3w", dict(capital_mult=0.6)),
                            ("no_filters", dict(exclude_st=False,
                                                exclude_limit_up=False,
                                                sell_limit_filter=False))):
            m2, tr2, _ = run_backtest(sc, {**cfg, **vcfg}, mkt)
            if m2 is None:
                continue
            rows.append(dict(score=s, cfg=name, variant=vname, cum=m2["cum_net"],
                             cum_trade=m2["cum_trade"], sharpe=m2["sharpe"],
                             sharpe_lo=np.nan, sharpe_hi=np.nan,
                             p_sharpe_neg=np.nan, maxdd=m2["maxdd"],
                             h1=m2["h1_cum"], h2=m2["h2_cum"],
                             t1=np.nan, t2=np.nan, t3=np.nan,
                             n=m2["n_trades"], win=m2["win_rate"],
                             cost_drag=m2["cost_drag"],
                             avg_cash=m2["avg_cash_ratio"],
                             drop1=rebuild_cum(tr2, slots, 1),
                             drop2=rebuild_cum(tr2, slots, 2)))
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "iter1_robust.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print("== base 变体 ==")
    print(df[df["variant"] == "base"][
        ["score", "cfg", "cum", "cum_trade", "sharpe", "sharpe_lo", "sharpe_hi",
         "p_sharpe_neg", "maxdd", "h1", "h2", "t1", "t2", "t3", "n", "win",
         "drop1", "drop2"]].round(3).to_string(index=False))
    print("\n== 其他变体 (cum) ==")
    piv = df.pivot_table(index=["score", "cfg"], columns="variant",
                         values="cum").round(3)
    print(piv.to_string())
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
