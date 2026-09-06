#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strategy_robustness.py — 主推普适策略 (S4/S5) 稳健性矩阵 (24 模型, 真实净值口径)

维度: 基准(含成本) / 无成本 / 1W 资金 / 3W 资金 / close_seq 卖出口径
输出: results/strategy_robustness.csv + 打印摘要
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import SCORE_SETS, load_calendar, load_market, load_scores, \
    run_backtest  # noqa: E402
from strategy_matrix import SETS, STRATEGIES  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FOCUS = ["S4_re300_t2", "S5_re300_sl8_tr15_t2"]


def main():
    market = load_market()
    tds, tdi = load_calendar()
    rows = []
    for s in SETS:
        sc = load_scores(s)
        for sname, _, _, cfg in STRATEGIES:
            if sname not in FOCUS:
                continue
            for var, vcfg in [("base", {}), ("nocost", {"use_cost": False}),
                              ("cap1w", {"capital_mult": 0.2}),
                              ("cap3w", {"capital_mult": 0.6}),
                              ("close_seq", {"sell_at": "close_seq"})]:
                c = {**cfg, **vcfg}
                m, tr, eq = run_backtest(sc, c, market, tds, tdi)
                if m is None:
                    continue
                rows.append(dict(set=s, strategy=sname, variant=var,
                                 cum_net=m["cum_net"], sharpe=m["sharpe"],
                                 maxdd=m["maxdd"], n=m["n_trades"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "strategy_robustness.csv"), index=False)
    base = df[df["variant"] == "base"].set_index(["set", "strategy"])["cum_net"]
    df["vs_base"] = df.apply(
        lambda r: r["cum_net"] - base.get((r["set"], r["strategy"]), float("nan")),
        axis=1)
    for var, g in df.groupby("variant"):
        print(f"{var:10s} median={g['cum_net'].median():+.1%} "
              f"pos={int((g['cum_net']>0).sum())}/{len(g)} "
              f"medianSharpe={g['sharpe'].median():.2f} "
              f"medianMaxDD={g['maxdd'].median():+.1%} medianN={g['n'].median():.0f}")
    print("→ results/strategy_robustness.csv")


if __name__ == "__main__":
    main()
