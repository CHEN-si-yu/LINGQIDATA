#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter1_grid.py — 新迭代第一轮: 协议网格 (修复利润再投资后的真实净值口径)

在冠军打分 (V11_ensw2 / V20_ensw2) 上做协议族网格:
  1. 纯持有期: top_n × hold × {无/止损/移动止损/止损+止盈}
  2. 排名退出: top_n × exit_rank × min_hold × {无/止损/移动/组合} × hold上限
  3. norep 续持变体
主指标 = cum_net (含再投资的现金级真实净值), 参考 cum_trade (旧口径复利),
并记录 H1/H2 (净值与逐笔)、T1/T2/T3 三等分、MaxDD、Sharpe、成本拖累、空仓日。

输出: results/iter1_protocol_grid.csv
"""
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import SCORE_SETS, load_calendar, load_market, load_scores, run_backtest

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TARGET_SETS = ["V11_ensw2", "V20_ensw2"]

_G = {}


def thirds(tr):
    if tr is None or len(tr) == 0:
        return np.nan, np.nan, np.nan
    tr = tr.sort_values("buy_dt")
    k = max(1, len(tr) // 3)
    out = []
    for part in (tr.iloc[:k], tr.iloc[k:2 * k], tr.iloc[2 * k:]):
        out.append(float((1 + part["net_pct"] / 100).prod() - 1))
    return tuple(out)


def build_grid():
    cfgs = []
    # 1) 纯持有期族
    for tn, h in product((1, 2, 3), (3, 5, 7, 10, 15, 20, 25, 30, 40, 60)):
        cfgs.append(dict(name=f"H_hold{h}_t{tn}", top_n=tn, hold=h))
        cfgs.append(dict(name=f"H_hold{h}_sl8_t{tn}", top_n=tn, hold=h, stop_loss=0.08))
        cfgs.append(dict(name=f"H_hold{h}_sl5_t{tn}", top_n=tn, hold=h, stop_loss=0.05))
        cfgs.append(dict(name=f"H_hold{h}_sl10_t{tn}", top_n=tn, hold=h, stop_loss=0.10))
        cfgs.append(dict(name=f"H_hold{h}_tr12_t{tn}", top_n=tn, hold=h, trail_pct=0.12))
        cfgs.append(dict(name=f"H_hold{h}_tr15_t{tn}", top_n=tn, hold=h, trail_pct=0.15))
        cfgs.append(dict(name=f"H_hold{h}_tr20_t{tn}", top_n=tn, hold=h, trail_pct=0.20))
        cfgs.append(dict(name=f"H_hold{h}_sl8tr15_t{tn}", top_n=tn, hold=h,
                         stop_loss=0.08, trail_pct=0.15))
    # 2) 排名退出族
    for tn, er, mh in product((1, 2, 3), (100, 200, 300, 400, 500), (2, 3, 5)):
        base = dict(top_n=tn, hold=None, exit_rank=er, min_hold=mh)
        cfgs.append(dict(name=f"R_re{er}_mh{mh}_t{tn}", **base))
        cfgs.append(dict(name=f"R_re{er}_mh{mh}_sl8_t{tn}", **base, stop_loss=0.08))
        cfgs.append(dict(name=f"R_re{er}_mh{mh}_tr12_t{tn}", **base, trail_pct=0.12))
        cfgs.append(dict(name=f"R_re{er}_mh{mh}_sl8tr15_t{tn}", **base,
                         stop_loss=0.08, trail_pct=0.15))
        cfgs.append(dict(name=f"R_re{er}_mh{mh}_h20_t{tn}",
                         **{**base, "hold": 20}))
        cfgs.append(dict(name=f"R_re{er}_mh{mh}_h20_sl8tr15_t{tn}",
                         **{**base, "hold": 20}, stop_loss=0.08,
                         trail_pct=0.15))
    # 3) norep 续持
    for tn, h in product((1, 2), (3, 5, 10)):
        cfgs.append(dict(name=f"NR_hold{h}_t{tn}", top_n=tn, hold=h, no_repeat=True))
        cfgs.append(dict(name=f"NR_hold{h}_sl8_t{tn}", top_n=tn, hold=h,
                         no_repeat=True, stop_loss=0.08))
    # 去重
    seen, out = set(), []
    for c in cfgs:
        key = tuple(sorted((k, v) for k, v in c.items() if k != "name"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def run(args):
    s, cfg = args
    sc = _G["scores"][s]
    m, tr, eq = run_backtest(sc, cfg, _G["mkt"], _G["tds"], _G["tdi"])
    if m is None:
        return None
    t1, t2, t3 = thirds(tr)
    return dict(set=s, cfg=cfg["name"], top_n=cfg["top_n"], hold=cfg.get("hold"),
                exit_rank=cfg.get("exit_rank"), min_hold=cfg.get("min_hold"),
                stop_loss=cfg.get("stop_loss"), trail_pct=cfg.get("trail_pct"),
                no_repeat=cfg.get("no_repeat", False),
                cum_net=m["cum_net"], cum_trade=m["cum_trade"],
                sharpe=m["sharpe"], sharpe_trade=m["sharpe_trade"],
                maxdd=m["maxdd"], h1_cum=m["h1_cum"], h2_cum=m["h2_cum"],
                h1_cum_t=m["h1_cum_t"], h2_cum_t=m["h2_cum_t"],
                win_rate=m["win_rate"], n_trades=m["n_trades"],
                avg_hold=m["avg_hold"], cost_drag=m["cost_drag"],
                turnover=m["turnover"], avg_cash=m["avg_cash_ratio"],
                empty_days=m["empty_days"], t1=t1, t2=t2, t3=t3,
                blocked_sell=m["blocked_sell"])


def main():
    import multiprocessing as mp
    mkt = load_market()
    tds, tdi = load_calendar()
    scores = {s: load_scores(s) for s in TARGET_SETS}
    _G.update(mkt=mkt, tds=tds, tdi=tdi, scores=scores)
    grid = build_grid()
    print(f"[grid] {len(grid)} 配置 × {len(TARGET_SETS)} 打分集 = "
          f"{len(grid) * len(TARGET_SETS)} 次回测", flush=True)
    tasks = list(product(TARGET_SETS, grid))
    pool = mp.Pool(20)
    rows = [r for r in pool.imap_unordered(run, tasks, chunksize=4) if r]
    pool.close(); pool.join()
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "iter1_protocol_grid.csv")
    df.to_csv(out, index=False)
    print(f"[grid] 完成 {len(df)} 行 → {out}")
    # 摘要: 每集 Top15 (按 cum_net)
    pd.set_option("display.width", 250)
    for s in TARGET_SETS:
        sub = df[df["set"] == s].sort_values("cum_net", ascending=False).head(15)
        print(f"\n== {s} Top15 by cum_net ==")
        print(sub[["cfg", "cum_net", "cum_trade", "sharpe", "maxdd", "h1_cum",
                   "h2_cum", "win_rate", "n_trades", "avg_hold", "t1", "t2", "t3"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
