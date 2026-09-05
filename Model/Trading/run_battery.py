#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_battery.py — 批量回测: 全部 score 集 × 策略电池 (多进程)

用法:
  python3 run_battery.py --battery B1            # 主电池
  python3 run_battery.py --battery B2            # 细化网格
  python3 run_battery.py --battery SELLMODE      # 卖出口径对照 (含净值曲线落盘)
  python3 run_battery.py --battery ALL           # 全部
  --workers 32  --sets V9_allz,V20_ensw2         # 限定集合 (默认全部 24 个)
"""
import argparse
import csv
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery import BATTERIES
from engine import (SCORE_SETS, load_calendar, load_market, load_scores,
                    run_backtest)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

METRIC_FIELDS = ["cum_net", "ann_net", "sharpe", "maxdd", "cum_add", "cum_trade",
                 "sharpe_trade", "win_rate", "avg_hold", "med_trade", "best_trade",
                 "worst_trade", "turnover", "cost_drag", "n_trades", "n_days",
                 "empty_days", "avg_npos", "avg_cash_ratio", "h1_cum", "h2_cum",
                 "h1_cum_t", "h2_cum_t", "h1_win", "h2_win", "skipped_buy",
                 "blocked_sell", "max_overlap"]
CFG_FIELDS = ["top_n", "hold", "sell_at", "no_repeat", "exit_rank", "min_hold",
              "stop_loss", "take_profit", "trail_pct", "rat_up", "rat_floor",
              "threshold_q"]

_G = {}


def _init_worker(market, tds, tdi, score_names):
    _G["market"], _G["tds"], _G["tdi"] = market, tds, tdi
    _G["score_names"] = score_names
    _G["scores"] = {s: load_scores(s) for s in score_names}


def _run_one(task):
    set_name, cfg, save_eq = (task + (False,))[:3]
    sc = _G["scores"][set_name]
    m, tr, eq = run_backtest(sc, cfg, _G["market"], _G["tds"], _G["tdi"])
    if m is None:
        return None, None, None
    row = dict(set=set_name, cfg=cfg["name"], group=cfg["group"])
    for f in CFG_FIELDS:
        row[f] = cfg.get(f)
    for f in METRIC_FIELDS:
        row[f] = m.get(f)
    if save_eq:
        return row, tr, eq
    return row, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="B1", choices=list(BATTERIES) + ["ALL"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--sets", default="", help="逗号分隔; 默认全部")
    args = ap.parse_args()

    if args.battery == "ALL":
        batteries = list(BATTERIES.keys())
    else:
        batteries = [args.battery]
    sets = args.sets.split(",") if args.sets else list(SCORE_SETS.keys())
    for s in sets:
        assert s in SCORE_SETS, f"未知 score 集: {s}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    market = load_market()
    tds, tdi = load_calendar()

    for bname in batteries:
        cfgs = BATTERIES[bname]()
        tasks = [(s, c) for s, c in product(sets, cfgs)]
        out_csv = os.path.join(RESULTS_DIR, f"battery_{bname}.csv")
        save_equity = (bname == "SELLMODE")
        eq_dir = os.path.join(RESULTS_DIR, "equity")
        if save_equity:
            os.makedirs(eq_dir, exist_ok=True)
        tasks = [(s, c, save_equity) for s, c in product(sets, cfgs)]

        fieldnames = ["set", "cfg", "group"] + CFG_FIELDS + METRIC_FIELDS
        t0 = time.time()
        done = 0
        n_ok = 0
        import multiprocessing as mp
        pool = mp.Pool(args.workers, initializer=_init_worker,
                       initargs=(market, tds, tdi, sets))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row, tr, eq in pool.imap_unordered(_run_one, tasks, chunksize=8):
                done += 1
                if row is None:
                    continue
                n_ok += 1
                w.writerow(row)
                if save_equity and eq is not None:
                    eq.to_csv(os.path.join(eq_dir, f"{row['set']}__{row['cfg']}.csv"),
                              index=False)
                if done % 1000 == 0:
                    print(f"[{bname}] {done}/{len(tasks)}  "
                          f"({(time.time()-t0):.0f}s, {done/(time.time()-t0):.1f}个/s)",
                          flush=True)
        pool.close(); pool.join()
        print(f"[{bname}] 完成: {n_ok}/{len(tasks)} 行 → {out_csv} "
              f"(总耗时 {(time.time()-t0):.0f}s)", flush=True)


if __name__ == "__main__":
    main()
