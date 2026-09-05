#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_equity.py — 重跑指定 (score集 × 策略) 并落盘净值/成交曲线 (供绘图)

用法:
  python3 dump_equity.py --cfgs A01_daily_top1,D01_h5_sl8_t1,K1h5_sl8 \
                         --sets V9_allz,V20_ensw2   # 默认全部 24 集
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery import battery_B1, battery_B2, battery_B3
from engine import SCORE_SETS, load_calendar, load_market, load_scores, run_backtest

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def find_cfg(name):
    for b in (battery_B1(), battery_B2(), battery_B3()):
        for c in b:
            if c["name"] == name:
                return c
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--sets", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfgs = [find_cfg(n) for n in args.cfgs.split(",")]
    sets = args.sets.split(",") if args.sets else list(SCORE_SETS.keys())
    eq_dir = os.path.join(RESULTS, "equity")
    tr_dir = os.path.join(RESULTS, "trades")
    os.makedirs(eq_dir, exist_ok=True); os.makedirs(tr_dir, exist_ok=True)

    market = load_market()
    tds, tdi = load_calendar()
    for s in sets:
        sc = load_scores(s)
        for c in cfgs:
            m, tr, eq = run_backtest(sc, c, market, tds, tdi)
            if m is None:
                print(f"[skip] {s} {c['name']} 无成交")
                continue
            eq.to_csv(os.path.join(eq_dir, f"{s}__{c['name']}.csv"), index=False)
            if tr is not None:
                tr.to_csv(os.path.join(tr_dir, f"{s}__{c['name']}.csv"), index=False)
    print(f"完成: {len(sets)} 集 × {len(cfgs)} 策略 → {eq_dir}")


if __name__ == "__main__":
    main()
