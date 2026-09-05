#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate.py — 引擎回归校准: 与旧引擎 (V11/strat_backtest.py) 逐笔对齐

锚点 (直接跑旧引擎得到, 与 Model/FINAL_REPORT.md 一致):
  - V9_allz  hold5+stop8% 开盘卖 : 净累计 +122.98%
  - V11_ensw2 hold5s8 收盘卖     : 净累计 +365.17% (FINAL_REPORT 冠军 +365.2%)
  - V20_ensw2 hold5s8 收盘卖     : 净累计 +277.79% (V20 复训打分)
  - V2_allz  每日换仓 Top1       : -50.7%
  - V9_allz  每日换仓 Top1       : +20.92%
本引擎口径差异 (有意为之): 期末强制清仓 → 多最后一笔平仓交易, 差异 < 3pp。
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/autodl-fs/data/lingqiData/Model/V11")
sys.path.insert(0, "/autodl-fs/data/lingqiData/Model/Trading")

import pandas as pd

import strat_backtest as sb
from engine import load_market, load_scores, run_backtest, load_calendar

CASES = [
    ("V9_allz", dict(top_n=1, hold=5, stop_loss=0.08), "open", 122.98),
    ("V11_ensw2", dict(top_n=1, hold=5, stop_loss=0.08), "close_overlap", 365.17),
    ("V20_ensw2", dict(top_n=1, hold=5, stop_loss=0.08), "close_overlap", 277.79),
    ("V2_allz", dict(top_n=1, hold=1), "open", -50.7),
    ("V9_allz", dict(top_n=1, hold=1), "open", 20.92),
]


def old_engine_run(score_df, cfg, sell_close):
    open_map, close_map, prev_close_map, amount_map, name_map = sb.load_prices()
    tds = sb.load_calendar()
    if sell_close:
        src = open(sb.__file__).read()
        old = "sp = open_map.get((p['planned_sell'], code))"
        assert old in src
        mod = {}
        exec(src.replace(old, "sp = close_map.get((p['planned_sell'], code))"), mod)
        rb = mod["run_backtest"]
    else:
        rb = sb.run_backtest
    m, tr, dr = rb(score_df, open_map, close_map, prev_close_map, name_map,
                   tds=tds, window_start="20250901", window_end="20260901",
                   use_cost=True, exclude_st=True, exclude_limit_up=True,
                   buy_gap_limit=0.095,
                   hold=cfg.get("hold"), stop_loss=cfg.get("stop_loss"))
    return m["net"]["cum"] * 100


def main():
    market = load_market()
    tds, tdi = load_calendar()
    from engine import SCORE_SETS, PROJECT_ROOT
    ok = True
    print(f"{'case':<34}{'old%':>10}{'new%':>10}{'diffpp':>8}  判定")
    for set_name, cfg, sell_at, _expect in CASES:
        df = pd.read_feather(PROJECT_ROOT + SCORE_SETS[set_name]).set_index("date").sort_index()
        old_cum = old_engine_run(df, cfg, sell_close=(sell_at == "close_overlap"))
        sc = load_scores(set_name)
        cfg2 = {**cfg, "sell_at": sell_at}
        m, tr, eq = run_backtest(sc, cfg2, market, tds, tdi)
        new_cum = m["cum_trade"] * 100
        diff = new_cum - old_cum
        verdict = "OK" if abs(diff) < 3.0 else "!! 需检查"
        if verdict != "OK":
            ok = False
        print(f"{set_name+' '+str(cfg)+' '+sell_at:<34}{old_cum:>10.2f}{new_cum:>10.2f}{diff:>8.2f}  {verdict}")
    print("\n全部 OK" if ok else "\n存在超差 case")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
