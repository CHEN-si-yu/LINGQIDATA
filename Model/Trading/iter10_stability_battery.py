#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter10_stability_battery.py — 稳健优先策略电池 (用户需求 2026-09-06)

需求: 持仓 1~3 只; 允许最终收益偏低; 要求收益稳定、回撤小、显式止盈止损;
在"IC 偏高"的好模型打分集上找普适的稳健策略 (规避单模型种子彩票依赖)。

高 IC 子集 (Test 窗 1d RankIC ≥ ~0.040, 官方口径来自各单元 model_pic):
  V7/V8/V9/V10/V11/V12/V14 allz (0.041~0.045) + V18_f1/f2 (0.062/0.065)
  + V19_meta (0.0505) + V20b/V20c_ensw2 (0.0406, 两个非幸运种子复刻)

排名口径 (稳健优先): 30% 正收益占比 + 25% 中位MaxDD + 20% H1/H2双正占比
+ 15% 中位Sharpe + 10% 中位收益 (z-秩和)。引擎: Trading/engine.py cum_net
真实净值 (开盘先卖后买, 1 份资金, 含成本, 利润再投资), Test 窗 20250901~20260901。

输出: results/iter10_stability_battery.csv (子集×配置全指标) + 控制台汇总
用法: python3 iter10_stability_battery.py [--workers 12]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import SCORE_SETS, load_calendar, load_market, load_scores, \
    run_backtest  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SCORE_SETS["V18_f1"] = "Model/V18/model_pred/v18_f1.fea"
SCORE_SETS["V18_f2"] = "Model/V18/model_pred/v18_f2.fea"
SCORE_SETS["V19_meta"] = "Model/V19/model_pred/2026q3/score_meta.fea"
SCORE_SETS["V20b_ensw2"] = "Model/V20b/model_pred/2026q3/score_ens_w2.fea"
SCORE_SETS["V20c_ensw2"] = "Model/V20c/model_pred/2026q3/score_ens_w2.fea"
SCORE_SETS["V13b_allz"] = "Model/V13b/model_pred/2026q3/all_zscore_score.fea"
SCORE_SETS["V13c_allz"] = "Model/V13c/model_pred/2026q3/all_zscore_score.fea"

GOOD = ["V7_allz", "V8_allz", "V9_allz", "V10_allz", "V11_allz", "V12_allz",
        "V14_allz", "V18_f1", "V18_f2", "V19_meta", "V20b_ensw2",
        "V20c_ensw2"]
# 跨种子稳健性复检 (选优后全正才算达标)
SEED_UNITS = ["V20_ensw2", "V20b_ensw2", "V20c_ensw2",
              "V13_allz", "V13b_allz", "V13c_allz"]

# ---- 稳健优先配置电池 (全部 Top2 双仓 = 2 只/腿, 满足 1~3 只约束) ----
CONFIGS = [
    # (name, cfg)
    ("r_re300",               dict(top_n=2, hold=None, exit_rank=300, min_hold=2)),
    ("r_re300_sl8_tr15",      dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.08, trail_pct=0.15)),  # S5 主推
    ("r_re200_sl8_tr15",      dict(top_n=2, hold=None, exit_rank=200, min_hold=2, stop_loss=0.08, trail_pct=0.15)),
    ("r_re150_sl8_tr15",      dict(top_n=2, hold=None, exit_rank=150, min_hold=2, stop_loss=0.08, trail_pct=0.15)),
    ("r_re300_sl6_tr15",      dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.06, trail_pct=0.15)),
    ("r_re300_sl10_tr15",     dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.10, trail_pct=0.15)),
    ("r_re300_sl8_tr10",      dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.08, trail_pct=0.10)),
    ("r_re300_sl8_tr20",      dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.08, trail_pct=0.20)),
    ("r_re300_sl8_tr15_tp25", dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.08, trail_pct=0.15, take_profit=0.25)),
    ("r_re300_sl8_tr15_tp20", dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.08, trail_pct=0.15, take_profit=0.20)),
    ("r_re300_sl8_tr15_q30",  dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.08, trail_pct=0.15, threshold_q=0.30)),
    ("r_re300_sl8_tr15_q50",  dict(top_n=2, hold=None, exit_rank=300, min_hold=2, stop_loss=0.08, trail_pct=0.15, threshold_q=0.50)),
    ("h_h20_tr20",            dict(top_n=2, hold=20, trail_pct=0.20)),   # S6
    ("h_h20_tr15",            dict(top_n=2, hold=20, trail_pct=0.15)),
    ("h_h20_tr12",            dict(top_n=2, hold=20, trail_pct=0.12)),
    ("h_h20_tr20_sl8",        dict(top_n=2, hold=20, trail_pct=0.20, stop_loss=0.08)),
    ("h_h15_tr15",            dict(top_n=2, hold=15, trail_pct=0.15)),
    ("c_h10_rat20_10_sl8",    dict(top_n=2, hold=10, rat_up=0.20, rat_floor=0.10, stop_loss=0.08)),  # S9
    ("c_h10_rat15_8_sl6",     dict(top_n=2, hold=10, rat_up=0.15, rat_floor=0.08, stop_loss=0.06)),
    ("c_re300_rat20_10_sl8",  dict(top_n=2, hold=None, exit_rank=300, min_hold=2, rat_up=0.20, rat_floor=0.10, stop_loss=0.08)),
    ("n_h10_tr12_t1",         dict(top_n=1, hold=10, trail_pct=0.12)),   # S7 单仓
    ("n_h5_sl8_t1",           dict(top_n=1, hold=5, stop_loss=0.08)),    # S1 单仓
    ("d_daily_t1",            dict(top_n=1, hold=1)),                    # 基线
    ("d_daily_t2",            dict(top_n=2, hold=1)),
]
# 叠加止盈上限 (把赢家封顶, 换回撤收窄) 的另一族
for tp in (0.30, 0.35):
    CONFIGS.append(
        (f"c2_h20_tr20_tp{int(tp*100)}",
         dict(top_n=2, hold=20, trail_pct=0.20, take_profit=tp)))

_G = {}


def _init_worker(market, tds, tdi, sets):
    _G["market"], _G["tds"], _G["tdi"] = market, tds, tdi
    _G["scores"] = {s: load_scores(s) for s in sets}


def _run_one(task):
    set_name, (cname, cfg) = task
    m, tr, eq = run_backtest(_G["scores"][set_name], cfg,
                             _G["market"], _G["tds"], _G["tdi"])
    if m is None:
        return None
    return dict(set=set_name, cfg=cname, cum_net=m["cum_net"],
                sharpe=m["sharpe"], maxdd=m["maxdd"], h1=m["h1_cum"],
                h2=m["h2_cum"], n_trades=m["n_trades"],
                win_rate=m["win_rate"], avg_hold=m["avg_hold"])


def zp(s):
    return s.rank(pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    market = load_market()
    tds, tdi = load_calendar()

    import multiprocessing as mp
    sets = GOOD
    tasks = [(s, c) for s in sets for c in CONFIGS]
    pool = mp.Pool(args.workers, initializer=_init_worker,
                   initargs=(market, tds, tdi, sets))
    rows = [r for r in pool.imap_unordered(_run_one, tasks, chunksize=8)
            if r is not None]
    pool.close()
    pool.join()
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "iter10_stability_battery.csv"),
              index=False)

    # ---- 稳健优先汇总 ----
    agg_rows = []
    for cn, g in df.groupby("cfg"):
        r = dict(cfg=cn, n_sets=len(g),
                 n_pos=int((g["cum_net"] > 0).sum()),
                 n_h1h2=int(((g["h1"] > 0) & (g["h2"] > 0)).sum()),
                 med_cum=g["cum_net"].median(),
                 med_sharpe=g["sharpe"].median(),
                 med_maxdd=g["maxdd"].median(),
                 worst_maxdd=g["maxdd"].min(),
                 med_hold=g["avg_hold"].median(),
                 med_trades=g["n_trades"].median())
        agg_rows.append(r)
    agg = pd.DataFrame(agg_rows)
    agg["U"] = (0.30 * zp(agg["n_pos"]) + 0.25 * (1 - zp(agg["med_maxdd"].abs()))
                + 0.20 * zp(agg["n_h1h2"]) + 0.15 * zp(agg["med_sharpe"])
                + 0.10 * zp(agg["med_cum"]))
    agg = agg.sort_values("U", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 220)
    print("===== 高IC子集 (%d 集) × %d 配置 — 稳健优先排序 ====="
          % (len(GOOD), len(CONFIGS)))
    cols = ["cfg", "n_pos", "n_h1h2", "med_cum", "med_sharpe", "med_maxdd",
            "worst_maxdd", "med_hold", "U"]
    print(agg[cols].round(3).to_string(index=False))

    # ---- 前 6 名做跨种子复检 (6 种子单元全正才算达标) ----
    top6 = agg.head(6)["cfg"].tolist()
    print("\n===== 前 6 配置 × 6 种子单元复检 (V20/V20b/V20c/V13/V13b/V13c) =====")
    srows = []
    for cn in top6:
        cfg = dict(CONFIGS)[cn]
        for u in SEED_UNITS:
            sc = load_scores(u)
            m, tr, eq = run_backtest(sc, cfg, market, tds, tdi)
            srows.append(dict(cfg=cn, unit=u,
                              cum=m["cum_net"] if m else np.nan,
                              sharpe=m["sharpe"] if m else np.nan))
    sdf = pd.DataFrame(srows)
    piv = sdf.pivot_table(index="cfg", columns="unit", values="cum")
    piv["n_pos_6"] = (piv > 0).sum(axis=1)
    piv["med"] = piv[SEED_UNITS].median(axis=1)
    print(piv.round(3).to_string())
    ok = [cn for cn in top6 if piv.loc[cn, "n_pos_6"] == 6]
    print(f"\n6/6 种子全正: {ok}")


if __name__ == "__main__":
    main()
