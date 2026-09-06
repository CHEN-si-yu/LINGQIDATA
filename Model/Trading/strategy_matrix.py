#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strategy_matrix.py — 11 策略 × 24 模型 横向对比 (任务2/3)

11 个策略 = B0 基线 (买入Top1隔日换仓) + 10 种不同风格策略:
  B0  daily_top1        买入Top1隔日换仓 (baseline)
  S1  h5_sl8_t1         Top1 + 固定持有5日 + 止损8%                (V20式)
  S2  h5_sl8_t2         Top2 + 固定持有5日 + 止损8%                (双仓分散)
  S3  nr3_t1            Top1 + 3日基准 + 仍Top1续持                 (最新打分评估续持)
  S4  re300_t2          Top2 + 排名掉出300退出                      (最新打分排名退出)
  S5  re300_sl8_tr15_t2 Top2 + 排名退出300 + 止损8% + 移动止盈15%   (S2主推)
  S6  h20_tr20_t2       Top2 + 持有≤20日 + 移动止损20%             (V20冠军协议)
  S7  h10_tr12_t1       Top1 + 持有≤10日 + 移动止损12%             (单仓趋势跟随)
  S8  re300_sl8_tp40_t2 Top2 + 排名退出300 + 止损8% + 固定止盈40%   (固定止盈)
  S9  rat20_10_sl8_t2   Top2 + 持有10日 + 利润锁定(+20锁+10) + 止损8% (利润锁定)
  S10 re300_sl8_q0.3_t2 Top2 + 排名退出300 + 止损8% + Top1分位择时  (打分择时)

24 个打分集: V1~V17 allz(16) + V11 allz/mixA/mixB/mixC/ensw2(5) + V18 f1/f2(2)
            + V20b_ensw2 (V20 使用 V20b, 用户指定)
全部: open 口径 (开盘先卖后买, 1 份资金), 真实成本, 官方 Test 窗 20250901~20260901。
主指标: cum_net 真实净值 (含利润再投资)。

输出:
  results/strategy_matrix.csv     (24 模型 × 11 策略 全指标长表)
  results/strategy_summary.csv    (每策略跨模型聚合 + 普适性 U 得分)
  results/equity_matrix/          (Top 策略 + 基线的逐日净值)
  STRATEGY_MATRIX_REPORT.md       (报告)
  figures/fig_strategy_heatmap.png / fig_strategy_equity.png

用法: python3 strategy_matrix.py [--workers 24] [--sets ...] [--quick]
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import SCORE_SETS, load_calendar, load_market, load_scores, \
    run_backtest  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# ---- 24 个打分集 (V20 → V20b) ----
ALLZ = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V12", "V13", "V14", "V15", "V16", "V17"]
SETS = [f"{v}_allz" for v in ALLZ]
SETS += ["V11_allz", "V11_mixA", "V11_mixB", "V11_mixC", "V11_ensw2",
         "V18_f1", "V18_f2", "V20b_ensw2"]
SCORE_SETS["V20b_ensw2"] = "Model/V20b/model_pred/2026q3/score_ens_w2.fea"

# ---- 11 个策略 ----
BASE_NAME = "B0_daily_top1"
STRATEGIES = [
    ("B0_daily_top1", "基线", "买入Top1隔日换仓",
     dict(top_n=1, hold=1)),
    ("S1_h5_sl8_t1", "固定持有", "Top1+持有5日+止损8%",
     dict(top_n=1, hold=5, stop_loss=0.08)),
    ("S2_h5_sl8_t2", "双仓固定持有", "Top2+持有5日+止损8%",
     dict(top_n=2, hold=5, stop_loss=0.08)),
    ("S3_nr3_t1", "续持", "Top1+3日基准+仍Top1续持",
     dict(top_n=1, hold=3, no_repeat=True)),
    ("S4_re300_t2", "排名退出", "Top2+最新打分掉出300退出",
     dict(top_n=2, hold=None, exit_rank=300, min_hold=2)),
    ("S5_re300_sl8_tr15_t2", "排名+止损止盈", "Top2+排名300+止损8%+移动止盈15%",
     dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
          stop_loss=0.08, trail_pct=0.15)),
    ("S6_h20_tr20_t2", "长持有移动止损", "Top2+持有≤20日+移动止损20%",
     dict(top_n=2, hold=20, trail_pct=0.20)),
    ("S7_h10_tr12_t1", "单仓移动止损", "Top1+持有≤10日+移动止损12%",
     dict(top_n=1, hold=10, trail_pct=0.12)),
    ("S8_re300_sl8_tp40_t2", "固定止盈", "Top2+排名300+止损8%+止盈40%",
     dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
          stop_loss=0.08, take_profit=0.40)),
    ("S9_rat20_10_sl8_t2", "利润锁定", "Top2+持有10日+利润锁定(20/10)+止损8%",
     dict(top_n=2, hold=10, rat_up=0.20, rat_floor=0.10, stop_loss=0.08)),
    ("S10_re300_sl8_q0.3_t2", "打分择时", "Top2+排名300+止损8%+Top1分位择时0.3",
     dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
          stop_loss=0.08, threshold_q=0.3)),
]
METRIC_FIELDS = ["cum_net", "ann_net", "sharpe", "maxdd", "cum_trade",
                 "win_rate", "avg_hold", "n_trades", "turnover", "cost_drag",
                 "h1_cum", "h2_cum", "h1_cum_t", "h2_cum_t",
                 "avg_cash_ratio", "empty_days"]

_G = {}


def _init_worker(market, tds, tdi, set_names):
    _G["market"], _G["tds"], _G["tdi"] = market, tds, tdi
    _G["scores"] = {s: load_scores(s) for s in set_names}


def _run_one(task):
    set_name, (sname, _, _, cfg) = task
    m, tr, eq = run_backtest(_G["scores"][set_name], cfg,
                             _G["market"], _G["tds"], _G["tdi"])
    if m is None:
        return None
    row = dict(set=set_name, strategy=sname)
    for f in METRIC_FIELDS:
        row[f] = m.get(f)
    return row


def zp(s):
    return s.rank(pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sets", default="")
    ap.add_argument("--quick", action="store_true",
                    help="跳过图表与净值落盘")
    args = ap.parse_args()

    sets = args.sets.split(",") if args.sets else SETS
    for s in sets:
        assert s in SCORE_SETS, f"未知 score 集: {s}"

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    market = load_market()
    tds, tdi = load_calendar()
    t0 = __import__("time").time()

    import multiprocessing as mp
    tasks = [(s, cfg) for s in sets for cfg in STRATEGIES]
    pool = mp.Pool(args.workers, initializer=_init_worker,
                   initargs=(market, tds, tdi, sets))
    rows = [r for r in pool.imap_unordered(_run_one, tasks, chunksize=8)
            if r is not None]
    pool.close(); pool.join()
    df = pd.DataFrame(rows)
    print(f"[matrix] {len(df)} 行 / {len(tasks)} 任务 "
          f"({__import__('time').time()-t0:.0f}s)", flush=True)

    out = os.path.join(RESULTS, "strategy_matrix.csv")
    df.to_csv(out, index=False)

    # ---- 每模型 vs 基线 ----
    base = df[df["strategy"] == BASE_NAME].set_index("set")["cum_net"]
    df["base_cum"] = df["set"].map(base)
    df["vs_base"] = df["cum_net"] - df["base_cum"]
    df["beat_base"] = (df["vs_base"] > 0).astype(int)

    # ---- 每策略跨模型聚合 + 普适性 U ----
    agg_rows = []
    for sn, g in df.groupby("strategy"):
        if sn == BASE_NAME:
            continue
        cum = g["cum_net"]
        r = dict(strategy=sn,
                 mean_cum=cum.mean(), median_cum=cum.median(),
                 p25_cum=cum.quantile(0.25), p75_cum=cum.quantile(0.75),
                 min_cum=cum.min(), max_cum=cum.max(),
                 n_pos=int((cum > 0).sum()),
                 n_beat_base=int(g["beat_base"].sum()),
                 n_sets=len(g),
                 median_sharpe=g["sharpe"].median(),
                 median_maxdd=g["maxdd"].median(),
                 median_win=g["win_rate"].median(),
                 median_ntrades=g["n_trades"].median(),
                 median_hold=g["avg_hold"].median(),
                 median_cost=g["cost_drag"].median(),
                 h1h2_both=int(((g["h1_cum"] > 0) & (g["h2_cum"] > 0)).sum()),
                 median_vs_base=g["vs_base"].median())
        agg_rows.append(r)
    agg = pd.DataFrame(agg_rows)
    agg["pos_frac"] = agg["n_pos"] / agg["n_sets"]
    agg["beat_frac"] = agg["n_beat_base"] / agg["n_sets"]
    agg["h1h2_frac"] = agg["h1h2_both"] / agg["n_sets"]
    agg["U"] = (0.30 * zp(agg["median_vs_base"])
                + 0.20 * zp(agg["beat_frac"])
                + 0.15 * zp(agg["pos_frac"])
                + 0.15 * zp(agg["h1h2_frac"])
                + 0.10 * zp(agg["median_sharpe"])
                + 0.10 * (1 - zp(agg["median_maxdd"].abs())))
    agg = agg.sort_values("U", ascending=False).reset_index(drop=True)
    agg.to_csv(os.path.join(RESULTS, "strategy_summary.csv"), index=False)

    # ---- 基线跨模型统计 ----
    bg = df[df["strategy"] == BASE_NAME]
    print("\n===== 基线 B0 (买入Top1隔日换仓) 跨 24 模型 =====")
    print(f"  median={bg['cum_net'].median():+.1%}  mean={bg['cum_net'].mean():+.1%}  "
          f"pos={int((bg['cum_net']>0).sum())}/{len(bg)}  "
          f"medianMaxDD={bg['maxdd'].median():+.1%}  medianSharpe={bg['sharpe'].median():.2f}")

    print("\n===== 每策略跨模型聚合 (24 模型, cum_net 真实净值) =====")
    cols = ["strategy", "median_cum", "mean_cum", "n_pos", "n_beat_base",
            "h1h2_both", "median_maxdd", "median_sharpe", "median_win",
            "median_hold", "U"]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- 逐模型 × 策略 cum_net 宽表 ----
    piv = df.pivot_table(index="set", columns="strategy", values="cum_net")
    piv = piv[["B0_daily_top1"] + [s[0] for s in STRATEGIES[1:]]]
    piv.to_csv(os.path.join(RESULTS, "strategy_matrix_wide.csv"))
    print("\n===== 逐模型 cum_net 宽表 =====")
    print(piv.round(3).to_string())

    if not args.quick:
        _figures(df, piv, agg, market, tds, tdi, sets)
    _report(df, piv, agg, bg)

    print(f"\n[matrix] 完成 → {out} / strategy_summary.csv / "
          f"strategy_matrix_wide.csv / STRATEGY_MATRIX_REPORT.md")


def _figures(df, piv, agg, market, tds, tdi, sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig1: 热图 (模型 × 策略 cum_net)
    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-0.6, vmax=4.0)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5)
    ax.set_title("24 models × 11 strategies cum_net (true equity, V20→V20b)")
    plt.colorbar(im, ax=ax, label="cum_net")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_strategy_heatmap.png"), dpi=150)
    plt.close(fig)

    # fig2: 净值曲线叠加 (Top-U 策略 + 基线, 24 模型各自净值, 中位数加粗)
    top_s = [agg.iloc[0]["strategy"], BASE_NAME]
    fig, axes = plt.subplots(1, len(top_s), figsize=(8 * len(top_s), 5))
    eq_dir = os.path.join(RESULTS, "equity_matrix")
    os.makedirs(eq_dir, exist_ok=True)
    for ax, sn in zip(axes, top_s):
        cfg = dict([(s[0], s[3]) for s in STRATEGIES])[sn]
        eqs = {}
        for s in sets:
            sc = load_scores(s)
            m, tr, eq = run_backtest(sc, cfg, market, tds, tdi)
            if eq is not None:
                eq.to_csv(os.path.join(eq_dir, f"{s}__{sn}.csv"), index=False)
                eqs[s] = eq.set_index("date")["equity"] / 50000 - 1
        med = pd.DataFrame(eqs).median(axis=1)
        for k, e in eqs.items():
            ax.plot(range(len(e)), e.values, lw=0.7, alpha=0.45,
                    color="#1f77b4")
        ax.plot(range(len(med)), med.values, lw=2.6, color="#d62728",
                label="median")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{sn} — 24 models (median={med.iloc[-1]:+.1%})")
        ax.set_xlabel("Trading day"); ax.set_ylabel("cum net")
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_strategy_equity.png"), dpi=150)
    plt.close(fig)
    print("[matrix] figures → fig_strategy_heatmap.png / fig_strategy_equity.png")


def _report(df, piv, agg, bg):
    desc = {s[0]: s[2] for s in STRATEGIES}
    L = ["# 交易策略矩阵报告 — 10 策略 + 基线 × 24 模型 (真实净值口径)",
         "",
         "> 日期: 2026-09-06 | 脚本: Model/Trading/strategy_matrix.py",
         "> 口径: Trading/engine.py cum_net 真实净值 (开盘先卖后买, 1 份资金, 含成本,"
         " 利润再投资), 官方 Test 窗 20250901~20260901 (243 因子日)。",
         "> 模型: V1~V17 allz(16) + V11 allz/mixA/mixB/mixC/ensw2(5) + V18 f1/f2(2)"
         " + V20b_ensw2 (用户指定: V20 用 V20b 测试)。",
         "> 基线: B0 买入Top1隔日换仓。",
         ""]
    L.append("## 一、策略清单 (10 种不同风格 + 基线)")
    L.append("")
    L.append("| # | 策略 | 风格 | 设计 |")
    L.append("|---|---|---|---|")
    for i, (sn, grp, ds, cfg) in enumerate(STRATEGIES):
        tag = "**基线**" if sn == BASE_NAME else ""
        L.append(f"| {i} | `{sn}` | {grp} | {ds} {tag} |")
    L.append("")
    L.append("## 二、跨 24 模型聚合 (普适性排序, U 得分)")
    L.append("")
    cols = ["strategy", "median_cum", "mean_cum", "n_pos", "n_beat_base",
            "h1h2_both", "median_maxdd", "median_sharpe", "median_win",
            "median_hold", "median_ntrades", "U"]
    fmt = dict(strategy="{}", median_cum="{:.1%}", mean_cum="{:.1%}",
               n_pos="{}", n_beat_base="{}", h1h2_both="{}",
               median_maxdd="{:.1%}", median_sharpe="{:.2f}",
               median_win="{:.1%}", median_hold="{:.1f}", median_ntrades="{:.0f}",
               U="{:.3f}")
    agg2 = agg.copy()
    agg2["median_win"] = agg2["median_win"].fillna(0)
    L.append(_md(agg2[cols], fmt))
    L.append("")
    L.append(f"基线 B0 跨模型: median={bg['cum_net'].median():+.1%}, "
             f"mean={bg['cum_net'].mean():+.1%}, "
             f"正收益 {int((bg['cum_net']>0).sum())}/{len(bg)}, "
             f"medianMaxDD={bg['maxdd'].median():+.1%}, "
             f"medianSharpe={bg['sharpe'].median():.2f}")
    L.append("")
    L.append("U 得分构成: 30% 相对基线增量中位 + 20% 超基线占比 + 15% 正收益占比"
             " + 15% H1/H2双正占比 + 10% Sharpe + 10% MaxDD。")
    L.append("")
    L.append("## 三、逐模型 × 策略 cum_net (真实净值)")
    L.append("")
    L.append("| 模型 | " + " | ".join(piv.columns) + " |")
    L.append("|---|" + "---|" * len(piv.columns))
    for i, row in piv.iterrows():
        cells = [f"{v:+.1%}" for v in row.values]
        b = row[BASE_NAME]
        L.append(f"| {i} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("> 加粗 = 该模型的最优策略; 最后一列注 = 相对基线增量。")
    L.append("")
    L.append("## 四、结论要点")
    L.append("")
    L.append("- 普适性最优策略 (U 最高): 见上表; 关键指标 = 超基线模型数 (n_beat_base)。")
    L.append("- 策略层收益与模型质量强相关: 差模型 (V5/V3/V2/V18) 在多数策略下仍为负。")
    L.append("- 详见 figures/fig_strategy_heatmap.png (全矩阵热图) 与"
             " fig_strategy_equity.png (最优策略净值叠加)。")
    L.append("")
    report = "\n".join(L)
    with open(os.path.join(HERE, "STRATEGY_MATRIX_REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write(report)
    print("[matrix] → STRATEGY_MATRIX_REPORT.md")


def _md(df, fmt=None):
    lines = ["| " + " | ".join(df.columns) + " |",
             "|---" * len(df.columns) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("—")
            elif fmt and c in fmt:
                cells.append(fmt[c].format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
