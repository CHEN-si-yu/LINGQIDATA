#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate.py — 跨模型聚合评估 / 普适性排名 / 报告表格与图表

输入: results/battery_B1.csv, battery_B2.csv, battery_SELLMODE.csv
输出:
  results/agg_strategies.csv   — 每个策略跨 24 个 score 集的聚合统计 + 普适性得分 U
  results/per_model_best.csv   — 每个 score 集内的 Top5 策略
  results/sellmode_summary.csv — 三种卖出口径对照
  figures/*.png                — 图表
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)


def load_batteries():
    frames = []
    for f in ["battery_B1.csv", "battery_B2.csv", "battery_B3.csv"]:
        p = os.path.join(RESULTS, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    # 基线相对增量: 每模型 vs 隔日换仓 Top1 (A01) 与 hold5+stop8% Top1 (D01)
    base = df[df["cfg"] == "A01_daily_top1"][["set", "cum_trade"]].rename(
        columns={"cum_trade": "_base_a01"})
    d01 = df[df["cfg"] == "D01_h5_sl8_t1"][["set", "cum_trade"]].rename(
        columns={"cum_trade": "_base_d01"})
    df = df.merge(base, on="set", how="left").merge(d01, on="set", how="left")
    df["vs_a01"] = df["cum_trade"] - df["_base_a01"]
    df["vs_d01"] = df["cum_trade"] - df["_base_d01"]
    return df


def aggregate(df):
    """每个 cfg 跨 score 集的聚合统计。"""
    rows = []
    for cfg, g in df.groupby("cfg"):
        r = dict(cfg=cfg, group=g["group"].iloc[0], n_sets=len(g))
        cum = g["cum_trade"]
        r.update(
            mean_net=cum.mean(), median_net=cum.median(),
            p25_net=cum.quantile(0.25), p75_net=cum.quantile(0.75),
            min_net=cum.min(), max_net=cum.max(),
            pos_frac=(cum > 0).mean(),
            mean_sharpe=g["sharpe"].mean(), median_sharpe=g["sharpe"].median(),
            median_maxdd=g["maxdd"].median(),
            mean_win=g["win_rate"].mean(),
            mean_trades=g["n_trades"].mean(), mean_turnover=g["turnover"].mean(),
            mean_cost=g["cost_drag"].mean(),
            mean_hold=g["avg_hold"].mean(),
            mean_rank=g["_rank"].mean(), median_rank=g["_rank"].median(),
        )
        h1 = g["h1_cum_t"].fillna(-1); h2 = g["h2_cum_t"].fillna(-1)
        r["h1h2_frac"] = ((h1 > 0) & (h2 > 0)).mean()
        r["h1_pos"] = (h1 > 0).mean(); r["h2_pos"] = (h2 > 0).mean()
        r["beat_a01_frac"] = (g["vs_a01"] > 0).mean()
        r["beat_d01_frac"] = (g["vs_d01"] > 0).mean()
        r["median_vs_a01"] = g["vs_a01"].median()
        r["median_vs_d01"] = g["vs_d01"].median()
        rows.append(r)
    agg = pd.DataFrame(rows)
    # 普适性得分 U (0..1): 相对基线增量为主 + 正收益占比 + 分半双正 + 风险调整
    def zp(s):
        return s.rank(pct=True)
    agg["U"] = (0.30 * zp(agg["median_vs_a01"])
                + 0.20 * zp(agg["beat_a01_frac"])
                + 0.15 * zp(agg["pos_frac"])
                + 0.15 * zp(agg["h1h2_frac"])
                + 0.10 * zp(agg["median_sharpe"])
                + 0.10 * (1 - zp(agg["median_maxdd"].abs())))
    return agg.sort_values("U", ascending=False).reset_index(drop=True)


def per_model_best(df, top_k=5):
    rows = []
    for s, g in df.groupby("set"):
        g = g.sort_values("cum_trade", ascending=False).head(top_k)
        for i, (_, r) in enumerate(g.iterrows(), 1):
            rows.append(dict(set=s, rank=i, cfg=r["cfg"], group=r["group"],
                             cum_trade=r["cum_trade"], sharpe=r["sharpe"],
                             maxdd=r["maxdd"], win_rate=r["win_rate"],
                             n_trades=r["n_trades"], h1=r["h1_cum_t"], h2=r["h2_cum_t"]))
    return pd.DataFrame(rows)


def sellmode_summary():
    p = os.path.join(RESULTS, "battery_SELLMODE.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    out = []
    for cfg in df["cfg"].unique():
        g = df[df["cfg"] == cfg].set_index("set")
        out.append(dict(cfg=cfg,
                        n=g["cum_trade"].notna().sum(),
                        mean_cum=g["cum_trade"].mean(),
                        median_cum=g["cum_trade"].median(),
                        pos_frac=(g["cum_trade"] > 0).mean(),
                        max_overlap_median=g["max_overlap"].median(),
                        mean_maxdd=g["maxdd"].mean()))
    return pd.DataFrame(out)


def md_table(df, cols, fmts=None):
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:.2f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_figures(df, agg):
    # fig1: 普适性散点 (x=正收益占比, y=中位净收益, 气泡=分半双正占比)
    top20 = agg.head(20)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(agg["pos_frac"] * 100, agg["median_net"] * 100,
               s=20 + 150 * agg["h1h2_frac"], alpha=0.35, c="#9bb8d3")
    ax.scatter(top20["pos_frac"] * 100, top20["median_net"] * 100,
               s=20 + 150 * top20["h1h2_frac"], alpha=0.9, c="#d4552b")
    for _, r in top20.iterrows():
        ax.annotate(r["cfg"].replace("K1h", "K1:").replace("K2h", "K2:"),
                    (r["pos_frac"] * 100, r["median_net"] * 100),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.7); ax.axvline(50, color="k", lw=0.7, ls="--")
    ax.set_xlabel("正收益模型占比 % (24 个 score 集)")
    ax.set_ylabel("中位净累计收益 %")
    ax.set_title("策略普适性: 中位收益 vs 正收益占比 (气泡=H1/H2双正占比)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_universality.png"), dpi=130)
    plt.close(fig)

    # fig2: 各组最优策略的中位收益/正占比条形
    gbest = agg.sort_values("median_net", ascending=False).drop_duplicates("group").head(12)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(gbest))
    ax.bar(x, gbest["median_net"] * 100, color="#4c7bd9", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(gbest["cfg"], rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("中位净累计收益 %")
    ax.set_title("各组内最优策略 (按中位净收益)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_group_best.png"), dpi=130)
    plt.close(fig)


def main():
    df = load_batteries()
    df["_rank"] = df.groupby("set")["cum_trade"].rank(ascending=False, method="min")
    print(f"载入 {len(df)} 行, {df['set'].nunique()} 个 score 集, {df['cfg'].nunique()} 个策略")

    agg = aggregate(df)
    agg.to_csv(os.path.join(RESULTS, "agg_strategies.csv"), index=False)

    pm = per_model_best(df)
    pm.to_csv(os.path.join(RESULTS, "per_model_best.csv"), index=False)

    sm = sellmode_summary()
    if sm is not None:
        sm.to_csv(os.path.join(RESULTS, "sellmode_summary.csv"), index=False)

    make_figures(df, agg)

    # ---- 控制台摘要 ----
    print("\n== 普适性 Top20 (U 得分, 以相对基线增量为核心) ==")
    show_cols = ["cfg", "group", "U", "median_net", "median_vs_a01", "beat_a01_frac",
                 "pos_frac", "h1h2_frac", "median_maxdd", "mean_trades"]
    print(md_table(agg.head(20), show_cols, None))
    print("\n== 按中位净收益 Top10 ==")
    top10 = agg.sort_values("median_net", ascending=False).head(10)
    print(md_table(top10, ["cfg", "group", "median_net", "mean_net", "pos_frac",
                           "h1h2_frac", "median_maxdd"]))
    print("\n== 相对基线 (vs A01 每日换仓) 增量 Top10 ==")
    topd = agg.sort_values("median_vs_a01", ascending=False).head(10)
    print(md_table(topd, ["cfg", "group", "median_vs_a01", "beat_a01_frac",
                          "median_net", "pos_frac"]))
    if sm is not None:
        print("\n== 卖出口径对照 ==")
        print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
