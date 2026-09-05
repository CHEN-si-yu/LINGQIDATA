#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_figs.py — 图表产出 (英文标签, 避免 CJK 字体缺失)

fig3: 卖出口径对照 (open vs close_overlap vs close_seq) — V11/V20 ens_w2
fig4: 主推策略跨 24 模型净值叠加 + 冠军族中位曲线对比
fig5: 逐模型净收益对比 (主推 vs 每日换仓 vs hold5s8)
fig6: V20 ens_w2 上的策略对比 (旧冠军口径 vs 新开盘换仓口径)
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
EQ = os.path.join(BASE, "results", "equity")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load_eq(set_name, cfg):
    p = os.path.join(EQ, f"{set_name}__{cfg}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["eq"] = df["equity"] / df["equity"].iloc[0]
    return df


def plot_sellmode():
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for ax, s in zip(axes, ["V11_ensw2", "V20_ensw2"]):
        for cfg, color, ls, label in [
            ("SM_h5_sl8_open", "#d4552b", "-", "h5+sl8 open-sell (new, 1x capital)"),
            ("SM_h5_sl8_close_overlap", "#2b6cd4", "--", "h5+sl8 close-sell (old champ, needs 2x capital)"),
            ("SM_h5_sl8_close_seq", "#3a9d5d", "-.", "h5+sl8 close-sell, buy next open"),
            ("SM_daily_open", "#888888", ":", "daily rebalance (baseline)"),
        ]:
            eq = load_eq(s, cfg)
            if eq is None:
                continue
            ax.plot(eq["date"], eq["eq"], color=color, ls=ls, lw=1.4, label=label)
        ax.set_title(f"{s}: sell-timing comparison (net, costs incl.)")
        ax.set_ylabel("equity / 50k")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_sellmode.png"), dpi=130)
    plt.close(fig)


def plot_universal():
    prim = "S2_re300_sl8_tr15"
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax = axes[0]
    meds = []
    for s in sorted(glob.glob(os.path.join(EQ, f"*__{prim}.csv"))):
        df = pd.read_csv(s)
        df["eq"] = df["equity"] / df["equity"].iloc[0]
        ax.plot(df["date"], df["eq"], color="#9bb8d3", lw=0.8, alpha=0.7)
        meds.append(df.set_index("date")["eq"])
    med = pd.concat(meds, axis=1).median(axis=1)
    ax.plot(med.index, med, color="#d4552b", lw=2.2, label="median across 24 models")
    ax.set_title(f"{prim} (rank-exit300 + SL8% + trail15%, Top2, open roll) "
                 "equity across all 24 score sets")
    ax.set_ylabel("equity / 50k")
    ax.legend(loc="upper left")
    ax = axes[1]
    for cfg, color, label in [
        ("S2_re300_sl8_tr15", "#d4552b", "S2 rank300+SL8+trail15 (recommended)"),
        ("N2_re300", "#2b6cd4", "N2 rank300 only"),
        ("T2_re300_mh5", "#3a9d5d", "T2 rank300 minhold5"),
        ("U2_re300_h20", "#8e44ad", "U2 rank300 hold-cap20"),
        ("M2_h20_tr20", "#e67e22", "M2 hold20+trail20"),
        ("D01_h5_sl8_t1", "#7f8c8d", "D01 hold5+SL8 (V20-style)"),
        ("A01_daily_top1", "#b0b0b0", "A01 daily rebalance (baseline)"),
    ]:
        meds = []
        for s in sorted(glob.glob(os.path.join(EQ, f"*__{cfg}.csv"))):
            df = pd.read_csv(s)
            meds.append(df.set_index("date")["equity"] / df["equity"].iloc[0])
        med = pd.concat(meds, axis=1).median(axis=1)
        ax.plot(med.index, med, color=color, lw=1.8, label=label)
    ax.set_title("median equity across 24 models: champion family vs baselines")
    ax.set_ylabel("equity / 50k")
    ax.set_xlabel("date")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_universal.png"), dpi=130)
    plt.close(fig)


def plot_per_model():
    d = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(BASE, "results", "battery_B*.csv"))])
    rows = []
    for s in d["set"].unique():
        g = d[d["set"] == s].set_index("cfg")
        rows.append(dict(set=s, S2=g.loc["S2_re300_sl8_tr15", "cum_trade"],
                         N2=g.loc["N2_re300", "cum_trade"],
                         A01=g.loc["A01_daily_top1", "cum_trade"],
                         D01=g.loc["D01_h5_sl8_t1", "cum_trade"]))
    m = pd.DataFrame(rows).sort_values("S2").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(m))
    w = 0.2
    ax.bar(x - 1.5 * w, m["A01"] * 100, w, label="A01 daily (baseline)", color="#b0b0b0")
    ax.bar(x - 0.5 * w, m["D01"] * 100, w, label="D01 hold5+SL8 (V20-style)", color="#7f8c8d")
    ax.bar(x + 0.5 * w, m["N2"] * 100, w, label="N2 rank300", color="#2b6cd4")
    ax.bar(x + 1.5 * w, m["S2"] * 100, w, label="S2 rank300+SL8+trail15 (recommended)", color="#d4552b")
    ax.set_xticks(x)
    ax.set_xticklabels(m["set"], rotation=60, fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("net cumulative return %")
    ax.set_title("per-model net return: recommended strategy vs baselines (24 score sets)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_per_model.png"), dpi=130)
    plt.close(fig)


def plot_v20_focus():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cfg, color, ls, label in [
        ("SM_h5_sl8_close_overlap", "#2b6cd4", "--", "old champion: hold5+SL8 close-sell (2x capital)"),
        ("SM_h5_sl8_open", "#3a9d5d", "-.", "hold5+SL8 open-sell (1x capital)"),
        ("S2_re300_sl8_tr15", "#d4552b", "-", "S2 rank300+SL8+trail15 (recommended, 1x capital)"),
        ("A01_daily_top1", "#b0b0b0", ":", "daily rebalance (baseline)"),
    ]:
        eq = load_eq("V20_ensw2", cfg)
        if eq is not None:
            ax.plot(eq["date"], eq["eq"], color=color, ls=ls, lw=1.8, label=label)
    ax.set_title("V20_ensw2 (best model): strategy comparison on the champion score")
    ax.set_ylabel("equity / 50k")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_v20_focus.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    plot_sellmode()
    plot_universal()
    plot_per_model()
    plot_v20_focus()
    print("figures →", FIG)
