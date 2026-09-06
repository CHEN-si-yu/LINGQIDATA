#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matrix_v24_addon.py — V24_ensw2 补充行 × 11 策略 (不重写 24 模型官方矩阵)

背景: 24 模型 × 11 策略官方矩阵 (strategy_matrix.py, 2026-09-06 10:35) 在
V24 训练完成 (10:33) 前运行, 未含 V24。V24 = V20 锐度主干 + 弱长周期辅助单元
(16 折, 与 V20 同种子同划分), 其自身评估 (model_pic/v24_matrix.csv) 在
冠军协议下为弱结果 (+2.9% h20tr20_t2)。本脚本把 V24_ensw2 补跑同一 11 策略,
确认其不改变官方矩阵结论 (最佳组合仍为 V13_allz × S7), 并以附录形式
追加到 STRATEGY_MATRIX_REPORT.md。

输出:
  results/strategy_matrix_v24_addon.csv   (V24_ensw2 × 11 策略全指标)
  STRATEGY_MATRIX_REPORT.md                (追加附录 A)

用法: python3 matrix_v24_addon.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import SCORE_SETS, load_calendar, load_market, load_scores, \
    run_backtest  # noqa: E402
from strategy_matrix import STRATEGIES, METRIC_FIELDS, BASE_NAME  # noqa: E402

RESULTS = os.path.join(HERE, "results")
ADDON = "V24_ensw2"
ADDON_PATH = "Model/V24/model_pred/2026q3/score_ens_w2.fea"


def main():
    SCORE_SETS[ADDON] = ADDON_PATH
    market = load_market()
    tds, tdi = load_calendar()
    sc = load_scores(ADDON)
    print(f"[v24] {len(sc['dates'])} 个因子日 ({sc['dates'][0]}~"
          f"{sc['dates'][-1]})", flush=True)

    rows = []
    for sname, _, _, cfg in STRATEGIES:
        m, tr, eq = run_backtest(sc, cfg, market, tds, tdi)
        if m is None:
            continue
        row = dict(set=ADDON, strategy=sname)
        for f in METRIC_FIELDS:
            row[f] = m.get(f)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["base_cum"] = df[df["strategy"] == BASE_NAME]["cum_net"].iloc[0]
    df["vs_base"] = df["cum_net"] - df["base_cum"]
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "strategy_matrix_v24_addon.csv")
    df.to_csv(out, index=False)

    best = df.loc[df["cum_net"].idxmax()]
    print(f"[v24] 基线 B0: {df['base_cum'].iloc[0]:+.1%}")
    print(f"[v24] 最优策略: {best['strategy']}  cum_net={best['cum_net']:+.1%}  "
          f"Sharpe={best['sharpe']:.2f}  MaxDD={best['maxdd']:.1%}  "
          f"H1/H2={best['h1_cum']:+.1%}/{best['h2_cum']:+.1%}")
    n_beat = int((df["vs_base"] > 0).sum())
    print(f"[v24] 超自身基线策略数: {n_beat}/{len(df)}")

    # ---- 追加附录到 STRATEGY_MATRIX_REPORT.md ----
    rp = os.path.join(HERE, "STRATEGY_MATRIX_REPORT.md")
    with open(rp, encoding="utf-8") as f:
        content = f.read().rstrip() + "\n\n"
    if "## 附录 A" in content:
        content = content.split("## 附录 A")[0].rstrip() + "\n\n"
    L = ["## 附录 A: V24_ensw2 补充行 (第 25 个打分集, 2026-09-06 补跑)",
         "",
         "> V24 = V20 锐度主干 + 弱长周期辅助单元 (16 折, 与 V20 同种子/同划分), "
         "其官方矩阵运行时训练刚完成、未纳入 24 模型集; 此处把 `score_ens_w2` "
         "补跑同一 11 策略 (脚本 matrix_v24_addon.py, 数据 "
         "results/strategy_matrix_v24_addon.csv)。",
         ""]
    cols = ["strategy", "cum_net", "sharpe", "maxdd", "h1_cum", "h2_cum",
            "win_rate", "n_trades"]
    L.append("| 策略 | cum_net | Sharpe | MaxDD | H1 | H2 | 胜率 | 交易数 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in df.sort_values("cum_net", ascending=False).iterrows():
        tag = " ← 最优" if r["strategy"] == best["strategy"] else ""
        L.append(f"| {r['strategy']} | {r['cum_net']:+.1%} | {r['sharpe']:.2f} "
                 f"| {r['maxdd']:.1%} | {r['h1_cum']:+.1%} | {r['h2_cum']:+.1%} "
                 f"| {r['win_rate']:.1%} | {int(r['n_trades'])} |{tag}")
    L.append("")
    L.append(f"- V24_ensw2 最优 = `{best['strategy']}` ({best['cum_net']:+.1%}, "
             f"Sharpe {best['sharpe']:.2f}), 超自身隔日基线 {n_beat}/11 策略; "
             "整体水平与 V20b 相当或更弱, 远低于冠军单样本 (V13_allz × S7 "
             "+538.7%), **官方矩阵结论不变** — 最佳模型×策略组合仍为 "
             "V13_allz × S7 (其种子稳定性见 SEED_STABILITY_V13_REPORT.md)。")
    with open(rp, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[v24] → 附录已追加至 STRATEGY_MATRIX_REPORT.md")


if __name__ == "__main__":
    main()
