#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter12_full_verify.py — 新策略 (混合体慢腿 / 双腿结构) 全模型普适性验证

用户问题: 新策略在"目前所有有结果的模型"上是否大部分 (尤其 IC 偏高者) 表现好?
全集 = 官方 24 矩阵集 + V19_meta + V24_ensw2 + V20_ensw2 + V20c_ensw2
     + V13b_allz + V13c_allz = 30 个打分集 (全部已含 Test 窗 20250901~20260901 打分)。

对每个打分集跑:
  B0   基线: Top1 隔日换仓
  D01  快腿: Top1 + hold5 + sl8
  HYB  新慢腿: Top2 + hold≤20 + 排名>300退出(min_hold2) + 移动止盈15%   ← 新策略
  PAIR 双腿 3 只: 0.5×[该集+HYB] + 0.5×[V11_ensw2+D01] (快腿固定 V11_ensw2)

输出: results/iter12_full_verify.csv + 控制台分组汇总 (高IC组/低IC组/全集)
用法: python3 iter12_full_verify.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import SCORE_SETS, load_calendar, load_market, run_backtest, \
    load_scores  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for k, p in [("V18_f1", "Model/V18/model_pred/v18_f1.fea"),
             ("V18_f2", "Model/V18/model_pred/v18_f2.fea"),
             ("V19_meta", "Model/V19/model_pred/2026q3/score_meta.fea"),
             ("V20_ensw2", "Model/V20/model_pred/2026q3/score_ens_w2.fea"),
             ("V20b_ensw2", "Model/V20b/model_pred/2026q3/score_ens_w2.fea"),
             ("V20c_ensw2", "Model/V20c/model_pred/2026q3/score_ens_w2.fea"),
             ("V24_ensw2", "Model/V24/model_pred/2026q3/score_ens_w2.fea"),
             ("V13b_allz", "Model/V13b/model_pred/2026q3/all_zscore_score.fea"),
             ("V13c_allz", "Model/V13c/model_pred/2026q3/all_zscore_score.fea")]:
    SCORE_SETS[k] = p

# ---- 30 个有结果的打分集 ----
SETS = ["V1_allz", "V2_allz", "V3_allz", "V4_allz", "V5_allz", "V6_allz",
        "V7_allz", "V8_allz", "V9_allz", "V10_allz",
        "V11_allz", "V11_mixA", "V11_mixB", "V11_mixC", "V11_ensw2",
        "V12_allz", "V13_allz", "V13b_allz", "V13c_allz", "V14_allz",
        "V15_allz", "V16_allz", "V17_allz",
        "V18_f1", "V18_f2", "V19_meta",
        "V20_ensw2", "V20b_ensw2", "V20c_ensw2", "V24_ensw2"]

# 实测 1d RankIC (Test 窗, 各单元 model_pic 官方值; None = 无文档值)
IC1D = {
    "V1_allz": None, "V2_allz": 0.0197, "V3_allz": -0.0022, "V4_allz": -0.0135,
    "V5_allz": 0.0116, "V6_allz": 0.0363, "V7_allz": 0.0410, "V8_allz": 0.0444,
    "V9_allz": 0.0453, "V10_allz": 0.0448, "V11_allz": 0.0446,
    "V11_mixA": None, "V11_mixB": None, "V11_mixC": None,
    "V11_ensw2": 0.0422, "V12_allz": 0.0448, "V13_allz": 0.0357,
    "V13b_allz": 0.0371, "V13c_allz": 0.0332, "V14_allz": 0.0418,
    "V15_allz": 0.0102, "V16_allz": 0.0347, "V17_allz": 0.0363,
    "V18_f1": 0.0645, "V18_f2": 0.0617, "V19_meta": 0.0505,
    "V20_ensw2": 0.0429, "V20b_ensw2": 0.0406, "V20c_ensw2": 0.0401,
    "V24_ensw2": None,
}
IC_FAM = {"V11_mixA": "V11族≈0.044", "V11_mixB": "V11族≈0.044",
          "V11_mixC": "V11族≈0.044", "V24_ensw2": "V20配方≈0.043",
          "V1_allz": "无文档值"}

B0 = dict(top_n=1, hold=1)
D01 = dict(top_n=1, hold=5, stop_loss=0.08)
HYB = dict(top_n=2, hold=20, exit_rank=300, min_hold=2, trail_pct=0.15)


def metrics_of(eq):
    r = eq["ret"].dropna()
    n = len(r)
    cum = float((1 + r).prod() - 1)
    sh = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq["equity"].values)
    dd = float((eq["equity"].values / peak - 1).min())
    h1 = float((1 + r.iloc[:n // 2]).prod() - 1)
    h2 = float((1 + r.iloc[n // 2:]).prod() - 1)
    pos = sum(1 for i in range(60, len(r) + 1)
              if (1 + r.iloc[i - 60:i]).prod() - 1 > 0)
    return dict(cum=cum, sharpe=sh, maxdd=dd, h1=h1, h2=h2,
                roll_pct=pos / max(1, len(r) - 59))


def main():
    market = load_market()
    tds, tdi = load_calendar()
    # 快腿 V11_ensw2+D01 equity (固定, 只算一次)
    _, _, eq_fast = run_backtest(load_scores("V11_ensw2"), D01,
                                 market, tds, tdi)
    eq_fast = eq_fast.set_index("date")["equity"]

    rows = []
    for s in SETS:
        sc = load_scores(s)
        _, _, eq0 = run_backtest(sc, B0, market, tds, tdi)
        m1, _, _ = run_backtest(sc, D01, market, tds, tdi)
        m2, _, eq2 = run_backtest(sc, HYB, market, tds, tdi)
        # 双腿 3 只组合
        eq2s = eq2.set_index("date")["equity"]
        j = eq2s.index.intersection(eq_fast.index)
        comb = (0.5 * eq2s.loc[j] + 0.5 * eq_fast.loc[j]).to_frame()
        comb.columns = ["equity"]
        comb["ret"] = comb["equity"].pct_change()
        mp = metrics_of(comb)
        rows.append(dict(set=s,
                         ic1d=IC1D[s],
                         b0=float((1 + eq0["ret"].dropna()).prod() - 1),
                         d01=m1["cum_net"],
                         hyb_cum=m2["cum_net"], hyb_dd=m2["maxdd"],
                         hyb_h1=m2["h1_cum"], hyb_h2=m2["h2_cum"],
                         hyb_n=m2["n_trades"],
                         pair_cum=mp["cum"], pair_dd=mp["maxdd"],
                         pair_sh=mp["sharpe"], pair_h1=mp["h1"],
                         pair_h2=mp["h2"], pair_roll=mp["roll_pct"]))
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "iter12_full_verify.csv")
    df.to_csv(out, index=False)

    hi = df[df["ic1d"].notna() & (df["ic1d"] >= 0.040)]
    rest = df[~(df["ic1d"].notna() & (df["ic1d"] >= 0.040))]
    for tag, g in [("全集 30", df), ("高IC组 (实测≥0.040, %d)" % len(hi), hi),
                   ("其余/低IC组 (%d)" % len(rest), rest)]:
        print(f"\n===== {tag} =====")
        print(f"  HYB 慢腿:  正收益 {int((g.hyb_cum>0).sum())}/{len(g)}, "
              f"中位 {g.hyb_cum.median():+.1%}, 中位MaxDD {g.hyb_dd.median():.1%}, "
              f"H1/H2双正 {int(((g.hyb_h1>0)&(g.hyb_h2>0)).sum())}/{len(g)}")
        print(f"  PAIR 双腿3只: 正收益 {int((g.pair_cum>0).sum())}/{len(g)}, "
              f"中位 {g.pair_cum.median():+.1%}, 中位MaxDD {g.pair_dd.median():.1%}, "
              f"中位Sharpe {g.pair_sh.median():.2f}, H1/H2双正 "
              f"{int(((g.pair_h1>0)&(g.pair_h2>0)).sum())}/{len(g)}")
        print(f"  超基线B0(隔日换仓): HYB {int((g.hyb_cum>g.b0).sum())}/{len(g)}, "
              f"PAIR {int((g.pair_cum>g.b0).sum())}/{len(g)}")

    pd.set_option("display.width", 260)
    cols = ["set", "ic1d", "b0", "d01", "hyb_cum", "hyb_dd", "hyb_h1",
            "hyb_h2", "pair_cum", "pair_dd", "pair_sh", "pair_h1", "pair_h2"]
    print("\n===== 逐模型明细 (按 ic1d 降序) =====")
    df["ic_sort"] = df["ic1d"].fillna(-1)
    print(df.sort_values("ic_sort", ascending=False)[cols]
          .round(4).to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
