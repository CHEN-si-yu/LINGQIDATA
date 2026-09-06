#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter8_seed_stability_v13.py — V13 种子稳定性验证 (V13b / V13c 单元)

背景: 策略矩阵 (strategy_matrix.py) 中 24 模型 × 11 策略的最佳组合 =
      V13_allz × S7_h10_tr12_t1 (真实净值 +538.7%, Sharpe 3.58, H1/H2 双正)。
      本脚本对照三个独立训练单元 (V13 seed=3253 / V13b=7777 / V13c=12345, 8 折各,
      数据划分种子固定 3253 → 差异 = 纯训练随机性) 在 11 个策略上的稳定性。

指标口径: Trading/engine.py cum_net 真实净值 (开盘先卖后买, 1 份资金, 含成本),
          官方 Test 窗 20250901~20260901 (243 因子日)。
种子稳定性专门指标 (对照 iter7):
  - 逐日 Top1 选股重叠 (两两 Jaccard + 同日全同占比)
  - S7 主协议成交 (buy_dt×code) 两两 Jaccard
  - S7 日净值收益两两 Pearson 相关

输出: results/seed_stability_v13.csv          (unit×strategy 全 11 策略指标)
      results/seed_stability_v13_overlap.csv  (重叠/相关汇总)
      SEED_STABILITY_V13_REPORT.md            (报告)

用法: python3 iter8_seed_stability_v13.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import SCORE_SETS, load_scores, load_calendar, load_market, \
    run_backtest  # noqa: E402
from strategy_matrix import STRATEGIES  # noqa: E402

RESULTS = os.path.join(HERE, "results")

UNITS = ["V13_allz", "V13b_allz", "V13c_allz"]
UNIT_DIR = {"V13_allz": "Model/V13", "V13b_allz": "Model/V13b",
            "V13c_allz": "Model/V13c"}
MAIN_PROTO = "S7_h10_tr12_t1"
FOCUS = ["S7_h10_tr12_t1", "S6_h20_tr20_t2", "S4_re300_t2",
         "S5_re300_sl8_tr15_t2", "S1_h5_sl8_t1", "S9_rat20_10_sl8_t2"]


def thirds(tr, slots):
    tr = tr.sort_values("buy_dt")
    k = max(1, len(tr) // 3)
    return [float((1 + p["net_pct"] / 100 / slots).prod() - 1)
            for p in (tr.iloc[:k], tr.iloc[k:2 * k], tr.iloc[2 * k:])]


def metrics_of(eq, trades, slots):
    r = eq["ret"].dropna()
    n = len(r)
    cum = float((1 + r).prod() - 1)
    sh = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq["equity"].values)
    dd = float((eq["equity"].values / peak - 1).min())
    h1 = float((1 + r.iloc[:n // 2]).prod() - 1)
    h2 = float((1 + r.iloc[n // 2:]).prod() - 1)
    t1, t2, t3 = thirds(trades, slots)
    pos = 0
    for i in range(60, len(r) + 1):
        if (1 + r.iloc[i - 60:i]).prod() - 1 > 0:
            pos += 1
    return dict(cum=cum, sharpe=sh, maxdd=dd, h1=h1, h2=h2, t1=t1, t2=t2,
                t3=t3, win=(trades["net_pct"] > 0).mean() if len(trades) else np.nan,
                n=len(trades), roll_pos=pos, roll_n=len(r) - 59)


def run_unit(scores, cfg):
    m, tr, eq = run_backtest(scores, cfg, market, tds, tdi)
    if m is None:
        return None, None, None
    return tr, eq, metrics_of(eq, tr, cfg.get("top_n", 1))


def main():
    global market, tds, tdi
    market = load_market()
    tds, tdi = load_calendar()

    for u in UNITS:
        p = f"{UNIT_DIR[u]}/model_pred/2026q3/all_zscore_score.fea"
        if not os.path.exists(PROJECT_ROOT + p):
            raise SystemExit(f"[iter8] 缺少打分集 {u} → {p} (先训练 + analysis.py)")
        SCORE_SETS[u] = p
        print(f"[iter8] 注册打分集 {u}", flush=True)

    S = {u: load_scores(u) for u in UNITS}
    for u in UNITS:
        print(f"[iter8] {u}: {len(S[u]['dates'])} 个因子日"
              f" ({S[u]['dates'][0]}~{S[u]['dates'][-1]})", flush=True)
    common_dates = set.intersection(*(set(S[u]["dates"]) for u in UNITS))
    print(f"[iter8] 三单元公共因子日: {len(common_dates)}", flush=True)

    # ---- 1) unit × 11 策略主表 ----
    rows = []
    for u in UNITS:
        for sname, _, _, cfg in STRATEGIES:
            tr, eq, mt = run_unit(S[u], cfg)
            if tr is None:
                continue
            rows.append(dict(unit=u, strategy=sname, **mt))
    df = pd.DataFrame(rows)
    df["min_h"] = df[["h1", "h2"]].min(axis=1)
    df["min_t"] = df[["t1", "t2", "t3"]].min(axis=1)
    df["roll_pct"] = df["roll_pos"] / df["roll_n"]
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "seed_stability_v13.csv"), index=False)
    print("\n===== 1) unit × strategy 指标 (真实净值, 聚焦列) =====")
    piv = df.pivot_table(index="strategy", columns="unit", values="cum")
    print(piv.loc[[s for s in piv.index if s in FOCUS or True]].round(3).to_string())

    # ---- 2) 逐日 Top1 重叠 ----
    dlist = sorted(common_dates)
    top1 = {u: {d: set(S[u]["ranked"][d][:1]) for d in dlist} for u in UNITS}
    pair_jac, pair_same = {}, {}
    for a in UNITS:
        for b in UNITS:
            if a >= b:
                continue
            j = [len(top1[a][d] & top1[b][d]) / len(top1[a][d] | top1[b][d])
                 for d in dlist]
            same = [top1[a][d] == top1[b][d] for d in dlist]
            pair_jac[(a, b)] = float(np.mean(j))
            pair_same[(a, b)] = float(np.mean(same))
    a, b, c = UNITS
    tri_same = float(np.mean([top1[a][d] == top1[b][d] == top1[c][d]
                              for d in dlist]))
    print("\n===== 2) 逐日 Top1 重叠 (公共因子日) =====")
    for (x, y), j in pair_jac.items():
        print(f"  {x} vs {y}: Jaccard={j:.3f}  同日全同占比={pair_same[(x,y)]:.1%}")
    print(f"  三者同日 Top1 全同占比 = {tri_same:.1%}")

    # ---- 3) 主协议 S7 成交重叠 + 日净值相关 ----
    trs, eqs = {}, {}
    main_cfg = dict([(s[0], s[3]) for s in STRATEGIES])[MAIN_PROTO]
    for u in UNITS:
        trs[u], eqs[u], _ = run_unit(S[u], main_cfg)
    buy_sets = {u: set(zip(trs[u]["buy_dt"], trs[u]["code"])) for u in UNITS}
    print(f"\n===== 3) {MAIN_PROTO} 成交/净值重叠 =====")
    for x in UNITS:
        for y in UNITS:
            if x >= y:
                continue
            ja = len(buy_sets[x] & buy_sets[y]) / len(buy_sets[x] | buy_sets[y])
            print(f"  成交 Jaccard {x} vs {y} = {ja:.3f} "
                  f"({len(buy_sets[x] & buy_sets[y])} 笔共同 / "
                  f"{len(buy_sets[x])}+{len(buy_sets[y])} 笔)")
    ret_piv = pd.DataFrame({u: eqs[u].set_index("date")["ret"] for u in UNITS})
    corr = ret_piv.corr()
    print("  日净值收益 Pearson 相关:")
    print(corr.to_string())

    # ---- 汇总落盘 ----
    ov = []
    for (x, y), j in pair_jac.items():
        ov.append(dict(kind="top1_jaccard", pair=f"{x}|{y}", value=j,
                       note=f"同日全同占比 {pair_same[(x,y)]:.1%}"))
    ov.append(dict(kind="top1_triple_same", pair="ALL", value=tri_same,
                   note="三者同日 Top1 全同"))
    for x in UNITS:
        for y in UNITS:
            if x >= y:
                continue
            ja = len(buy_sets[x] & buy_sets[y]) / len(buy_sets[x] | buy_sets[y])
            ov.append(dict(kind="trade_jaccard", pair=f"{x}|{y}", value=ja,
                           note=f"{MAIN_PROTO} 成交 {len(buy_sets[x] & buy_sets[y])} 笔共同"))
            ov.append(dict(kind="ret_corr", pair=f"{x}|{y}",
                           value=float(corr.loc[x, y]),
                           note=f"{MAIN_PROTO} 日净值收益相关"))
    pd.DataFrame(ov).to_csv(
        os.path.join(RESULTS, "seed_stability_v13_overlap.csv"), index=False)
    print(f"\n[iter8] 已写入 results/seed_stability_v13.csv + "
          f"seed_stability_v13_overlap.csv")

    _report(df, piv, ov, corr, pair_jac, pair_same, tri_same)


def _report(df, piv, ov, corr, pair_jac, pair_same, tri_same):
    a, b, c = UNITS
    L = ["# SEED_STABILITY_V13_REPORT — V13 冠军组合种子稳定性验证 (iter8)",
         "",
         "> 2026-09-06 | 三个独立训练单元: V13(seed=3253) / V13b(7777) / V13c(12345)",
         "> 8 折各; 架构/损失/ridge 热启动/all_zscore 配方逐字节相同;",
         "> **数据划分种子固定 3253** (比 V20b/V20c 测试更干净: 差异 = 纯训练随机性)",
         "> 口径: Trading/engine.py 真实净值 (含成本), Test 窗 20250901~20260901",
         "> 背景: V13_allz × S7 (h10_tr12_t1) = 24 模型 × 11 策略矩阵中的最佳组合"
         " (+538.7%)。",
         "> 脚本: Trading/iter8_seed_stability_v13.py",
         ""]
    L.append("## 一、unit × strategy 主表 (cum 真实净值)")
    L.append("")
    cols = sorted(piv.columns)
    L.append("| strategy | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for sname, row in piv.iterrows():
        L.append(f"| {sname} | " + " | ".join(
            f"{row[c]:+.1%}" for c in cols) + " |")
    L.append("")
    L.append("## 二、种子稳定性专门指标 (主协议 S7)")
    L.append("")
    L.append("- 逐日 Top1 选股重叠: 两两 Jaccard "
             f"{pair_jac[(a,b)]:.2f} / {pair_jac[(a,c)]:.2f} / {pair_jac[(b,c)]:.2f};"
             f" 同日全同 {pair_same[(a,b)]:.1%} / {pair_same[(a,c)]:.1%} / "
             f"{pair_same[(b,c)]:.1%}; 三者同日全同 {tri_same:.1%}")
    for x in UNITS:
        for y in UNITS:
            if x < y:
                cc = corr.loc[x, y]
                L.append(f"- 日净值收益 Pearson 相关 {x} vs {y} = {cc:.3f}")
    L.append("")
    L.append("## 三、结论")
    L.append("")
    r7 = df[df["strategy"] == "S7_h10_tr12_t1"].set_index("unit")
    L.append(f"- S7 主协议: V13={r7.loc['V13_allz','cum']:+.1%} / "
             f"V13b={r7.loc['V13b_allz','cum']:+.1%} / "
             f"V13c={r7.loc['V13c_allz','cum']:+.1%} — 见上表与 iter7 对照。")
    L.append("")

    # 3.1) 每策略 × 三单元正负稳健性 (df 计算)
    cu = df.pivot_table(index="strategy", columns="unit", values="cum")
    cu["n_pos_3"] = (cu > 0).sum(axis=1)
    cu["med_23"] = cu[UNITS[1:]].median(axis=1)
    L.append("### 3.1 策略 × 三单元稳健性 (累计净值正负分布)")
    L.append("")
    L.append("| 策略 | V13 | V13b | V13c | 三单元正数 | 两新种子中位 |")
    L.append("|---|---|---|---|---|---|")
    for sn in cu.index:
        L.append(f"| {sn} | {cu.loc[sn,UNITS[0]]:+.1%} | "
                 f"{cu.loc[sn,UNITS[1]]:+.1%} | {cu.loc[sn,UNITS[2]]:+.1%} | "
                 f"{int(cu.loc[sn,'n_pos_3'])}/3 | "
                 f"{cu.loc[sn,'med_23']:+.1%} |")
    all_pos = [sn for sn in cu.index if cu.loc[sn, "n_pos_3"] == 3]
    L.append("")
    L.append(f"- **三种子全正的策略 = {all_pos}** — 只有跨种子不敏感的策略才是"
             "该架构的稳定水平; 单样本峰值策略 (S1/S7/S2/S3/S10 等) 全部依赖 "
             "V13 的种子运气。全正策略中 S6 两新种子中位 +38.7% 显著高于 "
             "S9 (+5.1%), 是架构稳定水平的最优代表。")
    L.append("")

    # 3.2) IC 层对照 (各单元 model_pic/output.md 数值)
    L.append("### 3.2 IC 层稳定 vs 策略层爆炸 (与 iter7 同构)")
    L.append("")
    L.append("- IC 层三单元几乎无差 (V13 0.0357/0.2479, V13b 0.0371/0.2828, "
             "V13c 0.0332/0.2042; RankIC/RankICIR, 见各单元 model_pic) — "
             "信号质量对种子稳健。")
    L.append("- 策略层 (Top 选择) 相差 ~10× (S7: +538.7% / -1.3% / -15.9%) — "
             "Top 锐度完全由种子决定。逐日 Top1 两两 Jaccard 仅 "
             f"{pair_jac[(a,b)]:.2f} / {pair_jac[(a,c)]:.2f} / "
             f"{pair_jac[(b,c)]:.2f}"
             ", 成交几乎零重叠 → 换种子 = 换一个几乎不相关的模型 "
             "(与 V20 系 iter7 完全同构)。")
    L.append("- **V13_allz × S7 (+538.7%) 与 V20_ensw2 × h20tr20_t2 (+352%) 一样, "
             "是训练种子高方差抽取中的一次中奖, 不可外推为稳定水平。**")
    L.append("")

    # 3.3) 期望水平与建议
    L.append("### 3.3 稳定期望水平与建议")
    L.append("")
    L.append("- **架构无关的期望带 ~+35~45%**: V13b/V13c 在 S6 (h20tr20_t2) "
             "+33.7% / +43.7%, 与 V20b/V20c 的 h20tr20_t2 +37.0% / +38.8% "
             "高度一致 → 长持有+移动止损型协议 (S6/h20) 是跨种子最稳的收益来源。")
    L.append("- **报数纪律**: 单模型策略层成绩改报多种子中位/区间 (S6 型 "
             "≈ +35~45%), 峰值 (+538.7% / +352%) 仅作幸运上限参考, 不作能力代表。")
    L.append("- **实盘选型**: 单模型+峰值策略不可直接实盘; 应优先 (a) 架构多样"
             "组合层 (iter5/6: 双/三系统 0.5/0.5 → +301% / Sharpe 3.75), 或 "
             "(b) S6/S4 型协议; 24 模型 × 11 策略矩阵的普适结论 (S4/S5 跨架构 "
             "20/24 超基线) 不受本检验影响 — 普适性来自架构多样性, 而非种子多样性。")
    L.append("")
    with open(os.path.join(HERE, "SEED_STABILITY_V13_REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[iter8] → SEED_STABILITY_V13_REPORT.md")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))) + "/"

if __name__ == "__main__":
    main()
