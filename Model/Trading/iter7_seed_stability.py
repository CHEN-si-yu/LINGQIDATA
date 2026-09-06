#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter7_seed_stability.py — V20 种子稳定性验证 (V20b / V20c 单元)

对照三个独立训练单元 (V20 base seed=3253 / V20b=7777 / V20c=12345, 16 折各, 其余
逐字节相同) 在其适配协议 h20tr20_t2 上的稳定性, 并给 D01/hold20_t1/S2 作对照。

指标口径与 iter5_combos.py 完全一致 (真实净值 cum_net, 含成本, 官方 Test 窗
20250901~20260901):
  cum/sharpe/maxdd/h1/h2/t1..t3/win/n/滚动60日正率 (metrics_of 复制自 iter5)
种子稳定性专门指标:
  - 逐日 Top2 选股重叠 (两两 Jaccard + 三者同日全同占比 + 三者公共成员均值)
  - h20tr20_t2 成交 (buy_dt×code) 两两 Jaccard
  - h20tr20_t2 日净值收益两两 Pearson 相关
  - ens3 = 三种子逐日截面 z 平均 (V20+V20b+V20c) → h20tr20_t2 / D01 对照行

输出: results/seed_stability.csv (unit×proto 指标, 与 v24_matrix.csv 同构)
      results/seed_stability_overlap.csv (重叠/相关/ens3 汇总)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import SCORE_SETS, load_scores, load_calendar, load_market, \
    run_backtest  # noqa: E402

RESULTS = os.path.join(HERE, "results")

PROTO = {
    "h20tr20_t2": dict(top_n=2, hold=20, trail_pct=0.20),
    "D01": dict(top_n=1, hold=5, stop_loss=0.08),
    "hold20_t1": dict(top_n=1, hold=20),
    "S2": dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
               stop_loss=0.08, trail_pct=0.15),
}
UNITS = ["V20_ensw2", "V20b_ensw2", "V20c_ensw2"]
UNIT_DIR = {"V20_ensw2": "Model/V20", "V20b_ensw2": "Model/V20b",
            "V20c_ensw2": "Model/V20c"}
MAIN_PROTO = "h20tr20_t2"


# ---------- 指标 (复制自 iter5_combos.py, 保证口径一致) ----------
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


def run_unit(scores, pname):
    m, tr, eq = run_backtest(scores, PROTO[pname], market, tds, tdi)
    if m is None:
        return None, None, None
    return tr, eq, metrics_of(eq, tr, PROTO[pname]["top_n"])


def main():
    global market, tds, tdi
    market = load_market()
    tds, tdi = load_calendar()

    # 注册 V20b/V20c 打分集 (V20 已在 engine.SCORE_SETS)
    for u in UNITS:
        p = f"{UNIT_DIR[u]}/model_pred/2026q3/score_ens_w2.fea"
        if u in SCORE_SETS:
            assert SCORE_SETS[u] == p, f"{u} 已注册但路径不同: {SCORE_SETS[u]}"
        else:
            SCORE_SETS[u] = p
            print(f"[iter7] 注册打分集 {u} → {p}", flush=True)

    S = {u: load_scores(u) for u in UNITS}
    for u in UNITS:
        print(f"[iter7] {u}: {len(S[u]['dates'])} 个因子日"
              f" ({S[u]['dates'][0]}~{S[u]['dates'][-1]})", flush=True)
    common_dates = set(S["V20_ensw2"]["dates"]) & set(S["V20b_ensw2"]["dates"]) \
        & set(S["V20c_ensw2"]["dates"])
    print(f"[iter7] 三单元公共因子日: {len(common_dates)}", flush=True)

    # ---- 1) unit × proto 主表 ----
    rows = []
    for u in UNITS:
        for pname in PROTO:
            tr, eq, mt = run_unit(S[u], pname)
            if tr is None:
                continue
            rows.append(dict(unit=u, proto=pname, **mt))
    df = pd.DataFrame(rows)
    df["min_h"] = df[["h1", "h2"]].min(axis=1)
    df["min_t"] = df[["t1", "t2", "t3"]].min(axis=1)
    df["roll_pct"] = df["roll_pos"] / df["roll_n"]
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "seed_stability.csv"), index=False)
    print("\n===== 1) unit × proto 指标 (真实净值) =====")
    print(df.to_string(index=False))

    # ---- 2) 逐日 Top2 重叠 ----
    main_scores = S
    dlist = sorted(common_dates)
    top2 = {u: {d: set(main_scores[u]["ranked"][d][:2]) for d in dlist}
            for u in UNITS}
    pair_jac, pair_same = {}, {}
    for a in UNITS:
        for b in UNITS:
            if a >= b:
                continue
            j = [len(top2[a][d] & top2[b][d]) / len(top2[a][d] | top2[b][d])
                 for d in dlist]
            same = [top2[a][d] == top2[b][d] for d in dlist]
            pair_jac[(a, b)] = float(np.mean(j))
            pair_same[(a, b)] = float(np.mean(same))
    a, b, c = UNITS  # 三单元固定解包 (pair 循环泄漏的 a/b 不可依赖)
    tri_same = float(np.mean([top2[a][d] == top2[b][d] == top2[c][d]
                              for d in dlist]))
    tri_common = float(np.mean(
        [len(top2[a][d] & top2[b][d] & top2[c][d]) / 2.0 for d in dlist]))
    print("\n===== 2) 逐日 Top2 重叠 (主协议打分, 公共因子日) =====")
    for (a, b), j in pair_jac.items():
        print(f"  {a} vs {b}: 平均 Jaccard={j:.3f}  同日全同占比={pair_same[(a,b)]:.1%}")
    print(f"  三者同日 Top2 全同占比 = {tri_same:.1%}")
    print(f"  三者公共成员均值 (0~2) = {tri_common:.3f}")

    # ---- 3) 成交重叠 + 日净值相关 (h20tr20_t2) ----
    trs, eqs = {}, {}
    for u in UNITS:
        trs[u], eqs[u], _ = run_unit(S[u], MAIN_PROTO)
    print(f"\n===== 3) {MAIN_PROTO} 成交/净值重叠 =====")
    buy_sets = {u: set(zip(trs[u]["buy_dt"], trs[u]["code"])) for u in UNITS}
    for a in UNITS:
        for b in UNITS:
            if a >= b:
                continue
            ja = len(buy_sets[a] & buy_sets[b]) / len(buy_sets[a] | buy_sets[b])
            print(f"  成交 (buy_dt×code) Jaccard {a} vs {b} = {ja:.3f}"
                  f"  ({len(buy_sets[a] & buy_sets[b])} 笔共同 / "
                  f"{len(buy_sets[a])}+{len(buy_sets[b])} 笔)")
    ret_piv = pd.DataFrame({u: eqs[u].set_index("date")["ret"] for u in UNITS})
    corr = ret_piv.corr()
    print("  日净值收益 Pearson 相关:")
    print(corr.to_string())

    # ---- 4) ens3 (三种子逐日截面 z 平均) ----
    ens = build_ens3(S, dlist)
    for pname in (MAIN_PROTO, "D01"):
        tr, eq, mt = run_unit(ens, pname)
        rows.append(dict(unit="V20_ens3", proto=pname, **mt))
        print(f"\n===== 4) ens3 ({pname}) =====")
        print(f"  cum={mt['cum']:.3f} sharpe={mt['sharpe']:.2f} "
              f"maxdd={mt['maxdd']:.3f} h1={mt['h1']:.3f} h2={mt['h2']:.3f} "
              f"t1={mt['t1']:.3f} t2={mt['t2']:.3f} t3={mt['t3']:.3f} "
              f"win={mt['win']:.2%} n={mt['n']} roll={mt['roll_pos']}/{mt['roll_n']}")

    # ---- 汇总落盘 ----
    df2 = pd.DataFrame(rows)
    df2["min_h"] = df2[["h1", "h2"]].min(axis=1)
    df2["min_t"] = df2[["t1", "t2", "t3"]].min(axis=1)
    df2["roll_pct"] = df2["roll_pos"] / df2["roll_n"]
    df2.to_csv(os.path.join(RESULTS, "seed_stability.csv"), index=False)

    ov = []
    for (a, b), j in pair_jac.items():
        ov.append(dict(kind="top2_jaccard", pair=f"{a}|{b}", value=j,
                       note=f"同日全同占比 {pair_same[(a,b)]:.1%}"))
    ov.append(dict(kind="top2_triple_same", pair="ALL", value=tri_same,
                   note="三者同日 Top2 全同"))
    ov.append(dict(kind="top2_triple_common", pair="ALL", value=tri_common,
                   note="三者公共成员 0~2 均值"))
    for a in UNITS:
        for b in UNITS:
            if a >= b:
                continue
            ja = len(buy_sets[a] & buy_sets[b]) / len(buy_sets[a] | buy_sets[b])
            ov.append(dict(kind="trade_jaccard", pair=f"{a}|{b}", value=ja,
                           note=f"{MAIN_PROTO} 成交 {len(buy_sets[a] & buy_sets[b])} 笔共同"))
            ov.append(dict(kind="ret_corr", pair=f"{a}|{b}",
                           value=float(corr.loc[a, b]),
                           note=f"{MAIN_PROTO} 日净值收益相关"))
    pd.DataFrame(ov).to_csv(os.path.join(RESULTS, "seed_stability_overlap.csv"),
                            index=False)
    print(f"\n[iter7] 已写入: {RESULTS}/seed_stability.csv + seed_stability_overlap.csv")


def build_ens3(S, dlist):
    """ens3 = 逐日截面 z(三单元) 平均 → load_scores 同构 dict。"""
    codes_all = sorted({c for d in dlist for u in UNITS
                        for c in S[u]["scores"][d]})
    scores, ranked, top1 = {}, {}, {}
    for d in dlist:
        zs = []
        for u in UNITS:
            row = S[u]["scores"][d]
            v = np.array([row.get(c, 0.0) for c in codes_all])
            sd = v.std()
            zs.append((v - v.mean()) / sd if sd > 1e-12
                      else np.zeros_like(v))
        m = np.mean(zs, axis=0)
        scores[d] = {c: float(x) for c, x in zip(codes_all, m)}
        idx = np.argsort(-m)
        ranked[d] = [codes_all[i] for i in idx]
        top1[d] = float(m[idx[0]])
    return dict(scores=scores, ranked=ranked, top1=top1, dates=dlist)


if __name__ == "__main__":
    main()
