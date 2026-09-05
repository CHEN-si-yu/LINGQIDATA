#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter3_crossmodel.py — 跨模型综合评估 (不依赖单一协议/单一口径)

用户要求 (2026-09-05): V20 的极高表现利用了资金周转口径的缺陷, 不应作为衡量
其他模型的唯一标尺 → 本脚本给出**多维标尺矩阵**:
  A. IC 层: RankIC/IR × {1d,3d,5d,10d,20d,trail} (可交易池, Test 窗口)
  B. 策略层 (Trading 引擎真实净值 cum_net, 1 份资金, 开盘可执行):
     {D01(hold5s8), hold20_t1, hold20_t2, h20tr20_t2, S2_re300} × 全部打分集
  C. 稳定性: H1/H2 双正 + T1/T2/T3 无负段 (策略层主推协议口径)

打分集: V20_ensw2 / V21_ens21 / V22_10d / V22_20d / V23_ens23 (就绪后) /
        V11_ensw2 (历史冠军参照) / V18_v18 (GBDT 1d)
输出: results/iter3_crossmodel.csv + 打印矩阵
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import load_calendar, load_market, run_backtest

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ROOT = "/autodl-fs/data/lingqiData/Model/"

SETS = {
    "V20_ensw2": ROOT + "V20/model_pred/2026q3/score_ens_w2.fea",
    "V21_ens21": ROOT + "V21/model_pred/2026q3/score_ens21.fea",
    "V22_10d": ROOT + "V22/model_pred/v22_label_ret_10d.fea",
    "V22_20d": ROOT + "V22/model_pred/v22_label_ret_20d.fea",
    "V11_ensw2": ROOT + "V11/model_pred/2026q3/score_ens_w2.fea",
    "V18_v18": ROOT + "V18/model_pred/v18.fea",
}
V23_PATH = ROOT + "V23/model_pred/2026q3/score_ens23.fea"

PROTOCOLS = {
    "D01_h5s8_t1": dict(top_n=1, hold=5, stop_loss=0.08),
    "hold20_t1": dict(top_n=1, hold=20),
    "hold20_t2": dict(top_n=2, hold=20),
    "h20tr20_t2": dict(top_n=2, hold=20, trail_pct=0.20),
    "S2_re300": dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
                     stop_loss=0.08, trail_pct=0.15),
}

LABEL_FILES = {"1d": "label_ret_1d", "3d": "label_ret_3d", "5d": "label_ret_5d",
               "10d": "label_ret_10d", "20d": "label_ret_20d",
               "trail": "label_trail20d"}

_G = {}


def df_to_scores(df):
    return dict(
        scores={d: df.loc[d].dropna().to_dict() for d in df.index},
        ranked={d: df.loc[d].dropna().sort_values(ascending=False).index.tolist()
                for d in df.index},
        top1={d: float(df.loc[d].max()) for d in df.index},
        dates=sorted(df.index))


def eval_ic(score_df, buy):
    """多维 IC: 各标签周期 RankIC + RankICIR (可交易池, Test 窗口)。"""
    out = {}
    for h, fn in LABEL_FILES.items():
        p = ROOT.rsplit('Model/', 1)[0] + 'trainingdata/' + fn + '.fea'
        if not os.path.exists(p):
            continue
        lab = pd.read_feather(p).set_index('index')
        ics = []
        for d in score_df.index:
            if d not in lab.index:
                continue
            s = score_df.loc[d]
            y = lab.loc[d].reindex(s.index)
            if d in buy.index:
                b = pd.to_numeric(buy.loc[d].reindex(s.index), errors='coerce')
            else:
                b = pd.Series(np.nan, index=s.index)
            tr = b.gt(0.5) & y.notna()
            if tr.sum() < 50:
                continue
            ics.append(s[tr].rank().corr(y[tr].rank()))
        if ics:
            ic = np.mean(ics)
            out[f'ic_{h}'] = ic
            out[f'ir_{h}'] = ic / np.std(ics) if np.std(ics) > 0 else np.nan
    return out


def run_proto(args):
    name, sc, pn, pc = args
    m, tr, eq = run_backtest(sc, pc, _G["mkt"], _G["tds"], _G["tdi"])
    if m is None:
        return None
    tr = tr.sort_values("buy_dt")
    k = max(1, len(tr) // 3)
    ts = []
    for part in (tr.iloc[:k], tr.iloc[k:2 * k], tr.iloc[2 * k:]):
        ts.append(float((1 + part["net_pct"] / 100 / pc["top_n"]).prod() - 1))
    return dict(set=name, proto=pn, cum_net=m["cum_net"], sharpe=m["sharpe"],
                maxdd=m["maxdd"], h1=m["h1_cum"], h2=m["h2_cum"],
                t1=ts[0], t2=ts[1], t3=ts[2], win=m["win_rate"],
                n=m["n_trades"])


def main():
    import multiprocessing as mp
    buy = pd.read_feather(ROOT.rsplit('Model/', 1)[0] + 'trainingdata/buyable_mask.fea') \
        .set_index('date')
    sets = {}
    for n, p in SETS.items():
        if os.path.exists(p):
            sets[n] = pd.read_feather(p).set_index("date")
    if os.path.exists(V23_PATH):
        sets["V23_ens23"] = pd.read_feather(V23_PATH).set_index("date")
    mkt = load_market()
    tds, tdi = load_calendar()
    _G.update(mkt=mkt, tds=tds, tdi=tdi)

    ic_rows = []
    for n, df in sets.items():
        ic = eval_ic(df, buy)
        ic["set"] = n
        ic_rows.append(ic)
    ic_df = pd.DataFrame(ic_rows).set_index("set")
    print("== A. IC 层 (RankIC / RankICIR, 可交易池, Test) ==")
    cols = [c for c in ic_df.columns if c.startswith('ic_') or c.startswith('ir_')]
    print(ic_df[cols].round(3).to_string())

    scs = {n: df_to_scores(df) for n, df in sets.items()}
    tasks = [(n, scs[n], pn, pc) for n in scs for pn, pc in PROTOCOLS.items()]
    pool = mp.Pool(20)
    rows = [r for r in pool.imap_unordered(run_proto, tasks, chunksize=2) if r]
    pool.close(); pool.join()
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "iter3_crossmodel.csv")
    df.to_csv(out, index=False)
    print("\n== B/C. 策略层真实净值矩阵 (cum_net | H1/H2 | T1/T2/T3) ==")
    for pn in PROTOCOLS:
        sub = df[df["proto"] == pn].set_index("set")
        print(f"\n[{pn}]")
        print(sub[["cum_net", "h1", "h2", "t1", "t2", "t3", "sharpe", "maxdd"]]
              .round(2).to_string())
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
