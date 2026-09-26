#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_ops.py — Model/best 每日专用推演 + 操作指令 + 收益追踪 (20260906 起)

流程 (每日收盘后数据更新完成, 运行一次):
  ① 打分刷新: best (慢腿源) / V11 (快腿源) / V8 影子 (v8_shadow, 仅观察)
     落后于因子数据最新日 → 自动推演;
  ② 纸面状态推进: 结算今日开盘指令 → 收盘检查 → 产出明日开盘操作;
  ③ 输出: 当前持仓 / 明日买卖清单 (慢腿2只+快腿1只) / 双腿净值;
  ④ 收益记录: Model/best/equity_history.csv 追加当日净值 (20260906 空仓起点 10W);
  ⑤ V8 影子对照: 打印 V8 与 best 同因子日慢腿 Top2 候选及分歧 (仅观察,
     结论与建议一律以 best/V11 计划为准)。

状态: Model/best/paper_state.json; 成交日志: Model/best/paper_trades.csv
用法:
  python3 daily_ops.py            # 每日运行 (推演+结算+计划+记录)
  python3 daily_ops.py --history  # 只看收益历史
  python3 daily_ops.py --reset    # 重置为空仓起点 (谨慎!)
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRADING = os.path.join(ROOT, "Trading")
sys.path.insert(0, TRADING)
from engine import (load_calendar, load_market, CAPITAL, LOT)  # noqa: E402
import daily_frozen as DF  # noqa: E402

PY = "/autodl-fs/data/miniconda3/bin/python3"
FAC_ALL = os.path.join(os.path.dirname(ROOT), "trainingdata",
                      "fac_all.fea")
BASE_DATE = "20260906"
BASE_TOTAL = 2 * CAPITAL          # 10W = 慢5W + 快5W
SLOW_CFG = DF.SLOW                # 慢腿 u_h20_re300_tr15
FAST_CFG = DF.FAST                # 快腿 D01
STATE_PATH = os.path.join(HERE, "paper_state.json")
TRADE_PATH = os.path.join(HERE, "paper_trades.csv")
EQ_PATH = os.path.join(HERE, "equity_history.csv")

UNITS = {  # 打分源 → (目录, score 相对路径)
    "slow": ("best", "model_pred/2026q3/all_zscore_score.fea"),
    "fast": ("V11", "model_pred/2026q3/score_ens_w2.fea"),
    # V8 影子 (2026-09-07 加): V8 原版 9/2 checkpoint (训练截止 20250809),
    # 在 best/v8_shadow/ 下做同步实时推演, 仅供对照观察, 不参与 slow/fast 决策。
    "v8_shadow": ("best/v8_shadow", "model_pred/2026q3/all_zscore_score.fea"),
}


def latest_factor_date():
    import pyarrow.feather as pf
    d = pf.read_table(FAC_ALL, columns=["date"]).column("date") \
        .to_pandas().astype(str)
    return sorted(d.unique())[-1]


def score_max_date(unit, rel):
    p = os.path.join(ROOT, unit, rel)
    if not os.path.exists(p):
        return None
    df = pd.read_feather(p)
    return str(df["date"].max())


def refresh_if_stale(verbose=True):
    """打分落后于因子数据最新日 → 自动跑该单元 analysis.py 推演。"""
    fmax = latest_factor_date()
    ran = []
    for key, (unit, rel) in UNITS.items():
        smax = score_max_date(unit, rel)
        if smax is None or smax < fmax:
            if verbose:
                print(f"[daily_ops] {key} 打分落后 ({smax} < {fmax}), "
                      f"运行 {unit}/analysis.py 推演 ...")
            cwd = os.path.join(ROOT, unit)
            r = subprocess.run([PY, "-u", "analysis.py"], cwd=cwd,
                               capture_output=True, text=True, timeout=7200)
            if r.returncode != 0:
                print(f"[daily_ops] ⚠ {unit} 推演失败 rc={r.returncode}: "
                      f"{r.stderr[-300:]}")
            ran.append(unit)
    return fmax, ran


def load_score_df(unit, rel):
    return pd.read_feather(os.path.join(ROOT, unit, rel)) \
        .set_index("date").sort_index()


def state_init():
    return {k: DF.leg_state_init() for k in ["slow", "fast"]}


def leg_value(st, market, t):
    cash = float(st.get("cash", CAPITAL))
    for s in st.get("slots", []):
        px = market["close"].get((t, s["code"])) or s["buy_prc"]
        cash += px * s["lots"] * LOT
    return cash


def record_equity(today, slow_st, fast_st, market):
    sv = leg_value(slow_st, market, today)
    fv = leg_value(fast_st, market, today)
    total = sv + fv
    rows = []
    if os.path.exists(EQ_PATH):
        rows = pd.read_csv(EQ_PATH)
        rows["date"] = rows["date"].astype(str)   # CSV 数字日期会被读成 int,与日历 str 对齐
        rows = rows[rows["date"] != today]     # 同日重跑 → 覆盖该日行
    row = dict(date=today, slow_eq=round(sv, 2), fast_eq=round(fv, 2),
               total=round(total, 2), cum_pct=round(total / BASE_TOTAL - 1, 6))
    rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True) \
        if len(rows) else pd.DataFrame([row])
    rows = rows.drop_duplicates("date", keep="last").sort_values("date")
    rows.to_csv(EQ_PATH, index=False)
    return row


def run_daily():
    market = load_market()
    tds, tdi = load_calendar()
    fmax, refreshed = refresh_if_stale()
    slow_df = load_score_df(*UNITS["slow"])
    fast_df = load_score_df(*UNITS["fast"])
    today = tds[min(tdi.get(fmax, len(tds) - 1), len(tds) - 1)]
    names = market["name_last"]
    state = state_init()
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
    first_run = not state.get("base_date")
    if first_run:
        state = state_init()
        state["base_date"] = BASE_DATE
        print(f"[daily_ops] 首次运行: 以 {BASE_DATE} 周末空仓为起点 (10W=5W+5W)")
    CFG = dict(slow=SLOW_CFG, fast=FAST_CFG)
    SDF = dict(slow=slow_df, fast=fast_df)
    sold_all = bought_all = 0
    for leg in ["slow", "fast"]:
        st = state.setdefault(leg, DF.leg_state_init())
        _pend = st.get("pending") or {}
        if _pend.get("exec_day") and _pend["exec_day"] < today:
            print(f"[daily_ops] ⚠ [{leg}] 上份指令 (exec "
                  f"{_pend['exec_day']}) 因未在当日运行而失效 — 已丢弃, "
                  f"按 {today} 最新打分重新计划")
            st["pending"] = None
        ex_sells, ex_buys, tmr_sells, tmr_buys, st = \
            DF.settle_and_plan_leg(SDF[leg], market, st, CFG[leg],
                                   tds, tdi, today)
        state[leg] = st
        sold_all += len(ex_sells)
        bought_all += len(ex_buys)
        DF.TRADE_PATH = TRADE_PATH
        DF.log_trades(ex_sells, today, tdi, names, leg)
    if first_run:
        row = dict(date=BASE_DATE, slow_eq=float(CAPITAL),
                   fast_eq=float(CAPITAL), total=float(BASE_TOTAL),
                   cum_pct=0.0)
        pd.DataFrame([row]).to_csv(EQ_PATH, index=False)
        print(f"[daily_ops] 基准行已写入: {BASE_DATE} 空仓 100,000 (+0.0%)")
    else:
        row = record_equity(today, state["slow"], state["fast"], market)
    print(f"\n{'='*72}")
    print(f" Model/best 每日操作 @ {today} 收盘 (打分日 {fmax})"
          + (f" | 已自动刷新: {refreshed}" if refreshed else ""))
    print(f"{'='*72}")
    print(f" 今日开盘结算: 卖出 {sold_all} 笔 / 买入 {bought_all} 笔")
    for leg in ["slow", "fast"]:
        st = state[leg]
        cfgn = CFG[leg]["name"]
        val = leg_value(st, market, today)
        print(f"\n [{cfgn} {leg}] 池净值 {val:,.0f} 元 "
              f"(累计 {val/CAPITAL-1:+.2%})")
        if st["slots"]:
            for s in st["slots"]:
                cc = market["close"].get((today, s["code"])) or s["buy_prc"]
                pnl = cc / s["buy_prc"] - 1 if s["buy_prc"] else 0
                flag = " →明日卖" if s["sell_flag"] else ""
                print(f"   持仓 {s['code']} {names.get(s['code'],'?')} "
                      f"{s['lots']}手 买{s['buy_dt']}@{s['buy_prc']:.2f} "
                      f"现{cc:.2f} ({pnl:+.2%}) 峰{s['peak']:.2f}{flag}")
        else:
            print("   持仓: 空仓")
        pend = st.get("pending") or {}
        print(f" 明日 ({pend.get('exec_day')}) 开盘卖出: "
              + ("; ".join(f"{s['code']} {names.get(s['code'],'?')} "
                           f"{s['lots']}手 ({s['reason']})"
                           for s in pend.get("sells", [])) or "无"))
        print(f" 明日 ({pend.get('exec_day')}) 开盘买入 (按序成交前"
              + f"{CFG[leg]['top_n']} 只, 跳过 ST/涨停/无价): "
              + ("; ".join(f"{b['code']} {names.get(b['code'],'?')}"
                           for b in pend.get("buys", [])) or "无"))
    print(f"\n 双腿合计净值: {row['total']:,.0f} 元 / 10W "
          f"(累计 {row['cum_pct']:+.2%}) — 已记入 {os.path.basename(EQ_PATH)}")

    # ── V8 影子对照 (仅观察; 结论与操作建议一律以 best/V11 计划为准) ──
    try:
        v8_df = load_score_df(*UNITS["v8_shadow"])
        if len(v8_df):
            fd_v8 = str(v8_df.index.max())
            fd_best = str(slow_df.index.max())
            sync = "一致" if fd_v8 == fd_best else "⚠ 不一致"
            print(f"\n── V8 影子对照 (仅观察, 决策以 best/V11 为准) ──")
            print(f"  最新因子日: V8 {fd_v8} vs best {fd_best} ({sync})")
            best_row = slow_df.loc[fd_v8].dropna() \
                if fd_v8 in slow_df.index else None
            v8_row = v8_df.loc[fd_v8].dropna()
            if best_row is not None and len(v8_row):
                bt = best_row.sort_values(ascending=False).head(2)
                vt = v8_row.sort_values(ascending=False).head(2)
                fmt = lambda s: "  ".join(
                    f"{i} {names.get(str(i), '?')} ({v:+.2f})"
                    for i, v in s.items())
                print(f"  慢腿候选 Top2 @ {fd_v8}:")
                print(f"    best: {fmt(bt)}")
                print(f"    V8  : {fmt(vt)}")
                overlap = len(set(bt.index) & set(vt.index))
                print(f"  两源 Top2 重合 {overlap}/2" + (
                    "" if overlap else " — 分歧较大, 仅供观察, 不改 best 计划"))
            else:
                print("  ⚠ best 打分未覆盖该因子日, 跳过候选对比")
        else:
            print("\n[V8 影子] 尚无打分 (首次运行 v8_shadow/analysis.py 后生成)")
    except Exception as e:
        print(f"[V8 影子] ⚠ 对照不可用: {e}")

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f" 状态已写入 {os.path.basename(STATE_PATH)}")
    show_history(tail=10)


def show_history(tail=None):
    if not os.path.exists(EQ_PATH):
        print("[history] 尚无记录 (首次运行后生成)")
        return
    h = pd.read_csv(EQ_PATH)
    t = h.tail(tail) if tail else h
    print(f"\n--- 收益历史 (共 {len(h)} 行, {h['date'].min()} ~ {h['date'].max()}) ---")
    print(t.to_string(index=False,
                      formatters={"total": "{:,.0f}".format,
                                  "cum_pct": "{:+.2%}".format}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if args.reset:
        for p in (STATE_PATH, TRADE_PATH, EQ_PATH):
            if os.path.exists(p):
                os.remove(p)
        print("[reset] 已清除本目录纸面状态与收益历史 (重新从空仓起点开始)")
        return
    if args.history:
        show_history()
        return
    run_daily()


if __name__ == "__main__":
    main()
