#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_frozen.py — 冻结双腿策略每日决策器 (FIXED_STRATEGY.md v1.0)

慢腿 (5W 池): u_h20_re300_tr15 = Top2 + 持有≤20日 + 排名>300退出(min_hold2)
              + 移动止盈15% (无固定止损)
快腿 (5W 池): D01 = Top1 + 持有5日 + 止损8%
打分源 (默认): Model/V11/model_pred/2026q3/score_ens_w2.fea (V11_ensw2, 全场最优)

与 Trading/engine.py 口径一致 (open 开盘执行, 先卖后买, 收盘触发次日开盘执行,
利润再投资等权再平衡)。两阶段纸面流程 (每日收盘后运行):
  ① 结算: 上一份指令按今日开盘价执行;  ② 计划: 今日收盘检查 → 明日开盘指令。
执行过滤: ST/退、一字涨停买不进、跌停卖不出顺延、无价顺延、整手、T+1。

用法:
  python3 daily_frozen.py                            # 每日收盘后运行 (纸面推进)
  python3 daily_frozen.py --score <score.fea>        # 指定打分源
  python3 daily_frozen.py --replay                   # 回放校验: 双腿分别与引擎逐笔对齐
  python3 daily_frozen.py --reset                    # 重置纸面状态 (重新开始)
状态: Trading/holdings_frozen.json; 纸面成交: Trading/paper_trades_frozen.csv
"""
import argparse
import bisect
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (load_market, load_calendar, LIMIT_UP, CAPITAL,  # noqa: E402
                    buy_cost, sell_cost, LOT, run_backtest)

DEFAULT_SCORE = "/autodl-fs/data/lingqiData/Model/V11/model_pred/2026q3/score_ens_w2.fea"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "holdings_frozen.json")
TRADE_PATH = os.path.join(HERE, "paper_trades_frozen.csv")
BUY_SLACK = 8

# ---- 冻结双腿配置 (FIXED_STRATEGY.md v1.0, 不要改) ----
SLOW = dict(top_n=2, hold=20, exit_rank=300, min_hold=2,
            trail_pct=0.15, stop_loss=None, name="慢腿")
FAST = dict(top_n=1, hold=5, exit_rank=None, min_hold=1,
            trail_pct=None, stop_loss=0.08, name="快腿")
LEGS = ["slow", "fast"]


def latest_fd(td, fdates):
    j = bisect.bisect_right(fdates, td) - 1
    return fdates[j] if j >= 0 else None


def _slot_mv(s, market, t):
    px = market["open"].get((t, s["code"])) or market["close"].get((t, s["code"])) \
        or s["buy_prc"]
    return px * s["lots"] * LOT


def leg_state_init():
    return dict(cash=float(CAPITAL), slots=[], prev_day=None, pending=None)


def state_init():
    return {k: leg_state_init() for k in LEGS}


def settle_and_plan_leg(score_df, market, st, cfg, tds, tdi, today):
    """单腿两阶段: ①结算今日开盘 pending → ②收盘检查 → 明日开盘指令。"""
    t = today
    open_m, close_m, prev_m, name_m = (market["open"], market["close"],
                                       market["prev"], market["name"])
    name_last = market["name_last"]
    top_n = cfg["top_n"]
    liquid = float(st.get("cash", CAPITAL))
    slots = [s for s in st.get("slots", [])]
    pend = st.get("pending") or {}
    # 仅当 pending 的执行日 == 今日才结算 (防同日重复运行/状态滞后导致双结算)
    if pend.get("exec_day") and pend.get("exec_day") != t:
        pend = {}
    ex_sells, ex_buys = [], []

    # ---- ① 结算昨日指令 (今日开盘, 先卖后买) ----
    for s in list(pend.get("sells", [])):
        sp = open_m.get((t, s["code"]))
        if sp is None or sp <= 0:
            continue
        pc = prev_m.get((t, s["code"]))
        if pc and pc > 0 and sp / pc - 1 <= -0.095:
            continue
        notional = s["lots"] * LOT * sp
        liquid += notional - sell_cost(s["buy_notional"])
        ex_sells.append(dict(code=s["code"], price=sp, lots=s["lots"],
                             reason=s["reason"], buy_dt=s["buy_dt"],
                             buy_prc=s["buy_prc"],
                             buy_notional=s["buy_notional"]))
        slots = [q for q in slots if not (q["code"] == s["code"]
                                          and q["buy_dt"] == s["buy_dt"])]
    held_codes = {q["code"] for q in slots}
    mv_other = sum(_slot_mv(q, market, t) for q in slots)
    eq_now = liquid + mv_other
    budget = eq_now / top_n
    bought_codes = set()
    for b in list(pend.get("buys", [])):
        if len(slots) >= top_n:
            break
        if b["code"] in held_codes or b["code"] in bought_codes:
            continue
        nm = name_m.get((t, b["code"])) or name_last.get(b["code"], "")
        if "ST" in nm or "退" in nm:
            continue
        bp = open_m.get((t, b["code"]))
        pc = prev_m.get((t, b["code"]))
        if bp is None or bp <= 0:
            continue
        if pc and pc > 0 and bp / pc - 1 >= LIMIT_UP:
            continue
        avail = min(liquid, budget)
        lots = int(avail // (LOT * bp))
        while lots >= 1:
            if lots * LOT * bp + buy_cost(lots * LOT * bp) <= avail:
                break
            lots -= 1
        if lots < 1:
            continue
        notional = lots * LOT * bp
        liquid -= notional + buy_cost(notional)
        slots.append(dict(code=b["code"], buy_dt=t, buy_prc=bp, lots=lots,
                          peak=bp, peak_dt=t, sell_flag=False, reason="",
                          buy_notional=notional))
        bought_codes.add(b["code"])
        ex_buys.append(dict(code=b["code"], price=bp, lots=lots,
                            budget_used=notional))

    # ---- 今日打分排名 (收盘检查用: 排名退出) ----
    fdates = sorted(score_df.index)
    fd = latest_fd(t, fdates)
    rank_now = {}
    if cfg["exit_rank"] is not None and fd is not None:
        row = score_df.loc[fd].dropna()
        ranked = row.sort_values(ascending=False).index.tolist()
        rank_now = {c: i + 1 for i, c in enumerate(ranked)}

    # ---- ② 收盘检查 (峰值/排名/止损/移动止盈/到期 → 明日开盘卖) ----
    for s in slots:
        cc = close_m.get((t, s["code"]))
        if cc is None or cc <= 0:
            continue
        if cc > s["peak"]:
            s["peak"] = cc
            s["peak_dt"] = t
        held = tdi[t] - tdi[s["buy_dt"]]
        if held >= cfg["hold"] - 1:
            if not s["sell_flag"]:
                s["sell_flag"] = True
                s["reason"] = "expiry"
        elif held >= cfg["min_hold"] and cfg["exit_rank"] is not None \
                and rank_now.get(s["code"], len(rank_now) + 1) > cfg["exit_rank"]:
            if not s["sell_flag"]:
                s["sell_flag"] = True
                s["reason"] = "rank"
        elif held >= 1 and cfg["stop_loss"] is not None \
                and cc <= s["buy_prc"] * (1 - cfg["stop_loss"]):
            if not s["sell_flag"]:
                s["sell_flag"] = True
                s["reason"] = "stop"
        elif held >= 1 and cfg["trail_pct"] is not None \
                and cc <= s["peak"] * (1 - cfg["trail_pct"]):
            if not s["sell_flag"]:
                s["sell_flag"] = True
                s["reason"] = "trail"

    # ---- ③ 明日开盘指令 (基于今日收盘打分) ----
    tmr = tds[min(tdi[t] + 1, len(tds) - 1)]
    tmr_sells = [dict(code=s["code"], lots=s["lots"], buy_dt=s["buy_dt"],
                      buy_prc=s["buy_prc"], buy_notional=s["buy_notional"],
                      reason=s["reason"]) for s in slots if s["sell_flag"]]
    tmr_buys = []
    if fd is not None:
        row = score_df.loc[fd].dropna()
        ranked = row.sort_values(ascending=False).index.tolist()
        keep_codes = {q["code"] for q in slots if not q["sell_flag"]}
        planned = 0
        for code in ranked:
            if planned >= top_n + BUY_SLACK:
                break
            if code in keep_codes:
                continue
            nm = name_m.get((tmr, code)) or name_last.get(code, "")
            if "ST" in nm or "退" in nm:
                continue
            fc = close_m.get((fd, code))
            fpc = prev_m.get((fd, code))
            if fc and fpc and fpc > 0 and fc / fpc - 1 >= LIMIT_UP:
                continue
            tmr_buys.append(dict(code=code, factor_dt=fd))
            planned += 1
    st["slots"] = slots
    st["cash"] = liquid
    st["prev_day"] = t
    st["pending"] = dict(exec_day=tmr, factor_dt=fd,
                         sells=tmr_sells, buys=tmr_buys)
    return ex_sells, ex_buys, tmr_sells, tmr_buys, st


def log_trades(ex_sells, exec_day, tdi, name_last, leg):
    rows = []
    for s in ex_sells:
        notional = s["lots"] * LOT * s["price"]
        cost_b = max(5.0, s["buy_notional"] * 2.5e-4) + s["buy_notional"] * 1e-5
        cost_s = max(5.0, notional * 2.5e-4) + notional * 1e-5 + notional * 5e-4
        net = (notional - cost_s) / (s["buy_notional"] + cost_b) - 1.0
        rows.append(dict(leg=leg, buy_dt=s["buy_dt"], sell_dt=exec_day,
                         code=s["code"], name=name_last.get(s["code"], ""),
                         buy_prc=s["buy_prc"], sell_prc=s["price"],
                         lots=s["lots"], net_pct=net * 100, reason=s["reason"],
                         hold_days=tdi[exec_day] - tdi[s["buy_dt"]]))
    if rows:
        dfn = pd.DataFrame(rows)
        if os.path.exists(TRADE_PATH):
            dfn = pd.concat([pd.read_csv(TRADE_PATH), dfn], ignore_index=True)
        dfn.to_csv(TRADE_PATH, index=False)
    return len(rows)


def equity_now(st, market, t):
    cash = float(st["cash"])
    for s in st.get("slots", []):
        px = market["close"].get((t, s["code"])) or s["buy_prc"]
        cash += px * s["lots"] * LOT
    return cash


def run_daily(score_path, verbose=True):
    market = load_market()
    tds, tdi = load_calendar()
    score = pd.read_feather(score_path).set_index("date").sort_index()
    last_fd = score.index.max()
    today = tds[min(tdi.get(last_fd, len(tds) - 1), len(tds) - 1)]
    names = market["name_last"]
    state = state_init()
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
    CFG = dict(slow=SLOW, fast=FAST)
    total_sold = total_bought = 0
    for leg in LEGS:
        st = state.setdefault(leg, leg_state_init())
        ex_sells, ex_buys, tmr_sells, tmr_buys, st = \
            settle_and_plan_leg(score, market, st, CFG[leg], tds, tdi, today)
        state[leg] = st
        total_sold += len(ex_sells)
        total_bought += len(ex_buys)
        total_sold += log_trades(ex_sells, today, tdi, names, leg)
    if not verbose:
        return
    print(f"== 冻结双腿策略 每日决策 @ {today} 收盘 (打分日 {last_fd}) ==")
    print(f"  今日开盘结算: 卖出 {total_sold} 笔 / 买入 {total_bought} 笔")
    total_eq = 0.0
    for leg in LEGS:
        st = state[leg]
        eq = equity_now(st, market, today)
        total_eq += eq
        cfgn = CFG[leg]["name"]
        print(f"\n  [{cfgn} {leg}] 池净值 {eq:,.0f} 元 "
              f"(期初 {CAPITAL:,}, 累计 {eq/CAPITAL-1:+.1%})")
        if st["slots"]:
            for s in st["slots"]:
                cc = market["close"].get((today, s["code"])) or s["buy_prc"]
                pnl = (cc / s["buy_prc"] - 1) if s["buy_prc"] else 0
                flag = " →明日卖" if s["sell_flag"] else ""
                print(f"    持仓 {s['code']} {names.get(s['code'],'?')} "
                      f"{s['lots']}手 买{s['buy_dt']}@{s['buy_prc']:.2f} "
                      f"现{cc:.2f} ({pnl:+.1%}) 峰{s['peak']:.2f}{flag}")
        else:
            print("    持仓: 空仓")
        pend = st.get("pending") or {}
        print(f"  明日 ({pend.get('exec_day')}) 开盘卖出: "
              + ("; ".join(f"{s['code']} {names.get(s['code'],'?')} "
                           f"{s['lots']}手 ({s['reason']})"
                           for s in pend.get("sells", [])) or "无"))
        print(f"  明日 ({pend.get('exec_day')}) 开盘买入: "
              + ("; ".join(f"{b['code']} {names.get(b['code'],'?')}"
                           for b in pend.get("buys", [])) or "无"))
    print(f"\n  双腿合计净值 {total_eq:,.0f} 元 / 10W 本金 "
          f"(总累计 {total_eq/(2*CAPITAL)-1:+.1%})")
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"  状态已写入 {STATE_PATH}")


def replay_check(score_path):
    """回放校验: 全窗口逐日推进, 双腿分别与引擎 (HYB / D01) 逐笔对齐。"""
    score = pd.read_feather(score_path).set_index("date").sort_index()
    market = load_market()
    tds, tdi = load_calendar()
    scores_dict = dict(
        scores={d: score.loc[d].dropna().to_dict() for d in score.index},
        ranked={d: score.loc[d].dropna().sort_values(ascending=False)
                .index.tolist() for d in score.index},
        top1={d: float(score.loc[d].max()) for d in score.index},
        dates=sorted(score.index))
    eng_cfg = dict(
        slow=dict(top_n=2, hold=20, exit_rank=300, min_hold=2, trail_pct=0.15),
        fast=dict(top_n=1, hold=5, stop_loss=0.08))
    state = state_init()
    fdates = sorted(score.index)
    days = [d for d in fdates if d in tdi]
    # 与引擎同窗口: 仅回放至引擎窗口末日 (打分数据若超出则截断对齐)
    if days[-1] > "20260901":
        days = [d for d in days if d <= "20260901"]
    CFG = dict(slow=SLOW, fast=FAST)
    trades = {k: [] for k in LEGS}
    for d in days:
        for leg in LEGS:
            st = state[leg]
            fd = latest_fd(d, fdates)
            if st["pending"] is None:      # 首日: 只做计划
                tmr = tds[tdi[d] + 1]
                row = score.loc[fd].dropna()
                ranked = row.sort_values(ascending=False).index.tolist()
                name_m, name_last = market["name"], market["name_last"]
                plan = []
                for code in ranked:
                    if len(plan) >= CFG[leg]["top_n"] + BUY_SLACK:
                        break
                    nm = name_m.get((tmr, code)) or name_last.get(code, "")
                    if "ST" in nm or "退" in nm:
                        continue
                    fc = market["close"].get((fd, code))
                    fpc = market["prev"].get((fd, code))
                    if fc and fpc and fpc > 0 and fc / fpc - 1 >= LIMIT_UP:
                        continue
                    plan.append(dict(code=code, factor_dt=fd))
                st["pending"] = dict(exec_day=tmr, factor_dt=fd,
                                     sells=[], buys=plan)
            else:
                ex_sells, ex_buys, *_ , st = \
                    settle_and_plan_leg(score, market, st, CFG[leg],
                                        tds, tdi, d)
                for s in ex_sells:
                    trades[leg].append(dict(buy_dt=s["buy_dt"], sell_dt=d,
                                            code=s["code"],
                                            buy_prc=s["buy_prc"],
                                            sell_prc=s["price"],
                                            reason=s["reason"],
                                            hold_days=tdi[d] - tdi[s["buy_dt"]]))
            state[leg] = st
    ok_all = True
    for leg in LEGS:
        rt = pd.DataFrame(trades[leg])
        m, tr, eq = run_backtest(scores_dict, eng_cfg[leg], market, tds, tdi)
        is_final = tr["reason"].fillna("").astype(str).str.strip() == ""
        et = tr[~is_final][["buy_dt", "sell_dt", "code", "buy_prc",
                            "sell_prc", "hold_days", "reason"]].copy()
        n_match = 0
        for _, a in et.iterrows():
            b = rt[(rt["code"] == a["code"]) & (rt["buy_dt"] == a["buy_dt"])]
            if len(b) and abs(b.iloc[0]["sell_prc"] - a["sell_prc"]) < 1e-6 \
                    and b.iloc[0]["reason"] == a["reason"]:
                n_match += 1
        st = state[leg]
        eq_r = equity_now(st, market, days[-1])
        eq_e = float(eq.iloc[-1]["equity"])
        ratio = abs(eq_r - eq_e) / eq_e
        ok = n_match >= max(1, int(0.95 * len(et)))
        ok_all &= ok
        print(f"[replay:{leg}] 引擎 {len(et)} 笔 / 决策器 {len(rt)} 笔, "
              f"逐笔对齐 {n_match} 笔 ({n_match/max(1,len(et)):.0%}) | "
              f"期末净值 决策器 {eq_r:,.0f} vs 引擎 {eq_e:,.0f} "
              f"(差 {ratio:.2%}) → {'OK' if ok else 'MISMATCH'}")
    print(f"[replay] 双腿 {'全部 OK (与引擎口径一致)' if ok_all else '存在不一致, 请勿直接投产'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=DEFAULT_SCORE)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if args.reset and os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)
        print("[reset] 纸面状态已清除")
    if args.replay:
        replay_check(args.score)
    else:
        run_daily(args.score)


if __name__ == "__main__":
    main()
