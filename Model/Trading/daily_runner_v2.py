#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_runner_v2.py — h20tr20_t2 协议每日决策器 (Top2 + 持有≤20日 + 移动止损20%)

与 Trading/engine.py 口径一致 (open 开盘执行, 先卖后买, 收盘触发次日开盘执行,
利润再投资等权再平衡)。两阶段纸面流程 (收盘后运行):
  ① 结算: 上一份指令 (state['pending']) 按今日开盘价执行 (先卖后买, 资金闭环);
  ② 计划: 今日收盘检查 (峰值/移动止损/到期) → 产出明日开盘指令 (存回 pending)。
执行过滤: ST/退、一字涨停(±9.5%) 买不进、跌停卖不出顺延、无价顺延、整手。

用法:
  python3 daily_runner_v2.py <score.fea>            # 每日收盘后运行 (纸面推进)
  python3 daily_runner_v2.py <score.fea> --replay   # 回放校验: 与引擎逐笔对齐
  python3 daily_runner_v2.py <score.fea> --reset    # 重置纸面状态 (重新开始)
状态: Trading/holdings_v2.json; 纸面成交: Trading/paper_trades_v2.csv
"""
import bisect
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (load_market, load_calendar, LIMIT_UP, CAPITAL,
                    buy_cost, sell_cost, LOT, run_backtest)

TOPN = 2
HOLD = 20
TRAIL = 0.20
BUY_SLACK = 8     # 买入候选备选数 (开盘被阻时顺延, 与引擎的完整排名遍历对齐)
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, 'holdings_v2.json')
TRADE_PATH = os.path.join(HERE, 'paper_trades_v2.csv')


def latest_fd(td, fdates):
    j = bisect.bisect_right(fdates, td) - 1
    return fdates[j] if j >= 0 else None


def _slot_mv(s, market, t):
    px = market['open'].get((t, s['code'])) or market['close'].get((t, s['code'])) \
        or s['buy_prc']
    return px * s['lots'] * LOT


def settle_and_plan(score_df, market, state, tds, tdi, today):
    """today 收盘运行: 结算 pending 指令 (今日开盘) → 收盘检查 → 计划明日开盘。
    返回 (executed_sells, executed_buys, tomorrow_sells, tomorrow_buys, state)。"""
    t = today
    open_m, close_m, prev_m, name_m = (market['open'], market['close'],
                                       market['prev'], market['name'])
    name_last = market['name_last']
    liquid = state.get('cash', CAPITAL)
    slots = [s for s in state.get('slots', [])]
    pend = state.get('pending') or {}
    ex_sells, ex_buys = [], []

    # ---- ① 结算昨日指令 (今日开盘, 先卖后买) ----
    for s in list(pend.get('sells', [])):
        sp = open_m.get((t, s['code']))
        if sp is None or sp <= 0:
            continue                        # 无价 → 顺延
        pc = prev_m.get((t, s['code']))
        if pc and pc > 0 and sp / pc - 1 <= -0.095:
            continue                        # 一字跌停卖不出 → 顺延
        notional = s['lots'] * LOT * sp
        liquid += notional - sell_cost(s['buy_notional'])
        ex_sells.append(dict(code=s['code'], price=sp, lots=s['lots'],
                             reason=s['reason'], buy_dt=s['buy_dt'],
                             buy_prc=s['buy_prc'], buy_notional=s['buy_notional']))
        slots = [q for q in slots if not (q['code'] == s['code']
                                          and q['buy_dt'] == s['buy_dt'])]
    held_codes = {q['code'] for q in slots}
    eq_now = liquid + sum(_slot_mv(q, market, t) for q in slots)
    budget = eq_now / TOPN
    bought_codes = set()
    for b in list(pend.get('buys', [])):
        if len(slots) >= TOPN:
            break
        if b['code'] in held_codes or b['code'] in bought_codes:
            continue
        nm = name_m.get((t, b['code'])) or name_last.get(b['code'], '')
        if 'ST' in nm or '退' in nm:
            continue
        bp = open_m.get((t, b['code']))
        pc = prev_m.get((t, b['code']))
        if bp is None or bp <= 0:
            continue                        # 无价 → 顺延
        if pc and pc > 0 and bp / pc - 1 >= LIMIT_UP:
            continue                        # 一字涨停买不进 → 顺延
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
        slots.append(dict(code=b['code'], buy_dt=t, buy_prc=bp, lots=lots,
                          peak=bp, peak_dt=t, sell_flag=False, reason='',
                          buy_notional=notional))
        bought_codes.add(b['code'])
        ex_buys.append(dict(code=b['code'], price=bp, lots=lots,
                            budget_used=notional))

    # ---- ② 收盘检查: 峰值/移动止损/到期 → 明日开盘卖 ----
    for s in slots:
        cc = close_m.get((t, s['code']))
        if cc is None or cc <= 0:
            continue
        if cc > s['peak']:
            s['peak'] = cc
            s['peak_dt'] = t
        held = tdi[t] - tdi[s['buy_dt']]
        if held >= HOLD - 1:
            if not s['sell_flag']:
                s['sell_flag'] = True
                s['reason'] = 'expiry'
        elif held >= 1 and cc <= s['peak'] * (1 - TRAIL):
            if not s['sell_flag']:
                s['sell_flag'] = True
                s['reason'] = 'trail'

    # ---- ③ 明日开盘指令 (基于今日收盘打分) ----
    fdates = sorted(score_df.index)
    fd = latest_fd(t, fdates)
    tmr = tds[min(tdi[t] + 1, len(tds) - 1)]
    tmr_sells, tmr_buys = [], []
    for s in slots:
        if s['sell_flag']:
            tmr_sells.append(dict(code=s['code'], lots=s['lots'],
                                  buy_dt=s['buy_dt'], buy_prc=s['buy_prc'],
                                  buy_notional=s['buy_notional'],
                                  reason=s['reason']))
    if fd is not None:
        row = score_df.loc[fd].dropna()
        ranked = row.sort_values(ascending=False).index.tolist()
        # 与引擎一致: 明日开盘先卖后买 → 计划候选排除「持有且不卖」的代码,
        # 明日将被卖出的代码允许当日回买 (T+1 合规)
        held_codes = {q['code'] for q in slots if not q['sell_flag']}
        planned = 0
        for code in ranked:
            if planned >= TOPN + BUY_SLACK:
                break
            if code in held_codes:
                continue
            nm = name_m.get((tmr, code)) or name_last.get(code, '')
            if 'ST' in nm or '退' in nm:
                continue
            fc = close_m.get((fd, code))
            fpc = prev_m.get((fd, code))
            if fc and fpc and fpc > 0 and fc / fpc - 1 >= LIMIT_UP:
                continue                        # 因子日一字涨停 → 明日买不进
            tmr_buys.append(dict(code=code, factor_dt=fd))
            planned += 1
    state['slots'] = slots
    state['cash'] = liquid
    state['prev_day'] = t
    state['pending'] = dict(exec_day=tmr, factor_dt=fd,
                            sells=tmr_sells, buys=tmr_buys)
    return ex_sells, ex_buys, tmr_sells, tmr_buys, state


def log_trades(ex_sells, exec_day, tdi, name_last):
    rows = []
    for s in ex_sells:
        notional = s['lots'] * LOT * s['price']
        cost_b = max(5.0, s['buy_notional'] * 2.5e-4) + s['buy_notional'] * 1e-5
        cost_s = max(5.0, notional * 2.5e-4) + notional * 1e-5 + notional * 5e-4
        net = (notional - cost_s) / (s['buy_notional'] + cost_b) - 1.0
        rows.append(dict(buy_dt=s['buy_dt'], sell_dt=exec_day, code=s['code'],
                         name=name_last.get(s['code'], ''), buy_prc=s['buy_prc'],
                         sell_prc=s['price'], lots=s['lots'],
                         net_pct=net * 100, reason=s['reason'],
                         hold_days=tdi[exec_day] - tdi[s['buy_dt']]))
    if rows:
        dfn = pd.DataFrame(rows)
        if os.path.exists(TRADE_PATH):
            dfo = pd.read_csv(TRADE_PATH)
            dfn = pd.concat([dfo, dfn], ignore_index=True)
        dfn.to_csv(TRADE_PATH, index=False)
    return len(rows)


def run_daily(score_path):
    market = load_market()
    tds, tdi = load_calendar()
    score = pd.read_feather(score_path).set_index('date').sort_index()
    last_fd = score.index.max()
    today = tds[tdi.get(last_fd, len(tds) - 1)]
    state = {'slots': [], 'cash': CAPITAL, 'prev_day': None, 'pending': None}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
    ex_sells, ex_buys, tmr_sells, tmr_buys, state = \
        settle_and_plan(score, market, state, tds, tdi, today)
    names = market['name_last']
    tmr = state['pending']['exec_day']
    n_log = log_trades(ex_sells, today, tdi, names)
    eq = state['cash'] + sum(
        (market['close'].get((today, s['code'])) or s['buy_prc']) * s['lots'] * LOT
        for s in state['slots'])
    print(f'== h20tr20_t2 每日决策 @ {today} 收盘 (打分日 {last_fd}) ==')
    print(f'  今日开盘结算: 卖出 {len(ex_sells)} 笔 / 买入 {len(ex_buys)} 笔 '
          f'(纸面成交日志 +{n_log} 笔)')
    print(f'  当前净值: {eq:.0f} 元 (现金 {state["cash"]:.0f} + 持仓)')
    print(f'  明日 ({tmr}) 开盘卖出: ' + ('; '.join(
        f"{s['code']} {names.get(s['code'],'?')} {s['lots']}手 ({s['reason']})"
        for s in tmr_sells) if tmr_sells else '无'))
    print(f'  明日 ({tmr}) 开盘买入: ' + ('; '.join(
        f"{b['code']} {names.get(b['code'],'?')}" for b in tmr_buys)
        if tmr_buys else '无'))
    print(f'  收盘持仓: ' + ('; '.join(
        f"{s['code']} (买 {s['buy_dt']}, 峰 {s['peak']:.2f})"
        for s in state['slots']) if state['slots'] else '空仓'))
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f'  状态已写入 {STATE_PATH}')


def replay_check(score_path):
    """回放校验: 全窗口逐日 settle_and_plan, 与引擎 h20tr20_t2 逐笔对齐。"""
    score = pd.read_feather(score_path).set_index('date').sort_index()
    market = load_market()
    tds, tdi = load_calendar()
    scores_dict = dict(
        scores={d: score.loc[d].dropna().to_dict() for d in score.index},
        ranked={d: score.loc[d].dropna().sort_values(ascending=False).index.tolist()
                for d in score.index},
        top1={d: float(score.loc[d].max()) for d in score.index},
        dates=sorted(score.index))
    m, tr, eq = run_backtest(scores_dict, dict(top_n=TOPN, hold=HOLD,
                                               trail_pct=TRAIL),
                             market, tds, tdi)
    state = {'slots': [], 'cash': CAPITAL, 'prev_day': None, 'pending': None}
    trades = []
    fdates = sorted(score.index)
    # 先做首个因子日的计划 (无今日开盘结算)
    d0 = fdates[0]
    state['pending'] = None
    _ = None
    days = [d for d in fdates if d in tdi]
    first = True
    for d in days:
        if first:
            # 初始化: 以首因子日为今日 (只做计划, 无今日开盘结算)
            fd = latest_fd(d, fdates)
            row = score.loc[fd].dropna()
            ranked = row.sort_values(ascending=False).index.tolist()
            tmr = tds[tdi[d] + 1]
            open_m, close_m, prev_m = market['open'], market['close'], market['prev']
            name_m, name_last = market['name'], market['name_last']
            plan = []
            for code in ranked:
                if len(plan) >= TOPN + BUY_SLACK:
                    break
                nm = name_m.get((tmr, code)) or name_last.get(code, '')
                if 'ST' in nm or '退' in nm:
                    continue
                fc = close_m.get((fd, code))
                fpc = prev_m.get((fd, code))
                if fc and fpc and fpc > 0 and fc / fpc - 1 >= LIMIT_UP:
                    continue
                plan.append(dict(code=code, factor_dt=fd))
            state['pending'] = dict(exec_day=tmr, factor_dt=fd, sells=[],
                                    buys=plan)
            first = False
            continue
        ex_sells, ex_buys, tmr_sells, tmr_buys, state = \
            settle_and_plan(score, market, state, tds, tdi, d)
        for s in ex_sells:
            trades.append(dict(buy_dt=s['buy_dt'], sell_dt=d, code=s['code'],
                               buy_prc=s['buy_prc'], sell_prc=s['price'],
                               reason=s['reason'],
                               hold_days=tdi[d] - tdi[s['buy_dt']]))
    rt = pd.DataFrame(trades)
    is_final = tr['reason'].fillna('').astype(str).str.strip() == ''
    et = tr[~is_final][['buy_dt', 'sell_dt', 'code', 'buy_prc', 'sell_prc',
                        'hold_days', 'reason']].copy()
    n_match = 0
    for _, a in et.iterrows():
        b = rt[(rt['code'] == a['code']) & (rt['buy_dt'] == a['buy_dt'])]
        if len(b) and abs(b.iloc[0]['sell_prc'] - a['sell_prc']) < 1e-6:
            n_match += 1
    print(f'[replay] 引擎 {len(et)} 笔 (剔除 {int(is_final.sum())} 笔期末强制清仓) / '
          f'决策器 {len(rt)} 笔; 对齐 {n_match}/{len(et)} 笔')
    print(f'[replay] 引擎 cum_net {m["cum_net"]:+.2%} (基准)')
    ok = n_match == len(et)
    print('[replay] 结论: ' + ('一致 ✓ (runner 与 engine 同口径)' if ok
                               else '存在差异 ✗ — 检查二者口径'))
    return ok


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if p is None:
        print('usage: daily_runner_v2.py <score.fea> [--replay|--reset]')
        sys.exit(1)
    if '--reset' in sys.argv:
        for f in (STATE_PATH, TRADE_PATH):
            if os.path.exists(f):
                os.remove(f)
        print('纸面状态已重置')
    elif '--replay' in sys.argv:
        replay_check(p)
    else:
        run_daily(p)
