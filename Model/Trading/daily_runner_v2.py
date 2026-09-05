#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_runner_v2.py — h20tr20_t2 协议每日决策器 (Top2 + 持有≤20日 + 移动止损20%)

与 Trading/engine.py 口径一致 (open 开盘执行, 先卖后买, 收盘触发次日开盘执行):
  - 输入: 打分 fea (逐日宽表) + 纸面状态 holdings_v2.json
  - 每晚收盘后运行 → 输出次日开盘的卖出/买入指令, 并更新纸面状态
  - 卖出触发: ① 收盘 ≤ 持仓期峰值收盘×0.80 (移动止损); ② 持有 ≥ 20 个交易日
  - 买入: 空位按最新排名补入前 2 名未持有标的 (跳过 ST/退/一字涨停/无价)
  - 再投资: 每仓预算 = 当前净值/2 (等权再平衡, 与引擎一致)

用法:
  python3 daily_runner_v2.py <score.fea>            # 决策 + 状态推进 (收盘后跑)
  python3 daily_runner_v2.py <score.fea> --replay    # 回放校验: 与引擎逐笔对齐检查
状态: 同目录 holdings_v2.json {slots:[{code,buy_dt,buy_prc,peak,peak_dt}], prev_day}
"""
import bisect
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (load_market, load_calendar, LIMIT_UP, CAPITAL,
                    buy_cost, sell_cost, LOT, run_backtest, load_scores)

TOPN = 2
HOLD = 20
TRAIL = 0.20
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holdings_v2.json')


def latest_fd(td, fdates):
    j = bisect.bisect_right(fdates, td) - 1
    return fdates[j] if j >= 0 else None


def decide_orders(score_df, market, state, tds, tdi, exec_day):
    """在 exec_day 开盘的决策: 返回 (sells, buys, 更新后 state)。

    资金口径 (与引擎一致): state['cash'] = 流动现金; 决策时
    净值 = 现金 + 持仓按当日开盘估值; 每仓预算 = 净值/TOPN (等权再平衡);
    卖单成交资金当日可用 (先卖后买闭环)。"""
    t = exec_day
    fd = latest_fd(tds[tdi[t] - 1], sorted(score_df.index))  # 开盘可用打分 ≤ t-1
    open_m, close_m, prev_m, name_m = (market['open'], market['close'],
                                       market['prev'], market['name'])
    name_last = market['name_last']
    sells, buys = [], []
    slots = [s for s in state.get('slots', [])]
    liquid = state.get('cash', CAPITAL)
    # ---- 开盘: 先卖 (sell_flag 由前一收盘决策写好; 资金当日回笼) ----
    for s in list(slots):
        if s.get('sell_flag'):
            sp = open_m.get((t, s['code']))
            if sp and sp > 0:
                pc = prev_m.get((t, s['code']))
                if pc and pc > 0 and sp / pc - 1 <= -0.095:
                    continue  # 一字跌停卖不出 → 顺延
                notional = s['lots'] * LOT * sp
                liquid += notional - sell_cost(s.get('buy_notional',
                                                     s['buy_prc'] * s['lots'] * LOT))
                sells.append(dict(code=s['code'], price=sp, lots=s['lots'],
                                  reason=s.get('reason'), buy_dt=s['buy_dt'],
                                  buy_prc=s['buy_prc'],
                                  buy_notional=s.get('buy_notional',
                                                     s['buy_prc'] * s['lots'] * LOT)))
                slots.remove(s)
    # ---- 开盘: 买入 (空位补入; 每仓预算 = 当前净值/TOPN 再投资口径) ----
    if fd is not None:
        row = score_df.loc[fd].dropna()
        ranked = row.sort_values(ascending=False).index.tolist()
        held_codes = {s['code'] for s in slots}
        eq_now = liquid + sum(
            (open_m.get((t, s['code'])) or close_m.get((t, s['code']))
             or s['buy_prc']) * s['lots'] * LOT for s in slots)
        budget = eq_now / TOPN
        bought = set()
        for code in ranked:
            if len(slots) >= TOPN:
                break
            if code in held_codes or code in bought:
                continue
            nm = name_m.get((t, code)) or name_last.get(code, '')
            if 'ST' in nm or '退' in nm:
                continue
            bp = open_m.get((t, code))
            pc = prev_m.get((t, code))
            if bp is None or bp <= 0:
                continue
            if pc and pc > 0 and bp / pc - 1 >= LIMIT_UP:
                continue
            fc = close_m.get((fd, code))
            fpc = prev_m.get((fd, code))
            if fc and fpc and fpc > 0 and fc / fpc - 1 >= LIMIT_UP:
                continue
            avail = min(liquid, budget)
            lots = int(avail // (LOT * bp))
            while lots >= 1:
                if lots * LOT * bp + buy_cost(lots * LOT * bp) <= avail:
                    break
                lots -= 1
            if lots < 1:
                break  # 资金不足最小一手 → 停止补仓 (与引擎一致)
            notional = lots * LOT * bp
            liquid -= notional + buy_cost(notional)
            s = dict(code=code, buy_dt=t, buy_prc=bp, lots=lots, peak=bp,
                     peak_dt=t, sell_flag=False, reason='',
                     buy_notional=notional)
            slots.append(s)
            bought.add(code)
            buys.append(dict(code=code, price=bp, lots=lots, budget_used=notional))
    # ---- 收盘: 更新峰值 + 写 sell_flag (次日开盘执行) ----
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
    state['slots'] = slots
    state['cash'] = liquid
    state['prev_day'] = t
    return sells, buys, state


def run_daily(score_path):
    market = load_market()
    tds, tdi = load_calendar()
    score = pd.read_feather(score_path).set_index('date').sort_index()
    last_fd = score.index.max()
    # 最新因子日 → 其次日开盘为执行日 (若价格已有, 说明已开盘; 决策仍给出)
    exec_i = tdi.get(last_fd, len(tds) - 1) + 1
    exec_day = tds[min(exec_i, len(tds) - 1)]
    state = {'slots': [], 'cash': CAPITAL, 'prev_day': None}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
    sells, buys, state = decide_orders(score, market, state, tds, tdi, exec_day)
    names = market['name_last']
    print(f'== h20tr20_t2 每日决策 @ 因子日 {last_fd} (执行: {exec_day} 开盘) ==')
    print(f'  当前净值口径资金: {state["cash"]:.0f} 现金 + 持仓')
    print(f'  卖出 (开盘先卖): ' + ('; '.join(
        f"{s['code']} {names.get(s['code'],'?')} {s['lots']}手 @{s['price']:.2f} ({s['reason']})"
        for s in sells) if sells else '无'))
    print(f'  买入 (开盘后补): ' + ('; '.join(
        f"{b['code']} {names.get(b['code'],'?')} {b['lots']}手 @{b['price']:.2f}"
        for b in buys) if buys else '无'))
    print(f'  收盘后持仓: ' + ('; '.join(
        f"{s['code']} (买 {s['buy_dt']}, 峰 {s['peak']:.2f}, "
        f"{'明日卖('+s['reason']+')' if s['sell_flag'] else '持有'})"
        for s in state['slots']) if state['slots'] else '空仓'))
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f'  状态已写入 {STATE_PATH}')


def replay_check(score_path):
    """回放校验: 用本决策器逐日重放全窗口, 与引擎 h20tr20_t2 逐笔对齐。"""
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
    state = {'slots': [], 'cash': CAPITAL, 'prev_day': None}
    trades, open_m = [], market['open']
    for d in sorted(score.index):
        if d not in tdi:
            continue
        exec_day = tds[tdi[d] + 1]
        sells, buys, state = decide_orders(score, market, state, tds, tdi, exec_day)
        for s in sells:
            notional = s['lots'] * LOT * s['price']
            gross = s['price'] / s['buy_prc'] - 1.0
            cost_b = max(5.0, s['buy_notional'] * 2.5e-4) + s['buy_notional'] * 1e-5
            cost_s = max(5.0, notional * 2.5e-4) + notional * 1e-5 + notional * 5e-4
            net = (notional - cost_s) / (s['buy_notional'] + cost_b) - 1.0
            trades.append(dict(buy_dt=s['buy_dt'], sell_dt=exec_day, code=s['code'],
                               buy_prc=s['buy_prc'], sell_prc=s['price'],
                               gross_pct=gross * 100, net_pct=net * 100,
                               hold_days=tdi[exec_day] - tdi[s['buy_dt']],
                               reason=s['reason']))
    rt = pd.DataFrame(trades)
    # 引擎期末强制清仓 (reason 为空串) 是回测框架的人工结算, 实盘决策器不执行 → 剔除对齐
    is_final = tr['reason'].fillna('').astype(str).str.strip() == ''
    et = tr[~is_final][['buy_dt', 'sell_dt', 'code', 'buy_prc',
                        'sell_prc', 'hold_days', 'reason']].copy()
    n_match = 0
    for _, a in et.iterrows():
        b = rt[(rt['code'] == a['code']) & (rt['buy_dt'] == a['buy_dt'])]
        if len(b) and abs(b.iloc[0]['sell_prc'] - a['sell_prc']) < 1e-6:
            n_match += 1
    n_final = int(is_final.sum())
    print(f'[replay] 引擎 {len(et)} 笔 (剔除 {n_final} 笔期末强制清仓) / '
          f'决策器 {len(rt)} 笔; 按 (code, buy_dt, sell_prc) 对齐 {n_match}/{len(et)} 笔')
    print(f'[replay] 引擎 cum_net {m["cum_net"]:+.2%} (基准)')
    ok = n_match == len(et)
    print('[replay] 结论: ' + ('一致 ✓ (runner 与 engine 同口径)' if ok
                               else '存在差异 ✗ — 检查二者口径'))
    return ok


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if p is None:
        print('usage: daily_runner_v2.py <score.fea> [--replay]')
        sys.exit(1)
    if '--replay' in sys.argv:
        replay_check(p)
    else:
        run_daily(p)
