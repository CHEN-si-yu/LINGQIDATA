#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_trader.py — 冠军系统纸面交易追踪器 (前向滚动验证)

协议 (ens_w2 + hold5s8-收盘卖):
  - 买入: 决策日次日开盘价成交 (整数手, ~5W)
  - 卖出: 持有满 5 个交易日 → 当日收盘价成交
  - 止损: 持仓期间收盘价较买入价 ≤ -8% → 次日开盘价卖出
  - 成本: 佣金万2.5(最低5元)+印花税万5(卖)+过户费万0.1

用法: python3 paper_trader.py [score.fea] [--date YYYYMMDD]
每次运行: 1) 结算可成交挂单 2) 依据最新打分更新决策/挂单 3) 打印状态
状态: paper_state.json (持仓/挂单/净值/交易历史/决策日志)
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strat_backtest import load_calendar, load_prices, trade_cost, CAPITAL, LOT_SIZE

HOLD_DAYS = 5
STOP_LOSS = 0.08
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state.json')


def _next_td(d, n, tds):
    i = tds.index(d) if d in tds else None
    if i is None:
        # 最近一个 <= d 的交易日
        cand = [x for x in tds if x <= d]
        if not cand:
            return None
        i = tds.index(cand[-1])
    j = i + n
    return tds[j] if 0 <= j < len(tds) else None


def main():
    score_path = None
    args = sys.argv[1:]
    run_date = None
    for a in args:
        if a.startswith('--date'):
            run_date = args[args.index(a) + 1]
        elif not a.startswith('--'):
            score_path = a
    if score_path is None:
        score_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'model_pred/2026q3/score_ens_w2.fea')
    score = pd.read_feather(score_path).set_index('date').sort_index()
    tds = load_calendar()
    open_map, close_map, prev_close_map, amount_map, name_map = load_prices()

    latest_score_date = score.index.max()
    max_price_date = max(d for (d, _c) in close_map.keys())  # 实际价格数据最新日
    today = run_date or latest_score_date
    print(f'== 纸面交易 @ 因子日 {today} (价格数据至 {max_price_date}) ==')

    state = {'cash': CAPITAL, 'position': None, 'orders': [], 'trades': [],
             'decisions': [], 'last_run': None}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)

    # ── 1) 结算挂单 (成交日 <= 价格最新日) ──
    for order in list(state['orders']):
        if order['date'] <= max_price_date:
            code, side = order['code'], order['side']
            px = open_map.get((order['date'], code)) if order['mode'] == 'open' \
                else close_map.get((order['date'], code))
            if px is None or px <= 0:
                continue  # 停牌无价 → 保留挂单
            if side == 'buy':
                lots = max(1, int(state['cash'] // (LOT_SIZE * px)))
                notional = lots * LOT_SIZE * px
                cost = trade_cost(notional, 'buy')
                state['cash'] -= (notional + cost)
                state['position'] = dict(code=code, lots=lots, buy_price=px,
                                         buy_date=order['date'],
                                         stop_price=px * (1 - STOP_LOSS))
                print(f'  [成交] {order["date"]} 买入 {code} {name_map.get(code,"?")} '
                      f'{lots}手 @ {px:.2f} (成本 {cost:.1f} 元)')
                order['status'] = 'filled_buy'
            else:  # sell
                pos = state['position']
                notional = pos['lots'] * LOT_SIZE * px
                cost = trade_cost(notional, 'sell')
                proceeds = notional - cost
                ret = (proceeds / (pos['lots'] * LOT_SIZE * pos['buy_price'] +
                       trade_cost(pos['lots'] * LOT_SIZE * pos['buy_price'], 'buy'))) - 1
                state['cash'] += proceeds
                state['trades'].append(dict(code=pos['code'],
                                            name=name_map.get(pos['code'], '?'),
                                            buy_date=pos['buy_date'],
                                            buy_price=pos['buy_price'],
                                            sell_date=order['date'], sell_price=px,
                                            ret_pct=round(ret * 100, 2),
                                            reason=order['reason']))
                print(f'  [成交] {order["date"]} 卖出 {pos["code"]} @ {px:.2f} '
                      f'收益 {ret * 100:+.2f}% ({order["reason"]})')
                state['position'] = None
                order['status'] = 'filled_sell'
            state['orders'].remove(order)

    # ── 2) 今日决策 ──
    pos = state['position']
    if today == state.get('last_run'):
        pass  # 同日重复运行不重复决策
    else:
        decision = None
        if pos:
            held_days = (tds.index(today) - tds.index(pos['buy_date'])) \
                if today in tds and pos['buy_date'] in tds else None
            latest_close = close_map.get((today, pos['code']))
            if latest_close and latest_close / pos['buy_price'] - 1 <= -STOP_LOSS \
                    and held_days is not None and held_days < HOLD_DAYS:
                sell_date = _next_td(today, 1, tds)
                state['orders'].append(dict(side='sell', code=pos['code'],
                                            date=sell_date, mode='open',
                                            reason='stop'))
                decision = f'止损: {pos["code"]} 收盘 {latest_close:.2f} 较买入 ' \
                           f'{(latest_close / pos["buy_price"] - 1) * 100:+.1f}% ≤ -8% → {sell_date} 开盘卖'
            elif held_days is not None and held_days >= HOLD_DAYS:
                state['orders'].append(dict(side='sell', code=pos['code'],
                                            date=today, mode='close',
                                            reason='hold5'))
                decision = f'持有满 {HOLD_DAYS} 天 → 今日收盘卖 {pos["code"]}'
            else:
                decision = f'继续持有 {pos["code"]} (第 {held_days}/{HOLD_DAYS} 天)'
        if pos is None:
            top1 = score.loc[today].sort_values(ascending=False).index[0]
            buy_date = _next_td(today, 1, tds)
            state['orders'].append(dict(side='buy', code=top1, date=buy_date,
                                        mode='open', reason='top1'))
            decision = f'空仓 → {buy_date} 开盘买入 Top1 {top1} ' \
                       f'{name_map.get(top1, "?")}'
        if decision:
            state['decisions'].append({'date': today, 'decision': decision})
            print(f'  [决策] {decision}')
        state['last_run'] = today

    # ── 3) 状态汇总 ──
    pos = state['position']
    pos_val = 0.0
    if pos:
        latest_px = close_map.get((max_price_date, pos['code']))
        if latest_px:
            pos_val = pos['lots'] * LOT_SIZE * latest_px
    equity = state['cash'] + pos_val
    print(f'  --- 状态 ---')
    print(f'  现金: {state["cash"]:.0f} 元 | 持仓: '
          f'{(pos["code"] + " " + str(pos["lots"]) + "手 @" + f"{pos["buy_price"]:.2f}") if pos else "无"}'
          f' | 净值: {equity:.0f} 元 ({(equity / CAPITAL - 1) * 100:+.2f}%)')
    if state['trades']:
        rets = [t['ret_pct'] for t in state['trades']]
        print(f'  已实现交易 {len(rets)} 笔: 累计 '
              f'{sum(rets):+.2f}% (逐笔复利 {(1 + pd.Series(rets) / 100).prod() - 1:+.2%})')
    print(f'  挂单: {len(state["orders"])} 笔 | 决策日志 {len(state["decisions"])} 条')
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
