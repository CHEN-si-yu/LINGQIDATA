#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strat_backtest.py — V11 交易策略回测框架 (面向 ~5W 个人资金)

输入: 模型打分文件 (feather, index=因子日期, columns=股票代码) — 如 V9 的
      Model/V9/model_pred/2026q3/all_zscore_score.fea
输出: 各策略配置的绩效表 (净收益/年化/Sharpe/MaxDD/胜率/成本拖累/空仓天数)
      + 最优配置的逐笔交易日志

执行口径 (与 analysis.py 回测一致的真实价格, 但扩展):
  - 因子日 d 收盘打分 → d+1 开盘买入 → 持有 hold 个交易日 → 卖出日开盘卖出
  - hold=1: 每日换手 (与 V9 Top1 口径一致)
  - hold>=2: 顺序持仓 (单账户单票, 卖出次日再买下一只, 无重叠)
  - T+1 合规: 买入后至少持有 1 个交易日
成本模型 (5W 账户, A股):
  - 佣金 万2.5 (最低 5 元) 双向
  - 印花税 万5 卖出单边
  - 过户费 万0.1 双向
  - 整手买入 (100 股), 可用资金 50000 元
过滤 (可选):
  - exclude_st: 排除名称含 ST/退
  - exclude_limit_up: 因子日涨停 (close/prev_close-1 >= 0.095) 跳过
  - buy_gap_limit: 买入日开盘相对因子日收盘跳空 >= 阈值 (一字板买不进) 跳过
  - min_amount: 买入日成交额下限
择时 (可选):
  - score_threshold_q: 仅当 Top1 打分 >= 该日之前(含)60 日 Top1 打分的 q 分位时开仓,
    否则空仓 (expanding, 无未来函数)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
CAPITAL = 50000.0          # 个人资金 5W
LOT_SIZE = 100             # 整手
COMM_RATE = 2.5e-4         # 佣金万2.5
COMM_MIN = 5.0             # 最低佣金 5 元
STAMP_RATE = 5e-4          # 印花税万5 (卖出)
TRANSFER_RATE = 1e-5       # 过户费万0.1 (双向)

# 回测窗口 (与模型评价口径一致)
WINDOW_START = '20250901'
WINDOW_END = '20260901'


# ---------------------------------------------------------------
# 数据加载 (缓存)
# ---------------------------------------------------------------
def load_score(path):
    df = pd.read_feather(path)
    if 'date' in df.columns:
        df = df.set_index('date')
    df = df.sort_index()
    return df


def load_calendar():
    cal = pd.read_parquet(PROJECT_ROOT + 'data/calendar.parquet')
    tds = sorted(cal.loc[cal['is_open'] == 1, 'date'].astype(str)
                 .str.replace('-', '', regex=False).tolist())
    return tds


def load_prices():
    df = pd.read_parquet(PROJECT_ROOT + 'data/daily_adj.parquet',
                         columns=['stock_code', 'trade_date', 'open', 'close',
                                  'amount', 'stock_name'])
    df['trade_date'] = df['trade_date'].str.replace('-', '')
    df['code'] = df['stock_code'].str.replace('.SZ', '').str.replace('.SH', '')
    df = df.sort_values(['code', 'trade_date'])
    df['prev_close'] = df.groupby('code')['close'].shift(1)
    open_map = df.set_index(['trade_date', 'code'])['open'].to_dict()
    close_map = df.set_index(['trade_date', 'code'])['close'].to_dict()
    prev_close_map = df.set_index(['trade_date', 'code'])['prev_close'].to_dict()
    amount_map = df.set_index(['trade_date', 'code'])['amount'].to_dict()
    name_map = df.drop_duplicates('code').set_index('code')['stock_name'].to_dict()
    return open_map, close_map, prev_close_map, amount_map, name_map


# ---------------------------------------------------------------
# 交易成本
# ---------------------------------------------------------------
def trade_cost(notional, side):
    """side: 'buy' / 'sell' → 单边成本 (元)"""
    comm = max(COMM_MIN, notional * COMM_RATE)
    transfer = notional * TRANSFER_RATE
    stamp = notional * STAMP_RATE if side == 'sell' else 0.0
    return comm + transfer + stamp


# ---------------------------------------------------------------
# 回测引擎
# ---------------------------------------------------------------
def run_backtest(score_df, open_map, close_map, prev_close_map, name_map,
                 amount_map=None, hold=1, top_n=1, use_cost=True,
                 exclude_st=False, exclude_limit_up=False,
                 buy_gap_limit=None, min_amount=None, score_threshold_q=None,
                 no_repeat_skip=False, stop_loss=None, min_hold=None,
                 exit_rank=None,
                 window_start=WINDOW_START, window_end=WINDOW_END,
                 tds=None, verbose=False):
    """执行回测, 返回 (metrics dict, trade_df, daily_ret series)

    no_repeat_skip: Top1 与上一持仓相同则当日不换仓 (省成本, 继续持有)
    stop_loss: 持仓期间收盘价相对买入价跌幅 <= -X 则次日开盘止损卖出 (X 为正数)
    min_hold: 持仓最少交易日数 (早于此不因 exit_rank 退出)
    exit_rank: 自适应退出 — 持仓股当日打分排名 > exit_rank 且持仓 >= min_hold 天 → 次日开盘卖出
    """
    if tds is None:
        tds = load_calendar()
    td_idx = {d: i for i, d in enumerate(tds)}

    def next_td(d, n=1):
        i = td_idx.get(d)
        if i is None:
            return None
        j = i + n
        return tds[j] if 0 <= j < len(tds) else None

    factor_dates = [d for d in score_df.index if window_start <= d <= window_end]
    if not factor_dates:
        return None, None, None

    records = []
    day_rets = []        # (factor_date, gross_ret, net_ret, invested) — 每次开仓一笔
    threshold_hist = []  # Top1 打分历史 (用于 expanding 分位, 每日都记录)

    def pick_codes(scores, date, buy_date, n):
        """按打分取前 n 只 (逐次跳过不可执行候选)。"""
        picked = []
        for code in scores.index:
            if len(picked) >= n:
                break
            name = name_map.get(code, '')
            if exclude_st and ('ST' in name or '退' in name):
                continue
            if exclude_limit_up:
                fc = close_map.get((date, code))
                pc = prev_close_map.get((date, code))
                if fc is not None and pc is not None and pc > 0 and fc / pc - 1.0 >= 0.095:
                    continue
            if buy_gap_limit is not None:
                bo = open_map.get((buy_date, code))
                pc = close_map.get((date, code))
                if bo is not None and pc is not None and pc > 0 and bo / pc - 1.0 >= buy_gap_limit:
                    continue
            if min_amount is not None:
                amt = amount_map.get((buy_date, code)) if amount_map else None
                if amt is not None and amt < min_amount:
                    continue
            picked.append(code)
        return picked

    pos = None  # dict: code -> 持仓信息; 无持仓时 None
    cursor = 0
    first_factor_dt = None
    last_sell_dt = None
    while cursor < len(factor_dates):
        date = factor_dates[cursor]
        scores = score_df.loc[date].sort_values(ascending=False)
        buy_date = next_td(date, 1)
        if buy_date is None:
            break

        # ── 打分阈值: expanding 全历史分位 (每日记录 top1 打分, 无未来函数) ──
        top_score = float(scores.iloc[0]) if len(scores) else np.nan
        skip_thresh = False
        if score_threshold_q is not None and not np.isnan(top_score):
            if len(threshold_hist) >= 20:
                q = float(np.quantile(threshold_hist, score_threshold_q))
                if top_score < q:
                    skip_thresh = True
        threshold_hist.append(top_score)

        if pos is None:
            # ── 空仓: 尝试开仓 ──
            if skip_thresh:
                cursor += 1
                continue
            picked = pick_codes(scores, date, buy_date, top_n)
            if not picked:
                cursor += 1
                continue
            new_pos = {}
            budget_per = CAPITAL / len(picked)
            for code in picked:
                bp = open_map.get((buy_date, code))
                if bp is None or bp <= 0:
                    continue
                lots = max(1, int(budget_per // (LOT_SIZE * bp)))
                notional = lots * LOT_SIZE * bp
                cost_b = trade_cost(notional, 'buy') if use_cost else 0.0
                new_pos[code] = dict(buy_dt=buy_date, buy_prc=bp, lots=lots,
                                     notional=notional, cost_b=cost_b,
                                     planned_sell=next_td(buy_date, hold),
                                     factor_dt=date, score=float(scores[code]),
                                     stopped=False)
            if not new_pos:
                cursor += 1
                continue
            pos = new_pos
            if first_factor_dt is None:
                first_factor_dt = date
            cursor += 1
            continue

        # ── 持仓中: 先做 roll (重复不换仓) / 自适应退出 / 止损检查 ──
        if no_repeat_skip:
            picked = pick_codes(scores, date, buy_date, top_n)
            if picked and set(picked) == set(pos.keys()):
                for p in pos.values():
                    ns = next_td(p['planned_sell'], 1)
                    if ns is not None:
                        p['planned_sell'] = ns
                cursor += 1
                continue

        if exit_rank is not None:
            # 自适应退出: 持仓排名跌出前 exit_rank (且持仓 >= min_hold 天) → 次日开盘卖
            md = min_hold if min_hold is not None else hold
            for code, p in pos.items():
                if p['stopped']:
                    continue
                bi = td_idx.get(p['buy_dt'])
                di = td_idx.get(date)
                held_days = (di - bi) if (bi is not None and di is not None) else 0
                if held_days < md:
                    continue
                rank = 1 + int((scores > scores.get(code, -np.inf)).sum())
                if rank > exit_rank:
                    p['planned_sell'] = next_td(date, 1)
                    p['exited'] = True

        if stop_loss is not None:
            for code, p in pos.items():
                if p['stopped']:
                    continue
                i = td_idx.get(p['buy_dt'])
                planned_i = td_idx.get(p['planned_sell'])
                if i is None or planned_i is None:
                    continue
                for j in range(i + 1, planned_i):
                    dd = tds[j]
                    cc = close_map.get((dd, code))
                    if cc is not None and cc > 0 and cc / p['buy_prc'] - 1 <= -stop_loss:
                        p['planned_sell'] = next_td(dd, 1)
                        p['stopped'] = True
                        break

        sell_date = max(p['planned_sell'] for p in pos.values())
        if next_td(date, 1) == sell_date or next_td(date, 1) is not None \
                and td_idx.get(next_td(date, 1)) >= td_idx.get(sell_date, 0):
            # ── 平仓 (sell_date 开盘卖出) ──
            day_gross, day_net, day_inv = [], [], []
            for code, p in pos.items():
                sp = open_map.get((p['planned_sell'], code))
                if sp is None or sp <= 0:
                    continue
                gross = sp / p['buy_prc'] - 1.0
                if use_cost:
                    cost_s = trade_cost(p['notional'], 'sell')
                    net = (sp * p['lots'] * LOT_SIZE - cost_s) / \
                          (p['notional'] + p['cost_b']) - 1.0
                else:
                    net = gross
                day_gross.append(gross)
                day_net.append(net)
                day_inv.append(p['notional'])
                records.append({
                    'factor_dt': p['factor_dt'], 'buy_dt': p['buy_dt'],
                    'code': code, 'name': name_map.get(code, '?'),
                    'score': p['score'], 'lots': p['lots'],
                    'buy_prc': p['buy_prc'], 'sell_dt': p['planned_sell'],
                    'sell_prc': sp, 'gross_pct': gross * 100,
                    'net_pct': net * 100,
                    'cost_yuan': (gross - net) * p['notional'],
                    'stopped': p['stopped'],
                    'exited': p.get('exited', False),
                    'hold_days': td_idx[p['planned_sell']] - td_idx[p['buy_dt']],
                })
                last_sell_dt = p['planned_sell']
            if day_gross:
                day_rets.append((date, float(np.mean(day_gross)),
                                 float(np.mean(day_net)), float(np.sum(day_inv))))
            pos = None
            # 同一因子日可在次日开盘再开仓 → 不前进 cursor, 重新处理当日
            continue
        cursor += 1

    if not records:
        return None, None, None
    trade_df = pd.DataFrame(records)
    dr = pd.DataFrame(day_rets, columns=['date', 'gross', 'net', 'invested']).set_index('date')

    # 组合净值曲线 (逐笔复利)
    gross_r = dr['gross']
    net_r = dr['net']
    n_trades = len(trade_df)
    # 实际跨度: 首笔因子日 → 末笔卖出日 (交易日数)
    if first_factor_dt is not None and last_sell_dt is not None \
            and first_factor_dt in td_idx and last_sell_dt in td_idx:
        elapsed_days = td_idx[last_sell_dt] - td_idx[first_factor_dt]
    else:
        elapsed_days = len(dr)
    elapsed_days = max(elapsed_days, 1)
    tpy = n_trades * 252.0 / elapsed_days  # 年化换手次数

    def stats(r):
        r = r.dropna()
        if len(r) == 0:
            return {}
        cum = (1 + r).prod() - 1
        cum_add = r.sum()
        ann = (1 + cum) ** (252.0 / elapsed_days) - 1
        sharpe = r.mean() / r.std() * np.sqrt(tpy) if r.std() > 1e-12 else 0.0
        eq = (1 + r).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        win = (r > 0).mean()
        return {'cum': cum, 'cum_add': cum_add, 'ann': ann, 'sharpe': sharpe,
                'maxdd': dd, 'win': win, 'n': len(r), 'mean': r.mean()}

    metrics = {'gross': stats(gross_r), 'net': stats(net_r)}
    metrics['n_trades'] = n_trades
    metrics['n_days'] = len(dr)
    metrics['elapsed_days'] = elapsed_days
    metrics['skip_days'] = int((dr['invested'] == 0).sum())
    metrics['cost_drag'] = float((dr['gross'] - dr['net']).sum())  # 累计成本拖累
    return metrics, trade_df, dr


def summarize(metrics):
    if metrics is None:
        return {}
    g, n = metrics['gross'], metrics['net']
    return {
        '净累计%': round(n['cum'] * 100, 2),
        '净累加%': round(n['cum_add'] * 100, 2),
        '毛累计%': round(g['cum'] * 100, 2),
        '净年化%': round(n['ann'] * 100, 2),
        'Sharpe': round(n['sharpe'], 3),
        'MaxDD%': round(n['maxdd'] * 100, 2),
        '胜率%': round(n['win'] * 100, 2),
        '日均%': round(n['mean'] * 100, 4),
        '交易数': metrics['n_trades'],
        '空仓日': metrics['skip_days'],
        '成本拖累%': round(metrics['cost_drag'] * 100, 2),
    }


# ---------------------------------------------------------------
# IC 评估 (与 analysis.py 同口径: 可交易池 Spearman + Top1 label)
# ---------------------------------------------------------------
def eval_ic(score_df, top_n=1):
    ret = pd.read_feather(PROJECT_ROOT + 'trainingdata/label_ret_1d.fea').set_index('index')
    buy = pd.read_feather(PROJECT_ROOT + 'trainingdata/buyable_mask.fea').set_index('date')
    rank_ics, top_rets = [], []
    for date in score_df.index:
        if date not in ret.index or not (WINDOW_START <= date <= WINDOW_END):
            continue
        scores = score_df.loc[date]
        label = ret.loc[date].reindex(scores.index)
        if date in buy.index:
            b = pd.to_numeric(buy.loc[date].reindex(scores.index), errors='coerce')
        else:
            b = pd.Series(np.nan, index=scores.index)
        tradable = b.gt(0.5) & label.notna()
        if tradable.sum() < 50:
            continue
        x = scores[tradable].rank()
        y = label[tradable].rank()
        ic = x.corr(y)
        rank_ics.append(ic)
        top_codes = scores[tradable].sort_values(ascending=False).index[:top_n]
        top_rets.append(label[top_codes].mean() * 100)
    rank_ics = pd.Series(rank_ics, dtype=float)
    top_rets = pd.Series(top_rets, dtype=float)
    return {
        'RankIC': rank_ics.mean(),
        'RankICIR': rank_ics.mean() / rank_ics.std() if rank_ics.std() > 0 else 0.0,
        'top_return': top_rets.mean(),
    }


# ---------------------------------------------------------------
# 主入口: 策略网格
# ---------------------------------------------------------------
def main():
    score_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else \
        PROJECT_ROOT + 'Model/V9/model_pred/2026q3/all_zscore_score.fea'
    mode = 'focus2' if '--focus2' in sys.argv else 'base'
    if '--battery' in sys.argv:
        mode = 'battery'
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[strat] 加载打分: {score_path}")
    score_df = load_score(score_path)
    print(f"[strat] 打分 {score_df.shape[0]} 天 × {score_df.shape[1]} 股票, "
          f"{score_df.index.min()} ~ {score_df.index.max()}")

    ic = eval_ic(score_df)
    print(f"[strat] 打分 IC: RankIC={ic['RankIC']:+.4f}  "
          f"RankICIR={ic['RankICIR']:+.4f}  top_return={ic['top_return']:+.4f}")

    open_map, close_map, prev_close_map, amount_map, name_map = load_prices()
    tds = load_calendar()
    print(f"[strat] 价格/日历加载完成 ({len(tds)} 交易日)")

    base = dict(hold=1, top_n=1, use_cost=False)
    F = dict(use_cost=True, exclude_st=True, exclude_limit_up=True, buy_gap_limit=0.095)
    if mode == 'battery':
        # ── 标准策略电池: 跨打分文件的统一评估口径 ──
        configs = [
            dict(base, **F),
            dict(base, **F, hold=5, stop_loss=0.08),
            dict(base, **F, hold=10, stop_loss=0.08),
            dict(base, **F, hold=5, stop_loss=0.08, score_threshold_q=0.5),
            dict(base, **F, hold=5, stop_loss=0.08, no_repeat_skip=True),
            dict(base, **F, no_repeat_skip=True, hold=3),
        ]
        names = ['hold1', 'hold5s8', 'hold10s8', 'hold5s8q0.5', 'hold5s8norep', 'norep+hold3']
        table, rows, best = [], [], None
        for cfg_name, cfg in zip(names, configs):
            metrics, trade_df, dr = run_backtest(score_df, open_map, close_map,
                                                 prev_close_map, name_map,
                                                 amount_map=amount_map, tds=tds, **cfg)
            s = summarize(metrics)
            if s:
                s['配置'] = cfg_name
                rows.append(s)
                print(f"  {cfg_name:<14} 净累计={s['净累计%']:>8.2f}%  年化={s['净年化%']:>7.2f}%  "
                      f"Sharpe={s['Sharpe']:>6.3f}  MaxDD={s['MaxDD%']:>7.2f}%  "
                      f"胜率={s['胜率%']:>5.1f}%  交易={s['交易数']}  成本拖累={s['成本拖累%']:>6.2f}%")
        table = pd.DataFrame(rows).set_index('配置')
        print("\n===== 标准策略电池 =====")
        print(table.round(2).to_string())
        table.to_csv(os.path.join(out_dir, 'strat_battery.csv'))
        print(f"\n[strat] 电池结果已存 strat_battery.csv; "
              f"IC: RankIC={ic['RankIC']:+.4f} IR={ic['RankICIR']:+.4f} "
              f"top_return={ic['top_return']:+.4f}")
        return
    if mode == 'focus2':
        # ── 第二轮: hold/止损 稳健性 + 组合策略 ──
        configs = [
            dict(base, **F, hold=5, stop_loss=0.05),
            dict(base, **F, hold=5, stop_loss=0.08),
            dict(base, **F, hold=5, stop_loss=0.10),
            dict(base, **F, hold=5, stop_loss=0.12),
            dict(base, **F, hold=4, stop_loss=0.08),
            dict(base, **F, hold=6, stop_loss=0.08),
            dict(base, **F, hold=10, stop_loss=0.08),
            dict(base, **F, hold=10, stop_loss=0.10),
            dict(base, **F, hold=3, stop_loss=0.08),
            dict(base, **F, hold=5, stop_loss=0.08, score_threshold_q=0.25),
            dict(base, **F, hold=5, stop_loss=0.08, score_threshold_q=0.5),
            dict(base, **F, hold=5, stop_loss=0.08, no_repeat_skip=True),
            dict(base, **F, hold=2, stop_loss=0.08, no_repeat_skip=True),
            dict(base, **F, hold=5, stop_loss=0.10, score_threshold_q=0.25),
            dict(base, **F, hold=5, stop_loss=0.08, top_n=1),
        ]
        names = [
            'hold5+stop5%', 'hold5+stop8%', 'hold5+stop10%', 'hold5+stop12%',
            'hold4+stop8%', 'hold6+stop8%', 'hold10+stop8%', 'hold10+stop10%',
            'hold3+stop8%', 'hold5+stop8%+q0.25', 'hold5+stop8%+q0.5',
            'hold5+stop8%+不换仓', 'hold2+stop8%+不换仓',
            'hold5+stop10%+q0.25', 'hold5+stop8%(复跑)',
        ]
    else:
        configs = [
        # ── 基线 (V9 口径, 无成本无过滤) ──
        dict(base),
        # ── 成本影响 ──
        dict(base, use_cost=True),
        # ── 执行过滤 ──
        dict(base, **F),
        dict(base, **F, min_amount=5e6),
        # ── 持有期 (顺序持仓, 降换手省成本) ──
        dict(base, **F, hold=2),
        dict(base, **F, hold=3),
        dict(base, **F, hold=5),
        # ── 重复不换仓 (Top1 不变则继续持有, 省往返成本) ──
        dict(base, **F, no_repeat_skip=True),
        dict(base, **F, no_repeat_skip=True, hold=3),
        # ── 止损 ──
        dict(base, **F, hold=5, stop_loss=0.08),
        dict(base, **F, stop_loss=0.08),
        # ── 打分阈值择时 (空仓机制, expanding 分位) ──
        dict(base, **F, score_threshold_q=0.25),
        dict(base, **F, score_threshold_q=0.5),
        # ── Top-N 分散 ──
        dict(base, **F, top_n=2),
        dict(base, **F, top_n=3),
    ]
        names = [
            '基线(无成本)', '成本', '成本+过滤', '+流动性5e6',
            'hold2', 'hold3', 'hold5',
            '重复不换仓', '重复不换仓+hold3',
            'hold5+止损8%', 'hold1+止损8%',
            '阈值q0.25', '阈值q0.5',
            'Top2', 'Top3',
        ]

    rows = []
    best = None
    for cfg_name, cfg in zip(names, configs):
        metrics, trade_df, dr = run_backtest(score_df, open_map, close_map,
                                             prev_close_map, name_map,
                                             amount_map=amount_map, tds=tds, **cfg)
        s = summarize(metrics)
        if s:
            s['配置'] = cfg_name
            rows.append(s)
            print(f"  {cfg_name:<14} 净累计={s['净累计%']:>8.2f}%  年化={s['净年化%']:>7.2f}%  "
                  f"Sharpe={s['Sharpe']:>6.3f}  MaxDD={s['MaxDD%']:>7.2f}%  "
                  f"胜率={s['胜率%']:>5.1f}%  交易={s['交易数']}  空仓={s['空仓日']}  "
                  f"成本拖累={s['成本拖累%']:>6.2f}%")
            if best is None or s['净累计%'] > best[1]['净累计%']:
                best = (cfg_name, s, trade_df, dr)

    table = pd.DataFrame(rows).set_index('配置')
    print("\n===== 策略网格总表 (按净累计排序) =====")
    print(table.sort_values('净累计%', ascending=False).round(2).to_string())

    out_csv = os.path.join(out_dir, 'strat_grid.csv')
    table.to_csv(out_csv)
    print(f"\n[strat] 网格结果已存: {out_csv}")

    if best is not None:
        cfg_name, s, trade_df, dr = best
        out_log = os.path.join(out_dir, f'strat_best_{cfg_name}.csv')
        trade_df.to_csv(out_log, index=False)
        print(f"[strat] 最优配置 [{cfg_name}] 交易日志: {out_log} "
              f"(净累计 {s['净累计%']:.2f}%)")


if __name__ == '__main__':
    main()
