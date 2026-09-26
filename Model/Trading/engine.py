#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py — 普适性交易策略回测引擎 (5W 资金 / 1~2 只持仓 / 隔日开盘执行 / 开盘换仓)

设计目标 (解决 V20 hold5s8-收盘卖 的资金回笼问题):
  - 决策: 每晚收盘后, 用当日因子日打分 (score 集) 决策 → 只作用于次日开盘。
  - 换仓: 开盘时"先卖后买"同步进行 (同一开盘价成交), 卖出资金当日即可用于买入,
    全程无需双倍资金 (旧冠军口径在换仓日开盘买入、收盘卖出, 需要 ≈2× 资金)。
  - 持仓: 1~2 只 (top_n ∈ {1,2}), 每仓位预算 = 总资金 / top_n, 整手(100股)买入。
  - 止盈止损: 收盘价触发 → 次日开盘执行 (固定止损 / 固定止盈 / 移动止损 / 利润锁定)。
  - 退出: 固定持有期 hold、排名退出 exit_rank (自适应持有)、no_repeat 续持、
    Top1 打分 expanding 分位择时 (空仓)。
  - 执行真实性: ST/退 过滤、一字涨停买不进、一字跌停卖不出顺延、整手、佣金万2.5
    (最低5元) 双向 + 印花税万5 (卖) + 过户费万0.1 双向。
  - 三种卖出口径: open (本框架主口径) / close_overlap (旧冠军口径, 需双倍资金,
    用于复现) / close_seq (收盘卖、次日开盘买, 资金安全但非开盘换仓)。

用法 (见 run_battery.py):
  from engine import load_market, load_scores, run_backtest
"""
import bisect
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"

# ---------------- 全局口径常量 ----------------
CAPITAL = 50000.0          # 5W 资金
LOT = 100                  # 整手
COMM_RATE = 2.5e-4         # 佣金万2.5
COMM_MIN = 5.0             # 最低佣金
STAMP_RATE = 5e-4          # 印花税万5 (卖)
TRANSFER_RATE = 1e-5       # 过户费万0.1
LIMIT_UP = 0.095           # 一字涨停过滤阈值 (open/prev_close-1 >= 此值 → 买不进)
LIMIT_DN = -0.095          # 一字跌停 (open/prev_close-1 <= 此值 → 卖不出)
WINDOW_START = "20250901"  # 官方 Test 集窗口 (打分日)
WINDOW_END = "20260901"

# 打分文件登记: 每个可评估的"模型打分集" (V1~V20 的最终/可评估打分产物)
SCORE_SETS = {}
for _v in ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
           "V12", "V13", "V14", "V15", "V16", "V17"]:
    SCORE_SETS[f"{_v}_allz"] = f"Model/{_v}/model_pred/2026q3/all_zscore_score.fea"
SCORE_SETS["V11_allz"] = "Model/V11/model_pred/2026q3/all_zscore_score.fea"
SCORE_SETS["V11_mixA"] = "Model/V11/model_pred/2026q3/score_mixA_top.fea"
SCORE_SETS["V11_mixB"] = "Model/V11/model_pred/2026q3/score_mixB_ic.fea"
SCORE_SETS["V11_mixC"] = "Model/V11/model_pred/2026q3/score_mixC_bal.fea"
SCORE_SETS["V11_ensw2"] = "Model/V11/model_pred/2026q3/score_ens_w2.fea"
SCORE_SETS["V18_f1"] = "Model/V18/model_pred/v18_f1.fea"
SCORE_SETS["V18_f2"] = "Model/V18/model_pred/v18_f2.fea"
SCORE_SETS["V20_ensw2"] = "Model/V20/model_pred/2026q3/score_ens_w2.fea"
SCORE_SETS["V22_10d"] = "Model/V22/model_pred/v22_label_ret_10d.fea"
SCORE_SETS["V22_20d"] = "Model/V22/model_pred/v22_label_ret_20d.fea"
SCORE_SETS["V23_ens23"] = "Model/V23/model_pred/2026q3/score_ens23.fea"

DEFAULT_CFG = dict(top_n=1, hold=1, no_repeat=False, exit_rank=None, min_hold=1,
                   stop_loss=None, take_profit=None, trail_pct=None,
                   rat_up=None, rat_floor=None, threshold_q=None,
                   exclude_st=True, exclude_limit_up=True, sell_limit_filter=True,
                   use_cost=True, sell_at="open",
                   trail_on_high=False, max_entry_gap=None)


# ---------------- 市场数据 ----------------
def load_calendar():
    cal = pd.read_parquet(PROJECT_ROOT + "data/calendar.parquet")
    tds = sorted(cal.loc[cal["is_open"] == 1, "date"].astype(str)
                 .str.replace("-", "", regex=False).tolist())
    tdi = {d: i for i, d in enumerate(tds)}
    return tds, tdi


def load_market():
    """载入窗口内价格/名称映射 (open/close/prev_close/name, 按 (date,code) 键)。"""
    df = pd.read_parquet(PROJECT_ROOT + "data/daily_adj.parquet",
                         columns=["stock_code", "trade_date", "open", "high",
                                  "close", "amount", "stock_name"])
    df["trade_date"] = df["trade_date"].str.replace("-", "", regex=False)
    df["code"] = df["stock_code"].str.replace(".SZ", "", regex=False) \
                                  .str.replace(".SH", "", regex=False)
    df = df.sort_values(["code", "trade_date"])
    df["prev_close"] = df.groupby("code")["close"].shift(1)
    # 行情末端随 daily_adj.parquet 数据更新滚动 (实盘日结需要当日价格)
    mkt_end = df["trade_date"].max()
    df = df[(df["trade_date"] >= "20250801") & (df["trade_date"] <= mkt_end)]
    open_m = df.set_index(["trade_date", "code"])["open"].to_dict()
    high_m = df.set_index(["trade_date", "code"])["high"].to_dict()
    close_m = df.set_index(["trade_date", "code"])["close"].to_dict()
    prev_m = df.set_index(["trade_date", "code"])["prev_close"].to_dict()
    name_m = df.set_index(["trade_date", "code"])["stock_name"].to_dict()
    name_last = df.drop_duplicates("code", keep="last") \
                 .set_index("code")["stock_name"].to_dict()
    return dict(open=open_m, high=high_m, close=close_m, prev=prev_m,
                name=name_m, name_last=name_last, max_date=df["trade_date"].max())


def load_scores(set_name, window=(WINDOW_START, WINDOW_END)):
    """读打分 fea → {date: {code: score}}, ranked[date]=降序代码列表, top1_by_date。"""
    path = PROJECT_ROOT + SCORE_SETS[set_name]
    df = pd.read_feather(path)
    df = df.set_index("date").sort_index()
    df = df.loc[(df.index >= window[0]) & (df.index <= window[1])]
    scores, ranked, top1 = {}, {}, {}
    for date in df.index:
        row = df.loc[date].dropna()
        scores[date] = row.to_dict()
        ranked[date] = row.sort_values(ascending=False).index.tolist()
        top1[date] = float(row.max()) if len(row) else np.nan
    return dict(scores=scores, ranked=ranked, top1=top1,
                dates=sorted(scores.keys()))


# ---------------- 成本 ----------------
def buy_cost(notional):
    return max(COMM_MIN, notional * COMM_RATE) + notional * TRANSFER_RATE


def sell_cost(notional):
    return max(COMM_MIN, notional * COMM_RATE) + notional * TRANSFER_RATE \
        + notional * STAMP_RATE


# ---------------- 回测引擎 ----------------
def run_backtest(scores, cfg, market, tds=None, tdi=None,
                 window=(WINDOW_START, WINDOW_END)):
    """执行回测 → (metrics dict, trades DataFrame, equity DataFrame)

    scores: load_scores() 返回值; cfg: dict (见 DEFAULT_CFG); market: load_market()。
    """
    cfg = {**DEFAULT_CFG, **cfg}
    top_n = int(cfg["top_n"]); hold = cfg["hold"]
    sell_at = cfg["sell_at"]
    ranked, top1s = scores["ranked"], scores["top1"]
    fdates = scores["dates"]
    if not fdates:
        return None, None, None
    fdi = {d: i for i, d in enumerate(fdates)}
    if tds is None:
        tds, tdi = load_calendar()
    open_m, close_m, prev_m = market["open"], market["close"], market["prev"]
    high_m = market.get("high", {})
    name_m, name_last = market["name"], market["name_last"]

    i1 = tdi.get(window[1])
    # 首个开盘执行日 = 窗口首打分日的次日 (20250902)
    i_start = tdi[window[0]] + 1
    # 尾仓结算缓冲: 至价格数据末尾 (随行情滚动), 期末日强制清仓
    i_end = min(i1 + 4, len(tds) - 1, tdi.get(market.get("max_date"), len(tds) - 1))

    def latest_fd(td):
        j = bisect.bisect_right(fdates, td) - 1
        return fdates[j] if j >= 0 else None

    cash = CAPITAL * cfg.get("capital_mult", 1.0)
    budget = cash / top_n
    positions = {}   # pos_id -> pos dict (同一代码在 close_overlap 换仓日可同时存在旧仓(待收盘卖)与新仓)
    _pos_seq = 0
    trades = []
    equity_rows = []
    hist_top1 = []       # expanding Top1 打分历史 (严格早于当前决策因子日)
    hist_fd = None
    skipped_buy = 0      # 买不进候选 (涨停/ST/无价)
    blocked_sell = 0     # 卖不出顺延 (跌停/无价)
    max_overlap = 0.0    # close_overlap 口径下同时占用资金峰值
    realized_eq = CAPITAL  # close_overlap 口径: 已实现净值
    n_days = 0

    def rank_now(code, td):
        fd = latest_fd(td)
        if fd is None:
            return 10 ** 9
        sc = scores["scores"][fd]
        if code not in sc:
            return 10 ** 9
        return 1 + int(sum(1 for v in sc.values() if v > sc[code]))

    for i in range(i_start, i_end + 1):
        t = tds[i]
        n_days += 1
        fd_open = latest_fd(tds[i - 1])  # 开盘决策可用打分 (<= t-1)
        # ---- 阈值择时: expanding 历史 Top1 分位 (严格历史, 无未来) ----
        if cfg["threshold_q"] is not None and fd_open is not None:
            if hist_fd is None:
                hist_fd = fdates[0]
            while hist_fd != fd_open and fdi[hist_fd] + 1 < len(fdates) \
                    and fdates[fdi[hist_fd] + 1] <= fd_open:
                hist_top1.append(top1s[hist_fd])
                hist_fd = fdates[fdi[hist_fd] + 1]
            allow_entry = True
            cur = top1s.get(fd_open, np.nan)
            if len(hist_top1) >= 20 and not np.isnan(cur):
                allow_entry = cur >= float(np.quantile(hist_top1, cfg["threshold_q"]))
        else:
            allow_entry = True

        closing_today = [pid for pid, p in positions.items()
                         if p.get("sell_flag") and p.get("sell_date") == t]
        closing_codes = set(positions[pid]["code"] for pid in closing_today)

        # ================= 开盘阶段 =================
        # --- 1) 卖出 (open 口径: 开盘卖; close 口径在收盘阶段卖) ---
        if sell_at == "open":
            for pid in list(positions.keys()):
                p = positions[pid]
                code = p["code"]
                if not p.get("sell_flag"):
                    continue
                sp = open_m.get((t, code))
                if sp is None or sp <= 0:
                    blocked_sell += 1
                    continue
                if cfg["sell_limit_filter"]:
                    pc = prev_m.get((t, code))
                    if pc and pc > 0 and sp / pc - 1 <= LIMIT_DN:
                        blocked_sell += 1
                        continue
                _execute_sell(code, p, t, i, sp, trades, name_last, cfg)
                cash += p["proceeds_net"]
                positions.pop(pid, None)

        # --- 2) 买入 (补满空位; 开盘卖出的资金当日即可买入 → 资金回笼闭环) ---
        #     每个决策 (因子日 fd 收盘打分) 只在 fd 的下一个交易日开盘执行一次;
        #     陈旧打分不重复触发买入 (与旧引擎"逐因子日推进"一致)
        fresh_decision = fd_open is not None and i == tdi.get(fd_open, -999) + 1
        if fresh_decision and allow_entry:
            n_closing = len(closing_today)
            n_occ = len(positions) - (n_closing if sell_at == "close_overlap" else 0)
            empty = top_n - n_occ
            if empty > 0:
                bought_today = set()
                held_codes = set(p["code"] for p in positions.values())
                for code in ranked[fd_open]:
                    if empty <= 0:
                        break
                    if code in bought_today:
                        continue
                    if code in held_codes:
                        # open/close_seq: 已持仓不重复买; close_overlap 旧口径允许
                        # 当日开盘重买收盘卖出中的同一标的 (旧引擎行为, 隐含 T+1 瑕疵)
                        if sell_at == "close_overlap" and code in closing_codes:
                            pass
                        else:
                            continue
                    nm = name_m.get((t, code)) or name_last.get(code, "")
                    if cfg["exclude_st"] and ("ST" in nm or "退" in nm):
                        skipped_buy += 1
                        continue
                    bp = open_m.get((t, code))
                    pc = prev_m.get((t, code))
                    if bp is None or bp <= 0:
                        skipped_buy += 1
                        continue
                    if cfg["exclude_limit_up"] and pc and pc > 0 \
                            and bp / pc - 1 >= LIMIT_UP:
                        skipped_buy += 1
                        continue
                    # 入场跳空过滤 (避免追高开盘): 开盘较昨收跳空 ≥ 阈值 → 顺延
                    if cfg["max_entry_gap"] is not None and pc and pc > 0 \
                            and bp / pc - 1 >= cfg["max_entry_gap"]:
                        skipped_buy += 1
                        continue
                    # 因子日一字涨停过滤 (旧引擎口径: 因子日 close/prev_close-1 >= 0.095 → 次日买不进)
                    if cfg["exclude_limit_up"]:
                        fc = close_m.get((fd_open, code))
                        fpc = prev_m.get((fd_open, code))
                        if fc and fpc and fpc > 0 and fc / fpc - 1 >= LIMIT_UP:
                            skipped_buy += 1
                            continue
                    if sell_at == "close_overlap":
                        avail = budget  # 旧口径: 开盘买入按满额预算 (双倍资金场景)
                    elif cfg.get("reinvest", True):
                        # 利润再投资: 每仓目标 = 当前总净值/top_n (等权再平衡),
                        # 单仓买入动用全部可用现金 (以目标额为上限) — 修复旧版
                        # "盈利滞留现金不复投"导致的净值低估 (avg_cash 曾 ~33%)
                        mv_other = sum(q["lots"] * LOT *
                                       (open_m.get((t, q["code"]))
                                        or close_m.get((t, q["code"]))
                                        or q["buy_prc"])
                                       for q in positions.values())
                        eq_now = cash + mv_other
                        avail = min(cash, eq_now / top_n)
                    else:
                        avail = min(cash, budget)
                    lots = int(avail // (LOT * bp))
                    while lots >= 1:
                        notional = lots * LOT * bp
                        if notional + buy_cost(notional) <= avail:
                            break
                        lots -= 1
                    if lots < 1:
                        break  # 资金不足最小一手 → 停止补仓
                    notional = lots * LOT * bp
                    cb = buy_cost(notional) if cfg["use_cost"] else 0.0
                    if sell_at == "close_overlap":
                        max_overlap = max(max_overlap,
                                          notional + sum(q["notional"] for q in positions.values()))
                    else:
                        cash -= notional + cb
                    _pos_seq += 1
                    positions[_pos_seq] = dict(buy_dt=t, buy_i=i, buy_prc=bp, lots=lots,
                                               code=code, notional=notional, cost_b=cb,
                                               entry_score=scores["scores"][fd_open].get(code),
                                               factor_dt=fd_open, peak=bp, locked=-np.inf,
                                               sell_flag=False, reason="")
                    bought_today.add(code)
                    empty -= 1

        # ================= 收盘阶段 =================
        # --- close 口径卖出 (close_overlap: 换仓日开盘已买新仓, 收盘卖旧仓) ---
        if sell_at.startswith("close"):
            for pid in list(positions.keys()):
                p = positions[pid]
                code = p["code"]
                if not p.get("sell_flag") or p.get("sell_date") != t:
                    continue
                sp = close_m.get((t, code))
                if sp is None or sp <= 0:
                    p["sell_date"] = tds[min(i + 1, len(tds) - 1)]
                    blocked_sell += 1
                    continue
                if cfg["sell_limit_filter"]:
                    pc = prev_m.get((t, code))
                    if pc and pc > 0 and sp / pc - 1 <= LIMIT_DN:
                        p["sell_date"] = tds[min(i + 1, len(tds) - 1)]
                        blocked_sell += 1
                        continue
                _execute_sell(code, p, t, i, sp, trades, name_last, cfg)
                if sell_at == "close_seq":
                    cash += p["proceeds_net"]
                else:  # close_overlap: 买入不受现金约束, 只记录已实现净值
                    realized_eq *= 1 + p["net_ret"]
                positions.pop(pid, None)

        # --- 持仓收盘检查: 触发 → 次日执行 (买入当日不触发, 与旧引擎一致) ---
        for pid, p in list(positions.items()):
            code = p["code"]
            cc = close_m.get((t, code))
            if cc is None or cc <= 0:
                continue
            held = i - p["buy_i"]
            if cfg["trail_on_high"]:
                hh = high_m.get((t, code))
                pk = max(cc, hh) if (hh is not None and hh > 0) else cc
                p["peak"] = max(p["peak"], pk)
            else:
                p["peak"] = max(p["peak"], cc)
            reason = None
            # 到期: 开盘买入 → 买入日收盘即算持有 1 个交易日 → 收盘已满 hold 天则次日开盘卖
            # (hold=1: 买入次日开盘卖; hold=5: 第 5 个交易日收盘后 → 次日开盘卖)
            if hold is not None and held >= hold - 1:
                if cfg["no_repeat"] and rank_now(code, t) <= top_n:
                    pass  # 仍是 TopN → 续持
                else:
                    reason = "expiry"
            if held >= 1:
                # 其余触发 (止损/止盈/移动/排名) 买入当日不检查, 与旧引擎一致
                if cfg["exit_rank"] is not None and held >= cfg["min_hold"] \
                        and rank_now(code, t) > cfg["exit_rank"]:
                    reason = "rank"
                if cfg["stop_loss"] is not None \
                        and cc <= p["buy_prc"] * (1 - cfg["stop_loss"]):
                    reason = "stop"
                if cfg["take_profit"] is not None \
                        and cc >= p["buy_prc"] * (1 + cfg["take_profit"]):
                    reason = "take"
                if cfg["trail_pct"] is not None \
                        and cc <= p["peak"] * (1 - cfg["trail_pct"]):
                    reason = "trail"
                if cfg["rat_up"] is not None and cfg["rat_floor"] is not None:
                    if cc >= p["buy_prc"] * (1 + cfg["rat_up"]):
                        p["locked"] = max(p["locked"], p["buy_prc"] * (1 + cfg["rat_floor"]))
                    if cc <= p["locked"]:
                        reason = "ratchet"
            if reason and not p.get("sell_flag"):
                p["sell_flag"] = True
                p["reason"] = reason
                # open 口径: 次日开盘卖; close 口径: 次日收盘卖 (旧引擎一致)
                p["sell_date"] = tds[min(i + 1, len(tds) - 1)]

        # ---- 期末强制清仓 (尾日开盘, 不计过滤) ----
        if i == i_end:
            for pid in list(positions.keys()):
                p = positions[pid]
                code = p["code"]
                sp = open_m.get((t, code)) or close_m.get((t, code))
                if sp is None:
                    continue
                _execute_sell(code, p, t, i, sp, trades, name_last, cfg)
                if sell_at == "close_overlap":
                    realized_eq *= 1 + p["net_ret"]
                else:
                    cash += p["proceeds_net"]
                positions.pop(pid, None)

        # --- 净值标记 ---
        if sell_at == "close_overlap":
            eq = realized_eq
            csh = np.nan
        else:
            mv = sum(p["lots"] * LOT * close_m.get((t, p["code"]), p["buy_prc"])
                     for p in positions.values())
            eq = cash + mv
            csh = cash
        equity_rows.append((t, eq, len(positions), csh))

    if not equity_rows:
        return None, None, None
    eq_df = pd.DataFrame(equity_rows, columns=["date", "equity", "npos", "cash"])
    eq_df["ret"] = eq_df["equity"].pct_change()
    trades_df = pd.DataFrame(trades)
    cap_used = CAPITAL * cfg.get("capital_mult", 1.0)
    return _metrics(eq_df, trades_df, cfg, n_days, skipped_buy, blocked_sell,
                    max_overlap, cap_used), trades_df, eq_df


def _execute_sell(code, p, t, i, sp, trades, name_last, cfg):
    notional_in = p["notional"] + p["cost_b"]
    cs = sell_cost(p["notional"]) if cfg["use_cost"] else 0.0
    proceeds = sp * p["lots"] * LOT
    net_ret = (proceeds - cs) / notional_in - 1.0
    gross_ret = sp / p["buy_prc"] - 1.0
    p["proceeds_net"] = proceeds - cs
    p["net_ret"] = net_ret
    p["cost_s"] = cs
    trades.append(dict(buy_dt=p["buy_dt"], sell_dt=t, code=code,
                       name=name_last.get(code, ""), factor_dt=p["factor_dt"],
                       entry_score=p["entry_score"], lots=p["lots"],
                       buy_prc=p["buy_prc"], sell_prc=sp,
                       gross_pct=gross_ret * 100, net_pct=net_ret * 100,
                       cost_yuan=p["cost_b"] + cs,
                       hold_days=i - p["buy_i"],
                       reason=p.get("reason", "final")))


def _metrics(eq_df, trades_df, cfg, n_days, skipped_buy, blocked_sell, max_overlap,
            cap_used=CAPITAL):
    r = eq_df["ret"].dropna()
    n = len(r)
    if n == 0:
        return None
    cum = float((1 + r).prod() - 1)
    ann = float((1 + cum) ** (252.0 / max(n, 1)) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 1e-12 else 0.0
    eq = eq_df["equity"].values
    peak = np.maximum.accumulate(eq)
    maxdd = float((eq / peak - 1).min())
    h1 = r.iloc[: n // 2]; h2 = r.iloc[n // 2:]
    cum_h1 = float((1 + h1).prod() - 1); cum_h2 = float((1 + h2).prod() - 1)
    sh_h1 = float(h1.mean() / h1.std() * np.sqrt(252)) if h1.std() > 1e-12 else 0.0
    sh_h2 = float(h2.mean() / h2.std() * np.sqrt(252)) if h2.std() > 1e-12 else 0.0

    csh = eq_df["cash"].dropna()
    m = dict(
        cum_net=cum, ann_net=ann, sharpe=sharpe, maxdd=maxdd,
        cum_add=float(r.sum()), n_days=n, n_trades=int(len(trades_df)),
        empty_days=int((eq_df["npos"] == 0).sum()),
        avg_npos=float(eq_df["npos"].mean()),
        avg_cash_ratio=float((csh / eq_df.loc[csh.index, "equity"]).mean())
        if len(csh) else np.nan,
        h1_cum=cum_h1, h2_cum=cum_h2, h1_sharpe=sh_h1, h2_sharpe=sh_h2,
        skipped_buy=skipped_buy, blocked_sell=blocked_sell,
        max_overlap=max_overlap,
    )
    if len(trades_df):
        nets = trades_df["net_pct"].values
        m.update(
            win_rate=float((nets > 0).mean()),
            avg_hold=float(trades_df["hold_days"].mean()),
            med_trade=float(np.median(nets)),
            best_trade=float(nets.max()), worst_trade=float(nets.min()),
            turnover=float((trades_df["buy_prc"] * trades_df["lots"] * 100).sum()) / CAPITAL,
            cost_drag=float(trades_df["cost_yuan"].sum()) / cap_used,
            cum_trade=float((1 + nets / 100).prod() - 1),
        )
        tpy = m["n_trades"] * 252.0 / max(n_days, 1)
        m["sharpe_trade"] = float(nets.mean() / nets.std() * np.sqrt(tpy)) \
            if nets.std() > 1e-12 else 0.0
        mid = eq_df["date"].iloc[n // 2]
        h1t = trades_df[trades_df["buy_dt"] <= mid]["net_pct"]
        h2t = trades_df[trades_df["buy_dt"] > mid]["net_pct"]
        m["h1_cum_t"] = float((1 + h1t / 100).prod() - 1) if len(h1t) else np.nan
        m["h2_cum_t"] = float((1 + h2t / 100).prod() - 1) if len(h2t) else np.nan
        m["h1_win"] = float((h1t > 0).mean()) if len(h1t) else np.nan
        m["h2_win"] = float((h2t > 0).mean()) if len(h2t) else np.nan
    else:
        m.update(win_rate=np.nan, avg_hold=np.nan, med_trade=np.nan,
                 best_trade=np.nan, worst_trade=np.nan, turnover=0.0,
                 cost_drag=0.0, cum_trade=0.0, sharpe_trade=0.0,
                 h1_cum_t=np.nan, h2_cum_t=np.nan, h1_win=np.nan, h2_win=np.nan)
    return m
