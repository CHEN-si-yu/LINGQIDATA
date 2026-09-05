#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
battery.py — 策略电池定义 (全部可回测的策略变体)

三个电池:
  B1 主电池: 基线/持有期/续持/止损/止盈/组合/移动止损/排名退出/阈值择时/利润锁定 (~70 个)
  B2 细化网格: hold×止损 网格 + 止盈/移动/排名/阈值细化 (~150 个)
  SELLMODE: 三种卖出口径对照 (open / close_overlap / close_seq) × 每 score 集

口径约定 (与 engine.py 一致):
  - 全部真实成本 (佣金万2.5 最低5 + 印花税万5 卖 + 过户费万0.1), ST/退 过滤,
    一字涨停买不进, 一字跌停卖不出顺延, 整手买入。
  - 止盈/止损/移动/排名退出 = 收盘价触发 → 次日开盘执行。
"""


def _cfg(name, group, **kw):
    return dict(name=name, group=group, **kw)


def battery_B1():
    """主电池: 有代表性的策略变体。"""
    B = []
    # ---- A. 基线 (隔日换仓) ----
    B += [_cfg("A01_daily_top1", "A基线", top_n=1, hold=1),
          _cfg("A02_daily_top2", "A基线", top_n=2, hold=1)]
    # ---- B. 固定持有期 ----
    for h in (2, 3, 5, 7, 10, 20):
        B.append(_cfg(f"B01_hold{h}_t1", "B持有期", top_n=1, hold=h))
    for h in (2, 3, 5, 7, 10, 20):
        B.append(_cfg(f"B02_hold{h}_t2", "B持有期", top_n=2, hold=h))
    # ---- C. 续持 (norep): 仍在前 TopN 则顺延 ----
    for h in (2, 3, 5):
        B.append(_cfg(f"C01_nr_hold{h}_t1", "C续持", top_n=1, hold=h, no_repeat=True))
        B.append(_cfg(f"C02_nr_hold{h}_t2", "C续持", top_n=2, hold=h, no_repeat=True))
    # ---- D. 固定止损 (hold5) ----
    for sl in (3, 5, 8, 10):
        B.append(_cfg(f"D01_h5_sl{sl}_t1", "D止损", top_n=1, hold=5, stop_loss=sl / 100))
        B.append(_cfg(f"D02_h5_sl{sl}_t2", "D止损", top_n=2, hold=5, stop_loss=sl / 100))
    # ---- E. 固定止盈 (hold5) ----
    for tp in (10, 20, 30):
        B.append(_cfg(f"E01_h5_tp{tp}_t1", "E止盈", top_n=1, hold=5, take_profit=tp / 100))
        B.append(_cfg(f"E02_h5_tp{tp}_t2", "E止盈", top_n=2, hold=5, take_profit=tp / 100))
    # ---- F. 止损+止盈 组合 (hold5, sl8) ----
    for tp in (15, 25, 40):
        B.append(_cfg(f"F01_h5_sl8_tp{tp}_t1", "F组合", top_n=1, hold=5,
                      stop_loss=0.08, take_profit=tp / 100))
        B.append(_cfg(f"F02_h5_sl8_tp{tp}_t2", "F组合", top_n=2, hold=5,
                      stop_loss=0.08, take_profit=tp / 100))
    # ---- G. 移动止损 (从最高收盘回撤) ----
    for tr in (5, 8, 12):
        B.append(_cfg(f"G01_h10_tr{tr}_t1", "G移动止损", top_n=1, hold=10, trail_pct=tr / 100))
        B.append(_cfg(f"G02_h10_tr{tr}_t2", "G移动止损", top_n=2, hold=10, trail_pct=tr / 100))
    for tr in (8, 12, 20):
        B.append(_cfg(f"G03_h20_tr{tr}_t1", "G移动止损", top_n=1, hold=20, trail_pct=tr / 100))
    # ---- H. 排名退出 (自适应持有, 不设持有期上限) ----
    for er in (3, 5, 10, 20, 50):
        B.append(_cfg(f"H01_re{er}_t1", "H排名退出", top_n=1, hold=None,
                      exit_rank=er, min_hold=2))
    for er in (10, 20, 50, 100):
        B.append(_cfg(f"H02_re{er}_t2", "H排名退出", top_n=2, hold=None,
                      exit_rank=er, min_hold=2))
    for er in (20, 50):
        B.append(_cfg(f"H03_re{er}_sl8_t1", "H排名退出", top_n=1, hold=None,
                      exit_rank=er, min_hold=2, stop_loss=0.08))
        B.append(_cfg(f"H04_re{er}_sl8_t2", "H排名退出", top_n=2, hold=None,
                      exit_rank=er, min_hold=2, stop_loss=0.08))
    # ---- I. Top1 打分分位择时 (expanding, 空仓) ----
    B += [_cfg("I01_daily_t1_q0.3", "I择时", top_n=1, hold=1, threshold_q=0.3),
          _cfg("I02_daily_t1_q0.5", "I择时", top_n=1, hold=1, threshold_q=0.5),
          _cfg("I03_h5_sl8_t1_q0.3", "I择时", top_n=1, hold=5, stop_loss=0.08, threshold_q=0.3),
          _cfg("I04_h5_sl8_t2_q0.3", "I择时", top_n=2, hold=5, stop_loss=0.08, threshold_q=0.3)]
    # ---- J. 利润锁定 (ratchet): 浮盈达到 +rat_up 后保底 +rat_floor ----
    B += [_cfg("J01_h10_rat20_10_t1", "J利润锁定", top_n=1, hold=10, rat_up=0.20, rat_floor=0.10),
          _cfg("J02_h10_rat20_10_t2", "J利润锁定", top_n=2, hold=10, rat_up=0.20, rat_floor=0.10),
          _cfg("J03_h10_rat30_15_sl8_t1", "J利润锁定", top_n=1, hold=10, rat_up=0.30,
               rat_floor=0.15, stop_loss=0.08),
          _cfg("J04_h10_rat30_15_sl8_t2", "J利润锁定", top_n=2, hold=10, rat_up=0.30,
               rat_floor=0.15, stop_loss=0.08)]
    return B


def battery_B2():
    """细化网格: 在 B1 信号附近做系统扫描, 找稳健平台。"""
    B = []
    # ---- K. hold × stop_loss 全网格 (top_n 1/2) ----
    for tn in (1, 2):
        for h in (2, 3, 4, 5, 6, 7, 10, 15):
            for sl in (None, 3, 5, 8, 10):
                tag = f"K{tn}h{h}_sl{sl if sl else 'x'}"
                kw = dict(top_n=tn, hold=h)
                if sl:
                    kw["stop_loss"] = sl / 100
                B.append(_cfg(tag, "K网格", **kw))
    # ---- L. 止盈细化 (hold5 sl8, tp 网格) × (hold3 sl8) ----
    for tn in (1, 2):
        for h in (3, 5, 7):
            for tp in (15, 25, 40):
                B.append(_cfg(f"L{tn}_h{h}_sl8_tp{tp}", "L止盈细化", top_n=tn, hold=h,
                              stop_loss=0.08, take_profit=tp / 100))
    # ---- M. 移动止损细化 ----
    for tn in (1, 2):
        for h in (10, 20):
            for tr in (5, 8, 12, 15, 20):
                B.append(_cfg(f"M{tn}_h{h}_tr{tr}", "M移动细化", top_n=tn, hold=h,
                              trail_pct=tr / 100))
    # ---- N. 排名退出细化 ----
    for tn in (1, 2):
        for er in (2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500):
            B.append(_cfg(f"N{tn}_re{er}", "N排名细化", top_n=tn, hold=None,
                          exit_rank=er, min_hold=2))
    # ---- O. 择时细化 (hold5 sl8 上挂不同分位) ----
    for tn in (1, 2):
        for q in (0.2, 0.4, 0.6):
            B.append(_cfg(f"O{tn}_h5_sl8_q{q}", "O择时细化", top_n=tn, hold=5,
                          stop_loss=0.08, threshold_q=q))
    return B


def battery_sellmode():
    """三种卖出口径对照 (每 score 集): open / close_overlap / close_seq。"""
    B = []
    for sell_at in ("open", "close_overlap", "close_seq"):
        B.append(_cfg(f"SM_h5_sl8_{sell_at}", "卖出口径", top_n=1, hold=5,
                      stop_loss=0.08, sell_at=sell_at))
        B.append(_cfg(f"SM_daily_{sell_at}", "卖出口径", top_n=1, hold=1, sell_at=sell_at))
    return B


def battery_B3():
    """针对排名退出冠军族的组合电池: 止盈/止损/移动/持有上限/择时 与 re300 组合。"""
    B = []
    # ---- P. re300 + 固定止损 ----
    for tn in (1, 2):
        for sl in (5, 8, 10):
            B.append(_cfg(f"P{tn}_re300_sl{sl}", "P组合", top_n=tn, hold=None,
                          exit_rank=300, min_hold=2, stop_loss=sl / 100))
    # ---- Q. re300 + 固定止盈 (落袋) ----
    for tn in (1, 2):
        for tp in (20, 40, 60):
            B.append(_cfg(f"Q{tn}_re300_tp{tp}", "Q组合", top_n=tn, hold=None,
                          exit_rank=300, min_hold=2, take_profit=tp / 100))
    # ---- R. re300 + 移动止损 ----
    for tn in (1, 2):
        for tr in (10, 15, 20):
            B.append(_cfg(f"R{tn}_re300_tr{tr}", "R组合", top_n=tn, hold=None,
                          exit_rank=300, min_hold=2, trail_pct=tr / 100))
    # ---- S. re300 + 止损 + 止盈 ----
    for tn in (1, 2):
        B.append(_cfg(f"S{tn}_re300_sl8_tp40", "S组合", top_n=tn, hold=None,
                      exit_rank=300, min_hold=2, stop_loss=0.08, take_profit=0.40))
        B.append(_cfg(f"S{tn}_re300_sl8_tr15", "S组合", top_n=tn, hold=None,
                      exit_rank=300, min_hold=2, stop_loss=0.08, trail_pct=0.15))
    # ---- T. 更细的 exit_rank 与 min_hold ----
    for er in (150, 250, 400):
        B.append(_cfg(f"T2_re{er}", "T细化", top_n=2, hold=None, exit_rank=er, min_hold=2))
    for mh in (3, 5, 8):
        B.append(_cfg(f"T2_re300_mh{mh}", "T细化", top_n=2, hold=None,
                      exit_rank=300, min_hold=mh))
    # ---- U. re300 加持有期上限 (避免无限持有) ----
    for hcap in (10, 20, 30):
        B.append(_cfg(f"U2_re300_h{hcap}", "U持有上限", top_n=2, hold=hcap,
                      exit_rank=300, min_hold=2))
    # ---- V. re300 + 择时 ----
    for q in (0.2, 0.4):
        B.append(_cfg(f"V2_re300_q{q}", "V择时", top_n=2, hold=None,
                      exit_rank=300, min_hold=2, threshold_q=q))
    # ---- W. 持有期+移动止损 冠军族细化 (M2_h20_tr20 附近) ----
    for h in (10, 20, 30):
        for tr in (15, 20, 25):
            B.append(_cfg(f"W2_h{h}_tr{tr}", "W移动", top_n=2, hold=h, trail_pct=tr / 100))
    # ---- X. re300 口径的卖出口径对照 ----
    for sell_at in ("close_overlap", "close_seq"):
        B.append(_cfg(f"X_re300_{sell_at}", "X口径", top_n=2, hold=None,
                      exit_rank=300, min_hold=2, sell_at=sell_at))
    return B


BATTERIES = {"B1": battery_B1, "B2": battery_B2, "B3": battery_B3,
             "SELLMODE": battery_sellmode}


if __name__ == "__main__":
    for name, fn in BATTERIES.items():
        b = fn()
        print(name, len(b), "个配置; 组:", sorted(set(c["group"] for c in b)))
