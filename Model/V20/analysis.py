#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py — V20 统一入口: 冠军系统最终结果 (排行榜 + 策略指令 + 可选回测)

你只需要执行本文件, 无需关心内部实现:
  python3 analysis.py                 # 最新因子日 Top10 排行榜 + Top1 策略指令 (自动处理推演刷新)
  python3 analysis.py --top 20        # 榜单长度
  python3 analysis.py --days 3        # 最近 3 个因子日的排行榜块 (V9 output.md 风格)
  python3 analysis.py --date 20260902 # 指定某因子日的排行榜
  python3 analysis.py --backtest      # 冠军协议全窗口回测 (hold5+stop8%+收盘卖, 含成本, 附分半/IC)
  python3 analysis.py --update        # 先强制重跑 V11/V13 增量推演再出结果
  python3 analysis.py --no-refresh    # 跳过推演刷新检查 (直接用现有 heads)
  python3 analysis.py --no-md         # 不写 model_pic/output.md 镜像

说明: 当因子数据更新出新的交易日时, analysis.py 会自动在 V11/V13 目录运行其
analysis.py (增量重推最近10日, 需 GPU, 约数分钟), 然后组装 ens_w2 出榜。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import model  # noqa: E402


def _official_parity(score, date):
    """与 V11 官方 score_ens_w2.fea 做同口径一致性校验 (仅提示, 供信心参考)。"""
    if not os.path.exists(model.V11_OFFICIAL_SCORE):
        return None
    try:
        omax = model._read_date_max(model.V11_OFFICIAL_SCORE)
        if date > omax:      # 官方文件尚未覆盖该日, 跳过
            return None
        official = __import__('pandas').read_feather(
            model.V11_OFFICIAL_SCORE).set_index('date')
        if date not in official.index or date not in score.index:
            return None
        import numpy as np
        o = official.loc[date].reindex(score.loc[date].index)
        diff = (score.loc[date] - o).abs().max()
        return float(diff) if np.isfinite(diff) else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description='V20 冠军系统最终结果入口')
    ap.add_argument('--top', type=int, default=10, help='榜单长度 (默认 10)')
    ap.add_argument('--date', default=None, help='指定因子日 YYYYMMDD 出榜')
    ap.add_argument('--days', type=int, default=1, help='最近 N 个因子日出榜 (默认 1)')
    ap.add_argument('--backtest', action='store_true', help='追加冠军协议全窗口回测')
    ap.add_argument('--no-split', action='store_true', help='回测不做 H1/H2 分半')
    ap.add_argument('--update', action='store_true', help='强制重跑 V11/V13 增量推演')
    ap.add_argument('--no-refresh', action='store_true', help='跳过推演刷新检查')
    ap.add_argument('--no-md', action='store_true', help='不写 model_pic/output.md')
    args = ap.parse_args()

    # ── 1) 确保 heads 覆盖最新因子日 ──
    if not args.no_refresh:
        refreshed, msg = model.ensure_heads_fresh(
            force=args.update, log_dir=os.path.join(HERE, 'logs'))
        if refreshed:
            print(f'[V20] {msg}')
    else:
        print('[V20] 跳过推演刷新检查 (--no-refresh)')

    # ── 2) 组装冠军打分 ──
    score = model.score_ens_w2()
    print(f'[V20] ens_w2 组装完成: {score.shape[0]} 天 × {score.shape[1]} 股票 '
          f'({score.index.min()} ~ {score.index.max()})', flush=True)

    # ── 3) 选日期出榜 ──
    dates = list(score.index)
    if args.date:
        if args.date not in score.index:
            ap.error(f'--date {args.date} 不在打分范围 ({dates[0]}~{dates[-1]})')
        show_dates = [args.date]
    else:
        show_dates = dates[-max(1, min(args.days, len(dates))):]

    tds = model.load_trading_dates()
    name_map = model.load_name_map()
    blocks = []
    for d in show_dates:
        blocks.append(model.rank_block(d, score, topn=args.top, tds=tds,
                                       name_map=name_map))
        # 一致性校验 (与 V11 官方打分文件)
        if not args.date and d == show_dates[-1]:
            md = _official_parity(score, d)
            if md is not None:
                blocks.append(f'  [一致性] 与 V11 官方 score_ens_w2 同口径校验 '
                              f'max|Δ| = {md:.2e} (≈0 即组装公式一致)')
                blocks.append('')
    text = '\n'.join(blocks).rstrip() + '\n'

    # ── 4) 最新因子日策略指令 ──
    text += model.decision_block(show_dates[-1], score, tds=tds, name_map=name_map,
                                 holdings_path=os.path.join(model.V11_DIR,
                                                            'holdings.json')) + '\n'

    # ── 5) 输出 ──
    sys.stdout.write(text)
    sys.stdout.flush()
    if not args.no_md:
        md_dir = os.path.join(HERE, 'model_pic')
        os.makedirs(md_dir, exist_ok=True)
        with open(os.path.join(md_dir, 'output.md'), 'w', encoding='utf-8') as f:
            f.write('# V20 冠军系统每日结果 (ens_w2 + hold5s8-收盘卖)\n\n')
            f.write(text)

    # ── 6) 可选回测 ──
    if args.backtest:
        print()
        model.champion_backtest(score, split=not args.no_split, verbose=True)


if __name__ == '__main__':
    main()
