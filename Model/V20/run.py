#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — V20 启动入口: 刷新增量推演 → 组装冠军打分 → 落盘打分缓存

用法:
  python run.py                # 若因子数据有新交易日 → 自动刷新 V11/V13 推演, 再组装打分
  python run.py --update       # 强制重跑 V11/V13 增量推演后组装 (即使 heads 未落后)
  python run.py --no-refresh   # 跳过推演检查, 直接用现有 heads 组装

输出: model_pred/score_ens_w2.fea (全窗口打分缓存) + 控制台摘要。
日常出榜请直接运行 analysis.py (本入口只负责"刷新+组装")。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='V20 启动: 刷新推演 + 组装冠军打分')
    ap.add_argument('--update', action='store_true',
                    help='强制重跑 V11/V13 增量推演 (即使 heads 已最新)')
    ap.add_argument('--no-refresh', action='store_true',
                    help='跳过推演检查, 直接用现有 heads 组装')
    args = ap.parse_args()

    if not args.no_refresh:
        refreshed, msg = model.ensure_heads_fresh(force=args.update,
                                                  log_dir=os.path.join(HERE, 'logs'))
        print(f'[V20] {msg}')
    else:
        print('[V20] 跳过推演检查 (--no-refresh)')

    score = model.score_ens_w2()
    out_dir = os.path.join(HERE, 'model_pred')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'score_ens_w2.fea')
    score.reset_index().to_feather(out_path)

    print(f'[V20] 组装完成: {score.shape[0]} 天 × {score.shape[1]} 股票, '
          f'{score.index.min()} ~ {score.index.max()}')
    print(f'[V20] 打分缓存已落盘: {out_path}')
    latest = score.index.max()
    top1 = score.loc[latest].sort_values(ascending=False).index[0]
    names = model.load_name_map()
    print(f'[V20] 最新因子日 {latest}: Top1 = {top1} {names.get(top1, "?")} '
          f'(打分 {score.loc[latest, top1]:+.4f})')
    print('[V20] 完成。查看最终排行榜/策略指令请运行: python3 analysis.py')


if __name__ == '__main__':
    main()
