#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py — V22 (长周期 GBDT 单元) 报告: 模型结果记录 (对齐 V20/V21 的 model_pic 惯例)

输出: Model/V22/model_pic/output.md + figure_01.png
内容:
  1) IC 层: RankIC/IR × {1d,3d,5d,10d,20d,trail} (Test 可交易池)
  2) 策略层: 5 协议真实净值矩阵 (与 V20_ensw2 对照)
  3) 图: 主推协议 (h20tr20_t2) 净值曲线 + 逐笔收益
用法: python3 report.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'Trading'))

from engine import load_calendar, load_market, run_backtest  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(HERE)) + "/"   # 仓库根 (trainingdata/)
MODEL_DIR = os.path.dirname(HERE) + "/"                  # Model/
MODEL_PIC = os.path.join(HERE, 'model_pic')
SEP = '=' * 80
SEP2 = '=' * 55

SETS = {
    "V22_10d": HERE + "/model_pred/v22_label_ret_10d.fea",
    "V22_20d": HERE + "/model_pred/v22_label_ret_20d.fea",
    "V20_ensw2": MODEL_DIR + "V20/model_pred/2026q3/score_ens_w2.fea",
}
LABEL_FILES = {"1d": "label_ret_1d", "3d": "label_ret_3d", "5d": "label_ret_5d",
               "10d": "label_ret_10d", "20d": "label_ret_20d",
               "trail": "label_trail20d"}
PROTOCOLS = {
    "D01_h5s8_t1": dict(top_n=1, hold=5, stop_loss=0.08),
    "hold20_t1": dict(top_n=1, hold=20),
    "hold20_t2": dict(top_n=2, hold=20),
    "h20tr20_t2": dict(top_n=2, hold=20, trail_pct=0.20),
    "S2_re300": dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
                     stop_loss=0.08, trail_pct=0.15),
}


def df_to_scores(df):
    return dict(
        scores={d: df.loc[d].dropna().to_dict() for d in df.index},
        ranked={d: df.loc[d].dropna().sort_values(ascending=False).index.tolist()
                for d in df.index},
        top1={d: float(df.loc[d].max()) for d in df.index},
        dates=sorted(df.index))


def main():
    buy = pd.read_feather(PROJECT + 'trainingdata/buyable_mask.fea').set_index('date')
    sets = {n: pd.read_feather(p).set_index('date') for n, p in SETS.items()}
    # IC
    ic_rows = []
    for n, score in sets.items():
        row = {}
        for h, fn in LABEL_FILES.items():
            lab = pd.read_feather(PROJECT + f'trainingdata/{fn}.fea').set_index('index')
            ics = []
            for d in score.index:
                if d not in lab.index:
                    continue
                s = score.loc[d]
                y = lab.loc[d].reindex(s.index)
                b = buy.loc[d].reindex(s.index) if d in buy.index \
                    else pd.Series(np.nan, index=s.index)
                b = pd.to_numeric(b, errors='coerce')
                tr = b.gt(0.5) & y.notna()
                if tr.sum() < 50:
                    continue
                ics.append(s[tr].rank().corr(y[tr].rank()))
            if ics:
                row[f'ic_{h}'] = np.mean(ics)
        ic_rows.append(row)
    ic_df = pd.DataFrame(ic_rows, index=list(sets.keys()))

    # 策略层
    mkt = load_market()
    tds, tdi = load_calendar()
    proto_rows = []
    main_eq, main_tr = None, None
    for n, score in sets.items():
        sc = df_to_scores(score)
        for pn, pc in PROTOCOLS.items():
            m, tr, eq = run_backtest(sc, pc, mkt, tds, tdi)
            if m is None:
                continue
            proto_rows.append(dict(set=n, proto=pn, cum_net=m['cum_net'],
                                   sharpe=m['sharpe'], maxdd=m['maxdd'],
                                   h1=m['h1_cum'], h2=m['h2_cum'],
                                   n=m['n_trades'], win=m['win_rate']))
            if n == "V22_20d" and pn == "h20tr20_t2":
                main_eq, main_tr = eq, tr
    pv = pd.DataFrame(proto_rows).pivot_table(
        index='proto', columns='set', values='cum_net').round(3)

    text = ['# V22 报告 — 长周期 GBDT 排序器单元 (10d/20d 标签)',
            '', '> 日期: 2026-09-05 | 单元: Model/V22 (train_gbdt.py + 本报告)',
            '> 结论: GBDT 组件 (1d/10d/20d) 在 Top 选择策略层无正交 alpha, 封存;',
            '> 但其长周期 IC 显著高于 NN 集成 — "IC ≠ Top 收益" 的又一实例。',
            '']
    text.append(SEP)
    text.append('  一、IC 层 (RankIC, 可交易池, Test 窗口)')
    text.append(SEP)
    text.append(ic_df.round(4).to_string())
    text.append('')
    text.append(SEP)
    text.append('  二、策略层真实净值矩阵 (Trading 引擎, 1 份资金, 含成本)')
    text.append(SEP)
    text.append(pv.to_string())
    text.append('')
    text.append(SEP2)
    text.append('  三、主推协议 (h20tr20_t2) V22_20d 明细')
    text.append(SEP2)
    if main_eq is not None:
        text.append(f'  净值: {main_eq["equity"].iloc[-1]/50000-1:+.2%}  |  '
                    f'MaxDD: {(main_eq["equity"]/main_eq["equity"].cummax()-1).min():+.2%}'
                    f'  |  交易 {len(main_tr)}')
    text.append(SEP2)
    report = '\n'.join(text)
    sys.stdout.write(report + '\n')

    os.makedirs(MODEL_PIC, exist_ok=True)
    with open(os.path.join(MODEL_PIC, 'output.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if main_eq is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            eqd = main_eq['equity'] / 50000 - 1
            axes[0].plot(range(len(eqd)), eqd.values, color='#d62728', lw=1.6)
            axes[0].axhline(0, color='gray', lw=0.8, ls='--')
            axes[0].set_title('V22_20d h20tr20_t2 True Equity', fontsize=13)
            axes[0].set_xlabel('Trading day'); axes[0].set_ylabel('Cum net')
            axes[0].grid(alpha=0.3)
            net = main_tr['net_pct'] / 100.0
            axes[1].bar(range(len(net)), net.values,
                        color=['#2ca02c' if v >= 0 else '#d62728' for v in net])
            axes[1].axhline(0, color='gray', lw=0.8, ls='--')
            axes[1].set_title(f'Per-Trade Net ({len(net)})', fontsize=13)
            axes[1].grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_PIC, 'figure_01.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        print(f'[report] 出图跳过: {e}')
    print(f'[report] → {MODEL_PIC}/output.md (+figure_01.png)')


if __name__ == '__main__':
    main()
