#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py — best_new_seed1 推演/回测/最终结果入口 (对 best 自训的 16 折 checkpoint 工作)

使命: 用 model.py 定义的结构 + run.py/train.sh 训练出的 16 个 checkpoint,
全窗口逐日推演各头打分 → 组装冠军 ens_w2 → 输出**V9 风格完整报告**:
  1) Test 集打分 (RankIC/IR/top_return)
  2) 最新模型打分 Top 推荐 (最近 N 个交易日排行榜块, 默认 10, 增量滑动到最新因子日)
  3) Trade Log — 冠军协议成交明细 (hold5s8-收盘卖, 含成本; 含逐笔净收益 + 累加收益,
     随交易日推进增量更新, 末尾为当前待定持仓行)
  4) Summary / 分半 / Pending 说明
  5) 冠军策略指令 (Top1 行动)  +  model_pic/figure_01.png (累计收益+日收益图, 同 V9 风格)

用法:
  python3 analysis.py                  # 出完整 V9 风格报告 (输出到 stdout 与 model_pic/output.md)
  python3 analysis.py --days 3         # 推荐块数量 (默认 10)
  python3 analysis.py --date 20260902  # 只看指定因子日
  python3 analysis.py --top 5          # 每块榜单长度 (默认 10)
  python3 analysis.py --update         # 强制重推最近 10 个交易日
  python3 analysis.py --skip-infer     # 跳过推演 (直接用现有 heads)
  python3 analysis.py --no-md          # 不写 model_pic/output.md (也不出图)
"""
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import (PROJECT_ROOT, root_path, SEASON, MODEL_PRED, HEADS_A, HEADS_C,  # noqa: E402
                   CACHE_SCORE, FAC_PATH, TEST_START, TEST_END, HOLD_DAYS, STOP_LOSS,
                   fold_spec, N_FOLD_PER_FAMILY, TOTAL_FOLDS, PredictModel,
                   load_trading_dates, load_name_map, score_ens_w2,
                   champion_backtest, rank_block, decision_block,
                   latest_reportable_date, heads_last_date, next_td)

# ============================================================
# 配置
# ============================================================
MODEL_TRAIN = rf'{root_path}/model_train/{SEASON}'
MODEL_TEST = rf'{root_path}/model_test'
FEATURE_MAP = os.path.join(MODEL_TEST, 'feature_map.fea')
TAIL_DAYS = 10          # --update 时重推的天数
DEFAULT_DAYS = 10       # 推荐块滑动窗口 (同 V9: 最近 10 个交易日)
SEP = '=' * 100
SEP2 = '=' * 55
BAR = '─' * 60

# ============================================================
# 推演辅助 (与 V11/analysis.py 同口径)
# ============================================================
_VAL_RANKIC_RE = re.compile(r'val_rankic=(-?\d+\.\d+)')
_VAL_RANKIC_AVG_RE = re.compile(r'val_rankic_avg=(-?\d+\.\d+)')
_VAL_IC_RE = re.compile(r'val_icmean=(-?\d+\.\d+)')
_VAL_WEI_RE = re.compile(r'val_wei=(-?\d+\.\d+)')
_VAL_COMBO_RE = re.compile(r'val_combo=(-?\d+\.\d+)')  # best: 策略混合标尺优先


def load_feature_map():
    if not os.path.exists(FEATURE_MAP):
        raise FileNotFoundError(f'feature_map.fea 不存在 ({FEATURE_MAP}); 先训练 (bash train.sh)')
    with open(FEATURE_MAP, encoding='utf-8') as f:
        raw = f.read().replace('\\n', '\n')
    order = []
    for line in raw.split('\n'):
        if '=' in line:
            name, idx = line.rsplit('=', 1)
            order.append((int(idx.strip()), name.strip()))
    order.sort(key=lambda x: x[0])
    return [n for _, n in order]


def normed_infer(data, factor_list):
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    valid_mask = data[factor_list].notna().sum(axis=1) >= valid_threshold
    data = data.loc[valid_mask].copy()
    codes = data['Code'].values
    X = data[factor_list].rank(axis=0)
    X = (X - X.mean()) / X.std()
    X = X.fillna(0)
    x = np.nan_to_num(X.to_numpy(dtype=np.float32, copy=False),
                      nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(x), codes


def load_model(checkpoint_path, input_dim):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = {k.removeprefix('model.'): v for k, v in ck['state_dict'].items()
             if k.startswith('model.')}
    m = PredictModel(input_dim=input_dim)
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing:
        print(f'[warn] missing keys: {list(missing)[:5]}')
    m.eval()
    return m


def find_best_checkpoint(fold_dir):
    files = glob.glob(os.path.join(fold_dir, '**', '*.ckpt'), recursive=True)
    if not files:
        raise FileNotFoundError(f'No ckpt under {fold_dir}')
    best, bestv = None, -float('inf')
    for ck in files:
        m = (_VAL_COMBO_RE.search(os.path.basename(ck))   # best: 策略混合标尺优先
             or _VAL_RANKIC_AVG_RE.search(os.path.basename(ck))
             or _VAL_RANKIC_RE.search(os.path.basename(ck))
             or _VAL_IC_RE.search(os.path.basename(ck))
             or _VAL_WEI_RE.search(os.path.basename(ck)))
        if m is None:
            continue
        v = float(m.group(1))
        if v > bestv:
            bestv, best = v, ck
    if best is None:
        raise RuntimeError(f'无法从 checkpoint 名解析 val 指标: {fold_dir}')
    return best, bestv


def predict_heads(model, data, factor_list, device):
    X, codes = normed_infer(data, factor_list)
    with torch.no_grad():
        out = model(X.to(device))
    heads = {}
    if isinstance(out, tuple):
        for name, t in zip(('mixed', 'r1', 'r5', 'r3', 'top'), out[:5]):
            if t is not None:
                heads[name] = pd.DataFrame(t.detach().cpu().numpy(),
                                           index=codes, columns=['value'])
    return heads


# ============================================================
# 推演 16 折 → 各头逐折 z 宽表 (heads_{a,c})
# ============================================================
def ensure_heads(target_date, force_update=False):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as pf

    a_last, c_last = heads_last_date(HEADS_A), heads_last_date(HEADS_C)
    if a_last is None or c_last is None:
        print(f'[analysis] heads 缺失 (a={a_last}, c={c_last}) → 首次全窗口推演')
    cur = min(a_last, c_last) if (a_last and c_last) else None
    if cur is not None and cur >= target_date and not force_update:
        print(f'[analysis] heads 已覆盖至 {cur} (目标 {target_date}), 无需推演')
        return

    factor_list = load_feature_map()
    print(f'[analysis] feature_map: {len(factor_list)} 因子')

    tds = set(load_trading_dates())
    col = pf.read_table(FAC_PATH, columns=['date']).column('date')
    all_factor = sorted(set(pc.unique(col).to_pandas().astype(str)))
    full_dates = [d for d in all_factor if d >= TEST_START and d in tds]
    if force_update:
        run_dates = full_dates[-TAIL_DAYS:]
    elif cur is None:
        run_dates = full_dates
    else:
        run_dates = [d for d in full_dates if d > cur]
    if not run_dates:
        print('[analysis] 无新日期需要推演')
        return
    mode = f'重推最近 {TAIL_DAYS} 日' if force_update else '增量'
    print(f'[analysis] 待推演 {len(run_dates)} 日 ({mode}): '
          f'{run_dates[0]} ~ {run_dates[-1]}', flush=True)

    all_cols = pf.read_table(FAC_PATH, columns=[]).column_names
    available = [f for f in factor_list if f in all_cols]
    missing = [f for f in factor_list if f not in all_cols]
    cols = ['date', 'Code'] + available
    table = pf.read_table(FAC_PATH, columns=cols)
    mask = pc.is_in(table.column('date'), pa.array(run_dates))
    data = table.filter(mask).to_pandas().set_index('date').sort_index()
    del table
    for f in missing:
        data[f] = 0.0
    print(f'[analysis] 因子行 {len(data)} (缺失列补 0: {len(missing)})')

    fold_dirs, missing_folds = {}, []
    for g in range(1, TOTAL_FOLDS + 1):
        d = os.path.join(MODEL_TRAIN, f'fold{g}')
        if not os.path.exists(d):
            missing_folds.append(g)
        else:
            fold_dirs[g] = d
    if missing_folds:
        raise RuntimeError('缺少训练产物 fold ' + ','.join(map(str, missing_folds))
                           + f' (共 {TOTAL_FOLDS} 折) — 先 bash train.sh 完成 16 折训练')

    ckpt0, _ = find_best_checkpoint(fold_dirs[1])
    meta = torch.load(ckpt0, map_location='cpu', weights_only=False)
    dim0 = None
    for key, t in meta['state_dict'].items():
        if 'weight' in key and len(t.shape) == 2:
            dim0 = t.shape[1]
            break
    del meta
    flist = factor_list[:]
    if dim0 is not None and dim0 != len(flist):
        if dim0 > len(flist):
            for i in range(dim0 - len(flist)):
                flist.append(f'_pad_{i}')
                data[f'_pad_{i}'] = 0.0
        else:
            flist = flist[:dim0]
        print(f'[analysis] 输入维度校准: feature_map {len(factor_list)} → {len(flist)}')
    factor_cols = [c for c in flist if c in data.columns]
    print(f'[analysis] 实际使用特征列 {len(factor_cols)}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    for g in range(1, TOTAL_FOLDS + 1):
        fam, k = fold_spec(g)
        ck, v = find_best_checkpoint(fold_dirs[g])
        m = load_model(ck, len(factor_cols))
        m.to(device)
        models[g] = m
        print(f'[analysis]   fold{g} (族{fam}/k{k}): {os.path.basename(ck)} '
              f'(val={v:.4f})', flush=True)

    head_rows = {(fam, k, h): [] for fam in ('a', 'c')
                 for k in range(1, N_FOLD_PER_FAMILY + 1)
                 for h in ('r1', 'r3', 'r5', 'top')}
    total = len(run_dates) * TOTAL_FOLDS
    done = 0
    for d in run_dates:
        if d not in data.index:
            continue
        day = data.loc[d]
        if isinstance(day, pd.Series):
            day = day.to_frame().T
        for g in range(1, TOTAL_FOLDS + 1):
            fam, k = fold_spec(g)
            heads = predict_heads(models[g], day, factor_cols, device)
            for h in ('r1', 'r3', 'r5', 'top'):
                if h not in heads:
                    continue
                z = heads[h]['value']
                z = (z - z.mean()) / z.std()
                head_rows[(fam, k, h)].append(pd.Series(z.values,
                                                        index=heads[h].index,
                                                        name=d))
            done += 1
            if done % (TOTAL_FOLDS * 5) == 0 or done == total:
                print(f'[analysis] 推演进度 {done}/{total} (最新 {d})', flush=True)
    print('[analysis] 推演完成, 合并落盘 heads ...', flush=True)

    for fam, dir_ in (('a', HEADS_A), ('c', HEADS_C)):
        os.makedirs(dir_, exist_ok=True)
        for k in range(1, N_FOLD_PER_FAMILY + 1):
            for h in ('r1', 'r3', 'r5', 'top'):
                rows = head_rows[(fam, k, h)]
                if not rows:
                    continue
                new = pd.DataFrame(rows)
                new.index.name = 'date'
                path = os.path.join(dir_, f'{h}_f{k}.fea')
                if os.path.exists(path):
                    old = pd.read_feather(path).set_index('date')
                    new = pd.concat([old, new], axis=0)
                    new = new[~new.index.duplicated(keep='last')].sort_index()
                new.reset_index().to_feather(path)
    print(f'[analysis] heads 落盘完成: {HEADS_A} / {HEADS_C}')


# ============================================================
# V9 风格报告构建
# ============================================================
def fmt_date(d):
    s = str(d)
    return f'{int(s[:4])}年{int(s[4:6])}月{int(s[6:8])}日'


def _pct(v, signed=True):
    return f'{v * 100:+.2f}%' if signed else f'{v * 100:.2f}%'


def ic_block(ic):
    lines = ['---Test 集合打分---', f'  打分区间: Test 集合 {TEST_START} ~ {TEST_END} '
             f'(严格样本外, 与训练/验证无重叠)', '']
    for k, label in (('RankIC', 'RankIC'), ('RankICIR', 'RankICIR'),
                     ('top_return', 'top_return')):
        lines.append(f'  {k:<10} {ic.get(k, float("nan")):+.4f}')
    lines += ['[提示] 回测/交易日志章节使用同一区间的实际价格收益 (daily_adj), '
              '与训练 target 解耦', '']
    return '\n'.join(lines)


def champion_log(score, names, verbose=True):
    """冠军协议成交明细 (hold5s8-收盘卖, 含成本, 窗口 = 打分起点 ~ 最新因子日)。

    返回 (log_text, stats_dict, trades_df)。trades_df 为已实现成交; 若窗口末端存在
    未到期持仓 (卖出日无价格) 则 log 末尾追加一行待定 (sell '-'), 累加收益冻结。
    """
    mod_engine = __import__('model', fromlist=['x'])._close_sell_backtest_module()
    rb = mod_engine['run_backtest']
    open_map, close_map, prev_close_map, amount_map, nm = mod_engine['load_prices']()
    tds = mod_engine['load_calendar']()
    F = dict(use_cost=True, exclude_st=True, exclude_limit_up=True,
             buy_gap_limit=0.095, hold=HOLD_DAYS, stop_loss=STOP_LOSS)
    w_start, w_end = str(score.index.min()), str(score.index.max())
    m, trades, _ = rb(score, open_map, close_map, prev_close_map, nm,
                      amount_map=amount_map, tds=tds,
                      window_start=w_start, window_end=w_end, **F)

    # 逐笔: 顺序 = 开仓时间; 累加收益 = 复利 (与冠军净累计口径一致)
    tr = trades.sort_values('buy_dt').reset_index(drop=True)
    cum = 0.0
    rows = []
    for _, r in tr.iterrows():
        cum = (1.0 + cum) * (1.0 + r['net_pct'] / 100.0) - 1.0
        rows.append((str(r['factor_dt']), str(r['buy_dt']), r['code'],
                     r.get('name') or nm.get(r['code'], ''),
                     float(r['score']), float(r['buy_prc']),
                     str(r['sell_dt']), float(r['sell_prc']),
                     r['net_pct'] / 100.0, cum))
    realized = len(rows)
    last_sell_dt = str(tr['sell_dt'].max()) if realized else w_start

    # 待定持仓: 最后一个可开仓(买入价可得)且卖出日价格尚未出现的因子日
    pending = None
    factor_dates = [d for d in score.index if w_start <= str(d) <= w_end]
    for d in factor_dates:
        buy_dt = next_td(d, 1, tds)
        if buy_dt is None or buy_dt <= last_sell_dt:
            continue
        code = score.loc[d].sort_values(ascending=False).index[0]
        bp = open_map.get((buy_dt, code))
        if bp is None:
            continue
        sell_dt = next_td(buy_dt, HOLD_DAYS, tds)
        sp = close_map.get((sell_dt, code)) if sell_dt else None
        if sp is not None:
            continue            # 已能卖出 → 引擎应已实现, 非待定
        pending = (str(d), buy_dt, code, names.get(code, '?'),
                   float(score.loc[d, code]), float(bp), sell_dt)
        break                    # 单持仓: 仅最近一笔
    n_pending = 1 if pending else 0
    eq_series = pd.Series([r[9] for r in rows], dtype=float)  # 复利权益轨迹

    L = []
    L.append(SEP)
    L.append(f'  Trade Log — ens_w2 冠军协议 hold{HOLD_DAYS}s{int(STOP_LOSS*100)} 收盘卖 '
             f'| Top 1 | 含成本 | {w_start} ~ {w_end}')
    L.append(SEP)
    L.append(' FactorDt   BuyDt     Code     Name      Score   BuyPrc   SellDt  '
             'SellPrc   Net%   累加收益(净,复利)')
    L.append('-' * 100)
    for fd, bd, code, name, sc, bp, sd, sp, net, c in rows:
        L.append(f' {fd:<10}{bd:<10}{code:<7}{name:<9}{sc:+8.2f}{bp:9.2f}  '
                 f'{sd:<9}{sp:9.2f}  {net:+7.2%}  {c:+9.2%}')
    if pending:
        fd, bd, code, name, sc, bp, sd = pending
        L.append(f' {fd:<10}{bd:<10}{code:<7}{name:<9}{sc:+8.2f}{bp:9.2f}  '
                 f'{sd:<9}{"-":>9}  {"-":>7}  {cum:+9.2%}')
    L.append(SEP)
    win = int(np.sum(tr['net_pct'] > 0)) if realized else 0
    dr_mean = float(np.mean(tr['net_pct'] / 100.0)) if realized else 0.0
    eq_maxdd = 0.0
    if realized:
        peak = np.maximum.accumulate(eq_series.values)
        eq_maxdd = float((eq_series.values - peak).min())
    L.append(f'Realized: {realized}  |  Pending: {n_pending}')
    L.append(f'Cumulative(净,复利): {cum:+.4f} ({cum*100:+.2f}%)  |  '
             f'Win rate: {win}/{realized} ({win/max(1,realized)*100:.1f}%)  |  '
             f'Mean(笔均): {dr_mean:+.5f}  |  MaxDD: 见下方 Summary (官方口径)')
    stats = dict(realized=realized, pending=n_pending, cum=cum,
                 win=win, maxdd=eq_maxdd, mean=dr_mean)
    return '\n'.join(L), stats, tr


def figure_01(score, trades, out_path, title_window):
    """V9 风格图: 左 = 累计净收益曲线, 右 = 逐笔净收益直方 (model_pic/figure_01.png)。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[analysis] matplotlib 不可用, 跳过出图: {e}')
        return
    tr = trades.sort_values('buy_dt').reset_index(drop=True)
    if len(tr) == 0:
        return
    net = tr['net_pct'] / 100.0
    cum = ((1.0 + net).cumprod() - 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(range(len(cum)), cum.values, color='#d62728', lw=1.6)
    axes[0].axhline(0, color='gray', lw=0.8, ls='--')
    axes[0].set_title(f'Top 1 Cumulative Net Return (ens_w2 hold{HOLD_DAYS}) | '
                      f'{title_window}', fontsize=13)
    axes[0].set_xlabel('Trade #')
    axes[0].set_ylabel('Cumulative Net Return')
    axes[0].grid(alpha=0.3)
    axes[1].bar(range(len(net)), net.values, color=['#2ca02c' if v >= 0 else '#d62728'
                                                    for v in net])
    axes[1].axhline(0, color='gray', lw=0.8, ls='--')
    axes[1].set_title(f'Per-Trade Net Return ({len(net)} realized)', fontsize=13)
    axes[1].set_xlabel('Trade #')
    axes[1].set_ylabel('Net Return')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[analysis] 图已保存: {out_path}')


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='best_new_seed1 推演/回测/最终结果入口 (V9 风格报告)')
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--date', default=None, help='只展示指定因子日 YYYYMMDD')
    ap.add_argument('--days', type=int, default=DEFAULT_DAYS, help='推荐块数量 (默认 10)')
    ap.add_argument('--no-split', action='store_true', help='Summary 不做 H1/H2 分半')
    ap.add_argument('--update', action='store_true',
                    help=f'强制重推最近 {TAIL_DAYS} 个交易日')
    ap.add_argument('--skip-infer', action='store_true', help='跳过推演 (直接用 heads)')
    ap.add_argument('--no-md', action='store_true', help='不写 model_pic/output.md')
    args = ap.parse_args()

    # 1) 推演 (按需 / 强制)
    target = latest_reportable_date()
    if not args.skip_infer:
        ensure_heads(target, force_update=args.update)
    else:
        print('[analysis] 跳过推演 (--skip-infer)')

    # 2) 组装冠军打分
    try:
        score = score_ens_w2()
    except FileNotFoundError as e:
        print(f'[analysis] 无法组装打分: {e}')
        print('  请先完成 16 折训练 (bash train.sh) 并运行本文件推演 heads。')
        sys.exit(1)
    print(f'[analysis] ens_w2 组装完成: {score.shape[0]} 天 × {score.shape[1]} 股票 '
          f'({score.index.min()} ~ {score.index.max()})', flush=True)
    try:
        os.makedirs(MODEL_PRED, exist_ok=True)
        score.reset_index().to_feather(CACHE_SCORE)
        print(f'[analysis] 打分缓存: {CACHE_SCORE}')
    except OSError as e:
        print(f'[analysis] 打分缓存写入失败 (跳过, 不影响报告): {e}')

    # 3) 报告构建
    dates = list(score.index)
    tds, name_map = load_trading_dates(), load_name_map()
    latest = dates[-1]

    # 3.1 官方 Test 集指标 (含 IC; Summary 用, 与 FINAL_REPORT 口径一致)
    bt = champion_backtest(score, split=True, verbose=False)
    ic_res = bt['ic']
    text = []
    text.append('# best_new_seed1 冠军系统每日报告 (ens_w2 + hold5s8-收盘卖, 含成本)')
    text.append('')
    text.append(ic_block(ic_res))

    # 3.2 推荐块 (最近 N 日, 升序)
    text.append(SEP)
    text.append('                    最新模型打分 Top 推荐')
    text.append(SEP)
    text.append(f'  报告截至因子日期：{fmt_date(latest)} 收盘, '
                f'以下展示最近 {args.days} 个交易日')
    text.append('')
    if args.date:
        if args.date not in score.index:
            ap.error(f'--date {args.date} 不在打分范围 ({dates[0]}~{dates[-1]})')
        show = [args.date]
    else:
        show = dates[-max(1, min(args.days, len(dates))):]
    for d in show:
        text.append(rank_block(d, score, topn=args.top, tds=tds, name_map=name_map))
        text.append('')

    # 3.3 Trade Log (冠军协议, 窗口 = 打分起点 ~ 最新因子日, 增量更新) + Summary
    log_text, stats, trades = champion_log(score, name_map)
    text.append(log_text)
    text.append('')
    s = stats
    f_ = bt['full']
    text.append(SEP2)
    text.append(f'  Recent Summary | ens_w2 冠军协议 hold{HOLD_DAYS}s'
                f'{int(STOP_LOSS*100)} 收盘卖 | 含成本 | 官方 Test 集窗口')
    text.append(SEP2)
    text.append(f'  Test 窗口  : {TEST_START} ~ {TEST_END}  (243 交易日, 严格样本外)')
    text.append(f'  Net Cum    : {f_["净累计%"]:+.2f}%  |  净年化 {f_["净年化%"]:+.2f}%  |  '
                f'Sharpe {f_["Sharpe"]:.3f}')
    text.append(f'  MaxDD      : {f_["MaxDD%"]:+.2f}%  |  胜率 {f_["胜率%"]:.1f}%  '
                f'({s["win"]}/{s["realized"]} 已实现)  |  成本拖累 {f_["成本拖累%"]:.2f}%')
    text.append(f'  累计(复利) : 至 {score.index.max()} 因子日 = {s["cum"]*100:+.2f}% '
                f'(含 {s["pending"]} 笔待定未计入)')
    if not args.no_split:
        h1, h2 = bt['h1']['净累计%'], bt['h2']['净累计%']
        text.append(f'  分半 H1/H2 : {TEST_START}~20260227 {h1:+.2f}%  |  '
                    f'20260302~{TEST_END} {h2:+.2f}%')
    text.append(SEP2)
    text.append('')
    if s['pending']:
        text.append(f'[Pending] {s["pending"]} 笔持仓未到期 (卖出日价格尚未出现): 见 Trade Log 末行')
        text.append('')
    text.append(decision_block(latest, score, tds=tds, name_map=name_map,
                               holdings_path=os.path.join(PROJECT_ROOT,
                                                          'Model/V11/holdings.json')))
    text.append('')
    report = '\n'.join(text)

    sys.stdout.write(report)
    sys.stdout.flush()

    if not args.no_md:
        md_dir = os.environ.get('best_new_seed1_MD_DIR') or os.path.join(HERE, 'model_pic')
        try:
            os.makedirs(md_dir, exist_ok=True)
            with open(os.path.join(md_dir, 'output.md'), 'w', encoding='utf-8') as f:
                f.write(report)
            figure_01(score, trades, os.path.join(md_dir, 'figure_01.png'),
                      f'{score.index.min()} ~ {score.index.max()}')
            print(f'\n[analysis] 报告已写入: {md_dir}/output.md (+figure_01.png)')
        except OSError as e:
            print(f'\n[analysis] 报告写入 {md_dir} 失败 (跳过): {e}')
            print('  若需更新 V29/model_pic/*, 请以文件拥有者运行本脚本, 或设置 '
                  'best_new_seed1_MD_DIR 指向可写目录')


if __name__ == '__main__':
    main()


# ======================================================================
# 冻结双腿策略评估 + 当日操作 (FIXED_STRATEGY.md v1.0) — 本单元自测入口
# 运行本文件即: 推演→score→冻结策略回测→当日纸面操作 (打分源 = 本单元)
# ======================================================================
def _frozen_test_block():
    import os as _os
    import sys as _sys
    import pandas as _pd
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_HERE)
    _SCORE = _os.path.join(_HERE, "model_pred", "2026q3",
                           "score_ens_w2.fea")
    print("\n" + "=" * 80)
    print(" 冻结双腿策略 (FIXED_STRATEGY.md v1.0): 本单元打分 → 回测 → 操作")
    print("=" * 80)
    if not _os.path.exists(_SCORE):
        print(" ⚠ score_ens_w2.fea 不存在 (推演未完成?), 跳过冻结评估")
        return
    try:
        _sys.path.insert(0, _os.path.join(_ROOT, "Trading"))
        from engine import load_calendar, load_market, run_backtest  # noqa
        import daily_frozen as _df  # noqa
        _mkt = load_market()
        _tds, _tdi = load_calendar()
        _sc = _pd.read_feather(_SCORE).set_index("date").sort_index()
        _scores = dict(
            scores={_d: _sc.loc[_d].dropna().to_dict() for _d in _sc.index},
            ranked={_d: _sc.loc[_d].dropna().sort_values(ascending=False)
                    .index.tolist() for _d in _sc.index},
            top1={_d: float(_sc.loc[_d].max()) for _d in _sc.index},
            dates=sorted(_sc.index))
        _mS, _trS, _eqS = run_backtest(
            _scores, dict(top_n=2, hold=20, exit_rank=300, min_hold=2,
                          trail_pct=0.15), _mkt, _tds, _tdi)
        _mF, _trF, _eqF = run_backtest(
            _scores, dict(top_n=1, hold=5, stop_loss=0.08),
            _mkt, _tds, _tdi)
        print(" --- Test 窗 (20250901~20260901) 回测 (引擎口径, 含成本) ---")
        print(f"  慢腿 u_h20_re300_tr15: cum={_mS['cum_net']:+.1%} "
              f"sharpe={_mS['sharpe']:.2f} maxdd={_mS['maxdd']:.1%} "
              f"H1={_mS['h1_cum']:+.1%} H2={_mS['h2_cum']:+.1%} "
              f"(n={_mS['n_trades']})")
        print(f"  快腿 D01: cum={_mF['cum_net']:+.1%} "
              f"sharpe={_mF['sharpe']:.2f} maxdd={_mF['maxdd']:.1%} "
              f"H1={_mF['h1_cum']:+.1%} H2={_mF['h2_cum']:+.1%} "
              f"(n={_mF['n_trades']})")
        _eS = _eqS.set_index("date")["equity"]
        _eF = _eqF.set_index("date")["equity"]
        _j = _eS.index.intersection(_eF.index)
        _cb = (0.5 * _eS.loc[_j] + 0.5 * _eF.loc[_j]).to_frame()
        _cb.columns = ["equity"]
        _cb["ret"] = _cb["equity"].pct_change()
        _r = _cb["ret"].dropna()
        _cum = float((1 + _r).prod() - 1)
        _sh = float(_r.mean() / _r.std() * (252 ** 0.5))
        _pk = _cb["equity"].cummax()
        _dd = float((_cb["equity"] / _pk - 1).min())
        print(f"  双腿合计 (10W=慢5W+快5W): cum={_cum:+.1%} "
              f"sharpe={_sh:.2f} maxdd={_dd:.1%}")
        print(" --- 当日纸面操作 (状态文件与 V11 共用 holdings_frozen.json) ---")
        _df.run_daily(_SCORE)
    except Exception as _e:
        import traceback as _tb
        print(f" ⚠ 冻结评估/操作块失败: {_e}")
        _tb.print_exc()


_frozen_test_block()
