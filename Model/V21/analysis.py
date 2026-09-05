#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py — V21 推演/权重搜索/回测/最终结果入口 (对 V21 自训的 16 折 checkpoint 工作)

使命 (模型×策略耦合单元):
  1) 全窗口逐日推演 16 折各头 (r5/r10/r20/top) → heads_{a,c} 逐折 z 宽表
  2) 权重搜索: 家族内权重 × 跨族权重网格, 在目标协议 h20tr20_t2 (Top2 + 持有≤20日
     + 移动止损20%, 开盘换仓, 真实净值含再投资) 上回测, 选出 ens21 权重
     (按 cum_net 排序, H1/H2/T1-T3 全正优先; 结果落盘 weight_search.csv)
  3) 出 V9 风格完整报告: 长周期 IC + 排行榜 + Trade Log + Summary/分半 + 策略指令
  4) 目标协议回测 (Trading 引擎真实净值口径) + model_pic/figure_01.png

用法:
  python3 analysis.py                  # 出完整报告 (含权重搜索, 首次慢)
  python3 analysis.py --days 3         # 推荐块数量 (默认 10)
  python3 analysis.py --date 20260902  # 只看指定因子日
  python3 analysis.py --update         # 强制重推最近 10 个交易日
  python3 analysis.py --skip-infer     # 跳过推演 (直接用现有 heads)
  python3 analysis.py --no-search      # 跳过权重搜索 (用默认 ens21 权重)
  python3 analysis.py --no-md          # 不写 model_pic/output.md
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
                   CACHE_SCORE, FAC_PATH, TEST_START, TEST_END, PROTO_TOPN,
                   PROTO_HOLD, PROTO_TRAIL, TRADING_ENGINE,
                   fold_spec, N_FOLD_PER_FAMILY, TOTAL_FOLDS, PredictModel,
                   load_trading_dates, load_name_map, score_ens21, zn,
                   champion_backtest, rank_block, decision_block,
                   latest_reportable_date, heads_last_date, next_td)

# ============================================================
# 配置
# ============================================================
MODEL_TRAIN = rf'{root_path}/model_train/{SEASON}'
MODEL_TEST = rf'{root_path}/model_test'
FEATURE_MAP = os.path.join(MODEL_TEST, 'feature_map.fea')
TAIL_DAYS = 10          # --update 时重推的天数
DEFAULT_DAYS = 10       # 推荐块滑动窗口
SEP = '=' * 100
SEP2 = '=' * 55
BAR = '─' * 60
HEADS_ALL = ('r5', 'r10', 'r20', 'top')

_VAL_RANKIC_RE = re.compile(r'val_rankic=(-?\d+\.\d+)')
_VAL_RANKIC_AVG_RE = re.compile(r'val_rankic_avg=(-?\d+\.\d+)')


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


def load_model(checkpoint_path, input_dim, main_h='5d'):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = {k.removeprefix('model.'): v for k, v in ck['state_dict'].items()
             if k.startswith('model.')}
    m = PredictModel(input_dim=input_dim, main_h=main_h)
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
        m = (_VAL_RANKIC_AVG_RE.search(os.path.basename(ck))
             or _VAL_RANKIC_RE.search(os.path.basename(ck)))
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
        for name, t in zip(('mixed', 'r5', 'r10', 'r20', 'top'), out[:5]):
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
        m = load_model(ck, len(factor_cols), main_h='5d' if fam == 'a' else '10d')
        m.to(device)
        models[g] = m
        print(f'[analysis]   fold{g} (族{fam}/k{k}): {os.path.basename(ck)} '
              f'(val={v:.4f})', flush=True)

    head_rows = {(fam, k, h): [] for fam in ('a', 'c')
                 for k in range(1, N_FOLD_PER_FAMILY + 1)
                 for h in HEADS_ALL}
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
            for h in HEADS_ALL:
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
            for h in HEADS_ALL:
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
# ens21 权重搜索 (目标协议 h20tr20_t2, Trading 引擎真实净值)
# ============================================================
def _load_head_frames(heads_dir):
    frames = {h: {} for h in HEADS_ALL}
    for h in HEADS_ALL:
        for f in range(1, N_FOLD_PER_FAMILY + 1):
            p = os.path.join(heads_dir, f'{h}_f{f}.fea')
            frames[h][f] = pd.read_feather(p).set_index('date')
    return frames


def _combo(frames, w5, w10, w20, wt):
    score = None
    for f in range(1, N_FOLD_PER_FAMILY + 1):
        fs = w5 * frames['r5'][f]
        if w10:
            fs = fs + w10 * frames['r10'][f]
        if w20:
            fs = fs + w20 * frames['r20'][f]
        fs = fs + wt * frames['top'][f]
        fs = zn(fs)
        score = fs if score is None else score.add(fs, fill_value=0.0)
    return zn(score)


def _df_to_scores(df):
    ranked, scores, top1 = {}, {}, {}
    for d in df.index:
        row = df.loc[d].dropna()
        scores[d] = row.to_dict()
        ranked[d] = row.sort_values(ascending=False).index.tolist()
        top1[d] = float(row.max()) if len(row) else np.nan
    return dict(scores=scores, ranked=ranked, top1=top1, dates=sorted(scores.keys()))


_WS = {}   # 权重搜索共享状态 (fork 子进程继承)


def _ws_run(args):
    name, df = args
    sc = _df_to_scores(df)
    m, tr, eq = _WS['eng'].run_backtest(sc, _WS['proto'], _WS['mkt'],
                                        _WS['tds'], _WS['tdi'])
    if m is None:
        return None
    return dict(name=name, cum_net=m['cum_net'], sharpe=m['sharpe'],
                maxdd=m['maxdd'], h1=m['h1_cum'], h2=m['h2_cum'],
                win=m['win_rate'], n=m['n_trades'], hold=m['avg_hold'])


def weight_search(verbose=True):
    """家族权重 × 跨族权重网格 → h20tr20_t2 回测; 落盘 weight_search.csv。"""
    import importlib.util
    import multiprocessing as mp
    spec = importlib.util.spec_from_file_location('trading_engine', TRADING_ENGINE)
    eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eng)

    fa = _load_head_frames(HEADS_A)
    fc = _load_head_frames(HEADS_C)
    mkt = eng.load_market()
    tds, tdi = eng.load_calendar()
    F = dict(top_n=PROTO_TOPN, hold=PROTO_HOLD, trail_pct=PROTO_TRAIL)
    _WS.update(eng=eng, proto=F, mkt=mkt, tds=tds, tdi=tdi)

    # 家族内权重网格 (w5 或 w10 固定 1, z 缩放不变性)
    combos = {}
    for w10 in (0.0, 0.25, 0.5, 1.0):
        for w20 in (0.0, 0.25, 0.5):
            for wt in (1.0, 2.0, 3.0):
                combos[f"A_w10{w10}_w20{w20}_wt{wt}"] = \
                    ('A', _combo(fa, 1.0, w10, w20, wt))
    for w5 in (0.0, 0.25):
        for w20 in (0.0, 0.25, 0.5, 1.0):
            for wt in (0.5, 1.0, 2.0):
                combos[f"C_w5{w5}_w20{w20}_wt{wt}"] = \
                    ('C', _combo(fc, w5, 1.0, w20, wt))
    n_fam = len(combos)
    print(f'[search] 家族组合 {n_fam} 个 (A {n_fam//2} + C {n_fam//2})', flush=True)

    pool = mp.Pool(20)
    rows = [r for r in pool.imap_unordered(
        _ws_run, [(n, c[1]) for n, c in combos.items()], chunksize=2) if r]
    pool.close(); pool.join()
    fam_res = pd.DataFrame(rows)
    # 每族取前 N 名做跨族集成
    N = 6
    a_top = fam_res[fam_res['name'].str.startswith('A')].sort_values(
        'cum_net', ascending=False).head(N)['name'].tolist()
    c_top = fam_res[fam_res['name'].str.startswith('C')].sort_values(
        'cum_net', ascending=False).head(N)['name'].tolist()
    print(f'[search] 族A 前{N}: {a_top}')
    print(f'[search] 族C 前{N}: {c_top}')

    ens_tasks = []
    for na in a_top:
        for nc in c_top:
            da, dc = combos[na][1], combos[nc][1]
            for wc in (0.5, 1.0, 1.5, 2.0, 3.0):
                ens = zn(da).add(wc * zn(dc), fill_value=0.0)
                ens_tasks.append((f"{na}+{wc}x{nc}", ens))
    print(f'[search] 跨族集成 {len(ens_tasks)} 组合', flush=True)
    pool = mp.Pool(20)
    ens_rows = [r for r in pool.imap_unordered(_ws_run, ens_tasks, chunksize=2) if r]
    pool.close(); pool.join()
    ens_res = pd.DataFrame(ens_rows)
    all_res = pd.concat([fam_res.assign(kind='fam'), ens_res.assign(kind='ens')],
                        ignore_index=True)
    out_path = os.path.join(MODEL_PRED, 'weight_search.csv')
    os.makedirs(MODEL_PRED, exist_ok=True)
    all_res.to_csv(out_path, index=False)
    print(f'[search] 权重搜索结果 → {out_path}')
    if verbose:
        pd.set_option('display.width', 200)
        print('[search] ens Top10 (cum_net):')
        print(ens_res.sort_values('cum_net', ascending=False).head(10)
              [['name', 'cum_net', 'sharpe', 'maxdd', 'h1', 'h2', 'n']]
              .round(3).to_string(index=False))
    return all_res


# ============================================================
# 报告构建
# ============================================================
def fmt_date(d):
    s = str(d)
    return f'{int(s[:4])}年{int(s[4:6])}月{int(s[6:8])}日'


def ic_block(ic):
    lines = ['---Test 集合打分 (长周期, 可交易池)---',
             f'  打分区间: Test 集合 {TEST_START} ~ {TEST_END} (严格样本外)', '']
    for k in ('ic5', 'ic10', 'ic20', 'top5', 'top10'):
        lines.append(f'  {k:<8} {ic.get(k, float("nan")):+.4f}')
    lines += ['[说明] ic5/ic10/ic20 = RankIC vs 5d/10d/20d 前瞻收益; '
              'top5/top10 = Top1 的 5d/10d 平均收益', '']
    return '\n'.join(lines)


def champion_log(score, names, verbose=True):
    """目标协议成交明细 (h20tr20_t2, 含成本, 窗口 = 打分起点 ~ 最新因子日)。"""
    from model import _trading_engine, _df_to_scores
    eng = _trading_engine()
    mkt = eng.load_market()
    tds, tdi = eng.load_calendar()
    F = dict(top_n=PROTO_TOPN, hold=PROTO_HOLD, trail_pct=PROTO_TRAIL)
    sc = _df_to_scores(score)
    m, trades, eq = eng.run_backtest(sc, F, mkt, tds, tdi)
    tr = trades.sort_values('buy_dt').reset_index(drop=True) if trades is not None \
        else pd.DataFrame()
    L = [SEP,
         f'  Trade Log — ens21 目标协议 Top{PROTO_TOPN}+持有≤{PROTO_HOLD}日'
         f'+移动止损{int(PROTO_TRAIL*100)}% | 开盘换仓 | 含成本 | '
         f'{score.index.min()} ~ {score.index.max()}',
         SEP]
    L.append(' BuyDt     SellDt    Code     Name       BuyPrc   SellPrc  '
             'Hold  Net%     Reason')
    L.append('-' * 100)
    for _, r in tr.iterrows():
        L.append(f' {r["buy_dt"]:<10}{r["sell_dt"]:<10}{r["code"]:<7}'
                 f'{r.get("name") or names.get(r["code"], ""):<9}'
                 f'{r["buy_prc"]:9.2f} {r["sell_prc"]:9.2f} '
                 f'{int(r["hold_days"]):>4} {r["net_pct"]:+8.2%}  {r["reason"]}')
    L.append(SEP)
    if m is not None:
        L.append(f'Realized: {m["n_trades"]}  |  True equity (净值): '
                 f'{m["cum_net"]:+.2%}  |  逐笔复利: {m["cum_trade"]:+.2%}  |  '
                 f'Win rate: {m["win_rate"]:.1%}  |  Avg hold: {m["avg_hold"]:.1f} 日')
        L.append(f'成本拖累: {m["cost_drag"]:+.2%}  |  换手: {m["turnover"]:.2f}  |  '
                 f'被阻买入 {m["skipped_buy"]} / 被阻卖出 {m["blocked_sell"]}')
    stats = dict(realized=m['n_trades'] if m else 0, cum=m['cum_net'] if m else 0.0,
                 win=m['win_rate'] if m else 0.0)
    return '\n'.join(L), stats, tr


def figure_01(score, trades, out_path, title_window):
    """V9 风格图: 左 = 净值曲线 (引擎 equity), 右 = 逐笔净收益直方。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[analysis] matplotlib 不可用, 跳过出图: {e}')
        return
    from model import _trading_engine, _df_to_scores
    eng = _trading_engine()
    mkt = eng.load_market()
    tds, tdi = eng.load_calendar()
    F = dict(top_n=PROTO_TOPN, hold=PROTO_HOLD, trail_pct=PROTO_TRAIL)
    m, trades, eq = eng.run_backtest(_df_to_scores(score), F, mkt, tds, tdi)
    if m is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    eqd = eq.set_index('date')['equity'] / eq['equity'].iloc[0] - 1
    axes[0].plot(range(len(eqd)), eqd.values, color='#d62728', lw=1.6)
    axes[0].axhline(0, color='gray', lw=0.8, ls='--')
    axes[0].set_title(f'True Equity (Top{PROTO_TOPN} hold≤{PROTO_HOLD} '
                      f'trail{int(PROTO_TRAIL*100)}%) | {title_window}', fontsize=13)
    axes[0].set_xlabel('Trading day')
    axes[0].set_ylabel('Cumulative Net Return')
    axes[0].grid(alpha=0.3)
    net = trades['net_pct'] / 100.0
    axes[1].bar(range(len(net)), net.values,
                color=['#2ca02c' if v >= 0 else '#d62728' for v in net])
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
    ap = argparse.ArgumentParser(description='V21 推演/权重搜索/回测/最终结果入口')
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--date', default=None, help='只展示指定因子日 YYYYMMDD')
    ap.add_argument('--days', type=int, default=DEFAULT_DAYS, help='推荐块数量 (默认 10)')
    ap.add_argument('--no-split', action='store_true', help='Summary 不做 H1/H2 分半')
    ap.add_argument('--update', action='store_true',
                    help=f'强制重推最近 {TAIL_DAYS} 个交易日')
    ap.add_argument('--skip-infer', action='store_true', help='跳过推演 (直接用 heads)')
    ap.add_argument('--no-search', action='store_true',
                    help='跳过权重搜索 (用默认 ens21 权重)')
    ap.add_argument('--no-md', action='store_true', help='不写 model_pic/output.md')
    args = ap.parse_args()

    # 1) 推演 (按需 / 强制)
    target = latest_reportable_date()
    if not args.skip_infer:
        ensure_heads(target, force_update=args.update)
    else:
        print('[analysis] 跳过推演 (--skip-infer)')

    # 2) 权重搜索 (目标协议上选 ens21 权重)
    if not args.no_search:
        try:
            weight_search(verbose=True)
        except FileNotFoundError as e:
            print(f'[analysis] 权重搜索跳过: {e}')

    # 3) 组装 ens21 打分 (默认权重; 权重搜索供人工确认)
    try:
        score = score_ens21()
    except FileNotFoundError as e:
        print(f'[analysis] 无法组装打分: {e}')
        print('  请先完成 16 折训练 (bash train.sh) 并运行本文件推演 heads。')
        sys.exit(1)
    print(f'[analysis] ens21 组装完成: {score.shape[0]} 天 × {score.shape[1]} 股票 '
          f'({score.index.min()} ~ {score.index.max()})', flush=True)
    try:
        os.makedirs(MODEL_PRED, exist_ok=True)
        score.reset_index().to_feather(CACHE_SCORE)
        print(f'[analysis] 打分缓存: {CACHE_SCORE}')
    except OSError as e:
        print(f'[analysis] 打分缓存写入失败 (跳过, 不影响报告): {e}')

    # 4) 报告构建
    dates = list(score.index)
    tds, name_map = load_trading_dates(), load_name_map()
    latest = dates[-1]

    bt = champion_backtest(score, split=True, verbose=False)
    ic_res = bt['ic']
    text = [f'# V21 每日报告 (ens21 长周期打分 + Top{PROTO_TOPN} 持有≤{PROTO_HOLD}日'
            f'+移动止损{int(PROTO_TRAIL*100)}% 开盘换仓, 含成本)', '']
    text.append(ic_block(ic_res))

    # 排行榜块
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

    # Trade Log + Summary
    log_text, stats, trades = champion_log(score, name_map)
    text.append(log_text)
    text.append('')
    f_ = bt['full']
    text.append(SEP2)
    text.append(f'  Recent Summary | ens21 + Top{PROTO_TOPN} 持有≤{PROTO_HOLD}日'
                f'+移动止损{int(PROTO_TRAIL*100)}% | 开盘换仓 | 含成本 | 官方 Test 窗口')
    text.append(SEP2)
    text.append(f'  Test 窗口  : {TEST_START} ~ {TEST_END}  (243 交易日, 严格样本外)')
    if f_ is not None:
        text.append(f'  真实净值    : {f_["cum_net"]:+.2%}  |  逐笔复利 {f_["cum_trade"]:+.2%}'
                    f'  |  Sharpe {f_["sharpe"]:.3f}')
        text.append(f'  MaxDD      : {f_["maxdd"]:+.2%}  |  胜率 {f_["win_rate"]:.1%}  |  '
                    f'交易 {f_["n_trades"]}  |  成本拖累 {f_["cost_drag"]:+.2%}')
        text.append(f'  平均持有    : {f_["avg_hold"]:.1f} 交易日  |  资金闲置 '
                    f'{f_["avg_cash_ratio"]:.1%}')
        if not args.no_split:
            h1, h2 = bt['h1'], bt['h2']
            if h1 is not None and h2 is not None:
                text.append(f'  分半 H1/H2 : 20250901~20260227 {h1["cum_net"]:+.2%}  |  '
                            f'20260302~20260901 {h2["cum_net"]:+.2%}')
    text.append(SEP2)
    text.append('')
    text.append(decision_block(latest, score, tds=tds, name_map=name_map,
                               holdings_path=os.path.join(PROJECT_ROOT,
                                                          'Model/V11/holdings.json')))
    text.append('')
    report = '\n'.join(text)

    sys.stdout.write(report)
    sys.stdout.flush()

    if not args.no_md:
        md_dir = os.environ.get('V21_MD_DIR') or os.path.join(HERE, 'model_pic')
        try:
            os.makedirs(md_dir, exist_ok=True)
            with open(os.path.join(md_dir, 'output.md'), 'w', encoding='utf-8') as f:
                f.write(report)
            figure_01(score, trades, os.path.join(md_dir, 'figure_01.png'),
                      f'{score.index.min()} ~ {score.index.max()}')
            print(f'\n[analysis] 报告已写入: {md_dir}/output.md (+figure_01.png)')
        except OSError as e:
            print(f'\n[analysis] 报告写入 {md_dir} 失败 (跳过): {e}')


if __name__ == '__main__':
    main()
