#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py — V20 推演/回测/最终结果入口 (对 V20 自训的 16 折 checkpoint 工作)

使命: 用 model.py 定义的结构 + run.py/train.sh 训练出的 16 个 checkpoint,
全窗口逐日推演各头打分 → 组装冠军 ens_w2 → 输出排行榜/策略指令 (可选冠军回测)。

用法:
  python3 analysis.py                  # 若 heads 未覆盖最新因子日 → 自动增量推演 16 折, 再出榜
  python3 analysis.py --top 20         # 榜单长度
  python3 analysis.py --days 3         # 最近 3 个因子日排行榜块
  python3 analysis.py --date 20260902  # 指定因子日出榜
  python3 analysis.py --backtest       # 冠军协议全窗口回测 (hold5+stop8%+收盘卖, 含成本, 附分半/IC)
  python3 analysis.py --update         # 强制重推最近 10 个交易日 (换 checkpoint 后自愈用)
  python3 analysis.py --skip-infer     # 跳过推演 (直接用现有 heads_a/heads_c)
  python3 analysis.py --no-md          # 不写 model_pic/output.md

首次使用 (16 折训练完成后): 运行本文件触发全窗口推演 (约 245 交易日 × 16 模型,
GPU 约 30~40 分钟), 产物: model_pred/2026q3/heads_{a,c}/{r1,r3,r5,top}_f{1..8}.fea
+ score_ens_w2.fea。之后每日因子更新只需增量推演最新日期 (约数分钟)。
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
                   CACHE_SCORE, FAC_PATH, TEST_START, TEST_END,
                   fold_spec, N_FOLD_PER_FAMILY, TOTAL_FOLDS, PredictModel,
                   load_trading_dates, load_name_map, score_ens_w2,
                   champion_backtest, rank_block, decision_block,
                   latest_reportable_date, heads_last_date)

# ============================================================
# 配置
# ============================================================
MODEL_TRAIN = rf'{root_path}/model_train/{SEASON}'
MODEL_TEST = rf'{root_path}/model_test'
FEATURE_MAP = os.path.join(MODEL_TEST, 'feature_map.fea')
TAIL_DAYS = 10          # --update 时重推的天数

# ============================================================
# 推演辅助 (与 V11/analysis.py 同口径; 训练推理标准化见 model.normed_data)
# ============================================================
_VAL_RANKIC_RE = re.compile(r'val_rankic=(-?\d+\.\d+)')
_VAL_RANKIC_AVG_RE = re.compile(r'val_rankic_avg=(-?\d+\.\d+)')
_VAL_IC_RE = re.compile(r'val_icmean=(-?\d+\.\d+)')
_VAL_WEI_RE = re.compile(r'val_wei=(-?\d+\.\d+)')


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
    """推理标准化: 有效性过滤 → 每因子截面 rank → 标准化 → 缺失填 0。"""
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
        m = (_VAL_RANKIC_AVG_RE.search(os.path.basename(ck))
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
    """单日单模型各头打分 (未 zscore)。返回 {h: DataFrame(index=Code, value)}。"""
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
    """按需推演, 保证两族 heads 覆盖到 target_date。"""
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

    # 目标日期 = 因子日 ∩ 交易日, 且 >= TEST_START
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

    # 加载因子 (仅目标日期行, 全特征列)
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

    # 16 折 checkpoint
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

    # 输入维度 (与 feature_map 对齐; 若 checkpoint 维度不同则补零/裁剪)
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

    # 逐日 × 16 折推演, 各头逐折 z (逐日截面) 收集
    head_rows = {(fam, k, h): [] for fam in ('a', 'c')
                 for k in range(1, N_FOLD_PER_FAMILY + 1)
                 for h in ('r1', 'r3', 'r5', 'top')}
    total = len(run_dates) * TOTAL_FOLDS
    done = 0
    for d in run_dates:
        if d not in data.index:
            continue
        day = data.loc[d]
        if isinstance(day, pd.Series):      # 单行兜底
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
                print(f'[analysis] 推演进度 {done}/{total} '
                      f'(最新 {d})', flush=True)
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
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='V20 推演/回测/最终结果入口')
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--date', default=None, help='指定因子日 YYYYMMDD 出榜')
    ap.add_argument('--days', type=int, default=1, help='最近 N 个因子日出榜')
    ap.add_argument('--backtest', action='store_true', help='追加冠军协议全窗口回测')
    ap.add_argument('--no-split', action='store_true', help='回测不做 H1/H2 分半')
    ap.add_argument('--update', action='store_true',
                    help=f'强制重推最近 {TAIL_DAYS} 个交易日')
    ap.add_argument('--skip-infer', action='store_true',
                    help='跳过推演 (直接用现有 heads)')
    ap.add_argument('--no-md', action='store_true', help='不写 model_pic/output.md')
    args = ap.parse_args()

    # 1) 推演 (按需 / 强制)
    target = latest_reportable_date()
    if not args.skip_infer:
        ensure_heads(target, force_update=args.update)
    else:
        print(f'[analysis] 跳过推演 (--skip-infer)')

    # 2) 组装冠军打分
    try:
        score = score_ens_w2()
    except FileNotFoundError as e:
        print(f'[analysis] 无法组装打分: {e}')
        print('  请先完成 16 折训练 (bash train.sh) 并运行本文件推演 heads。')
        sys.exit(1)
    print(f'[analysis] ens_w2 组装完成: {score.shape[0]} 天 × {score.shape[1]} 股票 '
          f'({score.index.min()} ~ {score.index.max()})', flush=True)
    os.makedirs(MODEL_PRED, exist_ok=True)
    score.reset_index().to_feather(CACHE_SCORE)
    print(f'[analysis] 打分缓存: {CACHE_SCORE}')

    # 3) 选日期出榜
    dates = list(score.index)
    if args.date:
        if args.date not in score.index:
            ap.error(f'--date {args.date} 不在打分范围 ({dates[0]}~{dates[-1]})')
        show_dates = [args.date]
    else:
        show_dates = dates[-max(1, min(args.days, len(dates))):]
    tds, name_map = load_trading_dates(), load_name_map()
    text = '\n'.join(rank_block(d, score, topn=args.top, tds=tds, name_map=name_map)
                     for d in show_dates).rstrip() + '\n'
    text += decision_block(show_dates[-1], score, tds=tds, name_map=name_map,
                           holdings_path=os.path.join(PROJECT_ROOT,
                                                      'Model/V11/holdings.json')) + '\n'
    sys.stdout.write(text)
    sys.stdout.flush()
    if not args.no_md:
        md_dir = os.path.join(HERE, 'model_pic')
        os.makedirs(md_dir, exist_ok=True)
        with open(os.path.join(md_dir, 'output.md'), 'w', encoding='utf-8') as f:
            f.write('# V20 冠军系统每日结果 (ens_w2 + hold5s8-收盘卖)\n\n')
            f.write(text)

    # 4) 可选回测
    if args.backtest:
        print()
        champion_backtest(score, split=not args.no_split, verbose=True)


if __name__ == '__main__':
    main()
