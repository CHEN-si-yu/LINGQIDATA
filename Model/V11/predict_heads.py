#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_heads.py — 用 V9 的 8 个 checkpoint 提取各头打分 (CPU 推理)

输出 (Model/V11/mix_v9/):
  r1.fea / r5.fea / top.fea — 各头跨 fold z-score 求和集成矩阵 (date × code)
  mixed_orig.fea            — 原始 V9 集成定义 Σ_f z(z(r1)+z(top)) (一致性校验用)

后续 mix_grid.py 在这些矩阵上做权重网格 + 策略回测, 无需重推。
"""
import glob
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
V9_DIR = PROJECT_ROOT + "Model/V9"
OUT_DIR = PROJECT_ROOT + "Model/V11/mix_v9"
TEST_START, TEST_END = '20250901', '20260901'

# 导入 V9 model.py (会加载标签/掩码, ~30s)
spec = importlib.util.spec_from_file_location('v9model', V9_DIR + '/model.py')
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)


def load_feature_map(path):
    with open(path) as f:
        raw = f.read().replace('\\n', '\n')
    order = []
    for line in raw.split('\n'):
        if '=' in line:
            name, idx = line.rsplit('=', 1)
            order.append((int(idx.strip()), name.strip()))
    order.sort()
    return [n for _, n in order]


def find_best_ckpt(fold_dir):
    pat = re.compile(r'val_rankic=(-?\d+\.\d+)')
    best, bestv = None, -1e9
    for ck in glob.glob(os.path.join(fold_dir, '**', '*.ckpt'), recursive=True):
        m = pat.search(os.path.basename(ck))
        if m:
            v = float(m.group(1))
            if v > bestv:
                bestv, best = v, ck
    return best, bestv


def main():
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    os.makedirs(OUT_DIR, exist_ok=True)

    factors = load_feature_map(V9_DIR + '/model_test/feature_map.fea')
    print(f'[heads] {len(factors)} factors')

    # 只加载 Test 区间因子数据
    import pyarrow as pa, pyarrow.feather as pf, pyarrow.compute as pc
    fac_path = PROJECT_ROOT + 'trainingdata/fac_all.fea'
    all_cols = pf.read_table(fac_path, columns=[]).column_names
    available = [f for f in factors if f in all_cols]
    cols = ['date', 'Code'] + available
    table = pf.read_table(fac_path, columns=cols)
    dates_all = pc.unique(table.column('date')).to_pandas().astype(str).sort_values()
    dates = [d for d in dates_all if TEST_START <= d <= TEST_END]
    mask = pc.is_in(table.column('date'), pa.array(dates))
    pdf = table.filter(mask).to_pandas()
    del table
    data = pdf.set_index('date').sort_index().reset_index()
    del pdf
    print(f'[heads] 因子数据 {len(dates)} 天加载完成')

    folds = sorted(glob.glob(V9_DIR + '/model_train/2026q3/fold[0-9]*'))
    models = []
    for fd in folds:
        ck, v = find_best_ckpt(fd)
        print(f'[heads] {os.path.basename(fd)}: {os.path.basename(ck)} (val_rankic={v:.4f})')
        m = v9.PredictModel(input_dim=len(factors))
        st = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
        st = {k.removeprefix('model.'): vv for k, vv in st.items() if k.startswith('model.')}
        m.load_state_dict(st, strict=False)
        m.eval()
        models.append(m)

    def normed(dd):
        d = dd.copy()
        d = d.dropna(subset=factors, thresh=max(1, int(0.1 * len(factors))))
        X = d[factors].rank(axis=0)
        X = ((X - X.mean()) / X.std()).fillna(0)
        return torch.from_numpy(np.nan_to_num(
            X.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)), d['Code'].values

    saved = {h: [] for h in ['r1', 'r5', 'top', 'mixed_orig']}
    # 逐折头部矩阵: 每 head 一个 MultiIndex (fold, code) 宽表
    fold_saved = {h: [] for h in ['r1', 'r5', 'top']}
    with torch.no_grad():
        for di, date in enumerate(dates):
            dd = data[data['date'] == date]
            X, codes = normed(dd)
            fold_r1 = []
            fold_top = []
            h_sum = {h: None for h in ['r1', 'r5', 'top']}
            h_fold_rows = {h: {} for h in ['r1', 'r5', 'top']}
            for fi, m in enumerate(models):
                mixed, r1, r5, top = m(X)
                fold_r1.append(r1.numpy().ravel())
                fold_top.append(top.numpy().ravel())
                for h, t in [('r1', r1), ('r5', r5), ('top', top)]:
                    v = t.numpy().ravel()
                    z = (v - v.mean()) / v.std()
                    h_sum[h] = z if h_sum[h] is None else h_sum[h] + z
                    h_fold_rows[h][fi + 1] = z
            # 原始 V9 定义: Σ_f z(z(r1)+z(top))
            orig = None
            for r1v, topv in zip(fold_r1, fold_top):
                mixf = (r1v - r1v.mean()) / r1v.std() + \
                       (topv - topv.mean()) / topv.std()
                z = (mixf - mixf.mean()) / mixf.std()
                orig = z if orig is None else orig + z
            for h in ['r1', 'r5', 'top']:
                saved[h].append(pd.Series(h_sum[h], index=codes, name=date))
                row = pd.Series(
                    np.concatenate([h_fold_rows[h][fi + 1] for fi in range(len(models))]),
                    index=pd.MultiIndex.from_product(
                        [[fi + 1 for fi in range(len(models))], codes]),
                    name=date)
                fold_saved[h].append(row)
            saved['mixed_orig'].append(pd.Series(orig, index=codes, name=date))
            if (di + 1) % 25 == 0:
                print(f'[heads] {di + 1}/{len(dates)} done')

    for h, series in saved.items():
        df = pd.DataFrame(series)
        df.index.name = 'date'
        df.reset_index().to_feather(os.path.join(OUT_DIR, f'{h}.fea'))
        print(f'[heads] saved {h}: {df.shape}')
    for h, rows in fold_saved.items():
        # 每 (head, fold) 单独存文件 (feather 对 MultiIndex 列往返不可靠)
        # 注意各日有效股票数不同 → 每行独立切分
        nf = len(models)
        for fi in range(nf):
            sub = []
            for r in rows:
                codes_row = r.index.get_level_values(1)
                nc = len(codes_row) // nf
                sub.append(pd.Series(r.values[fi * nc:(fi + 1) * nc],
                                     index=codes_row[:nc], name=r.name))
            df = pd.DataFrame(sub)
            df.index.name = 'date'
            df.reset_index().to_feather(os.path.join(OUT_DIR, f'{h}_f{fi + 1}.fea'))
        print(f'[heads] saved {h}_f1..f{nf}: ({len(rows)} days)')

    # 一致性校验: mixed_orig vs 官方 all_zscore_score.fea
    official = pd.read_feather(V9_DIR + '/model_pred/2026q3/all_zscore_score.fea').set_index('date')
    mine = pd.DataFrame(saved['mixed_orig'])
    mine.index.name = 'date'
    common = official.index.intersection(mine.index)
    corr = official.loc[common].corrwith(mine.loc[common], axis=1)
    print(f'[heads] mixed_orig vs official score corr: mean={corr.mean():.6f} '
          f'min={corr.min():.6f}')


if __name__ == '__main__':
    main()
