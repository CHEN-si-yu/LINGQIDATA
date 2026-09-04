#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_valid.py — 推演 V11/V13 各折模型在验证期 (valid 窗口, 20250501~20250808)
的头部打分, 供 V19 堆叠元模型训练使用 (基模型未见过该期 = 元模型合法训练样本)

输出: Model/V19/meta_feats/{fam}{fold}_{head}.fea  (fam ∈ {a= V11, c= V13})
      + label_rg / buyable / 5d 标签 (逐日宽表)
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
OUT_DIR = PROJECT_ROOT + "Model/V19/meta_feats"
V11_DIR = PROJECT_ROOT + "Model/V11"
V13_DIR = PROJECT_ROOT + "Model/V13"
VALID_START, VALID_END = '20250501', '20250808'  # TRAIN_END 前 120 天窗口内


def load_model_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_feature_map(path):
    with open(path) as f:
        raw = f.read().replace('\\n', '\n')
    order = []
    for line in raw.split('\n'):
        if '=' in line:
            n, idx = line.rsplit('=', 1)
            order.append((int(idx.strip()), n.strip()))
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
    return best


def main():
    torch.set_num_threads(min(12, os.cpu_count() or 8))
    os.makedirs(OUT_DIR, exist_ok=True)
    v11 = load_model_module(V11_DIR + '/model.py', 'v11m')
    v13 = load_model_module(V13_DIR + '/model.py', 'v13m')
    factors = load_feature_map(V11_DIR + '/model_test/feature_map.fea')

    import pyarrow as pa, pyarrow.feather as pf, pyarrow.compute as pc
    fac_path = PROJECT_ROOT + 'trainingdata/fac_all.fea'
    all_cols = pf.read_table(fac_path, columns=[]).column_names
    available = [f for f in factors if f in all_cols]
    cols = ['date', 'Code'] + available
    table = pf.read_table(fac_path, columns=cols)
    dates_all = pc.unique(table.column('date')).to_pandas().astype(str).sort_values()
    dates = [d for d in dates_all if VALID_START <= d <= VALID_END]
    mask = pc.is_in(table.column('date'), pa.array(dates))
    data = table.filter(mask).to_pandas().set_index('date').sort_index().reset_index()
    del table
    print(f'[meta] valid 期因子 {len(dates)} 天加载完成')

    def normed(dd):
        d = dd.copy()
        d = d.dropna(subset=factors, thresh=max(1, int(0.1 * len(factors))))
        X = d[factors].rank(axis=0)
        X = ((X - X.mean()) / X.std()).fillna(0)
        return torch.from_numpy(np.nan_to_num(
            X.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)), \
            d['Code'].values

    def predict_family(mod, root, tag, fold_range, heads):
        frames = {h: [] for h in heads}
        for f in fold_range:
            fd = f'{root}/model_train/2026q3/fold{f}'
            ck = find_best_ckpt(fd)
            m = mod.PredictModel(input_dim=len(factors))
            st = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
            st = {k.removeprefix('model.'): v for k, v in st.items()
                  if k.startswith('model.')}
            m.load_state_dict(st, strict=False)
            m.eval()
            print(f'[meta] {tag} fold{f}: {os.path.basename(ck)}')
            rows = {h: [] for h in heads}
            with torch.no_grad():
                for date in dates:
                    dd = data[data['date'] == date]
                    X, codes = normed(dd)
                    out = m(X)
                    if isinstance(out, tuple) and len(out) >= 6:
                        mixed, r1, r5, r3, top1d, top3d = out
                        hvals = {'r1': r1, 'r5': r5, 'r3': r3,
                                 'top1d': top1d, 'top3d': top3d}
                    elif isinstance(out, tuple) and len(out) == 5:
                        mixed, r1, r5, r3, top = out
                        hvals = {'r1': r1, 'r5': r5, 'r3': r3, 'top': top}
                    else:
                        mixed, r1, r5, top = out
                        hvals = {'r1': r1, 'r5': r5, 'top': top}
                    for h in heads:
                        if h in hvals:
                            v = hvals[h].numpy().ravel()
                            z = (v - v.mean()) / v.std()
                            rows[h].append(pd.Series(z, index=codes, name=date))
            for h in heads:
                df = pd.DataFrame(rows[h])
                df.index.name = 'date'
                df.reset_index().to_feather(
                    os.path.join(OUT_DIR, f'{tag}{f}_{h}.fea'))
            print(f'[meta] {tag} fold{f} 各头落盘 ({len(dates)} 天)')

    predict_family(v11, V11_DIR, 'a', range(1, 9), ['r1', 'r5', 'r3', 'top'])
    predict_family(v13, V13_DIR, 'c', range(1, 9), ['r1', 'r5', 'r3', 'top'])
    print('[meta] V19 元特征推演完成')


if __name__ == '__main__':
    main()
