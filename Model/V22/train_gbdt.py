#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gbdt.py — V22: LightGBM 长周期排序器单元 (模型×策略耦合: 瞄准 h20tr20_t2 协议)

与 V18 修复后管线同口径, 标签换为协议耦合周期:
- 特征: 848 因子, 逐日截面 rank → 标准化 → 缺失填 0 (float32)
- 标签: 10d / 20d 收益 winsor(MAD5) → 逐日截面秩高斯化 (协议平均持有 ~19 交易日)
- 划分: 同 model.py 4 折 (seed 3253), train < 20250809, valid=最后120天, purge 5
- 权重: 时间衰减 hl=600d
- 输出: Model/V22/model_pred/v22_{10d,20d}_f{1..4}.fea + v22_{10d,20d}.fea (跨折 z 求和)
用法: python3 train_gbdt.py  (n_jobs 通过环境变量 V22_NJOBS 调, 默认 10)
"""
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
ROOT = PROJECT_ROOT + "Model/V22"
TEST_START, TEST_END = '20250901', '20260901'
TRAIN_END = '20250809'
VALID_DAYS, N_FOLDS, PURGE_DAYS = 120, 4, 5
LABEL_WINSOR_MAD = 5.0
HALF_LIFE = 600
SEED = 3253
BATCH_DAYS = 150
N_JOBS = int(os.environ.get('V22_NJOBS', '10'))


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


def time_weight(date_str):
    d = datetime.strptime(date_str, '%Y%m%d')
    ref = datetime.strptime(TRAIN_END, '%Y%m%d')
    return 0.5 ** (max(0, (ref - d).days) / HALF_LIFE)


def preprocess():
    """特征矩阵 (一次预处理, 两个标签周期共用)。"""
    factors = load_feature_map(PROJECT_ROOT + 'Model/V11/model_test/feature_map.fea')
    nf = len(factors)
    print(f'[V22] {nf} factors', flush=True)
    import pyarrow.feather as pf
    dates_all = sorted(pf.read_table(PROJECT_ROOT + 'trainingdata/fac_all.fea',
                                     columns=['date'])['date']
                       .to_pandas().astype(str).unique())
    print(f'[V22] {len(dates_all)} 个日期', flush=True)
    df_full = pd.read_feather(PROJECT_ROOT + 'trainingdata/fac_all.fea',
                              columns=['date', 'Code'] + factors)
    df_full['date'] = df_full['date'].astype(str)
    df_full[factors] = df_full[factors].astype(np.float32)
    print(f'[V22] {df_full.shape[0]} rows 已读入', flush=True)
    X_parts, date_parts, code_parts = [], [], []
    for di, d in enumerate(dates_all):
        m = df_full['date'] == d
        sub = df_full.loc[m, factors]
        X = sub.rank(axis=0).astype(np.float32)
        mu = X.mean(axis=0)
        sd = X.std(axis=0).replace(0, 1.0)
        X = ((X - mu) / sd).fillna(0.0)
        X_parts.append(X.to_numpy(np.float32))
        date_parts.append(df_full.loc[m, 'date'].to_numpy())
        code_parts.append(df_full.loc[m, 'Code'].to_numpy())
        if (di + 1) % 250 == 0:
            print(f'[V22] 预处理 {di + 1}/{len(dates_all)} 天', flush=True)
        del sub, X, mu, sd
    del df_full
    n_rows = sum(len(p) for p in X_parts)
    X_all = np.empty((n_rows, nf), dtype=np.float32)
    d_all = np.empty(n_rows, dtype=object)
    c_all = np.empty(n_rows, dtype=object)
    pos = 0
    for xp, dp, cp in zip(X_parts, date_parts, code_parts):
        k = len(xp)
        X_all[pos:pos + k] = xp
        d_all[pos:pos + k] = dp
        c_all[pos:pos + k] = cp
        pos += k
    del X_parts, date_parts, code_parts
    print(f'[V22] 特征矩阵 {X_all.shape} ({X_all.nbytes / 1e9:.1f}GB)', flush=True)
    buy = pd.read_feather(PROJECT_ROOT + 'trainingdata/buyable_mask.fea') \
        .set_index('date').reset_index().melt(id_vars='date', var_name='Code',
                                              value_name='buyable')
    buy['date'] = buy['date'].astype(str)
    key = pd.DataFrame({'date': d_all, 'Code': c_all}).merge(
        buy, on=['date', 'Code'], how='left')
    buyable = key['buyable'].fillna(False).to_numpy(bool)
    del buy, key
    return X_all, d_all, c_all, buyable, factors


def build_label(d_all, c_all, label_name):
    """逐日 winsor + 秩高斯标签 (与 V18 一致的修复版写法)。"""
    ret = pd.read_feather(PROJECT_ROOT + f'trainingdata/{label_name}.fea') \
        .set_index('index').reset_index().melt(id_vars='index', var_name='Code',
                                               value_name='label')
    ret.columns = ['date', 'Code', 'label']
    ret['date'] = ret['date'].astype(str)
    key = pd.DataFrame({'date': d_all, 'Code': c_all}).merge(
        ret, on=['date', 'Code'], how='left')
    label = key['label'].to_numpy(np.float32)
    del ret, key
    n_rows = len(d_all)
    label_rg = np.full(n_rows, np.nan, dtype=np.float32)
    tw = np.empty(n_rows, dtype=np.float32)
    dates_u = pd.unique(d_all)
    for i in range(0, len(dates_u), BATCH_DAYS):
        bd = dates_u[i:i + BATCH_DAYS]
        bm = np.isin(d_all, bd)
        tw[bm] = [time_weight(d) for d in d_all[bm]]
        for d in bd:
            dm = d_all == d
            v = label[dm]
            vv = v[~np.isnan(v)]
            if len(vv) < 50:
                continue
            med = np.median(vv)
            mad = np.median(np.abs(vv - med))
            if mad <= 1e-12:
                w = v
            else:
                w = np.clip(v, med - LABEL_WINSOR_MAD * mad,
                            med + LABEL_WINSOR_MAD * mad)
            ok = ~np.isnan(v)
            rg = pd.Series(w[ok]).rank().to_numpy(np.float64)
            rg = ((rg - rg.mean()) / rg.std()).astype(np.float32)
            pos = np.flatnonzero(dm)[ok]   # ok 相对当日行, 先取当日全局位置
            label_rg[pos] = rg
        print(f'[V22] {label_name} 标签 {min(i + BATCH_DAYS, len(dates_u))}/{len(dates_u)} 天',
              flush=True)
    return label_rg, tw


def train_one(X_all, d_all, c_all, buyable, label_rg, tw, label_name, fold_mats):
    valid_mask = buyable & ~np.isnan(label_rg)
    print(f'[V22] {label_name} 有效行 {valid_mask.sum()}/{len(d_all)}', flush=True)
    date_list = sorted(set(d_all[valid_mask]))
    train_allowed = [d for d in date_list if d < TRAIN_END]
    n = len(train_allowed)
    valid_pool = train_allowed[n - VALID_DAYS:]
    random.seed(SEED)
    random.shuffle(valid_pool)
    base, rem = divmod(len(valid_pool), N_FOLDS)
    test_dates = [d for d in date_list if TEST_START <= d <= TEST_END]
    for fold in range(1, N_FOLDS + 1):
        out_path = f'{ROOT}/model_pred/v22_{label_name}_f{fold}.fea'
        if os.path.exists(out_path):
            mat = pd.read_feather(out_path).set_index('date')
            fold_mats[fold] = mat
            print(f'[V22] {label_name} fold{fold} 已存在, 跳过 (resume)', flush=True)
            continue
        f = fold - 1
        start = f * base + min(f, rem)
        size = base + (1 if f < rem else 0)
        valid_dates = set(valid_pool[start:start + size])
        train_dates = set(train_allowed[:n - VALID_DAYS - PURGE_DAYS])
        tr_m = valid_mask & np.isin(d_all, list(train_dates))
        va_m = valid_mask & np.isin(d_all, list(valid_dates))
        print(f'[V22] {label_name} fold{fold}: train {tr_m.sum()} / valid {va_m.sum()}',
              flush=True)
        model = lgb.LGBMRegressor(
            objective='l2', n_estimators=3000, learning_rate=0.05,
            num_leaves=63, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
            n_jobs=N_JOBS, random_state=SEED + fold, verbosity=-1)
        model.fit(X_all[tr_m], label_rg[tr_m], sample_weight=tw[tr_m],
                  eval_set=[(X_all[va_m], label_rg[va_m])],
                  eval_sample_weight=[tw[va_m]],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        print(f'[V22] {label_name} fold{fold}: best_iter {model.best_iteration_}, '
              f'valid l2 {model.best_score_["valid_0"]["l2"]:.5f}', flush=True)
        te_m = valid_mask & np.isin(d_all, test_dates)
        pred = model.predict(X_all[te_m], num_iteration=model.best_iteration_)
        te = pd.DataFrame({'date': d_all[te_m], 'Code': c_all[te_m], 'pred': pred})
        te['z'] = te.groupby('date')['pred'].transform(
            lambda s: (s - s.mean()) / s.std())
        mat = te.pivot(index='date', columns='Code', values='z').sort_index()
        mat.index.name = 'date'
        mat.reset_index().to_feather(out_path)
        fold_mats[fold] = mat
        print(f'[V22] {label_name} fold{fold} 推演 {mat.shape[0]} 天落盘', flush=True)

    total = None
    for f, mat in fold_mats.items():
        total = mat if total is None else total.add(mat, fill_value=0.0)
    total.index.name = 'date'
    total.reset_index().to_feather(f'{ROOT}/model_pred/v22_{label_name}.fea')
    print(f'[V22] 集成 v22_{label_name}.fea: {total.shape} — DONE', flush=True)


def main():
    os.makedirs(ROOT + '/model_pred', exist_ok=True)
    X_all, d_all, c_all, buyable, factors = preprocess()
    for label_name in ('label_ret_10d', 'label_ret_20d'):
        label_rg, tw = build_label(d_all, c_all, label_name)
        train_one(X_all, d_all, c_all, buyable, label_rg, tw, label_name, {})
        del label_rg, tw
    print('[V22] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
