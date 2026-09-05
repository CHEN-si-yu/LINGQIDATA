#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gbdt.py — V18: LightGBM 排序器组件 (内存安全版, 90GB cgroup 上限内单任务运行)

与 V11/V13 神经网络完全同口径:
- 特征: 848 因子, 逐日截面 rank → 标准化 → 缺失填 0 (float32)
- 标签: 1d 收益 winsor(MAD5) → 逐日截面秩高斯化
- 划分: 同 model.py 4 折 (seed 3253), train < 20250809, valid=最后120天, purge 5
- 权重: 时间衰减 hl=600d
- 分批 (150 交易日/批) 预处理 → 预分配 float32 矩阵, 峰值内存 ~25GB
- 输出: Model/V18/model_pred/v18_f{1..4}.fea + v18.fea (跨折 z 求和)
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
ROOT = PROJECT_ROOT + "Model/V18"
TEST_START, TEST_END = '20250901', '20260901'
TRAIN_END = '20250809'
VALID_DAYS, N_FOLDS, PURGE_DAYS = 120, 4, 5
LABEL_WINSOR_MAD = 5.0
HALF_LIFE = 600
SEED = 3253
BATCH_DAYS = 150


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


def main():
    os.makedirs(ROOT + '/model_pred', exist_ok=True)
    factors = load_feature_map(PROJECT_ROOT + 'Model/V11/model_test/feature_map.fea')
    nf = len(factors)
    print(f'[V18] {nf} factors', flush=True)

    # ── 1) 日期列表 (轻量列读取) ──
    print('[V18] 统计日期 ...', flush=True)
    import pyarrow.feather as pf
    dates_all = sorted(pf.read_table(PROJECT_ROOT + 'trainingdata/fac_all.fea',
                                     columns=['date'])['date']
                       .to_pandas().astype(str).unique())
    print(f'[V18] {len(dates_all)} 个日期', flush=True)

    # ── 2) 一次性读入 (float32) + 逐日预处理 (与 NN normed_data 严格一致) ──
    # 口径: 每日截面内按因子列 rank(axis=0) → 列均值/标准差标准化 → NaN 填 0
    print('[V18] 读入 fac_all (float32) ...', flush=True)
    df_full = pd.read_feather(PROJECT_ROOT + 'trainingdata/fac_all.fea',
                              columns=['date', 'Code'] + factors)
    df_full['date'] = df_full['date'].astype(str)
    df_full[factors] = df_full[factors].astype(np.float32)
    print(f'[V18] {df_full.shape[0]} rows 已读入', flush=True)
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
            print(f'[V18] 预处理 {di + 1}/{len(dates_all)} 天', flush=True)
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
    print(f'[V18] 特征矩阵 {X_all.shape} ({X_all.nbytes / 1e9:.1f}GB)', flush=True)

    # ── 3) 标签 + buyable (长表) ──
    ret = pd.read_feather(PROJECT_ROOT + 'trainingdata/label_ret_1d.fea').set_index('index')
    buy = pd.read_feather(PROJECT_ROOT + 'trainingdata/buyable_mask.fea').set_index('date')
    ret_long = ret.reset_index().melt(id_vars='index', var_name='Code',
                                      value_name='label')
    ret_long.columns = ['date', 'Code', 'label']
    buy_long = buy.reset_index().melt(id_vars='date', var_name='Code',
                                      value_name='buyable')
    del ret, buy
    ret_long['date'] = ret_long['date'].astype(str)
    buy_long['date'] = buy_long['date'].astype(str)
    key = pd.DataFrame({'date': d_all, 'Code': c_all})
    key = key.merge(ret_long, on=['date', 'Code'], how='left')
    key = key.merge(buy_long, on=['date', 'Code'], how='left')
    del ret_long, buy_long
    label = key['label'].to_numpy(np.float32)
    buyable = key['buyable'].fillna(False).to_numpy(bool)
    del key

    # ── 4) 逐日 winsor + 秩高斯 (分批掩码) ──
    dates_u = pd.unique(d_all)
    label_rg = np.full(n_rows, np.nan, dtype=np.float32)
    tw = np.empty(n_rows, dtype=np.float32)
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
            pos = np.flatnonzero(dm)[ok]   # ok 相对当日行, 先取当日全局位置 (坑11: 链式索引静默丢弃)
            label_rg[pos] = rg
        print(f'[V18] 标签 {min(i + BATCH_DAYS, len(dates_u))}/{len(dates_u)} 天',
              flush=True)
    print('[V18] 标签处理完成', flush=True)

    # ── 5) 有效行掩码 ──
    valid_mask = buyable & ~np.isnan(label_rg)
    print(f'[V18] 有效行 {valid_mask.sum()}/{n_rows}', flush=True)

    # ── 6) 折划分 (与 model.py 一致) ──
    date_list = sorted(set(d_all[valid_mask]))
    train_allowed = [d for d in date_list if d < TRAIN_END]
    n = len(train_allowed)
    valid_pool = train_allowed[n - VALID_DAYS:]
    random.seed(SEED)
    random.shuffle(valid_pool)
    base, rem = divmod(len(valid_pool), N_FOLDS)
    test_dates = [d for d in date_list if TEST_START <= d <= TEST_END]

    fold_mats = {}
    for fold in range(1, N_FOLDS + 1):
        out_path = f'{ROOT}/model_pred/v18_f{fold}.fea'
        if os.path.exists(out_path):
            mat = pd.read_feather(out_path).set_index('date')
            fold_mats[fold] = mat
            print(f'[V18] fold{fold} 已存在, 跳过 (resume)', flush=True)
            continue
        f = fold - 1
        start = f * base + min(f, rem)
        size = base + (1 if f < rem else 0)
        valid_dates = set(valid_pool[start:start + size])
        train_dates = set(train_allowed[:n - VALID_DAYS - PURGE_DAYS])
        tr_m = valid_mask & np.isin(d_all, list(train_dates))
        va_m = valid_mask & np.isin(d_all, list(valid_dates))
        print(f'[V18] fold{fold}: train {tr_m.sum()} / valid {va_m.sum()}', flush=True)
        model = lgb.LGBMRegressor(
            objective='l2', n_estimators=3000, learning_rate=0.05,
            num_leaves=63, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
            n_jobs=20, random_state=SEED + fold, verbosity=-1)
        model.fit(X_all[tr_m], label_rg[tr_m], sample_weight=tw[tr_m],
                  eval_set=[(X_all[va_m], label_rg[va_m])],
                  eval_sample_weight=[tw[va_m]],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        print(f'[V18] fold{fold}: best_iter {model.best_iteration_}, '
              f'valid l2 {model.best_score_["valid_0"]["l2"]:.5f}', flush=True)
        # 推演 Test 区间
        te_m = valid_mask & np.isin(d_all, test_dates)
        pred = model.predict(X_all[te_m], num_iteration=model.best_iteration_)
        te = pd.DataFrame({'date': d_all[te_m], 'Code': c_all[te_m],
                           'pred': pred})
        te['z'] = te.groupby('date')['pred'].transform(
            lambda s: (s - s.mean()) / s.std())
        mat = te.pivot(index='date', columns='Code', values='z').sort_index()
        mat.index.name = 'date'
        mat.reset_index().to_feather(f'{ROOT}/model_pred/v18_f{fold}.fea')
        fold_mats[fold] = mat
        print(f'[V18] fold{fold} 推演 {mat.shape[0]} 天落盘', flush=True)

    total = None
    for f, mat in fold_mats.items():
        total = mat if total is None else total.add(mat, fill_value=0.0)
    total.index.name = 'date'
    total.reset_index().to_feather(f'{ROOT}/model_pred/v18.fea')
    print(f'[V18] 集成 v18.fea: {total.shape} — DONE', flush=True)


if __name__ == '__main__':
    main()
