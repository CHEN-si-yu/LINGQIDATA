#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v19.py — V19 堆叠元模型 (重启完成版, 2026-09-06)

背景 (FINAL_REPORT §四 / TRAINING_PLAYBOOK §7b):
  意图: 以基模型 (V11/V13, 各 8 折) 的头部打分作元特征, 训练轻量堆叠元模型。
  旧版问题: 元特征窗口 (20250501~20250808) 落在各折 valid 池内, 对部分折是
    "早停选定"的验证集 → 直接训练存在泄漏。
  本版修复 (真 OOF): 元训练窗口扩至完整 valid_pool (TRAIN_END 前 120 天,
    20250217~20250808); 每日期只使用【其所属 split 之外】的 6 折均值作特征
    (该 6 折从未见过该日期: 不在其 train, 也不在其 valid/早停窗口)。

阶段:
  1) --stage oof     V11/V13 各 8 折 best ckpt 推演 valid_pool 全部 120 天,
                     r1/r3/r5/top 头逐日 z → meta_oof/{fam}{f}_{h}.fea
  2) --stage meta    真 OOF 特征组装 → Ridge 元模型 (5d 秩高斯标签, 时间衰减,
                     可买池) → Test 窗 (V11/V13 heads 8 折均值) 推演
                     → model_pred/2026q3/score_meta.fea
  3) --stage report  IC + 5 协议真实净值 + 图 → model_pic/output.md + figure_01.png

用法: python3 run_v19.py --stage oof|meta|report   (或 --stage all)
"""
import argparse
import glob
import importlib.util
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
HERE = PROJECT_ROOT + "Model/V19"
V11_DIR = PROJECT_ROOT + "Model/V11"
V13_DIR = PROJECT_ROOT + "Model/V13"
OOF_DIR = HERE + "/meta_oof"
PRED_DIR = HERE + "/model_pred/2026q3"
TRAIN_END = '20250809'
VALID_DAYS = 120
SEED = 3253
N_FOLDS = 4
FAMILIES = [('a', V11_DIR), ('c', V13_DIR)]
HEADS = ['r1', 'r3', 'r5', 'top']
HALF_LIFE = 600.0
LABEL_WINSOR_MAD = 5.0


# ---------------------------------------------------------------- 工具
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


def time_weight(date_str, ref=TRAIN_END):
    from datetime import datetime
    d = datetime.strptime(date_str, '%Y%m%d')
    r = datetime.strptime(ref, '%Y%m%d')
    return float(np.exp(-(r - d).days / HALF_LIFE))


def zscore_row(s):
    s = s.astype(float)
    sd = s.std()
    if not np.isfinite(sd) or sd <= 1e-12:
        return s * 0.0
    return (s - s.mean()) / sd


def valid_pool_and_splits():
    """复刻 V11 model.py get_date_splits 的 valid_pool 与 4 折 split 成员表。"""
    lab = pd.read_feather(PROJECT_ROOT + 'trainingdata/label_ret_1d.fea',
                          columns=['index'])
    labd = set(str(d) for d in lab['index'])
    fac = pd.read_feather(PROJECT_ROOT + 'trainingdata/fac_all.fea',
                          columns=['date'])
    facd = sorted({str(d) for d in fac['date'].unique()})
    train_allowed = [d for d in facd if d < TRAIN_END and d in labd]
    n = len(train_allowed)
    valid_pool = train_allowed[n - VALID_DAYS:]
    import random
    random.seed(SEED)
    random.shuffle(valid_pool)
    base, rem = divmod(len(valid_pool), N_FOLDS)
    split_of = {}
    splits = {}
    for f in range(N_FOLDS):
        start = f * base + min(f, rem)
        size = base + (1 if f < rem else 0)
        chunk = sorted(valid_pool[start:start + size])
        splits[f + 1] = chunk
        for d in chunk:
            split_of[d] = f + 1
    return sorted(valid_pool), splits, split_of


# ---------------------------------------------------------------- stage oof
def stage_oof():
    os.makedirs(OOF_DIR, exist_ok=True)
    pool, _, _ = valid_pool_and_splits()
    print(f'[oof] valid_pool: {len(pool)} 天 {pool[0]}~{pool[-1]}', flush=True)

    import torch
    import pyarrow as pa
    import pyarrow.feather as pf
    import pyarrow.compute as pc
    torch.set_num_threads(min(12, os.cpu_count() or 8))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    factors = load_feature_map(V11_DIR + '/model_test/feature_map.fea')
    fac_path = PROJECT_ROOT + 'trainingdata/fac_all.fea'
    all_cols = pf.read_table(fac_path, columns=[]).column_names
    available = [f for f in factors if f in all_cols]
    cols = ['date', 'Code'] + available
    table = pf.read_table(fac_path, columns=cols)
    dates_all = pc.unique(table.column('date')).to_pandas().astype(str).sort_values()
    dates = [d for d in dates_all if d in set(pool)]
    mask = pc.is_in(table.column('date'), pa.array(dates))
    data = table.filter(mask).to_pandas().set_index('date').sort_index().reset_index()
    del table
    print(f'[oof] 因子 {len(dates)} 天加载完成, {len(available)} 因子', flush=True)

    def normed(dd):
        d = dd.copy()
        d = d.dropna(subset=factors, thresh=max(1, int(0.1 * len(factors))))
        X = d[factors].rank(axis=0)
        X = ((X - X.mean()) / X.std()).fillna(0)
        return torch.from_numpy(np.nan_to_num(
            X.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)), \
            d['Code'].values

    for fam, root in FAMILIES:
        mod = load_model_module(root + '/model.py', f'v19_{fam}')
        for f in range(1, 9):
            fd = f'{root}/model_train/2026q3/fold{f}'
            ck = find_best_ckpt(fd)
            m = mod.PredictModel(input_dim=len(factors)).to(dev)
            st = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
            st = {k.removeprefix('model.'): v for k, v in st.items()
                  if k.startswith('model.')}
            m.load_state_dict(st, strict=False)
            m.eval()
            rows = {h: [] for h in HEADS}
            with torch.no_grad():
                for date in dates:
                    dd = data[data['date'] == date]
                    X, codes = normed(dd)
                    out = m(X.to(dev))
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
                    for h in HEADS:
                        if h in hvals:
                            v = hvals[h].detach().cpu().numpy().ravel()
                            sd = v.std()
                            z = ((v - v.mean()) / sd) if np.isfinite(sd) \
                                and sd > 1e-12 else v * 0.0
                            rows[h].append(pd.Series(z, index=codes, name=date))
            for h in HEADS:
                df = pd.DataFrame(rows[h])
                df.index.name = 'date'
                df.reset_index().to_feather(
                    os.path.join(OOF_DIR, f'{fam}{f}_{h}.fea'))
            print(f'[oof] {fam}{f} ({os.path.basename(ck)}) 落盘', flush=True)
    print('[oof] DONE → meta_oof/', flush=True)


# ---------------------------------------------------------------- stage meta
def zscore_df(g):
    """Series/DataFrame 逐列 z (0 方差列 → 0)。"""
    g = g.astype(float)
    sd = g.std()
    if isinstance(sd, (float, np.floating)):
        if np.isfinite(sd) and sd > 1e-12:
            return (g - g.mean()) / sd
        return g * 0.0
    return (g - g.mean()) / sd.where(sd > 1e-12, 1.0)


def build_oof_features(pool, split_of):
    """真 OOF 特征: 每日期每头 = 其 split 之外的 6 折 (每族) 逐日 z 均值 → 8 列。"""
    frames = {}
    for fam, _ in FAMILIES:
        for f in range(1, 9):
            for h in HEADS:
                p = os.path.join(OOF_DIR, f'{fam}{f}_{h}.fea')
                if os.path.exists(p):
                    frames[(fam, f, h)] = pd.read_feather(p).set_index('date')
    parts = []
    for date in pool:
        s = split_of[date]
        row = {}
        for fam, _ in FAMILIES:
            for h in HEADS:
                fsub = [frames[(fam, f, h)].loc[date]
                        for f in range(1, 9)
                        if (f - 1) % N_FOLDS + 1 != s
                        and (fam, f, h) in frames and date in frames[(fam, f, h)].index]
                if not fsub:
                    row[f'{fam}_{h}'] = pd.Series(dtype=float)
                    continue
                m = pd.concat(fsub, axis=1).mean(axis=1)
                row[f'{fam}_{h}'] = m
        df = pd.DataFrame(row)
        df.index.name = 'Code'
        df['date'] = date
        parts.append(df.reset_index())
    X = pd.concat(parts, ignore_index=True)
    # 逐日每列 z (对齐量纲)
    X = X.set_index(['date', 'Code'])
    X = X.groupby('date').transform(zscore_df)
    return X.reset_index()


def build_label(pool):
    ret = pd.read_feather(PROJECT_ROOT + 'trainingdata/label_ret_5d.fea') \
        .set_index('index')
    buy = pd.read_feather(PROJECT_ROOT + 'trainingdata/buyable_mask.fea') \
        .set_index('date')
    rows = []
    for d in pool:
        y = ret.loc[d].astype(float) if d in ret.index else None
        if y is None:
            continue
        b = buy.loc[d].astype(float) if d in buy.index else None
        v = y.dropna()
        if len(v) < 50:
            continue
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        w = np.clip(v, med - LABEL_WINSOR_MAD * mad,
                    med + LABEL_WINSOR_MAD * mad) if mad > 1e-12 else v
        rg = w.rank()
        rg = ((rg - rg.mean()) / rg.std()).astype(np.float32)
        if b is not None:
            rg = rg[b.gt(0.5)]
        rows.append(rg.rename(d))
    lab = pd.DataFrame(rows).stack().rename('label').reset_index()
    lab.columns = ['date', 'Code', 'label']
    lab['date'] = lab['date'].astype(str)
    lab['Code'] = lab['Code'].astype(str)
    return lab


def stage_meta():
    pool, _, split_of = valid_pool_and_splits()
    print(f'[meta] 组装 OOF 特征 ({len(pool)} 天 × 8 列)', flush=True)
    X = build_oof_features(pool, split_of)
    lab = build_label(pool)
    M = X.merge(lab, on=['date', 'Code'], how='inner')
    M = M.dropna()
    tw = M['date'].map(time_weight).astype(np.float32)
    feats = [f'{fam}_{h}' for fam, _ in FAMILIES for h in HEADS]
    print(f'[meta] 训练矩阵 {M.shape}, 特征 {feats}', flush=True)

    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    # 时间序 holdout 选 alpha: 前 100 天训练 / 后 20 天验证
    dates_u = sorted(M['date'].unique())
    cut = dates_u[100]
    tr = M['date'] <= cut
    best_a, best_r2 = 1.0, -1e9
    for a in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        mm = Ridge(alpha=a)
        mm.fit(M.loc[tr, feats], M.loc[tr, 'label'],
               sample_weight=tw[tr.to_numpy()])
        p = mm.predict(M.loc[~tr, feats])
        r2 = r2_score(M.loc[~tr, 'label'], p)
        print(f'[meta] alpha={a}: holdout(后{len(dates_u)-100}天) r2={r2:.4f}',
              flush=True)
        if r2 > best_r2:
            best_a, best_r2 = a, r2
    print(f'[meta] 选择 alpha={best_a} (holdout r2={best_r2:.4f})', flush=True)
    model = Ridge(alpha=best_a)
    model.fit(M[feats], M['label'], sample_weight=tw)
    ins = model.predict(M[feats])
    print(f'[meta] 全量训练 r2={r2_score(M["label"], ins):.4f}  '
          f'(weighted {np.average((M["label"]-ins)**2, weights=tw):.4f})', flush=True)
    pd.DataFrame({'feat': feats, 'coef': model.coef_}).to_csv(
        HERE + '/meta_coefs.csv', index=False)
    print('[meta] coefs →', pd.Series(model.coef_, index=feats).round(4).to_dict(),
          flush=True)
    import joblib
    joblib.dump(model, HERE + '/meta_model.joblib')

    # ---- Test 窗特征: V11/V13 heads 8 折均值 (Test 对全部折均为 OOS) ----
    print('[meta] 组装 Test 特征 (V11/V13 heads 8 折均值)', flush=True)
    test_dates = None
    parts = []
    for fam, root in FAMILIES:
        hdir = root + '/model_pred/2026q3/heads'
        for h in HEADS:
            frames = []
            for f in range(1, 9):
                p = os.path.join(hdir, f'{h}_f{f}.fea')
                if not os.path.exists(p):
                    continue
                fr = pd.read_feather(p).set_index('date')
                if test_dates is None:
                    test_dates = sorted(fr.index)
                fr = fr.loc[fr.index <= '20260901']
                fr = fr.apply(zscore_row, axis=1)
                frames.append(fr)
            m = pd.concat(frames).groupby('date').mean()
            st = m.stack().rename(f'{fam}_{h}').reset_index()
            st.columns = ['date', 'Code', f'{fam}_{h}']
            parts.append(st)
    TX = parts[0]
    for p in parts[1:]:
        TX = TX.merge(p, on=['date', 'Code'], how='outer')
    TX = TX.sort_values(['date', 'Code']).reset_index(drop=True)
    TX_dates = sorted(TX['date'].unique())
    print(f'[meta] Test 特征 {TX.shape}, {len(TX_dates)} 天', flush=True)

    Xt = TX[feats].fillna(0.0)
    # 逐日列 z (与训练一致)
    Xt = TX.groupby('date')[feats].transform(zscore_df).fillna(0.0)
    pred = model.predict(Xt.to_numpy())
    TX['score'] = pred
    mat = TX.pivot_table(index='date', columns='Code', values='score')
    mat = mat.apply(zscore_row, axis=1)
    mat = mat.sort_index()
    mat.index.name = 'date'
    os.makedirs(PRED_DIR, exist_ok=True)
    mat.reset_index().to_feather(PRED_DIR + '/score_meta.fea')
    print(f'[meta] → {PRED_DIR}/score_meta.fea {mat.shape}', flush=True)

    # 快速 IC 检查 (Test 5d)
    ret5 = pd.read_feather(PROJECT_ROOT + 'trainingdata/label_ret_5d.fea') \
        .set_index('index')
    buy = pd.read_feather(PROJECT_ROOT + 'trainingdata/buyable_mask.fea') \
        .set_index('date')
    ics = []
    for d in mat.index:
        if d not in ret5.index:
            continue
        s = mat.loc[d]
        y = ret5.loc[d].astype(float).reindex(s.index)
        b = buy.loc[d].astype(float) if d in buy.index else None
        tr = y.notna() & (b.gt(0.5) if b is not None else True)
        if tr.sum() < 50:
            continue
        ics.append(s[tr].rank().corr(y[tr].rank()))
    print(f'[meta] Test 5d RankIC = {np.mean(ics):.4f} ({len(ics)} 天)', flush=True)


# ---------------------------------------------------------------- stage report
def stage_report():
    sys.path.insert(0, HERE)
    sys.path.insert(0, PROJECT_ROOT + 'Model/Trading')
    from engine import load_calendar, load_market, run_backtest

    SETS = {
        "V19_meta": HERE + "/model_pred/2026q3/score_meta.fea",
        "V11_ensw2": V11_DIR + "/model_pred/2026q3/score_ens_w2.fea",
        "V20b_ensw2": PROJECT_ROOT + "Model/V20b/model_pred/2026q3/score_ens_w2.fea",
    }
    LABEL_FILES = {"1d": "label_ret_1d", "3d": "label_ret_3d", "5d": "label_ret_5d",
                   "10d": "label_ret_10d", "20d": "label_ret_20d"}
    PROTOCOLS = {
        "D01_h5s8_t1": dict(top_n=1, hold=5, stop_loss=0.08),
        "hold20_t1": dict(top_n=1, hold=20),
        "hold20_t2": dict(top_n=2, hold=20),
        "h20tr20_t2": dict(top_n=2, hold=20, trail_pct=0.20),
        "S2_re300": dict(top_n=2, hold=None, exit_rank=300, min_hold=2,
                         stop_loss=0.08, trail_pct=0.15),
    }
    buy = pd.read_feather(PROJECT_ROOT + 'trainingdata/buyable_mask.fea').set_index('date')
    sets = {n: pd.read_feather(p).set_index('date') for n, p in SETS.items()}
    ic_rows = []
    for n, score in sets.items():
        row = {}
        for h, fn in LABEL_FILES.items():
            lab = pd.read_feather(PROJECT_ROOT + f'trainingdata/{fn}.fea').set_index('index')
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

    mkt = load_market()
    tds, tdi = load_calendar()
    proto_rows = []
    main_eq, main_tr = None, None
    for n, score in sets.items():
        sc = dict(scores={d: score.loc[d].dropna().to_dict() for d in score.index},
                  ranked={d: score.loc[d].dropna().sort_values(ascending=False)
                          .index.tolist() for d in score.index},
                  top1={d: float(score.loc[d].max()) for d in score.index},
                  dates=sorted(score.index))
        for pn, pc in PROTOCOLS.items():
            m, tr, eq = run_backtest(sc, pc, mkt, tds, tdi)
            if m is None:
                continue
            proto_rows.append(dict(set=n, proto=pn, cum_net=m['cum_net'],
                                   sharpe=m['sharpe'], maxdd=m['maxdd'],
                                   h1=m['h1_cum'], h2=m['h2_cum'],
                                   n=m['n_trades'], win=m['win_rate']))
            if n == "V19_meta" and pn == "h20tr20_t2":
                main_eq, main_tr = eq, tr
    pv = pd.DataFrame(proto_rows).pivot_table(
        index='proto', columns='set', values='cum_net').round(3)

    text = ['# V19 报告 — 堆叠元模型 (真 OOF 重启完成版)',
            '',
            '> 日期: 2026-09-06 | 单元: Model/V19 (run_v19.py)',
            '> 设计: 基模型 V11/V13 各 8 折 的 r1/r3/r5/top 头打分作元特征;',
            '> 元训练窗口 = valid_pool 20250217~20250808 (120 天);',
            '> 真 OOF: 每日期仅用其 split 之外的 6 折均值 (该 6 折从未见过该日),',
            '> 规避早停泄漏 (TRAINING_PLAYBOOK §7b)。',
            '> 元模型: Ridge (5d 秩高斯标签, 时间衰减 hl=600d, 可买池)。',
            '> Test 推演: V11/V13 heads 8 折均值 (Test 对全部折 OOS)。',
            '']
    text.append('=' * 80)
    text.append('  一、IC 层 (RankIC, 可交易池, Test 窗口)')
    text.append('=' * 80)
    text.append(ic_df.round(4).to_string())
    text.append('')
    text.append('=' * 80)
    text.append('  二、策略层真实净值矩阵 (Trading 引擎 cum_net, 1 份资金, 含成本)')
    text.append('=' * 80)
    text.append(pv.to_string())
    text.append('')
    text.append('=' * 55)
    text.append('  三、V19_meta h20tr20_t2 明细')
    text.append('=' * 55)
    if main_eq is not None:
        text.append(f'  净值: {main_eq["equity"].iloc[-1]/50000-1:+.2%}  |  '
                    f'MaxDD: {(main_eq["equity"]/main_eq["equity"].cummax()-1).min():+.2%}'
                    f'  |  交易 {len(main_tr)}')
    report = '\n'.join(text)
    sys.stdout.write(report + '\n')
    os.makedirs(HERE + '/model_pic', exist_ok=True)
    with open(HERE + '/model_pic/output.md', 'w', encoding='utf-8') as f:
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
            axes[0].set_title('V19_meta h20tr20_t2 True Equity', fontsize=13)
            axes[0].set_xlabel('Trading day'); axes[0].set_ylabel('Cum net')
            axes[0].grid(alpha=0.3)
            net = main_tr['net_pct'] / 100.0
            axes[1].bar(range(len(net)), net.values,
                        color=['#2ca02c' if v >= 0 else '#d62728' for v in net])
            axes[1].axhline(0, color='gray', lw=0.8, ls='--')
            axes[1].set_title(f'Per-Trade Net ({len(net)})', fontsize=13)
            axes[1].grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(HERE + '/model_pic/figure_01.png', dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        print(f'[report] 出图跳过: {e}')
    print(f'[report] → {HERE}/model_pic/output.md (+figure_01.png)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['oof', 'meta', 'report', 'all'])
    args = ap.parse_args()
    if args.stage in ('oof', 'all'):
        stage_oof()
    if args.stage in ('meta', 'all'):
        stage_meta()
    if args.stage in ('report', 'all'):
        stage_report()
