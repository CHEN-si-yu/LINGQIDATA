#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_epoch.py — 诊断「epoch=0 即最佳」现象 (用户提问 2026-09-05)

问题: 很多 fold 在 epoch=0 达到 val_rankic 最佳 (V20 9/16 折)。静态分析发现
val_top10_ret 的峰值普遍晚于 val_rankic (fold12: rankic@0 而 top_ret@7) —
epoch0 保存可能让顶部分支欠训练。

实验: 用 V20 model.py 训练 fold1 (fam a, cons, seed 4253), **保存全部 epoch
checkpoint** (save_top_k=-1, 无早停, 20 epoch), 然后对每个 epoch:
  A. val_rankic / val_top10_ret (验证期)
  B. Test 期单折打分 z(lin_1d)+z(top) 的策略层回测 (D01 与 h20tr20_t2, Trading 引擎)
  C. top 分支单独打分回测
输出: results/diag_epoch.csv
"""
import glob
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

PROJECT = "/autodl-fs/data/lingqiData/"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT + "Model/V20")


def load_v20_model():
    spec = importlib.util.spec_from_file_location("v20m", PROJECT + "Model/V20/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def train_fold_all_ckpts():
    """训练 V20 fold1 (fam a, cons), 保存全部 epoch; 返回 ckpt 列表。"""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    m = load_v20_model()
    out_dir = "/tmp/v20_epochdiag"
    os.system(f"rm -rf {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    # 打补丁: 保存全部 epoch, 去掉早停, 20 epoch
    import pytorch_lightning.callbacks as cb
    orig = m.DLLitModule.configure_callbacks
    m.DLLitModule.configure_callbacks = lambda self: [
        ModelCheckpoint(dirpath=out_dir, filename='{epoch}-{val_rankic:.4f}',
                        monitor='val_rankic', mode='max', save_top_k=-1,
                        save_on_train_epoch_end=True, every_n_epochs=1)]
    args = m.parse_args()
    args.epochs = 20
    # fold 1 → fam a, k=1, cons
    params_mod = m.params
    params_mod.all_data = pd.read_feather(PROJECT + "trainingdata/fac_all.fea")
    date_list = sorted(set(params_mod.all_data['date'].unique())
                       & set(params_mod.ret_data.index))
    params_mod.all_data = params_mod.all_data.set_index('date').sort_index()
    params_mod.factor_list = [c for c in params_mod.all_data.columns[1:]]
    params_mod.factor_num = len(params_mod.factor_list)
    m.seed_everything(args.seed + 1000)
    lit = m.DLLitModule(args, style='cons', fam='a')
    tr, va, te = m.get_date_splits(date_list, fold=1)
    dm = m.DLDataModule(args, tr, va)
    trainer = pl.Trainer(max_epochs=20, accelerator='gpu' if torch.cuda.is_available()
                         else 'cpu', devices=1, callbacks=lit.configure_callbacks(),
                         num_sanity_val_steps=0, enable_progress_bar=True,
                         log_every_n_steps=0, gradient_clip_val=1.0)
    trainer.fit(lit, dm)
    ckpts = sorted(glob.glob(out_dir + "/*.ckpt"),
                   key=lambda p: int(p.rsplit('-', 1)[0].rsplit('/', 1)[1]
                                     .split('-')[0]))
    return ckpts, m


def eval_ckpt(ck_path, m, market, tds, tdi, protos=(('D01', dict(top_n=1, hold=5, stop_loss=0.08)),
                                                 ('h20', dict(top_n=2, hold=20, trail_pct=0.20)))):
    """单折打分 → 策略层回测 (D01 + h20tr20_t2) + IC。"""
    from engine import run_backtest, buy_cost
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    state = {k.removeprefix('model.'): v for k, v in ck['state_dict'].items()
             if k.startswith('model.')}
    model = m.PredictModel(input_dim=len(m.params.factor_list))
    model.load_state_dict(state, strict=False)
    model.eval()
    factor_cols = m.params.factor_list
    data = m.params.all_data
    tdates = [d for d in data.index if '20250901' <= str(d) <= '20260901']
    rows = []
    for d in tdates:
        day = data.loc[d]
        if isinstance(day, pd.Series):
            day = day.to_frame().T
        # 与 analysis 一致的归一化
        valid_threshold = max(1, int(0.1 * len(factor_cols)))
        day = day.dropna(subset=factor_cols, thresh=valid_threshold).copy()
        X = day[factor_cols].rank(axis=0)
        X = ((X - X.mean()) / X.std()).fillna(0)
        x = torch.from_numpy(np.nan_to_num(X.to_numpy(dtype=np.float32)))
        with torch.no_grad():
            out_t = model(x)
        if len(out_t) >= 7:
            mixed, r1, r5, r3, top, r10, r20 = out_t
        else:
            mixed, r1, r5, r3, top = out_t
        z = lambda v: (v - v.mean()) / v.std()
        sm = z(mixed.squeeze(1)).numpy()
        st = z(top.squeeze(1)).numpy()
        rows.append((d, list(day['Code'].values), sm, st))
    # 打分宽表
    idx = [r[0] for r in rows]
    dfm = pd.DataFrame([r[2] for r in rows], index=idx,
                       columns=[str(c) for c in rows[0][1]])
    dft = pd.DataFrame([r[3] for r in rows], index=idx,
                       columns=[str(c) for c in rows[0][1]])
    ep = os.path.basename(ck_path).split('-')[0]
    def df_to_scores(dfx):
        return dict(scores={d: dfx.loc[d].dropna().to_dict() for d in dfx.index},
                    ranked={d: dfx.loc[d].dropna().sort_values(ascending=False).index.tolist()
                            for d in dfx.index},
                    top1={d: float(dfx.loc[d].max()) for d in dfx.index},
                    dates=sorted(dfx.index))
    out = dict(epoch=ep)
    for name, dfx in (('mixed', dfm), ('top', dft)):
        if name == 'top':
            protos = (('D01', dict(top_n=1, hold=5, stop_loss=0.08)),)
        for pn, pc in protos:
            mm, _, _ = run_backtest(df_to_scores(dfx), pc, market, tds, tdi)
            if mm:
                out[f'{name}_{pn}'] = mm['cum_net']
                out[f'{name}_{pn}_sh'] = mm['sharpe']
    # 验证指标 (从文件名解析)
    out['val_rankic'] = float(re.search(r'val_rankic=([-\d.]+)', ck_path).group(1))
    return out


def main():
    from engine import load_market, load_calendar
    if '--eval-only' in sys.argv and os.path.isdir('/tmp/v20_epochdiag'):
        m = load_v20_model()
        params_mod = m.params
        params_mod.all_data = pd.read_feather(PROJECT + 'trainingdata/fac_all.fea')
        params_mod.all_data = params_mod.all_data.set_index('date').sort_index()
        params_mod.factor_list = [c for c in params_mod.all_data.columns[1:]]
        params_mod.factor_num = len(params_mod.factor_list)
        ckpts = sorted(glob.glob('/tmp/v20_epochdiag/*.ckpt'),
                       key=lambda p: int(re.search(r"epoch=(\d+)", p).group(1)))
    else:
        ckpts, m = train_fold_all_ckpts()
    market = load_market()
    tds, tdi = load_calendar()
    rows = [eval_ckpt(p, m, market, tds, tdi) for p in ckpts]
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", "diag_epoch.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
