#!/usr/bin/env python3
"""
eval_ensemble.py — V6.9 4-fold 集成评估（240 天验证池逐日截面 rankIC）

对每个 fold 的 best checkpoint（model_train/2026q3/foldN/*/checkpoints/ 中
val_rankic_avg 最大者），在其自身 60 天验证集上逐日预测 4 个头，拼接成 240 天
验证池，逐日计算 Spearman(pred, raw_label)（可交易池 & label 有限），输出：
  - 各头 pooled mean rankIC / ICIR (mean/std*sqrt(n_days)) / 分月统计
  - 训练同口径对照（使用与训练验证一致的 winsor+fillna0 语义，便于与日志对比）

用法:
  python eval_ensemble.py [fold1 fold2 ...]   # 默认 1 2 3 4
输出: logs/ensemble_YYYYMMDD_HHMM.txt (同时打印到 stdout)
"""

import sys, os, re, glob, json
import numpy as np
import pandas as pd
import torch

# 与 train() 完全一致的数据装载
from model import (params, args, normed_data, DLLitModule, get_train_date_split,
                   PROJECT_ROOT, fac_path, fac_name, label_path)

torch.set_num_threads(4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_DIR = os.path.join(PROJECT_ROOT, 'Model/V6.9', 'model_train')
OUT_DIR = os.path.join(PROJECT_ROOT, 'Model/V6.9', 'logs')
SEASON = '2026q3'

_CKPT_RE = re.compile(r'val_rankic_avg=(-?\d+\.\d+)')


def load_all_data():
    """复刻 train(): fac_all + fac_new 合并 → 日期过滤 → 全零列剔除 → factor_list。"""
    all_data = pd.read_feather(rf'{fac_path}/{fac_name}.fea')
    fac_new = pd.read_feather(rf'{fac_path}/fac_new.fea')
    all_data = all_data.merge(fac_new, on=['date', 'Code'], how='left')
    date_list = list(all_data["date"].unique())
    date_list = [x for x in date_list if x in params.ret_data.index and x in params.liquid_data.index]
    date_list.sort()
    all_data = all_data.set_index("date").sort_index()
    all_zero_mask = (all_data == 0).all(axis=0)
    all_data = all_data.loc[:, ~all_zero_mask]
    feature_map = list(all_data.columns[1:])
    return all_data, date_list, feature_map


def find_best_checkpoint(fold):
    """在 fold 目录下递归找 val_rankic_avg 最大的 ckpt。"""
    best_path, best_val = None, -1e9
    for ckpt in glob.glob(os.path.join(MODEL_DIR, SEASON, f'fold{fold}', '*', 'checkpoints', '*.ckpt')):
        m = _CKPT_RE.search(os.path.basename(ckpt))
        if m:
            v = float(m.group(1))
            if v > best_val:
                best_val, best_path = v, ckpt
    return best_path, best_val


def raw_label_row(label_df, date):
    """日期不在索引 → 全 NaN 行；否则返回 (Code → raw label) 序列。"""
    if date in label_df.index:
        return label_df.loc[date]
    return pd.Series(np.nan, index=label_df.columns)


def evaluate_fold(fold, all_data, date_list, factor_list):
    """单折: 在其 60 天验证集上逐日预测, 返回逐日 (date, head→ic) 记录 + 训练同口径。"""
    train_dates, valid_dates, _ = get_train_date_split(fold=fold, season=SEASON, date_list=date_list)
    ckpt_path, best_val = find_best_checkpoint(fold)
    if ckpt_path is None:
        print(f"[fold{fold}] 无 checkpoint, 跳过")
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = DLLitModule(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()
    print(f"[fold{fold}] best={best_val:.4f} ckpt={os.path.basename(ckpt_path)} valid_days={len(valid_dates)}")

    rows = []          # 干净口径: raw label + buyable
    rows_trainstyle = []  # 训练同口径: winsor + fillna0 的 Label_*
    with torch.no_grad():
        for date in valid_dates:
            data = all_data.loc[date].copy()
            data_X, _, _, _, _, code_value, _, _, _ = normed_data(
                data, date, stage='val', factor_list=factor_list)
            x = data_X.float().to(device)
            pred_1d, pred_5d, pred_10d, pred_20d = [p.detach().cpu().numpy().ravel() for p in model.forward(x)]
            codes = np.asarray(code_value)

            # —— 干净口径: raw label feather + buyable mask ——
            buy_row = raw_label_row(params.buyable_mask, date)
            buy = pd.to_numeric(buy_row.reindex(codes), errors='coerce').values.astype(float)
            tradable = np.isfinite(buy) & (buy > 0.5)
            for name, pred, label_df in [
                ('1d', pred_1d, params.ret_1d_data),
                ('5d', pred_5d, params.ret_5d_data),
                ('10d', pred_10d, params.ret_10d_data),
                ('20d', pred_20d, params.ret_20d_data),
            ]:
                y = raw_label_row(label_df, date).reindex(codes).values
                m = tradable & np.isfinite(y)
                if m.sum() >= 50:
                    rows.append((date, name, _spearman(pred[m], y[m])))
    return rows


def evaluate_fold_trainstyle(fold, all_data, date_list, factor_list):
    """训练同口径逐日 rankIC (val_rankic / rankic5/10/20 语义):
    Ret1d=可交易池原始 label (不可买入→NaN); Label_5d/10d/20d=winsor+fillna0 全池。"""
    _, valid_dates, _ = get_train_date_split(fold=fold, season=SEASON, date_list=date_list)
    ckpt_path, best_val = find_best_checkpoint(fold)
    if ckpt_path is None:
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = DLLitModule(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()
    rows = []
    with torch.no_grad():
        for date in valid_dates:
            data = all_data.loc[date].copy()
            data_X, _, _, _, _, code_value, _, _, _ = normed_data(
                data, date, stage='val', factor_list=factor_list)
            x = data_X.float().to(device)
            pred_1d, pred_5d, pred_10d, pred_20d = [p.detach().cpu().numpy().ravel() for p in model.forward(x)]
            codes = np.asarray(code_value)
            # Ret1d (可交易池): 不可买入 → NaN
            ret1 = raw_label_row(params.ret_1d_data, date).reindex(codes).values
            buy_row = raw_label_row(params.buyable_mask, date)
            buy = pd.to_numeric(buy_row.reindex(codes), errors='coerce').values.astype(float)
            ret1_trad = ret1.copy()
            nb = ~(np.isfinite(buy) & (buy > 0.5))
            ret1_trad[nb] = np.nan
            m1 = np.isfinite(ret1_trad)
            if m1.sum() >= 50:
                rows.append((date, '1d_ts', _spearman(pred_1d[m1], ret1_trad[m1])))
            # Label_5d/10d/20d: winsor+fillna0 全池 (与训练 _evaluate_step 语义一致)
            for name, pred, df in [('5d_ts', pred_5d, params.ret_5d_data),
                                   ('10d_ts', pred_10d, params.ret_10d_data),
                                   ('20d_ts', pred_20d, params.ret_20d_data)]:
                y_raw = raw_label_row(df, date).reindex(codes).values
                if np.isfinite(pd.to_numeric(pd.Series(y_raw), errors='coerce').values).sum() < 50:
                    continue  # 尾部截断日 (20d 缺最后 ~21 交易日) 无有效 label, 与 clean 口径一致跳过
                y = np.asarray(y_raw, dtype=np.float64).copy()
                y = _winsor(y, 5.0)
                y[np.isnan(y)] = 0.0
                rows.append((date, name, _spearman(pred, y)))
    return rows


def _winsor(s, k=5.0):
    v = s[np.isfinite(s)]
    if len(v) == 0:
        return s
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad <= 1e-12:
        return s
    return np.clip(s, med - k * mad, med + k * mad)


def _spearman(a, b):
    a = pd.Series(a).rank().values
    b = pd.Series(b).rank().values
    return np.corrcoef(a, b)[0, 1]


def summarize(name, rows):
    if not rows:
        print(f"  {name}: 无数据")
        return None
    df = pd.DataFrame(rows, columns=['date', 'head', 'ic'])
    out = {}
    for head, grp in df.groupby('head'):
        ics = grp['ic'].values
        mean_ic = ics.mean()
        # ICIR 采用 V6.8 记录口径: mean/std (不乘 √n, 10d 标签逐日重叠, √n 会虚高)
        icir = mean_ic / (ics.std(ddof=1) + 1e-12)
        print(f"  {name} | {head:>4s}: mean IC={mean_ic:+.4f}  ICIR={icir:+.2f}  n_days={len(ics)}")
        # 分月
        months = {}
        for d, ic in zip(grp['date'], ics):
            months.setdefault(str(d)[:6], []).append(ic)
        for m in sorted(months):
            print(f"        {m}: {np.mean(months[m]):+.4f} (n={len(months[m])})")
        out[head] = dict(mean=float(mean_ic), icir=float(icir), n=len(ics),
                         months={m: round(float(np.mean(v)), 4) for m, v in sorted(months.items())})
    return out


def main():
    folds = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]
    all_data, date_list, factor_list = load_all_data()
    # train() 里才设置这两个全局, 评估需补上 (PredictModel 用 params.factor_num 定输入维度)
    params.factor_num = len(factor_list)
    params.factor_list = factor_list
    print(f"data: {all_data.shape}, dates: {len(date_list)}, factors: {len(factor_list)}")

    pooled_clean, pooled_ts = [], []
    for fold in folds:
        rows = evaluate_fold(fold, all_data, date_list, factor_list)
        if rows:
            pooled_clean.extend(rows)
        rows_ts = evaluate_fold_trainstyle(fold, all_data, date_list, factor_list)
        if rows_ts:
            pooled_ts.extend(rows_ts)

    print("\n===== 240 天验证池集成结果 (干净口径: raw label + buyable) =====")
    result = summarize('clean', pooled_clean)
    print("\n===== 训练同口径对照 (与各折验证日志可比) =====")
    result_ts = summarize('trainstyle', pooled_ts)

    # 四头平均 (干净口径)
    if result:
        avg = np.mean([result[h]['mean'] for h in ['1d', '5d', '10d', '20d'] if h in result])
        print(f"\n>>> 四头平均 rankIC = {avg:.4f}  目标 0.1")
        print(f">>> 判定: {'✅ 达标' if avg > 0.1 else '未达标 (弱折备选种子重训)'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'ensemble_result.json')
    with open(out_path, 'w') as f:
        json.dump({'clean': result, 'trainstyle': result_ts, 'avg_clean': avg if result else None,
                   'folds': folds}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
