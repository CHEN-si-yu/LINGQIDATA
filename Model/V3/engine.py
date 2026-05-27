#!/usr/bin/env python3
"""
V2 Model Engine — model architecture, inference, backtest, and evaluation.

Usage (CLI):
    python engine.py --season 2026q2 --dates 20250513,20250514
    python engine.py --season 2026q2 --dates 20250513 --per-fold

Usage (library):
    from engine import (predict_missing_dates, concat_model_4fold,
                        get_ret_ic, get_metrics, diversity_entropy,
                        ensemble_scores, run_individual_strategy)
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
from torch import nn
from numpy.linalg import eigh

# numpy compatibility (needed by older pickle files)
import numpy.core.numeric as _ncn
if 'numpy._core.numeric' not in sys.modules:
    sys.modules['numpy._core.numeric'] = _ncn

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
ROOT_PATH = r'/root/shared-nvme/lingqiData/Model/V3'
FAC_DIR = r'/root/shared-nvme/lingqiData/trainingdata/V3'
CALENDAR_PATH = r'/root/shared-nvme/lingqiData/data/calendar.parquet'
MODEL_PREFIX = r'nn'

# ============================================================
# Model naming
# ============================================================

_VAL_WEI_RE = re.compile(r'val_wei=(-?\d+\.\d+)')


def get_basic_name(fac_name='fac20260523', label_name='label',
                   dropout=True, dropout_rate=0.2):
    """Replicate model.py get_basic_name()."""
    name = rf'{MODEL_PREFIX}--{fac_name}--{label_name}'
    if dropout:
        name += rf'--dropout{dropout_rate}'
    return name


# ============================================================
# Model architecture (exact replica of model.py PredictModel)
# ============================================================

class PredictModel(nn.Module):
    """MLP: input_dim→256→128→32→1"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.output_layer = nn.Linear(32, 1)

    def forward(self, tsdata):
        x = tsdata.float()
        x = self.input_layer(x)
        x = self.net(x)
        x = self.output_layer(x)
        return x


# ============================================================
# Data preprocessing
# ============================================================

def normed_data(data, factor_list):
    """
    Replicate normed_data(stage='test'/'valid') from model.py.

    Steps:
      1. Filter stocks with >=10% valid factor values
      2. Winsorize at [0.005, 0.995]
      3. Z-score normalize (cross-sectional per date)
      4. Fill remaining NaN with 0
    """
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    valid_mask = data[factor_list].notna().sum(axis=1) >= valid_threshold
    data = data.loc[valid_mask].copy()

    code_value = data['Code'].values
    data_X = data[factor_list]

    quantiles = data_X.quantile([0.005, 0.995])
    data_X = data_X.clip(lower=quantiles.loc[0.005],
                         upper=quantiles.loc[0.995], axis=1)
    data_X = (data_X - data_X.mean()) / data_X.std()
    data_X = data_X.fillna(0)

    data_x_np = np.nan_to_num(
        data_X.to_numpy(dtype=np.float32, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    return torch.from_numpy(data_x_np), code_value


# ============================================================
# Model loading
# ============================================================

def load_model(checkpoint_path, input_dim):
    """Load PredictModel weights from a Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    state_dict = checkpoint['state_dict']

    model_state = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            model_state[k.removeprefix('model.')] = v

    model = PredictModel(input_dim)
    missing, unexpected = model.load_state_dict(model_state)
    if missing:
        raise RuntimeError(
            f'Missing keys when loading {checkpoint_path}: {missing}')
    if unexpected:
        raise RuntimeError(
            f'Unexpected keys when loading {checkpoint_path}: {unexpected}')

    model.eval()
    return model


def find_best_checkpoint(fold_dir):
    """Find the checkpoint with the highest val_wei in a fold directory."""
    ckpt_files = glob.glob(os.path.join(fold_dir, '**', '*.ckpt'),
                           recursive=True)
    if not ckpt_files:
        raise FileNotFoundError(f'No checkpoint files found under {fold_dir}')

    best_ckpt = None
    best_val = -float('inf')
    for ckpt in ckpt_files:
        m = _VAL_WEI_RE.search(os.path.basename(ckpt))
        if m is None:
            continue
        val = float(m.group(1))
        if val > best_val:
            best_val = val
            best_ckpt = ckpt

    if best_ckpt is None:
        raise RuntimeError(
            f'Could not parse val_wei from any checkpoint under {fold_dir}')
    return best_ckpt, best_val


def discover_folds(model_dir):
    """Discover fold directories (fold1..foldN) under model_dir."""
    folds = sorted(glob.glob(os.path.join(model_dir, 'fold[0-9]*')))
    if not folds:
        raise FileNotFoundError(f'No fold directories found under {model_dir}')
    return folds


# ============================================================
# Single-date prediction
# ============================================================

def predict_date(model, date, all_data, factor_list):
    """Run prediction for a single date. Returns DataFrame (index=Code, col='value')."""
    data = all_data.loc[date].copy()
    data_X, code_value = normed_data(data, factor_list)

    with torch.no_grad():
        preds = model(data_X)

    preds_np = preds.detach().cpu().numpy()
    result = pd.DataFrame(preds_np, index=code_value, columns=['value'])
    result.index.name = 'Code'
    return result


# ============================================================
# Core prediction: given explicit dates, run ensemble inference
# ============================================================

def predict_dates(dates, fac_path=None, model_train_base=None,
                  basic_name=None, season='2026q2'):
    """
    Predict specific dates using ensemble of all folds' best checkpoints.

    Parameters:
        dates: list of date strings (YYYYMMDD)
        fac_path, model_train_base, basic_name, season: model location

    Returns a DataFrame (index=date, columns=Code), or None if dates is empty.
    """
    if not dates:
        return None

    if fac_path is None:
        fac_path = rf'{FAC_DIR}/fac20260523.fea'
    if model_train_base is None:
        model_train_base = rf'{ROOT_PATH}/model_train'
    if basic_name is None:
        basic_name = get_basic_name()

    all_data = pd.read_feather(fac_path)
    all_data = all_data.set_index('date').sort_index()
    all_data = all_data.loc[:, all_data.replace(0, np.nan)
                                    .dropna(how="all", axis=1)
                                    .columns]

    factor_list = [c for c in all_data.columns if c != 'Code']
    input_dim = len(factor_list)

    model_dir = os.path.join(model_train_base, basic_name, season)
    folds = discover_folds(model_dir)
    print(f"[预测] 使用 {len(folds)} 个 fold 进行预测")

    models = {}
    for fold_dir in folds:
        fold_name = os.path.basename(fold_dir)
        best_ckpt, best_val = find_best_checkpoint(fold_dir)
        print(f"[预测]   [{fold_name}] {os.path.basename(best_ckpt)} "
              f"(val_wei={best_val:.4f})")
        models[fold_name] = load_model(best_ckpt, input_dim)

    new_scores = []
    for date in dates:
        if date not in all_data.index:
            print(f"[预测]   [{date}] 不在 factor 数据中，跳过")
            continue
        fold_preds = []
        for fold_name, model in models.items():
            result = predict_date(model, date, all_data, factor_list)
            fold_preds.append(result)
        zscored = [(f - f.mean()) / f.std() for f in fold_preds]
        ensemble = sum(zscored)
        date_score = ensemble['value']
        date_score.name = date
        new_scores.append(date_score)
        print(f"[预测]   [{date}] 完成, {len(ensemble)} 只股票")

    if not new_scores:
        return None

    new_score_df = pd.DataFrame(new_scores)
    new_score_df.index.name = 'date'
    print(f"[预测] 完成，共 {len(new_score_df)} 天数据。")
    return new_score_df


# ============================================================
# Batch prediction for missing dates (with calendar filter)
# ============================================================

def _load_trading_dates(calendar_path=None):
    """Load sorted list of trading dates (YYYYMMDD) from calendar.parquet."""
    if calendar_path is None:
        calendar_path = CALENDAR_PATH
    cal = pd.read_parquet(calendar_path)
    cal = cal[cal['is_open'] == 1]
    return set(cal['date'].astype(str).str.replace('-', '').tolist())


def predict_missing_dates(model_score, fac_path=None, model_train_base=None,
                          basic_name=None, season='2026q2',
                          calendar_path=None):
    """
    Given an existing model_score DataFrame (index=date, columns=Code),
    find factor data dates beyond model_score's last date that are also
    trading days, and predict them using an ensemble of all folds.

    Returns a DataFrame of new scores, or None if no missing dates.
    """
    if fac_path is None:
        fac_path = rf'{FAC_DIR}/fac20260523.fea'

    last_date = str(model_score.index.max())

    all_data = pd.read_feather(fac_path)
    all_data = all_data.set_index('date').sort_index()
    all_dates = sorted(all_data.index.unique())
    missing_dates = [d for d in all_dates if d > last_date]

    if not missing_dates:
        print("[预测] 所有 factor 日期已覆盖，无需补充预测。")
        return None

    # Filter to trading days only
    trading_dates = _load_trading_dates(calendar_path)
    non_trading = [d for d in missing_dates if d not in trading_dates]
    missing_dates = [d for d in missing_dates if d in trading_dates]

    if non_trading:
        print(f"[预测] 已排除 {len(non_trading)} 个非交易日: {non_trading}")

    print(f"[预测] 发现 {len(missing_dates)} 个未覆盖交易日: {missing_dates}")

    return predict_dates(missing_dates, fac_path=fac_path,
                         model_train_base=model_train_base,
                         basic_name=basic_name, season=season)


# ============================================================
# Save predictions to disk
# ============================================================

def save_predictions(score_df, output_dir):
    """
    Save each date in score_df as a .pkl file under output_dir.

    Parameters:
        score_df: DataFrame (index=date, columns=Code)
        output_dir: directory to save .pkl files
    """
    os.makedirs(output_dir, exist_ok=True)
    for date in score_df.index:
        s = score_df.loc[date].copy()
        s.index.name = 'Code'
        out_path = os.path.join(output_dir, f'{date}.pkl')
        s.to_pickle(out_path)
        print(f"[保存] {date} -> {out_path}")


# ============================================================
# Model result loading
# ============================================================

def concat_model_4fold(test_path, res_path):
    """Load 4-fold model predictions, z-score each fold, sum, and save."""
    market = 'ALL'
    model_res = []
    for fold in range(1, 5):
        fold_dirs = [d for d in os.listdir(test_path) if d.endswith(f'--fold{fold}')]
        if len(fold_dirs) == 0:
            print(f"Warning: No directory found for fold{fold} in {test_path}")
            continue
        data_path = rf"{test_path}/{fold_dirs[0]}"
        date_list = list(set([x[:8] for x in os.listdir(data_path)]))
        date_list.sort()
        all_res = []
        for date in date_list:
            date_res = pd.read_pickle(f'{data_path}/{date}.pkl').T
            date_res.index = [date] * len(date_res)
            all_res.append(date_res)
        all_res = pd.concat(all_res, axis=0).sort_index()
        all_res.index.name = "date"
        model_res.append(all_res.apply(lambda x: (x - x.mean()) / x.std(), axis=1))
    model_res = sum(model_res)
    os.makedirs(res_path, exist_ok=True)
    model_res.reset_index().to_feather(rf"{res_path}/{market}_zscore_score.fea")
    return model_res


# ============================================================
# Backtest: daily return and IC
# ============================================================

def get_ret_ic(score, ret_data, liquid_data, start='20230101', end='20241231',
               money=1.5e9):
    """
    Daily rebalance backtest with liquidity constraint (top 500).

    Returns (daily_ret_series, daily_ic_series).
    """
    model_score = score.copy()
    label_ret = []
    ic = []
    datelist = []

    for date in model_score.loc[start:end].index:
        datelist.append(date)
        code_rank = model_score.loc[date].sort_values(ascending=False)
        ret = ret_data.loc[date].reindex(code_rank.index).fillna(0) * 100
        liquid = liquid_data.loc[date].reindex(code_rank.index).fillna(0)
        total_hold = 0
        total_earned = 0
        for num, code in enumerate(code_rank.index):
            if num >= 500:
                break
            if (money - total_hold) < 1:
                break
            hold_money = min(money - total_hold, liquid[code])
            total_hold += hold_money
            total_earned += ret[code] * hold_money
        total_ret = total_earned / money
        label_ret.append(total_ret)
        ic.append(code_rank.corr(ret))

    ic = pd.Series(ic, index=datelist, dtype='float')
    label_ret = pd.Series(label_ret, index=datelist, dtype='float')
    return label_ret, ic


# ============================================================
# Metrics
# ============================================================

def get_metrics(ret, ic):
    """Compute evaluation metrics from return and IC series."""
    return {
        "IC": ic.mean(),
        "ICIR": ic.mean() / ic.std() if ic.std() != 0 else 0,
        "top_return": ret.mean(),
        "top_return_stability": ret.mean() / ret.std() if ret.std() != 0 else 0
    }


# ============================================================
# Diversity & Ensemble
# ============================================================

def diversity_entropy(score):
    """Compute diversity entropy of a score matrix."""
    if isinstance(score, pd.DataFrame):
        score = score.values
    m = score.shape[1]
    cov_matrix = np.cov(score, rowvar=False)
    eigenvalues, _ = eigh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    normalized_eigenvalues = eigenvalues / (np.sum(eigenvalues) + 1e-10)
    entropy = 0
    for p in normalized_eigenvalues:
        if p > 1e-10:
            entropy -= p * np.log(p)
    de = entropy / np.log(m)
    return de


def ensemble_scores(*dfs):
    """
    Z-score normalize each score DataFrame and stack into columns.

    Input: any number of score DataFrames (index=date, columns=Code)
    Output: stacked DataFrame with columns=score1, score2, ...
    """
    result_list = []
    for i, df in enumerate(dfs, 1):
        df_norm = df.copy()
        df_norm = df_norm.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        df_norm = df_norm.stack().rename(f"score{i}")
        result_list.append(df_norm)
    result = pd.concat(result_list, axis=1)
    return result.dropna()


# ============================================================
# Individual strategy backtest
# ============================================================

def run_individual_strategy(model_score, ret_data, top_n=5,
                            exclude_limit_up=True, limit_up_threshold=0.095,
                            start='20230101', end='20260331'):
    """
    Individual strategy: buy top_n stocks daily, equal weight.

    Parameters:
        model_score: score DataFrame (index=date, columns=Code)
        ret_data: daily return DataFrame (index=date, columns=Code), decimal form (0.01=1%)
        top_n: number of stocks to buy
        exclude_limit_up: skip stocks with return >= limit_up_threshold
        limit_up_threshold: limit-up threshold, default 0.095 (9.5%)
        start, end: backtest period

    Returns:
        (daily_ret_series, stats_dict)
    """
    model_score_sub = model_score.loc[start:end]
    daily_ret = []
    skip_total = 0
    skip_days = 0
    datelist = []

    for date in model_score_sub.index:
        ranked = model_score_sub.loc[date].sort_values(ascending=False)
        selected = []
        skipped = 0
        for code in ranked.index:
            if len(selected) >= top_n:
                break
            if exclude_limit_up:
                r_val = ret_data.loc[date, code] if code in ret_data.columns else 0.0
                if pd.notna(r_val) and r_val >= limit_up_threshold:
                    skipped += 1
                    continue
            selected.append(code)

        r = ret_data.loc[date, selected].mean() if len(selected) > 0 else 0.0

        datelist.append(date)
        daily_ret.append(r)
        skip_total += skipped
        if skipped > 0:
            skip_days += 1

    daily_ret = pd.Series(daily_ret, index=datelist, dtype='float')
    stats = {
        'skip_total': skip_total,
        'skip_days': skip_days,
        'total_days': len(datelist),
        'skip_day_pct': skip_days / len(datelist) * 100 if len(datelist) > 0 else 0
    }
    return daily_ret, stats


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='V2 Model Prediction — replicate model.py inference')
    parser.add_argument('--season', type=str, required=True,
                        help='Season, e.g. "2026q2"')
    parser.add_argument('--dates', type=str, required=True,
                        help='Comma-separated dates, e.g. "20250513,20250514"')
    parser.add_argument('--fac_name', type=str, default=r'fac20260523',
                        help='Factor data file name (without .fea extension)')
    parser.add_argument('--label_name', type=str, default=r'label',
                        help='Label name used in model directory naming')
    parser.add_argument('--dropout', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Whether dropout was used in model naming')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate used in model naming')
    parser.add_argument('--per-fold', action='store_true', default=False,
                        help='Save each fold separately instead of ensemble')
    return parser.parse_args()


def main():
    args = parse_args()
    dates = [d.strip() for d in args.dates.split(',')]

    basic_name = get_basic_name(args.fac_name, args.label_name,
                                args.dropout, args.dropout_rate)
    model_path = os.path.join(ROOT_PATH, 'model_train')
    model_dir = os.path.join(model_path, basic_name, args.season)

    fac_file = os.path.join(FAC_DIR, f'{args.fac_name}.fea')
    print(f"[INFO] Factor data: {fac_file}")
    all_data = pd.read_feather(fac_file)
    all_data = all_data.set_index('date').sort_index()

    all_data = all_data.loc[:, all_data.replace(0, np.nan)
                                    .dropna(how="all", axis=1)
                                    .columns]

    factor_list = [c for c in all_data.columns if c != 'Code']
    input_dim = len(factor_list)
    print(f"[INFO] Factor count: {input_dim}")

    folds = discover_folds(model_dir)
    print(f"[INFO] Model dir: {model_dir}")
    print(f"[INFO] Found {len(folds)} fold(s): "
          f"{[os.path.basename(f) for f in folds]}")

    models = {}
    for fold_dir in folds:
        fold_name = os.path.basename(fold_dir)
        best_ckpt, best_val = find_best_checkpoint(fold_dir)
        rel = os.path.relpath(best_ckpt, model_dir)
        print(f"[{fold_name}] Best ckpt: {rel}  (val_wei={best_val:.4f})")
        models[fold_name] = load_model(best_ckpt, input_dim)

    output_root = os.path.join(ROOT_PATH, 'model_pred', basic_name, args.season)
    os.makedirs(output_root, exist_ok=True)
    print(f"[INFO] Output: {output_root}")

    for date in dates:
        if date not in all_data.index:
            print(f"[WARN] Date {date} not in factor data, skip")
            continue

        fold_preds = {}
        for fold_name, model in models.items():
            result = predict_date(model, date, all_data, factor_list)
            fold_preds[fold_name] = result

        if args.per_fold:
            for fold_name, result in fold_preds.items():
                fold_dir = os.path.join(output_root, fold_name)
                os.makedirs(fold_dir, exist_ok=True)
                out = os.path.join(fold_dir, f'{date}.pkl')
                result.to_pickle(out)
            n_stocks = list(fold_preds.values())[0].shape[0]
            print(f"[OK] {date} -> {output_root}/fold*/{date}.pkl  "
                  f"(stocks: {n_stocks}, folds: {len(fold_preds)})")
        else:
            zscored = []
            for fold_result in fold_preds.values():
                z = (fold_result - fold_result.mean()) / fold_result.std()
                zscored.append(z)
            ensemble = sum(zscored)
            ensemble.index.name = 'Code'

            out = os.path.join(output_root, f'{date}.pkl')
            ensemble.to_pickle(out)
            print(f"[OK] {date} -> {os.path.relpath(out, ROOT_PATH)}  "
                  f"(stocks: {len(ensemble)}, folds: {len(fold_preds)})")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
