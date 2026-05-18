#!/usr/bin/env python3
"""
Prediction script for V2 factor model.

Replicates the exact inference behavior of model.py test mode for specified
dates using the best checkpoint (by val_wei) from each fold.  The per‑fold
forward pass is identical to model.py: same preprocessing, same architecture,
same weight loading.

By default the script z‑scores each fold's predictions cross‑sectionally then
sums them, matching min_test.ipynb's ALL_zscore_score.fea output.
Use --per-fold to save each fold's raw prediction separately.

Usage:
    python prediction.py --season 2026q2 --dates 20250513,20250514
    python prediction.py --season 2026q2 --dates 20250513 --per-fold
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

# ── numpy compatibility (needed by older pickle files) ──────────────────
import numpy.core.numeric as _ncn
if 'numpy._core.numeric' not in sys.modules:
    sys.modules['numpy._core.numeric'] = _ncn

warnings.filterwarnings("ignore")

# ── Paths (matching model.py) ───────────────────────────────────────────
root_path = r'/root/shared-nvme/lingqiData/Model/V2'
fac_path_default = r'/root/shared-nvme/lingqiData/trainingdata/V2'
model_prefix = r'nn'


def get_basic_name(fac_name, label_name, dropout, dropout_rate):
    """Replicate model.py get_basic_name()."""
    name = rf'{model_prefix}--{fac_name}--{label_name}'
    if dropout:
        name += rf'--dropout{dropout_rate}'
    return name


# Regex to extract val_wei from checkpoint filename like "epoch=7-val_wei=0.3807.ckpt"
_VAL_WEI_RE = re.compile(r'val_wei=(-?\d+\.\d+)')


# ── Model (exact replica of model.py PredictModel) ──────────────────────

class PredictModel(nn.Module):
    """MLP: 256→128→32→1 — identical to model.py"""
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


# ── Data preprocessing (exact replica of model.py normed_data, stage≠train) ──

def normed_data(data, factor_list):
    """
    Replicate normed_data(stage='test' / 'valid') from model.py.

    Steps:
      1. Filter stocks with >=10 % valid factor values
      2. Winsorize at [0.005, 0.995]
      3. Z-score normalize (cross-sectional per date)
      4. Fill remaining NaN with 0
    """
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    valid_mask = data[factor_list].notna().sum(axis=1) >= valid_threshold
    data = data.loc[valid_mask].copy()

    code_value = data['Code'].values
    data_X = data[factor_list]

    # winsorize
    quantiles = data_X.quantile([0.005, 0.995])
    data_X = data_X.clip(lower=quantiles.loc[0.005],
                         upper=quantiles.loc[0.995], axis=1)
    # z-score
    data_X = (data_X - data_X.mean()) / data_X.std()
    data_X = data_X.fillna(0)

    data_x_np = np.nan_to_num(
        data_X.to_numpy(dtype=np.float32, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    return torch.from_numpy(data_x_np), code_value


# ── Model loading ───────────────────────────────────────────────────────

def load_model(checkpoint_path, input_dim):
    """Load PredictModel weights from a Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    state_dict = checkpoint['state_dict']

    # Strip 'model.' prefix (Lightning wraps the nn.Module in DLLitModule)
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
        raise FileNotFoundError(
            f'No checkpoint files found under {fold_dir}')

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
        raise FileNotFoundError(
            f'No fold directories found under {model_dir}')
    return folds


# ── Predict ─────────────────────────────────────────────────────────────

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


# ── Batch prediction for missing dates ───────────────────────────────────

def predict_missing_dates(model_score, fac_path=None, model_train_base=None,
                          basic_name=None, season='2026q2'):
    """
    Given an existing model_score DataFrame (index=date, columns=Code),
    find factor data dates beyond model_score's last date and predict them
    using an ensemble of all folds' best checkpoints.

    Returns a DataFrame of new scores in the same format as model_score,
    or None if no missing dates are found.
    """
    if fac_path is None:
        fac_path = rf'{fac_path_default}/fac20260517.fea'
    if model_train_base is None:
        model_train_base = rf'{root_path}/model_train'
    if basic_name is None:
        basic_name = get_basic_name('fac20260517', 'label', True, 0.2)

    last_date = str(model_score.index.max())
    all_data = pd.read_feather(fac_path)
    all_data = all_data.set_index('date').sort_index()
    all_data = all_data.loc[:, all_data.replace(0, np.nan)
                                    .dropna(how="all", axis=1)
                                    .columns]

    all_dates = sorted(all_data.index.unique())
    missing_dates = [d for d in all_dates if d > last_date]

    if not missing_dates:
        print("[预测] 所有 factor 日期已覆盖，无需补充预测。")
        return None

    print(f"[预测] 发现 {len(missing_dates)} 个未覆盖日期: {missing_dates}")

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
    for date in missing_dates:
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

    new_score_df = pd.DataFrame(new_scores)
    new_score_df.index.name = 'date'
    print(f"[预测] 完成，新增 {len(new_score_df)} 天数据。")
    return new_score_df


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='V2 Model Prediction — replicate model.py inference')
    parser.add_argument('--season', type=str, required=True,
                        help='Season, e.g. "2026q2"')
    parser.add_argument('--dates', type=str, required=True,
                        help='Comma-separated dates, e.g. "20250513,20250514"')
    parser.add_argument('--fac_name', type=str, default=r'fac20260517',
                        help='Factor data file name (without .fea extension)')
    parser.add_argument('--label_name', type=str, default=r'label',
                        help='Label name used in model directory naming')
    parser.add_argument('--dropout', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Whether dropout was used in model naming')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate used in model naming')
    parser.add_argument('--per-fold', action='store_true', default=False,
                        help='Save each fold separately (matching model.py '
                             'test output) instead of ensemble averaging')
    return parser.parse_args()


def main():
    args = parse_args()
    dates = [d.strip() for d in args.dates.split(',')]

    # Build paths matching model.py conventions
    basic_name = get_basic_name(args.fac_name, args.label_name,
                                args.dropout, args.dropout_rate)
    model_path = os.path.join(root_path, 'model_train')
    model_dir = os.path.join(model_path, basic_name, args.season)

    # ── Load factor data (matching model.py) ──
    fac_file = os.path.join(fac_path_default, f'{args.fac_name}.fea')
    print(f"[INFO] Factor data: {fac_file}")
    all_data = pd.read_feather(fac_file)
    all_data = all_data.set_index('date').sort_index()

    # Match model.py line 568: drop columns that are entirely 0 or NaN
    all_data = all_data.loc[:, all_data.replace(0, np.nan)
                                    .dropna(how="all", axis=1)
                                    .columns]

    # Factor list: all columns except 'Code' (matches model.py columns[1:])
    factor_list = [c for c in all_data.columns if c != 'Code']
    input_dim = len(factor_list)
    print(f"[INFO] Factor count: {input_dim}")

    # ── Discover folds ──
    folds = discover_folds(model_dir)
    print(f"[INFO] Model dir: {model_dir}")
    print(f"[INFO] Found {len(folds)} fold(s): "
          f"{[os.path.basename(f) for f in folds]}")

    # ── Load best model from each fold ──
    models = {}
    for fold_dir in folds:
        fold_name = os.path.basename(fold_dir)
        best_ckpt, best_val = find_best_checkpoint(fold_dir)
        rel = os.path.relpath(best_ckpt, model_dir)
        print(f"[{fold_name}] Best ckpt: {rel}  (val_wei={best_val:.4f})")
        models[fold_name] = load_model(best_ckpt, input_dim)

    # ── Predict ──
    output_root = os.path.join(root_path, 'model_pred', basic_name,
                               args.season)
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
            # Save each fold separately (matching model.py test output)
            for fold_name, result in fold_preds.items():
                fold_dir = os.path.join(output_root, fold_name)
                os.makedirs(fold_dir, exist_ok=True)
                out = os.path.join(fold_dir, f'{date}.pkl')
                result.to_pickle(out)
            n_stocks = list(fold_preds.values())[0].shape[0]
            print(f"[OK] {date} -> {output_root}/fold*/{date}.pkl  "
                  f"(stocks: {n_stocks}, folds: {len(fold_preds)})")
        else:
            # Z-score each fold cross-sectionally, then sum (matches ALL_zscore_score.fea)
            zscored = []
            for fold_result in fold_preds.values():
                z = (fold_result - fold_result.mean()) / fold_result.std()
                zscored.append(z)
            ensemble = sum(zscored)
            ensemble.index.name = 'Code'

            out = os.path.join(output_root, f'{date}.pkl')
            ensemble.to_pickle(out)
            print(f"[OK] {date} -> {os.path.relpath(out, root_path)}  "
                  f"(stocks: {len(ensemble)}, folds: {len(fold_preds)})")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
