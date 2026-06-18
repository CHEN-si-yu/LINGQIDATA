"""
V3.0 — MVFP: Best Combination (4-Year Window + 10d Target + Full Factors)

Combines the two strongest improvements from V1 experiments:
  - 4-year training window (from V1.3, +33% Sharpe)
  - 10-day return target (from V1.7, +60% Sharpe)
  - Full 953-factor set (from V1.2, +9% IC)

This is the optimal configuration based on V1.X experimental results.

Architecture:
  Input(F d) → SE recalibration → Shared Backbone(256→128)  # F = factor_num (determined at runtime)
    ├── View 1: Normal Head₁ (128→32→1) + Feature Dropout₁
    ├── View 2: Normal Head₂ (128→32→1) + Feature Dropout₂
    ├── View 3: Normal Head₃ (128→32→1) + Feature Dropout₃
    └── Shared: Limit-Up Head (128→32→1) + gate

Loss:
  For each view v: L_v = L_wpcc(score_v) + α * L_rank(score_v)
  Diversity Loss: L_div = max(0, mean_corr(views) - target_corr)
  Total: L = mean(L_v) + λ_div * L_div + L_limit_up

Inference:
  score = mean(view_1_score, view_2_score, view_3_score)
  (Each view uses standard evaluation mode, no feature dropout at inference)

Key Configuration:
  - 4-year training window (extended from baseline 2 years)
  - 10-day return prediction target
  - Full 953-factor set (pre-filtering applied in data loading)
  - K=3 parallel Normal Heads with feature dropout diversity
"""

import gc
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
import warnings
import shutil
import pickle
from datetime import datetime, timedelta
import sys
import random
from dateutil.relativedelta import relativedelta
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging,
                                         TQDMProgressBar)
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler
from pathlib import Path

warnings.filterwarnings("ignore")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

# 自适应 CPU 线程数（不超过 16，避免跨 NUMA 性能退化）
cpu_num = min(16, os.cpu_count() or 8)
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
# bf16-mixed 训练：保持 fp32 累加精度，matmul 使用 TF32（Ampere+ GPU 加速 2x）
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(False)
# cuDNN benchmark 选择最优卷积算法（模型不含卷积，仅防御性设置）
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--seed', type=int, default=3253)
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--loss', default='wpcc')
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--min_epochs', type=int, default=15)
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--strategy', default='auto')
    parser.add_argument('--find_unused_parameters', default=False)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--check_test_every_n_epoch', type=int, default=1)

    parser.add_argument('--log_every_n_steps', type=int, default=10)
    parser.add_argument('--early_stop', action='store_true', default=True)
    parser.add_argument('--swa', action='store_true', default=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')

    # V9.5 args
    parser.add_argument('--num_views', type=int, default=3)
    parser.add_argument('--diversity_weight', type=float, default=0.1)
    parser.add_argument('--target_corr', type=float, default=0.7)

    args, unknown = parser.parse_known_args()
    return args


args = parse_args()

PROJECT_ROOT = "/root/autodl-fs/lingqiData/"
root_path = PROJECT_ROOT + 'Model/V3.1'
fac_path = PROJECT_ROOT + 'trainingdata/'
fac_name = r'fac20260614'
label_path = PROJECT_ROOT + 'trainingdata'
label_name = r'label_ret_5d'  # V3.1: 5-day return target (best from V1.7)
liquid_path = PROJECT_ROOT + 'trainingdata'
liquid_name = r'trade_amt'


class params:
    model_path = rf'{root_path}/model_train'
    profiler_path = rf'{root_path}/logs'
    model_prefix = rf'nn'
    liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")        # label_ret_10d — 训练 target（低噪声）
    ret_1d_data = pd.read_feather(rf"{label_path}/label_ret_1d.fea").set_index("index")      # label_ret_1d — 验证/测试真实收益（单日换手）
    dropout = True
    dropout_rate = 0.2
    normed_method = 'zscore'

    feature_dropout_rate = 0.25
    ranknet_alpha = 0.1
    time_decay_half_life_days = 365
    time_decay_ref_date = '20260101'

    near_limit_up_idx = None
    gate_threshold = 1.5
    gate_temperature = 5.0
    limit_up_loss_weight = 2.0

    # V9.5: Multi-View Factor Perturbation
    num_views = 3
    diversity_weight = 0.1       # weight for diversity loss
    target_corr = 0.7            # target mean correlation between views
    se_reduction = 16

    # Training stability
    swa_enabled = True
    swa_epoch_start = 0.6
    early_stop_patience = 8
    gradient_clip_val = 1.0
    warmup_epochs = 3


def get_basic_name():
    name = rf'{params.model_prefix}--{fac_name}--{label_name}'
    if params.dropout:
        name += rf'--dropout{params.dropout_rate}'
    return name


def normed_data(data, date, stage, factor_list, normed_method=params.normed_method):
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    valid_mask = data[factor_list].notna().sum(axis=1) >= valid_threshold
    data = data.loc[valid_mask].copy()
    liquid_data = params.liquid_data.loc[date]
    data['liquid'] = liquid_data.reindex(data["Code"]).values
    ret_data = params.ret_data.loc[date]
    data['Label'] = ret_data.reindex(data["Code"]).values
    if stage == "train":
        data['Label'] = (data['Label'] - data['Label'].mean()) / data['Label'].std()
    data['Label'] = data['Label'].fillna(0)
    # 加载真实 1 日收益（仅用于验证/测试评估，不参与训练 loss）
    ret_1d_series = params.ret_1d_data.loc[date]
    data['Ret1d'] = ret_1d_series.reindex(data["Code"]).values
    data['Ret1d'] = data['Ret1d'].fillna(0)
    code_value = data['Code'].values
    data_X = data.drop(['Code', 'Label', 'liquid', 'Ret1d'], axis=1)
    data_y = data['Label']
    data_liquid = data['liquid']

    if normed_method == 'zscore':
        quantiles = data_X.quantile([0.005, 0.995])
        data_X = data_X.clip(lower=quantiles.loc[0.005], upper=quantiles.loc[0.995], axis=1)
        data_X = (data_X - data_X.mean()) / data_X.std()
        data_X = data_X.fillna(0)
    else:
        raise NotImplementedError

    data_ret1d = data['Ret1d']
    data_x_np = np.nan_to_num(data_X.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_y_np = np.nan_to_num(data_y.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_liquid_np = np.nan_to_num(data_liquid.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_ret1d_np = np.nan_to_num(data_ret1d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    return torch.from_numpy(data_x_np), \
        torch.from_numpy(data_y_np), \
        code_value, torch.from_numpy(data_liquid_np), \
        torch.from_numpy(data_ret1d_np)


def collate_fn(datas):
    data_X, data_y, data_time, code_value, data_liquid, data_ret1d = zip(*datas)
    return list(data_X), list(data_y), list(data_time), list(code_value), list(data_liquid), list(data_ret1d)


class DLDataset(torch.utils.data.Dataset):
    def __init__(self, date_list, all_data, factor_list, stage='train'):
        self.date_list = date_list
        self.all_data = all_data
        self.factor_list = factor_list
        self.stage = stage
    def __getitem__(self, index):
        date = self.date_list[index]
        if date == 'out_sample':
            return 'out_sample', 'out_sample', 'out_sample', 'out_sample', 'out_sample', 'out_sample'
        data = self.all_data.loc[date].copy()
        data_X, data_y, code_value, data_liquid, data_ret1d = normed_data(data, date, stage=self.stage, factor_list=self.factor_list)
        return data_X, data_y, date, code_value, data_liquid, data_ret1d
    def __len__(self):
        return len(self.date_list)


class DLDataModule(pl.LightningDataModule):
    def __init__(self, args, train_date_list, valid_date_list, test_date_list):
        super().__init__()
        self.args = args
        all_data = params.all_data
        factor_list = params.factor_list
        self.tr = DLDataset(train_date_list, all_data=all_data, factor_list=factor_list, stage='train')
        self.val = DLDataset(valid_date_list, all_data=all_data, factor_list=factor_list, stage='valid')
        self.test = DLDataset(test_date_list, all_data=all_data, factor_list=factor_list, stage='test')
    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.args.batch_size, collate_fn=collate_fn,
                          num_workers=min(8, cpu_num // 2) if torch.cuda.is_available() else 0, shuffle=True,
                          persistent_workers=True, drop_last=False, pin_memory=True)
    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                          num_workers=0, persistent_workers=False, pin_memory=False, drop_last=False)
    def val_dataloader(self):
        return self._val_dataloader(self.val)
    def test_dataloader(self):
        return self._val_dataloader(self.test)


# ── WPCC loss ──────────────────────────────────────────────────────────

_WPCC_WEIGHT_CACHE = {}

def _get_wpcc_rank_weights(length, device, dtype):
    cache_key = (length, device.type, device.index, dtype)
    weight = _WPCC_WEIGHT_CACHE.get(cache_key)
    if weight is None:
        if length <= 1:
            weight = torch.ones((length, 1), device=device, dtype=dtype)
        else:
            exponents = torch.linspace(0, 1, steps=length, device=device, dtype=dtype)
            weight = torch.pow(torch.full((length,), 0.5, device=device, dtype=dtype), exponents).unsqueeze(1)
        _WPCC_WEIGHT_CACHE[cache_key] = weight
    return weight


def get_loss_fn(loss):
    def wpcc(preds, y):
        argsort = torch.argsort(preds, descending=True, dim=0)
        weight_new = _get_wpcc_rank_weights(preds.shape[0], preds.device, preds.dtype)
        weight = torch.empty_like(preds)
        weight.scatter_(0, argsort, weight_new.expand_as(preds))
        weight_sum = weight.sum(dim=0)
        weighted_pred_mean = (preds * weight).sum(dim=0) / weight_sum
        weighted_y_mean = (y * weight).sum(dim=0) / weight_sum
        wcov = (preds * y * weight).sum(dim=0) / weight_sum - weighted_pred_mean * weighted_y_mean
        pred_std = torch.sqrt(((preds - preds.mean(dim=0)) ** 2 * weight).sum(dim=0) / weight_sum)
        y_std = torch.sqrt(((y - y.mean(dim=0)) ** 2 * weight).sum(dim=0) / weight_sum)
        return -(wcov / (pred_std * y_std + 1e-12)).mean()
    def output(loss):
        return {'wpcc': wpcc}[loss]
    return output(loss)


def pairwise_ranking_loss(preds, y, n_pairs=2000):
    n = preds.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=preds.device)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    idx_i = torch.randint(0, n, (n_pairs,), device=preds.device)
    idx_j = torch.randint(0, n, (n_pairs,), device=preds.device)
    valid = idx_i != idx_j
    idx_i, idx_j = idx_i[valid], idx_j[valid]
    if len(idx_i) == 0:
        return torch.tensor(0.0, device=preds.device)
    si = preds[idx_i].squeeze(-1)
    sj = preds[idx_j].squeeze(-1)
    yi = y[idx_i].squeeze(-1)
    yj = y[idx_j].squeeze(-1)
    target = (yi > yj).float()
    logit = si - sj
    return F.binary_cross_entropy_with_logits(logit, target)


_TIME_WEIGHT_CACHE = {}

def _get_time_weight(date_str, ref_date=None, half_life_days=None):
    if ref_date is None:
        ref_date = params.time_decay_ref_date
    if half_life_days is None:
        half_life_days = params.time_decay_half_life_days
    cache_key = (date_str, ref_date, half_life_days)
    weight = _TIME_WEIGHT_CACHE.get(cache_key)
    if weight is not None:
        return weight
    try:
        date_dt = datetime.strptime(date_str, '%Y%m%d')
        ref_dt = datetime.strptime(ref_date, '%Y%m%d')
        days_diff = max(0, (ref_dt - date_dt).days)
        weight = 0.5 ** (days_diff / half_life_days)
    except (ValueError, TypeError):
        weight = 1.0
    _TIME_WEIGHT_CACHE[cache_key] = weight
    return weight


# ── SE-Net Layer ───────────────────────────────────────────────────────

class SELayer(nn.Module):
    def __init__(self, num_factors, reduction=16):
        super().__init__()
        bottleneck = max(8, num_factors // reduction)
        self.fc1 = nn.Linear(num_factors * 2, bottleneck)
        self.fc2 = nn.Linear(bottleneck, num_factors)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=0, keepdim=True)
        max_pool = x.max(dim=0, keepdim=True)[0]
        squeeze = torch.cat([avg_pool, max_pool], dim=1)
        excite = F.relu(self.fc1(squeeze))
        excite = self.fc2(excite)
        return x * self.sigmoid(excite)


# ── Single Normal Head (shared architecture per view) ──────────────────

class NormalHead(nn.Module):
    """Single view head: 128 → 32 → 1 with residual."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 32)
        self.bn = nn.BatchNorm1d(32)
        self.act = nn.GELU()
        self.proj = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, shared_repr):
        identity = self.proj(shared_repr)
        out = self.fc(shared_repr)
        out = self.bn(out)
        out = self.act(out)
        return self.output(out + identity)


# ── Diversity Loss ─────────────────────────────────────────────────────

def diversity_loss(view_scores, target_corr=0.7):
    """Penalize views that are too highly correlated.

    L_div = max(0, mean_pearson_corr(views) - target_corr)

    Args:
        view_scores: list of K tensors, each (N, 1) scalar scores
        target_corr: maximum allowed mean correlation

    Returns:
        scalar penalty (0 if views are already diverse enough)
    """
    K = len(view_scores)
    if K < 2:
        return torch.tensor(0.0, device=view_scores[0].device)

    # Stack: (N, K)
    stacked = torch.cat([s.squeeze(-1).unsqueeze(-1) for s in view_scores], dim=-1)

    # Center
    centered = stacked - stacked.mean(dim=0, keepdim=True)

    # Covariance matrix: (K, K)
    cov = (centered.T @ centered) / (centered.shape[0] - 1 + 1e-8)

    # Standard deviations
    stds = torch.sqrt(torch.diag(cov) + 1e-8)

    # Correlation matrix
    corr = cov / (stds.unsqueeze(0) * stds.unsqueeze(1) + 1e-8)

    # Mean off-diagonal correlation
    mask = ~torch.eye(K, dtype=torch.bool, device=corr.device)
    mean_corr = corr[mask].mean()

    return F.relu(mean_corr - target_corr)


# ── V9.5 PredictModel: SE-Net + Multi-View Heads ───────────────────────

class PredictModel(nn.Module):
    """SE-Net backbone → K parallel Normal Heads + shared Limit-Up Head."""
    def __init__(self, args):
        super(PredictModel, self).__init__()
        input_dim = params.factor_num

        self.se_layer = SELayer(input_dim, reduction=params.se_reduction)

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.res_fc1 = nn.Linear(256, 128)
        self.res_bn1 = nn.BatchNorm1d(128)
        self.res_act1 = nn.GELU()
        self.res_drop1 = nn.Dropout(0.1)
        self.res_proj1 = nn.Linear(256, 128)

        # K parallel Normal Heads
        self.normal_heads = nn.ModuleList([
            NormalHead() for _ in range(params.num_views)
        ])

        # Shared Limit-Up Head
        self.limit_up_fc2 = nn.Linear(128, 32)
        self.limit_up_bn2 = nn.BatchNorm1d(32)
        self.limit_up_act2 = nn.GELU()
        self.limit_up_proj2 = nn.Linear(128, 32)
        self.limit_up_output = nn.Linear(32, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def _make_feature_mask(self, x, rate, seed_offset=0):
        """Create a Bernoulli feature dropout mask.
        
        Uses a deterministic seed based on global args.seed + offset
        to ensure reproducibility across runs while maintaining
        diversity across views.
        """
        n_features = x.shape[1]
        keep_prob = 1.0 - rate
        # Deterministic seed: global seed + offset (reproducible across runs)
        generator = torch.Generator(device=x.device)
        base_seed = 3253  # args.seed default
        generator.manual_seed(base_seed + seed_offset)
        mask = torch.bernoulli(
            torch.full((1, n_features), keep_prob, device=x.device),
            generator=generator)
        return mask / keep_prob

    def forward(self, tsdata, feature_dropout_rate=0.0, training_views=True):
        """Forward pass.

        Args:
            tsdata: (N, F) input factor data
            feature_dropout_rate: dropout rate for feature masking
            training_views: if True, return all K view scores separately
                           if False, return mean score (inference mode)

        Returns:
            If training_views: (list of K scores, gate, limit_up_score)
            If not: (mean_score, gate, limit_up_score)
        """
        x = tsdata.float()

        x = self.se_layer(x)

        if params.near_limit_up_idx is not None and params.near_limit_up_idx < x.shape[1]:
            bias_20_zscore = x[:, params.near_limit_up_idx]
        else:
            bias_20_zscore = torch.zeros(x.shape[0], device=x.device)

        # Feature dropout on input (before shared backbone)
        if feature_dropout_rate > 0:
            x = x * self._make_feature_mask(x, feature_dropout_rate, seed_offset=0)

        x = self.input_layer(x)
        identity = self.res_proj1(x)
        out = self.res_fc1(x)
        out = self.res_bn1(out)
        out = self.res_act1(out)
        out = self.res_drop1(out)
        shared_repr = out + identity  # (N, 128)

        # Shared Limit-Up Head
        lu_identity = self.limit_up_proj2(shared_repr)
        lu_out = self.limit_up_fc2(shared_repr)
        lu_out = self.limit_up_bn2(lu_out)
        lu_out = self.limit_up_act2(lu_out)
        limit_up_score = self.limit_up_output(lu_out + lu_identity)

        gate = torch.sigmoid(
            (bias_20_zscore - params.gate_threshold) * params.gate_temperature
        ).unsqueeze(1)

        # K parallel Normal Heads
        view_scores = []
        for head in self.normal_heads:
            view_score = head(shared_repr)
            final_view = gate * limit_up_score + (1.0 - gate) * view_score
            view_scores.append(final_view)

        if training_views:
            return view_scores, gate, limit_up_score, shared_repr
        else:
            # Inference: mean of views
            mean_score = torch.stack(view_scores, dim=0).mean(dim=0)
            return mean_score, gate, limit_up_score


# ── WarmupReduceLROnPlateau ────────────────────────────────────────────

class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_epochs: int, **kwargs):
        self._warmup_base_lrs = [float(g['lr']) for g in optimizer.param_groups]
        super().__init__(optimizer, **kwargs)
        self.warmup_epochs = warmup_epochs
        self._warmup_step = 0
    def step(self, metrics=None):
        if self._warmup_step < self.warmup_epochs:
            progress = float(self._warmup_step + 1) / float(max(1, self.warmup_epochs))
            for param_group, base_lr in zip(self.optimizer.param_groups, self._warmup_base_lrs):
                param_group['lr'] = base_lr * progress
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            self._warmup_step += 1
        else:
            super().step(metrics)


# ── Lightning Module ───────────────────────────────────────────────────

class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel(args)
        print(self.model)
        self.loss_fn = get_loss_fn(self.args.loss)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_times = set()

    def forward(self, tsdata):
        """Inference: mean of all views."""
        mean_score, gate, limit_up_score = self.model(tsdata, feature_dropout_rate=0.0, training_views=False)
        return mean_score

    @staticmethod
    def _pearson_corr(preds, ret):
        preds = preds.reshape(-1)
        ret = ret.reshape(-1)
        preds_centered = preds - preds.mean()
        ret_centered = ret - ret.mean()
        denominator = torch.sqrt(preds_centered.square().sum() * ret_centered.square().sum())
        if denominator <= 1e-12:
            return preds.new_tensor(0.0)
        return (preds_centered * ret_centered).sum() / denominator

    def training_step(self, batch, batch_idx):
        tsdatas, rets, times, code_values, liquids, rets_1d = batch  # rets_1d not used in training loss
        total_loss_sum = 0.0
        for i in range(len(tsdatas)):
            tsdata, ret, liquid = tsdatas[i], rets[i], liquids[i]
            ret = ret.unsqueeze(1)

            # Forward with all K views
            view_scores, gate, limit_up_score, shared_repr = self.model(
                tsdata, feature_dropout_rate=params.feature_dropout_rate, training_views=True)

            # Per-view losses
            view_losses = []
            for v_score in view_scores:
                v_wpcc = self.loss_fn(v_score, ret)
                v_rank = pairwise_ranking_loss(v_score, ret)
                view_losses.append(v_wpcc + params.ranknet_alpha * v_rank)

            mean_view_loss = torch.stack(view_losses).mean()

            # Diversity loss: penalize high correlation between views
            div_loss = diversity_loss(view_scores, target_corr=params.target_corr)

            # Limit-up extra loss (on mean score)
            mean_score = torch.stack(view_scores, dim=0).mean(dim=0)
            lu_mask = (gate > 0.5).squeeze(1)
            lu_extra = torch.tensor(0.0, device=mean_score.device)
            if lu_mask.sum() > 5:
                lu_extra = self.loss_fn(mean_score[lu_mask], ret[lu_mask])

            time_str = times[i] if isinstance(times[i], str) else str(times[i])
            time_weight = _get_time_weight(time_str)

            total_loss = (
                mean_view_loss
                + params.diversity_weight * div_loss
                + params.limit_up_loss_weight * lu_extra
            ) * time_weight
            total_loss_sum += total_loss

        avg_loss = total_loss_sum / len(tsdatas)
        self.log('train_loss', avg_loss, prog_bar=True, on_step=True)
        self.log('div_loss', div_loss.detach(), prog_bar=False, on_step=True)
        return avg_loss

    def _evaluate_step(self, batch, batch_idx, stage):
        def get_excess_return(preds, ret, liquid, money):
            topk = min(500, preds.shape[0])
            sort = torch.argsort(preds.squeeze(1), descending=True, stable=True)[:topk]
            sorted_liquid = liquid[sort].reshape(-1)
            sorted_ret = ret[sort].reshape(-1)
            money_tensor = preds.new_tensor(money)
            previous_hold = torch.cat([sorted_liquid.new_zeros(1), sorted_liquid.cumsum(dim=0)[:-1]])
            remaining_before_buy = money_tensor - previous_hold
            hold_money = torch.minimum(remaining_before_buy, sorted_liquid)
            hold_money = torch.where(remaining_before_buy >= 1, torch.clamp(hold_money, min=0.0), torch.zeros_like(hold_money))
            total_ret = (sorted_ret * hold_money).sum() / money_tensor
            return total_ret

        excess_return_list = []
        ic_list = []
        tsdatas, rets, times, code_values, liquids, rets_1d = batch
        for i in range(len(tsdatas)):
            tsdata, ret_10d, time, code_value, liquid, ret_1d = tsdatas[i], rets[i], times[i], code_values[i], liquids[i], rets_1d[i]
            if isinstance(tsdata, str) and tsdata == 'out_sample':
                pass
            else:
                preds = self.forward(tsdata)
                if stage == "test":
                    self.test_times.add(time)
                    preds_cpu = preds.detach().cpu().numpy()
                    res = pd.DataFrame(preds_cpu, index=code_value, columns=['value'])
                    res.index.name = 'Code'
                    res.to_pickle(f'{params.test_save_path}/{time}.pkl')
                # 使用真实 1 日收益计算评估指标（非训练 10d label）
                excess_return = get_excess_return(preds, ret_1d, liquid, money=1.5e9)
                excess_return_list.append(excess_return)
                ic_list.append(self._pearson_corr(preds.squeeze(), ret_1d))
        try:
            res_list = [sum(excess_return_list) / len(excess_return_list), sum(ic_list) / len(ic_list)]
        except Exception:
            res_list = [np.nan, np.nan]
        if stage == 'val':
            self.validation_step_outputs.append(res_list)
        if stage == "test":
            self.test_step_outputs.append(res_list)
        return res_list

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'test')
    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'val')

    def on_validation_epoch_end(self):
        val_step_outputs = self.validation_step_outputs
        num_batch = len(val_step_outputs)
        self.log('val_ret', sum([(data[0] * 100) for data in val_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.log('val_icmean', sum([(data[1]) for data in val_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.log('val_wei', sum([(data[0] * 100 + data[1] * 0.7) for data in val_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
        gc.collect()

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        num_batch = len(test_step_outputs)
        self.log('test_wei', sum([data[0] * 100 for data in test_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.log('test_icmean', sum([(data[1]) for data in test_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.test_step_outputs.clear()
        self.test_times.clear()

    def on_before_optimizer_step(self, optimizer):
        """Manual gradient clipping — compatible with bf16-mixed precision.
        
        Lightning's built-in gradient_clip_val triggers
        'No inf checks were recorded for this optimizer' under bf16 autocast.
        Manual clipping in this hook runs outside autocast and avoids the bug.
        """
        if params.gradient_clip_val is not None and params.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=params.gradient_clip_val)
    
    def configure_optimizers(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]

        warmup_epochs = params.warmup_epochs
        if warmup_epochs > 0:
            scheduler = {
                'scheduler': WarmupReduceLROnPlateau(
                    optimizer, warmup_epochs=warmup_epochs,
                    mode='max', factor=0.5, patience=3, min_lr=5e-6, cooldown=2),
                'monitor': 'val_wei', 'interval': 'epoch', 'frequency': 1,
            }
        else:
            scheduler = {
                'scheduler': ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=5e-6, cooldown=2),
                'monitor': 'val_wei',
            }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_wei', mode='max', save_top_k=4, save_last=False,
                            filename='{epoch}-{val_wei:.4f}')
        ]
        if params.swa_enabled:
            callbacks.append(StochasticWeightAveraging(
                swa_lrs=self.args.lr * 0.1, swa_epoch_start=params.swa_epoch_start,
                device='cuda' if torch.cuda.is_available() else 'cpu'))
        callbacks.append(EarlyStopping(
            monitor='val_wei', mode='max', patience=params.early_stop_patience,
            check_on_train_epoch_end=False))
        return callbacks


# ── Trainer ────────────────────────────────────────────────────────────

def train_single(args, name, seed, train_date_list, valid_date_list, test_date_list):
    torch.set_num_threads(args.threads)
    seed_everything(seed)
    logger = TensorBoardLogger(save_dir=params.model_path, name=name)
    profiler = SimpleProfiler(dirpath=params.profiler_path, filename=name)
    args_for_trainer = dict()
    for key, value in vars(args).items():
        try:
            Trainer(**{key: value})
            args_for_trainer[key] = value
        except:
            pass
    enable_tqdm_progress = sys.stdout.isatty() or os.environ.get("FORCE_TQDM_PROGRESS") == "1"
    trainer_callbacks = [TQDMProgressBar(refresh_rate=10)] if enable_tqdm_progress else []
    trainer = Trainer(**args_for_trainer,
                      callbacks=trainer_callbacks,
                      num_sanity_val_steps=2,
                      profiler=profiler, logger=logger,
                      enable_progress_bar=enable_tqdm_progress,
                      deterministic=False,
                      precision='bf16-mixed')
    litmodel = DLLitModule(args)
    dm = DLDataModule(args, train_date_list, valid_date_list, test_date_list)
    trainer.fit(litmodel, dm)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_result = trainer.test(ckpt_path=best_ckpt, datamodule=dm, weights_only=False)
    print(test_result)


# ── Main train entry ───────────────────────────────────────────────────

def get_train_date_split(fold, season, date_list):
    """Fixed train split: 4yr train + random 1/4 valid + quarter test.

    Args:
        fold: fold index (1-4), used as random seed offset
        season: e.g. "2025q1"
        date_list: sorted list of date strings "YYYYMMDD"

    Returns:
        (train_dates, valid_dates, test_dates)
    """
    year = int(season[:4])
    q = int(season[5])
    test_start = datetime(year, (q - 1) * 3 + 1, 1)
    test_end = test_start + relativedelta(months=3)

    # Test: 目标季度
    test_dates = [d for d in date_list
                  if test_start <= datetime.strptime(d, "%Y%m%d") < test_end]

    # Valid: test 开始前 12 个月中随机抽取 1/4 天数
    valid_start = test_start - relativedelta(years=1)
    valid_end = test_start
    valid_pool = [d for d in date_list
                  if valid_start <= datetime.strptime(d, "%Y%m%d") < valid_end]

    random.seed(args.seed + fold)
    sample_size = max(1, len(valid_pool) // 4)
    valid_dates = sorted(random.sample(valid_pool, sample_size))

    # Train: valid 开始前固定 4 年
    train_start = valid_start - relativedelta(years=4)
    train_dates = [d for d in date_list
                   if train_start <= datetime.strptime(d, "%Y%m%d") < valid_start]

    return train_dates, valid_dates, test_dates


def train(args, name, market, season, fold, state='train'):
    save_path = rf"{root_path}/model_test/{get_basic_name()}"
    try:
        os.makedirs(save_path, exist_ok=True)
        current_file_path = os.path.abspath(__file__)
        shutil.copy(current_file_path, save_path)
    except Exception as e:
        print(e)

    params.model_name = f"{save_path}/{name[:len(market) + 19]}"
    params.test_save_path = f"{save_path}/{name[:len(market)] + name[len(market) + 6:len(market) + 19]}--fold{fold}"
    os.makedirs(params.test_save_path, exist_ok=True)

    params.all_data = pd.read_feather(rf'{fac_path}/{fac_name}.fea')
    date_list = list(params.all_data["date"].unique())
    date_list = [x for x in date_list if x in params.ret_data.index and x in params.liquid_data.index]
    date_list.sort()
    params.all_data = params.all_data.set_index("date").sort_index()

    train_date_list, valid_date_list, test_date_list = get_train_date_split(
        fold=fold, season=season, date_list=date_list)
    train_date_list.sort(); valid_date_list.sort(); test_date_list.sort()

    if len(test_date_list) == 0:
        test_date_list = ['out_sample']
    elif market == 'ALL':
        # Drop columns where ALL values are 0 (no signal)
        all_zero_mask = (params.all_data == 0).all(axis=0)
        params.all_data = params.all_data.loc[:, ~all_zero_mask]
    else:
        raise NotImplementedError

    feature_map = list(params.all_data.columns[1:])
    params.factor_list = feature_map[:]
    try:
        params.near_limit_up_idx = feature_map.index('bias_20')
    except ValueError:
        params.near_limit_up_idx = None

    with open(rf'{save_path}/{market}{name[len(market):len(market) + 6]}-feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(rf'{factor_name}={idx}\n')

    params.factor_num = params.all_data.shape[1] - 1

    print(f"[V3.0] MVFP: num_views={params.num_views}, diversity_weight={params.diversity_weight}")
    print(f"  Training window: 4 years (extended from baseline 2 years)")
    print(f"  target_corr={params.target_corr}")
    print(f"season: {season}, fold: {fold}")
    print(f"train: {len(train_date_list)} dates:  {train_date_list} ")
    print(f"valid: {len(valid_date_list)} dates:  {valid_date_list} ")
    print(f"test: {len(test_date_list)} dates:  {test_date_list} ")

    if state == 'train':
        train_name = f"{name}/{season}/fold{fold}"
        train_single(args, train_name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError
