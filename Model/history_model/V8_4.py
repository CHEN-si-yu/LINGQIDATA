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
from datetime import datetime
import sys
import random
from dateutil.relativedelta import relativedelta
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR

from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging,
                                         TQDMProgressBar)
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler

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
    parser.add_argument('--batch_size', type=int, default=8)            # V7.1: from V6.1, BS=8 + grad_accum=4 = eff32
    parser.add_argument('--weight_decay', type=float, default=3e-2)    # V7.1: from V6.1
    parser.add_argument('--seed', type=int, default=3253)
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--loss', default='wpcc')
    parser.add_argument('--lr', type=float, default=0.0005)            # V7.1: from V6.1

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

    args, unknown = parser.parse_known_args()
    return args


args = parse_args()

PROJECT_ROOT = "/root/autodl-fs/lingqiData/"
root_path = PROJECT_ROOT + 'Model/V8.4'
fac_path = PROJECT_ROOT + 'trainingdata/'
fac_name = r'fac20260614'
label_path = PROJECT_ROOT + 'trainingdata'
label_name = r'label_ret_1d'
liquid_path = PROJECT_ROOT + 'trainingdata'
liquid_name = r'trade_amt'


class params:
    model_path = rf'{root_path}/model_train'
    profiler_path = rf'{root_path}/logs'
    model_prefix = rf'nn'
    liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    ret_1d_data = pd.read_feather(rf"{label_path}/label_ret_1d.fea").set_index("index")
    # V7.1: Additional label data for multi-horizon training (from V6.7)
    ret_5d_data = pd.read_feather(rf"{label_path}/label_ret_5d.fea").set_index("index")
    ret_10d_data = pd.read_feather(rf"{label_path}/label_ret_10d.fea").set_index("index")
    dropout = True
    dropout_rate = 0.2
    normed_method = 'zscore'

    # V7.1: Multi-horizon loss weights (from V6.7)
    multi_horizon_weights = {'1d': 1.0, '5d': 0.5, '10d': 0.3}

    time_decay_half_life_days = 365
    time_decay_ref_date = '20260101'

    # Training stability
    swa_enabled = True
    swa_epoch_start = 0.6
    early_stop_patience = 8
    gradient_clip_val = 1.0

    # V7.1: OneCycleLR params (from V6.1)
    onecycle_max_lr = 0.001
    onecycle_pct_start = 0.3
    onecycle_div_factor = 10
    onecycle_final_div_factor = 100

    # V7.1: Light label smoothing (from V6.1)
    label_smooth_noise = 0.02


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
    # V7.1: Load 5d and 10d returns for multi-horizon training (from V6.7)
    ret_5d_series = params.ret_5d_data.loc[date]
    ret_10d_series = params.ret_10d_data.loc[date]
    data['Label_5d'] = ret_5d_series.reindex(data["Code"]).values
    data['Label_10d'] = ret_10d_series.reindex(data["Code"]).values
    if stage == "train":
        data['Label'] = (data['Label'] - data['Label'].mean()) / data['Label'].std()
        # V7.1: Apply same standardization to 5d and 10d labels (from V6.7)
        data['Label_5d'] = (data['Label_5d'] - data['Label_5d'].mean()) / data['Label_5d'].std()
        data['Label_10d'] = (data['Label_10d'] - data['Label_10d'].mean()) / data['Label_10d'].std()
        # V7.1: Light label smoothing for all three horizons
        noise_1d = np.random.normal(0, params.label_smooth_noise, size=len(data['Label']))
        noise_5d = np.random.normal(0, params.label_smooth_noise, size=len(data['Label_5d']))
        noise_10d = np.random.normal(0, params.label_smooth_noise, size=len(data['Label_10d']))
        data['Label'] = data['Label'] + noise_1d
        data['Label_5d'] = data['Label_5d'] + noise_5d
        data['Label_10d'] = data['Label_10d'] + noise_10d
    data['Label'] = data['Label'].fillna(0)
    data['Label_5d'] = data['Label_5d'].fillna(0)
    data['Label_10d'] = data['Label_10d'].fillna(0)
    # 加载真实 1 日收益（仅用于验证/测试评估，不参与训练 loss）
    ret_1d_series = params.ret_1d_data.loc[date]
    data['Ret1d'] = ret_1d_series.reindex(data["Code"]).values
    data['Ret1d'] = data['Ret1d'].fillna(0)
    code_value = data['Code'].values
    data_X = data.drop(['Code', 'Label', 'liquid', 'Ret1d', 'Label_5d', 'Label_10d'], axis=1)
    data_y_1d = data['Label']
    data_y_5d = data['Label_5d']
    data_y_10d = data['Label_10d']
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
    data_y_1d_np = np.nan_to_num(data_y_1d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_y_5d_np = np.nan_to_num(data_y_5d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_y_10d_np = np.nan_to_num(data_y_10d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_liquid_np = np.nan_to_num(data_liquid.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_ret1d_np = np.nan_to_num(data_ret1d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    return torch.from_numpy(data_x_np), \
        torch.from_numpy(data_y_1d_np), \
        torch.from_numpy(data_y_5d_np), \
        torch.from_numpy(data_y_10d_np), \
        code_value, torch.from_numpy(data_liquid_np), \
        torch.from_numpy(data_ret1d_np)


def collate_fn(datas):
    data_X, data_y_1d, data_y_5d, data_y_10d, data_time, code_value, data_liquid, data_ret1d = zip(*datas)
    return list(data_X), list(data_y_1d), list(data_y_5d), list(data_y_10d), list(data_time), list(code_value), list(data_liquid), list(data_ret1d)


class DLDataset(torch.utils.data.Dataset):
    def __init__(self, date_list, all_data, factor_list, stage='train'):
        self.date_list = date_list
        self.all_data = all_data
        self.factor_list = factor_list
        self.stage = stage
    def __getitem__(self, index):
        date = self.date_list[index]
        if date == 'out_sample':
            return 'out_sample', 'out_sample', 'out_sample', 'out_sample', 'out_sample', 'out_sample', 'out_sample', 'out_sample'
        data = self.all_data.loc[date].copy()
        data_X, data_y_1d, data_y_5d, data_y_10d, code_value, data_liquid, data_ret1d = normed_data(
            data, date, stage=self.stage, factor_list=self.factor_list)
        return data_X, data_y_1d, data_y_5d, data_y_10d, date, code_value, data_liquid, data_ret1d
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
                          persistent_workers=torch.cuda.is_available(), drop_last=False, pin_memory=True)
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
        weight.scatter_(0, argsort, weight_new.expand_as(preds).to(weight.dtype))
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


# ── PredictModel (V7.1: Wider 1024→512→128 shared body + 3 output heads 1d/5d/10d) ─────────

class FactorAttention(nn.Module):
    """V8.4: Multi-Head Attention over factors (953 factors as sequence).
    Uses 4 attention heads with d_model=64 projection to keep computation manageable.
    Includes residual connection + LayerNorm (standard Transformer block)."""
    def __init__(self, n_factors=953, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_factors = n_factors
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)  # Project each scalar factor to d_model
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.output_proj = nn.Linear(d_model, 1)  # Project back to scalar per factor

    def forward(self, x):
        # x shape: (batch, n_factors)
        b = x.shape[0]
        # Reshape to (batch, n_factors, 1) then project to d_model
        x = x.unsqueeze(-1)  # (B, F, 1)
        x = self.input_proj(x)  # (B, F, d_model)
        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # (B, F, d_model)
        x = self.norm1(x + attn_out)  # Residual + Norm
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual + Norm
        # Project back to scalar per factor
        x = self.output_proj(x).squeeze(-1)  # (B, F)
        return x



class PredictModel(nn.Module):
    """MLP-Wider + Multi-Horizon + FactorAttention -- V8.4: 1024→512→128 shared body, 3 output heads, factor attention, uncertainty-weighted loss"""
    def __init__(self, args=None, input_dim=None, **kwargs):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num

        # V8.4: Multi-Head Factor Attention — learn factor interactions before MLP
        # Projects each of F factors to d_model=64, applies 4-head self-attention,
        # then projects back to scalar per factor. Residual + LayerNorm at each stage.
        self.factor_attn = FactorAttention(n_factors=input_dim, d_model=64, n_heads=4, dropout=0.1)

        # V7.1: Shared layers — wider architecture from V6.5 (1024→512→128)
        # V8.4: MLP input is [original_factors | attended_factors] = 2 * input_dim
        mlp_input_dim = input_dim * 2
        # 层 1: 输入投射 → 1024
        self.input_layer = nn.Sequential(
            nn.Linear(mlp_input_dim, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )
        # 层 2: 1024 → 512
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 层 3: 512 → 128 (shared embedding)
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        # V7.1: Three output heads for multi-horizon prediction (from V6.7)
        self.output_1d = nn.Linear(128, 1)
        self.output_5d = nn.Linear(128, 1)
        self.output_10d = nn.Linear(128, 1)
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

    def forward(self, tsdata):
        x = tsdata.float()
        # V8.4: Apply factor attention — learn cross-factor interactions
        attended = self.factor_attn(x)  # (B, F)
        # Concatenate original factors with attention-augmented factors
        x = torch.cat([x, attended], dim=1)  # (B, 2*F)
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        # V7.1: Three output heads for 1d, 5d, 10d returns
        pred_1d = self.output_1d(x)
        pred_5d = self.output_5d(x)
        pred_10d = self.output_10d(x)
        return pred_1d, pred_5d, pred_10d

class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel(args)
        print(self.model)
        self.loss_fn = get_loss_fn(self.args.loss)
        # V7.2: Learnable log-variances for uncertainty-weighted multi-task (Kendall et al. 2018)
        self.log_sigmas = nn.Parameter(torch.zeros(3))  # (1d, 5d, 10d)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_times = set()

