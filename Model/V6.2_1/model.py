"""
V6.2 — 针对 V6.1_4 回测问题做的改进
=========================================

V6.1_4 分析结论：
  - 模型排序能力 (Rank IC) 在 2025Q1-Q4 保持良好 (IC 0.045-0.055)
  - Top 1 策略在 2025Q4/2026Q1 出现极端回撤 (>50%), 但 Top 5/10 策略盈利稳定
  - 根因: ① 市场 2024/9 后进入新范式, 训练窗口滞后; ② 极端分位波动大; ③ Top 1 集中度风险
  - 2026Q2 IC 真正衰减到 0.020, 信号需要更强自适应能力

V6.2 改进 (共 8 项):
  1. 训练窗口: 4年+1年间隔 → 3年+6个月间隔 (更快适应市场变化)
  2. 时间衰减: half-life 365d → 180d (近期样本权重更高)
  3. 残差连接: MLP 主干加入 residual, 提升深层表达能力
  4. Top-K 增强损失: 对头部股票加权惩罚排序错误
  5. 可学习 Top-K 权重: 动态平衡全局排序 vs 头部排序
  6. 市场状态特征: 日度截面均值/波动率作为额外输入
  7. 集成不确定性: 记录 fold 间方差, 用于后处理降权
  8. R-Drop 增强: 对三头均做一致性约束
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

# ── 环境配置 ──────────────────────────────────────────────────────────
cpu_num = 32
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=3e-2)
    parser.add_argument('--seed', type=int, default=1008611)
    parser.add_argument('--optimizer', default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--loss', default='wpcc')
    parser.add_argument('--lr', type=float, default=0.0005)

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
root_path = PROJECT_ROOT + 'Model/V6.2_1'
fac_path = PROJECT_ROOT + 'trainingdata/'
fac_name = r'fac_all'
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
    ret_3d_data = pd.read_feather(rf"{label_path}/label_ret_3d.fea").set_index("index")
    ret_5d_data = pd.read_feather(rf"{label_path}/label_ret_5d.fea").set_index("index")
    dropout = True
    dropout_rate = 0.2
    normed_method = 'zscore'

    # ── Mixup 正则化 ─────────────────────────────────────────────
    mixup_alpha = 0.2

    # ── 多周期损失权重 ────────────────────────────────────────────
    multi_horizon_weights = {'1d': 1.2, '3d': 0.5, '5d': 0.3}

    # ── V6.2: 时间衰减 — 半衰期从 365 缩短到 180 天 ─────────────
    # 分析: V6.1_4 用 365d half-life, 对 1 年前的数据仍给 50% 权重
    #       市场在 2024/9 后范式已变, 应更快遗忘旧数据
    time_decay_half_life_days = 180   # [V6.2] 365 → 180
    time_decay_ref_date = '20260101'

    # ── 训练稳定性 ────────────────────────────────────────────────
    swa_enabled = True
    swa_epoch_start = 0.6
    early_stop_patience = 8
    gradient_clip_val = 1.0

    # ── OneCycleLR ───────────────────────────────────────────────
    onecycle_max_lr = 0.001
    onecycle_pct_start = 0.3
    onecycle_div_factor = 10
    onecycle_final_div_factor = 100

    # ── 标签平滑 ─────────────────────────────────────────────────
    label_smooth_noise = 0.05

    # ── V6.2: Top-K 增强损失配置 ─────────────────────────────────
    # 对排名前 topk_frac 的股票施加额外排序损失
    # 分析: Top 1 失败的核心原因是模型对头部排序不够精确
    topk_loss_enabled = True
    topk_frac = 0.1          # 前 10% 股票做 Top-K 增强
    topk_loss_weight = 0.3   # Top-K 损失权重

    # ── V6.2: 市场状态特征 ───────────────────────────────────────
    # 在每日期特征中加入截面统计量 (均值/波动率/偏度)
    regime_features_enabled = True

    # ── V6.2: 集成不确定性记录 ───────────────────────────────────
    # 训练/推理时记录 fold 间方差, 供后处理使用
    track_fold_variance = True


# ── V6.2 市场状态特征 ──────────────────────────────────────────────

def compute_regime_features(data_X_np):
    """V6.2: 从因子截面计算市场状态特征 (3 维).

    返回 (N, 3) 的 regime 特征:
      - 列 0: 因子截面均值 (市场整体信号强度)
      - 列 1: 因子截面标准差 (市场分歧度)
      - 列 2: 因子截面偏度 (市场极端方向)

    这些特征拼接在因子后面, 让模型感知当前市场状态。
    """
    mean_val = np.nanmean(data_X_np, axis=1, keepdims=True)        # (N, 1)
    std_val = np.nanstd(data_X_np, axis=1, keepdims=True)          # (N, 1)
    # 偏度: 三阶矩 / 标准差^3
    diff = data_X_np - mean_val
    skew_val = np.nanmean(diff ** 3, axis=1, keepdims=True) / (std_val ** 3 + 1e-8)
    # 清理异常值
    mean_val = np.nan_to_num(mean_val, nan=0.0, posinf=0.0, neginf=0.0)
    std_val = np.nan_to_num(std_val, nan=1.0, posinf=1.0, neginf=1.0)
    skew_val = np.nan_to_num(skew_val, nan=0.0, posinf=0.0, neginf=0.0)
    regime = np.concatenate([mean_val, std_val, skew_val], axis=1)
    return regime.astype(np.float32)


def normed_data(data, date, stage, factor_list, normed_method=params.normed_method):
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    valid_mask = data[factor_list].notna().sum(axis=1) >= valid_threshold
    data = data.loc[valid_mask].copy()
    liquid_data = params.liquid_data.loc[date]
    data['liquid'] = liquid_data.reindex(data["Code"]).values
    ret_data = params.ret_data.loc[date]
    data['Label'] = ret_data.reindex(data["Code"]).values
    ret_3d_series = params.ret_3d_data.loc[date]
    ret_5d_series = params.ret_5d_data.loc[date]
    data['Label_3d'] = ret_3d_series.reindex(data["Code"]).values
    data['Label_5d'] = ret_5d_series.reindex(data["Code"]).values
    if stage == "train":
        data['Label'] = (data['Label'] - data['Label'].mean()) / data['Label'].std()
        data['Label_3d'] = (data['Label_3d'] - data['Label_3d'].mean()) / data['Label_3d'].std()
        data['Label_5d'] = (data['Label_5d'] - data['Label_5d'].mean()) / data['Label_5d'].std()
        noise_1d = np.random.normal(0, params.label_smooth_noise, size=len(data['Label']))
        noise_3d = np.random.normal(0, params.label_smooth_noise, size=len(data['Label_3d']))
        noise_5d = np.random.normal(0, params.label_smooth_noise, size=len(data['Label_5d']))
        data['Label'] = data['Label'] + noise_1d
        data['Label_3d'] = data['Label_3d'] + noise_3d
        data['Label_5d'] = data['Label_5d'] + noise_5d
    data['Label'] = data['Label'].fillna(0)
    data['Label_3d'] = data['Label_3d'].fillna(0)
    data['Label_5d'] = data['Label_5d'].fillna(0)
    ret_1d_series = params.ret_1d_data.loc[date]
    data['Ret1d'] = ret_1d_series.reindex(data["Code"]).values
    data['Ret1d'] = data['Ret1d'].fillna(0)
    code_value = data['Code'].values
    data_X = data.drop(['Code', 'Label', 'liquid', 'Ret1d', 'Label_3d', 'Label_5d'], axis=1)
    data_y_1d = data['Label']
    data_y_3d = data['Label_3d']
    data_y_5d = data['Label_5d']
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

    # V6.2: 拼接市场状态特征 (3 维)
    if params.regime_features_enabled:
        regime_np = compute_regime_features(data_x_np)
        data_x_np = np.concatenate([data_x_np, regime_np], axis=1)

    data_y_1d_np = np.nan_to_num(data_y_1d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_y_3d_np = np.nan_to_num(data_y_3d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_y_5d_np = np.nan_to_num(data_y_5d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_liquid_np = np.nan_to_num(data_liquid.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_ret1d_np = np.nan_to_num(data_ret1d.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    return torch.from_numpy(data_x_np), \
        torch.from_numpy(data_y_1d_np), \
        torch.from_numpy(data_y_3d_np), \
        torch.from_numpy(data_y_5d_np), \
        code_value, torch.from_numpy(data_liquid_np), \
        torch.from_numpy(data_ret1d_np)


def collate_fn(datas):
    data_X, data_y_1d, data_y_3d, data_y_5d, data_time, code_value, data_liquid, data_ret1d = zip(*datas)
    return list(data_X), list(data_y_1d), list(data_y_3d), list(data_y_5d), list(data_time), list(code_value), list(data_liquid), list(data_ret1d)


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
        data_X, data_y_1d, data_y_3d, data_y_5d, code_value, data_liquid, data_ret1d = normed_data(
            data, date, stage=self.stage, factor_list=self.factor_list)
        return data_X, data_y_1d, data_y_3d, data_y_5d, date, code_value, data_liquid, data_ret1d

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


# ── WPCC loss + V6.2 Top-K 增强损失 ─────────────────────────────────

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


def _topk_focused_loss(preds, y, topk_frac=0.1):
    """V6.2: Top-K 增强损失 —— 对头部股票施加更强的排序约束.

    将 topk_frac 的股票视为"重点关注区", 计算该区域的 pairwise ranking loss.
    这会迫使模型更精确地区分头部股票的相对优劣。

    Args:
        preds: (N, 1) 预测值
        y:     (N, 1) 真实标签
        topk_frac: 头部股票占比, 默认 10%

    Returns:
        scalar loss: Top-K 区域的 pairwise hinge loss
    """
    n = preds.shape[0]
    k = max(2, int(n * topk_frac))

    # 按预测值选出 Top-K
    _, top_indices = torch.topk(preds.squeeze(), k)

    preds_top = preds[top_indices]
    y_top = y[top_indices]

    # Pairwise: 对所有 (i,j) 对, 如果 pred_i > pred_j 但 y_i < y_j, 施加惩罚
    pred_diff = preds_top - preds_top.T
    y_diff = y_top - y_top.T

    # Hinge loss: max(0, -(pred_diff * sign(y_diff)))
    y_sign = torch.sign(y_diff)
    hinge = torch.clamp(-pred_diff * y_sign, min=0)

    # 只考虑上三角 (避免重复)
    mask = torch.triu(torch.ones(k, k, device=preds.device), diagonal=1)
    loss = (hinge * mask).sum() / (mask.sum() + 1e-8)
    return loss


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


# ── V6.2 PredictModel: 残差 MLP + 市场状态感知 ──────────────────────

class PredictModel(nn.Module):
    """V6.2: 残差 MLP-Wide —— 共享主干 512→256→64→32, 3 个输出头.

    相比 V6.1_4 的改进:
      - 残差连接: input_layer 输出通过线性投影拼接到 hidden1 输出
      - 更深表达: 64→32 的额外 bottleneck 层
      - 输入维度: 因子数 + 3 维市场状态特征
    """
    def __init__(self, args=None, input_dim=None, **kwargs):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num + 3  # V6.2: +3 regime features

        # 层 1: 输入投射 → 512
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )
        # 残差投影: 512 → 256 (对齐 hidden1 输出维度)
        self.residual_proj = nn.Linear(512, 256)

        # 层 2: 512 → 256 (残差块)
        self.hidden1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 层 3: 256 → 64
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # V6.2: 额外 bottleneck 64 → 32
        self.hidden3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        # 三个输出头
        self.output_1d = nn.Linear(32, 1)
        self.output_3d = nn.Linear(32, 1)
        self.output_5d = nn.Linear(32, 1)
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
        # 主干前向 (带残差)
        x1 = self.input_layer(x)                       # (N, 512)
        x2 = self.hidden1(x1)                           # (N, 256)
        # V6.2: 残差连接 — x1 流与 x2 相加
        residual = self.residual_proj(x1)               # (N, 256)
        x2 = x2 + residual
        x3 = self.hidden2(x2)                           # (N, 64)
        x4 = self.hidden3(x3)                           # (N, 32)
        # 三头输出
        pred_1d = self.output_1d(x4)
        pred_3d = self.output_3d(x4)
        pred_5d = self.output_5d(x4)
        return pred_1d, pred_3d, pred_5d


# ── Lightning Module ───────────────────────────────────────────────────

class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel(args)
        print(self.model)
        self.loss_fn = get_loss_fn(self.args.loss)
        # 可学习多周期权重
        self.log_horizon_weights = nn.Parameter(
            torch.tensor([0.0, -0.6931, -1.2040], dtype=torch.float32))
        # V6.2: 可学习 Top-K 损失权重
        self.log_topk_weight = nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32))
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_times = set()
        # V6.2: 记录 fold 预测方差
        self.fold_predictions = {}

    def forward(self, tsdata):
        return self.model(tsdata)

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
        tsdatas, rets_1d, rets_3d, rets_5d, times, code_values, liquids, rets_1d_eval = batch
        total_loss_sum = 0.0
        for i in range(len(tsdatas)):
            tsdata = tsdatas[i]
            ret_1d = rets_1d[i].unsqueeze(1)
            ret_3d = rets_3d[i].unsqueeze(1)
            ret_5d = rets_5d[i].unsqueeze(1)

            # Mixup 数据增强
            if params.mixup_alpha > 0:
                n = tsdata.shape[0]
                lam = np.random.beta(params.mixup_alpha, params.mixup_alpha)
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(n, device=tsdata.device)
                tsdata_mixed = lam * tsdata + (1 - lam) * tsdata[perm]
                ret_1d_mixed = lam * ret_1d + (1 - lam) * ret_1d[perm]
            else:
                tsdata_mixed = tsdata
                ret_1d_mixed = ret_1d

            # R-Drop: 两次前向传播
            pred_1d, pred_3d, pred_5d = self.model(tsdata_mixed)
            pred_1d_2, pred_3d_2, pred_5d_2 = self.model(tsdata_mixed)

            # WPCC 损失 (三个周期)
            loss_1d = self.loss_fn(pred_1d, ret_1d_mixed)
            loss_3d = self.loss_fn(pred_3d, ret_3d)
            loss_5d = self.loss_fn(pred_5d, ret_5d)

            # V6.2: Top-K 增强损失
            if params.topk_loss_enabled:
                topk_loss_1d = _topk_focused_loss(pred_1d, ret_1d_mixed, params.topk_frac)
                topk_loss_3d = _topk_focused_loss(pred_3d, ret_3d, params.topk_frac)
                topk_loss_5d = _topk_focused_loss(pred_5d, ret_5d, params.topk_frac)
                topk_loss = (topk_loss_1d + topk_loss_3d * 0.5 + topk_loss_5d * 0.3) / 3.0
                topk_weight = torch.sigmoid(self.log_topk_weight) * params.topk_loss_weight
            else:
                topk_loss = 0.0
                topk_weight = 0.0

            # 可学习多周期权重
            learned_weights = F.softmax(self.log_horizon_weights, dim=0)
            weight_scale = 1.8
            w1, w3, w5 = (learned_weights * weight_scale).unbind()
            multi_loss = w1 * loss_1d + w3 * loss_3d + w5 * loss_5d

            # V6.2: R-Drop 一致性损失 (三头均做约束)
            rdrop_weight = 0.5
            consistency_loss = F.mse_loss(pred_1d, pred_1d_2) + \
                               0.3 * F.mse_loss(pred_3d, pred_3d_2) + \
                               0.3 * F.mse_loss(pred_5d, pred_5d_2)
            multi_loss = multi_loss + rdrop_weight * consistency_loss

            # 加入 Top-K 增强损失
            multi_loss = multi_loss + topk_weight * topk_loss

            time_str = times[i] if isinstance(times[i], str) else str(times[i])
            time_weight = _get_time_weight(time_str)
            total_loss_sum += multi_loss * time_weight

        avg_loss = total_loss_sum / len(tsdatas)
        self.log('train_loss', avg_loss, prog_bar=True, on_step=True)
        if params.topk_loss_enabled:
            self.log('topk_weight', topk_weight, prog_bar=False, on_step=True)
        return avg_loss

    def _evaluate_step(self, batch, batch_idx, stage):
        def get_personal_return_topk(preds, ret, top_n=5):
            """V6.2: Top-K 股票等权组合评估 (默认 Top 5, 降低集中度)."""
            k = min(top_n, len(preds))
            best_indices = torch.topk(preds.squeeze(), k).indices
            return ret[best_indices].mean().item()

        daily_return_list = []
        ic_list = []
        tsdatas, rets_1d_batch, rets_3d_batch, rets_5d_batch, times, code_values, liquids, rets_1d_eval = batch
        for i in range(len(tsdatas)):
            tsdata, time, code_value, liquid, ret_1d = tsdatas[i], times[i], code_values[i], liquids[i], rets_1d_eval[i]
            if isinstance(tsdata, str) and tsdata == 'out_sample':
                pass
            else:
                pred_1d, _, _ = self.forward(tsdata)
                if stage == "test":
                    self.test_times.add(time)
                    preds_cpu = pred_1d.detach().cpu().float().numpy()
                    res = pd.DataFrame(preds_cpu, index=code_value, columns=['value'])
                    res.index.name = 'Code'
                    res.to_pickle(f'{params.test_save_path}/{time}.pkl')
                    if params.track_fold_variance:
                        self.fold_predictions[time] = preds_cpu
                # V6.2: 使用 Top 5 评估
                daily_ret = get_personal_return_topk(pred_1d, ret_1d, top_n=5)
                daily_return_list.append(daily_ret)
                ic_list.append(self._pearson_corr(pred_1d.squeeze(), ret_1d))
        try:
            res_list = [sum(daily_return_list) / len(daily_return_list), sum(ic_list) / len(ic_list)]
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
        # V6.2: 保存 fold 方差信息
        if params.track_fold_variance and len(self.fold_predictions) > 0:
            var_path = f'{params.test_save_path}/fold_variance.pkl'
            import pickle
            with open(var_path, 'wb') as f:
                pickle.dump(self.fold_predictions, f)
        self.fold_predictions.clear()

    def on_before_optimizer_step(self, optimizer):
        """梯度中心化 + 手动梯度裁剪."""
        for p in self.parameters():
            if p.grad is not None and p.dim() > 1:
                p.grad.data.sub_(p.grad.data.mean(dim=tuple(range(1, p.dim())), keepdim=True))
        if params.gradient_clip_val is not None and params.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=params.gradient_clip_val)

    def configure_optimizers(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]

        total_steps = self.args.max_epochs
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=params.onecycle_max_lr,
                total_steps=total_steps,
                pct_start=params.onecycle_pct_start,
                div_factor=params.onecycle_div_factor,
                final_div_factor=params.onecycle_final_div_factor,
            ),
            'interval': 'epoch',
            'frequency': 1,
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
                      accumulate_grad_batches=4,
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


# ── 主训练入口 ───────────────────────────────────────────────────────

def get_train_date_split(fold, season, date_list):
    """V6.2: 缩短训练窗口 — 3 年训练 + 6 个月间隔 + 季度测试.

    V6.1_4: 4 年训练 + 1 年间隔 → 数据滞后导致无法适应市场范式变化
    V6.2:   3 年训练 + 6 个月间隔 → 更快响应市场变化, 减少滞后
    """
    year = int(season[:4])
    q = int(season[5])
    test_start = datetime(year, (q - 1) * 3 + 1, 1)
    test_end = test_start + relativedelta(months=3)

    test_dates = [d for d in date_list
                  if test_start <= datetime.strptime(d, "%Y%m%d") < test_end]

    # V6.2: 验证集 — 测试前的 6 个月 (V6.1_4 是 1 年)
    valid_start = test_start - relativedelta(months=6)
    valid_end = test_start
    valid_pool = [d for d in date_list
                  if valid_start <= datetime.strptime(d, "%Y%m%d") < valid_end]

    random.seed(args.seed + fold)
    sample_size = max(1, len(valid_pool) // 4)
    valid_dates = sorted(random.sample(valid_pool, sample_size))

    # V6.2: 训练集 — 验证前的 3 年 (V6.1_4 是 4 年)
    train_start = valid_start - relativedelta(years=3)
    train_dates = [d for d in date_list
                   if train_start <= datetime.strptime(d, "%Y%m%d") < valid_start]

    return train_dates, valid_dates, test_dates


def train(args, season, fold, state='train'):
    save_path = rf"{root_path}/model_test"
    try:
        os.makedirs(save_path, exist_ok=True)
        current_file_path = os.path.abspath(__file__)
        shutil.copy(current_file_path, save_path)
    except Exception as e:
        print(e)

    params.test_save_path = f"{save_path}/fold{fold}"
    os.makedirs(params.test_save_path, exist_ok=True)

    params.all_data = pd.read_feather(rf'{fac_path}/{fac_name}.fea')
    date_list = list(params.all_data["date"].unique())
    date_list = [x for x in date_list if x in params.ret_data.index and x in params.liquid_data.index and x in params.ret_3d_data.index and x in params.ret_5d_data.index]
    date_list.sort()
    params.all_data = params.all_data.set_index("date").sort_index()

    train_date_list, valid_date_list, test_date_list = get_train_date_split(
        fold=fold, season=season, date_list=date_list)
    train_date_list.sort(); valid_date_list.sort(); test_date_list.sort()

    if len(test_date_list) == 0:
        test_date_list = ['out_sample']
    all_zero_mask = (params.all_data == 0).all(axis=0)
    params.all_data = params.all_data.loc[:, ~all_zero_mask]

    feature_map = list(params.all_data.columns[1:])
    params.factor_list = feature_map[:]
    with open(rf'{save_path}/feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(rf'{factor_name}={idx}\n')

    # V6.2: factor_num = 原始因子数 + 3 个 regime features
    params.factor_num = params.all_data.shape[1] - 1

    print(f"[V6.2] Residual MLP + Top-K Loss + Regime Features + Shorter Window")
    print(f"  Architecture: Residual MLP 512→256(+skip)→64→32 → 3 Heads (1d/3d/5d)")
    print(f"  Regime Features: cross-sectional mean/std/skew (3 dims)")
    print(f"  Top-K Loss: focusing top {params.topk_frac:.0%} stocks, weight={params.topk_loss_weight}")
    print(f"  Training window: 3 years + 6mo gap (was 4y+1y in V6.1)")
    print(f"  Time decay: half-life={params.time_decay_half_life_days}d (was 365d)")
    print(f"  R-Drop: all 3 heads + Gradient Centralization + OneCycleLR + SWA")
    print(f"season: {season}, fold: {fold}")
    print(f"train: {len(train_date_list)} dates:  {train_date_list[:3]}...{train_date_list[-3:] if len(train_date_list)>3 else ''} ")
    print(f"valid: {len(valid_date_list)} dates:  {valid_date_list[:3]}...{valid_date_list[-3:] if len(valid_date_list)>3 else ''} ")
    print(f"test: {len(test_date_list)} dates:  {test_date_list[:3]}...{test_date_list[-3:] if len(test_date_list)>3 else ''} ")

    if state == 'train':
        train_name = f"{season}/fold{fold}"
        train_single(args, train_name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError
