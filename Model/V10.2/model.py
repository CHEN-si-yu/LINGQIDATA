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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=3e-2)
    parser.add_argument('--seed', type=int, default=3253)
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'adamw'])
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
root_path = PROJECT_ROOT + 'Model/V10.2'
fac_path = PROJECT_ROOT + 'trainingdata/'
fac_name = r'fac_hybrid'
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
    # 多周期训练所需的额外标签数据
    ret_5d_data = pd.read_feather(rf"{label_path}/label_ret_5d.fea").set_index("index")
    ret_10d_data = pd.read_feather(rf"{label_path}/label_ret_10d.fea").set_index("index")
    dropout = True
    dropout_rate = 0.2
    normed_method = 'zscore'

    # V2.8: 强正则化 (融合 V1.7)
    mixup_alpha = 0.2  # Mixup Beta 分布参数

    # 多周期损失权重
    multi_horizon_weights = {'1d': 1.0, '5d': 0.5, '10d': 0.3}

    time_decay_half_life_days = 365
    time_decay_ref_date = '20260101'

    # 训练稳定性配置
    swa_enabled = True
    swa_epoch_start = 0.6
    early_stop_patience = 8
    gradient_clip_val = 1.0

    # OneCycleLR 参数
    onecycle_max_lr = 0.001
    onecycle_pct_start = 0.3
    onecycle_div_factor = 10
    onecycle_final_div_factor = 100

    # 强标签平滑 (V1.7 验证有效: IC +2.6%)
    label_smooth_noise = 0.05  # V2.8 融合 V1.7

    # 3-Fold 验证配置
    n_folds = 3
    valid_sample_ratio = 0.7  # 每个 Fold 随机抽取 70% 作为验证集


def normed_data(data, date, stage, factor_list, normed_method=params.normed_method):
    """数据标准化与特征工程。

    对输入数据执行: 有效性过滤、流动性对齐、标签计算、标准化、缺失值处理。
    """
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    # 使用 dropna 的 thresh 参数过滤因子覆盖率不足的样本
    data = data.dropna(subset=factor_list, thresh=valid_threshold).copy()
    liquid_data = params.liquid_data.loc[date]
    data['liquid'] = liquid_data.reindex(data["Code"]).values
    ret_data = params.ret_data.loc[date]
    data['Label'] = ret_data.reindex(data["Code"]).values
    # 加载 5 日和 10 日收益用于多周期训练
    ret_5d_series = params.ret_5d_data.loc[date]
    ret_10d_series = params.ret_10d_data.loc[date]
    data['Label_5d'] = ret_5d_series.reindex(data["Code"]).values
    data['Label_10d'] = ret_10d_series.reindex(data["Code"]).values
    if stage == "train":
        data['Label'] = (data['Label'] - data['Label'].mean()) / data['Label'].std()
        # 对 5 日和 10 日标签应用相同的标准化与标签平滑
        data['Label_5d'] = (data['Label_5d'] - data['Label_5d'].mean()) / data['Label_5d'].std()
        data['Label_10d'] = (data['Label_10d'] - data['Label_10d'].mean()) / data['Label_10d'].std()
        # 轻微标签平滑 —— 小方差高斯噪声可减轻过拟合
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
            return ('out_sample',) * 8
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
    """返回指定名称的损失函数。目前仅支持 'wpcc'。"""
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

    loss_fns = {'wpcc': wpcc}
    if loss not in loss_fns:
        raise ValueError(f"Unknown loss: {loss}. Supported: {list(loss_fns.keys())}")
    return loss_fns[loss]


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


# ── PredictModel（共享主干 512→256→64，3 个输出头分别预测 1d/5d/10d）─────────

class PredictModel(nn.Module):
    """MLP-Wide —— 共享主干 512→256→64，3 个输出头用于多周期联合训练"""
    def __init__(self, args=None, input_dim=None, **kwargs):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num

        # 共享层
        # 层 1: 输入投射 → 512 加宽首层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )
        # 层 2: 512 → 256
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
        # 三个输出头用于多周期预测
        self.output_1d = nn.Linear(64, 1)
        self.output_5d = nn.Linear(64, 1)
        self.output_10d = nn.Linear(64, 1)
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
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        # 三个输出头分别输出 1d/5d/10d 收益预测
        pred_1d = self.output_1d(x)
        pred_5d = self.output_5d(x)
        pred_10d = self.output_10d(x)
        return pred_1d, pred_5d, pred_10d


# ── Lightning Module ───────────────────────────────────────────────────

class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel(args)
        print(self.model)
        self.loss_fn = get_loss_fn(self.args.loss)
        # V2.8: 可学习多周期权重 (融合 V1.2)
        self.log_horizon_weights = nn.Parameter(
            torch.tensor([0.0, -0.6931, -1.2040], dtype=torch.float32))
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_times = set()

    def forward(self, tsdata):
        """推理前向传播，返回 (pred_1d, pred_5d, pred_10d)。"""
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
        # 解包 3 个标签用于多周期训练
        tsdatas, rets_1d, rets_5d, rets_10d, times, code_values, liquids, rets_1d_eval = batch
        total_loss_sum = 0.0
        for i in range(len(tsdatas)):
            tsdata = tsdatas[i]
            ret_1d = rets_1d[i].unsqueeze(1)
            ret_5d = rets_5d[i].unsqueeze(1)
            ret_10d = rets_10d[i].unsqueeze(1)

            # V2.8: Mixup 数据增强 (仅对 1d label, 融合 V1.7)
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

            # V4.1: R-Drop - 第一次前向传播
            pred_1d, pred_5d, pred_10d = self.model(tsdata_mixed)
            # V4.1: R-Drop - 第二次前向传播 (不同 dropout mask)
            pred_1d_2, pred_5d_2, pred_10d_2 = self.model(tsdata_mixed)

            # 为每个周期计算 WPCC 损失
            loss_1d = self.loss_fn(pred_1d, ret_1d_mixed)  # V2.8: Mixup label
            loss_5d = self.loss_fn(pred_5d, ret_5d)
            loss_10d = self.loss_fn(pred_10d, ret_10d)

            # V2.8: 可学习权重 (softmax 归一化, 融合 V1.2)
            learned_weights = F.softmax(self.log_horizon_weights, dim=0)
            weight_scale = 1.8
            w1, w5, w10 = (learned_weights * weight_scale).unbind()
            multi_loss = w1 * loss_1d + w5 * loss_5d + w10 * loss_10d
            # V4.1: R-Drop MSE 一致性损失 (仅对 1d 预测)
            rdrop_weight = 0.5
            consistency_loss = F.mse_loss(pred_1d, pred_1d_2)
            multi_loss = multi_loss + rdrop_weight * consistency_loss

            time_str = times[i] if isinstance(times[i], str) else str(times[i])
            time_weight = _get_time_weight(time_str)
            total_loss_sum += multi_loss * time_weight

        avg_loss = total_loss_sum / len(tsdatas)
        self.log('train_loss', avg_loss, prog_bar=True, on_step=True)
        return avg_loss

    def _evaluate_step(self, batch, batch_idx, stage):
        """评估步骤：计算日收益率和 IC。

        对每个 batch 中的日期:
        - 使用 pred_1d 选出最优股票（个人 ALL-IN 策略）
        - 计算 Pearson IC
        - 在 test 阶段保存预测结果
        """

        def get_personal_return(preds, ret):
            """Top-3 等权策略：选预测值最高的 3 只等权买入，隔日开盘卖出。

            Args:
                preds: (N, 1) 模型预测值
                ret:  (N,) 或 (N, 1) 真实隔日收益率 Ret1d

            Returns:
                float: top-3 选中股票的隔日真实收益率均值
            """
            top_k = min(3, len(preds))
            _, top_indices = torch.topk(preds.squeeze(1), top_k)
            return ret[top_indices].mean().item()

        daily_return_list = []
        ic_list = []
        # 解包全部标签（评估仅使用 ret_1d_eval）
        tsdatas, _rets_1d, _rets_5d, _rets_10d, times, code_values, liquids, rets_1d_eval = batch
        for i in range(len(tsdatas)):
            tsdata, time, code_value, liquid, ret_1d = tsdatas[i], times[i], code_values[i], liquids[i], rets_1d_eval[i]
            if isinstance(tsdata, str) and tsdata == 'out_sample':
                continue
            # 评估仅使用 pred_1d
            pred_1d, _, _ = self.forward(tsdata)
            if stage == "test":
                self.test_times.add(time)
                preds_cpu = pred_1d.detach().cpu().float().numpy()
                res = pd.DataFrame(preds_cpu, index=code_value, columns=['value'])
                res.index.name = 'Code'
                res.to_pickle(f'{params.test_save_path}/{time}.pkl')
            daily_ret = get_personal_return(pred_1d, ret_1d)
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

    def on_before_optimizer_step(self, optimizer):
        """梯度中心化 + 手动梯度裁剪 —— 兼容 bf16-mixed 精度。"""
        # Step 1: 梯度中心化 (GC) - 对权重矩阵梯度做零均值化
        for p in self.parameters():
            if p.grad is not None and p.dim() > 1:
                p.grad.data.sub_(p.grad.data.mean(dim=tuple(range(1, p.dim())), keepdim=True))
        # Step 2: 梯度裁剪
        if params.gradient_clip_val is not None and params.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=params.gradient_clip_val)

    def configure_optimizers(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]

        # OneCycleLR —— 快速探索 + 精细收敛
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
    # 梯度累积 batch_size=8 × 4 = 有效 batch 32
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


# ── 主训练入口 ───────────────────────────────────────────────────

def get_train_date_split(fold, season, date_list):
    """固定训练集划分：4 年训练 + 随机 70% 验证（3-Fold）+ 季度测试。

    验证集: 回溯 1 年，随机抽取 70% 的日期作为验证集。
    每个 Fold 使用不同的随机种子，确保 3 个 Fold 抽取不同的 70% 子集。
    训练集: 验证集开始前的 4 年数据。
    测试集: 当前季度的全部日期。
    """
    year = int(season[:4])
    q = int(season[5])
    test_start = datetime(year, (q - 1) * 3 + 1, 1)
    test_end = test_start + relativedelta(months=3)

    test_dates = [d for d in date_list
                  if test_start <= datetime.strptime(d, "%Y%m%d") < test_end]

    valid_start = test_start - relativedelta(years=1)
    valid_end = test_start
    valid_pool = [d for d in date_list
                  if valid_start <= datetime.strptime(d, "%Y%m%d") < valid_end]

    # 3-Fold: 每个 Fold 随机抽取 70% 的一年数据作为验证集
    random.seed(args.seed + fold)
    sample_size = max(1, int(len(valid_pool) * params.valid_sample_ratio))
    valid_dates = sorted(random.sample(valid_pool, sample_size))

    train_start = valid_start - relativedelta(years=4)
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
    date_list = [x for x in date_list if x in params.ret_data.index and x in params.liquid_data.index and x in params.ret_5d_data.index and x in params.ret_10d_data.index]
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

    params.factor_num = params.all_data.shape[1] - 1

    print(f"[V6.4] 3-Fold 70% Validation + R-Drop + GC Stacking (1d+5d+10d)")
    print(f"  Architecture: MLP-Wide Shared Body 512→256→64 → 3 Heads (1d/5d/10d)")
    print(f"  Gradient Centralization + OneCycleLR + GradAccum(8×4) + WD=3e-2 + SWA")
    print(f"  Valid: 1-year lookback, 70% random sample per fold ({params.n_folds}-Fold)")
    print(f"  Training window: 4 years")
    print(f"season: {season}, fold: {fold}/{params.n_folds}")
    print(f"train: {len(train_date_list)} dates:  {train_date_list} ")
    print(f"valid: {len(valid_date_list)} dates:  {valid_date_list} ")
    print(f"test:  {len(test_date_list)} dates:  {test_date_list} ")

    if state == 'train':
        train_name = f"{season}/fold{fold}"
        train_single(args, train_name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError
