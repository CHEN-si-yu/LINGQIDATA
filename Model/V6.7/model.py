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
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--seed', type=int, default=3253)
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--loss', default='wpcc')
    parser.add_argument('--lr', type=float, default=0.0005)

    parser.add_argument('--max_epochs', type=int, default=80)
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

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
root_path = PROJECT_ROOT + 'Model/V6.7'
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
    # V6.1 (学 V5.2): 多周期联合训练 —— 1d 为主目标, 5d/10d 为辅助目标
    # 评估/checkpoint 选择/回测统一用 1d 口径 (Ret1d = label_ret_1d 原始值, 仅评估)
    ret_1d_data = pd.read_feather(rf"{label_path}/label_ret_1d.fea").set_index("index")
    ret_5d_data = pd.read_feather(rf"{label_path}/label_ret_5d.fea").set_index("index")
    ret_10d_data = pd.read_feather(rf"{label_path}/label_ret_10d.fea").set_index("index")
    dropout = True
    dropout_rate = 0.2
    normed_method = 'zscore'

    # 标签平滑噪声 (V1.7 验证有效: IC +2.6%), 只加在训练标签上
    label_smooth_noise = 0.05

    # V6.1: 关闭 Mixup —— 截面选股任务中混合两只股票的因子向量会制造不存在的合成样本,
    # 破坏排序结构; 实测 1d 目标 IC≈0.02 的低信号场景下正则化收益有限, 改为 0
    mixup_alpha = 0.0

    # V6.2: 关闭时间衰减 —— V6.1 的 365d 半衰期使 2020 年数据权重仅 1.6%,
    # 有效训练数据≈2024-2025 两年; 弱信号场景下应使用全部 6.5 年数据
    time_decay_half_life_days = None
    time_decay_ref_date = '20260101'

    # V6.2: T+1 开盘可买入掩码 —— 一字涨停(T+1)无法买入的股票, label 不可实现,
    # 从训练 loss 与验证 IC 中剔除 (消除纸面 IC 虚高 + 减少 label 噪声)
    buyable_mask = pd.read_feather(rf"{label_path}/buyable_mask.fea").set_index("date")

    # V6.2: label 缩尾阈值 (MAD 倍数), 剔除 daily_adj 伪复权产生的垃圾极端值
    label_winsor_mad = 5.0

    # 训练稳定性配置
    swa_enabled = True
    swa_epoch_start = 0.2   # V6.2: 0.6 时 SWA 从未激活(早停在 ep14-20); 改为 0.15
    early_stop_patience = 12
    gradient_clip_val = 1.0

    # R-Drop MSE 一致性损失权重 (V6.2: 0.5 → 0.1, MSE 尺度与排序损失不协调)
    rdrop_weight = 0.25

    # OneCycleLR 参数 (V6.2: pct_start 0.3→0.1, 峰值提前到 ep6, 早停前完成退火段)
    onecycle_max_lr = 0.0002
    onecycle_pct_start = 0.3
    onecycle_div_factor = 10
    onecycle_final_div_factor = 100

    # K-Fold 验证配置: 验证池一次固定种子洗牌后均分为 K 份, 每 Fold 取 1/K (两两无交集且全覆盖)
    n_folds = 4

    # 数据划分配置: 最新 valid_size 个交易日作为验证集池, 验证集前隔断 gap_days 个交易日
    valid_size = 240  # 验证集池大小 N (最新 N 个交易日)
    gap_days = 10     # 训练集与验证集之间的隔断交易日数 (防泄漏)

    # 数据截止日期: 只使用 <= 该日期的数据训练与选 checkpoint (None = 用全部数据)。
    # 设 '20260630' 时, 20260701+ 完全不在训练/验证/选择流程中, 回测才是真实样本外。
    # 注意: 设了截止日期后, 该日期之后(含)的新数据需要在截止后重训才能用于预测。
    data_end_date = None


def _safe_label_row(label_df, date):
    """取标签行; 日期不在索引中(标签尾部按持仓周期截断, 如 10d 缺最后 11 个交易日)时
    返回全 NaN 行, 由后续 fillna(0) 处理 —— 缺失日期的标签本就不存在, 置 0 是正确语义。"""
    if date in label_df.index:
        return label_df.loc[date]
    return pd.Series(np.nan, index=label_df.columns)


def normed_data(data, date, stage, factor_list, normed_method=params.normed_method):
    """数据标准化与特征工程。

    对输入数据执行: 有效性过滤、流动性对齐、标签计算、标准化、缺失值处理。
    """
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    # 使用 dropna 的 thresh 参数过滤因子覆盖率不足的样本
    data = data.dropna(subset=factor_list, thresh=valid_threshold).copy()
    liquid_data = params.liquid_data.loc[date]
    data['liquid'] = liquid_data.reindex(data["Code"]).values
    ret_data = _safe_label_row(params.ret_data, date)
    data['Label'] = ret_data.reindex(data["Code"]).values
    # 辅助目标标签 (5d/10d)
    ret_5d_series = _safe_label_row(params.ret_5d_data, date)
    data['Label_5d'] = ret_5d_series.reindex(data["Code"]).values
    ret_10d_series = _safe_label_row(params.ret_10d_data, date)
    data['Label_10d'] = ret_10d_series.reindex(data["Code"]).values
    # V6.2: T+1 开盘不可买入的股票 → 三个 horizon 的 label 全部置 NaN (从 loss/IC 剔除)
    buyable_row = _safe_label_row(params.buyable_mask, date)
    data['buyable'] = buyable_row.reindex(data["Code"]).values
    not_buyable = ~(data['buyable'].fillna(False).astype(bool))
    if not_buyable.any():
        data.loc[not_buyable, ['Label', 'Label_5d', 'Label_10d']] = np.nan
    # V6.2: label 按 MAD 缩尾 (剔除 daily_adj 伪复权产生的垃圾极端值), 仅对有限值
    def _winsor_mad(s, k=params.label_winsor_mad):
        v = s.dropna()
        if len(v) == 0:
            return s
        med = v.median()
        mad = (v - med).abs().median()
        if mad <= 1e-12:
            return s
        return s.clip(med - k * mad, med + k * mad)
    data['Label'] = _winsor_mad(data['Label'])
    data['Label_5d'] = _winsor_mad(data['Label_5d'])
    data['Label_10d'] = _winsor_mad(data['Label_10d'])
    if stage == "train":
        # V6.2: NaN 感知 z-score (停牌/不可买入样本不参与截面统计)
        for col in ['Label', 'Label_5d', 'Label_10d']:
            m = data[col].mean(skipna=True)
            s = data[col].std(skipna=True)
            data[col] = (data[col] - m) / s if s and s > 0 else 0.0
        # 标签平滑噪声 (V1.7): 只加训练标签, 缓解截面噪声过拟合
        if params.label_smooth_noise > 0:
            for col in ['Label', 'Label_5d', 'Label_10d']:
                finite = data[col].notna()
                noise = np.random.normal(0, params.label_smooth_noise, size=int(finite.sum()))
                data.loc[finite, col] = data[col][finite] + noise
    data['Label'] = data['Label'].fillna(0)
    data['Label_5d'] = data['Label_5d'].fillna(0)
    data['Label_10d'] = data['Label_10d'].fillna(0)
    # 加载真实 1 日收益（仅用于验证/测试评估，不参与训练 loss）
    ret_1d_series = _safe_label_row(params.ret_1d_data, date)
    data['Ret1d'] = ret_1d_series.reindex(data["Code"]).values
    data['Ret1d'] = data['Ret1d'].fillna(0)
    # V6.2: 评估用双口径 —— Ret1d_raw 与 V6.1 同口径(全池, fillna0); Ret1d_trad 剔除不可买入
    data['Ret1d_raw'] = data['Ret1d']
    if not_buyable.any():
        data.loc[not_buyable, 'Ret1d'] = np.nan
    code_value = data['Code'].values
    data_X = data.drop(['Code', 'Label', 'Label_5d', 'Label_10d', 'liquid', 'Ret1d',
                        'Ret1d_raw', 'buyable'], axis=1)
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
    data_ret1d_raw = data['Ret1d_raw']
    data_x_np = np.nan_to_num(data_X.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    # V6.2: label 保留 NaN (不可买入/停牌样本在 loss 与 IC 中被过滤, 不再 fillna(0) 参与训练)
    data_y_1d_np = data_y_1d.to_numpy(dtype=np.float32, copy=False)
    data_y_5d_np = data_y_5d.to_numpy(dtype=np.float32, copy=False)
    data_y_10d_np = data_y_10d.to_numpy(dtype=np.float32, copy=False)
    data_liquid_np = np.nan_to_num(data_liquid.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_ret1d_np = data_ret1d.to_numpy(dtype=np.float32, copy=False)          # 可交易池口径 (含 NaN)
    data_ret1d_raw_np = data_ret1d_raw.to_numpy(dtype=np.float32, copy=False)  # V6.1 同口径 (全池, fillna0)

    return torch.from_numpy(data_x_np), \
        torch.from_numpy(data_y_1d_np), \
        torch.from_numpy(data_y_5d_np), \
        torch.from_numpy(data_y_10d_np), \
        code_value, torch.from_numpy(data_liquid_np), \
        torch.from_numpy(data_ret1d_np), \
        torch.from_numpy(data_ret1d_raw_np)


def collate_fn(datas):
    data_X, data_y_1d, data_y_5d, data_y_10d, data_time, code_value, data_liquid, data_ret1d, data_ret1d_raw = zip(*datas)
    return list(data_X), list(data_y_1d), list(data_y_5d), list(data_y_10d), \
        list(data_time), list(code_value), list(data_liquid), list(data_ret1d), list(data_ret1d_raw)


class DLDataset(torch.utils.data.Dataset):
    def __init__(self, date_list, all_data, factor_list, stage='train'):
        self.date_list = date_list
        self.all_data = all_data
        self.factor_list = factor_list
        self.stage = stage

    def __getitem__(self, index):
        date = self.date_list[index]
        if date == 'out_sample':
            return ('out_sample',) * 9
        data = self.all_data.loc[date].copy()
        data_X, data_y_1d, data_y_5d, data_y_10d, code_value, data_liquid, data_ret1d, data_ret1d_raw = normed_data(
            data, date, stage=self.stage, factor_list=self.factor_list)
        return data_X, data_y_1d, data_y_5d, data_y_10d, date, code_value, data_liquid, data_ret1d, data_ret1d_raw

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
    if half_life_days is None:  # V6.2: 时间衰减关闭, 全部数据等权
        return 1.0
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


# ── PredictModel（共享主干 512→256→64，单输出头预测 1d 收益）─────────

class PredictModel(nn.Module):
    """MLP-Wide —— 共享主干 512→256→64，单输出头仅预测 1 日收益"""
    def __init__(self, args=None, input_dim=None, **kwargs):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num

        # 共享层 (V6.3: 加宽 1024→512→256→64, 更强特征解耦能力)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.35)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # V6.1 (学 V5.2): 三个输出头用于多周期联合训练 (1d 为主, 5d/10d 辅助)
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
        x = self.hidden3(x)
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
        # V6.1 (学 V5.2 V2.8): 多周期损失权重 softmax 参数化。
        # 初始 softmax([0,-0.6931,-1.2040])×1.8 = {1d:1.0, 5d:0.5, 10d:0.3}。
        # 注: 优化器只更新 self.model 参数 (与 V5.2 相同), 该权重实际保持初始值 1.0/0.5/0.3,
        # 保证 1d 主目标不被辅助头稀释。
        self.log_horizon_weights = nn.Parameter(
            torch.tensor([0.0, 0.5108, 1.2039], dtype=torch.float32))
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
        # 解包 3 个标签用于多周期训练 (评估用 rets_1d_eval = 可交易池 1d label, rets_1d_raw = V6.1 同口径)
        tsdatas, rets_1d, rets_5d, rets_10d, times, code_values, liquids, rets_1d_eval, rets_1d_raw = batch
        total_loss_sum = 0.0
        n_contrib = 0
        for i in range(len(tsdatas)):
            tsdata = tsdatas[i]
            ret_1d = rets_1d[i]
            ret_5d = rets_5d[i]
            ret_10d = rets_10d[i]

            # V6.2: 逐头过滤 NaN label (停牌/不可买入样本不参与该头 loss)
            m1 = torch.isfinite(ret_1d)
            m5 = torch.isfinite(ret_5d)
            m10 = torch.isfinite(ret_10d)
            day_loss = None
            if m1.sum() >= 50:
                ret_1d_f = ret_1d[m1].unsqueeze(1)

                # V2.8: Mixup 数据增强 (仅对 1d label; V6.1 默认 mixup_alpha=0 关闭)
                if params.mixup_alpha > 0:
                    n = tsdata.shape[0]
                    lam = np.random.beta(params.mixup_alpha, params.mixup_alpha)
                    lam = max(lam, 1.0 - lam)
                    perm = torch.randperm(n, device=tsdata.device)
                    tsdata_mixed = lam * tsdata + (1 - lam) * tsdata[perm]
                    ret_1d_mixed = lam * ret_1d_f + (1 - lam) * ret_1d_f[perm[:ret_1d_f.shape[0]]]
                else:
                    tsdata_mixed = tsdata
                    ret_1d_mixed = ret_1d_f

                # V4.1: R-Drop - 第一次前向传播
                pred_1d, pred_5d, pred_10d = self.model(tsdata_mixed)
                # V4.1: R-Drop - 第二次前向传播 (不同 dropout mask)
                pred_1d_2, pred_5d_2, pred_10d_2 = self.model(tsdata_mixed)

                # V6.2: 各头 loss 只在其标签有限的行上计算
                loss_1d = self.loss_fn(pred_1d[m1], ret_1d_mixed)  # V2.8: Mixup label
                loss_5d = self.loss_fn(pred_5d[m5], ret_5d[m5].unsqueeze(1)) if m5.sum() >= 50 else None
                loss_10d = self.loss_fn(pred_10d[m10], ret_10d[m10].unsqueeze(1)) if m10.sum() >= 50 else None

                # V2.8: 可学习权重 (softmax 归一化, 融合 V1.2)
                learned_weights = F.softmax(self.log_horizon_weights, dim=0)
                weight_scale = 1.8
                w1, w5, w10 = (learned_weights * weight_scale).unbind()
                day_loss = w1 * loss_1d
                if loss_5d is not None:
                    day_loss = day_loss + w5 * loss_5d
                if loss_10d is not None:
                    day_loss = day_loss + w10 * loss_10d
                # V4.1: R-Drop MSE 一致性损失 (仅对 1d 预测; V6.2 权重 0.5→0.1)
                consistency_loss = F.mse_loss(pred_1d[m1], pred_1d_2[m1])
                day_loss = day_loss + params.rdrop_weight * consistency_loss

            if day_loss is not None:
                time_str = times[i] if isinstance(times[i], str) else str(times[i])
                time_weight = _get_time_weight(time_str)
                total_loss_sum += day_loss * time_weight
                n_contrib += 1

        if n_contrib == 0:
            return None
        avg_loss = total_loss_sum / n_contrib
        self.log('train_loss', avg_loss, prog_bar=True, on_step=True)
        return avg_loss

    def _evaluate_step(self, batch, batch_idx, stage):
        """V6.2 评估步骤: 多口径 IC。

        对每个 batch 中的日期计算:
        - rank_ic:    Spearman(pred_1d, 可交易池 Ret1d)   ← 主指标 (checkpoint/早停)
        - ic_raw:     Pearson(pred_1d, 全池 Ret1d_raw)     ← V6.1 同口径对照
        - rank_ic5/10: Spearman(pred_5d/10d, 对应 label)   ← 辅助头诊断
        - top10_ret:  可交易池 top10 平均收益 (参考)
        并统计分月 rank_ic (诊断哪段验证期贡献 IC)。
        """

        def _rank_corr(preds, ret):
            """Spearman rank correlation (GPU, 处理 NaN 由调用方过滤)。"""
            preds = preds.reshape(-1)
            ret = ret.reshape(-1)
            if preds.numel() < 30:
                return preds.new_tensor(0.0)
            def rk(x):
                return torch.argsort(torch.argsort(x)).float()
            p, r = rk(preds), rk(ret)
            p = p - p.mean()
            r = r - r.mean()
            denom = torch.sqrt(p.square().sum() * r.square().sum())
            if denom <= 1e-12:
                return preds.new_tensor(0.0)
            return (p * r).sum() / denom

        def get_personal_return(preds, ret):
            """TOP-10 回测：可交易池内选预测值最高的前 10 只等权买入。"""
            preds_flat = preds.squeeze(1)
            k = min(10, preds_flat.numel())
            topk_idx = torch.topk(preds_flat, k=k).indices
            return ret[topk_idx].mean().item()

        daily_return_list = []
        rank_ic_list = []
        ic_raw_list = []
        rank_ic5_list = []
        rank_ic10_list = []
        month_rank_ic = {}
        # 解包 (rets_1d/5d/10d 为 z 后或原始 label, 含 NaN; rets_1d_eval=可交易池 Ret1d, rets_1d_raw=全池)
        tsdatas, rets_1d, rets_5d, rets_10d, times, code_values, liquids, rets_1d_eval, rets_1d_raw = batch
        for i in range(len(tsdatas)):
            tsdata, time, code_value, liquid, ret_1d, ret_1d_raw = \
                tsdatas[i], times[i], code_values[i], liquids[i], rets_1d_eval[i], rets_1d_raw[i]
            if isinstance(tsdata, str) and tsdata == 'out_sample':
                continue
            pred_1d, pred_5d, pred_10d = self.forward(tsdata)
            if stage == "test":
                self.test_times.add(time)
                preds_cpu = pred_1d.detach().cpu().float().numpy()
                res = pd.DataFrame(preds_cpu, index=code_value, columns=['value'])
                res.index.name = 'Code'
                res.to_pickle(f'{params.test_save_path}/{time}.pkl')

            # 主指标: 可交易池 RankIC (过滤 NaN 的 Ret1d)
            m_trad = torch.isfinite(ret_1d)
            if m_trad.sum() >= 50:
                pred_trad = pred_1d.squeeze(1)[m_trad]
                rank_ic = _rank_corr(pred_trad, ret_1d[m_trad])
                rank_ic_list.append(rank_ic)
                daily_return_list.append(get_personal_return(pred_trad.unsqueeze(1), ret_1d[m_trad]))
                time_str = time if isinstance(time, str) else str(time)
                month = time_str[:6]
                month_rank_ic.setdefault(month, []).append(rank_ic.item())
            # V6.1 同口径对照: 全池 Pearson IC
            ic_raw = self._pearson_corr(pred_1d.squeeze(), ret_1d_raw)
            ic_raw_list.append(ic_raw)
            # 辅助头: 5d/10d RankIC (其 label 有限行)
            m5 = torch.isfinite(rets_5d[i])
            if m5.sum() >= 50:
                rank_ic5_list.append(_rank_corr(pred_5d.squeeze(1)[m5], rets_5d[i][m5]))
            m10 = torch.isfinite(rets_10d[i])
            if m10.sum() >= 50:
                rank_ic10_list.append(_rank_corr(pred_10d.squeeze(1)[m10], rets_10d[i][m10]))

        month_means = {m: float(np.mean(v)) for m, v in month_rank_ic.items()}
        res_list = [
            float(np.nanmean([v.item() for v in rank_ic_list])) if rank_ic_list else np.nan,
            float(np.nanmean([v.item() for v in ic_raw_list])) if ic_raw_list else np.nan,
            float(np.nanmean([v.item() for v in rank_ic5_list])) if rank_ic5_list else np.nan,
            float(np.nanmean([v.item() for v in rank_ic10_list])) if rank_ic10_list else np.nan,
            float(np.nanmean(daily_return_list)) if daily_return_list else np.nan,
            month_means,
        ]
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
        if num_batch == 0:
            return
        # res_list = [rank_ic(可交易池), ic_raw(V6.1同口径), rank_ic5, rank_ic10, top10_ret, month_means]
        val_rankic = np.nanmean([data[0] for data in val_step_outputs])
        val_rankic5 = np.nanmean([data[2] for data in val_step_outputs])
        val_rankic10 = np.nanmean([data[3] for data in val_step_outputs])
        self.log('val_rankic', val_rankic, prog_bar=True, sync_dist=True)
        self.log('val_icraw', np.nanmean([data[1] for data in val_step_outputs]), prog_bar=True, sync_dist=True)
        self.log('val_rankic5', val_rankic5, prog_bar=True, sync_dist=True)
        self.log('val_rankic10', val_rankic10, prog_bar=True, sync_dist=True)
        # V6.3: 平均 IC (三头均值) —— checkpoint/早停主指标
        self.log('val_rankic_avg', np.nanmean([val_rankic, val_rankic5, val_rankic10]), prog_bar=True, sync_dist=True)
        self.log('val_ret', np.nanmean([data[4] * 100 for data in val_step_outputs]), prog_bar=True, sync_dist=True)
        # 分月 rank_ic 诊断
        month_keys = sorted({m for data in val_step_outputs for m in data[5].keys()})
        for m in month_keys:
            vals = [data[5][m] for data in val_step_outputs if m in data[5]]
            if vals:
                self.log(f'val_rankic_{m}', float(np.nanmean(vals)), sync_dist=True)
        self.validation_step_outputs.clear()
        gc.collect()

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        num_batch = len(test_step_outputs)
        if num_batch == 0:
            return
        self.log('test_rankic', np.nanmean([data[0] for data in test_step_outputs]), prog_bar=True, sync_dist=True)
        self.log('test_icraw', np.nanmean([data[1] for data in test_step_outputs]), prog_bar=True, sync_dist=True)
        self.log('test_ret', np.nanmean([data[4] * 100 for data in test_step_outputs]), prog_bar=True, sync_dist=True)
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
        # V6.2: checkpoint 与早停由 val_rankic (可交易池 Spearman RankIC) 驱动。
        # val_icraw 为 V6.1 同口径 Pearson IC, 仅记录对照。
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_rankic_avg', mode='max', save_top_k=4, save_last=False,
                            filename='{epoch}-{val_rankic_avg:.4f}')
        ]
        if params.swa_enabled:
            callbacks.append(StochasticWeightAveraging(
                swa_lrs=self.args.lr * 0.1, swa_epoch_start=params.swa_epoch_start,
                device='cuda' if torch.cuda.is_available() else 'cpu'))
        callbacks.append(EarlyStopping(
            monitor='val_rankic_avg', mode='max', patience=params.early_stop_patience,
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
    # 无测试集（test_date_list 仅含 'out_sample' 占位）时跳过 test
    if test_date_list and test_date_list != ['out_sample']:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        test_result = trainer.test(ckpt_path=best_ckpt, datamodule=dm, weights_only=False)
        print(test_result)


# ── 主训练入口 ───────────────────────────────────────────────────

def get_train_date_split(fold, season, date_list):
    """固定训练集划分：最新 N 个交易日作为验证集（4-Fold 随机抽样），无测试集。

    date_list 按时间升序排列:
    验证集池: 最新的 params.valid_size 个交易日。
    隔断: 验证集池前 params.gap_days 个交易日丢弃（防泄漏）。
    训练集: 其余全部数据。
    K-Fold: 验证池按固定种子洗牌一次后均分为 K 份, 每 Fold 取 1/K (两两无交集且全覆盖)。
    season 参数保留以兼容调用方（仅用于 checkpoint 目录命名），不参与划分。
    """
    date_list = sorted(date_list)

    # 验证集池: 最新 N 个交易日
    valid_pool = date_list[-params.valid_size:]

    # 训练集: 验证集池前隔断 gap_days 个交易日，其余全部作为训练集
    train_cutoff_idx = len(date_list) - params.valid_size - params.gap_days
    train_dates = date_list[:max(0, train_cutoff_idx)]

    # K-Fold: 固定种子洗牌一次, 均分为 K 段, 每 Fold 取 1/K (两两无交集, 并集覆盖全池)
    random.seed(args.seed)  # 所有 Fold 共享同一次洗牌, 保证互斥且全覆盖
    random.shuffle(valid_pool)  # valid_pool 是 date_list 切片副本, 不影响 date_list
    k = params.n_folds
    base, rem = divmod(len(valid_pool), k)  # 余数 rem 由前 rem 个 Fold 各多分 1 个交易日
    f = fold - 1  # fold 从 1 开始
    start = f * base + min(f, rem)
    size = base + (1 if f < rem else 0)
    valid_dates = sorted(valid_pool[start:start + size])

    # 无测试集
    test_dates = []

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
    # V6.6: 追加 23 个新因子 (行业相对动量/行业离散度/隔夜日内分解/量能相对 等)
    fac_new = pd.read_feather(rf'{fac_path}/fac_new.fea')
    params.all_data = params.all_data.merge(fac_new, on=['date', 'Code'], how='left')
    print(f"[V6.6] merged fac_new: +{fac_new.shape[1] - 2} 新因子, all_data={params.all_data.shape}")
    date_list = list(params.all_data["date"].unique())
    date_list = [x for x in date_list if x in params.ret_data.index and x in params.liquid_data.index]
    date_list.sort()
    # V6.1: 数据截止 —— 截止日期之后的数据不参与训练/验证/checkpoint 选择,
    # 保证截止日期之后的回测窗口是真实样本外
    if params.data_end_date is not None:
        date_list = [d for d in date_list if d <= params.data_end_date]
        params.all_data = params.all_data[params.all_data['date'] <= params.data_end_date]
        print(f"[V6.1] data_end_date={params.data_end_date}: 数据截止, 仅用 {date_list[0]}~{date_list[-1]}")
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

    print(f"[V6.2] 多周期联合训练(1d主+5d/10d辅助) + 最新N日验证 + 4-Fold 均分(1/4, 互斥全覆盖) + 无测试集")
    print(f"  Architecture: MLP-Wide Shared Body 512→256→64 → 3 Heads (1d/5d/10d), 评估与回测统一 1d 口径")
    print(f"  Gradient Centralization + OneCycleLR(pct_start=0.1) + GradAccum(8×4) + WD=3e-2 + SWA(0.15) + Mixup=0")
    print(f"  V6.6: buyable mask + label MAD缩尾 + 时间衰减关闭 + 停牌/涨停剔除 loss + R-Drop 0.25 + 23新因子")
    print(f"  Checkpoint/EarlyStop 指标: val_rankic (可交易池 RankIC); 对照 val_icraw (V6.1 同口径 Pearson); 各 Fold 独立种子")
    print(f"  Valid: 最新 {params.valid_size} 个交易日为验证池, 洗牌后均分 {params.n_folds} 份, 每 Fold 取 1/{params.n_folds} (两两无交集且全覆盖)")
    print(f"  Gap: 验证集前隔断 {params.gap_days} 个交易日; Training: 其余全部数据; Test: 无")
    print(f"season: {season} (仅命名), fold: {fold}/{params.n_folds}")
    print(f"train: {len(train_date_list)} dates:  {train_date_list[:3]}...{train_date_list[-3:]} ")
    print(f"valid: {len(valid_date_list)} dates:  {valid_date_list} ")
    print(f"test:  {len(test_date_list)} dates:  {test_date_list} ")
    if len(train_date_list) == 0:
        print("WARNING: 训练集为空! valid_size + gap_days 超过可用日期数, 请调小 params.valid_size")

    if state == 'train':
        train_name = f"{season}/fold{fold}"
        # V6.1: 每个 Fold 独立种子 —— 旧版 4 个 Fold 同种子同训练集, 只是同一条训练轨迹
        # 上不同 epoch 的快照(伪集成); 独立种子后 4 个模型才构成真正集成
        fold_seed = args.seed + fold * 1000
        train_single(args, train_name, fold_seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError
