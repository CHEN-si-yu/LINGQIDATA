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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

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
    parser.add_argument('--batch_size', type=int, default=8)            # V5.7: BS=8 + grad_accum=4 = effective 32
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--seed', type=int, default=3253)
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--loss', default='wpcc')
    parser.add_argument('--lr', type=float, default=0.0005)            # V5.1: from V4.6

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
root_path = PROJECT_ROOT + 'Model/V5.7'
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
    dropout = True
    dropout_rate = 0.2
    normed_method = 'zscore'

    time_decay_half_life_days = 365
    time_decay_ref_date = '20260101'

    # Training stability
    swa_enabled = True
    swa_epoch_start = 0.6
    early_stop_patience = 8
    gradient_clip_val = 1.0
    warmup_epochs = 3
    # V5.1: CosineAnnealingWarmRestarts params (FIXED from V4.6 bug)
    cosine_t0 = 10
    cosine_t_mult = 2
    cosine_eta_min = 1e-6


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


# ── Warmup + Cosine Annealing Scheduler (V5.1: FIXED from V4.6 bug) ──

class WarmupThenCosine(CosineAnnealingWarmRestarts):
    """Linear warmup followed by CosineAnnealingWarmRestarts.

    V5.1 fix: V4.6 intended to use Cosine but the warmup branch always
    fell through to ReduceLROnPlateau. This class correctly chains:
      - First `warmup_epochs` steps: linear warmup from 0 → base_lr
      - Thereafter: CosineAnnealingWarmRestarts(T_0, T_mult, eta_min)

    NOTE: _in_warmup and _warmup_step MUST be set BEFORE super().__init__()
    because CosineAnnealingWarmRestarts._initial_step() calls self.step().
    """
    def __init__(self, optimizer, warmup_epochs, **kwargs):
        self._warmup_epochs = warmup_epochs
        self._warmup_base_lrs = [float(g['lr']) for g in optimizer.param_groups]
        # Set warmup state BEFORE super().__init__() — parent calls step() in _initial_step()
        self._warmup_step = 0
        self._in_warmup = True
        super().__init__(optimizer, **kwargs)

    def step(self, epoch=None):
        if self._in_warmup:
            self._warmup_step += 1
            progress = float(self._warmup_step) / float(max(1, self._warmup_epochs))
            for param_group, base_lr in zip(self.optimizer.param_groups, self._warmup_base_lrs):
                param_group['lr'] = base_lr * progress
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            if self._warmup_step >= self._warmup_epochs:
                self._in_warmup = False
                # Reset cosine state so first cycle starts fresh from T_0
                self.T_cur = 0
                self.last_epoch = 0
        else:
            super().step(epoch)


# ── PredictModel (V5.1: V4.5 wider architecture 512→256→64→1) ─────────

class PredictModel(nn.Module):
    """MLP-Wide -- V5.1: 512→256→64→1 (from V4.5 best Top-Ret)"""
    def __init__(self, args=None, input_dim=None, **kwargs):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num

        # 层 1: 输入投射 → 512 (V4.5 加宽首层)
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
        # 输出层
        self.output_layer = nn.Linear(64, 1)
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
        x = self.output_layer(x)
        return x


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
        """Inference forward pass."""
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
        tsdatas, rets, times, code_values, liquids, rets_1d = batch
        total_loss_sum = 0.0
        for i in range(len(tsdatas)):
            tsdata, ret = tsdatas[i], rets[i]
            ret = ret.unsqueeze(1)

            preds = self.model(tsdata)
            loss = self.loss_fn(preds, ret)

            time_str = times[i] if isinstance(times[i], str) else str(times[i])
            time_weight = _get_time_weight(time_str)
            total_loss_sum += loss * time_weight

        avg_loss = total_loss_sum / len(tsdatas)
        self.log('train_loss', avg_loss, prog_bar=True, on_step=True)
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
                    preds_cpu = preds.detach().cpu().float().numpy()
                    res = pd.DataFrame(preds_cpu, index=code_value, columns=['value'])
                    res.index.name = 'Code'
                    res.to_pickle(f'{params.test_save_path}/{time}.pkl')
                # 使用真实 1 日收益计算评估指标
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
        """Manual gradient clipping — compatible with bf16-mixed precision."""
        if params.gradient_clip_val is not None and params.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=params.gradient_clip_val)

    def configure_optimizers(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]

        # V5.1 FIXED: Warmup → CosineAnnealingWarmRestarts
        # (V4.6 bug was that warmup branch always used ReduceLROnPlateau)
        scheduler = {
            'scheduler': WarmupThenCosine(
                optimizer,
                warmup_epochs=params.warmup_epochs,
                T_0=params.cosine_t0,
                T_mult=params.cosine_t_mult,
                eta_min=params.cosine_eta_min,
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
                      profiler=profiler, logger=logger,
                      enable_progress_bar=enable_tqdm_progress,
                      deterministic=False,
                      precision='bf16-mixed',
                      accumulate_grad_batches=4)  # V5.7: BS=8 × 4 = eff32
    litmodel = DLLitModule(args)
    dm = DLDataModule(args, train_date_list, valid_date_list, test_date_list)
    trainer.fit(litmodel, dm)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_result = trainer.test(ckpt_path=best_ckpt, datamodule=dm, weights_only=False)
    print(test_result)


# ── Main train entry ───────────────────────────────────────────────────

def get_train_date_split(fold, season, date_list):
    """Fixed train split: 4yr train + random 1/4 valid + quarter test."""
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

    random.seed(args.seed + fold)
    sample_size = max(1, len(valid_pool) // 4)
    valid_dates = sorted(random.sample(valid_pool, sample_size))

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
        all_zero_mask = (params.all_data == 0).all(axis=0)
        params.all_data = params.all_data.loc[:, ~all_zero_mask]
    else:
        raise NotImplementedError

    feature_map = list(params.all_data.columns[1:])
    params.factor_list = feature_map[:]
    with open(rf'{save_path}/{market}{name[len(market):len(market) + 6]}-feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(rf'{factor_name}={idx}\n')

    params.factor_num = params.all_data.shape[1] - 1

    print(f"[V5.7] DNN Training -- Gradient Accum (BS=8 × 4 = eff32)")
    print(f"  Training window: 4 years")
    print(f"season: {season}, fold: {fold}")
    print(f"train: {len(train_date_list)} dates:  {train_date_list} ")
    print(f"valid: {len(valid_date_list)} dates:  {valid_date_list} ")
    print(f"test: {len(test_date_list)} dates:  {test_date_list} ")

    if state == 'train':
        train_name = f"{name}/{season}/fold{fold}"
        train_single(args, train_name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError
