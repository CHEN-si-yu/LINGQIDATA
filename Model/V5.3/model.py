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

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(False)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--seed', type=int, default=3253)
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--loss', default='wpcc')
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--min_epochs', type=int, default=30)
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--strategy', default='auto')
    parser.add_argument('--find_unused_parameters', default=False)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--check_test_every_n_epoch', type=int, default=1)

    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')

    args, unknown = parser.parse_known_args()
    return args


args = parse_args()

# ============================================================
# V5 paths — uses V3 factor data (best performer)
# ============================================================
root_path = r'/root/shared-nvme/lingqiData/Model/V5.3'
fac_path = r'/root/shared-nvme/lingqiData/trainingdata/V3/'
fac_name = r'fac20260523'
label_path = r'/root/shared-nvme/lingqiData/trainingdata/V3/'
label_name = r'label'
liquid_path = r'/root/shared-nvme/lingqiData/trainingdata/V3/'
liquid_name = r'trade_amt'


class params:
    model_path = rf'{root_path}/model_train'
    profiler_path = rf'{root_path}/logs'
    model_prefix = rf'nn'
    liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    dropout = True
    dropout_rate = 0.2
    normed_method = 'zscore'

    # ── V5 new hyperparameters ─────────────────────────────────────────
    feature_dropout_rate = 0.25       # fraction of features randomly masked during training
    ranknet_alpha = 0.1               # weight of pairwise ranking loss vs WPCC
    time_decay_half_life_days = 365   # time-decay half-life in days
    time_decay_ref_date = '20260101'  # reference date for time decay

    # ── V5.3 limit-up context gating ──────────────────────────────────
    near_limit_up_idx = None          # set at training time: index of 'bias_20' in factor_list
    gate_threshold = 1.5              # bias_20 z-score above which limit-up branch activates
    gate_temperature = 5.0            # sharpness of gating transition
    limit_up_loss_weight = 2.0        # extra weight on limit-up stocks in training loss


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
    code_value = data['Code'].values
    data_X = data.drop(['Code', 'Label', 'liquid'], axis=1)
    data_y = data['Label']
    data_liquid = data['liquid']

    if normed_method == 'zscore':
        quantiles = data_X.quantile([0.005, 0.995])
        data_X = data_X.clip(lower=quantiles.loc[0.005], upper=quantiles.loc[0.995], axis=1)
        data_X = (data_X - data_X.mean()) / data_X.std()
        data_X = data_X.fillna(0)
    else:
        raise NotImplementedError

    data_x_np = np.nan_to_num(data_X.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_y_np = np.nan_to_num(data_y.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    data_liquid_np = np.nan_to_num(data_liquid.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    return torch.from_numpy(data_x_np), \
        torch.from_numpy(data_y_np), \
        code_value, torch.from_numpy(data_liquid_np)


def collate_fn(datas):
    data_X, data_y, data_time, code_value, data_liquid = zip(*datas)
    return list(data_X), list(data_y), list(data_time), list(code_value), list(data_liquid)


class DLDataset(torch.utils.data.Dataset):

    def __init__(self, date_list, all_data, factor_list, stage='train'):
        self.date_list = date_list
        self.all_data = all_data
        self.factor_list = factor_list
        self.stage = stage

    def __getitem__(self, index):
        date = self.date_list[index]
        if date == 'out_sample':
            return 'out_sample', 'out_sample', 'out_sample', 'out_sample', 'out_sample'
        data = self.all_data.loc[date].copy()
        data_X, data_y, code_value, data_liquid = normed_data(
            data, date,
            stage=self.stage,
            factor_list=self.factor_list,
        )
        return data_X, data_y, date, code_value, data_liquid

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
                          num_workers=4 if torch.cuda.is_available() else 0, shuffle=True,
                          persistent_workers=False,
                          drop_last=False,
                          pin_memory=False)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                          num_workers=0, persistent_workers=False,
                          pin_memory=False, drop_last=False)

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


# ── RankNet pairwise ranking loss ─────────────────────────────────────

def pairwise_ranking_loss(preds, y, n_pairs=2000):
    """RankNet-style pairwise ranking loss.

    Samples pairs of stocks and penalizes ordering violations.
    A stock with higher return should receive a higher predicted score.
    """
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


# ── Time-decay weight ─────────────────────────────────────────────────

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


# ── V5.3 Two-Branch Model: Normal + Limit-Up with soft gating ──────────

class PredictModel(nn.Module):
    """V5.3: Shared 256→128 backbone, dual heads for normal vs limit-up regimes.

    A soft gate driven by bias_20 z-score routes each stock between:
      - Normal head (most stocks): standard factor→return mapping
      - Limit-up head (overbought stocks): continuation/reversal patterns

    Gate: sigmoid((bias_20_zscore - threshold) * temperature)
    """
    def __init__(self, args):
        super(PredictModel, self).__init__()
        input_dim = params.factor_num

        # ── Shared backbone ────────────────────────────────────────────
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Residual Block 1 (shared): 256 → 128
        self.res_fc1 = nn.Linear(256, 128)
        self.res_bn1 = nn.BatchNorm1d(128)
        self.res_act1 = nn.GELU()
        self.res_drop1 = nn.Dropout(0.1)
        self.res_proj1 = nn.Linear(256, 128)

        # ── Normal head: 128 → 32 → 1 ─────────────────────────────────
        self.normal_fc2 = nn.Linear(128, 32)
        self.normal_bn2 = nn.BatchNorm1d(32)
        self.normal_act2 = nn.GELU()
        self.normal_proj2 = nn.Linear(128, 32)
        self.normal_output = nn.Linear(32, 1)

        # ── Limit-up head: 128 → 32 → 1 (separate parameters) ─────────
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

    def forward(self, tsdata, feature_mask=None):
        x = tsdata.float()

        if feature_mask is not None:
            x = x * feature_mask

        # Extract near-limit-up signal from bias_20 z-score
        if params.near_limit_up_idx is not None and params.near_limit_up_idx < x.shape[1]:
            bias_20_zscore = x[:, params.near_limit_up_idx]  # (N,)
        else:
            bias_20_zscore = torch.zeros(x.shape[0], device=x.device)

        # ── Shared backbone ────────────────────────────────────────────
        x = self.input_layer(x)

        # Residual Block 1 (shared)
        identity = self.res_proj1(x)
        out = self.res_fc1(x)
        out = self.res_bn1(out)
        out = self.res_act1(out)
        out = self.res_drop1(out)
        shared_repr = out + identity                         # (N, 128)

        # ── Normal branch ─────────────────────────────────────────────
        n_identity = self.normal_proj2(shared_repr)
        n_out = self.normal_fc2(shared_repr)
        n_out = self.normal_bn2(n_out)
        n_out = self.normal_act2(n_out)
        normal_x = n_out + n_identity                         # (N, 32)
        normal_score = self.normal_output(normal_x)           # (N, 1)

        # ── Limit-up branch ───────────────────────────────────────────
        lu_identity = self.limit_up_proj2(shared_repr)
        lu_out = self.limit_up_fc2(shared_repr)
        lu_out = self.limit_up_bn2(lu_out)
        lu_out = self.limit_up_act2(lu_out)
        limit_up_x = lu_out + lu_identity                      # (N, 32)
        limit_up_score = self.limit_up_output(limit_up_x)      # (N, 1)

        # ── Soft gate: bias_20 z-score → gating weight ───────────────
        gate = torch.sigmoid(
            (bias_20_zscore - params.gate_threshold) * params.gate_temperature
        ).unsqueeze(1)                                         # (N, 1)

        score = gate * limit_up_score + (1.0 - gate) * normal_score
        return score, gate, normal_score, limit_up_score


# ── Lightning module (V5 enhanced) ─────────────────────────────────────

class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel(args)
        self.loss_fn = get_loss_fn(self.args.loss)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_times = set()

    def forward(self, tsdata, feature_mask=None):
        score, gate, normal_score, limit_up_score = self.model(tsdata, feature_mask=feature_mask)
        return score

    def forward_with_gate(self, tsdata, feature_mask=None):
        """Return full decomposition: (score, gate, normal_score, limit_up_score)."""
        return self.model(tsdata, feature_mask=feature_mask)

    @staticmethod
    def _pearson_corr(preds, ret):
        preds = preds.reshape(-1)
        ret = ret.reshape(-1)
        preds_centered = preds - preds.mean()
        ret_centered = ret - ret.mean()
        denominator = torch.sqrt(
            preds_centered.square().sum() * ret_centered.square().sum()
        )
        if denominator <= 1e-12:
            return preds.new_tensor(0.0)
        return (preds_centered * ret_centered).sum() / denominator

    def training_step(self, batch, batch_idx):
        tsdatas, rets, times, code_values, liquids = batch
        loss_list = []
        for i in range(len(tsdatas)):
            tsdata, ret, liquid = tsdatas[i], rets[i], liquids[i]
            ret = ret.unsqueeze(1)

            # Generate feature dropout mask (same for all stocks in this date)
            feature_mask = None
            if self.training and params.feature_dropout_rate > 0:
                n_features = tsdata.shape[1]
                keep_prob = 1.0 - params.feature_dropout_rate
                feature_mask = torch.bernoulli(
                    torch.full((1, n_features), keep_prob, device=tsdata.device)
                )
                feature_mask = feature_mask / keep_prob

            preds, gate, normal_score, limit_up_score = self.forward_with_gate(
                tsdata, feature_mask=feature_mask
            )

            # WPCC loss (all stocks)
            wpcc_loss = self.loss_fn(preds, ret)

            # Pairwise ranking loss
            rank_loss = pairwise_ranking_loss(preds, ret)

            # ── V5.3: extra WPCC loss on limit-up stocks ──────────────
            lu_mask = (gate > 0.5).squeeze(1)
            lu_extra = torch.tensor(0.0, device=preds.device)
            if lu_mask.sum() > 5:
                lu_extra = self.loss_fn(preds[lu_mask], ret[lu_mask])

            # Time decay weight
            time_str = times[i] if isinstance(times[i], str) else str(times[i])
            time_weight = _get_time_weight(time_str)

            total_loss = (
                wpcc_loss
                + params.ranknet_alpha * rank_loss
                + params.limit_up_loss_weight * lu_extra
            ) * time_weight
            loss_list.append(total_loss)

        total_loss = sum(loss_list) / len(loss_list)
        self.log('train_loss', total_loss, prog_bar=True, on_step=True)
        return total_loss

    def _evaluate_step(self, batch, batch_idx, stage):
        def get_excess_return(preds, ret, liquid, money):
            topk = min(500, preds.shape[0])
            sort = torch.argsort(preds.squeeze(1), descending=True, stable=True)[:topk]
            sorted_liquid = liquid[sort].reshape(-1)
            sorted_ret = ret[sort].reshape(-1)
            money_tensor = preds.new_tensor(money)
            previous_hold = torch.cat(
                [sorted_liquid.new_zeros(1), sorted_liquid.cumsum(dim=0)[:-1]]
            )
            remaining_before_buy = money_tensor - previous_hold
            hold_money = torch.minimum(remaining_before_buy, sorted_liquid)
            hold_money = torch.where(
                remaining_before_buy >= 1,
                torch.clamp(hold_money, min=0.0),
                torch.zeros_like(hold_money),
            )
            total_ret = (sorted_ret * hold_money).sum() / money_tensor
            return total_ret

        excess_return_list = []
        ic_list = []
        tsdatas, rets, times, code_values, liquids = batch
        for i in range(len(tsdatas)):
            tsdata, ret, time, code_value, liquid = tsdatas[i], rets[i], times[i], code_values[i], liquids[i]
            if isinstance(tsdata, str) and tsdata == 'out_sample':
                pass
            else:
                # No feature dropout during eval
                preds = self.forward(tsdata, feature_mask=None)
                if stage == "test":
                    self.test_times.add(time)
                    preds_cpu = preds.detach().cpu().numpy()
                    res = pd.DataFrame(preds_cpu, index=code_value, columns=['value'])
                    res.index.name = 'Code'
                    res.to_pickle(f'{params.test_save_path}/{time}.pkl')

                excess_return = get_excess_return(preds, ret, liquid, money=1.5e9)
                excess_return_list.append(excess_return)
                ic_list.append(self._pearson_corr(preds.squeeze(), ret))
        try:
            res_list = [sum(excess_return_list) / len(excess_return_list), sum(ic_list) / len(ic_list)]
        except:
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
        self.log('val_ret', sum([(data[0] * 100) for data in val_step_outputs]) / num_batch, prog_bar=True,
                 sync_dist=True)
        self.log('val_icmean', sum([(data[1]) for data in val_step_outputs]) / num_batch, prog_bar=True,
                 sync_dist=True)
        self.log('val_wei', sum([(data[0] * 100 + data[1] * 0.7) for data in val_step_outputs]) / num_batch, prog_bar=True,
                 sync_dist=True)
        self.validation_step_outputs.clear()
        gc.collect()

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        num_batch = len(test_step_outputs)
        self.log('test_wei', sum([data[0] * 100 for data in test_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.log('test_icmean', sum([(data[1]) for data in test_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.test_step_outputs.clear()
        self.test_times.clear()

    def configure_optimizers(self):
        kwargs = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay,
        }
        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=5e-6,
            cooldown=2
        )
        optim_config = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_wei',
            },
        }
        return optim_config

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_wei', mode='max', save_top_k=10, save_last=False,
                            filename='{epoch}-{val_wei:.4f}')
        ]
        if self.args.swa:
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=0.7,
                                                       device='cuda' if torch.cuda.is_available() else 'cpu'))
        if self.args.early_stop:
            callbacks.append(EarlyStopping(monitor='val_ret',
                                           mode='max', patience=6))
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
                      num_sanity_val_steps=10,
                      profiler=profiler,
                      logger=logger,
                      enable_progress_bar=enable_tqdm_progress,
                      deterministic=True)

    litmodel = DLLitModule(args)
    dm = DLDataModule(args, train_date_list, valid_date_list, test_date_list)
    trainer.fit(litmodel, dm)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_result = trainer.test(ckpt_path=best_ckpt, datamodule=dm)
    print(test_result)


# ── Main train entry ───────────────────────────────────────────────────

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

    from datetime import datetime, timedelta
    import random

    def get_train_date_split(kfold, date_list, season, period, interval=10, month=24):
        """Walk-forward K-fold: every fold's validation is strictly after its training.

        The year immediately before *test_start* is used for validation.

        When *shuffle_split* is True: each fold randomly samples 1/kfold of trading
        days from the full validation year; training is fixed at exactly 3 years
        ending at valid_year_start (same training set for all folds).

        When *shuffle_split* is False: validation is split into 4 consecutive
        quarters assigned to folds in temporal order (Q1→fold1 … Q4→fold4);
        training is 3 years ending just before each fold's validation quarter.
        """
        test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2)
        test_start = datetime.strptime(test_start, "%Y%m")
        test_end = test_start + relativedelta(months=3)

        # ── Test dates ──────────────────────────────────────────────────
        test_date_list = [x for x in date_list if test_start <= datetime.strptime(x, "%Y%m%d") < test_end]

        # ── Validation year: the full year before test_start ────────────
        valid_year_start = test_start - relativedelta(years=1)

        if args.shuffle_split:
            # ── Random sample: 1/kfold of trading days from full validation year
            random.seed(args.seed + period)
            all_valid_dates = [x for x in date_list
                              if valid_year_start <= datetime.strptime(x, "%Y%m%d") < test_start]
            all_valid_dates = all_valid_dates[interval:] if len(all_valid_dates) > interval else all_valid_dates
            sample_size = max(1, len(all_valid_dates) // kfold)
            valid_date_list = sorted(random.sample(all_valid_dates, sample_size))

            # ── Training: fixed 3 years ending at valid_year_start (same for all folds)
            train_end_dt = valid_year_start - timedelta(days=interval)
            train_start_dt = train_end_dt - relativedelta(years=3)
            train_date_list = [x for x in date_list
                              if train_start_dt <= datetime.strptime(x, "%Y%m%d") < train_end_dt]
        else:
            # ── Validation quarters: 4 consecutive quarters in the validation year
            valid_quarters = []  # (q_start_dt, q_end_dt, [trading_dates])
            for q in range(4):
                q_start = valid_year_start + relativedelta(months=3 * q)
                q_end = q_start + relativedelta(months=3)
                q_dates = [x for x in date_list if q_start <= datetime.strptime(x, "%Y%m%d") < q_end]
                if q_dates:
                    valid_quarters.append((q_start, q_end, q_dates))

            # ── Pick this fold's validation quarter ─────────────────────
            if period - 1 < len(valid_quarters):
                valid_start, valid_end, valid_all_dates = valid_quarters[period - 1]
            else:
                valid_start = test_start - relativedelta(months=3)
                valid_end = test_start
                valid_all_dates = [x for x in date_list if valid_start <= datetime.strptime(x, "%Y%m%d") < valid_end]

            # Embargo: strip the first *interval* trading days from validation
            valid_date_list = valid_all_dates[interval:] if len(valid_all_dates) > interval else valid_all_dates

            # ── Training: 3 years ending at valid_start ─────────────────
            train_end_dt = valid_start - timedelta(days=interval)
            train_start_dt = train_end_dt - relativedelta(years=3)
            train_date_list = [x for x in date_list
                              if train_start_dt <= datetime.strptime(x, "%Y%m%d") < train_end_dt]

        # ── Exclude Chinese New Year disruption period ──────────────────
        not_train_date = [x for x in date_list if (x >= "202402") & (x <= "20240223")]
        train_date_list = [x for x in train_date_list if x not in not_train_date]

        # ── 2024q3/q4 special boundary handling ─────────────────────────
        if season == '2024q3':
            test_date_list += [x for x in date_list if '20241001' <= x <= '20241016']
        if season == '2024q4':
            test_date_list = [x for x in test_date_list if '20241017' <= x]

        return train_date_list, valid_date_list, test_date_list

    args.all_train = False
    args.shuffle_split = False  # V5: sequential walk-forward (proven best in V4.1)
    train_date_list, valid_date_list, test_date_list = get_train_date_split(
        kfold=4,
        date_list=date_list,
        season=season,
        period=int(params.test_save_path[-1]),
        interval=10,
        month=48)
    train_date_list.sort()
    valid_date_list.sort()
    test_date_list.sort()

    if len(test_date_list) == 0:
        test_date_list = ['out_sample']
    elif market == 'ALL':
        params.all_data = params.all_data.loc[:, params.all_data.replace(0, np.nan).dropna(how="all", axis=1).columns]
    else:
        raise NotImplementedError

    feature_map = list(params.all_data.columns[1:])
    params.factor_list = feature_map[:]
    # V5.3: find bias_20 index for limit-up gating
    try:
        params.near_limit_up_idx = feature_map.index('bias_20')
        print(f"[V5.3] near_limit_up_idx (bias_20) = {params.near_limit_up_idx}")
    except ValueError:
        params.near_limit_up_idx = None
        print("[V5.3] WARNING: bias_20 not found, limit-up gating disabled")

    with open(rf'{save_path}/{market}{name[len(market):len(market) + 6]}-feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(rf'{factor_name}={idx}')
            file.write('\n')

    params.factor_num = params.all_data.shape[1] - 1

    print(f"season: {season}, period: {params.test_save_path[-1]}")
    print(f"len(train_date_list): {len(train_date_list)}, len(valid_date_list): {len(valid_date_list)}, len(test_date_list): {len(test_date_list)}")
    print(f"train_date_list: {train_date_list}")
    print(f"valid_date_list: {valid_date_list}")
    print(f"test_date_list: {test_date_list}")

    if state == 'train':
        train_name = f"{name}/{season}/fold{fold}"
        train_single(args, train_name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError
