import os
import random
import shutil
import sys
import warnings

from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

cpu_num = min(8, os.cpu_count() or 8)
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# ============================================================
# 数据划分边界 (与 analysis.py 对齐, 保持 V2 不变)
# ============================================================
TEST_START = '20250901'
TEST_END = '20260901'
TRAIN_END = '20250809'
VALID_DAYS = 120
N_FOLDS = 4
PURGE_DAYS = 5
MIN_DAY_STOCKS = 50
TOP_RET_FRAC = 0.1

# ============================================================
# V6 损失/优化常量 (线性主干 + 顶部小分支)
# ============================================================
LABEL_WINSOR_MAD = 5.0
RDROP_WEIGHT = 0.10
TIME_HALF_LIFE_DAYS = 600
WEIGHT_DECAY = 1e-3        # 线性头用小权重衰减, 顶部分支用较大
WEIGHT_DECAY_TOP = 3e-3
LR_MAX = 5e-4
LR_MIN = 1e-5
WARMUP_EPOCHS = 2
EARLY_STOP_PATIENCE = 10
RIDGE_INIT_FILE = 'ridge_init.csv'   # 相对版本目录; 存在则用于 1d 线性头初始化

# V11a 多目标: fold1-4 保守 (IC 主导), fold5-8 激进 (顶部主导)
# 相对 V9 新增: lin_3d 头 (RANK_W3) + 顶部分支 5d 辅助目标 (TOP5D_WEIGHT)
STYLE_CONS = {
    'RANK_W1': 4.0, 'RANK_W3': 2.0, 'RANK_W5': 2.0,
    'TOPRET_1D': 0.3, 'TOPRET_3D': 0.15,   # 联合顶部双头: 1d 主导
    'LISTNET_1D': 0.1, 'LISTNET_3D': 0.05,
    'TAU_FRAC_1D': 0.08, 'TAU_FRAC_3D': 0.08,
    'LISTNET_TAU_FRAC': 0.15, 'LISTNET_K': 20,
}
STYLE_AGGR = {
    'RANK_W1': 0.5, 'RANK_W3': 0.25, 'RANK_W5': 0.25,
    'TOPRET_1D': 1.0, 'TOPRET_3D': 2.0,   # 3d 主导
    'LISTNET_1D': 0.3, 'LISTNET_3D': 0.6,
    'TAU_FRAC_1D': 0.05, 'TAU_FRAC_3D': 0.03,
    'LISTNET_TAU_FRAC': 0.10, 'LISTNET_K': 10,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='每个 batch 的交易日数')
    parser.add_argument('--lr', type=float, default=LR_MAX, help='AdamW 学习率峰值')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数上限 (早停会提前结束)')
    parser.add_argument('--seed', type=int, default=3253)
    parser.add_argument('--data', default='fac_all', choices=['fac_all', 'fac_sample'],
                        help='因子数据文件 (trainingdata/{data}.fea)')
    args, _ = parser.parse_known_args()
    return args


args = parse_args()

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
root_path = PROJECT_ROOT + 'Model/V15'
fac_path = PROJECT_ROOT + 'trainingdata/'
fac_name = args.data
label_path = PROJECT_ROOT + 'trainingdata'
label_name = r'label_ret_1d'
label_3d_name = r'label_ret_3d'
label_5d_name = r'label_ret_5d'


class params:
    model_path = rf'{root_path}/model_train'
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    ret_1d_data = ret_data
    ret_3d_data = pd.read_feather(rf"{label_path}/{label_3d_name}.fea").set_index("index")
    ret_5d_data = pd.read_feather(rf"{label_path}/{label_5d_name}.fea").set_index("index")
    buyable_mask = pd.read_feather(rf"{label_path}/buyable_mask.fea").set_index("date")


def _safe_label_row(label_df, date):
    if date in label_df.index:
        return label_df.loc[date]
    return pd.Series(np.nan, index=label_df.columns)


def _winsor_mad(s, k=LABEL_WINSOR_MAD):
    v = s.dropna()
    if len(v) == 0:
        return s
    med = v.median()
    mad = (v - med).abs().median()
    if mad <= 1e-12:
        return s
    return s.clip(med - k * mad, med + k * mad)


def _rank_gauss(s):
    finite = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype='float64')
    if finite.sum() >= MIN_DAY_STOCKS:
        r = s[finite].rank()
        r = (r - r.mean()) / r.std()
        out.loc[finite] = r
    return out


def _time_weight(date_str, ref_date=TRAIN_END, half_life=TIME_HALF_LIFE_DAYS):
    if half_life is None:
        return 1.0
    try:
        d = datetime.strptime(date_str, '%Y%m%d')
        ref = datetime.strptime(ref_date, '%Y%m%d')
        return 0.5 ** (max(0, (ref - d).days) / half_life)
    except (ValueError, TypeError):
        return 1.0


def normed_data(data, date, factor_list):
    """与 analysis.py 完全一致: 有效性过滤 → 特征截面秩高斯化 → 缺失填 0;
    1d/3d/5d 标签 (winsor + 秩高斯化)。"""
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    data = data.dropna(subset=factor_list, thresh=valid_threshold).copy()

    label_row = _safe_label_row(params.ret_data, date)
    raw_label = label_row.reindex(data["Code"]).values
    label_3d_row = _safe_label_row(params.ret_3d_data, date)
    raw_label_3d = label_3d_row.reindex(data["Code"]).values
    label_5d_row = _safe_label_row(params.ret_5d_data, date)
    raw_label_5d = label_5d_row.reindex(data["Code"]).values

    data['Label_raw'] = np.nan_to_num(raw_label, nan=0.0)
    data['Label'] = _winsor_mad(pd.Series(raw_label, index=data.index))
    data['Label_3d'] = _winsor_mad(pd.Series(raw_label_3d, index=data.index))
    data['Label_5d'] = _winsor_mad(pd.Series(raw_label_5d, index=data.index))

    buyable_row = _safe_label_row(params.buyable_mask, date)
    data['buyable'] = buyable_row.reindex(data["Code"]).values
    not_buyable = ~(data['buyable'].fillna(False).astype(bool))
    if not_buyable.any():
        data.loc[not_buyable, ['Label', 'Label_3d', 'Label_5d']] = np.nan

    data['Label_rg'] = _rank_gauss(data['Label'])
    data['Label_3d_rg'] = _rank_gauss(data['Label_3d'])
    data['Label_5d_rg'] = _rank_gauss(data['Label_5d'])

    data_X = data[factor_list].rank(axis=0)
    data_X = (data_X - data_X.mean()) / data_X.std()
    data_X = data_X.fillna(0)
    data_x_np = np.nan_to_num(data_X.to_numpy(dtype=np.float32, copy=False),
                              nan=0.0, posinf=0.0, neginf=0.0)
    return (torch.from_numpy(data_x_np),
            torch.from_numpy(data['Label'].to_numpy(dtype=np.float32)),
            torch.from_numpy(data['Label_raw'].to_numpy(dtype=np.float32)),
            torch.from_numpy(data['Label_rg'].to_numpy(dtype=np.float32)),
            torch.from_numpy(data['Label_3d'].to_numpy(dtype=np.float32)),
            torch.from_numpy(data['Label_3d_rg'].to_numpy(dtype=np.float32)),
            torch.from_numpy(data['Label_5d'].to_numpy(dtype=np.float32)),
            torch.from_numpy(data['Label_5d_rg'].to_numpy(dtype=np.float32)),
            data['Code'].values,
            date)


def collate_fn(datas):
    return [list(t) for t in zip(*datas)]


class DLDataset(torch.utils.data.Dataset):
    def __init__(self, date_list, all_data, factor_list):
        self.date_list = date_list
        self.all_data = all_data
        self.factor_list = factor_list

    def __getitem__(self, index):
        date = self.date_list[index]
        data = self.all_data.loc[date].copy()
        return normed_data(data, date, self.factor_list)

    def __len__(self):
        return len(self.date_list)


class DLDataModule(pl.LightningDataModule):
    def __init__(self, args, train_date_list, valid_date_list):
        super().__init__()
        self.args = args
        all_data = params.all_data
        factor_list = params.factor_list
        self.tr = DLDataset(train_date_list, all_data=all_data, factor_list=factor_list)
        self.val = DLDataset(valid_date_list, all_data=all_data, factor_list=factor_list)

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.args.batch_size, collate_fn=collate_fn,
                          num_workers=min(4, cpu_num // 2) if torch.cuda.is_available() else 0,
                          shuffle=True, persistent_workers=torch.cuda.is_available(),
                          drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1, collate_fn=collate_fn,
                          num_workers=0, pin_memory=False, drop_last=False)


# ============================================================
# 损失与评估指标 (日截面)
# ============================================================

def _pearson(preds, y):
    p = preds.reshape(-1)
    y = y.reshape(-1)
    if p.numel() < 2:
        return p.new_zeros(())
    pc = p - p.mean()
    yc = y - y.mean()
    denom = pc.norm() * yc.norm()
    if not torch.isfinite(denom) or denom <= 1e-12:
        return p.new_zeros(())
    return (pc * yc).sum() / denom


def _spearman(preds, y):
    def rk(x):
        return torch.argsort(torch.argsort(x)).float()
    return _pearson(rk(preds.reshape(-1)), rk(y.reshape(-1)))


def _soft_rank(x):
    tau = x.std().detach().clamp_min(1e-6)
    diff = (x.unsqueeze(0) - x.unsqueeze(1)) / tau
    return 1.0 + torch.sigmoid(diff).sum(dim=1)


def _soft_rankic(preds, y):
    rp = _soft_rank(preds.reshape(-1))
    ry = torch.argsort(torch.argsort(y.reshape(-1))).float()
    return _pearson(rp, ry)


def _soft_top_ret(preds, y, tau_frac):
    p = preds.reshape(-1)
    tau = tau_frac * p.std().detach().clamp_min(1e-6)
    w = torch.softmax((p - p.mean()) / tau, dim=0)
    return (w * y).sum() - y.mean()


def _listnet_topk(preds, y, k, tau_frac):
    p = preds.reshape(-1)
    y = y.reshape(-1)
    n = p.numel()
    if n < k + 2:
        return p.new_zeros(())
    tau = tau_frac * p.std().detach().clamp_min(1e-6)
    log_w = torch.log_softmax(p / tau, dim=0)
    _, top_idx = torch.topk(y, k)
    target = p.new_zeros(n)
    target[top_idx] = 1.0 / k
    return -(target * log_w).sum() / np.log(n)


def rank_loss(preds, y_rg, w):
    ic = _pearson(preds, y_rg)
    rankic = _soft_rankic(preds, y_rg)
    return -w * (ic + rankic), ic, rankic


def top_loss(preds, y_rg, y_winsor, w_ret, tau_ret, w_ln, tau_ln, k_ln):
    ret = _soft_top_ret(preds, y_rg, tau_ret)
    ln = _listnet_topk(preds, y_winsor, k_ln, tau_ln)
    return -w_ret * ret + w_ln * ln, ret, ln


# ============================================================
# 模型: 线性排序头 (ridge 初始化) + 小型非线性顶部头
# ============================================================

def _znorm(v):
    mu = v.mean()
    sd = v.std().detach().clamp_min(1e-6)
    return (v - mu) / sd


class PredictModel(nn.Module):
    """V11a: lin_1d/lin_3d/lin_5d 线性排序头 (ridge 热启动) + 独立小 MLP 顶部选择分支。
    forward 返回 (混合分, lin_1d, lin_5d, lin_3d, top)。"""
    def __init__(self, input_dim=None, init_lin1=None, init_bias=None):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num
        self.lin_1d = nn.Linear(input_dim, 1, bias=True)
        self.lin_3d = nn.Linear(input_dim, 1, bias=True)
        self.lin_5d = nn.Linear(input_dim, 1, bias=True)
        self.top_trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.top_1d = nn.Linear(64, 1)
        self.top_3d = nn.Linear(64, 1)
        if init_lin1 is not None:
            with torch.no_grad():
                self.lin_1d.weight.copy_(init_lin1.view(1, -1))
                self.lin_1d.bias.copy_(torch.tensor([init_bias], dtype=torch.float32))
                # 3d/5d 头用 1d 解作热启动 (多周期信号强相关, SGD 再适配)
                self.lin_3d.weight.copy_(self.lin_1d.weight)
                self.lin_3d.bias.copy_(self.lin_1d.bias)
                self.lin_5d.weight.copy_(self.lin_1d.weight)
                self.lin_5d.bias.copy_(self.lin_1d.bias)
        else:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, tsdata):
        x = tsdata.float()
        r1 = self.lin_1d(x)
        r3 = self.lin_3d(x)
        r5 = self.lin_5d(x)
        t = self.top_trunk(x)
        top1d = self.top_1d(t)
        top3d = self.top_3d(t)
        mixed = _znorm(r1) + _znorm(top1d)
        return mixed, r1, r5, r3, top1d, top3d


def load_ridge_init(factor_list):
    """从版本目录读取 ridge_init.csv (按因子名对齐, 顺序无关)。"""
    path = os.path.join(root_path, RIDGE_INIT_FILE)
    if not os.path.exists(path):
        return None, None
    wmap = {}
    bias = 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            name, val = line.rsplit(',', 1)
            if name == '_bias_':
                bias = float(val)
            else:
                wmap[name] = float(val)
    w = [wmap.get(fn, 0.0) for fn in factor_list]
    return torch.tensor(w, dtype=torch.float32), bias


class DLLitModule(LightningModule):
    def __init__(self, args, style='cons'):
        super().__init__()
        self.args = args
        self.style = style
        self.cfg = STYLE_CONS if style == 'cons' else STYLE_AGGR
        init_w, init_b = load_ridge_init(params.factor_list)
        self.model = PredictModel(init_lin1=init_w, init_bias=init_b)
        print(f'[V15] style={style} cfg={self.cfg}')
        print(self.model)
        self.validation_step_outputs = []

    def forward(self, tsdata):
        return self.model(tsdata)

    def training_step(self, batch, batch_idx):
        cfg = self.cfg
        tsdatas, y1w_list, y1raw_list, y1rg_list, y3w_list, y3rg_list, y5w_list, y5rg_list, code_values, dates = batch
        losses, comps = [], []
        for i in range(len(tsdatas)):
            y1w, y1rg = y1w_list[i], y1rg_list[i]
            y3w, y3rg = y3w_list[i], y3rg_list[i]
            y5w, y5rg = y5w_list[i], y5rg_list[i]
            m = torch.isfinite(y1w) & torch.isfinite(y1rg) & torch.isfinite(y3rg) \
                & torch.isfinite(y5rg)
            if m.sum() < MIN_DAY_STOCKS:
                continue
            x = tsdatas[i]
            mixed, r1, r5, r3, top1d, top3d = self.model(x)
            r1m, r5m = r1.squeeze(1)[m], r5.squeeze(1)[m]
            r3m = r3.squeeze(1)[m]
            top1dm = top1d.squeeze(1)[m]
            top3dm = top3d.squeeze(1)[m]
            mixedm = mixed.squeeze(1)[m]
            if not (torch.isfinite(r1m).all() and torch.isfinite(r5m).all()
                    and torch.isfinite(r3m).all() and torch.isfinite(top1dm).all()
                    and torch.isfinite(top3dm).all()):
                continue

            lr1, ic1, rk1 = rank_loss(r1m, y1rg[m], cfg['RANK_W1'])
            lr3, ic3, rk3 = rank_loss(r3m, y3rg[m], cfg['RANK_W3'])
            lr5, ic5, rk5 = rank_loss(r5m, y5rg[m], cfg['RANK_W5'])
            ltop1d, ret1d, ln1d = top_loss(top1dm, y1rg[m], y1w[m],
                                       cfg['TOPRET_1D'], cfg['TAU_FRAC_1D'],
                                       cfg['LISTNET_1D'], cfg['LISTNET_TAU_FRAC'],
                                       cfg['LISTNET_K'])
            ltop3d, ret3d, ln3d = top_loss(top3dm, y3rg[m], y3w[m],
                                       cfg['TOPRET_3D'], cfg['TAU_FRAC_3D'],
                                       cfg['LISTNET_3D'], cfg['LISTNET_TAU_FRAC'],
                                       cfg['LISTNET_K'])

            rdrop = 0.0
            if RDROP_WEIGHT > 0:
                mixed2, _, _, _, _, _ = self.model(x)
                rdrop = nn.functional.mse_loss(mixedm, mixed2.squeeze(1)[m])

            day_loss = (lr1 + lr3 + lr5 + ltop1d + ltop3d + RDROP_WEIGHT * rdrop) \
                * _time_weight(dates[i])
            if torch.isfinite(day_loss):
                losses.append(day_loss)
                comps.append([ic1, rk1, ic3, rk3, ic5, rk5, ret1d, ret3d, ln3d, rdrop])
        if not losses:
            return None
        loss = torch.stack(losses).mean()
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        if comps:
            c = torch.stack([torch.stack(row) for row in comps]).mean(dim=0)
            self.log('tr_ic1', c[0], on_step=True)
            self.log('tr_rk1', c[1], on_step=True)
            self.log('tr_ic3', c[2], on_step=True)
            self.log('tr_rk3', c[3], on_step=True)
            self.log('tr_ic5', c[4], on_step=True)
            self.log('tr_rk5', c[5], on_step=True)
            self.log('tr_ret1d', c[6], on_step=True)
            self.log('tr_ret3d', c[7], on_step=True)
            self.log('tr_listnet', c[8], on_step=True)
            self.log('tr_rdrop', c[9], on_step=True)
        return loss

    def _evaluate_step(self, batch, batch_idx):
        tsdatas, y1w_list, y1raw_list, y1rg_list, y3w_list, y3rg_list, y5w_list, y5rg_list, code_values, dates = batch
        rank_ic_list, ic_raw_list, top_ret_list = [], [], []
        for i in range(len(tsdatas)):
            tsdata = tsdatas[i]
            y = y1w_list[i]
            y_raw = y1raw_list[i]
            mixed, _, _, _, _, _ = self.model(tsdata)
            pred = mixed.squeeze(1)
            m = torch.isfinite(y)
            if m.sum() >= MIN_DAY_STOCKS:
                rank_ic_list.append(_spearman(pred[m], y[m]))
                k = max(5, int(TOP_RET_FRAC * m.sum()))
                topk_idx = torch.topk(pred[m], k).indices
                top_ret_list.append(y[m][topk_idx].mean())
            ic_raw_list.append(_pearson(pred, y_raw))
        res_list = [
            float(np.nanmean([v.item() for v in rank_ic_list])) if rank_ic_list else np.nan,
            float(np.nanmean([v.item() for v in ic_raw_list])) if ic_raw_list else np.nan,
            float(np.nanmean([v.item() for v in top_ret_list])) if top_ret_list else np.nan,
        ]
        self.validation_step_outputs.append(res_list)
        return res_list

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return
        val_rankic = np.nanmean([d[0] for d in outputs])
        val_ic = np.nanmean([d[1] for d in outputs])
        val_ret = np.nanmean([d[2] for d in outputs])
        self.log('val_rankic', val_rankic, prog_bar=True, sync_dist=True)
        self.log('val_ic', val_ic, prog_bar=True, sync_dist=True)
        self.log('val_ret', val_ret, prog_bar=True, sync_dist=True)
        print(f'[epoch {self.current_epoch}] val_rankic={val_rankic:+.4f}  '
              f'val_ic={val_ic:+.4f}  val_top10_ret={val_ret:+.5f}')
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        lin_params = [p for n, p in self.model.named_parameters()
                      if n.startswith('lin_')]
        top_params = [p for n, p in self.model.named_parameters()
                      if n.startswith('top_')]
        opt = torch.optim.AdamW([
            {'params': lin_params, 'weight_decay': WEIGHT_DECAY},
            {'params': top_params, 'weight_decay': WEIGHT_DECAY_TOP},
        ], lr=self.args.lr)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(WARMUP_EPOCHS * total_steps / self.args.epochs))
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            ratio = LR_MIN / self.args.lr
            return ratio + 0.5 * (1.0 - ratio) * (1.0 + np.cos(np.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def configure_callbacks(self):
        return [
            ModelCheckpoint(monitor='val_rankic', mode='max', save_top_k=1,
                            filename='{epoch}-{val_rankic:.4f}'),
            EarlyStopping(monitor='val_rankic', mode='max', patience=EARLY_STOP_PATIENCE,
                          min_delta=1e-4),
        ]


# ============================================================
# 数据划分与训练入口 (与 V2 完全一致)
# ============================================================

def get_date_splits(date_list, fold=1):
    date_list = sorted(date_list)
    train_allowed = [d for d in date_list if d < TRAIN_END]
    test_dates = [d for d in date_list if TEST_START <= d <= TEST_END]
    n = len(train_allowed)
    if n < VALID_DAYS + PURGE_DAYS + 10:
        raise ValueError(f'训练可用日期不足: {n} < {VALID_DAYS + PURGE_DAYS + 10}')
    if not 1 <= fold <= N_FOLDS:
        raise ValueError(f'fold 必须在 1~{N_FOLDS} 之间, 收到 {fold}')
    valid_pool = train_allowed[n - VALID_DAYS:]
    random.seed(args.seed)
    random.shuffle(valid_pool)
    base, rem = divmod(len(valid_pool), N_FOLDS)
    f = fold - 1
    start = f * base + min(f, rem)
    size = base + (1 if f < rem else 0)
    valid_dates = sorted(valid_pool[start:start + size])
    train_dates = train_allowed[:n - VALID_DAYS - PURGE_DAYS]
    return train_dates, valid_dates, test_dates


def train_single(args, name, seed, train_date_list, valid_date_list, style='cons'):
    seed_everything(seed)
    logger = TensorBoardLogger(save_dir=params.model_path, name=name)
    litmodel = DLLitModule(args, style=style)
    dm = DLDataModule(args, train_date_list, valid_date_list)
    callbacks = litmodel.configure_callbacks()
    callbacks.append(TQDMProgressBar(refresh_rate=20))
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(litmodel, dm)


def train(args, season='2026q3', fold=1, state='train'):
    if state != 'train':
        raise NotImplementedError(state)
    save_path = rf"{root_path}/model_test"
    os.makedirs(save_path, exist_ok=True)
    try:
        shutil.copy(os.path.abspath(__file__), save_path)
    except Exception as e:
        print(e)

    print(f"[V15] 加载因子数据: {fac_path}/{fac_name}.fea")
    params.all_data = pd.read_feather(rf'{fac_path}/{fac_name}.fea')
    date_list = sorted(set(params.all_data['date'].unique()) & set(params.ret_data.index))
    params.all_data = params.all_data.set_index('date').sort_index()

    all_zero_mask = (params.all_data == 0).all(axis=0)
    params.all_data = params.all_data.loc[:, ~all_zero_mask]
    params.factor_list = list(params.all_data.columns[1:])
    params.factor_num = len(params.factor_list)
    with open(rf'{save_path}/feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(params.factor_list):
            file.write(rf'{factor_name}={idx}\n')

    # V11a 多目标: fold1-4 保守, fold5-8 激进 (同 4 折划分, 2 seeds × 2 风格)
    split_fold = (fold - 1) % N_FOLDS + 1
    style = 'cons' if fold <= 4 else 'aggr'
    train_dates, valid_dates, test_dates = get_date_splits(date_list, fold=split_fold)
    print('=' * 70)
    print('  V15: fold1-4 保守 + fold5-8 激进; 顶部双头联合 (1d+3d) 共享躯干')
    print('  联合双头: top1d/top3d 共享 128→64 躯干; lin_1d/3d/5d 排序头保留')
    print('  共用 848 因子 + 同一 ridge 热启动; 集成 = 8 个 z-score 求和')
    print('  打分 = z(lin_1d) + z(top) [+ w5*z(lin_5d) + w3*z(lin_3d), analysis 可调]')
    print(f'  本 fold 风格: {style}  cfg={STYLE_CONS if style == "cons" else STYLE_AGGR}')
    print('  训练: AdamW(lin wd=1e-3, top wd=3e-3) + warmup+cosine, 时间衰减 hl=600d')
    print('=' * 70)
    print('  数据划分:')
    print(f'    Test  : {TEST_START} ~ {TEST_END} ({len(test_dates)} 个交易日)')
    print(f'    Valid : 本 Fold({fold}/8, 划分复用 {split_fold}/{N_FOLDS}) 取 {len(valid_dates)} 天: {valid_dates[0]} ~ {valid_dates[-1]}')
    print(f'    Train : {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)} 天)')
    print(f'    factor_num={params.factor_num}, batch_size={args.batch_size}, '
          f'lr={args.lr}, epochs={args.epochs}, fold_seed={args.seed + fold * 1000}')
    print('=' * 70)

    train_name = f'{season}/fold{fold}'
    fold_seed = args.seed + fold * 1000
    train_single(args, train_name, fold_seed, train_dates, valid_dates, style=style)
