#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V20c model.py — 冠军系统独立可训练单元 (16 折 = 双族配方, 从零训练)

配方 = V11(多目标族) + V13(顶部3d族), 与冠军评估(FINAL_REPORT)完全一致:
  族 a (族 a (V11 多目标)): 顶部分支主目标 1d (+5d 辅助)   → 全局 fold 1..8
  族 c (V13 顶部3d): 顶部分支主目标 3d (+1d 辅助)   → 全局 fold 9..16
  每族 8 折 = 4 折划分 × 2 风格(保守/激进); 集成打分:
    ens_w2 = z(mixA) + 2·z(V13组件)
    mixA    = Σ_{a族8折} z( 1.0·r1 + 0.25·r3 + 2.0·top )
    V13组件 = Σ_{c族8折} z( 1.0·r1 + 1.0·top )

结构 (V9 以来不变): 线性排序头 lin_1d/lin_3d/lin_5d (ridge 热启动) + 独立小 MLP 顶部分支;
损失 = -(IC+RankIC) + 软Top收益(主/辅) + ListNet + R-Drop + 时间衰减(hl=600d) + AdamW(warmup+cosine)。

文件使命 (版本四文件之一): 模型架构 / 损失 / 数据划分 / 训练函数, 被 run.py 调用;
训练调度 = train.sh (分批并发数按 TRAINING_PLAYBOOK §2 内存规则: 120GB→6 折/批);
推演/回测/出结果 = analysis.py。
"""
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
# V20c 双族折体系: 全局 fold 1..16 (train.sh 按 §2 内存规则分批启动: 120GB→6 折/批)
# ============================================================
N_FOLD_PER_FAMILY = 8              # 每族折数 (族内: 4 折划分 × 2 风格)
TOTAL_FOLDS = N_FOLD_PER_FAMILY * 2  # 16 折 (族 a = fold 1..8, 族 c = fold 9..16)


def fold_spec(fold):
    """全局 fold 1..16 → (族, 族内折 k∈1..8)。族 a: 族 a (V11 多目标); 族 c: V13 顶部3d。"""
    fold = int(fold)
    if not 1 <= fold <= TOTAL_FOLDS:
        raise ValueError(f'V20c fold 必须在 1~{TOTAL_FOLDS}, 收到 {fold}')
    if fold <= N_FOLD_PER_FAMILY:
        return 'a', fold
    return 'c', fold - N_FOLD_PER_FAMILY

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

# V20c: 每族内 fold(k)1-4 保守 (IC 主导), k5-8 激进 (顶部主导)
# 顶部分支双目标: 主目标(TOPRET_WEIGHT) + 辅助目标(TOP_AUX_WEIGHT);
# 主/辅标签周期随族: 族a = 1d主/5d辅, 族c = 3d主/1d辅
STYLE_CONS = {
    'RANK_W1': 4.0,        # 1d 线性排序头权重
    'RANK_W3': 2.0,        # 3d 线性排序头权重 (新增)
    'RANK_W5': 2.0,        # 5d 线性排序头权重
    'TOPRET_WEIGHT': 0.3,  # 软 Top 收益项权重 (1d)
    'TOP_AUX_WEIGHT': 0.15,  # 顶部分支辅助目标权重 (a族:5d / c族:1d)
    'TOPRET_TAU_FRAC': 0.08,
    'LISTNET_WEIGHT': 0.1,
    'LISTNET_TAU_FRAC': 0.15,
    'LISTNET_K': 20,
}
STYLE_AGGR = {
    'RANK_W1': 0.5,
    'RANK_W3': 0.25,
    'RANK_W5': 0.25,
    'TOPRET_WEIGHT': 2.0,
    'TOP_AUX_WEIGHT': 1.0,   # 激进: 辅助目标权重加大
    'TOPRET_TAU_FRAC': 0.03,   # 更锐 → 权重集中在前 ~2~3%
    'LISTNET_WEIGHT': 0.6,
    'LISTNET_TAU_FRAC': 0.10,
    'LISTNET_K': 10,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='每个 batch 的交易日数')
    parser.add_argument('--lr', type=float, default=LR_MAX, help='AdamW 学习率峰值')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数上限 (早停会提前结束)')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--data', default='fac_all', choices=['fac_all', 'fac_sample'],
                        help='因子数据文件 (trainingdata/{data}.fea)')
    args, _ = parser.parse_known_args()
    return args


args = parse_args()

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
root_path = PROJECT_ROOT + 'Model/V20c'
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
    """V20c: lin_1d/lin_3d/lin_5d 线性排序头 (ridge 热启动) + 独立小 MLP 顶部选择分支。
    forward 返回 (混合分, lin_1d, lin_5d, lin_3d, top)。同族 a/c 共用结构,
    差异只在顶部分支的训练目标周期 (见 DLLitModule.training_step)。"""
    def __init__(self, input_dim=None, init_lin1=None, init_bias=None):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num
        self.lin_1d = nn.Linear(input_dim, 1, bias=True)
        self.lin_3d = nn.Linear(input_dim, 1, bias=True)
        self.lin_5d = nn.Linear(input_dim, 1, bias=True)
        self.top_branch = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
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
        top = self.top_branch(x)
        mixed = _znorm(r1) + _znorm(top)
        return mixed, r1, r5, r3, top


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
    def __init__(self, args, style='cons', fam='a'):
        super().__init__()
        self.args = args
        self.style = style
        self.fam = fam
        self.cfg = STYLE_CONS if style == 'cons' else STYLE_AGGR
        init_w, init_b = load_ridge_init(params.factor_list)
        self.model = PredictModel(init_lin1=init_w, init_bias=init_b)
        print(f'[V20c] fam={fam} style={style} cfg={self.cfg}')
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
            mixed, r1, r5, r3, top = self.model(x)
            r1m, r5m = r1.squeeze(1)[m], r5.squeeze(1)[m]
            r3m = r3.squeeze(1)[m]
            topm = top.squeeze(1)[m]
            mixedm = mixed.squeeze(1)[m]
            if not (torch.isfinite(r1m).all() and torch.isfinite(r5m).all()
                    and torch.isfinite(r3m).all() and torch.isfinite(topm).all()):
                continue

            lr1, ic1, rk1 = rank_loss(r1m, y1rg[m], cfg['RANK_W1'])
            lr3, ic3, rk3 = rank_loss(r3m, y3rg[m], cfg['RANK_W3'])
            lr5, ic5, rk5 = rank_loss(r5m, y5rg[m], cfg['RANK_W5'])
            if self.fam == 'c':   # 族 c (V13 顶部3d): 主 = 3d 软Top, 辅 = 1d
                top_main_rg, top_main_w = y3rg[m], y3w[m]
                top_aux_rg = y1rg[m]
            else:                 # 族 a (族 a (V11 多目标)): 主 = 1d 软Top, 辅 = 5d
                top_main_rg, top_main_w = y1rg[m], y1w[m]
                top_aux_rg = y5rg[m]
            ltop, ret, ln = top_loss(topm, top_main_rg, top_main_w,
                                     cfg['TOPRET_WEIGHT'], cfg['TOPRET_TAU_FRAC'],
                                     cfg['LISTNET_WEIGHT'], cfg['LISTNET_TAU_FRAC'],
                                     cfg['LISTNET_K'])
            # 顶部分支辅助目标 (族 a: 5d / 族 c: 1d 软Top)
            ltop5 = 0.0
            if cfg['TOP_AUX_WEIGHT'] > 0:
                ret5 = _soft_top_ret(topm, top_aux_rg, cfg['TOPRET_TAU_FRAC'])
                ltop5 = -cfg['TOP_AUX_WEIGHT'] * ret5
            else:
                ret5 = torch.zeros_like(ret)

            rdrop = 0.0
            if RDROP_WEIGHT > 0:
                mixed2, _, _, _, _ = self.model(x)
                rdrop = nn.functional.mse_loss(mixedm, mixed2.squeeze(1)[m])

            day_loss = (lr1 + lr3 + lr5 + ltop + ltop5 + RDROP_WEIGHT * rdrop) \
                * _time_weight(dates[i])
            if torch.isfinite(day_loss):
                losses.append(day_loss)
                comps.append([ic1, rk1, ic3, rk3, ic5, rk5, ret, ret5, ln, rdrop])
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
            self.log('tr_ret', c[6], on_step=True)
            self.log('tr_ret5', c[7], on_step=True)
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
            mixed, _, _, _, _ = self.model(tsdata)
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
                      if n.startswith('top_branch')]
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


def train_single(args, name, seed, train_date_list, valid_date_list,
                  style='cons', fam='a'):
    seed_everything(seed)
    logger = TensorBoardLogger(save_dir=params.model_path, name=name)
    litmodel = DLLitModule(args, style=style, fam=fam)
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

    print(f"[V20c] 加载因子数据: {fac_path}/{fac_name}.fea")
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

    # V20c: 16 折 = 族 a (fold 1..8) + 族 c (fold 9..16); 族内 k1-8: 4 折划分 × 2 风格
    fam, k = fold_spec(fold)
    split_fold = (k - 1) % N_FOLDS + 1
    style = 'cons' if k <= 4 else 'aggr'
    train_dates, valid_dates, test_dates = get_date_splits(date_list, fold=split_fold)
    print('=' * 70)
    print('  V20c 双族 16 折: 族 a (族 a (V11 多目标), 顶分支 1d 主) × 8 + 族 c (V13 顶部3d, 3d 主) × 8')
    print('  族内 k1-4 保守 (IC 主导) + k5-8 激进 (顶部主导); 同 4 折划分 × 2 风格 × 种子')
    print('  冠军集成: ens_w2 = z(Σ_a z(1·r1+0.25·r3+2·top)) + 2·z(Σ_c z(1·r1+1·top))')
    print('  共用 848 因子 + 同一 ridge 热启动; 打分 = z(lin_1d) + z(top) (analysis 组装)')
    print(f'  本折: 族={fam} 风格={style}  cfg={STYLE_CONS if style == "cons" else STYLE_AGGR}')
    print('  训练: AdamW(lin wd=1e-3, top wd=3e-3) + warmup+cosine, 时间衰减 hl=600d')
    print('=' * 70)
    print('  数据划分:')
    print(f'    Test  : {TEST_START} ~ {TEST_END} ({len(test_dates)} 个交易日)')
    print(f'    Valid : 全局 Fold({fold}/{TOTAL_FOLDS}, 族{k}, 划分复用 {split_fold}/{N_FOLDS}) '
          f'取 {len(valid_dates)} 天: {valid_dates[0]} ~ {valid_dates[-1]}')
    print(f'    Train : {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)} 天)')
    print(f'    factor_num={params.factor_num}, batch_size={args.batch_size}, '
          f'lr={args.lr}, epochs={args.epochs}, fold_seed={args.seed + k * 1000}')
    print('=' * 70)

    train_name = f'{season}/fold{fold}'
    fold_seed = args.seed + k * 1000
    train_single(args, train_name, fold_seed, train_dates, valid_dates,
                 style=style, fam=fam)


# ============================================================
# V20c 装配与报告工具 (analysis.py / run.py / train.sh 共用; 冠军协议与回测口径)
# ============================================================
import json as _json

SEASON = '2026q3'
MODEL_PRED = rf'{root_path}/model_pred/{SEASON}'
HEADS_A = MODEL_PRED + '/heads_a'     # 族 a 各头逐折 z 宽表: {h}_f{k}.fea, k=1..8
HEADS_C = MODEL_PRED + '/heads_c'     # 族 c 同上
CACHE_SCORE = MODEL_PRED + '/score_ens_w2.fea'
CALENDAR_PATH = PROJECT_ROOT + 'data/calendar.parquet'
PRICES_PATH = PROJECT_ROOT + 'data/daily_adj.parquet'
FAC_PATH = fac_path + fac_name + '.fea'
V11_STRAT = PROJECT_ROOT + 'Model/V11/strat_backtest.py'   # 冠军回测引擎 (全版本共用)
V11_HOLDINGS = PROJECT_ROOT + 'Model/V11/holdings.json'    # 纸面账户状态 (只读提示)

HOLD_DAYS = 5          # 冠军协议: 持有交易日数
STOP_LOSS = 0.08       # 收盘止损线
BAR = '─' * 60


# ---------- 打分装配 (与冠军定义一致) ----------
def zn(df):
    """逐日截面 z-score (行 = 日期, 列 = 股票)。"""
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]


def build_family(heads_dir, w1, w5, w3, wt, nf=N_FOLD_PER_FAMILY):
    """家族打分 = Σ_{f=1..8} z( w1·r1_f + w5·r5_f + w3·r3_f + wt·top_f )"""
    heads = ('r1', 'r3', 'r5', 'top')
    perfold = {h: {} for h in heads}
    for h in heads:
        for f in range(1, nf + 1):
            path = os.path.join(heads_dir, f'{h}_f{f}.fea')
            if os.path.exists(path):
                perfold[h][f] = pd.read_feather(path).set_index('date')
    if not perfold['r1']:
        raise FileNotFoundError(f'{heads_dir} 下无 r1 头文件 (先运行 analysis.py 推演)')
    score = None
    for f in sorted(perfold['r1'].keys()):
        fs = w1 * perfold['r1'][f]
        if w5 and 'r5' in perfold:
            fs = fs + w5 * perfold['r5'][f]
        if w3 and 'r3' in perfold:
            fs = fs + w3 * perfold['r3'][f]
        fs = fs + wt * perfold['top'][f]
        fs = zn(fs)
        score = fs if score is None else score.add(fs, fill_value=0.0)
    return score


def score_ens_w2():
    """冠军打分 ens_w2 = z(族a mixA) + 2·z(族c r1+top)。"""
    mixA = build_family(HEADS_A, 1.0, 0.0, 0.25, 2.0)
    v13 = build_family(HEADS_C, 1.0, 0.0, 0.0, 1.0)
    ens = zn(mixA).add(2.0 * zn(v13), fill_value=0.0)
    ens.index.name = 'date'
    ens = ens.sort_index()
    ens.columns = ens.columns.astype(str)
    return ens


# ---------- 交易日历 / 名称 / 日期 ----------
def load_trading_dates():
    cal = pd.read_parquet(CALENDAR_PATH)
    return sorted(cal.loc[cal['is_open'] == 1, 'date'].astype(str)
                  .str.replace('-', '', regex=False).tolist())


def load_name_map():
    df = pd.read_parquet(PRICES_PATH, columns=['stock_code', 'stock_name'])
    code = df['stock_code'].str.replace('.SZ', '', regex=False) \
                           .str.replace('.SH', '', regex=False)
    return (df.assign(code=code).drop_duplicates('code', keep='last')
            .set_index('code')['stock_name'].to_dict())


def next_td(d, n=1, tds=None):
    if tds is None:
        tds = load_trading_dates()
    i = tds.index(d) if d in tds else None
    if i is None:
        return None
    j = i + n
    return tds[j] if 0 <= j < len(tds) else None


def zh_date(s):
    s = str(s)
    return f'{int(s[:4])}年{int(s[4:6])}月{int(s[6:8])}日'


def factor_dates():
    """fac_all.fea 全部因子日期 (轻量读 date 列)。"""
    import pyarrow.compute as pc
    import pyarrow.feather as pf
    col = pf.read_table(FAC_PATH, columns=['date']).column('date')
    return sorted(set(pc.unique(col).to_pandas().astype(str)))


def latest_reportable_date():
    fd = set(factor_dates())
    td = set(load_trading_dates())
    inter = fd & td
    if not inter:
        raise RuntimeError('因子数据与交易日历无交集')
    return max(inter)


def heads_last_date(heads_dir):
    p = os.path.join(heads_dir, 'r1_f1.fea')
    if not os.path.exists(p):
        return None
    import pyarrow.compute as pc
    import pyarrow.feather as pf
    col = pf.read_table(p, columns=['date']).column('date')
    return max(pc.unique(col).to_pandas().astype(str))


# ---------- 冠军策略回测 (收盘卖引擎, 口径与冠军评估一致) ----------
def _close_sell_backtest_module():
    with open(V11_STRAT, encoding='utf-8') as f:
        src = f.read()
    old = "sp = open_map.get((p['planned_sell'], code))"
    assert old in src, 'strat_backtest.py 卖出价行未找到'
    mod = {}
    exec(src.replace(old, "sp = close_map.get((p['planned_sell'], code))"), mod)
    return mod


def champion_backtest(score=None, split=True, verbose=True):
    """冠军协议回测 (hold5 + stop8% + 收盘卖 + 含成本) 及可选 H1/H2 分半。

    返回 out, 含: full/h1/h2 (summarize 指标), ic, n_trades, trades (成交明细
    DataFrame), daily (逐笔日收益序列) 与 name_map (日志/出图用)。"""
    if score is None:
        score = score_ens_w2()
    mod = _close_sell_backtest_module()
    rb, summ = mod['run_backtest'], mod['summarize']
    open_map, close_map, prev_close_map, amount_map, name_map = mod['load_prices']()
    tds = mod['load_calendar']()
    F = dict(use_cost=True, exclude_st=True, exclude_limit_up=True,
             buy_gap_limit=0.095, hold=HOLD_DAYS, stop_loss=STOP_LOSS)

    def one(ws=None, we=None):
        kw = {}
        if ws:
            kw['window_start'] = ws
        if we:
            kw['window_end'] = we
        m, tr, dr = rb(score, open_map, close_map, prev_close_map, name_map,
                       amount_map=amount_map, tds=tds, **F, **kw)
        return m, tr, dr

    full, trades, daily = one()
    ic = mod['eval_ic'](score)
    out = {'full': summ(full), 'ic': ic, 'n_trades': len(trades),
           'trades': trades, 'daily': daily, 'name_map': name_map,
           'tds': tds, 'F': F}
    if split:
        h1, _, _ = one(ws=TEST_START, we='20260227')
        h2, _, _ = one(ws='20260302', we=TEST_END)
        out['h1'], out['h2'] = summ(h1), summ(h2)
    if verbose:
        s = out['full']
        print('===== 冠军策略回测 (hold5 + stop8% + 收盘卖, 含成本) =====')
        print(f"  RankIC={ic['RankIC']:+.4f}  RankICIR={ic['RankICIR']:+.4f}  "
              f"top_return={ic['top_return']:+.4f}")
        print(f"  净累计 {s['净累计%']:>8.2f}%  |  净年化 {s['净年化%']:>7.2f}%  |  "
              f"Sharpe {s['Sharpe']:>6.3f}  |  MaxDD {s['MaxDD%']:>7.2f}%  |  "
              f"胜率 {s['胜率%']:>5.1f}%  |  交易 {s['交易数']}  |  成本拖累 {s['成本拖累%']:>6.2f}%")
        if split:
            print(f"  分半 H1 ({TEST_START}~20260227): {out['h1']['净累计%']:>8.2f}%   |   "
                  f"H2 (20260302~{TEST_END}): {out['h2']['净累计%']:>8.2f}%")
    return out


# ---------- 报告块 (V9 风格排行榜 + 策略指令) ----------
def rank_block(date, score, topn=10, tds=None, name_map=None):
    if date not in score.index:
        raise KeyError(f'{date} 不在打分范围 ({score.index.min()}~{score.index.max()})')
    if name_map is None:
        name_map = load_name_map()
    if tds is None:
        tds = load_trading_dates()
    top = score.loc[date].sort_values(ascending=False).head(topn)
    buy_dt = next_td(date, 1, tds)
    sell_dt = next_td(buy_dt, HOLD_DAYS, tds) if buy_dt else None
    lines = [BAR,
             '  打分模型: ens_w2 = z(族a mixA) + 2·z(族c r1+top)  (V20c 自训 16 折)',
             f'  因子日期：{zh_date(date)} 收盘']
    if buy_dt:
        lines.append(f'  冠军策略：{zh_date(buy_dt)} 开盘买入 Top1 → 持有 {HOLD_DAYS} 个交易日'
                     f' ({(zh_date(sell_dt) if sell_dt else "?")} 收盘卖出); '
                     f'持仓期收盘较买入价下跌 ≥ {STOP_LOSS:.0%} → 次日开盘止损')
    lines += [BAR, '',
              '  排名   代码        名称                  打分',
              '  ' + '-' * 37]
    for i, code in enumerate(top.index, 1):
        lines.append(f'  {i:<4} {code:<9} {name_map.get(code, "?"):<10}'
                     f'   {score.loc[date, code]:+10.4f}')
    lines.append('')
    return '\n'.join(lines)


def decision_block(date, score, tds=None, name_map=None, holdings_path=None):
    if date not in score.index:
        raise KeyError(date)
    if name_map is None:
        name_map = load_name_map()
    if tds is None:
        tds = load_trading_dates()
    top = score.loc[date].sort_values(ascending=False)
    c1 = top.index[0]
    buy_dt = next_td(date, 1, tds)
    sell_dt = next_td(buy_dt, HOLD_DAYS, tds) if buy_dt else None
    lines = [BAR, '  冠军策略指令 (Top1 单票, 5W 账户口径)',
             f'  Top1: {c1}  {name_map.get(c1, "?")}   打分 {top[c1]:+.4f}',
             f'  动作: {zh_date(buy_dt) if buy_dt else "?"} 开盘买入 (整手) '
             f'→ 持有 {HOLD_DAYS} 个交易日 → '
             f'{zh_date(sell_dt) if sell_dt else "?"} 收盘卖出',
             f'  风控: 持仓期收盘较买入价下跌 ≥ {STOP_LOSS:.0%} → 次日开盘止损']
    if holdings_path and os.path.exists(holdings_path):
        try:
            st = _json.load(open(holdings_path, encoding='utf-8'))
            code = st.get('code')
            if code:
                lines.append(f'  纸面账户: 当前持仓 {code} {name_map.get(code, "?")}'
                             f' (买入 {st.get("buy_dt", "?")})')
                if code != c1:
                    lines.append(f'  ⚠ 纸面持仓 {code} 与今日 Top1 {c1} 不一致 — '
                                 f'状态可能过期/已人工调整, 以本指令为准')
            else:
                lines.append('  纸面账户: 空仓')
        except Exception:
            pass
    lines.append(BAR)
    return '\n'.join(lines)


# ---------- 冻结组件/训练产物健康检查 (train.sh / 启动前自检) ----------
def validate_frozen(verbose=True):
    """校验: 16 折训练产物齐备 + heads 覆盖 + 数据/日历。返回 (ok_all, [(ok,msg)])。"""
    import glob
    checks = []

    def chk(cond, msg):
        checks.append((bool(cond), msg))
        return bool(cond)

    train_base = rf'{root_path}/model_train/{SEASON}'
    folds = sorted(glob.glob(train_base + '/fold[0-9]*'))
    ckpts = glob.glob(train_base + '/fold*/version_*/checkpoints/*.ckpt')
    chk(len(folds) == TOTAL_FOLDS and len(ckpts) >= TOTAL_FOLDS,
        f'训练产物: {len(folds)}/16 个 fold, {len(ckpts)} 个最佳 checkpoint '
        f'(期望 16; 先 bash train.sh 全量训练)')
    chk(os.path.exists(rf'{root_path}/model_test/feature_map.fea'),
        'feature_map.fea 存在 (model_test; 训练时生成)')
    chk(os.path.exists(FAC_PATH) and os.path.exists(CALENDAR_PATH)
        and os.path.exists(PRICES_PATH), '基础数据存在 (fac_all/calendar/daily_adj)')
    chk(os.path.exists(rf'{root_path}/{RIDGE_INIT_FILE}'),
        f'ridge_init.csv 存在 ({RIDGE_INIT_FILE}, 线性头热启动)')
    try:
        latest = latest_reportable_date()
        a, c = heads_last_date(HEADS_A), heads_last_date(HEADS_C)
        cur = min(a, c) if (a and c) else None
        chk(cur is not None and cur >= latest,
            f'heads 覆盖因子最新日 ({cur} ≥ {latest}; 否则运行 analysis.py 增量推演)')
    except Exception as e:
        chk(False, f'日期覆盖检查异常: {e}')
    ok = all(o for o, _ in checks)
    if verbose:
        print('===== V20c 健康检查 =====')
        for o, m in checks:
            print(f'  [{"OK" if o else "FAIL"}] {m}')
        print(f'  → 结论: {"就绪 (analysis.py 可出结果)" if ok else "见 FAIL 项"}')
    return ok, checks


if __name__ == '__main__':
    ok, _ = validate_frozen()
    sys.exit(0 if ok else 1)
