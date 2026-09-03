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

# 4 个 Fold 并发跑在同一 GPU 上, 且容器可见 CPU 有限 (25 核):
# 每进程 8 个 torch 线程 + 4 个 DataLoader worker, 避免 4 进程互相过度抢占
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
TEST_START = '20250901'   # Test 集合起点 (含)
TEST_END = '20260901'     # Test 集合终点 (含)
TRAIN_END = '20250809'    # 训练数据上限 (不含)
VALID_DAYS = 120          # 验证池 = 最新 120 个交易日
N_FOLDS = 4
PURGE_DAYS = 5
MIN_DAY_STOCKS = 50
TOP_RET_FRAC = 0.1

# ============================================================
# V4 损失/优化常量
# ============================================================
LABEL_WINSOR_MAD = 5.0     # 标签 MAD 截尾
IC_WEIGHT = 4.0            # -(IC+RankIC) 项权重 (V3 此项被无界收益项淹没, V4 恢复排序质量)
TOPRET_WEIGHT = 0.5        # 软 Top 组合收益项权重 (秩高斯标签, 有界 ±~2.5σ)
TOPRET_TAU_FRAC = 0.06     # 软 Top 组合温度 → 权重集中在前 ~5~10%
AUX_5D_WEIGHT = 0.35       # 5d 辅助头权重 (平滑标签, 正则化 + 信号共享)
RDROP_WEIGHT = 0.10        # R-Drop 一致性 (两次 dropout 前向的 MSE)
LISTNET_WEIGHT = 0.20      # ListNet 顶前 20 对齐 (直接优化头部排序, 服务 Top1 回测)
LISTNET_TAU_FRAC = 0.15    # ListNet 温度 = tau_frac × std(preds)
LISTNET_K = 20             # ListNet 目标: 真实收益最高的前 K 只等权
TIME_HALF_LIFE_DAYS = 600  # 训练日时间衰减半衰期 (相对 TRAIN_END; None=关闭)
WEIGHT_DECAY = 1e-3        # AdamW 解耦权重衰减
LR_MAX = 5e-4              # 学习率峰值
LR_MIN = 1e-5              # cosine 末端学习率
WARMUP_EPOCHS = 2          # 线性 warmup 轮数
EARLY_STOP_PATIENCE = 10   # val_rankic 早停耐心 (无 Trainer min_epochs: V3 曾因此静默退出)


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
root_path = PROJECT_ROOT + 'Model/V4'
fac_path = PROJECT_ROOT + 'trainingdata/'
fac_name = args.data
label_path = PROJECT_ROOT + 'trainingdata'
label_name = r'label_ret_1d'
label_5d_name = r'label_ret_5d'


class params:
    model_path = rf'{root_path}/model_train'
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    ret_1d_data = ret_data           # 别名 (analysis.py 打分/回测同口径)
    ret_5d_data = pd.read_feather(rf"{label_path}/{label_5d_name}.fea").set_index("index")
    buyable_mask = pd.read_feather(rf"{label_path}/buyable_mask.fea").set_index("date")


def _safe_label_row(label_df, date):
    """取标签行; 日期不在索引中时返回全 NaN 行, 由后续逻辑处理。"""
    if date in label_df.index:
        return label_df.loc[date]
    return pd.Series(np.nan, index=label_df.columns)


def _winsor_mad(s, k=LABEL_WINSOR_MAD):
    """MAD 截尾 (稳健去极值)。"""
    v = s.dropna()
    if len(v) == 0:
        return s
    med = v.median()
    mad = (v - med).abs().median()
    if mad <= 1e-12:
        return s
    return s.clip(med - k * mad, med + k * mad)


def _rank_gauss(s):
    """截面秩高斯化: 用于 IC/RankIC 回归项, 与秩指标同口径且对尾部稳健。"""
    finite = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype='float64')
    if finite.sum() >= MIN_DAY_STOCKS:
        r = s[finite].rank()
        r = (r - r.mean()) / r.std()
        out.loc[finite] = r
    return out


def _time_weight(date_str, ref_date=TRAIN_END, half_life=TIME_HALF_LIFE_DAYS):
    """训练样本时间衰减权重 (近期市场状态更重要)。"""
    if half_life is None:
        return 1.0
    try:
        d = datetime.strptime(date_str, '%Y%m%d')
        ref = datetime.strptime(ref_date, '%Y%m%d')
        return 0.5 ** (max(0, (ref - d).days) / half_life)
    except (ValueError, TypeError):
        return 1.0


def normed_data(data, date, factor_list):
    """训练/评估统一的单日数据处理:
    有效性过滤 → 1d/5d 标签 (可交易口径 winsor, 1d 额外秩高斯化) →
    特征截面秩高斯化 (V4: 替代 zscore, 分布平移/尾部稳健) → 缺失填 0。
    """
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    data = data.dropna(subset=factor_list, thresh=valid_threshold).copy()

    label_row = _safe_label_row(params.ret_data, date)
    raw_label = label_row.reindex(data["Code"]).values
    label_5d_row = _safe_label_row(params.ret_5d_data, date)
    raw_label_5d = label_5d_row.reindex(data["Code"]).values

    data['Label_raw'] = np.nan_to_num(raw_label, nan=0.0)          # 全池原始收益 (ic_raw 口径)
    data['Label'] = _winsor_mad(pd.Series(raw_label, index=data.index))
    data['Label_5d'] = _winsor_mad(pd.Series(raw_label_5d, index=data.index))

    buyable_row = _safe_label_row(params.buyable_mask, date)
    data['buyable'] = buyable_row.reindex(data["Code"]).values
    not_buyable = ~(data['buyable'].fillna(False).astype(bool))
    if not_buyable.any():
        data.loc[not_buyable, ['Label', 'Label_5d']] = np.nan         # 可交易口径: 不可交易 → NaN

    # 秩高斯化 (仅对可交易池): 用于 IC/RankIC 损失项
    data['Label_rg'] = _rank_gauss(data['Label'])
    data['Label_5d_rg'] = _rank_gauss(data['Label_5d'])

    # 特征: 截面秩高斯化 (每因子独立, NaN-aware) → 标准正态; 与 analysis.py normed_data 完全一致
    data_X = data[factor_list].rank(axis=0)
    data_X = (data_X - data_X.mean()) / data_X.std()
    data_X = data_X.fillna(0)
    data_x_np = np.nan_to_num(data_X.to_numpy(dtype=np.float32, copy=False),
                              nan=0.0, posinf=0.0, neginf=0.0)
    return (torch.from_numpy(data_x_np),
            torch.from_numpy(data['Label'].to_numpy(dtype=np.float32)),      # winsor 1d (top-ret / val)
            torch.from_numpy(data['Label_raw'].to_numpy(dtype=np.float32)),  # 全池原始
            torch.from_numpy(data['Label_rg'].to_numpy(dtype=np.float32)),   # 秩高斯 1d
            torch.from_numpy(data['Label_5d'].to_numpy(dtype=np.float32)),   # winsor 5d
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
    """Pearson 相关系数 (IC)。"""
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
    """Spearman 秩相关系数 (RankIC), 硬秩版 (评估指标口径)。"""
    def rk(x):
        return torch.argsort(torch.argsort(x)).float()
    return _pearson(rk(preds.reshape(-1)), rk(y.reshape(-1)))


def _soft_rank(x):
    """可导软秩 (Blondel 软秩公式), 温度自适应取 x.std。"""
    tau = x.std().detach().clamp_min(1e-6)
    diff = (x.unsqueeze(0) - x.unsqueeze(1)) / tau
    return 1.0 + torch.sigmoid(diff).sum(dim=1)


def _soft_rankic(preds, y):
    """可导 RankIC: Pearson(soft_rank(pred), rank(y))。"""
    rp = _soft_rank(preds.reshape(-1))
    ry = torch.argsort(torch.argsort(y.reshape(-1))).float()
    return _pearson(rp, ry)


def _soft_top_ret(preds, y, tau_frac=TOPRET_TAU_FRAC):
    """可导收益项: softmax 加权组合相对等权组合的超额收益 (近似 Top-K 组合收益)。"""
    p = preds.reshape(-1)
    tau = tau_frac * p.std().detach().clamp_min(1e-6)
    w = torch.softmax((p - p.mean()) / tau, dim=0)
    return (w * y).sum() - y.mean()


def _listnet_topk(preds, y, k=LISTNET_K, tau_frac=LISTNET_TAU_FRAC):
    """ListNet 风格头部对齐损失: CE(softmax(preds/τ), 真实 Top-K 等权目标)。
    直接优化头部排序质量 —— 服务 Top1 累计回测口径。"""
    p = preds.reshape(-1)
    y = y.reshape(-1)
    n = p.numel()
    if n < k + 2:
        return p.new_zeros(())
    tau = tau_frac * p.std().detach().clamp_min(1e-6)
    log_w = torch.log_softmax(p / tau, dim=0)
    _, top_idx = torch.topk(y, k)          # 真实收益最高的 K 只
    target = p.new_zeros(n)
    target[top_idx] = 1.0 / k
    return -(target * log_w).sum()


def head_loss(preds, y_rg, y_winsor, use_ret_term, use_listnet):
    """单个头的日截面损失 (V4: 全部有界, 稳定性优先):
    -(IC + RankIC)×IC_WEIGHT 用秩高斯标签 (与 RankIC 口径一致);
    收益项用秩高斯标签 (有界 ±~2.5σ, 等价于头部排序, 且对涨跌停尾部稳健);
    ListNet 头部对齐项。
    """
    ic = _pearson(preds, y_rg)
    rankic = _soft_rankic(preds, y_rg)
    loss = -IC_WEIGHT * (ic + rankic)
    ret = preds.new_zeros(())
    if use_ret_term:
        ret = TOPRET_WEIGHT * _soft_top_ret(preds, y_rg)
        loss = loss - ret
    ln = preds.new_zeros(())
    if use_listnet:
        ln = LISTNET_WEIGHT * _listnet_topk(preds, y_winsor)
        loss = loss + ln
    return loss, ic, rankic, ret, ln


# ============================================================
# 模型: MLP-Res —— 残差主干 + 双周期头 (1d 主 / 5d 辅)
# ============================================================

class ResBlock(nn.Module):
    """Pre-activation 风格残差块。"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.lin = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.lin(self.drop1(self.act1(self.bn1(x))))
        h = self.drop2(self.act2(self.bn2(h)))
        return x + h


class PredictModel(nn.Module):
    """MLP-Res-2: 848 → 512 → 2×ResBlock(512) → 128 → 双头 (1d / 5d)。"""
    def __init__(self, input_dim=None):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.res1 = ResBlock(512, dropout=0.2)
        self.res2 = ResBlock(512, dropout=0.2)
        self.neck = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.head_1d = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_5d = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
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
        x = self.res1(x)
        x = self.res2(x)
        x = self.neck(x)
        pred_1d = self.head_1d(x)
        pred_5d = self.head_5d(x)
        return pred_1d, pred_5d


class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel()
        print(self.model)
        self.validation_step_outputs = []

    def forward(self, tsdata):
        return self.model(tsdata)

    def training_step(self, batch, batch_idx):
        tsdatas, y1w_list, y1raw_list, y1rg_list, y5w_list, y5rg_list, code_values, dates = batch
        losses, comps = [], []
        for i in range(len(tsdatas)):
            y1w, y1rg = y1w_list[i], y1rg_list[i]
            y5w, y5rg = y5w_list[i], y5rg_list[i]
            m = torch.isfinite(y1w) & torch.isfinite(y1rg) & torch.isfinite(y5rg)
            if m.sum() < MIN_DAY_STOCKS:
                continue
            x = tsdatas[i]
            pred_1d, pred_5d = self.model(x)
            pm1, ym1r, ym1w = pred_1d.squeeze(1)[m], y1rg[m], y1w[m]
            pm5, ym5r, ym5w = pred_5d.squeeze(1)[m], y5rg[m], y5w[m]
            if not (torch.isfinite(pm1).all() and torch.isfinite(pm5).all()):
                continue

            loss_1d, ic1, rk1, ret1, ln1 = head_loss(pm1, ym1r, ym1w,
                                                     use_ret_term=True, use_listnet=True)
            loss_5d, ic5, rk5, ret5, ln5 = head_loss(pm5, ym5r, ym5w,
                                                     use_ret_term=True, use_listnet=False)

            # R-Drop: 第二次前向 (不同 dropout 掩码), 保持两次预测一致
            rdrop = 0.0
            if RDROP_WEIGHT > 0:
                pred_1d_2, _ = self.model(x)
                rdrop = nn.functional.mse_loss(pm1, pred_1d_2.squeeze(1)[m])

            day_loss = loss_1d + AUX_5D_WEIGHT * loss_5d + RDROP_WEIGHT * rdrop
            day_loss = day_loss * _time_weight(dates[i])
            if torch.isfinite(day_loss):
                losses.append(day_loss)
                comps.append([ic1, rk1, ret1, ln1, rdrop])
        if not losses:
            return None
        loss = torch.stack(losses).mean()
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        if comps:
            c = torch.stack([torch.stack(row) for row in comps]).mean(dim=0)
            self.log('tr_ic', c[0], on_step=True)
            self.log('tr_rankic', c[1], on_step=True)
            self.log('tr_ret', c[2], on_step=True)
            self.log('tr_listnet', c[3], on_step=True)
            self.log('tr_rdrop', c[4], on_step=True)
        return loss

    def _evaluate_step(self, batch, batch_idx):
        """单日验证评估: 可交易池 RankIC / 全池 Pearson IC / Top10% 组合日均收益。
        (口径与 V2 完全一致, 保证 checkpoint 选择与结果可比)"""
        tsdatas, y1w_list, y1raw_list, y1rg_list, y5w_list, y5rg_list, code_values, dates = batch
        rank_ic_list, ic_raw_list, top_ret_list = [], [], []
        for i in range(len(tsdatas)):
            tsdata = tsdatas[i]
            y = y1w_list[i]
            y_raw = y1raw_list[i]
            pred_1d, _ = self.model(tsdata)
            pred = pred_1d.squeeze(1)
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
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,
                                weight_decay=WEIGHT_DECAY)
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
    """数据划分 (与 analysis.py 打分/回测区间对齐, 4-Fold): 同 V2。"""
    date_list = sorted(date_list)
    train_allowed = [d for d in date_list if d < TRAIN_END]
    test_dates = [d for d in date_list if TEST_START <= d <= TEST_END]
    n = len(train_allowed)
    if n < VALID_DAYS + PURGE_DAYS + 10:
        raise ValueError(f'训练可用日期不足: {n} < {VALID_DAYS + PURGE_DAYS + 10}')
    if not 1 <= fold <= N_FOLDS:
        raise ValueError(f'fold 必须在 1~{N_FOLDS} 之间, 收到 {fold}')
    valid_pool = train_allowed[n - VALID_DAYS:]
    random.seed(args.seed)          # 固定种子洗牌: 所有 Fold 用同一份洗牌结果, 再均分
    random.shuffle(valid_pool)
    base, rem = divmod(len(valid_pool), N_FOLDS)
    f = fold - 1
    start = f * base + min(f, rem)
    size = base + (1 if f < rem else 0)
    valid_dates = sorted(valid_pool[start:start + size])
    train_dates = train_allowed[:n - VALID_DAYS - PURGE_DAYS]
    return train_dates, valid_dates, test_dates


def train_single(args, name, seed, train_date_list, valid_date_list):
    seed_everything(seed)
    logger = TensorBoardLogger(save_dir=params.model_path, name=name)
    litmodel = DLLitModule(args)
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
    # 不做 test 推演: Test 集合 (20250901~20260901) 的推演与打分由 analysis.py 完成


def train(args, season='2026q3', fold=1, state='train'):
    if state != 'train':
        raise NotImplementedError(state)
    save_path = rf"{root_path}/model_test"
    os.makedirs(save_path, exist_ok=True)
    try:
        shutil.copy(os.path.abspath(__file__), save_path)
    except Exception as e:
        print(e)

    print(f"[V4] 加载因子数据: {fac_path}/{fac_name}.fea")
    params.all_data = pd.read_feather(rf'{fac_path}/{fac_name}.fea')
    date_list = sorted(set(params.all_data['date'].unique()) & set(params.ret_data.index))
    params.all_data = params.all_data.set_index('date').sort_index()

    all_zero_mask = (params.all_data == 0).all(axis=0)
    params.all_data = params.all_data.loc[:, ~all_zero_mask]
    params.factor_list = list(params.all_data.columns[1:])   # 第 1 列为 Code
    params.factor_num = len(params.factor_list)
    with open(rf'{save_path}/feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(params.factor_list):
            file.write(rf'{factor_name}={idx}\n')

    train_dates, valid_dates, test_dates = get_date_splits(date_list, fold=fold)
    print('=' * 70)
    print('  V4: MLP-Res 残差主干 (848→512→2×ResBlock→128) + 双头 (1d 主 / 5d 辅)')
    print('  特征: 截面秩高斯化 (替代 zscore, 与 analysis.py 推理一致)')
    print('  损失: 1d [-4×(IC+RankIC) - 0.5×TopRet(秩) + 0.2×ListNet-Top20] + 0.1×R-Drop')
    print('        + 0.35 × 5d [-4×(IC+RankIC) - 0.5×TopRet(秩)]')
    print('  标签: MAD 截尾 + 截面秩高斯化 (IC/RankIC/收益项统一用秩标签, 全有界)')
    print('  训练: AdamW(wd=1e-3) + warmup+cosine, 时间衰减 hl=600d, 早停 patience=10')
    print('=' * 70)
    print('  数据划分:')
    print(f'    Test  : {TEST_START} ~ {TEST_END} ({len(test_dates)} 个交易日)')
    print(f'    Valid : 本 Fold({fold}/{N_FOLDS}) 取 {len(valid_dates)} 天: {valid_dates[0]} ~ {valid_dates[-1]}')
    print(f'    Train : {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)} 天)')
    print(f'    factor_num={params.factor_num}, batch_size={args.batch_size}, '
          f'lr={args.lr}, epochs={args.epochs}, fold_seed={args.seed + fold * 1000}')
    print('=' * 70)

    train_name = f'{season}/fold{fold}'
    fold_seed = args.seed + fold * 1000     # 各 Fold 独立种子
    train_single(args, train_name, fold_seed, train_dates, valid_dates)
