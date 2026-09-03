import os
import random
import shutil
import sys
import warnings

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

cpu_num = min(16, os.cpu_count() or 8)
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# ============================================================
# 数据划分边界 (与 analysis.py 对齐)
# ============================================================
TEST_START = '20250901'   # Test 集合起点 (含): 打分/回测都基于这一时间段
TEST_END = '20260901'     # Test 集合终点 (含)
TRAIN_END = '20250809'    # 训练数据上限 (不含): 仅 20250809 之前 (即 20250810 以前) 可训练/验证
VALID_DAYS = 120          # 验证池 = 训练允许范围内最新 120 个交易日
N_FOLDS = 4               # 验证池均分为 4 份, 每 Fold 验证 1/4 (两两无交集、全覆盖)
PURGE_DAYS = 5            # 训练集与验证池之间的隔断交易日数
MIN_DAY_STOCKS = 50       # 截面最少有效股票数 (不足则该日跳过)
TOP_RET_FRAC = 0.1        # 收益率指标口径: 预测分最高的前 10%

# ============================================================
# 模型 / 损失常量 (硬编码, 不暴露为命令行超参)
# ============================================================
LABEL_WINSOR_MAD = 5.0    # 标签 MAD 截尾
RET_LOSS_SCALE = 100.0    # 收益损失项以百分比计, 与 IC/RankIC 同量级


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='每个 batch 的交易日数')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam 学习率')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--seed', type=int, default=3253)
    parser.add_argument('--data', default='fac_all', choices=['fac_all', 'fac_sample'],
                        help='因子数据文件 (trainingdata/{data}.fea)')
    args, _ = parser.parse_known_args()
    return args


args = parse_args()

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
root_path = PROJECT_ROOT + 'Model/V2'
fac_path = PROJECT_ROOT + 'trainingdata/'
fac_name = args.data
label_path = PROJECT_ROOT + 'trainingdata'
label_name = r'label_ret_1d'


class params:
    model_path = rf'{root_path}/model_train'
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    ret_1d_data = ret_data           # 别名 (analysis.py 打分/回测同口径)
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


def normed_data(data, date, factor_list):
    """训练/评估统一的单日数据处理:
    有效性过滤 → 拼接 1d 标签 (可交易口径 winsor + 全池原始口径) → 特征 0.5%/99.5% 截尾 → zscore → 填 0。
    """
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    data = data.dropna(subset=factor_list, thresh=valid_threshold).copy()
    label_row = _safe_label_row(params.ret_data, date)
    raw_label = label_row.reindex(data["Code"]).values
    data['Label_raw'] = np.nan_to_num(raw_label, nan=0.0)          # 全池原始收益 (ic_raw 口径)
    data['Label'] = _winsor_mad(pd.Series(raw_label, index=data.index))
    buyable_row = _safe_label_row(params.buyable_mask, date)
    data['buyable'] = buyable_row.reindex(data["Code"]).values
    not_buyable = ~(data['buyable'].fillna(False).astype(bool))
    if not_buyable.any():
        data.loc[not_buyable, 'Label'] = np.nan                    # 可交易口径: 不可交易 → NaN
    data_X = data[factor_list]
    quantiles = data_X.quantile([0.005, 0.995])
    data_X = data_X.clip(lower=quantiles.loc[0.005], upper=quantiles.loc[0.995], axis=1)
    data_X = (data_X - data_X.mean()) / data_X.std()
    data_X = data_X.fillna(0)
    data_x_np = np.nan_to_num(data_X.to_numpy(dtype=np.float32, copy=False),
                              nan=0.0, posinf=0.0, neginf=0.0)
    return (torch.from_numpy(data_x_np),
            torch.from_numpy(data['Label'].to_numpy(dtype=np.float32)),
            torch.from_numpy(data['Label_raw'].to_numpy(dtype=np.float32)),
            data['Code'].values,
            date)


def collate_fn(datas):
    data_X, data_y, data_y_raw, code_value, date = zip(*datas)
    return list(data_X), list(data_y), list(data_y_raw), list(code_value), list(date)


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
                          num_workers=min(8, cpu_num // 2) if torch.cuda.is_available() else 0,
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


def _soft_top_ret(preds, y):
    """可导收益项: softmax 加权组合相对等权组合的超额收益 (近似 Top-K 组合收益)。"""
    p = preds.reshape(-1)
    tau = 0.1 * p.std().detach().clamp_min(1e-6)
    w = torch.softmax((p - p.mean()) / tau, dim=0)
    return (w * y).sum() - y.mean()


def combined_loss(preds, y):
    """日截面组合损失 = -(IC + RankIC + 收益率), 三项等权。

    IC     = Pearson(pred, y)                        可导
    RankIC = 软秩 Spearman 近似                       可导
    收益率 = softmax 加权组合超额收益 ×100 (百分比)    可导
    """
    ic = _pearson(preds, y)
    rankic = _soft_rankic(preds, y)
    ret = RET_LOSS_SCALE * _soft_top_ret(preds, y)
    return -(ic + rankic + ret)


# ============================================================
# 模型
# ============================================================


class PredictModel(nn.Module):
    """MLP-4linear"""
    def __init__(self, input_dim=None):
        super(PredictModel, self).__init__()
        if input_dim is None:
            input_dim = params.factor_num

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )
        # 隐含层
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        # 输出层
        self.output_layer = nn.Linear(32, 1)
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
        x = self.net(x)
        x = self.output_layer(x)
        return x  


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
        tsdatas, labels, labels_raw, code_values, dates = batch
        losses = []
        for i in range(len(tsdatas)):
            y = labels[i]
            m = torch.isfinite(y)
            if m.sum() < MIN_DAY_STOCKS:
                continue
            pred = self.model(tsdatas[i]).squeeze(1)
            pm = pred[m]
            ym = y[m]
            if not torch.isfinite(pm).all():
                continue
            losses.append(combined_loss(pm, ym))
        if not losses:
            return None
        loss = torch.stack(losses).mean()
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss

    def _evaluate_step(self, batch, batch_idx):
        """单日验证评估: 可交易池 RankIC / 全池 Pearson IC / Top10% 组合日均收益。
        注意: Test 集合的推演/打分由 analysis.py 完成, 训练阶段不做任何 test 推演。
        """
        tsdatas, labels, labels_raw, code_values, dates = batch
        rank_ic_list, ic_raw_list, top_ret_list = [], [], []
        for i in range(len(tsdatas)):
            tsdata = tsdatas[i]
            y = labels[i]
            y_raw = labels_raw[i]
            pred = self.model(tsdata).squeeze(1)
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
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def configure_callbacks(self):
        return [ModelCheckpoint(monitor='val_rankic', mode='max', save_top_k=1,
                                filename='{epoch}-{val_rankic:.4f}')]


# ============================================================
# 数据划分与训练入口
# ============================================================

def get_date_splits(date_list, fold=1):
    """数据划分 (与 analysis.py 打分/回测区间对齐, 4-Fold):
    - Test : 20250901 ~ 20260901 (含), 固定打分区间, 不参与训练/验证, 各 Fold 共用。
    - 验证池: 训练允许范围 (<20250810) 内最新 VALID_DAYS 个交易日, 用固定种子洗牌后
      均分为 N_FOLDS 份; Fold f 取第 f 份作验证 (各 Fold 两两无交集且全覆盖)。
    - Train: 验证池之前隔断 PURGE_DAYS 个交易日的其余全部日期 (各 Fold 相同)。
    - 20250810 ~ 20250831 过渡区全部丢弃 (既不训练也不打分)。
    """
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
    # 进度条 refresh_rate=20: 输出频率比 V1(refresh_rate=10) 再降一半、比上一版(refresh_rate=1)
    # 降低 20 倍 —— 每个 epoch 仅 ~8 次更新, 日志体积大幅缩小且保留关键指标可见性
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

    print(f"[V2] 加载因子数据: {fac_path}/{fac_name}.fea")
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
    print(f'  V2 基线模型: 最基础全连接 MLP (input → 256 → 128 → 1, ReLU)')
    print(f'  损失: 日截面 -(IC + RankIC + 收益率), 三项等权;')
    print(f'  优化器: Adam(固定 lr), 无 lr 调度/早停/SWA/mixup/R-Drop;')
    print(f'  4-Fold: 验证池(最新 {VALID_DAYS} 交易日)洗牌均分 {N_FOLDS} 份, 各 Fold 验证 1/{N_FOLDS}')
    print('=' * 70)
    print('  数据划分:')
    print(f'    Test  : {TEST_START} ~ {TEST_END} ({len(test_dates)} 个交易日, '
          f'固定打分区间, 推演/打分由 analysis.py 完成, 训练不使用)')
    print(f'    Valid : 验证池最新 {VALID_DAYS} 个交易日, 本 Fold({fold}/{N_FOLDS}) 取其中 '
          f'{len(valid_dates)} 天: {valid_dates[0]} ~ {valid_dates[-1]}')
    print(f'    Train : {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)} 天), '
          f'与验证池隔断 {PURGE_DAYS} 天')
    print(f'    过渡区: {TRAIN_END} (不含) 起至 {TEST_START} (不含) 之前, 全部丢弃')
    print(f'    factor_num={params.factor_num}, batch_size={args.batch_size}, '
          f'lr={args.lr}, epochs={args.epochs}, fold_seed={args.seed + fold * 1000}')
    print('=' * 70)

    train_name = f'{season}/fold{fold}'
    fold_seed = args.seed + fold * 1000     # 各 Fold 独立种子
    train_single(args, train_name, fold_seed, train_dates, valid_dates)
