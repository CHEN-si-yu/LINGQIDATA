#!/usr/bin/env python3
"""
analysis.py — Converted from analysis.ipynb

Usage:
  cd /root/autodl-fs/lingqiData/Model/V6.2_1 && python analysis.py

Images saved to: ./model_pic/figure_*.png
Print output saved to: ./model_pic/output.md (also streamed to stdout)
"""

import sys, os, atexit

# ============================================================
# Setup: output redirection + plt.show → savefig
# ============================================================

# ── Output directory ──
PIC_DIR = './model_pic'
os.makedirs(PIC_DIR, exist_ok=True)

# ── Figure counter for auto-naming saved images ──
_fig_counter = [0]  # mutable counter so nested calls work

# ── Tee: duplicate print output to both stdout and a .md file ──
class _Tee:
    """Write to both the real stdout and a markdown log file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout  # keep reference to *real* stdout
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

# ── Activate output capture ──
_md_path = os.path.join(PIC_DIR, 'output.md')
_tee = _Tee(_md_path)
_real_stdout = sys.stdout  # capture real stdout before replacement below
sys.stdout = _tee

# ── Restore stdout & close .md file on exit ──
def _cleanup():
    if sys.stdout is _tee:
        sys.stdout = _tee.stdout  # restore real stdout
    _tee.close()
atexit.register(_cleanup)

# ── Monkey-patch plt.show() → plt.savefig() ──
import matplotlib.pyplot as _plt
_original_show = _plt.show
def _savefig_show(*args, **kwargs):
    """Replace plt.show() with savefig to PIC_DIR/."""
    _fig_counter[0] += 1
    fname = os.path.join(PIC_DIR, f"figure_{_fig_counter[0]:02d}.png")
    _plt.savefig(fname, dpi=150, bbox_inches="tight")
    # Use original stdout so this message is NOT double-logged in .md
    _real_stdout.write(f"[Figure saved] {fname}\n")
    _real_stdout.flush()
    _plt.close()
_plt.show = _savefig_show

# ======================================================================
# Cell 0 [code]
# ======================================================================
import re, os, sys, glob, warnings, gc, pickle, argparse, random
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.feather as pf, pyarrow.compute as pc

import torch, torch.nn.functional as F
from torch import nn
from numpy.linalg import eigh

import matplotlib.pyplot as plt

import numpy.core.numeric as _ncn
if 'numpy._core.numeric' not in sys.modules:
    sys.modules['numpy._core.numeric'] = _ncn

warnings.filterwarnings("ignore")
plt.style.use('default')  # 防止深色主题影响图表背景

# ======================================================================
# Cell 1 [code]
# ======================================================================
# 从 model.py 导入共享配置（路径、因子名、标签名、预加载数据）
from model import (PROJECT_ROOT, root_path, fac_path as _fac_dir, fac_name,
                   label_path, label_name, liquid_path, liquid_name, params,
                   PredictModel, parse_args)

# ── Analysis-only 配置 ──
fac_path = _fac_dir + fac_name + '.fea'    # 因子数据完整路径
ret_1d_name = r'label_ret_1d'              # 回测用的实际 1 日收益

start = '20250101'
end = '20260331'
calendar_path = PROJECT_ROOT + 'data/calendar.parquet'
watchlist_path = PROJECT_ROOT + 'MyCode.txt'

model_test_path = rf'{root_path}/model_test'
model_res_path = rf'{root_path}/model_res'
model_train_base = rf'{root_path}/model_train'

bench1_path = PROJECT_ROOT + "Model/bench/20260618.fea"
bench2_path = None
bench3_path = None
bench4_path = None

PERSONAL_TOP_N = 1           # 个人模式持仓股票数
TOP_N = 1                 # 买入股票数量
EXCLUDE_LIMIT_UP = False  # 是否排除涨停板（涨幅>=9.5%）
LABEL_HORIZON_DAYS = 1     # 持仓天数：单日换手（1天）
SEASON = '2026q1'         # 预测季度


# ======================================================================
# Cell 2 [code]
# ======================================================================
# ============================================================
# 所有函数与类定义（引擎函数 + 分析函数）
# ============================================================

# ── 引擎基础函数 ──────────────────────────────────────────────

_VAL_WEI_RE = re.compile(r'val_wei=(-?\d+\.\d+)')

def load_feature_map(model_test_path):
    import glob as _glob
    feat_path = os.path.join(model_test_path, 'feature_map.fea')
    if not os.path.exists(feat_path): return None
    factor_order = []
    with open(feat_path, 'r') as f:
        raw = f.read()
    raw = raw.replace('\\n', '\n')
    for line in raw.split('\n'):
        line = line.strip()
        if '=' in line:
            name, idx = line.rsplit('=', 1)
            factor_order.append((int(idx.strip()), name.strip()))
    factor_order.sort(key=lambda x: x[0])
    return [name for _, name in factor_order]

# ── Data preprocessing ─────────────────────────────────────────────────

def normed_data(data, factor_list):
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    valid_mask = data[factor_list].notna().sum(axis=1) >= valid_threshold
    data = data.loc[valid_mask].copy()
    code_value = data['Code'].values; data_X = data[factor_list]
    quantiles = data_X.quantile([0.005, 0.995])
    data_X = data_X.clip(lower=quantiles.loc[0.005], upper=quantiles.loc[0.995], axis=1)
    data_X = (data_X - data_X.mean()) / data_X.std()
    data_X = data_X.fillna(0)
    data_x_np = np.nan_to_num(data_X.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(data_x_np), code_value

def load_model(checkpoint_path, input_dim, **kwargs):
    """Load a trained checkpoint. Extra kwargs accepted for backward compatibility."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    model_state = {k.removeprefix('model.'): v for k, v in state_dict.items() if k.startswith('model.')}
    model = PredictModel(input_dim=input_dim, **kwargs)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing: print(f'[load_model] WARNING: missing keys (will use random init): {list(missing)[:10]}...')
    if unexpected:
        print(f"[load_model] WARNING: unexpected keys in checkpoint (will be ignored): {unexpected}")
    model.eval()
    return model

def find_best_checkpoint(fold_dir):
    ckpt_files = glob.glob(os.path.join(fold_dir, '**', '*.ckpt'), recursive=True)
    if not ckpt_files: raise FileNotFoundError(f'No ckpt under {fold_dir}')
    best_ckpt, best_val = None, -float('inf')
    for ckpt in ckpt_files:
        m = _VAL_WEI_RE.search(os.path.basename(ckpt))
        if m is None: continue
        val = float(m.group(1))
        if val > best_val: best_val = val; best_ckpt = ckpt
    if best_ckpt is None: raise RuntimeError(f'No val_wei parsed under {fold_dir}')
    return best_ckpt, best_val

def discover_folds(model_dir):
    folds = sorted(glob.glob(os.path.join(model_dir, 'fold[0-9]*')))
    if not folds: raise FileNotFoundError(f'No folds under {model_dir}')
    return folds

def predict_date(model, date, all_data, factor_list):
    data = all_data.loc[date].copy()
    data_X, code_value = normed_data(data, factor_list)
    with torch.no_grad():
        try:
            output = model(data_X, training_views=False)
        except TypeError:
            output = model(data_X)
        # Handle generic model output formats:
        # - tensor: use directly (e.g. V3.2 simple MLP)
        # - tuple with list: stack+mean (e.g. V3.0 multi-view training mode)
        # - tuple with tensor: use first (e.g. V3.0 inference mode)
        if isinstance(output, tuple):
            first = output[0]
            score = torch.stack(first, dim=0).mean(dim=0) if isinstance(first, list) else first
        elif isinstance(output, list):
            score = torch.stack(output, dim=0).mean(dim=0)
        else:
            score = output
    result = pd.DataFrame(score.detach().cpu().numpy(), index=code_value, columns=['value'])
    result.index.name = 'Code'
    return result

def predict_dates(dates, fac_path, model_train_base, season='2026q2', model_test_path=None):
    if not dates: return None
    if model_test_path is None: model_test_path = model_train_base.replace('model_train', 'model_test')

    # ── Step 1: read training factor names from feature_map.fea ──
    train_factors = load_feature_map(model_test_path)

    # ── Step 2: get ALL factor columns currently in the database ──
    all_cols = pd.read_feather(fac_path, columns=[]).columns.tolist()
    db_factor_cols = [c for c in all_cols if c not in ('date', 'Code')]

    if train_factors is not None:
        # Factors that exist both in feature_map AND in current database
        available = [f for f in train_factors if f in all_cols]
        # Factors in feature_map but MISSING from current database → fill with 0
        missing = [f for f in train_factors if f not in all_cols]
        if missing:
            print(f"[predict] WARNING: {len(missing)} training factors missing from DB, will fill with 0")
        cols_to_load = ['date', 'Code'] + available
        print(f"[predict] feature_map has {len(train_factors)} factors, "
              f"{len(available)} available in DB, {len(missing)} missing")
    else:
        cols_to_load = None
        missing = []
        available = []

    # ── Step 3: load factor data for the requested dates ──
    import pyarrow as pa, pyarrow.feather as pf, pyarrow.compute as pc
    table = pf.read_table(fac_path, columns=cols_to_load)
    mask = pc.is_in(table.column('date'), pa.array(dates))
    table = table.filter(mask)
    all_data = table.to_pandas(); del table
    all_data = all_data.set_index('date', drop=True).sort_index()

    # Fill missing training factors with 0
    if train_factors is not None and missing:
        for f in missing:
            all_data[f] = 0.0

    # ── Step 4: peek at a checkpoint to get the ACTUAL expected input dimension ──
    model_dir = os.path.join(model_train_base, season)
    folds = discover_folds(model_dir)
    first_ckpt_path, _ = find_best_checkpoint(folds[0])
    ckpt_meta = torch.load(first_ckpt_path, map_location='cpu', weights_only=False)
    ckpt_input_dim = None
    for key, tensor in ckpt_meta['state_dict'].items():
        if 'weight' in key and len(tensor.shape) == 2:
            ckpt_input_dim = tensor.shape[1]  # [out_features, in_features]
            break
    del ckpt_meta  # free memory

    # ── Step 5: build factor_list, aligned to checkpoint input dimension ──
    factor_list = train_factors if train_factors is not None else db_factor_cols

    if ckpt_input_dim is not None and ckpt_input_dim != len(factor_list):
        shortage = ckpt_input_dim - len(factor_list)
        print(f"[predict] Checkpoint expects {ckpt_input_dim} features, "
              f"feature_map has {len(factor_list)} ({shortage:+d})")

        if shortage > 0:
            # Need MORE factors: supplement from DB columns not already in factor_list
            extra_cols = [c for c in db_factor_cols if c not in factor_list]
            supplement = extra_cols[:shortage]
            factor_list = factor_list + supplement

            # Load the supplementary columns from DB
            if supplement:
                supp_table = pf.read_table(fac_path, columns=['date', 'Code'] + supplement)
                supp_mask = pc.is_in(supp_table.column('date'), pa.array(dates))
                supp_table = supp_table.filter(supp_mask)
                supp_df = supp_table.to_pandas().set_index('date', drop=True).sort_index()
                del supp_table
                for col in supplement:
                    all_data[col] = supp_df[col]
                print(f"[predict] Supplemented {len(supplement)} extra columns from DB: {supplement[:5]}..."
                      if len(supplement) > 5 else f"[predict] Supplemented {len(supplement)} extra columns: {supplement}")

            # If STILL not enough, pad with zero-filled placeholders
            still_short = ckpt_input_dim - len(factor_list)
            if still_short > 0:
                for i in range(still_short):
                    placeholder = f'_padding_{i}'
                    factor_list.append(placeholder)
                    all_data[placeholder] = 0.0
                print(f"[predict] Padded {still_short} zero placeholder(s) to reach {ckpt_input_dim}")
        else:
            # Too many factors: trim to checkpoint dimension
            factor_list = factor_list[:ckpt_input_dim]
            print(f"[predict] Trimmed factor_list from {len(train_factors)} to {ckpt_input_dim}")

    print(f"[predict] Final factor_list: {len(factor_list)} features → model input_dim={len(factor_list)}")

    try:
        bias_idx = factor_list.index('bias_20')
    except ValueError:
        bias_idx = None

    # ── Step 6: load models & predict ──
    print(f"[predict] Using {len(folds)} folds")
    models = {}
    for fold_dir in folds:
        fn = os.path.basename(fold_dir)
        ckpt, val = find_best_checkpoint(fold_dir)
        print(f"[predict]   [{fn}] {os.path.basename(ckpt)} (val_wei={val:.4f})")
        models[fn] = load_model(ckpt, len(factor_list), near_limit_up_idx=bias_idx)

    new_scores = []
    for date in dates:
        if date not in all_data.index: continue
        fold_preds = [predict_date(m, date, all_data, factor_list) for m in models.values()]
        zscored = [(f - f.mean()) / f.std() for f in fold_preds]
        ensemble = sum(zscored)
        date_score = ensemble['value']; date_score.name = date
        new_scores.append(date_score)
        print(f"[predict]   [{date}] done, {len(ensemble)} stocks")
    if not new_scores: return None
    new_score_df = pd.DataFrame(new_scores); new_score_df.index.name = 'date'
    print(f"[predict] Done, {len(new_score_df)} days.")
    return new_score_df

def _load_trading_dates(calendar_path):
    cal = pd.read_parquet(calendar_path)
    return set(cal[cal['is_open'] == 1]['date'].astype(str).str.replace('-', '').tolist())

def save_predictions(score_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for date in score_df.index:
        s = score_df.loc[date].copy(); s.index.name = 'Code'
        s.to_pickle(os.path.join(output_dir, f'{date}.pkl'))

def concat_model_4fold(test_path, res_path):
    market = 'ALL'; cache_path = rf"{res_path}/{market}_zscore_score.fea"
    if os.path.exists(cache_path): return pd.read_feather(cache_path).set_index("date")
    model_res = []
    for fold in range(1, 5):
        fold_dirs = [d for d in os.listdir(test_path) if d == f'fold{fold}']
        if len(fold_dirs) == 0: continue
        data_path = rf"{test_path}/{fold_dirs[0]}"
        date_list = sorted(set([x[:8] for x in os.listdir(data_path) if x[:8].isdigit()]))
        all_res = []
        for date in date_list:
            date_res = pd.read_pickle(f'{data_path}/{date}.pkl').T
            date_res.index = [date] * len(date_res)
            all_res.append(date_res)
        all_res = pd.concat(all_res, axis=0).sort_index()
        all_res.index.name = "date"
        model_res.append(all_res.apply(lambda x: (x - x.mean()) / x.std(), axis=1))
    model_res = sum(model_res)
    os.makedirs(res_path, exist_ok=True)
    model_res.reset_index().to_feather(cache_path)
    return model_res

def get_ret_ic(score, ret_data, start='20230101', end='20241231', top_n=1):
    """回测函数: ALL-IN Top-N 股票, 等权平均隔日收益, 无流动性约束.

    Returns:
        ret_series: 日均收益率 (百分比, 例如 0.5 = 0.5%)
        ic_series:  IC 序列
    """
    model_score = score.copy()
    label_ret, ic, datelist = [], [], []
    for date in model_score.loc[start:end].index:
        datelist.append(date)
        code_rank = model_score.loc[date].sort_values(ascending=False)
        ret = ret_data.loc[date].reindex(code_rank.index).fillna(0)

        # ALL-IN: 选 Top-N 只, 等权, 收益率 ×100 统一为百分比
        selected = code_rank.index[:top_n]
        daily_r = ret[selected].mean() * 100

        label_ret.append(daily_r)
        ic.append(code_rank.corr(ret * 100))
    return pd.Series(label_ret, index=datelist, dtype='float'), pd.Series(ic, index=datelist, dtype='float')

def get_metrics(ret, ic):
    return {"IC": ic.mean(), "ICIR": ic.mean() / ic.std() if ic.std() != 0 else 0,
            "top_return": ret.mean(), "top_return_stability": ret.mean() / ret.std() if ret.std() != 0 else 0}

def ensemble_scores(*dfs):
    result_list = []
    for i, df in enumerate(dfs, 1):
        df_norm = df.copy().apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        result_list.append(df_norm.stack().rename(f"score{i}"))
    return pd.concat(result_list, axis=1).dropna()

def run_individual_strategy(model_score, ret_data, top_n=5, exclude_limit_up=True,
                            limit_up_threshold=0.095, start='20230101', end='20260331'):
    model_score_sub = model_score.loc[start:end]
    daily_ret, skip_total, skip_days, datelist = [], 0, 0, []
    for date in model_score_sub.index:
        ranked = model_score_sub.loc[date].sort_values(ascending=False)
        selected, skipped = [], 0
        for code in ranked.index:
            if len(selected) >= top_n: break
            if exclude_limit_up:
                r_val = ret_data.loc[date, code] if code in ret_data.columns else 0.0
                if pd.notna(r_val) and r_val >= limit_up_threshold: skipped += 1; continue
            selected.append(code)
        r = ret_data.loc[date, selected].mean() if len(selected) > 0 else 0.0
        datelist.append(date); daily_ret.append(r)
        skip_total += skipped
        if skipped > 0: skip_days += 1
    daily_ret = pd.Series(daily_ret, index=datelist, dtype='float')
    stats = {'skip_total': skip_total, 'skip_days': skip_days, 'total_days': len(datelist),
             'skip_day_pct': skip_days / len(datelist) * 100 if len(datelist) > 0 else 0}
    return daily_ret, stats

# ── 分析函数 ──────────────────────────────────────────────────


def print_quarterly_metrics(ret, ic):
    """打印每个季度的收益和IC指标"""
    ret_series = ret.copy()
    ic_series = ic.copy()
    ret_series.index = pd.to_datetime(ret_series.index)
    ic_series.index = pd.to_datetime(ic_series.index)
    quarterly_ret = ret_series.groupby(pd.Grouper(freq='QE')).mean()
    quarterly_ic = ic_series.groupby(pd.Grouper(freq='QE')).mean()
    quarterly_ret.index = [f"{idx.year}Q{idx.quarter}" for idx in quarterly_ret.index]
    quarterly_ic.index = [f"{idx.year}Q{idx.quarter}" for idx in quarterly_ic.index]
    quarterly_df = pd.DataFrame({
        '季度平均收益': quarterly_ret.round(4),
        '季度平均IC': quarterly_ic.round(4)
    })
    print("\n===== 季度表现指标 =====")
    print(quarterly_df)
    print("=======================\n")


def print_metrics(model_metrics, bench_metrics=None, title="模型评估结果"):
    metric_keys = list(model_metrics.keys())
    df = pd.DataFrame(index=metric_keys)
    df.index.name = "指标"
    df["模型值"] = [model_metrics[k] for k in metric_keys]
    if bench_metrics is not None:
        df["基准值"] = [bench_metrics.get(k, None) for k in metric_keys]
        df["提升值"] = ((df["模型值"] - df["基准值"]) / df["基准值"]) * 100
    print(f"\n{title}")
    print(df.round(4))


def plot_model(model_score, bench_score, ret_1d_data, start='20250101', end='20260331', top_n=1):
    model_ret, _ = get_ret_ic(model_score, ret_1d_data, start=start, end=end, top_n=top_n)
    model_ret = model_ret / 100
    bench_ret, _ = get_ret_ic(bench_score, ret_1d_data, start=start, end=end, top_n=top_n)
    bench_ret = bench_ret / 100

    # 对齐日期：取两个序列的交集，避免 model_score / bench_score 日期不一致
    common_idx = model_ret.index.intersection(bench_ret.index)
    model_ret = model_ret.loc[common_idx]
    bench_ret = bench_ret.loc[common_idx]

    dates = model_ret.index.tolist()
    seen_months = set()
    month_ticks = []
    month_labels = []
    for i, d in enumerate(dates):
        month = d[:6]
        if month not in seen_months:
            seen_months.add(month)
            month_ticks.append(i)
            month_labels.append(d)

    # 1. 每日收益率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(dates, model_ret.cumsum(), label='model', color='blue')
    plt.plot(dates, bench_ret.cumsum(), label='bench', color='orange')
    plt.title('Cumulative Return Of Model & Bench', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.xticks(ticks=month_ticks, labels=month_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. 相对提升
    plt.figure(figsize=(12, 6))
    relative_improvement = model_ret - bench_ret
    plt.plot(dates, relative_improvement.cumsum(), label='model', color='blue')
    plt.title('Relative Improvement', fontsize=14)
    plt.ylabel('Relative Improvement', fontsize=12)
    plt.xticks(ticks=month_ticks, labels=month_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
# ---- 辅助函数 ----
def print_strategy(label, ret_series):
    ann = ret_series.mean() * 252 * 100
    sharpe = (ret_series.mean() / ret_series.std() * np.sqrt(252)
              if ret_series.std() > 0 else 0)
    q_avg = ret_series.groupby(
        pd.to_datetime(ret_series.index).to_period('Q')
    ).mean().mean() * 100
    print(f"{label:<35} {ann:>8.2f}% {sharpe:>8.4f} {q_avg:>10.4f}%")


def load_bench(bench_path):
    if bench_path is not None and os.path.exists(bench_path):
        return pd.read_feather(bench_path).set_index("date")
    return None

def run_new_top1_open_red(model_score, ret_data, open_red_dict, start, end):
    """new_Top 1: 从模型得分最高的股票中，选出当天开盘红的第一个（只买一只）。

    Args:
        ret_data: 应为 label_ret_1d（实际 1 日收益），非训练 target
    """
    model_score_sub = model_score.loc[start:end]
    daily_ret, datelist, hit_days = [], [], 0
    for date in model_score_sub.index:
        ranked = model_score_sub.loc[date].sort_values(ascending=False)
        red_set = open_red_dict.get(date, set())
        if not red_set:
            datelist.append(date)
            daily_ret.append(0.0)
            continue
        selected = None
        for code in ranked.index:
            if code in red_set:
                selected = code
                break
        if selected is not None:
            r = ret_data.loc[date, selected] if selected in ret_data.columns else 0.0
            hit_days += 1
        else:
            r = 0.0
        datelist.append(date)
        daily_ret.append(r)
    ret_series = pd.Series(daily_ret, index=datelist, dtype='float')
    print(f"  new_Top 1 (开盘红): 共 {len(datelist)} 天, 命中 {hit_days} 天 ({hit_days/len(datelist)*100:.1f}%)")
    return ret_series


# 原始 Top 1/3/5/10/20 —— 使用实际 1 日收益

def get_nth_next_trade_date(date_str, n=1):
    """获取 date_str 之后第 n 个交易日"""
    try:
        idx = _trade_dates.index(date_str)
        if idx + n < len(_trade_dates):
            return _trade_dates[idx + n]
    except (ValueError, IndexError):
        pass
    return None

def fmt_date(date_str):
    if date_str is None:
        return "待定（交易日历未覆盖）"
    return f"{date_str[:4]}年{int(date_str[4:6])}月{int(date_str[6:8])}日"

def _next_td(date_str, n=1):
    """Get the n-th trading day after (n>0) or before (n<0) date_str."""
    try:
        idx = _td_list.index(date_str)
        t = idx + n
        if 0 <= t < len(_td_list):
            return _td_list[t]
    except (ValueError, IndexError):
        pass
    return None


# ── CJK-aware display helpers ──────────────────────────────────────────

def _cjk_width(s):
    """Display width: CJK chars ≈ 2, ASCII ≈ 1."""
    w = 0
    for ch in str(s):
        w += 2 if ord(ch) > 127 else 1
    return w

def _pad_cjk(s, w, align='<'):
    """Pad string to display width w, accounting for CJK characters."""
    s = str(s)
    cur = _cjk_width(s)
    pad = max(0, w - cur)
    if align == '>': return ' ' * pad + s
    elif align == '^': l = pad // 2; r = pad - l; return ' ' * l + s + ' ' * r
    return s + ' ' * pad


# ── Backtest shared utilities ──────────────────────────────────────────

def _load_daily_adj():
    """Load daily_adj price data (cached). Returns DataFrame."""
    DAILY_ADJ_PATH = "/root/autodl-fs/lingqiData/data/daily_adj.parquet"
    try:
        return daily_adj
    except NameError:
        pass
    df = pd.read_parquet(DAILY_ADJ_PATH)
    df['trade_date'] = df['trade_date'].str.replace('-', '')
    df['code_clean'] = df['stock_code'].str.replace('.SZ', '').str.replace('.SH', '')
    df = df.sort_values(['code_clean', 'trade_date'])
    df['prev_close'] = df.groupby('code_clean')['close'].shift(1)
    return df


def _load_name_map():
    """Load stock code → name mapping (cached). Returns dict."""
    try:
        return code_to_name
    except NameError:
        pass
    stock_info = pd.read_parquet(PROJECT_ROOT + 'data/list.parquet')
    stock_info = stock_info[stock_info['list_status'] == 'L']
    return {
        re.sub(r'\.(SZ|SH|BJ)$', '', row['stock_code']): row['name']
        for _, row in stock_info.iterrows()
    }


def load_backtest_setup():
    """Load all shared data for backtest cells: price maps + name map.
    
    Also initializes module-level _td_list (used by _next_td) and _trade_dates
    (used by get_nth_next_trade_date).

    Returns:
        close_map      : dict (trade_date, code_clean) → close
        prev_close_map : dict (trade_date, code_clean) → prev_close
        code_to_name   : dict code → name
    """
    global _td_list, _trade_dates
    
    # --- Calendar ---
    cal = pd.read_parquet(PROJECT_ROOT + 'data/calendar.parquet')
    cal = cal[cal['is_open'] == 1]
    _td_list = sorted(cal['date'].astype(str).str.replace('-', '').tolist())
    _trade_dates = _td_list  # alias for get_nth_next_trade_date compatibility
    
    # --- Price data ---
    df = _load_daily_adj()
    print(f"Price data: {len(df)} rows, "
          f"dates {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    close_map      = df.set_index(['trade_date', 'code_clean'])['close'].to_dict()
    prev_close_map = df.set_index(['trade_date', 'code_clean'])['prev_close'].to_dict()
    code_to_name   = _load_name_map()

    return close_map, prev_close_map, code_to_name


def build_trade_log_df(model_score_extended, recent_dates, close_map, prev_close_map,
                        code_to_name, top_n=1, exclude_limit_up=False, label_horizon_days=1):
    """Build trade log records and daily return series from model scores & price data.

    Returns:
        trade_df   : DataFrame with columns [FactorDt, BuyDt, Code, Name, Score,
                     BuyPrc, SellDt, SellPrc, Ret%, 累加收益]
        ret_valid  : Series of realized daily returns
        skip_total : int, number of stocks skipped by limit-up filter
        skip_days  : int, number of days with at least one skip
    """
    records = []
    daily_returns = {}
    skip_total = 0
    skip_days = 0

    for date in recent_dates:
        scores = model_score_extended.loc[date].sort_values(ascending=False)

        # Stock selection: top N by model score
        picked = []
        day_skipped = 0
        for code in scores.index:
            if len(picked) >= top_n:
                break
            if exclude_limit_up:
                fc = close_map.get((date, code))
                pc = prev_close_map.get((date, code))
                if fc is not None and pc is not None and pc > 0:
                    if fc / pc - 1.0 >= 0.095:
                        day_skipped += 1
                        continue
            picked.append(code)
        if day_skipped > 0:
            skip_days += 1
            skip_total += day_skipped

        day_rets = []
        for rank_i, code in enumerate(picked):
            score = scores[code]
            buy_date = _next_td(date, 1)
            sell_date = _next_td(buy_date, label_horizon_days) if buy_date else None

            buy_price  = close_map.get((buy_date, code)) if buy_date else None
            sell_price = close_map.get((sell_date, code)) if sell_date else None

            if buy_price is not None and sell_price is not None and buy_price > 0:
                ret_val = sell_price / buy_price - 1.0
                ret_src  = 'OK'
            else:
                ret_val = None
                ret_src  = 'pending'

            day_rets.append(ret_val)

            name = code_to_name.get(code, '?')
            records.append({
                'factor_dt': date,
                'buy_dt':    buy_date or '-',
                'code':      code,
                'name':      name,
                'score':     score,
                'buy_prc':   buy_price,
                'sell_dt':   sell_date or '-',
                'sell_prc':  sell_price,
                'ret_pct':   ret_val * 100.0 if ret_val is not None else None,
                'status':    ret_src,
            })

        # Strategy daily return = mean of picked stocks' returns
        valid = [r for r in day_rets if r is not None]
        daily_returns[date] = float(np.mean(valid)) if valid else None

    # Build return series
    ret_series = pd.Series(daily_returns, name='daily_return').sort_index()
    ret_valid = ret_series.dropna()

    # Build trade_df
    disp_cols = {
        'factor_dt': 'FactorDt', 'buy_dt': 'BuyDt', 'code': 'Code',
        'name': 'Name', 'score': 'Score', 'buy_prc': 'BuyPrc',
        'sell_dt': 'SellDt', 'sell_prc': 'SellPrc', 'ret_pct': 'Ret%',
    }
    trade_df = pd.DataFrame(records).rename(columns=disp_cols)

    # Cumulative additive return
    cum = 0.0
    cum_list = []
    for _, row in trade_df.iterrows():
        val = row['Ret%']
        if pd.notna(val):
            cum += val
        cum_list.append(cum)
    trade_df['累加收益'] = cum_list
    trade_df = trade_df[list(disp_cols.values()) + ['累加收益']]

    return trade_df, ret_valid, skip_total, skip_days


def plot_backtest_curve(ret_valid, start_d, end_d, top_n=1, label_horizon_days=1,
                         predicted_dates=None):
    """Plot cumulative return + daily return bar chart."""
    dates_plot = ret_valid.index.tolist()
    vals_plot = ret_valid.values

    if len(dates_plot) == 0:
        print("\nWARNING: no return data available, skipping plot.")
        return

    step = max(1, len(dates_plot) // 10)
    tick_idx = list(range(0, len(dates_plot), step))
    tick_lbl = [dates_plot[i] for i in tick_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: Cumulative return
    cum = np.cumsum(vals_plot)
    axes[0].plot(dates_plot, cum, color='#1f77b4', linewidth=1.8, marker='o', markersize=3)
    axes[0].fill_between(range(len(dates_plot)), 0, cum, alpha=0.10, color='#1f77b4')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    if predicted_dates:
        p0 = predicted_dates[0]
        if p0 in dates_plot:
            axes[0].axvline(x=p0, color='red', linestyle='--', alpha=0.5, linewidth=1.2)
    axes[0].set_title(f'Top {top_n} Cumulative Return | {start_d} ~ {end_d}', fontsize=13)
    axes[0].set_ylabel('Cumulative Return', fontsize=11)
    axes[0].set_xticks(tick_idx)
    axes[0].set_xticklabels(tick_lbl, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)

    # Right: Daily return
    bar_colors = ['#d62728' if v < 0 else '#2ca02c' for v in vals_plot]
    axes[1].bar(range(len(dates_plot)), vals_plot, color=bar_colors, alpha=0.80, width=0.65)
    axes[1].axhline(y=0, color='black', linewidth=0.8)
    axes[1].set_title(f'Daily Return ({label_horizon_days}d holding)', fontsize=13)
    axes[1].set_ylabel('Daily Return', fontsize=11)
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels(tick_lbl, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_trade_log_table(trade_df, ret_valid, recent_dates, top_n=1, label_horizon_days=1):
    """Print CJK-aligned trade log table with cumulative return."""
    print(f"\n{'='*100}")
    print(f"  Trade Log — Top {top_n} | {len(recent_dates)} days | "
          f"hold {label_horizon_days}d | Returns from daily_adj close prices")
    print(f"{'='*100}")

    # Format
    fmt_df = trade_df.copy()
    fmt_df['Score']   = fmt_df['Score'].apply(lambda x: f'{x:+.2f}' if pd.notna(x) else '-')
    fmt_df['BuyPrc']  = fmt_df['BuyPrc'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-')
    fmt_df['SellPrc'] = fmt_df['SellPrc'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-')
    fmt_df['Ret%']    = fmt_df['Ret%'].apply(lambda x: f'{x:+.2f}%' if pd.notna(x) else '-')
    fmt_df['累加收益'] = fmt_df['累加收益'].apply(
        lambda x: f'{x:+.2f}%' if abs(x) >= 0.01 else f'{x:+.4f}%')

    # Column widths & alignment
    _cols = fmt_df.columns.tolist()
    _col_align = {c: ('<' if c == 'Name' else '>') for c in _cols}
    _col_w = [max(_cjk_width(c), max((_cjk_width(str(v)) for v in fmt_df[c]), default=0)) + 2
              for c in _cols]

    # Header (centered), separator, rows (numbers right, names left)
    print(''.join(_pad_cjk(c, _col_w[i], '^') for i, c in enumerate(_cols)))
    print(''.join('-' * _col_w[i] for i in range(len(_cols))))
    for _, row in fmt_df.iterrows():
        print(''.join(_pad_cjk(str(row[c]), _col_w[i], _col_align[c]) for i, c in enumerate(_cols)))

    print(f"{'='*100}\n")

    # Quick stats
    n_ok   = len(ret_valid)
    n_pend = len(recent_dates) - n_ok
    print(f"Realized: {n_ok}  |  Pending: {n_pend}")

    if len(ret_valid) > 0:
        cum_r = ret_valid.sum()
        win_r = int((ret_valid > 0).sum())
        print(f"Cumulative: {cum_r:+.4f} ({cum_r*100:+.2f}%)  |  "
              f"Win rate: {win_r}/{len(ret_valid)} ({win_r/len(ret_valid)*100:.1f}%)  |  "
              f"Mean: {ret_valid.mean():+.6f}  |  "
              f"MaxDD: {(np.cumsum(ret_valid.values) - np.maximum.accumulate(np.cumsum(ret_valid.values))).min():.4f}")
    print()


def print_backtest_summary(ret_valid, recent_dates, start_d, end_d, n_lookback,
                            top_n=1, label_horizon_days=1,
                            skip_total=0, skip_days=0, exclude_limit_up=False):
    """Print backtest summary statistics."""
    total_days = len(recent_dates)
    valid_days = len(ret_valid)
    pending_days = total_days - valid_days

    if valid_days > 0:
        vals = ret_valid.values
        win_days = int((vals > 0).sum())
        cum_ret = float(vals.sum())
        cummax_track = np.maximum.accumulate(np.cumsum(vals))
        drawdown = np.cumsum(vals) - cummax_track
        max_dd = float(drawdown.min())
        max_dd_idx = int(drawdown.argmin())
        best_idx = int(vals.argmax())
        worst_idx = int(vals.argmin())
        ann_ret = float(vals.mean() * 252 * 100)
        ann_sharpe = float(vals.mean() / vals.std() * np.sqrt(252)) if vals.std() > 0 else 0.0
        win_rate = win_days / valid_days * 100
    else:
        win_days = cum_ret = max_dd = ann_ret = ann_sharpe = win_rate = 0.0
        best_idx = worst_idx = max_dd_idx = 0

    print(f"{'='*55}")
    print(f"  Recent {n_lookback}d Summary  |  Top {top_n}  |  Hold {label_horizon_days}d")
    print(f"{'='*55}")
    print(f"  Window     : {start_d} ~ {end_d}  ({total_days} days)")
    print(f"  Realized   : {valid_days} days  |  Pending : {pending_days} days")
    if valid_days > 0:
        print(f"  Cum Return : {cum_ret:+.4f}  ({cum_ret*100:+.2f}%)")
        print(f"  Daily Mean : {vals.mean():+.6f}  ({vals.mean()*100:+.4f}%)")
        print(f"  Ann Return : {ann_ret:.2f}%")
        print(f"  Sharpe     : {ann_sharpe:.4f}")
        print(f"  Win Rate   : {win_rate:.1f}%  ({win_days}/{valid_days})")
        print(f"  Best Day   : {vals[best_idx]:+.4f}  ({ret_valid.index[best_idx]})")
        print(f"  Worst Day  : {vals[worst_idx]:+.4f}  ({ret_valid.index[worst_idx]})")
        if max_dd < 0:
            print(f"  Max Drawdown: {max_dd:.4f}  ({max_dd*100:.2f}%)  at {ret_valid.index[max_dd_idx]}")
        else:
            print(f"  Max Drawdown: 0 (no drawdown)")
    print(f"{'='*55}\n")

    # Pending-dates detail
    if pending_days > 0:
        pending_list = [d for d in recent_dates if d not in ret_valid.index]
        print(f"[Pending] {pending_days} date(s) — {label_horizon_days}d return not yet realized:")
        for d in pending_list:
            buy_d = _next_td(d, 1)
            sell_d = _next_td(buy_d, label_horizon_days) if buy_d else None
            print(f"  {d}: buy {buy_d or '?'}  ->  sell {sell_d or '?'}")
        print()

    # Limit-up filter stats
    if exclude_limit_up and skip_total > 0:
        print(f"[Limit-Up Filter] {skip_total} stocks filtered across "
              f"{skip_days}/{total_days} days ({skip_days/total_days*100:.1f}%)")

# ======================================================================
# Cell 3 [markdown]
# ======================================================================
# # 使用示例

# ======================================================================
# Cell 4 [code]
# ======================================================================
# bench
bench1 = load_bench(bench1_path)
bench2 = load_bench(bench2_path)
if bench2 == None:
    bench2 = bench1.copy()  
bench3 = load_bench(bench3_path)
if bench3 == None:
    bench3 = bench2.copy()
bench4 = load_bench(bench4_path)
if bench4 == None:
    bench4 = bench3.copy()

bench_all = ensemble_scores(bench1, bench2, bench3, bench4)
bench_all = bench_all.mean(axis=1).unstack()

# 收集模型训练结果
model_score = concat_model_4fold(test_path=model_test_path, res_path=model_res_path)
print("---单一模型评估---")
# 回测使用实际 1 日收益（单日换手），非训练 label
print(f"  持仓数: {PERSONAL_TOP_N} 只")
ret1, ic1 = get_ret_ic(model_score, params.ret_1d_data, start=start, end=end,
                       top_n=PERSONAL_TOP_N)
print_quarterly_metrics(ret1, ic1)
score1_metrics = get_metrics(ret1, ic1)
print_metrics(score1_metrics)
plot_model(model_score, bench_all, params.ret_1d_data, start=start, end=end,
           top_n=PERSONAL_TOP_N)
