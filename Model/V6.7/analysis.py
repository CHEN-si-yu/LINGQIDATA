#!/usr/bin/env python3
"""
analysis.py — V6.1 多周期联合训练模型推演脚本

配套 V6.1 model.py 改造（学 V5.2：多头 1d主+5d/10d辅助 联合训练、
最新 N 日验证、无测试集、checkpoint 按 val_icmean 选择）：
  - 直接从 model_train/{SEASON} 加载各 Fold 最佳 checkpoint，
    对 start 起至最新因子日期的所有交易日推演打分（4-Fold zscore 集成）
  - 推演/回测统一使用 pred_1d 头；回测 P&L 用正确的 1d 实际价格收益
    (open[t+2]/open[t+1]-1, 来自 daily_adj)，与训练 target 解耦

输出格式与 V5.2 model_pic/output.md 对齐：
  统一推演 → 合并/保存 → 一致性验证 → 最新模型打分 Top 推荐 → Recent 回测

Usage:
  cd /root/autodl-fs/lingqiData/Model/V6.1 && python analysis.py

预测结果落盘:
  ./model_pred/{SEASON}/{date}.pkl          每个交易日一只 pickle（index=Code）
  ./model_pred/{SEASON}/all_zscore_score.fea  全部推演日期的汇总（index=date）
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
import re, os, sys, glob, warnings, gc

import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.feather as pf, pyarrow.compute as pc

import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ======================================================================
# Cell 1 [code]
# ======================================================================
# 从 model.py 导入共享配置（路径、因子名、标签名、预加载数据）
from model import (PROJECT_ROOT, root_path, fac_path, fac_name, params,
                   PredictModel)

# ── 推演配置 ──
start = '20260701'          # 推演起始日期（含），终点自动取因子数据最新日期
SEASON = '2026q3'           # checkpoint 目录: model_train/{SEASON}/foldN
TOP_N = 1                   # 回测买入股票数量（trade log 展示用）
EXCLUDE_LIMIT_UP = False    # 是否排除涨停板（factor日涨幅>=9.5%，历史保留选项，默认关闭）
# 回测持仓天数 (策略口径): 1 = 单日换手。V6.1 学 V5.2 —— 训练目标是多周期(1d主+5d/10d辅助),
# 但回测 P&L 永远用正确的 1d 收益 (open[t+2]/open[t+1]-1) 计算, 与训练 target 解耦
LABEL_HORIZON_DAYS = 1

# V6.1: 执行可行性过滤（默认开启）
EXCLUDE_ST = True           # 排除 ST/退市整理股（涨跌幅限制不同、退市风险、机构禁买）
BUY_GAP_LIMIT = 0.095       # 买入日开盘相对昨收跳空 >= 9.5% 视为一字板不可成交, 跳过
MIN_AMOUNT = 5e7            # 买入日成交额下限（元, 5000万以下流动性不足）

# V6.1: 分散组合统计 —— 在 Top1 trade log 之外, 汇总展示多档 Top-N 与 Top 分数带等权组合
MULTI_TOPN = [5, 10, 20]    # 额外的 Top-N 等权档位
BAND_FRACS = [0.01, 0.05, 0.10]  # Top 分数带等权: 前 1% / 5% / 10%

fac_full_path = fac_path + fac_name + '.fea'      # 训练因子数据 fac_sample.fea
model_train_base = rf'{root_path}/model_train'
model_test_path = rf'{root_path}/model_test'      # 仅读取 feature_map.fea
model_pred_path = rf'{root_path}/model_pred/{SEASON}'
all_feather_path = rf'{model_pred_path}/all_zscore_score.fea'
calendar_path = PROJECT_ROOT + 'data/calendar.parquet'


# ======================================================================
# Cell 2 [code]
# ============================================================
# 所有函数定义（引擎函数 + 展示/回测函数）
# ============================================================

# V6.1: checkpoint 命名改为 {epoch}-{val_icmean:.4f}; 兼容旧 {epoch}-{val_wei:.4f} 命名
_VAL_IC_RE = re.compile(r'val_icmean=(-?\d+\.\d+)')
_VAL_WEI_RE = re.compile(r'val_wei=(-?\d+\.\d+)')
_VAL_RANKIC_AVG_RE = re.compile(r'val_rankic_avg=(-?\d+\.\d+)')

def load_feature_map(model_test_path):
    """读取训练时落盘的 feature_map.fea（因子顺序与 checkpoint 一致）。"""
    feat_path = os.path.join(model_test_path, 'feature_map.fea')
    if not os.path.exists(feat_path): return None
    with open(feat_path, 'r') as f:
        raw = f.read()
    raw = raw.replace('\\n', '\n')
    factor_order = []
    for line in raw.split('\n'):
        line = line.strip()
        if '=' in line:
            name, idx = line.rsplit('=', 1)
            factor_order.append((int(idx.strip()), name.strip()))
    factor_order.sort(key=lambda x: x[0])
    return [name for _, name in factor_order]


def normed_data(data, factor_list):
    """与训练一致的推理标准化: 有效性过滤 → 0.5%/99.5% 截尾 → zscore → 填 0。"""
    valid_threshold = max(1, int(0.1 * len(factor_list)))
    valid_mask = data[factor_list].notna().sum(axis=1) >= valid_threshold
    data = data.loc[valid_mask].copy()
    code_value = data['Code'].values
    data_X = data[factor_list]
    quantiles = data_X.quantile([0.005, 0.995])
    data_X = data_X.clip(lower=quantiles.loc[0.005], upper=quantiles.loc[0.995], axis=1)
    data_X = (data_X - data_X.mean()) / data_X.std()
    data_X = data_X.fillna(0)
    data_x_np = np.nan_to_num(data_X.to_numpy(dtype=np.float32, copy=False),
                              nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(data_x_np), code_value


def load_model(checkpoint_path, input_dim, **kwargs):
    """Load a trained checkpoint (strip lightning 'model.' prefix)。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    model_state = {k.removeprefix('model.'): v for k, v in state_dict.items()
                   if k.startswith('model.')}
    model = PredictModel(input_dim=input_dim, **kwargs)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing: print(f'[load_model] WARNING: missing keys: {list(missing)[:10]}...')
    if unexpected:
        print(f"[load_model] WARNING: unexpected keys (ignored): {unexpected}")
    model.eval()
    return model


def find_best_checkpoint(fold_dir):
    """在 fold 目录下找 val_icmean 最高的 checkpoint (V6.1 新命名), 兼容旧 val_wei 命名。"""
    ckpt_files = glob.glob(os.path.join(fold_dir, '**', '*.ckpt'), recursive=True)
    if not ckpt_files: raise FileNotFoundError(f'No ckpt under {fold_dir}')
    best_ckpt, best_val = None, -float('inf')
    for ckpt in ckpt_files:
        m = _VAL_RANKIC_AVG_RE.search(os.path.basename(ckpt))  # V6.3+: val_rankic_avg
        if m is None:
            m = _VAL_IC_RE.search(os.path.basename(ckpt))
        if m is None:
            m = _VAL_WEI_RE.search(os.path.basename(ckpt))  # 旧命名回退
        if m is None: continue
        val = float(m.group(1))
        if val > best_val: best_val = val; best_ckpt = ckpt
    if best_ckpt is None: raise RuntimeError(f'No val_icmean/val_wei parsed under {fold_dir}')
    return best_ckpt, best_val


def discover_folds(model_dir):
    folds = sorted(glob.glob(os.path.join(model_dir, 'fold[0-9]*')))
    if not folds: raise FileNotFoundError(f'No folds under {model_dir}')
    return folds


def predict_date(model, date, all_data, factor_list, device):
    data = all_data.loc[date].copy()
    data_X, code_value = normed_data(data, factor_list)
    data_X = data_X.to(device)
    with torch.no_grad():
        output = model(data_X)
        # V6.1 (学 V5.2): 多头模型返回 (pred_1d, pred_5d, pred_10d), 回测统一用 pred_1d
        score = output[0] if isinstance(output, tuple) else output
    result = pd.DataFrame(score.detach().cpu().numpy(), index=code_value, columns=['value'])
    result.index.name = 'Code'
    return result


def save_predictions(score_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for date in score_df.index:
        s = score_df.loc[date].copy(); s.index.name = 'Code'
        s.to_pickle(os.path.join(output_dir, f'{date}.pkl'))


def _load_trading_dates(calendar_path):
    cal = pd.read_parquet(calendar_path)
    return set(cal[cal['is_open'] == 1]['date'].astype(str).str.replace('-', '').tolist())


# ── 展示/回测辅助函数（沿用 V5.2 版本） ───────────────────────

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
    DAILY_ADJ_PATH = PROJECT_ROOT + "data/daily_adj.parquet"
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
    stock_info = pd.read_parquet(PROJECT_ROOT + 'data/stock_list.parquet')
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
    open_map       = df.set_index(['trade_date', 'code_clean'])['open'].to_dict()
    prev_close_map = df.set_index(['trade_date', 'code_clean'])['prev_close'].to_dict()
    code_to_name   = _load_name_map()

    # --- 成交额 (元): trade_amt.fea, index=date, columns=Code ---
    amt_df = pd.read_feather(PROJECT_ROOT + 'trainingdata/trade_amt.fea').set_index('index')
    amount_map = {}
    for d, row in amt_df.iterrows():
        for c, v in row.items():
            if pd.notna(v):
                amount_map[(str(d), str(c))] = v
    del amt_df

    return close_map, open_map, prev_close_map, code_to_name, amount_map


def build_trade_log_df(model_score_extended, recent_dates, close_map, open_map,
                        prev_close_map, code_to_name, amount_map=None,
                        top_n=1, exclude_limit_up=False, label_horizon_days=1,
                        exclude_st=False, buy_gap_limit=None, min_amount=None):
    """Build trade log records and daily return series from model scores & price data.

    V6.1: 新增执行可行性过滤 (默认在调用处开启):
      exclude_st     : 排除名称含 ST/退 的股票
      buy_gap_limit  : 买入日开盘相对昨收跳空 >= 该值(如 0.095)视为一字板不可成交, 跳过
      min_amount     : 买入日成交额下限(元), 低于则流动性不足跳过

    Returns:
        trade_df   : DataFrame with columns [FactorDt, BuyDt, Code, Name, Score,
                     BuyPrc, SellDt, SellPrc, Ret%, 累加收益]
        ret_valid  : Series of realized daily returns
        skip_total : int, number of stocks skipped by filters
        skip_days  : int, number of days with at least one skip
    """
    records = []
    daily_returns = {}
    skip_total = 0
    skip_days = 0
    skip_reason = {}

    for date in recent_dates:
        scores = model_score_extended.loc[date].sort_values(ascending=False)
        buy_date = _next_td(date, 1)

        # Stock selection: top N by model score（依次跳过不可执行的候选）
        picked = []
        day_skipped = 0
        for code in scores.index:
            if len(picked) >= top_n:
                break
            reason = None
            name = code_to_name.get(code, '')
            if exclude_st and ('ST' in name or '退' in name):
                reason = 'ST'
            if exclude_limit_up and reason is None:
                fc = close_map.get((date, code))
                pc = prev_close_map.get((date, code))
                if fc is not None and pc is not None and pc > 0 and fc / pc - 1.0 >= 0.095:
                    reason = 'factor日涨停'
            if buy_gap_limit is not None and reason is None and buy_date:
                bo = open_map.get((buy_date, code))
                pc = close_map.get((date, code))
                if bo is not None and pc is not None and pc > 0 and bo / pc - 1.0 >= buy_gap_limit:
                    reason = '一字板不可买'
            if min_amount is not None and reason is None and buy_date:
                amt = amount_map.get((buy_date, code)) if amount_map else None
                if amt is not None and amt < min_amount:
                    reason = '流动性不足'
            if reason is not None:
                day_skipped += 1
                skip_reason[reason] = skip_reason.get(reason, 0) + 1
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

            buy_price  = open_map.get((buy_date, code)) if buy_date else None
            sell_price = open_map.get((sell_date, code)) if sell_date else None

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

    return trade_df, ret_valid, skip_total, skip_days, skip_reason


def backtest_diversified(model_score_extended, recent_dates, close_map, open_map,
                         prev_close_map, code_to_name, amount_map=None,
                         multi_topn=(5, 10, 20), band_fracs=(0.01, 0.05, 0.10),
                         label_horizon_days=1, exclude_st=False,
                         buy_gap_limit=None, min_amount=None):
    """V6.1: 分散组合统计。

    Top1 单票是彩票 —— 单日 rank IC≈0.02 的信号下, 单票日盈亏由尾部事件(涨跌停)主导。
    这里统计多档 Top-N 等权与前 1%/5%/10% 分数带等权组合的日收益, 供横向对比。
    过滤规则与 trade log 一致。返回 DataFrame: 每行一个构造。
    """
    def pick_codes(date):
        scores = model_score_extended.loc[date].sort_values(ascending=False)
        buy_date = _next_td(date, 1)
        out = []
        for code in scores.index:
            name = code_to_name.get(code, '')
            if exclude_st and ('ST' in name or '退' in name):
                continue
            if buy_gap_limit is not None and buy_date:
                bo = open_map.get((buy_date, code))
                pc = close_map.get((date, code))
                if bo is not None and pc is not None and pc > 0 and bo / pc - 1.0 >= buy_gap_limit:
                    continue
            if min_amount is not None and buy_date:
                amt = amount_map.get((buy_date, code)) if amount_map else None
                if amt is not None and amt < min_amount:
                    continue
            out.append(code)
        return out

    def realized_ret(date, code):
        buy_date = _next_td(date, 1)
        sell_date = _next_td(buy_date, label_horizon_days) if buy_date else None
        bp = open_map.get((buy_date, code)) if buy_date else None
        sp = open_map.get((sell_date, code)) if sell_date else None
        if bp is not None and sp is not None and bp > 0:
            return sp / bp - 1.0
        return None

    series = {}  # 构造名 -> [(date, ret), ...]
    for date in recent_dates:
        codes = pick_codes(date)
        n = len(codes)
        if n == 0:
            continue
        for k in multi_topn:
            if k > n:
                continue
            rs = [r for r in (realized_ret(date, c) for c in codes[:k]) if r is not None]
            if rs:
                series.setdefault(f'Top{k}', []).append((date, np.mean(rs)))
        for frac in band_fracs:
            k = max(1, int(n * frac))
            rs = [r for r in (realized_ret(date, c) for c in codes[:k]) if r is not None]
            if rs:
                series.setdefault(f'前{frac:.0%}', []).append((date, np.mean(rs)))

    rows = []
    for name, items in series.items():
        rets = np.array([r for _, r in items])
        rows.append({
            '构造': name,
            '天数': len(rets),
            '日均': rets.mean() * 100,
            '累计': rets.sum() * 100,
            '胜率': (rets > 0).mean() * 100,
            '年化Sharpe': rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def build_daily_pnl_series(model_score_extended, recent_dates, open_map, close_map,
                           prev_close_map, code_to_name, amount_map=None,
                           top_n=1, label_horizon_days=1, exclude_st=False,
                           buy_gap_limit=None, min_amount=None):
    """V6.1: 重叠持仓策略的真实日度组合 P&L。

    多日持有 (label_horizon_days>1) 时, 每个交易日组合同时持有 H 个篮子,
    组合当日收益 = 所有在持仓位的 1 日 open-open 收益均值 (即用正确的 1d 收益口径,
    与账户净值一致)。逐笔 H 日收益直接相加会重复计算重叠区间, 是错误的。

    hold=1 时退化为每日 TopN 篮子收益, 与 build_trade_log_df 的 ret_valid 完全一致。
    返回 (daily_series, pending_factor_dates)。
    """
    idx_map = {d: i for i, d in enumerate(_td_list)}
    baskets = {}
    for date in recent_dates:
        scores = model_score_extended.loc[date].sort_values(ascending=False)
        buy_date = _next_td(date, 1)
        picked = []
        for code in scores.index:
            if len(picked) >= top_n:
                break
            name = code_to_name.get(code, '')
            if exclude_st and ('ST' in name or '退' in name):
                continue
            if buy_gap_limit is not None and buy_date:
                bo = open_map.get((buy_date, code))
                pc = close_map.get((date, code))
                if bo is not None and pc is not None and pc > 0 and bo / pc - 1.0 >= buy_gap_limit:
                    continue
            if min_amount is not None and buy_date:
                amt = amount_map.get((buy_date, code)) if amount_map else None
                if amt is not None and amt < min_amount:
                    continue
            picked.append(code)
        baskets[date] = picked

    first_buy = _next_td(recent_dates[0], 1)
    if first_buy is None:
        return pd.Series(dtype=float), list(recent_dates)

    out = {}
    for t in _td_list:
        if idx_map[t] < idx_map[first_buy]:
            continue
        t1 = _next_td(t, 1)
        if t1 is None:
            break
        # 区间 (t, t+1) 在持的篮子: factor date d 满足 d+1 <= t <= d+H
        basket_rets = []
        for d, codes in baskets.items():
            b = _next_td(d, 1)
            last_interval_start = _next_td(d, label_horizon_days)
            if b is None or last_interval_start is None:
                continue
            if idx_map[b] <= idx_map[t] <= idx_map[last_interval_start]:
                rs = []
                for c in codes:
                    o1, o2 = open_map.get((t, c)), open_map.get((t1, c))
                    if o1 and o2 and o1 > 0:
                        rs.append(o2 / o1 - 1.0)
                if rs:
                    basket_rets.append(np.mean(rs))
        if not basket_rets:
            break
        out[t] = float(np.mean(basket_rets))

    daily = pd.Series(out, name='daily_return').sort_index()
    # 尚未完全平仓的 factor date: 其最后一个持仓区间起点晚于最后一个已结算区间
    last_t = daily.index[-1] if len(daily) else None
    pending = []
    if last_t is not None:
        for d in recent_dates:
            last_start = _next_td(d, label_horizon_days)
            if last_start is None or idx_map[last_start] > idx_map[last_t]:
                pending.append(d)
    return daily, pending


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
          f"hold {label_horizon_days}d | Returns from daily_adj open prices")
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
                            skip_total=0, skip_days=0, exclude_limit_up=False,
                            pending_dates=None):
    """Print backtest summary statistics.

    V6.1: hold>1 时 ret_valid 为重叠持仓的日度组合序列 (见 build_daily_pnl_series),
    pending 由 pending_dates 显式给出 (默认按 ret_valid 缺失日期推断, 适用于 hold=1)。
    """
    total_days = len(recent_dates)
    valid_days = len(ret_valid)
    pending_days = len(pending_dates) if pending_dates is not None else (total_days - valid_days)

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
        if pending_dates is None:
            pending_dates = [d for d in recent_dates if d not in ret_valid.index]
        print(f"[Pending] {len(pending_dates)} date(s) — {label_horizon_days}d return not yet realized:")
        for d in pending_dates:
            buy_d = _next_td(d, 1)
            sell_d = _next_td(buy_d, label_horizon_days) if buy_d else None
            print(f"  {d}: buy {buy_d or '?'}  ->  sell {sell_d or '?'}")
        print()

    # Limit-up filter stats
    if exclude_limit_up and skip_total > 0:
        print(f"[Limit-Up Filter] {skip_total} stocks filtered across "
              f"{skip_days}/{total_days} days ({skip_days/total_days*100:.1f}%)")


# ======================================================================
# Cell 3 [code]
# ============================================================
# 1. 统一推演：加载 4 个 Fold 最佳 checkpoint，推演 start 起所有交易日
# ============================================================

# ── Step 0: 上次推演结果（用于增量合并与一致性验证） ──
model_score = None
if os.path.exists(all_feather_path):
    model_score = pd.read_feather(all_feather_path).set_index('date')

# ── Step 1: 读取训练因子顺序（feature_map.fea，与 checkpoint 对齐） ──
train_factors = load_feature_map(model_test_path)
if train_factors is None:
    raise FileNotFoundError(f'feature_map.fea not found under {model_test_path}')

# ── Step 2: 因子数据列（fac_sample 中实际存在的 feature_map 因子） ──
all_cols = pf.read_table(fac_full_path, columns=[]).column_names
available = [f for f in train_factors if f in all_cols]
missing = [f for f in train_factors if f not in all_cols]
if missing:
    print(f"[predict] WARNING: {len(missing)} training factors missing from DB, will fill with 0")
cols_to_load = ['date', 'Code'] + available

# ── Step 3: 待推演日期 = start 起的全部交易日 ──
all_factor_dates = sorted(pd.Series(
    pf.read_table(fac_full_path, columns=['date']).column('date').to_pandas()).unique())
trading_dates = _load_trading_dates(calendar_path)
dates = [d for d in all_factor_dates if d >= start and d in trading_dates]

print(f"\n>>> 统一推演 {len(dates)} 个日期: {dates[0]} ~ {dates[-1]}")

table = pf.read_table(fac_full_path, columns=cols_to_load)
mask = pc.is_in(table.column('date'), pa.array(dates))
table = table.filter(mask)
all_data = table.to_pandas(); del table
all_data = all_data.set_index('date', drop=True).sort_index()

# 缺失训练因子补 0
for f in missing:
    all_data[f] = 0.0

# ── Step 4: 核对 checkpoint 期望输入维度 ──
model_dir = os.path.join(model_train_base, SEASON)
folds = discover_folds(model_dir)
first_ckpt_path, _ = find_best_checkpoint(folds[0])
ckpt_meta = torch.load(first_ckpt_path, map_location='cpu', weights_only=False)
ckpt_input_dim = None
for key, tensor in ckpt_meta['state_dict'].items():
    if 'weight' in key and len(tensor.shape) == 2:
        ckpt_input_dim = tensor.shape[1]  # [out_features, in_features]
        break
del ckpt_meta

factor_list = train_factors[:]
if ckpt_input_dim is not None and ckpt_input_dim != len(factor_list):
    shortage = ckpt_input_dim - len(factor_list)
    if shortage > 0:
        for i in range(shortage):
            placeholder = f'_padding_{i}'
            factor_list.append(placeholder)
            all_data[placeholder] = 0.0
        print(f"[predict] Padded {shortage} zero placeholder(s) to reach {ckpt_input_dim}")
    else:
        factor_list = factor_list[:ckpt_input_dim]
        print(f"[predict] Trimmed factor_list to {ckpt_input_dim}")
print(f"[predict] feature_map has {len(train_factors)} factors, "
      f"{len(available)} available in DB, {len(missing)} missing")
print(f"[predict] Final factor_list: {len(factor_list)} features → model input_dim={len(factor_list)}")

# ── Step 5: 每个 Fold 加载各自最佳 checkpoint ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[predict] Using {len(folds)} folds")
models = []
for fold_dir in folds:
    fn = os.path.basename(fold_dir)
    ckpt, val = find_best_checkpoint(fold_dir)
    print(f"[predict]   [{fn}] {os.path.basename(ckpt)} (val_wei={val:.4f})")
    m = load_model(ckpt, len(factor_list))
    m.to(device)
    models.append(m)

# ── Step 6: 推演: 4 个模型分别预测, 逐日 zscore 后求和集成 ──
new_scores = []
for date in dates:
    if date not in all_data.index:
        continue
    fold_preds = [predict_date(m, date, all_data, factor_list, device) for m in models]
    zscored = [(f - f.mean()) / f.std() for f in fold_preds]
    ensemble = sum(zscored)
    date_score = ensemble['value']; date_score.name = date
    new_scores.append(date_score)
    print(f"[predict]   [{date}] done, {len(ensemble)} stocks")
new_score_df = pd.DataFrame(new_scores); new_score_df.index.name = 'date'
print(f"[predict] Done, {len(new_score_df)} days.")

# ======================================================================
# Cell 4 [code]
# ============================================================
# 2. 合并上次推演结果 + 落盘 + 一致性验证
# ============================================================

if model_score is not None and len(model_score) > 0:
    new_mask = ~new_score_df.index.isin(model_score.index)
    new_scores = new_score_df.loc[new_mask] if new_mask.any() else None
else:
    new_scores = new_score_df.copy()

if new_scores is not None and len(new_scores) > 0:
    model_score_extended = (pd.concat([model_score, new_scores], axis=0).sort_index()
                            if model_score is not None else new_scores.copy())
    print(f"\n合并完成：原始 {0 if model_score is None else len(model_score)} 天 + "
          f"新增 {len(new_scores)} 天 = {len(model_score_extended)} 天")
    print(f"新增日期: {list(new_scores.index)}")

    # 落盘: 新增日期逐日 pkl + 完整汇总 feather
    save_predictions(new_scores, model_pred_path)
    model_score_extended.reset_index().to_feather(all_feather_path)
    print(f"已保存到: {model_pred_path}")
else:
    model_score_extended = model_score.copy()
    print("无新增日期，使用上次推演结果。")

# ---- 与上次推演一致性验证（重叠的最后 3 天） ----
if model_score is not None and len(model_score) > 0:
    hist_dates = sorted(model_score.index)[-3:]
    hist_dates = [d for d in hist_dates if d in new_score_df.index]
    if hist_dates:
        print("\n--- 与上次推演一致性验证 ---")
        for date in hist_dates:
            old = model_score.loc[date]
            new = new_score_df.loc[date]
            common = old.index.intersection(new.index)
            if len(common) > 0:
                corr = old[common].corr(new[common])
                print(f"  {date}: corr={corr:.6f}, 共同股票数={len(common)}")
            else:
                print(f"  {date}: 无共同股票")

# ======================================================================
# Cell 5 [code]
# ============================================================
# 3. 最新推荐展示：股票名称映射、Top 打分、投资建议
# ============================================================

# --- 股票代码 → 名称映射 ---
code_to_name = _load_name_map()

# --- 交易日历 ---
cal = pd.read_parquet(calendar_path)
cal = cal[cal['is_open'] == 1]
_trade_dates = sorted(cal['date'].astype(str).str.replace('-', '').tolist())

# 原始（上次）结果的最新日期
last_label_date = model_score.index.max() if model_score is not None else None

# 本次推演的最新因子日期（即这一批数据的"当前日期"）
latest_factor_date = model_score_extended.index.max()

# 本次新增的预测日期（上次结果未覆盖的日期）
predicted_dates = sorted(set(model_score_extended.index) -
                         (set(model_score.index) if model_score is not None else set()))
predicted_dates.sort()

# ============================================================
# 展示预测日期的 Top 推荐
# ============================================================
print()
print("=" * 72)
print("                    最新模型打分 Top 推荐")
print("=" * 72)

# 无新增日期时（同日重跑）回退展示最新 10 天，保证每次运行都有推荐输出
block_dates = predicted_dates[-10:] if predicted_dates else sorted(model_score_extended.index)[-10:]

for date in block_dates:
    scores = model_score_extended.loc[date].sort_values(ascending=False)
    buy_date = get_nth_next_trade_date(date, 1)
    sell_date = get_nth_next_trade_date(buy_date, LABEL_HORIZON_DAYS) if buy_date else None

    print(f"\n{'─' * 60}")
    print(f"  当前日期：{fmt_date(latest_factor_date)} 收盘")
    if buy_date and sell_date:
        print(f"  模型预测{LABEL_HORIZON_DAYS}日收益率：{fmt_date(buy_date)} 买入 → {fmt_date(sell_date)} 卖出")
    elif buy_date:
        print(f"  模型预测{LABEL_HORIZON_DAYS}日收益率：{fmt_date(buy_date)} 买入 → {fmt_date(sell_date)}")
    else:
        print(f"  模型预测{LABEL_HORIZON_DAYS}日收益率：无法推算交易日（需更新交易日历）")
    print(f"{'─' * 60}")

    top_n = min(10, len(scores))
    print(f"\n  {'排名':<5}{'代码':<10}{'名称':<20}打分")
    print(f"  {'-' * 37}")
    for rank, (code, score) in enumerate(scores.head(top_n).items(), 1):
        name = code_to_name.get(code, '未知')
        sign = '+' if score > 0 else ''
        print(f"  {rank:<5}{code:<10}{name:<12}{sign}{score:>9.4f}")

# ======================================================================
# Cell 6 [code]
# ============================================================
# Recent N-Day Backtest
# Dynamic tracking: return curve + daily trade log (symbol / price / return)
# ALL returns computed from actual price data (daily_adj), NOT from label
# ============================================================

WINDOW_START = start  # '20260701' —— 回测窗口起点(含当天)，终点为最新因子日期(至今)

print(f"--- Backtest: {WINDOW_START} ~ 至今 ---")

# ── Setup ──
close_map, open_map, prev_close_map, code_to_name, amount_map = load_backtest_setup()

# ── Select window: WINDOW_START(20260701) ~ 最新日期(至今) ──
all_dates = sorted(model_score_extended.index)
recent_dates = [d for d in all_dates if d >= WINDOW_START]
start_d, end_d = recent_dates[0], recent_dates[-1]
N_LOOKBACK = len(recent_dates)  # 实际交易日数，仅供 summary 标题展示

_prev_idx = set(model_score.index) if model_score is not None else set()
predicted_dates = [d for d in recent_dates if d not in _prev_idx]

print(f"Window : {start_d} ~ {end_d}  ({len(recent_dates)} trading days)")
if predicted_dates:
    tag = f"{predicted_dates[0]}~{predicted_dates[-1]}" if len(predicted_dates) > 1 else predicted_dates[0]
    print(f"  predicted: {len(predicted_dates)} days  ({tag})")

# ── Build trade log ──
trade_df, ret_valid, skip_total, skip_days, skip_reason = build_trade_log_df(
    model_score_extended, recent_dates, close_map, open_map, prev_close_map, code_to_name,
    amount_map=amount_map, top_n=TOP_N, exclude_limit_up=EXCLUDE_LIMIT_UP,
    label_horizon_days=LABEL_HORIZON_DAYS, exclude_st=EXCLUDE_ST,
    buy_gap_limit=BUY_GAP_LIMIT, min_amount=MIN_AMOUNT,
)

# ── V6.1: hold>1 时组合日度 P&L 必须按重叠持仓的 1d 实际收益计算 (逐笔相加会重复计) ──
ret_for_stats = ret_valid
pending_for_summary = None
if LABEL_HORIZON_DAYS > 1:
    ret_for_stats, pending_for_summary = build_daily_pnl_series(
        model_score_extended, recent_dates, open_map, close_map, prev_close_map,
        code_to_name, amount_map=amount_map, top_n=TOP_N,
        label_horizon_days=LABEL_HORIZON_DAYS, exclude_st=EXCLUDE_ST,
        buy_gap_limit=BUY_GAP_LIMIT, min_amount=MIN_AMOUNT,
    )
    # 账本"累加收益"列改为组合日度序列的累计值 (按各笔卖出日对齐)
    if len(ret_for_stats):
        cum_map = ret_for_stats.cumsum().to_dict()
        trade_df['累加收益'] = trade_df['SellDt'].map(lambda x: cum_map.get(x, np.nan))

# ── Plot return curve ──
plot_backtest_curve(ret_for_stats, start_d, end_d, top_n=TOP_N,
                    label_horizon_days=LABEL_HORIZON_DAYS,
                    predicted_dates=predicted_dates if predicted_dates else None)

# ── Trade log table ──
print_trade_log_table(trade_df, ret_for_stats, recent_dates,
                      top_n=TOP_N, label_horizon_days=LABEL_HORIZON_DAYS)

# ── Summary statistics ──
print_backtest_summary(ret_for_stats, recent_dates, start_d, end_d, N_LOOKBACK,
                       top_n=TOP_N, label_horizon_days=LABEL_HORIZON_DAYS,
                       skip_total=skip_total, skip_days=skip_days,
                       exclude_limit_up=EXCLUDE_LIMIT_UP,
                       pending_dates=pending_for_summary)

if skip_reason:
    print(f"[执行过滤] 跳过明细: {skip_reason}")

# ── V6.1: 分散组合对比 (Top1 是彩票, 多档组合才反映信号质量) ──
div_df = backtest_diversified(
    model_score_extended, recent_dates, close_map, open_map, prev_close_map,
    code_to_name, amount_map=amount_map, multi_topn=MULTI_TOPN, band_fracs=BAND_FRACS,
    label_horizon_days=LABEL_HORIZON_DAYS, exclude_st=EXCLUDE_ST,
    buy_gap_limit=BUY_GAP_LIMIT, min_amount=MIN_AMOUNT,
)
if len(div_df):
    print(f"{'='*55}")
    print(f"  分散组合对比  |  Hold {LABEL_HORIZON_DAYS}d  |  {start_d} ~ {end_d}")
    print(f"{'='*55}")
    print(div_df.to_string(index=False, formatters={
        '日均': lambda x: f'{x:+.3f}%',
        '累计': lambda x: f'{x:+.2f}%',
        '胜率': lambda x: f'{x:.1f}%',
        '年化Sharpe': lambda x: f'{x:+.2f}',
    }))
    print(f"{'='*55}\n")
    print("[提示] Top1 单票的日盈亏由涨跌停等尾部事件主导, 统计意义很弱; "
          "请以上表分散组合为准判断信号质量。")

# ======================================================================
# End of analysis.py — cleanup will run automatically via atexit
# ======================================================================
