#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — V20: 冠军系统冻结模型定义 + 核心引擎 (库文件, 供 run.py / train.sh / analysis.py 调用)

V20 是 V11(多目标族) + V13(顶部3d族) 的冻结集成封装, 本身不训练任何新参数。
本文件是冠军打分的唯一权威定义, 与 Model/V11/run_daily_pipeline.sh 组装口径完全一致:

    ens_w2 = z(mixA) + 2·z(V13组件)
    mixA    = Σ_{f=1..8} z( 1.0·r1_f + 0.25·r3_f + 2.0·top_f )     # V11 家族
    V13组件 = Σ_{f=1..8} z( 1.0·r1_f + 1.0·top_f )                  # V13 家族

策略协议 (冠军): 因子日收盘打分 → Top1 次日开盘买入 (整手) → 持有 5 个交易日收盘卖出;
持仓期任一收盘 ≤ 买入价 -8% → 次日开盘止损。回测口径 = eval_v17 的"收盘卖引擎"。

模块职责:
  - 路径/常量 (打分公式权重、协议参数)
  - heads 逐折宽表加载 / 家族打分 build / ens_w2 组装
  - 增量推演编排: 因子数据出现新交易日时, 自动在 V11/V13 目录运行其 analysis.py
  - 交易日历 / 股票名称加载
  - 冠军策略回测 (收盘卖引擎, 含分半) / IC 评估
  - 报告块生成 (V9 风格排行榜 + 策略指令)
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# ============================================================
# 路径与常量
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..')) + '/'
DATA_ROOT = PROJECT_ROOT + 'data/'
TD_ROOT = PROJECT_ROOT + 'trainingdata/'
V11_DIR = PROJECT_ROOT + 'Model/V11'
V13_DIR = PROJECT_ROOT + 'Model/V13'
V11_HEADS = V11_DIR + '/model_pred/2026q3/heads'
V13_HEADS = V13_DIR + '/model_pred/2026q3/heads'
V11_OFFICIAL_SCORE = V11_DIR + '/model_pred/2026q3/score_ens_w2.fea'
V11_STRAT = V11_DIR + '/strat_backtest.py'
FAC_ALL = TD_ROOT + 'fac_all.fea'
CALENDAR = DATA_ROOT + 'calendar.parquet'
PRICES = DATA_ROOT + 'daily_adj.parquet'
FEATURE_MAP = V11_DIR + '/model_test/feature_map.fea'

TEST_START = '20250901'          # 回测/展示窗口起点 (与全项目一致)
HOLD_DAYS = 5                    # 持有交易日数
STOP_LOSS = 0.08                 # 收盘止损线
NFOLD = 8                        # 每家族折数

HEAD_NAMES = ('r1', 'r3', 'r5', 'top')
# 家族打分 (每折: 头部加权 → 逐日截面 z → 跨折求和)
V11_MIX = dict(w1=1.0, w5=0.0, w3=0.25, wt=2.0)   # mixA = V11 家族
V13_MIX = dict(w1=1.0, w5=0.0, w3=0.0, wt=1.0)    # V13 家族 (r1 + top)
ENS_W2_CROSS = 2.0                                # ens_w2 = z(mixA) + 2·z(V13)

# ============================================================
# 基础工具
# ============================================================
def zn(df):
    """逐日截面 z-score (行 = 日期, 列 = 股票)。"""
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]


def zh_date(s):
    """'20260903' → '2026年9月3日'"""
    s = str(s)
    return f'{int(s[:4])}年{int(s[4:6])}月{int(s[6:8])}日'


def load_trading_dates():
    cal = pd.read_parquet(CALENDAR)
    return sorted(cal.loc[cal['is_open'] == 1, 'date'].astype(str)
                  .str.replace('-', '', regex=False).tolist())


def load_name_map():
    df = pd.read_parquet(PRICES, columns=['stock_code', 'stock_name'])
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


def factor_dates():
    """fac_all.fea 中全部因子日期 (轻量读 date 列)。"""
    import pyarrow.compute as pc
    import pyarrow.feather as pf
    col = pf.read_table(FAC_ALL, columns=['date']).column('date')
    return sorted(set(pc.unique(col).to_pandas().astype(str)))


def latest_reportable_date():
    """最新因子日 ∩ 交易日 (V11/V13 analysis 只会推到这个日期)。"""
    fd = set(factor_dates())
    td = set(load_trading_dates())
    inter = fd & td
    if not inter:
        raise RuntimeError('因子数据与交易日历无交集')
    return max(inter)


def _read_date_max(path):
    import pyarrow.compute as pc
    import pyarrow.feather as pf
    col = pf.read_table(path, columns=['date']).column('date')
    return max(pc.unique(col).to_pandas().astype(str))


def heads_last_date(heads_dir):
    p = os.path.join(heads_dir, f'{HEAD_NAMES[0]}_f1.fea')
    if not os.path.exists(p):
        return None
    return _read_date_max(p)


# ============================================================
# heads 加载 / 家族打分 / ens_w2 组装 (与 run_daily_pipeline.sh 逐字同口径)
# ============================================================
def _load_perfold(heads_dir, nf=NFOLD):
    """heads/{h}_f{f}.fea → {h: {f: DataFrame(日期×股票)}}"""
    perfold = {h: {} for h in HEAD_NAMES}
    for h in HEAD_NAMES:
        for f in range(1, nf + 1):
            p = os.path.join(heads_dir, f'{h}_f{f}.fea')
            if os.path.exists(p):
                perfold[h][f] = pd.read_feather(p).set_index('date')
    return perfold


def build_family(heads_dir, w1, w5, w3, wt, nf=NFOLD):
    """家族打分 = Σ_{f} z( w1·r1_f + w5·r5_f + w3·r3_f + wt·top_f )"""
    perfold = _load_perfold(heads_dir, nf)
    if not perfold['r1']:
        raise FileNotFoundError(f'{heads_dir} 下无 r1 头逐折文件')
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
    """组装冠军打分 ens_w2 (全窗口, 日期×股票)。"""
    mixA = build_family(V11_HEADS, **V11_MIX)
    v13 = build_family(V13_HEADS, **V13_MIX)
    ens = zn(mixA).add(ENS_W2_CROSS * zn(v13), fill_value=0.0)
    ens.index.name = 'date'
    ens = ens.sort_index()
    ens.columns = ens.columns.astype(str)
    return ens


# ============================================================
# 增量推演编排: 因子新日期 → 自动刷新 V11/V13 heads
# ============================================================
def ensure_heads_fresh(force=False, log_dir=None):
    """若 V11/V13 heads 未覆盖到最新因子交易日, 分别在两目录运行 analysis.py (增量, 重推最近10日)。

    返回 (refresh_done, message)。刷新失败抛 RuntimeError (附日志尾部)。
    """
    target = latest_reportable_date()
    a, b = heads_last_date(V11_HEADS), heads_last_date(V13_HEADS)
    if a is None or b is None:
        raise RuntimeError(f'heads 缺失 (V11={a}, V13={b}); 需先由 V11/V13 各自全量推演一次')
    cur = min(a, b)
    if cur >= target and not force:
        return False, f'heads 已覆盖至 {cur} (因子最新 {target}), 无需推演'
    if log_dir is None:
        log_dir = os.path.join(HERE, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    action = '强制重推' if force else f'heads 落后 (截至 {cur}, 因子最新 {target})'
    print(f'[V20] {action} → 依次刷新 V11/V13 增量推演 ...', flush=True)
    for tag, d in (('V11', V11_DIR), ('V13', V13_DIR)):
        log_path = os.path.join(log_dir, f'analysis_{tag.lower()}.log')
        print(f'[V20]   运行 {d}/analysis.py (日志 {log_path})', flush=True)
        with open(log_path, 'w', encoding='utf-8') as lf:
            r = subprocess.run([sys.executable, 'analysis.py'], cwd=d,
                               stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = ''
            try:
                tail = '\n'.join(open(log_path, encoding='utf-8',
                                      errors='ignore').read().splitlines()[-15:])
            except OSError:
                pass
            raise RuntimeError(f'{tag} analysis.py 退出码 {r.returncode}; 日志尾部:\n{tail}')
        a2, b2 = heads_last_date(V11_HEADS), heads_last_date(V13_HEADS)
        print(f'[V20]   {tag} heads 现在截至: {b2 if tag == "V13" else a2}', flush=True)
    if min(a2, b2) < target:
        raise RuntimeError(f'刷新后 heads 仍落后 (V11={a2}, V13={b2}, 目标 {target})')
    return True, f'heads 已刷新至 {min(a2, b2)}'


# ============================================================
# 冠军策略回测 (收盘卖引擎, 与冠军评估口径一致)
# ============================================================
def _close_sell_backtest_module():
    """读取 V11/strat_backtest.py, 把卖出价 open→close 后 exec, 返回其函数命名空间。"""
    with open(V11_STRAT, encoding='utf-8') as f:
        src = f.read()
    old = "sp = open_map.get((p['planned_sell'], code))"
    assert old in src, 'strat_backtest.py 卖出价行未找到, 无法构造收盘卖引擎'
    mod = {}
    exec(src.replace(old, "sp = close_map.get((p['planned_sell'], code))"), mod)
    return mod


def champion_backtest(score=None, split=True, verbose=True):
    """冠军协议回测 (hold5 + stop8% + 收盘卖 + 含成本) 及可选 H1/H2 分半。

    score: 打分 DataFrame (日期×股票), 默认 = score_ens_w2()。
    返回 dict(full/h1/h2=summarize指标, ic=RankIC/IR/top_return)。
    """
    if score is None:
        score = score_ens_w2()
    mod = _close_sell_backtest_module()
    rb, lp, lc = mod['run_backtest'], mod['load_prices'], mod['load_calendar']
    summ = mod['summarize']
    open_map, close_map, prev_close_map, amount_map, name_map = lp()
    tds = lc()
    F = dict(use_cost=True, exclude_st=True, exclude_limit_up=True,
             buy_gap_limit=0.095, hold=HOLD_DAYS, stop_loss=STOP_LOSS)

    def one(window_start=None, window_end=None):
        kw = dict(window_start=window_start) if window_start else {}
        if window_end:
            kw['window_end'] = window_end
        m, tr, _ = rb(score, open_map, close_map, prev_close_map, name_map,
                      amount_map=amount_map, tds=tds, **F, **kw)
        return m, tr

    full, trades = one()
    ic = mod['eval_ic'](score)
    out = {'full': summ(full), 'ic': ic,
           'n_trades_full': full['n_trades'], 'n_trades': len(trades)}
    if split:
        h1, _ = one(window_start=TEST_START, window_end='20260227')
        h2, _ = one(window_start='20260302', window_end='20260901')
        out['h1'] = summ(h1)
        out['h2'] = summ(h2)
    if verbose:
        s = out['full']
        print('===== 冠军策略回测 (hold5 + stop8% + 收盘卖, 含成本) =====')
        print(f"  打分: ens_w2 | RankIC={ic['RankIC']:+.4f}  "
              f"RankICIR={ic['RankICIR']:+.4f}  top_return={ic['top_return']:+.4f}")
        print(f"  净累计 {s['净累计%']:>8.2f}%  |  净年化 {s['净年化%']:>7.2f}%  |  "
              f"Sharpe {s['Sharpe']:>6.3f}  |  MaxDD {s['MaxDD%']:>7.2f}%  |  "
              f"胜率 {s['胜率%']:>5.1f}%  |  交易 {s['交易数']}  |  成本拖累 {s['成本拖累%']:>6.2f}%")
        if split:
            h1s, h2s = out['h1'], out['h2']
            print(f"  分半 H1 (20250901~20260227): {h1s['净累计%']:>8.2f}%   |   "
                  f"H2 (20260302~20260901): {h2s['净累计%']:>8.2f}%")
    return out


# ============================================================
# 报告块生成 (V9 风格排行榜 + 策略指令)
# ============================================================
_BAR = '─' * 60


def _score_str(v):
    return f'{v:+10.4f}'


def rank_block(date, score, topn=10, tds=None, name_map=None):
    """单个因子日排行榜块文本 (含协议头)。"""
    if date not in score.index:
        raise KeyError(f'{date} 不在打分范围内 ({score.index.min()}~{score.index.max()})')
    if name_map is None:
        name_map = load_name_map()
    if tds is None:
        tds = load_trading_dates()
    top = score.loc[date].sort_values(ascending=False).head(topn)
    buy_dt = next_td(date, 1, tds)
    sell_dt = next_td(buy_dt, HOLD_DAYS, tds) if buy_dt else None
    lines = []
    lines.append(_BAR)
    lines.append('  打分模型: ens_w2 = z(V11 mixA) + 2·z(V13 r1+top)  (V20 冻结集成, 16 折)')
    lines.append(f'  因子日期：{zh_date(date)} 收盘')
    if buy_dt:
        lines.append(f'  冠军策略：{zh_date(buy_dt)} 开盘买入 Top1 → 持有 {HOLD_DAYS} 个交易日'
                     f' ({(zh_date(sell_dt) if sell_dt else "?")} 收盘卖出); '
                     f'持仓期收盘较买入价下跌 ≥ {STOP_LOSS:.0%} → 次日开盘止损')
    lines.append(_BAR)
    lines.append('')
    lines.append('  排名   代码        名称                  打分')
    lines.append('  ' + '-' * 37)
    for i, code in enumerate(top.index, 1):
        lines.append(f'  {i:<4} {code:<9} {name_map.get(code, "?"):<10}'
                     f'   {_score_str(top[code])}')
    lines.append('')
    return '\n'.join(lines)


def decision_block(date, score, tds=None, name_map=None, holdings_path=None):
    """最新因子日的 Top1 决策指令文本 (冠军协议执行提示)。"""
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
    lines = []
    lines.append(_BAR)
    lines.append('  冠军策略指令 (Top1 单票, 5W 账户口径)')
    lines.append(f'  Top1: {c1}  {name_map.get(c1, "?")}   打分 {top[c1]:+.4f}')
    lines.append(f'  动作: {zh_date(buy_dt) if buy_dt else "?"} 开盘买入 (整手) '
                 f'→ 持有 {HOLD_DAYS} 个交易日 → '
                 f'{zh_date(sell_dt) if sell_dt else "?"} 收盘卖出')
    lines.append(f'  风控: 持仓期收盘较买入价下跌 ≥ {STOP_LOSS:.0%} → 次日开盘止损')
    # 纸面账户提示 (V11/paper_trader 状态, 只读提示)
    if holdings_path and os.path.exists(holdings_path):
        try:
            st = __import__('json').load(open(holdings_path, encoding='utf-8'))
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
    lines.append(_BAR)
    return '\n'.join(lines)


# ============================================================
# 冻结组件健康检查 (train.sh / 启动前自检)
# ============================================================
def validate_frozen(verbose=True):
    """校验冠军组件齐备。返回 (ok_all, [(ok, msg), ...])。"""
    import glob
    checks = []

    def chk(cond, msg):
        checks.append((bool(cond), msg))
        return bool(cond)

    chk(os.path.isdir(V11_DIR) and os.path.isdir(V13_DIR),
        f'组件目录存在 (V11/V13)')
    for tag, d in (('V11', V11_DIR), ('V13', V13_DIR)):
        folds = sorted(glob.glob(d + '/model_train/2026q3/fold*'))
        ckpts = glob.glob(d + '/model_train/2026q3/fold*/version_*/checkpoints/*.ckpt')
        chk(len(folds) == NFOLD and len(ckpts) >= NFOLD,
            f'{tag}: {len(folds)} 个 fold / {len(ckpts)} 个最佳 checkpoint (期望 {NFOLD}+)')
        heads_dir = d + '/model_pred/2026q3/heads'
        nheads = sum(len(glob.glob(heads_dir + f'/{h}_f*.fea'))
                     for h in HEAD_NAMES)
        chk(nheads == NFOLD * len(HEAD_NAMES),
            f'{tag}: heads 逐折文件 {nheads}/{NFOLD * len(HEAD_NAMES)}')
    chk(os.path.exists(FEATURE_MAP), 'feature_map.fea 存在 (V11/model_test)')
    chk(os.path.exists(FAC_ALL) and os.path.exists(CALENDAR) and os.path.exists(PRICES),
        '基础数据存在 (fac_all.fea / calendar / daily_adj)')
    try:
        latest = latest_reportable_date()
        a, b = heads_last_date(V11_HEADS), heads_last_date(V13_HEADS)
        cur = min(a, b) if (a and b) else None
        chk(cur is not None and cur >= latest,
            f'heads 覆盖因子最新日 ({cur} ≥ {latest})')
    except Exception as e:
        chk(False, f'日期覆盖检查异常: {e}')
    ok = all(o for o, _ in checks)
    if verbose:
        print('===== V20 冻结组件健康检查 =====')
        for o, m in checks:
            print(f'  [{"OK" if o else "FAIL"}] {m}')
        print(f'  → 结论: {"健康, 可直接 analysis.py 出结果" if ok else "存在缺失, 请先补齐"}')
    return ok, checks


if __name__ == '__main__':
    ok, _ = validate_frozen()
    sys.exit(0 if ok else 1)
