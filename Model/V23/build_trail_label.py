#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_trail_label.py — V23 移动止损轨迹标签构造 (最紧模型×策略耦合)

标签定义 (与 Trading/engine.py h20tr20_t2 协议逐日一致):
  factor 日 d 收盘打分 → d+1 (E) 开盘买入价 open[E];
  peak 起点 = max(open[E], close[E]); 逐日 d'∈[E+1, E+19]:
    peak = max(peak, close[d']);
    close[d'] ≤ peak × 0.80 → 触发移动止损 → 卖出价 = open[d'+1];
  20 个交易日无触发 → 卖出价 = open[E+20];
  trail_ret = 卖出价/买入价 - 1。
不可交易/不可执行 (buyable_mask 外, 买入日一字涨停, 缺价) → NaN。

输出: trainingdata/label_trail20d.fea  (index=因子日, columns=代码, 与 label_ret_*.fea 同宽表)
"""
import numpy as np
import pandas as pd

PROJECT_ROOT = "/autodl-fs/data/lingqiData/"
TRAIL = 0.20
HOLD = 20
LIMIT_UP = 0.095


def main():
    df = pd.read_parquet(PROJECT_ROOT + 'data/daily_adj.parquet',
                         columns=['stock_code', 'trade_date', 'open', 'close'])
    df['trade_date'] = df['trade_date'].str.replace('-', '', regex=False)
    df['code'] = df['stock_code'].str.replace('.SZ', '', regex=False) \
                                   .str.replace('.SH', '', regex=False)
    df = df.drop(columns=['stock_code'])
    cal = pd.read_parquet(PROJECT_ROOT + 'data/calendar.parquet')
    tds = sorted(cal.loc[cal['is_open'] == 1, 'date'].astype(str)
                 .str.replace('-', '', regex=False).tolist())
    tdi = {d: i for i, d in enumerate(tds)}
    codes = sorted(df['code'].unique())
    cidx = {c: i for i, c in enumerate(codes)}
    n_c, n_t = len(codes), len(tds)
    open_m = np.full((n_t, n_c), np.nan, dtype=np.float32)
    close_m = np.full((n_t, n_c), np.nan, dtype=np.float32)
    for _, r in df.iterrows():
        i, j = tdi.get(r['trade_date']), cidx.get(r['code'])
        if i is None or j is None:
            continue
        open_m[i, j] = r['open']
        close_m[i, j] = r['close']
    prev_close = np.full_like(close_m, np.nan)
    prev_close[1:] = close_m[:-1]
    print(f'[trail] 价格矩阵 {n_t} 天 × {n_c} 代码', flush=True)

    buyable = pd.read_feather(PROJECT_ROOT + 'trainingdata/buyable_mask.fea') \
        .set_index('date')
    buyable = buyable.reindex(columns=codes).fillna(False).astype(np.float32)

    n_rows = 0
    rows = []
    dates_out = []
    for i in range(n_t - HOLD - 1):          # 因子日 i → 买入日 i+1, 需要价到 i+21
        E = i + 1
        entry = open_m[E].copy()
        entry_ok = np.isfinite(entry) & (entry > 0)
        # 执行过滤: 买入日一字涨停 (买不进) / 因子日一字涨停 (引擎同口径)
        gap_buy = np.where(prev_close[E] > 0, entry / prev_close[E] - 1.0, np.nan)
        gap_fd = np.where(prev_close[i] > 0, close_m[i] / prev_close[i] - 1.0, np.nan)
        entry_ok &= ~(gap_buy >= LIMIT_UP) & ~(gap_fd >= LIMIT_UP)
        peak = np.maximum(entry, close_m[E])
        exit_price = open_m[E + HOLD].copy()
        triggered = np.zeros(n_c, dtype=bool)
        for d in range(E + 1, E + HOLD):      # held 1..19 的收盘检查
            cc = close_m[d]
            peak = np.fmax(peak, cc)
            hit = (~triggered) & np.isfinite(cc) & (cc > 0) \
                & (cc <= peak * (1 - TRAIL))
            exit_price[hit] = open_m[d + 1][hit]
            triggered |= hit
        ret = exit_price / entry - 1.0
        ret = np.where(entry_ok & np.isfinite(exit_price) & (exit_price > 0),
                       ret, np.nan)
        dstr = tds[i]
        if dstr in buyable.index:
            ret = np.where(buyable.loc[dstr].values.astype(np.float32) > 0.5,
                           ret, np.nan)
        rows.append(ret)
        dates_out.append(dstr)
        if (i + 1) % 200 == 0:
            print(f'[trail] {i + 1}/{n_t} 天', flush=True)
    mat = pd.DataFrame(np.vstack(rows), index=dates_out, columns=codes)
    mat.index.name = 'index'
    mat = mat.astype(np.float32)
    out = PROJECT_ROOT + 'trainingdata/label_trail20d.fea'
    mat.reset_index().to_feather(out)
    finite = np.isfinite(mat.values).sum(axis=1)
    print(f'[trail] {mat.shape} → {out}; 有效日均 {finite.mean():.0f} 行',
          flush=True)
    print(f'[trail] 覆盖 {dates_out[0]} ~ {dates_out[-1]}; 检验均值 '
          f'{np.nanmean(mat.values):+.5f} std {np.nanstd(mat.values):.4f}',
          flush=True)


if __name__ == '__main__':
    main()
