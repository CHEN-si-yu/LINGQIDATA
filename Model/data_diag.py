"""数据侧诊断: 因子层面 rankIC/ICIR 统计 (训练区间, 可交易池口径)。
只读 trainingdata, 不改动任何因子文件。"""
import sys
import os
# 线程上限: 本脚本多为小矩阵运算, 多线程反而慢且与训练抢核
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
ROOT = '/autodl-fs/data/lingqiData/'

print("[1] 加载数据 ...", flush=True)
fac = pd.read_feather(ROOT + 'trainingdata/fac_all.fea')
lab1 = pd.read_feather(ROOT + 'trainingdata/label_ret_1d.fea').set_index('index')
lab5 = pd.read_feather(ROOT + 'trainingdata/label_ret_5d.fea').set_index('index')
buy = pd.read_feather(ROOT + 'trainingdata/buyable_mask.fea').set_index('date')
amt = pd.read_feather(ROOT + 'trainingdata/trade_amt.fea').set_index('index')

factors = [c for c in fac.columns if c not in ('date', 'Code')]
print(f"   因子数={len(factors)}, fac_all={fac.shape}", flush=True)

train_dates = sorted(d for d in fac['date'].unique() if d < '20250810' and d in lab1.index)
train_dates = train_dates[::3]          # 每 3 天采样: 因子筛选口径, 统计量几乎不变
print(f"   训练区间采样天数={len(train_dates)} (每3天)", flush=True)

# 可疑因子名扫描
import re
susp = [f for f in factors if re.search(r'ret|return|label|target|fwd|fut|next|pct_chg|chg', f, re.I)]
print(f"[2] 可疑因子名 (含 ret/return/label/target/fwd/fut/next/chg): {len(susp)} 个")
for s in susp[:40]:
    print("   ", s)

print("[3] 逐日截面 rankIC (1d 标签, 可交易池) ...", flush=True)
fac_by_date = {d: g.drop(columns=['date']).set_index('Code') for d, g in fac[fac['date'].isin(train_dates)].groupby('date')}
del fac

def day_rankic(Xr, yr):
    # Xr: [n, F] ranks, yr: [n] ranks -> per-factor Pearson
    n = Xr.shape[0]
    Xc = Xr - Xr.mean(axis=0)
    yc = yr - yr.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(denom > 1e-12, (Xc.T @ yc) / denom, np.nan)
    return out

ics = []          # (date, ic_vector 1d)
ics5 = []         # (date, ic_vector 5d)
pool_sizes = []
label_stats = []
amt_ics = []
for di, d in enumerate(train_dates):
    if di % 100 == 0:
        print(f"   ... {di}/{len(train_dates)} ({d})", flush=True)
    X = fac_by_date[d]
    codes = X.index
    y1 = lab1.loc[d].reindex(codes) if d in lab1.index else pd.Series(np.nan, index=codes)
    y5 = lab5.loc[d].reindex(codes) if d in lab5.index else pd.Series(np.nan, index=codes)
    b = buy.loc[d].reindex(codes) if d in buy.index else pd.Series(np.nan, index=codes)
    a = amt.loc[d].reindex(codes) if d in amt.index else pd.Series(np.nan, index=codes)
    tm1 = b.gt(0.5) & y1.notna()
    tm5 = b.gt(0.5) & y5.notna()
    pool_sizes.append((d, int(tm1.sum())))
    if tm1.sum() >= 50:
        Xr = X.where(tm1, axis=0).rank(axis=0)
        yr = y1.where(tm1).rank()
        ics.append((d, day_rankic(Xr.to_numpy(np.float32), yr.to_numpy(np.float32))))
        # 流动性: log(amt) 与 |label|、label 的关系
        ar = np.log1p(a.where(tm1)).rank().to_numpy(np.float32)
        amt_ics.append((d, np.corrcoef(ar, yr.to_numpy(np.float32))[0, 1]))
        label_stats.append((d, y1[tm1].mean(), y1[tm1].std(),
                            (y1[tm1].abs() >= 0.095).mean()))
    if tm5.sum() >= 50:
        Xr5 = X.where(tm5, axis=0).rank(axis=0)
        yr5 = y5.where(tm5).rank()
        ics5.append((d, day_rankic(Xr5.to_numpy(np.float32), yr5.to_numpy(np.float32))))

ic_df = pd.DataFrame({d: v for d, v in ics}, index=factors).T     # [days, F]
ic5_df = pd.DataFrame({d: v for d, v in ics5}, index=factors).T
print(f"   有效天数: 1d={len(ic_df)}, 5d={len(ic5_df)}", flush=True)

stats = pd.DataFrame({
    'IC1_mean': ic_df.mean(), 'IC1_std': ic_df.std(),
    'IC1_IR': ic_df.mean() / ic_df.std(),
    'IC5_mean': ic5_df.mean(), 'IC5_IR': ic5_df.mean() / ic5_df.std(),
}).sort_values('IC1_IR', ascending=False)
print("\n[4] 1d rankIC 最强 20 个因子 (IR 排序):")
print(stats.head(20).round(4).to_string())
print("\n    1d rankIC 最弱(负) 10 个因子:")
print(stats.tail(10).round(4).to_string())
print("\n    1d ICIR>0.05 的因子数:", int((stats['IC1_IR'] > 0.05).sum()),
      " | IC5_IR>0.05 的因子数:", int((stats['IC5_IR'] > 0.05).sum()))

# 前 40 因子相关性 (IC 序列相关)
top40 = stats.head(40).index
corr40 = ic_df[top40].corr()
print("\n[5] 前40因子 IC 序列平均相关: %.3f (绝对值), 最大非对角 %.3f" % (
    corr40.abs().values[np.triu_indices(40, 1)].mean(),
    corr40.abs().values[np.triu_indices(40, 1)].max()))

ls = pd.DataFrame(label_stats, columns=['d', 'mean', 'std', 'tail_frac'])
print("\n[6] 标签统计 (1d, 可交易池):")
print("    均值 %.4f%%/d, 截面std均值 %.3f%%, |label|>=9.5%% 占比 %.3f%%" % (
    ls['mean'].mean() * 100, ls['std'].mean() * 100, ls['tail_frac'].mean() * 100))

# 年度 IC 稳定性 (top20 因子)
ic_df.index = pd.to_datetime([d for d, _ in ics], format='%Y%m%d')
print("\n[7] top-20 因子按年 IC 均值 (1d):")
yearly = ic_df[top40[:20]].groupby(ic_df.index.year).mean()
print(yearly.round(4).to_string())

# 等权因子组合 IC
ew = ic_df.mean(axis=1)
print("\n[8] 全部因子等权 IC: mean=%.4f IR=%.3f" % (ew.mean(), ew.mean() / ew.std()))
top10ew = ic_df[stats.head(10).index].mean(axis=1)
print("    前10因子等权 IC: mean=%.4f IR=%.3f" % (top10ew.mean(), top10ew.mean() / top10ew.std()))
ic5ew = ic5_df.mean(axis=1)
print("    5d标签全因子等权 IC: mean=%.4f IR=%.3f" % (ic5ew.mean(), ic5ew.mean() / ic5ew.std()))

# 1d 与 5d IC 相关性 (因子层面)
print("\n[9] 因子 1d-IC 与 5d-IC 相关: %.3f" % stats['IC1_mean'].corr(stats['IC5_mean']))

amt_ic = pd.Series({d: v for d, v in amt_ics})
print("\n[10] log(成交额) 与 label 秩相关: mean=%.4f IR=%.3f (流动性因子强度)" % (
    amt_ic.mean(), amt_ic.mean() / amt_ic.std()))

stats.round(4).to_csv(ROOT + 'Model/factor_ic_stats.csv')
print("\n[DONE] 结果已存 Model/factor_ic_stats.csv")
