# 每日增量更新流程

三步走：**查现状 → 跑增量 → 看结果**。

**18:00 分界规则**：每日 18:00 前，当日交易数据尚未全部发布，理论最新日期为昨天（T-1）；18:00 后数据陆续到位，理论最新日期为今天（T）。延迟天数均以此 `ref_date` 为基准计算。

---

## 步骤 1：检查本地数据现状

按数据集类型分组、按延迟天数排序，逐一展示每个数据集的最新日期和状态。

```bash
cd /root/shared-nvme/lingqiData
python3 <<'PYEOF'
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import Counter

data_dir = Path('data')
now = datetime.now()
today = now.date()

# 18:00 分界：之前理论最新为 T-1，之后为 T
if now.hour < 18:
    ref_date = today - timedelta(days=1)
else:
    ref_date = today

date_candidates = ['trade_date','end_date','date']

# --- 读取所有数据集 ---
records = []
for f in sorted(data_dir.glob('*.parquet')):
    schema = pq.read_schema(f)
    date_col = next((c for c in date_candidates if c in schema.names), None)
    name = f.stem
    if date_col is None:
        records.append({'name': name, 'date_col': 'N/A', 'max_date': None, 'rows': 0, 'lag_days': None})
        continue
    t = pq.read_table(f, columns=[date_col])
    if t.num_rows == 0:
        records.append({'name': name, 'date_col': date_col, 'max_date': None, 'rows': 0, 'lag_days': None})
        continue
    s = pd.Series(t.column(0).to_pandas()).dropna()
    if len(s) > 0:
        max_str = str(s.max())[:10]
        max_date = datetime.strptime(max_str, '%Y-%m-%d').date()
        lag = (ref_date - max_date).days
    else:
        max_date = None
        lag = None
    records.append({
        'name': name, 'date_col': date_col,
        'max_date': str(max_date) if max_date else None,
        'rows': t.num_rows, 'lag_days': lag
    })

# --- 分类 ---
DAILY = {
    'daily','daily_adj','finance','margin_detail','main_fund_flow',
    'main_fund_flow_overview','dragon_tiger','top_list','limit_up','limit_list',
    'holder_number',
}
WEEKLY_MONTHLY = {
    'kline_weekly','kline_monthly','kline_adj_weekly','kline_adj_monthly',
}
QUARTERLY = {
    'balancesheet','cashflow','income','financial_indicator',
}
MONTHLY = {
    'pledge_stat',
}
REFERENCE = {
    'calendar','stock_list','list',
}

def category_of(name):
    if name in DAILY: return '日频'
    if name in WEEKLY_MONTHLY: return '周/月K线'
    if name in QUARTERLY: return '季报'
    if name in MONTHLY: return '月频'
    if name in REFERENCE: return '参考数据'
    return '其他'

# --- 状态判定（以 ref_date 为基准）---
def status_of(r):
    if r['lag_days'] is None or r['rows'] == 0:
        return ('empty', 'EMPTY')
    cat = category_of(r['name'])
    if cat == '季报':
        return ('ok', 'OK')
    if cat == '参考数据':
        return ('ok', 'OK')
    if cat == '月频':
        if r['lag_days'] <= 30: return ('ok', 'OK')
        return ('warn', 'LAG')
    # 日频 / 周线: 0~1d OK, 2~3d LAG, >3d STALE
    if r['lag_days'] <= 1:
        return ('ok', 'OK')
    elif r['lag_days'] <= 3:
        return ('warn', 'LAG')
    else:
        return ('stale', 'STALE')

for r in records:
    r['category'] = category_of(r['name'])
    s_code, s_label = status_of(r)
    r['status_code'] = s_code
    r['status_label'] = s_label

# --- 排序: 问题优先 (stale > warn > ok > empty) → 类别 → 名称 ---
severity = {'stale': 0, 'warn': 1, 'ok': 2, 'empty': 3}
cat_order = {'日频': 0, '周/月K线': 1, '月频': 2, '季报': 3, '参考数据': 4, '其他': 5}
records.sort(key=lambda r: (severity.get(r['status_code'], 9), cat_order.get(r['category'], 9), r['name']))

# --- 展示 ---
print(f'Now: {now.strftime("%Y-%m-%d %H:%M")} | Ref date: {ref_date} (cutoff 18:00)')
print()
print(f'{"Dataset":<28} {"Category":<10} {"Latest":<12} {"Lag":>5}  {"Status":<8} {"Rows":>12}')
print('-' * 98)

prev_cat = None
for r in records:
    if r['category'] != prev_cat:
        prev_cat = r['category']
        print(f'-- {prev_cat} --')
    lag_str = f'{r["lag_days"]}d' if r['lag_days'] is not None else '-'
    max_str = r['max_date'] or '-'
    print(f'{r["name"]:<28} {r["category"]:<10} {max_str:<12} {lag_str:>5}  {r["status_label"]:<8} {r["rows"]:>12,}')

# --- 汇总 ---
stales  = [r for r in records if r['status_code'] == 'stale']
warns   = [r for r in records if r['status_code'] == 'warn']
oks     = [r for r in records if r['status_code'] == 'ok']
empties = [r for r in records if r['status_code'] == 'empty']

print()
print(f'Summary: {len(records)} datasets -- '
      f'{len(oks)} OK | {len(warns)} lagging | {len(stales)} stale | {len(empties)} empty')

if stales:
    stale_names = ', '.join(r['name'] for r in stales)
    print(f'  STALE (>3d behind ref):  {stale_names}')
if warns:
    warn_names = ', '.join(r['name'] for r in warns)
    print(f'  LAG (2-3d):              {warn_names}')
if oks:
    ok_names = ', '.join(r['name'] for r in oks)
    print(f'  OK:                      {ok_names}')

# --- cyq_chips ---
chip_dir = data_dir / 'cyq_chips'
if chip_dir.exists():
    files = list(chip_dir.glob('*.parquet'))
    dates = Counter()
    for cf in files:
        try:
            t = pq.read_table(cf, columns=['trade_date'])
            if t.num_rows > 0:
                s = pd.Series(t.column(0).to_pandas())
                if not s.empty:
                    dates[str(s.max())[:10]] += 1
        except: pass
    print(f'\ncyq_chips: {len(files)} files')
    for d, n in sorted(dates.items(), reverse=True)[:5]:
        pct = n/len(files)*100
        bar = '#' * int(pct / 10)
        print(f'  {d}: {n:>5} files ({pct:5.1f}%) {bar}')

# --- 记录执行日志 ---
log_dir = data_dir / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
with open(log_dir / 'daily_check.log', 'a') as lf:
    lf.write(f'{now.strftime("%Y-%m-%d %H:%M:%S")} | step1 | ref={ref_date} | '
             f'{len(oks)} OK, {len(warns)} LAG, {len(stales)} STALE, {len(empties)} EMPTY\n')
PYEOF
```

预期输出：
- **日频数据**：18:00 前 ref_date = T-1，最新日期应与 ref_date 一致（0d lag），或差 1 天（T+1 正常）；18:00 后 ref_date = T，数据应更新到当天
- **季报数据**：停在最近季度末，lag 天数较大但状态始终 OK
- **cyq_chips**：100% 落在最新交易日为正常

---

## 步骤 2：运行增量更新

```bash
cd /root/shared-nvme/lingqiData
python3 scripts/incremental_update.py --parallel -w 6
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--parallel` | 多数据集并发（推荐） |
| `-w 6` | 每个数据集的内部并行线程数 |
| `--dataset daily margin_detail` | 只更新指定数据集 |
| `--dry-run` | 预览但不实际拉取 |
| `--overlap 5` | 扩大重叠窗口到 5 天 |

执行过程日志输出到 `data/logs/` 目录，同时打印到控制台。

---

## 步骤 3：对比更新结果

更新后重新诊断每个数据集的状态，与 `.incr_state.json` 对比展示 pending 情况。

```bash
cd /root/shared-nvme/lingqiData
python3 <<'PYEOF'
import json
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import Counter

data_dir = Path('data')
now = datetime.now()
today = now.date()

# 18:00 分界
if now.hour < 18:
    ref_date = today - timedelta(days=1)
else:
    ref_date = today

date_candidates = ['trade_date','end_date','date']

# --- 读取状态文件 ---
state_path = data_dir / '.incr_state.json'
if state_path.exists():
    with open(state_path) as f:
        state = json.load(f)
else:
    state = {}

# --- 从 parquet 读取最新日期 ---
def read_max_date(parquet_path):
    try:
        schema = pq.read_schema(parquet_path)
        date_col = next((c for c in date_candidates if c in schema.names), None)
        if date_col is None:
            return None, None
        t = pq.read_table(parquet_path, columns=[date_col])
        if t.num_rows == 0:
            return date_col, None
        s = pd.Series(t.column(0).to_pandas()).dropna()
        if len(s) > 0:
            max_str = str(s.max())[:10]
            return date_col, datetime.strptime(max_str, '%Y-%m-%d').date()
        return date_col, None
    except:
        return None, None

# --- 分类（同步骤1） ---
DAILY = {
    'daily','daily_adj','finance','margin_detail','main_fund_flow',
    'main_fund_flow_overview','dragon_tiger','top_list','limit_up','limit_list',
    'holder_number',
}
WEEKLY_MONTHLY = {'kline_weekly','kline_monthly','kline_adj_weekly','kline_adj_monthly'}
QUARTERLY = {'balancesheet','cashflow','income','financial_indicator'}
MONTHLY = {'pledge_stat'}
REFERENCE = {'calendar','stock_list','list'}

def category_of(name):
    if name in DAILY: return '日频'
    if name in WEEKLY_MONTHLY: return '周/月K线'
    if name in QUARTERLY: return '季报'
    if name in MONTHLY: return '月频'
    if name in REFERENCE: return '参考数据'
    return '其他'

def status_of(name, lag_days):
    cat = category_of(name)
    if cat == '季报' or cat == '参考数据':
        return 'ok'
    if cat == '月频':
        return 'ok' if (lag_days or 999) <= 30 else 'warn'
    if lag_days is None:
        return 'empty'
    if lag_days <= 1:
        return 'ok'
    elif lag_days <= 3:
        return 'warn'
    else:
        return 'stale'

severity_order = {'stale': 0, 'warn': 1, 'ok': 2, 'empty': 3}
cat_order = {'日频': 0, '周/月K线': 1, '月频': 2, '季报': 3, '参考数据': 4, '其他': 5}

rows = []
for f in sorted(data_dir.glob('*.parquet')):
    name = f.stem
    date_col, max_date = read_max_date(f)
    lag = (ref_date - max_date).days if max_date else None
    cat = category_of(name)
    st = status_of(name, lag)

    prev = state.get(name, {})
    is_pending = prev.get('pending', False)
    pending_since = prev.get('pending_since', '')

    rows.append({
        'name': name, 'category': cat,
        'max_date': str(max_date) if max_date else None,
        'lag_days': lag,
        'status': st,
        'pending': is_pending,
        'pending_since': pending_since,
    })

rows.sort(key=lambda r: (
    severity_order.get(r['status'], 9),
    cat_order.get(r['category'], 9),
    r['name']
))

# --- 展示 ---
print(f'Now: {now.strftime("%Y-%m-%d %H:%M")} | Ref date: {ref_date} (cutoff 18:00)')
print()
hdr = f'{"Dataset":<28} {"Cat":<8} {"Latest":<12} {"Lag":>5}  {"Status":<8} {"Pending":<16}'
print(hdr)
print('-' * 95)

prev_cat = None
st_map = {'ok': 'OK', 'warn': 'LAG', 'stale': 'STALE', 'empty': 'EMPTY'}

for r in rows:
    if r['category'] != prev_cat:
        prev_cat = r['category']
        print(f'-- {prev_cat} --')
    lag_str = f'{r["lag_days"]}d' if r['lag_days'] is not None else '-'
    max_str = r['max_date'] or '-'
    st_label = st_map.get(r['status'], '?')

    if r['pending']:
        pending_str = 'PENDING since ' + r['pending_since']
    elif r['status'] == 'ok':
        pending_str = '--'
    else:
        pending_str = 'check'

    print(f'{r["name"]:<28} {r["category"]:<8} {max_str:<12} {lag_str:>5}  {st_label:<8} {pending_str}')

# --- 汇总 ---
stales  = [r for r in rows if r['status'] == 'stale']
warns   = [r for r in rows if r['status'] == 'warn']
oks     = [r for r in rows if r['status'] == 'ok']
pending = [r for r in rows if r['pending']]

print()
print(f'Summary: {len(rows)} datasets -- '
      f'{len(oks)} OK | {len(warns)} lagging | {len(stales)} stale | {len(pending)} pending')

if stales:
    stale_names = ', '.join(r['name'] for r in stales)
    print(f'  STALE (server behind >3d):  {stale_names}')
if warns:
    warn_names = ', '.join(r['name'] for r in warns)
    print(f'  LAG (2-3d):                 {warn_names}')
if pending:
    pending_names = ', '.join(r['name'] for r in pending)
    print(f'  PENDING (server behind):    {pending_names}')
else:
    print(f'  All datasets up to date')

# --- cyq_chips ---
chip_dir = data_dir / 'cyq_chips'
if chip_dir.exists():
    files = list(chip_dir.glob('*.parquet'))
    dates = Counter()
    for cf in files:
        try:
            t = pq.read_table(cf, columns=['trade_date'])
            if t.num_rows > 0:
                s = pd.Series(t.column(0).to_pandas())
                if not s.empty:
                    dates[str(s.max())[:10]] += 1
        except: pass
    latest_date = max(dates.keys()) if dates else '?'
    latest_pct = dates[latest_date] / len(files) * 100 if dates else 0
    lagging_stocks = sum(n for d, n in dates.items() if d != latest_date)
    print(f'\ncyq_chips: {len(files)} files, latest={latest_date} ({latest_pct:.1f}%), '
          f'{lagging_stocks} lagging stocks')

# --- 记录执行日志 ---
log_dir = data_dir / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
with open(log_dir / 'daily_check.log', 'a') as lf:
    lf.write(f'{now.strftime("%Y-%m-%d %H:%M:%S")} | step3 | ref={ref_date} | '
             f'{len(oks)} OK, {len(warns)} LAG, {len(stales)} STALE, {len(pending)} PENDING\n')
PYEOF
```

### 解读

- **18:00 前** `ref_date = T-1`：日频数据 0d lag 为最佳，1d 为 T+1 正常范围
- **18:00 后** `ref_date = T`：当日数据陆续到位，0d 最佳
- **全部 OK 且无 pending**：所有数据已是最新，无需额外操作
- **有 LAG**：延迟 2-3 天，服务端滞后，下次增量自动扩大 overlap 覆盖
- **有 STALE**：延迟超过 3 天，可能是接口故障，需要排查
- **有 PENDING**：服务端数据落后于交易日历，脚本已记录，下次运行扩大 overlap 重拉
- **季报停在季度末**：正常现象，下一季度财报发布前不会更新

---

## 状态追踪机制

增量更新脚本维护 `data/.incr_state.json`，记录每次运行后每个数据集的实际最新日期。

```
{
  "margin_detail": {
    "last_run": "2026-05-14",
    "last_fetch_max": "2026-05-12",
    "pending": true,
    "pending_since": "2026-05-14"
  }
}
```

- `pending: true` → 上次运行时服务端数据没跟上，下次运行会扩大 overlap 窗口自动重拉
- 1 天容忍：日频 T+1 延迟不视为 pending
- 状态文件由脚本自动维护，无需手动修改

### 执行日志

步骤 1 和步骤 3 每次运行自动追加到 `data/logs/daily_check.log`：

```
2026-05-14 12:38:01 | step1 | ref=2026-05-13 | 19 OK, 2 LAG, 0 STALE, 2 EMPTY
2026-05-14 12:39:15 | step3 | ref=2026-05-13 | 21 OK, 2 LAG, 0 STALE, 7 PENDING
```

---

## 建议的 cron 配置

每天上午 9:30（开盘后）运行一次：

```bash
30 9 * * 1-5 cd /root/shared-nvme/lingqiData && python3 scripts/incremental_update.py --parallel -w 6 2>&1 | tee -a data/logs/cron_$(date +\%Y\%m\%d).log
```

周末不需要运行（无交易日数据）。
