# Debug：历史数据被今日数据污染的 Bug 追踪

## Bug 现象

每日运行 pipeline 后，**已经产生的历史日期的模型预测排名会发生变化**。

具体表现：
- 7月27日运行：因子日期 0727 预测，第1名是 **002141 贤丰控股**
- 7月28日运行：同一因子日期 0727 预测，第1名变成 **000506 招金黄金**

## 当前实验状态 (2026-07-29)

### 核心改进：逐日横截面数据指纹

原来的 snapshot.py 对每个 parquet 文件计算全文件 MD5——每天新增一行就变了，毫无意义。

**v2 改进**：对 6 个关键源文件的每个日期，独立计算横截面数据 MD5（该日期所有股票 × 所有数值列，按股票代码排序后 hash）。这样对比两个快照时，可以直接判断"同一个日期的数据是否被修改"。

### 已保存的数据

**debug/20260729/** 目录包含今天的完整快照（v2），核心新增文件：

| 文件 | 内容 | 用途 |
|------|------|------|
| `source_date_hashes.json` | 6个源文件最近10天的逐日横截面数据 MD5 | **明天对比的核心** |
| `fac_all_last3_sample.json` | fac_all 最后3天 × 16因子 × 全部股票值 | 精确对比因子值漂移 |
| `factor_sample.json` | 16个代表性因子的 top10/bottom10 | 对比因子排名变化 |
| `model_pred_full.json` | 最后5个因子日期的完整预测 | 对比预测排名变化 |

### 今天(0729)的逐日数据指纹

明天增量更新后，overlap 会重取 0727-0730。对比同一个日期的 MD5 即可判断数据是否被修订。

| 源文件 | 0727 MD5 (前16位) | 0728 MD5 (前16位) | 0729 MD5 (前16位) |
|--------|-------------------|-------------------|-------------------|
| daily | `e9f493e2e8495b35` | `d33327f80c80e039` | `209f9cabf2ad8178` |
| daily_adj | `8c80877480201266` | `8f904acca4739e8a` | `4e56286e25e7b4e3` |
| finance | `9ae4f6c85aa69862` | `71cf4065d649d3ea` | `42b2e9dea861802e` |
| cyq_perf | `58ea4ad210232d5d` | `646070ad718865c5` | `2cf1d73f3c91afc1` |
| main_fund_flow | `be3994cf908c1313` | `bf68d4321b17b26a` | `9a4865bd08f8c562` |
| margin_detail | `323c620a72b74f8a` | `1f4b722c05573a06` | (T+1, 无0729) |

完整 MD5 见 `debug/20260729/source_date_hashes.json`。

## 明天 (7/30) 操作步骤

### 步骤 1：跑完每日 pipeline

```
1. incremental.py — 爬取最新数据（overlap 会重取 0727-0730）
2. 删除 featureengineering/data/ 下所有 .fea 文件
3. build_factors.py --force — 全量重建因子
4. prepared_data.py — 合并到 fac_all.fea
5. analysis.py — 模型推演
```

### 步骤 2：生成快照

```bash
cd /autodl-fs/data/lingqiData
python debug/snapshot.py
```

自动创建 `debug/20260730/`，包含 `source_date_hashes.json`。

### 步骤 3：对比快照（核心步骤）

```bash
cd /autodl-fs/data/lingqiData
python debug/compare.py debug/20260729 debug/20260730
```

### 步骤 4：解读结果

compare.py 输出按优先级：

**🔴 级别 1 — 数据修订（source_date_hashes.json）：**
如果看到：
```
★★★ daily: 1 HISTORICAL DATE(S) CHANGED! ★★★
    20260729: 5524→5525 stocks, MD5 209f9cab... → a1b2c3d4...
```
→ 数据供应商修订了 0729 的数据！这是 Bug 的根因——overlap 机制把这个修订带入了 parquet。

**🟡 级别 2 — 因子值漂移（factor_sample.json）：**
```
★ FACTOR mom_20 DATE 20260729: top10 changed!
```
→ 即使源数据 MD5 相同，因子值也在变化（可能是非确定性）。

**🟠 级别 3 — 预测变化（model_pred_full.json）：**
```
★ 2026q3/20260729: TOP-10 RANKING CHANGED!
```
→ 模型预测漂移。

### 步骤 5：如果发现数据修订，手动深挖具体差异

```python
import json

a = json.load(open("debug/20260729/source_date_hashes.json"))
b = json.load(open("debug/20260730/source_date_hashes.json"))

for name in ["daily", "daily_adj", "finance", "cyq_perf", "main_fund_flow"]:
    # 注意：如果 snapshot.py v2 的 key 是 "dates"，这里需要改成 "dates"
    a_dates = a[name].get("last7_dates", a[name].get("dates", {}))
    b_dates = b[name].get("last7_dates", b[name].get("dates", {}))
    
    for d in sorted(set(a_dates.keys()) & set(b_dates.keys())):
        ha = a_dates[d].get("md5")
        hb = b_dates[d].get("md5")
        if ha and hb and ha != hb:
            print(f"★★★ {name} DATE {d}: DATA CHANGED!")
            print(f"    {ha} → {hb}")
```

---

## 实验逻辑

```
今天 (7/29)                      明天 (7/30)
─────────────                    ─────────────
incremental.py 跑完              incremental.py 跑完
parquet 中有 0727-0729 数据      overlap 重取 0727-0730
                                parquet 中 0727-0729 可能被修订
       │                               │
       ▼                               ▼
snapshot.py v2                    snapshot.py v2
source_date_hashes.json           source_date_hashes.json
记录 0729 数据指纹                 记录 0729 数据指纹
       │                               │
       └────────── 对比 ───────────────┘
                       │
                       ▼
            0729 的 MD5 相同？
            YES → 数据供应商没有修订，Bug 在别处
            NO  → 数据供应商修订了！overlap 带入污染！
```

## 关键代码路径

```
lingqiData/
├── scripts/incremental.py                          # 每日数据爬取 (overlap=3)
├── scripts/fetchers/incremental_core.py            # overlap 核心
│   └── merge_and_save() keep="last" → UPSERT
├── featureengineering/
│   ├── build_factors.py
│   └── src/featureengineering/
│       ├── utils.py:30-55                          # cross_sectional_rank()
│       ├── storage.py:41-100                       # align_to_reference()
│       └── dataset.py:48-124                       # _fill_source_gaps()
├── Model/
│   ├── prepared_data.py
│   └── V1.0/analysis.py
└── debug/
    ├── README.md                                   # ← 本文件
    ├── snapshot.py                                 # 快照 v2（逐日横截面指纹）
    ├── compare.py                                  # 对比 v2（逐日 MD5 对比）
    ├── 20260728/                                   # 7/28 快照（v1，无逐日指纹）
    └── 20260729/                                   # 7/29 快照（v2，含 source_date_hashes.json）
```

## 已知注意事项

1. `margin_detail.parquet` 比其他数据源少一天（T+1 延迟）
2. 因子存储的是 cross_sectional_rank 后的值（均值≈0.50028）
3. `incremental.py` overlap=3 自然日，`merge_and_save(keep="last")` 无条件覆盖
4. source_date_hashes.json 的 key 可能是 "last7_dates" 或 "dates"，取决于快照版本
