---
name: data-hash-monitor
description: 日频数据字段级哈希监控。当用户提到"数据快照"、"字段哈希"、"数据指纹"、"数据污染检查"、"历史数据是否变动"、"横截面哈希"时使用。提供两条路径：(A) snapshot_v5.py 生成完整的 field×date 哈希矩阵快照（原始字段+因子+目标,10日回溯）；(B) hash_analyze.py 与历史账本比对检测数据污染。两套哈希算法不同不可混用。
---

# 数据哈希监控 Skill

## 两条路径

| | snapshot_v5.py（快照） | hash_analyze.py（账本比对） |
|---|---|---|
| **用途** | 生成单次完整快照,记录所有字段的当日指纹 | 与历史账本比对,检测上游是否修订/污染了历史数据 |
| **粒度** | 字段级 (630 fields × 10 dates) | 字段级 (9源) |
| **覆盖** | 8 原始源 + 536 因子 + 5 目标 | 9 原始源 (含 indicator_1min) |
| **输出** | `debug/YYYYMMDD/field_date_matrix.json` | `debug/hash_analyze/ledger/` + `reports/` |
| **速度** | ~2-5 分钟 (24 workers, 1000股采样) | ~2-5 分钟 |
| **使用频率** | 每日 pipeline 后或按需 | 每日 pipeline 后例行检查 |

---

## 路径 A: 生成完整哈希快照 (snapshot_v5.py)

### 覆盖范围

**原始数据 (8源, 89字段)**:

| 源 | 文件 | 字段数 | 采样 | 滞后 |
|---|---|---|---|---|
| daily | daily.parquet | 11 | 全量 | 无 |
| finance | finance.parquet | 19 | 全量 | 无 |
| cyq_perf | cyq_perf.parquet | 10 | 全量 | 无 |
| main_fund_flow | main_fund_flow.parquet | 19 | 全量 | 无 |
| margin_detail | margin_detail.parquet | 9 | 全量 | **T+1** |
| ths_daily | ths_daily.parquet | 11 | 全量 | 无 |
| cyq_chips | cyq_chips/*.parquet | 3 (price, percent, stock_code) | **1000股** (总市值top) | 无 |
| history_1min | history_1min/*.parquet | 7 (open, high, low, close, vol, amount, stock_code) | **1000股** (总市值top) | 无 |

**因子 (536)**: `featureengineering/data/factors/*.fea`

**目标 (5)**: `featureengineering/data/targets/label_ret_{1d,3d,5d,10d,20d}.fea`

### 执行命令

```bash
cd /autodl-fs/data/lingqiData
/autodl-fs/data/miniconda3/bin/python3 debug/snapshot_v5.py \
  --days 10 --jobs 24 --out debug/YYYYMMDD
```

参数:
- `--days 10`: 回溯 10 个交易日（每源按自身实际日期独立求解窗口）
- `--jobs 24`: 并行 worker 数
- `--out debug/YYYYMMDD`: 输出目录
- `--sample-stocks 1000`: 目录型源采样股数（默认 1000）
- `--skip-per-file`: 跳过 cyq_chips/history_1min（加速）
- `--skip-factors`: 跳过因子（加速）

### 输出文件

| 文件 | 说明 |
|---|---|
| `field_date_matrix.json` | ★ 核心: field×date 哈希矩阵。每行一个字段(如 `daily.open`), 每列一个日期(如 `20260731`), 单元格=MD5 |
| `summary.json` | 按来源分类统计 |
| `raw_all.json` | 完整原始数据(含 row_hash) |

### 哈希算法

1. 同日期内按 sort_cols 稳定排序 (保证读序无关)
2. 逐字段取列值列表, NaN/None/NaT 归一化为 null
3. `json.dumps(arr, ensure_ascii=False, default=str)` 序列化
4. `md5(utf8_bytes)`

同一份数据无论读序/读库方式 (pandas/pyarrow), 哈希必一致。

### 解读快照

- `margin_detail.*` → 0731 列为 `·` 表示 T+1 滞后 (预期,非错误)
- `target.label_ret_*` → 滞后于因子日期 (label 需要未来数据, 天然滞后)
- `cyq_chips.stock_code` / `history_1min.stock_code` → 跨日恒定 (固定采样股)
- 对比两个快照目录的同日期同字段哈希 → 判断数据是否被修订

---

## 路径 B: 日常变更检测 (hash_analyze.py)

### 执行命令

```bash
cd /autodl-fs/data/lingqiData
/autodl-fs/data/miniconda3/bin/python3 debug/hash_analyze.py --report
```

- 默认回测窗口 = 最近 **3 个交易日** (`--days N` 可调)
- 首次运行或哈希算法升级后: `--init-run` 强制重建基线
- 回测历史某天: `--run-date YYYY-MM-DD`

### 解读结果

| 信号 | 含义 | 处理 |
|------|------|------|
| "未检测到任何变化" | 历史数据稳定 | 汇报结论,可放心继续 pipeline |
| `modified` 记录 | 某源某日期某字段哈希变了 → 数据被修订/污染 | 汇报 (source, date, field), 引导深挖 |
| `field_added` / `field_removed` | 上游改了 schema | 结构性变更,单独提示 |
| `date_removed` | 数据回退 | 立即告警 |
| 窗口内某日期"无数据" | 预期滞后 (margin/1分钟源 T-1) | 已标注,非污染 |

### 报告与账本

- 报告: `debug/hash_analyze/reports/run-<run_id>.md`
- 账本: `debug/hash_analyze/ledger/source/{源名}.json`
- 变更证据: `debug/hash_analyze/changes.json` (jsonl 追加)

⚠ **哈希算法与 snapshot_v5.py 不同, 两者哈希值不可互相套用。**

---

## 日常操作流程

```
每日 pipeline 跑完后:

  ┌─ 路径 A (快照): python debug/snapshot_v5.py --out debug/$(date +%Y%m%d)
  │  → 保存当日完整指纹, 供后续对比
  │
  └─ 路径 B (比对): python debug/hash_analyze.py --report
     → 检测历史数据是否被今日 pipeline 污染
     → 查看报告, 汇报结论
```

## 项目文件索引

```
debug/
├── snapshot_v5.py          ★ 字段级快照 v5 (当前推荐)
├── hash_analyze.py         ★ 字段级账本比对 (日常监控)
├── snapshot.py             历史快照 v3 (仅供参考)
├── compare.py              历史对比工具 (仅供参考)
├── README.md               项目说明
├── 20260802/               最新快照 (v5, 630 fields × 10d)
└── hash_analyze/           账本 & 报告
    ├── ledger/source/      各源账本 JSON
    ├── changes.json        变更证据
    └── reports/            Markdown 报告
```
