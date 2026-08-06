# Debug：10 交易日全量数据切片（找变动因子）

## 设计思路（2026-08-05 起，替代哈希快照方案）

**不再存哈希**——直接把最近 10 个交易日的**原始数据 + 因子的实际数据**拷一份到
`debug/<锚点日期>/data10d/`（2026-08-06 起只保留 raw + factors，不再切 targets/trainingdata）。
相邻两日的切片同构，逐文件对比即可定位变动因子，
且能直接看到"变成了什么"，无需回原目录。

## 正确执行顺序

```
① 上游数据刷新（daily / finance / cyq_chips / margin_detail 等落盘）
       ↓
② 因子工程全量重建（featureengineering/build_factors.py）
       ↓
③ python debug/build_debug_data_10d.py --anchor $(date +%Y%m%d)
   → 生成 debug/<日期>/data10d/
```

## 切片脚本 build_debug_data_10d.py

- 窗口：从 calendar.parquet 取截至锚点日期（默认 daily 最新交易日）的最近 **10 个交易日**
- 按股票存储的目录（cyq_chips / history_1min / indicator_1min）**全量拷贝全部股票**
  （2026-08-05 起不再抽样：因子宽表覆盖全部股票列，抽样 1000 只会让其余股票的原始
  数据缺失、无法定位其差异；清单仍存 `data10d/sample_1000.json`，内容为全量）
- 多进程并行（--jobs 24），约 5400 个任务（全量 1782 只 × 3 目录 + 长表 + 因子），20~30 分钟完成

**输出结构** `debug/<日期>/data10d/`：

| 路径 | 内容 |
|------|------|
| `raw/*.parquet` | daily / daily_adj / finance / cyq_perf / main_fund_flow / margin_detail / ths_daily（各 10 日切片）+ calendar / stock_list / ths 成分 / ths 板块（整体拷贝） |
| `raw/cyq_chips/` `history_1min/` `indicator_1min/` | 全部 1782 只股票 × 10 日 |
| `raw/daily_dump_1min/` | 窗口内 10 个日期文件 |
| `factors/*.fea` | 全量因子宽表（10 行 × 1782 列） |
| `sample_1000.json` / `summary.json` | 股票清单（全量）/ 切片汇总 |

## 浏览与对比

- **浏览**：`debug/Watch.ipynb`（当前快照的快速浏览）
- **对比找变动**：两份切片逐文件 diff（同构布局），如：

```python
# 示例：比较某个因子两天切片
import pyarrow.ipc as ipc
a = ipc.open_file("debug/20260804/data10d/factors/adx_14.fea").read_all().to_pandas()
b = ipc.open_file("debug/20260805/data10d/factors/adx_14.fea").read_all().to_pandas()
diff = (a != b).sum()   # 非零列即变动的因子日期×股票
```

## 重要约定

### 1. margin_detail T+1 延迟

margin_detail（融资融券明细）T 日数据在 **T+1 日收盘后**才落盘。切片中其最新日期
天然比 daily 少一个交易日（如 20260804 切片的 margin_detail 最新为 20260803），**不是缺失**。
讨论 T 日 margin 数据/因子，一律指 margin_detail 中 T 之前最近一个交易日（T=08-04 → 08-03；
T=08-03 → 07-31）。

### 2. 尾日扩展缺陷（因子构建侧）

`featureengineering/src/featureengineering/dataset.py::_fill_source_gaps` 前向扩展尾日时
曾用「全历史代码集合」创建扩展行，把已退出 margin 面板的股票数年前的最后值拉进尾日。
已修复为只向「最后数据日期实际存在的代码」扩展。若 margin 因子尾日非空数 > 1742，优先查此逻辑。

### 3. daily 的 ah_amount / ah_vol

2026-07-21 起 daily 新增 AH 股成交额/成交量两列，是上游补充字段，非修订/污染。

---

## 历史记录

- **2026-08-04**：旧哈希方案（snapshot_v5.py / hash_analyze.py 及全部哈希产物）已删除，
  原因：直接拷贝原件后哈希失去意义。曾用哈希方案发现 08-03 的 31 个因子变化是
  「因子重建滞后于数据刷新」而非数据污染（详见 git/聊天记录）。
- **2026-08-04**：因子全量重构（536→694 个因子），20260803 旧快照已清除。
- **2026-08-06**：切片范围精简为 raw + factors（不再切 targets/trainingdata）；重建 20260805
  时 per-stock 目录为全量 1782 只股票（此前两次构建实际是抽样 1000 只，seed=42）；
  20260804 / 20260805 旧切片（含对比报告）已清除。脚本根目录改为自动探测
  （/root/autodl-fs/lingqiData 或 /autodl-fs/data/lingqiData），root / claude 用户均可运行。
