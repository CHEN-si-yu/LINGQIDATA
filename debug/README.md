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
| `raw/*.parquet` | daily / daily_adj / finance / cyq_perf / main_fund_flow / margin_detail（各 10 日切片）+ calendar / stock_list（整体拷贝） |
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
- **2026-08-06（当日记录）**：
  - 数据刷新至 08-06（daily/daily_adj/finance 等；margin_detail 最新 08-05，T+1 正常）。
  - 因子 point-in-time 修复（commit 3c1a990）：`_fill_source_gaps` 改为按统一交易日历
    前向扩展、margin T+1 对齐等；重建时经 factor_loader 禁用 6 个非 point-in-time 因子
    （industry_relative_momentum_20 / pb_industry_adjusted / ps_ttm_sector_neutral /
    sector_amount_momentum_5d / sector_amount_rank / sector_mv_rank）。22:12–22:43 全量
    重建完成（116 组 ok、686 个 .fea、1195s）。重建前 root 用户 20:50 留了一份
    `20260805_copy/`（data + factors 快照）作为对比/备份。
  - 构建 20260806 切片：6043 任务全 ok、0 err（窗口 20260724~20260806，10 交易日，
    全量 1782 只，因子 686 个）。
  - 对比 20260805 vs 20260806（报告见 `20260806/compare_report_20260805_20260806.*`）：
    **686 个因子在 9 个重叠交易日上零变动** —— point-in-time 修复重建对历史日期无回归，
    仅新增 08-06 行。raw 侧变动均为正常上游修订：4 只更名（001232/002214/300311/920038.BJ，
    daily.stock_name）、~10 只复权价修订（688501.SH 等 08-05 除权，daily_adj OHLC）、
    finance.pe_ttm_percentile 随新数据全市场重算（08-05 共 2069 格）、ths_daily 08-05
    换手率等 355 行修订、stock_list +1 新股、calendar +1 开市日；cyq_chips /
    history_1min / indicator_1min / daily_dump_1min / margin_detail 全零变动。
- **2026-08-07（当日记录）**：
  - 数据刷新至 08-07（daily/daily_adj/finance 等；margin_detail 最新 08-06，T+1 正常）。
    因子 21:16 全量重建完成（686 个 .fea，含 08-07 行）。
  - 构建 20260807 切片：6043 任务全 ok、0 err（窗口 20260727~20260807，全量 1782 只）。
  - 对比 20260806 vs 20260807（报告见 `20260807/compare_report_20260806_20260807.*`）：
    **686 个因子在 9 个重叠交易日上零变动**。raw 侧正常修订：603468.SH（08-06 上市新股）
    名称空→N津富；14 只除权复权重算（000776/600060/600104/601336 等 08-07 除权，
    daily_adj 08-03~08-06 按比例 0.965~0.991 整体下调）；finance.pe_ttm_percentile 08-06
    回填（3916/5533 只旧值为 0 占位→真实分位，pe_ttm 本身零变动，与 08-05 模式相同）；
    ths_daily / margin_detail / cyq_perf / main_fund_flow / cyq_chips / history_1min /
    indicator_1min / daily_dump_1min / calendar 全零变动。
  - ⚠️ 上游两处结构性问题（不影响因子值，建议反馈上游）：
    ① stock_list.parquet 875 只代码重复各 ×2（行完全相同）：537 只在市 + 338 北交所
    退市股，昨日无此问题 → 上游生成去重缺陷；
    ② ths_constituent_stocks.parquet 从单指数（700001.TI 全A 5000 只）扩展为全部
    1664 个同花顺指数（356208 行），其中 700001.TI 新增 537 只、0 移除（与 ① 中重复的
    在市代码 507/537 重合 → 同源：~537 只新纳入全A）。sector 因子 2026-07-31 起已不
    依赖该表，故因子零变动不受影响；但切片体积涨 ~70 倍，下游 join 需按 index_code 过滤，
    stock_list join 需先去重。
- **2026-08-08**：ths_* 数据全线清除。data/ 下 ths_daily / ths_constituent_stocks /
  ths_sector_categories 三个原始文件删除，incremental.py 删除对应爬取任务、
  incremental_core.py 删除 canary 探测端点（此后不再爬取更新）；本目录三个历史切片
  （20260805 / 20260806 / 20260807）data10d/raw/ 中的 ths 拷贝一并删除，
  build_debug_data_10d.py 不再切片 ths。历史对比报告中的 ths 记录保留（属当时的事实记录）。
- **2026-08-08（当日记录）**：因子迭代更新（686 → 725 个 .fea，08-08 15:21~15:48 全量
  重建，最新行 20260807），旧切片的对比已失去意义。清除三份历史切片
  （20260805 / 20260806 / 20260807，含各自对比报告，共 ~3.3G / 18160 文件），
  重建 20260807 切片：6079 任务全 ok、0 err（窗口 20260727~20260807，10 交易日，
  全量 1782 只，因子 725 个），新切片作为后续对比的基线。
- **2026-08-09（当日记录）**：因子代码重构完成（builder.py / factor_loader.py /
  chip_deep_extra / indicator_minute(_extra) / intraday(_extra) / price_deep 等修改，
  新增 coupling_daily_extra3 / coupling_time / strategy_daily 模块），08-09 10:15~10:32
  因子全量重建（725 → 745 个 .fea，最新行 20260807，数据时间节点未刷新仍为 08-07），
  旧切片的对比已失去意义。清除 debug/20260807 旧快照（725 因子版，1.1G / 6091 文件）
  及根目录三份失效对比日志（compare_20260804_20260805 / _20260805_20260806 /
  _20260806_20260807，其对比对象均已删除）。
  重建 20260807 切片：6099 任务全 ok、0 err（窗口 20260727~20260807，10 交易日，
  全量 1782 只，因子 745 个），切片内 745 个因子均为 10 行全窗口覆盖，
  新切片作为重构后对比的基线。
- **2026-08-10（当日记录）**：
  - 数据刷新至 08-10（daily/daily_adj/finance/cyq_perf/main_fund_flow；margin_detail
    最新 08-07，T+1 正常）。因子 20:30~21:04 重建完成（395+141 个需更新因子 ok、0 错误，
    745 个 .fea 全部覆盖 08-10 行）。
  - 构建 20260810 切片：6099 任务全 ok、0 err（窗口 20260728~20260810，10 交易日，
    全量 1782 只，因子 745 个）。
  - 对比 20260807 vs 20260810（报告见 `20260810/compare_report_20260807_20260810.*`）：
    **745 个因子在 9 个重叠交易日上零变动**。raw 侧变动均为正常上游修订：4 只更名
    （300125.SZ *ST聆达→聆达股份 ST 摘帽、301677.SZ C欣兴工具→欣兴工具、
    301707.SZ 空→展芯股份 08-07 上市新股、920038.BJ N森合→森合高科，daily.stock_name）；
    7 只除权复权重算（000039/000429/300692/300750/601298/603093/688037 08-07 除权，
    daily_adj 08-04~08-07 按比例 0.687~0.999 整体下调，603093.SH 0.687 为高比例送转）；
    finance.pe_ttm_percentile 08-07 回填（5089 格 0 占位→真实分位，pe_ttm 本身零变动，
    与 08-05/08-06 模式相同）；stock_list +1 新股（301707.SZ，08-07 记录中的重复代码
    缺陷仍在：875→876 只）；calendar +3 未来开市日（09-07/08/09）；
    cyq_perf / main_fund_flow / margin_detail / cyq_chips / history_1min /
    indicator_1min / daily_dump_1min 全零变动。
