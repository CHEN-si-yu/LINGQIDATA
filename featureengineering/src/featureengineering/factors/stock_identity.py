"""Stock identity characteristic factors — 已全部删除（2026-08-05）。

原注册的因子：
  - hs_connect_flow_sensitivity   港股通资金敏感度（依赖 stock_list.is_hs）
  - owner_type_turnover_div       实控人类型换手偏离（依赖 stock_list.act_ent_type）

删除原因：两个因子依赖 stock_list.parquet 的静态属性（is_hs / act_ent_type /
上市状态按当前快照回填全部历史日期，文件自带"时点回溯"审计声明）。2026-08-05
切片对比（20260804 vs 20260805）发现这两个因子在重叠 9 个交易日全日期段出现差异，
根因是 stock_list 静态属性在两次构建间被修订（600530 的 is_hs、000550 的
act_ent_type 变更、+1 只新股）——属性一改，因子全历史截面跟着变，属于"用今日
信息回填历史"的时点风险类因子。按约定直接移除注册，不再重建。
"""
