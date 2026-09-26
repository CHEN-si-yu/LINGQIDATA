# v8_shadow — V8 原版模型影子打分单元 (仅观察, 2026-09-07 建)

## 用途

对 **V8 原版模型** (Model/V8 的 9/2 checkpoint, 训练截止 20250809) 做与 best
完全同口径的**每日实时推演** (20260901 起逐日打分), 与 best (0901 切点全量
重训版) 的打分并列对照。

背景: 同一冻结窗口 20250901~20260901 上, V8 (样本外) 慢腿 +79.1%, 而 best
0901 切点模型 (该窗口在训练集内, 样本内) 仅 -6.5% —— 见 Model/best/README_best
§六与当日实验记录。该对照不说明"谁更好", 只用于**前瞻跟踪**两源在实盘期的分歧。

## 组成

- `model.py` / `analysis.py`: best 实时窗口流水线的拷贝 (root_path 改指本目录),
  与 V8 目录的冻结窗口版本不同, 窗口 = 20260901 起随因子数据滚动。
- `model_train` / `model_test` → 符号链接指向 `../../V8/` 对应目录
  (V8 原版 9/2 checkpoint 与 feature_map, 勿替换)。
- 产物: `model_pred/2026q3/2026xxxx.pkl` (逐日) + `all_zscore_score.fea`;
  `model_pic/output.md` (推演报告)。

## 运行

```bash
cd /autodl-fs/data/lingqiData/Model/best/v8_shadow
python3 analysis.py        # 落后即增量推演最新因子日 (每次只算新日期)
```

正常无需手动: `../daily_ops.py` 已将本单元纳入自动刷新 (UNITS 键 `v8_shadow`),
并在输出尾部打印「V8 影子对照」(两源最新因子日一致性 + 慢腿 Top2 候选与重合)。

## 铁律

- 本单元输出**仅作对照观察**: 结论、操作建议、纸面修正一律以 best (慢腿)
  + V11 (快腿) 为准, 不得因 V8 影子排名不同而改动 best 计划。
- 本目录文件是 best 流水线的拷贝 + V8 checkpoint 的链接, 两者任一更新后
  本单元需同步 (root_path / 链接指向); 不要在本目录里改 V8 的 checkpoint。
