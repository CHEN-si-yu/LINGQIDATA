# V20 — 冠军系统统一版本 (冻结集成封装)

**V20 = V11(多目标族) + V13(顶部3d族) 冻结集成**, 将全项目最优模型与策略收敛为单个
规范版本, 统一"训练 / 启动 / 回测 / 出结果"入口。V20 自身**不训练新参数**。

## 冠军系统 (与 Model/FINAL_REPORT.md 完全一致)

```
打分: ens_w2 = z(mixA) + 2·z(V13组件)
  mixA    = Σ_{8折} z( 1.0·r1 + 0.25·r3 + 2.0·top )   # V11 家族
  V13组件 = Σ_{8折} z( 1.0·r1 + 1.0·top )             # V13 家族
策略: 因子日收盘打分 → Top1 次日开盘买入(整手) → 持有5个交易日收盘卖出;
      持仓期收盘 ≤ 买入价 -8% → 次日开盘止损
```

Test 集 (20250901~20260901, 含成本): 净累计 **+365.2%**, MaxDD -16.4%,
Sharpe 2.27 [0.86, 4.06], 分半 H1/H2 = +154.9%/+58.7%, 49 笔交易。

## 四个基本文件

| 文件 | 角色 | 用法 |
|---|---|---|
| `model.py` | 冠军模型定义 + 引擎库 (公式/组装/刷新/回测/出榜) | 被其余文件 import, 也可 `python3 model.py` 自检 |
| `run.py` | **启动**: 刷新增量推演 + 组装打分 + 落盘缓存 | `python run.py [--update] [--no-refresh]` |
| `train.sh` | **训练**(冻结校验): 组件/数据健康检查 | `bash train.sh [--full]` |
| `analysis.py` | **最终结果入口**: 排行榜 + 策略指令 (+ 回测) | `python analysis.py [选项]` |

## 快速开始 (日常只需一条命令)

```bash
cd /autodl-fs/data/lingqiData/Model/V20
python3 analysis.py                 # → 最新因子日 Top10 排行榜 + Top1 策略指令
python3 analysis.py --backtest      # → 冠军协议全窗口回测 (含分半/IC)
bash train.sh                       # → 启动前健康检查 (验证组件齐备)
python3 run.py                      # → 只刷新推演+组装打分 (供脚本/流程调用)
```

> 首次或数据更新后: 因子数据(fac_all.fea)出现新交易日时, analysis.py 会自动在
> V11/V13 目录运行各自 analysis.py 增量推演 (重推最近 10 日, 需 GPU, 约数分钟),
> 无需手动干预。已有 heads 覆盖最新日时秒级出榜。

## 关键选项 (analysis.py)

- `--top N` 榜单长度 (默认 10)
- `--days N` 最近 N 个因子日排行榜 (V9 output.md 风格)
- `--date YYYYMMDD` 指定因子日出榜
- `--backtest` / `--no-split` 冠军回测及分半开关
- `--update` 强制重跑 V11/V13 增量推演
- `--no-md` 不写 `model_pic/output.md`

## 依赖与说明

- 依赖: V11/V13 目录 (checkpoint + heads, 只读引用)、`trainingdata/`、`data/` 均在原位;
  V20 自身产物只写入 `Model/V20/` 下 (logs/model_pred/model_pic)。
- heads 落后时调用 V11/V13 的 `analysis.py` 会**更新 V11/V13 自己的 model_pred**——
  与既有每日流水线 `Model/V11/run_daily_pipeline.sh` 行为一致, 非破坏性。
- 回测口径 = eval_v17 收盘卖引擎 (卖出价取收盘, 含成本/ST排除/一字板过滤)。
- 结构已收敛 (FINAL_REPORT): 勿再叠加第三组件/种子/平滑。
