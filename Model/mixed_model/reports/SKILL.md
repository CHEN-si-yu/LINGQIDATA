# 每日模型预测深度分析技能

> **技能目标**: 对集成模型每日 Top 3 推荐进行因子级深度拆解，并从策略持有周期出发给出适配性判断。  
> **适用场景**: 每日运行 `ensemble_analysis.ipynb` 获得 Top 10 预测后触发。  
> **输出产物**: `reports/{YYYYMMDD}.md` — 当日深度分析报告。

---

## 一、分析框架总览

```
输入：当日 Top 3 股票代码
  │
  ├─ Step 1  模型共识度 ── 四个模型各自打分，计算 σ
  ├─ Step 2  因子画像   ── 极端因子提取 + 归类 + 方向解读
  ├─ Step 3  信号衰减分类 ── 快/中/慢三档归类
  ├─ Step 4  历史轨迹   ── 最近 20 日 z-score 稳定性
  │
  ▼
输出：策略适配排序（可能与模型原始排名不同）
```

### 核心洞察

> **模型的排名回答"谁的预期收益最高"，策略需要回答"谁明天最可能涨"。两个问题有本质区别，需要通过因子衰减速度来桥接。**

---

## 二、Step-by-Step 执行指南

### Step 1：提取模型共识度

**目的**: 判断高排名是多个模型的共识，还是单一模型的极端意见。

**代码模板**:

```python
import pandas as pd, numpy as np, os

MODEL_BASE = '/root/autodl-fs/lingqiData/Model'
models = ['V2.1', 'V3.2', 'V3.7', 'V1.2']  # 当前集成四人组
date = 'YYYYMMDD'  # 替换为实际预测日期

scores = {}
for m in models:
    df = pd.read_feather(os.path.join(MODEL_BASE, m, 'model_res', 'ALL_zscore_score.fea'))
    if 'date' in df.columns: df = df.set_index('date')
    elif 'index' in df.columns: df = df.set_index('index')
    # 取最新可用日期
    latest = sorted(df.index)[-1]
    s = df.loc[latest]
    z = (s - s.mean()) / s.std()
    scores[m] = z

ensemble_z = sum(scores[m] for m in models) / 4

# 对目标股票
for code in ['target1', 'target2', 'target3']:
    model_zs = {m: scores[m].get(code, np.nan) for m in models}
    sigma = np.std(list(model_zs.values()))
    print(f"{code}: z={ensemble_z[code]:+.4f}, σ={sigma:.2f}")
    for m in models:
        print(f"  {m}: {model_zs[m]:+.2f}")
```

**解读标准**:

| σ 范围 | 含义 | 策略含义 |
|--------|------|----------|
| < 0.5 | 高度共识 | 信号可靠，但可能弹性不足 |
| 0.5-0.8 | 中等分歧 | 方向一致，程度有差异 |
| > 0.8 | 显著分歧 | ⚠️ 信号可能依赖单一模型，需谨慎 |

---

### Step 2：因子画像 — 极端因子提取与归类

**目的**: 理解模型到底"看到了什么"，找出驱动高分的核心因子。

**代码模板**:

```python
# 加载当日因子数据
fac = pd.read_feather('/root/autodl-fs/lingqiData/trainingdata/fac_all.fea')
fac['date'] = fac['date'].astype(str)
all_stocks = fac[fac['date'] == date]

factor_cols = [c for c in all_stocks.columns if c not in ['date', 'Code']]
factor_means = all_stocks[factor_cols].mean()
factor_stds = all_stocks[factor_cols].std()
valid = factor_stds[factor_stds > 0.001].index  # 排除无方差因子

# 对目标股票计算截面 z-score
stock = all_stocks[all_stocks['Code'] == code]
stock_vals = stock[valid].iloc[0]
z_scores = ((stock_vals - factor_means[valid]) / factor_stds[valid]).sort_values(ascending=False)

# 极端偏高 (z > 1.5)
pos_extreme = [(col, z_scores[col]) for col in z_scores.index if z_scores[col] > 1.5]

# 极端偏低 (z < -1.5)  
neg_extreme = [(col, z_scores[col]) for col in z_scores.index if z_scores[col] < -1.5]
```

**因子归类函数**:

```python
# 因子关键词 → 类别映射（持续维护）
FACTOR_CATEGORIES = {
    '动量/技术': ['rsi', 'momentum', 'breakout', 'gap', 'obv', 'ret_autocorr',
                'consecutive_up', 'consecutive_down', 'close_to_high', 'limit_down',
                'chip_concentration', 'chip_winner', 'turnover_factor', 'ts_mom',
                'bollinger', 'distance_from_ma', 'overnight', 'volume_price',
                'parkinson', 'hl_range'],
    '资金流': ['mf_big_order', 'order_size', 'margin_flow', 'fundflow',
              'advance_receipts', 'sector_fund_flow', 'margin_buy',
              'sector_advance_decline', 'margin_sentiment', 'margin_smart'],
    '价值': ['bp', 'bps_rank', 'margin_of_safety', 'retained_earnings',
             'undistributed_profit', 'value_momentum', 'value_reversal',
             'growth_at_reasonable', 'dupont', 'factor_relative_value',
             'ev_to', 'fcf_yield'],
    '质量/盈利': ['roe', 'roa', 'earnings_quality', 'earnings_momentum',
                'gross_margin', 'profit_volatility', 'revenue_stability',
                'netprofit', 'ocf_to', 'ocfps', 'or_yoy', 'tr_yoy',
                'accruals', 'quality_growth', 'interest_income'],
    '财务健康': ['net_debt', 'financing_dependency', 'pledge', 'cash_ratio',
               'quick_ratio', 'current_ratio', 'working_capital', 'cf_short',
               'cash_adequacy', 'contract_liab', 'taxes_payable', 'payables',
               'capital_reserve', 'other_payables', 'dividend_payable',
               'tax_to_ebt', 'tax_rate', 'rd_intensity'],
}

def categorize_factor(name):
    for cat, keywords in FACTOR_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in name.lower():
                return cat
    return '其他'
```

**人工判断规则**:
- 某些"偏低"信号实为正面（如 `net_debt` 低 = 债务轻、`pledge` 低 = 无爆仓风险）
- 某个极端值特别高（|z|>3）的因子需特别标注，往往是该股票最显著的特征
- 需要区分"真实风险"（如 `or_yoy` 为负 = 营收下滑）和"假风险"（如 `net_debt` 为负 = 好事）

---

### Step 3：信号衰减速度分类（★ 核心步骤 ★）

**目的**: 这是分析框架的发动机。不同类型的因子，其对未来收益的预测能力衰减速度完全不同。隔日策略只能依赖快速衰减信号。

**衰减分类标准**:

| 类别 | 衰减周期 | 隔日有效性 | 典型因子族 |
|------|:--------:|:--------:|-----------|
| 🟢 **快速** | 1-5 天 | ⭐⭐⭐ | 动量、技术指标、资金流、大单、筹码变动 |
| 🟡 **中速** | 周-月 | ⭐⭐ | 盈利质量、ROE、现金流比率、利润率 |
| 🔴 **慢速** | 月-年 | ⭐ | 深度价值（BP、PE）、留存收益、安全边际、财务杠杆 |

**代码模板**:

```python
FAST_KEYWORDS = ['momentum', 'breakout', 'rsi', 'gap', 'obv', 'consecutive',
                 'fund_flow', 'big_order', 'order_size', 'margin_flow',
                 'chip_concentration', 'chip_winner', 'ret_autocorr',
                 'turnover_factor', 'ts_mom', 'bollinger', 'overnight',
                 'limit_down_open', 'limit_down_pct',
                 'margin_sentiment', 'margin_smart', 'margin_buy',
                 'sector_rotation', 'sector_turnover']

SLOW_KEYWORDS = ['bp', 'bps_rank', 'margin_of_safety', 'retained_earnings',
                 'undistributed_profit', 'value_momentum', 'value_reversal',
                 'growth_at_reasonable', 'dupont', 'ev_to',
                 'net_debt', 'financing_dependency', 'pledge',
                 'capital_reserve', 'fcf_yield', 'dividend']

def classify_decay(factor_name):
    for kw in FAST_KEYWORDS:
        if kw.lower() in factor_name.lower():
            return 'fast'
    for kw in SLOW_KEYWORDS:
        if kw.lower() in factor_name.lower():
            return 'slow'
    return 'medium'

# 统计信号构成
fast_count = sum(1 for col, _ in pos_extreme if classify_decay(col) == 'fast')
medium_count = sum(1 for col, _ in pos_extreme if classify_decay(col) == 'medium')
slow_count = sum(1 for col, _ in pos_extreme if classify_decay(col) == 'slow')
total = fast_count + medium_count + slow_count
print(f"快速:{fast_count}({100*fast_count/total:.0f}%) "
      f"中速:{medium_count}({100*medium_count/total:.0f}%) "
      f"慢速:{slow_count}({100*slow_count/total:.0f}%)")
```

**判断阈值**:
- 快速信号 > 50% → 适合隔日策略
- 慢速信号 > 50% → 不适合隔日策略（信号正确但时间尺子不对）
- 中速信号 > 50% → 边际适用，需结合模型共识度判断

---

### Step 4：历史轨迹验证

**目的**: 检查信号是突然飙升还是持续稳定，判断是否属于噪音。

**代码模板**:

```python
# 找共同日期
common_dates = sorted(set(scores[models[0]].index) & 
                      set(scores[models[1]].index) &
                      set(scores[models[2]].index) & 
                      set(scores[models[3]].index))
recent = common_dates[-20:]

# 逐日计算集成 z-score
for date in recent:
    zs = []
    for m in models:
        if date in scores[m].index and code in scores[m].columns:
            s = scores[m].loc[date]
            z = (s - s.mean()) / s.std()
            zs.append(z[code])
    avg_z = np.nanmean(zs) if len(zs) >= 3 else np.nan
    print(f"{date}: z={avg_z:+.4f}")
```

**判断标准**:
- 20 日内从未转负 + 波动小 → 高稳定性信号
- 近期突然飙升 + 历史有负值 → 信号可能不稳定
- 始终在低位 + 近期跳升 → 需关注是否因子突变

---

## 三、最终判断框架

### 3.1 策略适配矩阵

综合以上四步，将股票映射到以下矩阵：

```
                快速信号占比
                高 (>50%)    低 (<50%)
              ┌──────────┬──────────┐
模型  高(<0.5) │ 🥇 最佳   │ 🥈 共识  │
共识           │ 趋势+共识  │ 质量+共识 │
度   低(>0.8) │ 🥈 博弈   │ 🥉 不适  │
              │ 趋势+分歧  │ 价值+分歧 │
              └──────────┴──────────┘
```

### 3.2 最终推荐模板

```markdown
## 四、最终判断：[推荐股票名]

### 策略适配排序

| 排名 | 股票 | 适配逻辑 | 核心风险 |
|:----:|------|----------|----------|
| 🥇 | [最佳适配] | 快速信号占比 X%，σ=Y | [关键风险] |
| 🥈 | [次优] | [适配逻辑] | [关键风险] |
| 🥉 | [不适配] | [不适配原因] | [关键风险] |

### 核心逻辑

[用通俗语言解释为什么模型排名和策略排名可能不同]

持有周期建议：
- 1 天 → [推荐]
- 1 周 → [推荐]
- 1 月+ → [推荐]
```

---

## 四、报告生成清单

每日报告应包含以下章节：

- [ ] **一、预测结果与模型共识** — Top 3 表格 + 各模型得分 + σ
- [ ] **二、逐只深挖** — 每只股票一句话总结 + 核心因子信号表
- [ ] **三、核心问题：策略约束下的重新排序** — 因子衰减分类 + 模型分歧 + 历史轨迹
- [ ] **四、最终判断** — 策略适配排序 + 持有周期建议
- [ ] **五、风险提示** — 模型漂移、单日噪音、流动性、因子盲区

---

## 五、数据路径速查

| 数据 | 路径 |
|------|------|
| 因子数据 | `/root/autodl-fs/lingqiData/trainingdata/fac_all.fea` |
| 模型得分 | `/root/autodl-fs/lingqiData/Model/{version}/model_res/ALL_zscore_score.fea` |
| 集成模型列表 | 从 `ensemble_analysis.ipynb` 的 `ENSEMBLE_MODELS` 读取 |
| 1日收益标签 | `/root/autodl-fs/lingqiData/trainingdata/label_ret_1d.fea` |
| 流动性数据 | `/root/autodl-fs/lingqiData/trainingdata/trade_amt.fea` |
| 行情数据 | `/root/autodl-fs/lingqiData/data/daily_adj.parquet` |
| 搜索脚本 | `/root/autodl-fs/lingqiData/Model/mixed_model/search_best_ensemble.py` |
| 分析Notebook | `/root/autodl-fs/lingqiData/Model/mixed_model/ensemble_analysis.ipynb` |

---

## 六、维护说明

1. **因子分类映射**（`FACTOR_CATEGORIES`、`FAST_KEYWORDS`、`SLOW_KEYWORDS`）需要随因子库更新而维护
2. **集成模型列表**随 `search_best_ensemble.py` 重新搜索后更新
3. **衰减分类标准**基于经验法则，可通过回测（不同衰减速度因子的隔日 IC）进行校准
4. 每次分析完成后将报告保存为 `reports/{YYYYMMDD}.md`

---

## 七、关键注意事项与常见陷阱（2026-06-29 更新）

### 7.1 ⚠️ V4.1 单模型得分必须归一化（★ 极易出错 ★）

**陷阱**: V4.1 模型的原始打分（如 `analysis.ipynb` Cell 8/9 中的 "+11.2466"）是**单模型原始预测值**，其量纲远大于集成模型得分（集成模型是 4 个基模型的均值）。

**正确做法**:
```python
# V4.1 单模型得分 → 跨模型横向对比
v4_score_raw = 11.2466                    # V4.1 原始输出
v4_score_normalized = v4_score_raw / 4    # → +2.81，与集成得分可比

# 对比
ensemble_score = 2.9640                   # 集成模型对同一股票的得分
# 归一化后 +2.81 vs 集成 +1.01 → V4.1 对该股有 2.8 倍的模型特异性看多
```

**教训**: 2026-06-29 报告中，柯力传感 V4.1 +11.2466 若不除以 4，会错误地认为其得分远超珍宝岛 +2.9640。实际上归一化后 +2.81 在量级上相近，但集成模型仅给柯力 +1.01（排 132 名），这才是真正的共识信号。

### 7.2 ⚡ 历史轨迹优先使用 ensemble_pred（★ 数据源优先级 ★）

**陷阱**: 各基模型（V2.1/V3.2/V3.7/V1.2）的 `model_res/ALL_zscore_score.fea` 更新可能存在严重滞后（如 2026-06-29 时仅更新至 06-09/10，滞后 13 个交易日）。使用过期数据会导致结论完全错误。

**正确做法**: **优先使用集成预测 pickle 文件作为历史轨迹数据源**:

| 优先级 | 数据源 | 路径 | 时效性 | 用途 |
|--------|--------|------|--------|------|
| 🥇 第一优先 | **集成预测 pickle** | `ensemble_pred/{组合名}/2026q2/{YYYYMMDD}.pkl` | 每日更新，与预测同步 | **历史轨迹、当前排名、截面 z-score** |
| 🥈 第二优先 | 基模型得分 | `Model/{version}/model_res/ALL_zscore_score.fea` | 滞后 1-3 周 | 基模型分歧分析（辅助验证） |
| 🥉 第三参考 | 因子数据 | `trainingdata/fac_all.fea` | 每日更新 | 因子画像、衰减分类 |

**代码模板（正确方式）**:
```python
import pickle, pandas as pd, numpy as np, os, glob

# Step 1: 找到最新 ensemble_pred
ENSEMBLE_DIR = '/root/autodl-fs/lingqiData/Model/mixed_model/ensemble_pred'
# 优先使用当前活跃组合
active_ensemble = 'V1-2_V2-1_V3-2_V3-7'  # 从 ensemble_analysis.ipynb 确认
pred_dir = os.path.join(ENSEMBLE_DIR, active_ensemble, '2026q2')
files = sorted(glob.glob(os.path.join(pred_dir, '*.pkl')))

# Step 2: 加载最新预测
with open(files[-1], 'rb') as f:
    latest_scores = pickle.load(f)           # pd.Series: index=Code, values=score
latest_date = os.path.basename(files[-1]).replace('.pkl', '')

# Step 3: 计算截面 z-score
z_scores = (latest_scores - latest_scores.mean()) / latest_scores.std()
top10 = z_scores.sort_values(ascending=False).head(10)

# Step 4: 加载完整历史轨迹
history = {}
for f in files:
    date = os.path.basename(f).replace('.pkl', '')
    with open(f, 'rb') as fh:
        history[date] = pickle.load(fh)

# 对目标股票追踪历史
for code in ['603567', '600510']:
    print(f"\n{code} 历史轨迹:")
    for date in sorted(history.keys()):
        s = history[date]
        print(f"  {date}: {s[code]:+.4f}")
```

### 7.3 多 ensemble 交叉验证

当前有三个集成模型变体（排列组合不同），分析时应交叉验证：

| 组合名 | 路径 | 说明 |
|--------|------|------|
| V1-2_V2-1_V3-2_V3-7 | `ensemble_pred/V1-2_V2-1_V3-2_V3-7/` | 当前活跃（V1.2 替代 V1.0） |
| V1-0_V2-1_V3-2_V3-7 | `ensemble_pred/V1-0_V2-1_V3-2_V3-7/` | 前一版本（V1.0） |
| V2-1_V3-2_V3-7_V1-2 | `ensemble_pred/V2-1_V3-2_V3-7_V1-2/` | 不同排列顺序 |

**规则**: 如果一只股票在至少两个 ensemble 变体中同时排名前 10，信号可靠性大幅提升。如果仅在一个变体中排名靠前，需标注为"模型排列敏感"。

### 7.4 负向因子数量/比例也是信号

不要只看正向极端因子。负向因子远多于正向（如珍宝岛 201 vs 66 = 3:1）意味着：
- 集成模型的高得分来自少数因子的极端权重
- 如果这些关键因子在下一天失效，信号可能急转直下
- 需要特别关注正向因子中的"假正向"（如高质押、高现金周期）和负向因子中的"假负向"（如低商誉、低负债）
