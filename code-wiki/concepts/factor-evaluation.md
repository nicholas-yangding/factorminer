# Factor Evaluation (因子评估)

## 概述
因子评估是多阶段管道，判断候选因子是否值得加入因子库。评估考虑预测能力（IC/ICIR）和正交性。

## 评估指标

### IC (Information Coefficient)
因子与未来收益的 Spearman 相关系数：
```
IC = corr(factor_values, future_returns)
```

### ICIR (IC Irregularity)
IC 均值除以标准差：
```
ICIR = mean(IC) / std(IC)
```

### IC 胜率
IC > 0 的时间步占比：
```
win_rate = count(IC > 0) / count(total)
```

---

## IC Mean vs IC Abs Mean 使用指南

### 核心区别

| 指标 | 含义 | 告诉你什么 |
|------|------|-----------|
| **IC Mean** | IC 时间序列的均值（带符号） | 因子的**方向**和**一致性** |
| **IC Abs Mean** | IC 时间序列的绝对值均值 | 因子的**预测强度**（不看方向） |

### 公式

```python
IC Mean     = mean(IC_t)          # 可正可负
IC Abs Mean = mean(|IC_t|)         # 始终 >= 0
```

### 场景解读

#### 1. IC Mean ≈ 0，但 IC Abs Mean 很高
```
结构突破: IC Mean = +0.001, IC Abs Mean = 0.54
```
**解读：** 相关性很强（0.54），但方向不稳定 — 有时正相关有时负相关  
**结论：** 因子有效，但不能单向使用，需要配合其他因子或风控

#### 2. IC Mean > 0，且 IC Abs Mean 也很高
```
吸筹强度: IC Mean = +0.166, IC Abs Mean = 0.43
```
**解读：** 一致性强（43% 的日子正相关），且整体方向向上  
**结论：** 可直接使用，IC Mean 告诉你做多方向

#### 3. IC Mean 接近 0，IC Abs Mean 也很低
**解读：** 因子无效，预测能力差

### 实战阈值

| IC Mean | IC Abs Mean | 结论 |
|---------|-------------|------|
| > 0.02 | > 0.03 | ✅ 有效，做多 |
| < -0.02 | > 0.03 | ✅ 有效，做空 |
| ≈ 0 | > 0.03 | ⚠️ 有效但方向不稳，组合使用 |
| ≈ 0 | < 0.03 | ❌ 无效 |

### ICIR 的作用

```
ICIR = IC Mean / IC Std
```

- ICIR > 0.5 表示因子收益稳定
- ICIR < 0.5 但 IC Mean 和 IC Abs Mean 都高，说明因子方向稳定但波动大

### 示例

```python
from factorminer.evaluation.metrics import compute_ic, compute_ic_mean, compute_ic_abs_mean

ic_series = compute_ic(signals, returns)

ic_mean     = compute_ic_mean(ic_series)      # 决定方向: 做多 or 做空
ic_abs_mean = compute_ic_abs_mean(ic_series)  # 决定强度: > 0.03 才算有效
```

## 多阶段评估管道

```
候选因子
    |
    v
[Stage 1] Fast IC Screening
    | 快速计算 IC，筛选 > threshold
    v
[Stage 2] Library Correlation
    | 与库中因子相关性 < θ
    v
[Stage 2.5] Replacement Check
    | 相关但 IC 更高，可替换
    v
[Stage 3] Intra-batch Dedup
    | batch 内 pairwise rho < θ
    v
[Stage 4] Full Validation
    | 完整数据集 + 多时间框架
    |
    v
Admission?
```

## Admission Criteria (准入标准)

必须满足：
1. `IC > ic_threshold` (默认 0.02)
2. `ICIR > icir_threshold` (默认 0.5)
3. 与库中因子正交 (rho < correlation_threshold)

可选：
- Research admission: 多时间框架评分
- Causal admission: Granger 因果性检验
- Regime admission: 市场状态条件评估

## 信号失败处理

| 策略 | 行为 |
|------|------|
| `reject` | 直接拒绝 |
| `synthetic` | 生成合成信号继续 |
| `raise` | 抛出异常 |

## 来源
- `factorminer/evaluation/pipeline.py`
- `factorminer/evaluation/metrics.py`
- `factorminer/evaluation/admission.py`
