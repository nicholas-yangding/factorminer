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
