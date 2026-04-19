# 08 - 因子评估

## 评估管道概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    多阶段评估管道                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  候选因子 ──▶ Stage 1 ──▶ Stage 2 ──▶ Stage 2.5 ──▶ Stage 3 ──▶ Stage 4 ──▶ Admission │
│                │          │           │            │          │              │              │
│               IC        Corr       Replace      Dedup      Full IC        │              │
│              筛选      检查        检查         去重       验证            ▼              │
│               │          │           │            │          │         ┌────────┐         │
│               ▼          ▼           ▼            ▼          ▼         │ 准入   │         │
│            IC<0.04    rho>0.5    新IC<旧IC    rho>0.5    IC<0.04      │ 或拒绝 │         │
│            拒绝        拒绝        拒绝         拒绝       拒绝         └────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 核心指标

### IC (Information Coefficient)

```python
def compute_ic(factor_values, future_returns):
    """
    计算 IC (Spearman 相关系数)
    """
    from scipy.stats import spearmanr
    ic, _ = spearmanr(factor_values, future_returns)
    return ic
```

**IC 含义:**
- IC > 0: 正向预测
- IC < 0: 负向预测
- |IC| > 0.02: 有意义
- |IC| > 0.05: 较强

### ICIR (IC Irregularity)

```python
def compute_icir(ic_series):
    """
    ICIR = IC均值 / IC标准差
    衡量 IC 的稳定性
    """
    return np.mean(ic_series) / np.std(ic_series)
```

**ICIR 判断:**
- ICIR > 0.5: 稳定的高质量因子
- ICIR < 0.3: 不稳定，因子可能失效

### IC 胜率

```python
def compute_ic_win_rate(ic_series):
    """
    IC > 0 的时间步占比
    """
    return np.mean(ic_series > 0)
```

## Stage 1: Fast IC Screening

快速筛选，仅用部分资产：

```python
# 配置
fast_screen_assets = 100  # 快速筛选资产数

# 在 M_fast 资产上计算 IC
ic_fast = compute_ic(factor_values[:100], future_returns[:100])

# 筛选
if abs(ic_fast) < ic_threshold:
    reject(candidate)
```

## Stage 2: Correlation Check

与库中已有因子相关性检查：

```python
def check_correlation(candidate_ic, library_ic_series):
    """
    检查与库因子的相关性
    """
    from scipy.stats import pearsonr
    
    for existing_factor in library:
        rho, _ = pearsonr(candidate_ic, existing_factor.ic_series)
        if abs(rho) > correlation_threshold:
            return "correlated"  # 需要进一步检查
    return "independent"
```

## Stage 2.5: Replacement Check

相关因子是否值得替换：

```
条件: 新因子与库因子相关 (rho > 0.5)

判断:
- 新 IC > 旧 IC * 1.3  → 可替换
- 新 IC <= 旧 IC * 1.3 → 拒绝
```

## Stage 3: Intra-batch Dedup

Batch 内去重：

```python
def check_dedup(candidates):
    """
    Batch 内 pairwise 相关性检查
    """
    n = len(candidates)
    keep = []
    
    for i in range(n):
        is_unique = True
        for j in keep:
            rho = pearsonr(candidates[i].ic, candidates[j].ic)
            if abs(rho) > 0.5:
                is_unique = False
                break
        if is_unique:
            keep.append(i)
    
    return [candidates[i] for i in keep]
```

## Stage 4: Full Validation

全量数据验证：

```python
# 在 M_full 资产上计算
ic_full = compute_ic(all_factor_values, all_future_returns)
icir_full = compute_icir(ic_series_full)

# 多时间框架评估
multi_horizon_scores = []
for horizon in [1, 5, 20]:
    ic_h = compute_ic(factor_values, future_returns_shifted(horizon))
    multi_horizon_scores.append(ic_h)
```

## Admission 判断

```python
def passes_admission(factor):
    """
    综合判断是否准入
    """
    checks = {
        "ic": abs(factor.ic) >= ic_threshold,
        "icir": factor.icir >= icir_threshold,
        "independent": not factor.is_correlated,
        "unique": factor.is_unique,
    }
    
    if all(checks.values()):
        return "admit"
    elif checks["ic"] and checks["icir"]:
        return "admit"  # 即使相关也考虑
    else:
        return "reject"
```

## 信号失败处理

因子计算失败时的策略：

```yaml
evaluation:
  signal_failure_policy: "reject"  # reject | synthetic | raise
```

| 策略 | 行为 |
|------|------|
| `reject` | 直接拒绝该因子 |
| `synthetic` | 生成合成信号继续评估 |
| `raise` | 抛出异常 |

## 评估结果

```python
@dataclass
class EvaluationResult:
    factor_id: str
    ic: float
    icir: float
    ic_win_rate: float
    correlation_with_library: float
    passed: bool
    rejection_reason: Optional[str]
```

## 可视化

```python
# IC 时间序列图
plot_ic_series(factor.ic_series)

# 相关性热力图
plot_correlation_matrix(library)

# Quintile 分组收益图
plot_quintile_returns(factor, returns)
```

## 下一步

- [经验记忆](09-memory.md) - 记忆系统
- [Helix Loop](10-helix-loop.md) - Phase 2 扩展
