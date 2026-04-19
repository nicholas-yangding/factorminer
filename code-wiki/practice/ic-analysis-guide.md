# IC 分析指南

## 什么是 IC？

**IC (Information Coefficient)** 是衡量因子预测能力的重要指标，表示因子信号与未来收益的相关性。

```
IC = Corr_rank(factor_values, future_returns)
```

- 使用 **Spearman 秩相关**（而非 Pearson 线性相关）
- 计算的是每个时间截面的截面相关性
- 结果是一个时间序列 `IC_1, IC_2, ..., IC_T`

---

## IC 系列指标

| 指标 | 公式 | 含义 | 用途 |
|------|------|------|------|
| **IC Mean** | `mean(IC_t)` | IC 均值（带符号） | 决定因子**方向** |
| **IC Abs Mean** | `mean(\|IC_t\|)` | IC 绝对值均值 | 决定因子**强度** |
| **ICIR** | `IC_mean / IC_std` | IC 均值除以标准差 | 决定因子**稳定性** |
| **IC Win Rate** | `count(IC > 0) / T` | IC 为正的时间占比 | 决定**一致 性** |

---

## 深度解读：IC Mean vs IC Abs Mean

### 场景分析

#### 场景 1：IC Mean ≈ 0，但 IC Abs Mean 很高

```
示例：结构突破因子
IC Mean = +0.001 (≈0)
IC Abs Mean = 0.5354 (很高)
ICIR = 0.0014 (很低)
Win Rate = 39.9% (< 50%)
```

**解读**：
- 因子**有预测能力**（|IC| = 0.54 很大）
- 但方向**不稳定**（50% 的日子正相关，50% 负相关）
- IC ≈ 0 是因为正负相抵

**使用建议**：
- ❌ 不能单独使用（方向不稳）
- ✅ 配合其他方向稳定的因子
- ✅ 作为辅助信号

---

#### 场景 2：IC Mean > 0，且 IC Abs Mean 也很高

```
示例：吸筹强度因子
IC Mean = +0.1660 (正)
IC Abs Mean = 0.4326 (很高)
ICIR = 0.2837 (中等)
Win Rate = 43.0% (接近一半日子正相关)
```

**解读**：
- 因子方向**向上**（IC Mean > 0）
- 预测**强度大**（|IC| = 0.43）
- 但 **Win Rate 只有 43%**，说明有超过一半的日子 IC 为负

**使用建议**：
- ✅ 可直接使用
- ⚠️ 需要配合风控（因为有一半日子会亏钱）
- ⚠️ 仓位管理要谨慎

---

#### 场景 3：IC Mean < 0，但 IC Abs Mean 很高

```
IC Mean = -0.0836 (负)
IC Abs Mean = 0.3710 (很高)
Win Rate = 26.3% (很少正相关)
```

**解读**：
- 因子方向**向下**（IC Mean < 0）
- 预测强度很大
- 少数正相关的日子 IC 很大，但大多数日子 IC 为负

**使用建议**：
- ✅ 可做空
- ⚠️ 需要确认因子逻辑是否正确

---

#### 场景 4：IC Mean ≈ 0，IC Abs Mean 也很低

```
IC Mean ≈ 0
IC Abs Mean < 0.03
```

**解读**：因子**无效**，没有预测能力

**使用建议**：
- ❌ 不使用

---

## 实战阈值

| IC Mean | IC Abs Mean | ICIR | Win Rate | 结论 |
|---------|-------------|------|----------|------|
| > 0.02 | > 0.03 | > 0.5 | > 50% | ✅ **优质因子**，可重仓 |
| > 0.02 | > 0.03 | > 0.5 | < 50% | ⚠️ **有效但方向不稳**，需风控 |
| > 0.02 | > 0.03 | < 0.5 | > 50% | ⚠️ **有效但波动大**，轻仓 |
| > 0.02 | > 0.03 | < 0.5 | < 50% | ⚠️ **勉强可用**，需验证 |
| ≈ 0 | > 0.03 | - | - | ⚠️ **方向不稳**，组合使用 |
| ≈ 0 | < 0.03 | - | - | ❌ **无效**，不用 |

---

## ICIR 详解

```
ICIR = IC Mean / IC Std
```

| ICIR 范围 | 含义 |
|-----------|------|
| > 1.0 | 非常稳定，因子优秀 |
| 0.5 ~ 1.0 | 稳定，可使用 |
| 0.3 ~ 0.5 | 中等稳定，注意回撤 |
| < 0.3 | 不稳定，可能有周期性 |

**注意**：ICIR 低不一定意味着因子无效，只要 IC Mean 和 IC Abs Mean 够高，仍然可以使用。

---

## IC 时间序列可视化

建议在评估因子时绘制 IC 时间序列图，观察：

1. **趋势**：IC 是否随时间衰减？
2. **波动**：IC 的标准差是否太大？
3. **极端值**：是否有异常极端的 IC 值？
4. **周期性**：是否与市场周期相关？

```python
import matplotlib.pyplot as plt

def plot_ic_series(ic_series, name="Factor"):
    import matplotlib.pyplot as plt
    import numpy as np
    
    valid = ic_series[~np.isnan(ic_series)]
    t = np.arange(len(valid))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # IC 时间序列
    axes[0].plot(t, valid, 'b-', alpha=0.7)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0].axhline(y=np.mean(valid), color='r', linestyle='--', label=f'Mean={np.mean(valid):.4f}')
    axes[0].fill_between(t, valid, 0, alpha=0.3)
    axes[0].set_title(f'{name} - IC Time Series')
    axes[0].set_ylabel('IC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IC 分布
    axes[1].hist(valid, bins=50, edgecolor='k', alpha=0.7)
    axes[1].axvline(x=0, color='k', linestyle='--')
    axes[1].axvline(x=np.mean(valid), color='r', linestyle='--', label=f'Mean={np.mean(valid):.4f}')
    axes[1].set_title(f'{name} - IC Distribution')
    axes[1].set_xlabel('IC')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

---

## 多因子 IC 分析

### 因子 IC 对比表

```python
def compare_factors(results):
    """对比多个因子的 IC 指标"""
    print(f"\n{'Factor':<30} {'IC Mean':>10} {'IC Abs':>10} {'ICIR':>10} {'WinRate':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<30} {r['ic_mean']:>10.4f} {r['ic_abs_mean']:>10.4f} {r['icir']:>10.4f} {r['win_rate']*100:>9.1f}%")
```

### 因子 IC 相关性

评估因子间的相关性，避免高度相似的因子：

```python
from scipy.stats import spearmanr

def factor_correlation(signals_dict):
    """计算因子间的 IC 序列相关性"""
    names = list(signals_dict.keys())
    n = len(names)
    corr_matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                ic_i = compute_ic(signals_dict[name_i], returns)
                ic_j = compute_ic(signals_dict[name_j], returns)
                valid = ~(np.isnan(ic_i) | np.isnan(ic_j))
                corr_matrix[i, j], _ = spearmanr(ic_i[valid], ic_j[valid])
    
    return pd.DataFrame(corr_matrix, index=names, columns=names)
```

---

## IC 衰减分析

检验因子 IC 是否随时间衰减（因子拥挤）：

```python
def ic_decay_analysis(signals, returns, window=60):
    """分析 IC 随时间的变化"""
    ic_full = compute_ic(signals, returns)
    T = len(ic_full)
    
    # 滚动 IC Mean
    ic_series = ic_full[~np.isnan(ic_full)]
    rolling_mean = pd.Series(ic_series).rolling(window=window, min_periods=10).mean()
    
    # 分段 IC
    n_periods = 5
    period_len = len(ic_series) // n_periods
    
    print("\nIC 衰减分析:")
    print(f"{'Period':<10} {'IC Mean':>12} {'IC Abs Mean':>12}")
    print("-" * 40)
    for i in range(n_periods):
        start = i * period_len
        end = start + period_len if i < n_periods - 1 else len(ic_series)
        period_ic = ic_series[start:end]
        print(f"Period {i+1:<4} {np.mean(period_ic):>12.4f} {np.mean(np.abs(period_ic)):>12.4f}")
```

---

## 完整 IC 评估模板

```python
def full_ic_analysis(signals, returns, name="Factor"):
    """完整 IC 分析"""
    ic_series = compute_ic(signals, returns)
    valid_ic = ic_series[~np.isnan(ic_series)]
    
    if len(valid_ic) == 0:
        return None
    
    stats = {
        "name": name,
        "count": len(valid_ic),
        "ic_mean": np.mean(valid_ic),
        "ic_abs_mean": np.mean(np.abs(valid_ic)),
        "ic_std": np.std(valid_ic),
        "icir": np.mean(valid_ic) / np.std(valid_ic) if np.std(valid_ic) > 0 else 0,
        "win_rate": np.mean(valid_ic > 0),
        "ic_min": np.min(valid_ic),
        "ic_max": np.max(valid_ic),
        "ic_25": np.percentile(valid_ic, 25),
        "ic_75": np.percentile(valid_ic, 75),
    }
    
    print(f"\n{'='*50}")
    print(f"因子: {name}")
    print(f"{'='*50}")
    print(f"样本数:        {stats['count']}")
    print(f"IC Mean:       {stats['ic_mean']:>10.4f}")
    print(f"IC Abs Mean:   {stats['ic_abs_mean']:>10.4f}")
    print(f"IC Std:        {stats['ic_std']:>10.4f}")
    print(f"ICIR:          {stats['icir']:>10.4f}")
    print(f"Win Rate:      {stats['win_rate']*100:>10.1f}%")
    print(f"IC Range:      [{stats['ic_min']:.4f}, {stats['ic_max']:.4f}]")
    print(f"IC IQR:        [{stats['ic_25']:.4f}, {stats['ic_75']:.4f}]")
    
    # 判定
    if stats['ic_abs_mean'] < 0.03:
        verdict = "❌ 无效"
    elif stats['ic_mean'] > 0.02 and stats['ic_abs_mean'] > 0.03:
        verdict = "✅ 有效（做多）"
    elif stats['ic_mean'] < -0.02 and stats['ic_abs_mean'] > 0.03:
        verdict = "✅ 有效（做空）"
    else:
        verdict = "⚠️ 方向不稳，组合使用"
    print(f"结论:          {verdict}")
    
    return stats
```

---

## 常见误区

### 1. 只看 IC Mean 忽略 IC Abs Mean

```
IC Mean = 0.01 看起来正，但 IC Abs Mean = 0.02 说明实际没什么预测能力
```

### 2. 只看 IC Abs Mean 忽略 IC Mean

```
IC Abs Mean = 0.05 看起来不错，但 IC Mean = -0.04 说明做空才有效
```

### 3. 过度依赖 ICIR

```
ICIR = 0.3 不一定差，只要 IC Mean = 0.10, IC Abs = 0.40，仍然可用
ICIR 高不一定好，IC Mean 接近 0 的因子 ICIR 再高也没用
```

### 4. 样本量不足

```
T < 30 的 IC 统计不可靠，至少需要 60 个时间点
```
