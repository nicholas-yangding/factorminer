# 10 - Helix Loop (Phase 2)

## Helix Loop 概述

Helix Loop 是 Phase 2 的增强循环，在 Paper Lane 基础上增加了高级特性：

```
┌─────────────────────────────────────────────────────────────────┐
│                   Helix Loop vs Ralph Loop                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Ralph Loop                      Helix Loop                     │
│   ───────────                    ───────────                     │
│                                    │                            │
│   Memory ───────────────────────▶ Memory (增强)                   │
│       │                               │                        │
│       │                               ├─▶ Knowledge Graph       │
│       │                               ├─▶ Embeddings            │
│       │                               │                         │
│   Generator                      Generator (增强)                  │
│       │                               │                         │
│       │                               ├─▶ Debate               │
│       │                               │                         │
│   Evaluation                    Evaluation (增强)                 │
│       │                               │                         │
│       │                               ├─▶ Causal Validation    │
│       │                               ├─▶ Regime Analysis       │
│       │                               ├─▶ Capacity Estimation   │
│       │                               └─▶ Significance Testing  │
│       │                               │                         │
│       ▼                               ▼                         │
│   Canonicalizer (新增)              Canonicalizer ────────────▶ SymPy │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 启用方式

```bash
# 启用 Helix Lane
factorminer --cpu helix --mock -n 2 -b 8 -t 10

# 启用特定特性
factorminer --cpu helix --mock --debate --canonicalize -n 2 -b 8 -t 10
```

## 配置

```yaml
phase2:
  causal: true           # 因果验证
  regime: true         # 市场状态分析
  capacity: true       # 容量估计
  significance: true   # 显著性检验
  debate: true         # 辩论生成
  auto_inventor: true  # 自动算子发明
  helix: true          # Helix 模块
```

## Debate (辩论生成)

多专家辩论生成更好的因子：

```
┌─────────────────────────────────────────────────────────────┐
│                    Debate 流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   候选因子 ──▶ 统计专家 ──▶ 算子专家 ──▶ 领域专家 ──▶ 综合 │
│                 │           │           │           │       │
│               统计角度     算子角度     市场角度    评分   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
@dataclass
class DebateResult:
    score: float              # 综合评分
    statistical_merit: float # 统计优势
    operator_complexity: float # 算子复杂度
    market_understanding: float # 市场理解
    concerns: List[str]       # 担忧点
    suggestions: List[str]    # 建议
```

## Canonicalization (规范化)

使用 SymPy 规范化表达式：

```python
from factorminer.core.canonicalizer import SymPyCanonicalizer

canonicalizer = SymPyCanonicalizer()

# 规范化前
formula1 = "log(x) - log(y)"
formula2 = "log(x / y)"

# 规范化后 (应该相同)
canonical1 = canonicalizer.canonicalize(parse(formula1))
canonical2 = canonicalizer.canonicalize(parse(formula2))

# 发现等价表达式
if canonical1 == canonical2:
    print("等价表达式！")
```

**好处:**
- 发现重复因子
- 简化存储
- 消除冗余计算

## Causal Validation (因果验证)

验证因子是否具有因果预测能力：

```python
def causal_validation(factor, market_data):
    """
    Granger 因果性检验
    """
    # H0: x 不能 Granger-cause y
    # 如果拒绝 H0，则 x 对 y 有因果预测能力
    
    from statsmodels.tsa.stattools import grangercausalitytests
    
    result = grangercausalitytests(
        [factor_values, market_returns],
        maxlag=5
    )
    
    return result.p_value < 0.05
```

## Regime Analysis (市场状态分析)

评估因子在不同市场状态下的表现：

```python
@dataclass
class RegimeAnalysis:
    bull_ic: float       # 牛市 IC
    bear_ic: float       # 熊市 IC
    sideways_ic: float   # 震荡 IC
    best_regime: str     # 最佳状态
    regime_switch_aligned: bool
```

**市场状态定义:**
- **Bull**: 动量持续上升
- **Bear**: 动量持续下降
- **Sideways**: 低波动，无明显趋势

## Capacity Estimation (容量估计)

估算策略容量：

```python
def estimate_capacity(factor, market_data):
    """
    估算因子可承载的资金量
    """
    # 1. 计算换手率
    turnover = compute_turnover(factor)
    
    # 2. 估算市场规模
    market_cap = compute_market_cap(factor.universe)
    
    # 3. 计算容量
    # 经验公式: capacity ≈ market_cap * turnover_threshold
    capacity = market_cap * 0.01 * (1 / turnover)
    
    return capacity
```

## Significance Testing (显著性检验)

统计显著性检验：

```python
def significance_test(factor, ic_series):
    """
    检验 IC 是否显著不为零
    """
    from scipy import stats
    
    # t-test
    t_stat, p_value = stats.ttest_1samp(ic_series, 0)
    
    return {
        "significant": p_value < 0.05,
        "p_value": p_value,
        "t_stat": t_stat
    }
```

## Auto Inventor (自动算子发明)

Helix 可选自动发现新算子：

```bash
factorminer --cpu helix --mock --auto-inventor -n 2 -b 8 -t 10
```

```
┌─────────────────────────────────────────────────────────────┐
│                 Auto Inventor 流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. LLM 提议 ──▶ 2. 语法验证 ──▶ 3. IC 验证 ──▶ 4. 注册 │
│        │               │               │               │      │
│     提出新算子        检查 Python     在样本上测试     加入  │
│                      语法            IC 贡献        注册表  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 综合评估

Helix Loop 的完整评估流程：

```
候选因子
    │
    ├── 标准评估 (同 Ralph Loop)
    │       ├── IC 筛选
    │       ├── 相关性检查
    │       └── 去重
    │
    ├── Debate 评分
    │       └── 多专家评估
    │
    ├── Causal 验证
    │       └── Granger 检验
    │
    ├── Regime 分析
    │       └── 多状态评估
    │
    ├── Capacity 估计
    │       └── 容量预测
    │
    └── Significance 检验
            └── t-test
```

## 下一步

- [CLI 参考](11-cli-reference.md) - 完整命令行用法
