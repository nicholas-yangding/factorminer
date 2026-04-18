# Evaluation 模块

## Purpose
Evaluation 模块实现因子评估的完整管道，包括指标计算、相关性分析、回测、组合选择、因果验证等。

## 核心组件

### Metrics (`evaluation/metrics.py`)
因子指标计算：
- `compute_ic()`: Information Coefficient
- `compute_ic_mean()`, `compute_ic_abs_mean()`, `compute_icir()`
- `compute_ic_win_rate()`: IC 胜率
- `compute_factor_stats()`: 综合统计

### Pipeline (`evaluation/pipeline.py`)
评估管道：
- `EvaluationPipeline`: 多阶段评估
- `SignalComputationError`: 信号计算异常
- 阶段 1-4 评估流程

### Runtime (`evaluation/runtime.py`)
- `compute_tree_signals()`: 在数据上计算因子信号
- `SignalComputationError`: 处理计算失败

### Correlation (`evaluation/correlation.py`)
- `FactorCorrelator`: 因子相关性管理
- 维护相关性矩阵

### Admission (`evaluation/admission.py`)
- `passes_admission()`: 判断因子是否准入
- IC 阈值、ICIR 阈值、正交性检查

### Research (`evaluation/research.py`)
- `passes_research_admission()`: 研究级准入
- 多时间框架评分

### Backtest (`evaluation/backtest.py`)
- `backtest_factor()`: 因子回测
- 计算收益、风险指标

### Combination (`evaluation/combination.py`)
- `combine_factors()`: 因子组合
- 等权、IC 加权、风险平价

### Portfolio (`evaluation/portfolio.py`)
- `build_portfolio()`: 组合构建

### Regime (`evaluation/regime.py`)
- `RegimeClassifier`: 市场状态分类
- 多状态条件评估

### Causal (`evaluation/causal.py`)
- 因果验证
- Granger 因果性检验
- 干预分析

### Significance (`evaluation/significance.py`)
- `SignificanceTester`: 统计显著性检验

### Capacity (`evaluation/capacity.py`)
- `CapacityEstimator`: 策略容量估计

### Selection (`evaluation/selection.py`)
- `FactorSelector`: 因子选择模型

### Transaction Costs (`evaluation/transaction_costs.py`)
- `TransactionCostModel`: 交易成本模型

## 评估流程

```
候选因子 --> Pipeline --> Metrics --> Admission --> FactorLibrary
                      |
                      +--> Correlation (Orthogonality)
                      +--> Research (Multi-horizon)
                      +--> Backtest (Performance)
```

## 来源
- [因子评估](../concepts/factor-evaluation.md)
