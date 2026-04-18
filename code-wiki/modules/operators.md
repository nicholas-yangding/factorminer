# Operators 模块

## Purpose
Operators 模块包含 60+ 算子，覆盖算术、统计、时间序列、横截面、回归、平滑、逻辑等类别。算子注册表将算子名称映射到 NumPy/PyTorch 实现。

## 算子分类

### Arithmetic (`operators/arithmetic.py`)
基本数学运算：
- `add`, `sub`, `mul`, `div`
- `abs`, `log`, `log_abs`, `sqrt`
- `sign`, `pow`, `neg`, `inv`

### Statistical (`operators/statistical.py`)
统计运算：
- `ts_skew`, `ts_kurt` (高阶矩)
- `ts_median`, `ts_std`, `ts_mean`
- `ts_max`, `ts_min`, `ts_sum`

### Time Series (`operators/timeseries.py`)
时间序列运算：
- `ts_returns`, `ts_delta`, `ts_diff`
- `ts_zscore`, `ts_decay_linear`, `ts_rank`
- `ts_corr`, `ts_cov`

### Cross-sectional (`operators/crosssectional.py`)
横截面运算：
- `cs_rank`, `cs_zscore`
- `cs_mean`, `cs_std`, `cs_sum`
- `csCorr`, `csCov`

### Smoothing (`operators/smoothing.py`)
平滑运算：
- `ema`, `sma`, `wma`
- `ts_smooth`

### Regression (`operators/regression.py`)
回归运算：
- `ts_reg`, `ts_slope`, `ts_intercept`
- `ts_resi` (残差)

### Logical (`operators/logical.py`)
逻辑运算：
- `if_else`, `and_`, `or_`, `not_`
- 比较运算符

## Registry (`operators/registry.py`)

```python
OPERATOR_REGISTRY[name] = (OperatorSpec, numpy_fn, torch_fn)
```

核心函数：
- `get_operator(name)`: 获取算子规格
- `get_impl(name, backend)`: 获取实现函数

## 关键设计

### Dual Backend
- NumPy 实现用于 CPU
- PyTorch 实现用于 GPU (可选)

### 类型安全
- `OperatorSpec`: 包含参数类型、返回值类型、算子类别
- 表达式树验证时检查类型匹配

### 特征集
DSL 支持的原始特征：
`$open`, `$high`, `$low`, `$close`, `$volume`, `$amt`, `$vwap`, `$returns`

## 来源
- [DSL 设计](../design/typed-dsl.md)
