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

### Auto Inventor (`operators/auto_inventor.py`)
自动算子发明（Phase 2）：
- `ProposedOperator`: LLM 提议的算子
- `ValidationResult`: 验证结果
- `invent_operator()`: 发明新算子
- `register_proposed_operator()`: 注册新算子

### Custom (`operators/custom.py`)
用户自定义算子：
- `register_custom_operator()`: 注册自定义算子
- 支持用户通过配置文件添加新算子

### GPU Backend (`operators/gpu_backend.py`)
GPU 加速支持：
- `DeviceManager`: 设备选择和管理
- `to_gpu()`: NumPy 转 GPU 张量
- `to_cpu()`: GPU 张量转 NumPy
- 自动 CPU 回退

### Neuro Symbolic (`operators/neuro_symbolic.py`)
神经符号算子（Phase 2）：
- `NeuralLeafNode`: 神经网络叶子节点
- `SmallMLP`: 小型 MLP (< 5000 参数)
- `distill_to_symbolic()`: 蒸馏到符号公式

## 来源
- [DSL 设计](../design/typed-dsl.md)
- [Auto Inventor](../concepts/auto-inventor.md)
- [GPU Backend](../concepts/gpu-backend.md)
- [Neuro Symbolic](../concepts/neuro-symbolic.md)
