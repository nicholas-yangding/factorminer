# 06 - 算子参考

## 算子总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    60+ 算子一览                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ARITHMETIC (12)  │  STATISTICAL (8)  │  TIMESERIES (12)       │
│  ─────────────────│──────────────────│─────────────────────     │
│  add, sub         │  ts_mean        │  ts_returns             │
│  mul, div         │  ts_std         │  ts_delta               │
│  abs, log        │  ts_median      │  ts_diff                │
│  log_abs, sqrt   │  ts_skew        │  ts_zscore              │
│  sign, pow       │  ts_kurt        │  ts_rank                │
│  neg, inv        │  ts_max, ts_min │  ts_corr, ts_cov        │
│                  │  ts_sum         │  ts_decay_linear        │
│                                                                 │
│  CROSS_SECTION (8) │  SMOOTHING (4) │  REGRESSION (4)        │
│  ──────────────────│────────────────│────────────────────     │
│  cs_rank          │  ema           │  ts_reg                │
│  cs_zscore        │  sma           │  ts_slope               │
│  cs_mean          │  wma           │  ts_intercept          │
│  cs_std           │  ts_smooth     │  ts_resi               │
│  cs_sum           │                │                        │
│  csCorr, csCov   │                │                        │
│                                                                 │
│  LOGICAL (6)       │  NEURO_SYMBOLIC (2)                       │
│  ──────────────────│────────────────────────────────            │
│  if_else           │  neural_leaf_train                        │
│  and_, or_, not_   │  neural_leaf_eval                          │
│  >, <, ==         │                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 算子注册表

```python
from factorminer.operators.registry import OPERATOR_REGISTRY, get_operator

# 获取算子规格
op_spec = get_operator("ts_mean")
print(op_spec.name)      # "ts_mean"
print(op_spec.arity)      # 2
print(op_spec.category)    # OperatorType.TIMESERIES

# 获取实现
impl = get_impl("ts_mean", backend="numpy")
result = impl(input_array, 20)
```

## 算子分类详解

### 1. 算术算子 (ARITHMETIC)

```python
# 二元运算
add(x, y)      # x + y
sub(x, y)      # x - y
mul(x, y)      # x * y
div(x, y)      # x / y (Safe division)

# 一元运算
abs(x)         # |x|
log(x)         # log(|x|)
log_abs(x)     # log(|x|)
sqrt(x)        # sqrt(|x|)
sign(x)        # sign(x): -1, 0, 1
pow(x, n)      # x^n
neg(x)         # -x
inv(x)         # 1/x
```

**示例:**

```
# 价格动量
$close - $close / pow(2, 10)  # 简化: $close * (1 - 1/1024)

# 对数收益率
log($close)
```

### 2. 统计算子 (STATISTICAL)

```python
# 沿时间轴滚动计算
ts_mean(x, window)     # 滚动均值
ts_std(x, window)      # 滚动标准差
ts_median(x, window)   # 滚动中位数
ts_sum(x, window)      # 滚动和
ts_max(x, window)      # 滚动最大值
ts_min(x, window)      # 滚动最小值
ts_skew(x, window)     # 滚动偏度 (高阶矩)
ts_kurt(x, window)     # 滚动峰度 (高阶矩)
```

**示例:**

```
# 布林带
($close - ts_mean($close, 20)) / (2 * ts_std($close, 20))

# 价量相关性
ts_corr($close, $volume, 20)
```

### 3. 时间序列算子 (TIMESERIES)

```python
ts_returns(x, period=1)     # 收益率: x[t] / x[t-period] - 1
ts_delta(x, period=1)       # 变化量: x[t] - x[t-period]
ts_diff(x)                  # 一阶差分: x[t] - x[t-1]
ts_zscore(x, window=20)    # 标准化: (x - mean) / std
ts_rank(x, window=20)      # 滚动排名: rank(x[t]) in [0,1]
ts_corr(x, y, window)      # 滚动相关性
ts_cov(x, y, window)       # 滚动协方差
ts_decay_linear(x, window) # 线性衰减加权
```

**图示: ts_returns**

```
period=1:    period=5:
t0: -        t0: -
t1: (v1-v0)/v0    t5: (v5-v0)/v0
t2: (v2-v1)/v1    t6: (v6-v1)/v1
...           ...
```

**示例:**

```
# 动量
ts_returns($close, 20)

# 趋势强度
ts_corr(ts_returns($close, 1), ts_returns($volume, 1), 20)
```

### 4. 横截面算子 (CROSS_SECTION)

```python
cs_rank(x)      # 排名: 在每个时间点横截面排名
cs_zscore(x)    # 标准化: 在每个时间点横截面标准化
cs_mean(x)      # 横截面均值
cs_std(x)       # 横截面标准差
cs_sum(x)       # 横截面求和
csCorr(x, y)    # 横截面相关性
csCov(x, y)     # 横截面协方差
```

**图示: cs_rank**

```
时间 t:    股票A  股票B  股票C  股票D
价格:      10     20     15     25
cs_rank:   1      3      2      4   (排名 1-4)
```

**示例:**

```
# 行业相对强弱
$close / cs_mean($close)

# 量价背离选股
cs_zscore($close) - cs_zscore($volume)
```

### 5. 平滑算子 (SMOOTHING)

```python
ema(x, span)        # 指数移动平均
sma(x, window)      # 简单移动平均
wma(x, window)      # 加权移动平均
ts_smooth(x)        # 平滑（内部算法）
```

**图示: EMA vs SMA**

```
Price:  10 ─●──●──●──●──●──●──●──●──●
         │                              
EMA(5)  ─────────────●──────────────────
         │              ╲              
SMA(5)  ────────────────●──────────────
                            ╲           
                              ──────────
```

**示例:**

```
# 双均线
ema($close, 5) - ema($close, 20)

# 趋势确认
IfElse(ema($close, 10) > ema($close, 30), 1, -1)
```

### 6. 回归算子 (REGRESSION)

```python
ts_reg(x, y, window)     # 滚动回归系数 [slope, intercept]
ts_slope(x, window)       # 斜率
ts_intercept(x, window)   # 截距
ts_resi(x, window)        # 残差
```

**示例:**

```
# Beta
ts_slope($returns, $market_returns, 60)

# Alpha
ts_intercept($returns, $market_returns, 60)

# 回归残差（去除市场影响）
ts_resi($returns, $market_returns, 60)
```

### 7. 逻辑算子 (LOGICAL)

```python
IfElse(cond, then, else)   # 条件选择
and_(a, b)                 # 逻辑与
or_(a, b)                  # 逻辑或
not_(a)                    # 逻辑非
gt(a, b), lt(a, b)        # 大于/小于
ge(a, b), le(a, b)        # 大于等于/小于等于
eq(a, b)                   # 等于
```

**示例:**

```
# 趋势突破
IfElse($close > ts_max($close, 20), 1, -1)

# 波动率过滤
IfElse(ts_std($returns, 20) > 0.02, ts_returns($close, 5), 0)

# 区间震荡
IfElse(
    and_(
        $close < ts_mean($close, 20) * 1.05,
        $close > ts_mean($close, 20) * 0.95
    ),
    1,
    0
)
```

## 签名类型

| 签名 | 说明 | 输入输出 |
|------|------|---------|
| `TIME_SERIES_TO_TIME_SERIES` | 滚动运算 | (M,T) → (M,T) |
| `CROSS_SECTION_TO_CROSS_SECTION` | 截面运算 | (M,T) → (M,T) |
| `ELEMENT_WISE` | 点对点 | (M,T) → (M,T) |
| `REDUCE_TIME` | 时间压缩 | (M,T) → (M,) |

## 组合示例

```python
# 经典动量策略
ts_rank(ts_returns($close, 20), 60)

# 价值 + 动量
cs_rank($close / cs_mean($close, 20)) * ts_rank(ts_returns($close, 60), 20)

# 量价背离
ts_corr($close, $volume, 20) * ts_zscore($returns, 20)

# 高阶矩 + 趋势
IfElse(
    ts_skew($returns, 20) > 0,
    ts_returns($close, 20),
    -ts_returns($close, 20)
)
```

## 下一步

- [Ralph Loop](07-ralph-loop.md) - 挖掘循环
- [评估流程](08-evaluation.md) - 因子评估
