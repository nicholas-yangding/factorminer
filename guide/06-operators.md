# 06 - 算子参考

## 算子总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    60+ 算子一览                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ARITHMETIC (12)  │  STATISTICAL (8)  │  TIMESERIES (12)       │
│  ─────────────────│──────────────────│─────────────────────     │
│  add, sub         │  Mean        │  Return             │
│  mul, div         │  Std         │  Delta               │
│  abs, log        │  Median      │  Diff                │
│  log_abs, sqrt   │  Skew        │  Zscore              │
│  sign, pow       │  Kurt        │  TsRank                │
│  neg, inv        │  Max, Min │  Corr, Cov        │
│                  │  Sum         │  DecayLinear        │
│                                                                 │
│  CROSS_SECTION (8) │  SMOOTHING (4) │  REGRESSION (4)        │
│  ──────────────────│────────────────│────────────────────     │
│  CsRank          │  Ema           │  Reg                │
│  CsZscore        │  Sma           │  Slope               │
│  CsMean          │  Wma           │  Intercept          │
│  CsStd           │  Smooth     │  Resi               │
│  CsSum           │                │                        │
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
op_spec = get_operator("Mean")
print(op_spec.name)      # "Mean"
print(op_spec.arity)      # 2
print(op_spec.category)    # OperatorType.TIMESERIES

# 获取实现
impl = get_impl("Mean", backend="numpy")
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
Mean(x, window)     # 滚动均值
Std(x, window)      # 滚动标准差
Median(x, window)   # 滚动中位数
Sum(x, window)      # 滚动和
Max(x, window)      # 滚动最大值
Min(x, window)      # 滚动最小值
Skew(x, window)     # 滚动偏度 (高阶矩)
Kurt(x, window)     # 滚动峰度 (高阶矩)
```

**示例:**

```
# 布林带
($close - Mean($close, 20)) / (2 * Std($close, 20))

# 价量相关性
Corr($close, $volume, 20)
```

### 3. 时间序列算子 (TIMESERIES)

```python
Return(x, period=1)     # 收益率: x[t] / x[t-period] - 1
Delta(x, period=1)       # 变化量: x[t] - x[t-period]
Diff(x)                  # 一阶差分: x[t] - x[t-1]
Zscore(x, window=20)    # 标准化: (x - mean) / std
TsRank(x, window=20)      # 滚动排名: rank(x[t]) in [0,1]
Corr(x, y, window)      # 滚动相关性
Cov(x, y, window)       # 滚动协方差
DecayLinear(x, window) # 线性衰减加权
```

**图示: Return**

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
Return($close, 20)

# 趋势强度
Corr(Return($close, 1), Return($volume, 1), 20)
```

### 4. 横截面算子 (CROSS_SECTION)

```python
CsRank(x)      # 排名: 在每个时间点横截面排名
CsZscore(x)    # 标准化: 在每个时间点横截面标准化
CsMean(x)      # 横截面均值
CsStd(x)       # 横截面标准差
CsSum(x)       # 横截面求和
csCorr(x, y)    # 横截面相关性
csCov(x, y)     # 横截面协方差
```

**图示: CsRank**

```
时间 t:    股票A  股票B  股票C  股票D
价格:      10     20     15     25
CsRank:   1      3      2      4   (排名 1-4)
```

**示例:**

```
# 行业相对强弱
$close / CsMean($close)

# 量价背离选股
CsZscore($close) - CsZscore($volume)
```

### 5. 平滑算子 (SMOOTHING)

```python
Ema(x, span)        # 指数移动平均
Sma(x, window)      # 简单移动平均
Wma(x, window)      # 加权移动平均
Smooth(x)        # 平滑（内部算法）
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
Ema($close, 5) - Ema($close, 20)

# 趋势确认
IfElse(Ema($close, 10) > Ema($close, 30), 1, -1)
```

### 6. 回归算子 (REGRESSION)

```python
Reg(x, y, window)     # 滚动回归系数 [slope, intercept]
Slope(x, window)       # 斜率
Intercept(x, window)   # 截距
Resi(x, window)        # 残差
```

**示例:**

```
# Beta
Slope($returns, $market_returns, 60)

# Alpha
Intercept($returns, $market_returns, 60)

# 回归残差（去除市场影响）
Resi($returns, $market_returns, 60)
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
IfElse($close > Max($close, 20), 1, -1)

# 波动率过滤
IfElse(Std($returns, 20) > 0.02, Return($close, 5), 0)

# 区间震荡
IfElse(
    and_(
        $close < Mean($close, 20) * 1.05,
        $close > Mean($close, 20) * 0.95
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
TsRank(Return($close, 20), 60)

# 价值 + 动量
CsRank($close / CsMean($close, 20)) * TsRank(Return($close, 60), 20)

# 量价背离
Corr($close, $volume, 20) * Zscore($returns, 20)

# 高阶矩 + 趋势
IfElse(
    Skew($returns, 20) > 0,
    Return($close, 20),
    -Return($close, 20)
)
```

## 下一步

- [Ralph Loop](07-ralph-loop.md) - 挖掘循环
- [评估流程](08-evaluation.md) - 因子评估
