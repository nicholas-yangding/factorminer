# 06 - 算子参考

## 算子总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    70+ 算子一览                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ARITHMETIC (15)  │  STATISTICAL (16) │  TIMESERIES (14)      │
│  ──────────────────│───────────────────│─────────────────────── │
│  Add, Sub          │  Mean, Std       │  Delta, Delay          │
│  Mul, Div          │  Var, Skew, Kurt │  Return, LogReturn     │
│  Neg, Abs          │  Median, Sum     │  Corr, Cov, Beta       │
│  Sign, Log         │  Prod, TsMax     │  Resid, WMA           │
│  Sqrt, Square      │  TsMin, TsRank   │  Decay                │
│  Pow, Max, Min     │  TsArgMax,       │  CumSum, CumProd       │
│  Clip, Inv         │  TsArgMin,       │  CumMax, CumMin        │
│                    │  Quantile,        │                       │
│                    │  CountNaN,        │                       │
│                    │  CountNotNaN      │                       │
│                                                                 │
│  CROSS_SECTION (6)  │  SMOOTHING (5)   │  REGRESSION (4)       │
│  ──────────────────│──────────────────│─────────────────────── │
│  CsRank            │  EMA, DEMA       │  TsLinReg             │
│  CsZScore          │  SMA, KAMA       │  TsLinRegSlope        │
│  CsDemean          │  HMA             │  TsLinRegIntercept     │
│  CsScale           │                  │  TsLinRegResid        │
│  CsNeutralize       │                  │                       │
│  CsQuantile        │                  │                       │
│                                                                 │
│  LOGICAL (10)      │  NEURO_SYMBOLIC (2)                      │
│  ──────────────────│─────────────────────────────────────────  │
│  IfElse            │  neural_leaf_train                        │
│  Greater, Less     │  neural_leaf_eval                         │
│  GreaterEqual,    │                                          │
│  LessEqual        │                                          │
│  Equal, Ne        │                                          │
│  And, Or, Not     │                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 算子注册表

```python
from factorminer.core.types import OPERATOR_REGISTRY, get_operator

# 获取算子规格
op_spec = get_operator("Mean")
print(op_spec.name)      # "Mean"
print(op_spec.arity)      # 1 (参数个数)
print(op_spec.category)   # OperatorType.STATISTICAL

# 列出所有可用算子
print(sorted(OPERATOR_REGISTRY.keys()))
```

## 算子分类详解

### 1. 算术算子 (ARITHMETIC)

```python
# 二元运算
Add(x, y)      # x + y
Sub(x, y)      # x - y
Mul(x, y)      # x * y
Div(x, y)      # x / y (Safe division)
Pow(x, y)      # x^y
Max(x, y)      # element-wise max(x, y)
Min(x, y)      # element-wise min(x, y)

# 一元运算
Neg(x)         # -x
Abs(x)         # |x|
Sign(x)        # sign(x): -1, 0, 1
Log(x)         # log(1 + |x|) * sign(x)
Sqrt(x)        # sqrt(|x|) * sign(x)
Square(x)      # x^2
Clip(x)        # clip(x, lower, upper), 默认 [-3, 3]
Inv(x)         # 1 / x (safe)
```

**示例:**

```
# 价格动量
$close - $close / Pow(2, 10)  # 简化: $close * (1 - 1/1024)

# 对数收益率
Log($close)
```

### 2. 统计算子 (STATISTICAL)

```python
# 沿时间轴滚动计算 (所有算子都接受 window 参数)
Mean(x, window)       # 滚动均值
Std(x, window)       # 滚动标准差
Var(x, window)       # 滚动方差
Median(x, window)    # 滚动中位数
Sum(x, window)       # 滚动和
Prod(x, window)      # 滚动乘积
TsMax(x, window)    # 滚动最大值
TsMin(x, window)    # 滚动最小值
TsArgMax(x, window) # 滚动最大值位置
TsArgMin(x, window) # 滚动最小值位置
TsRank(x, window)   # 滚动排名 (当前值在窗口内的分位数)
Skew(x, window)     # 滚动偏度
Kurt(x, window)     # 滚动峰度
Quantile(x, window, q=0.5)  # 滚动分位数
CountNaN(x, window)         # 滚动 NaN 计数
CountNotNaN(x, window)      # 滚动非 NaN 计数
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
Return(x, period=1)      # 收益率: x[t] / x[t-period] - 1
LogReturn(x, period=1)   # 对数收益率: log(x[t] / x[t-period])
Delta(x, period=1)       # 变化量: x[t] - x[t-period]
Delay(x, period=1)        # 滞后: x[t-period]
Diff(x)                  # 一阶差分: x[t] - x[t-1]
Zscore(x, window=20)    # 标准化: (x - mean) / std
Corr(x, y, window)      # 滚动相关性
Cov(x, y, window)       # 滚动协方差
Beta(x, y, window)      # 滚动回归 beta
Resid(x, y, window)     # 滚动回归残差
WMA(x, window)          # 加权移动平均
Decay(x, window)        # 指数衰减和
CumSum(x)               # 累计求和
CumProd(x)              # 累计乘积
CumMax(x)               # 累计最大值
CumMin(x)               # 累计最小值
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

# Beta (市场敏感度)
Beta($returns, $market_returns, 60)
```

### 4. 横截面算子 (CROSS_SECTIONAL)

```python
CsRank(x)         # 排名: 在每个时间点横截面排名 (0-1 分位数)
CsZScore(x)       # 标准化: 在每个时间点横截面标准化
CsDemean(x)       # 去均值: x - cross-sectional mean
CsScale(x)        # 缩放: scale to unit L1 norm
CsNeutralize(x)   # 行业中性化
CsQuantile(x, n_bins=5)  # 横截面分桶
```

**图示: CsRank**

```
时间 t:    股票A  股票B  股票C  股票D
价格:      10     20     15     25
CsRank:   0.0    1.0    0.5    0.75  (排名分位数 0-1)
```

**示例:**

```
# 行业相对强弱
$close / CsDemean($close)

# 量价背离选股
CsZScore($close) - CsZScore($volume)
```

### 5. 平滑算子 (SMOOTHING)

```python
EMA(x, span)        # 指数移动平均
DEMA(x, span)       # 双指数移动平均
SMA(x, window)      # 简单移动平均
KAMA(x, window)     # Kaufman 自适应移动平均
HMA(x, window)      # Hull 移动平均
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
EMA($close, 5) - EMA($close, 20)

# 趋势确认
IfElse(EMA($close, 10) > EMA($close, 30), 1, -1)
```

### 6. 回归算子 (REGRESSION)

```python
TsLinReg(x, window)          # 滚动线性回归拟合值
TsLinRegSlope(x, window)     # 滚动线性回归斜率
TsLinRegIntercept(x, window) # 滚动线性回归截距
TsLinRegResid(x, window)     # 滚动线性回归残差
```

**注意:** 回归算子只接受一个主要输入 x 和 window 参数，用于对 x 本身做滚动回归。

**示例:**

```
# 斜率（趋势强度）
TsLinRegSlope($returns, 60)

# 残差（去除趋势后的波动）
TsLinRegResid($returns, 60)
```

### 7. 逻辑算子 (LOGICAL)

```python
IfElse(cond, then, else)       # 条件选择
Greater(x, y)                   # x > y 时返回 1，否则 0
Less(x, y)                      # x < y 时返回 1，否则 0
GreaterEqual(x, y)               # x >= y 时返回 1，否则 0
LessEqual(x, y)                 # x <= y 时返回 1，否则 0
Equal(x, y)                     # x == y 时返回 1，否则 0
Ne(x, y)                        # x != y 时返回 1，否则 0
And(x, y)                       # 逻辑与
Or(x, y)                        # 逻辑或
Not(x)                          # 逻辑非
```

**示例:**

```
# 趋势突破
IfElse(Greater($close, Max($close, 20)), 1, -1)

# 波动率过滤
IfElse(Greater(Std($returns, 20), 0.02), Return($close, 5), 0)

# 区间震荡
IfElse(
    And(
        Less($close, Mul(Mean($close, 20), 1.05)),
        Greater($close, Mul(Mean($close, 20), 0.95))
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
Mul(CsRank(Div($close, CsDemean($close))), TsRank(Return($close, 60), 20))

# 量价背离
Mul(Corr($close, $volume, 20), Zscore(Return($returns, 20), 20))

# 高阶矩 + 趋势
IfElse(
    Greater(Skew($returns, 20), 0),
    Return($close, 20),
    Neg(Return($close, 20))
)
```

## 下一步

- [Ralph Loop](07-ralph-loop.md) - 挖掘循环
- [评估流程](08-evaluation.md) - 因子评估
