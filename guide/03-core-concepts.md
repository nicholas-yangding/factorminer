# 03 - 核心概念

## 概览图

```
┌──────────────────────────────────────────────────────────────────┐
│                      FactorMiner 核心概念                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐     ┌─────────────┐     ┌──────────────┐        │
│   │  因子   │────▶│   IC/ICIR   │────▶│  因子评估    │        │
│   └─────────┘     └─────────────┘     └──────────────┘        │
│        │                                      │                 │
│        ▼                                      ▼                 │
│   ┌─────────┐     ┌─────────────┐     ┌──────────────┐        │
│   │表达式树│────▶│   DSL 语法   │────▶│  算子注册表  │        │
│   └─────────┘     └─────────────┘     └──────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 1. 因子 (Factor)

**因子**是一个数值函数，用于预测股票未来收益。

### 公式示例

| 因子名称 | 公式 | 含义 |
|---------|------|------|
| 均线偏离 | `$close / Mean($close, 20) - 1` | 价格相对均线的偏离 |
| 动量 | `Return($close, 20)` | 20日收益率 |
| 成交量异常 | `$volume / Mean($volume, 20)` | 量比 |

### 因子的结构

```python
@dataclass
class Factor:
    id: str              # 唯一标识 (如 "FM_001")
    name: str            # 名称 (如 "MA_20_Crossover")
    formula: str         # DSL 公式
    tree: ExpressionTree # 表达式树
    admitted_at: float   # 准入时间
    ic_series: np.ndarray # IC 时间序列
    stats: dict          # 统计指标
```

## 2. IC (Information Coefficient)

IC 是衡量因子预测能力的核心指标。

```
IC = Spearman_corr(因子值, 未来收益)
```

### 图示

```
时间 ─────────────────────────────────────────────────▶

因子值:  [0.5, 0.3, 0.8, -0.2, 0.6, ...]
            │    │    │    │    │
            ▼    ▼    ▼    ▼    ▼
未来收益: [0.4, 0.2, 0.7, -0.1, 0.5, ...]

IC = corr(因子值, 未来收益) ≈ 0.95  (强正相关)
```

### IC 解读

| IC 范围 | 预测能力 |
|---------|---------|
| \|IC\| < 0.02 | 弱，无意义 |
| 0.02 ≤ \|IC\| < 0.05 | 中等 |
| 0.05 ≤ \|IC\| < 0.1 | 较强 |
| \|IC\| ≥ 0.1 | 强 |

### ICIR (IC Irregularity)

ICIR = IC均值 / IC标准差，衡量 IC 的稳定性：

```
ICIR = mean(IC) / std(IC)
```

- ICIR > 0.5 被认为是高质量因子

## 3. 表达式树 (Expression Tree)

表达式树是因子的内部表示。

### 节点类型

```
Node (抽象基类)
├── LeafNode      # 市场数据引用 ($close, $volume)
├── ConstantNode  # 常量 (20, 0.5)
└── OperatorNode  # 算子应用
```

### 示例

表达式: `$close / Mean($close, 20) - 1`

```
                         div
                        /   \
                    Leaf    Mean
                   ($close)   /   \
                           Leaf  Constant
                          ($close)  (20)
                              \
                              neg
                                \
                             Constant(-1)
```

### 求值

```python
data = {
    "$close": np.array([[10, 11, 12, ...]]),  # (M, T) M=股票数, T=时间
}

tree.evaluate(data)  # 返回 (M, T) 的因子值
```

## 4. DSL (领域特定语言)

FactorMiner 使用 DSL 表达因子公式。

### 特征引用

```
$open   # 开盘价
$high   # 最高价
$low    # 最低价
$close  # 收盘价
$volume # 成交量
$amt    # 成交额
$vwap   # 成交量加权平均价
$returns # 收益率
```

### 函数调用

```
Mean($close, 20)    # 时间序列均值
CsRank($close)        # 横截面排名
IfElse(cond, a, b)     # 条件函数
```

### 算术运算

```
+ - * /           # 四则运算
abs log sqrt sign  # 一元运算
```

### 示例公式

```python
# 均线金叉/死叉
IfElse(
    Mean($close, 5) > Mean($close, 20),
    1,
    -1
)

# 量价背离
$close - Mean($close, 20) * ($volume / Mean($volume, 20) - 1)

# 波动率比率
Std($returns, 20) / Mean(Abs($returns), 20)
```

## 5. 算子 (Operator)

算子是表达式树中的操作。

### 算子分类

```
┌─────────────────────────────────────────────────────────────┐
│                      算子分类                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ARITHMETIC    │  +, -, *, /, abs, log, sqrt, pow        │
│  STATISTICAL   │  Mean, Std, Skew, Kurt        │
│  TIMESERIES    │  Return, Delta, Corr, TsRank    │
│  CROSS_SECTION │  CsRank, CsZscore, CsMean               │
│  SMOOTHING     │  ema, sma, wma                            │
│  REGRESSION    │  Reg, Slope, Resi                │
│  LOGICAL       │  IfElse, and_, or_, not_                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 签名类型

| 签名 | 说明 | 示例 |
|------|------|------|
| `TIME_SERIES_TO_TIME_SERIES` | 沿时间轴滚动 | `Mean($close, 20)` |
| `CROSS_SECTION_TO_CROSS_SECTION` | 截面运算 | `CsRank($close)` |
| `ELEMENT_WISE` | 点对点运算 | `abs($returns)` |
| `REDUCE_TIME` | 压缩时间轴 | `Sum($returns, 252)` |

## 6. 因子库 (Factor Library)

因子库是已准入因子的集合。

### 准入标准

```
因子 ──▶ IC > 0.02? ──▶ ICIR > 0.5? ──▶ 与库因子正交? ──▶ 准入
              │              │              │
             否             否             是
              ▼              ▼              ▼
           拒绝            拒绝         可替换
```

### 相关性管理

```
入库条件: |rho(新因子, 库因子)| < 0.5

如果相关:
- 新因子 IC 更高 → 可替换旧因子
- 新因子 IC 更低 → 拒绝
```

## 概念关系图

```
                    ┌─────────────┐
                    │   因子库    │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ 表达式树 │────▶│ DSL 语法 │────▶│ 算子注册表│
   └──────────┘     └──────────┘     └──────────┘
         │                                  │
         │         ┌──────────┐           │
         └────────▶│   IC/ICIR  │◀──────────┘
                   └──────────┘
```

## 下一步

- [架构概述](04-architecture.md) - 理解系统架构
- [表达式树详解](05-expression-tree.md) - 深入理解表达式树
