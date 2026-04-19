# Typed DSL (类型化领域特定语言)

## 概述
FactorMiner 使用类型化的 DSL 表达因子公式。DSL 是表达式树的文本表示，经过 Parser 解析后生成表达式树。

## 语法

### 特征引用
以 `$` 开头：
```
$open, $high, $low, $close, $volume, $amt, $vwap, $returns
```

### 函数调用
```
Mean($close, 20)        # 时间序列函数
CsRank($close)            # 横截面函数
IfElse(cond, then, else)    # 条件函数
```

### 算术运算
```
a + b, a - b, a * b, a / b
abs(x), log(x), sqrt(x), sign(x)
```

### 逻辑运算
```
IfElse(cond, then, else)
a > b, a < b, a == b
and_(a, b), or_(a, b), not_(a)
```

## 类型系统

### 基础类型
- `Number`: 数值
- `Boolean`: 布尔值
- `Series`: 时间序列 (M, T)

### 表达式类型
表达式树节点有类型：
- LeafNode: 从特征读取，类型由特征定义
- ConstantNode: Number
- OperatorNode: 由算子定义

### 类型检查
Parser 和 Evaluator 进行类型检查：
- 参数类型必须匹配
- 返回类型必须兼容上下文

## Parser 实现

```python
# core/parser.py
def parse(expr: str) -> ExpressionTree:
    """将 DSL 字符串解析为表达式树"""
    # 1. Tokenize
    # 2. Build AST
    # 3. Type check
    # 4. Return ExpressionTree
```

## 示例

| DSL 表达式 | 含义 |
|-----------|------|
| `$close` | 收盘价序列 |
| `Mean($close, 20)` | 20 日收盘价均线 |
| `$close / Mean($close, 20) - 1` | 收盘价相对均线的偏离 |
| `CsRank($close)` | 收盘价的横截面排名 |
| `IfElse($returns > 0, 1, -1)` | 收益率为正则 1，否则 -1 |

## 设计决策

1. **类型化**: 静态类型检查减少运行时错误
2. **显式特征**: 所有数据列必须显式引用，便于追踪
3. **函数式**: 纯函数风格，无副作用
4. **可组合**: 算子可嵌套任意深度

## 来源
- `factorminer/core/parser.py`
- `factorminer/core/types.py`
