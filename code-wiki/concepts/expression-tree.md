# Expression Tree (表达式树)

## 概述
表达式树是 FactorMiner 中因子公式的核心数据结构。它是一个 DAG（有向无环图），节点分为三类：

1. **LeafNode**: 引用市场数据列（如 `$close`、`$volume`）
2. **ConstantNode**: 数值常量
3. **OperatorNode**: 算子应用，可有多个子节点

## 节点层次

```
Node (ABC)
├── LeafNode       # 市场数据特征
├── ConstantNode   # 数值常量
└── OperatorNode   # 算子应用
```

## 求值机制

```python
def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    data: Dict[str, np.ndarray] - 特征名到 (M, T) 数组的映射
    M = 股票数量, T = 时间步数
    返回: (M, T) 的结果数组
    """
```

## 示例

表达式 `$close / Mean($close, 20) - 1` 解析为：

```
OperatorNode(div)
├── LeafNode($close)
└── OperatorNode(Mean)
    ├── LeafNode($close)
    └── ConstantNode(20)
```

## 关键操作

| 方法 | 说明 |
|------|------|
| `evaluate(data)` | 在数据上求值 |
| `to_string()` | 序列化为 DSL 公式 |
| `depth()` | 树深度 |
| `size()` | 节点数量 |
| `clone()` | 深拷贝 |
| `leaf_features()` | 获取所有特征名 |
| `iter_nodes()` | 遍历所有节点 |

## DAG vs 二叉树

表达式树是 **DAG** 而非二叉树，因为：
- 算子可以有任意数量的子节点（如 `IfElse(cond, then, else)` 是 3 个子节点）
- 允许节点共享（如 `$close` 出现在多个位置）

## 与 SymPy 的关系

可选通过 `SymPyCanonicalizer` 将表达式规范化为标准形式，用于：
- 发现等价的表达式
- 消除冗余计算
- 简化存储

## 来源
- `factorminer/core/expression_tree.py`
- `factorminer/core/canonicalizer.py`
