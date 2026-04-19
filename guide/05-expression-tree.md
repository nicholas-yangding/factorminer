# 05 - 表达式树与 DSL

## 表达式树结构

### 节点层次

```
                    Node (ABC)
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    LeafNode     ConstantNode   OperatorNode
        │              │              │
   $close, $volume    20         +, -, *, /
                                     Mean, CsRank
```

### 代码结构

```python
# factorminer/core/expression_tree.py

class Node(ABC):
    """所有节点的抽象基类"""
    
    @abstractmethod
    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """在数据上求值"""
    
    @abstractmethod
    def to_string(self) -> str:
        """序列化为 DSL 字符串"""
    
    @abstractmethod
    def depth(self) -> int:
        """返回树深度"""
    
    @abstractmethod
    def size(self) -> int:
        """返回节点数量"""
```

### 节点类型

```python
class LeafNode(Node):
    """引用市场数据"""
    def __init__(self, feature_name: str):  # "$close", "$volume"
        ...

class ConstantNode(Node):
    """数值常量"""
    def __init__(self, value: float):
        ...

class OperatorNode(Node):
    """算子应用（可多子节点）"""
    def __init__(self, operator, children: List[Node]):
        self.operator = operator
        self.children = children
```

## DSL 语法

### 特征

| 特征 | 说明 | 示例 |
|------|------|------|
| `$open` | 开盘价 | `$open` |
| `$high` | 最高价 | `$high` |
| `$low` | 最低价 | `$low` |
| `$close` | 收盘价 | `$close` |
| `$volume` | 成交量 | `$volume` |
| `$amt` | 成交额 | `$amt` |
| `$vwap` | 成交量加权均价 | `$vwap` |
| `$returns` | 收益率 | `$returns` |

### 算子函数

```
Mean(x, window)      # 时间序列均值
Std(x, window)       # 时间序列标准差
Return(x, period)  # 收益率
Delta(x, period)     # 变化量
Corr(x, y, window)   # 时间序列相关性
TsRank(x, window)      # 时间序列排名
CsRank(x)              # 横截面排名
CsZscore(x)            # 横截面标准化
ema(x, span)            # 指数移动平均
IfElse(cond, then, else)  # 条件
```

### 算术运算

```
+ - * /               # 四则运算
abs(x)                # 绝对值
log(x)                # 对数
sqrt(x)               # 平方根
sign(x)               # 符号
pow(x, n)             # 幂
neg(x)                # 相反数
```

## 解析示例

### 简单公式

公式: `$close`

```
LeafNode($close)
```

### 算术公式

公式: `$close + $volume`

```
OperatorNode(add)
├── LeafNode($close)
└── LeafNode($volume)
```

### 函数公式

公式: `Mean($close, 20)`

```
OperatorNode(Mean)
├── LeafNode($close)
└── ConstantNode(20)
```

### 复合公式

公式: `$close / Mean($close, 20) - 1`

```
OperatorNode(sub)
├── OperatorNode(div)
│   ├── LeafNode($close)
│   └── OperatorNode(Mean)
│       ├── LeafNode($close)
│       └── ConstantNode(20)
└── ConstantNode(1)
```

## 求值过程

### 数据格式

```python
# 数据格式: Dict[str, np.ndarray]
# shape: (M, T) 其中 M=股票数, T=时间步

data = {
    "$close": np.array([
        [10.0, 11.0, 12.0, 13.0],   # 股票1的收盘价
        [20.0, 21.0, 22.0, 23.0],   # 股票2的收盘价
    ]),  # shape: (2, 4)
    "$volume": np.array([
        [100, 110, 120, 130],
        [200, 210, 220, 230],
    ]),
    ...
}
```

### 求值流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     evaluate(data)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   data ──▶ LeafNode($close) ──▶ 获取列数据 (M,T)         │
│                │                                          │
│                ▼                                          │
│   ┌─────────────────────────────────────────────────┐    │
│   │  OperatorNode(Mean)                           │    │
│   │  ├── children[0] = LeafNode($close) ──▶ (M,T)  │    │
│   │  └── children[1] = ConstantNode(20)            │    │
│   │                                                  │    │
│   │  计算: np.mean((M,T), axis=-1, keepdims=True)   │    │
│   │  输出: (M,1) → 广播到 (M,T)                      │    │
│   └─────────────────────────────────────────────────┘    │
│                         │                                  │
│                         ▼                                  │
│   ┌─────────────────────────────────────────────────┐    │
│   │  OperatorNode(div)                               │    │
│   │  ├── left = LeafNode($close) ──▶ (M,T)          │    │
│   │  └── right = Mean 输出 ──▶ (M,T)             │    │
│   │                                                  │    │
│   │  计算: (M,T) / (M,T) → (M,T)                    │    │
│   └─────────────────────────────────────────────────┘    │
│                         │                                  │
│                         ▼                                  │
│                      (M, T) 因子矩阵                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Parser 使用

```python
from factorminer.core.parser import parse

# 解析 DSL 字符串
tree = parse("$close / Mean($close, 20) - 1")

# 获取 DSL 字符串
print(tree.to_string())
# "$close / Mean($close, 20) - 1"

# 求值
data = {"$close": np.array([[10, 11, 12], [20, 21, 22]])}
result = tree.evaluate(data)
# shape: (2, 3)
```

## 树操作

```python
# 深度
tree.depth()  # 如: 4

# 节点数
tree.size()   # 如: 7

# 克隆
tree_clone = tree.clone()

# 遍历节点
for node in tree.iter_nodes():
    print(node.to_string())

# 获取所有特征
features = tree.leaf_features()
# ["$close", "$volume"]
```

## DAG vs 二叉树

表达式树是 **DAG（有向无环图）** 而非二叉树：

### 二叉树限制

```
        op
       /   \
      a     op
           / \
          b   c
```

### DAG 优势

```
        op1
       /   \
      a     op2
            / \
           b   c
           
同一叶子节点可被多个父节点引用（不重复存储）
```

## SymPy 规范化

可选使用 SymPy 规范化表达式：

```python
from factorminer.core.canonicalizer import SymPyCanonicalizer

canonicalizer = SymPyCanonicalizer()
canonical_tree = canonicalizer.canonicalize(tree)

# 发现等价表达式
# log(x/y) - log(x) + log(y)
# 可能被规范化为 2*log(sqrt(x/y))
```

## 下一步

- [算子参考](06-operators.md) - 60+ 算子详解
- [Ralph Loop](07-ralph-loop.md) - 挖掘循环
