# Neuro-Symbolic Operators (神经符号算子)

## 概述
神经符号算子是 Phase 2 Helix 的核心创新，融合符号表达式的可解释性与神经网络的非线性建模能力。

## 为什么需要神经符号算子

**符号表达式的局限:**
- 依赖手工编码的算子词汇表
- 难以捕捉复杂的非线性模式
- 价格-成交量在特定市场状态下的分歧

**神经叶片的优势:**
- tiny MLP (< 5000 参数) 训练于历史数据
- 发现 hand-written 公式无法捕捉的模式
- 保持 (M, T) 输入输出形状

## 工作流程

```
1. 训练 NeuralLeaf
   输入: F 特征 x W 时间步 的滚动窗口
   输出: (M, T) 标量信号
   损失: 可微分 Pearson-IC 代理损失

2. 插入表达式树
   NeuralLeafNode 行为与任何算子相同

3. 验证
   在验证集上评估 IC

4. 蒸馏到符号
   distill_to_symbolic() 找到最接近的符号公式

5. 生产替换
   用蒸馏公式替换 NeuralLeafNode
```

## 架构约束

- **参数限制**: < 5,000 参数（适合 CPU，快速推理）
- **网络结构**: 2 层 MLP
  - input -> 32 hidden -> 1
  - LayerNorm + GELU
- **输入**: F 特征 x W 时间步 的展平窗口
- **输出**: (M, T) 形状

## NeuralLeafNode

```python
class NeuralLeafNode(Node):
    """神经网络叶子节点"""
    
    def __init__(
        self,
        input_features: List[str],      # ["$close", "$volume", ...]
        window_size: int = 20,           # 时间窗口
        hidden_size: int = 32,
    ):
        self.mlp = SmallMLP(input_features, window_size, hidden_size)
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        # 将市场数据转换为张量
        # 前向传播
        # 返回 (M, T) 信号
```

## 蒸馏过程

```python
def distill_to_symbolic(
    neural_leaf: NeuralLeafNode,
    operator_registry: Dict[str, OperatorSpec],
    n_samples: int = 1000
) -> ExpressionTree:
    """
    找到现有算子库中最接近神经叶片的符号公式
    返回: 蒸馏后的表达式树
    """
```

## 完整示例

```python
# 1. 训练神经叶片
neural_leaf = NeuralLeafNode(
    input_features=["$close", "$volume", "$vwap"],
    window_size=20,
    hidden_size=32
)
neural_leaf.train(market_data, target_returns, epochs=100)

# 2. 插入表达式树
tree = OperatorNode("mul",
    neural_leaf,  # 神经符号节点
    OperatorNode("ts_mean", LeafNode("$close"), ConstantNode(20))
)

# 3. 评估
signals = tree.evaluate(data)

# 4. 蒸馏
symbolic_formula = distill_to_symbolic(neural_leaf, OPERATOR_REGISTRY)
```

## 损失函数

可微分 Pearson-IC 代理损失：

```python
def pearson_ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    最大化 Pearson 相关系数的可微分代理
    """
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    
    correlation = (pred_centered * target_centered).mean()
    norm = pred_centered.std() * target_centered.std()
    
    return -correlation / (norm + 1e-8)  # 负号：因为是最小化
```

## 与 Helix Loop 的集成

HelixLoop 支持在生成循环中使用神经符号算子：

```python
class HelixLoop(RalphLoop):
    def __init__(self, ..., use_neuro_symbolic: bool = True):
        self.use_neuro_symbolic = use_neuro_symbolic
    
    def generate_candidates(self):
        if self.use_neuro_symbolic:
            # 可以生成包含 NeuralLeafNode 的候选
            ...
```

## 来源
- `factorminer/operators/neuro_symbolic.py`
