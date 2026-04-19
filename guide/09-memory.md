# 09 - 经验记忆系统

## 经验记忆概述

经验记忆使 Ralph Loop 能够从历史挖掘经验中学习：

```
┌─────────────────────────────────────────────────────────────────┐
│                      经验记忆系统                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     ┌─────────────────────────────────────────────────────┐     │
│     │                  记忆生命周期                         │     │
│     │                                                   │     │
│     │  ┌─────────┐    ┌─────────┐    ┌─────────┐      │     │
│     │  │ Formation│───▶│Retrieval│───▶│Evolution│      │     │
│     │  │  形成   │    │  检索   │    │  演化   │      │     │
│     │  └─────────┘    └─────────┘    └─────────┘      │     │
│     │       │              │              │             │     │
│     │       ▼              ▼              ▼             │     │
│     │  评估轨迹 ────▶  上下文查询  ────▶  反馈调整    │     │
│     │                                                   │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 记忆数据类型

### 1. SuccessPattern (成功模式)

```python
@dataclass
class SuccessPattern:
    name: str              # "Higher Moment Regimes"
    description: str        # 模式描述
    template: str          # 模板公式
    success_rate: str     # "High", "Medium", "Low"
    example_factors: List[str]
    occurrence_count: int  # 出现次数
```

**示例:**

```python
SuccessPattern(
    name="PV Corr Interaction",
    description="Price-volume correlation interaction...",
    template="CsRank(Corr($close, $volume, 20))",
    success_rate="High",
    example_factors=["PVC_001", "PVC_002"],
    occurrence_count=5
)
```

### 2. ForbiddenDirection (禁止方向)

```python
@dataclass
class ForbiddenDirection:
    name: str              # "Pure Price Ratio"
    description: str        # 为什么不work
    template: str           # 避免的模式
    reason: str            # 原因
    occurrence_count: int
```

### 3. StrategicInsight (战略洞察)

```python
@dataclass
class StrategicInsight:
    topic: str            # "Regime Awareness"
    content: str          # 洞察内容
    confidence: float     # 0.0 - 1.0
    supporting_factors: List[str]
```

## 检索流程

```python
def retrieve_memory(library_state):
    """
    基于当前库状态检索相关记忆
    """
    # 1. 分析库状态
    domain_saturation = analyze_library_saturation(library_state)
    
    # 2. 检索推荐方向
    recommended = []
    for pattern in memory.success_patterns:
        if pattern.relevant_to(domain_saturation):
            recommended.append(pattern)
    
    # 3. 检索禁止方向
    forbidden = memory.forbidden_directions
    
    # 4. 检索战略洞察
    insights = memory.strategic_insights
    
    return {
        "recommended_directions": recommended,
        "forbidden_directions": forbidden,
        "strategic_insights": insights,
        "domain_saturation": domain_saturation
    }
```

## 形成流程 (Formation)

从评估轨迹形成新记忆：

```python
def form_memory(trajectory):
    """
    从评估轨迹提取新模式
    """
    # 1. 分析成功因子
    for factor in trajectory.admitted:
        # 发现新模式
        pattern = extract_pattern(factor)
        if is_novel(pattern):
            memory.add_success_pattern(pattern)
    
    # 2. 分析失败因子
    for factor in trajectory.rejected:
        # 发现陷阱
        pitfall = extract_pitfall(factor)
        if is_novel(pitfall):
            memory.add_forbidden_direction(pitfall)
```

## 演化流程 (Evolution)

根据反馈演化记忆：

```python
def evolve_memory(memory, feedback):
    """
    调整记忆权重
    """
    # 1. 更新成功模式的 occurrence_count
    for pattern in memory.success_patterns:
        if pattern.name in feedback.successful_patterns:
            pattern.occurrence_count += 1
            # 调整成功率评估
            pattern.success_rate = calculate_success_rate(pattern)
    
    # 2. 合并相似模式
    memory.consolidate_similar_patterns()
    
    # 3. 移除过时模式
    memory.remove_stale_patterns()
```

## 知识图谱 (Phase 2)

可选的增强检索：

```python
class FactorKnowledgeGraph:
    """
    因子知识图谱
    """
    def add_factor(self, factor):
        # 添加因子节点
        node = FactorNode(factor.id, factor.formula)
        self.add_node(node)
        
        # 添加关系
        for op in extract_operators(factor):
            self.add_edge(factor.id, op, "uses")
        
        for feature in extract_features(factor):
            self.add_edge(factor.id, feature, "depends_on")
    
    def query_related(self, factor_id, depth=2):
        """查询相关因子"""
        return self.bfs(factor_id, depth)
```

## 嵌入检索 (Phase 2)

基于语义相似度检索：

```python
class FormulaEmbedder:
    """
    公式嵌入器
    """
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed(self, formula: str) -> np.ndarray:
        """将公式编码为向量"""
        return self.model.encode(formula)
    
    def find_similar(self, formula: str, top_k=5):
        """查找相似公式"""
        query_vec = self.embed(formula)
        scores = cosine_similarity(query_vec, self.index)
        return self.formulas[np.argsort(scores)[-top_k:]]
```

## 记忆检索信号

注入到 LLM 提示的信号格式：

```python
memory_signal = {
    "recommended_directions": [
        "Use higher moment regimes (Skew/Kurt) for regime switching",
        "Price-volume correlation interaction captures supply-demand imbalance",
        "Mean reversion triggers work well in sideways markets"
    ],
    "forbidden_directions": [
        "Avoid pure price ratios without volume confirmation",
        "Don't use very short windows (< 5) for stable estimation"
    ],
    "strategic_insights": [
        "Market regimes affect factor effectiveness significantly",
        "IC decay over horizons is a key indicator"
    ],
    "recent_rejections": [
        {"formula": "simple_ratio", "reason": "Low IC in backtesting"},
        {"formula": "complex_nested", "reason": "Computation unstable"}
    ]
}
```

## 双车道差异

| 功能 | Paper Lane | Helix Lane |
|------|------------|------------|
| 检索 | 基础检索 | + KG 检索 + 嵌入检索 |
| 形成 | 基础形成 | + 因果验证 |
| 演化 | 基础演化 | + 置信度追踪 |

## 下一步

- [Helix Loop](10-helix-loop.md) - Phase 2 详解
- [CLI 参考](11-cli-reference.md) - 命令行用法
