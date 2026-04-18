# Experience Memory (经验记忆)

## 概述
经验记忆系统存储和管理因子挖掘过程中的经验教训，包括成功模式、失败模式、战略洞察。它使 Ralph Loop 能够从历史经验中学习。

## 数据结构

### SuccessPattern (成功模式)
```python
SuccessPattern(
    name="Higher Moment Regimes",
    description="Use Skew/Kurt as IfElse conditions...",
    template="IfElse(Skew($close, 20), <factor_a>, <factor_b>)",
    success_rate="High",
    example_factors=["HMR_001", "HMR_002"],
    occurrence_count=0
)
```

### ForbiddenDirection (禁止方向)
```python
ForbiddenDirection(
    name="Pure Price Ratio",
    description="Avoid simple price ratios without volume...",
    template="<avoid_pattern>",
    reason="Low IC in backtesting",
    occurrence_count=0
)
```

### StrategicInsight (战略洞察)
```python
StrategicInsight(
    topic="Regime Awareness",
    content="Market regimes affect factor effectiveness...",
    confidence=0.8,
    supporting_factors=["因子1", "因子2"]
)
```

## 生命周期

```
形成 (Formation) --> 检索 (Retrieval) --> 演化 (Evolution)
     |                   |                  |
 评估轨迹             上下文查询           反馈调整
     |                   |                  |
 新记忆 <--------------+-------------------> 更新
```

### Formation (形成)
从评估轨迹中提取新模式：
- 成功因子揭示有效模式
- 失败因子揭示陷阱

### Retrieval (Retrieval)
基于当前库状态检索相关记忆：
- 上下文感知检索
- 可选：知识图谱检索
- 可选：嵌入向量检索

### Evolution (演化)
根据反馈调整记忆：
- 增加成功模式的 occurrence_count
- 调整置信度
- 合并相似模式

## 论文内置模式

来自 FactorMiner Paper Table 4 & 5：
- Higher Moment Regimes
- PV Corr Interaction
- Volume-Weighted Features
- Mean Reversion Triggers
- Momentum Acceleration

## Phase 2 增强

### Knowledge Graph
- 因子作为节点
- 算子、概念作为关系
- 支持语义关系发现

### Embeddings
- FormulaEmbedder: 将因子公式编码为向量
- 支持语义相似度检索

## 来源
- `factorminer/memory/experience_memory.py`
- `factorminer/memory/memory_store.py`
- `factorminer/memory/formation.py`
- `factorminer/memory/evolution.py`
