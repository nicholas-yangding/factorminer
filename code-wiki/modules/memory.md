# Memory 模块

## Purpose
Memory 模块实现经验记忆系统，包括记忆存储、检索、形成和演化。Phase 2 还支持知识图谱和嵌入向量检索。

## 核心组件

### Experience Memory (`memory/experience_memory.py`)
- `ExperienceMemoryManager`: 记忆管理器主类
  - `retrieve()`: 上下文相关检索
  - `update()`: 形成 + 演化记忆
  - 内置论文 Table 4/5 的默认模式

### Memory Store (`memory/memory_store.py`)
- `ExperienceMemory`: 记忆存储数据结构
- `SuccessPattern`: 成功模式（名称、描述、模板、成功率）
- `ForbiddenDirection`: 禁止方向
- `StrategicInsight`: 战略洞察

### Retrieval (`memory/retrieval.py`)
- `retrieve_memory()`: 基础检索

### KG Retrieval (`memory/kg_retrieval.py`)
- `retrieve_memory_enhanced()`: 增强检索（KG + Embeddings + 基础记忆）
- 结合知识图谱和嵌入向量的高级检索
- 用于 Helix Lane

### Knowledge Graph (`memory/knowledge_graph.py`)
- `FactorKnowledgeGraph`: 因子知识图谱
- `FactorNode`: 图节点（因子、算子、概念）
- 用于发现因子间的语义关系

### Embeddings (`memory/embeddings.py`)
- `FormulaEmbedder`: 公式嵌入器
- 基于 sentence-transformers
- 支持语义相似度检索

### Formation (`memory/formation.py`)
- `form_memory()`: 从评估轨迹形成新记忆

### Evolution (`memory/evolution.py`)
- `evolve_memory()`: 演化现有记忆
- 根据反馈调整成功模式

### Online Regime Memory (`memory/online_regime_memory.py`)
- `OnlineRegimeMemory`: 在线市场状态记忆

## 关系图

```
ExperienceMemoryManager
    |
    +---> MemoryStore (存储)
    +---> Retrieval (检索)
    |         |
    |         +---> KGRetrieval (知识图谱)
    |         +---> Embeddings (嵌入)
    |
    +---> Formation (形成)
    +---> Evolution (演化)
```

## 设计决策

- **双检索路径**: 基础检索用于 Paper Lane，增强检索用于 Helix Lane
- **延迟加载**: 知识图谱和嵌入为可选依赖
- **默认模式**: 内置论文验证过的成功/失败模式

## 来源
- [Experience Memory 设计](../concepts/experience-memory.md)
