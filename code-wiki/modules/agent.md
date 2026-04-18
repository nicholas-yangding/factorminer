# Agent 模块

## Purpose
Agent 模块负责 LLM 驱动的因子生成，包括提示构建、LLM 接口抽象、输出解析、多专家辩论。

## Key Classes/Functions

### LLM Interface (`agent/llm_interface.py`)
- `LLMProvider` (ABC): LLM 提供者抽象基类
- `MockProvider`: 用于测试的模拟 LLM
- 支持 OpenAI、Anthropic、Google Generative AI

### Factor Generator (`agent/factor_generator.py`)
- `FactorGenerator`: 核心因子生成智能体
  - `generate_batch()`: 生成一批候选因子
  - 注入记忆先验
  - 重试解析失败

### Prompt Builder (`agent/prompt_builder.py`)
- `PromptBuilder`: 构建 LLM 提示
  - 系统提示：DSL 语法、算子列表
  - 用户提示：记忆信号、库状态

### Output Parser (`agent/output_parser.py`)
- `CandidateFactor`: 解析后的候选因子数据结构
- `parse_llm_output()`: 解析 LLM 输出为候选因子列表

### Debate (`agent/debate.py`)
- `DebateOrchestrator`: 多专家辩论协调器
- 生成多方观点碰撞

### Specialists (`agent/specialists.py`)
- 领域专家智能体
- 统计专家、算子专家、领域专家

### Critic (`agent/critic.py`)
- `CriticAgent`: 评估候选因子质量

## 关系图

```
PromptBuilder --> LLMProvider --> FactorGenerator
                                      |
                                      v
                               OutputParser --> CandidateFactor
                                                       |
                                                       v
                                                  FactorLibrary
```

## 设计决策

- **Provider 抽象**: 支持多种 LLM 后端，便于切换和测试
- **记忆注入**: 通过 prompt 注入记忆先验，引导生成方向
- **重试机制**: 解析失败时自动重试生成

## 来源
- [Factor Generator](../concepts/factor-generator.md)
