# Factor Generator (因子生成器)

## 概述
FactorGenerator 是 LLM 驱动的因子生成智能体，负责构建提示、调用 LLM、解析输出为候选因子。

## 双生成器架构

FactorMiner 有两个 FactorGenerator：

### 1. Agent FactorGenerator (`agent/factor_generator.py`)
位于 agent 层，负责单批次生成：

```python
class FactorGenerator:
    def generate_batch(
        self,
        memory_signal: Optional[Dict[str, Any]] = None,
        library_state: Optional[Dict[str, Any]] = None,
        batch_size: int = 40,
    ) -> List[CandidateFactor]:
```

**职责:**
- 构建包含记忆先验的提示
- 调用 LLM 提供者
- 解析 LLM 输出
- 重试解析失败的情况

### 2. Core FactorGenerator (`core/ralph_loop.py`)
位于 core 层，是 RalphLoop 的内部组件：

```python
# ralph_loop.py 中的内部类
class FactorGenerator:
    def generate(self, batch_size: int) -> List[CandidateFactor]:
```

**职责:**
- 协调整个生成流程
- 与经验记忆交互
- 管理重试和验证

## 生成流程

```
1. 构建提示
   - PromptBuilder 生成系统提示（DSL 语法、算子）
   - 注入 memory_signal（推荐/禁止方向、战略洞察）
   - 注入 library_state（当前库大小、最近准入）

2. 调用 LLM
   - LLMProvider 生成候选公式
   - 支持 temperature、max_tokens 控制

3. 解析输出
   - OutputParser 解析为 CandidateFactor 列表
   - 验证公式语法
   - 重试失败的解析

4. 返回候选因子
```

## PromptBuilder

构建两类提示：

### 系统提示
```
你是一个因子挖掘专家。请根据以下 DSL 语法生成因子公式...

支持的算子：ts_mean, cs_rank, IfElse, ...

要求：
1. 因子必须有可解释的金融含义
2. 使用 1-3 个算子组合
3. 避免过于复杂的嵌套
```

### 用户提示
```
当前记忆先验：
- 推荐方向: [方向列表]
- 禁止方向: [方向列表]
- 战略洞察: [洞察列表]

当前因子库状态：
- 库大小: N/M
- 最近准入: [因子列表]

请生成 batch_size 个候选因子...
```

## 输出格式

解析后的 CandidateFactor：
```python
@dataclass
class CandidateFactor:
    formula: str           # DSL 公式
    name: str             # 因子名称
    description: str       # 金融含义描述
    expected_direction: str # 预期 IC 方向（正/负）
    confidence: float      # 生成置信度
```

## 重试机制

解析失败时自动重试：
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        candidates = parse_llm_output(response)
        return candidates
    except ParseError:
        if attempt < max_retries - 1:
            continue
        raise
```

## 与记忆的交互

```
Memory --> FactorGenerator --> RalphLoop
    |              |
    |<-- retrieve -|--> admission --> Library
```

- 生成前：检索记忆先验注入提示
- 生成后：记忆系统学习新因子的表现

## 来源
- `factorminer/agent/factor_generator.py`
- `factorminer/agent/prompt_builder.py`
- `factorminer/agent/output_parser.py`
- `factorminer/core/ralph_loop.py`
