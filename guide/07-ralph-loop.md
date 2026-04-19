# 07 - Ralph Loop (挖掘循环)

## Ralph Loop 概述

Ralph Loop 是 FactorMiner 的核心挖掘算法，实现自进化的因子发现：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Ralph Loop 流程图                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     ┌─────────────────────────────────────────────────────┐     │
│     │                    初始化                            │     │
│     │  • 加载配置                                          │     │
│     │  • 初始化因子库                                      │     │
│     │  • 初始化经验记忆                                    │     │
│     └────────────────────────┬────────────────────────┘     │
│                              │                                  │
│              ┌───────────────┼───────────────┐               │
│              ▼               ▼               ▼               │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│     │  R(M, L)  │  │  G(m, L)   │  │ E(candidates)│      │
│     │  记忆检索  │  │  LLM生成   │  │  多阶段评估  │        │
│     └─────┬──────┘  └──────┬─────┘  └──────┬──────┘        │
│           │                 │                 │               │
│           └─────────────────┼─────────────────┘               │
│                             ▼                                 │
│                  ┌──────────────────────┐                    │
│                  │    库更新           │                    │
│                  │  L ← L + {admitted}│                    │
│                  └──────────┬─────────┘                    │
│                             │                                │
│                             ▼                                 │
│                  ┌──────────────────────┐                    │
│                  │    记忆演化         │                    │
│                  │  E(M, F(M, τ))     │                    │
│                  └──────────┬─────────┘                    │
│                             │                                │
│                             ▼                                 │
│                  ┌──────────────────────┐                    │
│                  │   终止判断          │                    │
│                  │ |L| >= K or iter >= max│                   │
│                  └──────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 五个核心阶段

### Stage 1: 记忆检索 R(M, L)

从经验记忆检索先验知识：

```python
memory_signal = memory.retrieve(library_state)
# 返回:
# {
#     "recommended_directions": [...],   # 推荐方向
#     "forbidden_directions": [...],     # 禁止方向
#     "strategic_insights": [...],       # 战略洞察
#     "recent_rejections": [...]         # 最近拒绝原因
# }
```

### Stage 2: LLM 生成 G(m, L)

使用 LLM 生成候选因子：

```python
candidates = generator.generate_batch(
    memory_signal=memory_signal,
    library_state=library_state,
    batch_size=40
)
```

### Stage 3: 多阶段评估 E(candidates)

```
候选因子 ──▶ Stage 1 ──▶ Stage 2 ──▶ Stage 2.5 ──▶ Stage 3 ──▶ Stage 4
  │          │          │           │           │          │
  │         IC        Correlation  Replacement  Dedup     Full IC
  │        筛选        检查         检查        去重       验证
```

#### Stage 1: Fast IC Screening
- 快速计算 IC
- 仅用 100 个资产
- 阈值: IC > 0.04

#### Stage 2: Correlation Check
- 与库中因子相关性检查
- 阈值: |rho| < 0.5

#### Stage 2.5: Replacement Check
- 如果相关但 IC 更高
- 可替换旧因子

#### Stage 3: Intra-batch Dedup
- Batch 内去重
- Pairwise |rho| < 0.5

#### Stage 4: Full Validation
- 全量资产验证
- 多时间框架评估

### Stage 4: 库更新 L ← L + {α}

```python
for candidate in admitted:
    library.add(candidate)
```

### Stage 5: 记忆演化 E(M, F(M, τ))

```python
memory.update(trajectory)
# trajectory: 包含 IC、拒绝原因、准入因子等信息
```

## BudgetTracker

追踪资源消耗：

```python
@dataclass
class BudgetTracker:
    max_llm_calls: int = 0       # 0 = unlimited
    max_wall_seconds: float = 0    # 0 = unlimited
    
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    compute_seconds: float = 0.0
```

## 配置参数

```yaml
mining:
  target_library_size: 110   # 目标库大小
  batch_size: 40            # 每轮生成数
  max_iterations: 200       # 最大迭代
  ic_threshold: 0.04       # IC 阈值
  icir_threshold: 0.5       # ICIR 阈值
  correlation_threshold: 0.5 # 相关性阈值
```

## 循环终止条件

```
|L| >= K          # 库达到目标大小
OR
iterations >= max  # 达到最大迭代
OR
budget exhausted   # 资源耗尽
```

## 伪代码

```python
def run_ralph_loop(config):
    # 初始化
    library = FactorLibrary()
    memory = ExperienceMemoryManager()
    generator = FactorGenerator(llm_provider)
    
    while len(library) < config.target_library_size:
        # 1. 记忆检索
        memory_signal = memory.retrieve(library.state)
        
        # 2. LLM 生成
        candidates = generator.generate_batch(memory_signal)
        
        # 3. 评估
        admitted = evaluate_pipeline(candidates, library)
        
        # 4. 入库
        for factor in admitted:
            library.add(factor)
        
        # 5. 记忆演化
        memory.update(admitted, rejected)
    
    return library
```

## 输出示例

```
============================================================
FactorMiner -- Mining Session
============================================================
  Target library size: 110
  Batch size:          40
  Max iterations:      200
  IC threshold:        0.04
  Correlation limit:   0.5
------------------------------------------------------------
2024-04-18 10:30:15 [INFO] Starting iteration 1
2024-04-18 10:30:16 [INFO] Generated 40 candidates
2024-04-18 10:30:17 [INFO] Stage 1: 25 passed IC screen
2024-04-18 10:30:18 [INFO] Stage 2: 20 passed correlation
2024-04-18 10:30:19 [INFO] Stage 3: 18 passed deduplication
2024-04-18 10:30:20 [INFO] Stage 4: 15 admitted
2024-04-18 10:30:21 [INFO] Library size: 15/110
2024-04-18 10:30:22 [INFO] Memory updated
------------------------------------------------------------
...
2024-04-18 11:45:00 [INFO] Target reached! Library size: 110/110
============================================================
```

## 下一步

- [评估流程](08-evaluation.md) - 因子评估详解
- [经验记忆](09-memory.md) - 记忆系统
