# Ralph Loop (Ralph 循环)

## 概述
Ralph Loop 是 FactorMiner 的核心挖掘算法，实现论文 Algorithm 1。它是一个自我进化的因子发现循环，交替进行记忆检索、LLM 生成、评估、入库。

## 算法流程

```
while |L| < K and iterations < max:
    1. R(M, L)     <- 检索记忆先验
    2. G(m, L)     <- LLM 生成候选因子
    3. E(candidates) <- 多阶段评估
    4. L <- L + {admitted}  <- 更新库
    5. E(M, F(M, τ)) <- 演化记忆
```

## 详细阶段

### Stage 1: 记忆检索 (R)
从经验记忆系统中检索：
- 推荐方向 (recommended_directions)
- 禁止方向 (forbidden_directions)
- 战略洞察 (strategic_insights)
- 最近拒绝原因 (recent_rejections)

### Stage 2: LLM 生成 (G)
FactorGenerator 基于记忆先验生成候选因子：
- 构建提示（DSL 语法 + 记忆注入）
- 调用 LLM
- 解析输出为 CandidateFactor 列表

### Stage 3: 多阶段评估 (E)

```
候选因子
    |
    v
Stage 1: Fast IC screening on M_fast assets
    | (IC > threshold)
    v
Stage 2: Correlation check against library L
    | (not correlated with existing)
    v
Stage 2.5: Replacement check for correlated candidates
    | (better IC than existing)
    v
Stage 3: Intra-batch deduplication (pairwise rho < θ)
    |
    v
Stage 4: Full validation on M_full + trajectory collection
    |
    v
Admission
```

### Stage 4: 库更新 (L <- L + {α})
 admitted 因子加入因子库

### Stage 5: 记忆演化 (E)
根据新因子的表现更新经验记忆

## BudgetTracker

追踪资源消耗：
- LLM 调用次数和 token 数量
- GPU 计算时间
-  wall-clock 时间

支持早停：当预算耗尽时主动停止。

## 双车道架构

| 特性 | Paper Lane | Helix Lane |
|------|-----------|------------|
| 循环 | RalphLoop | HelixLoop |
| 评估 | 基础 | 增强（辩论、规范化） |
| 记忆 | 基础检索 | KG + Embedding |
| 准入 | IC/ICIR | + Causal/Regime |

## 来源
- `factorminer/core/ralph_loop.py`
- `factorminer/core/helix_loop.py`
