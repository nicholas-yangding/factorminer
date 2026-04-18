# Dual-Lane Architecture (双车道架构)

## 概述
FactorMiner 采用双车道架构：Paper Reproduction Lane（论文复现）和 Helix Research Lane（研究扩展）。

## Paper Reproduction Lane

严格复现论文方法的基准车道：

```
RalphLoop
    |
    +--> 基础记忆检索
    +--> 基础评估管道
    +--> 严格准入 (IC/ICIR)
    +--> 基础因子库
```

### 特点
- 可复现性优先
- 严格的重计算
- 基准测试兼容
- 无实验性功能

## Helix Research Lane

扩展的研究车道，启用 Phase 2 特性：

```
HelixLoop (extends RalphLoop)
    |
    +--> KG 检索增强
    +--> Embedding 检索增强
    +--> 辩论生成
    +--> 规范化
    +--> Causal 验证
    +--> Regime 评估
    +--> Capacity 估计
    +--> Significance 检验
```

### 特点
- 实验性功能
- 更深入的验证
- 知识图谱能力
- 嵌入向量语义检索

## 配置切换

```bash
# Paper Lane
factorminer --cpu mine --mock -n 2 -b 8 -t 10

# Helix Lane
factorminer --cpu helix --mock --debate --canonicalize -n 2 -b 8 -t 10
```

## 流程对比

| 阶段 | Paper Lane | Helix Lane |
|------|-----------|------------|
| 记忆检索 | 基础 | + KG + Embedding |
| 生成 | 基础 | + 辩论 |
| 评估 | IC/ICIR | + Causal/Regime |
| 规范化 | 无 | SymPy |
| 验证 | 基础 | + Significance |

## 设计意图

- **隔离**: 严格基准不受实验干扰
- **探索**: 研究车道可自由尝试新想法
- **对比**: 同一框架下对比不同配置

## 来源
- `factorminer/core/ralph_loop.py`
- `factorminer/core/helix_loop.py`
- `configs/paper_repro.yaml`
- `configs/helix_research.yaml`
