# FactorMiner Code Wiki

基于 FactorMiner 论文实现的代码知识库。

## 项目概述

**FactorMiner** 是一个 LLM 驱动的因子挖掘框架，使用经验记忆和自进化算法发现可解释的 Alpha 因子。

- 论文: *FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery* (Wang et al., 2026)
- 110+ 内置论文因子
- 60+ 算子
- Python 3.10+

## Modules (模块)

### 核心模块
- [core](modules/core.md) - 表达式树、解析器、因子库、Ralph/Helix 循环

### Agent 模块
- [agent](modules/agent.md) - LLM 接口、因子生成器、提示构建、辩论

### Operators 模块
- [operators](modules/operators.md) - 60+ 算子（算术、统计、时间序列、横截面、回归、逻辑）

### Memory 模块
- [memory](modules/memory.md) - 经验记忆、知识图谱、嵌入向量检索

### Evaluation 模块
- [evaluation](modules/evaluation.md) - 评估管道、指标、相关性、回测、组合

### Data 模块
- [data](modules/data.md) - 数据加载、预处理、合成数据

### Benchmark 模块
- [benchmark](modules/benchmark.md) - 基准测试、消融研究

### Utils 模块
- [utils](modules/utils.md) - 配置、日志、可视化

## Concepts (概念)

- [expression-tree](concepts/expression-tree.md) - 表达式树数据结构
- [factor-generator](concepts/factor-generator.md) - LLM 因子生成器
- [ralph-loop](concepts/ralph-loop.md) - Ralph 挖掘循环
- [experience-memory](concepts/experience-memory.md) - 经验记忆系统
- [factor-evaluation](concepts/factor-evaluation.md) - 因子评估流程
- [factor-library](concepts/factor-library.md) - 因子库管理

## Design (设计)

- [typed-dsl](design/typed-dsl.md) - 类型化领域特定语言
- [dual-lane](design/dual-lane.md) - 双车道架构（Paper Lane vs Helix Lane）

## 目录结构

```
factorminer/
├── agent/              # LLM 智能体
├── benchmark/          # 基准测试
├── configs/           # 配置文件
├── core/              # 核心（表达式树、循环、库）
├── data/              # 数据处理
├── evaluation/        # 评估管道
├── memory/            # 经验记忆
├── operators/         # 算子注册表
├── tests/             # 测试
└── utils/             # 工具
```

## 关键命令

```bash
# 挖掘因子
factorminer --cpu mine --mock -n 2 -b 8 -t 10

# Helix 循环
factorminer --cpu helix --mock --debate --canonicalize

# 评估因子库
factorminer --cpu evaluate output/factor_library.json --mock

# 基准测试
factorminer --cpu benchmark --config configs/paper_repro.yaml
```
