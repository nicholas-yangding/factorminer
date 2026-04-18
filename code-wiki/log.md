# Code Wiki Log

## 2026-04-18

### build | Initial wiki build

创建了 FactorMiner 仓库的代码知识库。

**创建的模块页面:**
- `modules/core.md` - 核心模块（表达式树、解析器、因子库、循环）
- `modules/agent.md` - Agent 模块（LLM 接口、因子生成器）
- `modules/operators.md` - 算子模块（60+ 算子）
- `modules/memory.md` - Memory 模块（经验记忆、知识图谱）
- `modules/evaluation.md` - Evaluation 模块（评估管道、指标）
- `modules/data.md` - Data 模块（数据加载、预处理）
- `modules/benchmark.md` - Benchmark 模块（基准测试）
- `modules/utils.md` - Utils 模块（配置、日志、可视化）

**创建的概念页面:**
- `concepts/expression-tree.md` - 表达式树
- `concepts/factor-generator.md` - 因子生成器（LLM 驱动）
- `concepts/ralph-loop.md` - Ralph 循环
- `concepts/experience-memory.md` - 经验记忆
- `concepts/factor-evaluation.md` - 因子评估
- `concepts/factor-library.md` - 因子库

**创建的设计页面:**
- `design/typed-dsl.md` - DSL 设计
- `design/dual-lane.md` - 双车道架构

**关键模块:**
- core: 表达式树、解析器、因子库、Ralph/Helix 循环
- agent: LLM 因子生成、提示构建、辩论
- operators: 60+ 算子覆盖 7 个类别
- memory: 经验记忆、知识图谱、嵌入
- evaluation: IC/ICIR 指标、多阶段评估管道

### fix | Fix broken links and add missing page

- 添加缺失的 `concepts/factor-generator.md` 概念页面
- 修复 `modules/data.md` 和 `modules/benchmark.md` 中的断链
- 更新 `index.md` 添加 factor-generator 链接
- 文档化双 FactorGenerator 架构（agent 层和 core 层）
