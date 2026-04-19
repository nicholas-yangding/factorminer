# Core 模块

## Purpose
Core 模块是 FactorMiner 的核心引擎，负责表达式树、DSL 解析器、因子库管理、主循环（ Ralph Loop 和 Helix Loop ）。

## Key Classes/Functions

### Expression Tree (`core/expression_tree.py`)
- `Node` (ABC): 表达式树节点基类
- `LeafNode`: 引用市场数据列（如 `$close`）
- `ConstantNode`: 数值常量节点
- `OperatorNode`: 算子应用节点
- `ExpressionTree`: 完整表达式树容器

### Parser (`core/parser.py`)
- `parse()`: 将 DSL 公式解析为表达式树
- `try_parse()`: 解析失败时返回 None

### Factor Library (`core/factor_library.py`)
- `Factor`: 单个因子（ID、名称、表达式树、元数据）
- `FactorLibrary`: 因子库管理器（添加、查询、相关性管理）

### Ralph Loop (`core/ralph_loop.py`)
- `RalphLoop`: 主挖掘循环实现
  - 迭代生成候选因子
  - 多阶段评估管道
  - 因子入库管理
  - 经验记忆更新
- `BudgetTracker`: 资源消耗追踪

### Helix Loop (`core/helix_loop.py`)
- `HelixLoop(RalphLoop)`: Phase 2 增强循环
  - 辩论生成
  - 规范化
  - 知识图谱检索增强

### Library I/O (`core/library_io.py`)
- `save_library()`: 保存因子库到 JSON + NPZ
- `load_library()`: 加载因子库
- `build_factor_miner_catalog()`: 内置 110 个论文因子

### Session (`core/session.py`)
- `MiningSession`: 挖掘会话管理
  - `session_id`: 唯一标识
  - `record_iteration()`: 记录迭代统计
  - `total_iterations`: 迭代总数
  - 持久化支持：保存/恢复会话状态

### Types (`core/types.py`)
- `OperatorType`: 算子类型枚举
  - `ARITHMETIC`, `STATISTICAL`, `TIMESERIES`, `CROSS_SECTIONAL`
  - `SMOOTHING`, `REGRESSION`, `LOGICAL`, `AUTO_INVENTED`
- `SignatureType`: 签名类型枚举
  - `TIME_SERIES_TO_TIME_SERIES`: 滚动/回溯运算
  - `CROSS_SECTION_TO_CROSS_SECTION`: 截面运算
  - `ELEMENT_WISE`: 点对点运算
  - `REDUCE_TIME`: 时间轴压缩
- `OperatorSpec`: 算子不可变描述符
- `FEATURES`: 规范特征集 `["$open", "$high", "$low", "$close", "$volume", "$amt", "$vwap", "$returns"]`

## 关系图

```
Parser --> ExpressionTree --> FactorLibrary
                |
                v
          OperatorNode --> OperatorRegistry
                               |
                               v
                         60+ Operators
```

## 设计决策

- **DAG 结构**: 表达式树是 DAG 而非二叉树，支持多子节点算子
- **延迟计算**: evaluate() 接受 data dict，按需计算
- **SymPy 规范化**: 可选通过 SymPy 规范化表达式

## 来源
- [Expression Tree 设计](../concepts/expression-tree.md)
- [Ralph Loop 机制](../concepts/ralph-loop.md)
- [DSL 设计](../design/typed-dsl.md)
