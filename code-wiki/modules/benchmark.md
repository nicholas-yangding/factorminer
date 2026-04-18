# Benchmark 模块

## Purpose
Benchmark 模块实现严格的论文复现和研究基准测试流程，包括消融研究、运行时效率对比、表格式输出。

## 核心组件

### Helix Benchmark (`benchmark/helix_benchmark.py`)
- `HelixBenchmark`: Phase 2 基准测试
- `run_table1_benchmark()`: 论文 Table 1 复现

### Ablation (`benchmark/ablation.py`)
- `AblationStudy`: 消融研究
- `run_full_ablation_study()`: 完整消融流程
- 评估各组件（记忆、辩论等）的贡献

### Runtime (`benchmark/runtime.py`)
- `SpeedBenchmark`: 运行时效率基准
- `run_efficiency_benchmark()`: 效率对比
- `run_runtime_mining_benchmark()`: 挖掘效率测试

### Catalogs (`benchmark/catalogs.py`)
- 基线因子目录
- `build_alpha101_adapted()`: Alpha101 适应版
- `build_random_exploration()`: 随机探索
- `build_gplearn_style()`: GPlearn 风格
- `build_factor_miner_catalog()`: FactorMiner 风格

## 基准测试类型

| 类型 | 函数 | 用途 |
|------|------|------|
| Table 1 | `run_table1_benchmark()` | 论文核心指标复现 |
| Ablation | `run_full_ablation_study()` | 组件重要性分析 |
| Memory | `run_ablation_memory_benchmark()` | 记忆模块消融 |
| Cost | `run_cost_pressure_benchmark()` | 成本压力测试 |
| Efficiency | `run_efficiency_benchmark()` | 运行时效率 |
| Suite | `run_benchmark_suite()` | 完整基准套件 |

## 设计决策

- **严格重计算**: 所有分析命令都重新计算，不信任存储的元数据
- **Manifest 制度**: 每次运行生成 manifest，记录配置和结果哈希
- **冻结集**: 基准测试使用冻结的 Top-K 因子集

## 来源
- [Benchmark 设计](../design/benchmark.md)
