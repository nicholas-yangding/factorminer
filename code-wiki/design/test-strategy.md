# Test Strategy (测试策略)

## 概述
FactorMiner 包含 27 个测试文件，覆盖所有核心模块，确保代码质量和功能正确性。

## 测试文件结构

```
factorminer/tests/
├── conftest.py              # pytest 配置和 fixtures
├── test_auto_inventor.py    # 自动算子发明测试
├── test_benchmark.py        # 基准测试
├── test_canonicalizer.py   # 规范化器测试
├── test_capacity.py         # 容量估计测试
├── test_causal.py          # 因果验证测试
├── test_cli_analysis.py     # CLI 分析命令测试
├── test_cli_helix.py       # CLI Helix 命令测试
├── test_combination.py     # 因子组合测试
├── test_data.py            # 数据处理测试
├── test_debate.py          # 辩论模块测试
├── test_evaluation.py      # 评估管道测试
├── test_expression_tree.py # 表达式树测试
├── test_helix_loop.py      # Helix 循环测试
├── test_knowledge_graph.py # 知识图谱测试
├── test_library.py         # 因子库测试
├── test_memory.py          # 记忆系统测试
├── test_operators.py      # 算子测试
├── test_provenance.py      # 溯源测试
├── test_ralph_loop.py     # Ralph 循环测试
├── test_regime.py          # 市场状态测试
├── test_research.py        # 研究评估测试
├── test_runtime_analysis.py # 运行时分析测试
└── test_significance.py    # 显著性测试
```

## conftest.py Fixtures

```python
# 常用 fixtures
@pytest.fixture
def mock_data():
    """生成模拟市场数据"""
    return generate_mock_data(n_assets=100, n_days=252)

@pytest.fixture
def sample_library():
    """创建示例因子库"""
    return FactorLibrary(ic_threshold=0.02)

@pytest.fixture
def operator_registry():
    """获取算子注册表"""
    return OPERATOR_REGISTRY
```

## 测试运行

```bash
# 运行所有测试
pytest factorminer/tests -v

# 运行特定测试文件
pytest factorminer/tests/test_operators.py -v

# 运行带覆盖率
pytest factorminer/tests --cov=factorminer --cov-report=html

# 运行快速测试（跳过慢速测试）
pytest factorminer/tests -m "not slow"
```

## 测试分类

### 单元测试
- `test_expression_tree.py`: 表达式树操作
- `test_operators.py`: 算子实现
- `test_parser.py`: DSL 解析

### 集成测试
- `test_ralph_loop.py`: 完整挖掘循环
- `test_helix_loop.py`: Helix 循环
- `test_evaluation.py`: 评估管道

### 基准测试
- `test_benchmark.py`: 基准测试运行
- `test_runtime_analysis.py`: 运行时分析

## Mock 数据

```python
from factorminer.data.mock_data import generate_mock_data

# 生成模拟数据
data = generate_mock_data(
    n_assets=100,
    n_days=252,
    start_date="2024-01-01",
    price_range=(10, 100),
    volume_range=(1000, 100000)
)
```

## 测试原则

1. **TDD**: 优先编写失败测试，再实现功能
2. **隔离**: 每个测试独立，不依赖其他测试
3. **可重复**: 测试结果稳定，不受随机因素影响
4. **快速**: 单元测试应能在秒级完成

## 来源
- `factorminer/tests/`
