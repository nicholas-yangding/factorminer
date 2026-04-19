# 11 - CLI 参考

## 命令概览

```bash
factorminer --help
```

```
usage: factorminer [-h] [--cpu] [--gpu] [--config CONFIG]
                   {mine,helix,evaluate,combine,visualize,benchmark,export}
                   ...

FactorMiner - LLM-driven formulaic alpha mining
```

## 通用选项

| 选项 | 说明 |
|------|------|
| `--cpu` | 使用 CPU 计算 |
| `--gpu` | 使用 GPU 计算 |
| `--config CONFIG` | 指定配置文件 |
| `--mock` | 使用模拟数据 |

## mine - 挖掘因子

```bash
factorminer --cpu mine --mock -n 2 -b 8 -t 10
```

**参数:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-n, --iterations` | 迭代次数 | 1 |
| `-b, --batch-size` | 每轮候选数 | 40 |
| `-t, --target-size` | 目标库大小 | 110 |

**示例:**

```bash
# Paper Lane 挖掘
factorminer --cpu mine --config configs/paper_repro.yaml -n 10 -b 40 -t 110

# 使用模拟数据
factorminer --cpu mine --mock -n 2 -b 8 -t 10
```

## helix - Helix 挖掘

```bash
factorminer --cpu helix --mock --debate --canonicalize -n 2 -b 8 -t 10
```

**Phase 2 选项:**

| 选项 | 说明 |
|------|------|
| `--debate` | 启用辩论生成 |
| `--canonicalize` | 启用表达式规范化 |
| `--causal` | 启用因果验证 |
| `--regime` | 启用市场状态分析 |
| `--capacity` | 启用容量估计 |
| `--significance` | 启用显著性检验 |
| `--auto-inventor` | 启用自动算子发明 |

## evaluate - 评估因子库

```bash
factorminer --cpu evaluate output/factor_library.json --mock --period both --top-k 10
```

**参数:**

| 参数 | 说明 | 选项 |
|------|------|------|
| `--period` | 评估周期 | `train`, `test`, `both` |
| `--top-k` | 只评估 Top K | 数字 |

**示例:**

```bash
# 评估整个库
factorminer --cpu evaluate output/factor_library.json --mock

# 只评估 Top 10
factorminer --cpu evaluate output/factor_library.json --mock --top-k 10

# 评估训练和测试期
factorminer --cpu evaluate output/factor_library.json --mock --period both
```

## combine - 因子组合

```bash
factorminer --cpu combine output/factor_library.json --mock -k 5
```

**参数:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-k, --n-factors` | 组合因子数 | 5 |

**示例:**

```bash
# 组合 Top 5 因子
factorminer --cpu combine output/factor_library.json --mock -k 5
```

## visualize - 可视化

```bash
factorminer --cpu visualize output/factor_library.json --mock --output-dir output/plots
```

**选项:**

| 选项 | 说明 |
|------|------|
| `--correlation` | 相关性热力图 |
| `--ic` | IC 时序图 |
| `--quintile` | Quintile 分组图 |
| `--tearsheet` | 完整 tearsheet |

**示例:**

```bash
# 生成所有图表
factorminer --cpu visualize output/factor_library.json --mock

# 只生成相关性图
factorminer --cpu visualize output/factor_library.json --mock --correlation

# 指定输出目录
factorminer --cpu visualize output/factor_library.json --mock --output-dir output/plots
```

## benchmark - 基准测试

```bash
factorminer --cpu benchmark --config configs/paper_repro.yaml
```

**可用配置:**

| 配置 | 说明 |
|------|------|
| `configs/default.yaml` | 默认配置 |
| `configs/paper_repro.yaml` | 论文复现 |
| `configs/helix_research.yaml` | Helix 研究 |
| `configs/benchmark_full.yaml` | 完整基准 |

**示例:**

```bash
# 论文复现基准
factorminer --cpu benchmark --config configs/paper_repro.yaml

# Helix 研究基准
factorminer --cpu benchmark --config configs/helix_research.yaml
```

## export - 导出

```bash
factorminer export output/factor_library.json --format json
```

**格式选项:**

| 格式 | 说明 |
|------|------|
| `--format json` | JSON 格式 |
| `--format csv` | CSV 格式 |
| `--format formulas` | 公式列表 |

**示例:**

```bash
# 导出为 JSON
factorminer export output/factor_library.json --format json

# 导出为 CSV
factorminer export output/factor_library.json --format csv

# 导出公式列表
factorminer export output/factor_library.json --format formulas
```

## 完整示例

### 1. 完整挖掘 + 评估流程

```bash
# 1. 挖掘因子
factorminer --cpu mine --mock -n 10 -b 40 -t 110 -o output/library.json

# 2. 评估因子库
factorminer --cpu evaluate output/library.json --mock --period both --top-k 20

# 3. 可视化
factorminer --cpu visualize output/library.json --mock --tearsheet

# 4. 组合 Top 10
factorminer --cpu combine output/library.json --mock -k 10
```

### 2. Helix 研究流程

```bash
# 1. Helix 挖掘
factorminer --cpu helix --mock --debate --canonicalize --causal --regime -n 5 -b 40 -t 50

# 2. 完整基准测试
factorminer --cpu benchmark --config configs/helix_research.yaml

# 3. 导出结果
factorminer export output/helix_library.json --format json
```

## 配置文件格式

```yaml
# configs/custom.yaml
mining:
  target_library_size: 50
  batch_size: 20
  max_iterations: 100
  ic_threshold: 0.03
  icir_threshold: 0.4
  correlation_threshold: 0.6

evaluation:
  backend: "cpu"
  num_workers: 20
  signal_failure_policy: "synthetic"

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.8
```

## 常见问题

### Q: 如何查看详细日志？

```bash
factorminer --cpu mine --mock -n 1 -b 4 -t 5 -v
```

### Q: 如何中断后恢复？

```bash
# 输出目录包含会话状态
ls output/

# 使用 --resume 恢复（如支持）
factorminer --cpu mine --mock --resume output/session.json
```

### Q: 如何调整 IC 阈值？

```bash
# 通过 CLI
factorminer --cpu mine --mock --ic-threshold 0.03 -n 2 -b 8 -t 10

# 或修改配置文件
```
