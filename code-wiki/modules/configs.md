# Configs 模块

## Purpose
Configs 模块包含 FactorMiner 的 YAML 配置文件，定义挖掘循环、评估、LLM、记忆等各项参数的默认值。

## 配置文件

### default.yaml
主配置文件，包含所有默认参数：

**Mining 参数:**
```yaml
mining:
  target_library_size: 110    # 目标库大小（论文：110）
  batch_size: 40               # 每轮生成候选因子数
  max_iterations: 200         # 最大迭代次数
  ic_threshold: 0.04          # IC 阈值
  icir_threshold: 0.5         # ICIR 阈值
  correlation_threshold: 0.5   # 相关性阈值
  replacement_ic_min: 0.10     # 替换最小 IC
  replacement_ic_ratio: 1.3   # 替换 IC 比率
```

**Evaluation 参数:**
```yaml
evaluation:
  num_workers: 40             # 并行工作进程数
  fast_screen_assets: 100     # 快速筛选资产数
  gpu_device: "cuda:0"        # GPU 设备
  backend: "gpu"               # 计算后端：gpu/numpy/c
  signal_failure_policy: "reject"  # 信号失败策略
```

**Data 参数:**
```yaml
data:
  market: "a_shares"          # 市场类型
  universe: "CSI500"          # 股票池
  frequency: "10min"          # 频率
  features: ["$open", "$high", "$low", "$close", "$volume", "$amt", "$vwap", "$returns"]
  train_period: ["2024-01-01", "2024-12-31"]
  test_period: ["2025-01-01", "2025-12-31"]
```

**LLM 参数:**
```yaml
llm:
  provider: "openai"           # 提供商：openai/anthropic/google
  model: "gpt-4"             # 模型
  temperature: 0.8            # 采样温度
  max_tokens: 4096            # 最大 token 数
```

**Memory 参数:**
```yaml
memory:
  success_pattern_limit: 50   # 成功模式上限
  failure_pattern_limit: 30   # 失败模式上限
  consolidation_interval: 10  # 合并间隔
```

### helix_research.yaml
Helix 研究车道配置，启用 Phase 2 特性：

```yaml
phase2:
  causal: true                 # 因果验证
  regime: true               # 市场状态评估
  capacity: true             # 容量估计
  significance: true          # 显著性检验
  debate: true               # 辩论生成
  auto_inventor: true        # 自动算子发明
  helix: true                # Helix 模块
```

### paper_repro.yaml
论文复现配置，严格复现论文设置：

```yaml
mining:
  target_library_size: 110
  ic_threshold: 0.04

evaluation:
  backend: "numpy"
  signal_failure_policy: "reject"
```

### demo_local.yaml
本地演示配置，小规模测试：

```yaml
mining:
  target_library_size: 10
  batch_size: 8
  max_iterations: 10
```

### benchmark_full.yaml
完整基准测试配置：

```yaml
# 包含所有基准测试的完整配置
```

## 配置优先级

```
defaults.yaml -> user_config.yaml -> CLI overrides
```

CLI 参数优先于配置文件。

## 来源
- `factorminer/configs/default.yaml`
- `factorminer/configs/helix_research.yaml`
- `factorminer/configs/paper_repro.yaml`
- `factorminer/configs/demo_local.yaml`
- `factorminer/configs/benchmark_full.yaml`
