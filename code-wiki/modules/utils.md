# Utils 模块

## Purpose
Utils 模块提供配置、日志、可视化、报告等辅助功能。

## 核心组件

### Config (`utils/config.py`)
- `load_config()`: 加载 YAML 配置
- 配置数据结构定义

### Logging (`utils/logging.py`)
- `MiningSessionLogger`: 挖掘会话日志
- `IterationRecord`: 迭代记录
- `FactorRecord`: 因子记录

### Visualization (`utils/visualization.py`)
- `FactorVisualizer`: 因子可视化
- IC 时间序列图
- 相关性热力图
- Quintile 分组图

### Tearsheet (`utils/tearsheet.py`)
- `generate_tearsheet()`: 生成分析报告
- 综合性能报告

### Reporting (`utils/reporting.py`)
- `generate_report()`: 生成文本报告
- Markdown/HTML 格式

## CLI (`cli.py`)
- `factorminer` 命令行工具
- 子命令: `mine`, `helix`, `evaluate`, `combine`, `visualize`, `benchmark`, `export`

## 来源
