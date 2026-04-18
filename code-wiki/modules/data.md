# Data 模块

## Purpose
Data 模块负责市场数据加载、预处理、合成数据生成、张量构建。

## 核心组件

### Loader (`data/loader.py`)
- `MarketDataLoader`: 市场数据加载器
- 支持 CSV、Parquet 等格式
- 标准化数据格式输出

### Preprocessor (`data/preprocessor.py`)
- `DataPreprocessor`: 数据预处理
  - 缺失值处理
  - 异常值处理
  - 标准化

### Mock Data (`data/mock_data.py`)
- `generate_mock_data()`: 生成合成市场数据
- 用于无 API Key 时的演示和测试
- 支持多种市场场景（牛市、熊市、震荡）

### Tensor Builder (`data/tensor_builder.py`)
- `TensorBuilder`: 构建评估用的张量
- 将 DataFrame 转换为 (M, T) 张量
- M = 股票数量, T = 时间步数

## 数据格式

市场数据应包含：
- `$open`, `$high`, `$low`, `$close`: OHLC 价格
- `$volume`: 成交量
- `$amt`: 成交额
- `$vwap`: 成交量加权平均价
- `$returns`: 收益率

## 来源
- [数据格式](../design/data-format.md)
