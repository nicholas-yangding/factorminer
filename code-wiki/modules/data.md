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

---

# AShareDataLoader (A股数据加载器)

## 概述

`AShareDataLoader` 是对接 `stock-data` SDK 的A股数据加载器，基于 DuckDB 本地存储，支持：

- **股票、ETF、指数** 日线行情（含前复权/后复权）
- **个股资金流向**：大单、中单、小单净流入
- 增量同步，复权支持

## 安装依赖

```bash
pip install stock-data
```

## 快速开始

```python
from factorminer.data import AShareDataLoader

# 加载单只股票
loader = AShareDataLoader(ts_codes=["000001.SZ"], count=60)
df = loader.load()
loader.close()

print(df.columns.tolist())
print(df.head())
```

## API 参考

### AShareDataLoader

```python
@dataclass
class AShareDataLoader:
    ts_codes: Optional[List[str]] = None    # 股票代码列表，None=全部
    adj: Literal["qfq", "hfq", None] = "qfq"  # 复权类型
    start: Optional[str] = None              # 开始日期 "2020-01-01"
    end: Optional[str] = None                # 结束日期 "2024-12-31"
    count: Optional[int] = None             # 最近交易日数量
    include_moneyflow: bool = True           # 是否加载资金流向
```

### 方法

| 方法 | 说明 |
|------|------|
| `load() -> pd.DataFrame` | 加载数据，返回长格式 DataFrame |
| `get_stock_list(industry, market) -> pd.DataFrame` | 获取股票列表 |
| `close()` | 关闭数据库连接 |

### load() 返回的 DataFrame 列

| 列名 | 类型 | 说明 |
|------|------|------|
| `asset_id` | str | 股票代码，如 "000001.SZ" |
| `datetime` | datetime | 交易日期 |
| `open`, `high`, `low`, `close` | float | OHLC 价格 |
| `volume` | float | 成交量（手） |
| `amount` | float | 成交额（元） |
| `vwap` | float | 成交量加权平均价 |
| `returns` | float | 日收益率（百分比） |
| `net_mf_vol` | int | **主力净流入量**（股） |
| `net_mf_amount` | float | **主力净流入额**（元） |
| `buy_lg_vol`, `sell_lg_vol` | int | 大单买入/卖出量 |
| `buy_elg_vol`, `sell_elg_vol` | int | 特大单买入/卖出量 |
| `buy_md_vol`, `sell_md_vol` | int | 中单买入/卖出量 |
| `buy_sm_vol`, `sell_sm_vol` | int | 小单买入/卖出量 |

## 复权类型选择

| 复权 | 含义 | 适用场景 |
|------|------|----------|
| `qfq` | 前复权（默认） | 技术分析、均线计算 |
| `hfq` | 后复权 | 回测、长期历史分析 |
| `None` | 不复权 | 原始价格查看 |

**推荐**：回测用 `hfq`（后复权），保持历史绝对价格感。

## 资金流向字段

资金流向数据通过 `moneyflow()` 接口获取，字段映射：

| DSL 特征名 | DataFrame 列名 | 说明 |
|------------|----------------|------|
| `$net_mf_vol` | `net_mf_vol` | 主力净流入量（股） |
| `$net_mf_amount` | `net_mf_amount` | 主力净流入额（元） |
| `$lg_buy_vol` | `buy_lg_vol` | 大单买入量 |
| `$lg_sell_vol` | `sell_lg_vol` | 大单卖出量 |
| `$elg_buy_vol` | `buy_elg_vol` | 特大单买入量 |
| `$elg_sell_vol` | `sell_elg_vol` | 特大单卖出量 |
| `$md_buy_vol` | `buy_md_vol` | 中单买入量 |
| `$md_sell_vol` | `sell_md_vol` | 中单卖出量 |
| `$sm_buy_vol` | `buy_sm_vol` | 小单买入量 |
| `$sm_sell_vol` | `sell_sm_vol` | 小单卖出量 |

**注意**：资金流向数据 tushare 接口仅保留约 **2 年**历史。

## 使用示例

### 1. 加载全部股票（近 60 交易日）

```python
loader = AShareDataLoader(count=60, adj="hfq")
df = loader.load()
loader.close()
```

### 2. 加载指定股票池

```python
loader = AShareDataLoader(
    ts_codes=["000001.SZ", "600519.SH", "600000.SH"],
    start="2023-01-01",
    end="2024-12-31",
    adj="qfq"
)
df = loader.load()
loader.close()
```

### 3. 获取股票列表（行业筛选）

```python
loader = AShareDataLoader()
stock_list = loader.get_stock_list(industry="银行", market="主板")
print(stock_list.head())
loader.close()
```

### 4. 上下文管理器写法

```python
with AShareDataLoader(count=60) as loader:
    df = loader.load()
    # loader.close() 自动调用
```

## Panel 数据构建

因子计算需要 `(M stocks, T periods)` 格式的 Panel 数据：

```python
import numpy as np

# 排序
df_sorted = df.sort_values(["asset_id", "datetime"])

# 获取股票和日期列表
assets = df_sorted["asset_id"].unique()
dates = np.sort(df_sorted["datetime"].unique())

# Pivot 构建 Panel
close_df = df_sorted.pivot(
    index="datetime",
    columns="asset_id",
    values="close"
).reindex(dates, columns=assets)

# 转换为 (M, T) 数组
close_arr = close_df.to_numpy(dtype=np.float64, na_value=0.0).T
```

## 完整因子计算流程

```python
from factorminer.data import AShareDataLoader
from factorminer.core.parser import parse
from factorminer.evaluation.metrics import compute_ic, compute_ic_mean, compute_ic_abs_mean

# 1. 加载数据
loader = AShareDataLoader(count=60, adj="hfq")
df = loader.load()
loader.close()

# 2. 构建 Panel
df_sorted = df.sort_values(["asset_id", "datetime"])
assets = df_sorted["asset_id"].unique()
dates = np.sort(df_sorted["datetime"].unique())

close_df = df_sorted.pivot(index="datetime", columns="asset_id", values="close").reindex(dates, columns=assets)
volume_df = df_sorted.pivot(index="datetime", columns="asset_id", values="volume").reindex(dates, columns=assets)
returns_df = df_sorted.pivot(index="datetime", columns="asset_id", values="returns").reindex(dates, columns=assets)
net_mf_df = df_sorted.pivot(index="datetime", columns="asset_id", values="net_mf_vol").reindex(dates, columns=assets)

# 3. 转换为数组
close_arr = close_df.to_numpy(dtype=np.float64, na_value=0.0).T
volume_arr = volume_df.to_numpy(dtype=np.float64, na_value=0.0).T
returns_arr = returns_df.to_numpy(dtype=np.float64, na_value=0.0).T
net_mf_arr = net_mf_df.to_numpy(dtype=np.float64, na_value=0.0).T

# 4. 构建 data_dict（DSL 需要 $ 前缀）
data_dict = {
    "$close": close_arr,
    "$volume": volume_arr,
    "$returns": returns_arr,
    "$net_mf_vol": net_mf_arr,
}

# 5. 解析并计算因子
formula = 'Div($net_mf_vol, Add($volume, 1))'
tree = parse(formula)
signals = tree.evaluate(data_dict)

# 6. IC 评估
ic_series = compute_ic(signals, returns_arr)
ic_mean = compute_ic_mean(ic_series)
ic_abs_mean = compute_ic_abs_mean(ic_series)

print(f"IC Mean: {ic_mean:.4f}, IC Abs Mean: {ic_abs_mean:.4f}")
```

## 来源

- `factorminer/data/astock.py`
