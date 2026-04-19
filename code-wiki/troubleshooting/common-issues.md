# 常见问题排查

## 数据加载问题

### 问题 1：stock-data SDK 未安装

**错误信息**：
```
ImportError: stock-data SDK 未安装。请运行: pip install stock-data
```

**解决方案**：
```bash
pip install stock-data
```

---

### 问题 2：数据加载为空

**可能原因**：
- 股票代码错误
- 日期范围无数据
- 网络连接问题

**排查步骤**：
```python
# 1. 检查股票代码格式
# 正确格式: "000001.SZ", "600000.SH"
# 常见错误: "1.SZ", "600000"

# 2. 检查日期范围
loader = AShareDataLoader(
    ts_codes=["000001.SZ"],
    start="2020-01-01",
    end="2024-12-31"
)
df = loader.load()
print(f"数据条数: {len(df)}")

# 3. 检查数据库连接
loader = AShareDataLoader()
stock_list = loader.get_stock_list()
print(f"股票总数: {len(stock_list)}")
loader.close()
```

---

### 问题 3：资金流向数据缺失

**错误信息**：
```
FutureWarning: The default fill_method='ffill' in SeriesGroupBy.pct_change...
```

**说明**：这是正常的警告，不影响计算。但资金流向数据确实只有约 2 年历史。

**解决方案**：
```python
# 如果只需要 OHLCV 数据，可以禁用资金流向
loader = AShareDataLoader(
    count=60,
    include_moneyflow=False  # 禁用资金流向
)
```

---

## Panel 构建问题

### 问题 4：NaN / NaT 类型错误

**错误信息**：
```
TypeError: float() argument must be a string or a real number, not 'NAType'
```

或：
```
TypeError: ufunc 'isnan' not supported for the input types...
```

**原因**：pandas 的 `Int64Dtype()` 包含 `<NA>` 值，numpy 的 `isnan()` 无法处理。

**解决方案**：
```python
# ❌ 错误：直接用 np.nan_to_num
arr = np.nan_to_num(df.values.T, nan=0.0)  # 报错

# ❌ 错误：用 np.isnan 判断 pandas nullable 类型
arr = np.isnan(df.values.T)  # 报错

# ✅ 正确：使用 pandas to_numpy
arr = df.to_numpy(dtype=np.float64, na_value=0.0).T
```

**完整 Panel 构建代码**：
```python
def build_panel(df, features):
    df = df.sort_values(["asset_id", "datetime"])
    assets = df["asset_id"].unique()
    dates = np.sort(df["datetime"].unique())
    
    panels = {}
    for feat in features:
        arr = df.pivot(index="datetime", columns="asset_id", values=feat) \
                .reindex(dates, columns=assets) \
                .to_numpy(dtype=np.float64, na_value=0.0).T
        panels[f"${feat}"] = arr
    return panels
```

---

### 问题 5：pivot 后数据量太大内存不足

**问题**：5,827 股票 × 4,174 天 = 2400 万个值，float64 约 200MB

**解决方案**：
```python
# 1. 减少股票池
loader = AShareDataLoader(ts_codes=["000001.SZ", "600519.SH"], count=60)

# 2. 减少时间范围
loader = AShareDataLoader(count=30)  # 只加载最近 30 天

# 3. 使用整数类型（如果数据允许）
arr = df.to_numpy(dtype=np.float32, na_value=0.0).T  # 减半内存

# 4. 分批处理
for chunk in range(0, T, 100):
    end_idx = min(chunk + 100, T)
    signals_chunk = tree.evaluate(data_dict[:, chunk:end_idx])
```

---

## DSL 解析问题

### 问题 6：Feature name not found

**错误信息**：
```
KeyError: Feature '$net_mf_vol' not found in data. Available: ['close', 'volume', ...]
```

**原因**：构建 `data_dict` 时忘记加 `$` 前缀

**解决方案**：
```python
# ❌ 错误
data_dict = {"close": close_arr}  # DSL 找不到

# ✅ 正确
data_dict = {"$close": close_arr}
```

---

### 问题 7：公式解析失败

**错误信息**：
```
ParseError: ...
```

**解决方案**：
```python
# 1. 检查括号匹配
formula = 'Sum(IfElse(Greater($net_mf_vol, 0), 1, 0), 20)'  # ✅ 正确

# 2. 检查引号
formula = "Div($net_mf_vol, Add($volume, 1))"  # ✅ 双引号也可以

# 3. 检查算子名称
# 正确: Greater, Less, Equal, IfElse, Sum, Mean, Div, Mul, Add, Sub
# 错误: gt, lt, >, < (必须用 DSL 规定的名称)
```

---

## IC 计算问题

### 问题 8：IC 全为 NaN

**错误信息**：所有 IC 值都是 NaN

**原因**：有效样本数太少（M < 5）

**解决方案**：
```python
# 需要足够的股票数量（M > 30）
print(f"股票数量: {M}")
if M < 30:
    print("警告: 股票数量太少，IC 计算不可靠")

# 检查是否有有效数据
valid = ~(np.isnan(signals) | np.isnan(returns))
print(f"有效数据比例: {valid.mean():.2%}")
```

---

### 问题 9：IC 值异常极端

**可能原因**：
- 因子值包含 Inf 或 -Inf
- 股票数量太少
- 数据有异常值

**解决方案**：
```python
# 1. 检查因子值范围
print(f"Signals min: {np.min(signals)}, max: {np.max(signals)}")
print(f"Inf count: {np.isinf(signals).sum()}")

# 2. 处理极端值
signals = np.clip(signals, -1e10, 1e10)

# 3. 去除极值
percentile = 1
low = np.percentile(signals, percentile)
high = np.percentile(signals, 100 - percentile)
signals = np.clip(signals, low, high)
```

---

## 表达式树评估问题

### 问题 10：evaluate 速度慢

**原因**：表达式树在 Python 中逐元素计算，大规模数据慢

**解决方案**：
```python
# 1. 使用向量化算子（如果可用）
# 查看可用算子
from factorminer.core.types import OPERATORS
print(OPERATORS.keys())

# 2. 减少计算频率
# 不要每次都 evaluate 全量数据

# 3. 使用 GPU 加速（如果配置了）
# 见 code-wiki/concepts/gpu-backend.md
```

---

### 问题 11：内存泄漏

**症状**：运行多次后内存持续增长

**原因**：表达式树每次 evaluate 创建临时数组

**解决方案**：
```python
# 1. 分批处理
batch_size = 1000
for i in range(0, T, batch_size):
    end_idx = min(i + batch_size, T)
    signals_batch = tree.evaluate({k: v[:, i:end_idx] for k, v in data_dict.items()})

# 2. 显式删除大数组
del large_intermediate_array

# 3. 使用 generator 而非 list
```

---

## 调试技巧

### 1. 打印中间结果

```python
# 逐步执行表达式树
tree = parse(formula)
print(f"公式: {tree.to_string()}")
print(f"根节点: {tree.root}")

# 打印部分评估结果
partial = signals[:5, :10]
print(f"信号样本 (前5股票, 前10天):\n{partial}")
```

### 2. 检查数据类型

```python
print(f"signals dtype: {signals.dtype}")
print(f"signals shape: {signals.shape}")
print(f"NaN count: {np.isnan(signals).sum()}")
print(f"Inf count: {np.isinf(signals).sum()}")
```

### 3. 使用 mock 数据验证

```python
from factorminer.data import generate_mock_data

# 用 mock 数据验证流程
mock_df = generate_mock_data(n_stocks=100, n_days=60)
print(mock_df.columns.tolist())
print(mock_df.head())
```

---

## 错误码速查

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| `ImportError: stock-data` | SDK 未安装 | `pip install stock-data` |
| `KeyError: '$xxx'` | 特征名缺失 | 检查 `data_dict` 是否加 `$` 前缀 |
| `TypeError: NAType` | pandas NA 类型问题 | 用 `df.to_numpy(na_value=0.0)` |
| `ParseError` | DSL 语法错误 | 检查公式语法 |
| `IC all NaN` | 样本太少 | 确保 M > 30 |
