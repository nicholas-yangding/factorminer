# A股数据使用速查

## 命令速查

### 1. 安装依赖

```bash
pip install stock-data
```

### 2. 快速加载

```python
from factorminer.data import AShareDataLoader

# 最近 60 交易日，全部股票
loader = AShareDataLoader(count=60, adj="hfq")
df = loader.load()
loader.close()

# 指定股票
loader = AShareDataLoader(ts_codes=["000001.SZ"], count=60)
```

### 3. Panel 构建（一行流）

```python
import numpy as np

def build_panel(df, features):
    """将 DataFrame 转换为 (M, T) Panel"""
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

data_dict = build_panel(df, ["close", "volume", "returns", "net_mf_vol"])
```

### 4. 因子计算

```python
from factorminer.core.parser import parse

# 吸筹强度
formula = 'Div($net_mf_vol, Add($volume, 1))'
tree = parse(formula)
signals = tree.evaluate(data_dict)  # shape: (M, T)
```

### 5. IC 评估

```python
from factorminer.evaluation.metrics import compute_ic, compute_ic_mean, compute_ic_abs_mean

ic_series = compute_ic(signals, data_dict["$returns"])
ic_mean = compute_ic_mean(ic_series)
ic_abs_mean = compute_ic_abs_mean(ic_series)

print(f"IC Mean: {ic_mean:.4f}, IC Abs Mean: {ic_abs_mean:.4f}")
```

---

## 常用 DSL 公式

| 因子 | 公式 |
|------|------|
| 主力净流入天数 | `Sum(IfElse(Greater($net_mf_vol, 0), 1, 0), 20)` |
| 吸筹强度 | `Div($net_mf_vol, Add($volume, 1))` |
| 价格 > MA20 | `Greater($close, Mean($close, 20))` |
| 成交量放大 | `Greater($volume, Mean($volume, 20))` |
| 20日动量 | `TsRank(Return($close, 20), 20)` |

---

## 股票代码格式

- 深圳: `000001.SZ`
- 上海: `600000.SH`
- 创业板: `300001.SZ`
- 科创板: `688001.SH`

---

## 复权选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 因子回测 | `hfq` | 保持历史价格可比性 |
| 实时行情 | `qfq` | 与行情软件一致 |
| 原始价格 | `None` | 查看真实价格 |

---

## 数据范围

| 字段 | 历史深度 |
|------|----------|
| OHLCV + 资金流向 | 约 2 年（tushare 限制） |
| 完整历史（无资金流） | 2000 年至今 |

---

## 常见错误

```python
# ❌ 错误：忘记加 $ 前缀
data_dict = {"close": close_arr}

# ✅ 正确：DSL 需要 $ 前缀
data_dict = {"$close": close_arr}
```

```python
# ❌ 错误：直接用 np.nan_to_num 处理 pandas nullable 类型
arr = np.nan_to_num(net_mf_df.values.T, nan=0.0)  # 会报错

# ✅ 正确：使用 pandas to_numpy
arr = net_mf_df.to_numpy(dtype=np.float64, na_value=0.0).T
```

---

## 快速扫描 TOP 10

```python
import pandas as pd

# 加载数据
loader = AShareDataLoader(count=60, adj="hfq")
df = loader.load()
loader.close()

# 构建 Panel
data_dict = build_panel(df, ["close", "volume", "returns", "net_mf_vol"])

# 计算吸筹强度
signals = parse('Div($net_mf_vol, Add($volume, 1))').evaluate(data_dict)

# 最新信号
latest = signals[:, -1]
returns = data_dict["$returns"][:, -1]

# 排序
results = pd.DataFrame({
    "asset_id": df["asset_id"].unique(),
    "strength": latest,
    "returns": returns
}).sort_values("strength", ascending=False)

print(results.head(10))
```
