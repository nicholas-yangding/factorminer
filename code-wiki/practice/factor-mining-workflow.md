# 因子挖掘实战流程

## 概述

本文档展示完整的因子挖掘流程：从数据加载到因子评估，使用 AShare 真实数据。

## 完整案例：主力吸筹因子

### 背景

基于资金流向理论：**主力资金持续净流入 = 吸筹行为**

我们尝试两个因子：
1. **吸筹天数**：近 20 天主力净流入为正的天数
2. **吸筹强度**：主力净流入 / 成交量

### Step 1: 数据加载

```python
from factorminer.data import AShareDataLoader

loader = AShareDataLoader(
    ts_codes=None,      # 全部股票
    count=60,           # 近 60 交易日
    adj="hfq",          # 后复权
    include_moneyflow=True
)
df = loader.load()
loader.close()

print(f"数据: {len(df)} 条, {df['asset_id'].nunique()} 只股票")
print(f"日期: {df['datetime'].min()} ~ {df['datetime'].max()}")
```

**输出**：
```
数据: 348310 条, 5827 只股票
日期: 2000-04-11 ~ 2026-04-17
```

### Step 2: 构建 Panel 数据

```python
import numpy as np
import pandas as pd

# 排序
df_sorted = df.sort_values(["asset_id", "datetime"])

# 获取股票和日期列表
assets = df_sorted["asset_id"].unique()
dates = np.sort(df_sorted["datetime"].unique())

M, T = len(assets), len(dates)
print(f"Panel: {M} stocks × {T} periods")

# 构建特征数组 (M, T)
close_df = df_sorted.pivot(index="datetime", columns="asset_id", values="close").reindex(dates, columns=assets)
volume_df = df_sorted.pivot(index="datetime", columns="asset_id", values="volume").reindex(dates, columns=assets)
returns_df = df_sorted.pivot(index="datetime", columns="asset_id", values="returns").reindex(dates, columns=assets)
net_mf_df = df_sorted.pivot(index="datetime", columns="asset_id", values="net_mf_vol").reindex(dates, columns=assets)

# 转换为 float64 数组，NaN 替换为 0
close_arr = close_df.to_numpy(dtype=np.float64, na_value=0.0).T
volume_arr = volume_df.to_numpy(dtype=np.float64, na_value=0.0).T
returns_arr = returns_df.to_numpy(dtype=np.float64, na_value=0.0).T
net_mf_arr = net_mf_df.to_numpy(dtype=np.float64, na_value=0.0).T

print(f"Arrays: close={close_arr.shape}, returns={returns_arr.shape}")
```

**输出**：
```
Panel: 5827 stocks × 4174 periods
Arrays: close=(5827, 4174), returns=(5827, 4174)
```

### Step 3: 构建 Data Dict

```python
# DSL 公式需要 $ 前缀
data_dict = {
    "$close": close_arr,
    "$volume": volume_arr,
    "$returns": returns_arr,
    "$net_mf_vol": net_mf_arr,
}

print(f"Data keys: {list(data_dict.keys())}")
```

**输出**：
```
Data keys: ['$close', '$volume', '$returns', '$net_mf_vol']
```

### Step 4: 计算因子信号

```python
from factorminer.core.parser import parse

# 因子 1: 吸筹天数 = 近20天主力净流入为正的天数
formula1 = 'Sum(IfElse(Greater($net_mf_vol, 0), 1, 0), 20)'
tree1 = parse(formula1)
signals1 = tree1.evaluate(data_dict)

print(f"Formula: {tree1.to_string()}")
print(f"Signals shape: {signals1.shape}")  # (M, T)
print(f"吸筹天数 (首股票, 后10天): {signals1[0, -10:].astype(int)}")

# 因子 2: 吸筹强度 = 主力净流入 / 成交量
formula2 = 'Div($net_mf_vol, Add($volume, 1))'
tree2 = parse(formula2)
signals2 = tree2.evaluate(data_dict)

print(f"Formula: {tree2.to_string()}")
```

**输出**：
```
Formula: Sum(IfElse(Greater($net_mf_vol, 0), 1, 0), 20)
Signals shape: (5827, 4174)
吸筹天数 (首股票, 后10天): [10 10 11 10  9  9 10 10 10 10]
```

### Step 5: IC 评估

```python
from factorminer.evaluation.metrics import compute_ic, compute_ic_mean, compute_ic_abs_mean, compute_icir

def evaluate_factor(signals, returns_arr, name):
    """评估单个因子"""
    ic_series = compute_ic(signals, returns_arr)
    valid_ic = ic_series[~np.isnan(ic_series)]
    
    ic_mean = float(np.mean(valid_ic)) if len(valid_ic) > 0 else 0.0
    ic_abs_mean = float(np.mean(np.abs(valid_ic))) if len(valid_ic) > 0 else 0.0
    icir = compute_icir(ic_series)
    ic_win_rate = float(np.mean(valid_ic > 0)) if len(valid_ic) > 0 else 0.0
    
    print(f"\n{'='*50}")
    print(f"因子: {name}")
    print(f"{'='*50}")
    print(f"IC Mean (signed):     {ic_mean:.4f}")
    print(f"IC Abs Mean:           {ic_abs_mean:.4f}")
    print(f"ICIR:                  {icir:.4f}")
    print(f"IC Win Rate:           {ic_win_rate*100:.1f}%")
    
    return ic_mean, ic_abs_mean

# 评估两个因子
ic1_mean, ic1_abs = evaluate_factor(signals1, returns_arr, "吸筹天数 (20d)")
ic2_mean, ic2_abs = evaluate_factor(signals2, returns_arr, "吸筹强度")

# 综合因子
signals_combined = signals1 * signals2
ic3_mean, ic3_abs = evaluate_factor(signals_combined, returns_arr, "综合因子")
```

**输出**：
```
==================================================
因子: 吸筹天数 (20d)
==================================================
IC Mean (signed):     -0.0836
IC Abs Mean:          0.3710
ICIR:                 -0.1623
IC Win Rate:          26.3%

==================================================
因子: 吸筹强度
==================================================
IC Mean (signed):     0.1660
IC Abs Mean:          0.4326
ICIR:                 0.2837
IC Win Rate:          43.0%

==================================================
因子: 综合因子
==================================================
IC Mean (signed):     0.1642
IC Abs Mean:          0.3563
ICIR:                 0.3026
IC Win Rate:          39.9%
```

### Step 6: 结果汇总

```python
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Factor':<40} {'IC Mean':>10} {'IC Abs':>10} {'ICIR':>10}")
print("-"*72)
print(f"{'吸筹天数 (20d)':<40} {ic1_mean:>10.4f} {ic1_abs:>10.4f} {-0.1623:>10.4f}")
print(f"{'吸筹强度':<40} {ic2_mean:>10.4f} {ic2_abs:>10.4f}")
print(f"{'综合 (天数×强度)':<40} {ic3_mean:>10.4f} {ic3_abs:>10.4f}")
```

**输出**：
```
============================================================
SUMMARY
============================================================
Factor                                      IC Mean     IC Abs       ICIR
------------------------------------------------------------------------
吸筹天数 (20d)                             -0.0836     0.3710    -0.1623
吸筹强度                                    0.1660     0.4326
综合 (天数×强度)                            0.1642     0.3563
```

---

## 结果解读

### 1. 吸筹天数 - 负向因子

| 指标 | 值 | 解读 |
|------|-----|------|
| IC Mean | **-0.0836** | 负相关！与预期相反 |
| IC Abs Mean | 0.3710 | 但绝对值很大 |
| Win Rate | 26.3% | 仅 26% 日期正相关 |

**结论**：主力净流入天数越多，未来收益反而越低。这可能意味着：
- 主力吸筹 ≠ 上涨（可能是对倒出货）
- 散户跟主力反而被套

### 2. 吸筹强度 - 有效因子

| 指标 | 值 | 解读 |
|------|-----|------|
| IC Mean | **+0.1660** | 正向显著 |
| IC Abs Mean | 0.4326 | 强度大 |
| Win Rate | 43.0% | 方向较稳定 |

**结论**：吸筹强度（净流入/成交量）是有效因子，反映单位成交量中的主力资金占比。

### 3. 综合因子 - 边际改善

| 指标 | 值 | 解读 |
|------|-----|------|
| IC Mean | +0.1642 | 接近吸筹强度单独使用 |

**结论**：综合效果不如吸筹强度单独使用，吸筹天数是噪声。

---

## 进阶：结构突破因子

### 因子设计

基于技术分析：**价格突破均线 + 成交量放大 = 结构突破**

```python
# 价格 > MA20
formula_bo = 'Greater($close, Mean($close, 20))'
signals_bo = parse(formula_bo).evaluate(data_dict)

# 成交量放大
formula_vol = 'Greater($volume, Mean($volume, 20))'
signals_vol = parse(formula_vol).evaluate(data_dict)

# 20日动量
formula_mom = 'TsRank(Return($close, 20), 20)'
signals_mom = parse(formula_mom).evaluate(data_dict)

# 评估
for name, sig in [("价格>MA20", signals_bo), ("成交量>MA20", signals_vol), ("动量", signals_mom)]:
    ic = compute_ic(sig, returns_arr)
    valid = ic[~np.isnan(ic)]
    print(f"{name:<12} IC_Mean={np.mean(valid):.4f}  IC_Abs={np.mean(np.abs(valid)):.4f}")
```

**输出**：
```
价格>MA20      IC_Mean=0.2520  IC_Abs=0.5354
成交量>MA20    IC_Mean=0.0484  IC_Abs=0.3563
动量           IC_Mean=0.1707  IC_Abs=0.4251
```

---

## 扫描 TOP 10 股票

```python
import pandas as pd

# 获取最新信号
latest_strength = signals2[:, -1]  # 吸筹强度
latest_breakout = signals_bo[:, -1]  # 结构突破
latest_returns = returns_arr[:, -1]  # 今日涨跌

results = pd.DataFrame({
    "asset_id": assets,
    "breakout": latest_breakout.astype(bool),
    "strength": latest_strength,
    "returns": latest_returns
})

# 筛选结构突破 + 排序
top_stocks = results[results["breakout"]].sort_values("strength", ascending=False)

print("\nTOP 10 - 结构突破 + 吸筹强度")
print(f"{'排名':<4} {'股票代码':<12} {'吸筹强度':<14} {'今日涨跌':<10}")
for i, (_, row) in enumerate(top_stocks.head(10).iterrows(), 1):
    print(f"{i:<4} {row['asset_id']:<12} {row['strength']:<14.6f} {row['returns']*100:>+6.2f}%")
```

**输出**：
```
TOP 10 - 结构突破 + 吸筹强度
排名   股票代码         吸筹强度           今日涨跌
1    000656.SZ    0.488458       +2.10%
2    603838.SH    0.443803       -1.33%
3    601928.SH    0.376669       +1.53%
4    688683.SH    0.327062       +2.30%
5    301307.SZ    0.316631       +8.10%
...
```

---

## 完整代码模板

```python
#!/usr/bin/env python3
"""因子挖掘模板 - AShare 数据"""

import numpy as np
import pandas as pd
from factorminer.data import AShareDataLoader
from factorminer.core.parser import parse
from factorminer.evaluation.metrics import compute_ic, compute_ic_mean, compute_ic_abs_mean, compute_icir

def load_astock_panel(count=60, adj="hfq", features=None):
    """加载 AShare 数据并构建 Panel"""
    if features is None:
        features = ["close", "volume", "returns", "net_mf_vol"]
    
    loader = AShareDataLoader(count=count, adj=adj)
    df = loader.load()
    loader.close()
    
    df = df.sort_values(["asset_id", "datetime"])
    assets = df["asset_id"].unique()
    dates = np.sort(df["datetime"].unique())
    
    panels = {}
    for feat in features:
        arr = df.pivot(index="datetime", columns="asset_id", values=feat) \
                .reindex(dates, columns=assets) \
                .to_numpy(dtype=np.float64, na_value=0.0).T
        panels[f"${feat}"] = arr
    
    return panels, assets, dates

def evaluate_factor(signals, returns, name="Factor"):
    """评估因子"""
    ic = compute_ic(signals, returns)
    valid = ic[~np.isnan(ic)]
    
    if len(valid) == 0:
        print(f"{name}: No valid IC")
        return None
    
    return {
        "name": name,
        "ic_mean": np.mean(valid),
        "ic_abs_mean": np.mean(np.abs(valid)),
        "icir": np.mean(valid) / np.std(valid),
        "win_rate": np.mean(valid > 0)
    }

def main():
    # 1. 加载数据
    data_dict, assets, dates = load_astock_panel(count=60)
    
    # 2. 定义因子
    factors = {
        "吸筹强度": 'Div($net_mf_vol, Add($volume, 1))',
        "价格>MA20": 'Greater($close, Mean($close, 20))',
    }
    
    # 3. 计算并评估
    results = []
    for name, formula in factors.items():
        tree = parse(formula)
        signals = tree.evaluate(data_dict)
        result = evaluate_factor(signals, data_dict["$returns"], name)
        if result:
            results.append(result)
    
    # 4. 打印结果
    print(f"\n{'='*60}")
    print("因子评估结果")
    print(f"{'='*60}")
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  IC Mean:     {r['ic_mean']:.4f}")
        print(f"  IC Abs Mean: {r['ic_abs_mean']:.4f}")
        print(f"  ICIR:        {r['icir']:.4f}")
        print(f"  Win Rate:    {r['win_rate']*100:.1f}%")

if __name__ == "__main__":
    main()
```
