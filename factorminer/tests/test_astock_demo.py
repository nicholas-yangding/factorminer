"""Demo: AShare Data Loading and Factor Computation.

This test demonstrates:
1. Loading A-share market data with moneyflow features
2. Building a factor using the DSL
3. Computing factor signals
4. Evaluating the factor with IC metrics

Run with:
    python -m pytest factorminer/tests/test_astock_demo.py -v -s
"""

import numpy as np
import pandas as pd
import pytest


def test_astock_accumulation_factor_demo():
    """Demo: Compute a 主力吸筹 factor using AShare data.
    
    Factor logic:
    - 近20天主力净流入天数越多越好
    - 吸筹强度 = 主力净流入 / 成交量
    """
    from factorminer.data import AShareDataLoader
    from factorminer.core.parser import parse
    from factorminer.evaluation.metrics import compute_ic, compute_ic_mean, compute_icir

    # Step 1: Load raw data (NOT preprocessed for IC calculation)
    # NOTE: Need enough stocks (M > 30) for meaningful IC calculation
    print("\n=== Step 1: Load Data ===")
    print("NOTE: Using all available stocks for meaningful IC calculation")
    loader = AShareDataLoader(
        ts_codes=None,  # Load all available stocks
        count=60,  # 3 months for faster test
        adj="hfq",  # 后复权
    )
    df = loader.load()
    loader.close()
    
    print(f"Loaded {len(df)} rows, {df['asset_id'].nunique()} stocks")
    print(f"Date range: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"Features: close, volume, returns, net_mf_vol, ...")

    # Step 2: Build panel format for factor computation
    print("\n=== Step 2: Build Panel Data ===")
    
    # Pivot to panel: (M stocks x T time periods)
    df_sorted = df.sort_values(["asset_id", "datetime"])
    
    # Get unique assets and dates
    assets = df_sorted["asset_id"].unique()
    dates = df_sorted["datetime"].unique()
    dates = np.sort(dates)
    
    M, T = len(assets), len(dates)
    print(f"Panel: {M} stocks x {T} time periods")
    
    # Build feature arrays (M, T) - transpose from (T, M) pivot result
    close_df = df_sorted.pivot(index="datetime", columns="asset_id", values="close").reindex(dates, columns=assets)
    volume_df = df_sorted.pivot(index="datetime", columns="asset_id", values="volume").reindex(dates, columns=assets)
    returns_df = df_sorted.pivot(index="datetime", columns="asset_id", values="returns").reindex(dates, columns=assets)
    net_mf_df = df_sorted.pivot(index="datetime", columns="asset_id", values="net_mf_vol").reindex(dates, columns=assets)
    
    # Convert to numeric arrays, replacing NA/NaN with 0
    close_arr = close_df.to_numpy(dtype=np.float64, na_value=0.0).T
    volume_arr = volume_df.to_numpy(dtype=np.float64, na_value=0.0).T
    returns_arr = returns_df.to_numpy(dtype=np.float64, na_value=0.0).T
    net_mf_arr = net_mf_df.to_numpy(dtype=np.float64, na_value=0.0).T
    
    print(f"Arrays: close={close_arr.shape}, returns={returns_arr.shape}")
    
    print(f"Arrays: close={close_arr.shape}, returns={returns_arr.shape}")

    # Step 3: Build data dict for DSL evaluation
    print("\n=== Step 3: Build Data Dict ===")
    data_dict = {
        "$close": close_arr,
        "$volume": volume_arr,
        "$returns": returns_arr,
        "$net_mf_vol": net_mf_arr,
    }
    print(f"Data keys: {list(data_dict.keys())}")

    # Step 4: Parse and evaluate factor
    print("\n=== Step 4: Factor - 主力吸筹天数 ===")
    
    # 近20天主力净流入天数
    formula1 = 'Sum(IfElse(Greater($net_mf_vol, 0), 1, 0), 20)'
    tree1 = parse(formula1)
    signals1 = tree1.evaluate(data_dict)
    print(f"Formula: {tree1.to_string()}")
    print(f"Signals shape: {signals1.shape}")  # (M, T)
    
    # Show accumulation days for first stock
    print(f"Accumulation days (stock 0, last 10 days): {signals1[0, -10:].astype(int)}")

    # Step 5: Compute IC
    print("\n=== Step 5: IC Evaluation ===")
    
    ic_series = compute_ic(signals1, returns_arr)
    valid_ic = ic_series[~np.isnan(ic_series)]
    
    ic_mean = float(np.mean(valid_ic)) if len(valid_ic) > 0 else 0.0
    ic_abs_mean = float(np.mean(np.abs(valid_ic))) if len(valid_ic) > 0 else 0.0
    icir = compute_icir(ic_series)
    ic_win_rate = float(np.mean(valid_ic > 0)) if len(valid_ic) > 0 else 0.0
    
    print(f"IC Mean (signed): {ic_mean:.4f}")
    print(f"IC Abs Mean: {ic_abs_mean:.4f}")
    print(f"ICIR: {icir:.4f}")
    print(f"IC Win Rate: {ic_win_rate*100:.1f}%")

    # Step 6: Another factor - 吸筹强度
    print("\n=== Step 6: Factor - 吸筹强度 ===")
    
    # 吸筹强度 = 主力净流入 / 成交量
    formula2 = 'Div($net_mf_vol, Add($volume, 1))'
    tree2 = parse(formula2)
    signals2 = tree2.evaluate(data_dict)
    print(f"Formula: {tree2.to_string()}")
    
    ic_series2 = compute_ic(signals2, returns_arr)
    valid_ic2 = ic_series2[~np.isnan(ic_series2)]
    ic_mean2 = float(np.mean(valid_ic2)) if len(valid_ic2) > 0 else 0.0
    print(f"IC Mean: {ic_mean2:.4f}")

    # Step 7: Combined factor
    print("\n=== Step 7: Combined Factor ===")
    
    # 综合: 吸筹天数 * 吸筹强度
    combined = signals1 * signals2
    ic_series3 = compute_ic(combined, returns_arr)
    valid_ic3 = ic_series3[~np.isnan(ic_series3)]
    ic_mean3 = float(np.mean(valid_ic3)) if len(valid_ic3) > 0 else 0.0
    print(f"Combined IC Mean: {ic_mean3:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Factor':<40} {'IC Mean':>10} {'IC Abs':>10} {'ICIR':>10}")
    print("-" * 72)
    print(f"{'Accumulation Days (20d)':<40} {ic_mean:>10.4f} {ic_abs_mean:>10.4f} {icir:>10.4f}")
    print(f"{'Accumulation Strength':<40} {ic_mean2:>10.4f}")
    print(f"{'Combined (Days x Strength)':<40} {ic_mean3:>10.4f}")


def test_astock_structure_factor_demo():
    """Demo: Compute a 结构突破 factor using AShare data.
    
    Factor logic:
    - 结构突破: 价格在均线 + 成交量放大
    - 近20天动量
    """
    from factorminer.data import AShareDataLoader
    from factorminer.core.parser import parse
    from factorminer.evaluation.metrics import compute_ic, compute_ic_mean

    print("\n" + "=" * 60)
    print("STRUCTURE BREAKOUT FACTOR DEMO")
    print("=" * 60)

    # Load data - use all stocks for meaningful IC
    print("NOTE: Using all available stocks for meaningful IC calculation")
    loader = AShareDataLoader(
        ts_codes=None,
        count=60,
        adj="hfq",
    )
    df = loader.load()
    loader.close()

    # Build panel
    df_sorted = df.sort_values(["asset_id", "datetime"])
    assets = df_sorted["asset_id"].unique()
    dates = np.sort(df_sorted["datetime"].unique())
    
    close_df = df_sorted.pivot(index="datetime", columns="asset_id", values="close").reindex(dates, columns=assets)
    volume_df = df_sorted.pivot(index="datetime", columns="asset_id", values="volume").reindex(dates, columns=assets)
    returns_df = df_sorted.pivot(index="datetime", columns="asset_id", values="returns").reindex(dates, columns=assets)
    
    close_arr = close_df.values.T
    volume_arr = volume_df.values.T
    returns_arr = returns_df.values.T
    
    data_dict = {
        "$close": close_arr,
        "$volume": volume_arr,
        "$returns": returns_arr,
    }

    # Factor: Price > MA20
    formula1 = 'Greater($close, Mean($close, 20))'
    tree1 = parse(formula1)
    signals1 = tree1.evaluate(data_dict)
    ic1 = compute_ic_mean(compute_ic(signals1, returns_arr))
    print(f"\nPrice > MA20: IC = {ic1:.4f}")

    # Volume surge
    formula2 = 'Greater($volume, Mean($volume, 20))'
    tree2 = parse(formula2)
    signals2 = tree2.evaluate(data_dict)
    ic2 = compute_ic_mean(compute_ic(signals2, returns_arr))
    print(f"Volume > MA20: IC = {ic2:.4f}")

    # Combined: price above MA20 AND volume surge
    formula3 = 'Mul(Greater($close, Mean($close, 20)), Greater($volume, Mean($volume, 20)))'
    tree3 = parse(formula3)
    signals3 = tree3.evaluate(data_dict)
    ic3 = compute_ic_mean(compute_ic(signals3, returns_arr))
    print(f"Combined (AND): IC = {ic3:.4f}")

    # Factor: 20-day momentum
    formula4 = 'TsRank(Return($close, 20), 20)'
    tree4 = parse(formula4)
    signals4 = tree4.evaluate(data_dict)
    ic4 = compute_ic_mean(compute_ic(signals4, returns_arr))
    print(f"Momentum TsRank: IC = {ic4:.4f}")

    # Combined: Structure + Momentum
    print("\n=== Structure Breakout + Momentum ===")
    combined = signals3 * signals4
    ic_combined = compute_ic_mean(compute_ic(combined, returns_arr))
    print(f"Combined IC: {ic_combined:.4f}")


if __name__ == "__main__":
    test_astock_accumulation_factor_demo()
    test_astock_structure_factor_demo()
