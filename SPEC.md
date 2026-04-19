# AShare Integration - Implementation Spec

## Status: Feature Complete

## Overview

Integrate `stock-data` SDK into FactorMiner to enable A-share market factor research with moneyflow features.

## Completed

### 1. AShareDataLoader (`factorminer/data/astock.py`)
- [x] Basic data loading (OHLCV)
- [x] Moneyflow integration (net_mf_vol, buy_lg_vol, etc.)
- [x] Adj type support (qfq/hfq)
- [x] Context manager support
- [x] Returns computation

### 2. Feature Registration (`factorminer/core/types.py`)
- [x] Added moneyflow features: `$net_mf_vol`, `$net_mf_amount`, `$lg_buy_vol`, `$lg_sell_vol`, `$elg_buy_vol`, `$elg_sell_vol`

### 3. Metrics Fix (`factorminer/evaluation/metrics.py`)
- [x] Added `compute_ic_abs_mean` function
- [x] Fixed `compute_ic_mean` to return signed IC (not absolute)

### 4. Tests
- [x] AShareDataLoader tests (`test_astock.py`)
- [x] IC Metrics TDD tests (`test_ic_metrics_tdd.py`)
- [x] Validation IC TDD tests (`test_validation_ic_tdd.py`)
- [x] AShare Demo tests (`test_astock_demo.py`) - factor computation with IC evaluation

## TDD Process

### RED Phase (Tests Written First)
1. Wrote failing tests for IC metrics bugs
2. Tests verified they FAIL with old code

### GREEN Phase (Fix Applied)
1. Fix already present in `origin/main` (commit `bcd371c`)
2. Tests now pass

### Current Status
- All 28 relevant tests passing
- TDD tests define correct behavior for IC metrics

## Architecture Notes

### Data Flow
```
stock-data (DuckDB)
    ↓
AShareDataLoader.load()
    ↓ (returns pd.DataFrame)
preprocess() + build_tensor()
    ↓ (returns TensorDataset)
ExpressionTree.evaluate(data_dict)
    ↓ (data_dict maps $feature → (M,T) arrays)
Factor signals
```

### Key Insight: $ Prefix Mapping
- DSL uses `$net_mf_vol` 
- DataFrame column is `net_mf_vol`
- TensorDataset feature_name is `net_mf_vol`
- Must add `$` prefix when building data_dict for evaluate()

## Test Commands

```bash
# Run all relevant tests
cd .worktrees/astock-loader
python -m pytest factorminer/tests/test_ic_metrics_tdd.py \
                 factorminer/tests/test_validation_ic_tdd.py \
                 factorminer/tests/test_astock.py \
                 factorminer/tests/test_astock_demo.py \
                 -v

# Manual verification
python3 -c "
from factorminer.data import AShareDataLoader
loader = AShareDataLoader(ts_codes=['000001.SZ'], count=60)
df = loader.load()
print(df.columns.tolist())
loader.close()
"
```

## Pending Items

### High Priority
- [x] TDD tests for IC metrics - DONE
- [x] Verify all tests pass - DONE

### Medium Priority  
- [ ] Document usage in guide
- [ ] CLI integration for AShareDataLoader

### Future Features (User Requested)
- [ ] 主力吸筹因子 DSL 模板
- [ ] 结构评分因子 DSL 模板
- [ ] 超参优化框架

## Git Log

```
93074bd Add test_astock_demo.py with AShare data factor demo tests
9692b0b test: add TDD tests for IC metrics and validation pipeline
4694ac6 feat: add AShareDataLoader for stock-data SDK integration with moneyflow features
bcd371c fix: correct IC metric assignments - use signed ic_mean, absolute comparison for threshold
```

## Demo Results

```
Factor                                      IC Mean     IC Abs       ICIR
------------------------------------------------------------------------
Accumulation Days (20d)                     -0.0836     0.3710    -0.1623
Accumulation Strength                        0.1660
Combined (Days x Strength)                   0.1642
```

## Next Steps

1. [x] AShare demo tests - DONE
2. [ ] Write usage documentation in guide/
3. [ ] Implement CLI integration for AShareDataLoader
4. [ ] Create DSL templates for 主力吸筹因子
