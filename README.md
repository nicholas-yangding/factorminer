# FactorMiner

**A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

FactorMiner is an LLM-powered framework for automated discovery of interpretable formulaic alpha factors. It combines a modular mining skill architecture with structured experience memory to iteratively explore the vast space of factor expressions while avoiding redundancy.

Based on the paper: *FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery* (Wang et al., 2026) &mdash; [arXiv:2602.14670](https://arxiv.org/abs/2602.14670)

---

## Overview

Formulaic alpha factor mining is central to quantitative trading. The search space of operator compositions is combinatorially vast, and as a factor library grows, finding novel signals that are both predictive and orthogonal to existing factors becomes increasingly difficult (the "Correlation Red Sea" problem).

FactorMiner addresses this through two synergistic mechanisms:

1. **Modular Skill Architecture** &mdash; Factor mining is encapsulated as a reusable Agent Skill with 60+ typed financial operators, a multi-stage validation pipeline (IC screening, correlation checking, deduplication, full validation), and standardized evaluation protocols.

2. **Experience Memory** &mdash; A structured knowledge base that accumulates insights across mining sessions: successful patterns (templates that consistently pass quality thresholds), forbidden regions (factor families with high correlation to existing library members), and strategic insights. The memory guides future exploration away from known dead ends and toward promising directions.

The system implements the **Ralph Loop** paradigm: *retrieve* memory priors, *generate* candidates via LLM, *evaluate* through the multi-stage pipeline, *admit* to the library, and *distill* outcomes back into memory.

## Key Features

- **110 pre-built alpha factors** from the paper with explicit formulaic expressions, validated on A-share equities and cryptocurrency markets
- **60+ typed financial operators** across 7 categories (arithmetic, statistical, time-series, cross-sectional, smoothing, regression, logical) with GPU-accelerated backends
- **Expression tree DSL** &mdash; domain-specific language for composing factor formulas as symbolic expression trees with recursive descent parsing
- **Multi-stage validation pipeline** &mdash; 4-stage cascade (fast IC screen &rarr; correlation check &rarr; batch deduplication &rarr; full validation) with factor replacement mechanism
- **Experience memory system** &mdash; formation, evolution, and retrieval operators for accumulating structural knowledge across mining sessions
- **Factor combination** &mdash; equal-weight, IC-weighted, and Gram-Schmidt orthogonal strategies
- **Factor selection** &mdash; Lasso, forward stepwise, and XGBoost methods
- **SymPy canonicalization** &mdash; algebraic equivalence detection (e.g., `Neg(Neg($close))` == `$close`)
- **Regime-aware evaluation** &mdash; market regime detection (bull/bear/neutral) with per-regime IC analysis
- **Statistical significance testing** &mdash; bootstrap confidence intervals, FDR correction, deflated Sharpe ratio
- **Knowledge graph memory** &mdash; graph-based representation of factor relationships, operator co-occurrence, and saturated clusters
- **Multi-backend acceleration** &mdash; NumPy (CPU), C-compiled (bottleneck), and PyTorch/CUDA (GPU) backends with 8&ndash;59x speedups

## Architecture

```
                    ┌─────────────────────────────┐
                    │   EXPERIENCE MEMORY (M)      │
                    │  P_succ │ P_fail │ Insights  │
                    └────┬────────────────┬────────┘
                         │                │
                  1. RETRIEVE        5. DISTILL
                         │                │
                         ▼                │
┌──────────┐    ┌────────────────┐    ┌───┴──────────────┐
│  MARKET  │───▶│  AGENT SKILL   │───▶│  FACTOR LIBRARY  │
│  DATA    │    │                │    │                  │
│  (D)     │    │ 2. GENERATE    │    │  4. ADMIT/       │
│          │    │    (LLM + Ω)   │    │     REPLACE      │
└──────────┘    │                │    └──────────────────┘
                │ 3. EVALUATE    │
                │  Stage 1: IC   │
                │  Stage 2: Corr │
                │  Stage 3: Dedup│
                │  Stage 4: Full │
                └────────────────┘
```

## Project Structure

```
alphadisk/
├── pyproject.toml                  # Package config and dependencies
├── run_demo.py                     # End-to-end demo (no API keys needed)
├── 2602.14670v1.pdf                # Source paper
├── data/
│   └── binance_crypto_5m.csv       # Sample Binance crypto 5-min bars
├── output/                         # Mining session outputs
│   ├── factor_library.json         # Discovered factor library
│   ├── mining.log                  # Detailed mining log
│   ├── mining_batches.jsonl        # Batch-level results
│   ├── session.json                # Session metadata
│   └── session_log.json            # Session event log
└── factorminer/                    # Main package
    ├── core/
    │   ├── expression_tree.py      # DAG-based factor expression trees
    │   ├── parser.py               # Recursive descent formula parser
    │   ├── factor_library.py       # Factor library with admission/replacement
    │   ├── ralph_loop.py           # Main mining loop (Algorithm 1)
    │   ├── helix_loop.py           # Extended loop with Phase 2 features
    │   ├── canonicalizer.py        # SymPy-based formula canonicalization
    │   ├── config.py               # Mining configuration
    │   ├── session.py              # Session persistence and resume
    │   ├── library_io.py           # Library serialization + 110 paper factors
    │   └── types.py                # Operator specs and type system
    ├── agent/
    │   ├── llm_interface.py        # LLM provider abstraction (+ MockProvider)
    │   ├── factor_generator.py     # LLM-based factor generation
    │   ├── prompt_builder.py       # Prompt construction with memory injection
    │   ├── output_parser.py        # Parse LLM output to CandidateFactor
    │   ├── debate.py               # Multi-agent debate framework
    │   ├── critic.py               # Quality validation critic
    │   └── specialists.py          # Domain-specialist generators
    ├── operators/
    │   ├── registry.py             # Central operator registry
    │   ├── arithmetic.py           # Add, Sub, Mul, Div, Log, Sqrt, ...
    │   ├── statistical.py          # Mean, Std, Skew, Kurt, Median, ...
    │   ├── timeseries.py           # Delta, Delay, Corr, Cov, Beta, ...
    │   ├── crosssectional.py       # CsRank, CsZScore, CsDemean, ...
    │   ├── smoothing.py            # EMA, SMA, WMA, KAMA, HMA
    │   ├── regression.py           # Slope, Rsquare, Resi (rolling OLS)
    │   ├── logical.py              # IfElse, Greater, Less, And, Or
    │   ├── gpu_backend.py          # PyTorch/CUDA implementations
    │   ├── custom.py               # User-defined operators
    │   └── auto_inventor.py        # Automatic operator discovery
    ├── evaluation/
    │   ├── metrics.py              # IC, ICIR, win rate, factor stats
    │   ├── pipeline.py             # Multi-stage validation pipeline
    │   ├── admission.py            # Admission and replacement logic
    │   ├── correlation.py          # Pairwise Spearman correlation
    │   ├── combination.py          # EW, IC-weighted, orthogonal combination
    │   ├── selection.py            # Lasso, stepwise, XGBoost selection
    │   ├── backtest.py             # Backtesting framework
    │   ├── portfolio.py            # Portfolio construction
    │   ├── regime.py               # Market regime detection
    │   ├── significance.py         # Bootstrap CI, FDR, deflated Sharpe
    │   ├── causal.py               # Causal factor analysis
    │   └── capacity.py             # Factor capacity estimation
    ├── memory/
    │   ├── memory_store.py         # Experience memory data structures
    │   ├── formation.py            # Memory formation operator F
    │   ├── evolution.py            # Memory evolution operator E
    │   ├── retrieval.py            # Memory retrieval operator R
    │   ├── knowledge_graph.py      # Factor knowledge graph
    │   ├── kg_retrieval.py         # KG-based retrieval
    │   ├── embeddings.py           # Formula/pattern embeddings
    │   └── experience_memory.py    # Unified memory interface
    ├── data/
    │   ├── loader.py               # CSV/Parquet/HDF5 data loading
    │   ├── preprocessor.py         # Cleaning, NaN handling, normalization
    │   ├── tensor_builder.py       # DataFrame to (M, T) array conversion
    │   └── mock_data.py            # Synthetic OHLCV data generation
    ├── utils/
    │   ├── logging.py              # Session logging and iteration tracking
    │   ├── visualization.py        # IC plots, correlation heatmaps
    │   ├── tearsheet.py            # Factor tearsheet generation
    │   └── reporting.py            # HTML/PDF/Markdown reports
    ├── configs/
    │   └── __init__.py             # Configuration defaults
    └── tests/                      # Comprehensive pytest suite
        ├── test_expression_tree.py
        ├── test_operators.py
        ├── test_library.py
        ├── test_data.py
        ├── test_evaluation.py
        ├── test_memory.py
        ├── test_ralph_loop.py
        ├── test_combination.py
        ├── test_regime.py
        ├── test_significance.py
        ├── test_knowledge_graph.py
        ├── test_helix_loop.py
        ├── test_debate.py
        └── conftest.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/factorminer/factorminer.git
cd factorminer

# Install base package
pip install -e .

# Install with LLM support (OpenAI, Anthropic, Google)
pip install -e ".[llm]"

# Install with GPU acceleration
pip install -e ".[gpu]"

# Install everything (GPU + LLM + dev tools)
pip install -e ".[all]"
```

**Requirements:** Python 3.10+

## Quick Start

### Run the Demo (No API Keys Needed)

The demo runs the full pipeline on synthetic data using a MockProvider for LLM generation:

```bash
python run_demo.py
```

This demonstrates:
1. Synthetic market data generation (100 assets, 500 periods)
2. Evaluation of the paper's 110 factors
3. Factor library construction with admission rules (IC > 0.02, correlation < 0.5)
4. Factor combination (equal-weight, IC-weighted, orthogonal)
5. Regime detection (bull/bear/neutral)
6. Statistical significance testing (bootstrap CI + FDR)
7. SymPy formula canonicalization
8. Knowledge graph memory
9. Mining loop (3 iterations)

### Run with a Real LLM

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# Run the mining loop
factorminer mine --config factorminer/configs/default.yaml
```

### Run with Real Market Data

Place a CSV file with columns `[datetime, asset_id, open, high, low, close, volume, amount]` at `data/market.csv` and update the config accordingly.

A sample Binance crypto dataset (`data/binance_crypto_5m.csv`) is included.

## Factor Expression Language

Factors are defined as symbolic expression trees using a domain-specific language:

```python
from factorminer.core.parser import parse

# Simple momentum factor
tree = parse("Neg(CsRank(Delta($close, 3)))")

# VWAP deviation
tree = parse("Neg(Div(Sub($close, $vwap), $vwap))")

# Regime-switching factor with conditional logic
tree = parse("""
    IfElse(Greater(Std($returns, 12), Mean(Std($returns, 12), 48)),
           Neg(CsRank(Delta($close, 3))),
           Neg(CsRank(Div(Sub($close, $low), Add(Sub($high, $low), 0.0001)))))
""")

# Evaluate on data
import numpy as np
data = {"$close": close_array, "$high": high_array, ...}  # shape (M, T)
signals = tree.evaluate(data)  # shape (M, T)
```

### Available Features

| Feature | Description |
|---------|-------------|
| `$open` | Open price |
| `$high` | High price |
| `$low` | Low price |
| `$close` | Close price |
| `$volume` | Trading volume |
| `$amt` | Trading amount |
| `$vwap` | Volume-weighted average price |
| `$returns` | Log returns |

### Operator Categories

| Category | Representative Operators | Description |
|----------|------------------------|-------------|
| Arithmetic | Add, Sub, Mul, Div, Neg, Log, Sqrt, Abs | Element-wise transformations |
| Statistical | Mean, Std, Skew, Kurt, Median, Sum | Rolling window statistics |
| Time-series | Delta, Delay, TsRank, TsMax, TsMin, TsDecay | Temporal pattern capture |
| Cross-sectional | CsRank, CsZScore, CsDemean, Scale | Cross-asset transforms |
| Smoothing | SMA, EMA, WMA, KAMA, HMA | Trend extraction |
| Regression | Slope, Rsquare, Resi | Rolling OLS trend/residuals |
| Logical | IfElse, Greater, Less, And, Or | Conditional regime switching |

## Core Concepts

### The Ralph Loop (Algorithm 1)

The mining loop iterates through five stages:

```
repeat until |L| >= K or budget exhausted:
    1. RETRIEVE: m <- R(M, L)           # Get memory priors
    2. GENERATE: C ~ pi(alpha | m)      # LLM generates candidates
    3. EVALUATE:                         # Multi-stage validation
       Stage 1: Fast IC screening (subset of assets)
       Stage 2: Correlation check against library
       Stage 2.5: Replacement check for high-IC correlated factors
       Stage 3: Intra-batch deduplication
       Stage 4: Full validation (all assets)
    4. ADMIT: L <- L ∪ C_admitted       # Update library
    5. DISTILL: M <- E(M, F(tau))       # Update memory
```

### Admission Criteria

A factor alpha is admitted to the library L if:

- **IC gate:** |IC(alpha)| >= tau_IC (default: 0.04 for A-shares)
- **Correlation gate:** max |rho(alpha, g)| < theta for all g in L (default: theta = 0.5)

A **replacement** occurs when a high-IC candidate is correlated with exactly one existing factor and exceeds it by 30%+.

### Experience Memory

The memory M stores three types of knowledge:

- **Successful Patterns (P_succ):** Templates that consistently produce admitted factors (e.g., "higher moment regimes via Skew/Kurt", "price-volume correlation interaction")
- **Forbidden Directions (P_fail):** Regions identified as "Red Seas" due to persistent high correlation (e.g., "VWAP Deviation variants", "simple Delta reversals")
- **Strategic Insights (I):** High-level lessons (e.g., "non-linear combination via XGBoost outperforms linear")

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **IC** | Spearman rank correlation between factor signal and forward returns |
| **ICIR** | IC mean / IC std &mdash; measures consistency |
| **IC Win Rate** | Fraction of periods with positive IC |
| **Avg \|rho\|** | Average pairwise absolute correlation in library |
| **Monotonicity** | Whether quintile returns are monotonically ordered |
| **Turnover** | Average daily portfolio turnover |

## Testing

```bash
# Run all tests
pytest factorminer/tests/ -v

# Run specific test module
pytest factorminer/tests/test_expression_tree.py -v
pytest factorminer/tests/test_ralph_loop.py -v
```

## Configuration

Mining parameters are set via `MiningConfig`:

```python
from factorminer.core.config import MiningConfig

config = MiningConfig(
    target_library_size=110,   # Target number of factors
    batch_size=40,             # Candidates per iteration
    max_iterations=200,        # Maximum mining iterations
    ic_threshold=0.04,         # Minimum |IC| for admission
    correlation_threshold=0.5, # Maximum pairwise |rho|
    fast_screen_assets=50,     # Assets for Stage 1 screening
    num_workers=40,            # Parallel evaluation workers
    backend="numpy",           # "numpy", "c", or "gpu"
)
```

## Citation

```bibtex
@inproceedings{wang2026factorminer,
  title={FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery},
  author={Wang, Yanlong and Xu, Jian and Zhang, Hongkang and Huang, Shao-Lun and Sun, Danny Dongning and Zhang, Xiao-Ping},
  booktitle={Preprint},
  year={2026},
  publisher={ACM},
  address={New York, NY, USA},
  url={https://arxiv.org/abs/2602.14670}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

This software is for research and educational purposes. The discovered factors should not be used directly for live trading without proper risk management, compliance review, and transaction cost analysis. The authors are not responsible for any financial losses incurred from the use of this software.
