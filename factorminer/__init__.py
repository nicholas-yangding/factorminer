"""FactorMiner: LLM-powered quantitative factor mining with evolutionary search."""

__version__ = "0.1.0"
__author__ = "FactorMiner Team"

from factorminer.utils.config import (
    Config,
    MiningConfig,
    EvaluationConfig,
    DataConfig,
    LLMConfig,
    MemoryConfig,
    Phase2Config,
    CausalConfig,
    RegimeConfig,
    CapacityConfig,
    SignificanceConfig,
    DebateConfig,
    AutoInventorConfig,
    HelixConfig,
    load_config,
)

__all__ = [
    "__version__",
    "Config",
    "MiningConfig",
    "EvaluationConfig",
    "DataConfig",
    "LLMConfig",
    "MemoryConfig",
    "load_config",
    # Phase 2 configs
    "Phase2Config",
    "CausalConfig",
    "RegimeConfig",
    "CapacityConfig",
    "SignificanceConfig",
    "DebateConfig",
    "AutoInventorConfig",
    "HelixConfig",
]
