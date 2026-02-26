"""Specialist agent configurations for domain-focused factor generation.

Each specialist focuses on a particular alpha factor domain (momentum,
volatility, liquidity) with preferred operators, features, and temperature
settings.  ``SpecialistPromptBuilder`` extends the base ``PromptBuilder``
to inject domain-specific directives into the system and user prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from factorminer.agent.prompt_builder import SYSTEM_PROMPT, PromptBuilder


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpecialistConfig:
    """Configuration for a domain-specialist factor generator.

    Attributes
    ----------
    name : str
        Human-readable specialist name (e.g. ``"momentum"``).
    domain : str
        Domain description used in prompt directives.
    preferred_operators : list[str]
        Operator names this specialist should emphasise.
    preferred_features : list[str]
        Raw features this specialist should lean towards.
    temperature : float
        Sampling temperature for LLM calls.
    system_prompt_suffix : str
        Extra paragraph appended to the system prompt.
    provider_config : dict
        Optional provider-level overrides (model, max_tokens, etc.).
    """

    name: str
    domain: str
    preferred_operators: List[str]
    preferred_features: List[str]
    temperature: float = 0.8
    system_prompt_suffix: str = ""
    provider_config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pre-defined specialist constants
# ---------------------------------------------------------------------------

MOMENTUM_SPECIALIST = SpecialistConfig(
    name="momentum",
    domain="trend-following and momentum",
    preferred_operators=[
        "TsRank", "Delta", "Delay", "Return", "LogReturn",
        "TsLinRegSlope", "Slope",
    ],
    preferred_features=["$close", "$returns"],
    temperature=0.85,
    system_prompt_suffix=(
        "You are a MOMENTUM specialist.  Focus on trend-following and "
        "mean-reversion signals.  Exploit price persistence, serial "
        "correlation in returns, and time-series rank dynamics.  "
        "Prefer directional operators (Delta, Return, LogReturn, "
        "TsLinRegSlope) to capture price trajectory information."
    ),
)

VOLATILITY_SPECIALIST = SpecialistConfig(
    name="volatility",
    domain="volatility and regime-switching",
    preferred_operators=[
        "Std", "Var", "Kurt", "Skew", "IfElse", "Greater", "Less",
    ],
    preferred_features=["$returns", "$high", "$low"],
    temperature=0.9,
    system_prompt_suffix=(
        "You are a VOLATILITY specialist.  Focus on regime-switching "
        "signals, conditional dispersion, and higher-moment dynamics.  "
        "Combine statistical operators (Std, Var, Kurt, Skew) with "
        "logical branching (IfElse, Greater, Less) to capture "
        "asymmetric behaviour in up-vs-down volatility regimes."
    ),
)

LIQUIDITY_SPECIALIST = SpecialistConfig(
    name="liquidity",
    domain="cross-sectional liquidity",
    preferred_operators=[
        "CsRank", "CsZScore", "Corr", "Cov", "CsScale",
    ],
    preferred_features=["$volume", "$amt", "$vwap"],
    temperature=0.85,
    system_prompt_suffix=(
        "You are a LIQUIDITY specialist.  Focus on cross-sectional "
        "liquidity patterns: volume-price divergence, turnover "
        "anomalies, and VWAP-based microstructure signals.  Lean "
        "heavily on cross-sectional normalization (CsRank, CsZScore, "
        "CsScale) and correlation/covariance operators to capture "
        "relative liquidity positioning across the stock universe."
    ),
)

DEFAULT_SPECIALISTS: List[SpecialistConfig] = [
    MOMENTUM_SPECIALIST,
    VOLATILITY_SPECIALIST,
    LIQUIDITY_SPECIALIST,
]


# ---------------------------------------------------------------------------
# SpecialistPromptBuilder
# ---------------------------------------------------------------------------

class SpecialistPromptBuilder(PromptBuilder):
    """Prompt builder that injects domain-specific specialist directives.

    Extends the base system prompt with a specialist suffix and biases
    the user prompt towards the specialist's preferred operators and
    features.

    Parameters
    ----------
    specialist_config : SpecialistConfig
        The specialist configuration to use.
    base_system_prompt : str or None
        Override for the base system prompt.  Defaults to the global
        ``SYSTEM_PROMPT`` from :mod:`factorminer.agent.prompt_builder`.
    """

    def __init__(
        self,
        specialist_config: SpecialistConfig,
        base_system_prompt: Optional[str] = None,
    ) -> None:
        base = base_system_prompt or SYSTEM_PROMPT
        # Append specialist domain directives to the system prompt.
        modified_system = (
            f"{base}\n\n"
            f"## SPECIALIST DOMAIN DIRECTIVE\n"
            f"{specialist_config.system_prompt_suffix}"
        )
        super().__init__(system_prompt=modified_system)
        self._specialist = specialist_config

    @property
    def specialist_config(self) -> SpecialistConfig:
        """Return the underlying specialist configuration."""
        return self._specialist

    def build_user_prompt(
        self,
        memory_signal: Dict[str, Any],
        library_state: Dict[str, Any],
        batch_size: int = 40,
    ) -> str:
        """Build user prompt with specialist operator/feature bias.

        Calls the base ``PromptBuilder.build_user_prompt`` and appends a
        directive asking the specialist to focus roughly 60 % of its
        candidates on its preferred operators and features.

        Parameters
        ----------
        memory_signal : dict
            Memory priors (recommended/forbidden directions, etc.).
        library_state : dict
            Current library state (size, saturation, etc.).
        batch_size : int
            Number of candidates to generate.

        Returns
        -------
        str
            Assembled user prompt with specialist bias section.
        """
        base_prompt = super().build_user_prompt(
            memory_signal=memory_signal,
            library_state=library_state,
            batch_size=batch_size,
        )

        ops = ", ".join(self._specialist.preferred_operators)
        feats = ", ".join(self._specialist.preferred_features)

        specialist_section = (
            f"\n## SPECIALIST FOCUS\n"
            f"As the {self._specialist.domain} specialist, focus ~60% of "
            f"candidates on {{{ops}}} operators applied to {{{feats}}} "
            f"features.  The remaining ~40% should explore creative "
            f"cross-domain combinations to maintain diversity."
        )

        return base_prompt + specialist_section
