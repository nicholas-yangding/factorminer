"""LLM agent integration for factor generation."""

from factorminer.agent.factor_generator import FactorGenerator
from factorminer.agent.llm_interface import (
    AnthropicProvider,
    GoogleProvider,
    LLMProvider,
    MockProvider,
    OpenAIProvider,
    create_provider,
)
from factorminer.agent.output_parser import CandidateFactor, parse_llm_output
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.agent.specialists import (
    LIQUIDITY_SPECIALIST,
    MOMENTUM_SPECIALIST,
    VOLATILITY_SPECIALIST,
    DEFAULT_SPECIALISTS,
    SpecialistConfig,
    SpecialistPromptBuilder,
)
from factorminer.agent.critic import CriticAgent, CriticScore
from factorminer.agent.debate import DebateConfig, DebateGenerator

__all__ = [
    # Generator
    "FactorGenerator",
    # LLM providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MockProvider",
    "create_provider",
    # Parsing
    "CandidateFactor",
    "parse_llm_output",
    # Prompt
    "PromptBuilder",
    # Specialists
    "SpecialistConfig",
    "SpecialistPromptBuilder",
    "MOMENTUM_SPECIALIST",
    "VOLATILITY_SPECIALIST",
    "LIQUIDITY_SPECIALIST",
    "DEFAULT_SPECIALISTS",
    # Critic
    "CriticAgent",
    "CriticScore",
    # Debate
    "DebateGenerator",
    "DebateConfig",
]
