"""Tests for the multi-agent debate orchestrator (agent/debate.py)."""

from __future__ import annotations

import pytest

from factorminer.agent.critic import CriticAgent
from factorminer.agent.debate import DebateConfig, DebateGenerator
from factorminer.agent.llm_interface import MockProvider
from factorminer.agent.output_parser import CandidateFactor
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.agent.specialists import (
    SpecialistConfig,
    SpecialistPromptBuilder,
)


# -----------------------------------------------------------------------
# SpecialistConfig and SpecialistPromptBuilder
# -----------------------------------------------------------------------

def test_specialist_config_creation():
    cfg = SpecialistConfig(
        name="test_spec",
        domain="testing domain",
        preferred_operators=["CsRank", "Neg"],
        preferred_features=["$close"],
        temperature=0.7,
    )
    assert cfg.name == "test_spec"
    assert "CsRank" in cfg.preferred_operators


def test_specialist_prompt_builder_inherits():
    """SpecialistPromptBuilder should be a subclass of PromptBuilder."""
    assert issubclass(SpecialistPromptBuilder, PromptBuilder)


def test_specialist_prompt_builder_creates():
    cfg = SpecialistConfig(
        name="momentum",
        domain="trend-following",
        preferred_operators=["Delta"],
        preferred_features=["$close"],
        system_prompt_suffix="Focus on momentum.",
    )
    pb = SpecialistPromptBuilder(specialist_config=cfg)
    assert "SPECIALIST DOMAIN DIRECTIVE" in pb.system_prompt
    assert "Focus on momentum." in pb.system_prompt


# -----------------------------------------------------------------------
# CriticAgent with MockProvider
# -----------------------------------------------------------------------

def test_critic_agent_with_mock():
    """CriticAgent should produce scores when given proposals."""
    provider = MockProvider()
    critic = CriticAgent(llm_provider=provider)

    candidates = [
        CandidateFactor(name="f1", formula="Neg($close)", category="test"),
        CandidateFactor(name="f2", formula="CsRank($volume)", category="test"),
    ]
    proposals = {"test_specialist": candidates}

    scores = critic.review_candidates(
        proposals=proposals,
        library_state={"size": 0},
        memory_signal={},
    )
    # Should return scores (fallback uniform if parsing fails)
    assert len(scores) >= 2
    assert all(hasattr(s, "final_score") for s in scores)


# -----------------------------------------------------------------------
# DebateGenerator.generate_batch returns List[CandidateFactor]
# -----------------------------------------------------------------------

def test_debate_generator_returns_candidates():
    provider = MockProvider()
    gen = DebateGenerator(
        llm_provider=provider,
        debate_config=DebateConfig(
            enable_critic=False,
            candidates_per_specialist=5,
        ),
    )
    result = gen.generate_batch(batch_size=10)
    assert isinstance(result, list)
    # Should have some candidates (specialists produce them)
    assert len(result) > 0
    assert all(isinstance(c, CandidateFactor) for c in result)


# -----------------------------------------------------------------------
# DebateGenerator with critic produces non-empty results
# -----------------------------------------------------------------------

def test_debate_generator_with_critic():
    provider = MockProvider()
    gen = DebateGenerator(
        llm_provider=provider,
        debate_config=DebateConfig(
            enable_critic=True,
            candidates_per_specialist=5,
            top_k_after_critic=10,
        ),
    )
    result = gen.generate_batch(batch_size=10)
    assert isinstance(result, list)
    assert len(result) > 0
