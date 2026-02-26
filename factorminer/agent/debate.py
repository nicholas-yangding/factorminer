"""Multi-agent debate orchestrator for factor generation.

``DebateGenerator`` is a **drop-in replacement** for ``FactorGenerator``.
It runs multiple domain-specialist generators, collects their proposals,
optionally passes them through a ``CriticAgent`` for scoring and ranking,
and returns a single ``List[CandidateFactor]`` with the same interface as
``FactorGenerator.generate_batch()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from factorminer.agent.critic import CriticAgent, CriticScore
from factorminer.agent.factor_generator import FactorGenerator
from factorminer.agent.llm_interface import LLMProvider
from factorminer.agent.output_parser import CandidateFactor
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.agent.specialists import (
    DEFAULT_SPECIALISTS,
    SpecialistConfig,
    SpecialistPromptBuilder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DebateConfig
# ---------------------------------------------------------------------------

@dataclass
class DebateConfig:
    """Configuration for the multi-agent debate pipeline.

    Attributes
    ----------
    specialists : list[SpecialistConfig]
        Specialist configurations to run.  Defaults to the three
        pre-defined specialists (momentum, volatility, liquidity).
    enable_critic : bool
        Whether to run the CriticAgent for scoring/ranking.
    candidates_per_specialist : int
        Number of candidates each specialist generates.
    top_k_after_critic : int
        How many candidates the critic retains after ranking.
    critic_temperature : float
        Sampling temperature for the critic LLM call.
    """

    specialists: List[SpecialistConfig] = field(
        default_factory=lambda: list(DEFAULT_SPECIALISTS)
    )
    enable_critic: bool = True
    candidates_per_specialist: int = 15
    top_k_after_critic: int = 40
    critic_temperature: float = 0.3


# ---------------------------------------------------------------------------
# DebateGenerator
# ---------------------------------------------------------------------------

class DebateGenerator:
    """Multi-agent debate-based factor generator.

    Drop-in replacement for :class:`FactorGenerator`.  Runs multiple
    specialist generators in sequence, optionally scores them with a
    ``CriticAgent``, and returns a ranked list of ``CandidateFactor``
    objects.

    Parameters
    ----------
    llm_provider : LLMProvider
        LLM backend shared across all specialists and the critic.
    debate_config : DebateConfig or None
        Pipeline configuration.  Uses defaults if ``None``.
    prompt_builder : PromptBuilder or None
        Optional base prompt builder (its system prompt may be used as
        the base for specialist prompt builders).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        debate_config: Optional[DebateConfig] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.config = debate_config or DebateConfig()

        base_system_prompt = (
            prompt_builder.system_prompt if prompt_builder else None
        )

        # Create one FactorGenerator per specialist
        self._specialist_generators: Dict[str, FactorGenerator] = {}
        for spec in self.config.specialists:
            specialist_pb = SpecialistPromptBuilder(
                specialist_config=spec,
                base_system_prompt=base_system_prompt,
            )
            gen = FactorGenerator(
                llm_provider=self.llm_provider,
                prompt_builder=specialist_pb,
                temperature=spec.temperature,
            )
            self._specialist_generators[spec.name] = gen

        # Create critic if enabled
        self._critic: Optional[CriticAgent] = None
        if self.config.enable_critic:
            self._critic = CriticAgent(
                llm_provider=self.llm_provider,
                temperature=self.config.critic_temperature,
            )

        self._generation_count = 0

    def generate_batch(
        self,
        memory_signal: Optional[Dict[str, Any]] = None,
        library_state: Optional[Dict[str, Any]] = None,
        batch_size: int = 40,
    ) -> List[CandidateFactor]:
        """Generate a batch of candidate factors via multi-agent debate.

        Signature is **identical** to ``FactorGenerator.generate_batch``
        so this class is a true drop-in replacement.

        Steps:
        1. Each specialist generates ``candidates_per_specialist`` factors.
        2. All proposals are collected into a dict keyed by specialist name.
        3. If the critic is enabled, it scores and ranks all proposals,
           returning the top-K.
        4. CriticScores are mapped back to CandidateFactor objects with
           specialist source metadata.

        Parameters
        ----------
        memory_signal : dict or None
            Memory priors for prompt injection.
        library_state : dict or None
            Current factor library state.
        batch_size : int
            Target number of candidates to return.

        Returns
        -------
        list[CandidateFactor]
            Ranked candidate factors (same type as ``FactorGenerator``).
        """
        memory_signal = memory_signal or {}
        library_state = library_state or {}

        self._generation_count += 1
        batch_id = self._generation_count

        logger.info(
            "Debate batch #%d: %d specialists, critic=%s, per_specialist=%d",
            batch_id,
            len(self._specialist_generators),
            self.config.enable_critic,
            self.config.candidates_per_specialist,
        )

        # --- Step 1: Run all specialists ---
        proposals: Dict[str, List[CandidateFactor]] = {}
        total_candidates = 0

        for spec_name, generator in self._specialist_generators.items():
            logger.info("Running specialist: %s", spec_name)
            candidates = generator.generate_batch(
                memory_signal=memory_signal,
                library_state=library_state,
                batch_size=self.config.candidates_per_specialist,
            )
            proposals[spec_name] = candidates
            total_candidates += len(candidates)
            logger.info(
                "Specialist %s produced %d valid candidates",
                spec_name,
                len(candidates),
            )

        if total_candidates == 0:
            logger.warning("Debate batch #%d: no candidates from any specialist.", batch_id)
            return []

        # --- Step 2: Critic review (if enabled) ---
        if self._critic is not None:
            logger.info(
                "Running critic on %d total candidates (top_k=%d)",
                total_candidates,
                self.config.top_k_after_critic,
            )
            critic_scores = self._critic.review_candidates(
                proposals=proposals,
                library_state=library_state,
                memory_signal=memory_signal,
                top_k=min(batch_size, self.config.top_k_after_critic),
            )

            # Map CriticScores back to CandidateFactor objects
            result = self._scores_to_candidates(critic_scores, proposals)
        else:
            # No critic -- just merge all proposals
            result = []
            for spec_name, candidates in proposals.items():
                for c in candidates:
                    result.append(c)
            # Trim to batch_size
            result = result[:batch_size]

        # --- Step 3: Tag specialist source in metadata ---
        # We store the source specialist as a simple attribute suffix
        # on the factor name if not already present.
        result = self._tag_specialist_source(result, proposals)

        logger.info(
            "Debate batch #%d complete: returning %d candidates",
            batch_id,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scores_to_candidates(
        scores: List[CriticScore],
        proposals: Dict[str, List[CandidateFactor]],
    ) -> List[CandidateFactor]:
        """Map CriticScore objects back to their CandidateFactor instances.

        Parameters
        ----------
        scores : list[CriticScore]
            Ranked scores from the critic.
        proposals : dict[str, list[CandidateFactor]]
            Original proposals for lookup.

        Returns
        -------
        list[CandidateFactor]
            Candidate factors in critic-ranked order.
        """
        # Build lookup: factor_name -> CandidateFactor
        lookup: Dict[str, CandidateFactor] = {}
        for candidates in proposals.values():
            for c in candidates:
                lookup[c.name] = c

        result: List[CandidateFactor] = []
        seen: set = set()
        for score in scores:
            candidate = lookup.get(score.factor_name)
            if candidate is not None and score.factor_name not in seen:
                result.append(candidate)
                seen.add(score.factor_name)

        return result

    @staticmethod
    def _tag_specialist_source(
        candidates: List[CandidateFactor],
        proposals: Dict[str, List[CandidateFactor]],
    ) -> List[CandidateFactor]:
        """Add specialist source information to each candidate's name.

        Builds a reverse index from factor name to specialist name and
        prefixes the candidate name for traceability, but only if it
        doesn't already carry a specialist tag.

        Parameters
        ----------
        candidates : list[CandidateFactor]
            The candidates to tag.
        proposals : dict[str, list[CandidateFactor]]
            Original proposals for specialist lookup.

        Returns
        -------
        list[CandidateFactor]
            Same list (mutated in-place for efficiency).
        """
        # Build reverse map: factor_name -> specialist_name
        source_map: Dict[str, str] = {}
        for spec_name, spec_candidates in proposals.items():
            for c in spec_candidates:
                source_map[c.name] = spec_name

        for c in candidates:
            spec_name = source_map.get(c.name, "unknown")
            # Store as category suffix rather than mutating the name,
            # so downstream pipeline still works with the original name.
            if not c.category.startswith("specialist:"):
                c.category = f"specialist:{spec_name}/{c.category}"

        return candidates
