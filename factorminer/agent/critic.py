"""Critic agent that reviews and scores candidate factors from multiple specialists.

The ``CriticAgent`` takes proposals from all specialists, sends a single
consolidated review prompt to the LLM, and returns ranked ``CriticScore``
objects.  This enables efficient quality gating with a single API call.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from factorminer.agent.llm_interface import LLMProvider
from factorminer.agent.output_parser import CandidateFactor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CriticScore dataclass
# ---------------------------------------------------------------------------

@dataclass
class CriticScore:
    """Scored review of a single candidate factor.

    Attributes
    ----------
    factor_name : str
        Name of the candidate factor being reviewed.
    formula : str
        DSL formula string of the candidate.
    source_specialist : str
        Name of the specialist that proposed this candidate.
    novelty_score : float
        Novelty rating in [0, 1] -- how structurally distinct is this
        factor from existing library members?
    quality_score : float
        Quality rating in [0, 1] -- how likely is this factor to be
        economically meaningful and predictive?
    diversity_bonus : float
        Bonus for contributing to domain diversity across the batch.
    critic_rationale : str
        Free-text rationale from the critic explaining the scores.
    final_score : float
        Weighted aggregate: 0.4 * novelty + 0.4 * quality + 0.2 * diversity.
    """

    factor_name: str
    formula: str
    source_specialist: str
    novelty_score: float
    quality_score: float
    diversity_bonus: float
    critic_rationale: str
    final_score: float


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class CriticAgent:
    """LLM-powered critic that ranks candidate factors from multiple specialists.

    Uses a single LLM call to score all proposals on novelty, quality,
    and diversity.  Falls back to uniform scores when LLM parsing fails.

    Parameters
    ----------
    llm_provider : LLMProvider
        LLM backend for the critic review call.
    temperature : float
        Sampling temperature for the review (low for consistency).
    max_tokens : int
        Max response tokens for the review.
    """

    _SYSTEM_PROMPT = (
        "You are a quantitative research critic.  Your job is to evaluate "
        "candidate alpha factors proposed by specialist agents and score "
        "each one on novelty and quality.  Be rigorous but fair."
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    def review_candidates(
        self,
        proposals: Dict[str, List[CandidateFactor]],
        library_state: Dict[str, Any],
        memory_signal: Dict[str, Any],
        top_k: int = 40,
    ) -> List[CriticScore]:
        """Review all specialist proposals and return ranked scores.

        Parameters
        ----------
        proposals : dict[str, list[CandidateFactor]]
            Mapping from specialist name to its list of candidates.
        library_state : dict
            Current factor library state for context.
        memory_signal : dict
            Memory priors (unused directly but available for context).
        top_k : int
            Number of top-scoring candidates to return.

        Returns
        -------
        list[CriticScore]
            Top-K candidates sorted by ``final_score`` descending.
        """
        # Flatten all candidates for counting
        all_candidates: List[tuple[str, CandidateFactor]] = []
        for specialist_name, candidates in proposals.items():
            for c in candidates:
                all_candidates.append((specialist_name, c))

        if not all_candidates:
            return []

        # Build and send review prompt
        review_prompt = self._build_review_prompt(proposals, library_state)

        try:
            raw_response = self.llm_provider.generate(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=review_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            scores = self._parse_review_response(raw_response, proposals)
        except Exception as e:
            logger.warning(
                "Critic LLM call or parsing failed (%s); falling back to "
                "uniform scores.",
                e,
            )
            scores = self._fallback_uniform_scores(proposals)

        # If parsing produced no scores, use fallback
        if not scores:
            logger.warning(
                "Critic produced no parsed scores; falling back to uniform."
            )
            scores = self._fallback_uniform_scores(proposals)

        # Sort descending and return top-K
        scores.sort(key=lambda s: s.final_score, reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_review_prompt(
        self,
        proposals: Dict[str, List[CandidateFactor]],
        library_state: Dict[str, Any],
    ) -> str:
        """Build the review prompt listing all candidates grouped by specialist.

        Parameters
        ----------
        proposals : dict[str, list[CandidateFactor]]
            Specialist name -> candidate list.
        library_state : dict
            Library context (size, recent admissions, saturation).

        Returns
        -------
        str
            Fully assembled review prompt.
        """
        sections: List[str] = []

        # Library context
        lib_size = library_state.get("size", 0)
        target = library_state.get("target_size", 110)
        sections.append(
            f"Current library: {lib_size}/{target} factors."
        )

        recent = library_state.get("recent_admissions", [])
        if recent:
            sections.append(
                "Recently admitted:\n"
                + "\n".join(f"  - {f}" for f in recent[-5:])
            )

        # List all candidates grouped by specialist
        sections.append("\n## CANDIDATES TO REVIEW\n")
        idx = 1
        for specialist_name, candidates in proposals.items():
            sections.append(f"### From {specialist_name} specialist:")
            for c in candidates:
                sections.append(f"  {idx}. {c.name}: {c.formula}")
                idx += 1
            sections.append("")

        # Scoring instructions
        sections.append(
            "## SCORING INSTRUCTIONS\n"
            "For each candidate, provide a JSON object on its own line:\n"
            '{"name": "<factor_name>", "novelty": <0.0-1.0>, '
            '"quality": <0.0-1.0>, "rationale": "<brief reason>"}\n\n'
            "Criteria:\n"
            "- novelty: structural distinctiveness from existing library "
            "members and other candidates in this batch.\n"
            "- quality: economic meaningfulness, appropriate complexity "
            "(depth 3-7), and likelihood of being predictive.\n\n"
            "Output ONLY the JSON objects, one per line.  No other text."
        )

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_review_response(
        self,
        raw: str,
        proposals: Dict[str, List[CandidateFactor]],
    ) -> List[CriticScore]:
        """Parse the LLM review response into CriticScore objects.

        Robust to various formatting issues.  Falls back to uniform
        scores for any candidates that cannot be matched.

        Parameters
        ----------
        raw : str
            Raw LLM text containing JSON score objects.
        proposals : dict[str, list[CandidateFactor]]
            Original proposals for reverse-lookup of formulas.

        Returns
        -------
        list[CriticScore]
            Parsed scores (may be empty if parsing totally fails).
        """
        # Build lookup: name -> (specialist, candidate)
        lookup: Dict[str, tuple[str, CandidateFactor]] = {}
        for specialist_name, candidates in proposals.items():
            for c in candidates:
                lookup[c.name] = (specialist_name, c)

        # Track which specialists have been seen for diversity scoring
        specialist_counts: Dict[str, int] = {}
        scores: List[CriticScore] = []
        seen_names: set = set()

        # Try to extract JSON objects from the response
        json_pattern = re.compile(r'\{[^{}]+\}')
        matches = json_pattern.findall(raw)

        for match_str in matches:
            try:
                obj = json.loads(match_str)
            except json.JSONDecodeError:
                continue

            name = obj.get("name", "")
            if name not in lookup or name in seen_names:
                continue

            specialist_name, candidate = lookup[name]
            novelty = float(max(0.0, min(1.0, obj.get("novelty", 0.5))))
            quality = float(max(0.0, min(1.0, obj.get("quality", 0.5))))
            rationale = str(obj.get("rationale", ""))

            # Diversity bonus: reward specialists that are underrepresented
            specialist_counts[specialist_name] = (
                specialist_counts.get(specialist_name, 0) + 1
            )
            total_so_far = sum(specialist_counts.values())
            n_specialists = len(proposals)
            ideal_share = 1.0 / max(n_specialists, 1)
            actual_share = specialist_counts[specialist_name] / max(total_so_far, 1)
            diversity_bonus = max(0.0, ideal_share - actual_share + 0.5)
            diversity_bonus = min(1.0, diversity_bonus)

            final_score = (
                0.4 * novelty + 0.4 * quality + 0.2 * diversity_bonus
            )

            scores.append(CriticScore(
                factor_name=name,
                formula=candidate.formula,
                source_specialist=specialist_name,
                novelty_score=novelty,
                quality_score=quality,
                diversity_bonus=diversity_bonus,
                critic_rationale=rationale,
                final_score=final_score,
            ))
            seen_names.add(name)

        # Any candidates not scored get appended with fallback scores
        for name, (specialist_name, candidate) in lookup.items():
            if name not in seen_names:
                scores.append(CriticScore(
                    factor_name=name,
                    formula=candidate.formula,
                    source_specialist=specialist_name,
                    novelty_score=0.5,
                    quality_score=0.5,
                    diversity_bonus=0.5,
                    critic_rationale="Not scored by critic; using default.",
                    final_score=0.4 * 0.5 + 0.4 * 0.5 + 0.2 * 0.5,
                ))

        return scores

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_uniform_scores(
        proposals: Dict[str, List[CandidateFactor]],
    ) -> List[CriticScore]:
        """Generate uniform scores when LLM review fails.

        Every candidate gets equal scores (0.5) so that the downstream
        pipeline still works.

        Parameters
        ----------
        proposals : dict[str, list[CandidateFactor]]
            Specialist name -> candidate list.

        Returns
        -------
        list[CriticScore]
            All candidates with uniform 0.5 scores.
        """
        default_score = 0.4 * 0.5 + 0.4 * 0.5 + 0.2 * 0.5
        scores: List[CriticScore] = []
        for specialist_name, candidates in proposals.items():
            for c in candidates:
                scores.append(CriticScore(
                    factor_name=c.name,
                    formula=c.formula,
                    source_specialist=specialist_name,
                    novelty_score=0.5,
                    quality_score=0.5,
                    diversity_bonus=0.5,
                    critic_rationale="Fallback uniform score (critic unavailable).",
                    final_score=default_score,
                ))
        return scores
