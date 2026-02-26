"""The Helix Loop: 5-stage self-evolving factor discovery with Phase 2 extensions.

Extends the base Ralph Loop with:
  1. RETRIEVE  -- KG + embeddings + flat memory hybrid retrieval
  2. PROPOSE   -- Multi-agent debate (specialists + critic) or standard generation
  3. SYNTHESIZE -- SymPy canonicalization to eliminate mathematical duplicates
  4. VALIDATE  -- Standard pipeline + causal + regime + capacity + significance
  5. DISTILL   -- Standard memory evolution + KG update + online forgetting

All Phase 2 components are optional: when none are enabled the Helix Loop
behaves identically to the Ralph Loop and is a full drop-in replacement.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from factorminer.core.ralph_loop import (
    BudgetTracker,
    EvaluationResult,
    FactorGenerator,
    RalphLoop,
    ValidationPipeline,
)
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.parser import try_parse
from factorminer.evaluation.metrics import compute_ic
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.memory.retrieval import retrieve_memory
from factorminer.memory.formation import form_memory
from factorminer.memory.evolution import evolve_memory
from factorminer.agent.llm_interface import LLMProvider
from factorminer.utils.logging import IterationRecord, FactorRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional imports -- resolved at call time with graceful fallback
# ---------------------------------------------------------------------------

def _try_import_debate():
    try:
        from factorminer.agent.debate import DebateGenerator, DebateConfig
        return DebateGenerator, DebateConfig
    except ImportError:
        return None, None


def _try_import_canonicalizer():
    try:
        from factorminer.core.canonicalizer import FormulaCanonicalizer
        return FormulaCanonicalizer
    except ImportError:
        return None


def _try_import_causal():
    try:
        from factorminer.evaluation.causal import CausalValidator, CausalConfig
        return CausalValidator, CausalConfig
    except ImportError:
        return None, None


def _try_import_regime():
    try:
        from factorminer.evaluation.regime import (
            RegimeDetector,
            RegimeAwareEvaluator,
            RegimeConfig,
        )
        return RegimeDetector, RegimeAwareEvaluator, RegimeConfig
    except ImportError:
        return None, None, None


def _try_import_capacity():
    try:
        from factorminer.evaluation.capacity import CapacityEstimator, CapacityConfig
        return CapacityEstimator, CapacityConfig
    except ImportError:
        return None, None


def _try_import_significance():
    try:
        from factorminer.evaluation.significance import (
            BootstrapICTester,
            FDRController,
            DeflatedSharpeCalculator,
            SignificanceConfig,
        )
        return BootstrapICTester, FDRController, DeflatedSharpeCalculator, SignificanceConfig
    except ImportError:
        return None, None, None, None


def _try_import_kg():
    try:
        from factorminer.memory.knowledge_graph import FactorKnowledgeGraph, FactorNode
        return FactorKnowledgeGraph, FactorNode
    except ImportError:
        return None, None


def _try_import_kg_retrieval():
    try:
        from factorminer.memory.kg_retrieval import retrieve_memory_enhanced
        return retrieve_memory_enhanced
    except ImportError:
        return None


def _try_import_embedder():
    try:
        from factorminer.memory.embeddings import FormulaEmbedder
        return FormulaEmbedder
    except ImportError:
        return None


def _try_import_auto_inventor():
    try:
        from factorminer.operators.auto_inventor import OperatorInventor
        return OperatorInventor
    except ImportError:
        return None


def _try_import_custom_store():
    try:
        from factorminer.operators.custom import CustomOperatorStore
        return CustomOperatorStore
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# HelixLoop
# ---------------------------------------------------------------------------

class HelixLoop(RalphLoop):
    """Enhanced 5-stage Helix Loop for self-evolving factor discovery.

    Extends the Ralph Loop with:
    1. RETRIEVE: KG + embeddings + flat memory hybrid retrieval
    2. PROPOSE: Multi-agent debate (specialists + critic) or standard generation
    3. SYNTHESIZE: SymPy canonicalization to eliminate mathematical duplicates
    4. VALIDATE: Standard pipeline + causal + regime + capacity + significance
    5. DISTILL: Standard memory evolution + KG update + online forgetting

    All Phase 2 components are optional and default to off. When none are
    enabled, the Helix Loop behaves identically to the Ralph Loop.

    Parameters
    ----------
    config : Any
        Mining configuration object.
    data_tensor : np.ndarray
        Market data tensor D in R^(M x T x F).
    returns : np.ndarray
        Forward returns array R in R^(M x T).
    llm_provider : LLMProvider, optional
        LLM provider for factor generation.
    memory : ExperienceMemory, optional
        Pre-populated experience memory.
    library : FactorLibrary, optional
        Pre-populated factor library.
    debate_config : DebateConfig, optional
        Configuration for multi-agent debate generation.
        When provided, replaces standard FactorGenerator.
    enable_knowledge_graph : bool
        Enable factor knowledge graph for lineage and structural analysis.
    enable_embeddings : bool
        Enable semantic formula embeddings for similarity search.
    enable_auto_inventor : bool
        Enable periodic auto-invention of new operators.
    auto_invention_interval : int
        Run auto-invention every N iterations.
    canonicalize : bool
        Enable SymPy-based formula canonicalization for deduplication.
    forgetting_lambda : float
        Exponential decay factor for online forgetting (0-1).
    causal_config : CausalConfig, optional
        Configuration for causal validation (Granger + intervention).
    regime_config : RegimeConfig, optional
        Configuration for regime-aware IC evaluation.
    capacity_config : CapacityConfig, optional
        Configuration for capacity-aware cost evaluation.
    significance_config : SignificanceConfig, optional
        Configuration for bootstrap CI + FDR + deflated Sharpe.
    volume : np.ndarray, optional
        Dollar volume array (M, T) required for capacity estimation.
    """

    def __init__(
        self,
        config: Any,
        data_tensor: np.ndarray,
        returns: np.ndarray,
        llm_provider: Optional[LLMProvider] = None,
        memory: Optional[ExperienceMemory] = None,
        library: Optional[FactorLibrary] = None,
        # Phase 2 extensions
        debate_config: Optional[Any] = None,
        enable_knowledge_graph: bool = False,
        enable_embeddings: bool = False,
        enable_auto_inventor: bool = False,
        auto_invention_interval: int = 10,
        canonicalize: bool = True,
        forgetting_lambda: float = 0.95,
        causal_config: Optional[Any] = None,
        regime_config: Optional[Any] = None,
        capacity_config: Optional[Any] = None,
        significance_config: Optional[Any] = None,
        volume: Optional[np.ndarray] = None,
    ) -> None:
        # Initialize base RalphLoop
        super().__init__(
            config=config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=llm_provider,
            memory=memory,
            library=library,
        )

        # Store Phase 2 configuration
        self._debate_config = debate_config
        self._enable_kg = enable_knowledge_graph
        self._enable_embeddings = enable_embeddings
        self._enable_auto_inventor = enable_auto_inventor
        self._auto_invention_interval = auto_invention_interval
        self._canonicalize = canonicalize
        self._forgetting_lambda = forgetting_lambda
        self._causal_config = causal_config
        self._regime_config = regime_config
        self._capacity_config = capacity_config
        self._significance_config = significance_config
        self._volume = volume

        # Track iterations without admissions for forgetting
        self._no_admission_streak: int = 0

        # Initialize Phase 2 components
        self._debate_generator: Optional[Any] = None
        self._canonicalizer: Optional[Any] = None
        self._causal_validator: Optional[Any] = None
        self._regime_detector: Optional[Any] = None
        self._regime_evaluator: Optional[Any] = None
        self._regime_classification: Optional[Any] = None
        self._capacity_estimator: Optional[Any] = None
        self._bootstrap_tester: Optional[Any] = None
        self._fdr_controller: Optional[Any] = None
        self._kg: Optional[Any] = None
        self._embedder: Optional[Any] = None
        self._auto_inventor: Optional[Any] = None
        self._custom_op_store: Optional[Any] = None

        self._init_phase2_components(llm_provider)

    # ------------------------------------------------------------------
    # Phase 2 component initialization
    # ------------------------------------------------------------------

    def _init_phase2_components(self, llm_provider: Optional[LLMProvider]) -> None:
        """Initialize all Phase 2 components based on configuration."""

        # -- Debate generator --
        if self._debate_config is not None:
            DebateGeneratorCls, _ = _try_import_debate()
            if DebateGeneratorCls is not None:
                try:
                    self._debate_generator = DebateGeneratorCls(
                        llm_provider=llm_provider or self.generator.llm,
                        debate_config=self._debate_config,
                    )
                    logger.info("Helix: multi-agent debate generator enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init debate generator: %s", exc)
            else:
                logger.warning(
                    "Helix: debate_config provided but debate module unavailable"
                )

        # -- Canonicalizer --
        if self._canonicalize:
            FormulaCanonCls = _try_import_canonicalizer()
            if FormulaCanonCls is not None:
                try:
                    self._canonicalizer = FormulaCanonCls()
                    logger.info("Helix: SymPy canonicalization enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init canonicalizer: %s", exc)
            else:
                logger.warning(
                    "Helix: canonicalize=True but sympy/canonicalizer unavailable"
                )

        # -- Causal validator --
        if self._causal_config is not None:
            CausalValidatorCls, _ = _try_import_causal()
            if CausalValidatorCls is not None:
                logger.info("Helix: causal validation enabled")
            else:
                logger.warning(
                    "Helix: causal_config provided but causal module unavailable"
                )

        # -- Regime evaluator --
        if self._regime_config is not None:
            RegimeDetectorCls, RegimeEvalCls, _ = _try_import_regime()
            if RegimeDetectorCls is not None and RegimeEvalCls is not None:
                try:
                    self._regime_detector = RegimeDetectorCls(self._regime_config)
                    self._regime_classification = self._regime_detector.classify(
                        self.returns
                    )
                    self._regime_evaluator = RegimeEvalCls(
                        returns=self.returns,
                        regime=self._regime_classification,
                        config=self._regime_config,
                    )
                    logger.info("Helix: regime-aware evaluation enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init regime evaluator: %s", exc)
            else:
                logger.warning(
                    "Helix: regime_config provided but regime module unavailable"
                )

        # -- Capacity estimator --
        if self._capacity_config is not None:
            CapacityEstCls, _ = _try_import_capacity()
            if CapacityEstCls is not None:
                if self._volume is not None:
                    try:
                        self._capacity_estimator = CapacityEstCls(
                            returns=self.returns,
                            volume=self._volume,
                            config=self._capacity_config,
                        )
                        logger.info("Helix: capacity-aware evaluation enabled")
                    except Exception as exc:
                        logger.warning(
                            "Helix: failed to init capacity estimator: %s", exc
                        )
                else:
                    logger.warning(
                        "Helix: capacity_config provided but no volume data supplied"
                    )
            else:
                logger.warning(
                    "Helix: capacity_config provided but capacity module unavailable"
                )

        # -- Significance testing --
        if self._significance_config is not None:
            BootstrapCls, FDRCls, _, _ = _try_import_significance()
            if BootstrapCls is not None and FDRCls is not None:
                try:
                    self._bootstrap_tester = BootstrapCls(self._significance_config)
                    self._fdr_controller = FDRCls(self._significance_config)
                    logger.info("Helix: significance testing enabled")
                except Exception as exc:
                    logger.warning(
                        "Helix: failed to init significance testing: %s", exc
                    )
            else:
                logger.warning(
                    "Helix: significance_config provided but significance module unavailable"
                )

        # -- Knowledge graph --
        if self._enable_kg:
            KGCls, _ = _try_import_kg()
            if KGCls is not None:
                try:
                    self._kg = KGCls()
                    logger.info("Helix: knowledge graph enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init knowledge graph: %s", exc)
            else:
                logger.warning(
                    "Helix: enable_knowledge_graph=True but knowledge_graph module unavailable"
                )

        # -- Embedder --
        if self._enable_embeddings:
            EmbedderCls = _try_import_embedder()
            if EmbedderCls is not None:
                try:
                    self._embedder = EmbedderCls()
                    logger.info("Helix: formula embeddings enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init embedder: %s", exc)
            else:
                logger.warning(
                    "Helix: enable_embeddings=True but embeddings module unavailable"
                )

        # -- Auto inventor --
        if self._enable_auto_inventor:
            InventorCls = _try_import_auto_inventor()
            CustomStoreCls = _try_import_custom_store()
            if InventorCls is not None:
                try:
                    self._auto_inventor = InventorCls(
                        llm_provider=llm_provider or self.generator.llm,
                        data_tensor=self.data_tensor,
                        returns=self.returns,
                    )
                    logger.info("Helix: auto operator invention enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init auto inventor: %s", exc)

            if CustomStoreCls is not None:
                output_dir = getattr(self.config, "output_dir", "./output")
                try:
                    self._custom_op_store = CustomStoreCls(
                        store_dir=str(Path(output_dir) / "custom_operators")
                    )
                    logger.info("Helix: custom operator store enabled")
                except Exception as exc:
                    logger.warning(
                        "Helix: failed to init custom operator store: %s", exc
                    )
            else:
                logger.warning(
                    "Helix: enable_auto_inventor=True but custom operator store unavailable"
                )

    # ------------------------------------------------------------------
    # Override: _run_iteration with 5-stage Helix flow
    # ------------------------------------------------------------------

    def _run_iteration(self, batch_size: int) -> Dict[str, Any]:
        """Execute one iteration of the 5-stage Helix Loop.

        Stages:
          1. RETRIEVE  -- enhanced memory retrieval (KG + embeddings + flat)
          2. PROPOSE   -- debate or standard factor generation
          3. SYNTHESIZE -- canonicalize and deduplicate candidates
          4. VALIDATE  -- standard pipeline + causal + regime + capacity + significance
          5. DISTILL   -- memory evolution + KG update + forgetting

        Returns
        -------
        dict
            Iteration statistics.
        """
        t0 = time.time()
        helix_stats: Dict[str, Any] = {}

        # ==================================================================
        # Stage 1: RETRIEVE
        # ==================================================================
        library_state = self.library.get_state_summary()
        memory_signal = self._helix_retrieve(library_state)

        # ==================================================================
        # Stage 2: PROPOSE
        # ==================================================================
        t_gen = time.time()
        candidates = self._helix_propose(memory_signal, library_state, batch_size)
        self.budget.record_llm_call()

        if not candidates:
            logger.warning(
                "Helix iteration %d: generator produced 0 candidates",
                self.iteration,
            )
            return self._empty_stats()

        helix_stats["candidates_before_canon"] = len(candidates)

        # ==================================================================
        # Stage 3: SYNTHESIZE (canonicalize + dedup)
        # ==================================================================
        candidates, n_canon_dupes = self._canonicalize_and_dedup(candidates)
        helix_stats["canonical_duplicates_removed"] = n_canon_dupes

        if not candidates:
            logger.warning(
                "Helix iteration %d: all candidates removed by canonicalization",
                self.iteration,
            )
            return self._empty_stats()

        # ==================================================================
        # Stage 4: VALIDATE
        # ==================================================================
        results = self.pipeline.evaluate_batch(candidates)
        admitted_results = self._update_library(results)

        # Phase 2 extended validation on admitted candidates
        rejected_by_phase2 = self._helix_validate(results, admitted_results)
        helix_stats["phase2_rejections"] = rejected_by_phase2

        # ==================================================================
        # Stage 5: DISTILL
        # ==================================================================
        trajectory = self._build_trajectory(results)
        formed = form_memory(self.memory, trajectory, self.iteration)
        self.memory = evolve_memory(self.memory, formed)

        # KG + embeddings + forgetting
        self._helix_distill(results, admitted_results)

        # Auto-invention check
        if (
            self._auto_inventor is not None
            and self.iteration % self._auto_invention_interval == 0
        ):
            self._run_auto_invention()

        # Build stats
        elapsed = time.time() - t0
        self.budget.record_compute(elapsed)
        stats = self._compute_stats(results, admitted_results, elapsed)
        stats.update(helix_stats)
        stats["iteration"] = self.iteration

        # Log to reporter and session logger
        self.reporter.log_batch(**stats)
        if self._session_logger:
            ic_values = [r.ic_mean for r in results if r.parse_ok]
            record = IterationRecord(
                iteration=self.iteration,
                candidates_generated=len(candidates) + n_canon_dupes,
                ic_passed=stats["ic_passed"],
                correlation_passed=stats["corr_passed"],
                admitted=stats["admitted"],
                rejected=len(candidates) + n_canon_dupes - stats["admitted"],
                replaced=stats["replaced"],
                library_size=self.library.size,
                best_ic=max(ic_values) if ic_values else 0.0,
                mean_ic=float(np.mean(ic_values)) if ic_values else 0.0,
                elapsed_seconds=elapsed,
            )
            self._session_logger.log_iteration(record)

            for r in results:
                factor_rec = FactorRecord(
                    expression=r.formula,
                    ic=r.ic_mean if r.parse_ok else None,
                    icir=r.icir if r.parse_ok else None,
                    max_correlation=r.max_correlation if r.parse_ok else None,
                    admitted=r.admitted,
                    rejection_reason=r.rejection_reason or None,
                    replaced_factor=str(r.replaced) if r.replaced else None,
                )
                self._session_logger.log_factor(factor_rec)

        return stats

    # ------------------------------------------------------------------
    # Stage 1: Enhanced retrieval
    # ------------------------------------------------------------------

    def _helix_retrieve(
        self, library_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 1 RETRIEVE: KG + embeddings + flat memory hybrid retrieval.

        Falls back to standard retrieve_memory if no KG/embedder is available.
        """
        retrieve_enhanced_fn = _try_import_kg_retrieval()

        if retrieve_enhanced_fn is not None and (
            self._kg is not None or self._embedder is not None
        ):
            try:
                return retrieve_enhanced_fn(
                    memory=self.memory,
                    library_state=library_state,
                    kg=self._kg,
                    embedder=self._embedder,
                )
            except Exception as exc:
                logger.warning(
                    "Helix: enhanced retrieval failed, falling back: %s", exc
                )

        return retrieve_memory(self.memory, library_state=library_state)

    # ------------------------------------------------------------------
    # Stage 2: Debate or standard proposal
    # ------------------------------------------------------------------

    def _helix_propose(
        self,
        memory_signal: Dict[str, Any],
        library_state: Dict[str, Any],
        batch_size: int,
    ) -> List[Tuple[str, str]]:
        """Stage 2 PROPOSE: Use debate generator or standard generator.

        Returns list of (name, formula) tuples compatible with the
        validation pipeline.
        """
        if self._debate_generator is not None:
            try:
                debate_candidates = self._debate_generator.generate_batch(
                    memory_signal=memory_signal,
                    library_state=library_state,
                    batch_size=batch_size,
                )
                # Convert CandidateFactor objects to (name, formula) tuples
                tuples: List[Tuple[str, str]] = []
                for c in debate_candidates:
                    tuples.append((c.name, c.formula))
                if tuples:
                    logger.info(
                        "Helix: debate generator produced %d candidates",
                        len(tuples),
                    )
                    return tuples
                logger.warning(
                    "Helix: debate generator returned 0 candidates, "
                    "falling back to standard generator"
                )
            except Exception as exc:
                logger.warning(
                    "Helix: debate generation failed, falling back: %s", exc
                )

        # Standard generation
        return self.generator.generate_batch(
            memory_signal=memory_signal,
            library_state=library_state,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Stage 3: Canonicalization + deduplication
    # ------------------------------------------------------------------

    def _canonicalize_and_dedup(
        self, candidates: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], int]:
        """Stage 3 SYNTHESIZE: Remove mathematically equivalent candidates.

        Uses SymPy-based canonicalization to detect algebraic duplicates
        before evaluation, saving compute.

        Returns
        -------
        tuple of (deduplicated_candidates, n_duplicates_removed)
        """
        if self._canonicalizer is None:
            return candidates, 0

        seen_hashes: Dict[str, str] = {}  # hash -> first factor name
        unique: List[Tuple[str, str]] = []
        n_dupes = 0

        for name, formula in candidates:
            tree = try_parse(formula)
            if tree is None:
                # Keep unparseable candidates; the pipeline will reject them
                unique.append((name, formula))
                continue

            try:
                canon_hash = self._canonicalizer.canonicalize(tree)
            except Exception as exc:
                logger.debug(
                    "Helix: canonicalization failed for '%s': %s", name, exc
                )
                unique.append((name, formula))
                continue

            if canon_hash in seen_hashes:
                n_dupes += 1
                logger.debug(
                    "Helix: canonical duplicate '%s' matches '%s'",
                    name,
                    seen_hashes[canon_hash],
                )
            else:
                seen_hashes[canon_hash] = name
                unique.append((name, formula))

        if n_dupes > 0:
            logger.info(
                "Helix: canonicalization removed %d/%d duplicate candidates",
                n_dupes,
                len(candidates),
            )

        return unique, n_dupes

    # ------------------------------------------------------------------
    # Stage 4: Extended validation
    # ------------------------------------------------------------------

    def _helix_validate(
        self,
        results: List[EvaluationResult],
        admitted_results: List[EvaluationResult],
    ) -> int:
        """Stage 4 extended VALIDATE: causal + regime + capacity + significance.

        Runs Phase 2 validation on admitted candidates and revokes admission
        for those that fail. Returns the number of Phase 2 rejections.
        """
        if not admitted_results:
            self._no_admission_streak += 1
            return 0

        self._no_admission_streak = 0
        rejected = 0

        # Collect admitted results that still have signals for extended checks
        to_check = [r for r in admitted_results if r.signals is not None]
        if not to_check:
            return 0

        # -- Causal validation --
        if self._causal_config is not None:
            rejected += self._validate_causal(to_check, results)

        # -- Regime validation --
        if self._regime_evaluator is not None:
            rejected += self._validate_regime(to_check, results)

        # -- Capacity validation --
        if self._capacity_estimator is not None:
            rejected += self._validate_capacity(to_check, results)

        # -- Significance testing (batch-level FDR) --
        if self._bootstrap_tester is not None and self._fdr_controller is not None:
            rejected += self._validate_significance(to_check, results)

        if rejected > 0:
            logger.info(
                "Helix: Phase 2 validation rejected %d/%d admitted candidates",
                rejected,
                len(admitted_results),
            )

        return rejected

    def _validate_causal(
        self,
        to_check: List[EvaluationResult],
        all_results: List[EvaluationResult],
    ) -> int:
        """Run causal validation (Granger + intervention) on admitted candidates."""
        CausalValidatorCls, _ = _try_import_causal()
        if CausalValidatorCls is None:
            return 0

        # Collect library signals for controls
        library_signals: Dict[str, np.ndarray] = {}
        for f in self.library.list_factors():
            if f.signals is not None:
                library_signals[f.name] = f.signals

        try:
            validator = CausalValidatorCls(
                returns=self.returns,
                data_tensor=self.data_tensor,
                library_signals=library_signals,
                config=self._causal_config,
            )
        except Exception as exc:
            logger.warning("Helix: causal validator creation failed: %s", exc)
            return 0

        rejected = 0
        threshold = getattr(
            self._causal_config, "robustness_threshold", 0.4
        )

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = validator.validate(r.factor_name, r.signals)
                if not result.passes:
                    self._revoke_admission(r, all_results,
                        f"Causal: robustness_score={result.robustness_score:.3f} < {threshold}"
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: causal rejection for '%s' (score=%.3f)",
                        r.factor_name,
                        result.robustness_score,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: causal validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_regime(
        self,
        to_check: List[EvaluationResult],
        all_results: List[EvaluationResult],
    ) -> int:
        """Run regime-aware IC evaluation on admitted candidates."""
        if self._regime_evaluator is None:
            return 0

        rejected = 0
        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = self._regime_evaluator.evaluate(r.factor_name, r.signals)
                if not result.passes:
                    self._revoke_admission(r, all_results,
                        f"Regime: only {result.n_regimes_passing} regimes passing "
                        f"(need {getattr(self._regime_config, 'min_regimes_passing', 2)})"
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: regime rejection for '%s' (%d regimes passing)",
                        r.factor_name,
                        result.n_regimes_passing,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: regime validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_capacity(
        self,
        to_check: List[EvaluationResult],
        all_results: List[EvaluationResult],
    ) -> int:
        """Run capacity-aware cost evaluation on admitted candidates."""
        if self._capacity_estimator is None:
            return 0

        rejected = 0
        net_icir_threshold = getattr(
            self._capacity_config, "net_icir_threshold", 0.3
        )

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = self._capacity_estimator.net_cost_evaluation(
                    factor_name=r.factor_name,
                    signals=r.signals,
                )
                if not result.passes_net_threshold:
                    self._revoke_admission(r, all_results,
                        f"Capacity: net_icir={result.net_icir:.3f} < {net_icir_threshold}"
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: capacity rejection for '%s' (net_icir=%.3f)",
                        r.factor_name,
                        result.net_icir,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: capacity validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_significance(
        self,
        to_check: List[EvaluationResult],
        all_results: List[EvaluationResult],
    ) -> int:
        """Run bootstrap CI + batch-level FDR correction on admitted candidates."""
        if self._bootstrap_tester is None or self._fdr_controller is None:
            return 0

        # Compute IC series for each admitted candidate and gather p-values
        ic_series_map: Dict[str, np.ndarray] = {}
        result_map: Dict[str, EvaluationResult] = {}

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                ic_series = compute_ic(r.signals, self.returns)
                ic_series_map[r.factor_name] = ic_series
                result_map[r.factor_name] = r
            except Exception as exc:
                logger.warning(
                    "Helix: IC computation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        if not ic_series_map:
            return 0

        try:
            fdr_result = self._fdr_controller.batch_evaluate(
                ic_series_map, self._bootstrap_tester
            )
        except Exception as exc:
            logger.warning("Helix: FDR batch evaluation failed: %s", exc)
            return 0

        rejected = 0
        for name, is_sig in fdr_result.significant.items():
            if not is_sig:
                r = result_map.get(name)
                if r is not None and r.admitted:
                    adj_p = fdr_result.adjusted_p_values.get(name, 1.0)
                    self._revoke_admission(r, all_results,
                        f"Significance: FDR-adjusted p={adj_p:.4f} > "
                        f"{getattr(self._significance_config, 'fdr_level', 0.05)}"
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: significance rejection for '%s' (adj_p=%.4f)",
                        name,
                        adj_p,
                    )

        return rejected

    def _revoke_admission(
        self,
        result: EvaluationResult,
        all_results: List[EvaluationResult],
        reason: str,
    ) -> None:
        """Revoke a previously admitted candidate from the library.

        Updates the EvaluationResult and removes the factor from the library.
        """
        result.admitted = False
        result.rejection_reason = reason

        # Find and remove from library by name+formula match
        try:
            for factor in self.library.list_factors():
                if (
                    factor.name == result.factor_name
                    and factor.formula == result.formula
                ):
                    del self.library.factors[factor.id]
                    logger.debug(
                        "Helix: revoked factor '%s' (id=%d): %s",
                        result.factor_name,
                        factor.id,
                        reason,
                    )
                    return
        except Exception as exc:
            logger.warning(
                "Helix: failed to revoke factor '%s': %s",
                result.factor_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Stage 5: Enhanced distillation
    # ------------------------------------------------------------------

    def _helix_distill(
        self,
        results: List[EvaluationResult],
        admitted_results: List[EvaluationResult],
    ) -> None:
        """Stage 5 DISTILL: KG update + embeddings + online forgetting."""

        # -- Knowledge graph updates --
        if self._kg is not None:
            self._update_knowledge_graph(results, admitted_results)

        # -- Embed newly admitted factors --
        if self._embedder is not None:
            for r in admitted_results:
                if r.admitted:
                    try:
                        self._embedder.embed(r.factor_name, r.formula)
                    except Exception as exc:
                        logger.debug(
                            "Helix: embedding failed for '%s': %s",
                            r.factor_name,
                            exc,
                        )

        # -- Online forgetting --
        self._apply_forgetting()

    def _update_knowledge_graph(
        self,
        results: List[EvaluationResult],
        admitted_results: List[EvaluationResult],
    ) -> None:
        """Update the knowledge graph with new factor nodes and edges."""
        _, FactorNodeCls = _try_import_kg()
        if FactorNodeCls is None or self._kg is None:
            return

        for r in admitted_results:
            if not r.admitted:
                continue

            # Extract operators and features from formula
            operators = self._extract_operators(r.formula)
            features = self._extract_features(r.formula)

            node = FactorNodeCls(
                factor_id=r.factor_name,
                formula=r.formula,
                ic_mean=r.ic_mean,
                category=self._infer_category(r.formula),
                operators=operators,
                features=features,
                batch_number=self.iteration,
                admitted=True,
            )

            try:
                self._kg.add_factor(node)
            except Exception as exc:
                logger.debug(
                    "Helix: failed to add factor to KG: %s", exc
                )
                continue

            # Add correlation edges with existing library factors
            if r.signals is not None:
                for factor in self.library.list_factors():
                    if factor.name == r.factor_name:
                        continue
                    if factor.signals is not None:
                        try:
                            corr = self.library._compute_correlation_vectorized(
                                r.signals, factor.signals
                            )
                            self._kg.add_correlation_edge(
                                r.factor_name,
                                factor.name,
                                rho=corr,
                                threshold=0.4,
                            )
                        except Exception:
                            pass

            # Detect derivation (mutation) relationships
            self._detect_derivation(r, operators)

    def _detect_derivation(
        self,
        result: EvaluationResult,
        new_operators: List[str],
    ) -> None:
        """Detect if a new factor is a mutation of an existing one.

        Compares operator sets: if the new factor shares >50% of operators
        with an existing factor but has at least one different operator,
        it is considered a derivation (mutation).
        """
        if self._kg is None:
            return

        new_ops = set(new_operators)
        if not new_ops:
            return

        for factor in self.library.list_factors():
            if factor.name == result.factor_name:
                continue

            existing_ops = set(self._extract_operators(factor.formula))
            if not existing_ops:
                continue

            shared = new_ops & existing_ops
            if not shared:
                continue

            # More than 50% shared but not identical
            overlap = len(shared) / max(len(new_ops), len(existing_ops))
            if 0.5 <= overlap < 1.0:
                diff_ops = (new_ops - existing_ops) | (existing_ops - new_ops)
                mutation_type = f"operator_change:{','.join(sorted(diff_ops))}"
                try:
                    self._kg.add_derivation_edge(
                        child=result.factor_name,
                        parent=factor.name,
                        mutation_type=mutation_type,
                    )
                except Exception:
                    pass

    def _apply_forgetting(self) -> None:
        """Apply online forgetting: exponential decay on memory patterns.

        - Decay occurrence_count of all success patterns by forgetting_lambda.
        - If no admissions for 20+ consecutive iterations, demote success_rate.
        """
        lam = self._forgetting_lambda

        for pattern in self.memory.success_patterns:
            # Decay occurrence count
            if hasattr(pattern, "occurrence_count"):
                pattern.occurrence_count = int(
                    pattern.occurrence_count * lam
                )

        # Demote success_rate after prolonged drought
        if self._no_admission_streak >= 20:
            for pattern in self.memory.success_patterns:
                if hasattr(pattern, "success_rate"):
                    current = pattern.success_rate
                    if current == "High":
                        pattern.success_rate = "Medium"
                    elif current == "Medium":
                        pattern.success_rate = "Low"
            logger.info(
                "Helix: demoted success rates after %d iterations without admissions",
                self._no_admission_streak,
            )

    # ------------------------------------------------------------------
    # Auto-invention
    # ------------------------------------------------------------------

    def _run_auto_invention(self) -> None:
        """Periodically propose, validate, and register new operators.

        Uses the OperatorInventor to generate novel operators from
        successful pattern context, then validates and registers them
        via CustomOperatorStore.
        """
        if self._auto_inventor is None:
            return

        logger.info("Helix: running auto-invention at iteration %d", self.iteration)

        # Gather existing operators
        try:
            from factorminer.core.types import OPERATOR_REGISTRY as SPEC_REG
            existing_ops = dict(SPEC_REG)
        except ImportError:
            existing_ops = {}

        # Gather successful pattern descriptions
        patterns = []
        for pat in self.memory.success_patterns[:10]:
            patterns.append(f"{pat.name}: {pat.description}")

        try:
            proposals = self._auto_inventor.propose_operators(
                existing_operators=existing_ops,
                successful_patterns=patterns,
            )
        except Exception as exc:
            logger.warning("Helix: auto-invention proposal failed: %s", exc)
            return

        self.budget.record_llm_call()

        validated = 0
        for proposal in proposals:
            try:
                val_result = self._auto_inventor.validate_operator(proposal)
                if val_result.valid:
                    self._register_invented_operator(proposal, val_result)
                    validated += 1
                else:
                    logger.debug(
                        "Helix: operator '%s' failed validation: %s",
                        proposal.name,
                        val_result.error,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: operator validation error for '%s': %s",
                    proposal.name,
                    exc,
                )

        logger.info(
            "Helix: auto-invention: %d/%d proposals validated and registered",
            validated,
            len(proposals),
        )

    def _register_invented_operator(
        self,
        proposal: Any,
        val_result: Any,
    ) -> None:
        """Register a validated auto-invented operator."""
        if self._custom_op_store is None:
            logger.warning(
                "Helix: no custom operator store; cannot register '%s'",
                proposal.name,
            )
            return

        try:
            from factorminer.operators.custom import CustomOperator
            from factorminer.core.types import OperatorSpec, OperatorType, SignatureType

            spec = OperatorSpec(
                name=proposal.name,
                arity=proposal.arity,
                category=OperatorType.AUTO_INVENTED,
                signature=SignatureType.TIME_SERIES_TO_TIME_SERIES,
                param_names=proposal.param_names,
                param_defaults=proposal.param_defaults,
                param_ranges={
                    k: tuple(v) for k, v in proposal.param_ranges.items()
                },
                description=proposal.description,
            )

            # Compile the function
            from factorminer.operators.custom import _compile_operator_code
            fn = _compile_operator_code(proposal.numpy_code)
            if fn is None:
                logger.warning(
                    "Helix: failed to compile invented operator '%s'",
                    proposal.name,
                )
                return

            custom_op = CustomOperator(
                name=proposal.name,
                spec=spec,
                numpy_code=proposal.numpy_code,
                numpy_fn=fn,
                validation_ic=val_result.ic_contribution,
                invention_iteration=self.iteration,
                rationale=proposal.rationale,
            )

            self._custom_op_store.register(custom_op)
            logger.info(
                "Helix: registered auto-invented operator '%s' (IC=%.4f)",
                proposal.name,
                val_result.ic_contribution,
            )
        except Exception as exc:
            logger.warning(
                "Helix: failed to register operator '%s': %s",
                proposal.name,
                exc,
            )

    # ------------------------------------------------------------------
    # Enhanced checkpointing
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Save a periodic checkpoint including Phase 2 state."""
        try:
            self.save_session()
        except Exception as exc:
            logger.warning("Helix: checkpoint failed: %s", exc)

    def save_session(self, path: Optional[str] = None) -> str:
        """Save the full mining session state including Phase 2 components.

        Extends the base RalphLoop save with:
        - Knowledge graph serialization
        - Custom operator store persistence

        Parameters
        ----------
        path : str, optional
            Directory for the checkpoint.

        Returns
        -------
        str
            Path to the saved session directory.
        """
        # Base save
        checkpoint_path = super().save_session(path)
        checkpoint_dir = Path(checkpoint_path)

        # Save knowledge graph
        if self._kg is not None:
            try:
                kg_path = checkpoint_dir / "knowledge_graph.json"
                self._kg.save(kg_path)
                logger.debug("Helix: saved knowledge graph to %s", kg_path)
            except Exception as exc:
                logger.warning("Helix: failed to save knowledge graph: %s", exc)

        # Save custom operators
        if self._custom_op_store is not None:
            try:
                self._custom_op_store.save()
                logger.debug("Helix: saved custom operators")
            except Exception as exc:
                logger.warning("Helix: failed to save custom operators: %s", exc)

        # Save helix-specific state
        helix_state = {
            "no_admission_streak": self._no_admission_streak,
            "forgetting_lambda": self._forgetting_lambda,
            "canonicalize": self._canonicalize,
            "enable_knowledge_graph": self._enable_kg,
            "enable_embeddings": self._enable_embeddings,
            "enable_auto_inventor": self._enable_auto_inventor,
        }
        try:
            with open(checkpoint_dir / "helix_state.json", "w") as f:
                json.dump(helix_state, f, indent=2)
        except Exception as exc:
            logger.warning("Helix: failed to save helix state: %s", exc)

        return checkpoint_path

    def load_session(self, path: str) -> None:
        """Resume a mining session from a saved checkpoint.

        Extends the base RalphLoop load with Phase 2 state restoration.

        Parameters
        ----------
        path : str
            Path to the checkpoint directory.
        """
        super().load_session(path)
        checkpoint_dir = Path(path)

        # Load knowledge graph
        if self._kg is not None:
            kg_path = checkpoint_dir / "knowledge_graph.json"
            if kg_path.exists():
                KGCls, _ = _try_import_kg()
                if KGCls is not None:
                    try:
                        self._kg = KGCls.load(kg_path)
                        logger.info(
                            "Helix: loaded knowledge graph (%d factors, %d edges)",
                            self._kg.get_factor_count(),
                            self._kg.get_edge_count(),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Helix: failed to load knowledge graph: %s", exc
                        )

        # Load custom operators
        if self._custom_op_store is not None:
            try:
                self._custom_op_store.load()
            except Exception as exc:
                logger.warning(
                    "Helix: failed to load custom operators: %s", exc
                )

        # Load helix-specific state
        helix_state_path = checkpoint_dir / "helix_state.json"
        if helix_state_path.exists():
            try:
                with open(helix_state_path) as f:
                    helix_state = json.load(f)
                self._no_admission_streak = helix_state.get(
                    "no_admission_streak", 0
                )
                logger.info(
                    "Helix: restored helix state (streak=%d)",
                    self._no_admission_streak,
                )
            except Exception as exc:
                logger.warning(
                    "Helix: failed to load helix state: %s", exc
                )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_operators(formula: str) -> List[str]:
        """Extract operator names from a DSL formula string."""
        return re.findall(r"([A-Z][a-zA-Z]+)\(", formula)

    @staticmethod
    def _extract_features(formula: str) -> List[str]:
        """Extract feature names (e.g. $close, $volume) from a formula."""
        return re.findall(r"\$[a-zA-Z_]+", formula)
