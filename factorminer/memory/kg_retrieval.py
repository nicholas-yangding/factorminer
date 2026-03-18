"""Enhanced memory retrieval combining Knowledge Graph + Embeddings + flat memory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from factorminer.memory.memory_store import ExperienceMemory
from factorminer.memory.retrieval import retrieve_memory

# Optional imports -- presence checked at call time
try:
    from factorminer.memory.knowledge_graph import FactorKnowledgeGraph
except ImportError:
    FactorKnowledgeGraph = None  # type: ignore[assignment,misc]

try:
    from factorminer.memory.embeddings import FormulaEmbedder
except ImportError:
    FormulaEmbedder = None  # type: ignore[assignment,misc]


def retrieve_memory_enhanced(
    memory: ExperienceMemory,
    library_state: Optional[Dict[str, Any]] = None,
    max_success: int = 8,
    max_forbidden: int = 10,
    max_insights: int = 10,
    kg: Optional[FactorKnowledgeGraph] = None,  # type: ignore[type-arg]
    embedder: Optional[FormulaEmbedder] = None,  # type: ignore[type-arg]
) -> Dict[str, Any]:
    """Enhanced memory retrieval operator R+(M, L, KG, E).

    Calls the base :func:`retrieve_memory` first, then augments the
    returned dict with additional prompt-oriented keys derived from the
    knowledge graph and embedder.

    Parameters
    ----------
    memory : ExperienceMemory
        The flat experience memory.
    library_state : dict, optional
        Current library diagnostics.
    max_success, max_forbidden, max_insights : int
        Limits forwarded to the base retrieval.
    kg : FactorKnowledgeGraph, optional
        Knowledge graph instance.
    embedder : FormulaEmbedder, optional
        Formula embedder instance.

    Returns
    -------
    dict
        Base memory signal plus the four additional keys above.
    """
    # Base retrieval
    result = retrieve_memory(
        memory,
        library_state=library_state,
        max_success=max_success,
        max_forbidden=max_forbidden,
        max_insights=max_insights,
    )

    # Default augmented keys
    result["complementary_patterns"] = []
    result["conflict_warnings"] = []
    result["operator_cooccurrence"] = []
    result["semantic_gaps"] = []

    # ----------------------------------------------------------------
    # Knowledge-graph augmentations
    # ----------------------------------------------------------------
    if kg is not None:
        # Complementary patterns: for each recently admitted factor,
        # find structurally complementary neighbours.
        complementary: List[str] = []
        seen: Set[str] = set()
        for admission in memory.state.recent_admissions[-5:]:
            fid = admission.get("factor_id", "")
            if not fid:
                continue
            for comp in kg.find_complementary_patterns(fid, max_hops=2):
                if comp not in seen:
                    seen.add(comp)
                    complementary.append(_describe_factor_node(kg, comp))
        result["complementary_patterns"] = complementary

        # Conflict warnings: saturated regions
        saturated_regions = kg.find_saturated_regions(threshold=0.5)
        result["conflict_warnings"] = [
            _describe_conflict_cluster(kg, region) for region in saturated_regions
        ]

        # Operator co-occurrence
        cooc = kg.get_operator_cooccurrence()
        # Sort by count descending, take top 20
        top_cooc = sorted(cooc.items(), key=lambda x: x[1], reverse=True)[:20]
        result["operator_cooccurrence"] = [
            f"{a} + {b} (seen {count} times)" for (a, b), count in top_cooc
        ]

    # ----------------------------------------------------------------
    # Embedding-based augmentations
    # ----------------------------------------------------------------
    if embedder is not None:
        result["semantic_gaps"] = _find_semantic_gaps(memory, kg, embedder)

    # ----------------------------------------------------------------
    # Augment prompt text
    # ----------------------------------------------------------------
    extra_sections: List[str] = []

    if result["complementary_patterns"]:
        extra_sections.append("=== COMPLEMENTARY PATTERNS (explore) ===")
        extra_sections.append(
            "Factors structurally complementary to recent admissions:"
        )
        for fid in result["complementary_patterns"][:8]:
            extra_sections.append(f"  - {fid}")
        extra_sections.append("")

    if result["conflict_warnings"]:
        extra_sections.append("=== SATURATION WARNINGS ===")
        extra_sections.append(
            "The following factor clusters are highly correlated -- "
            "avoid generating variants:"
        )
        for cluster in result["conflict_warnings"][:5]:
            extra_sections.append(f"  Cluster: {', '.join(cluster[:6])}")
        extra_sections.append("")

    if result["semantic_gaps"]:
        extra_sections.append("=== SEMANTIC GAPS (underexplored) ===")
        extra_sections.append(
            "Operators present in success patterns but underused in the library:"
        )
        for op in result["semantic_gaps"][:10]:
            extra_sections.append(f"  - {op}")
        extra_sections.append("")

    if extra_sections:
        result["prompt_text"] += "\n" + "\n".join(extra_sections)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_semantic_gaps(
    memory: ExperienceMemory,
    kg: Optional[FactorKnowledgeGraph],  # type: ignore[type-arg]
    embedder: Optional[FormulaEmbedder],  # type: ignore[type-arg]
) -> List[str]:
    """Identify operators that appear in success-pattern templates but
    are under-represented in admitted factors in the knowledge graph.
    """
    import re

    # Operators mentioned in success-pattern templates
    template_ops: Set[str] = set()
    op_pattern = re.compile(r"\b([A-Z][a-zA-Z]+)\(")
    for pat in memory.success_patterns:
        for match in op_pattern.finditer(pat.template):
            template_ops.add(match.group(1))

    if kg is None or kg.get_factor_count() == 0:
        # No graph data -- return all template ops as potential gaps
        return sorted(template_ops)

    # Count operator usage in admitted factors
    cooc = kg.get_operator_cooccurrence()
    used_ops: Set[str] = set()
    for (a, b) in cooc:
        used_ops.add(a)
        used_ops.add(b)

    # Also scan individual factors for single-operator formulas
    try:
        for node_id, attrs in kg._graph.nodes(data=True):
            if attrs.get("node_type") != "factor":
                continue
            data = attrs.get("data", {})
            if data.get("admitted"):
                for op in data.get("operators", []):
                    used_ops.add(op)
    except Exception:
        pass

    gaps = template_ops - used_ops
    return sorted(gaps)


def _describe_factor_node(
    kg: FactorKnowledgeGraph,  # type: ignore[type-arg]
    factor_id: str,
) -> str:
    """Render a factor node into short prompt-friendly text."""
    try:
        attrs = kg._graph.nodes.get(factor_id, {})
        data = attrs.get("data", {})
    except Exception:
        return factor_id

    category = data.get("category", "") or "unknown"
    ic_mean = data.get("ic_mean")
    formula = data.get("formula", "")
    summary = factor_id
    if category:
        summary += f" [{category}]"
    if ic_mean is not None:
        summary += f" IC={float(ic_mean):.4f}"
    if formula:
        summary += f": {formula[:80]}"
        if len(formula) > 80:
            summary += "..."
    return summary


def _describe_conflict_cluster(
    kg: FactorKnowledgeGraph,  # type: ignore[type-arg]
    cluster: Set[str],
) -> str:
    """Render one saturated cluster into short text."""
    described = [_describe_factor_node(kg, factor_id) for factor_id in sorted(cluster)]
    return " | ".join(described[:3])
