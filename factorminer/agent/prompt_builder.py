"""Build prompts for LLM-driven factor generation using memory priors.

The system prompt encodes the full operator library, syntax rules, feature
list, and task description.  The user prompt injects per-iteration context:
memory signals, library state, and output format instructions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from factorminer.core.types import (
    FEATURES,
    OPERATOR_REGISTRY,
    OperatorSpec,
    OperatorType,
)


def _format_operator_table() -> str:
    """Build a human-readable operator reference table grouped by category."""
    grouped: Dict[str, List[OperatorSpec]] = {}
    for spec in OPERATOR_REGISTRY.values():
        cat = spec.category.name
        grouped.setdefault(cat, []).append(spec)

    lines: List[str] = []
    for cat_name in [
        "ARITHMETIC",
        "STATISTICAL",
        "TIMESERIES",
        "SMOOTHING",
        "CROSS_SECTIONAL",
        "REGRESSION",
        "LOGICAL",
        "AUTO_INVENTED",
    ]:
        specs = grouped.get(cat_name, [])
        if not specs:
            continue
        lines.append(f"\n### {cat_name} operators")
        for spec in sorted(specs, key=lambda s: s.name):
            params_str = ""
            if spec.param_names:
                parts = []
                for pname in spec.param_names:
                    default = spec.param_defaults.get(pname, "")
                    lo, hi = spec.param_ranges.get(pname, (None, None))
                    range_str = f"[{lo}-{hi}]" if lo is not None else ""
                    parts.append(f"{pname}={default}{range_str}")
                params_str = f"  params: {', '.join(parts)}"
            arity_args = ", ".join([f"expr{i+1}" for i in range(spec.arity)])
            if spec.param_names:
                arity_args += ", " + ", ".join(spec.param_names)
            lines.append(f"- {spec.name}({arity_args}): {spec.description}{params_str}")
    return "\n".join(lines)


def _format_feature_list() -> str:
    """Build a description of available raw features."""
    descriptions = {
        "$open": "opening price",
        "$high": "highest price in the bar",
        "$low": "lowest price in the bar",
        "$close": "closing price",
        "$volume": "trading volume (shares)",
        "$amt": "trading amount (currency value)",
        "$vwap": "volume-weighted average price",
        "$returns": "close-to-close returns",
    }
    lines = []
    for feat in FEATURES:
        desc = descriptions.get(feat, "")
        lines.append(f"  {feat}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a quantitative researcher mining formulaic alpha factors for stock selection.

Your goal is to generate novel, predictive factor expressions using a tree-structured domain-specific language (DSL). Each factor is a composition of operators applied to raw market features.

## RAW FEATURES (leaf nodes)
{_format_feature_list()}

## OPERATOR LIBRARY
{_format_operator_table()}

## EXPRESSION SYNTAX RULES
1. Every expression is a nested function call: Operator(args...)
2. Leaf nodes are raw features ($close, $volume, etc.) or numeric constants.
3. Operators are called by name with expression arguments first, then numeric parameters:
   - Mean($close, 20) = 20-day rolling mean of $close
   - Corr($close, $volume, 10) = 10-day rolling correlation of close and volume
   - IfElse(Greater($returns, 0), $volume, Neg($volume)) = conditional
4. No infix operators; use Add(x,y) instead of x+y, Sub(x,y) instead of x-y, etc.
5. Parameters like window sizes are trailing numeric arguments after expression children.
6. Valid window sizes are integers; check each operator's parameter ranges above.
7. Cross-sectional operators (CsRank, CsZScore, CsDemean, CsScale, CsNeutralize) operate across all stocks at each time step -- they are crucial for making factors comparable.

## EXAMPLES OF WELL-FORMED FACTORS
- Neg(CsRank(Delta($close, 5)))
  Short-term reversal: rank of 5-day price change, negated.
- CsZScore(Div(Sub($volume, Mean($volume, 20)), Std($volume, 20)))
  Volume surprise: standardized deviation from 20-day mean volume.
- CsRank(Div(Sub($close, $vwap), $vwap))
  Intraday deviation from VWAP, cross-sectionally ranked.
- Neg(Corr($volume, $close, 10))
  Negative price-volume correlation over 10 days.
- CsRank(TsLinRegSlope($volume, 20))
  Trend in trading volume over 20 days, ranked.
- IfElse(Greater($returns, 0), Std($returns, 10), Neg(Std($returns, 10)))
  Conditional volatility: positive for up-moves, negative for down-moves.
- CsRank(Div(Sub($close, TsMin($low, 20)), Sub(TsMax($high, 20), TsMin($low, 20))))
  Position within 20-day price range, ranked.

## KEY PRINCIPLES FOR HIGH-QUALITY FACTORS
- Always wrap the outermost expression with a cross-sectional operator (CsRank, CsZScore) for comparability.
- Combine DIFFERENT operator types for novelty (e.g., time-series + cross-sectional + arithmetic).
- Use diverse window sizes; avoid always defaulting to 10.
- Explore uncommon feature combinations ($amt, $vwap are underused).
- Factors with depth 3-7 tend to be best: deep enough to capture non-trivial patterns but not so deep they overfit.
- Prefer economically meaningful combinations over random nesting.
"""


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Constructs system and user prompts for factor generation.

    The system prompt is static (operator library + rules).
    The user prompt varies each iteration based on memory signals.
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self._system_prompt = system_prompt or SYSTEM_PROMPT

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def build_user_prompt(
        self,
        memory_signal: Dict[str, Any],
        library_state: Dict[str, Any],
        batch_size: int = 40,
    ) -> str:
        """Build the per-iteration user prompt injecting memory priors.

        Parameters
        ----------
        memory_signal : dict
            Keys:
            - ``"recommended_directions"`` : list[str] -- patterns to explore
            - ``"forbidden_directions"`` : list[str] -- patterns to avoid
            - ``"strategic_insights"`` : list[str] -- high-level lessons
            - ``"recent_rejections"`` : list[dict] -- recent rejection reasons
        library_state : dict
            Keys:
            - ``"size"`` : int -- current library size
            - ``"target_size"`` : int -- target library size
            - ``"recent_admissions"`` : list[str] -- recently admitted factor names
            - ``"domain_saturation"`` : dict[str, float] -- per-domain saturation
        batch_size : int
            Number of candidates to generate this iteration.

        Returns
        -------
        str
            The fully assembled user prompt.
        """
        sections: List[str] = []

        # --- Task directive ---
        sections.append(
            f"Generate exactly {batch_size} novel, diverse alpha factor candidates."
        )

        # --- Library status ---
        lib_size = library_state.get("size", 0)
        target = library_state.get("target_size", 110)
        sections.append(
            f"\n## CURRENT LIBRARY STATUS\n"
            f"Library size: {lib_size} / {target} factors."
        )

        recent = library_state.get("recent_admissions", [])
        if recent:
            sections.append(
                "Recently admitted factors:\n"
                + "\n".join(f"  - {f}" for f in recent[-10:])
            )

        saturation = library_state.get("domain_saturation", {})
        if saturation:
            sat_lines = [f"  {domain}: {pct:.0%} saturated" for domain, pct in saturation.items()]
            sections.append(
                "Domain saturation:\n" + "\n".join(sat_lines)
            )

        # --- Memory signal: recommended directions ---
        rec_dirs = memory_signal.get("recommended_directions", [])
        if rec_dirs:
            sections.append(
                "\n## RECOMMENDED DIRECTIONS (focus on these successful patterns)\n"
                + "\n".join(f"  * {d}" for d in rec_dirs)
            )

        # --- Memory signal: forbidden directions ---
        forbidden = memory_signal.get("forbidden_directions", [])
        if forbidden:
            sections.append(
                "\n## FORBIDDEN DIRECTIONS (AVOID these -- they produce correlated/weak factors)\n"
                + "\n".join(f"  X {d}" for d in forbidden)
            )

        # --- Memory signal: strategic insights ---
        insights = memory_signal.get("strategic_insights", [])
        if insights:
            sections.append(
                "\n## STRATEGIC INSIGHTS\n"
                + "\n".join(f"  Note: {ins}" for ins in insights)
            )

        helix_prompt_text = memory_signal.get("prompt_text", "").strip()
        if helix_prompt_text:
            sections.append(
                "\n## HELIX RETRIEVAL SUMMARY\n"
                f"{helix_prompt_text}"
            )

        complementary_patterns = memory_signal.get("complementary_patterns", [])
        if complementary_patterns:
            sections.append(
                "\n## COMPLEMENTARY PATTERNS\n"
                + "\n".join(f"  + {pattern}" for pattern in complementary_patterns)
            )

        conflict_warnings = memory_signal.get("conflict_warnings", [])
        if conflict_warnings:
            sections.append(
                "\n## SATURATION WARNINGS\n"
                + "\n".join(f"  ! {warning}" for warning in conflict_warnings)
            )

        operator_cooccurrence = memory_signal.get("operator_cooccurrence", [])
        if operator_cooccurrence:
            sections.append(
                "\n## OPERATOR CO-OCCURRENCE PRIORS\n"
                + "\n".join(f"  - {pair}" for pair in operator_cooccurrence)
            )

        semantic_gaps = memory_signal.get("semantic_gaps", [])
        if semantic_gaps:
            sections.append(
                "\n## SEMANTIC GAPS\n"
                + "\n".join(
                    f"  - Underused but promising: {gap}" for gap in semantic_gaps
                )
            )

        # --- Recent rejection reasons ---
        rejections = memory_signal.get("recent_rejections", [])
        if rejections:
            rej_lines = []
            for rej in rejections[-10:]:
                name = rej.get("name", "unknown")
                reason = rej.get("reason", "unknown")
                rej_lines.append(f"  - {name}: rejected because {reason}")
            sections.append(
                "\n## RECENT REJECTIONS (learn from these failures)\n"
                + "\n".join(rej_lines)
            )

        # --- Orthogonality directive ---
        sections.append(
            "\n## CRITICAL REQUIREMENT: ORTHOGONALITY\n"
            "Generate factors that are UNCORRELATED with existing library members. "
            "Each candidate should explore a DIFFERENT structural pattern. "
            "Vary your operator choices, window sizes, feature combinations, and "
            "nesting depth across candidates. Do NOT generate trivial variations "
            "of the same formula (e.g., changing only the window size)."
        )

        # --- Output format ---
        sections.append(
            f"\n## OUTPUT FORMAT\n"
            f"Output exactly {batch_size} factors, one per line.\n"
            f"Format each line as: <number>. <factor_name>: <formula>\n"
            f"Example:\n"
            f"1. momentum_reversal: Neg(CsRank(Delta($close, 5)))\n"
            f"2. volume_surprise: CsZScore(Div(Sub($volume, Mean($volume, 20)), Std($volume, 20)))\n"
            f"\nRules:\n"
            f"- factor_name: lowercase_with_underscores, descriptive, unique\n"
            f"- formula: valid DSL expression using ONLY operators and features listed above\n"
            f"- No markdown, no explanations, no extra text -- just the numbered list\n"
            f"- Every formula must parse correctly with the operator library"
        )

        return "\n".join(sections)
