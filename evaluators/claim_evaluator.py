"""
claim_evaluator.py — Evaluate a biological claim against IPF domain priors.

Answers "how well-supported is this claim a priori?" without reference to
any specific paper. Uses pathway centrality priors from fibrosis_priors.PATHWAYS,
model translation hierarchy from fibrosis_priors.MODELS, and contested biology
from fibrosis_priors.CONTESTED_BIOLOGY.

This evaluator is intentionally distinct from evaluate_paper(): papers are
scored on the quality of evidence they contain; claims are scored on how well
their biological assertions align with established IPF domain priors. A claim
about a central pathway (TGF-β/SMAD) starts with high prior support; a claim
attributing findings to bleomycin mouse data starts with a model penalty.

Contested biology acts as a hard tier override, not a score penalty. A claim
invoking myofibroblast reversibility cannot be classified WELL_SUPPORTED
without resolving an active debate — which this evaluator refuses to do.

Tier classification:
    WELL_SUPPORTED — high pathway prior support, no model penalty, no contested flags.
    CONTESTED      — hard override when contested biology is detected; also used as the
                     conservative default for claims in the ambiguous mid-range.
    OVERCLAIMED    — strong pathway prior but resting on poor-translation model evidence,
                     OR no recognized pathway support at all.

Known limitations of prior-based evaluation:
    - Claims about model validity (e.g. "bleomycin faithfully recapitulates IPF") may
      be classified OVERCLAIMED rather than CONTESTED because model-only claims score
      pathway_support = 0.0, which falls below the OVERCLAIMED_THRESHOLD.
    - Claims that require clinical outcome knowledge (e.g. failed trial X) to classify
      as OVERCLAIMED will land in the ambiguous CONTESTED bucket — the evaluator cannot
      know trial results without paper evidence.
    - The prior_support_score reflects pathway biology only, not clinical efficacy data.
"""

import logging
import re
from dataclasses import dataclass, field

from domain_knowledge.fibrosis_priors import (
    get_pathway_prior,
    get_model,
    has_flag,
)
from evaluators.contradiction_detector import ContestedFlag, detect_contested_claims
from evaluators.evidence_quality import PATHWAY_PATTERNS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier classification thresholds
# ---------------------------------------------------------------------------

WELL_SUPPORTED_THRESHOLD: float = 0.70
OVERCLAIMED_THRESHOLD: float = 0.35

# ---------------------------------------------------------------------------
# Model penalty magnitudes — applied when a claim mentions a model system with
# known poor IPF translational relevance.
# ---------------------------------------------------------------------------

POOR_TRANSLATION_PENALTY: float = 0.40   # models with has_flag("poor_ipf_translation")
LOW_TRANSLATABILITY_PENALTY: float = 0.25 # models with score < 0.50 (e.g. tgfb_overexpression)

# ---------------------------------------------------------------------------
# Cell-type and population entity patterns — detect biological entities that
# have strong IPF domain support but are not represented in PATHWAY_PATTERNS.
# Each key maps to CLAIM_ENTITY_PRIORS for scoring.
# ---------------------------------------------------------------------------

CLAIM_ENTITY_PATTERNS: dict[str, list[str]] = {
    # SPP1hi macrophage population: defined in human IPF scRNA-seq atlases (Adams 2020,
    # Habermann 2019). Distinct from M1/M2 framework; SPP1+ macrophages are a specific
    # disease-associated population with strong paracrine activation of myofibroblasts.
    "spp1_macrophage": [
        r"\bSPP1\b",
        r"SPP1\+",
        r"SPP1.{0,10}macrophage",
    ],
    # CTHRC1+ myofibroblast axis: CTHRC1 is the canonical marker of pathological
    # myofibroblast subpopulation in human IPF atlases. α-SMA (ACTA2) is the classical
    # myofibroblast marker, well-validated in human biopsy tissue.
    "myofibroblast_axis": [
        r"\bmyofibroblast\b",
        r"\bCTHRC1\b",
        r"\b[aα][.-]?SMA\b",
        r"\bACTA2\b",
    ],
    # Aberrant basaloid cells: KRT17+ cells co-expressing epithelial and mesenchymal
    # markers are a disease-specific population identified in human IPF single-cell atlases
    # (Kathiriya 2023). Their existence in human IPF lung is well-supported; their
    # mechanistic interpretation (partial EMT vs. aberrant differentiation) is contested.
    "aberrant_basaloid": [
        r"\baberrant basaloid\b",
        r"\bKRT17\b",
    ],
}

CLAIM_ENTITY_PRIORS: dict[str, float] = {
    "spp1_macrophage":    0.85,  # strong human scRNA-seq consensus across multiple IPF atlases
    "myofibroblast_axis": 0.80,  # well-characterized in human biopsy, primary fibroblast data
    "aberrant_basaloid":  0.75,  # human-specific; well-supported as a cell population
}

# ---------------------------------------------------------------------------
# Model mention patterns — detect when a claim invokes a preclinical model
# system with known poor IPF translational relevance. Penalty is applied as
# a multiplier suppressing pathway_support_score.
# Keys match canonical model identifiers in fibrosis_priors.MODELS so that
# has_flag() and get_model() can be called directly.
# ---------------------------------------------------------------------------

CLAIM_MODEL_PATTERNS: list[tuple[str, list[str]]] = [
    ("bleomycin_mouse_acute", [
        r"(intratracheal|i\.t\.).{0,10}bleomycin",
        r"bleomycin model",
        r"bleomycin.{0,30}(mouse|mice|murine|rat)",
    ]),
    ("bleomycin_mouse_chronic", [
        r"chronic bleomycin",
        r"repeated bleomycin",
        r"bleomycin.{0,5}(28|35|42|56).?day",
    ]),
    ("tgfb_overexpression", [
        r"TGF.?[bβ].{0,10}(transgenic|overexpression|overexpressing)",
        r"(dox(ycycline)?|tet).{0,10}inducible.{0,10}TGF.?[bβ]",
        r"TGF.?[bβ].{0,10}inducible",
    ]),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClaimEvaluationResult:
    """
    Prior-based evaluation of a single biological claim against IPF domain knowledge.

    This result reflects how well the claim aligns with established IPF biology
    priors — NOT whether any paper supports the claim. Papers are evaluated separately
    by evidence_quality.evaluate_paper(); this result is the domain-knowledge baseline
    against which paper-level evidence should be interpreted.

    detected_pathways includes both PATHWAY_PATTERNS keys and CLAIM_ENTITY_PATTERNS
    keys (cell-type level entities), unified under a single list with their
    prior scores captured in the rationale.

    contested_flags stores full ContestedFlag objects (including competing positions)
    so callers can surface the full debate text without needing to go back to
    fibrosis_priors. This differs from EvidenceQualityReport.contested_flags which
    stores only keys — claim evaluation is consumed interactively (benchmark runs)
    where the full debate text is immediately useful.
    """
    claim: str
    disease: str

    # Detection results
    detected_pathways: list[str]          # pathway + entity keys with recognized priors
    detected_model_mentions: list[str]    # canonical model keys from fibrosis_priors.MODELS

    # Full ContestedFlag objects — surface all positions, never synthesize
    contested_flags: list[ContestedFlag]

    # Intermediate scores
    pathway_support_score: float   # max pathway/entity prior across detections (0.0–1.0)
    model_penalty: float           # deduction applied as multiplier (0.0–1.0)

    # Composite prior support
    prior_support_score: float     # pathway_support_score * (1.0 - model_penalty)

    # Tier classification
    tier: str  # "WELL_SUPPORTED" | "CONTESTED" | "OVERCLAIMED"

    # Audit trail — one entry per scoring decision, matching rationale format in
    # evidence_quality.EvidenceQualityReport for consistency across evaluators.
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_claim_pathways(claim: str) -> list[tuple[str, float]]:
    """
    Return (key, prior_score) pairs for all pathways and biological entities
    detected in the claim text.

    Checks PATHWAY_PATTERNS first (canonical IPF pathways from fibrosis_priors),
    then CLAIM_ENTITY_PATTERNS (cell-type and population markers with their own
    domain priors). Uses the same first-match-per-key logic as _detect_pathways
    in evidence_quality.py.

    Args:
        claim: Biological claim as a declarative string.

    Returns:
        List of (key, score) tuples, one per detected pathway or entity.
    """
    results: list[tuple[str, float]] = []

    for pathway_key, patterns in PATHWAY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, claim, re.IGNORECASE | re.DOTALL):
                results.append((pathway_key, get_pathway_prior(pathway_key)))
                break

    for entity_key, patterns in CLAIM_ENTITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, claim, re.IGNORECASE | re.DOTALL):
                results.append((entity_key, CLAIM_ENTITY_PRIORS[entity_key]))
                break

    return results


def _detect_model_mentions(claim: str) -> list[str]:
    """
    Return canonical model keys for any preclinical model systems mentioned
    in the claim text.

    Checks CLAIM_MODEL_PATTERNS in order; all matches are collected (a claim
    may mention multiple model systems). Uses first-match-per-model logic to
    avoid duplicate detection of the same model via different patterns.

    Args:
        claim: Biological claim as a declarative string.

    Returns:
        List of canonical model keys (keys in fibrosis_priors.MODELS).
    """
    detected: list[str] = []
    for model_key, patterns in CLAIM_MODEL_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, claim, re.IGNORECASE | re.DOTALL):
                detected.append(model_key)
                break
    return detected


def _compute_model_penalty(
    detected_model_mentions: list[str],
    rationale: list[str],
    warnings: list[str],
) -> float:
    """
    Compute the maximum model-based penalty across all detected model mentions.

    A claim citing bleomycin AND TGF-β overexpression is penalized for the
    worst of the two, not doubly penalized. The penalty is meant to be applied
    as a multiplier: prior_support_score = pathway_score * (1.0 - penalty).

    Penalty tiers:
        POOR_TRANSLATION_PENALTY (0.40)  — models flagged "poor_ipf_translation"
        LOW_TRANSLATABILITY_PENALTY (0.25) — models with score < 0.50 that lack
                                             the explicit flag (tgfb_overexpression)

    Args:
        detected_model_mentions: Canonical model keys from _detect_model_mentions.
        rationale: Mutated in place — one entry appended per model.
        warnings: Mutated in place — model flags appended when present.

    Returns:
        Maximum penalty value in [0.0, 1.0]. 0.0 if no models detected.
    """
    if not detected_model_mentions:
        return 0.0

    max_penalty: float = 0.0
    for model_key in detected_model_mentions:
        model = get_model(model_key)
        if model is None:
            # Unknown model key — should not happen with CLAIM_MODEL_PATTERNS, but
            # defend against future pattern additions that lack a MODELS entry.
            rationale.append(
                f"[model] {model_key}: unrecognized model key, skipping penalty"
            )
            continue

        if has_flag(model_key, "poor_ipf_translation"):
            penalty = POOR_TRANSLATION_PENALTY
            for flag in model.flags:
                if flag not in warnings:
                    warnings.append(flag)
            rationale.append(
                f"[model] {model_key}: score={model.score:.2f} "
                f"poor_ipf_translation flag → penalty={penalty:.2f}"
            )
        elif model.score < 0.50:
            penalty = LOW_TRANSLATABILITY_PENALTY
            rationale.append(
                f"[model] {model_key}: score={model.score:.2f} "
                f"(low translatability, < 0.50) → penalty={penalty:.2f}"
            )
        else:
            penalty = 0.0
            rationale.append(
                f"[model] {model_key}: score={model.score:.2f} → no penalty applied"
            )

        max_penalty = max(max_penalty, penalty)

    return max_penalty


def _classify_tier(
    prior_support_score: float,
    contested_flags: list[ContestedFlag],
    model_penalty: float,
) -> str:
    """
    Classify a claim into WELL_SUPPORTED, CONTESTED, or OVERCLAIMED.

    Classification order (each condition is checked in sequence; first match wins):
    1. Contested flags present → CONTESTED (hard override; resolving a live debate
       is outside the scope of prior-based evaluation).
    2. High pathway support AND no model penalty → WELL_SUPPORTED.
    3. High pathway support BUT model penalty present → OVERCLAIMED (real biology
       invoked, but claim rests on poor-translation model evidence — the exact
       pattern of OC-01, OC-03).
    4. Very low pathway support → OVERCLAIMED (claim invokes no recognized pathway
       or cell-type prior, suggesting biologically unsupported or extrapolated claim).
    5. Default (ambiguous mid-range) → CONTESTED (conservative; the evaluator lacks
       sufficient priors to confidently classify, so surfaces as needing scrutiny).

    Known edge case: claims about model system fidelity (e.g. CT-06 "bleomycin
    faithfully recapitulates IPF") have pathway_support=0.0 and model_penalty>0,
    which places them in OVERCLAIMED rather than CONTESTED. This is a limitation
    of prior-based evaluation — the debate about model validity is a contested one,
    but the evaluator cannot distinguish "claim about model fidelity" from "claim
    attributing biology to model data" without semantic parsing.

    Args:
        prior_support_score: pathway_support_score * (1.0 - model_penalty).
        contested_flags:     Detected contested biology flags.
        model_penalty:       Applied penalty value (0.0 = none detected).

    Returns:
        "WELL_SUPPORTED" | "CONTESTED" | "OVERCLAIMED"
    """
    if contested_flags:
        return "CONTESTED"

    if prior_support_score >= WELL_SUPPORTED_THRESHOLD:
        if model_penalty == 0.0:
            return "WELL_SUPPORTED"
        else:
            return "OVERCLAIMED"

    if prior_support_score < OVERCLAIMED_THRESHOLD:
        return "OVERCLAIMED"

    # Mid-range (OVERCLAIMED_THRESHOLD <= score < WELL_SUPPORTED_THRESHOLD):
    # insufficient prior support to declare well-supported, but not obviously
    # wrong — surface as CONTESTED for further scrutiny.
    return "CONTESTED"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_claim(
    claim: str,
    disease: str = "ipf",
) -> ClaimEvaluationResult:
    """
    Evaluate a biological claim against IPF domain priors.

    Answers "how well-supported is this claim a priori?" using only domain
    knowledge encoded in fibrosis_priors. Does NOT evaluate whether any paper
    supports the claim — that is the retrieval layer's job.

    Three detection passes are applied to the claim text:
    1. Pathway + entity detection: identifies recognized IPF pathways (PATHWAY_PATTERNS)
       and cell-type markers (CLAIM_ENTITY_PATTERNS). pathway_support_score is the
       maximum prior across all detected pathways/entities.
    2. Model mention detection: identifies preclinical model systems with poor IPF
       translational relevance (CLAIM_MODEL_PATTERNS). model_penalty suppresses the
       pathway_support_score multiplicatively.
    3. Contested biology detection: reuses detect_contested_claims() from
       contradiction_detector. Any contested flag forces tier = CONTESTED regardless
       of pathway support — the evaluator refuses to resolve live debates.

    prior_support_score = pathway_support_score * (1.0 - model_penalty)

    This score reflects only pathway/entity biology and model-system caveats.
    It does NOT encode clinical trial outcomes, organ-specificity arguments, or
    validation status claims — those require paper-level evidence.

    Args:
        claim:   Biological claim as a declarative string.
        disease: Target disease context. Currently only "ipf" is supported.

    Returns:
        ClaimEvaluationResult with pathway support, model penalty, contested
        positions, prior_support_score, tier classification, and full audit
        rationale.
    """
    rationale: list[str] = []
    warnings: list[str] = []

    # 1. Pathway + entity detection
    detected_pathway_pairs = _detect_claim_pathways(claim)
    detected_pathways = [key for key, _ in detected_pathway_pairs]

    if detected_pathway_pairs:
        pathway_support_score = max(score for _, score in detected_pathway_pairs)
        for key, score in detected_pathway_pairs:
            rationale.append(f"[pathway] {key}: prior={score:.2f}")
        rationale.append(
            f"[pathway] pathway_support_score={pathway_support_score:.2f} "
            f"(max across {len(detected_pathway_pairs)} detected)"
        )
    else:
        pathway_support_score = 0.0
        rationale.append("[pathway] no recognized pathways or entities detected → score=0.00")

    # 2. Model mention detection + penalty
    detected_model_mentions = _detect_model_mentions(claim)
    model_penalty = _compute_model_penalty(detected_model_mentions, rationale, warnings)

    if not detected_model_mentions:
        rationale.append("[model] no model system mentions detected → penalty=0.00")

    # 3. Contested biology detection
    # detect_contested_claims takes (title, abstract) — pass claim as title and
    # empty string as abstract so all claim text is scanned via the title path.
    contested_flags = detect_contested_claims(claim, "", disease)

    if contested_flags:
        for flag in contested_flags:
            rationale.append(
                f"[contested] {flag.debate_name}: matched='{flag.matched_snippet[:60]}' "
                f"({len(flag.positions)} competing positions — not resolved)"
            )
    else:
        rationale.append("[contested] no contested biology detected")

    # 4. Compute composite prior support
    prior_support_score = pathway_support_score * (1.0 - model_penalty)
    rationale.append(
        f"[prior_support] {pathway_support_score:.2f} × (1.0 - {model_penalty:.2f}) "
        f"= {prior_support_score:.3f}"
    )

    # 5. Tier classification
    tier = _classify_tier(prior_support_score, contested_flags, model_penalty)
    rationale.append(
        f"[tier] {tier}: "
        f"prior_support={prior_support_score:.3f} "
        f"model_penalty={model_penalty:.2f} "
        f"contested_flags={[f.debate_name for f in contested_flags]}"
    )

    logger.info(
        "Claim '%s...' | tier=%s prior_support=%.3f pathway=%.2f "
        "model_penalty=%.2f contested=%s",
        claim[:60],
        tier,
        prior_support_score,
        pathway_support_score,
        model_penalty,
        [f.debate_name for f in contested_flags],
    )

    return ClaimEvaluationResult(
        claim=claim,
        disease=disease,
        detected_pathways=detected_pathways,
        detected_model_mentions=detected_model_mentions,
        contested_flags=contested_flags,
        pathway_support_score=pathway_support_score,
        model_penalty=model_penalty,
        prior_support_score=prior_support_score,
        tier=tier,
        rationale=rationale,
        warnings=warnings,
    )
