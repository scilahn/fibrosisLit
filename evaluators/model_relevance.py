"""
model_relevance.py — Detect model systems in paper text and score translational
relevance to human IPF using the model hierarchy from fibrosis_priors.

Detection is keyword/regex-based: fast, deterministic, and auditable. No LLM
inference is used here; the biology is encoded in fibrosis_priors.MODELS.

Clinical trial patterns are checked first so they score 1.0. Bleomycin acute
patterns are checked last to avoid masking chronic variants — a paper describing
"chronic bleomycin (42 days)" must not be mis-classified as acute simply because
"bleomycin mouse" also matches.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from domain_knowledge.fibrosis_priors import (
    MODELS,
    UNKNOWN_MODEL_FALLBACK_SCORE,
    get_model,
    get_model_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection patterns — ordered highest-evidence first.
# Clinical trials are checked first (score = 1.0). Within preclinical models,
# chronic bleomycin is checked before acute to prevent false positives.
# Each entry: (canonical_model_key, [regex_patterns]).
# Patterns are matched case-insensitively against title + abstract.
# ---------------------------------------------------------------------------

MODEL_PATTERNS: list[tuple[str, list[str]]] = [
    ("phase_3_clinical_trial", [
        r"\bphase[ -]?3\b.{0,60}(trial|study|RCT)",
        r"\bphase[ -]?III\b.{0,60}(trial|study|RCT)",
        r"(randomized|randomised).{0,30}(phase[ -]?3|phase[ -]?III)",
        r"(FIBRONEER|INBUILD|TOMORROW|CAPACITY|ASCEND).{0,30}(trial|study)",
    ]),
    ("phase_2_clinical_trial", [
        r"\bphase[ -]?2\b.{0,60}(trial|study|RCT)",
        r"\bphase[ -]?II\b.{0,60}(trial|study|RCT)",
        r"(randomized|randomised).{0,30}(phase[ -]?2|phase[ -]?II)",
        r"(INTEGRIS|FIBRONEER|ISABELA|ZEPHYR).{0,30}(trial|study)",
    ]),
    ("phase_1_clinical_trial", [
        r"\bphase[ -]?1\b.{0,60}(trial|study|dose.escalation|first.in.human)",
        r"\bphase[ -]?I\b.{0,60}(trial|study|dose.escalation|first.in.human)",
        r"first.in.human.{0,30}(IPF|fibrosis|pulmonary)",
        r"dose.escalation.{0,30}(IPF|fibrosis|pulmonary)",
    ]),
    ("human_biopsy_scrnaseq", [
        r"(single.cell|snRNA.seq|scRNA.seq).{0,40}(human|patient|IPF|biopsy)",
        r"(human|patient|IPF).{0,20}biopsy.{0,20}single.cell",
    ]),
    ("human_explant_pcls", [
        r"\bPCLS\b",
        r"precision.cut lung slice",
        r"human lung explant",
        r"ex vivo.{0,15}human.{0,15}lung",
    ]),
    ("primary_human_fibroblasts", [
        r"primary human (lung )?fibroblast",
        r"IPF fibroblast",
        r"HFL.?1",
    ]),
    ("ipsc_derived", [
        r"\biPSC\b",
        r"induced pluripotent",
    ]),
    ("humanized_mouse", [
        r"humanized mouse",
        r"human.{0,10}engraft",
    ]),
    ("bleomycin_mouse_chronic", [
        r"chronic bleomycin",
        r"repeated bleomycin",
        r"bleomycin.{0,5}(28|35|42|56).?day",
    ]),
    ("tgfb_overexpression", [
        r"TGF.?[bβ].{0,10}(transgenic|overexpression|overexpressing)",
        r"(dox(ycycline)?|tet).{0,10}TGF.?[bβ].{0,10}inducible",
    ]),
    ("bleomycin_mouse_acute", [   # checked last — fallback for any bleomycin+animal hit
        r"bleomycin.{0,30}(mouse|mice|murine|rat)",
        r"(intratracheal|i\.t\.).{0,10}bleomycin",
    ]),
]


@dataclass
class ModelRelevanceResult:
    """
    Result of preclinical model detection for a single paper.

    detected_models are ordered by their position in MODEL_PATTERNS (highest-
    confidence human models first). primary_model is the one with the highest
    translational relevance score from MODELS — not necessarily
    the first detected, since a paper may use both human biopsies and bleomycin
    mice, and we want to score it by its strongest evidence system.
    """
    detected_models: list[str]     # canonical keys, in detection order
    primary_model: str             # key with highest score among detected
    primary_score: float           # score of primary_model
    warnings: list[str]            # model flags: "frequently_overcited", etc.
    rationale: list[str]           # one entry per detected model


def _match_patterns(text: str, patterns: list[str]) -> str | None:
    """Return the first matching snippet, or None if no pattern matches."""
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(0)
    return None


def score_model_relevance(
    title: str,
    abstract: str,
    disease: str = "ipf",   # reserved for future PSC priors
) -> ModelRelevanceResult:
    """
    Detect preclinical model systems in paper text and return translational
    relevance scores from the IPF model hierarchy in fibrosis_priors.

    Detection is keyword-based (regex, case-insensitive). When multiple models
    are detected, the highest-scoring model is used as the primary score.
    Bleomycin acute detection runs last to avoid masking chronic patterns.

    Args:
        title:    Paper title.
        abstract: Paper abstract.
        disease:  Target disease context. Currently only "ipf" is supported;
                  PSC priors are planned for a future session.

    Returns:
        ModelRelevanceResult with detected models, primary score, and audit log.
    """
    text = f"{title} {abstract}"
    detected: list[str] = []
    rationale: list[str] = []

    for model_key, patterns in MODEL_PATTERNS:
        snippet = _match_patterns(text, patterns)
        if snippet is not None and model_key not in detected:
            score = get_model_score(model_key)
            model = get_model(model_key)
            model_rationale = model.rationale if model else "Unknown model."
            rationale.append(
                f"[{model_key}] score={score:.2f} | matched: '{snippet[:80]}' | "
                f"rationale: {model_rationale[:120]}"
            )
            detected.append(model_key)
            logger.debug("Detected model '%s' (score=%.2f) via pattern match.", model_key, score)

    if not detected:
        # No recognised model — use fallback and log
        primary = "unknown"
        primary_score = UNKNOWN_MODEL_FALLBACK_SCORE
        rationale.append(
            f"[unknown] score={primary_score:.2f} | No recognised model system detected. "
            "Conservative fallback applied."
        )
        logger.debug("No model detected; applying fallback score %.2f.", primary_score)
        return ModelRelevanceResult(
            detected_models=[],
            primary_model=primary,
            primary_score=primary_score,
            warnings=["unrecognised_model_system"],
            rationale=rationale,
        )

    # Select primary model as the one with the highest translational score
    primary_model = max(detected, key=lambda k: get_model_score(k))
    primary_score = get_model_score(primary_model)

    # Collect all flags from detected models
    warnings: list[str] = []
    for model_key in detected:
        model = get_model(model_key)
        if model:
            for flag in model.flags:
                if flag not in warnings:
                    warnings.append(flag)

    logger.info(
        "Model relevance: primary=%s score=%.2f detected=%s warnings=%s",
        primary_model, primary_score, detected, warnings,
    )

    return ModelRelevanceResult(
        detected_models=detected,
        primary_model=primary_model,
        primary_score=primary_score,
        warnings=warnings,
        rationale=rationale,
    )
