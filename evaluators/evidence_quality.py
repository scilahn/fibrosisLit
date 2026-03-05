"""
evidence_quality.py — Orchestrate sub-evaluators into a single EvidenceQualityReport.

Combines model translational relevance, pathway centrality, study design tier,
and contested-biology flags into a weighted composite score. Every scoring
decision is logged to the rationale list for downstream auditability.

Score weights rationale:
  model (0.40)        — translational relevance is the primary failure mode in IPF;
                        bleomycin mouse results routinely fail to translate clinically.
  pathway (0.30)      — pathway centrality reflects depth of human mechanistic evidence.
  study_design (0.30) — study design independently captures methodological rigor beyond
                        the model system (e.g. RCT vs. cohort vs. in vitro).
"""

import logging
import re
from dataclasses import dataclass, field

from domain_knowledge.fibrosis_priors import get_pathway_prior
from evaluators.contradiction_detector import detect_contested_claims
from evaluators.model_relevance import score_model_relevance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Study design tiers — ordered strongest to weakest; first match wins.
# Each entry: (tier_name, score, [regex_patterns]).
# ---------------------------------------------------------------------------

STUDY_DESIGN_TIERS: list[tuple[str, float, list[str]]] = [
    ("clinical_trial", 1.00, [
        r"randomized.{0,20}(trial|study)",
        r"\bRCT\b",
        r"phase [23]",
        r"NCT\d{8}",
    ]),
    ("single_cell_human", 0.90, [
        r"(scRNA.seq|snRNA.seq).{0,30}(IPF|patient|human|biopsy)",
    ]),
    ("human_cohort", 0.85, [
        r"\b\d+\s+patients?\b",
        r"(BAL|bronchoalveolar lavage).{0,20}patient",
        r"surgical lung biopsy",
        r"serum.{0,15}IPF.{0,15}cohort",
    ]),
    ("ex_vivo_human", 0.80, [
        r"\bPCLS\b",
        r"precision.cut lung",
        r"human.{0,10}explant",
    ]),
    ("primary_cells", 0.60, [
        r"primary human (lung )?fibroblast",
        r"primary.{0,10}(alveolar|bronchial)",
    ]),
    ("animal_model", 0.50, [
        r"(mouse|mice|murine|rat).{0,20}(bleomycin|fibrosis|model)",
        r"(in vivo).{0,20}(fibrosis|IPF)",
    ]),
    ("cell_line", 0.30, [
        r"\bcell line\b",
        r"\bin vitro\b",
        r"A549",
        r"MRC.?5",
        r"NIH.?3T3",
    ]),
]

# ---------------------------------------------------------------------------
# Pathway detection patterns — keys map to PATHWAYS in fibrosis_priors.
# ---------------------------------------------------------------------------

PATHWAY_PATTERNS: dict[str, list[str]] = {
    "tgfb_smad":    [r"TGF.?[bβ]", r"\bSMAD[234]\b", r"TGFB[123]"],
    "integrin_avb6":[r"[aα]v[bβ]6", r"\bITGB6\b", r"αvβ6"],
    "tnik":         [r"\bTNIK\b"],
    "pde4":         [r"\bPDE4[ABCD]?\b", r"phosphodiesterase.{0,5}4"],
    "wnt_pathway":  [r"\bWNT\b", r"[bβ].catenin", r"\bTCF\b|\bLEF\b"],
    "hedgehog":     [r"\bSHH\b", r"\bGLI[123]\b", r"\bSMO\b", r"\bHedgehog\b"],
    "il13_tslp":    [r"\bIL.?13\b", r"\bTSLP\b"],
    "lpa1":         [r"\bLPA1\b", r"\bLPAR1\b"],
    "lpa2":         [r"\bLPA2\b", r"\bLPAR2\b"],
    "csf":          [r"\bCSF1R?\b", r"\bM.CSF\b", r"\bGM.CSF\b"],
    "autotaxin":    [r"\bATX\b|\bENPP2\b", r"\bautotaxin\b", r"\bLPA\b"],
}

# Composite score weights
SCORE_WEIGHTS: dict[str, float] = {
    "model": 0.40,
    "pathway": 0.30,
    "study_design": 0.30,
}


@dataclass
class EvidenceQualityReport:
    """
    Composite evidence quality assessment for a single paper.

    Sub-scores are all in [0, 1]. overall_score is a weighted sum using
    SCORE_WEIGHTS. contested_flags list the debate keys that were triggered,
    not the full ContestedFlag objects — use contradiction_detector directly
    if you need the full debate text.

    rationale contains one log entry per scoring decision and is the primary
    mechanism for auditing why a particular score was assigned.
    """
    pmid: str
    title: str

    # Sub-scores (0–1)
    model_score: float
    pathway_score: float
    study_design_score: float

    # Detected context
    detected_models: list[str]
    detected_pathways: list[str]
    study_design_tier: str
    contested_flags: list[str]    # debate keys only

    # Composite
    overall_score: float

    # Audit trail
    rationale: list[str]
    warnings: list[str]           # model flags: "poor_ipf_translation", etc.


def _detect_study_design(text: str) -> tuple[str, float, str]:
    """
    Return (tier_name, score, matched_snippet) for the strongest matching
    study design tier. Falls back to ("unknown", 0.0, "") if nothing matches.
    """
    for tier_name, score, patterns in STUDY_DESIGN_TIERS:
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                return tier_name, score, m.group(0)[:80]
    return "unknown", 0.0, ""


def _detect_pathways(text: str) -> list[str]:
    """Return canonical pathway keys for all pathways detected in text."""
    detected: list[str] = []
    for pathway_key, patterns in PATHWAY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                detected.append(pathway_key)
                break
    return detected


def evaluate_paper(
    paper: dict,
    disease: str = "ipf",   # reserved for future PSC priors
) -> EvidenceQualityReport:
    """
    Compute an evidence quality score for a single paper.

    `paper` must contain at minimum: pmid, title, abstract.
    Orchestrates model_relevance, contradiction_detector, and pathway detection,
    then computes a weighted composite score. Every decision is logged to
    `rationale` for downstream auditability.

    Args:
        paper:   Dict with keys 'pmid', 'title', 'abstract' (at minimum).
        disease: Target disease context. Currently only "ipf" is supported.

    Returns:
        EvidenceQualityReport with sub-scores, detections, and full audit trail.
    """
    pmid = paper.get("pmid", "unknown")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    text = f"{title} {abstract}"

    rationale: list[str] = []

    # --- Model relevance ---
    model_result = score_model_relevance(title, abstract, disease=disease)
    model_score = model_result.primary_score
    rationale.append(
        f"[model] primary={model_result.primary_model} score={model_score:.2f} "
        f"detected={model_result.detected_models}"
    )
    rationale.extend(model_result.rationale)

    # --- Pathway detection ---
    detected_pathways = _detect_pathways(text)
    if detected_pathways:
        pathway_score = max(get_pathway_prior(p) for p in detected_pathways)
        rationale.append(
            f"[pathway] detected={detected_pathways} "
            f"pathway_score={pathway_score:.2f} (max prior across detected)"
        )
    else:
        pathway_score = 0.0
        rationale.append("[pathway] No recognised pathways detected; pathway_score=0.0")

    # --- Study design ---
    tier_name, study_design_score, snippet = _detect_study_design(text)
    rationale.append(
        f"[study_design] tier={tier_name} score={study_design_score:.2f} "
        f"matched='{snippet}'"
    )

    # --- Contested biology ---
    contested_flags_full = detect_contested_claims(title, abstract, disease=disease)
    contested_flag_keys = [f.debate_name for f in contested_flags_full]
    for cf in contested_flags_full:
        rationale.append(
            f"[contested] debate='{cf.debate_name}' snippet='{cf.matched_snippet[:60]}' "
            f"— {len(cf.positions)} competing positions surfaced, not synthesized"
        )

    # --- Composite score ---
    overall_score = round(
        model_score * SCORE_WEIGHTS["model"]
        + pathway_score * SCORE_WEIGHTS["pathway"]
        + study_design_score * SCORE_WEIGHTS["study_design"],
        3,
    )
    rationale.append(
        f"[composite] overall={overall_score:.3f} = "
        f"model({model_score:.2f})*0.40 + "
        f"pathway({pathway_score:.2f})*0.30 + "
        f"study_design({study_design_score:.2f})*0.30"
    )

    logger.info(
        "PMID %s | overall=%.3f model=%.2f pathway=%.2f study_design=%.2f "
        "contested=%s warnings=%s",
        pmid, overall_score, model_score, pathway_score, study_design_score,
        contested_flag_keys, model_result.warnings,
    )

    return EvidenceQualityReport(
        pmid=pmid,
        title=title,
        model_score=model_score,
        pathway_score=pathway_score,
        study_design_score=study_design_score,
        detected_models=model_result.detected_models,
        detected_pathways=detected_pathways,
        study_design_tier=tier_name,
        contested_flags=contested_flag_keys,
        overall_score=overall_score,
        rationale=rationale,
        warnings=model_result.warnings,
    )
