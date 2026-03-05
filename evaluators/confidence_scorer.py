"""
confidence_scorer.py — Derive a confidence score from an EvidenceQualityReport.

Confidence differs from evidence quality: a high-quality paper (human biopsy,
central pathway) touching contested biology should score high on quality but LOW
on confidence, because making a definitive claim requires resolving a debate that
the field has not resolved.

The per-flag penalty reflects epistemic caution, not a judgment about the paper's
methods or rigor. Consumers of this score should log both quality and confidence
to make the distinction visible.
"""

import logging

from evaluators.evidence_quality import EvidenceQualityReport

logger = logging.getLogger(__name__)

# Three flags (e.g. M1/M2 + EMT + myofibroblast reversibility) reduce a perfect
# 1.0 quality score to 0.70 — still moderate confidence but clearly flagged.
CONTESTED_PENALTY_PER_FLAG: float = 0.10


def compute_confidence(report: EvidenceQualityReport) -> float:
    """
    Apply a per-flag penalty for contested biology to produce a confidence score.

    Confidence is floored at 0.0 and rounded to 3 decimal places.
    """
    penalty = len(report.contested_flags) * CONTESTED_PENALTY_PER_FLAG
    confidence = round(max(0.0, report.overall_score - penalty), 3)

    logger.info(
        "PMID %s | confidence=%.3f (quality=%.3f penalty=%.2f flags=%s)",
        report.pmid, confidence, report.overall_score, penalty, report.contested_flags,
    )

    return confidence
