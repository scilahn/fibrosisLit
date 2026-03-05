"""
contradiction_detector.py — Detect contested biology claims in paper text.

Flags papers that touch known unresolved debates in IPF biology, importing
positions from fibrosis_priors.CONTESTED_BIOLOGY. The detector does NOT
determine which side of the debate a paper takes — that would require synthesis.
Callers must surface all positions as alternatives, never collapsing them.

This enforces the project's core constraint: contested biology is surfaced,
not resolved.
"""

import logging
import re
from dataclasses import dataclass, field

from domain_knowledge.fibrosis_priors import (
    CONTESTED_BIOLOGY,
    get_contested,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection patterns — each key maps to contested biology in CONTESTED_BIOLOGY.
# Multiple patterns per debate increase recall across different terminologies.
# ---------------------------------------------------------------------------

CONTESTED_PATTERNS: dict[str, list[str]] = {
    "macrophage_polarization": [
        r"M1.{0,5}M2",
        r"\bM[12]\b.{0,10}macrophage",
        r"alternatively activated macrophage",
        r"classically activated macrophage",
        r"macrophage polarization",
    ],
    "myofibroblast_reversibility": [
        r"myofibroblast.{0,30}(dedifferentiat|reversion|reversib|plasticity)",
        r"(dedifferentiat|reversion).{0,30}myofibroblast",
    ],
    "epithelial_mesenchymal_transition": [
        r"\bEMT\b",
        r"epithelial.mesenchymal transition",
        r"epithelial.to.mesenchymal",
    ],
    "fibrosis_resolution_capacity": [
        r"fibrosis.{0,15}(resolv|resolution|regression|reversal|reversib)",
        r"(resolv|reversal).{0,15}fibrosis",
    ],
}


@dataclass
class ContestedFlag:
    """
    A single contested-biology flag triggered by paper text.

    matched_snippet is retained for the audit log so reviewers can verify
    that the detection was appropriate and not a false positive.

    positions lists the competing stances from fibrosis_priors — they are
    presented as-is and must never be synthesized into a single conclusion.
    """
    debate_name: str        # key into CONTESTED_BIOLOGY
    matched_snippet: str    # text fragment that triggered detection
    debate: str             # human-readable statement of the disagreement
    positions: list[str]    # competing positions from fibrosis_priors (NOT synthesized)


def detect_contested_claims(
    title: str,
    abstract: str,
    disease: str = "ipf",   # reserved for future PSC priors
) -> list[ContestedFlag]:
    """
    Scan paper text for claims touching contested IPF biology.

    Returns flags for each contested debate touched. Does NOT attempt to
    determine which side the paper takes — that would require synthesis.
    The caller must surface all positions from fibrosis_priors.CONTESTED_BIOLOGY
    as alternatives, never collapsing them into a single conclusion.

    Args:
        title:    Paper title.
        abstract: Paper abstract.
        disease:  Target disease context. Currently only "ipf" is supported.

    Returns:
        List of ContestedFlag, one per triggered debate (deduplicated).
    """
    text = f"{title} {abstract}"
    flags: list[ContestedFlag] = []
    triggered: set[str] = set()

    for debate_key, patterns in CONTESTED_PATTERNS.items():
        if debate_key in triggered:
            continue

        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                snippet = m.group(0)
                contested = get_contested(debate_key)
                if contested is None:
                    logger.warning(
                        "Pattern matched debate_key '%s' but no entry in CONTESTED_BIOLOGY.",
                        debate_key,
                    )
                    continue

                flags.append(ContestedFlag(
                    debate_name=debate_key,
                    matched_snippet=snippet[:120],
                    debate=contested.debate,
                    positions=list(contested.positions),
                ))
                triggered.add(debate_key)
                logger.info(
                    "Contested biology detected: '%s' via snippet '%s'",
                    debate_key, snippet[:60],
                )
                break  # one flag per debate is sufficient

    return flags
