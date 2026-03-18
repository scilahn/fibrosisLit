"""
claim_evaluator.py — Evaluate a biological claim using a two-vote system.

Vote 1 (deterministic): prior-based pathway support, model translational penalty, and
contested biology detection — fully local, no API calls. This vote answers "how well
does this claim align with IPF domain priors encoded in fibrosis_priors.py?"

Vote 2 (LLM): Claude API call with papers retrieved from ChromaDB as evidence context.
Retrieved papers are pre-scored by evidence_quality.evaluate_paper() before being
formatted into the prompt, so the LLM receives structured quality metadata rather than
raw abstracts.

Note: pipeline.embed.query() returns titles but not abstracts. evaluate_paper() is
therefore called with abstract="" — an accepted title-only limitation. The LLM
compensates via training knowledge about the cited papers' content. This is the reason
the LLM vote exists: it does not depend solely on metadata scores.

Adjudication rules (deterministic tier → verdict mapping first):
    WELL_SUPPORTED → SUPPORTED,  CONTESTED → CONTESTED,  OVERCLAIMED → UNSUPPORTED

    det=SUPPORTED  + llm=SUPPORTED  → SUPPORTED    (HIGH confidence)
    det=CONTESTED  + llm=CONTESTED  → CONTESTED    (HIGH confidence)
    det=UNSUPPORTED + llm=UNSUPPORTED → UNSUPPORTED (HIGH confidence)
    det=UNSUPPORTED + llm=CONTESTED → CONTESTED    (HIGH — LLM found genuine debate)
    Any other combination           → LOW_CONFIDENCE (LOW — flag both verdicts)

INSUFFICIENT_EVIDENCE from LLM → LOW_CONFIDENCE (retrieval gap, not a biological verdict).

The authoritative output is `result.verdict`. The original `result.tier` (deterministic tier)
is preserved unchanged for benchmark_runner.py backward compatibility.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

from domain_knowledge.fibrosis_priors import (
    get_pathway_prior,
    get_model,
    has_flag,
)
from evaluators.contradiction_detector import ContestedFlag, detect_contested_claims
from evaluators.evidence_quality import PATHWAY_PATTERNS, evaluate_paper
from evaluators.confidence_scorer import compute_confidence
from pipeline.embed import query as chroma_query

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier classification thresholds (deterministic vote)
# ---------------------------------------------------------------------------

WELL_SUPPORTED_THRESHOLD: float = 0.70
OVERCLAIMED_THRESHOLD: float = 0.35

# ---------------------------------------------------------------------------
# Model penalty magnitudes
# ---------------------------------------------------------------------------

POOR_TRANSLATION_PENALTY: float = 0.40
LOW_TRANSLATABILITY_PENALTY: float = 0.25

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

LLM_MODEL: str = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS: int = 1024
LLM_TEMPERATURE: float = 0.0   # deterministic for auditable scoring
N_EVIDENCE_DEFAULT: int = 8

# ---------------------------------------------------------------------------
# Adjudication — normalize deterministic tier to verdict vocabulary
# ---------------------------------------------------------------------------

_DET_TIER_NORM: dict[str, str] = {
    "WELL_SUPPORTED": "SUPPORTED",
    "CONTESTED":      "CONTESTED",
    "OVERCLAIMED":    "UNSUPPORTED",
}

# ---------------------------------------------------------------------------
# Cell-type and population entity patterns
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
    # markers, disease-specific to human IPF (Kathiriya 2023). Existence is well-supported;
    # mechanistic interpretation (partial EMT vs. aberrant differentiation) is contested.
    "aberrant_basaloid": [
        r"\baberrant basaloid\b",
        r"\bKRT17\b",
    ],
}

CLAIM_ENTITY_PRIORS: dict[str, float] = {
    "spp1_macrophage":    0.85,
    "myofibroblast_axis": 0.80,
    "aberrant_basaloid":  0.75,
}

# ---------------------------------------------------------------------------
# Model mention patterns
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
# LLM system prompt
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT: str = """\
You are an expert in IPF (idiopathic pulmonary fibrosis) biology with knowledge of \
translational medicine and clinical trial history. Evaluate whether a biological claim \
is SUPPORTED, CONTESTED, or UNSUPPORTED based on retrieved evidence.

Critical rules:
1. Human biopsy / single-cell atlas / clinical trial data outweighs animal model data. \
Weight evidence accordingly.
2. Bleomycin acute mouse results are weak IPF evidence — flag claims resting primarily \
on them as having poor translational support.
3. The M1/M2 macrophage framework is contested in fibrosis. Do NOT return SUPPORTED for \
claims invoking M1/M2 polarization without flagging this as a contested framework.
4. Myofibroblast reversibility is an active debate. Surface both positions; do NOT resolve it.
5. Clinical trial failures outweigh preclinical mechanistic papers — cite them if relevant.
6. Never synthesize contradictory evidence into a confident conclusion. If genuine \
disagreement exists in the evidence, return CONTESTED.
7. INSUFFICIENT_EVIDENCE is correct when the retrieved papers do not meaningfully address \
the claim — not when papers disagree.

Return ONLY valid JSON matching this exact schema — no text outside the JSON object:
{
  "verdict": "SUPPORTED" | "CONTESTED" | "UNSUPPORTED" | "INSUFFICIENT_EVIDENCE",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-4 sentences explaining the verdict, citing PMIDs>",
  "supporting_pmids": ["<pmid>", ...],
  "contesting_pmids": ["<pmid>", ...],
  "paper_stances": {"<pmid>": "supporting" | "contesting" | "neutral", ...}
}\
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClaimEvaluationResult:
    """
    Two-vote evaluation of a single biological claim.

    The authoritative output is `verdict` (adjudicated from both votes).
    The original `tier` field (deterministic tier: WELL_SUPPORTED|CONTESTED|OVERCLAIMED)
    is preserved unchanged for benchmark_runner.py backward compatibility.

    `references` is the list of papers retrieved from ChromaDB that were used as
    LLM evidence context, each enriched with pre-scored quality metrics and the
    LLM's assessed stance (supporting/contesting/neutral).

    When llm_verdict is None or "LLM_ERROR", verdict will be "LOW_CONFIDENCE" and
    verdict_confidence will be "LOW". The deterministic tier remains valid in all cases.
    """
    claim: str
    disease: str

    # ----- Deterministic vote (all fields from original evaluate_claim) -----
    detected_pathways: list[str]
    detected_model_mentions: list[str]
    contested_flags: list[ContestedFlag]   # full objects with positions
    pathway_support_score: float
    model_penalty: float
    prior_support_score: float
    tier: str                              # "WELL_SUPPORTED" | "CONTESTED" | "OVERCLAIMED"
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ----- LLM vote -----
    llm_verdict: str | None = None         # SUPPORTED|CONTESTED|UNSUPPORTED|INSUFFICIENT_EVIDENCE|LLM_ERROR
    llm_confidence: float | None = None
    llm_reasoning: str | None = None
    llm_raw_response: str | None = None    # raw JSON string for audit

    # ----- Adjudicated verdict -----
    verdict: str = "PENDING"              # SUPPORTED|CONTESTED|UNSUPPORTED|LOW_CONFIDENCE
    verdict_confidence: str = "LOW"       # HIGH|LOW
    verdict_rationale: str = ""

    # ----- Retrieved references -----
    references: list[dict] = field(default_factory=list)
    # Each reference dict keys:
    #   pmid, title, journal, pub_date, doi, authors, distance   ← from embed.query()
    #   overall_score, study_design_tier, detected_pathways,      ← from evaluate_paper()
    #   detected_models, contested_flags, model_warnings,
    #   confidence                                                 ← from compute_confidence()
    #   llm_stance   "supporting"|"contesting"|"neutral"          ← from LLM paper_stances


# ---------------------------------------------------------------------------
# Deterministic vote helpers (unchanged from original evaluate_claim body)
# ---------------------------------------------------------------------------

def _detect_claim_pathways(claim: str) -> list[tuple[str, float]]:
    """Return (key, prior_score) pairs for all pathways and entities detected in claim."""
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
    """Return canonical model keys for preclinical model systems mentioned in claim."""
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
    """Compute maximum model penalty across detected model mentions."""
    if not detected_model_mentions:
        return 0.0
    max_penalty: float = 0.0
    for model_key in detected_model_mentions:
        model = get_model(model_key)
        if model is None:
            rationale.append(f"[model] {model_key}: unrecognized model key, skipping penalty")
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
            rationale.append(f"[model] {model_key}: score={model.score:.2f} → no penalty applied")
        max_penalty = max(max_penalty, penalty)
    return max_penalty


def _classify_tier(
    prior_support_score: float,
    contested_flags: list[ContestedFlag],
    model_penalty: float,
) -> str:
    """Classify claim into WELL_SUPPORTED, CONTESTED, or OVERCLAIMED."""
    if contested_flags:
        return "CONTESTED"
    if prior_support_score >= WELL_SUPPORTED_THRESHOLD:
        return "WELL_SUPPORTED" if model_penalty == 0.0 else "OVERCLAIMED"
    if prior_support_score < OVERCLAIMED_THRESHOLD:
        return "OVERCLAIMED"
    return "CONTESTED"


def _deterministic_vote(claim: str, disease: str) -> ClaimEvaluationResult:
    """
    Run the deterministic prior-based evaluation.

    Identical logic to the original evaluate_claim(). Returns a ClaimEvaluationResult
    with all deterministic fields populated; LLM fields remain at defaults.
    """
    rationale: list[str] = []
    warnings: list[str] = []

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

    detected_model_mentions = _detect_model_mentions(claim)
    model_penalty = _compute_model_penalty(detected_model_mentions, rationale, warnings)
    if not detected_model_mentions:
        rationale.append("[model] no model system mentions detected → penalty=0.00")

    contested_flags = detect_contested_claims(claim, "", disease)
    if contested_flags:
        for flag in contested_flags:
            rationale.append(
                f"[contested] {flag.debate_name}: matched='{flag.matched_snippet[:60]}' "
                f"({len(flag.positions)} competing positions — not resolved)"
            )
    else:
        rationale.append("[contested] no contested biology detected")

    prior_support_score = pathway_support_score * (1.0 - model_penalty)
    rationale.append(
        f"[prior_support] {pathway_support_score:.2f} × (1.0 - {model_penalty:.2f}) "
        f"= {prior_support_score:.3f}"
    )

    tier = _classify_tier(prior_support_score, contested_flags, model_penalty)
    rationale.append(
        f"[tier] {tier}: prior_support={prior_support_score:.3f} "
        f"model_penalty={model_penalty:.2f} "
        f"contested_flags={[f.debate_name for f in contested_flags]}"
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


# ---------------------------------------------------------------------------
# LLM vote helpers
# ---------------------------------------------------------------------------

def _score_retrieved_papers(papers: list[dict], disease: str) -> list[dict]:
    """
    Score each retrieved paper with evaluate_paper() + compute_confidence().

    Because embed.query() returns no abstract, evaluate_paper() is called with
    abstract="" — title-only scoring. This is an accepted limitation documented
    in the module header.

    Returns a list of reference dicts with all embed.query() fields plus
    quality scoring fields, with llm_stance defaulting to "neutral".
    """
    scored: list[dict] = []
    for p in papers:
        paper_dict = {
            "pmid":     p.get("pmid", ""),
            "title":    p.get("title", ""),
            "abstract": "",   # not available from embed.query()
            "journal":  p.get("journal", ""),
            "pub_date": p.get("pub_date", ""),
        }
        try:
            report = evaluate_paper(paper_dict, disease=disease)
            conf = compute_confidence(report)
        except Exception as exc:
            logger.warning("evaluate_paper failed for pmid=%s: %s", p.get("pmid"), exc)
            report = None
            conf = 0.0

        scored.append({
            # From embed.query()
            "pmid":             p.get("pmid", ""),
            "title":            p.get("title", ""),
            "journal":          p.get("journal", ""),
            "pub_date":         p.get("pub_date", ""),
            "doi":              p.get("doi", ""),
            "authors":          p.get("authors", ""),
            "distance":         p.get("distance", 1.0),
            # From evaluate_paper() — or defaults if scoring failed
            "overall_score":    report.overall_score    if report else 0.0,
            "study_design_tier": report.study_design_tier if report else "unknown",
            "detected_pathways": report.detected_pathways if report else [],
            "detected_models":  report.detected_models  if report else [],
            "contested_flags":  report.contested_flags  if report else [],
            "model_warnings":   report.warnings         if report else [],
            "confidence":       conf,
            # LLM stance assigned later by _assign_stances_to_references()
            "llm_stance":       "neutral",
        })
    return scored


def _format_evidence_block(scored_papers: list[dict]) -> str:
    """Format scored papers into the evidence context block for the LLM prompt."""
    lines: list[str] = []
    for i, p in enumerate(scored_papers, start=1):
        pathways_str   = ", ".join(p["detected_pathways"])  or "none detected"
        models_str     = ", ".join(p["detected_models"])    or "none"
        flags_str      = ", ".join(p["contested_flags"])    or "none"
        warnings_str   = ", ".join(p["model_warnings"])     or "none"
        lines.append(
            f"[{i}] PMID {p['pmid']} | {p['title']}\n"
            f"    Journal: {p['journal']} ({p['pub_date']})\n"
            f"    Evidence score: {p['overall_score']:.2f} | "
            f"Study design: {p['study_design_tier']} | "
            f"Retrieval distance: {p['distance']:.4f}\n"
            f"    Pathways: {pathways_str}\n"
            f"    Model flags: {warnings_str}\n"
            f"    Contested flags: {flags_str}"
        )
    return "\n\n".join(lines)


def _build_llm_messages(claim: str, evidence_block: str) -> list[dict]:
    """Return [system, user] message dicts for the Claude API call."""
    user_content = (
        f"Claim: {claim}\n\n"
        f"Retrieved evidence papers (pre-scored by deterministic evaluator):\n"
        f"{evidence_block}\n\n"
        f"Evaluate the claim based on the evidence above and your domain knowledge."
    )
    return [
        {"role": "user", "content": user_content},
    ]


def _parse_llm_response(raw_json: str) -> dict:
    """
    Parse and validate the LLM JSON response.

    Raises ValueError if required fields are missing or values are out of range.
    """
    # Strip markdown code fence if the LLM wraps its response (e.g. ```json ... ```)
    text = raw_json.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    valid_verdicts = {"SUPPORTED", "CONTESTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"}
    if data.get("verdict") not in valid_verdicts:
        raise ValueError(f"Invalid verdict: {data.get('verdict')!r}")

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        raise ValueError(f"Invalid confidence: {confidence!r}")

    if not isinstance(data.get("reasoning"), str) or not data["reasoning"].strip():
        raise ValueError("Missing or empty 'reasoning' field")

    if not isinstance(data.get("supporting_pmids"), list):
        data["supporting_pmids"] = []
    if not isinstance(data.get("contesting_pmids"), list):
        data["contesting_pmids"] = []
    if not isinstance(data.get("paper_stances"), dict):
        data["paper_stances"] = {}

    return data


def _assign_stances_to_references(
    references: list[dict],
    llm_raw: str | None,
) -> list[dict]:
    """
    Merge paper_stances from the parsed LLM JSON into each reference dict.

    Defaults to "neutral" for any pmid not in paper_stances, or when llm_raw is None.
    """
    stances: dict[str, str] = {}
    if llm_raw:
        try:
            parsed = json.loads(llm_raw)
            stances = parsed.get("paper_stances", {})
        except (json.JSONDecodeError, AttributeError):
            pass

    valid_stances = {"supporting", "contesting", "neutral"}
    for ref in references:
        raw_stance = stances.get(str(ref["pmid"]), "neutral")
        ref["llm_stance"] = raw_stance if raw_stance in valid_stances else "neutral"
    return references


def _llm_vote(
    claim: str,
    disease: str,
    n_evidence: int,
) -> tuple[str, float | None, str | None, str | None, list[dict]]:
    """
    Retrieve evidence from ChromaDB, score papers, and call Claude API.

    Returns (llm_verdict, confidence, reasoning, raw_response, references).

    On any failure returns ("LLM_ERROR", None, error_string, None, []) so
    adjudication can produce LOW_CONFIDENCE without crashing.
    """
    # 1. Retrieve papers from ChromaDB
    try:
        papers = chroma_query(claim, n_results=n_evidence)
    except Exception as exc:
        logger.warning("ChromaDB query failed: %s", exc)
        return ("LLM_ERROR", None, f"ChromaDB query failed: {exc}", None, [])

    if not papers:
        logger.warning("ChromaDB returned no results for claim: '%s...'", claim[:60])
        return ("INSUFFICIENT_EVIDENCE", 0.0, "No papers retrieved from ChromaDB.", None, [])

    # 2. Score retrieved papers
    references = _score_retrieved_papers(papers, disease)

    # 3. Format evidence block and build messages
    evidence_block = _format_evidence_block(references)
    messages = _build_llm_messages(claim, evidence_block)

    # 4. Call Claude API — client instantiated here to avoid import-time failure
    #    when ANTHROPIC_API_KEY is not set (e.g. in deterministic-only test runs)
    raw_response: str | None = None
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=_LLM_SYSTEM_PROMPT,
            messages=messages,
        )
        raw_response = response.content[0].text
    except Exception as exc:
        logger.warning("Anthropic API call failed: %s", exc)
        return ("LLM_ERROR", None, f"API call failed: {exc}", None, references)

    # 5. Parse and validate
    try:
        parsed = _parse_llm_response(raw_response)
    except ValueError as exc:
        logger.warning("LLM response parse failed: %s | raw=%s", exc, raw_response[:200])
        return ("LLM_ERROR", None, f"Parse error: {exc}", raw_response, references)

    return (
        parsed["verdict"],
        float(parsed["confidence"]),
        parsed["reasoning"],
        raw_response,
        references,
    )


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------

def _adjudicate(det_tier: str, llm_verdict: str | None) -> tuple[str, str, str]:
    """
    Apply the two-vote adjudication rules table.

    Returns (verdict, verdict_confidence, verdict_rationale).
    """
    if llm_verdict in (None, "LLM_ERROR"):
        return (
            "LOW_CONFIDENCE",
            "LOW",
            f"LLM vote unavailable (llm_verdict={llm_verdict!r}); "
            f"deterministic tier={det_tier}.",
        )

    det_norm = _DET_TIER_NORM.get(det_tier, "UNKNOWN")

    if det_norm == "SUPPORTED" and llm_verdict == "SUPPORTED":
        return ("SUPPORTED", "HIGH", "Both votes SUPPORTED — high confidence.")

    if det_norm == "CONTESTED" and llm_verdict == "CONTESTED":
        return ("CONTESTED", "HIGH", "Both votes CONTESTED — high confidence surfacing debate.")

    if det_norm == "UNSUPPORTED" and llm_verdict in ("UNSUPPORTED", "INSUFFICIENT_EVIDENCE"):
        note = (
            "Deterministic OVERCLAIMED aligned with LLM UNSUPPORTED — high confidence rejection."
            if det_tier == "OVERCLAIMED"
            else "Both votes UNSUPPORTED — high confidence."
        )
        return ("UNSUPPORTED", "HIGH", note)

    if det_norm == "UNSUPPORTED" and llm_verdict == "CONTESTED":
        return (
            "CONTESTED",
            "HIGH",
            "Deterministic OVERCLAIMED/UNSUPPORTED but LLM found genuine debate — "
            "CONTESTED verdict (LLM debate detection takes precedence).",
        )

    # All other combinations
    return (
        "LOW_CONFIDENCE",
        "LOW",
        f"Votes diverge: deterministic={det_tier} ({det_norm}), "
        f"llm={llm_verdict} — insufficient agreement to classify.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_claim(
    claim: str,
    disease: str = "ipf",
    n_evidence: int = N_EVIDENCE_DEFAULT,
) -> ClaimEvaluationResult:
    """
    Evaluate a biological claim using a two-vote system.

    Vote 1 (deterministic): prior-based pathway support, model penalty, and contested
    biology detection — fully local.

    Vote 2 (LLM): Claude API call with ChromaDB-retrieved papers as evidence context.
    Papers are pre-scored by evaluate_paper() before being sent to Claude.

    The authoritative output is `result.verdict`. The original `result.tier`
    (deterministic tier) is preserved for benchmark_runner.py backward compatibility.

    Args:
        claim:      Biological claim as a declarative string.
        disease:    Target disease context. Currently only "ipf" is supported.
        n_evidence: Number of papers to retrieve from ChromaDB for LLM context.
                    Pass 0 to skip the LLM vote (verdict will be LOW_CONFIDENCE).

    Returns:
        ClaimEvaluationResult with both votes, adjudicated verdict, and references.
    """
    # 1. Deterministic vote
    det_result = _deterministic_vote(claim, disease)

    # 2. LLM vote
    if n_evidence > 0:
        llm_verdict, llm_conf, llm_reasoning, llm_raw, references = _llm_vote(
            claim, disease, n_evidence
        )
    else:
        llm_verdict  = "LLM_ERROR"
        llm_conf     = None
        llm_reasoning = "LLM vote skipped (n_evidence=0)."
        llm_raw      = None
        references   = []

    # 3. Adjudicate
    verdict, verdict_confidence, verdict_rationale = _adjudicate(det_result.tier, llm_verdict)

    # 4. Assign LLM stances back onto references
    references = _assign_stances_to_references(references, llm_raw)

    logger.info(
        "Claim '%s...' | det=%s llm=%s → verdict=%s (%s)",
        claim[:60], det_result.tier, llm_verdict, verdict, verdict_confidence,
    )

    return ClaimEvaluationResult(
        claim=det_result.claim,
        disease=det_result.disease,
        detected_pathways=det_result.detected_pathways,
        detected_model_mentions=det_result.detected_model_mentions,
        contested_flags=det_result.contested_flags,
        pathway_support_score=det_result.pathway_support_score,
        model_penalty=det_result.model_penalty,
        prior_support_score=det_result.prior_support_score,
        tier=det_result.tier,
        rationale=det_result.rationale,
        warnings=det_result.warnings,
        llm_verdict=llm_verdict,
        llm_confidence=llm_conf,
        llm_reasoning=llm_reasoning,
        llm_raw_response=llm_raw,
        verdict=verdict,
        verdict_confidence=verdict_confidence,
        verdict_rationale=verdict_rationale,
        references=references,
    )
