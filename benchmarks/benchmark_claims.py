"""
benchmark_claims.py — FibrosisLit evaluation benchmark test set.

25 biological claims across three tiers designed to stress-test domain-specific
evidence evaluation against general retrieval tools (Elicit, Consensus,
Semantic Scholar).

Tier structure:
    WELL_SUPPORTED  — Claims with strong human evidence consensus. All tools
                      should get these right. Concordance validates retrieval.
    CONTESTED       — Claims touching genuinely unresolved debates. FibrosisLit
                      should surface competing positions; general tools will tend
                      to synthesize into false confidence.
    OVERCLAIMED     — Plausible-sounding claims that are wrong, model-limited,
                      or contradicted by higher-quality evidence. Requires domain
                      knowledge to detect; naive retrieval will find supporting
                      papers because they exist.

Scoring rubric fields (used by human scorer during benchmark runs):
    expected_verdict:       SUPPORTED | CONTESTED | UNSUPPORTED | INSUFFICIENT_EVIDENCE
    domain_flags:           Priors a correct evaluator must apply
    failure_mode:           How a general tool is likely to fail on this claim
    scoring_notes:          What a correct response must include to pass

Domain knowledge encoded by Richard Ahn, PhD.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    WELL_SUPPORTED = "well_supported"
    CONTESTED      = "contested"
    OVERCLAIMED    = "overclaimed"


class ExpectedVerdict(str, Enum):
    SUPPORTED             = "SUPPORTED"
    CONTESTED             = "CONTESTED"
    UNSUPPORTED           = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class BenchmarkClaim:
    """
    A single benchmark claim with scoring metadata.

    claim_id:        Unique identifier for tracking across benchmark runs.
    claim:           The biological claim to evaluate, stated as a declarative
                     sentence. Phrased to match how a scientist would query a
                     tool, not as a yes/no question.
    tier:            Tier classification (see module docstring).
    expected_verdict: The correct verdict a domain-expert evaluator would assign.
    domain_flags:    List of domain priors that must be applied for a correct
                     evaluation. Used to assess whether tools apply fibrosis-
                     specific reasoning or treat all evidence as equivalent.
    failure_mode:    The specific way a general retrieval tool is expected to
                     fail on this claim. Defines what FibrosisLit must improve on.
    scoring_notes:   Criteria a response must meet to be scored as correct.
                     Used during human scoring of benchmark outputs.
    """
    claim_id: str
    claim: str
    tier: Tier
    expected_verdict: ExpectedVerdict
    domain_flags: list[str]
    failure_mode: str
    scoring_notes: str


BENCHMARK_CLAIMS: list[BenchmarkClaim] = [

    # -----------------------------------------------------------------------
    # TIER 1 — WELL-SUPPORTED
    # Strong human evidence consensus. All tools should get these right.
    # FibrosisLit concordance here validates retrieval and scoring layers.
    # -----------------------------------------------------------------------

    BenchmarkClaim(
        claim_id="WS-01",
        claim="SPP1+ macrophages promote myofibroblast activation in IPF lung.",
        tier=Tier.WELL_SUPPORTED,
        expected_verdict=ExpectedVerdict.SUPPORTED,
        domain_flags=["human_biopsy_scrnaseq", "spp1_macrophage_axis"],
        failure_mode="None expected — general tools should retrieve supporting scRNA-seq atlas papers.",
        scoring_notes=(
            "Correct response cites human single-cell data (Adams, Habermann, or equivalent). "
            "Bonus: notes that SPP1hi macrophage is a defined population in IPF atlases, "
            "distinct from M1/M2 classification."
        ),
    ),

    BenchmarkClaim(
        claim_id="WS-02",
        claim="αvβ6 integrin expression is upregulated in IPF epithelial cells relative to healthy lung.",
        tier=Tier.WELL_SUPPORTED,
        expected_verdict=ExpectedVerdict.SUPPORTED,
        domain_flags=["integrin_avb6", "human_biopsy_scrnaseq"],
        failure_mode="None expected.",
        scoring_notes=(
            "Correct response references human biopsy or IHC evidence for ITGB6 upregulation "
            "in IPF vs. control lung. Clinical context (bexotegrast program) is a bonus but "
            "not required for a passing score."
        ),
    ),

    BenchmarkClaim(
        claim_id="WS-03",
        claim="TGF-β1 drives fibroblast-to-myofibroblast differentiation via SMAD3 phosphorylation in IPF.",
        tier=Tier.WELL_SUPPORTED,
        expected_verdict=ExpectedVerdict.SUPPORTED,
        domain_flags=["tgfb_smad", "central_pathway"],
        failure_mode="None expected — extensively documented in human and preclinical systems.",
        scoring_notes=(
            "Must reference SMAD3 specifically (not just SMAD2/3 generically) as the more potent "
            "fibrosis effector. Acceptable to note non-canonical branches as amplifiers."
        ),
    ),

    BenchmarkClaim(
        claim_id="WS-04",
        claim="Nerandomilast (BI 1015550) met its primary endpoint in the Phase 3 FIBRONEER-IPF trial.",
        tier=Tier.WELL_SUPPORTED,
        expected_verdict=ExpectedVerdict.SUPPORTED,
        domain_flags=["pde4", "phase_3_clinical_trial"],
        failure_mode="May fail if tool's knowledge cutoff predates FIBRONEER-IPF results.",
        scoring_notes=(
            "Must confirm Phase 3 success and FDA approval. If tool returns uncertainty or "
            "absence of evidence, that is a retrieval/currency failure, not an evaluation failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="WS-05",
        claim="CTHRC1+ fibroblasts are enriched in IPF relative to non-fibrotic lung in single-cell atlases.",
        tier=Tier.WELL_SUPPORTED,
        expected_verdict=ExpectedVerdict.SUPPORTED,
        domain_flags=["human_biopsy_scrnaseq", "myofibroblast_populations"],
        failure_mode="None expected for tools with access to Adams 2020 or Habermann 2019.",
        scoring_notes=(
            "Must reference human scRNA-seq data. CTHRC1 as a marker of pathological "
            "myofibroblast subpopulation (not generic fibroblast activation) is the key point."
        ),
    ),

    BenchmarkClaim(
        claim_id="WS-06",
        claim="Bexotegrast inhibits both αvβ6 and αvβ1 integrins to block latent TGF-β activation.",
        tier=Tier.WELL_SUPPORTED,
        expected_verdict=ExpectedVerdict.SUPPORTED,
        domain_flags=["integrin_avb6", "tgfb_smad", "phase_2_clinical_trial"],
        failure_mode="None expected.",
        scoring_notes=(
            "Must specify dual inhibition (αvβ6 epithelial + αvβ1 mesenchymal compartments). "
            "Single-integrin framing is incomplete and should not score as fully correct."
        ),
    ),

    BenchmarkClaim(
        claim_id="WS-07",
        claim="Aberrant basaloid cells co-expressing KRT17 and mesenchymal markers are present in IPF lung.",
        tier=Tier.WELL_SUPPORTED,
        expected_verdict=ExpectedVerdict.SUPPORTED,
        domain_flags=["human_biopsy_scrnaseq", "aberrant_basaloid"],
        failure_mode="None expected for tools with access to Kathiriya 2023 or equivalent.",
        scoring_notes=(
            "Must reference human single-cell data identifying this population. "
            "Note: whether this constitutes classical EMT is contested — a correct response "
            "distinguishes the cell population (well-supported) from its mechanistic "
            "interpretation (contested)."
        ),
    ),

    BenchmarkClaim(
        claim_id="WS-08",
        claim="IPF fibroblasts show elevated PDE4 activity and are hyperresponsive to TGF-β relative to normal lung fibroblasts.",
        tier=Tier.WELL_SUPPORTED,
        expected_verdict=ExpectedVerdict.SUPPORTED,
        domain_flags=["pde4", "primary_human_fibroblasts"],
        failure_mode="None expected.",
        scoring_notes=(
            "Correct response should note human primary fibroblast evidence, not cell line data. "
            "PDE4B subtype specificity (vs. PDE4D) is a bonus."
        ),
    ),

    # -----------------------------------------------------------------------
    # TIER 2 — CONTESTED
    # Genuinely unresolved debates in the IPF field. FibrosisLit must surface
    # competing positions; general tools will tend to return false confidence.
    # -----------------------------------------------------------------------

    BenchmarkClaim(
        claim_id="CT-01",
        claim="Myofibroblasts can dedifferentiate under reduced TGF-β signaling conditions in IPF.",
        tier=Tier.CONTESTED,
        expected_verdict=ExpectedVerdict.CONTESTED,
        domain_flags=["myofibroblast_reversibility", "contested_biology"],
        failure_mode=(
            "General tools will retrieve dedifferentiation papers and return SUPPORTED, "
            "missing the countervailing terminal differentiation literature and the "
            "in vivo relevance question."
        ),
        scoring_notes=(
            "Must surface all three positions from fibrosis_priors.CONTESTED_BIOLOGY: "
            "(1) terminal differentiation, (2) in vitro plasticity demonstrated, "
            "(3) in vivo relevance unclear. Resolving to either SUPPORTED or UNSUPPORTED "
            "without flagging the debate is a failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="CT-02",
        claim="M2 macrophage polarization is the primary driver of fibrosis progression in IPF.",
        tier=Tier.CONTESTED,
        expected_verdict=ExpectedVerdict.CONTESTED,
        domain_flags=["macrophage_polarization", "contested_biology", "spp1_macrophage_axis"],
        failure_mode=(
            "General tools will retrieve M2/TGF-β papers and return SUPPORTED. "
            "They will not flag that single-cell data have replaced M1/M2 with "
            "functionally distinct populations that do not map onto this binary."
        ),
        scoring_notes=(
            "Must flag M1/M2 framework as contested in fibrosis context. "
            "Must note SPP1hi population as the current leading macrophage candidate "
            "from scRNA-seq data. Returning SUPPORTED without these flags is a failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="CT-03",
        claim="Epithelial-mesenchymal transition contributes substantially to the myofibroblast pool in human IPF lung.",
        tier=Tier.CONTESTED,
        expected_verdict=ExpectedVerdict.CONTESTED,
        domain_flags=["epithelial_mesenchymal_transition", "contested_biology"],
        failure_mode=(
            "General tools will retrieve EMT-supporting papers and return SUPPORTED, "
            "missing lineage tracing evidence against full EMT and the partial EMT / "
            "aberrant basaloid cell nuance."
        ),
        scoring_notes=(
            "Must surface the lineage tracing counterevidence. Must distinguish "
            "full EMT (contested) from partial EMT / hybrid E/M states (better supported). "
            "Aberrant basaloid cells as a disease-specific partial EMT population should be noted."
        ),
    ),

    BenchmarkClaim(
        claim_id="CT-04",
        claim="IL-13 is a therapeutically actionable driver of IPF progression.",
        tier=Tier.CONTESTED,
        expected_verdict=ExpectedVerdict.CONTESTED,
        domain_flags=["il13_tslp", "clinical_trial_failure", "contested_biology"],
        failure_mode=(
            "General tools will retrieve IL-13 biology papers and return SUPPORTED. "
            "They will not surface the tralokinumab and lebrikizumab Phase 2 failures "
            "that contradict the preclinical hypothesis."
        ),
        scoring_notes=(
            "Must cite failed anti-IL-13 trials (tralokinumab, lebrikizumab) as "
            "contradicting evidence. Biology supporting IL-13 role is real; clinical "
            "failure is also real. A correct response holds both. SUPPORTED without "
            "trial failures is a scoring failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="CT-05",
        claim="Established pulmonary fibrosis in humans has meaningful capacity to resolve.",
        tier=Tier.CONTESTED,
        expected_verdict=ExpectedVerdict.CONTESTED,
        domain_flags=["fibrosis_resolution_capacity", "contested_biology"],
        failure_mode=(
            "General tools may return UNSUPPORTED (citing irreversibility consensus) "
            "or SUPPORTED (citing animal resolution models). Neither captures the "
            "stabilization-vs-resolution distinction or the organ-specificity debate."
        ),
        scoring_notes=(
            "Must distinguish fibrotic regression from disease stabilization. "
            "Must note that lung differs from liver in ECM composition and "
            "epithelial progenitor biology. Must surface animal model vs. human "
            "IPF evidence asymmetry."
        ),
    ),

    BenchmarkClaim(
        claim_id="CT-06",
        claim="The bleomycin mouse model faithfully recapitulates IPF disease progression.",
        tier=Tier.CONTESTED,
        expected_verdict=ExpectedVerdict.CONTESTED,
        domain_flags=["bleomycin_mouse_acute", "poor_ipf_translation", "frequently_overcited"],
        failure_mode=(
            "General tools will retrieve bleomycin methodology papers and return "
            "a mixed or SUPPORTED verdict, failing to weight the clinical translation "
            "failure history appropriately."
        ),
        scoring_notes=(
            "Must note acute bleomycin is inflammation-driven and self-limiting. "
            "Must note history of translation failures from bleomycin to IPF. "
            "Chronic bleomycin as a modest improvement is acceptable to note but "
            "does not rescue the general claim. SUPPORTED is a failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="CT-07",
        claim="ATX inhibition will slow FVC decline in IPF patients based on current evidence.",
        tier=Tier.CONTESTED,
        expected_verdict=ExpectedVerdict.CONTESTED,
        domain_flags=["autotaxin", "insufficient_clinical_validation"],
        failure_mode=(
            "General tools will retrieve ATX biology and Phase 1 LPA suppression data "
            "and return SUPPORTED, conflating pharmacodynamic proof-of-concept with "
            "clinical efficacy evidence."
        ),
        scoring_notes=(
            "Must distinguish LPA suppression (shown) from FVC slowing (not yet shown). "
            "Phase 2a is ongoing; no efficacy data available. Preclinical support is "
            "real but insufficient to support the clinical efficacy claim. "
            "SUPPORTED is a failure."
        ),
    ),

    # -----------------------------------------------------------------------
    # TIER 3 — OVERCLAIMED
    # Claims that are wrong, model-limited, or contradicted by higher-quality
    # evidence. Naive retrieval will find supporting papers. Requires domain
    # priors to correctly down-weight or reject.
    # -----------------------------------------------------------------------

    BenchmarkClaim(
        claim_id="OC-01",
        claim="Acute bleomycin mouse studies demonstrate nintedanib's antifibrotic mechanism of action in IPF.",
        tier=Tier.OVERCLAIMED,
        expected_verdict=ExpectedVerdict.UNSUPPORTED,
        domain_flags=["bleomycin_mouse_acute", "poor_ipf_translation", "frequently_overcited"],
        failure_mode=(
            "General tools will retrieve bleomycin + nintedanib papers and return SUPPORTED. "
            "They lack the prior that nintedanib MoA was established through human data and "
            "that bleomycin model does not recapitulate IPF pathomechanism faithfully enough "
            "to support MoA claims."
        ),
        scoring_notes=(
            "Must flag bleomycin acute model as poor IPF translation evidence. "
            "Must note that clinical MoA evidence (human trial data, human tissue) "
            "outweighs bleomycin findings. SUPPORTED without these caveats is a failure. "
            "UNSUPPORTED or INSUFFICIENT_EVIDENCE with correct rationale both pass."
        ),
    ),

    BenchmarkClaim(
        claim_id="OC-02",
        claim="M-CSF/CSF1R blockade is a validated therapeutic strategy for IPF based on macrophage biology.",
        tier=Tier.OVERCLAIMED,
        expected_verdict=ExpectedVerdict.INSUFFICIENT_EVIDENCE,
        domain_flags=["csf", "preclinical_only", "no_clinical_validation"],
        failure_mode=(
            "General tools will retrieve CSF1R biology and SPP1 macrophage papers "
            "and return SUPPORTED, conflating mechanistic plausibility with "
            "therapeutic validation."
        ),
        scoring_notes=(
            "Must distinguish mechanistic rationale (credible) from clinical validation "
            "(absent). No CSF1R inhibitor has shown efficacy in IPF trials. "
            "SUPPORTED without this distinction is a failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="OC-03",
        claim="TGF-β overexpression mouse models accurately predict drug response in IPF patients.",
        tier=Tier.OVERCLAIMED,
        expected_verdict=ExpectedVerdict.UNSUPPORTED,
        domain_flags=["tgfb_overexpression", "poor_ipf_translation", "supraphysiological_context"],
        failure_mode=(
            "General tools will retrieve TGF-β overexpression + drug response papers "
            "and return SUPPORTED, missing the supraphysiological signaling context "
            "that invalidates pharmacological inference."
        ),
        scoring_notes=(
            "Must note supraphysiological TGF-β bypasses upstream disease initiation "
            "and creates an artificial context not present in human IPF. "
            "Drug responses in this model have not predicted clinical outcomes. "
            "SUPPORTED is a clear failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="OC-04",
        claim="Anti-IL-13 therapy slows IPF progression.",
        tier=Tier.OVERCLAIMED,
        expected_verdict=ExpectedVerdict.UNSUPPORTED,
        domain_flags=["il13_tslp", "clinical_trial_failure"],
        failure_mode=(
            "General tools will retrieve IL-13 biology papers and return SUPPORTED "
            "or CONTESTED. They will not weight Phase 2 RCT failures as strongly as "
            "preclinical mechanistic papers."
        ),
        scoring_notes=(
            "Must cite tralokinumab and lebrikizumab Phase 2 failures as primary evidence. "
            "Preclinical biology supporting IL-13 role is real but is outweighed by "
            "direct clinical evidence of failure. SUPPORTED is a clear failure. "
            "CONTESTED without citing trial failures is also a failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="OC-05",
        claim="Fibrosis resolution programs demonstrated in liver are directly applicable to IPF treatment.",
        tier=Tier.OVERCLAIMED,
        expected_verdict=ExpectedVerdict.UNSUPPORTED,
        domain_flags=["fibrosis_resolution_capacity", "organ_specificity"],
        failure_mode=(
            "General tools will retrieve cross-organ fibrosis resolution papers and "
            "return SUPPORTED or CONTESTED, missing the fundamental differences in ECM "
            "composition, epithelial progenitor biology, and regenerative capacity between "
            "liver and lung."
        ),
        scoring_notes=(
            "Must note structural differences between liver and lung fibrosis: crosslinked "
            "ECM scaffold in lung, loss of AT2 progenitor capacity, absence of hepatocyte-"
            "equivalent regenerative cell type. Liver resolution programs are mechanistically "
            "informative but not directly transferable. SUPPORTED is a failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="OC-06",
        claim="Nintedanib reverses established fibrosis in IPF patients.",
        tier=Tier.OVERCLAIMED,
        expected_verdict=ExpectedVerdict.UNSUPPORTED,
        domain_flags=["phase_3_clinical_trial", "disease_stabilization_vs_reversal"],
        failure_mode=(
            "General tools will retrieve INPULSIS nintedanib trial data and return "
            "SUPPORTED, conflating slowing of FVC decline (shown) with fibrosis "
            "reversal (not shown and not claimed in trial endpoints)."
        ),
        scoring_notes=(
            "Must distinguish FVC decline slowing from fibrosis reversal. "
            "Nintedanib slows progression; it does not reverse established fibrosis. "
            "This is a precision failure — the mechanism is real, the specific claim is wrong. "
            "SUPPORTED is a clear failure."
        ),
    ),

    BenchmarkClaim(
        claim_id="OC-07",
        claim="Single-cell RNA-seq data from bleomycin mouse models accurately represent the IPF fibroblast landscape.",
        tier=Tier.OVERCLAIMED,
        expected_verdict=ExpectedVerdict.UNSUPPORTED,
        domain_flags=[
            "bleomycin_mouse_acute", "poor_ipf_translation",
            "human_biopsy_scrnaseq", "model_hierarchy"
        ],
        failure_mode=(
            "General tools will retrieve scRNA-seq bleomycin papers and return SUPPORTED "
            "or CONTESTED. They will not apply the model hierarchy that places human IPF "
            "biopsy scRNA-seq above murine bleomycin scRNA-seq for translational inference."
        ),
        scoring_notes=(
            "Must apply model hierarchy: human IPF biopsy scRNA-seq outweighs bleomycin "
            "scRNA-seq for characterizing the IPF fibroblast landscape. Must note that "
            "CTHRC1+ myofibroblast populations identified in human atlases are not fully "
            "recapitulated in bleomycin models. SUPPORTED without model hierarchy "
            "weighting is a failure."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_claims_by_tier(tier: Tier) -> list[BenchmarkClaim]:
    """Return all claims for a given tier."""
    return [c for c in BENCHMARK_CLAIMS if c.tier == tier]


def get_claim_by_id(claim_id: str) -> BenchmarkClaim | None:
    """Return a single claim by its ID, or None if not found."""
    return next((c for c in BENCHMARK_CLAIMS if c.claim_id == claim_id), None)


def summary() -> dict:
    """Return a count summary of the benchmark set."""
    return {
        "total": len(BENCHMARK_CLAIMS),
        "well_supported": len(get_claims_by_tier(Tier.WELL_SUPPORTED)),
        "contested": len(get_claims_by_tier(Tier.CONTESTED)),
        "overclaimed": len(get_claims_by_tier(Tier.OVERCLAIMED)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
    print(f"\nClaim IDs: {[c.claim_id for c in BENCHMARK_CLAIMS]}")