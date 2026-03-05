"""
fibrosis_priors.py — Biological priors for IPF/PSC evidence evaluation.

Domain knowledge encoded by Richard Ahn, PhD 
Structure provided by Claude Code
Evaluators import from this module to weight evidence by translational relevance,
flag contested frameworks, and surface known model limitations rather than
silently treating all peer-reviewed data as equivalent.
"""

from dataclasses import dataclass, field
from enum import Enum

# Score assigned to any model not explicitly listed in the hierarchy.
# Conservative prior: uncharacterized models are treated as weak evidence.
UNKNOWN_MODEL_FALLBACK_SCORE: float = 0.2


@dataclass(frozen=True)
class PreclinicalModel:
    """
    A preclinical or clinical model system with its IPF translational relevance score
    and the biological reasoning behind that score.

    Attributes:
        name:      Canonical identifier used across evaluators.
        score:     Translational relevance score in [0, 1]. Higher = more directly
                   translatable to human IPF biology.
        rationale: Biological justification for the assigned score. Should explain
                   *why* this model does or does not recapitulate human disease,
                   not merely describe the model itself.
        flags:     Machine-readable tags for downstream evaluator logic.
                   e.g. "frequently_overcited", "poor_ipf_translation", "contested".
    """
    name: str
    score: float
    rationale: str
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preclinical model hierarchy — IPF translational relevance
#
# Ranked by how faithfully each system recapitulates human IPF biology:
# cell type composition, fibrotic microenvironment, disease chronicity,
# and pharmacological translatability to clinical outcomes.
# ---------------------------------------------------------------------------

PRECLINICAL_MODELS: dict[str, PreclinicalModel] = {
    "human_biopsy_scrnaseq": PreclinicalModel(
        name="human_biopsy_scrnaseq",
        score=1.0,
        rationale=(
            "Single-cell RNA-seq of human IPF biopsy tissue is the gold standard. "
            "Captures the full fibrotic microenvironment — SPP1+ macrophages, "
            "myofibroblast populations, aberrant basaloid epithelial cells — at "
            "single-cell resolution in actively diseased human lung. "
            "Directly reflects the biology any therapeutic must engage."
        ),
    ),
    "human_explant_pcls": PreclinicalModel(
        name="human_explant_pcls",
        score=0.95,
        rationale=(
            "Precision-cut lung slices from human explants preserve native tissue "
            "architecture, cell-cell interactions, and fibrotic ECM composition. "
            "Functionally tractable (drug treatment, imaging) while retaining "
            "human disease context. Slight discount vs. in vivo biopsy due to "
            "ex vivo culture artifacts and loss of systemic signals."
        ),
    ),
    "primary_human_fibroblasts": PreclinicalModel(
        name="primary_human_fibroblasts",
        score=0.75,
        rationale=(
            "Human-derived and genetically authentic, but stripped of the in vivo "
            "fibrotic microenvironment. Loss of paracrine signals from macrophages, "
            "epithelial cells, and ECM stiffness cues reduces fidelity. Useful for "
            "mechanistic dissection of fibroblast-intrinsic pathways."
        ),
    ),
    "ipsc_derived": PreclinicalModel(
        name="ipsc_derived",
        score=0.65,
        rationale=(
            "iPSC-derived models carry patient genetic backgrounds, enabling "
            "disease-relevant genotype-phenotype studies. However, differentiation "
            "fidelity to mature lung cell types is variable, and the fibrotic "
            "microenvironment is not recapitulated without co-culture systems."
        ),
    ),
    "humanized_mouse": PreclinicalModel(
        name="humanized_mouse",
        score=0.55,
        rationale=(
            "Engraftment of human immune components improves relevance for "
            "immune-mediated fibrosis mechanisms, but mouse lung architecture "
            "and stromal biology remain. Partial translational improvement over "
            "standard murine models for immune-fibroblast crosstalk studies."
        ),
    ),
    "bleomycin_mouse_chronic": PreclinicalModel(
        name="bleomycin_mouse_chronic",
        score=0.45,
        rationale=(
            "Repeated or chronic bleomycin dosing extends the fibrotic phase and "
            "reduces the acute inflammatory confound, making it a modest improvement "
            "over the standard acute protocol. Still fundamentally inflammation-driven "
            "and self-limiting in a way that human IPF is not."
        ),
        flags=["poor_ipf_translation"],
    ),
    "tgfb_overexpression": PreclinicalModel(
        name="tgfb_overexpression",
        score=0.30,
        rationale=(
            "Transgenic TGF-β overexpression drives fibrosis through the canonical "
            "SMAD pathway and can model downstream effector mechanisms. However, "
            "supraphysiological TGF-β levels bypass upstream disease initiation "
            "and create an artificial signaling context not present in human IPF."
        ),
    ),
    "bleomycin_mouse_acute": PreclinicalModel(
        name="bleomycin_mouse_acute",
        score=0.25,
        rationale=(
            "Acute bleomycin induces an inflammation-driven lung injury that largely "
            "resolves, in contrast to the progressive, irreversible fibrosis of human "
            "IPF. Drug responses in this model have repeatedly failed to translate "
            "clinically. Frequently overcited; findings should be down-weighted "
            "unless corroborated by human or PCLS data."
        ),
        flags=["frequently_overcited", "poor_ipf_translation"],
    ),
}


# ---------------------------------------------------------------------------
# Pathway centrality — IPF fibrotic signaling hierarchy
#
# Reflects the weight of human genetic, mechanistic, and clinical evidence
# supporting each pathway as a driver of IPF progression. "Central" pathways
# have deep mechanistic support and/or clinical validation. "Emerging" pathways
# have biologically plausible roles but limited or contested human evidence.
#
# Centrality tiers and their numeric priors:
#   central  = 1.0  — well-validated, mechanistically central, clinical support
#   strong   = 0.8  — strong mechanistic evidence, active clinical programs
#   moderate = 0.6  — credible role, incomplete human validation
#   emerging = 0.4  — biologically plausible, limited or early human evidence
# ---------------------------------------------------------------------------

class PathwayCentrality(str, Enum):
    CENTRAL  = "central"
    STRONG   = "strong"
    MODERATE = "moderate"
    EMERGING = "emerging"


CENTRALITY_SCORES: dict[PathwayCentrality, float] = {
    PathwayCentrality.CENTRAL:  1.0,
    PathwayCentrality.STRONG:   0.8,
    PathwayCentrality.MODERATE: 0.6,
    PathwayCentrality.EMERGING: 0.4,
}


@dataclass(frozen=True)
class FibrosisPathway:
    """
    A fibrosis-relevant signaling pathway with its centrality tier, numeric
    prior weight, and biological rationale for that classification.

    Attributes:
        name:        Canonical identifier used across evaluators.
        centrality:  Qualitative tier reflecting depth of mechanistic and
                     clinical evidence in human IPF.
        prior:       Numeric score derived from centrality tier. Used by
                     evaluators to weight positive or negative findings in
                     papers studying this pathway.
        rationale:   Biological justification for the centrality assignment.
                     Explains the pathway's role in IPF, relevant cell types,
                     and any important caveats (e.g. failed trials, crosstalk).
    """
    name: str
    centrality: PathwayCentrality
    prior: float
    rationale: str


PATHWAYS: dict[str, FibrosisPathway] = {
    "tgfb_smad": FibrosisPathway(
        name="tgfb_smad",
        centrality=PathwayCentrality.CENTRAL,
        prior=CENTRALITY_SCORES[PathwayCentrality.CENTRAL],
        rationale=(
            "TGF-β1/SMAD signaling is the master pro-fibrotic driver in IPF, acting "
            "downstream of repeated alveolar epithelial injury to promote "
            "fibroblast-to-myofibroblast differentiation, ECM deposition, and "
            "autocrine fibrogenesis in AT2-lineage cells. The canonical axis proceeds "
            "through SMAD3 phosphorylation (more potent than SMAD2 in this context) "
            "and nuclear Smad2/3-Smad4 complex driving α-SMA, PAI-1, and collagen "
            "transcription. Non-canonical branches (MAPK, PI3K/AKT, Rho GTPases) "
            "amplify the response; bidirectional crosstalk with Wnt/β-catenin is "
            "well-documented. No direct SMAD inhibitor is approved; αvβ6 integrin "
            "inhibition (bexotegrast) targets upstream latent TGF-β activation."
        ),
    ),
    "integrin_avb6": FibrosisPathway(
        name="integrin_avb6",
        centrality=PathwayCentrality.STRONG,
        prior=CENTRALITY_SCORES[PathwayCentrality.STRONG],
        rationale=(
            "αvβ6 is a lung epithelial-restricted integrin minimally expressed in "
            "healthy adult lung but strongly upregulated after injury. It binds the "
            "RGD motif in latent TGF-β LAP, physically activating TGF-β1 and driving "
            "local SMAD2/3 phosphorylation without protease cleavage. The companion "
            "integrin αvβ1 performs an analogous activation function in the "
            "mesenchymal compartment. Dual inhibition is the mechanistic basis for "
            "bexotegrast (PLN-74809), which showed dose-dependent receptor occupancy "
            "and exploratory FVC signals in Phase 2 INTEGRIS-IPF."
        ),
    ),
    "tnik": FibrosisPathway(
        name="tnik",
        centrality=PathwayCentrality.STRONG,
        prior=CENTRALITY_SCORES[PathwayCentrality.STRONG],
        rationale=(
            "TNIK is a serine/threonine kinase that functions as a required "
            "co-activator of TCF4-dependent Wnt target gene transcription: it binds "
            "active β-catenin in the nucleus and phosphorylates TCF4 to drive "
            "fibrosis-related Wnt targets. TNIK also participates in TGF-β, Hippo, "
            "JNK, and NF-κB signaling, making it a convergence node across multiple "
            "fibrotic pathways. INS018-055, a first-in-class oral TNIK inhibitor, "
            "showed significant FVC improvement vs. placebo in a randomized Phase 2a "
            "trial — the strongest Wnt-targeting clinical signal in IPF to date."
        ),
    ),
    "pde4": FibrosisPathway(
        name="pde4",
        centrality=PathwayCentrality.STRONG,
        prior=CENTRALITY_SCORES[PathwayCentrality.STRONG],
        rationale=(
            "PDE4 degrades intracellular cAMP; elevated PDE4 activity in IPF "
            "fibroblasts lowers cAMP/PKA tone, removing a brake on myofibroblast "
            "differentiation and ECM synthesis. PDE4 inhibition restores cAMP, "
            "suppresses TGF-β-driven fibrogenesis, and induces myofibroblast "
            "dedifferentiation. The selective PDE4B inhibitor nerandomilast "
            "(BI 1015550) met its primary endpoint in Phase 3 FIBRONEER-IPF and "
            "received FDA approval, validating this mechanism clinically. PDE4B "
            "selectivity over PDE4D is important — PDE4D inhibition drives the "
            "nausea/emesis that limited earlier pan-PDE4 inhibitors."
        ),
    ),
    "wnt_pathway": FibrosisPathway(
        name="wnt_pathway",
        centrality=PathwayCentrality.MODERATE,
        prior=CENTRALITY_SCORES[PathwayCentrality.MODERATE],
        rationale=(
            "Wnt/β-catenin signaling is aberrantly activated in IPF lung, "
            "particularly in AT2 cells and lung-resident mesenchymal cells, where "
            "it drives abnormal epithelial regeneration, mesenchymal-to-myofibroblast "
            "transition, and collagen synthesis. Increased nuclear β-catenin and "
            "activated TCF/LEF transcription have been documented in IPF fibroblasts. "
            "Bidirectional crosstalk with TGF-β/SMAD is well-established. No direct "
            "Wnt inhibitor has achieved regulatory approval in IPF; TNIK inhibition "
            "represents the strongest indirect Wnt-targeting clinical proof-of-concept."
        ),
    ),
    "hedgehog": FibrosisPathway(
        name="hedgehog",
        centrality=PathwayCentrality.EMERGING,
        prior=CENTRALITY_SCORES[PathwayCentrality.EMERGING],
        rationale=(
            "Hedgehog (Hh) signaling, normally silenced in adult lung, re-emerges "
            "in IPF epithelial cells, fibroblasts, myofibroblasts, and macrophages. "
            "SHH-mediated SMO/GLI1 activation promotes fibroblast-to-myofibroblast "
            "transition, with evidence that primary cilia-dependent GLI transcription "
            "is required for TGF-β1-driven differentiation of IPF fibroblasts — "
            "indicating direct Hh/TGF-β crosstalk. GLI inhibition reduced collagen "
            "accumulation in mouse models, but no Hh inhibitor has entered "
            "IPF-specific clinical trials. Translational context remains limited."
        ),
    ),
    "il13_tslp": FibrosisPathway(
        name="il13_tslp",
        centrality=PathwayCentrality.EMERGING,
        prior=CENTRALITY_SCORES[PathwayCentrality.EMERGING],
        rationale=(
            "IL-13 is a type 2 cytokine elevated in IPF BAL that promotes fibroblast "
            "activation, collagen synthesis, and alternative macrophage polarization; "
            "IPF fibroblasts are hyperresponsive to IL-13 via elevated IL-13Rα1. "
            "TSLP is upregulated in IPF lung and acts on fibroblasts via STAT3 to "
            "induce CCL2-driven monocyte recruitment. Despite compelling biology, "
            "both anti-IL-13 agents tested in IPF (tralokinumab, lebrikizumab) "
            "failed to meet primary endpoints, suggesting IL-13 alone is insufficient "
            "to drive progression in the heterogeneous IPF population."
        ),
    ),
    "lpa1": FibrosisPathway(
        name="lpa1",
        centrality=PathwayCentrality.EMERGING,
        prior=CENTRALITY_SCORES[PathwayCentrality.EMERGING],
        rationale=(
            "LPA1 (LPAR1) is the dominant LPA receptor on lung fibroblasts, mediating "
            "fibroblast chemotaxis, resistance to apoptosis, and vascular leak that "
            "amplifies the injury response. LPA levels are elevated in IPF BAL, and "
            "LPA1 is upregulated in IPF fibroblasts. LPA1 simultaneously promotes "
            "epithelial apoptosis, shifting the epithelial/mesenchymal balance toward "
            "fibrosis. Multiple selective LPA1 antagonists have shown antifibrotic "
            "activity preclinically, but no LPA1 inhibitor has received regulatory "
            "approval for IPF."
        ),
    ),
    "lpa2": FibrosisPathway(
        name="lpa2",
        centrality=PathwayCentrality.EMERGING,
        prior=CENTRALITY_SCORES[PathwayCentrality.EMERGING],
        rationale=(
            "LPA2 (LPAR2) is expressed on lung epithelial and mesenchymal cells in "
            "IPF and plays a mechanistically distinct role from LPA1: LPA2 signaling "
            "via Gαq activates αvβ6 integrin on bronchial epithelial cells, which "
            "then mediates latent TGF-β activation — directly linking the autotaxin/"
            "LPA axis to the αvβ6-TGF-β pathway. LPA2-deficient mice show delayed "
            "disease onset and protection against bleomycin-induced mortality. LPA2 "
            "may act as an upstream amplifier of the integrin-TGF-β axis rather than "
            "an independent driver; its relative contribution vs. LPA1 in human IPF "
            "is not fully resolved."
        ),
    ),
    "csf": FibrosisPathway(
        name="csf",
        centrality=PathwayCentrality.EMERGING,
        prior=CENTRALITY_SCORES[PathwayCentrality.EMERGING],
        rationale=(
            "CSF1 (M-CSF) and its receptor CSF1R regulate the monocyte/macrophage "
            "lineage central to fibrotic remodeling in IPF. CSF1R is highly expressed "
            "in peripheral monocytes and IPF lung macrophages; soluble CSF1R in BAL "
            "predicts disease progression. CSF1-driven expansion of SPP1hi macrophages "
            "— a population causally implicated in myofibroblast activation by "
            "single-cell data — places CSF at the macrophage→myofibroblast axis. "
            "CSF2 (GM-CSF) biology is more complex: it drives inflammatory macrophage "
            "responses but is also required for alveolar macrophage homeostasis, "
            "making its net role context-dependent."
        ),
    ),
    "autotaxin": FibrosisPathway(
        name="autotaxin",
        centrality=PathwayCentrality.EMERGING,
        prior=CENTRALITY_SCORES[PathwayCentrality.EMERGING],
        rationale=(
            "Autotaxin (ATX; ENPP2) is the secreted lysophospholipase D responsible "
            "for the majority of extracellular LPA production. ATX is expressed in "
            "IPF bronchial epithelial cells, fibroblasts, and inflammatory cells, and "
            "both ATX activity and LPA concentrations are elevated in IPF BAL and "
            "plasma, correlating with progression. The ATX-LPA axis is upstream of "
            "both LPA1-mediated fibroblast recruitment and LPA2-mediated αvβ6-TGF-β "
            "activation, making ATX inhibition an attractive single point to block "
            "multiple downstream pro-fibrotic signals. BBT-877 is in Phase 2a with "
            "CT-based imaging biomarkers; Phase 1 data showed up to 90% LPA "
            "suppression. No ATX inhibitor has yet received approval for IPF."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Contested biology — debates that must be surfaced, never silently resolved
#
# Evaluators MUST NOT synthesize contradictory papers on these topics into
# confident summaries. The disagreement is the signal; resolving it would
# misrepresent the state of the field.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContestedBiology:
    """
    A biological question that remains genuinely unresolved in the IPF field.

    Attributes:
        name:      Canonical identifier used across evaluators.
        debate:    One-sentence statement of the core disagreement.
        positions: Competing positions, stated neutrally. Evaluators should
                   present these as alternatives, not synthesize them.
    """
    name: str
    debate: str
    positions: list[str]


CONTESTED_BIOLOGY: dict[str, ContestedBiology] = {
    "myofibroblast_reversibility": ContestedBiology(
        name="myofibroblast_reversibility",
        debate="Whether differentiated myofibroblasts can dedifferentiate and whether this is relevant to fibrosis resolution in human IPF.",
        positions=[
            "Myofibroblasts are terminally differentiated; once activated they are committed to an ECM-secreting state and cannot revert, making fibrosis irreversible at the cellular level.",
            "Myofibroblasts retain plasticity under certain conditions (reduced TGF-β, mechanical unloading, PDE4 inhibition); dedifferentiation has been demonstrated in vitro and in some in vivo models.",
            "The relevance of in vitro dedifferentiation to human IPF is unclear; the fibrotic microenvironment may continuously re-activate any reverted cells, making reversibility functionally moot even if mechanistically possible.",
        ],
    ),
    "macrophage_polarization": ContestedBiology(
        name="macrophage_polarization",
        debate="Whether the M1/M2 binary framework meaningfully describes macrophage states in fibrotic lung, and which macrophage populations drive versus resolve fibrosis.",
        positions=[
            "M1/M2 is an oversimplification; single-cell data from IPF lung identify distinct populations (SPP1hi, TREM2hi, MoAM) that do not map cleanly onto the M1/M2 axis and carry independent functional significance.",
            "M2-like (alternatively activated) macrophages promote fibrosis via TGF-β secretion and tissue remodeling, and the M1/M2 framework retains utility as a first-order approximation for pathway-level analysis.",
            "Macrophage states in IPF are dynamically regulated by the local microenvironment; the same population may be pro- or anti-fibrotic depending on disease stage and spatial context.",
        ],
    ),
    "epithelial_mesenchymal_transition": ContestedBiology(
        name="epithelial_mesenchymal_transition",
        debate="Whether epithelial-mesenchymal transition contributes meaningfully to the myofibroblast pool in human IPF lung.",
        positions=[
            "Lineage tracing in mice shows minimal contribution of epithelial-derived cells to the myofibroblast population; EMT may be an in vitro artifact not recapitulated in vivo.",
            "Partial EMT (hybrid E/M states) is prevalent in IPF epithelium and contributes to fibrosis through paracrine signaling and basement membrane disruption, even if full mesenchymal fate conversion is rare.",
            "Aberrant basaloid cells in IPF, which co-express epithelial and mesenchymal markers, may represent a disease-specific partial EMT state that is functionally relevant without constituting classical EMT.",
        ],
    ),
    "fibrosis_resolution_capacity": ContestedBiology(
        name="fibrosis_resolution_capacity",
        debate="Whether established pulmonary fibrosis in humans can meaningfully resolve, or whether the adult lung lacks the regenerative capacity for true fibrotic regression.",
        positions=[
            "Unlike liver fibrosis, established pulmonary fibrosis does not resolve; the crosslinked ECM scaffold and loss of epithelial progenitor populations make structural regression implausible in adult human IPF.",
            "Fibrosis resolution has been demonstrated in animal models and in some post-inflammatory contexts; the question is whether the right molecular intervention could activate analogous resolution programs in IPF.",
            "Stabilization rather than resolution is the realistic therapeutic target; the debate conflates fibrotic regression with disease progression arrest, which are distinct outcomes.",
        ],
    ),
}


def is_contested(topic_name: str) -> bool:
    """
    Return True if a biological topic is flagged as contested.

    Evaluators should call this before summarizing any finding that touches
    a contested area, and surface the debate rather than resolving it.
    """
    return topic_name in CONTESTED_BIOLOGY


def get_contested(topic_name: str) -> ContestedBiology | None:
    """Return the full ContestedBiology entry, or None if not recognized."""
    return CONTESTED_BIOLOGY.get(topic_name)


def get_pathway(pathway_name: str) -> FibrosisPathway | None:
    """Return the full FibrosisPathway entry, or None if not recognized."""
    return PATHWAYS.get(pathway_name)


def get_pathway_prior(pathway_name: str) -> float:
    """
    Return the numeric prior weight for a pathway.

    Falls back to the EMERGING score (0.4) for unrecognized pathways —
    a conservative prior that acknowledges possible biological relevance
    without overstating it.
    """
    pathway = PATHWAYS.get(pathway_name)
    return pathway.prior if pathway is not None else CENTRALITY_SCORES[PathwayCentrality.EMERGING]


# ---------------------------------------------------------------------------
# Biomarker evidence levels — pharmacodynamic and predictive biomarkers
#
# Reflects the evidentiary weight of a biomarker finding based on the context
# in which it was measured. Clinical trial validation outweighs preclinical
# observations; human tissue data outweighs animal data at equivalent stages.
# ---------------------------------------------------------------------------

BIOMARKER_EVIDENCE_LEVELS: dict[str, float] = {
    "validated_clinical_trial":  1.00,  # Prospectively validated in a powered trial
    "phase2_exploratory":        0.80,  # Exploratory endpoint; signal without confirmation
    "phase1_pdmarker":           0.65,  # PD marker in dose-escalation; target engagement only
    "preclinical_human_tissue":  0.70,  # Human biopsy / PCLS — outweighs phase 1 animal PD
    "preclinical_animal":        0.40,  # Animal model; subject to translation failure
    "in_vitro_only":             0.25,  # No in vivo context; weakest evidence tier
}

# Conservative fallback for unrecognized biomarker evidence contexts.
UNKNOWN_BIOMARKER_FALLBACK_SCORE: float = 0.2


def get_biomarker_score(evidence_level: str) -> float:
    """
    Return the evidentiary weight for a biomarker finding at a given evidence level.

    Note that preclinical_human_tissue (0.70) intentionally outscores
    phase1_pdmarker (0.65): human tissue data is more directly informative
    about disease-relevant biology than target engagement measured in a
    dose-escalation study.

    Falls back to UNKNOWN_BIOMARKER_FALLBACK_SCORE for unrecognized levels.
    """
    return BIOMARKER_EVIDENCE_LEVELS.get(evidence_level, UNKNOWN_BIOMARKER_FALLBACK_SCORE)


def get_model_score(model_name: str) -> float:
    """
    Return the translational relevance score for a preclinical model.

    Falls back to UNKNOWN_MODEL_FALLBACK_SCORE for unrecognized models rather
    than raising, so evaluators can process papers with novel or ambiguously
    described model systems without crashing. The fallback score is deliberately
    conservative — unknown models have not been validated for IPF translatability.

    Args:
        model_name: Canonical model identifier (key in PRECLINICAL_MODELS).

    Returns:
        Score in [0, 1]. Higher = more translationally relevant to human IPF.
    """
    model = PRECLINICAL_MODELS.get(model_name)
    return model.score if model is not None else UNKNOWN_MODEL_FALLBACK_SCORE


def get_model(model_name: str) -> PreclinicalModel | None:
    """
    Return the full PreclinicalModel entry, or None if not recognized.

    Prefer this over get_model_score() when evaluators need flags or rationale
    for logging and auditable scoring decisions.
    """
    return PRECLINICAL_MODELS.get(model_name)


def has_flag(model_name: str, flag: str) -> bool:
    """
    Check whether a model carries a specific flag.

    Useful for evaluators that need to emit warnings or apply score penalties
    for known problem models (e.g. "frequently_overcited").

    Returns False for unrecognized models rather than raising.
    """
    model = PRECLINICAL_MODELS.get(model_name)
    return flag in model.flags if model is not None else False
