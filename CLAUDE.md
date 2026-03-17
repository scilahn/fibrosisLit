# FibrosisLit — Claude Code Project Context

## What This Project Is
A domain-expert-guided biological evidence evaluation framework for IPF and PSC research.
The primary contribution is domain knowledge encoded as evaluation logic, NOT retrieval
infrastructure. General biomedical AI tools (Elicit, Consensus, Semantic Scholar) handle
retrieval fine. This project addresses what they get wrong: evidence quality assessment
for a disease area with contested biology and known preclinical model translation failures.

## Owner Background
Richard Ahn, PhD — computational biologist with 3+ years analyzing IPF and PSC datasets
at Pliant Therapeutics (clinical-stage biotech). Direct experience with:
- snRNA-seq on precision-cut lung and liver slices from IPF/PSC/PBC patients
- Olink and SomaScan proteomics for pharmacodynamic biomarker identification
- αvβ6/αvβ1 integrin biology (bexotegrast program)
- SPP1+ macrophage and myofibroblast populations in fibrotic tissue

## Primary Contribution Areas
1. domain_knowledge/ — biological priors, evidence taxonomy, contested biology docs
   THIS is where domain expertise lives. Claude Code assists structure; biology is mine.
2. evaluators/ — multi-criteria evidence scoring using those priors
3. benchmarks/ — head-to-head comparison vs. Elicit, Consensus, AI2 on real queries

## Key Biological Priors to Respect
- Bleomycin-acute mouse model has poor IPF translation — do not treat as strong evidence
- M1/M2 macrophage framework is contested in fibrosis — always flag
- Human biopsy / PCLS data outweighs animal model data for translational relevance
- αvβ6 integrin, TGF-β/SMAD, SPP1+ macrophage → myofibroblast axis are well-supported
- Myofibroblast reversibility is actively debated — surface disagreement, don't resolve it

## Tech Stack
- Python 3.11+
- ChromaDB (local vector store)
- Claude API (claude-sonnet-4-20250514) for evaluation agents
- AWS S3 for paper storage and ChromaDB snapshots
- PubMed E-utilities API (no key needed for low volume; NCBI key for higher volume)

## Environment Variables (in .env, never commit)
ANTHROPIC_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-west-2
S3_BUCKET_NAME=fibrosislens-papers
NCBI_API_KEY=  # optional but recommended

## Code Style
- Type hints on all functions
- Docstrings explaining the biological rationale, not just what the code does
- Explicit logging of evidence scoring decisions (auditable outputs)
- Never silently resolve contested biology — always surface the debate

## What Claude Code Should NOT Do
- Suggest generic relevance scoring without fibrosis-specific model hierarchy weighting
- Treat all peer-reviewed papers as equivalent evidence quality
- Resolve mechanistic debates by synthesizing contradictory papers into confident summaries
- Propose M1/M2 macrophage classification as established in fibrosis context

## Next Build: evaluators/claim_evaluator.py

### Purpose
Takes a biological claim as input, retrieves evidence from ChromaDB, scores it
using the existing evaluator stack, and calls the Claude API for a verdict.
This is the layer that connects pipeline/ (retrieval) to evaluators/ (scoring).

### Data flow
1. query ChromaDB via pipeline.embed.query(claim, n_results=n_evidence)
2. For each retrieved paper: evaluate_paper(paper) → EvidenceQualityReport
3. compute_confidence(report) on each report
4. Partition into supporting / contesting / flagged_contested lists
5. Build structured Claude API prompt with pre-scored evidence (NOT raw abstracts)
6. Call claude-sonnet-4-20250514, request JSON output only
7. Parse into ClaimEvaluationResult dataclass, log full audit trail

### ClaimEvaluationResult fields
- claim: str
- verdict: str  # SUPPORTED | CONTESTED | UNSUPPORTED | INSUFFICIENT_EVIDENCE
- confidence: float  # aggregate, weighted by evidence quality scores
- supporting_evidence: list[EvidenceQualityReport]
- contesting_evidence: list[EvidenceQualityReport]
- contested_biology_flags: list[str]  # surfaced, never resolved
- model_hierarchy_warnings: list[str]  # e.g. frequently_overcited
- claude_reasoning: str  # raw LLM output, logged for auditability
- rationale: list[str]  # full deterministic audit trail pre-LLM

### Critical constraints (same as rest of project)
- Claude API receives pre-scored, pre-flagged evidence — NOT raw abstracts
- Deterministic scoring runs before LLM inference
- System prompt must instruct Claude to surface contested biology as unresolved
  alternatives, never synthesize competing positions into a confident summary
- Bleomycin-only evidence must be flagged as weak in the prompt context
- Claude API returns JSON only — no prose wrapper
- All scoring decisions logged to rationale list

### First test claim
WS-01: "SPP1+ macrophages promote myofibroblast activation in IPF lung."
Expected: SUPPORTED, human scRNA-seq evidence, no contested flags, no model warnings.

## Next Build: benchmarks/benchmark_runner.py

### Purpose
Runs all 25 claims in benchmark_claims.py through evaluate_claim() and writes
structured results for analysis and comparison against Elicit/Consensus/Semantic Scholar.

### Output
Two files written to benchmarks/results/:
  - results_{timestamp}.json  — full ClaimEvaluationResult per claim (for audit)
  - results_{timestamp}.csv   — flat summary table (for analysis and README table)

### CSV columns
claim_id, tier, claim, expected_verdict, actual_verdict, correct (bool),
confidence, pathway_support_score, model_penalty, prior_support_score,
contested_flags (semicolon-joined), model_warnings (semicolon-joined)

### Behavior
- Runs all 25 claims sequentially
- Logs progress to stdout: "Running WS-01 (1/25)..."
- Catches and logs exceptions per claim without stopping the run
- Prints summary at end: overall accuracy, accuracy by tier
- Rate limit: 1 second sleep between claims (Anthropic API)

### No comparison tool calls yet
Elicit/Consensus comparison is a manual scoring step — runner only covers
FibrosisLit. Comparison tool outputs will be scored by hand against the
same rubric and added to a separate CSV column later.
