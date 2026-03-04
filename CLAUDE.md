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
