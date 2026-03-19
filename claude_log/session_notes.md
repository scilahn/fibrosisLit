# FibrosisLit — Session Notes

**Project:** Domain-expert-guided biological evidence evaluation for IPF and PSC research
**Owner:** Richard Ahn, PhD — computational biologist
**Primary goal:** Evidence *quality* assessment for fibrosis claims, not retrieval. Address what general biomedical AI tools (Elicit, Consensus, Semantic Scholar) get wrong in a disease area with contested biology and known preclinical model translation failures.

---

## Architecture Overview

```
domain_knowledge/fibrosis_priors.py   ← encoded biological priors (MODELS, PATHWAYS, CONTESTED_BIOLOGY)
       ↓
evaluators/
  model_relevance.py         ← regex-based preclinical model detection → translational score
  evidence_quality.py        ← orchestrates model + pathway + study_design → EvidenceQualityReport
  contradiction_detector.py  ← contested biology flag detection
  confidence_scorer.py       ← confidence = overall_score - (n_contested_flags × 0.10)
  claim_evaluator.py         ← two-vote system (deterministic + LLM) → ClaimEvaluationResult
       ↓
pipeline/
  ingest.py    ← PubMed ESearch/efetch, make_mesh_query(), upload_to_s3()
  embed.py     ← SPECTER2 proximity adapter, ChromaDB upsert/query
       ↓
benchmarks/
  benchmark_claims.py   ← 25 annotated claims (WS/CT/OC tiers, expected verdicts)
  benchmark_runner.py   ← run all 25, write JSON+CSV to results/, seed_chromadb()
       ↓
scripts/search_eval.py         ← CLI: PubMed search → embed → evaluate_paper → ranked table
notebooks/search_eval_ui.ipynb ← ipywidgets UI wrapping search_eval.run()
test/
  test_claim_evaluator_two_vote.ipynb  ← 3-claim integration test (WS-01, CT-02, OC-01)
  benchmark_runner_test.ipynb          ← full 25-claim benchmark notebook
```

---

## Build Log (chronological)

### Phase 1 — Foundation (commits a12e96d → 6639be2)

**`domain_knowledge/fibrosis_priors.py`**
The biological foundation of the whole project. Encodes:
- `MODELS` (11 entries, scored 0.25–1.0): preclinical model hierarchy from `phase_3_clinical_trial` (1.0) down to `bleomycin_mouse_acute` (0.25), each with `rationale` and `flags` (`poor_ipf_translation`, `frequently_overcited`)
- `PATHWAYS` (11 entries, scored 0.4–1.0): IPF pathway centrality — `tgfb_smad`, `integrin_avb6`, `tnik`, `pde4` are `central` (1.0); `autotaxin`, `csf`, `lpa1/2`, `il13_tslp` are `emerging` (0.4)
- `CONTESTED_BIOLOGY` (4 debates): `macrophage_polarization`, `myofibroblast_reversibility`, `fibrosis_resolution_capacity`, `bleomycin_translatability` — each with multiple `positions` that must be surfaced, never synthesized
- Helper functions: `get_pathway_prior()`, `get_model()`, `get_model_score()`, `has_flag()`
- `UNKNOWN_MODEL_FALLBACK_SCORE = 0.2`

### Phase 2 — Ingestion Pipeline (commits 9424ed1 → 428fae9)

**`pipeline/ingest.py`**
- `search_pubmed(query, max_results)` → ESearch PMIDs → `fetch_by_pmids()`
- `fetch_by_pmids()` → ESummary metadata + `_fetch_abstract()` (efetch plain text)
- `upload_to_s3()` → raw JSON per paper to S3
- `make_mesh_query(text, disease)` → prepends MeSH anchor (`"idiopathic pulmonary fibrosis"[MeSH Terms]`) and strips disease name variants from the text to avoid duplication
- Rate limiting: 0.11s with NCBI API key, 0.34s without

**`pipeline/embed.py`**
- SPECTER2 base model + proximity adapter (allenai/specter2) via `adapters` library
- `embed_and_store(papers)` → embed `"title [SEP] abstract"` → ChromaDB upsert by PMID
- `query(query_text, n_results)` → embed query → cosine similarity search → returns metadatas + distances + **documents** (title [SEP] abstract, added later for abstract recovery — see Phase 6)
- ChromaDB at `./chroma_db`, collection `fibrosis_papers`
- **Dependency note:** `adapters==1.2.0` requires `transformers~=4.51.3` — `BERT_INPUTS_DOCSTRING` was removed in 4.52+, causing `ImportError` if versions are mismatched

### Phase 3 — Evidence Evaluators (commit a818e8b)

**`evaluators/model_relevance.py`**
Regex-based model system detection against `title + abstract`. `MODEL_PATTERNS` ordered highest-evidence first (clinical trials → human biopsy scRNA-seq → ... → bleomycin acute). `primary_model` = highest-scoring detected key.

**`evaluators/evidence_quality.py`**
Orchestrates sub-evaluators → `EvidenceQualityReport`. Composite score:
```
overall_score = model_score × 0.40 + pathway_score × 0.30 + study_design_score × 0.30
```
`STUDY_DESIGN_TIERS` ordered strongest→weakest (first match wins): `clinical_trial` (1.0) → `single_cell_human` (0.90) → `human_cohort` (0.85) → `ex_vivo_human` (0.80) → `primary_cells` (0.60) → `animal_model` (0.50) → `cell_line` (0.30) → `unknown` (0.0).

**`evaluators/contradiction_detector.py`**
Detects contested biology flags from title + abstract. `fibrosis_resolution_capacity` regex gap was fixed in an earlier session (window `{0,15}` → `{0,40}`).

**`evaluators/confidence_scorer.py`**
`confidence = max(0, overall_score - len(contested_flags) × 0.10)`

### Phase 4 — Search Eval CLI + Notebook UI (commit ed2b6ed)

**`scripts/search_eval.py`**
Full pipeline: PubMed search → embed → ChromaDB query → evaluate_paper + compute_confidence → ranked table.
- `ingest_papers(query, max_results=200)` — PubMed → ChromaDB, returns count upserted
- `run(query, max_results=200)` — full evaluation pipeline, returns rows sorted by confidence
- CLI with interactive query + max_results prompt
- Default `max_results` raised from 50 → 200 after early testing showed sparse results

**`notebooks/search_eval_ui.ipynb`**
ipywidgets UI wrapping `run()`. Max results widget default raised to 200.

### Phase 5 — Claim Evaluator + Benchmark Runner (commits 50c1e2b → 5b4189e)

**`evaluators/claim_evaluator.py`** — Two-vote system:

**Voter 1 (deterministic):**
Prior-based scoring using `evaluate_paper()` + `contradiction_detector`. Returns `tier` ∈ {`WELL_SUPPORTED`, `CONTESTED`, `OVERCLAIMED`} and full rationale list.

**Voter 2 (LLM — claude-sonnet-4-20250514):**
- Retrieves 8 papers from ChromaDB (`embed.query`)
- Scores each with `evaluate_paper()` (now including abstract from ChromaDB documents field — see Phase 6)
- Formats evidence block: PMID, title, journal, evidence score, study design, detected models, pathways, model flags, contested flags, 400-char abstract excerpt
- Calls Claude API with `_build_system_prompt()` — dynamically injects `MODELS`, `PATHWAYS`, `CONTESTED_BIOLOGY` from `fibrosis_priors.py` at call time so the LLM stays in sync without manual prompt updates
- Returns JSON: `{verdict, confidence, reasoning, paper_stances}`
- **Code-fence stripping** added in `_parse_llm_response()` — LLM occasionally returned ` ```json ... ``` ` wrappers despite the system prompt; `json.loads()` was failing on the fence chars

**Adjudication rules (fixed table, in priority order):**

| Voter 1 | Voter 2 | Final verdict | Confidence |
|---------|---------|---------------|-----------|
| Any | — | CONTESTED | HIGH (Voter 1 contested_flag override) |
| WELL_SUPPORTED | SUPPORTED | SUPPORTED | HIGH |
| CONTESTED | CONTESTED | CONTESTED | HIGH |
| OVERCLAIMED | UNSUPPORTED | UNSUPPORTED | HIGH |
| WELL_SUPPORTED or OVERCLAIMED | CONTESTED | CONTESTED | MEDIUM |
| OVERCLAIMED | CONTESTED | CONTESTED | MEDIUM |
| Any disagreement | Any | LOW_CONFIDENCE | LOW |

**`ClaimEvaluationResult` fields:**
`claim`, `tier`, `prior_support_score`, `model_penalty`, `pathway_support_score`, `contested_flags`, `warnings`, `rationale`, `llm_verdict`, `llm_confidence`, `llm_reasoning`, `verdict`, `verdict_confidence`, `verdict_rationale`, `references`

**`benchmarks/benchmark_claims.py`** — 25 annotated claims:
- 8 WELL_SUPPORTED (WS-01 to WS-08): SPP1+ macrophages, αvβ6 integrin, TGF-β/SMAD3, nerandomilast FIBRONEER, CTHRC1+ fibroblasts, bexotegrast, aberrant basaloid, PDE4/IPF fibroblasts
- 7 CONTESTED (CT-01 to CT-07): myofibroblast dediff, M2 macrophage as primary driver, EMT contribution, IL-13 actionability, fibrosis resolution, bleomycin model fidelity, ATX/FVC
- 10 OVERCLAIMED (OC-01 to OC-10): bleomycin→nintedanib MoA, CSF1R validation, TGF-β overexpression model, anti-IL-13, liver→IPF applicability, nintedanib reversal, scRNA from bleomycin mice, etc.

**`benchmarks/benchmark_runner.py`**
- `run(disease, output_dir, seed)` — iterates all 25, writes `results_{timestamp}.json` + `.csv`
- `seed_chromadb(disease, max_results)` — 13 MeSH-anchored queries covering all 25 claim domains
- `--seed` and `--disease` CLI flags
- `_is_correct()` — treats INSUFFICIENT_EVIDENCE == UNSUPPORTED for scoring (OC-02 expected INSUFFICIENT_EVIDENCE counts as correct if actual_verdict is UNSUPPORTED)

---

### Phase 6 — ChromaDB Coverage Fixes (commits d927a7d → 759552c)

**Problem:** Early integration test runs showed only 1–2 papers upserted for SPP1/macrophage queries, leaving the LLM with INSUFFICIENT_EVIDENCE for WS-01.

**Root causes and fixes:**

**1. Compound AND queries too restrictive**
PubMed ANDs all free-text terms by default. "SPP1 macrophage myofibroblast IPF single-cell atlas" requires ALL 6 terms in every matching paper — result: 1 paper. Fixed by shortening all ingest queries to 2–3 core terms (`62a4ccd`). The `_SEED_QUERIES` list was expanded from 8 long queries to 13 short queries.

**2. MeSH anchoring**
`make_mesh_query()` prepends `"idiopathic pulmonary fibrosis"[MeSH Terms]`, triggering MeSH expansion and dramatically broadening the hit set vs. raw free-text.

**3. Abstract missing from ChromaDB retrieval**
`embed.query()` only requested `metadatas` and `distances` from ChromaDB — not `documents`. All papers were scored with `abstract=""` → model system undetected → `unrecognised_model_system` warning → minimum fallback score 0.08.
Fixed in `759552c`: added `"documents"` to the ChromaDB include list, split the stored `"title [SEP] abstract"` string to recover the abstract, and propagated it through `_score_retrieved_papers()` to both `evaluate_paper()` and the LLM evidence block (as a 400-char excerpt).

---

### Phase 7 — Pattern Detection Fixes (commits 3832e52 → 2c0ffd9)

**`bleomycin_mouse_acute` one-directional bug (3832e52)**
Original pattern: `bleomycin.{0,30}(mouse|mice|murine|rat)` — only matches when "bleomycin" appears *before* "mice/mouse/murine/rat".
Abstract phrasing "C57BL/6 mice were treated with bleomycin" → "mice" comes first → no match → fallback to `unrecognised_model_system` (score 0.08 instead of 0.25).
Fix: added `(mouse|mice|murine|rat).{0,30}bleomycin` (reverse direction) + BLM abbreviation patterns.
Also widened `study_design_tier` `animal_model` window `{0,20}` → `{0,30}` and added reverse direction.

**`human_biopsy_scrnaseq` proximity too tight (2c0ffd9)**
Original pattern: `(single.cell|scRNA.seq|snRNA.seq).{0,40}(human|patient|IPF|biopsy)` — 40-char window fails for typical 2019 abstracts using "single-cell RNA sequencing...IPF patient samples" (>40 chars apart).
PMID 31221805 ("Proliferating SPP1/MERTK-expressing macrophages in idiopathic pulmonary fibrosis") — a human scRNA-seq paper scoring 0.08 instead of ≥0.40.
Fix: widened to `{0,200}` and added reverse direction. Same fix to `single_cell_human` study design tier (also added `single.cell` alongside `scRNA.seq|snRNA.seq`).

---

### Phase 8 — Comparison Infrastructure (2026-03-19)

**`benchmarks/comparison/`** — new directory for head-to-head scoring:

**`benchmarks/comparison/README.md`**
Documents the full comparison workflow: querying Elicit, Consensus, and Semantic Scholar
by hand, recording verdicts in the template CSV, and running the analysis notebook.
Includes:
- Verdict vocabulary (SUPPORTED / CONTESTED / UNSUPPORTED / INSUFFICIENT_EVIDENCE)
- Per-tool output → verdict mapping rules (Elicit confidence language, Consensus Meter %, Semantic Scholar retrieval-only default)
- Correctness rules per tier (OVERCLAIMED: UNSUPPORTED or INSUFFICIENT_EVIDENCE both count as correct)
- Binary scoring policy (correct = 1/0; nuance captured in `*_response` free text)

**`benchmarks/comparison/generate_template.py`**
Imports `BENCHMARK_CLAIMS` and writes `comparison_template.csv` (22 rows, 15 columns).
Columns: `claim_id`, `tier`, `claim`, `expected_verdict`, `failure_mode`, `scoring_notes`,
then `*_response`, `*_verdict`, `*_correct` for each of three tools (blank for manual entry).
Re-runnable if claims change.

**`benchmarks/comparison/comparison_template.csv`**
22 rows pre-filled from `BenchmarkClaim` data. Tool verdict columns blank — filled by
Richard after querying each tool.

**`notebooks/comparison_analysis.ipynb`**
12-cell analysis notebook. Once the CSV is filled:
1. Loads latest `benchmarks/results/results_*.csv` → FibrosisLit two-vote verdicts
2. Loads `comparison_template.csv` → external tool verdicts
3. Merges on `claim_id`, computes `*_correct` for all four tools
4. Displays overall accuracy table, per-tier accuracy table, full verdict table (with ✓/✗),
   and failure mode analysis (which claims each tool failed + `failure_mode` text)

---

## Current State (2026-03-19)

**Integration tests:** All 3 pass (after Phases 6–7 fixes)
- WS-01: SUPPORTED ✓
- CT-02: CONTESTED ✓
- OC-01: UNSUPPORTED ✓

**Benchmark notebook:** `test/benchmark_runner_test.ipynb` created — seeds ChromaDB, runs all 22 claims, displays summary table + per-tier accuracy + failures table.

**Two-vote benchmark result (2026-03-18):** 17/22 (77.3%) overall — contested 6/7 (85.7%), overclaimed 4/7 (57.1%), well_supported 7/8 (87.5%).

**Comparison infrastructure:** `benchmarks/comparison/` created — template CSV (22 rows), scoring rubric, and analysis notebook ready for manual tool scoring (Elicit, Consensus, Semantic Scholar).

---

## Key Design Decisions

1. **Deterministic scoring before LLM** — LLM receives pre-scored, pre-flagged evidence blocks, not raw abstracts. Auditable, reproducible, not subject to LLM hallucination on evidence quality.
2. **Contested biology is surfaced, never resolved** — `contested_flags` Voter 1 override is a hard rule. Debates (M1/M2, myofibroblast reversibility, fibrosis resolution) are presented as competing positions and never synthesized into a confident summary.
3. **Model hierarchy is the primary discriminator** — bleomycin mouse data (0.25) is not treated as equivalent to human scRNA-seq (1.0). The 40% model weight in the composite score encodes this priority.
4. **Fibrosis priors injected at LLM call time** — `_build_system_prompt()` reads `MODELS`, `PATHWAYS`, `CONTESTED_BIOLOGY` from `fibrosis_priors.py` dynamically, so LLM stays in sync with any prior updates without manual prompt maintenance.
5. **Adjudication rules are fixed, not learned** — deterministic table prevents LLM from overriding hard-coded biology constraints (e.g., contested flag cannot be overridden by LLM returning SUPPORTED).

---

## Known Gaps / Future Work

- **PSC priors** not yet implemented — `disease="psc"` is reserved but `_MESH_ANCHORS` only has IPF and PSC entries; PSC-specific MODELS/PATHWAYS/CONTESTED_BIOLOGY not in `fibrosis_priors.py`
- **Elicit/Consensus/Semantic Scholar comparison** template ready (`benchmarks/comparison/comparison_template.csv`) — manual scoring step; fill tool verdict columns by querying each tool per `benchmarks/comparison/README.md`, then run `notebooks/comparison_analysis.ipynb`
- **Regex pattern coverage** — model detection and study design tier remain regex-based; novel phrasing (e.g., "10x Genomics", "Visium spatial") not always caught; monitored via notebook reference tables
- **Abstract text quality** — efetch plain text includes citation header (authors, journal, DOI) before the abstract paragraph; evaluators function correctly but noise is higher than a clean abstract would be
- **Asymmetric embedding** — `embed.query()` uses the same proximity adapter for both indexing (full abstract) and querying (short claim text); the adhoc_query adapter is designed for this asymmetric use case and could improve retrieval quality

---

## File Reference

| File | Purpose |
|------|---------|
| `domain_knowledge/fibrosis_priors.py` | MODELS, PATHWAYS, CONTESTED_BIOLOGY — all biological priors |
| `evaluators/model_relevance.py` | Regex-based preclinical model detection + translational score |
| `evaluators/evidence_quality.py` | Composite EvidenceQualityReport (model + pathway + study design) |
| `evaluators/contradiction_detector.py` | Contested biology flag detection |
| `evaluators/confidence_scorer.py` | Final confidence with contested flag penalty |
| `evaluators/claim_evaluator.py` | Two-vote system + fixed adjudication rules |
| `pipeline/ingest.py` | PubMed ingestion, make_mesh_query(), upload_to_s3() |
| `pipeline/embed.py` | SPECTER2 embedding + ChromaDB upsert/query |
| `scripts/search_eval.py` | Full pipeline CLI + ingest_papers() |
| `benchmarks/benchmark_claims.py` | 25 annotated benchmark claims |
| `benchmarks/benchmark_runner.py` | Full benchmark run + seed_chromadb() + CLI flags |
| `test/test_claim_evaluator_two_vote.ipynb` | 3-claim integration test (all passing) |
| `test/benchmark_runner_test.ipynb` | Full 22-claim benchmark notebook |
| `notebooks/search_eval_ui.ipynb` | ipywidgets search UI |
| `benchmarks/comparison/README.md` | Comparison workflow, scoring rubric, verdict mapping rules |
| `benchmarks/comparison/generate_template.py` | Generates blank comparison_template.csv from BENCHMARK_CLAIMS |
| `benchmarks/comparison/comparison_template.csv` | 22-row scoring sheet for Elicit/Consensus/Semantic Scholar |
| `notebooks/comparison_analysis.ipynb` | Accuracy + failure mode analysis after manual tool scoring |
