# FibrosisLit — Head-to-Head Comparison Against General Biomedical AI Tools

## Purpose

Measures FibrosisLit accuracy against Elicit, Consensus, and Semantic Scholar on 25 annotated
IPF claims across three evidence tiers. The comparison is **manual**: each tool is queried by
hand with each claim text; verdicts are recorded in `comparison_template.csv`; the analysis
notebook computes per-tool per-tier accuracy.

---

## Workflow

1. **Run the FibrosisLit benchmark** (if not already done):
   ```
   cd /path/to/fibrosisLit
   python benchmarks/benchmark_runner.py --seed
   ```
   Results written to `benchmarks/results/results_{timestamp}.csv`.

2. **Query each tool for all 25 claims:**
   - Open `comparison_template.csv`
   - For each row, copy the `claim` text into each tool's search box
   - Record the tool's response summary in `*_response` (brief excerpt)
   - Map the response to a verdict in `*_verdict` per the scoring rubric below

3. **Run the analysis notebook:**
   ```
   jupyter notebook notebooks/comparison_analysis.ipynb
   ```

---

## Files

| File | Description |
|------|-------------|
| `comparison_template.csv` | 25-row scoring sheet; tool columns filled in manually |
| `generate_template.py` | Script to (re-)generate the blank template from `benchmark_claims.py` |
| `README.md` | This file |

---

## Scoring Rubric

### Verdict vocabulary

All tools are scored against the same four verdicts:

| Verdict | Meaning |
|---------|---------|
| `SUPPORTED` | Tool confirms the claim with human-level evidence |
| `CONTESTED` | Tool surfaces competing positions or flags uncertainty |
| `UNSUPPORTED` | Tool explicitly refutes the claim or finds contradicting evidence |
| `INSUFFICIENT_EVIDENCE` | Tool finds no relevant papers or cannot assess |

### Mapping tool outputs → verdicts

**Elicit** (elicit.com — "What does the research say?" format)
- "The evidence supports…" / high confidence summary → `SUPPORTED`
- "Mixed evidence…" / "Some studies find… while others…" → `CONTESTED`
- "The evidence does not support…" / contradicting majority → `UNSUPPORTED`
- "No relevant studies found" / "I couldn't find papers on this" → `INSUFFICIENT_EVIDENCE`

**Consensus** (consensus.app — Consensus Meter + GPT-4 synthesis)
- Consensus Meter ≥ 70% Yes → `SUPPORTED`
- Consensus Meter 30–70% / mixed synthesis → `CONTESTED`
- Consensus Meter ≤ 30% / synthesis explicitly rejects → `UNSUPPORTED`
- No papers returned / "not enough evidence" → `INSUFFICIENT_EVIDENCE`

**Semantic Scholar** (semanticscholar.org — primarily retrieval, limited synthesis)
- S2 is a retrieval tool. If only paper lists are returned without a synthesized verdict,
  record `INSUFFICIENT_EVIDENCE` by default. If the Research Feed or AI Summary
  returns explicit language ("supports" / "does not support"), map accordingly.
- Record the TLDR of the most relevant returned paper in `semantic_scholar_response`.

### Correctness by tier

| Tier | Expected verdict | Correct if | Failure if |
|------|-----------------|-----------|-----------|
| WELL_SUPPORTED (WS) | SUPPORTED | `SUPPORTED` | CONTESTED, UNSUPPORTED, INSUFFICIENT_EVIDENCE |
| CONTESTED (CT) | CONTESTED | `CONTESTED` | SUPPORTED (false confidence), UNSUPPORTED (over-rejection) |
| OVERCLAIMED (OC) | UNSUPPORTED or INSUFFICIENT_EVIDENCE | `UNSUPPORTED` or `INSUFFICIENT_EVIDENCE` | SUPPORTED, CONTESTED |

**Binary scoring:** correct = 1 or 0. No partial credit in accuracy metrics. Nuance is
captured in the `*_response` free-text column for qualitative review.

---

## Per-Claim Scoring Notes

The `scoring_notes` column in `comparison_template.csv` (pulled directly from
`BenchmarkClaim.scoring_notes`) describes the specific criteria a tool's response must
meet to be scored correct. Key patterns:

**CONTESTED claims** — a tool returning `SUPPORTED` passes only if the response explicitly
surfaces the competing positions described in `scoring_notes`. Record the nuance in
`*_response`; the verdict is still `SUPPORTED` but it informs qualitative analysis.

**OVERCLAIMED claims** — a tool returning `CONTESTED` passes only if it cites the specific
failure evidence (trial failures, model translation limitations). `CONTESTED` without citing
these is scored as incorrect.

---

## Claim Tiers

| Tier | Count | What FibrosisLit must do better than general tools |
|------|-------|-----------------------------------------------------|
| WELL_SUPPORTED | 8 | Concordance validates retrieval; all tools expected to pass |
| CONTESTED | 7 | Surface competing positions; general tools return false confidence |
| OVERCLAIMED | 10 | Apply domain priors (model hierarchy, trial failures); general tools return SUPPORTED |
