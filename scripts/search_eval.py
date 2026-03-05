#!/usr/bin/env python3
"""
Search PubMed for papers matching a query, embed and store results in ChromaDB,
augment with semantically similar papers already in the index, evaluate evidence
quality, and display all results in a formatted table.

Flow mirrors test_ingest_embed.ipynb:
  1. search_pubmed(query, max_results=N)
  2. embed_and_store → upsert into ChromaDB
  3. chroma_query(query, n_results=N) → ranked by semantic similarity
  4. fetch_by_pmids for any ChromaDB-only PMIDs (from prior sessions)
  5. evaluate_paper + compute_confidence on all
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pipeline.ingest import search_pubmed, fetch_by_pmids
from pipeline.embed import embed_and_store, query as chroma_query
from evaluators.evidence_quality import evaluate_paper
from evaluators.confidence_scorer import compute_confidence

def summarize_rationale(rationale: list[str]) -> str:
    """Return the composite score line + any model/pathway highlights (≤2 lines)."""
    composite = next((l for l in rationale if l.startswith("[composite]")), "")
    highlights = [l for l in rationale if l.startswith(("[model]", "[pathway]"))]
    parts = ([composite] if composite else []) + highlights[:1]
    return " | ".join(parts) if parts else "; ".join(rationale[:2])


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No papers met the confidence threshold.")
        return

    # Column widths
    W_PMID, W_CONF, W_OVR, W_MDL, W_PTH, W_SDS = 10, 6, 5, 5, 5, 5
    W_TITLE = 50
    W_RAT = 60

    header = (
        f"{'PMID':<{W_PMID}} {'Title':<{W_TITLE}} {'Conf':>{W_CONF}} "
        f"{'Ovrl':>{W_OVR}} {'Mdl':>{W_MDL}} {'Pth':>{W_PTH}} {'Sds':>{W_SDS}}  "
        f"{'Rationale':<{W_RAT}}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        title = r["title"][:W_TITLE-1] + "…" if len(r["title"]) > W_TITLE else r["title"]
        rat   = r["rationale_summary"][:W_RAT-1] + "…" if len(r["rationale_summary"]) > W_RAT else r["rationale_summary"]
        src   = "(chroma)" if r.get("from_chroma") else ""
        print(
            f"{r['pmid']:<{W_PMID}} {title:<{W_TITLE}} {r['confidence']:>{W_CONF}.3f} "
            f"{r['overall']:>{W_OVR}.3f} {r['model']:>{W_MDL}.3f} "
            f"{r['pathway']:>{W_PTH}.3f} {r['study_design']:>{W_SDS}.3f}  "
            f"{rat:<{W_RAT}} {src}"
        )
    print(sep)
    print(f"\n{len(rows)} paper(s)\n")

    # Detailed rationale block per paper
    print("=== Full Rationale ===")
    for r in rows:
        src = " [from ChromaDB index]" if r.get("from_chroma") else ""
        print(f"\nPMID {r['pmid']}: {r['title']}{src}")
        if r.get("abstract"):
            print(f"  Abstract       : {r['abstract']}")
        print(f"  Warnings       : {r['warnings']}")
        print(f"  Detected models: {r['detected_models']}")
        print(f"  Pathways       : {r['detected_pathways']}")
        print(f"  Contested flags: {r['contested_flags']}")
        for line in r["rationale_lines"]:
            print(f"    {line}")


def run(query: str, max_results: int = 50) -> list[dict]:
    """Execute the full search → embed → evaluate pipeline and return rows.

    Returns a list of dicts (one per paper) sorted by confidence descending,
    suitable for both CLI display and notebook widget consumption.
    """
    # ── 1. PubMed search ────────────────────────────────────────────────────
    print(f"\nSearching PubMed for up to {max_results} papers…")
    pubmed_papers = search_pubmed(query, max_results=max_results)
    print(f"Found {len(pubmed_papers)} papers from PubMed.")

    # ── 2. Embed & store PubMed results in ChromaDB ─────────────────────────
    if pubmed_papers:
        print(f"Embedding and storing {len(pubmed_papers)} papers in ChromaDB…")
        n_stored = embed_and_store(pubmed_papers)
        print(f"Stored {n_stored} documents in ChromaDB.")

    # ── 3. Query ChromaDB for semantically similar papers ───────────────────
    print(f"Querying ChromaDB for up to {max_results} semantically similar papers…")
    chroma_results = chroma_query(query, n_results=max_results)

    pubmed_pmids = {p["pmid"] for p in pubmed_papers}
    chroma_only_pmids = [r["pmid"] for r in chroma_results if r["pmid"] not in pubmed_pmids]

    # ── 4. Fetch full data for ChromaDB-only PMIDs ──────────────────────────
    extra_papers: list[dict] = []
    if chroma_only_pmids:
        print(f"Fetching full data for {len(chroma_only_pmids)} additional papers from ChromaDB…")
        extra_papers = fetch_by_pmids(chroma_only_pmids)
        for p in extra_papers:
            p["_from_chroma"] = True

    all_papers = pubmed_papers + extra_papers
    print(f"\nEvaluating {len(all_papers)} papers total "
          f"({len(pubmed_papers)} PubMed + {len(extra_papers)} ChromaDB-only)…\n")

    # ── 5. Evaluate ─────────────────────────────────────────────────────────
    rows = []
    for paper in all_papers:
        report = evaluate_paper(paper)
        confidence = compute_confidence(report)
        rows.append({
                "pmid": report.pmid,
                "title": report.title,
                "confidence": confidence,
                "overall": report.overall_score,
                "model": report.model_score,
                "pathway": report.pathway_score,
                "study_design": report.study_design_score,
                "warnings": report.warnings,
                "detected_models": report.detected_models,
                "detected_pathways": report.detected_pathways,
                "contested_flags": report.contested_flags,
                "rationale_summary": summarize_rationale(report.rationale),
                "rationale_lines": report.rationale,
                "abstract": paper.get("abstract", ""),
                "from_chroma": paper.get("_from_chroma", False),
        })

    rows.sort(key=lambda r: r["confidence"], reverse=True)
    return rows


def main() -> None:
    if len(sys.argv) >= 2:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ").strip()
        if not query:
            print("No query provided.")
            sys.exit(1)

    raw = input("How many papers to search/query? [default 50]: ").strip()
    max_results = int(raw) if raw.isdigit() else 50

    rows = run(query, max_results)
    print_table(rows)


if __name__ == "__main__":
    main()
