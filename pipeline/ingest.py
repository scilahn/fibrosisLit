"""
pipeline/ingest.py — Minimal PubMed ingestion and S3 upload for FibrosisLit.

Retrieval quality is not the differentiator in this project. This module is
intentionally thin: fetch abstracts + metadata from NCBI E-utilities, upload
raw JSON to S3. Evidence evaluation happens downstream in evaluators/.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NCBI E-utilities configuration
# ---------------------------------------------------------------------------

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.getenv("NCBI_API_KEY")  # Optional; increases rate limit to 10 req/s
NCBI_TOOL = "fibrosisLit"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "fibrosislit@example.com")

# NCBI requires a delay between requests without an API key (max 3 req/s).
# With an API key the limit is 10 req/s; we use 0.11s to stay safely under.
_REQUEST_DELAY_S = 0.11 if NCBI_API_KEY else 0.34

# ---------------------------------------------------------------------------
# S3 configuration
# ---------------------------------------------------------------------------

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "fibrosislens-papers")


def _ncbi_params(**kwargs: Any) -> dict[str, Any]:
    """Build base NCBI E-utilities query params, injecting credentials."""
    params: dict[str, Any] = {
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
        "retmode": "json",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    params.update(kwargs)
    return params


def _get(url: str, params: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
    """
    GET wrapper with logging and basic error handling.
    Raises requests.HTTPError on non-2xx responses.
    """
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    time.sleep(_REQUEST_DELAY_S)
    return response.json()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_by_pmids(pmids: list[str]) -> list[dict[str, Any]]:
    """
    Fetch abstracts and metadata for a list of PubMed IDs via efetch.

    Uses the ESummary endpoint (rettype=abstract is not available in JSON mode;
    ESummary returns structured metadata including title, authors, journal,
    publication date, and abstract when available).

    Args:
        pmids: List of PubMed ID strings (e.g. ["38949181", "37802650"]).

    Returns:
        List of paper dicts, each containing at minimum:
            pmid, title, abstract, authors, journal, pub_date, fetched_at.
        Papers where the API returns no abstract are included with
        abstract set to None and a warning logged.
    """
    if not pmids:
        return []

    logger.info("Fetching %d PMIDs from NCBI", len(pmids))

    # ESummary accepts comma-separated IDs; batch to stay within URL limits.
    batch_size = 200
    results: list[dict[str, Any]] = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = _ncbi_params(
            db="pubmed",
            id=",".join(batch),
            rettype="abstract",
        )
        data = _get(f"{NCBI_BASE_URL}/esummary.fcgi", params)
        result_map = data.get("result", {})

        for pmid in batch:
            entry = result_map.get(pmid)
            if entry is None:
                logger.warning("PMID %s not found in ESummary response", pmid)
                continue

            abstract = _fetch_abstract(pmid)
            paper = {
                "pmid": pmid,
                "title": entry.get("title", ""),
                "abstract": abstract,
                "authors": [
                    a.get("name", "") for a in entry.get("authors", [])
                ],
                "journal": entry.get("fulljournalname", entry.get("source", "")),
                "pub_date": entry.get("pubdate", ""),
                "doi": next(
                    (
                        aid["value"]
                        for aid in entry.get("articleids", [])
                        if aid.get("idtype") == "doi"
                    ),
                    None,
                ),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            if not abstract:
                logger.warning("No abstract available for PMID %s", pmid)

            results.append(paper)
            logger.debug("Fetched PMID %s: %s", pmid, paper["title"][:80])

    logger.info("Fetched %d papers", len(results))
    return results


def _fetch_abstract(pmid: str) -> str | None:
    """
    Fetch the full abstract text for a single PMID via efetch (plain text mode).
    ESummary does not include abstract text in JSON mode.
    """
    params = _ncbi_params(
        db="pubmed",
        id=pmid,
        rettype="abstract",
        retmode="text",
    )
    response = requests.get(
        f"{NCBI_BASE_URL}/efetch.fcgi", params=params, timeout=15
    )
    response.raise_for_status()
    time.sleep(_REQUEST_DELAY_S)

    text = response.text.strip()
    # efetch plain text includes title, authors, etc. before the abstract.
    # The abstract section starts after a blank line following the citation block.
    # Return the full text and let evaluators parse as needed; keep ingest minimal.
    return text if text else None


def search_pubmed(query: str, max_results: int = 50) -> list[dict[str, Any]]:
    """
    Search PubMed by query string and return full paper records for the hits.

    Runs ESearch to get PMIDs, then passes them to fetch_by_pmids.

    Args:
        query:       PubMed query string, e.g. "IPF biomarker 2024" or
                     "idiopathic pulmonary fibrosis[MeSH] AND TGF-beta".
        max_results: Maximum number of results to return. NCBI caps ESearch
                     at 10,000; keep this low for exploratory use.

    Returns:
        List of paper dicts in the same format as fetch_by_pmids.
    """
    logger.info("Searching PubMed: %r (max %d)", query, max_results)

    params = _ncbi_params(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    data = _get(f"{NCBI_BASE_URL}/esearch.fcgi", params)

    pmids: list[str] = data.get("esearchresult", {}).get("idlist", [])
    logger.info("ESearch returned %d PMIDs", len(pmids))

    if not pmids:
        return []

    return fetch_by_pmids(pmids)


def upload_to_s3(
    papers: list[dict[str, Any]],
    prefix: str = "raw/",
) -> list[str]:
    """
    Upload a list of paper dicts to S3 as individual JSON files.

    Each paper is stored at s3://<bucket>/<prefix><pmid>.json.
    Existing objects are overwritten — this is idempotent by PMID.

    Args:
        papers: List of paper dicts as returned by fetch_by_pmids.
        prefix: S3 key prefix. Default "raw/" keeps raw ingestion separate
                from any processed outputs written by evaluators.

    Returns:
        List of S3 keys for successfully uploaded papers.
    """
    if not papers:
        logger.warning("upload_papers_to_s3 called with empty list; nothing to do")
        return []

    s3 = boto3.client("s3")
    uploaded: list[str] = []

    for paper in papers:
        pmid = paper.get("pmid", "unknown")
        key = f"{prefix}{pmid}.json"
        body = json.dumps(paper, ensure_ascii=False, indent=2)

        try:
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            uploaded.append(key)
            logger.debug("Uploaded s3://%s/%s", S3_BUCKET, key)
        except Exception:
            logger.exception("Failed to upload PMID %s to S3", pmid)

    logger.info(
        "Uploaded %d/%d papers to s3://%s/%s",
        len(uploaded),
        len(papers),
        S3_BUCKET,
        prefix,
    )

    return uploaded
