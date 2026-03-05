"""
pipeline/embed.py — Embedding and ChromaDB storage for FibrosisLit.

Uses allenai/specter2 with the 'proximity' adapter, which is trained on
scientific paper similarity and outperforms general-purpose models for
biomedical abstract retrieval. Embeddings are stored in a local ChromaDB
instance. A snapshot function archives the ChromaDB directory to S3 so the
index can be restored across sessions without re-embedding.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import chromadb
from adapters import AutoAdapterModel
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPECTER2_BASE = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"          # proximity adapter
SPECTER2_ADAPTER_NAME = "proximity"

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "fibrosis_papers")

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "fibrosislens-papers")
S3_SNAPSHOT_PREFIX = "chroma_snapshots/"

# SPECTER2 max token length
MAX_TOKENS = 512

# Batch size for embedding; reduce if running out of memory
EMBED_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_tokenizer: Any = None
_model: Any = None


def _load_model() -> tuple[Any, Any]:
    """
    Load SPECTER2 base model and attach the proximity adapter.

    Model is cached in module-level singletons so repeated calls within a
    session don't reload from disk. First call will download ~500MB if the
    model is not already cached by HuggingFace.
    """
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        logger.info("Loading SPECTER2 tokenizer and base model")
        _tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE)
        _model = AutoAdapterModel.from_pretrained(SPECTER2_BASE)

        logger.info("Loading SPECTER2 proximity adapter")
        _model.load_adapter(SPECTER2_ADAPTER, source="hf", set_active=True)
        _model.eval()
        logger.info("SPECTER2 model ready")

    return _tokenizer, _model


# ---------------------------------------------------------------------------
# ChromaDB client (lazy singleton)
# ---------------------------------------------------------------------------

_chroma_client: chromadb.PersistentClient | None = None
_chroma_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, initializing the client if needed."""
    global _chroma_client, _chroma_collection

    if _chroma_client is None:
        logger.info("Initializing ChromaDB at %s", CHROMA_DIR)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    if _chroma_collection is None:
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d documents)",
            CHROMA_COLLECTION,
            _chroma_collection.count(),
        )

    return _chroma_collection


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using SPECTER2 proximity adapter.

    Truncates to MAX_TOKENS. Processes in batches to avoid OOM on large inputs.

    Returns:
        List of float vectors, one per input text.
    """
    import torch

    tokenizer, model = _load_model()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)

        # SPECTER2 uses the [CLS] token embedding as the document representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.extend(embeddings.tolist())
        logger.debug("Embedded batch %d–%d", i, i + len(batch))

    return all_embeddings


def _paper_to_text(paper: dict[str, Any]) -> str:
    """
    Concatenate title and abstract for embedding, matching SPECTER2's
    training format (title [SEP] abstract).
    """
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "") or ""
    abstract = abstract.strip()
    return f"{title} [SEP] {abstract}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_and_store(papers: list[dict[str, Any]]) -> int:
    """
    Embed a list of paper dicts and upsert them into ChromaDB.

    Uses PMID as the document ID, so re-ingesting the same paper updates
    its record rather than creating a duplicate.

    Args:
        papers: Paper dicts as returned by pipeline.ingest.fetch_by_pmids.
                Each must have a 'pmid' field.

    Returns:
        Number of documents upserted.
    """
    if not papers:
        logger.warning("embed_and_store called with empty list; nothing to do")
        return 0

    collection = _get_collection()

    texts = [_paper_to_text(p) for p in papers]
    ids = [p["pmid"] for p in papers]
    metadatas = [
        {
            "pmid": p.get("pmid", ""),
            "title": p.get("title", ""),
            "journal": p.get("journal", ""),
            "pub_date": p.get("pub_date", ""),
            "doi": p.get("doi") or "",
            "authors": ", ".join(p.get("authors", [])),
        }
        for p in papers
    ]

    logger.info("Embedding %d papers with SPECTER2 (proximity)", len(papers))
    embeddings = _embed_texts(texts)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info("Upserted %d documents into ChromaDB collection '%s'", len(papers), CHROMA_COLLECTION)
    return len(papers)


def query(
    query_text: str,
    n_results: int = 10,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Embed a query string and return the nearest neighbours from ChromaDB.

    Note: query_text is embedded with the same proximity adapter used at
    index time. For asymmetric search (short query vs. full abstract),
    consider switching to the adhoc_query adapter in a future version.

    Args:
        query_text: Free-text query, e.g. "SPP1 macrophage myofibroblast IPF".
        n_results:  Number of results to return.
        where:      Optional ChromaDB metadata filter dict.

    Returns:
        List of result dicts with keys: pmid, title, journal, pub_date,
        doi, authors, distance.
    """
    collection = _get_collection()

    embeddings = _embed_texts([query_text])
    query_args: dict[str, Any] = {
        "query_embeddings": embeddings,
        "n_results": n_results,
        "include": ["metadatas", "distances"],
    }
    if where:
        query_args["where"] = where

    results = collection.query(**query_args)

    output: list[dict[str, Any]] = []
    for meta, dist in zip(
        results["metadatas"][0], results["distances"][0]
    ):
        output.append({**meta, "distance": dist})

    return output


def snapshot_to_s3(label: str | None = None) -> str:
    """
    Archive the local ChromaDB directory as a .tar.gz and upload to S3.

    The snapshot is stored at:
        s3://<bucket>/chroma_snapshots/<label>.tar.gz

    where label defaults to a UTC timestamp (YYYYMMDD_HHMMSS). Use a fixed
    label (e.g. "latest") to maintain a single overwriteable snapshot, or
    timestamp labels to keep a history.

    Args:
        label: S3 object name without extension. Defaults to UTC timestamp.

    Returns:
        S3 key of the uploaded snapshot.
    """
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {CHROMA_DIR}. "
            "Run embed_and_store first."
        )

    if label is None:
        label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    s3_key = f"{S3_SNAPSHOT_PREFIX}{label}.tar.gz"

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info("Archiving ChromaDB directory %s → %s", CHROMA_DIR, tmp_path)
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(chroma_path, arcname=chroma_path.name)

        archive_size_mb = Path(tmp_path).stat().st_size / 1_048_576
        logger.info("Archive size: %.1f MB — uploading to s3://%s/%s", archive_size_mb, S3_BUCKET, s3_key)

        s3 = boto3.client("s3")
        s3.upload_file(tmp_path, S3_BUCKET, s3_key)
        logger.info("Snapshot uploaded: s3://%s/%s", S3_BUCKET, s3_key)

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return s3_key


def restore_from_s3(label: str) -> None:
    """
    Download a ChromaDB snapshot from S3 and extract it to CHROMA_DIR.

    Replaces the current local ChromaDB directory. The in-process ChromaDB
    client is reset so the next operation reinitializes against the restored
    data.

    Args:
        label: Snapshot label as used in snapshot_to_s3 (without .tar.gz).
    """
    global _chroma_client, _chroma_collection

    s3_key = f"{S3_SNAPSHOT_PREFIX}{label}.tar.gz"
    chroma_path = Path(CHROMA_DIR)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info("Downloading s3://%s/%s", S3_BUCKET, s3_key)
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, s3_key, tmp_path)

        if chroma_path.exists():
            logger.info("Removing existing ChromaDB at %s", CHROMA_DIR)
            shutil.rmtree(chroma_path)

        chroma_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting snapshot to %s", chroma_path.parent)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=chroma_path.parent)

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Reset singletons so the next call reinitializes against restored data
    _chroma_client = None
    _chroma_collection = None
    logger.info("ChromaDB restored from snapshot '%s'", label)
