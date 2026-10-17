"""Phases 4-8: embeddings + FAISS, retrieval, filtering, RAG context, provenance.

This module prepares trusted evidence. It does NOT write the final answer;
the next module (Llama 3 / Gemini) consumes RagContext.
"""
import logging
import re

import faiss
import numpy as np
from pydantic import BaseModel

from app.config import settings
from app.ingestion import IngestReport, build_chunks, read_json, run_ingestion, write_json
from app.source_manager import TrustPolicy

log = logging.getLogger("trusted_rag.rag")
_cache: dict = {}  # holds the loaded model, FAISS index and chunk metadata


class IndexNotReadyError(Exception):
    pass


# ------------------------------------------------------------------ response models (Phase 8 lives here)
class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str
    domain: str
    title: str
    url: str
    page: int | None = None      # only when the source really has pages (PDF)
    section: str | None = None
    topic: str


class Citation(BaseModel):
    ref: int
    source: str
    domain: str
    document_id: str
    title: str
    url: str
    page: int | None = None
    chunk_id: str
    score: float


class RagContext(BaseModel):
    query: str
    context_text: str
    citations: list[Citation]


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    rag_context: RagContext
    message: str | None = None


class UpdateResult(BaseModel):
    ingestion: IngestReport
    index_rebuilt: bool
    num_chunks: int


# ------------------------------------------------------------------ Phase 4: embeddings + FAISS
def get_model():
    if "model" not in _cache:
        from sentence_transformers import SentenceTransformer  # lazy: slow import
        _cache["model"] = SentenceTransformer(settings.embedding_model)
    return _cache["model"]


def embed(texts: list[str]) -> np.ndarray:
    """Normalised vectors, so inner product == cosine similarity."""
    vectors = get_model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def embedding_text(chunk: dict) -> str:
    return f"{chunk['section']}. {chunk['text']}" if chunk.get("section") else chunk["text"]


def build_index(chunks: list[dict]) -> int:
    """Embed all chunks, build FAISS, save index + metadata. Returns number of chunks indexed."""
    if not chunks:
        log.warning("no chunks to index")
        return 0
    vectors = embed([embedding_text(c) for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    settings.vector_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.index_file))
    write_json(settings.meta_file, {"embedding_model": settings.embedding_model, "chunks": chunks})
    _cache["index"], _cache["chunks"] = index, chunks
    log.info("FAISS index built with %d chunks", len(chunks))
    return len(chunks)


def load_index() -> bool:
    """Load a saved index instead of re-embedding. False if none/incompatible."""
    if "index" in _cache:
        return True
    if not (settings.index_file.exists() and settings.meta_file.exists()):
        return False
    meta = read_json(settings.meta_file, {})
    index = faiss.read_index(str(settings.index_file))
    if meta.get("embedding_model") != settings.embedding_model or index.ntotal != len(meta.get("chunks", [])):
        log.warning("saved index is incompatible with current settings; rebuild required")
        return False
    _cache["index"], _cache["chunks"] = index, meta["chunks"]
    return True


def get_chunks() -> list[dict]:
    return _cache["chunks"] if load_index() else []


# ------------------------------------------------------------------ Phase 5: retrieval
def search(query: str, k: int) -> list[tuple[dict, float]]:
    """Embed the query and return the k nearest (chunk, score) pairs."""
    if not load_index():
        raise IndexNotReadyError("No index found. Run POST /api/v1/ingest first.")
    index = _cache["index"]
    scores, ids = index.search(embed([query]), min(k, index.ntotal))
    return [(_cache["chunks"][i], float(sc)) for sc, i in zip(scores[0], ids[0]) if i >= 0]


# ------------------------------------------------------------------ Phase 6: filtering
def filter_results(candidates: list[tuple[dict, float]], top_k: int, topic: str | None,
                   policy: TrustPolicy) -> list[RetrievedChunk]:
    """Score threshold, optional topic filter, trust re-check, duplicate removal.

    (A cross-encoder re-ranker could be added here later; see docs/PHASES.md.)
    """
    seen: set[str] = set()
    results: list[RetrievedChunk] = []
    for chunk, score in candidates:  # already sorted by score, best first
        if score < settings.similarity_threshold:
            continue
        if topic and chunk["topic"] != topic.lower():
            continue
        if not policy.is_trusted(chunk["url"]):  # defence in depth
            continue
        key = re.sub(r"\s+", " ", chunk["text"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        results.append(RetrievedChunk(
            chunk_id=chunk["chunk_id"], document_id=chunk["document_id"], text=chunk["text"],
            score=round(score, 4), source=chunk["source_name"], domain=chunk["domain"],
            title=chunk["title"], url=chunk["url"], page=chunk.get("page"),
            section=chunk.get("section"), topic=chunk["topic"]))
        if len(results) == top_k:
            break
    return results


def retrieve(query: str, top_k: int, topic: str | None, policy: TrustPolicy) -> list[RetrievedChunk]:
    candidates = search(query, top_k * settings.candidate_multiplier)
    return filter_results(candidates, top_k, topic, policy)


# ------------------------------------------------------------------ Phases 7 + 8: context and citations
def build_context(query: str, results: list[RetrievedChunk]) -> RagContext:
    """Numbered evidence blocks for the next (LLM) module, plus a matching citation list."""
    blocks, citations = [], []
    for n, r in enumerate(results, start=1):
        where = f"page {r.page}" if r.page is not None else (f"section: {r.section}" if r.section else "web page")
        blocks.append(f"[{n}] {r.source} - {r.title} ({where})\n{r.text}")
        citations.append(Citation(ref=n, source=r.source, domain=r.domain, document_id=r.document_id,
                                  title=r.title, url=r.url, page=r.page, chunk_id=r.chunk_id, score=r.score))
    return RagContext(query=query, context_text="\n\n".join(blocks), citations=citations)


# ------------------------------------------------------------------ Phase 10: update pipeline
def ingest_and_index(policy: TrustPolicy, sources: list[str] | None = None,
                     urls: list[str] | None = None, force: bool = False) -> UpdateResult:
    """Discover -> ingest -> chunk -> embed -> FAISS. The index is rebuilt only when needed."""
    report = run_ingestion(policy, settings, sources, urls)
    changed = report.new + report.updated > 0
    if force or changed or not load_index():
        chunks = build_chunks(settings)
        _cache.pop("index", None)
        _cache.pop("chunks", None)
        count = build_index(chunks)
        return UpdateResult(ingestion=report, index_rebuilt=count > 0, num_chunks=len(chunks))
    return UpdateResult(ingestion=report, index_rebuilt=False, num_chunks=len(_cache["chunks"]))
