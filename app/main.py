"""FastAPI layer. Two main endpoints (ingest, retrieve) plus /health."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import rag
from app.config import settings
from app.source_manager import load_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
policy = load_policy(settings.sources_file)  # Phase 1 output, reused everywhere


@asynccontextmanager
async def lifespan(_: FastAPI):
    if rag.load_index():  # reuse the saved index; never re-embed on startup
        rag.get_model()
    yield


app = FastAPI(
    title="Trusted RAG - Mental Health Information",
    description="Trusted document retrieval. Returns evidence + provenance for a later LLM module.",
    version="1.0.0",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    sources: list[str] | None = Field(default=None, description="Limit to these source names, e.g. ['NIMH']")
    urls: list[str] = Field(default_factory=list, description="Extra document URLs; each must pass the trust policy")
    force_rebuild: bool = Field(default=False, description="Rebuild the FAISS index even if nothing changed")


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500, examples=["What are common symptoms of anxiety?"])
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)
    topic: str | None = Field(default=None, description="Optional topic filter, e.g. 'anxiety'")


@app.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok", "trusted_sources": [s.name for s in policy.sources], "index_ready": rag.load_index()}


@app.post("/api/v1/ingest", response_model=rag.UpdateResult, tags=["Pipeline"])
def ingest(req: IngestRequest | None = None):
    """Discover -> download -> extract -> chunk -> embed -> FAISS. Safe to call again (incremental)."""
    req = req or IngestRequest()
    known = {s.name.lower() for s in policy.sources}
    unknown = {n.lower() for n in req.sources or []} - known
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown source(s): {sorted(unknown)}")
    return rag.ingest_and_index(policy, req.sources, req.urls, req.force_rebuild)


@app.post("/api/v1/retrieve", response_model=rag.RetrieveResponse, tags=["Pipeline"])
def retrieve(req: RetrieveRequest):
    """Query -> embedding -> FAISS -> filtering -> trusted chunks + RAG context + citations."""
    try:
        results = rag.retrieve(req.query, req.top_k, req.topic, policy)
    except rag.IndexNotReadyError as err:
        raise HTTPException(status_code=409, detail=str(err))
    message = None if results else "No trusted content passed the relevance threshold for this query."
    return rag.RetrieveResponse(query=req.query, results=results,
                                rag_context=rag.build_context(req.query, results), message=message)
