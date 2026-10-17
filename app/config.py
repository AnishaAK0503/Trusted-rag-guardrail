"""Central configuration. Every tunable value lives here."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # optional .env overrides; no secrets are required
BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default, cast=str):
    return cast(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # ---- paths ----
    data_dir: Path = Path(_env("DATA_DIR", BASE_DIR / "data"))
    sources_file: Path = BASE_DIR / "config" / "trusted_sources.json"
    eval_file: Path = BASE_DIR / "config" / "eval_queries.json"

    # ---- Phase 2: controlled crawling / download limits ----
    max_pages_per_source: int = _env("MAX_PAGES_PER_SOURCE", 15, int)
    max_depth: int = _env("MAX_DEPTH", 2, int)
    request_timeout: int = _env("REQUEST_TIMEOUT", 15, int)  # seconds
    max_document_bytes: int = _env("MAX_DOCUMENT_MB", 10, int) * 1024 * 1024
    max_redirects: int = 5
    min_document_words: int = 50  # shorter pages (hub/menu pages) are skipped
    user_agent: str = "TrustedRAG-Academic/1.0 (student research project)"

    # ---- Phase 3: chunking (sizes are in words) ----
    chunk_size: int = _env("CHUNK_SIZE", 200, int)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 40, int)
    min_chunk_words: int = 10
    generic_topics: tuple[str, ...] = ("mental health", "mental disorders")

    # ---- Phase 4: embeddings ----
    embedding_model: str = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # ---- Phases 5-6: retrieval and filtering ----
    default_top_k: int = 5
    max_top_k: int = 20
    candidate_multiplier: int = 3  # fetch top_k * 3 candidates, then filter
    similarity_threshold: float = _env("SIMILARITY_THRESHOLD", 0.30, float)

    # ---- derived paths ----
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def documents_dir(self) -> Path:
        return self.processed_dir / "documents"

    @property
    def manifest_file(self) -> Path:
        return self.processed_dir / "manifest.json"

    @property
    def chunks_file(self) -> Path:
        return self.processed_dir / "chunks.json"

    @property
    def failures_file(self) -> Path:
        return self.processed_dir / "failures.json"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @property
    def index_file(self) -> Path:
        return self.vector_dir / "index.faiss"

    @property
    def meta_file(self) -> Path:
        return self.vector_dir / "metadata.json"

    def ensure_dirs(self) -> None:
        for folder in (self.raw_dir, self.documents_dir, self.vector_dir):
            folder.mkdir(parents=True, exist_ok=True)


settings = Settings()
