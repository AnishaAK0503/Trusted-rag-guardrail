# Pipeline phases

Every chunk carries `source_name`, `domain`, `document_id`, `url`, `title`, `page`, `section`, `chunk_id`, `topic`
from Phase 2 onwards. Provenance is never dropped or invented (`page` is `null` for HTML).

---
## Phase 1 - Trusted Source Management
**Objective:** decide deterministically which URLs may ever enter the system.
**Files:** `config/trusted_sources.json`, `app/source_manager.py`
**Input:** trusted-source configuration (name, domain, base URL, allowed types, topics).
**Processing:** load JSON into `TrustedSource` models; `TrustPolicy.validate_url()` requires HTTPS, a hostname, no
embedded credentials, port 443 only, and `host == domain` or `host.endswith("." + domain)`. So `who.int.fakewebsite.com`,
`fakewho.int` and `https://who.int@evil.com` are rejected. Shared error codes (`ErrorCode`, `PipelineError`) live here.
**Output:** a `TrustPolicy` object reused by every later phase (created once in `main.py`).
**Connects to:** Phase 2 validates every crawled URL and redirect with it; Phase 6 re-checks every result.

## Phase 2 - Document Discovery & Ingestion
**Objective:** collect relevant documents from trusted sources under strict limits.
**Files:** `app/ingestion.py` (`crawl_source`, `fetch_document`, `extract_html`, `extract_pdf`, `clean_text`, `process_document`, `run_ingestion`), `app/config.py` (limits)
**Input:** `TrustPolicy` (+ optional source names / extra URLs).
**Processing:** breadth-first crawl from each base URL (max pages, max depth, timeout, max size). Links are validated,
kept only if on the same source and matching topic keywords. Redirects are followed manually and each hop is re-validated.
Type is detected (HTML/PDF). HTML: BeautifulSoup+lxml, noise tags removed, headings + paragraphs kept. PDF: pypdf, page by page,
page numbers preserved. Cleaning only fixes whitespace; text is never paraphrased. Failures are logged, categorised and skipped.
**Output:** `data/raw/<id>.html|pdf`, `data/processed/documents/<id>.json`, `failures.json`.
**Connects to:** Phase 3 chunks the processed documents.

## Phase 3 - Preprocessing & Chunking
**Objective:** split documents into retrieval-sized pieces without losing provenance.
**Files:** `app/ingestion.py` (`split_words`, `chunk_document`, `build_chunks`)
**Input:** processed document JSON files.
**Processing:** each section (heading block or PDF page) is split with a word window (`chunk_size`=200, `chunk_overlap`=40);
tiny fragments are dropped; ID = `<document_id>_<nnn>`.
**Output:** `data/processed/chunks.json`.
**Connects to:** Phase 4 embeds these chunks.

## Phase 4 - Embeddings + FAISS
**Objective:** make chunks searchable by meaning.
**Files:** `app/rag.py` (`embed`, `build_index`, `load_index`)
**Input:** chunk list.
**Processing:** `all-MiniLM-L6-v2` produces normalised vectors; `faiss.IndexFlatIP` (inner product = cosine).
The index and chunk metadata (same row order) are saved; on startup the saved index is loaded, not rebuilt.
**Output:** `data/vector_store/index.faiss`, `data/vector_store/metadata.json`.
**Connects to:** Phase 5 searches it.

## Phase 5 - Query Retrieval
**Objective:** find candidate chunks for a user query.
**Files:** `app/rag.py` (`search`)
**Input:** query text, `top_k`.
**Processing:** embed query -> FAISS search for `top_k x 3` candidates (extra candidates leave room for filtering).
**Output:** list of `(chunk, score)`.
**Connects to:** Phase 6.

## Phase 6 - Re-ranking / Filtering
**Objective:** keep only good, trusted, non-duplicate evidence.
**Files:** `app/rag.py` (`filter_results`)
**Input:** candidates.
**Processing:** similarity threshold (0.30), optional topic filter, trust re-validation of each URL, duplicate-text removal, cut to `top_k`.
Currently retrieval-score filtering only. A cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) could be added by re-scoring the candidates inside this function.
**Output:** `list[RetrievedChunk]`.
**Connects to:** Phase 7.

## Phase 7 - RAG Context Preparation
**Objective:** package evidence for the LLM module.
**Files:** `app/rag.py` (`build_context`, `RagContext`)
**Input:** query + filtered chunks.
**Processing:** numbered blocks `[1] NIMH - Title (page 2)` followed by the exact chunk text. No answer is generated.
**Output:** `RagContext(query, context_text, citations)`.
**Connects to:** the next module (Llama 3 / Gemini) and Phase 8.

## Phase 8 - Provenance / Citation
**Objective:** every result is traceable.
**Files:** `app/rag.py` (`Citation`, `RetrievedChunk`), metadata created in `app/ingestion.py`
**Input:** chunk metadata.
**Processing:** each result and citation carries source, domain, document ID, title, URL, page (if any), chunk ID, score; `[n]` markers match `citations[n-1]`.
**Output:** visible in the `/api/v1/retrieve` response.

## Phase 9 - Evaluation
**Objective:** measure retrieval quality, trusted-source usage and provenance.
**Files:** `app/evaluation.py`, `config/eval_queries.json`
**Input:** sample queries with `relevant_keywords`.
**Processing:** relevant = URL/title contains a keyword. Precision@K, Recall@K (distinct relevant documents found / relevant documents in index), MRR, Hit Rate@K, trusted rate, provenance completeness. Queries with no relevant document in the index are skipped.
**Output:** JSON summary. Run `python -m app.evaluation`.

## Phase 10 - Automated Updating
**Objective:** refresh the knowledge base without redundant work.
**Files:** `app/ingestion.py` (manifest, `process_document`), `app/rag.py` (`ingest_and_index`), `POST /api/v1/ingest`
**Input:** the same call as Phase 2.
**Processing:** stable document ID from the URL; SHA-256 of cleaned text in `manifest.json`. Same hash -> `unchanged` (logged `DUPLICATE_DOCUMENT`); different hash -> `updated` (document rewritten); new ID -> `new`. The FAISS index is rebuilt only if something is new/updated (or `force_rebuild`).
**Output:** `IngestReport` + `index_rebuilt` flag.

---
## Complete system flow
```text
trusted_sources.json -> TrustPolicy (P1)
 -> crawl + validate + download (P2) -> HTML/PDF extract + clean + metadata -> raw/ + processed/documents/
 -> chunks.json (P3) -> embeddings + FAISS (P4) -> vector_store/
 POST /api/v1/retrieve: query -> embedding -> FAISS search (P5) -> filter (P6) -> RAG context (P7) -> citations (P8)
 -> next module: Llama 3 / Gemini
 P9: evaluation.py measures P5-P8.   P10: re-running /ingest updates only what changed.
```
