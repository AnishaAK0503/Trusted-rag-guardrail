# Automated Trusted Document Retrieval + RAG for Mental Health Information

## Problem statement
LLMs answering mental-health questions can hallucinate or quote unreliable websites. Answers must be grounded in
evidence from **verifiably trusted organisations**, with a traceable citation for every statement.

## Objective
Build a locally runnable pipeline that (1) accepts content only from whitelisted organisations (WHO, NIMH, NICE),
(2) retrieves the most relevant trusted passages for a query, and (3) hands that evidence, with full provenance, to a
separate LLM module (Llama 3 / Gemini). This project does **not** generate the final answer and does **not** do safety
evaluation (Llama Guard is handled elsewhere).

| Concern | Question | Handled by |
|---|---|---|
| Trusted-source management | Can this source be trusted? | `source_manager.py` (deterministic config, no LLM) |
| Retrieval | Which trusted information is relevant? | `ingestion.py`, `rag.py` |
| Generation | How to turn evidence into an answer? | Next module (out of scope) |

## Architecture
```text
Trusted sources -> URL validation -> discovery -> download -> HTML/PDF extraction -> cleaning -> chunking
 -> embeddings -> FAISS  ...  query -> embedding -> search -> filtering -> provenance -> RAG context -> LLM module
```

## Technologies
Python 3.11+, FastAPI, Uvicorn, Requests, BeautifulSoup4 + lxml, pypdf, sentence-transformers
(`all-MiniLM-L6-v2`), FAISS, NumPy, Pydantic, pytest. No LangChain, no database.

## Project structure
```text
app/main.py            FastAPI endpoints          app/rag.py         Phases 4-8
app/config.py          all settings               app/evaluation.py  Phase 9
app/source_manager.py  Phase 1                    config/            trusted sources + eval queries
app/ingestion.py       Phases 2, 3, 10            data/              raw, processed, vector_store
tests/  docs/  requirements.txt
```

## Pipeline (10 phases)
1 Source management · 2 Discovery & ingestion · 3 Chunking · 4 Embeddings + FAISS · 5 Retrieval ·
6 Filtering · 7 RAG context · 8 Provenance/citations · 9 Evaluation · 10 Automated updating.
Details: `docs/PHASES.md`.

## API endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/ingest` | discover, ingest, chunk, embed, build/update FAISS |
| POST | `/api/v1/retrieve` | query -> trusted chunks + RAG context + citations |
| GET | `/health` | service check (not a functional API) |

Swagger UI: `http://127.0.0.1:8000/docs`

## Setup and running
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
uvicorn app.main:app --reload
```
Full instructions: `docs/RUNNING.md`. Manual validation: `docs/MANUAL_TESTING.md`.

## Testing
`pytest` runs offline tests (trust policy, HTML/PDF extraction, chunking, crawling, duplicates, retrieval, API).
Retrieval quality: `python -m app.evaluation`.

## Future improvements
Cross-encoder re-ranking, hybrid BM25 + vector search, incremental embedding (only new chunks), sitemap-based
discovery, scheduled updates, publication-date extraction, the Llama 3 / Gemini answer module.
