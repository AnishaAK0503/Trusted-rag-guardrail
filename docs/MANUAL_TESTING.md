# Manual testing guide

Run these in order. Open a second terminal (with the venv activated) for the `python -c` commands.
Base URL: `http://127.0.0.1:8000`; use `/docs` (Swagger) for the POST requests.

## Test 1 - Start the server
- **Objective:** the app starts. **Action:** `uvicorn app.main:app --reload`
- **Expected:** "Application startup complete", no traceback. **Check:** `/docs` opens.

## Test 2 - Health
- **Action:** open `/health` (or `curl http://127.0.0.1:8000/health`).
- **Expected:** `{"status":"ok","trusted_sources":["WHO","NIMH","NICE"],"index_ready":false}` (`false` before the first ingestion).
- **Check:** three sources are listed.

## Test 3 - Run ingestion
- **Action:** Swagger -> `POST /api/v1/ingest` -> body `{"sources":["NIMH"]}` -> Execute (then repeat later with `{}` for all sources).
- **Expected:** `discovered` > 0, `new` > 0, `index_rebuilt: true`, `num_chunks` > 0; failures (if any) have categories such as `HTTP_ERROR`.
- **Check:** `data/raw/`, `data/processed/documents/`, `data/processed/chunks.json`, `data/vector_store/` now exist; server log shows progress.

## Test 4 - Trusted-source validation
- **Action:**
```bash
python -c "from app.config import settings; from app.source_manager import load_policy; p=load_policy(settings.sources_file); print(p.is_trusted('https://www.who.int/x'), p.is_trusted('https://who.int.fakewebsite.com'), p.is_trusted('http://www.who.int/x'))"
```
- **Expected:** `True False False`. **Check:** the fake domain that merely contains "who.int" is rejected, and so is plain HTTP.

## Test 5 - HTML extraction
- **Action:** `pytest -k html -v`, then open one `data/processed/documents/nimh_*.json`.
- **Expected:** test passes; the JSON has `sections` with `heading` and `text`, `page: null`.
- **Check:** no menu/footer/script text; wording matches the website (not paraphrased).

## Test 6 - PDF extraction
- **Action:** `pytest -k pdf -v`. For a real PDF, pick any PDF link from a who.int / nimh.nih.gov / nice.org.uk page and call
  `POST /api/v1/ingest` with `{"urls":["<that https PDF URL>"]}`.
- **Expected:** test passes; the real PDF appears as `data/raw/<id>.pdf` and a processed JSON with `"kind": "pdf"`.
- **Check:** text is readable. Scanned (image-only) PDFs give `EMPTY_DOCUMENT` (no OCR).

## Test 7 - PDF page numbers
- **Action:** open the PDF's processed JSON.
- **Expected:** `sections[i].page` = 1, 2, 3...
- **Check:** compare page 1 text with the PDF's first page.

## Test 8 - Chunking
- **Action:** `pytest -k chunk -v`; open `data/processed/chunks.json`.
- **Expected:** each chunk has `chunk_id`, `document_id`, `text`, `source_name`, `domain`, `url`, `title`, `page`, `section`, `topic`; IDs end `_001`, `_002`...
- **Check:** chunks of a long section overlap by ~40 words; PDF chunks keep their `page`.

## Test 9 - FAISS index
- **Action:**
```bash
python -c "import faiss,json; i=faiss.read_index('data/vector_store/index.faiss'); m=json.load(open('data/vector_store/metadata.json')); print(i.ntotal, len(m['chunks']), m['embedding_model'])"
```
- **Expected:** two equal numbers and the model name. Restart the server: log shows no re-embedding and `/health` shows `"index_ready": true`.

## Test 10 - Retrieval query
- **Action:** `POST /api/v1/retrieve` `{"query":"What are common symptoms of anxiety?","top_k":5}`.
- **Expected:** `results` (best first, scores 0.3-1.0) about anxiety, plus `rag_context.context_text` with `[1]`, `[2]`... and `citations`.

## Test 11 - Source metadata
- **Action:** inspect the Test 10 response.
- **Expected:** every result has `source` (WHO/NIMH/NICE), `domain`, `document_id`, `url` (https, trusted domain), `chunk_id`; `page` is `null` for web pages and a number only for PDFs.
- **Check:** open two URLs: the text really appears there. No invented dates or pages.

## Test 12 - Unrelated query
- **Action:** `{"query":"How do I bake a chocolate cake?"}`.
- **Expected:** `results: []`, empty `context_text`, and a `message` saying no trusted content passed the threshold.

## Test 13 - Duplicate / previously processed documents
- **Action:** run the same ingest request as Test 3 again.
- **Expected:** `new: 0`, `unchanged` > 0 (or `updated` if a page changed), `index_rebuilt: false`. The server log shows `DUPLICATE_DOCUMENT ... unchanged`.
- **Extra:** `{"force_rebuild": true}` gives `index_rebuilt: true`.

## Test 14 - Invalid / untrusted URL
- **Action:** `POST /api/v1/ingest` with
  `{"sources":["NIMH"],"urls":["https://who.int.fakewebsite.com/page","http://www.who.int/x","https://evil.com/a.pdf"]}`.
- **Expected:** `failures` contains `SOURCE_NOT_TRUSTED` (fake domain, evil.com) and `INVALID_URL` (http). No request is ever sent to those hosts.
- **Check:** `data/processed/failures.json` lists the same entries; other documents were still processed.

## Bonus - Evaluation
`python -m app.evaluation` prints Precision@K, Recall@K, MRR, HitRate@K, TrustedRate (should be 1.0) and ProvenanceComplete (should be 1.0).
