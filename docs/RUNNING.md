# Running the project

Requires Python 3.11+. The first run downloads the embedding model (~90 MB) and internet is needed for crawling.

## 1. Open the project
```bash
cd trusted-rag
```

## 2. Create a virtual environment
```bash
python -m venv venv
```

## 3. Activate
Windows CMD:
```bash
venv\Scripts\activate
```
PowerShell:
```powershell
venv\Scripts\Activate.ps1
```
(If PowerShell blocks scripts: `Set-ExecutionPolicy -Scope Process Bypass`.)
Linux/macOS: `source venv/bin/activate`

## 4. Upgrade pip
```bash
python -m pip install --upgrade pip
```

## 5. Install dependencies
```bash
pip install -r requirements.txt
```

## 6. Run tests (offline)
```bash
pytest
```

## 7. Start FastAPI
```bash
uvicorn app.main:app --reload
```

## 8. Swagger
Open http://127.0.0.1:8000/docs. Expand an endpoint, click **Try it out**, edit the JSON, click **Execute**.

## 9. Call the two main APIs
**Ingest** (start small with one source):
```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest -H "Content-Type: application/json" -d "{\"sources\": [\"NIMH\"]}"
```
Empty body `{}` ingests all three sources. Crawling takes a minute or more.

**Retrieve:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/retrieve -H "Content-Type: application/json" -d "{\"query\": \"What are common symptoms of anxiety?\", \"top_k\": 5}"
```
PowerShell alternative:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/retrieve -ContentType "application/json" -Body '{"query":"What are common symptoms of anxiety?","top_k":5}'
```

## 10. Rebuild / update the knowledge base
- Update: call `POST /api/v1/ingest` again. Unchanged documents are skipped; the index is rebuilt only if something is new/changed.
- Force a rebuild: body `{"force_rebuild": true}`.
- Full reset: stop the server, delete the `data/` folder, ingest again.
- Evaluate: `python -m app.evaluation`

## Configuration
Limits (pages, depth, timeout, size, chunk size, threshold) are in `app/config.py`, overridable via `.env` (see `.env.example`).
Source URLs are in `config/trusted_sources.json`; if an organisation restructures its site, update `base_url` there.
