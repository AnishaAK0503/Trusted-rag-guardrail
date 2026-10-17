"""Offline tests: no internet and no model download (embeddings are faked)."""
import dataclasses
import re
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import evaluation, ingestion, rag
from app.config import settings
from app.main import app
from app.source_manager import ErrorCode, PipelineError, load_policy

policy = load_policy(settings.sources_file)
LONG = " ".join(["Anxiety disorders involve excessive fear and worry that affect daily life."] * 8)
HTML = f"""<html><head><title>Anxiety Disorders | NIMH</title></head><body>
<script>var tracking = 1;</script><nav><ul><li>Home menu item</li></ul></nav>
<h1>Anxiety Disorders</h1><p>{LONG}</p>
<h2>Symptoms</h2><p>Common symptoms include restlessness and trouble sleeping.</p><ul><li>Feeling nervous</li></ul>
<footer><p>Copyright footer text</p></footer></body></html>""".encode()


@pytest.fixture(autouse=True)
def clean_cache():
    rag._cache.clear()
    yield
    rag._cache.clear()


@pytest.fixture
def s(tmp_path):
    return dataclasses.replace(settings, data_dir=tmp_path)


def make_pdf(pages: list[str]) -> bytes:
    """Tiny valid PDF generator (no extra library needed). Text must not contain parentheses."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def fake_embed(texts):
    """Deterministic bag-of-words vectors so retrieval can be tested without a model."""
    out = np.zeros((len(texts), 512), dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            out[row, zlib.crc32(word.encode()) % 512] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


def make_chunk(cid, text, url, source, domain, topic, page=None):
    return {"chunk_id": cid, "document_id": cid.rsplit("_", 1)[0], "text": text, "source_name": source,
            "domain": domain, "url": url, "title": "T", "page": page, "section": None, "topic": topic}


# ---------------------------------------------------------------- Phase 1
@pytest.mark.parametrize("url", ["https://who.int", "https://www.who.int/x", "https://WWW.WHO.INT/x",
                                 "https://www.nimh.nih.gov/health/topics", "https://www.nice.org.uk/guidance/ng222"])
def test_trust_policy_accepts_whitelisted_hosts(url):
    assert policy.is_trusted(url)


@pytest.mark.parametrize("url,code", [
    ("https://who.int.fakewebsite.com", ErrorCode.SOURCE_NOT_TRUSTED),
    ("https://fakewho.int/page", ErrorCode.SOURCE_NOT_TRUSTED),
    ("https://evil.com/?next=who.int", ErrorCode.SOURCE_NOT_TRUSTED),
    ("https://who.int@evil.com/", ErrorCode.INVALID_URL),
    ("http://www.who.int/health-topics", ErrorCode.INVALID_URL),
    ("ftp://www.who.int/file", ErrorCode.INVALID_URL),
    ("https://www.who.int:8443/x", ErrorCode.INVALID_URL),
    ("not a url", ErrorCode.INVALID_URL),
])
def test_trust_policy_rejects_untrusted_urls(url, code):
    with pytest.raises(PipelineError) as err:
        policy.validate_url(url)
    assert err.value.code == code


# ---------------------------------------------------------------- Phase 2
def test_html_extraction_removes_noise_and_keeps_headings():
    title, sections = ingestion.extract_html(HTML)
    text = " ".join(sec.text for sec in sections)
    assert title == "Anxiety Disorders | NIMH"
    assert [sec.heading for sec in sections] == ["Anxiety Disorders", "Symptoms"]
    assert "excessive fear and worry" in text and "Feeling nervous" in text
    assert "Home menu item" not in text and "Copyright" not in text and "tracking" not in text


def test_pdf_extraction_keeps_page_numbers():
    title, sections = ingestion.extract_pdf(make_pdf(["First page alpha", "Second page beta"]), "file.pdf")
    assert [sec.page for sec in sections] == [1, 2]
    assert "alpha" in sections[0].text and "beta" in sections[1].text


def test_crawl_stays_on_trusted_domain(monkeypatch, s):
    base = "https://www.nimh.nih.gov/health/topics"
    anxiety = "https://www.nimh.nih.gov/health/topics/anxiety-disorders"
    site = {
        base: b'<a href="/health/topics/anxiety-disorders">Anxiety disorders</a>'
              b'<a href="https://evil.com/anxiety">Anxiety</a>'
              b'<a href="http://www.nimh.nih.gov/health/topics/depression">Depression</a>'
              b'<a href="/health/topics/sports">Sports news</a>',
        anxiety: HTML,
    }

    def fake_fetch(url, pol, settings_):
        if url in site:
            return site[url], "html"
        raise PipelineError(ErrorCode.NETWORK_ERROR, "not found")

    monkeypatch.setattr(ingestion, "fetch_document", fake_fetch)
    report = ingestion.IngestReport()
    docs = list(ingestion.crawl_source(policy.get_source("NIMH"), policy, s, report))
    assert [d[0] for d in docs] == [anxiety]
    assert report.rejected_links == 2 and report.failed == 0


def test_duplicate_and_update_detection(s):
    url = "https://www.nimh.nih.gov/health/topics/anxiety-disorders"
    source, manifest = policy.get_source("NIMH"), {}
    assert ingestion.process_document(url, source, HTML, "html", manifest, s) == "new"
    assert ingestion.process_document(url, source, HTML, "html", manifest, s) == "unchanged"
    changed = HTML.replace(b"excessive fear", b"intense fear")
    assert ingestion.process_document(url, source, changed, "html", manifest, s) == "updated"
    assert len(list(s.documents_dir.glob("*.json"))) == 1 and len(list(s.raw_dir.glob("*.html"))) == 1


# ---------------------------------------------------------------- Phase 3
def test_chunking_keeps_provenance_and_overlap(s):
    doc = {"document_id": "nimh_x_abc123", "source_name": "NIMH", "domain": "nimh.nih.gov",
           "url": "https://www.nimh.nih.gov/x", "title": "X", "topic": "anxiety",
           "sections": [{"page": 2, "heading": "H", "text": " ".join(f"w{i}" for i in range(450))}]}
    chunks = ingestion.chunk_document(doc, dataclasses.replace(s, chunk_size=200, chunk_overlap=40))
    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == "nimh_x_abc123_001"
    assert chunks[0]["text"].split()[-40:] == chunks[1]["text"].split()[:40]
    assert all(c["page"] == 2 and c["url"] == doc["url"] and c["source_name"] == "NIMH" for c in chunks)


# ---------------------------------------------------------------- Phases 4-8
def build_test_index(s, monkeypatch):
    monkeypatch.setattr(rag, "settings", s)
    monkeypatch.setattr(rag, "embed", fake_embed)
    nimh, who = "https://www.nimh.nih.gov/a", "https://www.who.int/b"
    chunks = [
        make_chunk("nimh_anx_001", "Anxiety symptoms include worry restlessness and trouble sleeping", nimh, "NIMH", "nimh.nih.gov", "anxiety", page=2),
        make_chunk("nimh_anx_002", "Anxiety symptoms include worry restlessness and trouble sleeping", nimh, "NIMH", "nimh.nih.gov", "anxiety"),
        make_chunk("who_dep_001", "Depression symptoms include low mood and loss of interest", who, "WHO", "who.int", "depression"),
        make_chunk("evil_001", "Anxiety symptoms cured by miracle pills buy now cheap", "https://evil.example.com/x", "EVIL", "evil.example.com", "anxiety"),
    ]
    assert rag.build_index(chunks) == 4


def test_retrieval_filtering_and_provenance(s, monkeypatch):
    build_test_index(s, monkeypatch)
    results = rag.retrieve("anxiety symptoms", 5, None, policy)
    assert len(results) == 1                                  # duplicate + untrusted chunk removed
    top = results[0]
    assert top.chunk_id in {"nimh_anx_001", "nimh_anx_002"} and top.source == "NIMH"
    assert top.url.startswith("https://") and top.score >= settings.similarity_threshold
    assert [r.chunk_id for r in rag.retrieve("depression symptoms", 5, "depression", policy)] == ["who_dep_001"]
    assert rag.retrieve("depression symptoms", 5, "anxiety", policy) == []   # topic filter
    assert rag.retrieve("chocolate cake recipe", 5, None, policy) == []      # unrelated query


def test_index_persistence_and_context(s, monkeypatch):
    build_test_index(s, monkeypatch)
    assert s.index_file.exists() and s.meta_file.exists()
    rag._cache.clear()
    assert rag.load_index() and len(rag.get_chunks()) == 4     # loaded from disk, no re-embedding
    results = rag.retrieve("depression symptoms", 3, None, policy)
    context = rag.build_context("depression symptoms", results)
    assert context.context_text.startswith("[1] WHO")
    assert context.citations[0].chunk_id == "who_dep_001" and context.citations[0].page is None


def test_index_not_ready(s, monkeypatch):
    monkeypatch.setattr(rag, "settings", s)
    assert rag.load_index() is False
    with pytest.raises(rag.IndexNotReadyError):
        rag.retrieve("anxiety symptoms", 5, None, policy)


# ---------------------------------------------------------------- Phase 9
def test_evaluation_metrics():
    assert evaluation.precision_at_k([True, False, True], 3) == pytest.approx(2 / 3)
    assert evaluation.reciprocal_rank([False, True]) == 0.5
    assert evaluation.reciprocal_rank([False, False]) == 0.0
    assert evaluation.hit_rate_at_k([False, True], 2) == 1.0


# ---------------------------------------------------------------- API
def test_api_endpoints(s, monkeypatch):
    monkeypatch.setattr(rag, "settings", s)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/api/v1/retrieve", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/retrieve", json={"query": "anxiety symptoms"}).status_code == 409
    assert client.post("/api/v1/ingest", json={"sources": ["UNKNOWN"]}).status_code == 400
