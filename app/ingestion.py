"""Phases 2, 3 and 10.

Phase 2  - discover, download, detect type, extract, clean, save documents
Phase 3  - split processed documents into chunks that keep provenance
Phase 10 - manifest + content hashes so unchanged documents are skipped
"""
import hashlib
import io
import json
import logging
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from pypdf import PdfReader

from app.config import Settings
from app.source_manager import ErrorCode, PipelineError, TrustedSource, TrustPolicy

log = logging.getLogger("trusted_rag.ingestion")
NOISE_TAGS = ["script", "style", "nav", "footer", "aside", "form", "noscript", "svg", "iframe", "button"]


# ------------------------------------------------------------------ models / helpers
class Section(BaseModel):
    page: int | None = None      # PDF page number (never invented for HTML)
    heading: str | None = None   # nearest HTML heading
    text: str


class Failure(BaseModel):
    url: str
    code: str
    message: str


class IngestReport(BaseModel):
    discovered: int = 0       # documents downloaded and examined
    new: int = 0
    updated: int = 0
    unchanged: int = 0        # already processed, content identical
    rejected_links: int = 0   # discovered links that failed trust validation
    failed: int = 0
    failures: list[Failure] = []

    def fail(self, url: str, err: PipelineError) -> None:
        self.failed += 1
        self.failures.append(Failure(url=url, code=err.code.value, message=err.message))
        log.warning("%s | %s | %s", err.code.value, url, err.message)


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm(text: str) -> str:
    return re.sub(r"[-_/]+", " ", text.lower())


def is_relevant(text: str, keywords: list[str]) -> bool:
    """True if any topic keyword appears in the text (hyphens/slashes ignored)."""
    haystack = _norm(text)
    return any(_norm(k) in haystack for k in keywords)


# ------------------------------------------------------------------ Phase 2: download
def detect_kind(content_type: str, body: bytes) -> str:
    if body[:5] == b"%PDF-" or content_type == "application/pdf":
        return "pdf"
    if content_type in ("text/html", "application/xhtml+xml"):
        return "html"
    raise PipelineError(ErrorCode.UNSUPPORTED_DOCUMENT_TYPE, f"content-type {content_type!r}")


def fetch_document(url: str, policy: TrustPolicy, s: Settings) -> tuple[bytes, str]:
    """Download one URL. Redirects are followed manually so EVERY hop is re-validated."""
    current = url
    for _ in range(s.max_redirects + 1):
        policy.validate_url(current)
        try:
            with requests.get(current, stream=True, timeout=s.request_timeout,
                              allow_redirects=False, headers={"User-Agent": s.user_agent}) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        raise PipelineError(ErrorCode.HTTP_ERROR, "redirect without Location header")
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 400:
                    raise PipelineError(ErrorCode.HTTP_ERROR, f"HTTP {resp.status_code}")
                declared = resp.headers.get("Content-Length", "")
                if declared.isdigit() and int(declared) > s.max_document_bytes:
                    raise PipelineError(ErrorCode.DOCUMENT_TOO_LARGE, f"{declared} bytes")
                body = bytearray()
                for part in resp.iter_content(chunk_size=65536):
                    body.extend(part)
                    if len(body) > s.max_document_bytes:
                        raise PipelineError(ErrorCode.DOCUMENT_TOO_LARGE, "size limit exceeded")
                ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                return bytes(body), detect_kind(ctype, bytes(body))
        except requests.RequestException as exc:
            raise PipelineError(ErrorCode.NETWORK_ERROR, str(exc))
    raise PipelineError(ErrorCode.NETWORK_ERROR, "too many redirects")


def extract_links(html: bytes, base_url: str) -> Iterator[tuple[str, str]]:
    """Yield (absolute_url_without_query_or_fragment, anchor_text)."""
    for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
        parsed = urlparse(urljoin(base_url, a["href"]))._replace(query="", fragment="")
        yield parsed.geturl(), a.get_text(" ", strip=True)


def crawl_source(source: TrustedSource, policy: TrustPolicy, s: Settings,
                 report: IngestReport) -> Iterator[tuple[str, bytes, str]]:
    """Breadth-first crawl of ONE trusted source. Yields (url, bytes, kind) for relevant documents.

    Limits: max_pages_per_source, max_depth. Only links that pass the trust policy,
    stay on the same source and look mental-health related are followed.
    """
    queue = deque([(source.base_url, 0)])
    seen = {source.base_url}
    attempts = 0
    while queue and attempts < s.max_pages_per_source:
        url, depth = queue.popleft()
        attempts += 1
        try:
            data, kind = fetch_document(url, policy, s)
        except PipelineError as err:
            report.fail(url, err)
            continue
        if kind not in source.allowed_document_types:
            report.fail(url, PipelineError(ErrorCode.UNSUPPORTED_DOCUMENT_TYPE, f"{kind} not allowed"))
            continue
        if is_relevant(url, source.topics):  # the hub/base page itself is only a document if relevant
            yield url, data, kind
        if kind == "html" and depth < s.max_depth:
            for link, text in extract_links(data, url):
                if link in seen:
                    continue
                seen.add(link)
                try:
                    linked = policy.validate_url(link)
                except PipelineError:
                    report.rejected_links += 1  # untrusted/http/mailto etc.
                    log.info("rejected link: %s", link)
                    continue
                if linked.name == source.name and is_relevant(f"{link} {text}", source.topics):
                    queue.append((link, depth + 1))


# ------------------------------------------------------------------ Phase 2: extraction + cleaning
def clean_text(text: str) -> str:
    """Formatting-only cleanup. The wording of the source is never changed."""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _close_section(sections: list[Section], heading: str | None, paragraphs: list[str]) -> None:
    text = clean_text("\n".join(paragraphs))
    if text:
        sections.append(Section(heading=heading, text=text))


def extract_html(data: bytes) -> tuple[str, list[Section]]:
    """Return (title, sections). A section = a heading plus the paragraphs under it."""
    soup = BeautifulSoup(data, "lxml")
    title = clean_text(soup.title.get_text(" ")) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in root.find_all(NOISE_TAGS):
        tag.decompose()
    sections: list[Section] = []
    heading, paragraphs = None, []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = clean_text(el.get_text(" ", strip=True))
        if not text:
            continue
        if el.name.startswith("h"):
            _close_section(sections, heading, paragraphs)
            heading, paragraphs = text, []
        elif el.name == "li" and el.find(["p", "li"]):
            continue  # its inner <p>/<li> are collected separately (avoids duplicates)
        else:
            paragraphs.append(text)
    _close_section(sections, heading, paragraphs)
    return title, sections


def extract_pdf(data: bytes, fallback_title: str) -> tuple[str, list[Section]]:
    """Extract text page by page; page numbers start at 1."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PipelineError(ErrorCode.EXTRACTION_ERROR, "PDF is encrypted")
        meta_title = reader.metadata.title if reader.metadata and reader.metadata.title else ""
        sections = [Section(page=n, text=clean_text(page.extract_text() or ""))
                    for n, page in enumerate(reader.pages, start=1)]
        return meta_title or fallback_title, sections
    except PipelineError:
        raise
    except Exception as exc:  # pypdf raises many different exceptions
        raise PipelineError(ErrorCode.EXTRACTION_ERROR, str(exc))


def make_document_id(source: TrustedSource, url: str) -> str:
    """Stable ID: same URL -> same ID, so a re-crawl finds the previous version."""
    slug = re.sub(r"[^a-z0-9]+", "-", urlparse(url).path.rstrip("/").split("/")[-1].lower()).strip("-")[:30]
    prefix = re.sub(r"\W+", "", source.name.lower())
    return f"{prefix}_{slug or 'index'}_{hashlib.sha1(url.encode()).hexdigest()[:6]}"


def detect_topic(text: str, keywords: list[str], s: Settings) -> str:
    """Most frequent specific topic keyword in the document, else 'general'."""
    haystack = _norm(text)
    specific = [k for k in keywords if k.lower() not in s.generic_topics]
    counts = {k: haystack.count(_norm(k)) for k in specific}
    best = max(counts, key=counts.get) if counts else None
    return best if best and counts[best] > 0 else "general"


def process_document(url: str, source: TrustedSource, data: bytes, kind: str,
                     manifest: dict, s: Settings) -> str:
    """Extract, clean, hash and save one document. Returns 'new', 'updated' or 'unchanged'."""
    if kind == "pdf":
        title, sections = extract_pdf(data, url.rsplit("/", 1)[-1])
    else:
        title, sections = extract_html(data)
    sections = [sec for sec in sections if sec.text]
    if sum(len(sec.text.split()) for sec in sections) < s.min_document_words:
        raise PipelineError(ErrorCode.EMPTY_DOCUMENT, "too little text extracted")

    doc_id = make_document_id(source, url)
    content_hash = hashlib.sha256("\n".join(sec.text for sec in sections).encode("utf-8")).hexdigest()

    # Phase 10: change detection
    previous = manifest.get(doc_id)
    if previous and previous["content_hash"] == content_hash:
        log.info("%s | %s | already processed, content unchanged", ErrorCode.DUPLICATE_DOCUMENT.value, url)
        return "unchanged"
    twin = next((d for d, m in manifest.items() if m["content_hash"] == content_hash and d != doc_id), None)
    if twin:
        log.info("%s | %s | same content as %s", ErrorCode.DUPLICATE_DOCUMENT.value, url, twin)
        return "unchanged"

    s.ensure_dirs()
    (s.raw_dir / f"{doc_id}.{'pdf' if kind == 'pdf' else 'html'}").write_bytes(data)
    sample = f"{title} {url} " + " ".join(sec.text for sec in sections)[:3000]
    now = datetime.now(timezone.utc).isoformat()
    write_json(s.documents_dir / f"{doc_id}.json", {
        "document_id": doc_id, "source_name": source.name, "domain": source.domain,
        "url": url, "title": title or doc_id, "kind": kind,
        "topic": detect_topic(sample, source.topics, s),
        "content_hash": content_hash, "fetched_at": now,
        "sections": [sec.model_dump() for sec in sections],
    })
    manifest[doc_id] = {"url": url, "content_hash": content_hash, "updated_at": now}
    return "updated" if previous else "new"


def run_ingestion(policy: TrustPolicy, s: Settings, source_names: list[str] | None = None,
                  urls: list[str] | None = None) -> IngestReport:
    """Phases 2 + 10: discover, download and process documents; one failure never stops the run."""
    s.ensure_dirs()
    report = IngestReport()
    manifest = read_json(s.manifest_file, {})
    wanted = {n.lower() for n in source_names or []}

    def candidates() -> Iterator[tuple[str, TrustedSource, bytes, str]]:
        for source in policy.sources:
            if wanted and source.name.lower() not in wanted:
                continue
            for url, data, kind in crawl_source(source, policy, s, report):
                yield url, source, data, kind
        for url in urls or []:  # user-supplied URLs are never trusted blindly
            try:
                source = policy.validate_url(url)
                data, kind = fetch_document(url, policy, s)
            except PipelineError as err:
                report.fail(url, err)
                continue
            yield url, source, data, kind

    for url, source, data, kind in candidates():
        report.discovered += 1
        try:
            status = process_document(url, source, data, kind, manifest, s)
            setattr(report, status, getattr(report, status) + 1)
        except PipelineError as err:
            report.fail(url, err)
        except Exception as exc:  # last safety net: keep processing other documents
            report.fail(url, PipelineError(ErrorCode.EXTRACTION_ERROR, str(exc)))

    write_json(s.manifest_file, manifest)
    write_json(s.failures_file, [f.model_dump() for f in report.failures])
    log.info("ingestion finished: %s", report.model_dump(exclude={"failures"}))
    return report


# ------------------------------------------------------------------ Phase 3: chunking
def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Sliding word window with overlap."""
    words = text.split()
    step = max(size - overlap, 1)
    pieces, i = [], 0
    while i < len(words):
        pieces.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += step
    return pieces


def chunk_document(doc: dict, s: Settings) -> list[dict]:
    """Chunk each section separately so page/heading provenance is never lost."""
    chunks, n = [], 0
    for sec in doc["sections"]:
        for piece in split_words(sec["text"], s.chunk_size, s.chunk_overlap):
            if len(piece.split()) < s.min_chunk_words:
                continue
            n += 1
            chunks.append({
                "chunk_id": f"{doc['document_id']}_{n:03d}",
                "document_id": doc["document_id"],
                "text": piece,
                "source_name": doc["source_name"],
                "domain": doc["domain"],
                "url": doc["url"],
                "title": doc["title"],
                "page": sec.get("page"),
                "section": sec.get("heading"),
                "topic": doc["topic"],
            })
    return chunks


def build_chunks(s: Settings) -> list[dict]:
    """Chunk every processed document and save data/processed/chunks.json."""
    chunks: list[dict] = []
    for path in sorted(s.documents_dir.glob("*.json")):
        chunks.extend(chunk_document(read_json(path, {}), s))
    write_json(s.chunks_file, chunks)
    return chunks
