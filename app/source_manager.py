"""Phase 1 - Trusted source management.

Answers one question deterministically: "Can this URL be trusted?"
No LLM is involved; trust comes only from config/trusted_sources.json.
All other modules reuse TrustPolicy instead of re-implementing validation.
"""
import json
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Error categories shared by the whole pipeline."""
    SOURCE_NOT_TRUSTED = "SOURCE_NOT_TRUSTED"
    INVALID_URL = "INVALID_URL"
    NETWORK_ERROR = "NETWORK_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    UNSUPPORTED_DOCUMENT_TYPE = "UNSUPPORTED_DOCUMENT_TYPE"
    DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
    EXTRACTION_ERROR = "EXTRACTION_ERROR"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"
    DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"


class PipelineError(Exception):
    """A failure that affects one URL/document but must not stop the pipeline."""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message


class TrustedSource(BaseModel):
    name: str
    domain: str  # lowercase registrable domain, e.g. "who.int"
    base_url: str
    allowed_document_types: list[str] = ["html", "pdf"]
    topics: list[str] = []


class TrustPolicy:
    """Whitelist of trusted sources plus strict URL validation."""

    def __init__(self, sources: list[TrustedSource]):
        self.sources = sources

    def validate_url(self, url: str) -> TrustedSource:
        """Return the matching trusted source or raise PipelineError."""
        try:
            parsed = urlparse(url.strip())
            host, port = parsed.hostname, parsed.port
        except ValueError:
            raise PipelineError(ErrorCode.INVALID_URL, f"cannot parse {url!r}")
        if parsed.scheme != "https":
            raise PipelineError(ErrorCode.INVALID_URL, "only https URLs are allowed")
        if not host:
            raise PipelineError(ErrorCode.INVALID_URL, "URL has no hostname")
        if parsed.username or parsed.password:
            raise PipelineError(ErrorCode.INVALID_URL, "credentials in URL are not allowed")
        if port not in (None, 443):
            raise PipelineError(ErrorCode.INVALID_URL, f"port {port} is not allowed")
        # Exact host match OR a real sub-domain. "who.int.fakewebsite.com" and
        # "fakewho.int" both fail because the host must END with ".who.int".
        for source in self.sources:
            if host == source.domain or host.endswith("." + source.domain):
                return source
        raise PipelineError(ErrorCode.SOURCE_NOT_TRUSTED, f"{host} is not a whitelisted domain")

    def is_trusted(self, url: str) -> bool:
        try:
            self.validate_url(url)
            return True
        except PipelineError:
            return False

    def get_source(self, name: str) -> TrustedSource:
        for source in self.sources:
            if source.name.lower() == name.lower():
                return source
        raise KeyError(name)


def load_policy(path: Path) -> TrustPolicy:
    """Load the JSON whitelist and check that each base_url obeys its own rules."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    policy = TrustPolicy([TrustedSource(**item) for item in raw["sources"]])
    for source in policy.sources:
        if policy.validate_url(source.base_url).name != source.name:
            raise ValueError(f"base_url of {source.name} is not on its own domain")
    return policy
