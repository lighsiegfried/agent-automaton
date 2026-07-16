"""Document / chunk / evidence models + lifecycle states (Phase 6A). Pure dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# --- ingestion lifecycle -----------------------------------------------------------
QUEUED = "queued"
EXTRACTING = "extracting"
CHUNKING = "chunking"
EMBEDDING = "embedding"
READY = "ready"
FAILED = "failed"
ARCHIVED = "archived"
STATES = frozenset({QUEUED, EXTRACTING, CHUNKING, EMBEDDING, READY, FAILED, ARCHIVED})

# --- supported formats -------------------------------------------------------------
FORMAT_PDF = "pdf"
FORMAT_TXT = "txt"
FORMAT_MD = "md"
FORMAT_DOCX = "docx"
FORMAT_CSV = "csv"
SUPPORTED_FORMATS = frozenset({FORMAT_PDF, FORMAT_TXT, FORMAT_MD, FORMAT_DOCX, FORMAT_CSV})

# --- collections (item 8) ----------------------------------------------------------
DEFAULT_COLLECTIONS = ("projects", "contracts", "policies", "technical", "notes")


@dataclass
class Block:
    """One extracted structural unit, carrying its location metadata."""
    text: str
    page: int | None = None
    heading: str | None = None
    section: str | None = None
    row: int | None = None


@dataclass
class ExtractedDoc:
    title: str
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0

    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    owner: str
    position: int
    text: str
    page: int | None = None
    heading: str | None = None
    section: str | None = None
    row: int | None = None
    char_start: int = 0
    char_end: int = 0
    embedding: list[float] = field(default_factory=list)

    def location(self) -> str:
        if self.page is not None:
            return f"page {self.page}"
        if self.section:
            return f"section “{self.section}”"
        if self.heading:
            return f"“{self.heading}”"
        if self.row is not None:
            return f"row {self.row}"
        return f"chunk {self.position}"

    def to_row(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "owner": self.owner,
                "position": self.position, "text": self.text, "page": self.page,
                "heading": self.heading, "section": self.section, "row": self.row,
                "char_start": self.char_start, "char_end": self.char_end,
                "embedding": json.dumps(self.embedding)}

    @classmethod
    def from_row(cls, row) -> "Chunk":
        d = dict(row)
        return cls(chunk_id=d["chunk_id"], doc_id=d["doc_id"], owner=d["owner"],
                   position=d["position"], text=d["text"], page=d["page"], heading=d["heading"],
                   section=d["section"], row=d["row"], char_start=d["char_start"],
                   char_end=d["char_end"], embedding=json.loads(d["embedding"] or "[]"))


@dataclass
class Document:
    doc_id: str
    owner: str
    filename: str
    title: str
    sha256: str
    format: str
    size_bytes: int = 0
    page_count: int = 0
    chunk_count: int = 0
    status: str = QUEUED
    collections: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str = "upload"
    injection_flags: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_reason: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_row(self) -> dict:
        return {"doc_id": self.doc_id, "owner": self.owner, "filename": self.filename,
                "title": self.title, "sha256": self.sha256, "format": self.format,
                "size_bytes": self.size_bytes, "page_count": self.page_count,
                "chunk_count": self.chunk_count, "status": self.status,
                "collections": json.dumps(self.collections, ensure_ascii=False),
                "tags": json.dumps(self.tags, ensure_ascii=False), "source": self.source,
                "injection_flags": json.dumps(self.injection_flags),
                "error_code": self.error_code, "error_reason": self.error_reason,
                "created_at": self.created_at, "updated_at": self.updated_at}

    @classmethod
    def from_row(cls, row) -> "Document":
        d = dict(row)
        return cls(doc_id=d["doc_id"], owner=d["owner"], filename=d["filename"], title=d["title"],
                   sha256=d["sha256"], format=d["format"], size_bytes=d["size_bytes"],
                   page_count=d["page_count"], chunk_count=d["chunk_count"], status=d["status"],
                   collections=json.loads(d["collections"] or "[]"), tags=json.loads(d["tags"] or "[]"),
                   source=d["source"], injection_flags=json.loads(d["injection_flags"] or "[]"),
                   error_code=d["error_code"], error_reason=d["error_reason"],
                   created_at=d["created_at"], updated_at=d["updated_at"])

    def public(self) -> dict:
        return {"doc_id": self.doc_id, "filename": self.filename, "title": self.title,
                "sha256": self.sha256[:16], "format": self.format, "size_bytes": self.size_bytes,
                "page_count": self.page_count, "chunk_count": self.chunk_count, "status": self.status,
                "collections": list(self.collections), "tags": list(self.tags),
                "injection_flagged": bool(self.injection_flags),
                "error_code": self.error_code, "created_at": self.created_at,
                "updated_at": self.updated_at}


@dataclass
class Evidence:
    doc_id: str
    filename: str
    chunk_id: str
    location: str
    excerpt: str
    score: float
    page: int | None = None
    section: str | None = None

    def public(self) -> dict:
        return {"doc_id": self.doc_id, "filename": self.filename, "chunk_id": self.chunk_id,
                "location": self.location, "page": self.page, "section": self.section,
                "excerpt": self.excerpt, "score": round(self.score, 4)}

    def citation(self) -> str:
        return f"{self.filename} ({self.location})"
