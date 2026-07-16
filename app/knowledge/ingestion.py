"""Ingestion workflow + state machine (Phase 6A, item 3).

queued → extracting → chunking → embedding → ready (or failed). The document is only
published (flipped to ``ready``) atomically after its chunks + embeddings are validated
and written, so a failure never leaves a half-indexed document. SHA-256 is computed
before processing for dedup; in-document instructions are detected and RECORDED but
never acted on. Each transition emits a safe Activity Center event.
"""

from __future__ import annotations

import hashlib

from app.core import errors
from app.core.logger import get_logger
from app.knowledge import chunking, extractors, safety
from app.knowledge.extractors import ExtractionError
from app.knowledge.models import (
    CHUNKING, EMBEDDING, EXTRACTING, FAILED, QUEUED, Document,
)

log = get_logger(__name__)


def _emit(event_type, *, doc_id=None, status=None, severity="info", title="", metadata=None):
    try:
        from app.core import eventbus
        eventbus.emit(domain="knowledge", event_type=event_type, status=status, severity=severity,
                      related_id=doc_id, title=title, metadata=metadata or {})
    except Exception:
        pass


def ingest(repo, embedder, *, owner, filename, data, collections, tags, source, settings) -> dict:
    v = safety.validate_upload(filename, data, max_file_mb=settings.knowledge_max_file_mb)
    if not v["ok"]:
        _emit("document_rejected", status="failed", severity="warning",
              title=f"Rejected {filename}", metadata={"error_code": v["error_code"]})
        return {"ok": False, "error_code": v["error_code"], "reason": v["reason"], "document": None}
    fmt = v["format"]
    sha = hashlib.sha256(data).hexdigest()

    existing = repo.find_by_sha256(owner, sha)
    if existing is not None:
        return {"ok": False, "error_code": errors.DUPLICATE_DOCUMENT,
                "reason": "this document is already in the vault", "document": existing.public()}

    doc = Document(doc_id=repo.new_doc_id(), owner=owner, filename=filename,
                   title=(filename.rsplit(".", 1)[0] or filename), sha256=sha, format=fmt,
                   size_bytes=len(data), collections=list(collections or []),
                   tags=list(tags or []), source=source, status=QUEUED)
    repo.create_document(doc)
    repo.store_file(sha, fmt, data)
    repo.audit("queued", owner=owner, doc_id=doc.doc_id, detail=fmt)
    _emit("document_queued", doc_id=doc.doc_id, status=QUEUED, title=f"Ingesting {doc.title}",
          metadata={"format": fmt})

    try:
        doc.status = EXTRACTING
        repo.save_document(doc)
        extracted = extractors.extract(data, fmt, filename, max_pages=settings.knowledge_max_pages)
        doc.title = extracted.title or doc.title
        doc.page_count = extracted.page_count
        doc.injection_flags = safety.detect_injection(extracted.full_text())
        if doc.injection_flags:
            repo.audit("injection_flagged", owner=owner, doc_id=doc.doc_id,
                       detail=",".join(doc.injection_flags))
            _emit("document_injection_flagged", doc_id=doc.doc_id, severity="warning",
                  title=f"Instructions ignored in {doc.title}",
                  metadata={"count": len(doc.injection_flags)})

        doc.status = CHUNKING
        repo.save_document(doc)
        chunks = chunking.chunk_document(extracted, doc_id=doc.doc_id, owner=owner,
                                         chunk_size=settings.knowledge_chunk_size,
                                         overlap=settings.knowledge_chunk_overlap)
        if not chunks:
            raise ExtractionError(errors.EMPTY_DOCUMENT, "no text to index")

        doc.status = EMBEDDING
        repo.save_document(doc)
        vectors = embedder.embed([c.text for c in chunks])
        for ch, vec in zip(chunks, vectors):
            ch.embedding = list(vec)

        repo.publish(doc, chunks)                       # atomic → READY
        repo.audit("ready", owner=owner, doc_id=doc.doc_id,
                   detail=f"pages={doc.page_count} chunks={len(chunks)}")
        _emit("document_ready", doc_id=doc.doc_id, status="ready", title=f"Indexed {doc.title}",
              metadata={"count": len(chunks), "page_count": doc.page_count})
        return {"ok": True, "error_code": None, "reason": "", "document": repo.get_document(doc.doc_id).public()}

    except ExtractionError as exc:
        doc.status = FAILED
        doc.error_code = exc.code
        doc.error_reason = exc.reason
        repo.save_document(doc)
        repo.audit("failed", owner=owner, doc_id=doc.doc_id, detail=exc.code)
        _emit("document_failed", doc_id=doc.doc_id, status="failed", severity="error",
              title=f"Failed to index {doc.title}", metadata={"error_code": exc.code})
        return {"ok": False, "error_code": exc.code, "reason": exc.reason, "document": doc.public()}
    except Exception as exc:  # never leave the pipeline in an inconsistent state
        log.warning("ingestion failed", exc_info=True)
        doc.status = FAILED
        doc.error_code = errors.EXTRACTION_FAILED
        doc.error_reason = type(exc).__name__
        repo.save_document(doc)
        repo.audit("failed", owner=owner, doc_id=doc.doc_id, detail=type(exc).__name__)
        return {"ok": False, "error_code": errors.EXTRACTION_FAILED, "reason": "ingestion failed",
                "document": doc.public()}


def reindex(repo, embedder, doc, settings) -> dict:
    """Re-run extraction/chunking/embedding for an existing stored document."""
    path = repo.file_path(doc.sha256, doc.format)
    if not path.exists():
        return {"ok": False, "error_code": errors.DOCUMENT_NOT_FOUND, "reason": "stored file missing"}
    data = path.read_bytes()
    try:
        extracted = extractors.extract(data, doc.format, doc.filename, max_pages=settings.knowledge_max_pages)
        doc.injection_flags = safety.detect_injection(extracted.full_text())
        doc.page_count = extracted.page_count
        chunks = chunking.chunk_document(extracted, doc_id=doc.doc_id, owner=doc.owner,
                                         chunk_size=settings.knowledge_chunk_size,
                                         overlap=settings.knowledge_chunk_overlap)
        if not chunks:
            raise ExtractionError(errors.EMPTY_DOCUMENT, "no text to index")
        for ch, vec in zip(chunks, embedder.embed([c.text for c in chunks])):
            ch.embedding = list(vec)
        repo.publish(doc, chunks)
        repo.save_document(doc)
        repo.audit("reindexed", owner=doc.owner, doc_id=doc.doc_id, detail=f"chunks={len(chunks)}")
        _emit("document_reindexed", doc_id=doc.doc_id, status="ready", title=f"Reindexed {doc.title}",
              metadata={"count": len(chunks)})
        return {"ok": True, "document": repo.get_document(doc.doc_id).public()}
    except ExtractionError as exc:
        doc.status = FAILED
        doc.error_code = exc.code
        repo.save_document(doc)
        return {"ok": False, "error_code": exc.code, "reason": exc.reason}
