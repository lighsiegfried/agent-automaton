"""SQLite persistence + FTS5 + local file store for the Knowledge Vault (Phase 6A).

Metadata and chunks (with their local embeddings) live in a WAL SQLite database with
schema migrations, owner scope, an audit trail, and corruption recovery. Chunk text is
mirrored into an FTS5 index for lexical search. Original files are stored content-
addressed under ``files/`` (deduped by sha256). Publishing is atomic: a document only
becomes ``ready`` when its chunks + FTS rows are written and its status flipped in one
transaction, so a failure never leaves a half-indexed document.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.core.logger import get_logger
from app.knowledge.models import ARCHIVED, READY, Chunk, Document

log = get_logger(__name__)

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    owner          TEXT NOT NULL,
    filename       TEXT NOT NULL,
    title          TEXT NOT NULL,
    sha256         TEXT NOT NULL,
    format         TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL DEFAULT 0,
    page_count     INTEGER NOT NULL DEFAULT 0,
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'queued',
    collections    TEXT NOT NULL DEFAULT '[]',
    tags           TEXT NOT NULL DEFAULT '[]',
    source         TEXT NOT NULL DEFAULT 'upload',
    injection_flags TEXT NOT NULL DEFAULT '[]',
    error_code     TEXT,
    error_reason   TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_owner_status ON documents(owner, status);
CREATE INDEX IF NOT EXISTS idx_docs_sha ON documents(owner, sha256);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    owner       TEXT NOT NULL,
    position    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    page        INTEGER,
    heading     TEXT,
    section     TEXT,
    row         INTEGER,
    char_start  INTEGER NOT NULL DEFAULT 0,
    char_end    INTEGER NOT NULL DEFAULT 0,
    embedding   TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_owner ON chunks(owner);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, doc_id UNINDEXED, text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS knowledge_audit (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    owner   TEXT,
    doc_id  TEXT,
    event   TEXT NOT NULL,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_know_audit ON knowledge_audit(doc_id, id);
"""

_DOC_COLS = ("doc_id, owner, filename, title, sha256, format, size_bytes, page_count, "
             "chunk_count, status, collections, tags, source, injection_flags, error_code, "
             "error_reason, created_at, updated_at")
_CHUNK_COLS = ("chunk_id, doc_id, owner, position, text, page, heading, section, row, "
               "char_start, char_end, embedding")


def _integrity_ok(conn) -> bool:
    try:
        row = conn.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError:
        return False
    return bool(row) and row[0] == "ok"


def _backup_corrupt(db_path: Path) -> None:
    if not db_path.exists():
        return
    i = 0
    while (cand := db_path.with_suffix(db_path.suffix + f".corrupt.{i}")).exists():
        i += 1
    try:
        db_path.replace(cand)
    except OSError:
        try:
            db_path.unlink()
        except OSError:
            return
    for side in ("-wal", "-shm"):
        s = db_path.with_name(db_path.name + side)
        if s.exists():
            try:
                s.unlink()
            except OSError:
                pass
    log.warning("knowledge: corrupt database backed up/reset (%s)", db_path)


def open_database(db_path: Path, *, recover: bool = True):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if recover and db_path.exists() and not _integrity_ok(conn):
        conn.close()
        _backup_corrupt(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        log.info("knowledge: schema migrated to v%d", SCHEMA_VERSION)
    return conn


class KnowledgeRepository:
    def __init__(self, db_path, files_dir, indexes_dir, *, now_fn=None, id_factory=None):
        self.db_path = Path(db_path)
        self.files_dir = Path(files_dir)
        self.indexes_dir = Path(indexes_dir)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._id = id_factory or (lambda: uuid.uuid4().hex)
        open_database(self.db_path).close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def now_iso(self) -> str:
        return self._now().astimezone(timezone.utc).isoformat(timespec="seconds")

    def new_doc_id(self) -> str:
        return "doc_" + self._id()

    # -- files (content-addressed) -------------------------------------------------

    def file_path(self, sha256: str, fmt: str) -> Path:
        sub = self.files_dir / sha256[:2]
        return sub / f"{sha256}.{fmt}"

    def store_file(self, sha256: str, fmt: str, data: bytes) -> Path:
        path = self.file_path(sha256, fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return path

    def delete_file(self, sha256: str, fmt: str) -> None:
        try:
            self.file_path(sha256, fmt).unlink(missing_ok=True)
        except OSError:
            pass

    # -- documents -----------------------------------------------------------------

    def create_document(self, doc: Document) -> Document:
        now = self.now_iso()
        doc.created_at = doc.created_at or now
        doc.updated_at = now
        with self._cursor() as conn:
            conn.execute(
                f"INSERT INTO documents ({_DOC_COLS}) VALUES "
                "(:doc_id, :owner, :filename, :title, :sha256, :format, :size_bytes, :page_count, "
                ":chunk_count, :status, :collections, :tags, :source, :injection_flags, :error_code, "
                ":error_reason, :created_at, :updated_at)", doc.to_row())
        return doc

    def save_document(self, doc: Document) -> Document:
        doc.updated_at = self.now_iso()
        with self._cursor() as conn:
            conn.execute(
                "UPDATE documents SET filename=:filename, title=:title, format=:format, "
                "size_bytes=:size_bytes, page_count=:page_count, chunk_count=:chunk_count, "
                "status=:status, collections=:collections, tags=:tags, source=:source, "
                "injection_flags=:injection_flags, error_code=:error_code, error_reason=:error_reason, "
                "updated_at=:updated_at WHERE doc_id=:doc_id", doc.to_row())
        return doc

    def get_document(self, doc_id: str) -> Document | None:
        with self._cursor() as conn:
            r = conn.execute(f"SELECT {_DOC_COLS} FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return Document.from_row(r) if r else None

    def find_by_sha256(self, owner: str, sha256: str) -> Document | None:
        with self._cursor() as conn:
            r = conn.execute(
                f"SELECT {_DOC_COLS} FROM documents WHERE owner=? AND sha256=? "
                "AND status!='archived' LIMIT 1", (owner, sha256)).fetchone()
        return Document.from_row(r) if r else None

    def list_documents(self, owner: str, *, statuses=None, collection=None, limit=100) -> list[Document]:
        sql = f"SELECT {_DOC_COLS} FROM documents WHERE owner=?"
        params: list = [owner]
        if statuses:
            sql += " AND status IN (" + ",".join("?" for _ in statuses) + ")"
            params.extend(statuses)
        sql += " ORDER BY datetime(updated_at) DESC LIMIT ?"
        params.append(limit)
        with self._cursor() as conn:
            rows = conn.execute(sql, params).fetchall()
        docs = [Document.from_row(r) for r in rows]
        if collection:
            docs = [d for d in docs if collection in d.collections]
        return docs

    # -- chunks + atomic publish ---------------------------------------------------

    def publish(self, doc: Document, chunks: list[Chunk]) -> Document:
        """Atomically replace a document's chunks + FTS rows and flip it to READY."""
        doc.chunk_count = len(chunks)
        doc.status = READY
        doc.updated_at = self.now_iso()
        with self._cursor() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc.doc_id,))
            conn.execute("DELETE FROM chunks_fts WHERE doc_id=?", (doc.doc_id,))
            for ch in chunks:
                conn.execute(
                    f"INSERT INTO chunks ({_CHUNK_COLS}) VALUES "
                    "(:chunk_id, :doc_id, :owner, :position, :text, :page, :heading, :section, "
                    ":row, :char_start, :char_end, :embedding)", ch.to_row())
                conn.execute("INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES (?,?,?)",
                             (ch.chunk_id, ch.doc_id, ch.text))
            conn.execute(
                "UPDATE documents SET status=?, chunk_count=?, updated_at=? WHERE doc_id=?",
                (READY, len(chunks), doc.updated_at, doc.doc_id))
        return doc

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        with self._cursor() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks WHERE doc_id=? ORDER BY position", (doc_id,)).fetchall()
        return [Chunk.from_row(r) for r in rows]

    def active_chunks(self, owner: str, *, doc_ids=None, limit=5000) -> list[Chunk]:
        """Chunks of READY documents for the owner (for vector scoring)."""
        sql = (f"SELECT c.chunk_id, c.doc_id, c.owner, c.position, c.text, c.page, c.heading, "
               "c.section, c.row, c.char_start, c.char_end, c.embedding FROM chunks c "
               "JOIN documents d ON d.doc_id=c.doc_id WHERE c.owner=? AND d.status=?")
        params: list = [owner, READY]
        if doc_ids:
            sql += " AND c.doc_id IN (" + ",".join("?" for _ in doc_ids) + ")"
            params.extend(doc_ids)
        sql += " LIMIT ?"
        params.append(limit)
        with self._cursor() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Chunk.from_row(r) for r in rows]

    def search_fts(self, owner: str, query: str, *, limit=40, doc_ids=None) -> list[tuple[str, float]]:
        if not (query or "").strip():
            return []
        sql = ("SELECT f.chunk_id AS cid, bm25(chunks_fts) AS score FROM chunks_fts f "
               "JOIN documents d ON d.doc_id=f.doc_id WHERE chunks_fts MATCH ? "
               "AND d.owner=? AND d.status=?")
        params: list = [query, owner, READY]
        if doc_ids:
            sql += " AND f.doc_id IN (" + ",".join("?" for _ in doc_ids) + ")"
            params.extend(doc_ids)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        with self._cursor() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(r["cid"], float(r["score"])) for r in rows]

    def delete_document(self, doc: Document) -> None:
        with self._cursor() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc.doc_id,))
            conn.execute("DELETE FROM chunks_fts WHERE doc_id=?", (doc.doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc.doc_id,))
        # Remove the stored file only if no other document shares the hash.
        with self._cursor() as conn:
            shared = conn.execute("SELECT COUNT(*) FROM documents WHERE sha256=?",
                                  (doc.sha256,)).fetchone()[0]
        if not shared:
            self.delete_file(doc.sha256, doc.format)

    def counts(self, owner: str) -> dict:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) n FROM documents WHERE owner=? GROUP BY status",
                (owner,)).fetchall()
        by = {r["status"]: r["n"] for r in rows}
        return {"total": sum(by.values()), "ready": by.get(READY, 0),
                "archived": by.get(ARCHIVED, 0), "by_status": by}

    # -- audit ---------------------------------------------------------------------

    def audit(self, event: str, *, owner=None, doc_id=None, detail=None) -> None:
        with self._cursor() as conn:
            conn.execute(
                "INSERT INTO knowledge_audit(ts, owner, doc_id, event, detail) VALUES (?,?,?,?,?)",
                (self.now_iso(), owner, doc_id, event, detail))

    def audit_recent(self, doc_id=None, limit=100) -> list[dict]:
        with self._cursor() as conn:
            if doc_id:
                rows = conn.execute("SELECT ts, owner, doc_id, event, detail FROM knowledge_audit "
                                    "WHERE doc_id=? ORDER BY id DESC LIMIT ?", (doc_id, limit)).fetchall()
            else:
                rows = conn.execute("SELECT ts, owner, doc_id, event, detail FROM knowledge_audit "
                                    "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
