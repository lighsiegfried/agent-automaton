"""Typed memory records + pending-action payloads (Phase 5A).

Pure dataclasses, enums, and stdlib-only helpers (normalization, hashing) — no
SQLite, no service logic — so every other memory module can import this without a
cycle and the whole layer stays unit-testable.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(str, Enum):
    """The kinds of memory Fifi may keep. Tasks/reminders are deliberately absent
    in this phase (no autonomous long-term planning yet)."""

    PREFERENCE = "preference"
    PERSON = "person"
    PROJECT = "project"
    DECISION = "decision"
    WORKFLOW = "workflow"
    FACT = "fact"
    INSTRUCTION = "instruction"


MEMORY_TYPES: frozenset[str] = frozenset(t.value for t in MemoryType)

# --- lifecycle status --------------------------------------------------------------
ACTIVE = "active"        # retrievable
FORGOTTEN = "forgotten"  # soft-deleted: excluded from retrieval, still on disk
DELETED = "deleted"      # permanently removed (transient marker for the audit trail)

STATUSES: frozenset[str] = frozenset({ACTIVE, FORGOTTEN, DELETED})

# --- audit events (item 10). Never carry full sensitive content. -------------------
EV_PROPOSAL = "proposal"
EV_CONFIRMATION = "confirmation"
EV_RETRIEVAL = "retrieval"
EV_UPDATE = "update"
EV_FORGET = "forget"
EV_PERMANENT_DELETE = "permanent_delete"
EV_CANCEL = "cancel"
EV_BLOCKED = "blocked"
EV_RESTORE = "restore"


def normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip accents — for stable keys/tags.

    Diacritics are folded so "preferido" and "preferído" compare equal, matching
    the FTS tokenizer's ``remove_diacritics`` behaviour.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(folded.strip().lower().split())


def content_hash(owner: str, mem_type: str, content: str) -> str:
    """Stable hash for duplicate detection: same owner+type+normalized content.

    Title is intentionally excluded — it is cosmetic/derived, so two proposals
    that say the same thing collide even if their titles differ."""
    basis = "\n".join([owner or "", mem_type or "", normalize(content)])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


@dataclass
class Memory:
    """One stored memory. ``entities``/``tags`` are normalized for exact-match
    retrieval; ``subject`` is the normalized property key used for conflict
    detection ("editor preferido" → a new value conflicts with the old)."""

    id: str
    owner: str
    type: str
    title: str
    content: str
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    subject: str = ""
    source: str = "api"
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str | None = None
    confidence: float = 0.8
    expires_at: str | None = None
    status: str = ACTIVE
    content_hash: str = ""
    revision: int = 1

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "owner": self.owner,
            "type": self.type,
            "title": self.title,
            "content": self.content,
            "entities": json.dumps(self.entities, ensure_ascii=False),
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "subject": self.subject,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
            "confidence": self.confidence,
            "expires_at": self.expires_at,
            "status": self.status,
            "content_hash": self.content_hash,
            "revision": self.revision,
        }

    @classmethod
    def from_row(cls, row) -> "Memory":
        data = dict(row)
        return cls(
            id=data["id"],
            owner=data["owner"],
            type=data["type"],
            title=data["title"],
            content=data["content"],
            entities=_as_list(data.get("entities")),
            tags=_as_list(data.get("tags")),
            subject=data.get("subject", ""),
            source=data.get("source", "api"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            last_used_at=data.get("last_used_at"),
            confidence=data.get("confidence", 0.8),
            expires_at=data.get("expires_at"),
            status=data.get("status", ACTIVE),
            content_hash=data.get("content_hash", ""),
            revision=data.get("revision", 1),
        )

    def public(self) -> dict:
        """Full record for the API/tray/export (the user's own data, no secrets —
        secrets are blocked before storage)."""
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "content": self.content,
            "entities": list(self.entities),
            "tags": list(self.tags),
            "subject": self.subject,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
            "confidence": self.confidence,
            "expires_at": self.expires_at,
            "status": self.status,
            "revision": self.revision,
        }

    def context_view(self, max_chars: int = 280) -> dict:
        """A compact view for the bounded retrieval bundle: id/type/date + the
        relevant content, truncated to ``max_chars``."""
        content = self.content if len(self.content) <= max_chars else self.content[:max_chars].rstrip() + "…"
        return {
            "id": self.id,
            "type": self.type,
            "date": (self.updated_at or self.created_at or "")[:10],
            "content": content,
        }


@dataclass
class MemoryProposal:
    """A pending create/update awaiting an explicit confirmation phrase.

    Memory-only and expiring — it is registered with the global pending broker as
    the ``memory_write`` domain, so a new proposal (or any other domain's pending
    action) supersedes it."""

    action_id: str
    owner: str
    mode: str                       # "create" | "update"
    memory: Memory
    target_id: str | None = None    # the memory being updated, when mode == "update"
    old_title: str = ""
    old_content: str = ""
    conflict: bool = False          # update triggered by a same-subject conflict
    expires_at_monotonic: float = 0.0
    expires_at_iso: str = ""
    used: bool = False

    def preview(self, limit: int = 200) -> dict:
        content = self.memory.content
        preview = content if len(content) <= limit else content[:limit].rstrip() + "…"
        data = {
            "action_id": self.action_id,
            "mode": self.mode,
            "type": self.memory.type,
            "title": self.memory.title,
            "content_preview": preview,
            "entities": list(self.memory.entities),
            "tags": list(self.memory.tags),
            "subject": self.memory.subject,
            "expires_at": self.expires_at_iso,
        }
        if self.mode == "update":
            old = self.old_content
            data["target_id"] = self.target_id
            data["conflict"] = self.conflict
            data["old_value"] = old if len(old) <= limit else old[:limit].rstrip() + "…"
        return data


@dataclass
class ForgetAction:
    """A pending forget awaiting confirmation. ``mode`` is ``soft`` (excluded from
    retrieval) or ``permanent`` (row removed) — each needs its OWN exact phrase and
    is registered as a distinct broker domain (``memory_forget`` / ``memory_delete``)."""

    action_id: str
    owner: str
    mode: str                       # "soft" | "permanent"
    memory_ids: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    expires_at_monotonic: float = 0.0
    expires_at_iso: str = ""
    used: bool = False

    def preview(self) -> dict:
        return {
            "action_id": self.action_id,
            "mode": self.mode,
            "count": len(self.memory_ids),
            "matches": [
                {"id": mid, "title": title, "type": mtype}
                for mid, title, mtype in zip(self.memory_ids, self.titles, self.types)
            ],
            "expires_at": self.expires_at_iso,
        }
