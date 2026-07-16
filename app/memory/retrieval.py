"""Deterministic hybrid retrieval + bounded context bundle (Phase 5A, item 5).

Combines three signals, no embeddings required:

1. **Exact entity/tag/subject match** — the strongest signal.
2. **SQLite FTS5 full-text** relevance (bm25 ordering).
3. **Recency** — more recently updated/used memories rank higher.

The result is a small, BOUNDED bundle: at most ``max_items`` memories and at most
``max_chars`` of content, each reduced to id/type/date + the relevant text. The full
database is never returned and never injected into the planner.
"""

from __future__ import annotations

import re

from app.memory.models import EV_RETRIEVAL, Memory, normalize

# Signal weights (fixed, so ranking is fully deterministic).
_W_ENTITY = 100.0     # query token equals a memory entity/tag
_W_SUBJECT = 80.0     # query matches the memory's subject key
_W_FTS = 6.0          # per-rank full-text relevance (best hit gets the most)
_W_RECENCY = 4.0      # per-rank recency (most recent gets the most)

_TOKEN = re.compile(r"[0-9a-záéíóúüñ.+#-]{2,}", re.IGNORECASE)


def query_tokens(text: str) -> list[str]:
    return [normalize(t) for t in _TOKEN.findall(text or "") if normalize(t)]


def build_fts_query(text: str) -> str:
    """A safe FTS5 MATCH string: each token quoted and OR-ed, so stray quotes or
    operators in user speech can never produce a malformed query."""
    tokens = query_tokens(text)
    return " OR ".join(f'"{t}"' for t in tokens)


def _entity_hit(memory: Memory, tokens: set[str]) -> bool:
    haystack = {normalize(e) for e in (*memory.entities, *memory.tags)}
    if haystack & tokens:
        return True
    # A multi-word entity may contain a query token, or vice-versa.
    for entity in haystack:
        for tok in tokens:
            if tok and (tok in entity or entity in tok):
                return True
    return False


def _subject_hit(memory: Memory, tokens: set[str], query_norm: str) -> bool:
    subject = normalize(memory.subject)
    if not subject:
        return False
    if subject in query_norm or query_norm in subject:
        return True
    return bool(set(subject.split()) & tokens)


def rank_candidates(query: str, exact_pool: list[Memory],
                    fts_hits: list[tuple[Memory, float]]) -> list[tuple[Memory, float]]:
    """Score and order candidates deterministically. Returns (memory, score) desc."""
    tokens = set(query_tokens(query))
    query_norm = normalize(query)

    # FTS rank position -> relevance (best hit first). Preserve id order for ties.
    fts_rank = {mem.id: i for i, (mem, _score) in enumerate(fts_hits)}

    # Merge unique candidates.
    by_id: dict[str, Memory] = {}
    for mem in exact_pool:
        by_id.setdefault(mem.id, mem)
    for mem, _score in fts_hits:
        by_id.setdefault(mem.id, mem)
    candidates = list(by_id.values())

    # Recency rank across the merged candidate set.
    recency_order = sorted(
        candidates,
        key=lambda m: (m.last_used_at or m.updated_at or m.created_at or "", m.id),
        reverse=True,
    )
    recency_rank = {mem.id: i for i, mem in enumerate(recency_order)}

    scored: list[tuple[Memory, float]] = []
    for mem in candidates:
        score = 0.0
        if tokens and _entity_hit(mem, tokens):
            score += _W_ENTITY
        if _subject_hit(mem, tokens, query_norm):
            score += _W_SUBJECT
        if mem.id in fts_rank:
            score += max(0.0, _W_FTS * (len(fts_hits) - fts_rank[mem.id]))
        score += max(0.0, _W_RECENCY * (len(candidates) - recency_rank[mem.id]))
        scored.append((mem, score))

    scored.sort(
        key=lambda pair: (
            pair[1],
            pair[0].last_used_at or pair[0].updated_at or "",
            pair[0].id,
        ),
        reverse=True,
    )
    return scored


def search(repo, owner: str, query: str, *, limit: int) -> list[Memory]:
    """Ranked active memories for a management/voice search (not char-bounded).

    Unlike ``retrieve``, this returns whole memories (the caller formats them) and
    does not touch ``last_used_at`` or the audit trail beyond what the caller adds."""
    query = (query or "").strip()
    exact_pool = repo.active(owner)
    fts_query = build_fts_query(query)
    fts_hits = repo.search_fts(owner, fts_query, limit=max(limit * 2, 20)) if fts_query else []
    ranked = rank_candidates(query, exact_pool, fts_hits)
    floor = _W_RECENCY * len(exact_pool)
    hits = [m for m, s in ranked if s > floor]
    return hits[:limit]


def retrieve(repo, owner: str, query: str, *, max_items: int, max_chars: int,
             audit: bool = True) -> dict:
    """Return a bounded context bundle for ``query`` (only active, owner-scoped)."""
    query = (query or "").strip()
    exact_pool = repo.active(owner)
    fts_query = build_fts_query(query)
    fts_hits = repo.search_fts(owner, fts_query, limit=max(20, max_items * 4)) if fts_query else []

    if not query:
        return {"query": "", "count": 0, "memories": [], "char_budget": max_chars,
                "chars_used": 0, "item_budget": max_items, "truncated": False}

    ranked = rank_candidates(query, exact_pool, fts_hits)
    # Keep only candidates with a positive content signal — a bare recency floor is
    # not a match, so recency-only memories never leak into the bundle.
    ranked = [(m, s) for m, s in ranked if s > _W_RECENCY * len(exact_pool)]

    memories: list[dict] = []
    used_ids: list[str] = []
    chars_used = 0
    per_item_cap = max(80, max_chars // max(1, max_items))
    truncated = False
    for mem, _score in ranked:
        if len(memories) >= max_items:
            truncated = True
            break
        view = mem.context_view(per_item_cap)
        cost = len(view["content"])
        if memories and chars_used + cost > max_chars:
            truncated = True
            break
        memories.append(view)
        used_ids.append(mem.id)
        chars_used += cost

    if used_ids:
        repo.touch_used(used_ids)
        if audit:
            repo.audit(EV_RETRIEVAL, owner=owner,
                       detail=f"query_len={len(query)} returned={len(used_ids)}")

    return {
        "query": query,
        "count": len(memories),
        "memories": memories,
        "char_budget": max_chars,
        "chars_used": chars_used,
        "item_budget": max_items,
        "truncated": truncated or len(ranked) > len(memories),
    }
