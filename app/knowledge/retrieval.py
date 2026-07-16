"""Hybrid retrieval → bounded evidence bundle (Phase 6A, item 6).

Combines lexical FTS5 (bm25) with local semantic vector similarity, applies document/
collection filters, and uses recency only as a minor tie-breaker. It returns a small,
bounded evidence bundle (document id, filename, page/section, chunk id, excerpt, score)
— never whole documents — so the planner/answer only ever sees a handful of cited
snippets.
"""

from __future__ import annotations

import re

from app.knowledge.embeddings import cosine
from app.knowledge.models import READY, Evidence

_TOKEN = re.compile(r"[0-9a-záéíóúüñ]{2,}", re.IGNORECASE)
_W_VECTOR = 0.7
_W_LEXICAL = 0.3
# A chunk is evidence only with a lexical (FTS) hit OR a genuinely semantic match —
# this keeps stopword-only overlap from making an off-topic question look answerable.
_SEMANTIC_FLOOR = 0.30


def build_fts_query(text: str) -> str:
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    return " OR ".join(f'"{t}"' for t in toks)


def retrieve(repo, embedder, owner: str, query: str, *, max_results: int, max_chars: int,
             doc_ids=None, collection=None) -> dict:
    query = (query or "").strip()
    if not query:
        return {"query": "", "evidence": [], "count": 0, "chars_used": 0, "truncated": False}

    if collection and not doc_ids:
        doc_ids = [d.doc_id for d in repo.list_documents(owner, statuses=[READY],
                                                         collection=collection, limit=1000)]
        if not doc_ids:
            return {"query": query, "evidence": [], "count": 0, "chars_used": 0, "truncated": False}

    chunks = repo.active_chunks(owner, doc_ids=doc_ids)
    if not chunks:
        return {"query": query, "evidence": [], "count": 0, "chars_used": 0, "truncated": False}

    docs = {d.doc_id: d for d in repo.list_documents(owner, statuses=[READY], limit=1000)}
    qvec = embedder.embed_one(query)

    fts = repo.search_fts(owner, build_fts_query(query), limit=max_results * 6, doc_ids=doc_ids)
    fts_ranked = [cid for cid, _ in sorted(fts, key=lambda kv: kv[1])]   # ascending bm25 = best first
    n_fts = max(1, len(fts_ranked))
    fts_bonus = {cid: (n_fts - i) / n_fts for i, cid in enumerate(fts_ranked)}

    scored: list[tuple] = []
    for c in chunks:
        vec = cosine(qvec, c.embedding)
        lex = fts_bonus.get(c.chunk_id, 0.0)
        if lex <= 0 and vec < _SEMANTIC_FLOOR:      # not a real match — skip (precision)
            continue
        combined = _W_VECTOR * max(0.0, vec) + _W_LEXICAL * lex
        doc = docs.get(c.doc_id)
        recency = doc.updated_at if doc else ""
        scored.append((combined, recency, c, doc))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    evidence: list[Evidence] = []
    chars = 0
    truncated = False
    for combined, _rec, c, doc in scored:
        if len(evidence) >= max_results:
            truncated = True
            break
        excerpt = c.text if len(c.text) <= 400 else c.text[:400].rstrip() + "…"
        if evidence and chars + len(excerpt) > max_chars:
            truncated = True
            break
        evidence.append(Evidence(
            doc_id=c.doc_id, filename=(doc.filename if doc else c.doc_id), chunk_id=c.chunk_id,
            location=c.location(), excerpt=excerpt, score=combined, page=c.page, section=c.section))
        chars += len(excerpt)

    return {"query": query, "evidence": [e.public() for e in evidence],
            "evidence_objs": evidence, "count": len(evidence), "chars_used": chars,
            "truncated": truncated or len(scored) > len(evidence)}
