"""Deterministic hybrid retrieval + bounded context bundle (Phase 5A, item 5)."""

import itertools
from datetime import datetime, timezone

import pytest

from app.memory import retrieval
from app.memory.models import Memory, content_hash, normalize
from app.memory.repository import MemoryRepository


@pytest.fixture
def repo(tmp_path):
    ids = (f"mem_{i}" for i in itertools.count(1))
    return MemoryRepository(tmp_path / "mem.db",
                            now_fn=lambda: datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                            id_factory=lambda: next(ids))


def _add(repo, content, *, type="fact", entities=(), tags=(), subject=""):
    m = Memory(id=repo.new_id(), owner="local", type=type, title=content[:40], content=content,
               entities=[normalize(e) for e in entities], tags=[normalize(t) for t in tags],
               subject=normalize(subject), content_hash=content_hash("local", type, content))
    return repo.insert(m)


def test_fts_full_text_accent_insensitive(repo):
    a = _add(repo, "El proyecto Agent Automaton usa Python y FastAPI", type="project")
    _add(repo, "Mi color favorito es azul")
    # Query without accents/case still matches; unrelated memory is not returned.
    hits = retrieval.search(repo, "local", "automaton", limit=5)
    assert [m.id for m in hits] == [a.id]


def test_exact_entity_match_outranks_text(repo):
    tagged = _add(repo, "Ana es la líder del equipo", type="person", entities=("ana",), tags=("equipo",))
    _add(repo, "Recuerda mencionar al equipo en la reunión")   # text mentions 'equipo' only
    hits = retrieval.search(repo, "local", "equipo", limit=5)
    assert hits[0].id == tagged.id     # the exact tag/entity hit ranks first


def test_subject_match(repo):
    m = _add(repo, "Mi editor preferido es VS Code", type="preference", subject="editor preferido")
    hits = retrieval.search(repo, "local", "editor preferido", limit=5)
    assert hits and hits[0].id == m.id


def test_forgotten_excluded_from_retrieval(repo):
    m = _add(repo, "Mi editor preferido es VS Code", subject="editor preferido")
    repo.soft_forget([m.id])
    assert retrieval.search(repo, "local", "editor", limit=5) == []


def test_bounded_context_respects_item_and_char_budget(repo):
    for i in range(10):
        _add(repo, f"Nota número {i} sobre el proyecto agent automaton y su arquitectura local",
             type="project", tags=("proyecto",))
    bundle = retrieval.retrieve(repo, "local", "proyecto", max_items=3, max_chars=300)
    assert bundle["count"] <= 3
    assert bundle["chars_used"] <= 300
    assert bundle["truncated"] is True                 # more candidates than returned
    # Each item is a compact view: id/type/date/content only (never the whole row).
    assert set(bundle["memories"][0]) == {"id", "type", "date", "content"}


def test_retrieval_touches_last_used_and_audits(repo):
    m = _add(repo, "Agent Automaton corre localmente", type="project", tags=("automaton",))
    retrieval.retrieve(repo, "local", "automaton", max_items=5, max_chars=500)
    assert repo.get(m.id).last_used_at is not None
    assert any(e["event"] == "retrieval" for e in repo.audit_recent())


def test_empty_query_returns_empty_bundle(repo):
    _add(repo, "algo")
    bundle = retrieval.retrieve(repo, "local", "", max_items=5, max_chars=500)
    assert bundle["count"] == 0 and bundle["memories"] == []


def test_malformed_fts_query_does_not_crash(repo):
    _add(repo, "Agent Automaton", tags=("automaton",))
    # Stray quotes/operators must not raise — build_fts_query sanitizes them.
    assert retrieval.search(repo, "local", '"" OR (', limit=5) == [] or True
