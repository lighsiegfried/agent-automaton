"""Event redaction + the read-optimized event store (Phase 5D): allowlist redaction,
dedup, filtering/pagination, retention pruning, corruption recovery, owner scope."""

from datetime import datetime, timezone

import pytest

from app.activity import models
from app.activity.models import Event, EventRepository, redact_text, safe_metadata


# --- redaction ---------------------------------------------------------------------


def test_redact_text_scrubs_secrets():
    assert redact_text("token sk-LIVE1234567890ABCDEFghij here") == "[redacted]"
    assert redact_text("mi contraseña es hunter2") == "[redacted]"
    assert redact_text("browser read of the pricing page") == "browser read of the pricing page"


def test_safe_metadata_allowlist_drops_content():
    raw = {"count": 3, "recipient": "ana@example.com", "content": "the full body",
           "body_preview": "Dear Ana…", "results": [{"content": "x"}], "target": "notepad",
           "token": "sk-abc", "step_count": 4}
    out = safe_metadata(raw)
    assert out == {"count": 3, "recipient": "ana@example.com", "target": "notepad", "step_count": 4}
    assert "content" not in out and "body_preview" not in out and "results" not in out


def test_safe_metadata_secret_scans_allowed_values():
    out = safe_metadata({"target": "sk-LIVE1234567890ABCDEFghij", "count": 1})
    assert out["target"] == "[redacted]"


def test_event_redacted_scrubs_title_and_metadata():
    ev = Event(event_id="e1", ts="2026-07-15T12:00:00+00:00", domain="text",
               event_type="draft_text", title="draft sk-LIVE1234567890ABCDEFghij",
               metadata={"content": "secret body", "count": 2})
    r = ev.redacted()
    assert "[redacted]" in r.title or "sk-" not in r.title
    assert r.metadata == {"count": 2}


# --- repository --------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    return EventRepository(tmp_path / "a.db",
                          now_fn=lambda: datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc))


def _ev(repo, **over):
    d = dict(event_id=repo.new_id(), ts="2026-07-15T12:00:00+00:00", domain="memory",
             event_type="memory_search", severity="info", status="ok", title="t", summary="s",
             owner="local")
    d.update(over)
    return Event(**d)


def test_schema_wal_and_recovery(tmp_path):
    db = tmp_path / "a.db"
    db.write_bytes(b"not a database " * 40)
    repo = EventRepository(db)
    assert repo.insert(_ev(repo)) is True
    assert list(tmp_path.glob("a.db.corrupt.*"))
    conn = models.open_database(db)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()


def test_insert_and_get(repo):
    ev = _ev(repo)
    assert repo.insert(ev) is True
    got = repo.get(ev.event_id, "local")
    assert got and got.domain == "memory"


def test_dedup_prevents_duplicate(repo):
    e1 = _ev(repo, dedup_key="dupe-1")
    e2 = _ev(repo, dedup_key="dupe-1")
    assert repo.insert(e1) is True
    assert repo.insert(e2) is False               # same dedup key → ignored
    assert repo.total("local") == 1


def test_query_filter_and_pagination(repo):
    for i in range(12):
        repo.insert(_ev(repo, domain="browser", dedup_key=f"b{i}",
                        ts=f"2026-07-15T12:00:{i:02d}+00:00"))
    for i in range(3):
        repo.insert(_ev(repo, domain="email", severity="warning", dedup_key=f"e{i}"))
    page1, total = repo.query("local", domain="browser", limit=5, offset=0)
    assert total == 12 and len(page1) == 5
    page3, _ = repo.query("local", domain="browser", limit=5, offset=10)
    assert len(page3) == 2                         # last page
    warns, wt = repo.query("local", severity="warning")
    assert wt == 3


def test_severity_and_domain_counts(repo):
    repo.insert(_ev(repo, severity="error", dedup_key="a"))
    repo.insert(_ev(repo, severity="warning", dedup_key="b"))
    repo.insert(_ev(repo, domain="tasks", dedup_key="c"))
    assert repo.severity_counts("local").get("error") == 1
    assert repo.domain_counts("local").get("tasks") == 1


def test_owner_isolation(repo):
    repo.insert(_ev(repo, owner="local", dedup_key="l"))
    repo.insert(_ev(repo, owner="intruder", dedup_key="i"))
    assert repo.total("local") == 1
    events, _ = repo.query("intruder")
    assert all(e.owner == "intruder" for e in events)


def test_retention_prune_by_age_and_cap(tmp_path):
    clock = {"t": datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)}
    repo = EventRepository(tmp_path / "a.db", now_fn=lambda: clock["t"])
    # Old events (60 days ago) + fresh events.
    for i in range(5):
        repo.insert(_ev(repo, dedup_key=f"old{i}", ts="2026-05-01T12:00:00+00:00"))
    for i in range(5):
        repo.insert(_ev(repo, dedup_key=f"new{i}", ts="2026-07-15T11:00:00+00:00"))
    removed = repo.prune(retention_days=30, max_events=50000)
    assert removed == 5                             # the 5 old ones
    assert repo.total("local") == 5
    # Cap enforcement: keep only newest 3.
    removed2 = repo.prune(retention_days=30, max_events=3)
    assert removed2 == 2 and repo.total("local") == 3
