"""Phase 6C — the SSE bridge: safe-event whitelisting, frame formatting, and the
non-blocking, bounded event-bus sink."""

import json

from app.core import eventbus
from app.uibridge.stream import EventStream, format_sse, heartbeat, safe_event


def test_safe_event_whitelists_fields():
    ev = safe_event({
        "domain": "email", "event_type": "send", "status": "ok", "severity": "success",
        "title": "Sent", "error_code": None, "related_id": "a1",
        # everything below must be dropped:
        "secret": "hunter2", "excerpt": "the message body", "path": "C:/x", "hash": "ab",
        "metadata": {"draft": "private"},
    })
    assert set(ev) == {"domain", "event_type", "status", "severity", "error_code",
                       "title", "related_id"}
    assert "secret" not in ev and "excerpt" not in ev and "metadata" not in ev


def test_safe_event_clips_and_strips_newlines():
    ev = safe_event({"title": "line1\nline2\rline3" + "x" * 500})
    assert "\n" not in ev["title"] and "\r" not in ev["title"]
    assert len(ev["title"]) <= 120


def test_safe_event_normalizes_bad_severity():
    assert safe_event({"severity": "CRITICAL"})["severity"] == "info"
    assert safe_event({"severity": "warning"})["severity"] == "warning"


def test_format_sse_frame():
    frame = format_sse({"domain": "ui", "status": "ok"}, event_id=7, event_name="activity")
    assert "id: 7\n" in frame
    assert "event: activity\n" in frame
    assert frame.endswith("\n\n")
    data_line = [ln for ln in frame.splitlines() if ln.startswith("data: ")][0]
    assert json.loads(data_line[6:]) == {"domain": "ui", "status": "ok"}


def test_heartbeat_is_a_comment():
    assert heartbeat().startswith(":")


def test_event_stream_receives_bus_events():
    eventbus.reset()
    try:
        es = EventStream().subscribe()
        eventbus.emit(domain="knowledge", event_type="ask", status="ok", title="Q", secret="x")
        got = es.get(timeout=1.0)
        assert got["domain"] == "knowledge" and "secret" not in got
    finally:
        es.unsubscribe()
        eventbus.reset()


def test_event_stream_drops_when_full_never_blocks():
    es = EventStream(maxsize=2)
    for i in range(5):
        es.push_for_test({"domain": f"d{i}"})
    assert es.dropped == 3                     # 2 queued, 3 shed
    assert es.get(timeout=0.1) is not None     # still drains what fit


def test_event_stream_get_timeout_returns_none():
    es = EventStream()
    assert es.get(timeout=0.05) is None
