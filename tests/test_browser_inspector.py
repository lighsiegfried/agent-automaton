"""Safe page snapshot shaping + element targeting (Phase 4B)."""

from app.browser import inspector


def _raw(**over):
    raw = {
        "url": "https://example.com/pricing",
        "title": "Pricing",
        "text": "Our pricing is simple. Contact sales for details.",
        "headings": [{"level": 1, "text": "Pricing"}, {"level": 2, "text": "Plans"}],
        "links": [{"text": "Contact", "href": "https://example.com/contact"},
                  {"text": "Docs", "href": "https://example.com/docs"}],
        "buttons": [{"text": "Sign up", "role": "button"}],
        "form_controls": [
            {"name": "email", "type": "email", "label": "Email", "autocomplete": "email"},
            {"name": "password", "type": "password", "label": "Password"},
            {"name": "csrf", "type": "hidden", "label": ""},
        ],
        "landmarks": ["main"],
    }
    raw.update(over)
    return raw


def test_snapshot_excludes_hidden_and_values_flags_sensitive():
    snap = inspector.build_snapshot(_raw(), max_chars=1000)
    names = [c["name"] for c in snap["form_controls"]]
    assert "csrf" not in names                       # hidden field dropped
    controls = {c["name"]: c for c in snap["form_controls"]}
    assert controls["email"]["fillable"] is True
    assert controls["password"]["fillable"] is False  # sensitive, never fill
    # No control ever carries a 'value'.
    assert all("value" not in c for c in snap["form_controls"])


def test_snapshot_caps_text():
    snap = inspector.build_snapshot(_raw(text="x" * 100), max_chars=10)
    assert snap["visible_text"] == "x" * 10
    assert snap["text_truncated"] is True


def test_find_text_matches_headings_and_links():
    snap = inspector.build_snapshot(_raw(), max_chars=1000)
    result = inspector.find_text(snap, "plans")
    assert result["found"] is True
    assert any(m["kind"] == "heading" for m in result["matches"])


def test_resolve_link_unique_and_ambiguous():
    snap = inspector.build_snapshot(_raw(links=[
        {"text": "Docs", "href": "https://example.com/a"},
        {"text": "Docs", "href": "https://example.com/b"},
    ]), max_chars=1000)
    assert inspector.resolve_link(snap, text="Docs")["ok"] is False   # ambiguous
    snap2 = inspector.build_snapshot(_raw(), max_chars=1000)
    ok = inspector.resolve_link(snap2, text="Contact")
    assert ok["ok"] is True and ok["link"]["href"].endswith("/contact")


def test_resolve_target_rejects_ambiguous():
    snap = inspector.build_snapshot(_raw(buttons=[
        {"text": "Open", "role": "button"}, {"text": "Open", "role": "button"},
    ]), max_chars=1000)
    result = inspector.resolve_target(snap, text="Open")
    assert result["ok"] is False and "ambiguous" in result["reason"]


def test_resolve_target_unique_match():
    snap = inspector.build_snapshot(_raw(), max_chars=1000)
    result = inspector.resolve_target(snap, role="button", text="Sign up")
    assert result["ok"] is True and result["confidence"] == 1.0


def test_resolve_target_requires_descriptor():
    snap = inspector.build_snapshot(_raw(), max_chars=1000)
    assert inspector.resolve_target(snap)["ok"] is False
