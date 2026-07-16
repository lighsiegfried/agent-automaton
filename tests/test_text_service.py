"""TextActionService: draft/confirm/cancel gates (Phase 4A).

No real typing: the inserter is a fake and the target is injected. Covers
drafting-without-execution, phrase confirmation, expiry, changed-target/draft
rejection, password/terminal blocking, length, secrets, Unicode, no duplicate
insertion.
"""

import itertools
import types

import pytest

from app.text import models
from app.text.models import TargetSummary
from app.text.service import TextActionService


def make_settings(**over):
    base = dict(
        enable_real_text_input=False,
        text_input_require_confirmation=True,
        text_action_expires_seconds=45,
        text_max_characters=5000,
        text_allowed_apps_list=["notepad", "wordpad", "winword", "chrome", "edge"],
        text_block_password_fields=True,
        text_allow_clipboard_fallback=False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def make_target(**over):
    base = dict(
        application="notepad", window_title="Untitled - Notepad", process="notepad.exe",
        executable="notepad.exe", control_type="Edit", editable=True, supported=True,
    )
    base.update(over)
    return TargetSummary(**base)


@pytest.fixture
def clock():
    return {"t": 100.0}


@pytest.fixture
def service(clock):
    captured = []

    def fake_inserter(pending, settings):
        captured.append(pending)
        return {"ok": True, "method": "uia", "chars": len(pending.text), "pressed_enter": False}

    ids = (f"id{i}" for i in itertools.count(1))
    svc = TextActionService(
        clock=lambda: clock["t"],
        id_factory=lambda: next(ids),
        inserter=fake_inserter,
    )
    svc.captured = captured
    return svc


# --- drafting is safe, never inserts ----------------------------------------------


def test_draft_creates_pending_without_inserting(service):
    s = make_settings()
    result = service.draft(action="type_text", text="Hola mundo", target=make_target(), settings=s)
    assert result["status"] == "needs_confirmation"
    assert result["preview"]["character_count"] == len("Hola mundo")
    assert result["preview"]["action_id"] == "id1"
    assert service.pending_preview(settings=s) is not None
    assert service.captured == []          # nothing was inserted


def test_confirm_simulated_when_real_input_disabled(service):
    s = make_settings(enable_real_text_input=False)
    drafted = service.draft(action="type_text", text="Hola", target=make_target(), settings=s)
    aid = drafted["preview"]["action_id"]
    result = service.confirm(action_id=aid, phrase="insert text", target=make_target(), settings=s)
    assert result["status"] == "simulated" and result["inserted"] is False
    assert service.captured == []


def test_confirm_executes_when_real_input_enabled(service):
    s = make_settings(enable_real_text_input=True)
    drafted = service.draft(action="type_text", text="Hola", target=make_target(), settings=s)
    aid = drafted["preview"]["action_id"]
    result = service.confirm(action_id=aid, phrase="escribirlo", target=make_target(), settings=s)
    assert result["status"] == "executed" and result["inserted"] is True
    assert result["pressed_enter"] is False
    assert len(service.captured) == 1


# --- confirmation phrase strictness -----------------------------------------------


def test_plain_si_is_not_enough(service):
    s = make_settings(enable_real_text_input=True)
    aid = service.draft(action="type_text", text="Hola", target=make_target(), settings=s)["preview"]["action_id"]
    result = service.confirm(action_id=aid, phrase="sí", target=make_target(), settings=s)
    assert result["status"] == "needs_confirmation"
    assert service.captured == []                 # not inserted
    assert service.pending_preview(settings=s) is not None   # still pending, can retry


# --- expiry ------------------------------------------------------------------------


def test_expired_action_cannot_be_confirmed(service, clock):
    s = make_settings(text_action_expires_seconds=45)
    aid = service.draft(action="type_text", text="Hola", target=make_target(), settings=s)["preview"]["action_id"]
    clock["t"] += 46
    result = service.confirm(action_id=aid, phrase="insert text", target=make_target(), settings=s)
    assert result["status"] == "rejected" and "expired" in result["reason"]
    assert service.pending_preview(settings=s) is None


# --- changed target / changed draft ------------------------------------------------


def test_changed_target_rejects_confirmation(service):
    s = make_settings(enable_real_text_input=True)
    aid = service.draft(action="type_text", text="Hola", target=make_target(), settings=s)["preview"]["action_id"]
    moved = make_target(window_title="Other - Notepad")   # different signature
    result = service.confirm(action_id=aid, phrase="insert text", target=moved, settings=s)
    assert result["status"] == "rejected" and "changed" in result["reason"]
    assert service.captured == []


def test_changed_draft_action_id_mismatch(service):
    s = make_settings()
    first = service.draft(action="type_text", text="One", target=make_target(), settings=s)
    old_id = first["preview"]["action_id"]
    service.draft(action="type_text", text="Two", target=make_target(), settings=s)  # replaces
    result = service.confirm(action_id=old_id, phrase="insert text", target=make_target(), settings=s)
    assert result["status"] == "rejected" and "does not match" in result["reason"]


# --- blocked targets ---------------------------------------------------------------


def test_password_target_is_rejected_at_draft(service):
    s = make_settings()
    blocked = make_target(is_password_field=True, blocked=True,
                          blocked_reason="password/secret field — text insertion is blocked")
    result = service.draft(action="type_text", text="secret words", target=blocked, settings=s)
    assert result["status"] == "rejected" and "password" in result["reason"]
    assert service.pending_preview(settings=s) is None


def test_terminal_target_is_rejected_at_draft(service):
    s = make_settings()
    blocked = make_target(is_terminal=True, blocked=True, blocked_reason="terminal/shell window")
    result = service.draft(action="type_text", text="rm -rf", target=blocked, settings=s)
    assert result["status"] == "rejected" and "terminal" in result["reason"]


# --- length + secrets --------------------------------------------------------------


def test_over_max_length_rejected(service):
    s = make_settings(text_max_characters=10)
    result = service.draft(action="type_text", text="x" * 20, target=make_target(), settings=s)
    assert result["status"] == "rejected" and "TEXT_MAX_CHARACTERS" in result["reason"]


def test_secret_draft_rejected_and_redacted(service):
    s = make_settings()
    result = service.draft(action="type_text", text="card 4111 1111 1111 1111",
                           target=make_target(), settings=s)
    assert result["status"] == "rejected"
    assert "payment_card_number" in result["secret_reasons"]
    assert result["preview"]["text"] == "[redacted]"


# --- Unicode + no duplicate insertion ---------------------------------------------


def test_unicode_accents_preserved_through_insertion(service):
    s = make_settings(enable_real_text_input=True)
    text = "Añadí una canción para ti — ¡qué día tan lindo!"
    aid = service.draft(action="type_text", text=text, target=make_target(), settings=s)["preview"]["action_id"]
    service.confirm(action_id=aid, phrase="confirmar escritura", target=make_target(), settings=s)
    assert service.captured[0].text == text     # accents/ñ/¡ preserved exactly


def test_no_duplicate_insertion(service):
    s = make_settings(enable_real_text_input=True)
    aid = service.draft(action="type_text", text="Hola", target=make_target(), settings=s)["preview"]["action_id"]
    first = service.confirm(action_id=aid, phrase="insert text", target=make_target(), settings=s)
    assert first["status"] == "executed"
    second = service.confirm(action_id=aid, phrase="insert text", target=make_target(), settings=s)
    assert second["status"] == "rejected" and "no pending" in second["reason"]
    assert len(service.captured) == 1           # inserted exactly once


def test_cancel_discards_pending(service):
    s = make_settings()
    service.draft(action="type_text", text="Hola", target=make_target(), settings=s)
    result = service.cancel()
    assert result["status"] == "cancelled" and result["had_pending"] is True
    assert service.pending_preview(settings=s) is None


def test_append_action_type_recorded(service):
    s = make_settings()
    result = service.draft(action="append_text", text="more", target=make_target(), settings=s)
    assert result["preview"]["action_type"] == models.APPEND_TEXT
