"""WhatsApp Web safety machine (Phase 4C) — fake session, no browser."""

import itertools
import types

import pytest

from app.config import get_settings
from app.core import conversation, errors, pending
from app.core.nl import ServiceCommand
from app.integrations.whatsapp import inspector, safety
from app.integrations.whatsapp import service as wa_service
from app.integrations.whatsapp.service import WhatsAppService


class FakeSession:
    def __init__(self, logged_in=True, contacts=None, chat="Ana"):
        self._logged_in = logged_in
        self._contacts = contacts if contacts is not None else ["Ana", "Ana García", "Beto"]
        self._chat = chat
        self.composer = ""
        self.sent = 0

    def logged_in(self): return self._logged_in
    def search_contacts(self, name): return self._contacts
    def open_chat(self, name): self._chat = name; return True
    def chat_title(self): return self._chat
    def composer_text(self): return self.composer
    def set_composer(self, text): self.composer = text
    def click_send(self): self.sent += 1
    def close(self): pass


@pytest.fixture
def env(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_whatsapp_automation", True)
    monkeypatch.setattr(settings, "enable_real_whatsapp_send", False)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    clock = {"t": 1000.0}
    session = FakeSession()
    ids = (f"wa{i}" for i in itertools.count(1))
    svc = WhatsAppService(session_provider=lambda s: session, clock=lambda: clock["t"],
                          id_factory=lambda: next(ids))
    monkeypatch.setattr(wa_service, "_service", svc)
    return types.SimpleNamespace(settings=settings, clock=clock, session=session, svc=svc,
                                 broker=pending.get_pending_broker())


def _dispatch(intent, **args):
    return conversation.dispatch(ServiceCommand(intent=intent, arguments=args), language="es")


def _confirm(phrase, wake=False):
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=phrase),
        language="es", wake=wake)


def _draft_and_place(env, text="Hola Ana"):
    _dispatch("whatsapp_draft_message", text=text)
    return _confirm("colocar borrador")


# --- pure helpers ------------------------------------------------------------------


def test_match_contacts_exact_ambiguous_notfound():
    assert inspector.match_contacts(["Ana", "Beto"], "Ana")["status"] == "ok"
    assert inspector.match_contacts(["Ana García", "Ana López"], "Ana")["status"] == "ambiguous"
    assert inspector.match_contacts(["Beto"], "Ana")["status"] == "not_found"


def test_recipient_matching_and_phrases():
    assert safety.is_place_phrase("colocar borrador") is True
    assert safety.is_send_phrase("confirmar envío a Ana") is True
    assert safety.is_send_phrase("confirmar") is False
    assert safety.recipient_matches("confirmar envío a Ana", "Ana") is True
    assert safety.recipient_matches("confirmar envío a Beto", "Ana") is False


# --- login + contacts --------------------------------------------------------------


def test_login_required(env):
    env.session._logged_in = False
    result = _dispatch("whatsapp_status")
    assert result["error_code"] == errors.LOGIN_REQUIRED


def test_ambiguous_contact_not_auto_selected(env):
    env.session._contacts = ["Ana García", "Ana López"]
    result = _dispatch("whatsapp_find_contact", name="Ana")
    assert result["error_code"] == errors.CONTACT_AMBIGUOUS
    assert len(result["data"]["candidates"]) == 2


def test_exact_contact_found(env):
    result = _dispatch("whatsapp_find_contact", name="Ana")
    assert result["status"] == "ok" and result["data"]["contact"] == "Ana"


# --- draft + place (no send) -------------------------------------------------------


def test_draft_registers_place_pending(env):
    result = _dispatch("whatsapp_draft_message", text="Hola Ana")
    assert result["status"] == "needs_confirmation"
    assert env.broker.summary()["domain"] == "whatsapp_place"
    assert env.session.composer == ""          # nothing placed yet


def test_place_inserts_without_sending(env):
    placed = _draft_and_place(env)
    assert placed["status"] == "placed"
    assert env.session.composer == "Hola Ana"  # inserted once
    assert env.session.sent == 0                # never sent
    assert env.broker.summary()["domain"] == "whatsapp_send"   # now armed to send


def test_secret_message_blocked(env):
    result = _dispatch("whatsapp_draft_message", text="mi contraseña: hunter2")
    assert result["error_code"] == errors.SEND_BLOCKED
    assert env.broker.summary() is None


def test_place_rejected_when_chat_changed(env):
    _dispatch("whatsapp_draft_message", text="Hola")
    env.session._chat = "Beto"                  # chat changed before placing
    result = _confirm("colocar borrador")
    assert result["error_code"] == errors.CHAT_CHANGED
    assert env.session.composer == ""


# --- send confirmation -------------------------------------------------------------


def test_send_requires_exact_recipient(env):
    _draft_and_place(env)
    wrong = _confirm("confirmar envío a Beto")   # wrong recipient
    assert wrong["error_code"] == errors.CONFIRMATION_MISMATCH
    assert env.session.sent == 0


def test_send_simulated_by_default(env):
    _draft_and_place(env)
    result = _confirm("confirmar envío a Ana")
    assert result["status"] == "simulated"
    assert env.session.sent == 0                 # simulated => no click


def test_send_real_clicks_send_once(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_real_whatsapp_send", True)
    _draft_and_place(env)
    result = _confirm("confirmar envío a Ana")
    assert result["status"] == "sent"
    assert env.session.sent == 1                 # exactly one Send click, never Enter


def test_duplicate_send_prevented(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_real_whatsapp_send", True)
    _draft_and_place(env)
    _confirm("confirmar envío a Ana")
    again = _confirm("confirmar envío a Ana")
    assert again["error_code"] == errors.CONFIRMATION_REQUIRED    # nothing pending
    assert env.session.sent == 1


def test_composer_hash_mismatch_blocks_send(env):
    _draft_and_place(env)
    env.session.composer = "Hola Ana EDITADO"    # composer changed after placing
    result = _confirm("confirmar envío a Ana")
    assert result["error_code"] == errors.DRAFT_CHANGED
    assert env.session.sent == 0


def test_chat_change_blocks_send(env):
    _draft_and_place(env)
    env.session._chat = "Beto"
    result = _confirm("confirmar envío a Ana")
    assert result["error_code"] == errors.CHAT_CHANGED


def test_send_expiration(env):
    _draft_and_place(env)
    env.clock["t"] += 100
    result = _confirm("confirmar envío a Ana")
    assert result["error_code"] == errors.ACTION_EXPIRED


def test_rate_limited(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_real_whatsapp_send", True)
    _draft_and_place(env)
    _confirm("confirmar envío a Ana")            # first send (sets cooldown)
    _draft_and_place(env, text="Otro mensaje")   # a fresh action, same instant
    blocked = _confirm("confirmar envío a Ana")
    assert blocked["error_code"] == errors.SEND_BLOCKED
    assert env.session.sent == 1                 # the rate-limited one did not send


# --- cross-domain + wake safety ----------------------------------------------------


def test_plain_si_never_sends(env):
    _draft_and_place(env)
    assert _confirm("sí")["error_code"] == errors.CONFIRMATION_MISMATCH
    assert env.session.sent == 0


def test_wake_never_sends(env):
    _draft_and_place(env)
    assert _confirm("Fifi", wake=True)["error_code"] == errors.CONFIRMATION_MISMATCH


def test_cross_domain_text_phrase_cannot_send(env):
    _draft_and_place(env)                         # whatsapp_send pending
    result = _confirm("confirmar escritura")      # a TEXT phrase
    assert result["error_code"] == errors.CONFIRMATION_MISMATCH
    assert env.session.sent == 0
