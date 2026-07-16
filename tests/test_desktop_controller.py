"""Phase 6C — the desktop controller with a mocked transport.

Send → reply + confirmation card, confirm sends the EXACT phrase (never a generic
'yes'), cancel, grounded ask, and the no-database / thin-client guarantees."""

import pytest

from desktop.client import ApiClient
from desktop.controller import DesktopController, Turn
from desktop.models import ConfirmationCardVM


class FakeTransport:
    """Scriptable (method, path) -> (status, body). Records calls for assertions."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, path, *, json=None, params=None, headers=None):
        self.calls.append({"method": method, "path": path, "json": json, "headers": headers})
        key = (method, path)
        handler = self.routes.get(key)
        body = handler(json) if callable(handler) else (handler or {})
        return 200, body


def make(routes):
    transport = FakeTransport(routes)
    return DesktopController(ApiClient(transport=transport)), transport


# The bridge's /ui/pending carries the EXACT machine phrase (confirm_phrase) plus the
# human display string (required_confirmation_phrase).
PENDING_EMAIL = {"data": {"pending": {
    "domain": "email_send", "action_id": "a1", "target": "ana@example.com",
    "expires_at": "2026-07-15T12:10:00",
    "required_confirmation_phrase": "confirmar envío a <destinatarios>",
    "confirm_phrase": "confirmar envío a ana@example.com", "status": "pending"}}}


def test_send_returns_reply_and_card():
    ctrl, _ = make({
        ("POST", "/command"): lambda j: {"status": "needs_confirmation",
                                         "assistant_message": "Please confirm.",
                                         "result": {"domain": "email", "action_id": "a1"}},
        ("GET", "/ui/pending"): PENDING_EMAIL,
    })
    turn = ctrl.send("email ana")
    assert isinstance(turn, Turn)
    assert turn.reply.text == "Please confirm."
    # the card SENDS the exact machine phrase (recipient included), not the display str
    assert turn.card is not None and turn.card.phrase == "confirmar envío a ana@example.com"
    assert turn.card.can_confirm is True
    assert turn.user.text == "email ana" and turn.user.role == "user"


def test_confirm_sends_exact_phrase():
    ctrl, transport = make({
        ("POST", "/command"): lambda j: {"status": "executed", "assistant_message": "Sent."},
    })
    card = ConfirmationCardVM(domain="email", action_id="a1", target="ana",
                             phrase="confirmar envío", can_confirm=True)
    turn = ctrl.confirm(card)
    assert turn.reply.text == "Sent."
    # the exact phrase was sent as the command text — no confirm flag, no bare 'yes'
    posted = [c for c in transport.calls if c["path"] == "/command"][-1]
    assert posted["json"]["text"] == "confirmar envío"
    assert posted["json"]["confirm"] is False


def test_confirm_refuses_card_without_phrase():
    ctrl, transport = make({("POST", "/command"): {"status": "executed"}})
    card = ConfirmationCardVM(domain="x", action_id="a", target="t", phrase="", can_confirm=False)
    turn = ctrl.confirm(card)
    assert turn.reply.role == "system" and turn.reply.status == "refused"
    # nothing was sent to the backend
    assert not any(c["path"] == "/command" for c in transport.calls)


def test_client_confirm_rejects_blank_phrase():
    client = ApiClient(transport=FakeTransport({}))
    with pytest.raises(ValueError):
        client.confirm("   ")


def test_ask_returns_grounded_answer_with_citations():
    ctrl, _ = make({
        ("POST", "/knowledge/ask"): lambda j: {"status": "ok", "data": {
            "answer": "Net 30 days.", "grounded": True,
            "citations": [{"filename": "contract.pdf", "location": "p. 3",
                           "citation": "contract.pdf (p. 3)"}]}},
    })
    ans = ctrl.ask("payment terms?")
    assert ans.grounded is True and ans.message.text == "Net 30 days."
    assert ans.citations[0].document == "contract.pdf"


def test_cancel():
    ctrl, _ = make({("POST", "/pending/cancel"): {"status": "cancelled"}})
    turn = ctrl.cancel()
    assert turn.reply.role == "system" and "Cancelled" in turn.reply.text


def test_controller_holds_no_database_or_store():
    ctrl, _ = make({})
    # the thin client owns only its transport + token; no db/engine/store/connection
    forbidden = ("db", "engine", "store", "conn", "connection", "session_db", "repo")
    assert not any(hasattr(ctrl, a) for a in forbidden)
    assert not any(hasattr(ctrl.client, a) for a in forbidden)


def test_locked_backend_reply_is_surfaced_verbatim():
    ctrl, _ = make({
        ("POST", "/command"): lambda j: {"status": "rejected",
                                         "assistant_message": "Unlock Fifi to do that.",
                                         "result": {"domain": "security", "error_code": "LOCKED"}},
    })
    turn = ctrl.send("open my memory")
    assert turn.reply.error_code == "LOCKED" and turn.card is None
