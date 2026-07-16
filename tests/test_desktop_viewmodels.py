"""Phase 6C — pure view-model builders: sanitization, confirmation cards carrying the
EXACT phrase, and grounded citations (filename + location only, never content)."""

from desktop import viewmodels as vm
from desktop.models import CitationVM, ConfirmationCardVM


def test_message_from_command_sanitizes_and_reads_result():
    resp = {"status": "rejected", "assistant_message": "<b>Denied</b>",
            "result": {"domain": "email", "error_code": "LOCKED"}}
    m = vm.message_from_command(resp)
    assert "<b>" not in m.text and "&lt;b&gt;" in m.text
    assert m.status == "rejected" and m.error_code == "LOCKED" and m.domain == "email"


def test_message_prefers_assistant_message():
    resp = {"status": "ok", "message": "raw", "assistant_message": "spoken"}
    assert vm.message_from_command(resp).text == "spoken"


def test_confirmation_card_carries_exact_phrase():
    summary = {"domain": "email", "action_id": "a1", "target": "ana@example.com",
               "expires_at": "2026-07-15T12:10:00", "required_confirmation_phrase": "confirmar envío",
               "status": "pending"}
    card = vm.confirmation_from_pending(summary)
    assert isinstance(card, ConfirmationCardVM)
    assert card.phrase == "confirmar envío" and card.can_confirm is True
    assert card.target == "ana@example.com" and card.domain == "email"


def test_confirmation_card_without_phrase_cannot_confirm():
    summary = {"domain": "x", "action_id": "a", "target": "t",
               "required_confirmation_phrase": ""}
    card = vm.confirmation_from_pending(summary)
    assert card.can_confirm is False and card.phrase == ""


def test_no_pending_no_card():
    assert vm.confirmation_from_pending(None) is None


def test_confirmation_card_sanitizes_target():
    summary = {"domain": "email", "action_id": "a", "target": "<img src=x onerror=1>",
               "required_confirmation_phrase": "confirmar envío"}
    card = vm.confirmation_from_pending(summary)
    assert "<img" not in card.target


def test_citations_filename_and_location_only():
    ask = {"answer": "Net 30.", "grounded": True, "citations": [
        {"filename": "contract.pdf", "location": "p. 3", "page": 3,
         "citation": "contract.pdf (p. 3)", "excerpt": "SECRET BODY TEXT"}]}
    ans = vm.answer_from_ask(ask)
    assert ans.grounded is True and len(ans.citations) == 1
    c = ans.citations[0]
    assert isinstance(c, CitationVM)
    assert c.document == "contract.pdf" and c.location == "p. 3"
    assert c.label == "contract.pdf (p. 3)"
    # the document excerpt/content is never carried into the citation VM
    assert "SECRET BODY TEXT" not in c.label and not hasattr(c, "excerpt")


def test_insufficient_evidence_has_no_citations():
    ask = {"answer": "I don't have enough document evidence.", "grounded": False, "citations": []}
    ans = vm.answer_from_ask(ask)
    assert ans.grounded is False and ans.citations == tuple()
    assert ans.message.status == "insufficient_evidence"


def test_citations_sanitize_malicious_filename():
    ask = {"answer": "x", "grounded": True, "citations": [
        {"filename": "<script>evil</script>.pdf", "location": "p.1",
         "citation": "<script>evil</script>.pdf (p.1)"}]}
    c = vm.answer_from_ask(ask).citations[0]
    assert "<script>" not in c.label and "<script>" not in c.document


def test_activity_from_event_sanitizes_title():
    a = vm.activity_from_event({"domain": "email", "event_type": "send", "status": "ok",
                                "severity": "success", "title": "<b>x</b>"})
    assert "<b>" not in a.title and a.severity == "success"
