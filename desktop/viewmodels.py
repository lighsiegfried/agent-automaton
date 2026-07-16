"""Pure builders: raw API JSON → sanitized view-models (Phase 6C).

Every function here is deterministic and side-effect free, and EVERY dynamic string it
copies is passed through :mod:`desktop.safety` first. This is the one place document/
web/email-derived text is turned into display data, so escaping here guarantees the Qt
layer only ever receives inert text."""

from __future__ import annotations

from desktop.models import ActivityVM, AnswerVM, CitationVM, ConfirmationCardVM, MessageVM
from desktop.safety import confirmable_phrase, sanitize, sanitize_line


def _result_field(command_response: dict, key: str, default=None):
    result = command_response.get("result")
    if isinstance(result, dict):
        if key in result:
            return result[key]
        data = result.get("data")
        if isinstance(data, dict) and key in data:
            return data[key]
    return default


def message_from_command(command_response: dict) -> MessageVM:
    """Build the assistant reply VM from a /command response (already-safe spoken line,
    re-sanitized defensively)."""
    text = command_response.get("assistant_message") or command_response.get("message") or ""
    return MessageVM(
        role="assistant",
        text=sanitize(text),
        status=sanitize_line(command_response.get("status", ""), max_len=40),
        error_code=(_result_field(command_response, "error_code") or None),
        domain=sanitize_line(_result_field(command_response, "domain", "") or "", max_len=40),
    )


def user_message(text: str) -> MessageVM:
    return MessageVM(role="user", text=sanitize(text))


def citations_from_ask(ask_data: dict) -> tuple:
    """CitationVMs from a knowledge/ask ``data`` block. Each citation exposes ONLY the
    filename + page/section location — never the document excerpt/content."""
    out = []
    for c in (ask_data or {}).get("citations", []) or []:
        if not isinstance(c, dict):
            continue
        label = c.get("citation") or f"{c.get('filename', '')} ({c.get('location', '')})"
        out.append(CitationVM(
            label=sanitize_line(label),
            document=sanitize_line(c.get("filename", "")),
            location=sanitize_line(c.get("location", "")),
        ))
    return tuple(out)


def answer_from_ask(ask_data: dict) -> AnswerVM:
    """Grounded-answer VM. When ``grounded`` is false (insufficient evidence) there are
    no citations and the message says so — the UI must render that plainly."""
    data = ask_data or {}
    msg = MessageVM(role="assistant", text=sanitize(data.get("answer", "")),
                    domain="knowledge",
                    status="grounded" if data.get("grounded") else "insufficient_evidence")
    citations = citations_from_ask(data) if data.get("grounded") else tuple()
    return AnswerVM(message=msg, citations=citations, grounded=bool(data.get("grounded")))


def confirmation_from_pending(pending_summary: dict | None) -> ConfirmationCardVM | None:
    """Build a confirmation card from a /pending summary, or None when nothing pends.

    The card carries the EXACT domain phrase; ``can_confirm`` is False when no concrete
    phrase is available, so the UI shows the card without a one-click confirm (never a
    generic 'yes')."""
    if not pending_summary:
        return None
    phrase = confirmable_phrase(pending_summary)      # the exact phrase to SEND
    display = (pending_summary.get("required_confirmation_phrase") or phrase or "")
    return ConfirmationCardVM(
        domain=sanitize_line(pending_summary.get("domain", ""), max_len=40),
        action_id=sanitize_line(pending_summary.get("action_id", ""), max_len=80),
        target=sanitize_line(pending_summary.get("target", "")),
        phrase=phrase or "",
        display_phrase=sanitize_line(display, max_len=80),
        expires_at=sanitize_line(pending_summary.get("expires_at", ""), max_len=40),
        can_confirm=phrase is not None,
    )


def activity_from_event(event: dict) -> ActivityVM:
    """A safe SSE activity event → ActivityVM (titles re-sanitized before display)."""
    return ActivityVM(
        domain=sanitize_line(event.get("domain", ""), max_len=40),
        event_type=sanitize_line(event.get("event_type", ""), max_len=60),
        status=sanitize_line(event.get("status", ""), max_len=40),
        severity=sanitize_line(event.get("severity", "info"), max_len=20) or "info",
        title=sanitize_line(event.get("title", "")),
        error_code=(sanitize_line(event.get("error_code", ""), max_len=60) or None),
    )
