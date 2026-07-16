"""Immutable view-models the desktop UI renders (Phase 6C).

These are the ONLY things the Qt layer ever paints. Every string field is already
sanitized (see :mod:`desktop.viewmodels`), so a widget can render them directly without
re-escaping. They carry NO business logic — just safe, display-ready data."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MessageVM:
    """One conversational turn's reply."""
    role: str                       # "assistant" | "user" | "system"
    text: str                       # sanitized
    status: str = ""                # normalized dispatcher status
    error_code: str | None = None
    domain: str = ""


@dataclass(frozen=True)
class CitationVM:
    """A single grounded citation — filename + location, never document content."""
    label: str                      # sanitized "filename (page/section)"
    document: str = ""              # sanitized filename
    location: str = ""              # sanitized "p. 3" / "§ Heading"


@dataclass(frozen=True)
class ConfirmationCardVM:
    """A confirmation prompt. The button, when clicked, sends EXACTLY ``phrase`` — the
    UI never offers a generic 'yes'. ``display_phrase`` is the human-readable string
    shown on the card (may list both languages); ``phrase`` is the single machine phrase
    actually sent. ``can_confirm`` is False when no exact phrase is available, so the
    card is display-only."""
    domain: str
    action_id: str
    target: str                     # sanitized (e.g. recipient / app / document)
    phrase: str                     # the EXACT phrase sent on confirm
    display_phrase: str = ""        # sanitized human-readable phrase for the label
    expires_at: str = ""
    can_confirm: bool = False


@dataclass(frozen=True)
class ActivityVM:
    """One safe activity-stream line."""
    domain: str
    event_type: str
    status: str
    severity: str = "info"
    title: str = ""
    error_code: str | None = None


@dataclass(frozen=True)
class AnswerVM:
    """A grounded answer: a message plus its citations (possibly empty → insufficient
    evidence)."""
    message: MessageVM
    citations: tuple = field(default_factory=tuple)
    grounded: bool = False
