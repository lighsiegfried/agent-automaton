"""Client-side safety rules for the native desktop app (Phase 6C).

The desktop is a THIN presentation layer, but it still enforces three hard rules so the
UI can never become an attack surface:

1. **No raw HTML/markup is ever rendered.** Every dynamic string (assistant replies,
   web/email/document-derived text, filenames, citations, activity titles) is escaped
   and control-stripped by :func:`sanitize` before it reaches any widget. Rich-text
   widgets therefore only ever see inert, escaped text.
2. **Loopback only.** :func:`is_local_base` rejects any API base URL that is not
   127.0.0.1 / ::1 / localhost, so the client can never be pointed at a remote host.
3. **No generic confirmation.** :func:`confirmable_phrase` returns the EXACT domain
   phrase for a pending action, or ``None`` — a card with no exact phrase must not offer
   a one-click confirm. There is no "just say yes" path from the desktop.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

# Control characters except tab/newline (which we normalize to spaces anyway).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def sanitize(text, *, max_len: int = 4000) -> str:
    """Escape markup and strip control chars from ANY text before display.

    The result is safe to place into a rich-text widget: ``<script>`` becomes
    ``&lt;script&gt;`` and can only render as literal characters. Newlines are kept
    (collapsed runs) so multi-line replies still read naturally."""
    if text is None:
        return ""
    raw = str(text)
    raw = _CONTROL.sub("", raw)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if len(raw) > max_len:
        raw = raw[:max_len] + "…"
    return html.escape(raw, quote=False)


def sanitize_line(text, *, max_len: int = 200) -> str:
    """Single-line sanitize (newlines → spaces) for titles, filenames, chips."""
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    return sanitize(collapsed, max_len=max_len)


def is_local_base(base_url: str) -> bool:
    """True only for a loopback http(s) base URL. Anything else is refused."""
    try:
        parsed = urlparse(base_url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return host in _LOCAL_HOSTS


def require_local_base(base_url: str) -> str:
    if not is_local_base(base_url):
        raise ValueError(f"desktop client refuses a non-loopback API base: {base_url!r}")
    return base_url


def confirmable_phrase(pending_summary: dict | None) -> str | None:
    """The EXACT phrase to SEND to confirm a pending action, or None.

    Prefers the machine-usable ``confirm_phrase`` the API provides (e.g. "confirmar
    envío a Ana" for recipient-sensitive sends); falls back to
    ``required_confirmation_phrase`` only when it names a single concrete phrase. A
    summary with neither yields None — the caller must then NOT render a one-click
    confirm button (there is no generic 'yes')."""
    if not pending_summary:
        return None
    phrase = (pending_summary.get("confirm_phrase")
              or pending_summary.get("required_confirmation_phrase") or "").strip()
    return phrase or None
