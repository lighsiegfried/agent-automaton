"""Security audit redaction (Phase 6B, item 11-12).

Security audit records carry only safe fields (event, domain, effect, profile, safe
identifier, decision, code). This scrubs any value that would leak a password hash,
salt, secret, or token — a defensive backstop, since the service never passes those.
"""

from __future__ import annotations

import re

_FORBIDDEN = re.compile(r"\b(salt|hash|password|passwd|secret|token|verifier|pbkdf2)\b\s*[:=]?\s*\S+",
                        re.IGNORECASE)
_HEXY = re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE)   # a hash/salt-looking hex blob


def redact(detail) -> str:
    if not detail:
        return ""
    text = str(detail)
    text = _FORBIDDEN.sub("[redacted]", text)
    text = _HEXY.sub("[redacted]", text)
    return text


def safe_events(events: list[dict]) -> list[dict]:
    return [{**e, "detail": redact(e.get("detail"))} for e in events]
