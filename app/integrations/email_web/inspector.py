"""Interpret email session reads: thread signatures, body hashing, safe summaries.

Operates on already-extracted, SAFE data (visible sender/subject/date/text) — the
adapter's extractor excludes scripts, hidden elements, and tracking pixels. This
module hashes state for change-detection and builds summaries that carry only
visible content (never raw HTML, cookies, or tokens).
"""

from __future__ import annotations

import hashlib


def thread_signature(subject: str, sender: str) -> str:
    return hashlib.sha256(f"thread:{subject}|{sender}".encode("utf-8")).hexdigest()[:16]


def body_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def compose_signature(compose: dict) -> str:
    """Identity over To/CC/subject/body — a change here invalidates a send."""
    to = ",".join(sorted(compose.get("to") or []))
    cc = ",".join(sorted(compose.get("cc") or []))
    payload = f"{to}|{cc}|{compose.get('subject','')}|{compose.get('body','')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def safe_message_summary(message: dict, max_chars: int) -> dict:
    """A SAFE per-message summary — visible fields + truncated visible body."""
    body = (message.get("body") or "")[:max_chars]
    return {
        "sender": message.get("sender", ""),
        "recipients": message.get("recipients", []),
        "date": message.get("date", ""),
        "subject": message.get("subject", ""),
        "body": body,
        "truncated": len(message.get("body") or "") > max_chars,
    }


def safe_thread_summary(thread: dict, max_chars: int) -> dict:
    messages = thread.get("messages") or []
    return {
        "subject": thread.get("subject", ""),
        "sender": thread.get("sender", ""),
        "recipients": thread.get("recipients", []),
        "date": thread.get("date", ""),
        "message_count": len(messages),
        "attachment_count": thread.get("attachment_count", 0),
        "messages": [safe_message_summary(m, max_chars) for m in messages],
    }
