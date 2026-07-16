"""Server-Sent Events bridge for the desktop app (Phase 6C).

Subsystems ``emit`` safe observability events onto the in-process event bus (Phase 5D).
This module turns those into an SSE byte stream for the desktop client. Two rules:

- Only WHITELISTED, non-content fields ever leave the process — never draft text,
  message bodies, document contents, file paths, secrets, or hashes.
- The bridge NEVER blocks the command path: the event-bus sink pushes onto a bounded
  queue and DROPS (with a counter) if the client stalls, rather than back-pressuring.

The pure pieces — ``safe_event`` and ``format_sse`` — are unit-tested; only the async
drain loop in ``app/uibridge/api.py`` needs a running server.
"""

from __future__ import annotations

import json
import queue

from app.core import eventbus

# Fields that are safe to surface to the UI. Everything else is dropped.
_ALLOWED_SEVERITY = {"info", "success", "warning", "error"}
_TITLE_MAX = 120


def _clip(value, limit: int = _TITLE_MAX) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text[:limit]


def safe_event(event: dict) -> dict:
    """Project a raw event bus dict onto the SAFE, whitelisted shape sent to the UI.

    Titles are clipped and newline-stripped; the desktop client sanitizes again before
    display (defense in depth), so even a maliciously-crafted title can only ever be
    shown as inert, escaped text."""
    severity = str(event.get("severity", "info") or "info").lower()
    if severity not in _ALLOWED_SEVERITY:
        severity = "info"
    return {
        "domain": _clip(event.get("domain"), 40),
        "event_type": _clip(event.get("event_type"), 60),
        "status": _clip(event.get("status"), 40),
        "severity": severity,
        "error_code": _clip(event.get("error_code"), 60) or None,
        "title": _clip(event.get("title")),
        "related_id": _clip(event.get("related_id"), 80) or None,
    }


def format_sse(event: dict, *, event_id=None, event_name: str = "activity") -> str:
    """Encode ONE already-safe event as an SSE frame."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_name}")
    lines.append("data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"


def heartbeat() -> str:
    """An SSE comment line — keeps the connection alive without delivering an event."""
    return ": keep-alive\n\n"


class EventStream:
    """Bridges the synchronous event bus to a bounded queue an async loop drains.

    ``subscribe`` attaches to the bus; the sink is non-blocking and drops events when
    the queue is full so a slow/absent client can never stall a command. ``dropped``
    counts what was shed so the UI/telemetry can note the gap."""

    def __init__(self, maxsize: int = 1000):
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def _sink(self, event: dict) -> None:
        try:
            self._q.put_nowait(safe_event(event))
        except queue.Full:
            self.dropped += 1        # never block the emitter

    def subscribe(self) -> "EventStream":
        eventbus.subscribe(self._sink)
        return self

    def unsubscribe(self) -> None:
        eventbus.unsubscribe(self._sink)

    def get(self, timeout: float):
        """Block up to ``timeout`` for the next safe event; ``None`` on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def push_for_test(self, event: dict) -> None:
        """Feed the bus sink directly (tests, without a live emitter)."""
        self._sink(event)
