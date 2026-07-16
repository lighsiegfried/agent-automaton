"""Client-side Server-Sent Events consumer for the desktop app (Phase 6C).

Parses the SSE frames from ``GET /ui/stream`` into sanitized :class:`ActivityVM`s and
hands them to a callback. The line source is injected (any iterable of decoded lines),
so the parser is unit-tested without a socket; the default source is a lazy httpx
streaming response. The activity events are already whitelisted server-side, and are
re-sanitized here before display (defense in depth)."""

from __future__ import annotations

import json

from desktop.models import ActivityVM
from desktop.safety import require_local_base
from desktop import viewmodels as vm


def parse_sse_lines(lines):
    """Yield ``(event_name, data_dict)`` per complete SSE frame.

    A frame ends on a blank line; ``event:`` names it (default ``message``); ``data:``
    lines are concatenated and JSON-decoded. Comment lines (``:`` heartbeats) and
    malformed data are skipped, never raised."""
    event_name = "message"
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line == "":                       # dispatch the buffered frame
            if data_parts:
                payload = "\n".join(data_parts)
                try:
                    yield event_name, json.loads(payload)
                except (ValueError, TypeError):
                    pass
            event_name, data_parts = "message", []
            continue
        if line.startswith(":"):             # comment / heartbeat
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or "message"
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
        # any other field (id:, retry:) is ignored
    # flush a trailing frame with no terminating blank line
    if data_parts:
        payload = "\n".join(data_parts)
        try:
            yield event_name, json.loads(payload)
        except (ValueError, TypeError):
            pass


def events_from_lines(lines):
    """Yield sanitized :class:`ActivityVM` for each ``activity`` frame (the ``hello``
    handshake and heartbeats are skipped)."""
    for name, data in parse_sse_lines(lines):
        if name == "activity" and isinstance(data, dict):
            yield vm.activity_from_event(data)


class StreamClient:
    """Streams ``/ui/stream`` and calls ``on_event(ActivityVM)`` per activity frame."""

    def __init__(self, base_url: str, token: str, *, line_source=None):
        self.base_url = require_local_base(base_url.rstrip("/"))
        self.token = token
        self._line_source = line_source
        self._stop = False

    def _default_lines(self):  # pragma: no cover - needs a running server
        import httpx

        headers = {"Authorization": f"Bearer {self.token}", "Accept": "text/event-stream"}
        with httpx.stream("GET", f"{self.base_url}/ui/stream", headers=headers,
                          timeout=None) as resp:
            for line in resp.iter_lines():
                yield line

    def run(self, on_event) -> None:
        """Block, delivering activity events until :meth:`stop` or the source ends."""
        source = self._line_source if self._line_source is not None else self._default_lines()
        for event in events_from_lines(source):
            if self._stop:
                break
            on_event(event)

    def stop(self) -> None:
        self._stop = True
