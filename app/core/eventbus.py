"""A tiny, neutral in-process event bus (Phase 5D).

Subsystems ``emit`` safe observability signals here; the Activity Center subscribes.
This keeps subsystems coupled only to a dependency-free core primitive — never to the
UI. ``emit`` is fire-and-forget: a failing or slow sink can never raise into, or
block, the command path (each sink is wrapped and best-effort).

This module has NO imports beyond stdlib logging, so any subsystem can use it without
creating an import cycle.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_SINKS: list = []


def subscribe(fn) -> None:
    """Register a sink ``fn(event: dict)``. Idempotent — the same callable is only
    added once, so re-imported hook installers never duplicate delivery."""
    if fn not in _SINKS:
        _SINKS.append(fn)


def unsubscribe(fn) -> None:
    if fn in _SINKS:
        _SINKS.remove(fn)


def emit(**event) -> None:
    """Publish a safe event to all sinks. Never raises; never blocks the caller on a
    sink failure. Callers pass only safe, redaction-ready fields."""
    for sink in list(_SINKS):
        try:
            sink(event)
        except Exception:  # a broken sink must never affect the command path
            log.debug("event sink failed", exc_info=True)


def reset() -> None:
    """Drop all sinks (tests)."""
    _SINKS.clear()
