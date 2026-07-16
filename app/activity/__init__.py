"""Local Activity Center — unified observability + safe control UI (Phase 5D).

A read-optimized, localhost-only view over every Fifi subsystem. Safe events are
AGGREGATED from each domain through additive event hooks (subsystems emit to a
neutral core bus and never depend on this package or the UI) into a separate local
SQLite database. Only redacted, allowlisted metadata is stored — never secrets,
tokens, cookies, form values, or full email/message/draft bodies.

Every control the UI offers (confirm/cancel a pending action; approve/resume/cancel a
task; pause/resume/cancel a schedule; retry a health check; restart an OWNED service;
release VRAM) is routed through the EXISTING services and the one global pending
broker. The Activity Center never mutates a subsystem database directly, never offers
a generic "confirm everything", and never kills unrelated processes.

Importing this package installs the (idempotent) event sink.
"""

from app.activity import events as _events  # noqa: F401  (installs the sink)

_events.install()
