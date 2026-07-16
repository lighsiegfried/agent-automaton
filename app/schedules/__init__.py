"""Persistent reminders + safely scheduled task execution for Fifi (Phase 5C).

A schedule fires a reminder or runs a validated Phase-5B task template at an
absolute time or on a recurrence. Instants are stored in UTC and resolved through
the configured IANA timezone with explicit daylight-saving handling; relative
Spanish/English dates are normalized deterministically and ambiguous ones ("el
viernes" with no time) are refused. Creating or modifying a schedule is a sensitive
write confirmed through the global pending broker ("confirmar programación" — a
plain "sí" or a wake never confirms).

One owned scheduler polls on a bounded interval and acquires an atomic per-run lease
so a restart or clock change can't double-run. At run time the plan is REVALIDATED
and memory context recomputed: reminders notify, read-only and local-draft steps may
auto-run, but every external effect (WhatsApp/email send, text insertion, form fill)
pauses in the broker and needs a fresh domain-specific confirmation — schedule
approval never pre-confirms an effect, and a task paused for confirmation is never
auto-resumed.

Importing this package registers the schedule service's intents, NL patterns, and
pending-broker domain with the core dispatcher.
"""

from app.schedules import service as _service  # noqa: F401  (import registers)
