"""Deterministic, resumable multi-step tasks for Fifi (Phase 5B).

A task is an ordered plan over the EXISTING services (memory search; browser
search/read/summarize; local text drafting + insertion; WhatsApp drafting/sending;
email search/summarize/drafting/sending). An LLM may PROPOSE a plan, but
deterministic code validates every intent and argument against the service
registry — arbitrary tools, Python, shell, selectors, and unknown URLs are
rejected, cycles and over-long plans are refused, and every step is classified as
read-only, local mutation, or external effect. The LLM never executes steps.

Execution reuses the one global pending broker and every domain's exact
confirmation: safe steps run automatically after plan approval, but each effect
(typing, form-fill, WhatsApp/email sending) pauses and requires its own domain
phrase. A plain "sí", a wake event, and plan approval can never confirm an effect.
State is persisted after every transition to a separate local SQLite database, so a
paused task can be inspected after a restart — but nothing auto-resumes, completed
effects never run twice, and cancellation never pretends an effect was rolled back.

Importing this package registers the task service's intents, NL patterns, and the
effect-completion listener with the core dispatcher.
"""

from app.tasks import service as _service  # noqa: F401  (import registers)
