"""Explicit, local, auditable personal memory for Fifi (Phase 5A).

Fifi never stores anything merely because it appeared in conversation. Creating,
updating, or forgetting a memory requires an EXPLICIT user intent ("recuerda
que…", "remember that…") that becomes a pending proposal, and then an EXACT
confirmation phrase ("confirmar memoria" / "confirm memory") — a plain "sí", a
wake detection, or email/web-page content never store memory. Sensitive data
(passwords, OTPs, API keys, cards, private keys, cookies, precise addresses,
sensitive personal information) is blocked by default and never inferred.

Storage is a local SQLite file (WAL + schema migrations); there is no cloud
dependency. Retrieval is deterministic (exact entity/tag match + SQLite FTS5 +
recency) and returns only a small, bounded context bundle — never the whole
database.

Importing this package registers the memory service's intents, confirmation
handlers, NL patterns, and pending-broker domains with the core dispatcher.
"""

from app.memory import service as _service  # noqa: F401  (import registers)
