"""Native desktop app for Fifi (Phase 6C).

A PySide6 client (NOT Electron) that is a THIN presentation layer over the local API:

- :mod:`desktop.client`      — loopback-only HTTP client (injectable transport)
- :mod:`desktop.controller`  — conversational orchestration (send / confirm / ask)
- :mod:`desktop.viewmodels`  — raw JSON → sanitized, display-ready view-models
- :mod:`desktop.models`      — the immutable VMs the UI renders
- :mod:`desktop.stream`      — SSE consumer for the safe activity stream
- :mod:`desktop.settingsview`— read-only, no-secret settings rows
- :mod:`desktop.safety`      — sanitize / loopback / no-generic-confirm rules
- :mod:`desktop.qt_app`      — the thin Qt adapter (lazy PySide6; not unit-tested)

No business logic and no database live here; the backend remains the single source of
truth and the single place effects are authorized and confirmed."""
