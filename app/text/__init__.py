"""Safe text drafting + controlled text insertion (Phase 4A).

Drafting and rewriting text are safe. Inserting text into a Windows application
is sensitive: it is simulated unless ENABLE_REAL_TEXT_INPUT=true, requires an
explicit confirmation phrase, never types into an unverified/terminal/elevated
window or a password/secret field, never presses Enter or any submit shortcut,
and never uses the clipboard unless explicitly configured (restoring it after).
Only one pending action may exist; it is memory-only and expires.
"""
