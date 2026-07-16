"""Deterministic profiles, permissions and trust boundaries (Phase 6B).

Profiles (locked / guest / standard / trusted / developer) are explicit policy
bundles — never opaque AI scores. A single central ``authorize`` gates every service
dispatch path with default-deny. NO profile, including developer, may bypass the
mandatory invariants (recipient-specific confirmation, changed-target/hash
revalidation, secret blocking, the document prompt-injection boundary, no automatic
purchases/deletion/account changes, wake-cannot-confirm, duplicate-effect prevention);
developer only exposes diagnostics. The authenticated OS user is the identity
boundary, with an optional local unlock (Windows Hello or a salted password verifier —
never plaintext). Temporary elevation is capability-scoped, visibly expiring, audited,
and cleared on reboot.

Importing this package registers the security intents and pending-broker domain, and
installs the central authorization gate on the dispatcher.
"""

from app.security import service as _service  # noqa: F401  (import registers + installs the gate)
