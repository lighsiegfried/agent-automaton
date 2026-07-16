"""The five profiles as explicit policy bundles (Phase 6B, item 2)."""

from __future__ import annotations

LOCKED = "locked"
GUEST = "guest"
STANDARD = "standard"
TRUSTED = "trusted"
DEVELOPER = "developer"

PROFILES = (LOCKED, GUEST, STANDARD, TRUSTED, DEVELOPER)
PROFILE_RANK = {LOCKED: 0, GUEST: 1, STANDARD: 2, TRUSTED: 3, DEVELOPER: 4}

# Human-readable, explicit descriptions (shown in the UI/tray; not AI-generated).
PROFILE_SUMMARY = {
    LOCKED: "Health/status only; unlock required for anything else.",
    GUEST: "Public browser reading and non-sensitive chat; no personal memory, "
           "documents or accounts.",
    STANDARD: "Owner-scoped memory and Knowledge Vault, tasks/schedules/local drafts; "
              "external effects are simulated.",
    TRUSTED: "Real text/browser/message/email effects when their feature flags are on; "
             "every domain-specific confirmation still required.",
    DEVELOPER: "Diagnostics, traces and test controls — no secret display and no safety "
               "bypass.",
}


def is_profile(value: str) -> bool:
    return value in PROFILE_RANK
