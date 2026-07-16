"""Models for text actions: the verified target summary and the pending draft.

The public preview deliberately excludes anything sensitive — no full window
contents, no filesystem paths (the executable is a basename only), no internal
hashes. Change-detection data (target signature, draft hash) stays server-side.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import BaseModel

# Action kinds. Drafting/rewriting produce text; the rest describe how a
# confirmed insertion writes it.
DRAFT_TEXT = "draft_text"
REWRITE_TEXT = "rewrite_text"
TYPE_TEXT = "type_text"
APPEND_TEXT = "append_text"
REPLACE_SELECTED_TEXT = "replace_selected_text"

INSERTION_ACTIONS = {TYPE_TEXT, APPEND_TEXT, REPLACE_SELECTED_TEXT}

# Safety status of a draft/preview.
SAFE = "safe"
NEEDS_CONFIRMATION = "needs_confirmation"
BLOCKED = "blocked"


class TargetSummary(BaseModel):
    """A SAFE summary of the focused window/control — no contents, no paths."""

    application: str            # canonical app name, e.g. "notepad"
    window_title: str           # the visible title (not the document body)
    process: str                # process image name, e.g. "notepad.exe"
    executable: str             # BASENAME only — never a full filesystem path
    control_type: str           # e.g. "Edit", "Document", "Text"
    editable: bool
    supported: bool             # app is in TEXT_ALLOWED_APPS
    is_password_field: bool = False
    is_terminal: bool = False
    is_elevated: bool = False
    blocked: bool = False
    blocked_reason: str = ""

    def signature(self) -> str:
        """Identity used to detect that the target changed between draft/confirm."""
        return f"{self.process.lower()}|{self.window_title}|{self.control_type}"


@dataclass
class PendingAction:
    """The single in-memory pending text action. Never persisted to disk."""

    action_id: str
    action_type: str
    text: str
    language: str
    target: TargetSummary
    safety_status: str
    created_monotonic: float
    expires_at_monotonic: float
    expires_at_iso: str

    def is_expired(self, now_monotonic: float) -> bool:
        return now_monotonic >= self.expires_at_monotonic

    def target_signature(self) -> str:
        return self.target.signature()

    def draft_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def preview(self) -> dict:
        """Public draft-preview object (item 3) — no secrets, no paths, no hash."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "text": self.text,
            "language": self.language,
            "character_count": len(self.text),
            "target_application": self.target.application,
            "target_window_title": self.target.window_title,
            "target_control_type": self.target.control_type,
            "expires_at": self.expires_at_iso,
            "safety_status": self.safety_status,
        }
