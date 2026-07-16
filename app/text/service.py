"""The text-action service: one pending draft, confirmed before any insertion.

Flow:
    draft()   -> composes text (safe), verifies the target, stores ONE pending
                 action (memory-only, expiring), returns a preview.
    confirm() -> requires an exact confirmation phrase, re-verifies the target is
                 unchanged and still safe, then inserts (simulated unless
                 ENABLE_REAL_TEXT_INPUT). Consumes the pending action so it can
                 never be inserted twice.
    cancel()  -> discards the pending action.

Every sensitive gate lives here; the API and voice/tray layers are thin.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.text import drafting, secrets
from app.text.insertion import perform_insertion
from app.text.models import (
    BLOCKED,
    INSERTION_ACTIONS,
    NEEDS_CONFIRMATION,
    TYPE_TEXT,
    PendingAction,
    TargetSummary,
)
from app.text.targets import active_target

VALID_ACTIONS = {"draft_text", "rewrite_text", "type_text", "append_text",
                 "replace_selected_text"}


def _preview_dict(action_type, text, language, target: TargetSummary, safety_status,
                  expires_at="") -> dict:
    return {
        "action_id": None,
        "action_type": action_type,
        "text": text,
        "language": language,
        "character_count": len(text),
        "target_application": target.application,
        "target_window_title": target.window_title,
        "target_control_type": target.control_type,
        "expires_at": expires_at,
        "safety_status": safety_status,
    }


class TextActionService:
    def __init__(self, *, clock=time.monotonic, now_utc=None, id_factory=None,
                 target_probe=None, inserter=perform_insertion, generator=None):
        self._pending: PendingAction | None = None
        self._clock = clock
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._target_probe = target_probe
        self._inserter = inserter
        self._generator = generator

    # -- helpers -------------------------------------------------------------------

    def _clear(self) -> None:
        self._pending = None

    def _resolve_target(self, target, settings) -> TargetSummary:
        if target is not None:
            return target
        return active_target(self._target_probe, settings)

    def pending_preview(self, settings=None) -> dict | None:
        settings = settings or get_settings()
        if self._pending is None:
            return None
        if self._pending.is_expired(self._clock()):
            self._clear()
            return None
        return self._pending.preview()

    # -- draft ---------------------------------------------------------------------

    def draft(self, *, action, text=None, instruction=None, base_text="", language="es",
              target=None, settings=None) -> dict:
        settings = settings or get_settings()
        if action not in VALID_ACTIONS:
            return {"status": "rejected", "reason": f"unknown action {action!r}"}

        target = self._resolve_target(target, settings)
        final_text = drafting.compose(
            action_type=action, text=text, instruction=instruction,
            base_text=base_text, language=language, generator=self._generator,
        )
        insertion_type = action if action in INSERTION_ACTIONS else TYPE_TEXT

        # Safety gates — produce the (safe) draft but refuse to make it pending.
        if not final_text.strip():
            return {"status": "rejected", "reason": "the draft is empty"}
        if len(final_text) > settings.text_max_characters:
            return {"status": "rejected",
                    "reason": f"draft exceeds TEXT_MAX_CHARACTERS ({settings.text_max_characters})",
                    "preview": _preview_dict(insertion_type, final_text[:200], language, target, BLOCKED)}
        is_secret, reasons = secrets.contains_likely_secret(final_text)
        if is_secret:
            return {"status": "rejected", "reason": "the draft looks like it contains a secret",
                    "secret_reasons": reasons,
                    "preview": _preview_dict(insertion_type, "[redacted]", language, target, BLOCKED)}
        if target.blocked:
            return {"status": "rejected", "reason": target.blocked_reason,
                    "preview": _preview_dict(insertion_type, final_text, language, target, BLOCKED)}

        # Create the single pending action (replaces any prior one).
        expires_seconds = settings.text_action_expires_seconds
        expires_iso = (self._now_utc() + timedelta(seconds=expires_seconds)).isoformat(
            timespec="seconds")
        pending = PendingAction(
            action_id=self._id_factory(),
            action_type=insertion_type,
            text=final_text,
            language=language,
            target=target,
            safety_status=NEEDS_CONFIRMATION,
            created_monotonic=self._clock(),
            expires_at_monotonic=self._clock() + expires_seconds,
            expires_at_iso=expires_iso,
        )
        self._pending = pending
        return {
            "status": "needs_confirmation",
            "preview": pending.preview(),
            "confirmation_phrases": sorted(secrets.CONFIRMATION_PHRASES),
            "require_real_input": settings.enable_real_text_input,
        }

    # -- confirm -------------------------------------------------------------------

    def confirm(self, *, action_id, phrase, target=None, settings=None) -> dict:
        settings = settings or get_settings()
        pending = self._pending
        if pending is None:
            return {"status": "rejected", "reason": "there is no pending text action"}
        if action_id and action_id != pending.action_id:
            # A different (or changed) draft — refuse without touching the pending.
            return {"status": "rejected",
                    "reason": "action_id does not match the pending draft (it changed)"}
        if pending.is_expired(self._clock()):
            self._clear()
            return {"status": "rejected", "reason": "the pending action expired — draft again"}
        if not secrets.is_confirmation_phrase(phrase):
            # A plain "sí" lands here: kept pending, but NOT confirmed.
            return {
                "status": "needs_confirmation",
                "reason": "say or type an exact confirmation phrase (a normal 'sí' is not enough)",
                "confirmation_phrases": sorted(secrets.CONFIRMATION_PHRASES),
                "preview": pending.preview(),
            }

        current = self._resolve_target(target, settings)
        if current.signature() != pending.target_signature():
            self._clear()
            return {"status": "rejected",
                    "reason": "the target window changed since the draft — insertion cancelled"}
        if current.blocked:
            self._clear()
            return {"status": "rejected", "reason": current.blocked_reason}
        is_secret, _ = secrets.contains_likely_secret(pending.text)
        if is_secret:
            self._clear()
            return {"status": "rejected", "reason": "the draft contains a likely secret"}

        # Passed every gate. Simulate unless real input is explicitly enabled.
        if not settings.enable_real_text_input:
            preview = pending.preview()
            self._clear()
            return {"status": "simulated", "inserted": False, "action_id": action_id,
                    "message": "ENABLE_REAL_TEXT_INPUT=false — nothing was typed (simulated).",
                    "preview": preview}

        insertion = self._inserter(pending, settings)
        self._clear()   # consumed — cannot be inserted twice
        return {
            "status": "executed" if insertion.get("ok") else "error",
            "inserted": bool(insertion.get("ok")),
            "action_id": action_id,
            **insertion,
        }

    # -- cancel --------------------------------------------------------------------

    def cancel(self) -> dict:
        had = self._pending is not None
        self._clear()
        return {"status": "cancelled", "had_pending": had}


_service: TextActionService | None = None


def get_text_service() -> TextActionService:
    global _service
    if _service is None:
        _service = TextActionService()
    return _service
