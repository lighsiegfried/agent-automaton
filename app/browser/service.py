"""BrowserService: one isolated session + one pending form-fill plan.

Safe operations (navigate/search/read/find/scroll/open-link) run directly.
Filling a form is a two-step, confirmed action: prepare_form() builds a single
pending plan (skipping sensitive fields), and confirm_form() requires an exact
phrase, re-verifies the page is unchanged, then fills the fields WITHOUT ever
submitting. Every URL is validated before navigation; observability logs only
the URL origin, action, and result — never form values, page text, or cookies.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from app.browser import actions, inspector, safety
from app.browser.session import create_session
from app.config import get_settings
from app.core.logger import get_logger

log = get_logger(__name__)

_DISABLED = {"status": "disabled",
             "message": "Browser automation is off. Set ENABLE_BROWSER_AUTOMATION=true."}


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else "(none)"


class BrowserService:
    def __init__(self, *, session_factory=create_session, clock=time.monotonic,
                 now_utc=None, id_factory=None):
        self._session = None
        self._pending = None            # pending form plan
        self._pending_meta = None       # {expires_monotonic, signature, values}
        self._last_activity = None      # service-tracked idle clock
        self._session_factory = session_factory
        self._clock = clock
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)

    # -- session lifecycle ---------------------------------------------------------

    def _enabled(self, settings) -> bool:
        return bool(settings.enable_browser_automation)

    def _touch(self) -> None:
        self._last_activity = self._clock()

    def _idle_expired(self, settings) -> bool:
        if self._session is None or self._last_activity is None:
            return False
        return (self._clock() - self._last_activity) > settings.browser_session_idle_seconds

    def start_session(self, settings=None) -> dict:
        settings = settings or get_settings()
        if not self._enabled(settings):
            return _DISABLED
        if self._session is not None and not self._idle_expired(settings):
            return {"status": "ok", "reused": True, **self.status(settings)}
        if self._session is not None:
            self.close_session(settings)
        self._session = self._session_factory(settings)
        self._touch()
        log.info("browser session started (engine=%s headless=%s isolated=%s)",
                 settings.browser_engine, settings.browser_headless,
                 settings.browser_profile_mode)
        return {"status": "ok", "reused": False, **self.status(settings)}

    def status(self, settings=None) -> dict:
        settings = settings or get_settings()
        if self._session is None:
            return {"active": False}
        return {
            "active": True,
            "url": self._session.current_url(),
            "title": self._session.title(),
            "has_pending_form": self._pending is not None,
        }

    def close_session(self, settings=None) -> dict:
        self._pending, self._pending_meta = None, None
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        return {"status": "closed"}

    def _require(self, settings) -> dict | None:
        if not self._enabled(settings):
            return _DISABLED
        if self._session is None:
            return {"status": "error", "reason": "no active browser session — start one first"}
        if self._idle_expired(settings):
            self.close_session(settings)
            return {"status": "error", "reason": "session expired (idle) — start a new one"}
        return None

    # -- navigation / reading (safe) ----------------------------------------------

    def _snapshot(self, settings) -> dict:
        self._touch()
        raw = self._session.snapshot_raw()
        return inspector.build_snapshot(raw, settings.browser_max_page_text_chars)

    def navigate(self, url: str, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        verdict = safety.validate_url(url, settings.browser_allowed_schemes_list,
                                      settings.browser_block_private_networks)
        if not verdict["ok"]:
            log.info("browser navigate blocked: %s", verdict["reason"])
            return {"status": "rejected", "reason": verdict["reason"], "url": verdict["url"]}
        self._session.goto(verdict["url"])
        log.info("browser navigate origin=%s", _origin(verdict["url"]))
        return {"status": "ok", "page": self._snapshot(settings)}

    def search(self, query: str, settings=None) -> dict:
        return self.navigate(actions.search_url(query), settings)

    def go_back(self, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        self._session.back()
        return {"status": "ok", "page": self._snapshot(settings)}

    def go_forward(self, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        self._session.forward()
        return {"status": "ok", "page": self._snapshot(settings)}

    def page(self, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        return {"status": "ok", "page": self._snapshot(settings)}

    def find(self, query: str, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        return {"status": "ok", **inspector.find_text(self._snapshot(settings), query)}

    def scroll(self, direction: str, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        self._session.scroll(actions.scroll_delta(direction))
        self._touch()
        return {"status": "ok", "direction": direction}

    def open_link(self, text: str, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        resolved = inspector.resolve_link(self._snapshot(settings), text=text)
        if not resolved["ok"]:
            return {"status": "rejected", "reason": resolved["reason"]}
        href = resolved["link"]["href"]
        verdict = safety.validate_url(href, settings.browser_allowed_schemes_list,
                                      settings.browser_block_private_networks)
        if not verdict["ok"]:
            return {"status": "rejected", "reason": verdict["reason"], "url": href}
        self._session.goto(verdict["url"])
        log.info("browser open_link origin=%s", _origin(verdict["url"]))
        return {"status": "ok", "page": self._snapshot(settings)}

    # -- controlled form filling (sensitive) --------------------------------------

    def prepare_form(self, requested_fields: dict, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        snapshot = self._snapshot(settings)
        expires_iso = (self._now_utc() + timedelta(
            seconds=settings.browser_action_timeout_seconds)).isoformat(timespec="seconds")
        plan = actions.build_form_plan(
            snapshot, requested_fields, action_id=self._id_factory(), expires_at_iso=expires_iso)
        if not plan["fields"]:
            return {"status": "rejected",
                    "reason": "no fillable fields (all were sensitive or unknown)",
                    "plan": plan}
        self._pending = plan
        self._pending_meta = {
            "expires_monotonic": self._clock() + settings.browser_action_timeout_seconds,
            "signature": actions.page_signature(snapshot["url"], snapshot["title"]),
            "values": {f["name"]: f["value"] for f in plan["fields"]},
        }
        log.info("browser form prepared fields=%d skipped_sensitive=%d origin=%s",
                 len(plan["fields"]), len(plan["skipped_sensitive_fields"]),
                 _origin(snapshot["url"]))
        return {"status": "needs_confirmation", "plan": plan,
                "confirmation_phrases": sorted(safety.CONFIRMATION_PHRASES)}

    def pending_form(self, settings=None) -> dict | None:
        settings = settings or get_settings()
        if self._pending is None:
            return None
        if self._clock() >= self._pending_meta["expires_monotonic"]:
            self.cancel_form()
            return None
        return self._pending

    def confirm_form(self, action_id: str, phrase: str, settings=None) -> dict:
        settings = settings or get_settings()
        guard = self._require(settings)
        if guard:
            return guard
        if self._pending is None:
            return {"status": "rejected", "reason": "there is no pending form action"}
        if action_id and action_id != self._pending["action_id"]:
            return {"status": "rejected", "reason": "action_id does not match the pending form"}
        if self._clock() >= self._pending_meta["expires_monotonic"]:
            self.cancel_form()
            return {"status": "rejected", "reason": "the pending form expired — prepare again"}
        if not safety.is_confirmation_phrase(phrase):
            return {"status": "needs_confirmation",
                    "reason": "say or type an exact confirmation phrase",
                    "confirmation_phrases": sorted(safety.CONFIRMATION_PHRASES),
                    "plan": self._pending}

        # Re-verify the page has not changed (URL, title, or the field set).
        snapshot = self._snapshot(settings)
        signature = actions.page_signature(snapshot["url"], snapshot["title"])
        if signature != self._pending_meta["signature"]:
            self.cancel_form()
            return {"status": "rejected",
                    "reason": "the page changed since the form was prepared — cancelled"}
        current_fillable = {c["name"] for c in snapshot.get("form_controls", []) if c.get("fillable")}
        plan_fields = {f["name"] for f in self._pending["fields"]}
        if not plan_fields.issubset(current_fillable):
            self.cancel_form()
            return {"status": "rejected",
                    "reason": "the form fields changed since preparation — cancelled"}

        # Fill each field. NEVER submit / press Enter (submit is blocked this phase).
        filled = []
        for field in self._pending["fields"]:
            self._session.fill(field["name"], field["value"])
            filled.append(field["name"])
        plan = self._pending
        self.cancel_form()   # consume — no duplicate fill
        log.info("browser form filled fields=%d submitted=False origin=%s",
                 len(filled), _origin(snapshot["url"]))
        return {"status": "filled", "filled_fields": filled, "submitted": False,
                "action_id": plan["action_id"]}

    def cancel_form(self, settings=None) -> dict:
        had = self._pending is not None
        self._pending, self._pending_meta = None, None
        return {"status": "cancelled", "had_pending": had}


_service: BrowserService | None = None


def get_browser_service() -> BrowserService:
    global _service
    if _service is None:
        _service = BrowserService()
    return _service
