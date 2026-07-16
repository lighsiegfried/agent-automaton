"""Deterministic dispatcher for conversational service intents (Phase 4B.1).

The LLM/NL layer PROPOSES a structured ``{intent, arguments}``; this module
validates it, checks the feature flag, delegates the actual sensitive work to
TextActionService / BrowserService (which own every safety gate), coordinates the
single cross-domain pending action through the broker, and returns a NORMALIZED
result with a stable error code, a command state, and a SAFE spoken line. It never
executes arbitrary code, never bypasses confirmation, and never speaks secrets.
"""

from __future__ import annotations

from app.browser.service import get_browser_service
from app.config import get_settings
from app.core import errors, spoken
from app.core.logger import get_logger
from app.core.pending import get_pending_broker
from app.schemas.commands import Intent
from app.text.service import get_text_service

log = get_logger(__name__)

TEXT_INTENTS = {
    Intent.DRAFT_TEXT.value, Intent.REWRITE_TEXT.value, Intent.TYPE_TEXT.value,
    Intent.APPEND_TEXT.value, Intent.REPLACE_SELECTED_TEXT.value,
    Intent.CANCEL_TEXT_ACTION.value,
}
BROWSER_INTENTS = {
    Intent.BROWSER_OPEN.value, Intent.BROWSER_SEARCH.value, Intent.BROWSER_READ.value,
    Intent.BROWSER_SUMMARIZE.value, Intent.BROWSER_FIND.value, Intent.BROWSER_SCROLL.value,
    Intent.BROWSER_OPEN_LINK.value, Intent.BROWSER_PREPARE_FORM.value,
    Intent.BROWSER_CONFIRM_FORM.value, Intent.BROWSER_CANCEL_FORM.value,
    Intent.BROWSER_CLOSE.value,
}
SERVICE_INTENTS = TEXT_INTENTS | BROWSER_INTENTS | {"confirm", "cancel"}

TEXT_CONFIRM_PHRASE = "confirmar escritura / insert text"
BROWSER_CONFIRM_PHRASE = "confirmar formulario / confirm form fill"

# Allowed argument keys per service intent — the planner's arguments are
# validated against this (structured only; no arbitrary Python/selectors/URLs).
SERVICE_INTENT_ARGS: dict[str, set[str]] = {
    Intent.DRAFT_TEXT.value: {"instruction", "language"},
    Intent.REWRITE_TEXT.value: {"instruction", "base_text", "language"},
    Intent.TYPE_TEXT.value: {"text", "mode"},
    Intent.APPEND_TEXT.value: {"text", "mode"},
    Intent.REPLACE_SELECTED_TEXT.value: {"text", "mode"},
    Intent.CANCEL_TEXT_ACTION.value: set(),
    Intent.BROWSER_OPEN.value: {"url"},
    Intent.BROWSER_SEARCH.value: {"query"},
    Intent.BROWSER_READ.value: set(),
    Intent.BROWSER_SUMMARIZE.value: set(),
    Intent.BROWSER_FIND.value: {"query"},
    Intent.BROWSER_SCROLL.value: {"direction"},
    Intent.BROWSER_OPEN_LINK.value: {"text"},
    Intent.BROWSER_PREPARE_FORM.value: {"fields"},
    Intent.BROWSER_CONFIRM_FORM.value: {"phrase"},
    Intent.BROWSER_CANCEL_FORM.value: set(),
    Intent.BROWSER_CLOSE.value: set(),
    "confirm": {"phrase"},
    "cancel": set(),
}


def validate_service_args(intent: str, arguments: dict) -> str | None:
    """Reject arguments the intent does not accept. None means valid."""
    if intent not in SERVICE_INTENT_ARGS:
        return f"unsupported service intent {intent!r}"
    unexpected = set(arguments or {}) - SERVICE_INTENT_ARGS[intent]
    if unexpected:
        return f"unexpected arguments for {intent!r}: {sorted(unexpected)}"
    return None


# Extension registries so integrations (WhatsApp, email) add their intents and
# confirmation domains WITHOUT editing this module. Built-in text/browser stay
# hardcoded below (no risk to their tests).
INTENT_DISPATCHERS: dict[str, object] = {}    # intent -> fn(args, language, settings, broker)
CONFIRM_HANDLERS: dict[str, object] = {}       # domain -> fn(action_id, phrase, language, broker)


def register_service_intent(intent: str, arg_keys: set, dispatcher) -> None:
    SERVICE_INTENT_ARGS[intent] = set(arg_keys)
    INTENT_DISPATCHERS[intent] = dispatcher


def register_confirm_handler(domain: str, handler) -> None:
    CONFIRM_HANDLERS[domain] = handler


# Effect-completion listeners (Phase 5B). Notified with the normalized result
# after ANY confirmation resolves, so the task executor can advance a task whose
# sensitive step was just confirmed — WITHOUT the broker or the domain services
# knowing anything about tasks. Purely additive: no existing behaviour changes.
EFFECT_LISTENERS: list = []


def register_effect_listener(fn) -> None:
    EFFECT_LISTENERS.append(fn)


# The single central authorization gate (Phase 6B). When a security profile is active,
# every service command is authorized here BEFORE any work; when no gate is installed
# (or security is disabled) dispatch is unchanged. Confirmations complete an
# already-authorized action and are not re-gated.
_AUTHORIZATION_GATE = None


def register_authorization_gate(fn) -> None:
    global _AUTHORIZATION_GATE
    _AUTHORIZATION_GATE = fn


def _notify_effect_listeners(result, action_id=None) -> None:
    """``action_id`` is the pending action that was ACTIVE before the confirmation
    resolved (the email place→send flow makes the result's own action_id point at
    the next stage, so the executor needs the one it was actually waiting on)."""
    for fn in list(EFFECT_LISTENERS):
        try:
            fn(result, action_id)
        except Exception:
            log.warning("effect listener failed", exc_info=True)


def _result(status, *, state, spoken_text, code=None, domain=None, action_id=None, data=None):
    return {"status": status, "error_code": code, "domain": domain,
            "action_id": action_id, "state": state, "spoken": spoken_text,
            "data": data or {}}


# --- text ---------------------------------------------------------------------------


_TEXT_ACTION = {
    Intent.DRAFT_TEXT.value: "draft_text",
    Intent.REWRITE_TEXT.value: "rewrite_text",
    Intent.TYPE_TEXT.value: "type_text",
    Intent.APPEND_TEXT.value: "append_text",
    Intent.REPLACE_SELECTED_TEXT.value: "replace_selected_text",
}


def _dispatch_text(intent, args, language, settings, broker):
    service = get_text_service()
    if intent == Intent.CANCEL_TEXT_ACTION.value:
        service.cancel()
        broker.clear()
        return _result("cancelled", state=errors.COMPLETED,
                       spoken_text=spoken.speak("cancelled", language), domain="text")

    result = service.draft(
        action=_TEXT_ACTION[intent], text=args.get("text"),
        instruction=args.get("instruction"), base_text=args.get("base_text", ""),
        language=language,
    )
    status = result.get("status")
    if status == "needs_confirmation":
        preview = result["preview"]
        broker.register(domain="text", action_id=preview["action_id"],
                        target=preview["target_application"], expires_at=preview["expires_at"],
                        cancel=service.cancel)
        return _result("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                       domain="text", action_id=preview["action_id"],
                       spoken_text=spoken.speak("draft_prepared", language,
                                                count=preview["character_count"],
                                                app=preview["target_application"] or "the window",
                                                phrase="confirmar escritura"),
                       data={"character_count": preview["character_count"],
                             "target": preview["target_application"]})
    # rejected — classify without leaking the draft.
    reason = (result.get("reason") or "").lower()
    if "secret" in reason:
        return _result("rejected", state=errors.BLOCKED, code=errors.SENSITIVE_FIELD_BLOCKED,
                       domain="text", spoken_text=spoken.speak("secret_blocked", language))
    if "password" in reason or "terminal" in reason or "elevated" in reason \
            or "unsupported" in reason or "editable" in reason or "inspector" in reason:
        return _result("rejected", state=errors.BLOCKED, code=errors.TARGET_UNAVAILABLE,
                       domain="text", spoken_text=spoken.speak("field_blocked", language))
    return _result("rejected", state=errors.FAILED, code=errors.EXECUTION_FAILED,
                   domain="text", spoken_text=spoken.speak("failed", language))


# --- browser ------------------------------------------------------------------------


def _browser_summary(page: dict) -> dict:
    """A SAFE, compact summary (title + heading count) — never full text."""
    title = page.get("title", "")
    headings = page.get("headings", [])
    return {"title": title, "heading_count": len(headings),
            "first_heading": headings[0]["text"] if headings else ""}


def _dispatch_browser(intent, args, language, settings, broker):
    if not settings.enable_browser_automation:
        return _result("rejected", state=errors.BLOCKED, code=errors.FEATURE_DISABLED,
                       domain="browser", spoken_text=spoken.speak("feature_disabled", language))
    service = get_browser_service()

    # confirm/cancel form are routed through the confirmation path.
    if intent == Intent.BROWSER_CONFIRM_FORM.value:
        return _handle_confirmation(args.get("phrase", ""), language, False, settings, broker)
    if intent == Intent.BROWSER_CANCEL_FORM.value:
        return _handle_cancel(language, broker)

    started = service.start_session(settings)
    if started.get("status") == "disabled":
        return _result("rejected", state=errors.BLOCKED, code=errors.FEATURE_DISABLED,
                       domain="browser", spoken_text=spoken.speak("feature_disabled", language))

    try:
        return _run_browser(service, intent, args, language, settings, broker)
    except Exception as exc:   # a dead/unavailable browser must not crash the pipeline
        log.warning("browser dispatch failed: %s", exc)
        return _result("error", state=errors.FAILED, code=errors.BROWSER_UNAVAILABLE,
                       domain="browser", spoken_text=spoken.speak("browser_unavailable", language))


def _run_browser(service, intent, args, language, settings, broker):
    if intent == Intent.BROWSER_OPEN.value:
        url = (args.get("url") or "").strip()
        if not url:
            return _result("ok", state=errors.COMPLETED, domain="browser",
                           spoken_text=spoken.speak("opened", language, title=""))
        res = service.navigate(url, settings)
        return _browser_nav_result(res, language, "opened")
    if intent == Intent.BROWSER_SEARCH.value:
        res = service.search(args.get("query", ""), settings)
        return _browser_nav_result(res, language, "searched")
    if intent in (Intent.BROWSER_READ.value, Intent.BROWSER_SUMMARIZE.value):
        res = service.page(settings)
        if res.get("status") != "ok":
            return _browser_error(res, language)
        summary = _browser_summary(res["page"])
        situation = "summarized" if intent == Intent.BROWSER_SUMMARIZE.value else "opened"
        title = f" — {summary['title']}" if summary["title"] else ""
        return _result("ok", state=errors.COMPLETED, domain="browser",
                       spoken_text=spoken.speak(situation, language, title=title), data=summary)
    if intent == Intent.BROWSER_FIND.value:
        res = service.find(args.get("query", ""), settings)
        if res.get("status") != "ok":
            return _browser_error(res, language)
        count = res.get("count", 0)
        situation = "found" if count else "not_found"
        return _result("ok", state=errors.COMPLETED, domain="browser",
                       spoken_text=spoken.speak(situation, language, count=count),
                       data={"count": count})
    if intent == Intent.BROWSER_SCROLL.value:
        direction = args.get("direction", "down")
        service.scroll(direction, settings)
        return _result("ok", state=errors.COMPLETED, domain="browser",
                       spoken_text=spoken.speak("scrolled", language, direction=direction))
    if intent == Intent.BROWSER_OPEN_LINK.value:
        res = service.open_link(args.get("text", ""), settings)
        if res.get("status") == "rejected" and "ambiguous" in (res.get("reason") or "").lower():
            return _result("rejected", state=errors.BLOCKED, code=errors.AMBIGUOUS_TARGET,
                           domain="browser", spoken_text=spoken.speak("ambiguous", language))
        return _browser_nav_result(res, language, "opened")
    if intent == Intent.BROWSER_PREPARE_FORM.value:
        res = service.prepare_form(args.get("fields", {}) or {}, settings)
        if res.get("status") == "needs_confirmation":
            plan = res["plan"]
            broker.register(domain="browser", action_id=plan["action_id"],
                            target=plan.get("page_title") or plan.get("page_url", ""),
                            expires_at=plan["expires_at"], cancel=service.cancel_form)
            return _result("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                           domain="browser", action_id=plan["action_id"],
                           spoken_text=spoken.speak("form_prepared", language,
                                                    count=len(plan["fields"]),
                                                    phrase="confirmar formulario"),
                           data={"field_count": len(plan["fields"]),
                                 "skipped_sensitive": len(plan["skipped_sensitive_fields"])})
        return _result("rejected", state=errors.BLOCKED, code=errors.SENSITIVE_FIELD_BLOCKED,
                       domain="browser", spoken_text=spoken.speak("field_blocked", language))
    if intent == Intent.BROWSER_CLOSE.value:
        service.close_session(settings)
        return _result("ok", state=errors.COMPLETED, domain="browser",
                       spoken_text=spoken.speak("closed", language))
    return _result("rejected", state=errors.FAILED, code=errors.UNSUPPORTED_INTENT,
                   domain="browser", spoken_text=spoken.speak("unsupported", language))


def _browser_nav_result(res, language, situation):
    if res.get("status") != "ok":
        return _browser_error(res, language)
    title = res.get("page", {}).get("title", "")
    return _result("ok", state=errors.COMPLETED, domain="browser",
                   spoken_text=spoken.speak(situation, language,
                                            title=f" — {title}" if title else ""),
                   data=_browser_summary(res.get("page", {})))


def _browser_error(res, language):
    reason = (res.get("reason") or "").lower()
    if "no active" in reason or "expired" in reason and "session" in reason:
        code = errors.BROWSER_UNAVAILABLE
    else:
        code = errors.TARGET_UNAVAILABLE
    return _result("rejected", state=errors.BLOCKED, code=code, domain="browser",
                   spoken_text=spoken.speak("browser_unavailable" if code == errors.BROWSER_UNAVAILABLE
                                            else "failed", language))


# --- confirmation / cancel ----------------------------------------------------------


def _handle_confirmation(phrase, language, wake, settings, broker):
    route = broker.route_confirmation(phrase)
    if not route["ok"]:
        situation = ("nothing_pending" if route["code"] == errors.CONFIRMATION_REQUIRED
                     else "confirmation_mismatch")
        return _result("rejected", state=errors.BLOCKED, code=route["code"],
                       domain=broker.active_domain(),
                       spoken_text=spoken.speak(situation, language))
    domain, action_id = route["domain"], route["action_id"]
    if domain == "text":
        res = get_text_service().confirm(action_id=action_id, phrase=phrase)
        return _finish_text_confirm(res, action_id, language, broker)
    if domain == "browser":
        res = get_browser_service().confirm_form(action_id, phrase)
        return _finish_browser_confirm(res, action_id, language, broker)
    handler = CONFIRM_HANDLERS.get(domain)   # WhatsApp / email / memory register theirs
    if handler is not None:
        # A handler may opt into the wake flag (memory refuses a wake-originated
        # confirmation); older handlers keep the 4-arg signature unchanged.
        import inspect

        try:
            accepts_wake = "wake" in inspect.signature(handler).parameters
        except (TypeError, ValueError):
            accepts_wake = False
        if accepts_wake:
            return handler(action_id, phrase, language, broker, wake=wake)
        return handler(action_id, phrase, language, broker)
    return _result("rejected", state=errors.FAILED, code=errors.UNSUPPORTED_INTENT,
                   domain=domain, spoken_text=spoken.speak("unsupported", language))


def _finish_text_confirm(res, action_id, language, broker):
    broker.clear(action_id)
    status = res.get("status")
    if status == "simulated":
        return _result("simulated", state=errors.COMPLETED, domain="text", action_id=action_id,
                       spoken_text=spoken.speak("inserted_simulated", language))
    if status == "executed":
        return _result("executed", state=errors.COMPLETED, domain="text", action_id=action_id,
                       spoken_text=spoken.speak("inserted", language))
    return _rejected_confirm(res, "text", language)


def _finish_browser_confirm(res, action_id, language, broker):
    broker.clear(action_id)
    if res.get("status") == "filled":
        return _result("filled", state=errors.COMPLETED, domain="browser", action_id=action_id,
                       spoken_text=spoken.speak("form_filled", language))
    return _rejected_confirm(res, "browser", language)


def _rejected_confirm(res, domain, language):
    reason = (res.get("reason") or "").lower()
    if "changed" in reason:
        return _result("rejected", state=errors.BLOCKED, code=errors.TARGET_CHANGED,
                       domain=domain, spoken_text=spoken.speak("target_changed", language))
    if "expired" in reason:
        return _result("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED,
                       domain=domain, spoken_text=spoken.speak("action_expired", language))
    if res.get("status") == "needs_confirmation":
        return _result("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                       domain=domain, spoken_text=spoken.speak("confirmation_required", language))
    return _result("error", state=errors.FAILED, code=errors.EXECUTION_FAILED,
                   domain=domain, spoken_text=spoken.speak("failed", language))


def _handle_cancel(language, broker):
    had = broker.cancel_active()
    situation = "cancelled" if had else "nothing_pending"
    return _result("cancelled" if had else "rejected",
                   state=errors.COMPLETED if had else errors.BLOCKED,
                   spoken_text=spoken.speak(situation, language))


# --- public entry -------------------------------------------------------------------


def is_service_intent(intent: str) -> bool:
    return intent in SERVICE_INTENTS or intent in INTENT_DISPATCHERS


def dispatch(command, *, language="es", wake=False, settings=None) -> dict:
    """Dispatch a validated ServiceCommand, then publish ONE safe observability event
    (Phase 5D). The event carries only the normalized status/domain/action id and the
    result's already-safe ``data`` (redacted/allowlisted downstream) — never content."""
    settings = settings or get_settings()
    if _AUTHORIZATION_GATE is not None:      # central default-deny gate (Phase 6B)
        try:
            denied = _AUTHORIZATION_GATE(command, settings)
        except Exception:
            denied = None                    # a gate failure must never open access nor crash
        if denied is not None:
            _emit_activity(command, denied)
            return denied
    result = _dispatch_inner(command, language=language, wake=wake, settings=settings)
    _emit_activity(command, result)
    return result


def _emit_activity(command, result) -> None:
    try:
        from app.core import eventbus

        eventbus.emit(
            domain=result.get("domain") or "command",
            event_type=getattr(command, "intent", "") or "command",
            status=result.get("state") or result.get("status"),
            error_code=result.get("error_code"),
            related_id=result.get("action_id"),
            metadata=result.get("data") or {})
    except Exception:
        pass


def _dispatch_inner(command, *, language="es", wake=False, settings=None) -> dict:
    """Dispatch a validated ServiceCommand. ``command`` has .intent/.arguments/
    .is_confirmation/.confirmation_phrase (see app/core/nl.ServiceCommand)."""
    settings = settings or get_settings()
    broker = get_pending_broker()
    intent = command.intent

    if command.is_confirmation or intent == "confirm":
        active = broker.active()
        pre_action_id = active["action_id"] if active else None
        result = _handle_confirmation(command.confirmation_phrase, language, wake, settings, broker)
        _notify_effect_listeners(result, pre_action_id)   # let a waiting task advance (5B)
        return result
    if intent == "cancel":
        return _handle_cancel(language, broker)
    if intent in TEXT_INTENTS:
        return _dispatch_text(intent, command.arguments, language, settings, broker)
    if intent in BROWSER_INTENTS:
        return _dispatch_browser(intent, command.arguments, language, settings, broker)
    handler = INTENT_DISPATCHERS.get(intent)   # WhatsApp / email integrations
    if handler is not None:
        return handler(command.arguments, language, settings, broker)
    return _result("rejected", state=errors.FAILED, code=errors.UNSUPPORTED_INTENT,
                   spoken_text=spoken.speak("unsupported", language))
