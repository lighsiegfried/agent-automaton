"""EmailService: safe reading + drafting + two-stage confirmed sending (Phase 4D).

Email content is UNTRUSTED: it is summarized but never executed, and in-email
instructions are ignored (flagged as PROMPT_INJECTION_DETECTED, never acted on).
Recipients come only from explicit user input or the verified thread sender —
never inferred from body text. Reply-all and BCC are blocked; attachments must be
zero. Placing a draft and sending are separate confirmations, and a send must name
every recipient. Simulated unless ENABLE_REAL_EMAIL_SEND. The provider adapter is
injectable, so the safety machine is testable without a browser.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.core import conversation, errors, nl, pending
from app.core.logger import get_logger
from app.integrations.email_web import inspector, safety
from app.integrations.email_web.adapters.base import create_adapter
from app.integrations.email_web.models import EmailDraft, EmailSendAction
from app.schemas.commands import Intent

log = get_logger(__name__)

_SPOKEN = {
    "login_required": ("Email needs a manual login first.", "El correo necesita que inicies sesión primero."),
    "provider_unsupported": ("That email provider isn't supported.", "Ese proveedor de correo no es compatible."),
    "found": ("I found {count} message(s).", "Encontré {count} mensaje(s)."),
    "opened": ("Opened the thread.", "Abrí la conversación."),
    "summary": ("Here's a summary of the thread.", "Aquí tienes un resumen de la conversación."),
    "draft_ready": ("Draft ready for {recipients} ({count} chars). Say 'colocar borrador' to place it.",
                    "Borrador listo para {recipients} ({count} caracteres). Di 'colocar borrador' para colocarlo."),
    "placed": ("Placed the draft for {recipients}. Say 'confirmar envío a {recipients}' to send.",
               "Coloqué el borrador para {recipients}. Di 'confirmar envío a {recipients}' para enviar."),
    "sent": ("Sent the email to {recipients}.", "Envié el correo a {recipients}."),
    "sent_sim": ("Simulated — real email sending is off, nothing was sent.",
                 "Simulado — el envío real de correo está desactivado, no se envió nada."),
    "thread_changed": ("The thread changed, so I cancelled it.", "La conversación cambió, así que lo cancelé."),
    "recipient_changed": ("The recipients changed, so I cancelled it.", "Los destinatarios cambiaron, así que lo cancelé."),
    "draft_changed": ("The message changed, so I cancelled it.", "El mensaje cambió, así que lo cancelé."),
    "recipient_mismatch": ("The named recipients don't match the pending email.",
                           "Los destinatarios nombrados no coinciden con el correo pendiente."),
    "attachment_blocked": ("I won't send an email with attachments in this phase.",
                           "No enviaré un correo con adjuntos en esta fase."),
    "send_blocked": ("I won't send that.", "No enviaré eso."),
    "expired": ("That action expired.", "Esa acción expiró."),
    "blocked": ("I won't do that.", "No haré eso."),
    "rate_limited": ("Please wait before sending another email.", "Espera antes de enviar otro correo."),
    "no_thread": ("Open a thread first.", "Abre una conversación primero."),
    "cancelled": ("Cancelled.", "Cancelado."),
    "closed": ("Closed email.", "Cerré el correo."),
    "disabled": ("Email automation is off.", "La automatización de correo está desactivada."),
    "failed": ("That didn't work.", "Eso no funcionó."),
}


def _say(situation, language, **ctx):
    en, es = _SPOKEN.get(situation, _SPOKEN["failed"])
    template = en if str(language).lower().startswith("en") else es
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


def _r(status, *, state, spoken, code=None, action_id=None, data=None):
    return {"status": status, "error_code": code, "domain": "email", "action_id": action_id,
            "state": state, "spoken": spoken, "data": data or {}}


def _recipients(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value or "").replace(";", ",").split(",") if p.strip()]


class EmailService:
    def __init__(self, *, adapter_factory=create_adapter, clock=time.monotonic,
                 now_utc=None, id_factory=None):
        self._adapter = None
        self._provider = None
        self._thread = None          # {subject, sender, signature}
        self._draft: EmailDraft | None = None
        self._send: EmailSendAction | None = None
        self._last_send_monotonic: float | None = None
        self._factory = adapter_factory
        self._clock = clock
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._id = id_factory or (lambda: uuid.uuid4().hex)

    # -- provider / adapter --------------------------------------------------------

    def _resolve_provider(self, args, settings):
        provider = (args.get("provider") or self._provider
                    or (settings.email_allowed_providers_list or [""])[0]).lower()
        if provider not in settings.email_allowed_providers_list:
            return None, provider
        if self._adapter is None or self._provider != provider:
            self._adapter = self._factory(provider, settings)
            self._provider = provider
        return self._adapter, provider

    def _clear(self):
        self._draft, self._send = None, None

    def _guard(self, args, settings, language):
        if not settings.enable_email_web_automation:
            return _r("rejected", state=errors.BLOCKED, code=errors.FEATURE_DISABLED,
                      spoken=_say("disabled", language)), None
        adapter, provider = self._resolve_provider(args, settings)
        if adapter is None:
            return _r("rejected", state=errors.BLOCKED, code=errors.PROVIDER_UNSUPPORTED,
                      spoken=_say("provider_unsupported", language)), None
        if not adapter.logged_in():
            return _r("rejected", state=errors.BLOCKED, code=errors.LOGIN_REQUIRED,
                      spoken=_say("login_required", language)), None
        return None, adapter

    # -- status / read -------------------------------------------------------------

    def status(self, args, language, settings, broker):
        if not settings.enable_email_web_automation:
            return _r("rejected", state=errors.BLOCKED, code=errors.FEATURE_DISABLED,
                      spoken=_say("disabled", language))
        adapter, provider = self._resolve_provider(args, settings)
        if adapter is None:
            return _r("rejected", state=errors.BLOCKED, code=errors.PROVIDER_UNSUPPORTED,
                      spoken=_say("provider_unsupported", language))
        logged_in = bool(adapter.logged_in())
        data = {"provider": provider, "logged_in": logged_in,
                "real_send": settings.enable_real_email_send}
        if not logged_in:
            return _r("rejected", state=errors.BLOCKED, code=errors.LOGIN_REQUIRED,
                      spoken=_say("login_required", language), data=data)
        return _r("ok", state=errors.COMPLETED, spoken="", data=data)

    def search(self, args, language, settings, broker):
        guard, adapter = self._guard(args, settings, language)
        if guard:
            return guard
        results = adapter.search(args.get("query", ""))[:10]   # never enumerate the whole mailbox
        summaries = [inspector.safe_message_summary(m, settings.email_max_body_chars) for m in results]
        return _r("ok", state=errors.COMPLETED, spoken=_say("found", language, count=len(summaries)),
                  data={"provider": self._provider, "results": summaries, "count": len(summaries)})

    def open_thread(self, args, language, settings, broker):
        guard, adapter = self._guard(args, settings, language)
        if guard:
            return guard
        thread = adapter.open_thread(args.get("thread_id") or args.get("query") or "")
        if not thread:
            return _r("rejected", state=errors.BLOCKED, code=errors.THREAD_NOT_FOUND,
                      spoken=_say("no_thread", language))
        self._thread = {"subject": thread.get("subject", ""), "sender": thread.get("sender", ""),
                        "signature": inspector.thread_signature(thread.get("subject", ""),
                                                                thread.get("sender", ""))}
        summary = inspector.safe_thread_summary(thread, settings.email_max_body_chars)
        return _r("ok", state=errors.COMPLETED, spoken=_say("opened", language), data=summary)

    def summarize_thread(self, args, language, settings, broker):
        result = self.open_thread(args, language, settings, broker)
        if result["status"] != "ok":
            return result
        # Untrusted-content check: flag in-email instructions, but NEVER act on them.
        bodies = " ".join(m.get("body", "") for m in result["data"].get("messages", []))
        injected, reasons = safety.detect_prompt_injection(bodies)
        result["data"]["prompt_injection_detected"] = injected
        result["data"]["prompt_injection_reasons"] = reasons
        result["spoken"] = _say("summary", language)
        return result

    # -- drafting (safe) -----------------------------------------------------------

    def _make_draft(self, mode, recipients, subject, body, cc, thread_sig, language, settings):
        expires = settings.email_action_expires_seconds
        return EmailDraft(
            action_id=self._id(), provider=self._provider, mode=mode, recipients=recipients,
            cc=cc, subject=subject, body=body, language=language, thread_signature=thread_sig,
            expires_at_monotonic=self._clock() + expires,
            expires_at_iso=(self._now_utc() + timedelta(seconds=expires)).isoformat(timespec="seconds"),
        )

    def _register_draft(self, broker, language):
        broker.register(domain="email_place", action_id=self._draft.action_id,
                        target=", ".join(self._draft.recipients), expires_at=self._draft.expires_at_iso,
                        cancel=self._clear)
        return _r("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                  action_id=self._draft.action_id,
                  spoken=_say("draft_ready", language, recipients=", ".join(self._draft.recipients),
                              count=len(self._draft.body)),
                  data=self._draft.preview())

    def draft_new(self, args, language, settings, broker):
        guard, adapter = self._guard(args, settings, language)
        if guard:
            return guard
        recipients = _recipients(args.get("to"))     # ONLY from explicit input
        if not recipients:
            return _r("rejected", state=errors.BLOCKED, code=errors.RECIPIENT_AMBIGUOUS,
                      spoken=_say("blocked", language))
        body = args.get("body") or args.get("instruction") or ""
        ok, reason = safety.check_body(body, settings.email_max_body_chars)
        if not ok:
            return _r("rejected", state=errors.BLOCKED, code=errors.SEND_BLOCKED,
                      spoken=_say("blocked", language))
        self._draft = self._make_draft("new", recipients, args.get("subject", ""), body,
                                       _recipients(args.get("cc")), "", language, settings)
        return self._register_draft(broker, language)

    def draft_reply(self, args, language, settings, broker):
        guard, adapter = self._guard(args, settings, language)
        if guard:
            return guard
        if not self._thread:
            return _r("rejected", state=errors.BLOCKED, code=errors.THREAD_NOT_FOUND,
                      spoken=_say("no_thread", language))
        # Reply recipient = the VERIFIED thread sender only (no reply-all here).
        recipients = [self._thread["sender"]] if self._thread["sender"] else []
        if not recipients:
            return _r("rejected", state=errors.BLOCKED, code=errors.RECIPIENT_AMBIGUOUS,
                      spoken=_say("blocked", language))
        body = args.get("body") or args.get("instruction") or ""
        ok, reason = safety.check_body(body, settings.email_max_body_chars)
        if not ok:
            return _r("rejected", state=errors.BLOCKED, code=errors.SEND_BLOCKED,
                      spoken=_say("blocked", language))
        subject = self._thread["subject"]
        subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        self._draft = self._make_draft("reply", recipients, subject, body, [],
                                       self._thread["signature"], language, settings)
        return self._register_draft(broker, language)

    # -- place ---------------------------------------------------------------------

    def confirm_place(self, action_id, phrase, language, broker):
        settings = get_settings()
        draft = self._draft
        if draft is None or draft.action_id != action_id:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                      spoken=_say("failed", language))
        if self._clock() >= draft.expires_at_monotonic:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED,
                      spoken=_say("expired", language))
        adapter = self._adapter
        if adapter is None or not adapter.logged_in():
            return _r("rejected", state=errors.BLOCKED, code=errors.LOGIN_REQUIRED,
                      spoken=_say("login_required", language))
        if draft.mode == "reply":
            current = adapter.current_thread()
            sig = inspector.thread_signature(current.get("subject", ""), current.get("sender", ""))
            if sig != draft.thread_signature:
                self._clear(); broker.clear(action_id)
                return _r("rejected", state=errors.BLOCKED, code=errors.THREAD_CHANGED,
                          spoken=_say("thread_changed", language))
            adapter.start_reply()
        else:
            adapter.start_new()
        adapter.set_compose(draft.recipients, draft.subject, draft.body, cc=draft.cc)
        compose = adapter.compose_state()
        if compose.get("attachment_count", 0) and not settings.email_allow_attachments:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.ATTACHMENT_BLOCKED,
                      spoken=_say("attachment_blocked", language))
        draft.placed = True
        broker.clear(action_id)   # retire PLACE without firing its cancel
        expires = settings.email_action_expires_seconds
        self._send = EmailSendAction(
            action_id=self._id(), provider=self._provider, recipients=list(draft.recipients),
            cc=list(draft.cc), subject=draft.subject, thread_signature=draft.thread_signature,
            body_hash=inspector.body_hash(draft.body),
            attachment_count=compose.get("attachment_count", 0),
            expires_at_monotonic=self._clock() + expires,
            expires_at_iso=(self._now_utc() + timedelta(seconds=expires)).isoformat(timespec="seconds"),
        )
        broker.register(domain="email_send", action_id=self._send.action_id,
                        target=", ".join(draft.recipients), expires_at=self._send.expires_at_iso,
                        cancel=self._clear)
        log.info("email draft placed (no send)")
        return _r("placed", state=errors.AWAITING_CONFIRMATION, action_id=self._send.action_id,
                  spoken=_say("placed", language, recipients=", ".join(draft.recipients)),
                  data=self._send.summary(draft.preview()["body_preview"]))

    def prepare_send(self, args, language, settings, broker):
        if self._send is None or self._draft is None or not self._draft.placed:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                      spoken=_say("no_thread", language))
        return _r("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                  action_id=self._send.action_id,
                  spoken=_say("placed", language, recipients=", ".join(self._send.recipients)),
                  data=self._send.summary(self._draft.preview()["body_preview"]))

    # -- send ----------------------------------------------------------------------

    def confirm_send(self, action_id, phrase, language, broker):
        settings = get_settings()
        send, draft = self._send, self._draft
        if send is None or send.action_id != action_id or send.used:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                      spoken=_say("failed", language))
        # Every recipient must be named in the confirmation phrase.
        if not safety.recipients_match(phrase, send.recipients):
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_MISMATCH,
                      spoken=_say("recipient_mismatch", language))
        if self._clock() >= send.expires_at_monotonic:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED,
                      spoken=_say("expired", language))
        if self._last_send_monotonic is not None and (
                self._clock() - self._last_send_monotonic) < settings.email_send_cooldown_seconds:
            return _r("rejected", state=errors.BLOCKED, code=errors.SEND_BLOCKED,
                      spoken=_say("rate_limited", language))
        adapter = self._adapter
        if adapter is None or not adapter.logged_in():
            return _r("rejected", state=errors.BLOCKED, code=errors.LOGIN_REQUIRED,
                      spoken=_say("login_required", language))
        # Immediately-before-send revalidation of the live compose state.
        compose = adapter.compose_state()
        compose["mode"] = draft.mode if draft else compose.get("mode")
        ok, code, _reason = safety.check_send_compose(compose, settings)
        if not ok:
            self._clear(); broker.clear(action_id)
            spoken = "attachment_blocked" if code == errors.ATTACHMENT_BLOCKED else "send_blocked"
            return _r("rejected", state=errors.BLOCKED, code=code, spoken=_say(spoken, language))
        if sorted(_recipients(compose.get("to"))) != sorted(send.recipients):
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.RECIPIENT_CHANGED,
                      spoken=_say("recipient_changed", language))
        if inspector.body_hash(compose.get("body", "")) != send.body_hash:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.DRAFT_CHANGED,
                      spoken=_say("draft_changed", language))

        recipients = ", ".join(send.recipients)
        if not settings.enable_real_email_send:
            send.used = True; self._last_send_monotonic = self._clock()
            self._clear(); broker.clear(action_id)
            return _r("simulated", state=errors.COMPLETED, spoken=_say("sent_sim", language),
                      data={"recipients": send.recipients})
        adapter.click_send()   # exactly once; never a keyboard shortcut
        send.used = True; self._last_send_monotonic = self._clock()
        self._clear(); broker.clear(action_id)
        log.info("email sent (recipients hidden)")
        return _r("sent", state=errors.COMPLETED, spoken=_say("sent", language, recipients=recipients),
                  data={"recipients": send.recipients})

    def cancel(self, args, language, settings, broker):
        self._clear(); broker.cancel_active()
        return _r("cancelled", state=errors.COMPLETED, spoken=_say("cancelled", language))

    def close(self, args, language, settings, broker):
        if self._adapter is not None:
            try:
                self._adapter.close()
            except Exception:
                pass
            self._adapter = None
        self._provider, self._thread = None, None
        self._clear()
        return _r("ok", state=errors.COMPLETED, spoken=_say("closed", language))


_service: EmailService | None = None


def get_email_service() -> EmailService:
    global _service
    if _service is None:
        _service = EmailService()
    return _service


def _svc():
    return get_email_service()


def _register() -> None:
    pending.register_domain("email_place", phrases=safety.PLACE_PHRASES,
                            primary="colocar borrador / place email draft")
    pending.register_domain("email_send", matcher=safety.is_send_phrase,
                            primary="confirmar envío a <destinatarios>")

    for intent, keys, method in [
        (Intent.EMAIL_STATUS.value, {"provider"}, "status"),
        (Intent.EMAIL_SEARCH.value, {"query", "provider"}, "search"),
        (Intent.EMAIL_OPEN_THREAD.value, {"thread_id", "query", "provider"}, "open_thread"),
        (Intent.EMAIL_SUMMARIZE_THREAD.value, {"thread_id", "query", "provider"}, "summarize_thread"),
        (Intent.EMAIL_DRAFT_NEW.value, {"to", "cc", "subject", "body", "instruction", "provider"}, "draft_new"),
        (Intent.EMAIL_DRAFT_REPLY.value, {"body", "instruction"}, "draft_reply"),
        (Intent.EMAIL_PREPARE_SEND.value, set(), "prepare_send"),
        (Intent.EMAIL_CANCEL.value, set(), "cancel"),
        (Intent.EMAIL_CLOSE.value, set(), "close"),
    ]:
        conversation.register_service_intent(
            intent, keys, (lambda m: lambda a, l, s, b: getattr(_svc(), m)(a, l, s, b))(method))

    conversation.register_confirm_handler(
        "email_place", lambda aid, phrase, lang, b: _svc().confirm_place(aid, phrase, lang, b))
    conversation.register_confirm_handler(
        "email_send", lambda aid, phrase, lang, b: _svc().confirm_send(aid, phrase, lang, b))

    import re
    nl.register_patterns([
        (re.compile(r"\bbusca(?:r)?\s+(?:el\s+[uú]ltimo\s+)?correos?\s+de\s+(.+)", re.I),
         Intent.EMAIL_SEARCH.value, lambda m: {"query": m.group(1).strip()}),
        (re.compile(r"\bsearch\s+(?:for\s+)?emails?\s+from\s+(.+)", re.I),
         Intent.EMAIL_SEARCH.value, lambda m: {"query": m.group(1).strip()}),
        (re.compile(r"\bresume\s+(?:esta|este)\s+(?:conversaci[oó]n|hilo|correo)\b", re.I),
         Intent.EMAIL_SUMMARIZE_THREAD.value, lambda m: {}),
        (re.compile(r"\bsummari[sz]e\s+(?:this\s+)?(?:thread|conversation|email)\b", re.I),
         Intent.EMAIL_SUMMARIZE_THREAD.value, lambda m: {}),
        (re.compile(r"\bredacta\s+una\s+respuesta(?:\s+que\s+diga|:)?\s*(.*)", re.I),
         Intent.EMAIL_DRAFT_REPLY.value, lambda m: {"body": m.group(1).strip()}),
        (re.compile(r"\bdraft\s+a\s+reply(?:\s+saying|:)?\s*(.*)", re.I),
         Intent.EMAIL_DRAFT_REPLY.value, lambda m: {"body": m.group(1).strip()}),
        (re.compile(r"\b(?:estado\s+del\s+correo|email\s+status)\b", re.I),
         Intent.EMAIL_STATUS.value, lambda m: {}),
        (re.compile(r"\b(?:cierra|close)\s+(?:el\s+)?(?:correo|email)\b", re.I),
         Intent.EMAIL_CLOSE.value, lambda m: {}),
    ])


_register()
