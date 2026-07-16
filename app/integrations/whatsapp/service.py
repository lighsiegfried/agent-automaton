"""WhatsAppService: safe drafting + two-stage confirmed sending (Phase 4C).

Flow (all sensitive steps go through the unified pending broker, one at a time):
    find_contact -> open_chat          (safe; verifies the visible chat)
    draft_message                       (safe; local only) -> registers PLACE pending
    "colocar borrador"                  -> confirm_place: inserts once, no Enter
                                         -> registers SEND pending
    "enviar mensaje a <recipient>"      -> confirm_send: re-verifies recipient +
                                            chat signature + composer hash, then
                                            clicks Send exactly once (never Enter)

Simulated unless ENABLE_REAL_WHATSAPP_SEND. The session adapter is injectable, so
the whole safety machine is testable without a browser. Cookies/tokens/QR are
never read or logged.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.core import conversation, errors, nl, pending
from app.core.logger import get_logger
from app.core.pending import get_pending_broker
from app.integrations.whatsapp import inspector, safety
from app.integrations.whatsapp.models import SendAction, WhatsAppDraft
from app.schemas.commands import Intent

log = get_logger(__name__)

_SPOKEN = {
    "login_required": ("WhatsApp needs a manual QR login first.",
                       "WhatsApp necesita que inicies sesión con el QR primero."),
    "not_found": ("I couldn't find that contact.", "No encontré ese contacto."),
    "ambiguous": ("Several contacts match — please be more specific.",
                  "Varios contactos coinciden — sé más específico."),
    "chat_open": ("Opened the chat with {recipient}.", "Abrí el chat con {recipient}."),
    "draft_ready": ("Draft ready for {recipient} ({count} chars). Say 'colocar borrador' to place it.",
                    "Borrador listo para {recipient} ({count} caracteres). Di 'colocar borrador' para colocarlo."),
    "placed": ("Placed the draft for {recipient}. Say 'confirmar envío a {recipient}' to send.",
               "Coloqué el borrador para {recipient}. Di 'confirmar envío a {recipient}' para enviar."),
    "sent": ("Sent your message to {recipient}.", "Envié tu mensaje a {recipient}."),
    "sent_sim": ("Simulated — real WhatsApp sending is off, nothing was sent.",
                 "Simulado — el envío real de WhatsApp está desactivado, no se envió nada."),
    "chat_changed": ("The chat changed, so I cancelled it.", "El chat cambió, así que lo cancelé."),
    "draft_changed": ("The message changed, so I cancelled it.", "El mensaje cambió, así que lo cancelé."),
    "recipient_mismatch": ("The named recipient doesn't match the pending message.",
                           "El destinatario nombrado no coincide con el mensaje pendiente."),
    "expired": ("That action expired.", "Esa acción expiró."),
    "blocked": ("I won't do that.", "No haré eso."),
    "rate_limited": ("Please wait a moment before sending again.",
                     "Espera un momento antes de enviar de nuevo."),
    "no_chat": ("Open a chat first.", "Abre un chat primero."),
    "cancelled": ("Cancelled.", "Cancelado."),
    "closed": ("Closed WhatsApp.", "Cerré WhatsApp."),
    "disabled": ("WhatsApp automation is off.", "La automatización de WhatsApp está desactivada."),
    "failed": ("That didn't work.", "Eso no funcionó."),
}


def _say(situation: str, language: str, **ctx) -> str:
    en, es = _SPOKEN.get(situation, _SPOKEN["failed"])
    template = en if str(language).lower().startswith("en") else es
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


def _r(status, *, state, spoken, code=None, action_id=None, data=None):
    return {"status": status, "error_code": code, "domain": "whatsapp",
            "action_id": action_id, "state": state, "spoken": spoken, "data": data or {}}


class WhatsAppService:
    def __init__(self, *, session_provider=None, clock=time.monotonic, now_utc=None,
                 id_factory=None):
        self._session = None
        self._draft: WhatsAppDraft | None = None
        self._send: SendAction | None = None
        self._last_send_monotonic: float | None = None
        self._provider = session_provider or _default_session_provider
        self._clock = clock
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._id = id_factory or (lambda: uuid.uuid4().hex)

    # -- session -------------------------------------------------------------------

    def _session_or_none(self, settings):
        if self._session is None:
            self._session = self._provider(settings)
        return self._session

    def _clear(self):
        self._draft, self._send = None, None

    # -- status / contacts / chat --------------------------------------------------

    def status(self, args, language, settings, broker) -> dict:
        if not settings.enable_whatsapp_automation:
            return _r("rejected", state=errors.BLOCKED, code=errors.FEATURE_DISABLED,
                      spoken=_say("disabled", language))
        session = self._session_or_none(settings)
        logged_in = bool(session and session.logged_in())
        data = {"logged_in": logged_in,
                "active_chat": session.chat_title() if logged_in else None,
                "real_send": settings.enable_real_whatsapp_send}
        if not logged_in:
            return _r("rejected", state=errors.BLOCKED, code=errors.LOGIN_REQUIRED,
                      spoken=_say("login_required", language), data=data)
        return _r("ok", state=errors.COMPLETED, spoken="", data=data)

    def _require_login(self, settings, language):
        if not settings.enable_whatsapp_automation:
            return _r("rejected", state=errors.BLOCKED, code=errors.FEATURE_DISABLED,
                      spoken=_say("disabled", language))
        session = self._session_or_none(settings)
        if session is None or not session.logged_in():
            return _r("rejected", state=errors.BLOCKED, code=errors.LOGIN_REQUIRED,
                      spoken=_say("login_required", language))
        return None

    def find_contact(self, args, language, settings, broker) -> dict:
        guard = self._require_login(settings, language)
        if guard:
            return guard
        query = (args.get("name") or "").strip()
        candidates = self._session.search_contacts(query)
        match = inspector.match_contacts(candidates, query)
        if match["status"] == "not_found":
            return _r("rejected", state=errors.BLOCKED, code=errors.CONTACT_NOT_FOUND,
                      spoken=_say("not_found", language))
        if match["status"] == "ambiguous":
            return _r("rejected", state=errors.BLOCKED, code=errors.CONTACT_AMBIGUOUS,
                      spoken=_say("ambiguous", language),
                      data={"candidates": match["candidates"]})
        return _r("ok", state=errors.COMPLETED, spoken="", data={"contact": match["contact"]})

    def open_chat(self, args, language, settings, broker) -> dict:
        guard = self._require_login(settings, language)
        if guard:
            return guard
        name = (args.get("name") or "").strip()
        candidates = self._session.search_contacts(name)
        match = inspector.match_contacts(candidates, name)
        if match["status"] != "ok":
            code = errors.CONTACT_AMBIGUOUS if match["status"] == "ambiguous" else errors.CONTACT_NOT_FOUND
            return _r("rejected", state=errors.BLOCKED, code=code,
                      spoken=_say(match["status"], language),
                      data={"candidates": match.get("candidates", [])})
        if not self._session.open_chat(match["contact"]):
            return _r("rejected", state=errors.BLOCKED, code=errors.CONTACT_NOT_FOUND,
                      spoken=_say("not_found", language))
        title = self._session.chat_title()
        return _r("ok", state=errors.COMPLETED,
                  spoken=_say("chat_open", language, recipient=title),
                  data={"recipient": title, "chat_signature": inspector.chat_signature(title)})

    # -- drafting (safe) + place ---------------------------------------------------

    def draft_message(self, args, language, settings, broker) -> dict:
        guard = self._require_login(settings, language)
        if guard:
            return guard
        recipient = self._session.chat_title()
        if not recipient:
            return _r("rejected", state=errors.BLOCKED, code=errors.CHAT_CHANGED,
                      spoken=_say("no_chat", language))
        text = args.get("text") or args.get("instruction") or ""
        ok, reason = safety.check_message(text, settings.whatsapp_max_message_chars)
        if not ok:
            code = errors.SEND_BLOCKED if "secret" in reason or "character" in reason else errors.EXECUTION_FAILED
            return _r("rejected", state=errors.BLOCKED, code=code, spoken=_say("blocked", language))
        expires = settings.whatsapp_action_expires_seconds
        self._draft = WhatsAppDraft(
            action_id=self._id(), recipient=recipient, text=text, language=language,
            chat_signature=inspector.chat_signature(recipient),
            expires_at_monotonic=self._clock() + expires,
            expires_at_iso=(self._now_utc() + timedelta(seconds=expires)).isoformat(timespec="seconds"),
        )
        broker.register(domain="whatsapp_place", action_id=self._draft.action_id,
                        target=recipient, expires_at=self._draft.expires_at_iso,
                        cancel=self._clear)
        return _r("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                  action_id=self._draft.action_id,
                  spoken=_say("draft_ready", language, recipient=recipient, count=len(text)),
                  data=self._draft.preview())

    def confirm_place(self, action_id, phrase, language, broker) -> dict:
        settings = get_settings()
        draft = self._draft
        if draft is None or draft.action_id != action_id:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                      spoken=_say("failed", language))
        if self._clock() >= draft.expires_at_monotonic:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED,
                      spoken=_say("expired", language))
        session = self._session_or_none(settings)
        if not session or not session.logged_in():
            return _r("rejected", state=errors.BLOCKED, code=errors.LOGIN_REQUIRED,
                      spoken=_say("login_required", language))
        if inspector.chat_signature(session.chat_title()) != draft.chat_signature:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.CHAT_CHANGED,
                      spoken=_say("chat_changed", language))
        # Insert the draft once — never press Enter or Send.
        session.set_composer(draft.text)
        draft.placed = True
        # Retire the PLACE pending without firing its cancel (which would wipe the
        # draft we just placed) before arming the separate SEND pending.
        broker.clear(action_id)
        expires = settings.whatsapp_action_expires_seconds
        self._send = SendAction(
            action_id=self._id(), recipient=draft.recipient, character_count=len(draft.text),
            chat_signature=draft.chat_signature, composer_hash=inspector.composer_hash(draft.text),
            expires_at_monotonic=self._clock() + expires,
            expires_at_iso=(self._now_utc() + timedelta(seconds=expires)).isoformat(timespec="seconds"),
        )
        broker.register(domain="whatsapp_send", action_id=self._send.action_id,
                        target=draft.recipient, expires_at=self._send.expires_at_iso,
                        cancel=self._clear)
        log.info("whatsapp draft placed for a recipient (no send)")
        return _r("placed", state=errors.AWAITING_CONFIRMATION, action_id=self._send.action_id,
                  spoken=_say("placed", language, recipient=draft.recipient),
                  data=self._send.summary(draft.preview()["body_preview"]))

    def prepare_send(self, args, language, settings, broker) -> dict:
        """Explicit (re)arm of the send pending after a draft is placed."""
        if self._send is None or self._draft is None or not self._draft.placed:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                      spoken=_say("no_chat", language))
        return _r("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                  action_id=self._send.action_id,
                  spoken=_say("placed", language, recipient=self._send.recipient),
                  data=self._send.summary(self._draft.preview()["body_preview"]))

    # -- send ----------------------------------------------------------------------

    def confirm_send(self, action_id, phrase, language, broker) -> dict:
        settings = get_settings()
        send, draft = self._send, self._draft
        if send is None or send.action_id != action_id or send.used:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                      spoken=_say("failed", language))
        # The confirmation phrase MUST name the correct recipient.
        if settings.whatsapp_require_recipient_in_confirmation and not safety.recipient_matches(
                phrase, send.recipient):
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_MISMATCH,
                      spoken=_say("recipient_mismatch", language))
        if self._clock() >= send.expires_at_monotonic:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED,
                      spoken=_say("expired", language))
        # Rate-limit consecutive sends.
        if self._last_send_monotonic is not None and (
                self._clock() - self._last_send_monotonic) < settings.whatsapp_send_cooldown_seconds:
            return _r("rejected", state=errors.BLOCKED, code=errors.SEND_BLOCKED,
                      spoken=_say("rate_limited", language))
        session = self._session_or_none(settings)
        if not session or not session.logged_in():
            return _r("rejected", state=errors.BLOCKED, code=errors.LOGIN_REQUIRED,
                      spoken=_say("login_required", language))
        # Immediately-before-send revalidation.
        if inspector.chat_signature(session.chat_title()) != send.chat_signature:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.CHAT_CHANGED,
                      spoken=_say("chat_changed", language))
        if inspector.composer_hash(session.composer_text()) != send.composer_hash:
            self._clear(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.DRAFT_CHANGED,
                      spoken=_say("draft_changed", language))

        recipient = send.recipient
        if not settings.enable_real_whatsapp_send:
            send.used = True
            self._last_send_monotonic = self._clock()
            self._clear(); broker.clear(action_id)
            return _r("simulated", state=errors.COMPLETED,
                      spoken=_say("sent_sim", language), data={"recipient": recipient})
        session.click_send()   # exactly once; never Enter
        send.used = True
        self._last_send_monotonic = self._clock()
        self._clear(); broker.clear(action_id)
        log.info("whatsapp message sent to a recipient")
        return _r("sent", state=errors.COMPLETED, spoken=_say("sent", language, recipient=recipient),
                  data={"recipient": recipient})

    # -- cancel / close ------------------------------------------------------------

    def cancel(self, args, language, settings, broker) -> dict:
        self._clear()
        broker.cancel_active()
        return _r("cancelled", state=errors.COMPLETED, spoken=_say("cancelled", language))

    def close(self, args, language, settings, broker) -> dict:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        self._clear()
        return _r("ok", state=errors.COMPLETED, spoken=_say("closed", language))


def _default_session_provider(settings):  # pragma: no cover - requires a browser
    from app.integrations.whatsapp.session import WhatsAppSession

    return WhatsAppSession(settings)


_service: WhatsAppService | None = None


def get_whatsapp_service() -> WhatsAppService:
    global _service
    if _service is None:
        _service = WhatsAppService()
    return _service


# --- registration with the core dispatcher -----------------------------------------


def _svc():
    return get_whatsapp_service()


def _register() -> None:
    pending.register_domain("whatsapp_place", phrases=safety.PLACE_PHRASES,
                            primary="colocar borrador / place draft")
    pending.register_domain("whatsapp_send", matcher=safety.is_send_phrase,
                            primary="confirmar envío a <destinatario>")

    conversation.register_service_intent(
        Intent.WHATSAPP_STATUS.value, set(), lambda a, l, s, b: _svc().status(a, l, s, b))
    conversation.register_service_intent(
        Intent.WHATSAPP_FIND_CONTACT.value, {"name"}, lambda a, l, s, b: _svc().find_contact(a, l, s, b))
    conversation.register_service_intent(
        Intent.WHATSAPP_OPEN_CHAT.value, {"name"}, lambda a, l, s, b: _svc().open_chat(a, l, s, b))
    conversation.register_service_intent(
        Intent.WHATSAPP_DRAFT_MESSAGE.value, {"text", "instruction", "recipient"},
        lambda a, l, s, b: _svc().draft_message(a, l, s, b))
    conversation.register_service_intent(
        Intent.WHATSAPP_PREPARE_SEND.value, set(), lambda a, l, s, b: _svc().prepare_send(a, l, s, b))
    conversation.register_service_intent(
        Intent.WHATSAPP_CANCEL.value, set(), lambda a, l, s, b: _svc().cancel(a, l, s, b))
    conversation.register_service_intent(
        Intent.WHATSAPP_CLOSE.value, set(), lambda a, l, s, b: _svc().close(a, l, s, b))

    conversation.register_confirm_handler(
        "whatsapp_place", lambda aid, phrase, lang, b: _svc().confirm_place(aid, phrase, lang, b))
    conversation.register_confirm_handler(
        "whatsapp_send", lambda aid, phrase, lang, b: _svc().confirm_send(aid, phrase, lang, b))

    import re
    nl.register_patterns([
        (re.compile(r"\b(?:busca|buscar)\s+a?\s*(.+?)\s+en\s+whats?app\b", re.I),
         Intent.WHATSAPP_FIND_CONTACT.value, lambda m: {"name": m.group(1).strip()}),
        (re.compile(r"\bsearch\s+(?:for\s+)?(.+?)\s+(?:on|in)\s+whats?app\b", re.I),
         Intent.WHATSAPP_FIND_CONTACT.value, lambda m: {"name": m.group(1).strip()}),
        (re.compile(r"\b(?:abre|open)\s+(?:el\s+)?chat\s+(?:de|con|with)\s+(.+)", re.I),
         Intent.WHATSAPP_OPEN_CHAT.value, lambda m: {"name": m.group(1).strip()}),
        (re.compile(r"\b(?:redacta|escribe)\s+un\s+mensaje\s+(?:de\s+whats?app\s+)?"
                    r"para\s+(?:\w+)\s*(?:que\s+diga|:|,)\s*(.+)", re.I),
         Intent.WHATSAPP_DRAFT_MESSAGE.value, lambda m: {"text": m.group(1).strip()}),
        (re.compile(r"\bdraft\s+a\s+whats?app\s+message\s+(?:to\s+\w+\s+)?(?:saying|:)\s*(.+)", re.I),
         Intent.WHATSAPP_DRAFT_MESSAGE.value, lambda m: {"text": m.group(1).strip()}),
        (re.compile(r"\bwhats?app\s+status\b|\bestado\s+de\s+whats?app\b", re.I),
         Intent.WHATSAPP_STATUS.value, lambda m: {}),
        (re.compile(r"\b(?:cierra|close)\s+whats?app\b", re.I),
         Intent.WHATSAPP_CLOSE.value, lambda m: {}),
    ])


_register()
