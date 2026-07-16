"""MemoryService: explicit-consent create/update/forget + deterministic retrieval.

Nothing is ever stored implicitly. A create/update is PROPOSED (a memory-only,
expiring pending action registered with the global broker) and committed only by an
EXACT confirmation phrase; forgetting is two-staged (soft "confirmar olvido", then a
separate permanent "eliminar memoria permanentemente"). A plain "sí", a wake
detection, or a store request originating from email/web content are all refused.
Sensitive content is blocked before it can be persisted, duplicates are detected by
content hash, and a same-subject conflict is surfaced (old vs proposed) and requires
an explicit update confirmation — never a silent overwrite. Revisions are preserved.

Importing this module registers the service's intents, confirmation handlers, NL
patterns, and pending-broker domains with the core dispatcher.
"""

from __future__ import annotations

import time
import uuid

from app.config import get_settings
from app.core import conversation, errors, nl, pending
from app.core.logger import get_logger
from app.memory import retrieval, safety
from app.memory.models import (
    EV_BLOCKED,
    EV_CANCEL,
    EV_CONFIRMATION,
    EV_FORGET,
    EV_PERMANENT_DELETE,
    EV_PROPOSAL,
    EV_UPDATE,
    ForgetAction,
    Memory,
    MemoryProposal,
    content_hash,
    normalize,
)
from app.memory.repository import MemoryRepository

log = get_logger(__name__)

# Broker domains (each carries its own exact confirmation phrase set).
WRITE_DOMAIN = "memory_write"
FORGET_DOMAIN = "memory_forget"
DELETE_DOMAIN = "memory_delete"

# A store request coming from these channels is never allowed to self-store —
# external content may be summarized, but it cannot create memory.
UNTRUSTED_SOURCES = frozenset({"email", "web", "browser", "whatsapp", "page", "wake"})

_SPOKEN = {
    "propose_create": (
        "I can remember that {title}. Say 'confirmar memoria' to save it.",
        "Puedo recordar que {title}. Di 'confirmar memoria' para guardarlo.",
    ),
    "propose_update": (
        "You told me {subject} is \"{old}\". Update it to \"{new}\"? Say 'confirmar memoria'.",
        "Me dijiste que {subject} es «{old}». ¿Lo actualizo a «{new}»? Di 'confirmar memoria'.",
    ),
    "duplicate": (
        "I already remember that — nothing to add.",
        "Ya recuerdo eso — no hay nada que añadir.",
    ),
    "saved": (
        "Saved. I'll remember that.",
        "Guardado. Lo recordaré.",
    ),
    "updated": (
        "Updated the memory.",
        "Actualicé la memoria.",
    ),
    "sensitive_blocked": (
        "I won't store that — it looks like sensitive information ({category}).",
        "No guardaré eso — parece información sensible ({category}).",
    ),
    "consent_required": (
        "I only store something when you explicitly ask me to remember it.",
        "Solo guardo algo cuando me pides explícitamente que lo recuerde.",
    ),
    "search_results": (
        "I found {count} memory(ies) about that.",
        "Encontré {count} memoria(s) sobre eso.",
    ),
    "search_empty": (
        "I don't have anything remembered about that.",
        "No tengo nada recordado sobre eso.",
    ),
    "list_summary": (
        "I'm keeping {count} memory(ies).",
        "Estoy guardando {count} memoria(s).",
    ),
    "forget_prepared": (
        "I found {count} matching memory(ies). Say 'confirmar olvido' to forget them.",
        "Encontré {count} memoria(s) que coinciden. Di 'confirmar olvido' para olvidarlas.",
    ),
    "delete_prepared": (
        "This permanently deletes {count} memory(ies). Say 'eliminar memoria permanentemente'.",
        "Esto elimina {count} memoria(s) de forma permanente. Di 'eliminar memoria permanentemente'.",
    ),
    "forgotten": (
        "Done — I'll no longer bring that up ({count}).",
        "Listo — ya no lo mencionaré ({count}).",
    ),
    "deleted": (
        "Permanently deleted {count} memory(ies).",
        "Eliminé permanentemente {count} memoria(s).",
    ),
    "nothing_to_forget": (
        "I couldn't find a memory matching that. Nothing was changed.",
        "No encontré una memoria que coincida. No cambié nada.",
    ),
    "not_found": (
        "I don't have a memory like that to update.",
        "No tengo una memoria así para actualizar.",
    ),
    "need_new_value": (
        "What should the new value be?",
        "¿Cuál debería ser el nuevo valor?",
    ),
    "nothing_pending": (
        "There's no pending memory to confirm.",
        "No hay memoria pendiente que confirmar.",
    ),
    "mismatch": (
        "That's not the confirmation I need for the pending memory.",
        "Esa no es la confirmación que necesito para la memoria pendiente.",
    ),
    "wake_blocked": (
        "A wake word can't confirm a memory — please say the confirmation yourself.",
        "Una palabra de activación no puede confirmar una memoria — di la confirmación tú.",
    ),
    "expired": (
        "That memory proposal expired. Please ask again.",
        "Esa propuesta de memoria expiró. Pídemelo de nuevo.",
    ),
    "cancelled": (
        "Cancelled the pending memory.",
        "Cancelé la memoria pendiente.",
    ),
    "disabled": (
        "Memory is turned off.",
        "La memoria está desactivada.",
    ),
    "failed": (
        "That didn't work.",
        "Eso no funcionó.",
    ),
}


def _say(situation, language, **ctx):
    en, es = _SPOKEN.get(situation, _SPOKEN["failed"])
    template = en if str(language).lower().startswith("en") else es
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


def _r(status, *, state, spoken, code=None, domain="memory", action_id=None, data=None):
    return {"status": status, "error_code": code, "domain": domain, "action_id": action_id,
            "state": state, "spoken": spoken, "data": data or {}}


def _short(text: str, limit: int = 80) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


class MemoryService:
    def __init__(self, *, repository=None, db_path=None, clock=time.monotonic,
                 now_fn=None, id_factory=None):
        if repository is None:
            repository = MemoryRepository(
                db_path or get_settings().memory_db_path, now_fn=now_fn, id_factory=id_factory)
        self._repo = repository
        self._clock = clock
        self._proposal: MemoryProposal | None = None
        self._forget: ForgetAction | None = None

    # -- helpers -------------------------------------------------------------------

    @property
    def repo(self) -> MemoryRepository:
        return self._repo

    def _owner(self, settings) -> str:
        return settings.memory_owner

    def _clear_proposal(self) -> None:
        self._proposal = None

    def _clear_forget(self) -> None:
        self._forget = None

    def _enabled(self, settings, language):
        if not settings.enable_memory:
            return _r("rejected", state=errors.BLOCKED, code=errors.MEMORY_DISABLED,
                      spoken=_say("disabled", language))
        return None

    # -- propose (create / conflict-update) ----------------------------------------

    def propose(self, args, language, settings, broker):
        blocked = self._enabled(settings, language)
        if blocked:
            return blocked
        owner = self._owner(settings)
        content = (args.get("content") or args.get("instruction") or args.get("text") or "").strip()
        if not content:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONSENT_REQUIRED,
                      spoken=_say("consent_required", language))
        if len(content) > settings.memory_max_content_chars:
            content = content[: settings.memory_max_content_chars]

        # A store request from an external/automated channel can never self-store.
        source = normalize(args.get("source") or "voice")
        if source in UNTRUSTED_SOURCES:
            self._repo.audit(EV_BLOCKED, owner=owner, detail=f"untrusted_source:{source}")
            return _r("rejected", state=errors.BLOCKED, code=errors.CONSENT_REQUIRED,
                      spoken=_say("consent_required", language))

        mem_type = args.get("type") or safety.infer_type(content)
        if not safety.valid_type(mem_type):
            mem_type = "fact"
        subject = normalize(args.get("subject") or safety.infer_subject(content))
        title = (args.get("title") or safety.infer_title(content, subject))[: settings.memory_max_title_chars]

        # Sensitive-data gate (never weakened by a flag; never echoes the content).
        if settings.memory_block_sensitive:
            ok, category, _reasons = safety.check_storable(title, content)
            if not ok:
                self._repo.audit(EV_BLOCKED, owner=owner, memory_type=mem_type,
                                 detail=f"category:{category}")
                return _r("rejected", state=errors.BLOCKED, code=errors.SENSITIVE_CONTENT_BLOCKED,
                          spoken=_say("sensitive_blocked", language, category=category))

        entities = [normalize(e) for e in (args.get("entities") or safety.infer_entities(content))]
        tags = [normalize(t) for t in (args.get("tags") or [])]
        chash = content_hash(owner, mem_type, content)

        # Duplicate: identical content already remembered → nothing to store.
        if self._repo.find_duplicate(owner, mem_type, chash):
            return _r("ok", state=errors.COMPLETED, code=errors.DUPLICATE_MEMORY,
                      spoken=_say("duplicate", language), data={"duplicate": True})

        memory = Memory(
            id=self._repo.new_id(), owner=owner, type=mem_type, title=title, content=content,
            entities=entities, tags=tags, subject=subject, source=source,
            confidence=float(args.get("confidence", settings.memory_default_confidence)),
            expires_at=args.get("expires_at"), content_hash=chash,
        )

        # Conflict: same subject, different value → propose an UPDATE (never silent).
        conflict = self._repo.find_conflict(owner, mem_type, subject, chash)
        if conflict is not None:
            return self._register_proposal(
                memory, settings, language, broker, mode="update",
                target=conflict, conflict=True)
        return self._register_proposal(memory, settings, language, broker, mode="create")

    def _register_proposal(self, memory, settings, language, broker, *, mode,
                           target=None, conflict=False):
        expires = settings.memory_action_expires_seconds
        proposal = MemoryProposal(
            action_id=self._repo.new_id(), owner=memory.owner, mode=mode, memory=memory,
            target_id=target.id if target else None,
            old_title=target.title if target else "",
            old_content=target.content if target else "",
            conflict=conflict,
            expires_at_monotonic=self._clock() + expires,
            expires_at_iso=self._repo.future_iso(expires),
        )
        self._proposal = proposal
        broker.register(domain=WRITE_DOMAIN, action_id=proposal.action_id,
                        target=_short(memory.title), expires_at=proposal.expires_at_iso,
                        cancel=self._clear_proposal)
        self._repo.audit(EV_PROPOSAL, owner=memory.owner, memory_type=memory.type,
                         detail=f"mode:{mode} title:{_short(memory.title, 40)}")
        if mode == "update":
            spoken = _say("propose_update", language, subject=memory.subject or memory.title,
                          old=_short(proposal.old_content, 60), new=_short(memory.content, 60))
        else:
            spoken = _say("propose_create", language, title=_short(memory.title, 80))
        return _r("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                  action_id=proposal.action_id, spoken=spoken, data=proposal.preview())

    # -- update (explicit "actualiza la memoria de X a Y") -------------------------

    def propose_update(self, args, language, settings, broker):
        blocked = self._enabled(settings, language)
        if blocked:
            return blocked
        owner = self._owner(settings)
        new_value = (args.get("content") or args.get("text") or args.get("instruction") or "").strip()

        target = None
        if args.get("id"):
            target = self._repo.get(args["id"])
        if target is None:
            query = (args.get("query") or "").strip()
            if query:
                hits = retrieval.search(self._repo, owner, query, limit=1)
                target = hits[0] if hits else None
        if target is None or target.owner != owner:
            return _r("rejected", state=errors.BLOCKED, code=errors.MEMORY_NOT_FOUND,
                      spoken=_say("not_found", language))
        if not new_value:
            return _r("needs_input", state=errors.AWAITING_CONFIRMATION,
                      code=errors.CONFIRMATION_REQUIRED, action_id=None,
                      spoken=_say("need_new_value", language),
                      data={"target_id": target.id, "title": target.title})

        if len(new_value) > settings.memory_max_content_chars:
            new_value = new_value[: settings.memory_max_content_chars]
        if settings.memory_block_sensitive:
            ok, category, _ = safety.check_storable(target.title, new_value)
            if not ok:
                self._repo.audit(EV_BLOCKED, owner=owner, memory_type=target.type,
                                 detail=f"category:{category}")
                return _r("rejected", state=errors.BLOCKED, code=errors.SENSITIVE_CONTENT_BLOCKED,
                          spoken=_say("sensitive_blocked", language, category=category))

        mem_type = args.get("type") or target.type
        updated = Memory(
            id=target.id, owner=owner, type=mem_type, title=target.title, content=new_value,
            entities=[normalize(e) for e in (args.get("entities") or safety.infer_entities(new_value))] or target.entities,
            tags=[normalize(t) for t in (args.get("tags") or target.tags)],
            subject=normalize(args.get("subject") or target.subject),
            source=target.source, confidence=target.confidence, expires_at=target.expires_at,
            content_hash=content_hash(owner, mem_type, new_value), revision=target.revision,
            created_at=target.created_at, status=target.status,
        )
        return self._register_proposal(updated, settings, language, broker, mode="update",
                                       target=target, conflict=False)

    # -- confirm write (create / update) -------------------------------------------

    def confirm_write(self, action_id, phrase, language, broker, wake=False):
        if wake:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_MISMATCH,
                      spoken=_say("wake_blocked", language))
        proposal = self._proposal
        if proposal is None or proposal.action_id != action_id or proposal.used:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                      spoken=_say("nothing_pending", language))
        if self._clock() >= proposal.expires_at_monotonic:
            self._clear_proposal(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED,
                      spoken=_say("expired", language))

        memory = proposal.memory
        if proposal.mode == "update" and proposal.target_id:
            target = self._repo.get(proposal.target_id)
            if target is None:
                # The target vanished — fall back to creating it.
                saved = self._repo.insert(memory, reason="created")
                event, situation = EV_CONFIRMATION, "saved"
            else:
                target.type = memory.type
                target.title = memory.title
                target.content = memory.content
                target.entities = memory.entities
                target.tags = memory.tags
                target.subject = memory.subject
                target.confidence = memory.confidence
                target.content_hash = memory.content_hash
                target.status = memory.status or target.status
                saved = self._repo.update(target, reason="updated")
                event, situation = EV_UPDATE, "updated"
        else:
            saved = self._repo.insert(memory, reason="created")
            event, situation = EV_CONFIRMATION, "saved"

        proposal.used = True
        self._clear_proposal(); broker.clear(action_id)
        self._repo.audit(event, owner=saved.owner, memory_id=saved.id, memory_type=saved.type,
                         detail=f"rev:{saved.revision}")
        log.info("memory %s (id hidden) rev=%d", situation, saved.revision)
        return _r("executed", state=errors.COMPLETED, action_id=action_id,
                  spoken=_say(situation, language),
                  data={"id": saved.id, "type": saved.type, "revision": saved.revision})

    # -- search / list / retrieval -------------------------------------------------

    def search(self, args, language, settings, broker):
        blocked = self._enabled(settings, language)
        if blocked:
            return blocked
        owner = self._owner(settings)
        query = (args.get("query") or "").strip()
        hits = retrieval.search(self._repo, owner, query, limit=settings.memory_search_max_results)
        if hits:
            self._repo.touch_used([m.id for m in hits])
            self._repo.audit("retrieval", owner=owner,
                             detail=f"search returned={len(hits)}")
        results = [
            {"id": m.id, "type": m.type, "title": m.title, "content": _short(m.content, 200),
             "date": (m.updated_at or m.created_at or "")[:10], "confidence": m.confidence}
            for m in hits
        ]
        situation = "search_results" if results else "search_empty"
        return _r("ok", state=errors.COMPLETED,
                  spoken=_say(situation, language, count=len(results)),
                  data={"count": len(results), "results": results})

    def context_bundle(self, query, settings, *, max_items=None, max_chars=None) -> dict:
        """Bounded retrieval bundle for planner injection — never the whole DB.

        ``max_items``/``max_chars`` override the defaults so a caller (e.g. the task
        planner, which has its own TASK_CONTEXT_* budget) stays bounded too."""
        return retrieval.retrieve(
            self._repo, self._owner(settings), query,
            max_items=settings.memory_context_max_items if max_items is None else max_items,
            max_chars=settings.memory_context_max_chars if max_chars is None else max_chars)

    def list(self, args, language, settings, broker):
        blocked = self._enabled(settings, language)
        if blocked:
            return blocked
        owner = self._owner(settings)
        mem_type = args.get("type") or None
        if mem_type and not safety.valid_type(mem_type):
            mem_type = None
        memories = self._repo.list(owner, mem_type=mem_type, limit=settings.memory_list_max_results)
        data = {
            "count": len(memories),
            "memories": [
                {"id": m.id, "type": m.type, "title": m.title, "content": _short(m.content, 200),
                 "date": (m.updated_at or m.created_at or "")[:10], "confidence": m.confidence,
                 "revision": m.revision}
                for m in memories
            ],
        }
        return _r("ok", state=errors.COMPLETED,
                  spoken=_say("list_summary", language, count=len(memories)), data=data)

    # -- forget (soft) / delete (permanent) ----------------------------------------

    def prepare_forget(self, args, language, settings, broker):
        blocked = self._enabled(settings, language)
        if blocked:
            return blocked
        owner = self._owner(settings)
        mode = "permanent" if normalize(args.get("mode")) == "permanent" else "soft"

        ids = args.get("memory_ids") or ([args["id"]] if args.get("id") else [])
        matches: list[Memory] = []
        if ids:
            for mid in ids:
                mem = self._repo.get(mid)
                if mem and mem.owner == owner:
                    matches.append(mem)
        else:
            query = (args.get("query") or "").strip()
            if not query:
                # Never forget by a broad, empty query.
                return _r("rejected", state=errors.BLOCKED, code=errors.NOTHING_TO_FORGET,
                          spoken=_say("nothing_to_forget", language))
            matches = retrieval.search(self._repo, owner, query, limit=10)

        if not matches:
            return _r("rejected", state=errors.BLOCKED, code=errors.NOTHING_TO_FORGET,
                      spoken=_say("nothing_to_forget", language))

        expires = settings.memory_action_expires_seconds
        forget = ForgetAction(
            action_id=self._repo.new_id(), owner=owner, mode=mode,
            memory_ids=[m.id for m in matches], titles=[m.title for m in matches],
            types=[m.type for m in matches],
            expires_at_monotonic=self._clock() + expires,
            expires_at_iso=self._repo.future_iso(expires),
        )
        self._forget = forget
        domain = DELETE_DOMAIN if mode == "permanent" else FORGET_DOMAIN
        broker.register(domain=domain, action_id=forget.action_id,
                        target=f"{len(matches)} memory(ies)", expires_at=forget.expires_at_iso,
                        cancel=self._clear_forget)
        self._repo.audit(EV_PROPOSAL, owner=owner,
                         detail=f"forget mode:{mode} count:{len(matches)}")
        situation = "delete_prepared" if mode == "permanent" else "forget_prepared"
        return _r("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                  action_id=forget.action_id, domain=domain,
                  spoken=_say(situation, language, count=len(matches)),
                  data=forget.preview())

    def _validate_forget(self, action_id, mode, language, broker):
        forget = self._forget
        if forget is None or forget.action_id != action_id or forget.used or forget.mode != mode:
            return None, _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                            spoken=_say("nothing_pending", language))
        if self._clock() >= forget.expires_at_monotonic:
            self._clear_forget(); broker.clear(action_id)
            return None, _r("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED,
                            spoken=_say("expired", language))
        return forget, None

    def confirm_forget(self, action_id, phrase, language, broker, wake=False):
        if wake:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_MISMATCH,
                      spoken=_say("wake_blocked", language))
        forget, err = self._validate_forget(action_id, "soft", language, broker)
        if err:
            return err
        changed = self._repo.soft_forget(forget.memory_ids)
        forget.used = True
        self._clear_forget(); broker.clear(action_id)
        for mid, mtype in zip(forget.memory_ids, forget.types):
            self._repo.audit(EV_FORGET, owner=forget.owner, memory_id=mid, memory_type=mtype)
        return _r("executed", state=errors.COMPLETED, action_id=action_id,
                  spoken=_say("forgotten", language, count=changed), data={"forgotten": changed})

    def confirm_delete(self, action_id, phrase, language, broker, wake=False):
        if wake:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_MISMATCH,
                      spoken=_say("wake_blocked", language))
        forget, err = self._validate_forget(action_id, "permanent", language, broker)
        if err:
            return err
        # Audit BEFORE deletion (id/type only — never content), then remove rows.
        for mid, mtype in zip(forget.memory_ids, forget.types):
            self._repo.audit(EV_PERMANENT_DELETE, owner=forget.owner, memory_id=mid,
                             memory_type=mtype)
        removed = self._repo.permanent_delete(forget.memory_ids)
        forget.used = True
        self._clear_forget(); broker.clear(action_id)
        return _r("executed", state=errors.COMPLETED, action_id=action_id,
                  spoken=_say("deleted", language, count=removed), data={"deleted": removed})

    # -- cancel / status / export --------------------------------------------------

    def cancel(self, args, language, settings, broker):
        had = self._proposal is not None or self._forget is not None
        self._clear_proposal(); self._clear_forget()
        broker.cancel_active()
        if had:
            self._repo.audit(EV_CANCEL, owner=self._owner(settings))
        return _r("cancelled", state=errors.COMPLETED, spoken=_say("cancelled", language))

    def status(self, args, language, settings, broker):
        owner = self._owner(settings)
        counts = self._repo.counts(owner) if settings.enable_memory else {}
        summary = broker.summary()
        pending_memory = summary if (summary and summary.get("domain", "").startswith("memory")) else None
        return _r("ok", state=errors.COMPLETED, spoken="", data={
            "enabled": settings.enable_memory,
            "owner": owner,
            "db_path": str(settings.memory_db_path),
            "counts": counts,
            "pending": pending_memory,
            "blocked_categories": list(safety.BLOCKED_CATEGORIES),
            "context_limits": {
                "max_items": settings.memory_context_max_items,
                "max_chars": settings.memory_context_max_chars,
            },
        })

    def export(self, settings, *, include_forgotten=False) -> dict:
        owner = self._owner(settings)
        statuses = ("active", "forgotten") if include_forgotten else ("active",)
        memories = self._repo.list(owner, statuses=statuses, limit=100000)
        out = []
        for m in memories:
            # Defensive: never export anything that would trip the sensitive gate.
            ok, _cat, _r = safety.check_storable(m.title, m.content)
            if not ok:
                continue
            out.append(m.public())
        self._repo.audit("retrieval", owner=owner, detail=f"export count:{len(out)}")
        return {
            "owner": owner,
            "exported_at": self._repo.now_iso(),
            "count": len(out),
            "memories": out,
        }

    def audit_log(self, limit: int = 50) -> list[dict]:
        return self._repo.audit_recent(limit)

    def revisions(self, memory_id: str) -> list[dict]:
        return self._repo.revisions(memory_id)


_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    global _service
    if _service is None:
        _service = MemoryService()
    return _service


def _svc():
    return get_memory_service()


def _memory_confirm(args, language, settings, broker):
    """The memory_confirm intent routes a phrase through the normal confirm path."""
    phrase = args.get("phrase", "")
    return conversation.dispatch(
        nl.ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=phrase),
        language=language, settings=settings)


def _register() -> None:
    pending.register_domain(WRITE_DOMAIN, phrases=safety.WRITE_CONFIRM_PHRASES,
                            primary="confirmar memoria / confirm memory")
    pending.register_domain(FORGET_DOMAIN, phrases=safety.FORGET_CONFIRM_PHRASES,
                            primary="confirmar olvido / confirm forget")
    pending.register_domain(DELETE_DOMAIN, phrases=safety.DELETE_CONFIRM_PHRASES,
                            primary="eliminar memoria permanentemente")

    for intent, keys, method in [
        ("memory_propose",
         {"content", "instruction", "text", "type", "title", "subject", "entities",
          "tags", "source", "confidence", "expires_at"}, "propose"),
        ("memory_search", {"query"}, "search"),
        ("memory_list", {"type"}, "list"),
        ("memory_update",
         {"query", "id", "content", "text", "instruction", "type", "subject",
          "entities", "tags"}, "propose_update"),
        ("memory_forget", {"query", "mode", "memory_ids", "id"}, "prepare_forget"),
        ("memory_cancel", set(), "cancel"),
    ]:
        conversation.register_service_intent(
            intent, keys, (lambda m: lambda a, l, s, b: getattr(_svc(), m)(a, l, s, b))(method))

    conversation.register_service_intent("memory_confirm", {"phrase"}, _memory_confirm)

    conversation.register_confirm_handler(
        WRITE_DOMAIN,
        lambda aid, phrase, lang, b, wake=False: _svc().confirm_write(aid, phrase, lang, b, wake=wake))
    conversation.register_confirm_handler(
        FORGET_DOMAIN,
        lambda aid, phrase, lang, b, wake=False: _svc().confirm_forget(aid, phrase, lang, b, wake=wake))
    conversation.register_confirm_handler(
        DELETE_DOMAIN,
        lambda aid, phrase, lang, b, wake=False: _svc().confirm_delete(aid, phrase, lang, b, wake=wake))

    import re
    nl.register_patterns([
        # --- propose (explicit consent) ---------------------------------------------
        (re.compile(r"\b(?:recu[eé]rdame|recuerda)\s+que\s+(.+)", re.I),
         "memory_propose", lambda m: {"content": m.group(1).strip(), "source": "voice"}),
        (re.compile(r"\bguarda(?:r)?\s+(?:esto\s*:?\s*|que\s+)(.+)", re.I),
         "memory_propose", lambda m: {"content": m.group(1).strip(), "source": "voice"}),
        (re.compile(r"\b(?:apunta|toma\s+nota\s+de)\s+que\s+(.+)", re.I),
         "memory_propose", lambda m: {"content": m.group(1).strip(), "source": "voice"}),
        (re.compile(r"\bremember\s+that\s+(.+)", re.I),
         "memory_propose", lambda m: {"content": m.group(1).strip(), "source": "voice"}),
        (re.compile(r"\b(?:save\s+this|take\s+note\s+that|note\s+that)\s*:?\s*(.+)", re.I),
         "memory_propose", lambda m: {"content": m.group(1).strip(), "source": "voice"}),
        (re.compile(r"\bremember\s+(.+)", re.I),
         "memory_propose", lambda m: {"content": m.group(1).strip(), "source": "voice"}),
        # --- update (before the broad forget/search) --------------------------------
        (re.compile(r"\bactualiza\s+(?:la\s+)?memoria\s+(?:de(?:l)?|sobre|de\s+la)\s+(.+?)"
                    r"(?:\s+a\s+|\s*:\s*)(.+)", re.I),
         "memory_update", lambda m: {"query": m.group(1).strip(), "content": m.group(2).strip()}),
        (re.compile(r"\bupdate\s+(?:the\s+|your\s+)?memory\s+(?:of|about|for)\s+(.+?)\s+to\s+(.+)", re.I),
         "memory_update", lambda m: {"query": m.group(1).strip(), "content": m.group(2).strip()}),
        (re.compile(r"\bactualiza\s+(?:la\s+)?memoria\s+(?:de(?:l)?|sobre)\s+(.+)", re.I),
         "memory_update", lambda m: {"query": m.group(1).strip()}),
        # --- forget: permanent (explicit) before soft ------------------------------
        (re.compile(r"\b(?:elimina|borra)\s+(?:permanentemente|para\s+siempre)\s+(.+)", re.I),
         "memory_forget", lambda m: {"query": m.group(1).strip(), "mode": "permanent"}),
        (re.compile(r"\bpermanently\s+(?:delete|forget)\s+(.+)", re.I),
         "memory_forget", lambda m: {"query": m.group(1).strip(), "mode": "permanent"}),
        # --- forget: soft ----------------------------------------------------------
        (re.compile(r"\b(?:olvida|olv[ií]date\s+de)\s+(?:esa\s+|esta\s+|la\s+|el\s+|mi\s+|el\s+)?(.+)", re.I),
         "memory_forget", lambda m: {"query": m.group(1).strip(), "mode": "soft"}),
        (re.compile(r"\bforget\s+(?:about\s+|that\s+|the\s+|my\s+)?(.+)", re.I),
         "memory_forget", lambda m: {"query": m.group(1).strip(), "mode": "soft"}),
        # --- search ----------------------------------------------------------------
        (re.compile(r"\bqu[eé]\s+(?:recuerdas|sabes|tienes)\s+(?:de|sobre|acerca\s+de)\s+(.+)", re.I),
         "memory_search", lambda m: {"query": m.group(1).strip().rstrip("?")}),
        (re.compile(r"\b(?:recuerdas|sabes)\s+algo\s+(?:de|sobre)\s+(.+)", re.I),
         "memory_search", lambda m: {"query": m.group(1).strip().rstrip("?")}),
        (re.compile(r"\bwhat\s+do\s+you\s+(?:remember|know)\s+about\s+(.+)", re.I),
         "memory_search", lambda m: {"query": m.group(1).strip().rstrip("?")}),
        (re.compile(r"\b(?:busca|buscar)\s+en\s+(?:tu\s+)?memoria\s+(.+)", re.I),
         "memory_search", lambda m: {"query": m.group(1).strip().rstrip("?")}),
        # --- list ------------------------------------------------------------------
        (re.compile(r"\b(?:lista|muestra|mu[eé]strame|ense[nñ]a)\s+(?:tus\s+)?"
                    r"(?:memorias|recuerdos)\b", re.I),
         "memory_list", lambda m: {}),
        (re.compile(r"\blist\s+(?:your\s+)?memories\b", re.I),
         "memory_list", lambda m: {}),
        (re.compile(r"\b(?:qu[eé]\s+recuerdas|what\s+do\s+you\s+remember)\s*\??$", re.I),
         "memory_list", lambda m: {}),
    ])


_register()
