"""MemoryService behaviour (Phase 5A): explicit consent, exact confirmation, wake
rejection, sensitive blocking, duplicates, conflicts, revisions, two-stage
forgetting, and cross-domain rejection — all through the real dispatcher."""

import itertools
import types
from datetime import datetime, timezone

import pytest

from app.config import get_settings
from app.core import conversation, errors, pending
from app.core.nl import ServiceCommand
from app.memory import service as memory_service
from app.memory.service import MemoryService


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_memory", True)
    monkeypatch.setattr(settings, "memory_owner", "local")
    monkeypatch.setattr(settings, "memory_db_path", tmp_path / "mem.db")
    monkeypatch.setattr(settings, "memory_block_sensitive", True)
    monkeypatch.setattr(settings, "memory_action_expires_seconds", 120)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    clock = {"t": 1000.0}
    ids = (f"mem_{i}" for i in itertools.count(1))
    svc = MemoryService(db_path=tmp_path / "mem.db", clock=lambda: clock["t"],
                        now_fn=lambda: datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                        id_factory=lambda: next(ids))
    monkeypatch.setattr(memory_service, "_service", svc)
    return types.SimpleNamespace(settings=settings, clock=clock, svc=svc,
                                 broker=pending.get_pending_broker())


def _d(intent, **args):
    return conversation.dispatch(ServiceCommand(intent=intent, arguments=args), language="es")


def _confirm(phrase, wake=False):
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=phrase),
        language="es", wake=wake)


def _count(env):
    return env.svc.repo.counts("local")["active"]


def _propose(env, content="mi editor preferido es VS Code", **extra):
    return _d("memory_propose", content=content, source="voice", **extra)


# --- explicit consent: nothing stored without an explicit proposal + confirmation --


def test_propose_does_not_store_until_confirmed(env):
    result = _propose(env)
    assert result["status"] == "needs_confirmation"
    assert env.broker.summary()["domain"] == "memory_write"
    assert _count(env) == 0                       # NOT stored yet


def test_empty_content_requires_consent(env):
    assert _d("memory_propose", content="", source="voice")["error_code"] == errors.CONSENT_REQUIRED


@pytest.mark.parametrize("phrase", ["confirmar memoria", "guardar memoria", "confirm memory"])
def test_each_accepted_phrase_commits(env, phrase):
    _propose(env)
    assert _confirm(phrase)["status"] == "executed"
    assert _count(env) == 1


@pytest.mark.parametrize("phrase", ["sí", "si", "yes", "ok", "confirmar", "vale", "dale"])
def test_plain_yes_never_commits(env, phrase):
    _propose(env)
    assert _confirm(phrase)["error_code"] == errors.CONFIRMATION_MISMATCH
    assert _count(env) == 0
    # …and the proposal is still pending, so the exact phrase still works afterwards.
    assert _confirm("confirmar memoria")["status"] == "executed"


def test_wake_cannot_confirm(env):
    _propose(env)
    assert _confirm("confirmar memoria", wake=True)["error_code"] == errors.CONFIRMATION_MISMATCH
    assert _count(env) == 0
    # A deliberate (non-wake) confirmation still works.
    assert _confirm("confirmar memoria")["status"] == "executed"


def test_expired_proposal_rejected(env):
    _propose(env)
    env.clock["t"] += 200                          # past the 120s expiry
    assert _confirm("confirmar memoria")["error_code"] == errors.ACTION_EXPIRED
    assert _count(env) == 0


# --- untrusted content cannot self-store -------------------------------------------


@pytest.mark.parametrize("source", ["email", "web", "browser", "whatsapp", "page", "wake"])
def test_external_source_cannot_self_store(env, source):
    result = _d("memory_propose", content="algo interesante del correo", source=source)
    assert result["error_code"] == errors.CONSENT_REQUIRED
    assert env.broker.summary() is None and _count(env) == 0


# --- sensitive-data blocking -------------------------------------------------------


@pytest.mark.parametrize("content", [
    "mi contraseña es hunter2",
    "guarda el token sk-ABCDEF1234567890ABCDEF12",
    "mi tarjeta 4111 1111 1111 1111",
    "session_id=abc123DEF456ghi789xyz",
])
def test_sensitive_blocked_and_not_pending(env, content):
    result = _propose(env, content=content)
    assert result["error_code"] == errors.SENSITIVE_CONTENT_BLOCKED
    assert env.broker.summary() is None and _count(env) == 0
    # The offending content is never echoed back.
    assert "hunter2" not in result["spoken"] and "sk-" not in result["spoken"]
    assert any(e["event"] == "blocked" for e in env.svc.audit_log())


# --- duplicates --------------------------------------------------------------------


def test_duplicate_detected(env):
    _propose(env)
    _confirm("confirmar memoria")
    dup = _propose(env)
    assert dup["error_code"] == errors.DUPLICATE_MEMORY
    assert env.broker.summary() is None          # no new pending for a duplicate


# --- conflicts + revisions ---------------------------------------------------------


def test_conflict_presents_old_and_new_and_requires_confirmation(env):
    _propose(env, content="mi editor preferido es VS Code")
    _confirm("confirmar memoria")
    conflict = _propose(env, content="mi editor preferido es Neovim")
    assert conflict["status"] == "needs_confirmation"
    assert conflict["data"]["mode"] == "update" and conflict["data"]["conflict"] is True
    assert "VS Code" in conflict["data"]["old_value"]
    # Not overwritten until confirmed.
    assert env.svc.repo.active("local")[0].content == "mi editor preferido es VS Code"
    done = _confirm("confirmar memoria")
    assert done["status"] == "executed" and done["data"]["revision"] == 2
    assert _count(env) == 1                        # updated in place, not duplicated


def test_revision_history_preserved(env):
    _propose(env, content="mi editor preferido es VS Code")
    res = _confirm("confirmar memoria")
    mem_id = res["data"]["id"]
    _propose(env, content="mi editor preferido es Neovim")
    _confirm("confirmar memoria")
    revs = env.svc.revisions(mem_id)
    assert [(r["revision"], r["reason"]) for r in revs] == [(1, "created"), (2, "updated")]
    assert revs[0]["content"] == "mi editor preferido es VS Code"


def test_explicit_update_flow(env):
    _propose(env, content="el endpoint de deploy es staging")
    _confirm("confirmar memoria")
    upd = _d("memory_update", query="endpoint de deploy", content="el endpoint de deploy es production")
    assert upd["status"] == "needs_confirmation" and upd["data"]["mode"] == "update"
    assert _confirm("confirmar memoria")["status"] == "executed"
    hit = env.svc.search({"query": "endpoint de deploy"}, "es", env.settings, env.broker)
    assert "production" in hit["data"]["results"][0]["content"]


def test_update_unknown_target(env):
    assert _d("memory_update", query="algo que no existe", content="x")["error_code"] == errors.MEMORY_NOT_FOUND


# --- two-stage forgetting ----------------------------------------------------------


def _store(env, content, **extra):
    _propose(env, content=content, **extra)
    return _confirm("confirmar memoria")


def test_soft_forget_excludes_from_retrieval(env):
    _store(env, "mi editor preferido es VS Code")
    prep = _d("memory_forget", query="editor preferido", mode="soft")
    assert prep["status"] == "needs_confirmation" and prep["domain"] == "memory_forget"
    assert prep["data"]["count"] == 1                    # matches shown before forgetting
    assert _confirm("confirmar olvido")["status"] == "executed"
    assert env.svc.search({"query": "editor"}, "es", env.settings, env.broker)["data"]["count"] == 0


def test_soft_phrase_cannot_trigger_permanent_and_vice_versa(env):
    _store(env, "mi editor preferido es VS Code")
    _d("memory_forget", query="editor", mode="permanent")   # prepared as permanent
    # The soft phrase must not confirm a permanent action.
    assert _confirm("confirmar olvido")["error_code"] == errors.CONFIRMATION_MISMATCH
    # The correct permanent phrase works.
    assert _confirm("eliminar memoria permanentemente")["status"] == "executed"


def test_permanent_delete_requires_its_own_phrase(env):
    res = _store(env, "una decisión temporal")
    mem_id = res["data"]["id"]
    _d("memory_forget", query="decisión temporal", mode="permanent")
    assert _confirm("eliminar memoria permanentemente")["status"] == "executed"
    assert env.svc.repo.get(mem_id) is None                 # row gone
    assert any(e["event"] == "permanent_delete" for e in env.svc.audit_log())


def test_forget_by_empty_query_refused(env):
    _store(env, "algo")
    assert _d("memory_forget", query="", mode="soft")["error_code"] == errors.NOTHING_TO_FORGET


def test_forget_no_match(env):
    assert _d("memory_forget", query="inexistente", mode="soft")["error_code"] == errors.NOTHING_TO_FORGET


# --- cross-domain rejection --------------------------------------------------------


def test_text_phrase_cannot_confirm_memory(env):
    _propose(env)
    assert _confirm("confirmar escritura")["error_code"] == errors.CONFIRMATION_MISMATCH
    assert _count(env) == 0


def test_forget_phrase_cannot_confirm_a_write(env):
    _propose(env)
    assert _confirm("confirmar olvido")["error_code"] == errors.CONFIRMATION_MISMATCH
    assert _count(env) == 0


def test_memory_phrase_cannot_confirm_text(env, monkeypatch):
    # Register a text draft as the active pending, then a memory phrase must miss.
    from app.text import service as text_service

    ts = text_service.TextActionService(
        target_probe=lambda: {"title": "Untitled - Notepad", "process_name": "notepad.exe",
                              "exe_path": r"C:\Windows\notepad.exe", "control_type": "Edit",
                              "is_password": False, "is_elevated": False, "editable": True,
                              "error": None},
        id_factory=lambda: "t1", inserter=lambda p, s: {"ok": True, "method": "uia"})
    monkeypatch.setattr(text_service, "_service", ts)
    monkeypatch.setattr(env.settings, "enable_real_text_input", False)
    _d("type_text", text="hola", mode="insert")
    assert env.broker.summary()["domain"] == "text"
    assert _confirm("confirmar memoria")["error_code"] == errors.CONFIRMATION_MISMATCH


# --- cancel + disabled + owner isolation -------------------------------------------


def test_cancel_clears_pending(env):
    _propose(env)
    assert _d("memory_cancel")["status"] == "cancelled"
    assert env.broker.summary() is None


def test_disabled_blocks_everything(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_memory", False)
    assert _propose(env)["error_code"] == errors.MEMORY_DISABLED


def test_owner_isolation_at_service_level(env, monkeypatch):
    _store(env, "mi editor preferido es VS Code")
    monkeypatch.setattr(env.settings, "memory_owner", "someone_else")
    assert env.svc.search({"query": "editor"}, "es", env.settings, env.broker)["data"]["count"] == 0
    assert env.svc.list({}, "es", env.settings, env.broker)["data"]["count"] == 0
    monkeypatch.setattr(env.settings, "memory_owner", "local")
    assert env.svc.search({"query": "editor"}, "es", env.settings, env.broker)["data"]["count"] == 1
