"""Plan proposal + bounded memory context (Phase 5B).

The planner only PROPOSES candidate steps — it never executes and its output is
always handed to ``validator.validate_plan`` before anything is stored or run.
Three sources, in priority order:

1. explicit steps supplied by the caller (API/tests) — fully deterministic;
2. an optional LLM proposal (only when ENABLE_MULTI_STEP_TASKS and Ollama are
   available) — strictly JSON, and still validated downstream;
3. a small deterministic keyword heuristic for simple spoken requests.

Relevant memories are retrieved as a BOUNDED bundle (TASK_CONTEXT_MAX_*), and only
their IDs are recorded on the task — the full memory database is never copied into
planner context, and task outputs are never written back as memories.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.core.logger import get_logger
from app.tasks import models

log = get_logger(__name__)


@dataclass
class ProposedPlan:
    title: str
    raw_steps: list[dict]
    memory_refs: list[str] = field(default_factory=list)
    memory_context: list[dict] = field(default_factory=list)
    source: str = "explicit"


def _title(user_request: str) -> str:
    text = " ".join((user_request or "").split())
    return text if len(text) <= 80 else text[:80].rsplit(" ", 1)[0] + "…"


def gather_memory_context(user_request: str, settings) -> tuple[list[str], list[dict]]:
    """A BOUNDED memory bundle for the request. Returns (ids, compact views)."""
    if not settings.enable_memory:
        return [], []
    try:
        from app.memory.service import get_memory_service

        bundle = get_memory_service().context_bundle(
            user_request, settings,
            max_items=settings.task_context_max_memories,
            max_chars=settings.task_context_max_chars)
    except Exception:
        log.warning("task planner: memory context unavailable", exc_info=True)
        return [], []
    memories = bundle.get("memories", [])
    return [m["id"] for m in memories], memories


# --- deterministic heuristic (spoken requests without explicit steps) --------------

_RESEARCH = re.compile(r"\b(investiga(?:r)?|busca(?:r)?|research|look\s+up|find\s+info)\b", re.I)
_SUMMARIZE = re.compile(r"\b(resume|resumir|resumen|summari[sz]e|summary)\b", re.I)
_ABOUT = re.compile(r"\b(?:sobre|acerca\s+de|de|about|on|for)\s+(.+)", re.I)


def heuristic_steps(user_request: str) -> list[dict]:
    """A conservative read-only research composition. Effect steps (drafting,
    sending) are NEVER inferred from free text — those must be explicit."""
    text = user_request or ""
    steps: list[dict] = []
    if _RESEARCH.search(text):
        topic = text
        m = _ABOUT.search(text)
        if m:
            topic = m.group(1).strip()
        steps.append({"intent": "browser_search", "arguments": {"query": topic}})
        if _SUMMARIZE.search(text) or True:
            steps.append({"intent": "browser_summarize", "arguments": {},
                          "dependencies": [0]})
    elif _SUMMARIZE.search(text):
        steps.append({"intent": "browser_summarize", "arguments": {}})
    return steps


# --- optional LLM proposal ----------------------------------------------------------


def _intent_catalog() -> str:
    from app.core.conversation import SERVICE_INTENT_ARGS

    lines = []
    for intent, (domain, risk) in sorted(models.SUPPORTED_INTENTS.items()):
        keys = sorted(SERVICE_INTENT_ARGS.get(intent, set()))
        lines.append(f"- {intent} | {domain} | {risk} | args: {', '.join(keys) or '(none)'}")
    return "\n".join(lines)


def llm_steps(user_request: str, memory_context: list[dict], settings) -> list[dict] | None:
    """Ask the local LLM for a JSON plan. Returns None on any failure — the result
    is validated downstream regardless, so the LLM can never execute anything."""
    try:
        from app.llm.ollama_client import OllamaClient
    except Exception:
        return None
    context = json.dumps(memory_context, ensure_ascii=False)
    prompt = (
        "You plan multi-step tasks for a local assistant by composing ONLY the "
        "allowed steps below. Never invent tools, URLs, selectors, shell or code.\n\n"
        f"Allowed steps:\n{_intent_catalog()}\n\n"
        f"Relevant memories (bounded): {context}\n\n"
        f"User request: {user_request}\n\n"
        "Return ONLY a JSON array of steps, each {\"intent\": str, \"arguments\": obj, "
        "\"dependencies\": [int]}. dependencies are indices of earlier steps."
    )
    try:
        client = OllamaClient()
        result = client.generate(prompt=prompt, model=settings.llm_planner_model,
                                 timeout=settings.llm_planner_timeout_seconds, json_format=True)
    except Exception:
        return None
    if result.get("simulated") or not result.get("response"):
        return None
    raw = str(result["response"]).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except ValueError:
        # Some models wrap the array in an object.
        return None
    if isinstance(data, dict):
        data = data.get("steps") or data.get("plan") or []
    return data if isinstance(data, list) else None


def propose(user_request: str, settings, *, explicit_steps=None, title: str = "") -> ProposedPlan:
    refs, context = gather_memory_context(user_request, settings)
    if explicit_steps is not None:
        raw, source = list(explicit_steps), "explicit"
    else:
        raw, source = None, "heuristic"
        if settings.enable_multi_step_tasks:
            raw = llm_steps(user_request, context, settings)
            source = "llm" if raw else "heuristic"
        if not raw:
            raw = heuristic_steps(user_request)
            source = "heuristic"
    return ProposedPlan(title=title or _title(user_request), raw_steps=raw or [],
                        memory_refs=refs, memory_context=context, source=source)
