"""Deterministic plan validation (Phase 5B, item 3).

An LLM (or any caller) only PROPOSES a plan; nothing runs until this module has
proven, step by step, that:

- every intent is in the supported service registry (no arbitrary tools),
- arguments match the existing service schemas (no Python/shell/selectors/unknown
  URLs sneaking in as arguments),
- the dependency graph is acyclic and references only earlier, real steps,
- the plan is non-empty and within ``TASK_MAX_STEPS``,

and then classifies every step as read-only / local / external effect. A single
bad step blocks the whole plan — a partial run of an unvalidated plan is never
attempted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.core import conversation, errors
from app.tasks import models
from app.tasks.models import Step

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


@dataclass
class BlockedStep:
    position: int
    intent: str
    code: str
    reason: str


@dataclass
class PlanValidation:
    ok: bool
    steps: list[Step] = field(default_factory=list)
    blocked: list[BlockedStep] = field(default_factory=list)
    error_code: str | None = None
    reason: str = ""


_ALLOWED_URL_SCHEMES = ("http", "https")


def _validate_url(intent: str, args: dict) -> str | None:
    """Reject unknown/unsafe URLs at plan time (the browser service re-checks at
    dispatch). Only browser_open carries a URL; an empty URL just opens a browser."""
    if intent != "browser_open":
        return None
    url = (args.get("url") or "").strip()
    if not url:
        return None
    # A bare host ("example.com") is fine; anything with an explicit scheme must be
    # http/https — file:, javascript:, data:, about:, etc. are rejected.
    has_scheme = bool(_SCHEME_RE.match(url))
    parsed = urlparse(url if has_scheme else "https://" + url)
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        return f"unsupported URL scheme {parsed.scheme!r}"
    if not parsed.netloc:
        return "URL has no host"
    return None


def _validate_dependencies(position: int, deps, step_count: int) -> str | None:
    """Dependencies form a DAG (prerequisite → step). Any other valid step index is
    allowed; a true cycle is caught separately by ``_has_cycle``."""
    for dep in deps:
        if not isinstance(dep, int):
            return f"dependency {dep!r} is not a step index"
        if dep == position:
            return "a step cannot depend on itself"
        if dep < 0 or dep >= step_count:
            return f"dependency {dep} is out of range"
    return None


def _has_cycle(steps: list[Step]) -> bool:
    """Kahn's algorithm over the dependency DAG (deps point to prerequisites)."""
    indeg = {s.position: 0 for s in steps}
    adj: dict[int, list[int]] = {s.position: [] for s in steps}
    for s in steps:
        for dep in s.dependencies:
            if dep in adj:
                adj[dep].append(s.position)
                indeg[s.position] += 1
    queue = [p for p, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return seen != len(steps)


def validate_plan(raw_steps: list[dict], *, task_id: str, max_steps: int,
                  repo=None) -> PlanValidation:
    """Validate a proposed plan. ``raw_steps`` is a list of {intent, arguments?,
    dependencies?}. Returns validated Step objects (unsaved) or a blocked result."""
    if not raw_steps:
        return PlanValidation(ok=False, error_code=errors.EMPTY_PLAN,
                              reason="the plan has no steps")
    if len(raw_steps) > max_steps:
        return PlanValidation(ok=False, error_code=errors.PLAN_TOO_LARGE,
                              reason=f"plan has {len(raw_steps)} steps (max {max_steps})")

    steps: list[Step] = []
    blocked: list[BlockedStep] = []
    step_count = len(raw_steps)

    for position, raw in enumerate(raw_steps):
        intent = str(raw.get("intent", "")).strip()
        args = raw.get("arguments") or raw.get("args") or {}
        deps = raw.get("dependencies") or raw.get("depends_on") or []
        if not isinstance(args, dict):
            blocked.append(BlockedStep(position, intent, errors.INVALID_STEP_ARGS,
                                       "arguments must be an object"))
            continue

        if not models.is_supported(intent):
            blocked.append(BlockedStep(position, intent or "(missing)", errors.UNSUPPORTED_STEP,
                                       "intent is not a supported task step"))
            continue

        # Arguments must match the existing service schema (rejects arbitrary keys,
        # e.g. raw selectors, shell/python payloads, unexpected fields).
        arg_error = conversation.validate_service_args(intent, args)
        if arg_error:
            blocked.append(BlockedStep(position, intent, errors.INVALID_STEP_ARGS, arg_error))
            continue

        url_error = _validate_url(intent, args)
        if url_error:
            blocked.append(BlockedStep(position, intent, errors.INVALID_STEP_ARGS, url_error))
            continue

        dep_error = _validate_dependencies(position, deps, step_count)
        if dep_error:
            blocked.append(BlockedStep(position, intent, errors.INVALID_DEPENDENCY, dep_error))
            continue

        step = Step(
            step_id="", task_id=task_id, position=position, intent=intent,
            arguments=dict(args), domain=models.domain_of(intent),
            risk_level=models.risk_of(intent), dependencies=[int(d) for d in deps],
            requires_confirmation=models.requires_confirmation(intent),
            confirmation_phrase=models.expected_confirmation_phrase(intent, args),
        )
        steps.append(step)

    if blocked:
        return PlanValidation(ok=False, steps=steps, blocked=blocked,
                              error_code=blocked[0].code,
                              reason="the plan contains steps I can't run safely")

    if _has_cycle(steps):
        return PlanValidation(ok=False, steps=steps, error_code=errors.PLAN_HAS_CYCLE,
                              reason="the plan has a dependency cycle")

    return PlanValidation(ok=True, steps=steps)
