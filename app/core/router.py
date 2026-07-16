"""Command router: intent detection + dispatch through the safety layer.

Two planners feed the same execution path:
- rule-based bilingual keyword rules (default, and the fallback)
- LLM planner (ENABLE_LLM_PLANNER=true): proposes a plan that is strictly
  validated in app/llm/command_planner.py before it gets here.

Whatever the source, every tool call passes through safety.evaluate() — the
LLM can tighten safety (request confirmation for a safe action) but never
loosen it.
"""

import re

from app.config import get_settings
from app.core import conversation, memory, nl, safety
from app.core.logger import get_logger
from app.llm.command_planner import CommandPlan, plan_command
from app.llm.response_generator import generate_assistant_message, resolve_language
from app.schemas.commands import (
    CommandRequest,
    CommandResponse,
    ExecutionStatus,
    Intent,
    PlanInfo,
    PlannerSource,
    SafetyLevel,
)

# Normalized dispatcher status -> the legacy ExecutionStatus for CommandResponse.
_SERVICE_STATUS_MAP: dict[str, ExecutionStatus] = {
    "needs_confirmation": ExecutionStatus.NEEDS_CONFIRMATION,
    "executed": ExecutionStatus.EXECUTED,
    "filled": ExecutionStatus.EXECUTED,
    "ok": ExecutionStatus.EXECUTED,
    "simulated": ExecutionStatus.SIMULATED,
    "cancelled": ExecutionStatus.EXECUTED,
    "rejected": ExecutionStatus.REJECTED,
    "error": ExecutionStatus.REJECTED,
}
from app.tools.registry import INTENT_TOOL_MAP, Tool, registry

log = get_logger(__name__)

# Optional leading invocation forms: "Fifi", "Fifi,", "hey Fifi", "oye Fifi".
# Stripped for routing only — never from the middle of text, and the original
# transcription is preserved in the response and command log.
_GREETING = r"(?:hey|hi|hello|ok|okay|oye|hola)\s+"


def strip_invocation_prefix(text: str) -> str:
    """Remove an optional leading assistant invocation ("Fifi", "hey Fifi", …).

    Anchored at the start only, so "Fifi" appearing mid-sentence is untouched.
    Returns the original text if stripping would leave nothing.
    """
    name = re.escape(get_settings().agent_name)
    pattern = re.compile(rf"^\s*(?:{_GREETING})?{name}\b[\s,]*", re.IGNORECASE)
    stripped = pattern.sub("", text, count=1).strip()
    return stripped or text


_ARTICLE = r"(?:the\s+|my\s+|la\s+|el\s+|los\s+|las\s+|mi\s+|un\s+|una\s+)?"

# Checked in order; the first match wins. Capture group 1 becomes the tool
# parameter, sliced from the original text so casing is preserved.
_RULES: list[tuple[re.Pattern[str], Intent, str]] = [
    (
        re.compile(
            rf"\b(?:open|show|abre|abrir|muestra)\s+{_ARTICLE}(?:folder|carpeta)\s*(.*)",
            re.IGNORECASE,
        ),
        Intent.OPEN_FOLDER,
        "path",
    ),
    (
        re.compile(
            r"\b(?:search(?:\s+the\s+web)?(?:\s+for)?|google|look\s+up"
            r"|busca(?:r)?(?:\s+en\s+(?:internet|la\s+web|google))?)\s+(.+)",
            re.IGNORECASE,
        ),
        Intent.SEARCH_WEB,
        "query",
    ),
    (
        re.compile(
            r"\b(?:say|speak|read\s+aloud|di|habla|lee\s+en\s+voz\s+alta)\s+(.+)",
            re.IGNORECASE,
        ),
        Intent.SPEAK,
        "text",
    ),
    (
        re.compile(r"\b(?:type|write|escribe)\s+(.+)", re.IGNORECASE),
        Intent.TYPE_TEXT,
        "text",
    ),
    (
        re.compile(
            rf"\b(?:open|launch|start|run|abre|abrir|inicia|ejecuta|lanza)\s+"
            rf"{_ARTICLE}(?:app(?:lication)?\s+|aplicaci[oó]n\s+)?(.+)",
            re.IGNORECASE,
        ),
        Intent.OPEN_APP,
        "app",
    ),
]


def detect_intent(text: str) -> tuple[Intent, dict[str, str]]:
    from app.tools.windows_tools import FOLDER_ALIASES

    for pattern, intent, param in _RULES:
        match = pattern.search(text)
        if not match:
            continue
        value = text[match.start(1) : match.end(1)].strip().strip('"')
        # "abre descargas" names a known folder, not an app.
        if intent is Intent.OPEN_APP and value.lower() in FOLDER_ALIASES:
            return Intent.OPEN_FOLDER, {"path": value}
        return intent, {param: value}
    return Intent.UNKNOWN, {}


def _unknown_response(request: CommandRequest) -> CommandResponse:
    return CommandResponse(
        input_text=request.text,
        intent=Intent.UNKNOWN,
        status=ExecutionStatus.NOT_HANDLED,
        message=(
            "Could not detect an intent. Try e.g. 'open folder <path>', "
            "'open app <name>', 'search for <query>', 'say <text>' — "
            "or in Spanish: 'abre la carpeta <ruta>', 'abre <app>', "
            "'busca en internet <consulta>', 'di <texto>'."
        ),
    )


def _execute(
    request: CommandRequest,
    intent: Intent,
    params: dict[str, str],
    tool: Tool,
    tighten: bool = False,
) -> CommandResponse:
    """Safety-check and run one tool. `tighten` may only raise the bar:
    a safe tool is treated as sensitive when the LLM asked for confirmation."""
    effective_level = tool.safety_level
    if tighten and effective_level is SafetyLevel.SAFE:
        effective_level = SafetyLevel.SENSITIVE

    decision = safety.evaluate(effective_level, request.confirm)
    if not decision.allowed:
        return CommandResponse(
            input_text=request.text,
            intent=intent,
            tool=tool.name,
            safety_level=tool.safety_level,
            status=decision.status,
            message=decision.reason,
        )

    result = tool.handler(**params)
    if result.get("error"):
        status = ExecutionStatus.REJECTED
        message = str(result["error"])
    elif result.get("simulated", True):
        status = ExecutionStatus.SIMULATED
        message = f"Simulated '{tool.name}'. {decision.reason}"
    else:
        status = ExecutionStatus.EXECUTED
        message = f"Executed '{tool.name}'. {decision.reason}"

    try:  # additive Activity Center hook (Phase 5D) — safe fields only
        from app.core import eventbus

        eventbus.emit(domain="command", event_type=intent.value, status=status.value,
                      metadata={"intent": intent.value})
    except Exception:
        pass

    return CommandResponse(
        input_text=request.text,
        intent=intent,
        tool=tool.name,
        safety_level=tool.safety_level,
        status=status,
        message=message,
        result=result,
    )


def _dispatch_rules(request: CommandRequest, routing_text: str | None = None) -> CommandResponse:
    intent, params = detect_intent(routing_text if routing_text is not None else request.text)
    log.info("rule intent=%s params=%s", intent.value, params)

    if intent is Intent.UNKNOWN:
        return _unknown_response(request)

    tool = registry.get(INTENT_TOOL_MAP[intent])
    if tool is None:  # should not happen once load_tools() ran
        response = _unknown_response(request)
        response.intent = intent
        response.message = f"No tool registered for intent '{intent.value}'."
        return response
    return _execute(request, intent, params, tool)


def _dispatch_plan(request: CommandRequest, plan: CommandPlan) -> CommandResponse:
    # A validated service-intent plan goes to the deterministic dispatcher.
    if plan.intent.value in conversation.SERVICE_INTENT_ARGS:
        command = nl.ServiceCommand(
            intent=plan.intent.value, arguments=dict(plan.arguments),
            is_confirmation=(plan.intent.value == "confirm"),
            confirmation_phrase=str(plan.arguments.get("phrase", "")),
        )
        response = _dispatch_service(request, command)
        response.plan = PlanInfo(confidence=plan.confidence,
                                 reasoning_summary=plan.reasoning_summary, language=plan.language)
        return response

    tool = registry.get(plan.tool_name)  # validated to exist and be non-destructive
    assert tool is not None
    response = _execute(
        request, plan.intent, plan.arguments, tool, tighten=plan.requires_confirmation
    )
    response.plan = PlanInfo(
        confidence=plan.confidence,
        reasoning_summary=plan.reasoning_summary,
        language=plan.language,
    )
    return response


def _service_language(request: CommandRequest, routing_text: str) -> str:
    if request.language:
        return "en" if request.language.strip().lower().startswith("en") else "es"
    # Reuse the response generator's language heuristic on the routing text.
    probe = CommandResponse(input_text=routing_text, intent=Intent.UNKNOWN,
                            status=ExecutionStatus.NOT_HANDLED, message="")
    return resolve_language(probe)


def _dispatch_service(request: CommandRequest, command: "nl.ServiceCommand") -> CommandResponse:
    """Route a conversational service command through the deterministic dispatcher."""
    language = _service_language(request, request.text)
    result = conversation.dispatch(command, language=language)
    try:
        intent_enum = Intent(command.intent)
    except ValueError:
        intent_enum = Intent.UNKNOWN
    response = CommandResponse(
        input_text=request.text,
        intent=intent_enum,
        status=_SERVICE_STATUS_MAP.get(result["status"], ExecutionStatus.REJECTED),
        message=result["spoken"],
        result={k: v for k, v in result.items() if k != "spoken"},
    )
    response.assistant_message = result["spoken"]   # already safe + localized
    return response


def handle_command(request: CommandRequest) -> CommandResponse:
    # Strip a leading "Fifi"/"hey Fifi" for routing only; request.text (the
    # original transcription) is preserved for the response and the command log.
    routing_text = strip_invocation_prefix(request.text)

    # Conversational service intents (text/browser drafting, confirmation, cancel)
    # are handled by the deterministic dispatcher BEFORE the tool router, so the
    # legacy tool behaviour ("busca X", "abre <app>") is unchanged.
    service_command = nl.parse(routing_text)
    if service_command is not None:
        log.info("service intent=%s", service_command.intent)
        response = _dispatch_service(request, service_command)
        memory.log_command(
            input_text=request.text, intent=response.intent.value, tool=None,
            status=response.status.value, result=response.result,
        )
        return response

    if get_settings().enable_llm_planner:
        outcome = plan_command(routing_text)
        if outcome.plan is not None:
            log.info(
                "llm plan: tool=%s confidence=%.2f", outcome.plan.tool_name, outcome.plan.confidence
            )
            response = _dispatch_plan(request, outcome.plan)
            response.planner = PlannerSource.LLM_PLANNER
        else:
            log.info("llm planner rejected (%s); falling back to rules", outcome.failure)
            response = _dispatch_rules(request, routing_text)
            response.planner = PlannerSource.FALLBACK_ROUTER
    else:
        response = _dispatch_rules(request, routing_text)
        response.planner = PlannerSource.RULE_ROUTER

    # Service dispatches already produced a safe, localized spoken line; only the
    # tool path needs the template/LLM response generator.
    if not response.assistant_message:
        response.assistant_message = generate_assistant_message(
            response, language=request.language
        )

    memory.log_command(
        input_text=request.text,
        intent=response.intent.value,
        tool=response.tool,
        status=response.status.value,
        result=response.result,
    )
    return response
