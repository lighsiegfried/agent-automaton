"""Assistant response generator: short natural replies for command outcomes.

Deterministic bilingual templates by default. With
ENABLE_RESPONSE_GENERATOR=true, Ollama phrases the reply instead — but the
model only rephrases an already-decided outcome (it cannot change what
happened), its output is length-capped, and any failure falls back to the
templates. Chain-of-thought is never requested or exposed.

Language: detected command language when available (voice/planner), else a
small keyword heuristic, else Spanish.
"""

import re

import httpx

from app.config import get_settings
from app.core.logger import get_logger
from app.llm.ollama_client import OllamaClient
from app.llm.prompts import RESPONSE_SYSTEM_PROMPT, build_response_prompt
from app.schemas.commands import CommandResponse, ExecutionStatus

log = get_logger(__name__)

_MAX_MESSAGE_LENGTH = 300

_SPANISH_HINTS = re.compile(
    r"[áéíóúñ¿¡]|\b(?:abre|abrir|busca|buscar|carpeta|escribe|descargas|documentos"
    r"|calculadora|internet|hola|di|habla|muestra|ejecuta|lanza)\b",
    re.IGNORECASE,
)
_ENGLISH_HINTS = re.compile(
    r"\b(?:open|search|folder|type|say|speak|launch|start|show|the|web|downloads"
    r"|documents|calculator)\b",
    re.IGNORECASE,
)

_TOOL_LABELS: dict[str, tuple[str, str]] = {
    "open_folder": ("the folder", "la carpeta"),
    "open_app": ("the app", "la aplicación"),
    "search_web": ("the web search", "la búsqueda en internet"),
    "type_text": ("the text input", "la escritura de texto"),
    "speak": ("the spoken message", "el mensaje de voz"),
}

# (english, spanish) per status; {label} = localized tool label,
# {message} = the technical message (e.g. a rejection reason),
# {name} = the agent name (AGENT_NAME, default Fifi) — used sparingly, only
# where introducing the assistant is natural.
_TEMPLATES: dict[ExecutionStatus, tuple[str, str]] = {
    ExecutionStatus.EXECUTED: (
        "Done — I completed {label}.",
        "Listo — completé {label}.",
    ),
    ExecutionStatus.SIMULATED: (
        "Done, I simulated {label} — no real action was taken.",
        "Listo, simulé {label} — no se realizó ninguna acción real.",
    ),
    ExecutionStatus.NEEDS_CONFIRMATION: (
        "I need confirmation before doing that. Please repeat the command "
        "with confirmation to proceed.",
        "Necesito confirmación antes de hacer eso. Repite el comando "
        "con confirmación para continuar.",
    ),
    ExecutionStatus.REJECTED: (
        "I can't do that: {message}",
        "No puedo hacer eso: {message}",
    ),
    ExecutionStatus.BLOCKED: (
        "I won't do that — destructive actions are blocked.",
        "No haré eso — las acciones destructivas están bloqueadas.",
    ),
    ExecutionStatus.NOT_HANDLED: (
        "I'm {name}. I didn't understand that command — what would you like me to do? "
        "Try, for example, 'open folder downloads' or 'search the web for something'.",
        "Soy {name}. No entendí el comando — ¿qué necesitas que haga? "
        "Prueba, por ejemplo, 'abre la carpeta descargas' o 'busca en internet algo'.",
    ),
}


def _guess_language(text: str) -> str | None:
    spanish = len(_SPANISH_HINTS.findall(text))
    english = len(_ENGLISH_HINTS.findall(text))
    if spanish == english == 0:
        return None
    return "es" if spanish >= english else "en"


def resolve_language(response: CommandResponse, override: str | None = None) -> str:
    """'en' or 'es'; unknown languages fall back to Spanish."""
    candidates = [
        override,
        response.plan.language if response.plan else None,
        _guess_language(response.input_text),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate = candidate.strip().lower()
        if candidate.startswith("en"):
            return "en"
        if candidate.startswith("es"):
            return "es"
    return "es"


def _template_message(response: CommandResponse, language: str) -> str:
    english, spanish = _TEMPLATES[response.status]
    template = english if language == "en" else spanish
    labels = _TOOL_LABELS.get(response.tool or "")
    if labels is None:
        labels = ("that action", "esa acción")
    label = labels[0] if language == "en" else labels[1]
    return template.format(
        label=label, message=response.message, name=get_settings().agent_name
    )


def _llm_message(response: CommandResponse, language: str) -> str | None:
    """One short sentence from Ollama, sanitized; None on any failure."""
    settings = get_settings()
    client = OllamaClient()
    try:
        result = client.generate(
            prompt=build_response_prompt(
                command=response.input_text,
                status=response.status.value,
                tool=response.tool,
                message=response.message,
                language=language,
                agent_name=settings.agent_name,
            ),
            system=RESPONSE_SYSTEM_PROMPT,
            model=settings.response_model,
            timeout=settings.response_timeout_seconds,
        )
    except httpx.HTTPError as exc:
        log.warning("response generator request failed: %s", exc)
        return None
    if result.get("simulated") or not result.get("response"):
        return None
    text = " ".join(str(result["response"]).split()).strip().strip('"')
    if not text:
        return None
    return text[:_MAX_MESSAGE_LENGTH]


def generate_assistant_message(
    response: CommandResponse, language: str | None = None
) -> str:
    resolved = resolve_language(response, language)
    if get_settings().enable_response_generator:
        message = _llm_message(response, resolved)
        if message:
            return message
        log.info("response generator unavailable; using template")
    return _template_message(response, resolved)
