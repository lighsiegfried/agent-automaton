import pytest

from app.core.router import detect_intent, handle_command
from app.schemas.commands import CommandRequest, ExecutionStatus, Intent


@pytest.mark.parametrize(
    ("text", "intent", "params"),
    [
        # English
        ("open folder C:\\Temp", Intent.OPEN_FOLDER, {"path": "C:\\Temp"}),
        ("open downloads", Intent.OPEN_FOLDER, {"path": "downloads"}),
        ("open app notepad", Intent.OPEN_APP, {"app": "notepad"}),
        ("launch chrome", Intent.OPEN_APP, {"app": "chrome"}),
        (
            "search the web for local voice models",
            Intent.SEARCH_WEB,
            {"query": "local voice models"},
        ),
        ("google rust tutorials", Intent.SEARCH_WEB, {"query": "rust tutorials"}),
        ("say hello world", Intent.SPEAK, {"text": "hello world"}),
        ("type hello", Intent.TYPE_TEXT, {"text": "hello"}),
        # Spanish
        ("abre descargas", Intent.OPEN_FOLDER, {"path": "descargas"}),
        ("abre la carpeta documentos", Intent.OPEN_FOLDER, {"path": "documentos"}),
        ("abre notepad", Intent.OPEN_APP, {"app": "notepad"}),
        ("abrir calculadora", Intent.OPEN_APP, {"app": "calculadora"}),
        (
            "busca en internet modelos locales de voz",
            Intent.SEARCH_WEB,
            {"query": "modelos locales de voz"},
        ),
        ("di hola", Intent.SPEAK, {"text": "hola"}),
        ("escribe hola mundo", Intent.TYPE_TEXT, {"text": "hola mundo"}),
        # Unknown stays unknown
        ("what time is it", Intent.UNKNOWN, {}),
        ("qué hora es", Intent.UNKNOWN, {}),
    ],
)
def test_detect_intent(text, intent, params):
    assert detect_intent(text) == (intent, params)


def test_unknown_command_is_graceful():
    response = handle_command(CommandRequest(text="hazme un sandwich"))
    assert response.status is ExecutionStatus.NOT_HANDLED
    assert response.intent is Intent.UNKNOWN
    assert "abre" in response.message  # hint covers Spanish too


def test_rejected_tool_input_surfaces_as_rejected(settings):
    settings.require_confirmation = True
    response = handle_command(CommandRequest(text="open app photoshop", confirm=True))
    assert response.status is ExecutionStatus.REJECTED
    assert "allowlist" in response.message
