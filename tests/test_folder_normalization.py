"""Phase 3C.2 — invocation-prefix stripping and deterministic known-folder
resolution. No LLM, GPU, or real launches (open_folder simulates by default)."""

from pathlib import Path

import pytest

from app.core.router import handle_command, strip_invocation_prefix
from app.schemas.commands import CommandRequest, ExecutionStatus, Intent
from app.tools.windows_tools import normalize_known_folder, validate_folder

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- invocation prefix -----------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Fifi abre la carpeta de descargas", "abre la carpeta de descargas"),
        ("Fifi, abre descargas", "abre descargas"),
        ("hey Fifi open my downloads folder", "open my downloads folder"),
        ("oye Fifi abre documentos", "abre documentos"),
        ("open the downloads folder", "open the downloads folder"),  # no prefix
    ],
)
def test_strip_invocation_prefix(settings, text, expected):
    assert strip_invocation_prefix(text) == expected


def test_prefix_only_stripped_at_start(settings):
    # "Fifi" mid-sentence is preserved (e.g. searching for the word).
    assert strip_invocation_prefix("busca en internet quien es Fifi") == (
        "busca en internet quien es Fifi"
    )


# --- known-folder normalization --------------------------------------------------


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("descargas", "Downloads"),
        ("carpeta de descargas", "Downloads"),
        ("mi carpeta de descargas", "Downloads"),
        ("de descargas", "Downloads"),  # what the rule router extracts
        ("documentos", "Documents"),
        ("carpeta de documentos", "Documents"),
        ("escritorio", "Desktop"),
        ("downloads", "Downloads"),
        ("downloads folder", "Downloads"),
        ("my downloads folder", "Downloads"),
        ("documents", "Documents"),
        ("documents folder", "Documents"),
        ("desktop", "Desktop"),
    ],
)
def test_known_folder_aliases(phrase, expected):
    assert normalize_known_folder(phrase) == expected


def test_arbitrary_text_is_not_a_known_folder():
    assert normalize_known_folder("facturas pendientes") is None
    assert normalize_known_folder("de la mi") is None  # only filler -> nothing
    assert normalize_known_folder("downloads documents") is None  # ambiguous


# --- validate_folder safety ------------------------------------------------------


def test_alias_resolves_under_home_not_repo():
    path, error = validate_folder("carpeta de descargas")
    if error is not None:
        pytest.skip("no Downloads folder on this machine")
    assert path == (Path.home() / "Downloads").resolve()
    # Crucially, NOT under the repository working directory.
    assert REPO_ROOT not in path.parents
    assert path != (REPO_ROOT / "de descargas")


def test_arbitrary_text_never_becomes_a_relative_path():
    path, error = validate_folder("facturas pendientes")
    assert path is None  # not turned into REPO_ROOT/"facturas pendientes"
    assert error is not None and "not sure which folder" in error.lower()


def test_no_alias_resolves_under_repository_dir():
    for alias in ("descargas", "documentos", "escritorio", "downloads", "documents", "desktop"):
        path, error = validate_folder(alias)
        if error is not None:
            continue  # folder may not exist on this machine
        assert REPO_ROOT not in path.parents
        assert str(path).startswith(str(Path.home().resolve()))


# --- end-to-end through the router (planner off in tests) ------------------------


@pytest.mark.skipif(
    not (Path.home() / "Downloads").is_dir(), reason="no Downloads folder"
)
def test_fifi_abre_descargas_resolves_to_home_downloads(settings, launches):
    response = handle_command(CommandRequest(text="Fifi abre la carpeta de descargas"))
    assert response.intent is Intent.OPEN_FOLDER
    assert response.status is ExecutionStatus.SIMULATED
    downloads = str((Path.home() / "Downloads").resolve())
    assert downloads in response.result["would_do"]
    # Original transcription preserved in the response (and thus history).
    assert response.input_text == "Fifi abre la carpeta de descargas"


def test_arbitrary_folder_request_is_rejected_not_cwd(settings, launches):
    response = handle_command(CommandRequest(text="abre la carpeta de facturas pendientes"))
    assert response.intent is Intent.OPEN_FOLDER
    assert response.status is ExecutionStatus.REJECTED
    # Rejected with a clarification, never resolved under the repo/cwd.
    assert response.result and "error" in response.result
    assert str(REPO_ROOT) not in response.result["error"]