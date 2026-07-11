"""Voice API endpoints.

Voice never bypasses safety: /voice/command feeds the transcription into the
exact same handle_command() pipeline as POST /command — router/planner,
safety layer, tool registry, command log.

All endpoints answer {"status": "disabled", ...} while ENABLE_VOICE=false.
"""

import os
import tempfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.router import handle_command
from app.schemas.commands import CommandRequest
from app.voice.stt import get_stt_service
from app.voice.tts import TextToSpeechService

router = APIRouter(prefix="/voice", tags=["voice"])

_DISABLED: dict[str, str] = {
    "status": "disabled",
    "message": "Voice is disabled. Set ENABLE_VOICE=true in .env and restart the server.",
}


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)


async def _save_upload(file: UploadFile) -> str:
    """Persist an uploaded .wav to a temp file; caller must delete it."""
    if not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav uploads are accepted.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(data)
    finally:
        tmp.close()
    return tmp.name


def _transcribe_upload_result(path: str) -> dict[str, Any]:
    result = get_stt_service().transcribe_file(path)
    if result.get("error"):
        status = "unavailable" if "not installed" in result["error"] else "error"
        return {"status": status, "message": result["error"]}
    return {"status": "ok", **result}


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, Any]:
    """Transcribe an uploaded .wav file (Spanish and English)."""
    if not get_settings().enable_voice:
        return _DISABLED
    path = await _save_upload(file)
    try:
        return _transcribe_upload_result(path)
    finally:
        os.unlink(path)


@router.post("/speak")
def speak(request: SpeakRequest) -> dict[str, Any]:
    """Speak text via the configured TTS engine (simulated if unavailable)."""
    if not get_settings().enable_voice:
        return _DISABLED
    result = TextToSpeechService().speak(request.text)
    if result.get("error"):
        return {"status": "error", "message": result["error"]}
    return {"status": "ok", **result}


@router.post("/command")
async def voice_command(
    file: UploadFile = File(...),
    confirm: bool = Query(
        default=False,
        description="Approve a sensitive action, same as /command's confirm field",
    ),
) -> dict[str, Any]:
    """Transcribe a .wav and run it through the normal /command pipeline."""
    if not get_settings().enable_voice:
        return _DISABLED

    path = await _save_upload(file)
    try:
        transcription = _transcribe_upload_result(path)
    finally:
        os.unlink(path)
    if transcription["status"] != "ok":
        return transcription

    text = transcription["text"].strip()
    if not text:
        return {"status": "error", "message": "Transcription was empty — nothing to run."}

    response = handle_command(
        CommandRequest(text=text, confirm=confirm, language=transcription.get("language"))
    )
    payload: dict[str, Any] = {
        "status": "ok",
        "transcription": transcription["text"],
        "language": transcription.get("language"),
        "assistant_message": response.assistant_message,
        "command": response,
    }
    if get_settings().voice_speak_command_response:
        payload["speech"] = _speak_assistant_message(response.assistant_message)
    return payload


def _speak_assistant_message(message: str) -> dict[str, Any]:
    """Speak the reply out loud; report clearly when TTS is unavailable."""
    result = TextToSpeechService().speak(message)
    if result.get("error"):
        return {"status": "error", "message": result["error"]}
    if result.get("simulated"):
        return {"status": "simulated", "note": result.get("note", "TTS unavailable.")}
    return {"status": "spoken", "engine": result.get("engine")}
