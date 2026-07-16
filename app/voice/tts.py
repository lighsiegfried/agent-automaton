"""Text-to-speech.

Engine selected via TTS_ENGINE:
- "voice_lab": neural voices from the isolated Voice Lab worker (Phase 3D.0).
  The main process NEVER loads TTS models — it POSTs text to the loopback
  worker (VOICE_LAB_URL), which speaks with the profile from
  config/voices/active.json. If the worker is unavailable and
  VOICE_LAB_ALLOW_FALLBACK=true, speech falls back to the Windows engine.
- "windows": Windows SAPI through pyttsx3 (optional dependency from
  requirements-voice.txt). Falls back to simulation when not installed.
- "simulated": always simulate.

TTS never blocks or breaks the API: with ENABLE_VOICE=false, a missing
dependency, a dead worker, or an engine failure, speak() returns a structured
result instead of raising. Callers on the event loop must run speak() in the
threadpool (app/voice/api.py does).
"""

import json
from typing import Any

import httpx

from app.config import PROJECT_ROOT, get_settings
from app.core.logger import get_logger

log = get_logger(__name__)

try:
    import pyttsx3
except ImportError:  # optional dependency
    pyttsx3 = None

INSTALL_HINT = (
    "pyttsx3 is not installed. Install the optional voice dependencies: "
    "pip install -r requirements-voice.txt"
)

# Shared integration file, written ATOMICALLY by the Voice Lab (never by us).
ACTIVE_VOICE_PATH = PROJECT_ROOT / "config" / "voices" / "active.json"

# Hot reload (Phase 3D.0.1): the active profile is re-read whenever the file's
# mtime changes, so `voice_lab.py select` takes effect on the NEXT spoken reply
# — no restart. updated_utc from the file is kept as change metadata.
_active_cache: dict[str, Any] = {"mtime": None, "profile": None, "updated_utc": None}


def active_voice_info() -> dict[str, Any]:
    """{"profile", "updated_utc"} from active.json, mtime-cached; hot-reloads."""
    try:
        mtime = ACTIVE_VOICE_PATH.stat().st_mtime_ns
    except OSError:
        _active_cache.update(mtime=None, profile=None, updated_utc=None)
        return {"profile": None, "updated_utc": None}
    if mtime != _active_cache["mtime"]:
        try:
            data = json.loads(ACTIVE_VOICE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        name = data.get("profile") if isinstance(data, dict) else None
        name = name if isinstance(name, str) and name else None
        updated = data.get("updated_utc") if isinstance(data, dict) else None
        if name != _active_cache["profile"]:
            log.info(
                "active voice profile changed: %r -> %r (updated_utc=%s)",
                _active_cache["profile"], name, updated,
            )
        _active_cache.update(mtime=mtime, profile=name, updated_utc=updated)
    return {"profile": _active_cache["profile"], "updated_utc": _active_cache["updated_utc"]}


def active_voice_profile() -> str | None:
    """Profile name from config/voices/active.json, or None when unreadable."""
    return active_voice_info()["profile"]


# Shared profile store (written by the Voice Lab; read-only here). We only need
# the engine/model to know whether the active voice is a heavy VoiceDesign one —
# the main process never loads a neural model.
_PROFILES_DIR = PROJECT_ROOT / "config" / "voices" / "profiles"


def profile_engine_info(name: str | None) -> dict[str, Any]:
    """{engine, model, is_voice_design} for a profile, from its shared JSON.

    Never raises; an unknown/unreadable profile returns engine=None. Used by the
    wake TTS guard to keep VoiceDesign out of hands-free replies without importing
    anything from the isolated Voice Lab package.
    """
    info: dict[str, Any] = {"engine": None, "model": "", "is_voice_design": False}
    if not name:
        return info
    try:
        data = json.loads((_PROFILES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return info
    engine = data.get("engine")
    model = data.get("model")
    info["engine"] = engine if isinstance(engine, str) else None
    info["model"] = model if isinstance(model, str) else ""
    info["is_voice_design"] = (
        info["engine"] == "qwen3_tts" and "voicedesign" in info["model"].lower()
    )
    return info


class TextToSpeechService:
    def __init__(self, engine: str | None = None) -> None:
        self.engine = (engine or get_settings().tts_engine).lower()

    def available(self) -> bool:
        if self.engine == "windows":
            return pyttsx3 is not None
        return self.engine == "voice_lab"  # worker liveness is checked per call

    def _simulate(self, text: str, note: str) -> dict[str, Any]:
        return {
            "simulated": True,
            "action": "speak",
            "would_do": f"Speak with {self.engine}: {text!r}",
            "note": note,
        }

    def speak(self, text: str, wake: bool = False) -> dict[str, Any]:
        """Speak `text`. In wake mode (Phase 3D.1) the Voice Lab path applies the
        daily latency guard: a shorter budget, Qwen->Kokoro fallback, and never
        VoiceDesign — the command already ran, so the reply must not stall."""
        text = (text or "").strip()
        if not text:
            return {"action": "speak", "error": "Empty text."}

        if not get_settings().enable_voice:
            return self._simulate(text, "Voice is disabled (ENABLE_VOICE=false).")
        if self.engine == "voice_lab":
            return self._speak_voice_lab(text, wake=wake)
        if self.engine != "windows":
            return self._simulate(text, f"TTS engine {self.engine!r} has no real backend yet.")
        return self._speak_windows(text)

    # -- Windows SAPI ----------------------------------------------------------------

    def _speak_windows(self, text: str) -> dict[str, Any]:
        if pyttsx3 is None:
            return self._simulate(text, INSTALL_HINT)
        try:
            driver = pyttsx3.init()
            driver.say(text)
            driver.runAndWait()
        except Exception as exc:  # audio failures must not crash the API
            log.exception("TTS failed")
            return {"action": "speak", "error": f"TTS failed: {exc}"}
        return {"simulated": False, "action": "speak", "spoke": text, "engine": "windows"}

    # -- Voice Lab worker (Phase 3D.0) -------------------------------------------------

    def _speak_voice_lab(self, text: str, wake: bool = False) -> dict[str, Any]:
        """Ask the isolated worker to synthesize + play. No models load here.

        Wake mode (Phase 3D.1) enforces the daily latency guard:
        - a shorter budget (WAKE_MAX_TTS_WAIT_SECONDS) so a slow/cold Qwen voice
          never keeps the user waiting;
        - if the active voice can't answer in time, fall back to KOKORO (never
          Windows), reporting the requested vs actual engine;
        - a VoiceDesign active voice is never loaded — the reply uses Kokoro.
        Outside wake mode the behavior is unchanged (worker timeout -> Windows).
        """
        settings = get_settings()
        active = active_voice_profile() or settings.voice_profile
        info = profile_engine_info(active)
        requested_engine = info["engine"] or "voice_lab"
        kokoro_profile = settings.voice_profile
        timeout = (
            settings.wake_max_tts_wait_seconds if wake else settings.voice_lab_timeout_seconds
        )

        # Wake mode NEVER loads VoiceDesign — speak with the Kokoro voice instead.
        if wake and info["is_voice_design"]:
            result, status, reason = self._post_synthesize(kokoro_profile, text, timeout)
            if status == "ok":
                return self._voice_lab_payload(
                    text, result, requested_engine="voice_design", fallback_used=True,
                    fallback_reason="VoiceDesign no se carga en modo wake — voz Kokoro",
                )
            return self._voice_lab_fallback(
                text, f"VoiceDesign bloqueado en wake y Kokoro falló: {reason}"
            )

        result, status, reason = self._post_synthesize(active, text, timeout)
        if status == "ok":
            return self._voice_lab_payload(text, result, requested_engine=requested_engine)

        # Wake latency fallback: the active (Qwen) voice was too slow -> Kokoro,
        # NEVER Windows here. The command already executed; only the voice changes.
        if wake and status == "timeout" and active != kokoro_profile:
            result, kstatus, kreason = self._post_synthesize(
                kokoro_profile, text, settings.wake_max_tts_wait_seconds
            )
            if kstatus == "ok":
                return self._voice_lab_payload(
                    text, result, requested_engine=requested_engine, fallback_used=True,
                    fallback_reason=(
                        f"{requested_engine} tardó más de {timeout:g}s — voz Kokoro"
                    ),
                )
            reason = kreason  # Kokoro also failed -> last-resort Windows below

        return self._voice_lab_fallback(text, reason)

    def _post_synthesize(
        self, profile: str, text: str, timeout: float
    ) -> tuple[dict[str, Any] | None, str, str]:
        """POST /synthesize for one profile. Returns (result, status, reason)
        where status is 'ok' | 'timeout' | 'error'. Never raises."""
        settings = get_settings()
        try:
            response = httpx.post(
                f"{settings.voice_lab_url}/synthesize",
                json={"text": text, "profile": profile, "play": True},
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            return None, "error", f"Voice Lab worker unavailable ({type(exc).__name__})"
        except httpx.TimeoutException as exc:
            # Up, but synthesis took too long (e.g. a cold Qwen load).
            return None, "timeout", (
                f"Voice Lab synthesis timed out after {timeout:g}s ({type(exc).__name__})"
            )
        except (httpx.HTTPError, ValueError) as exc:
            return None, "error", f"Voice Lab request failed ({type(exc).__name__})"
        if result.get("status") != "ok":
            return None, "error", result.get("message") or "Voice Lab synthesis failed"
        return result, "ok", ""

    def _voice_lab_payload(
        self,
        text: str,
        result: dict[str, Any],
        *,
        requested_engine: str,
        fallback_used: bool | None = None,
        fallback_reason: str = "",
    ) -> dict[str, Any]:
        actual = result.get("engine")  # the engine that ACTUALLY produced audio
        payload: dict[str, Any] = {
            "simulated": False,
            "action": "speak",
            "spoke": text,
            "engine": "voice_lab",
            "profile": result.get("profile"),
            "requested_engine": requested_engine,
            "actual_engine": actual,
            "synthesis_engine": actual,  # legacy key
            "fallback_used": (
                bool(result.get("fallback_used")) if fallback_used is None else fallback_used
            ),
        }
        if fallback_reason:
            payload["fallback_reason"] = fallback_reason
        return payload

    def _voice_lab_fallback(self, text: str, reason: str) -> dict[str, Any]:
        """Worker down/broken: degrade to Windows TTS when allowed, never raise."""
        if not get_settings().voice_lab_allow_fallback:
            log.warning("voice_lab TTS failed and fallback is disabled: %s", reason)
            return {"action": "speak", "error": f"Voice Lab TTS failed: {reason}"}
        log.warning("voice_lab TTS failed (%s) — falling back to Windows TTS", reason)
        result = self._speak_windows(text)
        return {**result, "fallback_from": "voice_lab", "fallback_reason": reason}
