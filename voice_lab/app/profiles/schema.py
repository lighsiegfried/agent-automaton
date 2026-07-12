"""Voice profile schema — the shared contract between voice_lab and Fifi.

Profiles are plain JSON files under config/voices/profiles/ so both sides
(and the user) can read them. They contain NO secrets and NO absolute paths:
`model` is always a model id / short name, never a filesystem location.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

KNOWN_ENGINES = ("windows_sapi", "kokoro", "qwen3_tts")


class VoiceProfile(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9_\-]+$")
    engine: Literal["windows_sapi", "kokoro", "qwen3_tts"]
    model: str = ""  # model id or engine-specific short name (never a path)
    language: str = "en"
    speaker: str = ""  # engine voice id (e.g. kokoro "af_heart", SAPI voice name)
    speed: float = Field(default=1.0, gt=0.25, lt=4.0)
    style_instruction: str = ""  # natural-language style hint (qwen3_tts)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    device: str = "cuda"  # "cuda" | "cpu" (engines fall back safely)
    dtype: str = "float16"
    fallback_engine: Literal["windows_sapi", "kokoro", "qwen3_tts", ""] = "windows_sapi"
    provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("model", "speaker", "style_instruction")
    @classmethod
    def _no_paths(cls, value: str) -> str:
        # Profiles are shared config: never smuggle filesystem paths through them.
        if ":\\" in value or value.startswith(("/", "\\")):
            raise ValueError("profile fields must not contain filesystem paths")
        return value

    def public_dict(self) -> dict:
        """The profile exactly as stored — safe to expose (no paths/secrets)."""
        return self.model_dump()
