"""Ollama client — talks to a local Ollama server.

Used by the LLM command planner (app/llm/command_planner.py) when
ENABLE_LLM_PLANNER=true. Returns a placeholder result when no server is
reachable so callers can fall back gracefully.
"""

from typing import Any

import httpx

from app.config import get_settings
from app.core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_TIMEOUT = 120.0


class OllamaClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.default_model

    def is_available(self) -> bool:
        """True if a local Ollama server responds."""
        try:
            response = httpx.get(f"{self.base_url}/api/version", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        json_format: bool = False,
    ) -> dict[str, Any]:
        """Single non-streaming completion against a local model.

        Raises httpx.HTTPError on network/timeout failures — callers decide
        how to fall back.
        """
        model = model or self.model
        if not self.is_available():
            return {
                "simulated": True,
                "would_do": f"Ask {model!r} at {self.base_url}: {prompt[:80]!r}...",
                "response": None,
                "note": "Ollama server not reachable; returning placeholder.",
            }
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if json_format:
            payload["format"] = "json"
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=timeout or _DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return {"simulated": False, "response": data.get("response"), "model": model}
