"""Compose the text a draft will insert. Drafting/rewriting are always safe.

If the caller supplies explicit ``text`` it is used verbatim (e.g. "type this").
Otherwise a generator composes it from an instruction (draft) or rewrites a base
text. The generator is injectable; the default is a template fallback so drafting
works offline and in tests, and an optional Ollama-backed generator can be wired
in without changing callers.
"""

from __future__ import annotations

from app.text.models import REWRITE_TEXT


def _fallback_generator(prompt: str, language: str, base_text: str = "") -> str:
    """Offline fallback: no LLM. Returns the base text (rewrite) or the prompt."""
    return base_text if base_text else prompt


def compose(
    *,
    action_type: str,
    text: str | None = None,
    instruction: str | None = None,
    base_text: str = "",
    language: str = "es",
    generator=None,
) -> str:
    """Return the final text to insert (never performs any insertion)."""
    if text is not None and text != "":
        return text
    generator = generator or _fallback_generator
    if action_type == REWRITE_TEXT:
        prompt = instruction or "Rewrite the text."
        return generator(prompt, language, base_text)
    prompt = instruction or ""
    return generator(prompt, language, "")
