"""Data models for WhatsApp drafting/sending. Memory-only, never persisted."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContactCandidate:
    name: str


@dataclass
class WhatsAppDraft:
    """A safe, local draft — nothing has touched WhatsApp yet."""

    action_id: str
    recipient: str
    text: str
    language: str
    chat_signature: str
    expires_at_monotonic: float
    expires_at_iso: str
    placed: bool = False

    def preview(self, limit: int = 120) -> dict:
        """Safe preview — a TRUNCATED body, never the full sensitive message."""
        body = self.text if len(self.text) <= limit else self.text[:limit] + "…"
        return {
            "action_id": self.action_id,
            "recipient": self.recipient,
            "body_preview": body,
            "character_count": len(self.text),
            "chat_signature": self.chat_signature,
            "expires_at": self.expires_at_iso,
            "placed": self.placed,
        }


@dataclass
class SendAction:
    """A pending send — created only after the draft is visibly placed."""

    action_id: str
    recipient: str
    character_count: int
    chat_signature: str
    composer_hash: str
    expires_at_monotonic: float
    expires_at_iso: str
    used: bool = False

    def summary(self, body_preview: str) -> dict:
        return {
            "action_id": self.action_id,
            "recipient": self.recipient,
            "body_preview": body_preview,
            "character_count": self.character_count,
            "chat_signature": self.chat_signature,
            "expires_at": self.expires_at_iso,
            "required_phrase": f"confirmar envío a {self.recipient}",
        }
