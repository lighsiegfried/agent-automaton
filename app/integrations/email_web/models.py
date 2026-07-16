"""Data models for email drafting/sending. Memory-only, never persisted."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmailDraft:
    action_id: str
    provider: str
    mode: str                       # "new" | "reply"
    recipients: list[str]           # To
    cc: list[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    language: str = "es"
    thread_signature: str = ""      # set when replying
    expires_at_monotonic: float = 0.0
    expires_at_iso: str = ""
    placed: bool = False

    def preview(self, limit: int = 160) -> dict:
        body = self.body if len(self.body) <= limit else self.body[:limit] + "…"
        return {
            "action_id": self.action_id,
            "provider": self.provider,
            "mode": self.mode,
            "recipients": list(self.recipients),
            "cc": list(self.cc),
            "subject": self.subject,
            "body_preview": body,
            "character_count": len(self.body),
            "thread_signature": self.thread_signature,
            "expires_at": self.expires_at_iso,
            "placed": self.placed,
        }


@dataclass
class EmailSendAction:
    action_id: str
    provider: str
    recipients: list[str]
    cc: list[str]
    subject: str
    thread_signature: str
    body_hash: str
    attachment_count: int
    expires_at_monotonic: float
    expires_at_iso: str
    used: bool = False

    def summary(self, body_preview: str) -> dict:
        return {
            "action_id": self.action_id,
            "provider": self.provider,
            "recipients": list(self.recipients),
            "cc": list(self.cc),
            "subject": self.subject,
            "body_preview": body_preview,
            "attachment_count": self.attachment_count,
            "expires_at": self.expires_at_iso,
            "required_phrase": "confirmar envío a " + ", ".join(self.recipients),
        }
