"""Outlook web adapter (Phase 4D) — controlled browser, dedicated profile.

Lazy/real (needs manual login); the safety machine is tested via a fake adapter.
Reads only visible content; never cookies/tokens.
"""

from __future__ import annotations

from app.integrations.email_web.adapters.base import BrowserEmailAdapter


class OutlookAdapter(BrowserEmailAdapter):  # pragma: no cover - requires real Outlook login
    PROVIDER = "outlook"
    URL = "https://outlook.office.com/mail/"
    PROFILE_SUBDIR = "outlook"
    SELECTORS = {
        "logged_in": "div[role='main'], div[aria-label='Message list']",
        "search_box": "input[aria-label='Search'], input[placeholder='Search']",
        "thread_rows": "div[role='option']",
        "sender": "span[title][class*='sender'], div[aria-label*='From']",
        "subject": "div[role='heading']",
        "message_body": "div[aria-label='Message body']",
        "compose_to": "div[aria-label='To']",
        "compose_subject": "input[aria-label='Add a subject']",
        "compose_body": "div[aria-label='Message body'][contenteditable='true']",
        "reply_button": "button[aria-label='Reply']",
        "compose_button": "button[aria-label='New mail']",
        "send_button": "button[aria-label='Send']",
        "attachment_chip": "div[aria-label*='attachment']",
    }

    def search(self, query: str) -> list[dict]:
        raise NotImplementedError("live Outlook DOM scraping — run with a real session")

    def open_thread(self, thread_id: str) -> dict:
        raise NotImplementedError

    def current_thread(self) -> dict:
        return {"subject": "", "sender": ""}

    def start_new(self) -> None:
        raise NotImplementedError

    def start_reply(self) -> None:
        raise NotImplementedError

    def set_compose(self, to, subject, body, cc=None) -> None:
        raise NotImplementedError

    def compose_state(self) -> dict:
        raise NotImplementedError

    def click_send(self) -> None:
        button = self._page.query_selector(self.SELECTORS["send_button"])
        if button is None:
            raise RuntimeError("send button not found")
        button.click()   # one click; never Enter
