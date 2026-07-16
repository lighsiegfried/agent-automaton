"""Gmail web adapter (Phase 4D) — controlled browser, dedicated profile.

Lazy/real (needs manual login); the safety machine is tested via a fake adapter,
so this is not unit-tested. Reads only visible content; never cookies/tokens.
"""

from __future__ import annotations

from app.integrations.email_web.adapters.base import BrowserEmailAdapter


class GmailAdapter(BrowserEmailAdapter):  # pragma: no cover - requires real Gmail login
    PROVIDER = "gmail"
    URL = "https://mail.google.com/"
    PROFILE_SUBDIR = "gmail"
    SELECTORS = {
        "logged_in": "div[role='main'], div[gh='tl']",
        "search_box": "input[aria-label='Search mail'], input[name='q']",
        "thread_rows": "tr.zA",
        "sender": "span.yP, span[email]",
        "subject": "h2.hP",
        "message_body": "div.a3s",
        "compose_to": "textarea[name='to'], input[aria-label='To']",
        "compose_subject": "input[name='subjectbox']",
        "compose_body": "div[aria-label='Message Body']",
        "reply_button": "div[aria-label='Reply']",
        "compose_button": "div[gh='cm']",
        "send_button": "div[role='button'][data-tooltip*='Send'], div[aria-label*='Send']",
        "attachment_chip": "div.dtx, div[aria-label*='attachment']",
    }

    def search(self, query: str) -> list[dict]:
        raise NotImplementedError("live Gmail DOM scraping — run with a real session")

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
