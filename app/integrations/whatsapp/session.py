"""Real WhatsApp Web session adapter (Phase 4C) — a CONTROLLED, isolated browser.

Uses the same controlled Playwright pattern as BrowserService with a DEDICATED
persistent profile (never the user's normal browser). Login is manual by QR. This
adapter only ever reads visible names/titles/composer text and types/clicks — it
never reads cookies, localStorage, tokens, or QR content. Lazy import; tests use a
fake adapter, so this is not unit-tested.
"""

from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT, get_settings
from app.integrations.whatsapp.selectors import SELECTORS, WHATSAPP_URL


class WhatsAppSession:  # pragma: no cover - requires a real browser + manual login
    def __init__(self, settings=None):
        from playwright.sync_api import sync_playwright

        settings = settings or get_settings()
        profile = PROJECT_ROOT / settings.whatsapp_profile_dir
        profile.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        self._context = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(profile), headless=settings.browser_headless,
            accept_downloads=False,
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._timeout = settings.browser_action_timeout_seconds * 1000
        if not self._page.url.startswith(WHATSAPP_URL):
            self._page.goto(WHATSAPP_URL, wait_until="domcontentloaded", timeout=self._timeout)

    def logged_in(self) -> bool:
        # Logged in when the chat list is present and the QR canvas is not.
        try:
            return self._page.query_selector(SELECTORS["chat_list"]) is not None
        except Exception:
            return False

    def search_contacts(self, name: str) -> list[str]:
        try:
            box = self._page.query_selector(SELECTORS["search_box"])
            if box is None:
                return []
            box.click()
            box.fill("")
            box.type(name, delay=10)
            self._page.wait_for_timeout(600)
            rows = self._page.query_selector_all(SELECTORS["contact_rows"])
            names = [r.get_attribute("title") or r.inner_text() for r in rows]
            return [n.strip() for n in names if n and n.strip()]
        except Exception:
            return []

    def open_chat(self, name: str) -> bool:
        try:
            rows = self._page.query_selector_all(SELECTORS["contact_rows"])
            for row in rows:
                if (row.get_attribute("title") or "").strip() == name.strip():
                    row.click()
                    self._page.wait_for_timeout(400)
                    return True
            return False
        except Exception:
            return False

    def chat_title(self) -> str:
        try:
            el = self._page.query_selector(SELECTORS["chat_title"])
            return (el.get_attribute("title") or el.inner_text() or "").strip() if el else ""
        except Exception:
            return ""

    def composer_text(self) -> str:
        try:
            el = self._page.query_selector(SELECTORS["composer"])
            return el.inner_text().strip() if el else ""
        except Exception:
            return ""

    def set_composer(self, text: str) -> None:
        el = self._page.query_selector(SELECTORS["composer"])
        if el is None:
            raise RuntimeError("composer not found")
        el.click()
        el.fill("")
        el.type(text, delay=5)   # types text only — never Enter

    def click_send(self) -> None:
        button = self._page.query_selector(SELECTORS["send_button"])
        if button is None:
            raise RuntimeError("send button not found")
        button.click()           # exactly one click; never Enter

    def close(self) -> None:
        try:
            self._context.close()
        finally:
            self._pw.stop()
