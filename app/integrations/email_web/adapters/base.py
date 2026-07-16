"""Email adapter interface + factory.

A provider adapter drives a dedicated, isolated browser profile through the same
controlled Playwright pattern as BrowserService (never a second uncontrolled
runtime, never the user's normal browser). It only reads visible content and
types/clicks; it never reads cookies, localStorage, or tokens. The concrete
Gmail/Outlook adapters are lazy (real browser) — tests inject a fake.

Required method surface (duck-typed):
  provider() -> str
  logged_in() -> bool
  search(query) -> list[{sender, subject, date, snippet, thread_id}]   # SAFE
  open_thread(thread_id) -> {subject, sender, recipients, date, messages, attachment_count}
  current_thread() -> {subject, sender}
  start_new() / start_reply() -> None            # open a compose pane
  set_compose(to, subject, body, cc=[]) -> None  # fill fields; NEVER submit
  compose_state() -> {to, cc, bcc, subject, body, attachment_count, mode}
  click_send() -> None                           # exactly one click; never Enter
  close() -> None
"""

from __future__ import annotations


def create_adapter(provider: str, settings):
    provider = (provider or "").lower()
    if provider == "gmail":
        from app.integrations.email_web.adapters.gmail import GmailAdapter

        return GmailAdapter(settings)
    if provider == "outlook":
        from app.integrations.email_web.adapters.outlook import OutlookAdapter

        return OutlookAdapter(settings)
    return None


class BrowserEmailAdapter:  # pragma: no cover - requires a real browser + login
    """Shared Playwright plumbing; providers supply URL, profile subdir, selectors."""

    URL = ""
    PROFILE_SUBDIR = ""
    SELECTORS: dict = {}
    PROVIDER = ""

    def __init__(self, settings=None):
        from pathlib import Path

        from playwright.sync_api import sync_playwright

        from app.config import PROJECT_ROOT, get_settings

        settings = settings or get_settings()
        profile = Path(PROJECT_ROOT) / "storage" / "browser" / "email" / self.PROFILE_SUBDIR
        profile.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        self._context = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(profile), headless=settings.browser_headless,
            accept_downloads=False,
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._timeout = settings.browser_action_timeout_seconds * 1000
        if not self._page.url.startswith(self.URL):
            self._page.goto(self.URL, wait_until="domcontentloaded", timeout=self._timeout)

    def provider(self) -> str:
        return self.PROVIDER

    def logged_in(self) -> bool:
        try:
            return self._page.query_selector(self.SELECTORS.get("logged_in", "")) is not None
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._context.close()
        finally:
            self._pw.stop()

    # search / open_thread / compose etc. are provider-specific and implemented
    # in the subclasses using self._page + self.SELECTORS.
