"""Playwright session wrapper — a dedicated, ISOLATED Chromium profile.

The browser state lives under storage/browser/profile (gitignored). We never
reuse the user's real Chrome/Edge profile and never import their cookies. The
in-page extractor returns only visible structure (URL, title, headings, text,
links, buttons, form controls, a11y labels) — never scripts, cookies, storage,
or a DOM/HTML dump.

Playwright is imported lazily (optional dependency). The service depends on the
small method surface below, so it is fully testable with a fake session.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.config import PROJECT_ROOT

BROWSER_STORAGE = PROJECT_ROOT / "storage" / "browser"
PROFILE_DIR = BROWSER_STORAGE / "profile"

# In-page extractor: returns ONLY visible structure. No scripts, cookies, or
# storage are read; hidden inputs are surfaced with type="hidden" so the snapshot
# builder can drop them. Values are never returned.
_EXTRACT_JS = r"""
() => {
  const visible = (el) => {
    const s = window.getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const headings = [...document.querySelectorAll('h1,h2,h3,h4')]
    .filter(visible).slice(0, 100)
    .map(h => ({level: +h.tagName[1], text: h.innerText.trim()}));
  const links = [...document.querySelectorAll('a[href]')]
    .filter(visible).slice(0, 300)
    .map(a => ({text: a.innerText.trim(), href: a.href}));
  const buttons = [...document.querySelectorAll('button,[role=button],input[type=submit],input[type=button]')]
    .filter(visible).slice(0, 150)
    .map(b => ({text: (b.innerText || b.value || b.getAttribute('aria-label') || '').trim(),
                role: b.getAttribute('role') || 'button'}));
  const form_controls = [...document.querySelectorAll('input,textarea,select')]
    .slice(0, 150)
    .map(c => {
      const id = c.id;
      let label = c.getAttribute('aria-label') || '';
      if (!label && id) { const l = document.querySelector(`label[for="${id}"]`); if (l) label = l.innerText.trim(); }
      return {name: c.name || c.id || '', type: (c.type || c.tagName).toLowerCase(),
              label: label, autocomplete: c.getAttribute('autocomplete') || '',
              role: c.getAttribute('role') || ''};
    });
  const landmarks = [...document.querySelectorAll('[role=main],main,[role=navigation],nav')]
    .filter(visible).slice(0, 20)
    .map(l => l.getAttribute('role') || l.tagName.toLowerCase());
  return {url: location.href, title: document.title,
          text: (document.body ? document.body.innerText : '').slice(0, 200000),
          headings, links, buttons, form_controls, landmarks};
}
"""


class PlaywrightSession:  # pragma: no cover - requires a real browser
    """A single isolated browser context/page. Methods mirror the service's needs."""

    def __init__(self, settings):
        from playwright.sync_api import sync_playwright

        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        engine = getattr(self._pw, settings.browser_engine, self._pw.chromium)
        self._context = engine.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=settings.browser_headless,
            accept_downloads=False,             # no automatic downloads
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._timeout = settings.browser_action_timeout_seconds * 1000
        self.last_activity = time.monotonic()

    def _touch(self):
        self.last_activity = time.monotonic()

    def current_url(self) -> str:
        return self._page.url

    def title(self) -> str:
        return self._page.title()

    def goto(self, url: str) -> None:
        self._page.goto(url, timeout=self._timeout, wait_until="domcontentloaded")
        self._touch()

    def back(self) -> None:
        self._page.go_back(timeout=self._timeout)
        self._touch()

    def forward(self) -> None:
        self._page.go_forward(timeout=self._timeout)
        self._touch()

    def scroll(self, dy: int) -> None:
        self._page.mouse.wheel(0, dy)
        self._touch()

    def snapshot_raw(self) -> dict:
        self._touch()
        return self._page.evaluate(_EXTRACT_JS)

    def fill(self, name: str, value: str) -> None:
        """Fill a field by name/id — NEVER presses Enter or submits."""
        selector = f'[name="{name}"], #{name}'
        self._page.fill(selector, value, timeout=self._timeout)
        self._touch()

    def close(self) -> None:
        try:
            self._context.close()
        finally:
            self._pw.stop()


def create_session(settings) -> PlaywrightSession:  # pragma: no cover
    return PlaywrightSession(settings)
