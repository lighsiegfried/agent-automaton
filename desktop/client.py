"""Localhost API client for the desktop app (Phase 6C).

A thin, dependency-light HTTP client that talks ONLY to the local Fifi API. It holds no
business logic: it sends conversational commands, reads the pending action, confirms
with an EXACT phrase, asks the Knowledge Vault, and reads safe status/overview. The
transport is injected (``transport(method, path, *, json, params, headers) -> (status,
body)``) so every test runs without a real socket; the default transport is a lazy
httpx call bound to the loopback base URL.

Two invariants live here: the base URL MUST be loopback (:func:`desktop.safety.
require_local_base`), and the client NEVER persists anything — it has no database.
"""

from __future__ import annotations

from desktop.safety import require_local_base


def _httpx_transport(base_url: str):
    def transport(method: str, path: str, *, json=None, params=None, headers=None):
        import httpx

        resp = httpx.request(method, f"{base_url}{path}", json=json, params=params,
                             headers=headers, timeout=30.0)
        try:
            body = resp.json()
        except Exception:
            body = {}
        return resp.status_code, body
    return transport


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, transport=None,
                 token: str | None = None):
        self.base_url = require_local_base(base_url.rstrip("/"))
        self._transport = transport or _httpx_transport(self.base_url)
        self.token = token

    # -- low level -----------------------------------------------------------------

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _get(self, path, *, params=None):
        return self._transport("GET", path, params=params, headers=self._headers())

    def _post(self, path, *, json=None):
        return self._transport("POST", path, json=json or {}, headers=self._headers())

    # -- UI session (short-lived token) --------------------------------------------

    def mint_session(self) -> dict:
        status, body = self._post("/ui/session")
        if status == 200 and body.get("status") == "ok":
            self.token = body["data"]["token"]
        return body

    def revoke_session(self) -> dict:
        _, body = self._post("/ui/session/revoke")
        self.token = None
        return body

    # -- conversation --------------------------------------------------------------

    def send_command(self, text: str, *, confirm: bool = False, language=None) -> dict:
        payload = {"text": text, "confirm": confirm}
        if language:
            payload["language"] = language
        _, body = self._post("/command", json=payload)
        return body

    def pending(self) -> dict | None:
        """The pending action WITH its exact ``confirm_phrase`` (via the token-gated
        bridge endpoint). Falls back to the plain summary if the bridge is unavailable."""
        _, body = self._get("/ui/pending")
        data = (body or {}).get("data")
        if isinstance(data, dict):
            return data.get("pending")
        _, plain = self._get("/pending")          # bridge off → best-effort summary
        return (plain or {}).get("pending")

    def confirm(self, phrase: str) -> dict:
        """Confirm by sending the EXACT domain phrase as a command. There is no
        generic 'yes' path: an empty/blank phrase is refused before any network call."""
        if not (phrase or "").strip():
            raise ValueError("refusing to confirm without an exact phrase")
        return self.send_command(phrase)

    def cancel_pending(self) -> dict:
        _, body = self._post("/pending/cancel")
        return body

    # -- knowledge (grounded answers with citations) -------------------------------

    def knowledge_ask(self, question: str, *, collection=None) -> dict:
        _, body = self._post("/knowledge/ask", json={"question": question,
                                                     "collection": collection or ""})
        return (body or {}).get("data") or {}

    # -- safe status / overview ----------------------------------------------------

    def security_status(self) -> dict:
        _, body = self._get("/security/status")
        return (body or {}).get("data") or {}

    def overview(self) -> dict:
        _, body = self._get("/ui/overview")
        return (body or {}).get("data") or {}

    # -- safe, read-only identity/health (for the settings pane) -------------------

    def identity(self) -> dict:
        _, body = self._get("/identity")
        return body or {}

    def health(self) -> dict:
        _, body = self._get("/health")
        return body or {}
