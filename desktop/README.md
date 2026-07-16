# Fifi desktop app (Phase 6C)

A native **PySide6** conversational client for Fifi — not Electron, no embedded browser,
no web runtime. It is a **thin presentation layer** over the local API: it holds no
business logic and writes no database. The backend stays the single source of truth and
the single place effects are authorized and confirmed.

## Run (Windows host)

```powershell
# 1. Enable the bridge in .env
ENABLE_DESKTOP_BRIDGE=true

# 2. Start the API (as usual), then install the UI deps and launch the window
.venv\Scripts\python.exe -m pip install -r requirements-desktop-ui.txt
.venv\Scripts\python.exe -m desktop
```

## Architecture

| Module | Responsibility |
| --- | --- |
| `desktop/client.py` | Loopback-only HTTP client (injectable transport). |
| `desktop/controller.py` | Conversational orchestration: send / confirm / ask / cancel. |
| `desktop/viewmodels.py` | Raw API JSON → **sanitized** view-models. |
| `desktop/models.py` | The immutable VMs the UI renders. |
| `desktop/stream.py` | SSE consumer for the safe activity stream. |
| `desktop/settingsview.py` | Read-only, no-secret settings rows. |
| `desktop/safety.py` | sanitize / loopback / no-generic-confirm rules. |
| `desktop/qt_app.py` | The thin Qt adapter (lazy PySide6; not unit-tested). |

Server side (`app/uibridge/`): `POST /ui/session` mints a short-lived, in-memory UI
session token; `GET /ui/stream` streams **whitelisted** activity events (SSE);
`GET /ui/overview` returns a safe snapshot. All are loopback-only and flag-gated.

## Safety rules the desktop enforces

1. **No raw HTML/markup.** Every dynamic string (replies, web/email/document text,
   filenames, citations, activity titles) is escaped and control-stripped by
   `sanitize()` before it reaches a widget, and rich-text is disabled on the labels.
2. **Loopback only.** The client refuses any non-`127.0.0.1`/`::1`/`localhost` base URL.
3. **No generic confirmation.** A confirmation card shows the action and its **exact**
   domain phrase; the button sends that exact phrase. There is no "just say yes" path.
4. **Short-lived UI session token.** Minted per process, in memory, expiring; a server
   restart invalidates every token. It gates the stream/overview endpoints.
5. **No database writes.** The desktop only calls the API; it has no local store.

All logic modules are unit-tested with a mocked transport (`tests/test_desktop_*.py`,
`tests/test_uibridge_*.py`); only `qt_app.py` needs a display and is excluded from
coverage.
