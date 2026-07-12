"""Thread-local progress reporting — engines narrate, jobs listen.

Long operations run inside job threads (app.jobs). A job installs a reporter
for its thread; any code it calls (engines, designer) emits stages with
report(). Outside a job the calls are no-ops, so engines stay usable from the
plain synchronous endpoints and the CLI without any coupling.
"""

import threading
from typing import Any, Callable

_local = threading.local()

Reporter = Callable[..., None]  # (step: str, **fields) -> None


def set_reporter(reporter: Reporter) -> None:
    _local.reporter = reporter


def current_reporter() -> Reporter | None:
    """This thread's reporter, so helper threads (e.g. the download watcher)
    can inherit it — thread-locals do not cross thread boundaries on their own."""
    return getattr(_local, "reporter", None)


def clear_reporter() -> None:
    _local.reporter = None


def report(step: str, **fields: Any) -> None:
    """Emit a progress stage from anywhere inside a job thread. Never raises."""
    reporter = getattr(_local, "reporter", None)
    if reporter is None:
        return
    try:
        reporter(step, **fields)
    except Exception:
        pass


class Cancelled(Exception):
    """Raised cooperatively inside a job when the user cancelled it."""


def check_cancelled() -> None:
    """Raise Cancelled if this job thread's cancel flag is set. No-op otherwise."""
    event = getattr(_local, "cancel_event", None)
    if event is not None and event.is_set():
        raise Cancelled()


def set_cancel_event(event: threading.Event) -> None:
    _local.cancel_event = event


def clear_cancel_event() -> None:
    _local.cancel_event = None
