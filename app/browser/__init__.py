"""Safe browser navigation, page understanding, and controlled form filling (Phase 4B).

Reading, searching, scrolling, and navigating are safe. Filling form fields is
sensitive and requires an explicit confirmation phrase. Submitting, purchasing,
publishing, uploading, deleting, accepting legal terms, sending messages, and
logging in are blocked in this phase. A dedicated, isolated Chromium profile is
used — never the user's real browser — and passwords, cookies, tokens, and full
browser storage are never exposed to the LLM.
"""
