"""Third-party web integrations that reuse the controlled BrowserService.

Each integration (WhatsApp Web, web email) drives a dedicated, isolated browser
profile through the same controlled Playwright abstraction — never a second
uncontrolled runtime, never the user's normal browser — and routes all sensitive
actions through the unified pending-action broker with explicit confirmation.
"""
