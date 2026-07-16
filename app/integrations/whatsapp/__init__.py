"""Safe WhatsApp Web drafting + explicitly confirmed sending (Phase 4C).

Reuses the controlled BrowserService with a dedicated persistent profile
(storage/browser/whatsapp) and manual QR login. No unofficial APIs, reverse
engineering, groups, attachments, calls, reactions, deletion, forwarding, status
posting, or bulk operations. Cookies, tokens, and QR data are never read, logged,
or sent to the LLM.

Importing this package registers its intents, confirmation handlers, NL patterns,
and pending-broker domains with the core dispatcher.
"""

from app.integrations.whatsapp import service as _service  # noqa: F401  (side-effect: registers)
