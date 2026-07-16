"""WhatsApp Web DOM selectors (data only) used by the session adapter.

Kept in one place so a WhatsApp UI change is a one-file fix. These are read/typed
through the controlled browser session; nothing here reads cookies, localStorage,
or tokens. Groups/attachments/etc. are intentionally absent — unsupported.
"""

WHATSAPP_URL = "https://web.whatsapp.com/"

SELECTORS = {
    # Presence of the QR canvas => not logged in; presence of the chat list =>
    # logged in. (The adapter only checks presence, never reads QR content.)
    "qr_canvas": "canvas[aria-label*='scan'], div[data-testid='qrcode']",
    "chat_list": "div[aria-label='Chat list'], div[data-testid='chat-list']",
    "search_box": "div[contenteditable='true'][data-tab='3'], div[title='Search input textbox']",
    "contact_rows": "div[aria-label='Chat list'] span[title], div[data-testid='cell-frame-title'] span[title]",
    "chat_title": "header span[title], header div[data-testid='conversation-info-header'] span[title]",
    "composer": "div[contenteditable='true'][data-tab='10'], footer div[contenteditable='true']",
    "send_button": "button[aria-label='Send'], span[data-testid='send'], button[data-testid='compose-btn-send']",
}
