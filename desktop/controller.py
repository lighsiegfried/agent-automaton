"""Conversational controller for the desktop app (Phase 6C).

Orchestrates a turn: send a message, surface the reply and (if the backend asked for
one) a confirmation card, confirm with the EXACT domain phrase, cancel, or ask the
Knowledge Vault for a grounded answer. It is UI-agnostic and fully testable — the Qt
layer only calls these methods and renders the returned view-models.

Guarantees enforced here (not duplicated from the backend, just honored):
- **No business logic / no database.** The controller only calls :class:`ApiClient`;
  it holds no store and performs no persistence.
- **No generic confirmation.** :meth:`confirm` refuses a card that has no exact phrase,
  and always sends that exact phrase — never a bare "yes"/confirm flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from desktop.client import ApiClient
from desktop.models import AnswerVM, ConfirmationCardVM, MessageVM
from desktop import viewmodels as vm


@dataclass(frozen=True)
class Turn:
    """The result of one interaction: the reply, and an optional confirmation card."""
    reply: MessageVM
    card: ConfirmationCardVM | None = None
    user: MessageVM | None = None


_NEEDS_CONFIRMATION = "needs_confirmation"


class DesktopController:
    def __init__(self, client: ApiClient):
        self.client = client

    # -- session -------------------------------------------------------------------

    def connect(self) -> dict:
        """Mint the short-lived UI session token used for the stream/overview."""
        return self.client.mint_session()

    # -- conversation --------------------------------------------------------------

    def send(self, text: str) -> Turn:
        user = vm.user_message(text)
        resp = self.client.send_command(text)
        reply = vm.message_from_command(resp)
        card = self._card_if_pending(reply.status)
        return Turn(reply=reply, card=card, user=user)

    def confirm(self, card: ConfirmationCardVM) -> Turn:
        """Confirm ``card`` by sending its EXACT phrase. A card without a concrete
        phrase (``can_confirm`` False) is refused with a system message — the desktop
        never turns that into a generic 'yes'."""
        if card is None or not card.can_confirm or not card.phrase.strip():
            return Turn(reply=MessageVM(
                role="system",
                text="This action can only be confirmed by voice or with its exact phrase.",
                status="refused"))
        resp = self.client.confirm(card.phrase)          # sends the exact phrase
        reply = vm.message_from_command(resp)
        return Turn(reply=reply, card=self._card_if_pending(reply.status))

    def cancel(self) -> Turn:
        resp = self.client.cancel_pending()
        status = (resp or {}).get("status", "")
        text = "Cancelled the pending action." if status == "cancelled" else "Nothing to cancel."
        return Turn(reply=MessageVM(role="system", text=text, status=status))

    def ask(self, question: str) -> AnswerVM:
        """Ask the Knowledge Vault; returns a grounded answer + citations (or an
        insufficient-evidence answer with no citations)."""
        return vm.answer_from_ask(self.client.knowledge_ask(question))

    # -- state ---------------------------------------------------------------------

    def pending_card(self) -> ConfirmationCardVM | None:
        return vm.confirmation_from_pending(self.client.pending())

    def overview(self) -> dict:
        return self.client.overview()

    # -- helpers -------------------------------------------------------------------

    def _card_if_pending(self, status: str) -> ConfirmationCardVM | None:
        if status == _NEEDS_CONFIRMATION:
            return vm.confirmation_from_pending(self.client.pending())
        return None
