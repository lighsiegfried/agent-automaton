"""Thin PySide6 adapter for the Fifi desktop app (Phase 6C).

This is the ONLY module that imports Qt, and it contains no business logic: it wires the
:class:`DesktopController` and the sanitized view-models to widgets. All heavy imports
are lazy so the rest of the package (and its tests) run without PySide6 installed.

Rendering rule: every widget is fed text that :mod:`desktop.viewmodels` already
sanitized, and rich-text is DISABLED on the labels (``setTextFormat(PlainText)``), so
even if an un-sanitized string slipped through it could only render as literal text.
Run on the Windows host:  python -m desktop  (after installing requirements-desktop-ui).
"""

from __future__ import annotations

import sys

from desktop.client import ApiClient
from desktop.controller import DesktopController
from desktop import settingsview


def _require_qt():  # pragma: no cover - environment probe
    try:
        import PySide6  # noqa: F401
        return True
    except Exception:
        print("Desktop UI dependencies missing. Install them on the host:")
        print("  .venv\\Scripts\\python.exe -m pip install -r requirements-desktop-ui.txt")
        return False


class DesktopWindow:  # pragma: no cover - requires a display + PySide6
    """The main window: a transcript, a confirmation-card area, a citation strip, an
    input box, and a read-only settings pane. Thin — it delegates every action to the
    controller and only paints sanitized view-models."""

    def __init__(self, controller: DesktopController):
        from PySide6.QtWidgets import (
            QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
        )

        self.controller = controller
        self.window = QWidget()
        self.window.setWindowTitle("Fifi")
        layout = QVBoxLayout(self.window)

        self._transcript = QVBoxLayout()
        transcript_host = QWidget()
        transcript_host.setLayout(self._transcript)
        scroll = QScrollArea()
        scroll.setWidget(transcript_host)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        self._card_area = QVBoxLayout()
        layout.addLayout(self._card_area)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Message Fifi…")
        self._input.returnPressed.connect(self._on_send)
        layout.addWidget(self._input)

        send = QPushButton("Send")
        send.clicked.connect(self._on_send)
        layout.addWidget(send)

        self._QLabel = QLabel
        self._QPushButton = QPushButton

    def _add_label(self, layout, text: str) -> None:
        from PySide6.QtGui import Qt

        label = self._QLabel(text)
        label.setTextFormat(Qt.PlainText)          # never interpret markup
        label.setWordWrap(True)
        layout.addWidget(label)

    def _clear_cards(self) -> None:
        while self._card_area.count():
            item = self._card_area.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_turn(self, turn) -> None:
        if turn.user is not None:
            self._add_label(self._transcript, f"You: {turn.user.text}")
        self._add_label(self._transcript, f"Fifi: {turn.reply.text}")
        self._clear_cards()
        if turn.card is not None:
            self._render_card(turn.card)

    def _render_card(self, card) -> None:
        prompt = f"Confirm {card.domain} for {card.target}?"
        if card.display_phrase:
            prompt += f"  (phrase: {card.display_phrase})"
        self._add_label(self._card_area, prompt)
        if card.can_confirm:
            button = self._QPushButton(f"Confirm: “{card.phrase}”")
            button.clicked.connect(lambda: self._render_turn(self.controller.confirm(card)))
            self._card_area.addWidget(button)
        else:
            self._add_label(self._card_area,
                            "This action can only be confirmed by voice or its exact phrase.")

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._render_turn(self.controller.send(text))

    def show(self) -> None:
        self.window.show()


def run(argv=None) -> int:  # pragma: no cover - requires a display + PySide6
    if not _require_qt():
        return 1
    from PySide6.QtWidgets import QApplication

    client = ApiClient()
    controller = DesktopController(client)
    controller.connect()                            # mint the short-lived UI token

    app = QApplication(argv or sys.argv)
    window = DesktopWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
