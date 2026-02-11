"""
Logs tab — scrollable, filterable log viewer.

Subscribes to ``log_entry`` events and appends them.  The user can
filter by category (Allowed / Filtered / Rewritten).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QWidget,
)

from .base_tab import BaseTab


class LogsTab(BaseTab):

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Event Log"))

        # ── Toolbar row ───────────────────────────────────────
        toolbar = QWidget()
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(8)

        self._filter_combo = QComboBox()
        self._filter_combo.setObjectName("SettingsCombo")
        self._filter_combo.addItems(["All", "Allowed", "Filtered", "Rewritten"])
        self._filter_combo.currentTextChanged.connect(self._apply_filter)
        tb.addWidget(self._filter_combo, 1)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("ModeButton")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._clear)
        tb.addWidget(clear_btn)

        lay.addWidget(toolbar)

        # ── Log display ───────────────────────────────────────
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setObjectName("LogView")
        self._log_view.setStyleSheet(
            "background:#0a0f18; border:1px solid #2a3b55; border-radius:8px; "
            "color:#c7d3e6; padding:8px; font-family:'Consolas','Courier New',monospace; "
            "font-size:12px;"
        )
        lay.addWidget(self._log_view, 1)

        # Internal storage
        self._entries: list[dict] = []

        # Subscribe
        self.bus.subscribe("log_entry", self._on_log_entry)

    def _on_log_entry(self, data: dict) -> None:
        """
        Expected data: {"category": "Allowed"|"Filtered"|"Rewritten",
                        "text": "..."}
        """
        self._entries.append(data)
        self._apply_filter(self._filter_combo.currentText())

    def _apply_filter(self, category: str) -> None:
        if category == "All":
            visible = self._entries
        else:
            visible = [e for e in self._entries if e.get("category") == category]

        lines = []
        for e in visible:
            tag = e.get("category", "?")
            text = e.get("text", "")
            lines.append(f"[{tag}] {text}")

        self._log_view.setPlainText("\n".join(lines))

    def _clear(self) -> None:
        self._entries.clear()
        self._log_view.clear()
