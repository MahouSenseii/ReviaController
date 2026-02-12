"""
Control center tab — per-user retention policies, privacy controls,
feature toggles for emotion coupling / proactive recall / persona lock,
and manual memory freeze/reset for safety incidents.

Publishes commands through the EventBus to control backend systems.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .base_tab import BaseTab


class _ToggleRow(QWidget):
    """Feature toggle row with label, description, and on/off button."""

    def __init__(self, label: str, description: str, initial: bool = True):
        super().__init__()
        self._enabled = initial

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        name_lbl = QLabel(label)
        name_lbl.setStyleSheet("color:#e5e7eb; font-size:12px; font-weight:600;")
        text_col.addWidget(name_lbl)
        desc_lbl = QLabel(description)
        desc_lbl.setStyleSheet("color:#6b7280; font-size:10px;")
        desc_lbl.setWordWrap(True)
        text_col.addWidget(desc_lbl)
        layout.addLayout(text_col, 1)

        self._btn = QPushButton("ON" if initial else "OFF")
        self._btn.setFixedSize(50, 26)
        self._update_style()
        self._btn.clicked.connect(self._toggle)
        layout.addWidget(self._btn)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _toggle(self) -> None:
        self._enabled = not self._enabled
        self._btn.setText("ON" if self._enabled else "OFF")
        self._update_style()

    def _update_style(self) -> None:
        if self._enabled:
            self._btn.setStyleSheet(
                "QPushButton { background:#052e16; color:#22c55e; border:1px solid #22c55e; "
                "border-radius:4px; font-size:10px; font-weight:700; }"
                "QPushButton:hover { background:#14532d; }"
            )
        else:
            self._btn.setStyleSheet(
                "QPushButton { background:#1c1917; color:#6b7280; border:1px solid #374151; "
                "border-radius:4px; font-size:10px; font-weight:700; }"
                "QPushButton:hover { background:#292524; }"
            )


class _DangerButton(QPushButton):
    """Red-styled button for destructive actions."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setFixedHeight(30)
        self.setStyleSheet(
            "QPushButton { background:#450a0a; color:#ef4444; border:1px solid #ef4444; "
            "border-radius:6px; padding:4px 12px; font-size:11px; font-weight:600; }"
            "QPushButton:hover { background:#7f1d1d; }"
        )


class ControlTab(BaseTab):
    """Control center with feature toggles and safety controls."""

    def _build(self) -> None:
        lay = self._layout

        # ── Feature Toggles ──────────────────────────────────
        lay.addWidget(self._heading("Feature Toggles"))

        self._emotion_toggle = _ToggleRow(
            "Emotion Coupling",
            "Tag memories with emotional metadata and bias retrieval accordingly",
        )
        self._proactive_toggle = _ToggleRow(
            "Proactive Recall",
            "Automatically reference relevant memories in conversation",
        )
        self._persona_lock = _ToggleRow(
            "Persona Lock",
            "Enforce strict persona consistency and flag drift",
        )
        self._intent_inference = _ToggleRow(
            "Intent Inference",
            "Infer user goals and motivations from conversation context",
        )
        self._continuity_tracking = _ToggleRow(
            "Continuity Tracking",
            "Track open threads, commitments, and pending tasks",
        )
        self._correction_awareness = _ToggleRow(
            "Correction Awareness",
            "Prefer corrected facts over prior incorrect recalls",
        )

        for toggle in (
            self._emotion_toggle, self._proactive_toggle,
            self._persona_lock, self._intent_inference,
            self._continuity_tracking, self._correction_awareness,
        ):
            lay.addWidget(toggle)

        # Apply button
        apply_row = QHBoxLayout()
        apply_btn = QPushButton("Apply Toggles")
        apply_btn.setObjectName("ModeButton")
        apply_btn.clicked.connect(self._apply_toggles)
        apply_row.addWidget(apply_btn)
        apply_row.addStretch()
        lay.addLayout(apply_row)

        # ── Recall Policy ─────────────────────────────────────
        lay.addWidget(self._heading("Recall Policy"))

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._recall_mode = QComboBox()
        self._recall_mode.addItems(["balanced", "concise", "proactive", "silent"])
        self._recall_mode.setObjectName("SettingsCombo")
        self._recall_mode.setFixedWidth(120)
        mode_row.addWidget(self._recall_mode)
        mode_row.addStretch()

        apply_mode_btn = QPushButton("Set Mode")
        apply_mode_btn.setObjectName("ModeButton")
        apply_mode_btn.clicked.connect(self._set_recall_mode)
        mode_row.addWidget(apply_mode_btn)
        lay.addLayout(mode_row)

        # ── Retention Policy ──────────────────────────────────
        lay.addWidget(self._heading("Retention Policy"))

        ret_info = QLabel(
            "Memories decay over time based on the configured half-life.\n"
            "Entries below the importance floor are archived during consolidation.\n"
            "TTL-expired entries are pruned automatically."
        )
        ret_info.setStyleSheet("color:#9ca3af; font-size:10px;")
        ret_info.setWordWrap(True)
        lay.addWidget(ret_info)

        # ── Safety Controls ───────────────────────────────────
        lay.addWidget(self._heading("Safety Controls"))

        safety_info = QLabel(
            "Use these controls carefully. Actions cannot be undone."
        )
        safety_info.setStyleSheet("color:#f59e0b; font-size:10px;")
        lay.addWidget(safety_info)

        danger_row = QHBoxLayout()

        freeze_btn = _DangerButton("Freeze Memory")
        freeze_btn.setToolTip("Stop all memory writes until unfrozen")
        freeze_btn.clicked.connect(self._freeze_memory)
        danger_row.addWidget(freeze_btn)

        clear_st_btn = _DangerButton("Clear Short-Term")
        clear_st_btn.setToolTip("Delete all short-term memories")
        clear_st_btn.clicked.connect(self._clear_short_term)
        danger_row.addWidget(clear_st_btn)

        reset_btn = _DangerButton("Reset All Memory")
        reset_btn.setToolTip("Delete ALL memories for the active profile")
        reset_btn.clicked.connect(self._reset_all)
        danger_row.addWidget(reset_btn)

        danger_row.addStretch()
        lay.addLayout(danger_row)

        # Subscribe
        self.bus.subscribe("recall_policy_changed", self._on_policy_changed)

    def _apply_toggles(self) -> None:
        self.bus.publish("feature_toggles_changed", {
            "emotion_coupling": self._emotion_toggle.enabled,
            "proactive_recall": self._proactive_toggle.enabled,
            "persona_lock": self._persona_lock.enabled,
            "intent_inference": self._intent_inference.enabled,
            "continuity_tracking": self._continuity_tracking.enabled,
            "correction_awareness": self._correction_awareness.enabled,
        })

    def _set_recall_mode(self) -> None:
        mode = self._recall_mode.currentText()
        self.bus.publish("recall_policy_command", {
            "action": "set_mode",
            "mode": mode,
        })

    def _freeze_memory(self) -> None:
        reply = QMessageBox.warning(
            self, "Freeze Memory",
            "This will stop all memory writes until manually unfrozen.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.bus.publish("memory_freeze", {"frozen": True})

    def _clear_short_term(self) -> None:
        reply = QMessageBox.warning(
            self, "Clear Short-Term",
            "This will delete all short-term memories for the active profile.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.bus.publish("memory_command", {"action": "end_session"})

    def _reset_all(self) -> None:
        reply = QMessageBox.critical(
            self, "Reset All Memory",
            "THIS WILL DELETE ALL MEMORIES for the active profile.\n"
            "This action CANNOT be undone.\n\nAre you absolutely sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.bus.publish("memory_reset_all", {})

    def _on_policy_changed(self, data: dict) -> None:
        mode = data.get("mode", "balanced")
        idx = self._recall_mode.findText(mode)
        if idx >= 0:
            self._recall_mode.setCurrentIndex(idx)
