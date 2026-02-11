"""
Left sidebar panel — avatar, module pills, mode selectors, profiles.

Publishes events when the user toggles modules or selects a mode/profile.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.events import EventBus
from ui.widgets import Pill, SectionLabel, StatusDot

from .base_panel import BasePanel


class _ModeButton(QPushButton):
    """Flat selectable button used for Modes and Profiles."""

    def __init__(self, text: str, group: str, event_bus: EventBus):
        super().__init__(text)
        self._group = group
        self._bus = event_bus
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("ModeButton")
        self.clicked.connect(self._on_click)

    def _on_click(self) -> None:
        self._bus.publish(f"{self._group}_selected", {"value": self.text()})


class SidebarPanel(BasePanel):
    """Left-hand sidebar: avatar, status pills, modes, profiles."""

    def __init__(self, event_bus: EventBus, config: Config):
        super().__init__(event_bus, config)
        self.setFixedWidth(260)

    def _build(self) -> None:
        lay = self._inner_layout

        # ── Avatar ────────────────────────────────────────────
        avatar = QLabel()
        avatar.setObjectName("AvatarImage")
        avatar.setFixedSize(220, 220)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap("assets/avatar.png")
        if not pix.isNull():
            avatar.setPixmap(
                pix.scaled(
                    avatar.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        lay.addWidget(avatar, alignment=Qt.AlignmentFlag.AlignHCenter)

        # ── Name row ──────────────────────────────────────────
        name_row = QWidget()
        nr = QHBoxLayout(name_row)
        nr.setContentsMargins(0, 0, 0, 0)
        nr.setSpacing(8)

        self._status_dot = StatusDot("on", size=10)
        self._status_dot.setObjectName("StatusDot")

        name = QLabel(
            '<span style="color:#ffffff; font-weight:600;">Astra</span> '
            '<span style="color:#7fb3ff; font-weight:300;">(Assistant)</span>'
        )
        nr.addStretch(1)
        nr.addWidget(self._status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        nr.addWidget(name, alignment=Qt.AlignmentFlag.AlignVCenter)
        nr.addStretch(1)
        lay.addWidget(name_row)
        lay.addSpacing(6)

        # ── Module pills ──────────────────────────────────────
        self._stt_pill = Pill("STT Active", "Errors: 0 / 10m", status="on", toggle=True, checked=True)
        self._stt_pill.toggled.connect(lambda on: self.bus.publish("stt_toggled", {"enabled": on}))
        lay.addWidget(self._stt_pill)

        self._tts_pill = Pill("TTS Active", "Errors: 0 / 10m", status="on", toggle=True, checked=True)
        self._tts_pill.toggled.connect(lambda on: self.bus.publish("tts_toggled", {"enabled": on}))
        lay.addWidget(self._tts_pill)

        self._vision_pill = Pill("Vision Active", "Errors: 0 / 10m", status="on", toggle=True, checked=True)
        self._vision_pill.toggled.connect(lambda on: self.bus.publish("vision_toggled", {"enabled": on}))
        lay.addWidget(self._vision_pill)

        # ── Modes ─────────────────────────────────────────────
        lay.addSpacing(14)
        lay.addWidget(SectionLabel("Modes"))

        self._mode_buttons: list[_ModeButton] = []
        for label in ("Passive (Background)", "Interactive (Voice)", "Teaching / Explain", "Debug"):
            btn = _ModeButton(label, "mode", self.bus)
            self._mode_buttons.append(btn)
            lay.addWidget(btn)

        # ── Profiles ──────────────────────────────────────────
        lay.addSpacing(14)
        lay.addWidget(SectionLabel("Profiles"))

        self._profile_buttons: list[_ModeButton] = []
        for label in ("Default", "Technical", "Custom"):
            btn = _ModeButton(label, "profile", self.bus)
            self._profile_buttons.append(btn)
            lay.addWidget(btn)

        lay.addStretch(1)

        # ── Wire up mutual exclusion ─────────────────────────
        self.bus.subscribe("mode_selected", self._on_mode_selected)
        self.bus.subscribe("profile_selected", self._on_profile_selected)

        # ── Wire up module status from plugins ────────────────
        self.bus.subscribe("module_status", self._on_module_status)

    # ── Event handlers ────────────────────────────────────────

    def _on_mode_selected(self, data: dict) -> None:
        value = data.get("value")
        for btn in self._mode_buttons:
            btn.setChecked(btn.text() == value)
        self.config.set("mode", value)

    def _on_profile_selected(self, data: dict) -> None:
        value = data.get("value")
        for btn in self._profile_buttons:
            btn.setChecked(btn.text() == value)
        self.config.set("profile", value)

    def _on_module_status(self, data: dict) -> None:
        """
        Expected data: {"module": "stt"|"tts"|"vision",
                        "status": "on"|"warn"|"off",
                        "subtitle": "Errors: 2 / 10m"}
        """
        module = data.get("module")
        pill_map = {"stt": self._stt_pill, "tts": self._tts_pill, "vision": self._vision_pill}
        p = pill_map.get(module)
        if p is None:
            return
        if "status" in data:
            p.set_status(data["status"])
        if "subtitle" in data:
            p.set_subtitle(data["subtitle"])
